"""
Plotting utilities for DirectionalSensitivityExplainer benchmark results.

Updated to support the NEW benchmark format:

{
    "classification": {
        "dataset": {
            "model": {
                "dist_method": [ records... ]
            }
        }
    },
    "regression": { ... }
}

This file is a fully patched version of your original plotter.
"""

import json
import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── palette ──────────────────────────────────────────────────────────────────
BLUE = "#2563EB"
TEAL = "#0D9488"
AMBER = "#D97706"
RED = "#DC2626"
PURPLE = "#7C3AED"
SLATE = "#475569"
LIGHT = "#F1F5F9"
GRID = "#E2E8F0"

DIST_COLORS = {
    "percentile": BLUE,
    "wasserstein": TEAL,
    "mahalanobis": AMBER,
    "counterfactual": RED,
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _flatten(results, task):
    """
    Flatten the nested dict into a list of dicts, one per record.

    NEW expected structure:
        results[dataset][model][dist_method] = list of records
    """
    rows = []
    for ds, models in results.items():
        for model, dists in models.items():
            for dist, records in dists.items():
                if not isinstance(records, list):
                    continue
                for r in records:
                    row = dict(r)
                    row["dataset"] = ds
                    row["model"] = model
                    row["dist_method"] = dist
                    row["group_size"] = len(r.get("feature_group", []))
                    rows.append(row)
    return pd.DataFrame(rows)


def _hit_counts(results):
    """
    Returns a DataFrame: (dataset, model, dist_method) → n_hits
    """
    rows = []
    for ds, models in results.items():
        for model, dists in models.items():
            for dist, records in dists.items():
                n = len(records) if isinstance(records, list) else 0
                rows.append(
                    {"dataset": ds, "model": model, "dist_method": dist, "n_hits": n}
                )
    return pd.DataFrame(rows)


def _feature_counts(df, task):
    """Count how often each feature index appears across all records."""
    counts = Counter()
    for _, row in df.iterrows():
        for feat in row.get("feature_group", []):
            counts[int(feat)] += 1
    return counts


def _move_magnitudes(df, task):
    """Extract per-feature perturbation magnitudes from move_values."""
    rows = []
    for _, row in df.iterrows():
        mv = row.get("move_values", {})
        if isinstance(mv, dict):
            for feat, val in mv.items():
                # val = ["continuous", old, new]
                if isinstance(val, list) and len(val) == 3:
                    ftype, old, new = val
                    if ftype == "continuous":
                        rows.append(
                            {
                                "feature": int(feat),
                                "delta": abs(new - old),
                                "direction": row.get("direction", ""),
                                "dist_method": row.get("dist_method", ""),
                                "dataset": row.get("dataset", ""),
                                "model": row.get("model", ""),
                            }
                        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────


class ExplainerPlotter:
    def __init__(self, json_path: str, task: str):
        """
        Parameters
        ----------
        json_path : str
            Path to the JSON file produced by the explainer benchmark.
        task : str
            "classification" or "regression"
        """
        with open(json_path) as f:
            full = json.load(f)

        if task not in full:
            raise ValueError(
                f"Task '{task}' not found in JSON. "
                f"Top-level keys: {list(full.keys())}"
            )

        # NEW: extract only the task-specific block
        self.results = full[task]

        self.task = task
        self._df = _flatten(self.results, task)
        self._hits = _hit_counts(self.results)

        self.datasets = sorted(self.results.keys())
        self.models = sorted({m for ds in self.results.values() for m in ds})
        self.dist_methods = sorted(
            {d for ds in self.results.values() for ms in ds.values() for d in ms}
        )

    # ────────────────────────────────────────────────────────────────────────
    # 1. Hit-rate heatmap
    # ────────────────────────────────────────────────────────────────────────

    def plot_hit_rate_heatmap(self, agg="sum", figsize=None):
        pivot = (
            self._hits.groupby(["model", "dist_method"])["n_hits"]
            .agg(agg)
            .unstack("dist_method")
            .reindex(columns=self.dist_methods)
        ).fillna(0)

        fig, ax = plt.subplots(
            figsize=figsize or (7, max(3, len(self.models) * 0.55 + 1))
        )
        cmap = LinearSegmentedColormap.from_list("blue_white", [LIGHT, BLUE])
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=f"{agg} hits")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > pivot.values.max() * 0.6 else SLATE,
                )

        ax.set_title(f"[{self.task.upper()}] Hits by Model × Distance Method ({agg})")
        ax.set_xlabel("Distance method")
        ax.set_ylabel("Model")
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # 2. Top features
    # ────────────────────────────────────────────────────────────────────────

    def plot_top_features(
        self, dataset=None, model=None, dist_method=None, top_n=15, figsize=None
    ):

        sub = self._df.copy()
        if dataset:
            sub = sub[sub["dataset"] == dataset]
        if model:
            sub = sub[sub["model"] == model]
        if dist_method:
            sub = sub[sub["dist_method"] == dist_method]

        counts = _feature_counts(sub, self.task)
        if not counts:
            print("No data for the given filters.")
            return

        top = counts.most_common(top_n)
        feats, vals = zip(*top)

        fig, ax = plt.subplots(figsize=figsize or (7, max(3, top_n * 0.38 + 1)))
        bars = ax.barh(
            [f"feature {f}" for f in feats],
            vals,
            color=BLUE,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Times perturbed")

        title_parts = [f"[{self.task.upper()}] Top features"]
        if dataset:
            title_parts.append(f"ds={dataset}")
        if model:
            title_parts.append(f"model={model}")
        if dist_method:
            title_parts.append(f"dist={dist_method}")
        ax.set_title(" | ".join(title_parts))

        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # 3. Group-size sensitivity
    # ────────────────────────────────────────────────────────────────────────

    def plot_group_size_sensitivity(self, figsize=None):
        if self._df.empty:
            print("No data.")
            return

        agg = (
            self._df.groupby(["group_size", "dist_method"])
            .size()
            .reset_index(name="count")
        )
        sizes = sorted(agg["group_size"].unique())
        methods = self.dist_methods
        x = np.arange(len(sizes))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        for i, dist in enumerate(methods):
            sub = agg[agg["dist_method"] == dist].set_index("group_size")
            vals = [sub.loc[s, "count"] if s in sub.index else 0 for s in sizes]
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width * 0.9,
                label=dist,
                color=DIST_COLORS.get(dist, SLATE),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"size {s}" for s in sizes])
        ax.set_xlabel("Feature group size")
        ax.set_ylabel("Threshold-crossing perturbations")
        ax.set_title(f"[{self.task.upper()}] Sensitivity by Group Size")
        ax.legend(title="Distance")
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # 4. Hits per dataset
    # ────────────────────────────────────────────────────────────────────────

    def plot_hits_per_dataset(self, model=None, figsize=None):
        sub = self._hits.copy()
        if model:
            sub = sub[sub["model"] == model]

        pivot = (
            sub.groupby(["dataset", "dist_method"])["n_hits"]
            .sum()
            .unstack("dist_method")
            .fillna(0)
            .reindex(columns=self.dist_methods)
        )

        fig, ax = plt.subplots(figsize=figsize or (max(6, len(pivot) * 1.0), 5))
        bottom = np.zeros(len(pivot))
        for dist in self.dist_methods:
            vals = pivot[dist].values
            ax.bar(
                pivot.index,
                vals,
                bottom=bottom,
                label=dist,
                color=DIST_COLORS.get(dist, SLATE),
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += vals

        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_ylabel("Total hits")
        title = f"[{self.task.upper()}] Hits per Dataset"
        if model:
            title += f" — {model}"
        ax.set_title(title)
        ax.legend(title="Distance", bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # 5. Model comparison
    # ────────────────────────────────────────────────────────────────────────

    def plot_model_comparison(self, dataset=None, figsize=None):
        sub = self._hits.copy()
        if dataset:
            sub = sub[sub["dataset"] == dataset]

        pivot = (
            sub.groupby(["model", "dist_method"])["n_hits"]
            .sum()
            .unstack("dist_method")
            .fillna(0)
            .reindex(columns=self.dist_methods)
        )

        models = list(pivot.index)
        x = np.arange(len(models))
        width = 0.8 / len(self.dist_methods)

        fig, ax = plt.subplots(figsize=figsize or (max(7, len(models) * 1.1), 5))
        for i, dist in enumerate(self.dist_methods):
            offset = (i - len(self.dist_methods) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                pivot[dist].values,
                width=width * 0.9,
                label=dist,
                color=DIST_COLORS.get(dist, SLATE),
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Total hits")
        title = f"[{self.task.upper()}] Model Comparison"
        if dataset:
            title += f" — {dataset}"
        ax.set_title(title)
        ax.legend(title="Distance")
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # 6. Perturbation magnitude distribution
    # ────────────────────────────────────────────────────────────────────────

    def plot_delta_distribution(
        self, dataset=None, model=None, top_features=8, figsize=None
    ):

        sub = self._df.copy()
        if dataset:
            sub = sub[sub["dataset"] == dataset]
        if model:
            sub = sub[sub["model"] == model]

        mag = _move_magnitudes(sub, self.task)
        if mag.empty:
            print("No continuous-feature deltas found.")
            return

        top_feats = (
            mag.groupby("feature")["delta"]
            .count()
            .nlargest(top_features)
            .index.tolist()
        )
        mag = mag[mag["feature"].isin(top_feats)]

        methods = self.dist_methods
        n_feats = len(top_feats)
        fig, axes = plt.subplots(
            1, n_feats, figsize=figsize or (max(8, n_feats * 1.8), 5), sharey=False
        )
        if n_feats == 1:
            axes = [axes]

        for ax, feat in zip(axes, top_feats):
            data = [
                mag[(mag["feature"] == feat) & (mag["dist_method"] == d)][
                    "delta"
                ].values
                for d in methods
            ]
            bp = ax.boxplot(
                data, patch_artist=True, medianprops={"color": "white", "linewidth": 2}
            )
            for patch, dist in zip(bp["boxes"], methods):
                patch.set_facecolor(DIST_COLORS.get(dist, SLATE))
                patch.set_alpha(0.85)
            ax.set_title(f"f{feat}", fontsize=9)
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels([d[:4] for d in methods], fontsize=7, rotation=30)
            ax.set_ylabel("|Δ|" if feat == top_feats[0] else "")

        title = f"[{self.task.upper()}] Perturbation Magnitude |Δ| per Feature"
        if dataset:
            title += f" — {dataset}"
        if model:
            title += f" / {model}"
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # REGRESSION-SPECIFIC
    # ────────────────────────────────────────────────────────────────────────

    def plot_change_distribution(self, dataset=None, model=None, figsize=None):
        if self.task != "regression":
            raise ValueError("plot_change_distribution is for regression only.")

        sub = self._df.copy()
        if dataset:
            sub = sub[sub["dataset"] == dataset]
        if model:
            sub = sub[sub["model"] == model]

        if sub.empty:
            print("No data.")
            return

        methods = self.dist_methods
        fig, axes = plt.subplots(
            1, len(methods), figsize=figsize or (4 * len(methods), 4), sharey=True
        )
        if len(methods) == 1:
            axes = [axes]

        for ax, dist in zip(axes, methods):
            d = sub[sub["dist_method"] == dist]["change"].dropna()
            if d.empty:
                ax.set_title(dist)
                continue
            inc = d[d >= 0]
            dec = d[d < 0]
            bins = np.linspace(d.min(), d.max(), 30)
            ax.hist(inc, bins=bins, color=TEAL, alpha=0.75, label="increase")
            ax.hist(dec, bins=bins, color=RED, alpha=0.75, label="decrease")
            ax.axvline(0, color=SLATE, lw=1, ls="--")
            ax.set_title(dist)
            ax.set_xlabel("Δŷ")
            if ax is axes[0]:
                ax.set_ylabel("Count")
                ax.legend()

        title = "[REGRESSION] Prediction Change Distribution"
        if dataset:
            title += f" — {dataset}"
        if model:
            title += f" / {model}"
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_change_vs_group_size(self, dataset=None, figsize=None):
        if self.task != "regression":
            raise ValueError("regression only.")

        sub = self._df.copy()
        if dataset:
            sub = sub[sub["dataset"] == dataset]

        if sub.empty:
            print("No data.")
            return

        sub = sub.dropna(subset=["change"])

        sub["abs_change"] = sub["change"].abs()

        fig, ax = plt.subplots(figsize=figsize or (7, 5))
        for dist in self.dist_methods:
            d = sub[sub["dist_method"] == dist]
            jitter = np.random.uniform(-0.15, 0.15, len(d))
            ax.scatter(
                d["group_size"] + jitter,
                d["abs_change"],
                alpha=0.25,
                s=12,
                color=DIST_COLORS.get(dist, SLATE),
                label=dist,
            )

        ax.set_xlabel("Feature group size")
        ax.set_ylabel("|Δŷ|")
        ax.set_xticks(sorted(sub["group_size"].unique()))
        title = "[REGRESSION] |Δŷ| vs Feature Group Size"
        if dataset:
            title += f" — {dataset}"
        ax.set_title(title)
        ax.legend(title="Distance", markerscale=2)
        fig.tight_layout()
        return fig

    def plot_increase_vs_decrease_rate(self, figsize=None):
        """
        Side-by-side bars: share of threshold_met_increase vs threshold_met_decrease
        per dataset × model.
        Regression only.
        """
        if self.task != "regression":
            raise ValueError("regression only.")

        rows = []
        for ds, models in self.results.items():
            for model, dists in models.items():
                for dist, records in dists.items():
                    if not isinstance(records, list):
                        continue
                    n_inc = sum(1 for r in records if r.get("threshold_met_increase"))
                    n_dec = sum(1 for r in records if r.get("threshold_met_decrease"))
                    rows.append(
                        {
                            "dataset": ds,
                            "model": model,
                            "dist_method": dist,
                            "n_increase": n_inc,
                            "n_decrease": n_dec,
                        }
                    )

        df = pd.DataFrame(rows)
        agg = df.groupby("dataset")[["n_increase", "n_decrease"]].sum()

        x = np.arange(len(agg))
        w = 0.38
        fig, ax = plt.subplots(figsize=figsize or (max(6, len(agg) * 1.1), 5))
        ax.bar(x - w / 2, agg["n_increase"], w, color=TEAL, label="Increase ≥ σ")
        ax.bar(x + w / 2, agg["n_decrease"], w, color=RED, label="Decrease ≤ -σ")

        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=35, ha="right")
        ax.set_ylabel("Hit count")
        ax.set_title("[REGRESSION] Threshold-Met Increase vs Decrease by Dataset")
        ax.legend()
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION-SPECIFIC
    # ────────────────────────────────────────────────────────────────────────

    def plot_class_transition_matrix(
        self, dataset, model, dist_method=None, figsize=None
    ):
        if self.task != "classification":
            raise ValueError("classification only.")

        sub = self._df[(self._df["dataset"] == dataset) & (self._df["model"] == model)]
        if dist_method:
            sub = sub[sub["dist_method"] == dist_method]

        if sub.empty:
            print("No data for the given filters.")
            return

        classes = sorted(
            set(sub["original_class"].unique()) | set(sub["new_class"].unique())
        )
        mat = pd.DataFrame(0, index=classes, columns=classes)
        for _, row in sub.iterrows():
            mat.loc[row["original_class"], row["new_class"]] += 1

        fig, ax = plt.subplots(figsize=figsize or (5, 4))
        cmap = LinearSegmentedColormap.from_list("white_blue", ["white", BLUE])
        im = ax.imshow(mat.values, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="Count")

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"class {c}" for c in classes])
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([f"class {c}" for c in classes])
        ax.set_xlabel("Predicted after perturbation")
        ax.set_ylabel("Original class")

        title = f"[CLF] Class Transitions — {dataset} / {model}"
        if dist_method:
            title += f" / {dist_method}"
        ax.set_title(title)

        for i in range(len(classes)):
            for j in range(len(classes)):
                v = mat.values[i, j]
                if v > 0:
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if v > mat.values.max() * 0.6 else SLATE,
                    )

        fig.tight_layout()
        return fig

    def plot_direction_breakdown(self, dataset=None, figsize=None):
        if self.task != "classification":
            raise ValueError("classification only.")

        sub = self._df.copy()
        if dataset:
            sub = sub[sub["dataset"] == dataset]

        agg = (
            sub.groupby(["dist_method", "direction"])
            .size()
            .unstack("direction", fill_value=0)
        )

        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        bottom = np.zeros(len(agg))
        dir_colors = {"plus": TEAL, "minus": AMBER}

        for direction in agg.columns:
            vals = agg[direction].values
            ax.bar(
                agg.index,
                vals,
                bottom=bottom,
                label=direction,
                color=dir_colors.get(direction, SLATE),
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += vals

        ax.set_xlabel("Distance method")
        ax.set_ylabel("Hits")
        title = "[CLF] Perturbation Direction — plus vs minus"
        if dataset:
            title += f" — {dataset}"
        ax.set_title(title)
        ax.legend(title="Direction")
        fig.tight_layout()
        return fig

    # ────────────────────────────────────────────────────────────────────────
    # Convenience: plot everything
    # ────────────────────────────────────────────────────────────────────────

    def plot_all(self, save_dir=None):
        import os

        figs = {}

        figs["hit_rate_heatmap"] = self.plot_hit_rate_heatmap()
        figs["group_size_sensitivity"] = self.plot_group_size_sensitivity()
        figs["hits_per_dataset"] = self.plot_hits_per_dataset()
        figs["model_comparison"] = self.plot_model_comparison()

        first_ds = self.datasets[0]
        first_model = self.models[0]

        figs["top_features"] = self.plot_top_features(
            dataset=first_ds, model=first_model
        )
        figs["delta_dist"] = self.plot_delta_distribution(
            dataset=first_ds, model=first_model
        )

        if self.task == "regression":
            figs["change_dist"] = self.plot_change_distribution(
                dataset=first_ds, model=first_model
            )
            figs["change_vs_group"] = self.plot_change_vs_group_size(dataset=first_ds)
            figs["increase_vs_decrease"] = self.plot_increase_vs_decrease_rate()

        if self.task == "classification":
            figs["class_transitions"] = self.plot_class_transition_matrix(
                first_ds, first_model
            )
            figs["direction_breakdown"] = self.plot_direction_breakdown()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figs.items():
                if fig is not None:
                    path = os.path.join(save_dir, f"{self.task}_{name}.png")
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    print(f"Saved {path}")

        return figs


# ─────────────────────────────────────────────────────────────────────────────
# DEMO ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    json_path = "benchmark_results.json"

    print("=== Classification ===")
    clf = ExplainerPlotter(json_path, task="classification")
    clf_figs = clf.plot_all(save_dir="plots_classification")

    print("\n=== Regression ===")
    reg = ExplainerPlotter(json_path, task="regression")
    reg_figs = reg.plot_all(save_dir="plots_regression")
