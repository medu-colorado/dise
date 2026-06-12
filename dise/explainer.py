import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .detectors import column_check
from .distances import (
    percentile_difference,
    wasserstein_vector,
    mahalanobis_delta,
    minimal_counterfactual,
)
from .perturbation import move_group_classification, move_group_regression


class DirectionalSensitivityExplainer:
    """
    Model-agnostic, post-hoc interpretability framework that identifies
    the minimal feature-group perturbations sufficient to change a model's
    prediction by a practically meaningful amount.

    Parameters
    ----------
    model : object
        Trained model with a ``predict()`` method (scikit-learn compatible).
    X : array-like, shape (n, p)
        Feature matrix.
    y : array-like, shape (n,)
        Target vector.
    distance_method : {"percentile", "wasserstein", "mahalanobis", "counterfactual"}
        Method used to determine per-feature perturbation magnitudes.
    scale : float, default 0.5
        Multiplier applied to continuous perturbation magnitudes.
        Has no effect on discrete features.
    """

    def __init__(self, model, X, y, distance_method="percentile", scale=1.0):
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y)
        self.distance_method = distance_method
        self.scale = scale

        self.is_classification = len(np.unique(self.y)) < 0.2 * len(
            self.y
        ) and np.allclose(self.y, self.y.astype(int))
        self.is_regression = not self.is_classification

        if self.is_classification:
            self.classes = np.unique(self.y)
        else:
            self.y_std = np.std(self.y)

    def _unpack(self, value):
        """
        Unpacks a value if it's a numpy array.

        Parameters
        ----------
        value:
            An iterable or scalar

        Returns
        -------

        A singular value.
            
        """
        if isinstance(value, np.ndarray):
            return value.item() if value.size == 1 else value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value[0] if len(value) == 1 else value
        return value

    def _build_type_map(self, feature_indices):
        """
        Iterates through the columns in X feature space check if
        the columns are continuous or discrete.

        Parameters
        ----------
        feature_indicies:
            The top features that are being considered.

        Returns
        -------
        A dictionary of the column index (key) and whether or not
        the column is discrete or continuous.
        """
        return {i: column_check(self.X[:, i]) for i in feature_indices}

    def _discrete_alternatives(self, col_idx, current_val):
        """
        Gets all values aside from the current value.

        Parameters
        ----------
        col_idx:
            The column index
        current_val:
            The current discrete value
        
        Returns
        -------
        All values that are not the current value.
        """
        all_vals = np.unique(self.X[:, col_idx])
        return all_vals[all_vals != current_val]

    def get_feature_importance(self, n=3):
        """
        Return the indices of the top-n most important features.

        Uses model coefficients, tree importances, or permutation importance,
        in that order.

        Parameters
        ----------
        n: the number of features to consider

        Returns
        -------
        The top features as a numpy array
        """
        model = self.model
        X, y = self.X, self.y

        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_).sum(axis=0)
            return np.argsort(importance)[-n:][::-1]

        if hasattr(model, "feature_importances_"):
            return np.argsort(model.feature_importances_)[-n:][::-1]

        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        return np.argsort(perm.importances_mean)[-n:][::-1]

    def compute_distance(self, A, B=None, type_map=None):
        """
        Dispatch to the chosen distance method and zero out discrete columns.

        Parameters
        ----------
        A: np.array
            the first feature matrix to consider

        B: np.array
            the second feature matrix to consider. Does not exist for regression.

        Returns
        -------
        The distances to be used the perturbations
        """
        if self.distance_method == "percentile":
            dist = percentile_difference(A, B)
        elif self.distance_method == "wasserstein":
            dist = wasserstein_vector(A, B)
        elif self.distance_method == "mahalanobis":
            dist = mahalanobis_delta(A, B, type_map=type_map)
        elif self.distance_method == "counterfactual":
            dist = minimal_counterfactual(A, B, self.y)
        else:
            raise ValueError(f"Unknown distance_method: {self.distance_method!r}")

        if type_map:
            for col_idx, col_type in type_map.items():
                if col_type == "discrete":
                    dist[col_idx] = 0.0

        return dist

    def run(self, n=3, sample_threshold=1000, all_combos=True):
        """
        Run the directional sensitivity analysis.

        Parameters
        ----------
        n : int
            Number of top features to include.
        sample_threshold : int
            Maximum rows to process (random subset if exceeded).
        all_combos : bool
            If True, test all non-empty subsets of the top-n features.
            If False, test only cumulative prefixes.

        Returns
        -------
        pd.DataFrame
            Threshold-crossing events that met the change criterion.
        """
        X, y = self.X, self.y

        if X.shape[0] > sample_threshold:
            idx = np.random.choice(len(X), size=sample_threshold, replace=False)
            X, y = X[idx], y[idx]

        self.X, self.y = X, y

        top_features = self.get_feature_importance(n=n)

        if all_combos:
            combos = [
                c
                for i in range(1, len(top_features) + 1)
                for c in itertools.combinations(top_features.tolist(), i)
            ]
            feature_groups = [np.array(c) for c in combos]
        else:
            feature_groups = [top_features[:i] for i in range(1, len(top_features) + 1)]

        type_map = self._build_type_map(top_features.tolist())
        results = []

        # Classification
        if self.is_classification:
            clf_cols = [
                "sample_idx",
                "feature_group",
                "move_values",
                "original_class",
                "new_class",
                "changed",
                "direction",
            ]
            for a in self.classes:
                for b in self.classes:
                    if a >= b:
                        continue
                    A = X[y == a]
                    B = X[y == b]
                    dist_vals = self.compute_distance(A, B, type_map=type_map)
                    sample_idx = np.random.choice(len(A))
                    sample = A[sample_idx : sample_idx + 1]

                    for group in feature_groups:
                        for direction in ("plus", "minus"):
                            results.append(
                                move_group_classification(
                                    self.model,
                                    sample.copy(),
                                    group,
                                    dist_vals,
                                    type_map,
                                    a,
                                    sample_idx,
                                    direction,
                                    self.scale,
                                    X,
                                    self._unpack,
                                )
                            )

            if not results:
                return pd.DataFrame(columns=clf_cols)
            df = pd.DataFrame(results)
            filtered = df[df["changed"]]
            return filtered if not filtered.empty else pd.DataFrame(columns=clf_cols)

        # Regression
        reg_cols = [
            "sample_idx",
            "feature_group",
            "move_values",
            "original_y",
            "new_y",
            "change",
            "threshold_met_increase",
            "threshold_met_decrease",
            "direction",
        ]
        dist_vals = self.compute_distance(X, type_map=type_map)

        for sample_idx in np.random.choice(len(X), size=min(50, len(X)), replace=False):
            sample = X[sample_idx : sample_idx + 1]
            for group in feature_groups:
                for direction in ("plus", "minus"):
                    results.append(
                        move_group_regression(
                            self.model,
                            sample.copy(),
                            group,
                            dist_vals,
                            type_map,
                            sample_idx,
                            direction,
                            self.scale,
                            X,
                            self.y_std,
                            self._unpack,
                        )
                    )

        if not results:
            return pd.DataFrame(columns=reg_cols)
        df = pd.DataFrame(results)
        filtered = df[df["threshold_met_increase"] | df["threshold_met_decrease"]]
        return filtered if not filtered.empty else pd.DataFrame(columns=reg_cols)
