# DISE — Directional Sensitivity Explainer

[![PyPI version](https://img.shields.io/pypi/v/dise.svg)](https://pypi.org/project/dise/)
[![Python](https://img.shields.io/pypi/pyversions/dise.svg)](https://pypi.org/project/dise/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A model-agnostic, post-hoc interpretability framework that identifies
which feature groups, when perturbed at data-driven scales, produce
practically meaningful changes in a model's predictions.

## Installation

```bash
pip install dise                  # core only
pip install "dise[plotting]"      # + matplotlib visualisations
pip install "dise[xgboost]"       # + XGBoost support
pip install "dise[all]"           # everything + dev tools
```

## Quick start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from dise import DirectionalSensitivityExplainer

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier().fit(X, y)

exp = DirectionalSensitivityExplainer(
    model, X, y,
    distance_method="wasserstein",  # "percentile" | "wasserstein" | "mahalanobis" | "counterfactual"
    scale=0.5,
)
results = exp.run(n=3)             # top-3 features, all non-empty subsets
print(results.head())
```

## Visualisation

```python
from dise.plotting import ExplainerPlotter

plotter = ExplainerPlotter("results.json", task="classification")
plotter.plot_hit_rate_heatmap()
plotter.plot_top_features(dataset="iris", model="random_forest")
plotter.plot_all(save_dir="plots/")
```

## Distance methods

| Method | Description | Best for |
|---|---|---|
| `percentile` | IQR-based, fast, non-parametric | General use |
| `wasserstein` | Earth mover's distance | Skewed / multimodal distributions |
| `mahalanobis` | Covariance-aware, decorrelates features | Correlated features |
| `counterfactual` | Nearest-neighbour distance between groups | Interpretable scales |

## Supported task types

- **Classification** — threshold: predicted class changes
- **Regression** — threshold: |Δŷ| ≥ σ_y

## Feature types

Mixed numeric/categorical data is handled automatically.
Pass dummy-encoded matrices directly from `pd.get_dummies(...).values`.

## Running tests

```bash
pip install "dise[dev]"
pytest
```

## Citation

If you use DISE in research, please cite:

```bibtex
@misc{dise2025,
  title  = {Directional Sensitivity Explainability: A Feature-Group
             Perturbation Framework for Supervised Machine Learning Models},
  author = {Melvin Dunn},
  year   = {2025},
  url    = {https://github.com/medu-colorado/dise},
}
```
