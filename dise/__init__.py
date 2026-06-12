"""
DISE — Directional Sensitivity Explainer

A model-agnostic, interpretability framework that identifies
the feature-group perturbations sufficient to change a model's
prediction by a practically meaningful amount.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dise")
except PackageNotFoundError:  # package not installed (e.g. in dev from source)
    __version__ = "0.0.0.dev"

from .explainer import DirectionalSensitivityExplainer
from .detectors import column_check
from .distances import (
    percentile_difference,
    wasserstein_vector,
    mahalanobis_delta,
    minimal_counterfactual,
)

__all__ = [
    "DirectionalSensitivityExplainer",
    "column_check",
    "percentile_difference",
    "wasserstein_vector",
    "mahalanobis_delta",
    "minimal_counterfactual",
    "__version__",
]
