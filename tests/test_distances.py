import numpy as np
import pytest
from dise.distances import (
    percentile_difference,
    wasserstein_vector,
    mahalanobis_delta,
    minimal_counterfactual,
)

np.random.seed(42)
A = np.random.normal(0, 1, (100, 4))
B = np.random.normal(2, 1, (100, 4))   # clearly separated from A


def test_percentile_regression_shape():
    d = percentile_difference(A)
    assert d.shape == (4,)
    assert np.all(d >= 0)

def test_percentile_classification_shape():
    d = percentile_difference(A, B)
    assert d.shape == (4,)
    assert np.all(d >= 0)

def test_percentile_separated_classes():
    # B is shifted by 2: P75(A)~0.67, P75(B)~2.67 → diff ~2
    d = percentile_difference(A, B)
    assert np.all(d > 1.0)

def test_wasserstein_regression_shape():
    d = wasserstein_vector(A)
    assert d.shape == (4,)
    assert np.all(d >= 0)

def test_wasserstein_classification_shape():
    d = wasserstein_vector(A, B)
    assert d.shape == (4,)

def test_wasserstein_zero_for_identical():
    d = wasserstein_vector(A, A)
    assert np.allclose(d, 0, atol=1e-10)

def test_mahalanobis_regression_shape():
    d = mahalanobis_delta(A)
    assert d.shape == (4,)
    assert np.all(d >= 0)

def test_mahalanobis_with_type_map():
    type_map = {0: "discrete", 1: "continuous", 2: "continuous", 3: "discrete"}
    d = mahalanobis_delta(A, type_map=type_map)
    assert d[0] == 0.0
    assert d[3] == 0.0
    assert d[1] > 0
    assert d[2] > 0

def test_mahalanobis_single_continuous_col():
    """Should not crash when only one continuous column."""
    type_map = {0: "discrete", 1: "continuous", 2: "discrete", 3: "discrete"}
    d = mahalanobis_delta(A, type_map=type_map)
    assert d.shape == (4,)

def test_mahalanobis_object_array():
    """Object arrays (from get_dummies) must be castable to float."""
    A_obj = A.astype(object)
    d = mahalanobis_delta(A_obj)
    assert d.shape == (4,)

def test_counterfactual_shape():
    y = np.random.normal(0, 1, 100)
    d = minimal_counterfactual(A, y=y)
    assert d.shape == (4,)
    assert np.all(d >= 0)

def test_counterfactual_classification():
    d = minimal_counterfactual(A, B=B)
    assert d.shape == (4,)
