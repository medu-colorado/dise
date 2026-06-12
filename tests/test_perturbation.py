import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dise.perturbation import move_group_classification, move_group_regression

np.random.seed(42)
X = np.random.normal(0, 1, (100, 4))
y_clf = (X[:, 0] > 0).astype(int)
y_reg = X[:, 0] * 2 + X[:, 1]

clf = DecisionTreeClassifier(random_state=42).fit(X, y_clf)
reg = DecisionTreeRegressor(random_state=42).fit(X, y_reg)

type_map  = {0: "continuous", 1: "continuous", 2: "discrete", 3: "continuous"}
dist_vals = np.array([0.5, 0.5, 0.0, 0.5])
unpack    = lambda v: v.item() if hasattr(v, "item") else v


def test_classification_result_keys():
    sample = X[0:1].copy()
    result = move_group_classification(
        clf, sample, [0, 1], dist_vals, type_map,
        int(y_clf[0]), 0, "plus", 1.0, X, unpack
    )
    expected = {"sample_idx", "feature_group", "move_values",
                "original_class", "new_class", "changed", "direction"}
    assert expected == set(result.keys())

def test_regression_result_keys():
    sample = X[0:1].copy()
    result = move_group_regression(
        reg, sample, [0, 1], dist_vals, type_map,
        0, "plus", 1.0, X, np.std(y_reg), unpack
    )
    expected = {"sample_idx", "feature_group", "move_values",
                "original_y", "new_y", "change",
                "threshold_met_increase", "threshold_met_decrease", "direction"}
    assert expected == set(result.keys())

def test_continuous_feature_perturbed():
    sample = X[0:1].copy()
    original_val = float(sample[0, 0])
    result = move_group_classification(
        clf, sample, [0], dist_vals, type_map,
        int(y_clf[0]), 0, "plus", 1.0, X, unpack
    )
    new_val = result["move_values"][0][2]
    assert new_val != original_val

def test_minus_direction_decreases_value():
    sample = X[5:6].copy()
    original_val = float(sample[0, 0])
    result = move_group_regression(
        reg, sample, [0], dist_vals, type_map,
        5, "minus", 1.0, X, np.std(y_reg), unpack
    )
    new_val = result["move_values"][0][2]
    assert new_val < original_val

def test_discrete_feature_swapped():
    # Create dataset where col 2 is genuinely discrete
    X_d = X.copy()
    X_d[:, 2] = np.random.choice([10, 20, 30], 100)
    sample = X_d[0:1].copy()
    original_val = sample[0, 2]
    result = move_group_regression(
        reg, sample, [2], dist_vals, type_map,
        0, "plus", 1.0, X_d, np.std(y_reg), unpack
    )
    new_val = result["move_values"][2][2]
    assert new_val != original_val or result["move_values"][2][0] == "discrete_no_alternative"
