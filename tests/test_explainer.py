import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dise import DirectionalSensitivityExplainer

np.random.seed(42)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_data():
    X, y = load_iris(return_X_y=True)
    m = RandomForestClassifier(n_estimators=20, random_state=42)
    m.fit(X, y)
    return X, y, m

@pytest.fixture(scope="module")
def reg_data():
    X, y = load_diabetes(return_X_y=True)
    m = RandomForestRegressor(n_estimators=20, random_state=42)
    m.fit(X, y)
    return X, y, m

@pytest.fixture(scope="module")
def mixed_data():
    """Dummy-encoded dataset simulating get_dummies().values (object array)."""
    n = 500
    df = pd.DataFrame({
        "age":      np.random.randint(18, 80, n),
        "income":   np.random.normal(55000, 12000, n),
        "contract": np.random.choice(["month-to-month", "one-year", "two-year"], n),
        "churn":    np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    X = pd.get_dummies(df.drop("churn", axis=1), columns=["contract"]).values
    y = df["churn"].values
    m = DecisionTreeClassifier(random_state=42)
    m.fit(X, y)
    return X, y, m


# ── Return type and columns ───────────────────────────────────────────────────

def test_clf_returns_dataframe(clf_data):
    X, y, model = clf_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="percentile")
    df  = exp.run(n=3)
    assert isinstance(df, pd.DataFrame)

def test_clf_columns_present(clf_data):
    X, y, model = clf_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="percentile")
    df  = exp.run(n=3)
    expected = {"sample_idx", "feature_group", "move_values",
                "original_class", "new_class", "changed", "direction"}
    assert expected.issubset(set(df.columns))

def test_reg_returns_dataframe(reg_data):
    X, y, model = reg_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="percentile")
    df  = exp.run(n=3)
    assert isinstance(df, pd.DataFrame)

def test_reg_columns_present(reg_data):
    X, y, model = reg_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="percentile")
    df  = exp.run(n=3)
    expected = {"sample_idx", "feature_group", "move_values",
                "original_y", "new_y", "change",
                "threshold_met_increase", "threshold_met_decrease", "direction"}
    assert expected.issubset(set(df.columns))


# ── Empty result still has correct columns ────────────────────────────────────

def test_empty_result_has_columns(reg_data):
    """If no TCEs are found, the DataFrame should still have all columns."""
    X, y, model = reg_data
    exp = DirectionalSensitivityExplainer(model, X, y,
                                          distance_method="counterfactual",
                                          scale=0.001)
    df = exp.run(n=2)
    expected_cols = {"sample_idx", "feature_group", "move_values",
                     "original_y", "new_y", "change",
                     "threshold_met_increase", "threshold_met_decrease", "direction"}
    assert expected_cols.issubset(set(df.columns))


# ── All distance methods run without error ────────────────────────────────────

@pytest.mark.parametrize("dist", ["percentile", "wasserstein", "mahalanobis", "counterfactual"])
def test_all_dist_methods_clf(clf_data, dist):
    X, y, model = clf_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method=dist)
    df  = exp.run(n=3)
    assert isinstance(df, pd.DataFrame)

@pytest.mark.parametrize("dist", ["percentile", "wasserstein", "mahalanobis", "counterfactual"])
def test_all_dist_methods_reg(reg_data, dist):
    X, y, model = reg_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method=dist)
    df  = exp.run(n=3)
    assert isinstance(df, pd.DataFrame)


# ── Mixed (dummy-encoded) data ────────────────────────────────────────────────

@pytest.mark.parametrize("dist", ["percentile", "wasserstein", "mahalanobis", "counterfactual"])
def test_mixed_data_no_error(mixed_data, dist):
    X, y, model = mixed_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method=dist)
    df  = exp.run(n=3)
    assert isinstance(df, pd.DataFrame)


# ── Regression threshold flags are consistent ─────────────────────────────────

def test_reg_threshold_flags(reg_data):
    X, y, model = reg_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="wasserstein")
    df  = exp.run(n=3)
    if not df.empty:
        # Every row should meet at least one threshold
        assert (df["threshold_met_increase"] | df["threshold_met_decrease"]).all()

def test_reg_change_sign_consistent(reg_data):
    X, y, model = reg_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="percentile")
    df  = exp.run(n=3)
    if not df.empty:
        assert (df.loc[df["threshold_met_increase"], "change"] >= 0).all()
        assert (df.loc[df["threshold_met_decrease"], "change"] <= 0).all()


# ── Classification changed flag is correct ────────────────────────────────────

def test_clf_changed_flag(clf_data):
    X, y, model = clf_data
    exp = DirectionalSensitivityExplainer(model, X, y, distance_method="percentile")
    df  = exp.run(n=3)
    if not df.empty:
        assert (df["original_class"] != df["new_class"]).all()
        assert df["changed"].all()


# ── Invalid distance method raises ───────────────────────────────────────────

def test_invalid_distance_raises(reg_data):
    X, y, model = reg_data
    with pytest.raises(ValueError, match="Unknown distance_method"):
        exp = DirectionalSensitivityExplainer(model, X, y, distance_method="bogus")
        exp.run()
