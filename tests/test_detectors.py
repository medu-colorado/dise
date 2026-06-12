import numpy as np
import pandas as pd
import pytest
from dise.detectors import column_check


# ── Continuous ────────────────────────────────────────────────────────────────

def test_float_continuous():
    assert column_check(np.random.normal(0, 1, 500)) == "continuous"

def test_integer_high_cardinality():
    assert column_check(np.random.randint(18, 80, 1000)) == "continuous"

def test_lognormal_continuous():
    assert column_check(np.random.lognormal(10, 0.5, 500)) == "continuous"


# ── Discrete ──────────────────────────────────────────────────────────────────

def test_binary_integer():
    assert column_check(np.random.choice([0, 1], 500)) == "discrete"

def test_binary_float():
    assert column_check(np.array([0.0, 1.0] * 250)) == "discrete"

def test_low_cardinality_integer():
    assert column_check(np.random.choice([1, 2, 3, 4, 5], 500)) == "discrete"

def test_string_column():
    assert column_check(np.array(["M", "F"] * 250)) == "discrete"

def test_boolean_column():
    assert column_check(np.array([True, False] * 250)) == "discrete"

def test_constant_column():
    assert column_check(np.ones(100)) == "discrete"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_nan_heavy_column():
    arr = np.array([np.nan] * 90 + [1.0, 2.0, 3.0, 4.0, 5.0,
                                     6.0, 7.0, 8.0, 9.0, 10.0])
    # After dropping NaN only 10 values remain → discrete
    assert column_check(arr) == "discrete"

def test_object_array_from_get_dummies():
    """Object arrays produced by pd.get_dummies().values should work."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "age":      np.random.randint(18, 80, n),
        "income":   np.random.normal(60000, 15000, n),
        "gender":   np.random.choice(["M", "F"], n),
        "contract": np.random.choice(["A", "B", "C"], n),
    })
    X = pd.get_dummies(df, columns=["gender", "contract"]).values
    assert X.dtype == object  # confirm we're testing the right thing

    assert column_check(X[:, 0]) == "continuous"   # age
    assert column_check(X[:, 1]) == "continuous"   # income
    assert column_check(X[:, 2]) == "discrete"     # gender_F (dummy)

def test_pandas_series():
    s = pd.Series(np.random.normal(0, 1, 300))
    assert column_check(s) == "continuous"
