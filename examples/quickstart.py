"""
Quickstart examples for DISE.
Run from the repo root: python examples/quickstart.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dise import DirectionalSensitivityExplainer


# ── Classification ────────────────────────────────────────────────────────────

print("=== Classification (Iris) ===")
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

for dist in ["percentile", "wasserstein", "mahalanobis", "counterfactual"]:
    exp = DirectionalSensitivityExplainer(clf, X, y, distance_method=dist, scale=0.5)
    df  = exp.run(n=3)
    print(f"  {dist:15s}: {len(df)} threshold-crossing events")


# ── Regression ────────────────────────────────────────────────────────────────

print("\n=== Regression (Diabetes) ===")
X, y = load_diabetes(return_X_y=True)
reg  = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)

for dist in ["percentile", "wasserstein", "mahalanobis", "counterfactual"]:
    exp = DirectionalSensitivityExplainer(reg, X, y, distance_method=dist, scale=0.5)
    df  = exp.run(n=3)
    print(f"  {dist:15s}: {len(df)} threshold-crossing events")


# ── Mixed (dummy-encoded) ─────────────────────────────────────────────────────

print("\n=== Mixed categorical data ===")
np.random.seed(42)
n = 500
df_raw = pd.DataFrame({
    "age":      np.random.randint(18, 75, n),
    "income":   np.random.lognormal(10.8, 0.5, n),
    "contract": np.random.choice(["month-to-month", "one-year", "two-year"], n),
    "churn":    np.random.choice([0, 1], n, p=[0.7, 0.3]),
})
X_enc = pd.get_dummies(df_raw.drop("churn", axis=1), columns=["contract"]).values
y_enc = df_raw["churn"].values

from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier(random_state=42).fit(X_enc, y_enc)
exp  = DirectionalSensitivityExplainer(clf2, X_enc, y_enc,
                                        distance_method="wasserstein", scale=0.5)
df_out = exp.run(n=3)
print(f"  wasserstein: {len(df_out)} threshold-crossing events")
if not df_out.empty:
    print(df_out[["feature_group", "original_class", "new_class", "direction"]].head())
