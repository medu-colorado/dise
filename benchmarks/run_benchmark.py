"""
Benchmark for DirectionalSensitivityExplainer across:
  - Classification and regression tasks
  - Diverse feature categories (pure numeric, mixed categorical, high cardinality,
    skewed/heavy-tailed, binary-heavy, ordinal, temporal)
  - Diverse empirical distributions (normal, log-normal, Pareto, Poisson,
    Bernoulli, uniform, bimodal, zero-inflated, exponential, beta)
"""

import numpy as np
import pandas as pd
import json
import traceback

from sklearn.datasets import (
    load_breast_cancer, load_iris, load_wine,
    load_diabetes, make_friedman1, make_friedman2
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from dise import DirectionalSensitivityExplainer


# ================================================================
# DISTRIBUTION HELPERS
# ================================================================

def normal(n, mu=0, sigma=1):
    return np.random.normal(mu, sigma, n)

def lognormal(n, mu=0, sigma=1):
    return np.random.lognormal(mu, sigma, n)

def pareto(n, alpha=2.0, scale=1.0):
    return (np.random.pareto(alpha, n) + 1) * scale

def poisson(n, lam=5):
    return np.random.poisson(lam, n).astype(float)

def bernoulli(n, p=0.5):
    return np.random.binomial(1, p, n).astype(float)

def uniform(n, low=0, high=1):
    return np.random.uniform(low, high, n)

def bimodal(n, mu1=-2, mu2=2, sigma=0.8):
    mask = np.random.rand(n) > 0.5
    return np.where(mask, np.random.normal(mu1, sigma, n), np.random.normal(mu2, sigma, n))

def zero_inflated(n, p_zero=0.6, lam=3):
    vals = np.random.poisson(lam, n).astype(float)
    vals[np.random.rand(n) < p_zero] = 0.0
    return vals

def exponential(n, scale=1.0):
    return np.random.exponential(scale, n)

def beta_dist(n, a=2, b=5):
    return np.random.beta(a, b, n)


# ================================================================
# CLASSIFICATION DATASETS
# ================================================================

def load_classification_datasets():
    np.random.seed(42)
    datasets = {}

    # --- Built-in sklearn datasets ---
    X, y = load_breast_cancer(return_X_y=True)
    datasets["breast_cancer"] = (X, y)

    X, y = load_iris(return_X_y=True)
    datasets["iris_multiclass"] = (X, y)

    X, y = load_wine(return_X_y=True)
    datasets["wine_multiclass"] = (X, y)

    # --- 1. Customer churn — mixed categorical, moderate imbalance (70/30) ---
    def make_churn(n=800):
        df = pd.DataFrame({
            "age":            np.random.randint(18, 75, n),          # uniform int
            "income":         lognormal(n, 10.8, 0.5),               # log-normal (income)
            "tenure_months":  poisson(n, lam=24),                    # Poisson count
            "monthly_charge": normal(n, 65, 20).clip(10),            # near-normal
            "num_products":   np.random.randint(1, 6, n).astype(float),
            "gender":         np.random.choice(["M", "F"], n),
            "contract":       np.random.choice(["month-to-month", "one-year", "two-year"], n,
                                               p=[0.55, 0.30, 0.15]),
            "internet":       np.random.choice(["DSL", "Fiber", "None"], n),
            "paperless":      bernoulli(n, p=0.6),
        })
        logit = (
            -2.5
            + 0.015 * (df["monthly_charge"] - 65)
            - 0.008 * df["tenure_months"]
            + 0.3 * (df["contract"] == "month-to-month").astype(float)
            - 0.2 * (df["internet"] == "None").astype(float)
            + np.random.normal(0, 0.3, n)
        )
        y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
        X = pd.get_dummies(df, columns=["gender", "contract", "internet"])
        return X.values, y

    datasets["churn_imbalanced"] = make_churn()

    # --- 2. Loan default — heavy tails, severe imbalance (90/10) ---
    def make_loan_default(n=1000):
        df = pd.DataFrame({
            "loan_amount":    pareto(n, alpha=2.5, scale=5000),      # heavy-tailed
            "annual_income":  lognormal(n, 10.9, 0.6),
            "credit_score":   normal(n, 680, 80).clip(300, 850),
            "debt_ratio":     beta_dist(n, a=2, b=5),                # beta [0,1]
            "num_late_pays":  zero_inflated(n, p_zero=0.7, lam=2),   # zero-inflated
            "employment":     np.random.choice(["Employed", "Self-employed", "Unemployed"], n,
                                               p=[0.65, 0.25, 0.10]),
            "loan_purpose":   np.random.choice(["Home", "Auto", "Education", "Personal"], n),
            "home_ownership": np.random.choice(["Own", "Rent", "Mortgage"], n),
        })
        logit = (
            -3.5
            - 0.003 * (df["credit_score"] - 680)
            + 2.0 * df["debt_ratio"]
            + 0.4 * df["num_late_pays"]
            + 0.8 * (df["employment"] == "Unemployed").astype(float)
            + np.random.normal(0, 0.2, n)
        )
        y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
        X = pd.get_dummies(df, columns=["employment", "loan_purpose", "home_ownership"])
        return X.values, y

    datasets["loan_default_severe_imbalance"] = make_loan_default()

    # --- 3. Medical diagnosis — bimodal features, multiclass (3) ---
    def make_medical(n=900):
        df = pd.DataFrame({
            "biomarker_a":   bimodal(n, mu1=1.2, mu2=4.5),          # bimodal
            "biomarker_b":   bimodal(n, mu1=-1.0, mu2=3.0),
            "age":           normal(n, 55, 12).clip(18, 95),
            "bmi":           normal(n, 27, 5).clip(15, 50),
            "systolic_bp":   normal(n, 125, 18).clip(80, 200),
            "cholesterol":   lognormal(n, 5.1, 0.2),
            "smoker":        bernoulli(n, p=0.25),
            "exercise_freq": np.random.choice(["Never", "Sometimes", "Regular"], n,
                                              p=[0.30, 0.45, 0.25]),
            "diet":          np.random.choice(["Poor", "Average", "Good"], n),
        })
        score = (
            0.5 * df["biomarker_a"]
            + 0.4 * df["biomarker_b"]
            + 0.02 * (df["age"] - 55)
            + 0.1 * df["bmi"]
            + 0.5 * df["smoker"]
            + np.random.normal(0, 0.5, n)
        )
        y = pd.cut(score, bins=3, labels=[0, 1, 2]).astype(int).values
        X = pd.get_dummies(df, columns=["exercise_freq", "diet"])
        return X.values, y

    datasets["medical_multiclass_bimodal"] = make_medical()

    # --- 4. Fraud detection — extreme imbalance (98/2), exponential features ---
    def make_fraud(n=2000):
        df = pd.DataFrame({
            "transaction_amount": exponential(n, scale=80),          # exponential
            "time_since_last":    exponential(n, scale=12),
            "num_transactions":   poisson(n, lam=8),
            "distance_from_home": pareto(n, alpha=3, scale=1),       # heavy-tailed
            "account_age_days":   np.random.randint(1, 3650, n).astype(float),
            "card_type":          np.random.choice(["Debit", "Credit", "Prepaid"], n,
                                                   p=[0.5, 0.4, 0.1]),
            "merchant_cat":       np.random.choice(["Retail", "Food", "Travel", "Online", "ATM"], n),
            "country":            np.random.choice(["Domestic", "Foreign"], n, p=[0.85, 0.15]),
        })
        logit = (
            -5.0
            + 0.005 * df["transaction_amount"]
            + 0.3 * df["distance_from_home"]
            + 0.8 * (df["country"] == "Foreign").astype(float)
            - 0.001 * df["account_age_days"]
            + np.random.normal(0, 0.3, n)
        )
        y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
        X = pd.get_dummies(df, columns=["card_type", "merchant_cat", "country"])
        return X.values, y

    datasets["fraud_extreme_imbalance"] = make_fraud()

    # --- 5. Employee attrition — ordinal features, zero-inflated, uniform ---
    def make_attrition(n=800):
        df = pd.DataFrame({
            "satisfaction":      uniform(n, 1, 10),                  # uniform [1,10]
            "years_at_company":  np.random.randint(0, 30, n).astype(float),
            "salary":            lognormal(n, 10.7, 0.4),
            "overtime_hours":    zero_inflated(n, p_zero=0.5, lam=10),
            "promotions":        poisson(n, lam=1.2),
            "distance_to_work":  exponential(n, scale=15),
            "department":        np.random.choice(["Engineering", "Sales", "HR", "Finance"], n),
            "job_level":         np.random.choice([1, 2, 3, 4, 5], n,
                                                  p=[0.15, 0.30, 0.30, 0.15, 0.10]),
            "travel_freq":       np.random.choice(["None", "Rarely", "Frequently"], n,
                                                  p=[0.40, 0.40, 0.20]),
        })
        logit = (
            -1.5
            - 0.15 * df["satisfaction"]
            - 0.04 * df["years_at_company"]
            + 0.02 * df["overtime_hours"]
            + 0.6 * (df["travel_freq"] == "Frequently").astype(float)
            + np.random.normal(0, 0.4, n)
        )
        y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
        X = pd.get_dummies(df, columns=["department", "travel_freq"])
        return X.values, y

    datasets["employee_attrition"] = make_attrition()

    return datasets


# ================================================================
# REGRESSION DATASETS
# ================================================================

def load_regression_datasets():
    np.random.seed(42)
    datasets = {}

    # --- Built-in sklearn datasets ---
    X, y = load_diabetes(return_X_y=True)
    datasets["diabetes"] = (X, y)

    X, y = make_friedman1(n_samples=1000, noise=0.1)
    datasets["friedman1"] = (X, y)

    X, y = make_friedman2(n_samples=1000, noise=0.1)
    datasets["friedman2"] = (X, y)

    # --- 1. House price — log-normal target, mixed distributions ---
    def make_house_price(n=1000):
        df = pd.DataFrame({
            "sqft":           pareto(n, alpha=3, scale=800),         # heavy-tailed size
            "bedrooms":       np.random.randint(1, 7, n).astype(float),
            "bathrooms":      np.random.randint(1, 5, n).astype(float),
            "house_age":      exponential(n, scale=20),              # exponential age
            "lot_size":       lognormal(n, 7.5, 0.8),                # log-normal lot
            "garage":         bernoulli(n, p=0.65),
            "pool":           bernoulli(n, p=0.15),
            "neighbourhood":  np.random.choice(["Downtown", "Suburbs", "Rural", "Waterfront"], n,
                                               p=[0.20, 0.50, 0.20, 0.10]),
            "condition":      np.random.choice(["Poor", "Fair", "Good", "Excellent"], n,
                                               p=[0.05, 0.20, 0.50, 0.25]),
            "style":          np.random.choice(["Ranch", "Colonial", "Contemporary", "Craftsman"], n),
        })
        nbhd = df["neighbourhood"].map({"Rural": 0, "Suburbs": 40000, "Downtown": 80000, "Waterfront": 160000})
        cond = df["condition"].map({"Poor": -25000, "Fair": 0, "Good": 25000, "Excellent": 60000})
        y = (
            90 * df["sqft"]
            + 12000 * df["bedrooms"]
            + 9000  * df["bathrooms"]
            - 600   * df["house_age"]
            + 10000 * df["garage"]
            + 20000 * df["pool"]
            + nbhd + cond
            + np.random.normal(0, 20000, n)
        ).clip(50000).values
        X = pd.get_dummies(df, columns=["neighbourhood", "condition", "style"])
        return X.values, y

    datasets["house_price_lognormal"] = make_house_price()

    # --- 2. Insurance premium — multiplicative structure, beta/Pareto features ---
    def make_insurance(n=1000):
        df = pd.DataFrame({
            "driver_age":      np.random.randint(17, 80, n).astype(float),
            "years_no_claim":  np.random.randint(0, 20, n).astype(float),
            "annual_mileage":  lognormal(n, 9.5, 0.4),              # log-normal mileage
            "credit_score":    normal(n, 680, 80).clip(300, 850),
            "claim_history":   zero_inflated(n, p_zero=0.6, lam=1.5),
            "vehicle_value":   pareto(n, alpha=2.5, scale=8000),    # heavy-tailed value
            "vehicle_type":    np.random.choice(["Hatchback", "Sedan", "SUV", "Sports"], n,
                                                p=[0.25, 0.35, 0.25, 0.15]),
            "coverage":        np.random.choice(["Basic", "Standard", "Comprehensive"], n,
                                                p=[0.20, 0.50, 0.30]),
            "region":          np.random.choice(["Urban", "Suburban", "Rural"], n,
                                                p=[0.35, 0.45, 0.20]),
        })
        veh_f  = df["vehicle_type"].map({"Hatchback": 0.8, "Sedan": 1.0, "SUV": 1.2, "Sports": 1.6})
        cov_f  = df["coverage"].map({"Basic": 0.7, "Standard": 1.0, "Comprehensive": 1.4})
        reg_f  = df["region"].map({"Rural": 0.85, "Suburban": 1.0, "Urban": 1.25})
        y = (
            (700
             + 12  * df["annual_mileage"] / 1000
             - 25  * df["years_no_claim"]
             + 150 * df["claim_history"]
             + np.where(df["driver_age"] < 25, 500, 0))
            * veh_f * cov_f * reg_f
            + np.random.normal(0, 80, n)
        ).clip(100).values
        X = pd.get_dummies(df, columns=["vehicle_type", "coverage", "region"])
        return X.values, y

    datasets["insurance_premium_multiplicative"] = make_insurance()

    # --- 3. Energy consumption — Poisson-like counts, seasonal (bimodal), skewed ---
    def make_energy(n=1000):
        df = pd.DataFrame({
            "floor_area":      lognormal(n, 5.0, 0.5),              # log-normal area
            "num_occupants":   poisson(n, lam=2.5),                  # Poisson count
            "insulation_score":beta_dist(n, a=3, b=2),              # beta quality
            "outside_temp":    bimodal(n, mu1=5, mu2=25, sigma=4),  # bimodal (winter/summer)
            "solar_panels":    bernoulli(n, p=0.20),
            "appliance_age":   exponential(n, scale=8),
            "building_type":   np.random.choice(["House", "Apartment", "Office", "Industrial"], n,
                                                p=[0.40, 0.35, 0.15, 0.10]),
            "heating_type":    np.random.choice(["Gas", "Electric", "HeatPump", "None"], n,
                                                p=[0.45, 0.30, 0.15, 0.10]),
            "tariff_plan":     np.random.choice(["Standard", "Economy7", "EV"], n,
                                                p=[0.60, 0.30, 0.10]),
        })
        bldg_f = df["building_type"].map({"Apartment": 0.7, "House": 1.0, "Office": 1.4, "Industrial": 2.5})
        heat_f = df["heating_type"].map({"None": 0.5, "HeatPump": 0.8, "Gas": 1.0, "Electric": 1.2})
        y = (
            2.5  * df["floor_area"]
            + 300 * df["num_occupants"]
            - 800 * df["insulation_score"]
            + 30  * np.abs(df["outside_temp"] - 18)   # deviation from comfort temp
            - 1200 * df["solar_panels"]
            + 50  * df["appliance_age"]
            * bldg_f * heat_f
            + np.random.normal(0, 300, n)
        ).clip(100).values
        X = pd.get_dummies(df, columns=["building_type", "heating_type", "tariff_plan"])
        return X.values, y

    datasets["energy_consumption_bimodal"] = make_energy()

    # --- 4. Employee salary — ordinal education, zero-inflated bonus, log-normal base ---
    def make_salary(n=1000):
        df = pd.DataFrame({
            "years_experience":  np.random.randint(0, 35, n).astype(float),
            "age":               normal(n, 38, 10).clip(22, 65),
            "performance_score": beta_dist(n, a=5, b=2),            # skewed-high beta
            "overtime_hours":    zero_inflated(n, p_zero=0.55, lam=8),
            "publications":      zero_inflated(n, p_zero=0.8, lam=3),
            "education":         np.random.choice(["HighSchool", "Bachelor", "Master", "PhD"], n,
                                                  p=[0.10, 0.50, 0.30, 0.10]),
            "department":        np.random.choice(["Engineering", "Sales", "HR", "Finance", "Research"], n),
            "employment_type":   np.random.choice(["Full-time", "Part-time", "Contract"], n,
                                                  p=[0.70, 0.15, 0.15]),
            "location":          np.random.choice(["NYC", "SF", "Austin", "Chicago", "Remote"], n,
                                                  p=[0.20, 0.20, 0.15, 0.15, 0.30]),
        })
        edu  = df["education"].map({"HighSchool": 0, "Bachelor": 15000, "Master": 30000, "PhD": 55000})
        dept = df["department"].map({"HR": 0, "Sales": 5000, "Finance": 15000, "Engineering": 25000, "Research": 35000})
        loc  = df["location"].map({"Remote": 0, "Chicago": 5000, "Austin": 8000, "NYC": 20000, "SF": 25000})
        emp  = df["employment_type"].map({"Part-time": -18000, "Contract": -5000, "Full-time": 0})
        y = (
            38000
            + 2200  * df["years_experience"]
            + 15000 * df["performance_score"]
            + 25    * df["overtime_hours"]
            + 1000  * df["publications"]
            + edu + dept + loc + emp
            + np.random.normal(0, 6000, n)
        ).clip(18000).values
        X = pd.get_dummies(df, columns=["education", "department", "employment_type", "location"])
        return X.values, y

    datasets["employee_salary_skewed"] = make_salary()

    # --- 5. Hospital length of stay — Poisson target, highly skewed features ---
    def make_hospital(n=1000):
        df = pd.DataFrame({
            "age":             normal(n, 58, 18).clip(0, 100),
            "bmi":             normal(n, 27, 6).clip(15, 55),
            "num_comorbidities": poisson(n, lam=1.8),               # Poisson count
            "prior_admissions":  zero_inflated(n, p_zero=0.65, lam=2),
            "lab_score":         bimodal(n, mu1=40, mu2=80, sigma=8), # bimodal severity
            "systolic_bp":       normal(n, 128, 20).clip(80, 210),
            "admission_type":    np.random.choice(["Emergency", "Elective", "Urgent"], n,
                                                  p=[0.40, 0.35, 0.25]),
            "ward":              np.random.choice(["General", "ICU", "Surgical", "Cardiology"], n,
                                                  p=[0.40, 0.20, 0.25, 0.15]),
            "insurance":         np.random.choice(["Private", "Medicare", "Medicaid", "Uninsured"], n,
                                                  p=[0.40, 0.30, 0.20, 0.10]),
        })
        adm_d  = df["admission_type"].map({"Elective": 1, "Urgent": 2, "Emergency": 4})
        ward_d = df["ward"].map({"General": 2, "Cardiology": 4, "Surgical": 5, "ICU": 9})
        y = (
            adm_d + ward_d
            + 0.04  * df["age"]
            + 0.5   * df["num_comorbidities"]
            + 0.3   * df["prior_admissions"]
            + 0.03  * (df["lab_score"] - 60).clip(0)
            + np.random.exponential(0.8, n)
        ).clip(1).values
        X = pd.get_dummies(df, columns=["admission_type", "ward", "insurance"])
        return X.values, y

    datasets["hospital_los_poisson"] = make_hospital()

    return datasets


# ================================================================
# MODELS
# ================================================================

def get_classifiers():
    return {
        "logistic":          LogisticRegression(max_iter=500),
        "random_forest":     RandomForestClassifier(n_estimators=100),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100),
        "decision_tree":     DecisionTreeClassifier(),
        "knn":               make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "xgboost":           XGBClassifier(n_estimators=100, use_label_encoder=False,
                                           eval_metric="logloss", verbosity=0),
    }

def get_regressors():
    return {
        "linear":            LinearRegression(),
        "ridge":             Ridge(),
        "lasso":             Lasso(),
        "random_forest":     RandomForestRegressor(n_estimators=100),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100),
        "decision_tree":     DecisionTreeRegressor(),
        "knn":               make_pipeline(StandardScaler(), KNeighborsRegressor()),
        "xgboost":           XGBRegressor(n_estimators=100, objective="reg:squarederror",
                                          verbosity=0),
    }


DISTANCE_METHODS = ["percentile", "wasserstein", "mahalanobis", "counterfactual"]


# ================================================================
# HARNESS
# ================================================================

def run_suite(datasets, models, task, n_features=4):
    results = {}
    for dname, (X, y) in datasets.items():
        print(f"\n  Dataset: {dname}  shape={X.shape}  "
              f"{'classes=' + str(np.unique(y)) if task == 'classification' else 'y_std=' + f'{np.std(y):.1f}'}")
        results[dname] = {}
        for mname, model in models.items():
            results[dname][mname] = {}
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"    [{mname}] FIT ERROR: {e}")
                for d in DISTANCE_METHODS:
                    results[dname][mname][d] = f"FIT ERROR: {e}"
                continue

            for dist in DISTANCE_METHODS:
                try:
                    exp = DirectionalSensitivityExplainer(
                        model, X, y, distance_method=dist, scale=0.5
                    )
                    df = exp.run(n=n_features)
                    results[dname][mname][dist] = df
                    print(f"    [{mname}/{dist}] OK — {len(df)} rows")
                except Exception as e:
                    print(f"    [{mname}/{dist}] ERROR — {e}")
                    traceback.print_exc()
                    results[dname][mname][dist] = f"ERROR: {e}"

    return results


def run_all():
    np.random.seed(42)

    print("\n" + "="*65)
    print("CLASSIFICATION BENCHMARK")
    print("="*65)
    clf_results = run_suite(
        load_classification_datasets(),
        get_classifiers(),
        task="classification"
    )

    print("\n" + "="*65)
    print("REGRESSION BENCHMARK")
    print("="*65)
    reg_results = run_suite(
        load_regression_datasets(),
        get_regressors(),
        task="regression"
    )

    return {"classification": clf_results, "regression": reg_results}


# ================================================================
# JSON EXPORT
# ================================================================

def to_json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return [to_json_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return to_json_safe(obj.tolist())
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    results = run_all()

    # Quick summary
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    for task, task_results in results.items():
        errors = sum(
            1 for dr in task_results.values()
            for mr in dr.values()
            for v in mr.values()
            if isinstance(v, str) and v.startswith("ERROR")
        )
        runs = sum(
            len(mr) for dr in task_results.values() for mr in dr.values()
        )
        print(f"  {task:20s}  runs={runs:4d}  errors={errors}")

    with open("benchmark_results.json", "w") as f:
        json.dump(to_json_safe(results), f, indent=2)
    print("\nSaved → benchmark_results.json")
