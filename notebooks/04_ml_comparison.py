"""
notebooks/04_ml_comparison.py
------------------------------
LightGBM predictive model vs causal DiD model.
Shows why prediction ≠ causation — the key insight
that separates this project from Kaggle work.

Run as: python notebooks/04_ml_comparison.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables",  exist_ok=True)

PANEL_PATH = "data/processed/panel_final.csv"

# ============================================================
# 0. Load data
# ============================================================
print("Loading panel data...")
df = pd.read_csv(PANEL_PATH)

FEATURES = [
    "log_exp_lakh",
    "log_person_days",
    "avg_days_per_hh",
    "sc_share",
    "st_share",
    "women_share",
    "pct_agri_works",
    "pct_timely_payment",
    "budget_utilisation",
    "works_completed",
    "hh_100days",
    "time",           # year trend
    "district_id",    # district identity (for predictive only)
]
TARGET = "log_wage"

# Drop rows with missing values in key columns
cols_needed = [TARGET] + [f for f in FEATURES if f in df.columns]
df_ml = df[cols_needed].dropna().copy()
print(f"  ML dataset shape: {df_ml.shape}")

X = df_ml[[f for f in FEATURES if f in df_ml.columns]]
y = df_ml[TARGET]

# ============================================================
# 1. LightGBM with cross-validation
# ============================================================
print("\n[1] Training LightGBM (5-fold CV)...")

lgbm = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(lgbm, X, y, cv=kf, scoring="r2")
cv_mae = cross_val_score(lgbm, X, y, cv=kf,
                         scoring="neg_mean_absolute_error")

print(f"  CV R²:  {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print(f"  CV MAE: {(-cv_mae).mean():.4f} ± {(-cv_mae).std():.4f}")

# Fit on full data for SHAP
lgbm.fit(X, y)

# ============================================================
# 2. SHAP feature importance
# ============================================================
print("\n[2] Computing SHAP values...")
explainer  = shap.TreeExplainer(lgbm)
shap_vals  = explainer.shap_values(X)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_vals, X, plot_type="bar",
                  show=False, color="#2563EB")
plt.title("SHAP Feature Importance\n(LightGBM — Predictive Model for log(Wage))",
          fontsize=12)
plt.tight_layout()
plt.savefig("outputs/figures/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_importance.png")

# SHAP beeswarm
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_vals, X, show=False)
plt.title("SHAP Value Distribution by Feature", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/figures/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_beeswarm.png")

# ============================================================
# 3. Causal vs Predictive comparison table
# ============================================================
print("\n[3] Causal vs Predictive comparison...")

# Reload DiD results if available
did_result_path = "outputs/tables/regression_results.txt"

comparison_text = f"""
CAUSAL (Two-Way FE DiD) vs PREDICTIVE (LightGBM) — MODEL COMPARISON
=====================================================================
West Bengal District Panel, 2018-2024
Dependent variable: log(Average Daily Wage)

PREDICTIVE MODEL (LightGBM)
-----------------------------
  Purpose      : Maximize prediction accuracy
  CV R²        : {cv_r2.mean():.3f} (± {cv_r2.std():.3f})
  CV MAE       : {(-cv_mae).mean():.4f} log-wage units
  Key features : See SHAP plots
  Limitation   : Cannot tell us WHETHER spending CAUSES wages to rise.
                 High R² may reflect reverse causation or omitted variables.

CAUSAL MODEL (Two-Way Fixed Effects DiD)
-----------------------------------------
  Purpose      : Estimate causal effect of MGNREGS spending on wages
  Identification: District FE (removes time-invariant confounders)
                  Year FE (removes common macro shocks like COVID, inflation)
                  Clustered SE (accounts for within-district serial correlation)
  Coefficient  : See outputs/tables/regression_results.txt
  Interpretation: β = % change in wage for 1% increase in MGNREGS expenditure,
                  AFTER controlling for all time-invariant district characteristics
                  and all common year effects.

WHY THIS MATTERS FOR POLICY
-----------------------------
  A high LightGBM R² tells us we can PREDICT which districts will have
  high wages. The DiD β tells us whether SPENDING MORE on MGNREGS in a
  district actually CAUSES wages to rise — a completely different question,
  and the one policymakers actually need answered.
"""

with open("outputs/tables/causal_vs_predictive.txt", "w", encoding="utf-8") as f:
    f.write(comparison_text)
print(comparison_text)
print("  Saved: outputs/tables/causal_vs_predictive.txt")

# ============================================================
# 4. Actual vs Predicted plot
# ============================================================
print("\n[4] Actual vs Predicted plot...")
y_pred = lgbm.predict(X)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y, y_pred, alpha=0.5, color="#2563EB", s=40)
mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual log(Wage)", fontsize=12)
ax.set_ylabel("Predicted log(Wage)", fontsize=12)
ax.set_title(f"LightGBM: Actual vs Predicted\nR² = {r2_score(y, y_pred):.3f}",
             fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/actual_vs_predicted.png", dpi=150)
plt.close()
print("  Saved: outputs/figures/actual_vs_predicted.png")

print("\nML analysis complete. All outputs in outputs/figures/ and outputs/tables/")
