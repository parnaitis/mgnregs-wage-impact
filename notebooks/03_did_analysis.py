"""
notebooks/03_did_analysis.py
-----------------------------
Two-way fixed effects DiD analysis:
log(wage_it) = α + β·log_exp_it + δ_i + λ_t + ε_it

Run as: python notebooks/03_did_analysis.py
All outputs saved to outputs/figures/ and outputs/tables/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.panel import compare
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

warnings.filterwarnings("ignore")
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables",  exist_ok=True)

PANEL_PATH = "data/processed/panel_final.csv"

# ============================================================
# 0. Load data
# ============================================================
print("Loading panel data...")
df = pd.read_csv(PANEL_PATH)
print(f"  Shape: {df.shape}")
print(f"  Districts: {df['district'].nunique()}, Years: {sorted(df['year'].unique())}")

# Set panel index required by linearmodels
df = df.set_index(["district", "year"])

# ============================================================
# 1. Pooled OLS (naive baseline — biased, for comparison only)
# ============================================================
print("\n[1] Pooled OLS (naive baseline)...")
y   = df["log_wage"].dropna()
X_b = df[["log_exp_lakh"]].dropna()
idx = y.index.intersection(X_b.index)

mod_ols = PooledOLS(y.loc[idx], sm.add_constant(X_b.loc[idx]))
res_ols = mod_ols.fit(cov_type="clustered", cluster_entity=True)
print(res_ols.summary.tables[1])

# ============================================================
# 2. Two-Way Fixed Effects (preferred causal model)
#    District FE + Year FE + clustered SE
# ============================================================
print("\n[2] Two-Way Fixed Effects DiD (preferred)...")

controls = ["sc_share", "st_share", "women_share",
            "pct_agri_works", "pct_timely_payment"]

# Build clean subset
cols_needed = ["log_wage", "log_exp_lakh"] + controls
sub = df[cols_needed].dropna()

mod_fe = PanelOLS(
    sub["log_wage"],
    sm.add_constant(sub[["log_exp_lakh"] + controls]),
    entity_effects=True,   # district fixed effects
    time_effects=True,     # year fixed effects
)
res_fe = mod_fe.fit(cov_type="clustered", cluster_entity=True)
print(res_fe.summary.tables[1])

# ============================================================
# 3. Event-study / parallel trends plot
#    Regress log_wage on year dummies × log_exp to test
#    whether treatment effect emerges post-treatment
# ============================================================
print("\n[3] Plotting year-by-year coefficients (pre-trend test)...")

df_reset = df.reset_index()
years = sorted(df_reset["year"].unique())
base_year = years[0]

year_coefs = {}
for yr in years[1:]:
    sub_yr = df_reset[df_reset["year"].isin([base_year, yr])].copy()
    sub_yr = sub_yr.set_index(["district", "year"])
    cols = ["log_wage", "log_exp_lakh"] + controls
    sub_yr = sub_yr[cols].dropna()
    try:
        m = PanelOLS(
            sub_yr["log_wage"],
            sm.add_constant(sub_yr[["log_exp_lakh"] + controls]),
            entity_effects=True,
            time_effects=True
        ).fit(cov_type="clustered", cluster_entity=True)
        coef = m.params["log_exp_lakh"]
        se   = m.std_errors["log_exp_lakh"]
        year_coefs[yr] = (coef, se)
    except Exception as e:
        print(f"  Skipping year {yr}: {e}")

if year_coefs:
    fig, ax = plt.subplots(figsize=(9, 5))
    yrs   = list(year_coefs.keys())
    coefs = [year_coefs[y][0] for y in yrs]
    ses   = [year_coefs[y][1] for y in yrs]
    ax.errorbar(yrs, coefs, yerr=[1.96*s for s in ses],
                fmt="o-", color="#2563EB", capsize=5, linewidth=2,
                markersize=7, label="β ± 1.96 SE")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Effect of log(MGNREGS Exp) on log(Wage)", fontsize=11)
    ax.set_title("Year-by-Year DiD Coefficients\n(Two-Way FE, Clustered SE)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/event_study.png", dpi=150)
    plt.close()
    print("  Saved: outputs/figures/event_study.png")

# ============================================================
# 4. Wage trend by district (visual evidence)
# ============================================================
print("\n[4] Plotting district wage trends...")
df_plot = df.reset_index()

fig, ax = plt.subplots(figsize=(11, 6))
for dist, grp in df_plot.groupby("district"):
    grp_s = grp.sort_values("year")
    ax.plot(grp_s["year"], grp_s["avg_wage_daily"],
            alpha=0.4, linewidth=1.2, color="#6B7280")

# Highlight top 3 and bottom 3 districts by mean wage
mean_wages = df_plot.groupby("district")["avg_wage_daily"].mean().sort_values()
highlight = list(mean_wages.index[:2]) + list(mean_wages.index[-2:])
colors = ["#DC2626","#B91C1C","#1D4ED8","#1E40AF"]
for dist, col in zip(highlight, colors):
    grp = df_plot[df_plot["district"]==dist].sort_values("year")
    ax.plot(grp["year"], grp["avg_wage_daily"],
            linewidth=2.5, color=col,
            label=dist.title()[:20])

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Avg Daily Wage (₹)", fontsize=12)
ax.set_title("Rural Agricultural Wage Trends by District — West Bengal (2018–2024)", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("outputs/figures/wage_trends.png", dpi=150)
plt.close()
print("  Saved: outputs/figures/wage_trends.png")

# ============================================================
# 5. Expenditure vs Wage scatter (by year)
# ============================================================
print("\n[5] Expenditure vs Wage scatter...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
axes = axes.flatten()
for i, yr in enumerate(sorted(df_plot["year"].unique())):
    ax = axes[i]
    sub = df_plot[df_plot["year"]==yr]
    ax.scatter(sub["log_exp_lakh"], sub["avg_wage_daily"],
               alpha=0.7, color="#2563EB", s=60)
    # Trend line
    # Trend line — align on non-null rows for both axes
mask = sub["log_exp_lakh"].notna() & sub["avg_wage_daily"].notna()
if mask.sum() > 2:
    z = np.polyfit(sub.loc[mask, "log_exp_lakh"],
                   sub.loc[mask, "avg_wage_daily"], 1)
    p = np.poly1d(z)
    xr = np.linspace(sub.loc[mask, "log_exp_lakh"].min(),
                     sub.loc[mask, "log_exp_lakh"].max(), 50)
    ax.plot(xr, p(xr), color="#DC2626", linewidth=1.5)

# Hide unused subplot if years < 8
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("MGNREGS Expenditure vs Rural Wage — By Year (West Bengal Districts)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("outputs/figures/exp_wage_scatter.png", dpi=150)
plt.close()
print("  Saved: outputs/figures/exp_wage_scatter.png")

# ============================================================
# 6. Save regression table
# ============================================================
print("\n[6] Saving regression tables...")

comparison = compare({"Pooled OLS": res_ols, "Two-Way FE (DiD)": res_fe})
with open("outputs/tables/regression_results.txt", "w") as f:
    f.write("MGNREGS CAUSAL IMPACT ON RURAL WAGES\n")
    f.write("West Bengal District Panel, 2018-2024\n")
    f.write("Dependent variable: log(Average Daily Wage)\n\n")
    f.write(str(comparison.summary))

print("  Saved: outputs/tables/regression_results.txt")

# ============================================================
# 7. Summary of key finding
# ============================================================
print("\n" + "="*55)
print("KEY FINDING:")
coef = res_fe.params["log_exp_lakh"]
se   = res_fe.std_errors["log_exp_lakh"]
pval = res_fe.pvalues["log_exp_lakh"]
sig  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
print(f"  Two-Way FE β (log_exp → log_wage): {coef:.4f}{sig} (SE={se:.4f}, p={pval:.3f})")
print(f"  Interpretation: 1% increase in MGNREGS expenditure")
print(f"  associated with {coef:.3f}% change in rural daily wages")
print(f"  (after controlling for district and year fixed effects)")
print("="*55)
print("\nAll outputs saved to outputs/figures/ and outputs/tables/")
