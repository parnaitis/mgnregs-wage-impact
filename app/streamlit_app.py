"""
app/streamlit_app.py
---------------------
Interactive dashboard for the MGNREGS Causal Impact study.
Deploy: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

st.set_page_config(
    page_title="MGNREGS Causal Impact — West Bengal",
    page_icon="📊",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────────
@st.cache_data
def load_panel():
    path = "data/processed/panel_final.csv"
    if not os.path.exists(path):
        # Try relative path from app/
        path = "../data/processed/panel_final.csv"
    return pd.read_csv(path)

df = load_panel()

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Controls")
years = sorted(df["year"].dropna().unique().astype(int))
selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-2)

districts = sorted(df["district"].dropna().unique())
selected_districts = st.sidebar.multiselect(
    "Highlight Districts", districts,
    default=districts[:3]
)

# ── Header ─────────────────────────────────────────────────
st.title("Causal Impact of MGNREGS Spending on Rural Wages")
st.markdown("**West Bengal District Panel · 2018–2024 · Two-Way Fixed Effects DiD**")
st.markdown("---")

# ── Key metrics row ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

yr_data = df[df["year"] == selected_year]
with col1:
    avg_wage = yr_data["avg_wage_daily"].mean()
    st.metric("Avg Daily Wage (₹)", f"₹{avg_wage:.0f}" if not np.isnan(avg_wage) else "N/A")
with col2:
    avg_exp = yr_data["total_exp_lakh"].mean()
    st.metric("Avg District Exp (₹ Lakh)", f"₹{avg_exp:,.0f}" if not np.isnan(avg_exp) else "N/A")
with col3:
    st.metric("Districts", f"{yr_data['district'].nunique()}")
with col4:
    st.metric("Two-Way FE β", "-0.027 (p=0.635)", delta="Not significant", delta_color="off")

st.markdown("---")

# ── Two column layout ──────────────────────────────────────
left, right = st.columns(2)

# ── Plot 1: Wage trends ────────────────────────────────────
with left:
    st.subheader("District Wage Trends (2018–2024)")
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for dist in df["district"].unique():
        sub = df[df["district"] == dist].sort_values("year")
        sub_valid = sub[sub["avg_wage_daily"].notna()]
        if len(sub_valid) > 1:
            ax.plot(sub_valid["year"], sub_valid["avg_wage_daily"],
                    alpha=0.25, linewidth=1, color="#94A3B8")

    colors = ["#2563EB","#DC2626","#16A34A","#D97706","#7C3AED"]
    for i, dist in enumerate(selected_districts):
        sub = df[df["district"] == dist].sort_values("year")
        sub_valid = sub[sub["avg_wage_daily"].notna()]
        if len(sub_valid) > 1:
            ax.plot(sub_valid["year"], sub_valid["avg_wage_daily"],
                    linewidth=2.5, color=colors[i % len(colors)],
                    label=dist.title(), marker="o", markersize=5)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Avg Daily Wage (₹)", fontsize=11)
    ax.set_title("Rural Agricultural Wage by District", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if selected_districts:
        ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Plot 2: Expenditure vs Wage scatter ────────────────────
with right:
    st.subheader(f"Expenditure vs Wage — {selected_year}")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))

    plot_df = yr_data.dropna(subset=["log_exp_lakh","avg_wage_daily"])
    if len(plot_df) > 2:
        ax2.scatter(plot_df["log_exp_lakh"], plot_df["avg_wage_daily"],
                    color="#2563EB", s=80, alpha=0.7, zorder=3)

        # Label each dot
        for _, row in plot_df.iterrows():
            ax2.annotate(row["district"].title()[:12],
                         (row["log_exp_lakh"], row["avg_wage_daily"]),
                         fontsize=6.5, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")

        # OLS trend line
        z = np.polyfit(plot_df["log_exp_lakh"], plot_df["avg_wage_daily"], 1)
        p = np.poly1d(z)
        xr = np.linspace(plot_df["log_exp_lakh"].min(),
                         plot_df["log_exp_lakh"].max(), 100)
        ax2.plot(xr, p(xr), color="#DC2626", linewidth=2,
                 linestyle="--", label="OLS trend (biased)")

    ax2.set_xlabel("log(MGNREGS Expenditure, ₹ Lakh)", fontsize=11)
    ax2.set_ylabel("Avg Daily Wage (₹)", fontsize=11)
    ax2.set_title("Note: Cross-sectional correlation ≠ causal effect", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# ── Regression results ─────────────────────────────────────
st.subheader("Regression Results Summary")

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.markdown("**Model Comparison**")
    results_df = pd.DataFrame({
        "Model": ["Pooled OLS (naive)", "Two-Way FE DiD (preferred)"],
        "β (log_exp)": ["-0.072***", "-0.027"],
        "Std Error": ["0.017", "0.057"],
        "p-value": ["<0.001", "0.635"],
        "District FE": ["No", "Yes"],
        "Year FE": ["No", "Yes"],
        "Clustered SE": ["Yes", "Yes"],
    })
    st.dataframe(results_df, hide_index=True, use_container_width=True)

    st.markdown("""
    **Key insight:** The OLS estimate is large and significant, but entirely
    driven by *selection bias* — poorer districts receive more MGNREGS funds
    AND have lower wages. Once we add district fixed effects (Two-Way FE),
    the effect disappears. This is the identification problem that makes
    naive regression misleading for policy evaluation.
    """)

with res_col2:
    st.markdown("**LightGBM vs DiD: Prediction vs Causation**")
    ml_df = pd.DataFrame({
        "Metric": ["CV R²", "CV MAE", "Purpose", "Can answer 'does spending cause wages?'"],
        "LightGBM": ["0.258", "0.096 log-units", "Prediction", "❌ No"],
        "Two-Way FE DiD": ["N/A", "N/A", "Causal inference", "✅ Yes"],
    })
    st.dataframe(ml_df, hide_index=True, use_container_width=True)

    st.info("""
    **Why this matters:** A hiring manager from a quant/policy role will ask
    "how do you know spending *causes* wages to rise?" The DiD framework
    answers this. LightGBM cannot — high R² may just reflect that rich
    districts spend more AND pay more, with no causal link.
    """)

st.markdown("---")

# ── District data table ────────────────────────────────────
st.subheader(f"District-Level Data — {selected_year}")
display_cols = ["district","avg_wage_daily","total_exp_lakh",
                "avg_days_per_hh","women_share","sc_share","pct_agri_works"]
display_cols = [c for c in display_cols if c in df.columns]
yr_display = df[df["year"]==selected_year][display_cols].dropna(
    subset=["avg_wage_daily"]).sort_values("avg_wage_daily", ascending=False)
yr_display.columns = ["District","Avg Wage (₹)","Total Exp (₹L)",
                       "Avg Days/HH","Women Share","SC Share","% Agri Works"][:len(display_cols)]
st.dataframe(yr_display.reset_index(drop=True),
             hide_index=True, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Data:** MGNREGS MIS via data.gov.in · **Period:** 2018–2024 · **Geography:** West Bengal (23 districts)  
**Method:** Two-Way Fixed Effects DiD with clustered standard errors (linearmodels)  
**Code:** [github.com/parnaitis/mgnregs-wage-impact](https://github.com/parnaitis/mgnregs-wage-impact)
""")
