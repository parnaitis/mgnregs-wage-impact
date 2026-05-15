# Causal Impact of MGNREGS Spending on Rural Wage Growth in India
### A District-Level Panel Econometrics Study (2011–2022)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Research Question

Does increased MGNREGS (Mahatma Gandhi National Rural Employment Guarantee Scheme)
expenditure **causally** raise agricultural wages for rural workers at the district level —
after controlling for rainfall, crop price cycles, and district/year fixed effects?

---

## Why This Matters

MGNREGS is one of the world's largest public works programmes (~₹70,000 crore annual
budget), yet causal evidence on its wage effects remains contested. Most studies either
rely on state-level aggregates or fail to address the selection problem (districts with
worse labour markets may receive more funds). This project uses the **staggered rollout**
of MGNREGS Phase 1 and Phase 2 across districts as a source of quasi-experimental
variation, enabling a cleaner causal estimate.

---

## Methodology

| Stage | Technique | Library |
|---|---|---|
| Data cleaning | Pandas, fuzzy district name matching | `pandas`, `thefuzz` |
| Exploratory analysis | District choropleth maps, trend plots | `geopandas`, `matplotlib` |
| Causal estimation | Two-way fixed effects DiD (district + year FE) | `linearmodels` |
| Robustness check | Parallel trends test, event-study plot | `linearmodels` |
| Predictive benchmark | LightGBM with SHAP feature importance | `lightgbm`, `shap` |
| Dashboard | Interactive district map + results viewer | `streamlit` |

### Core Model (Difference-in-Differences)

```
log(wage_idt) = α + β · MGNREGS_spend_idt + γ · X_idt + δ_i + λ_t + ε_idt
```

Where:
- `wage_idt` = real agricultural wage in district `i`, state `s`, year `t`
- `MGNREGS_spend_idt` = per-capita MGNREGS expenditure (₹)
- `X_idt` = controls: rainfall deviation, MSP crop price index
- `δ_i` = district fixed effects (absorb all time-invariant district characteristics)
- `λ_t` = year fixed effects (absorb common macro shocks)
- Standard errors clustered at district level

---

## Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| MGNREGS district expenditure & person-days | [nrega.nic.in MIS](https://nreganarep.nic.in/netnrega/home.aspx) | 2006–2023 |
| Rural agricultural wages (PLFS) | [mospi.gov.in](https://mospi.gov.in/web/plfs) | 2017–2022 |
| District-level rainfall | [IMD / data.gov.in](https://data.gov.in) | 2001–2022 |
| India district shapefiles | [Datameet / SHR](https://github.com/datameet/maps) | 2011 Census |

All raw data is preserved unmodified in `data/raw/`. Cleaning scripts in `src/` are
fully reproducible.

---

## Key Findings

- **Pooled OLS** (naive): β = -0.072 (p<0.001) — spending *negatively* correlated
  with wages. This is selection bias: poorer districts receive more MGNREGS funds.

- **Two-Way FE DiD** (preferred causal estimate): β = -0.027 (p=0.635, SE=0.057)
  — effect is small and statistically insignificant after controlling for
  district and year fixed effects. The OLS result was entirely driven by
  selection bias, not a true causal effect.

- **LightGBM predictive benchmark**: CV R² = 0.258 — district identity
  (fixed characteristics) explains most wage variation, not spending intensity.

- **Interpretation**: MGNREGS spending intensity at the district level does not
  appear to causally drive agricultural wage growth in West Bengal (2018–2024).
  State-level wage floors (notified MGNREGS rates) likely dominate local
  spending variation as the wage-setting mechanism — consistent with
  Imbert & Papp (2015) and Muralidharan et al. (2017).

- **Methodological contribution**: The divergence between OLS (β=-0.072***)
  and Two-Way FE (β=-0.027, ns) demonstrates the importance of controlling
  for district-level selection into MGNREGS funding — a common failure in
  policy evaluation.

## Repo Structure

```
mgnregs-wage-impact/
├── data/
│   ├── raw/          # Original downloads — never edited
│   ├── processed/    # Cleaned panel dataset
│   └── sources.md    # All source URLs with download dates
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_did_analysis.ipynb
│   └── 04_ml_comparison.ipynb
├── src/
│   ├── clean_mgnregs.py
│   ├── clean_plfs.py
│   └── merge_panel.py
├── app/
│   └── streamlit_app.py
└── requirements.txt
```

---

## How to Reproduce

```bash
git clone https://github.com/YOUR_USERNAME/mgnregs-wage-impact
cd mgnregs-wage-impact
pip install -r requirements.txt

# Run cleaning pipeline
python src/clean_mgnregs.py
python src/clean_plfs.py
python src/merge_panel.py

# Launch notebooks (in order)
jupyter notebook notebooks/

# Run Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## Requirements

See `requirements.txt`. Core dependencies:

```
pandas==2.2.2
numpy==1.26.4
geopandas==0.14.4
linearmodels==6.0
lightgbm==4.3.0
shap==0.45.0
streamlit==1.35.0
matplotlib==3.9.0
seaborn==0.13.2
thefuzz==0.22.1
jupyter==1.0.0
```

---

## Author

**[Your Name]**  
Economics + Data Science  
[LinkedIn] · [Email]

*Built as a portfolio project demonstrating applied causal inference on Indian labour market data.*
