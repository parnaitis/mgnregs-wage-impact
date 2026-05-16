# Causal Impact of MGNREGS Spending on Rural Wages — West Bengal
### A District-Level Panel Econometrics Study (2018–2024)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)

---

## Research Question

Does increased MGNREGS (Mahatma Gandhi National Rural Employment Guarantee Scheme)
expenditure **causally** raise agricultural wages for rural workers at the district level,
after controlling for district-level fixed characteristics and common year-level shocks?

---

## Why This Matters

MGNREGS is one of the world's largest public works programmes (~70,000 crore annual
budget). A naive cross-sectional analysis suggests spending is associated with lower
wages — but this reflects selection bias: poorer districts with structurally weaker
labour markets both receive more funds and have lower wages. This project uses a
**two-way fixed effects** strategy to isolate within-district variation over time,
removing that selection problem and producing a credible causal estimate.

---

## Key Findings

| Model | β (log expenditure) | Std Error | p-value | District FE | Year FE |
|---|---|---|---|---|---|
| Pooled OLS (naive) | -0.072*** | 0.017 | <0.001 | No | No |
| **Two-Way FE DiD (preferred)** | **-0.027** | **0.057** | **0.635** | **Yes** | **Yes** |

- **The OLS result is entirely selection bias.** Poorer districts receive more
  MGNREGS funds AND have lower wages — not because the programme lowers wages,
  but because the programme targets distressed areas.

- **After two-way fixed effects**, the causal estimate is small and statistically
  insignificant (p = 0.635). District-level spending intensity does not appear
  to causally drive wage growth in West Bengal over this period.

- **Interpretation:** State-level notified MGNREGS wage floors likely dominate
  local spending variation as the wage-setting mechanism — consistent with
  Imbert & Papp (2015) and Muralidharan et al. (2017).

- **Methodological contribution:** The divergence between OLS (B=-0.072***)
  and TWFE (B=-0.027, ns) demonstrates the identification problem that makes
  naive policy regression misleading for evaluation — a common failure in applied work.

- **LightGBM predictive benchmark:** CV R2 = 0.258. District fixed characteristics
  explain most wage variation, not spending intensity. High predictive R2 is not
  the same as a causal effect.

---

## Methodology

| Stage | Technique | Library |
|---|---|---|
| Data ingestion | Monthly MIS data from data.gov.in, HTML-as-XLS parsing | `pandas`, `beautifulsoup4` |
| Data cleaning | Monthly to annual aggregation, outlier detection, missing value treatment | `pandas`, `numpy` |
| Exploratory analysis | District wage trend plots, expenditure scatter by year | `matplotlib` |
| Causal estimation | Two-way fixed effects DiD (district + year FE), clustered SE | `linearmodels` |
| Robustness check | Year-by-year event study plot, parallel trends visual | `linearmodels` |
| Predictive benchmark | LightGBM with 5-fold CV, SHAP feature importance | `lightgbm`, `shap` |
| Dashboard | Interactive district explorer, regression results viewer | `streamlit` |

### Core Model

```
log(wage_it) = a + B * log(exp_it) + d_i + l_t + e_it
```

Where:
- `wage_it` = average daily agricultural wage in district `i`, year `t` (Rs.)
- `exp_it` = total MGNREGS expenditure in district `i`, year `t` (Rs. lakhs)
- `d_i` = district fixed effects (absorb all time-invariant district characteristics)
- `l_t` = year fixed effects (absorb common macro shocks: COVID, inflation, policy changes)
- Standard errors clustered at district level

---

## Data

| Dataset | Source | Coverage |
|---|---|---|
| MGNREGS district-level MIS data (expenditure, wages, person-days) | [data.gov.in](https://data.gov.in/resource/district-wise-mgnrega-data-glance) | West Bengal, 2018-2024 |

**Geography:** West Bengal — 23 districts  
**Period:** Financial years 2018-19 through 2024-25  
**Observations:** 161 district-year pairs (23 districts x 7 years)  
**Key variables:** Average daily wage (Rs.), total expenditure (Rs. lakhs),
person-days generated, SC/ST share, women's participation share, % agriculture-allied works

All raw data preserved unmodified in `data/raw/financial/`. Cleaning scripts in
`src/` are fully reproducible.

### Data Engineering Challenges

- Government portal exports HTML tables disguised as `.xls` files — detected
  and parsed using BeautifulSoup rather than pandas Excel reader
- 2024 data file had 2,258 rows vs 276 for all other years — silent format
  change requiring asymmetric aggregation handling
- Monthly observations aggregated to annual with variable-specific rules:
  `sum` for flow variables (expenditure), `max` for stock variables (job cards),
  `mean` for rate variables (wage)
- Zero wages detected and treated as missing (not real zeros); outlier wages
  above Rs. 500/day flagged and removed

---

## Repo Structure

```
mgnregs-wage-impact/
├── data/
│   ├── raw/financial/    # Original monthly CSVs from data.gov.in
│   ├── processed/        # Cleaned annual panel dataset
│   └── sources.md        # All source URLs with download dates
├── notebooks/
│   ├── 03_did_analysis.py    # Two-way FE DiD + event study
│   └── 04_ml_comparison.py   # LightGBM + SHAP analysis
├── src/
│   ├── build_panel.py                     # Monthly to annual aggregation pipeline
│   ├── clean_mgnregs.py                   # Raw file cleaning (HTML-XLS detection)
│   └── download_mgnregs_financial.py      # Data download utility
├── outputs/
│   ├── figures/          # All plots (wage trends, event study, SHAP)
│   └── tables/           # Regression results, model comparison
├── app/
│   └── streamlit_app.py  # Interactive dashboard
└── requirements.txt
```

---

## How to Reproduce

```bash
git clone https://github.com/parnaitis/mgnregs-wage-impact
cd mgnregs-wage-impact
pip install -r requirements.txt

# Download data from data.gov.in (West Bengal, FY 2018-19 to 2024-25)
# Save CSVs to data/raw/financial/ as wb_2018.csv ... wb_2024.csv

# Build annual panel
python src/build_panel.py

# Run causal analysis
python notebooks/03_did_analysis.py

# Run ML benchmark
python notebooks/04_ml_comparison.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## References

- Imbert, C. & Papp, J. (2015). Labor Market Effects of Social Programs:
  Evidence from India's Employment Guarantee. *American Economic Journal: Applied Economics.*
- Muralidharan, K., Niehaus, P. & Sukhtankar, S. (2017). General Equilibrium
  Effects of (Improving) Public Employment Programs. *NBER Working Paper.*

---

## Requirements

```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.0
streamlit==1.35.0
lightgbm==4.3.0
shap==0.45.0
linearmodels==6.0
scikit-learn==1.4.2
statsmodels
beautifulsoup4
lxml
```

---

## Author

**Arna Parnaitis**  
Economics + Data Science  
[LinkedIn](https://linkedin.com/in/YOUR-LINKEDIN) · [GitHub](https://github.com/parnaitis)

*District-level causal inference study on Indian rural labour markets using
panel econometrics and machine learning.*