"""
src/build_panel.py
------------------
Loads all yearly CSVs from data/raw/financial/,
aggregates monthly data to annual, and builds the
final analysis panel.

Output: data/processed/panel_final.csv
"""

import os
import glob
import re
import pandas as pd
import numpy as np

RAW_DIR  = "data/raw/financial"
OUT_PATH = "data/processed/panel_final.csv"


def load_and_combine():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "wb_*.csv")))
    if not files:
        print("No files found in data/raw/financial/. Expected wb_2018.csv etc.")
        return None
    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")

    dfs = []
    for f in files:
        year_match = re.search(r"(\d{4})", os.path.basename(f))
        year = int(year_match.group(1)) if year_match else None
        df = pd.read_csv(f)
        df["file_year"] = year
        dfs.append(df)
        print(f"  Loaded {os.path.basename(f)}: {df.shape}")

    return pd.concat(dfs, ignore_index=True)


def build_panel(df):
    """Aggregate monthly district data to annual panel."""

    # Standardise district names
    df["district"] = (df["district_name"]
                      .str.lower().str.strip()
                      .str.replace(r"[^\w\s]", "", regex=True)
                      .str.replace(r"\s+", " ", regex=True))

    # Parse financial year to integer (2018-2019 -> 2018)
    df["year"] = df["fin_year"].str.extract(r"(\d{4})").astype(int)

    # ----------------------------------------------------------------
    # Aggregate to district × year
    # For flow variables (expenditure, person-days): SUM across months
    # For rate variables (wage, days per HH): MEAN across months
    # ----------------------------------------------------------------
    agg = df.groupby(["district", "year"]).agg(

        # --- Core analysis variables ---
        total_exp_lakh          = ("Total_Exp",                                    "sum"),
        total_wages_lakh        = ("Wages",                                        "sum"),
        person_days             = ("Persondays_of_Central_Liability_so_far",       "max"),
        avg_wage_daily          = ("Average_Wage_rate_per_day_per_person",         "mean"),
        avg_days_per_hh         = ("Average_days_of_employment_provided_per_Household", "mean"),

        # --- Household/worker counts ---
        total_hh_worked         = ("Total_Households_Worked",                      "max"),
        total_individuals       = ("Total_Individuals_Worked",                     "max"),
        hh_100days              = ("Total_No_of_HHs_completed_100_Days_of_Wage_Employment", "max"),
        active_jobcards         = ("Total_No_of_Active_Job_Cards",                 "max"),
        total_jobcards          = ("Total_No_of_JobCards_issued",                  "max"),

        # --- Demographic controls ---
        sc_persondays           = ("SC_persondays",                                "sum"),
        st_persondays           = ("ST_persondays",                                "sum"),
        women_persondays        = ("Women_Persondays",                             "sum"),

        # --- Work quality indicators ---
        works_completed         = ("Number_of_Completed_Works",                    "sum"),
        works_ongoing           = ("Number_of_Ongoing_Works",                      "max"),
        pct_agri_works          = ("percent_of_Expenditure_on_Agriculture_Allied_Works", "mean"),
        pct_nrm_works           = ("percent_of_NRM_Expenditure",                   "mean"),
        pct_timely_payment      = ("percentage_payments_gererated_within_15_days", "mean"),
        labour_budget           = ("Approved_Labour_Budget",                       "max"),

    ).reset_index()

    # ----------------------------------------------------------------
    # Derived variables
    # ----------------------------------------------------------------
    agg["log_wage"]         = np.log(agg["avg_wage_daily"].clip(lower=1))
    agg["log_exp_lakh"]     = np.log1p(agg["total_exp_lakh"])
    agg["log_person_days"]  = np.log1p(agg["person_days"])

    # Expenditure per jobcard household (intensity measure)
    agg["exp_per_hh"] = np.where(
        agg["total_hh_worked"] > 0,
        (agg["total_exp_lakh"] * 1e5) / agg["total_hh_worked"],
        np.nan
    )
    agg["log_exp_per_hh"] = np.log1p(agg["exp_per_hh"])

    # Share of SC/ST person-days (inequality controls)
    total_pd = agg["person_days"].clip(lower=1)
    agg["sc_share"] = agg["sc_persondays"] / total_pd
    agg["st_share"] = agg["st_persondays"] / total_pd
    agg["women_share"] = agg["women_persondays"] / total_pd

    # Budget utilisation rate
    agg["budget_utilisation"] = np.where(
        agg["labour_budget"] > 0,
        agg["person_days"] / agg["labour_budget"],
        np.nan
    )

    # ----------------------------------------------------------------
    # Treatment variable for DiD:
    # High-intensity districts = those with above-median spending
    # We use within-district variation over time (FE handles levels)
    # ----------------------------------------------------------------
    median_exp = agg.groupby("year")["total_exp_lakh"].transform("median")
    agg["high_spend"] = (agg["total_exp_lakh"] > median_exp).astype(int)

    # ----------------------------------------------------------------
    # Time trend variable (for pre-trend tests)
    # ----------------------------------------------------------------
    min_year = agg["year"].min()
    agg["time"] = agg["year"] - min_year

    # ----------------------------------------------------------------
    # District fixed effect index (numeric ID for linearmodels)
    # ----------------------------------------------------------------
    agg["district_id"] = pd.Categorical(agg["district"]).codes

    # Sort
    agg = agg.sort_values(["district", "year"]).reset_index(drop=True)

    # ----------------------------------------------------------------
    # Data quality fixes
    # ----------------------------------------------------------------
    # Zero wages are missing data, not actual zero wages
    agg.loc[agg["avg_wage_daily"] <= 0, "avg_wage_daily"] = np.nan
    agg.loc[agg["avg_wage_daily"] > 500, "avg_wage_daily"] = np.nan  # outliers

    # Zero expenditure means no data reported, not zero spending
    agg.loc[agg["total_exp_lakh"] <= 0, "total_exp_lakh"] = np.nan

    # Timely payment % > 150 is a data entry error
    agg.loc[agg["pct_timely_payment"] > 150, "pct_timely_payment"] = np.nan

    # Recompute log transforms after cleaning
    agg["log_wage"]     = np.log(agg["avg_wage_daily"].clip(lower=1))
    agg["log_exp_lakh"] = np.log1p(agg["total_exp_lakh"])

    return agg


def main():
    print("=" * 55)
    print("Building MGNREGS Analysis Panel")
    print("=" * 55)

    raw = load_and_combine()
    if raw is None:
        return

    print(f"\nRaw combined shape: {raw.shape}")
    print(f"Years in raw data:  {sorted(raw['fin_year'].unique())}")

    panel = build_panel(raw)

    print(f"\nFinal panel shape:  {panel.shape}")
    print(f"Districts:          {panel['district'].nunique()}")
    print(f"Years:              {sorted(panel['year'].unique())}")
    print(f"Obs (dist × year):  {len(panel)}")
    print(f"\nKey variable summary:")
    print(panel[["avg_wage_daily","total_exp_lakh","person_days",
                 "avg_days_per_hh","pct_timely_payment"]].describe().round(2))

    os.makedirs("data/processed", exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to: {OUT_PATH}")
    print("\nSample rows:")
    print(panel[["district","year","avg_wage_daily","total_exp_lakh",
                 "log_wage","log_exp_lakh"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
