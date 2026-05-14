"""
src/merge_panel.py
------------------
Merges cleaned MGNREGS expenditure data with PLFS wage data to produce
a final district-level panel dataset ready for econometric analysis.

Handles the key data engineering challenge: district names differ across
datasets (e.g. "Purba Medinipur" vs "East Midnapore"). We use fuzzy
matching via thefuzz to align names.

Input :
    data/processed/mgnregs_clean.csv
    data/processed/plfs_wages_clean.csv

Output:
    data/processed/panel_final.csv

Final columns:
    district_clean      : standardised district name
    state_clean         : standardised state name
    year                : calendar year
    total_exp_lakh      : MGNREGS expenditure (₹ lakhs)
    person_days         : MGNREGS person-days
    exp_per_pd          : MGNREGS expenditure per person-day
    wage_rural_agri     : rural agricultural daily wage (₹, state-level from PLFS)
    log_wage            : log of wage_rural_agri
    log_exp_lakh        : log of (total_exp_lakh + 1)
"""

import os
import pandas as pd
import numpy as np
from thefuzz import process as fuzz_process

MGNREGS_PATH = "data/processed/mgnregs_clean.csv"
PLFS_PATH    = "data/processed/plfs_wages_clean.csv"
OUT_PATH     = "data/processed/panel_final.csv"

FUZZY_THRESHOLD = 80   # minimum match score (0-100) to accept


def fuzzy_match_states(mgnregs_states, plfs_states):
    """
    Build a mapping from MGNREGS state names to PLFS state names
    using fuzzy string matching.
    Returns a dict {mgnregs_state: plfs_state}.
    """
    mapping = {}
    plfs_list = list(plfs_states)
    for s in mgnregs_states:
        result = fuzz_process.extractOne(s, plfs_list)
        if result and result[1] >= FUZZY_THRESHOLD:
            mapping[s] = result[0]
        else:
            mapping[s] = None
            print(f"  WARNING: No fuzzy match for state '{s}' (best score: "
                  f"{result[1] if result else 'N/A'})")
    return mapping


def merge_panel():
    # 1. Load cleaned data
    if not os.path.exists(MGNREGS_PATH):
        print(f"ERROR: {MGNREGS_PATH} not found. Run clean_mgnregs.py first.")
        return
    if not os.path.exists(PLFS_PATH):
        print(f"ERROR: {PLFS_PATH} not found. Run clean_plfs.py first.")
        return

    mgnregs = pd.read_csv(MGNREGS_PATH)
    wages   = pd.read_csv(PLFS_PATH)

    print(f"MGNREGS shape: {mgnregs.shape}")
    print(f"PLFS wages shape: {wages.shape}")

    # 2. Fuzzy-match state names across datasets
    print("\nMatching state names (MGNREGS → PLFS)...")
    mgnregs_states = mgnregs["state_clean"].dropna().unique()
    plfs_states    = wages["state_clean"].dropna().unique()
    state_map = fuzzy_match_states(mgnregs_states, plfs_states)

    # Show the mapping for review
    print("\nState name mapping:")
    for k, v in sorted(state_map.items()):
        flag = "✓" if v else "✗ UNMATCHED"
        print(f"  {flag}  '{k}' → '{v}'")

    # Apply mapping
    mgnregs["state_plfs"] = mgnregs["state_clean"].map(state_map)

    # 3. Merge: MGNREGS (district × year) LEFT JOIN wages (state × year)
    panel = mgnregs.merge(
        wages.rename(columns={"state_clean": "state_plfs"}),
        on=["state_plfs", "year"],
        how="left"
    )

    # 4. Log transformations
    panel["log_wage"] = np.log(panel["wage_rural_agri"].clip(lower=1))
    panel["log_exp_lakh"] = np.log1p(panel["total_exp_lakh"])   # log(1 + x)

    # 5. Sort and clean
    panel = panel.sort_values(["state_clean", "district_clean", "year"])

    # 6. Diagnostics
    n_district_years = len(panel)
    n_districts = panel["district_clean"].nunique()
    n_years = panel["year"].nunique()
    wage_coverage = panel["wage_rural_agri"].notna().mean()

    print(f"\nFinal panel:")
    print(f"  Observations      : {n_district_years}")
    print(f"  Districts         : {n_districts}")
    print(f"  Years             : {sorted(panel['year'].unique())}")
    print(f"  Wage coverage     : {100*wage_coverage:.1f}%")

    missing_wage = panel[panel["wage_rural_agri"].isna()]["state_clean"].unique()
    if len(missing_wage) > 0:
        print(f"\n  States with missing wage data: {list(missing_wage)}")
        print("  → These states had no PLFS match. Check state name mapping above.")

    # 7. Save
    final_cols = [
        "district_clean", "state_clean", "year",
        "total_exp_lakh", "person_days", "exp_per_pd",
        "wage_rural_agri", "log_wage", "log_exp_lakh"
    ]
    final_cols = [c for c in final_cols if c in panel.columns]
    panel[final_cols].to_csv(OUT_PATH, index=False)
    print(f"\nSaved panel to: {OUT_PATH}")
    print(panel[final_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    merge_panel()
