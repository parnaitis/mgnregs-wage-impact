"""
src/clean_mgnregs.py
--------------------
Cleans and standardises raw MGNREGS district-level expenditure data
downloaded from nreganarep.nic.in MIS reports.

Input  : data/raw/mgnregs_FY_XXXX.csv  (one file per financial year)
Output : data/processed/mgnregs_clean.csv  (long-format panel)

Columns in output:
    district_clean  : standardised district name (lowercase, stripped)
    state_clean     : standardised state name
    year            : calendar year (FY 2014-15 → 2014)
    total_exp_lakh  : total MGNREGS expenditure (₹ lakhs)
    person_days     : total person-days generated
    exp_per_pd      : expenditure per person-day (₹)
"""

import os
import glob
import pandas as pd
import numpy as np
import re

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/mgnregs_clean.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standardise_name(s: str) -> str:
    """Lowercase, strip, remove special chars, collapse whitespace."""
    if not isinstance(s, str):
        return np.nan
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)          # remove punctuation
    s = re.sub(r"\s+", " ", s)             # collapse spaces
    # Common abbreviation expansions
    s = s.replace("dt ", "district ")
    s = s.replace(" dt", " district")
    return s


def fy_to_year(fy_str: str) -> int:
    """
    Convert financial year string to start calendar year.
    '2014-15' → 2014, '2014-2015' → 2014, '201415' → 2014
    """
    fy_str = str(fy_str).strip()
    match = re.search(r"(\d{4})", fy_str)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse financial year: {fy_str}")


def infer_year_from_filename(filepath: str) -> int:
    """
    Fallback: extract year from filename like mgnregs_FY_2014.csv
    or mgnregs_2014-15.csv
    """
    basename = os.path.basename(filepath)
    match = re.search(r"(\d{4})", basename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer year from filename: {basename}")


# ---------------------------------------------------------------------------
# Column name normalisation map
# Different years/exports have different column names — we map them all.
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    # District
    "district": "district",
    "district name": "district",
    "dist name": "district",
    "district_name": "district",

    # State
    "state": "state",
    "state name": "state",
    "state_name": "state",

    # Expenditure
    "total expenditure (in lakhs)": "total_exp_lakh",
    "total expenditure": "total_exp_lakh",
    "expenditure (lakh)": "total_exp_lakh",
    "exp (lakh)": "total_exp_lakh",
    "total exp": "total_exp_lakh",
    "total_expenditure": "total_exp_lakh",
    "expenditure": "total_exp_lakh",

    # Person-days
    "total persondays generated": "person_days",
    "person days": "person_days",
    "persondays": "person_days",
    "total person days": "person_days",
    "person_days_generated": "person_days",

    # Financial year (if present as column)
    "financial year": "fin_year",
    "year": "fin_year",
    "fy": "fin_year",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using the map above."""
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_single_file(filepath: str) -> pd.DataFrame:
    """Load one raw MGNREGS file, normalise, add year."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        # Try UTF-8, fall back to latin-1 (common in govt exports)
        try:
            df = pd.read_csv(filepath, encoding="utf-8", skiprows=0)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1", skiprows=0)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        print(f"  Skipping unsupported file type: {filepath}")
        return pd.DataFrame()

    df = normalise_columns(df)

    # Determine year
    if "fin_year" in df.columns and df["fin_year"].notna().any():
        year_val = df["fin_year"].dropna().iloc[0]
        year = fy_to_year(year_val)
    else:
        year = infer_year_from_filename(filepath)

    df["year"] = year
    return df


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_mgnregs():
    # 1. Find all raw files
    patterns = [
        os.path.join(RAW_DIR, "mgnregs*.csv"),
        os.path.join(RAW_DIR, "mgnregs*.xlsx"),
        os.path.join(RAW_DIR, "mgnregs*.xls"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        print("No MGNREGS raw files found in data/raw/")
        print("Expected filenames like: mgnregs_FY_2014.csv")
        return

    print(f"Found {len(files)} raw file(s): {[os.path.basename(f) for f in files]}")

    # 2. Load and concatenate
    dfs = []
    for f in sorted(files):
        print(f"  Loading: {os.path.basename(f)}")
        df = load_single_file(f)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No data loaded. Check file formats.")
        return

    data = pd.concat(dfs, ignore_index=True)
    print(f"\nRaw combined shape: {data.shape}")

    # 3. Check required columns
    required = ["district", "state", "year"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        print(f"\nERROR: Missing required columns after normalisation: {missing}")
        print(f"Columns found: {list(data.columns)}")
        print("Check COLUMN_MAP in this script and add the column names from your files.")
        return

    # 4. Standardise district and state names
    data["district_clean"] = data["district"].apply(standardise_name)
    data["state_clean"] = data["state"].apply(standardise_name)

    # 5. Numeric coercion
    for col in ["total_exp_lakh", "person_days"]:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # 6. Derived column: expenditure per person-day
    if "total_exp_lakh" in data.columns and "person_days" in data.columns:
        # ₹ lakhs × 1e5 / person_days = ₹ per person-day
        data["exp_per_pd"] = np.where(
            (data["person_days"] > 0) & data["person_days"].notna(),
            (data["total_exp_lakh"] * 1e5) / data["person_days"],
            np.nan
        )

    # 7. Drop duplicates
    id_cols = ["district_clean", "state_clean", "year"]
    before = len(data)
    data = data.drop_duplicates(subset=id_cols, keep="first")
    print(f"Dropped {before - len(data)} duplicate rows (same district + year).")

    # 8. Drop rows where district or state is null
    data = data.dropna(subset=["district_clean", "state_clean"])
    data = data[data["district_clean"].str.len() > 0]

    # 9. Select and order final columns
    keep_cols = ["district_clean", "state_clean", "year"]
    for c in ["total_exp_lakh", "person_days", "exp_per_pd"]:
        if c in data.columns:
            keep_cols.append(c)
    data = data[keep_cols].sort_values(["state_clean", "district_clean", "year"])

    # 10. Summary diagnostics
    print(f"\nCleaned dataset shape: {data.shape}")
    print(f"Years covered: {sorted(data['year'].unique())}")
    print(f"States: {data['state_clean'].nunique()}")
    print(f"Districts: {data['district_clean'].nunique()}")
    if "total_exp_lakh" in data.columns:
        missing_exp = data["total_exp_lakh"].isna().sum()
        print(f"Missing expenditure values: {missing_exp} ({100*missing_exp/len(data):.1f}%)")

    # 11. Save
    os.makedirs("data/processed", exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to: {OUT_PATH}")
    print("\nSample rows:")
    print(data.head(10).to_string(index=False))


if __name__ == "__main__":
    clean_mgnregs()
