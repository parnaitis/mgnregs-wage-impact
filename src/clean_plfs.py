"""
src/clean_plfs.py
-----------------
Cleans PLFS (Periodic Labour Force Survey) rural agricultural wage data.

PLFS annual reports contain statement tables with average daily wages
by industry × sector (rural/urban) × state. We extract rural agricultural
worker wages and reshape to a district-compatible long-format panel.

NOTE: PLFS data is state-level, not district-level. We will later
      assume wages are uniform within a state-year and merge onto
      the district panel. District-level wage data (if available from
      state Labour Bureaus) can replace this.

Input  : data/raw/plfs_YYYY.xlsx   (Annual Report Excel, one per year)
Output : data/processed/plfs_wages_clean.csv

Columns in output:
    state_clean     : standardised state name
    year            : survey year (2017, 2018, ...)
    wage_rural_agri : average daily wage (₹), rural agricultural workers
"""

import os
import glob
import re
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/plfs_wages_clean.csv"


def standardise_name(s: str) -> str:
    if not isinstance(s, str):
        return np.nan
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def infer_year_from_filename(filepath: str) -> int:
    basename = os.path.basename(filepath)
    match = re.search(r"(\d{4})", basename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer year from filename: {basename}")


def clean_plfs():
    files = glob.glob(os.path.join(RAW_DIR, "plfs*.xlsx")) + \
            glob.glob(os.path.join(RAW_DIR, "plfs*.xls")) + \
            glob.glob(os.path.join(RAW_DIR, "plfs*.csv"))

    if not files:
        print("No PLFS raw files found in data/raw/")
        print("Expected filenames like: plfs_2018.xlsx")
        return

    print(f"Found {len(files)} PLFS file(s).")

    dfs = []
    for f in sorted(files):
        year = infer_year_from_filename(f)
        print(f"  Loading: {os.path.basename(f)}  (year={year})")

        ext = os.path.splitext(f)[1].lower()
        try:
            if ext in (".xlsx", ".xls"):
                # PLFS reports embed wage tables — try reading all sheets,
                # find the one with 'wage' or 'earning' in column headers.
                xl = pd.ExcelFile(f, engine="openpyxl")
                target_df = None
                for sheet in xl.sheet_names:
                    tmp = xl.parse(sheet, header=None)
                    flat = " ".join(tmp.astype(str).values.flatten()).lower()
                    if "wage" in flat or "earning" in flat:
                        target_df = tmp
                        break
                if target_df is None:
                    print(f"    WARNING: No wage table found in {os.path.basename(f)}")
                    continue
                df = target_df
            else:
                df = pd.read_csv(f, header=None)

            # Heuristic: find header row containing 'state' and 'wage'/'earning'
            header_row = None
            for i, row in df.iterrows():
                row_str = " ".join(str(v).lower() for v in row.values)
                if "state" in row_str and ("wage" in row_str or "earning" in row_str):
                    header_row = i
                    break

            if header_row is not None:
                df.columns = df.iloc[header_row]
                df = df.iloc[header_row + 1:].reset_index(drop=True)
                df.columns = [str(c).lower().strip() for c in df.columns]

                # Find state column
                state_col = next((c for c in df.columns if "state" in c), None)
                # Find rural agricultural wage column
                # PLFS typically labels this as "rural" + "agriculture" or similar
                wage_col = next(
                    (c for c in df.columns
                     if ("rural" in c or "agri" in c) and ("wage" in c or "earn" in c)),
                    None
                )
                if wage_col is None:
                    # Fallback: take the second numeric column
                    numeric_cols = [c for c in df.columns if c != state_col and
                                    pd.to_numeric(df[c], errors="coerce").notna().sum() > 5]
                    if numeric_cols:
                        wage_col = numeric_cols[0]

                if state_col and wage_col:
                    out = pd.DataFrame({
                        "state_clean": df[state_col].apply(standardise_name),
                        "year": year,
                        "wage_rural_agri": pd.to_numeric(
                            df[wage_col].astype(str).str.replace(",", "").str.strip(),
                            errors="coerce"
                        )
                    })
                    out = out.dropna(subset=["state_clean", "wage_rural_agri"])
                    out = out[out["state_clean"].str.len() > 2]
                    dfs.append(out)
                    print(f"    Extracted {len(out)} state rows.")
                else:
                    print(f"    WARNING: Could not find state/wage columns. "
                          f"Columns found: {list(df.columns)[:10]}")
            else:
                print(f"    WARNING: Could not find header row in {os.path.basename(f)}")

        except Exception as e:
            print(f"    ERROR loading {os.path.basename(f)}: {e}")
            continue

    if not dfs:
        print("\nNo data extracted. See warnings above.")
        print("You may need to manually inspect the PLFS Excel files and adjust the")
        print("header detection logic in this script.")
        return

    wages = pd.concat(dfs, ignore_index=True)
    wages = wages.drop_duplicates(subset=["state_clean", "year"], keep="first")
    wages = wages.sort_values(["state_clean", "year"])

    print(f"\nCleaned PLFS shape: {wages.shape}")
    print(f"Years: {sorted(wages['year'].unique())}")
    print(f"States: {wages['state_clean'].nunique()}")
    print(f"Sample:\n{wages.head(10).to_string(index=False)}")

    os.makedirs("data/processed", exist_ok=True)
    wages.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    clean_plfs()
