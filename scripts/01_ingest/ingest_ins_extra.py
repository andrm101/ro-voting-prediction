"""
Ingest additional INS (National Statistics) datasets into a per-county feature table.

Sources (data/raw/ins/):
  ins_emigranti_permanenti_judete_91-24.csv.csv  → permanent emigrants 2022
  ins_emigranti_temporari_judete_12-24.csv.csv   → temporary (abroad) emigrants 2022
  ins_invatamant_judete_90-24.csv.csv            → school units 2022
  ins_unitati-sanatate_judete_90-24.csv.csv      → hospitals 2022
  ins_unitati-turism_judete_90-25.csv.csv        → tourism accommodation units 2022
  ins_condamnari_judete_90-24.csv.csv            → criminal convictions 2022
  ins_gospodarii_2021.csv.xlsx                   → household size from 2021 census

Output: data/processed/ins_extra_clean.parquet
  One row per SIRUTA (42 counties). Raw counts; normalisation by population done
  in build_features.py which has access to pop_total from Eurostat.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import JUDET_SIRUTA

RAW   = ROOT / "data" / "raw" / "ins"
OUT   = ROOT / "data" / "processed" / "ins_extra_clean.parquet"

BASELINE_YEAR = 2022
CENSUS_YEAR   = 2021


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _nfkd(text: str) -> str:
    """Strip diacritics, uppercase — for fuzzy geo-name matching."""
    nfkd = unicodedata.normalize("NFKD", str(text).upper())
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").strip()


# Build normalised-name → SIRUTA lookup (42 units)
_GEO_LOOKUP: dict[str, int] = {_nfkd(k): v for k, v in JUDET_SIRUTA.items()}
# Remove duplicates introduced by "București" / "Municipiul București" alias
_GEO_LOOKUP[_nfkd("Municipiul București")] = 179


def _map_geo(raw_name: str) -> int | None:
    """Map a raw INS county name to SIRUTA. Returns None if no match."""
    return _GEO_LOOKUP.get(_nfkd(raw_name))


# ---------------------------------------------------------------------------
# Common CSV parser (long-format INS series)
# ---------------------------------------------------------------------------

def _parse_csv(
    path: Path,
    year: int,
    *,
    cat_col: str | None = None,
    cat_value: str | None = None,
    prop_col: str | None = None,
    sex_col: str | None = None,
    age_col: str | None = None,
    medii_col: str | None = None,
    medii_value: str | None = None,
) -> dict[int, float]:
    """
    Read a long-format INS CSV and return {siruta: value} for the given year.

    Filters applied in order:
      1. year == year
      2. cat_col == cat_value (if provided)
      3. prop_col == 'Total' (if provided)
      4. medii_col == medii_value (if provided)
    Then aggregates over sex and/or age by summing (no 'Total' rows for those dims).
    Finally maps county names to SIRUTA.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    geo_col = next(c for c in df.columns if "Macroregiuni" in c)
    val_col = next(c for c in df.columns if "Valoare" in c)
    yr_col  = next(c for c in df.columns if "Ani" in c)

    # Strip all string columns
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Year filter
    df = df[df[yr_col] == f"Anul {year}"].copy()

    # Category filter
    if cat_col and cat_value is not None:
        df = df[df[cat_col] == cat_value]

    # Ownership/form filter (always 'Total' when present)
    if prop_col:
        df = df[df[prop_col] == "Total"]

    # Residence medium filter
    if medii_col and medii_value:
        df = df[df[medii_col] == medii_value]

    # Convert value to numeric; drop missing
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    # Group-by geo (summing over sex, age, or other remaining dims)
    group_cols = [geo_col]
    if sex_col and sex_col in df.columns:
        pass  # left out of group → summed
    agg = df.groupby(geo_col)[val_col].sum()

    # Map to SIRUTA
    result: dict[int, float] = {}
    for name, value in agg.items():
        siruta = _map_geo(name)
        if siruta is not None:
            result[siruta] = value

    return result


# ---------------------------------------------------------------------------
# Gospodarii XLSX (2021 Census, Tab 6.1)
# ---------------------------------------------------------------------------

def _parse_gospodarii(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    """
    Extract average household size and total census population per county.
    County rows are identifiable by exactly 3 leading spaces in column 0.
    Returns ({siruta: avg_household_size}, {siruta: pop_census}).
    """
    df = pd.read_excel(path, sheet_name="TAB 6.1", header=None)

    avg_hh: dict[int, float] = {}
    pop:    dict[int, float] = {}
    for _, row in df.iterrows():
        name_raw = row.iloc[0]
        if not isinstance(name_raw, str):
            continue
        # County rows have 3 leading spaces (region=2, macroregion=1)
        if not name_raw.startswith("   ") or name_raw.startswith("    "):
            continue
        name = name_raw.strip()
        if not name:
            continue
        siruta = _map_geo(name)
        if siruta is None:
            continue
        # Column 1 = total households, column 2 = total persons
        try:
            n_hh      = float(row.iloc[1])
            n_persons = float(row.iloc[2])
            if n_hh > 0:
                avg_hh[siruta] = n_persons / n_hh
            if n_persons > 0:
                pop[siruta] = n_persons
        except (TypeError, ValueError):
            continue
    return avg_hh, pop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: dict[int, dict[str, float]] = {s: {} for s in set(JUDET_SIRUTA.values())}

    print("Parsing permanent emigrants (2022) ...")
    r = _parse_csv(
        RAW / "ins_emigranti_permanenti_judete_91-24.csv.csv",
        BASELINE_YEAR,
        sex_col="Sexe",   # M+F summed (no Total row)
    )
    print(f"  {len(r)} counties matched")
    for s, v in r.items():
        all_rows[s]["emigrants_permanent"] = v

    print("Parsing temporary (abroad) emigrants (2022) ...")
    r = _parse_csv(
        RAW / "ins_emigranti_temporari_judete_12-24.csv.csv",
        BASELINE_YEAR,
        sex_col="Sexe",
        age_col="Varste si grupe de varsta",  # all age groups summed
        medii_col="Medii de rezidenta",
        medii_value="Total",
    )
    print(f"  {len(r)} counties matched")
    for s, v in r.items():
        all_rows[s]["emigrants_temporary"] = v

    print("Parsing school units (2022) ...")
    r = _parse_csv(
        RAW / "ins_invatamant_judete_90-24.csv.csv",
        BASELINE_YEAR,
        cat_col="Categorii de unitati de invatamant",
        cat_value="Total",
        prop_col="Forme de proprietate",
    )
    print(f"  {len(r)} counties matched")
    for s, v in r.items():
        all_rows[s]["school_units"] = v

    print("Parsing hospitals (2022) ...")
    r = _parse_csv(
        RAW / "ins_unitati-sanatate_judete_90-24.csv.csv",
        BASELINE_YEAR,
        cat_col="Categorii de unitati sanitare",
        cat_value="Spitale",
        prop_col="Forme de proprietate",
    )
    print(f"  {len(r)} counties matched")
    for s, v in r.items():
        all_rows[s]["hospitals"] = v

    print("Parsing tourism units (2022) ...")
    r = _parse_csv(
        RAW / "ins_unitati-turism_judete_90-25.csv.csv",
        BASELINE_YEAR,
        cat_col="Tipuri de structuri de primire turistica",
        cat_value="Total",
    )
    print(f"  {len(r)} counties matched")
    for s, v in r.items():
        all_rows[s]["tourism_units"] = v

    print("Parsing criminal convictions (2022) ...")
    r = _parse_csv(
        RAW / "ins_condamnari_judete_90-24.csv.csv",
        BASELINE_YEAR,
    )
    print(f"  {len(r)} counties matched")
    for s, v in r.items():
        all_rows[s]["convictions"] = v

    print("Parsing household size + census population (2021 Census) ...")
    r_hh, r_pop = _parse_gospodarii(RAW / "ins_gospodarii_2021.csv.xlsx")
    print(f"  {len(r_hh)} counties matched (household size), {len(r_pop)} (population)")
    for s, v in r_hh.items():
        all_rows[s]["avg_household_size"] = v
    for s, v in r_pop.items():
        all_rows[s]["pop_census"] = v

    df = pd.DataFrame.from_dict(all_rows, orient="index").reset_index()
    df.columns = ["siruta"] + [c for c in df.columns if c != "index"]
    df = df.sort_values("siruta").reset_index(drop=True)

    print(f"\nCoverage ({len(df)} counties):")
    for col in df.columns:
        if col == "siruta":
            continue
        n = df[col].notna().sum()
        print(f"  {col:35s}: {n}/42")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
