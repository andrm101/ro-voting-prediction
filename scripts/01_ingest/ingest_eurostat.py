"""
Ingest Eurostat bulk-download TSV files for Romanian NUTS 3 indicators.

Eurostat bulk downloads follow the format:
  <freq,unit,...,geo>\time\t<year1>\t<year2>\t...
  Values may carry quality flags: "12345 p", "12345 e", ": " (missing)

All indicators pinned to baseline year 2022 (most recent complete year
before the 2024 elections). Missing 2022 values fall back to 2021, then 2023,
in that order — documented in the output.

Expected files in data/raw/eurostat/ (see README.md for dataset codes):
  nama_10r_2gdp_RO_*.tsv / .csv
  lfst_r_lfu3rt_RO_*.tsv / .csv
  edat_lfse_04_RO_*.tsv / .csv
  edat_lfse_14_RO_*.tsv / .csv
  ilc_peps13_RO_*.tsv / .csv
  isoc_r_iuse_i_RO_*.tsv / .csv
  demo_r_d2jan_RO_*.tsv / .csv
  demo_r_pjanaggr3_RO_*.tsv / .csv
  demo_r_gind3_RO_*.tsv / .csv
  nama_10r_3empers_RO_*.tsv / .csv

Output: data/processed/eurostat_raw.parquet
  Columns: geo | nuts_level | dataset | year_used | value | flag | fallback_used
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "eurostat"
OUT_DIR = ROOT / "data" / "processed"

BASELINE_YEAR = 2022
FALLBACK_YEARS = [2021, 2023]

# Codes we recognise; script will also process any file it can identify
KNOWN_DATASETS: dict[str, str] = {
    "nama_10r_2gdp": "gdp_per_capita_nuts2",
    "nama_10r_3gdp": "gdp_per_capita_nuts3",
    "lfst_r_lfu3rt": "unemployment_rate",
    "edat_lfse_04": "tertiary_education_pct",
    "edat_lfse_14": "early_school_leaving_pct",
    "ilc_peps13": "poverty_risk_pct",
    "isoc_r_iuse_i": "internet_use_pct",
    "demo_r_d2jan": "population_density",
    "demo_r_pjanaggr3": "population_by_age",
    "demo_r_gind3": "net_migration_rate",
    "nama_10r_3empers": "employment_by_sector",
}

RO_NUTS3_PATTERN = re.compile(r"^RO\d{3}$")
RO_NUTS2_PATTERN = re.compile(r"^RO\d{2}$")


def _identify_dataset(path: Path) -> str:
    """Match file stem to a known dataset code."""
    for code, label in KNOWN_DATASETS.items():
        if code.lower() in path.stem.lower():
            return label
    return path.stem.split("_RO")[0].lower()


def _strip_flag(val: str) -> tuple[float | None, str | None]:
    """Split '12345.6 p' into (12345.6, 'p'); ':' into (None, 'missing')."""
    val = str(val).strip()
    if val in (":", "", "n.a.", "na", "N/A"):
        return None, "missing"
    match = re.match(r"^([\d.,]+)\s*([a-z]?)$", val)
    if match:
        numeric_str = match.group(1).replace(",", ".")
        flag = match.group(2) or None
        try:
            return float(numeric_str), flag
        except ValueError:
            return None, "parse_error"
    return None, "parse_error"


def _parse_eurostat_tsv(path: Path) -> pd.DataFrame | None:
    """
    Parse Eurostat bulk-download TSV into a tidy DataFrame.

    The first column encodes multiple dimensions separated by commas or tabs
    depending on the download format. Year columns are integer headers.
    """
    try:
        raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str,
                          keep_default_na=False)
    except Exception:
        try:
            raw = pd.read_csv(path, sep=",", encoding="utf-8-sig", dtype=str,
                              keep_default_na=False)
        except Exception as exc:
            print(f"  [ERROR] Cannot read {path.name}: {exc}", file=sys.stderr)
            return None

    # First column: "freq,unit,...,geo\time" or similar — geo is last segment
    first_col = raw.columns[0]
    raw[first_col] = raw[first_col].str.strip()

    # Extract geo code from the last comma-separated field in the first column
    raw["geo"] = raw[first_col].str.split(r"[,\\]").str[-1].str.strip()

    # Filter to Romania only
    ro_mask = raw["geo"].str.startswith("RO", na=False)
    raw = raw[ro_mask].copy()
    if raw.empty:
        print(f"  [WARN] No Romanian NUTS codes found in {path.name}", file=sys.stderr)
        return None

    raw["nuts_level"] = raw["geo"].apply(
        lambda g: 3 if RO_NUTS3_PATTERN.match(g)
        else 2 if RO_NUTS2_PATTERN.match(g)
        else 1 if g == "RO" else 0
    )

    # Identify year columns (integer-parseable column names)
    year_cols: dict[int, str] = {}
    for col in raw.columns:
        try:
            year_cols[int(col.strip())] = col
        except ValueError:
            pass

    if not year_cols:
        print(f"  [WARN] No year columns detected in {path.name}", file=sys.stderr)
        return None

    # Extract value for baseline year with fallback
    records: list[dict] = []
    for _, row in raw.iterrows():
        geo = row["geo"]
        nuts_level = row["nuts_level"]
        value = None
        flag = None
        year_used = None
        fallback = False

        for yr in [BASELINE_YEAR] + FALLBACK_YEARS:
            col = year_cols.get(yr)
            if col is None:
                continue
            v, f = _strip_flag(row[col])
            if v is not None:
                value = v
                flag = f
                year_used = yr
                fallback = yr != BASELINE_YEAR
                break

        records.append({
            "geo": geo,
            "nuts_level": nuts_level,
            "value": value,
            "flag": flag,
            "year_used": year_used,
            "fallback_used": fallback,
        })

    return pd.DataFrame(records)


def main() -> None:
    files = (
        sorted(RAW_DIR.glob("*.tsv"))
        + sorted(RAW_DIR.glob("*.csv"))
        + sorted(RAW_DIR.glob("*.gz"))
    )
    # Exclude READMEs
    files = [f for f in files if f.suffix not in (".md", ".txt")]

    if not files:
        print(
            f"No files found in {RAW_DIR}.\n"
            "Download bulk TSV files from https://ec.europa.eu/eurostat/databrowser/\n"
            "and place them in data/raw/eurostat/ following the naming convention in\n"
            "data/raw/eurostat/README.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for f in files:
        print(f"Processing {f.name} …")
        df = _parse_eurostat_tsv(f)
        if df is None or df.empty:
            continue
        dataset_label = _identify_dataset(f)
        df["dataset"] = dataset_label
        missing_pct = df["value"].isna().mean()
        fallback_pct = df["fallback_used"].mean()
        print(f"  [OK] {f.name} → {len(df)} rows | "
              f"missing: {missing_pct:.0%} | fallback to non-2022: {fallback_pct:.0%}")
        if missing_pct > 0.3:
            print(f"  [WARN] High missingness ({missing_pct:.0%}) in {dataset_label} — "
                  "check filter criteria or download a broader year range.", file=sys.stderr)
        frames.append(df)

    if not frames:
        print("No files successfully parsed.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eurostat_raw.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined):,} rows → {out_path}")
    print(f"Datasets: {sorted(combined['dataset'].unique())}")
    print(f"NUTS levels present: {sorted(combined['nuts_level'].unique())}")
    nuts3_count = combined[combined["nuts_level"] == 3]["geo"].nunique()
    print(f"Distinct NUTS 3 codes (RO): {nuts3_count}")


if __name__ == "__main__":
    main()
