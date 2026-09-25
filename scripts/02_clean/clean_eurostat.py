"""
Clean Eurostat indicators into a wide NUTS 3 feature matrix.

Input:  data/processed/eurostat_raw.parquet
        data/raw/eurostat/*.tsv  (re-read for multi-dimensional datasets)
Output: data/processed/eurostat_clean.parquet

Strategy:
  - Single-indicator datasets (GDP, unemployment, poverty, internet, education,
    density): pivot directly from eurostat_raw.parquet.
  - Multi-dimensional datasets (population by age, employment by sector):
    re-parse raw TSV with explicit dimension filters to extract the required
    slices (e.g. age group Y_GE65, NACE sector A for agriculture).
  - NUTS 2 fill-down: for indicators only available at NUTS 2, broadcast the
    NUTS 2 value to all constituent NUTS 3 counties. Flag filled rows.
  - Join all indicators on NUTS 3 code; attach SIRUTA for downstream joins.

Output columns:
  geo_nuts3 | siruta | nuts2_code
  gdp_per_capita | unemployment_rate | poverty_risk_pct
  tertiary_education_pct | early_school_leaving_pct | internet_use_pct
  population_density | net_migration_rate
  pct_pop_over65 | pct_pop_under25
  pct_employment_agriculture | pct_employment_industry | pct_employment_services
  source_nuts_level_<indicator>  (2 or 3, for transparency)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import NUTS2_NUTS3, NUTS3_NUTS2, NUTS3_SIRUTA

RAW_EUROSTAT = ROOT / "data" / "raw" / "eurostat"
IN_PATH  = ROOT / "data" / "processed" / "eurostat_raw.parquet"
OUT_PATH = ROOT / "data" / "processed" / "eurostat_clean.parquet"

BASELINE_YEAR = 2022
FALLBACK_YEARS = [2021, 2023]

ALL_NUTS3_RO = sorted(NUTS3_SIRUTA.keys())

# ---------------------------------------------------------------------------
# Single-indicator dataset → output column name mapping
# These are straightforward: one value per geo per year.
# ---------------------------------------------------------------------------
SIMPLE_DATASET_MAP: dict[str, str] = {
    "gdp_per_capita_nuts2":        "gdp_per_capita",
    "gdp_per_capita_nuts3":        "gdp_per_capita",
    "unemployment_rate":           "unemployment_rate",
    "poverty_risk_pct":            "poverty_risk_pct",
    "tertiary_education_pct":      "tertiary_education_pct",
    "early_school_leaving_pct":    "early_school_leaving_pct",
    "internet_use_pct":            "internet_use_pct",
    "population_density":          "population_density",
    "net_migration_rate":          "net_migration_rate",
}

# ---------------------------------------------------------------------------
# Multi-dimensional dataset specs
# Dimension filter: substring patterns against the full first-column value.
# ---------------------------------------------------------------------------
class MultiDimSpec:
    def __init__(
        self,
        dataset_code: str,
        dimension_filters: dict[str, str],  # dim_position_pattern: value_pattern
        output_col: str,
        value_is_share: bool = False,       # if True, value is already a %; else count
    ):
        self.dataset_code = dataset_code
        self.dimension_filters = dimension_filters
        self.output_col = output_col
        self.value_is_share = value_is_share


MULTIDIM_SPECS: list[MultiDimSpec] = [
    # Population ≥65: demo_r_pjanaggr3 — Romania has Y_GE65, sex=T, unit=NR
    MultiDimSpec(
        "demo_r_pjanaggr3",
        {"age": "Y_GE65", "sex": "T", "unit": "NR"},
        "pop_over65_count",
    ),
    # Population <15 (best available proxy for young population in this dataset)
    MultiDimSpec(
        "demo_r_pjanaggr3",
        {"age": "Y_LT15", "sex": "T", "unit": "NR"},
        "pop_under25_count",
    ),
    # Total population: age=TOTAL
    MultiDimSpec(
        "demo_r_pjanaggr3",
        {"age": "TOTAL", "sex": "T", "unit": "NR"},
        "pop_total",
    ),
    # Employment in agriculture (NACE A): nama_10r_3empers — unit=THS, wstatus=EMP
    MultiDimSpec(
        "nama_10r_3empers",
        {"nace_r2": r"^A$", "unit": "THS", "wstatus": "EMP"},
        "emp_agriculture_ths",
    ),
    # Total employment
    MultiDimSpec(
        "nama_10r_3empers",
        {"nace_r2": r"^TOTAL$", "unit": "THS", "wstatus": "EMP"},
        "emp_total_ths",
    ),
    # Industry (NACE B-E: mining, manufacturing, utilities — Romania aggregate available as B-E)
    MultiDimSpec(
        "nama_10r_3empers",
        {"nace_r2": r"^B-E$", "unit": "THS", "wstatus": "EMP"},
        "emp_industry_ths",
    ),
    # Services: NACE G-I or G-J as available in Romania
    MultiDimSpec(
        "nama_10r_3empers",
        {"nace_r2": r"^G-[IJ]$", "unit": "THS", "wstatus": "EMP"},
        "emp_services_ths",
    ),
]


# Additional simple indicators sourced directly from TSV (NUTS 2 → NUTS 3 fill-down)
# These are not in eurostat_raw.parquet (different dataset codes from original download).
EXTRA_SIMPLE_SPECS: list[tuple[str, dict, str]] = [
    # Early school leaving at NUTS 2 (edat_lfse_16), sex=T, age=Y18-24, unit=PC
    ("edat_lfse_16", {"sex": "T", "age": "Y18-24", "unit": "PC"}, "early_school_leaving_pct"),
    # Poverty risk at NUTS 2 (ilc_peps11), unit=PC
    ("ilc_peps11", {"unit": "PC"}, "poverty_risk_pct"),
]


def _strip_flag(val: str) -> float | None:
    val = str(val).strip()
    if val in (":", "", "n.a.", "na", "N/A"):
        return None
    m = re.match(r"^([\d.,]+)\s*[a-z]?$", val)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def _parse_multidim_tsv(dataset_code: str, dim_filters: dict[str, str]) -> pd.DataFrame | None:
    """
    Re-parse a raw Eurostat TSV and return (geo, value) for the specified
    dimension slice at the baseline year with fallback.
    """
    candidates = list(RAW_EUROSTAT.glob(f"*{dataset_code}*"))
    if not candidates:
        print(f"  [WARN] No raw TSV found for {dataset_code}", file=sys.stderr)
        return None

    path = candidates[0]
    try:
        raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig",
                          dtype=str, keep_default_na=False)
    except Exception as exc:
        print(f"  [ERROR] Cannot read {path.name}: {exc}", file=sys.stderr)
        return None

    raw.columns = [c.strip() for c in raw.columns]
    first_col = raw.columns[0]

    # Dimension names from first column header: "freq,sex,age,unit,geo\time"
    dim_names = re.split(r"[,\\]", first_col.lower().replace(r"\time", ""))

    # Filter rows by dimension values
    mask = pd.Series(True, index=raw.index)
    for dim_name, value_pattern in dim_filters.items():
        # Find which position this dimension occupies
        try:
            pos = next(i for i, d in enumerate(dim_names) if dim_name in d)
        except StopIteration:
            print(f"  [WARN] Dimension '{dim_name}' not found in {path.name} "
                  f"header: {dim_names}", file=sys.stderr)
            continue
        dim_values = raw[first_col].str.split(r"[,\\]").str[pos].str.strip()
        mask &= dim_values.str.match(value_pattern, na=False)

    filtered = raw[mask].copy()
    if filtered.empty:
        print(f"  [WARN] No rows matched dimension filter {dim_filters} in {path.name}",
              file=sys.stderr)
        return None

    # Extract geo (last segment of first column)
    filtered["geo"] = filtered[first_col].str.split(r"[,\\]").str[-1].str.strip()
    filtered = filtered[filtered["geo"].str.match(r"^RO", na=False)]

    # Extract year columns
    year_cols: dict[int, str] = {}
    for col in filtered.columns:
        try:
            year_cols[int(col.strip())] = col
        except ValueError:
            pass

    # All available years sorted descending (most recent first) as last-resort fallback
    all_years_desc = sorted(year_cols.keys(), reverse=True)
    preferred_years = [BASELINE_YEAR] + FALLBACK_YEARS

    records: list[dict] = []
    for _, row in filtered.iterrows():
        geo = row["geo"]
        value = None
        for yr in preferred_years:
            col = year_cols.get(yr)
            if col is None:
                continue
            v = _strip_flag(row[col])
            if v is not None:
                value = v
                break
        if value is None:
            # Fall back to most recent year with actual data
            for yr in all_years_desc:
                if yr in preferred_years:
                    continue
                col = year_cols.get(yr)
                if col is None:
                    continue
                v = _strip_flag(row[col])
                if v is not None:
                    value = v
                    break
        records.append({"geo": geo, "value": value})

    df_out = pd.DataFrame(records).dropna(subset=["geo"])
    # Keep first non-null value per geo in case multiple dimension combos match
    df_out = (df_out
              .sort_values("value", na_position="last")
              .drop_duplicates(subset=["geo"], keep="first"))
    return df_out


def _build_simple_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract single-value indicators from eurostat_raw.parquet.
    Returns a DataFrame indexed on geo with one column per indicator.
    Prefer NUTS 3 values; fill NUTS 2 → NUTS 3 where needed.
    """
    frames: dict[str, pd.DataFrame] = {}

    for dataset_label, col_name in SIMPLE_DATASET_MAP.items():
        subset = raw[raw["dataset"] == dataset_label][["geo", "nuts_level", "value"]].copy()
        if subset.empty:
            continue

        nuts3 = (subset[subset["nuts_level"] == 3]
                 .dropna(subset=["value"])
                 .groupby("geo")["value"].first())
        nuts2 = (subset[subset["nuts_level"] == 2]
                 .dropna(subset=["value"])
                 .groupby("geo")["value"].first())

        col_data: dict[str, float] = {}
        source_level: dict[str, int] = {}

        for nuts3_code in ALL_NUTS3_RO:
            if nuts3_code in nuts3.index and pd.notna(nuts3[nuts3_code]):
                col_data[nuts3_code] = nuts3[nuts3_code]
                source_level[nuts3_code] = 3
            else:
                # Fill down from NUTS 2
                nuts2_code = NUTS3_NUTS2.get(nuts3_code)
                if nuts2_code and nuts2_code in nuts2.index:
                    col_data[nuts3_code] = nuts2[nuts2_code]
                    source_level[nuts3_code] = 2
                else:
                    col_data[nuts3_code] = np.nan
                    source_level[nuts3_code] = 0

        frames[col_name] = pd.DataFrame({
            "geo": list(col_data.keys()),
            col_name: list(col_data.values()),
            f"source_nuts_level_{col_name}": list(source_level.values()),
        })

    if not frames:
        return pd.DataFrame({"geo": ALL_NUTS3_RO})

    result = pd.DataFrame({"geo": ALL_NUTS3_RO})
    for df in frames.values():
        result = result.merge(df, on="geo", how="left")
    return result


def _build_population_shares(result: pd.DataFrame) -> pd.DataFrame:
    """Derive pct_pop_over65 and pct_pop_under25 from raw counts."""
    specs_needed = ["pop_over65_count", "pop_under25_count", "pop_total"]
    for spec_name in specs_needed:
        spec = next((s for s in MULTIDIM_SPECS if s.output_col == spec_name), None)
        if spec is None:
            continue
        df = _parse_multidim_tsv(spec.dataset_code, spec.dimension_filters)
        if df is not None:
            df = df.rename(columns={"value": spec_name})
            result = result.merge(df, on="geo", how="left")

    for age_band, count_col, share_col in [
        ("over65",  "pop_over65_count",  "pct_pop_over65"),
        ("under25", "pop_under25_count", "pct_pop_under25"),
    ]:
        if count_col in result.columns and "pop_total" in result.columns:
            result[share_col] = result[count_col] / result["pop_total"].replace(0, np.nan)
        result = result.drop(columns=[count_col], errors="ignore")

    result = result.drop(columns=["pop_total"], errors="ignore")
    return result


def _build_employment_shares(result: pd.DataFrame) -> pd.DataFrame:
    """Derive employment shares by sector from NACE breakdowns."""
    emp_cols: dict[str, pd.Series] = {}

    for spec in MULTIDIM_SPECS:
        if "emp_" not in spec.output_col:
            continue
        df = _parse_multidim_tsv(spec.dataset_code, spec.dimension_filters)
        if df is not None:
            df = df.rename(columns={"value": spec.output_col})
            result = result.merge(df, on="geo", how="left")
            emp_cols[spec.output_col] = result[spec.output_col]

    total = result.get("emp_total_ths", pd.Series(np.nan, index=result.index))

    for src_col, share_col in [
        ("emp_agriculture_ths", "pct_employment_agriculture"),
        ("emp_industry_ths",    "pct_employment_industry"),
        ("emp_services_ths",    "pct_employment_services"),
    ]:
        if src_col in result.columns:
            result[share_col] = result[src_col] / total.replace(0, np.nan)
        result = result.drop(columns=[src_col], errors="ignore")

    result = result.drop(columns=["emp_total_ths"], errors="ignore")
    return result


def _build_extra_simple_indicators(result: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch ESL and poverty directly from their TSV files (not in eurostat_raw.parquet).
    Applies NUTS 2 fill-down to NUTS 3.
    """
    for dataset_code, dim_filters, col_name in EXTRA_SIMPLE_SPECS:
        if col_name in result.columns and result[col_name].notna().sum() > 0:
            continue  # already present

        nuts2_df = _parse_multidim_tsv(dataset_code, dim_filters)
        if nuts2_df is None or nuts2_df.empty:
            result[col_name] = np.nan
            continue

        # nuts2_df has geo=NUTS2 code, value=indicator
        nuts2_series = (nuts2_df
                        .dropna(subset=["value"])
                        .groupby("geo")["value"].first())

        values = []
        for geo in result["geo"]:
            nuts2_code = NUTS3_NUTS2.get(geo)
            if nuts2_code and nuts2_code in nuts2_series.index:
                values.append(nuts2_series[nuts2_code])
            else:
                values.append(np.nan)
        result[col_name] = values
        n_filled = sum(v is not np.nan and not pd.isna(v) for v in values)
        print(f"  {col_name}: {n_filled}/42 from NUTS 2 ({dataset_code})")

    return result


# ---------------------------------------------------------------------------
# New: GVA deindustrialisation, long-term unemployment, household income
# ---------------------------------------------------------------------------

def _parse_gva_sector(nace_pattern: str, years: list[int]) -> pd.DataFrame | None:
    """
    Parse nama_10r_3gva for a given NACE category across multiple years.
    Returns DataFrame with columns: geo, gva_<year> for each year requested.
    Unit: CP_MEUR (current prices in million EUR — Eurostat code for this dataset).
    """
    candidates = list(RAW_EUROSTAT.glob("*nama_10r_3gva*"))
    if not candidates:
        print("  [WARN] nama_10r_3gva not found", file=sys.stderr)
        return None

    path = candidates[0]
    try:
        raw = pd.read_csv(path, sep="\t", encoding="utf-8-sig",
                          dtype=str, keep_default_na=False)
    except Exception as exc:
        print(f"  [ERROR] {path.name}: {exc}", file=sys.stderr)
        return None

    raw.columns = [c.strip() for c in raw.columns]
    first_col = raw.columns[0]
    dim_names = re.split(r"[,\\]", first_col.lower().replace(r"\time", ""))

    def _dim(col_vals, name, pattern):
        try:
            pos = next(i for i, d in enumerate(dim_names) if name in d)
        except StopIteration:
            return pd.Series(True, index=col_vals.index)
        vals = col_vals.str.split(r"[,\\]").str[pos].str.strip()
        return vals.str.match(pattern, na=False)

    # Unit code in this dataset is CP_MEUR (current prices, million EUR)
    mask = (
        _dim(raw[first_col], "unit",    r"^CP_MEUR$") &
        _dim(raw[first_col], "nace_r2", nace_pattern)
    )
    filtered = raw[mask].copy()
    if filtered.empty:
        print(f"  [WARN] GVA: no rows for unit=CP_MEUR nace={nace_pattern}", file=sys.stderr)
        return None

    filtered["geo"] = filtered[first_col].str.split(r"[,\\]").str[-1].str.strip()
    filtered = filtered[filtered["geo"].str.match(r"^RO", na=False)]

    year_cols = {}
    for col in filtered.columns:
        try:
            year_cols[int(col.strip())] = col
        except ValueError:
            pass

    records: dict[str, dict] = {}
    for _, row in filtered.iterrows():
        geo = row["geo"]
        records.setdefault(geo, {"geo": geo})
        for yr in years:
            col = year_cols.get(yr)
            if col is None:
                continue
            v = _strip_flag(row[col])
            if v is not None:
                records[geo][f"gva_{yr}"] = v

    df_out = pd.DataFrame(list(records.values()))
    return df_out if not df_out.empty else None


def _build_gva_features(result: pd.DataFrame) -> pd.DataFrame:
    """
    Compute industry GVA share (2022) and deindustrialisation index (2010->2022).

    New columns:
      industry_gva_share      — B-E share of total GVA, 2022
      gva_deindustrial_index  — (industry_share_2010 - industry_share_2022),
                                positive = county lost industrial base
      household_gva_per_cap   — total GVA / population density proxy (removed;
                                use nama_10r_2hhinc instead)
    """
    print("  Parsing GVA total (TOTAL) for 2010 + 2022 ...")
    gva_total = _parse_gva_sector(r"^TOTAL$", years=[2010, 2022])
    print("  Parsing GVA industry (B-E) for 2010 + 2022 ...")
    gva_industry = _parse_gva_sector(r"^B-E$", years=[2010, 2022])

    if gva_total is None or gva_industry is None:
        print("  [WARN] GVA data missing; skipping deindustrialisation features",
              file=sys.stderr)
        result["industry_gva_share"]     = np.nan
        result["gva_deindustrial_index"] = np.nan
        return result

    gva = gva_total.merge(gva_industry, on="geo", how="outer",
                          suffixes=("_total", "_industry"))

    # Shares
    for yr in [2010, 2022]:
        t_col = f"gva_{yr}_total"
        i_col = f"gva_{yr}_industry"
        if t_col in gva.columns and i_col in gva.columns:
            gva[f"ind_share_{yr}"] = (
                gva[i_col] / gva[t_col].replace(0, np.nan)
            )

    result = result.merge(gva[["geo", "ind_share_2022", "ind_share_2010"]],
                          on="geo", how="left")
    result = result.rename(columns={"ind_share_2022": "industry_gva_share"})

    if "ind_share_2010" in result.columns and "industry_gva_share" in result.columns:
        result["gva_deindustrial_index"] = (
            result["ind_share_2010"] - result["industry_gva_share"]
        ).clip(lower=0)  # only positive = actual deindustrialisation
        result = result.drop(columns=["ind_share_2010"])

    filled_share = result["industry_gva_share"].notna().sum()
    filled_dind  = result["gva_deindustrial_index"].notna().sum() if "gva_deindustrial_index" in result.columns else 0
    print(f"  industry_gva_share: {filled_share}/42 NUTS3 filled")
    print(f"  gva_deindustrial_index: {filled_dind}/42 NUTS3 filled")
    return result


def _build_long_term_unemployment(result: pd.DataFrame) -> pd.DataFrame:
    """
    Long-term unemployment rate (12+ months) from lfst_r_lfu2ltu (NUTS2).
    Dimensions: isced11=TOTAL, sex=T, age=Y15-74, unit=PC_ACT.
    Fill-down to NUTS3.
    """
    print("  Parsing lfst_r_lfu2ltu (long-term unemployment, NUTS2) ...")
    # Correct dimension filters from TSV inspection:
    # header = freq,isced11,sex,age,unit,geo  — unit is PC_ACT not PC
    for age_pat in ["Y15-74", "Y15-64", "Y_GE15"]:
        df = _parse_multidim_tsv(
            "lfst_r_lfu2ltu",
            {"isced11": "TOTAL", "sex": "T", "age": age_pat, "unit": "PC_ACT"}
        )
        if df is not None and not df.empty:
            break
    else:
        print("  [WARN] Long-term unemployment: no data found", file=sys.stderr)
        result["long_term_unemployment_rate"] = np.nan
        return result

    nuts2_series = (df.dropna(subset=["value"])
                    .groupby("geo")["value"].first())

    values = []
    for geo in result["geo"]:
        nuts2_code = NUTS3_NUTS2.get(geo)
        if nuts2_code and nuts2_code in nuts2_series.index:
            values.append(nuts2_series[nuts2_code])
        else:
            values.append(np.nan)
    result["long_term_unemployment_rate"] = values
    filled = sum(1 for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))
    print(f"  long_term_unemployment_rate: {filled}/42 (NUTS2 fill-down)")
    return result


def _build_household_income(result: pd.DataFrame) -> pd.DataFrame:
    """
    Household income per capita in PPS from nama_10r_2hhinc (NUTS2).
    Unit: PPS_EU27_2020_HAB, direct=BAL, na_item=B6N (gross disposable income).
    Fill-down to NUTS3.
    """
    print("  Parsing nama_10r_2hhinc (household income PPS, NUTS2) ...")
    # Correct dimension filters: unit=PPS_EU27_2020_HAB, direct=BAL, na_item=B6N
    df = _parse_multidim_tsv(
        "nama_10r_2hhinc",
        {"unit": "PPS_EU27_2020_HAB", "direct": "BAL", "na_item": "B6N"}
    )
    if df is None or df.empty:
        # Fallback: EUR per habitant
        df = _parse_multidim_tsv(
            "nama_10r_2hhinc",
            {"unit": "EUR_HAB", "direct": "BAL", "na_item": "B6N"}
        )
    if df is None or df.empty:
        print("  [WARN] Household income: no data found", file=sys.stderr)
        result["household_income_pps"] = np.nan
        return result

    nuts2_series = (df.dropna(subset=["value"])
                    .groupby("geo")["value"].first())

    values = []
    for geo in result["geo"]:
        nuts2_code = NUTS3_NUTS2.get(geo)
        if nuts2_code and nuts2_code in nuts2_series.index:
            values.append(nuts2_series[nuts2_code])
        else:
            values.append(np.nan)
    result["household_income_pps"] = values
    filled = sum(1 for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))
    print(f"  household_income_pps: {filled}/42 (NUTS2 fill-down)")
    return result


def _report_coverage(df: pd.DataFrame) -> None:
    indicator_cols = [c for c in df.columns
                      if not c.startswith(("geo", "siruta", "nuts", "source_"))]
    print(f"\nCoverage report ({len(df)} NUTS 3 units):")
    for col in indicator_cols:
        filled = df[col].notna().sum()
        nuts2_filled = 0
        src_col = f"source_nuts_level_{col}"
        if src_col in df.columns:
            nuts2_filled = (df[src_col] == 2).sum()
        flag = f" ({nuts2_filled} from NUTS 2 fill-down)" if nuts2_filled else ""
        print(f"  {col:40s}: {filled}/42{flag}")


def main() -> None:
    if not IN_PATH.exists():
        print(f"Input not found: {IN_PATH}\nRun scripts/01_ingest/ingest_eurostat.py first.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading eurostat_raw.parquet ...")
    raw = pd.read_parquet(IN_PATH)
    print(f"  {len(raw):,} rows | datasets: {sorted(raw['dataset'].unique())}")

    print("Building simple indicators ...")
    result = _build_simple_indicators(raw)

    print("Building population age shares (multi-dim) ...")
    result = _build_population_shares(result)

    print("Building employment sector shares (multi-dim) ...")
    result = _build_employment_shares(result)

    print("Building ESL and poverty from regional TSVs ...")
    result = _build_extra_simple_indicators(result)

    print("Building GVA deindustrialisation features (nama_10r_3gva) ...")
    result = _build_gva_features(result)

    print("Building long-term unemployment (lfst_r_lfu2ltu) ...")
    result = _build_long_term_unemployment(result)

    print("Building household income PPS (nama_10r_2hhinc) ...")
    result = _build_household_income(result)

    # Attach SIRUTA and NUTS 2 codes
    result["siruta"]    = result["geo"].map(NUTS3_SIRUTA)
    result["nuts2_code"] = result["geo"].map(NUTS3_NUTS2)
    result = result.rename(columns={"geo": "geo_nuts3"})

    unmatched = result[result["siruta"].isna()]["geo_nuts3"].unique()
    if len(unmatched):
        print(f"  [WARN] No SIRUTA for NUTS3 codes: {list(unmatched)}", file=sys.stderr)

    result = result.sort_values("siruta").reset_index(drop=True)
    _report_coverage(result)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(result)} rows x {len(result.columns)} columns -> {OUT_PATH}")


if __name__ == "__main__":
    main()
