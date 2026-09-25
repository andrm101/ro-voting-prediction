"""
Join all clean sources into a single modelling-ready feature matrix.

Inputs:
  data/processed/eurostat_clean.parquet
  data/processed/census_clean.parquet
  data/processed/elections_clean.parquet
  data/processed/ins_extra_clean.parquet   (optional — skipped if absent)

Output:
  data/processed/features_2022_judet.parquet  — 42 rows × ~80 columns
  app/src/data/features.json                  — app export (no target columns)

Column groups in output:
  Identity      : siruta, judet, geo_nuts3, nuts2_code
  Eurostat      : gdp_per_capita … pct_employment_services
  Census        : pct_maghiari … pct_urban
  INS extra     : emigrants_permanent_rate, emigrants_temporary_rate,
                  school_units_per_10k, hospitals_per_100k,
                  tourism_units_per_10k, convictions_per_100k,
                  avg_household_size
  Composites    : idx_grievance, idx_modernity, idx_demographic_pressure
  Engineered    : ethno_linguistic_frag, religious_frag,
                  rural_elderly, young_urban, agri_emigration,
                  orthodox_poverty, anti_estab_persistence,
                  nationalist_consolidation_ratio
  Sentiment     : ep_2024_anti_estab, georgescu_r1_2024, simion_r1_2025,
                  pres19r2_IOHANNIS  (baseline establishment support)
  Targets       : t_AUR, t_PSD, t_PNL, t_USR, t_UDMR, t_SOS, t_POT,
                  t_ESTABLISHMENT, t_ANTI_ESTABLISHMENT, t_turnout
  Validation    : pres25r2_DAN, pres25r2_SIMION, pres19r2_DANCILA
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import PARLIAMENTARY_MODEL_PARTIES

EURO_PATH  = ROOT / "data" / "processed" / "eurostat_clean.parquet"
CENS_PATH  = ROOT / "data" / "processed" / "census_clean.parquet"
ELEC_PATH  = ROOT / "data" / "processed" / "elections_clean.parquet"
INS_PATH   = ROOT / "data" / "processed" / "ins_extra_clean.parquet"
OUT_PATH   = ROOT / "data" / "processed" / "features_2022_judet.parquet"
APP_PATH   = ROOT / "app" / "src" / "data" / "features.json"

np.random.seed(42)

# Feature columns used in models (all non-target, non-identity columns)
EUROSTAT_FEATURES = [
    "gdp_per_capita", "unemployment_rate", "poverty_risk_pct",
    "tertiary_education_pct", "early_school_leaving_pct", "internet_use_pct",
    "population_density", "net_migration_rate",
    "pct_pop_over65", "pct_pop_under25",
    "pct_employment_agriculture", "pct_employment_industry", "pct_employment_services",
    # New: deindustrialisation thesis
    "industry_gva_share",        # industry's current share of GVA (2022)
    "gva_deindustrial_index",    # (industry_share_2010 - industry_share_2022), > 0 = lost base
    "long_term_unemployment_rate",  # % unemployed 12+ months — structural joblessness
    "household_income_pps",         # gross disposable income per capita, PPS
]
CENSUS_FEATURES = [
    "pct_maghiari", "pct_romi", "pct_romani",
    "pct_orthodox", "pct_reformed", "pct_pentecostal",
    "pct_greco_catholic", "pct_roman_catholic",
    "pct_urban",
    # Added: NW cluster anchors + Bukovina signal
    "pct_germani",    # Transylvanian Saxon/Swabian heritage counties (Sibiu, Timiș, Brașov)
    "pct_ucraineni",  # Northern Bukovina signal (Suceava, Maramureș) — separates NE Moldova
]
COMPOSITE_FEATURES = ["idx_grievance", "idx_modernity", "idx_demographic_pressure"]
INS_EXTRA_FEATURES = [
    "emigrants_permanent_rate", "emigrants_temporary_rate",
    "school_units_per_10k", "hospitals_per_100k",
    "tourism_units_per_10k", "convictions_per_100k",
    "avg_household_size",
]
ENGINEERED_FEATURES = [
    "ethno_linguistic_frag", "religious_frag",
    "rural_elderly", "young_urban", "agri_emigration",
    "orthodox_poverty", "anti_estab_persistence",
    "nationalist_consolidation_ratio",
]
SENTIMENT_FEATURES = [
    "ep_2024_anti_estab",
    "georgescu_r1_2024", "simion_r1_2025",
    "pres19r2_IOHANNIS",
]

TARGET_COLS = (
    [f"t_{p}" for p in PARLIAMENTARY_MODEL_PARTIES]
    + ["t_ESTABLISHMENT", "t_ANTI_ESTABLISHMENT", "t_turnout"]
)
VALIDATION_COLS = [
    "pres25r2_DAN", "pres25r2_SIMION",
    "pres19r2_DANCILA",
]


def _z(s: pd.Series) -> pd.Series:
    """Z-score a series, filling NaN with median before standardising."""
    filled = s.fillna(s.median())
    std = filled.std()
    return (filled - filled.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)


def _composite(df: pd.DataFrame, cols: list[str], signs: list[int]) -> pd.Series:
    """Weighted-sign average of z-scores. sign=+1 means higher→higher index."""
    z_parts = [sign * _z(df[c]) for c, sign in zip(cols, signs) if c in df.columns]
    if not z_parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(z_parts, axis=1).mean(axis=1)


def _extract_election(
    elections: pd.DataFrame, election_id: str, prefix: str
) -> pd.DataFrame:
    """Pull one election's share columns, rename with prefix, keep siruta."""
    sub = elections[elections["election_id"] == election_id].copy()
    if sub.empty:
        print(f"  [WARN] Election '{election_id}' not found in elections_clean.",
              file=sys.stderr)
        return pd.DataFrame(columns=["siruta"])
    share_cols = [c for c in sub.columns if c.startswith("share_")]
    renames = {c: f"{prefix}_{c.removeprefix('share_')}" for c in share_cols}
    if "turnout" in sub.columns:
        renames["turnout"] = f"{prefix}_turnout"
    return sub[["siruta"] + list(renames.keys())].rename(columns=renames)


def build_ins_extra_features(df: pd.DataFrame, ins: pd.DataFrame) -> pd.DataFrame:
    """
    Merge INS extra data and normalise raw counts by census population.

    Rates:
      emigrants_permanent_rate  per 1 000 population
      emigrants_temporary_rate  per 1 000 population
      school_units_per_10k      per 10 000 population
      hospitals_per_100k        per 100 000 population
      tourism_units_per_10k     per 10 000 population
      convictions_per_100k      per 100 000 population
      avg_household_size        already a ratio — kept as-is
    """
    df = df.merge(ins, on="siruta", how="left")
    pop = df["pop_census"]

    df["emigrants_permanent_rate"] = df["emigrants_permanent"] / pop * 1_000
    df["emigrants_temporary_rate"] = df["emigrants_temporary"] / pop * 1_000
    df["school_units_per_10k"]     = df["school_units"]        / pop * 10_000
    df["hospitals_per_100k"]       = df["hospitals"]           / pop * 100_000
    df["tourism_units_per_10k"]    = df["tourism_units"]       / pop * 10_000
    df["convictions_per_100k"]     = df["convictions"]         / pop * 100_000

    # Drop raw counts and census pop — rates are sufficient for modelling
    df = df.drop(columns=["emigrants_permanent", "emigrants_temporary",
                           "school_units", "hospitals", "tourism_units",
                           "convictions", "pop_census"])
    return df


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct interaction terms and fragmentation indices from existing columns.

    All inputs are expected to already be in [0, 1] share space (proportions).
    Missing inputs produce NaN outputs rather than errors.
    """
    df = df.copy()

    # --- Fragmentation indices (Herfindahl–Hirschman complement) ---------------
    eth_shares = ["pct_romani", "pct_maghiari", "pct_romi",
                  "pct_germani", "pct_ucraineni", "pct_tatari",
                  "pct_turci", "pct_lipoveni", "ethnicity_other"]
    eth_present = [c for c in eth_shares if c in df.columns]
    if len(eth_present) >= 3:
        eth_mat = df[eth_present].fillna(0).clip(lower=0)
        # Rescale each row to sum to 1 (absorbs small gaps from "other" rounding)
        row_sum = eth_mat.sum(axis=1).replace(0, np.nan)
        eth_mat = eth_mat.div(row_sum, axis=0)
        df["ethno_linguistic_frag"] = 1 - (eth_mat ** 2).sum(axis=1)
    else:
        df["ethno_linguistic_frag"] = np.nan

    rel_shares = ["pct_orthodox", "pct_roman_catholic", "pct_greco_catholic",
                  "pct_reformed", "pct_pentecostal", "pct_baptist",
                  "pct_adventist", "pct_unitarian", "pct_muslim",
                  "religion_other"]
    rel_present = [c for c in rel_shares if c in df.columns]
    if len(rel_present) >= 3:
        rel_mat = df[rel_present].fillna(0).clip(lower=0)
        row_sum = rel_mat.sum(axis=1).replace(0, np.nan)
        rel_mat = rel_mat.div(row_sum, axis=0)
        df["religious_frag"] = 1 - (rel_mat ** 2).sum(axis=1)
    else:
        df["religious_frag"] = np.nan

    # --- Interaction terms -------------------------------------------------------
    if "pct_urban" in df.columns and "pct_pop_over65" in df.columns:
        df["rural_elderly"] = (1 - df["pct_urban"]) * df["pct_pop_over65"]

    if "pct_urban" in df.columns and "pct_pop_under25" in df.columns:
        df["young_urban"] = df["pct_urban"] * df["pct_pop_under25"]

    # Agricultural employment × emigration pressure (both strong in same counties)
    if "pct_employment_agriculture" in df.columns and "net_migration_rate" in df.columns:
        df["agri_emigration"] = (
            df["pct_employment_agriculture"] * (-df["net_migration_rate"])
        )

    if "pct_orthodox" in df.columns and "poverty_risk_pct" in df.columns:
        df["orthodox_poverty"] = df["pct_orthodox"] * df["poverty_risk_pct"]

    # --- Election-derived interactions (only after sentiment columns are present) ---
    if "ep_2024_anti_estab" in df.columns and "simion_r1_2025" in df.columns:
        df["anti_estab_persistence"] = (
            df["ep_2024_anti_estab"] * df["simion_r1_2025"]
        )

    # Georgescu→Simion vote consolidation ratio:
    # How much of their combined vote went to Georgescu in 2024?
    # High → county's radical-nationalist vote was Georgescu-specific (personality-driven)
    # Low  → Simion inherited more proportionally (party consolidation)
    if "georgescu_r1_2024" in df.columns and "simion_r1_2025" in df.columns:
        combined = df["georgescu_r1_2024"] + df["simion_r1_2025"]
        combined = combined.replace(0, np.nan)
        df["nationalist_consolidation_ratio"] = df["georgescu_r1_2024"] / combined

    return df


def build_composite_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add three composite index columns to the feature DataFrame."""
    df = df.copy()

    # Economic grievance: high unemployment/poverty/ESL, low GDP → high grievance
    df["idx_grievance"] = _composite(
        df,
        ["unemployment_rate", "poverty_risk_pct", "early_school_leaving_pct", "gdp_per_capita"],
        [+1, +1, +1, -1],
    )

    # Modernity: high education/internet, low ESL → high modernity
    df["idx_modernity"] = _composite(
        df,
        ["tertiary_education_pct", "internet_use_pct", "early_school_leaving_pct"],
        [+1, +1, -1],
    )

    # Demographic pressure: ageing + net emigration (negative net_migration = outflow)
    # Net migration convention: negative = net loss → multiply by -1 for pressure direction
    df["idx_demographic_pressure"] = _composite(
        df,
        ["pct_pop_over65", "net_migration_rate", "pct_pop_under25"],
        [+1, -1, -1],
    )

    return df


def build_sentiment_proxies(
    df: pd.DataFrame, elections: pd.DataFrame
) -> pd.DataFrame:
    """Add EP 2024 anti-establishment proxy and 2020→2024 parliamentary swing."""
    df = df.copy()

    ep = _extract_election(elections, "ep_2024", "ep24")
    if "ep24_ANTI_ESTABLISHMENT" in ep.columns:
        df = df.merge(
            ep[["siruta", "ep24_ANTI_ESTABLISHMENT"]].rename(
                columns={"ep24_ANTI_ESTABLISHMENT": "ep_2024_anti_estab"}
            ),
            on="siruta", how="left",
        )
    else:
        df["ep_2024_anti_estab"] = np.nan

    # Swing: change in anti-establishment share from 2020 to 2024 parliamentary
    p2024 = _extract_election(elections, "parlamentare_2024", "p24")
    p2020 = _extract_election(elections, "parlamentare_2020", "p20")

    swing_key_2024 = "p24_ANTI_ESTABLISHMENT"
    swing_key_2020 = "p20_ANTI_ESTABLISHMENT"

    if swing_key_2024 in p2024.columns and swing_key_2020 in p2020.columns:
        swing_df = p2024[["siruta", swing_key_2024]].merge(
            p2020[["siruta", swing_key_2020]], on="siruta", how="left"
        )
        swing_df["swing_anti_estab_2020_2024"] = (
            swing_df[swing_key_2024] - swing_df[swing_key_2020]
        )
        df = df.merge(swing_df[["siruta", "swing_anti_estab_2020_2024"]],
                      on="siruta", how="left")
    else:
        df["swing_anti_estab_2020_2024"] = np.nan

    # Georgescu R1 2024 (annulled) and Simion R1 2025
    for election_id, candidate, out_col in [
        ("prezidentiale_2024_r1", "GEORGESCU", "georgescu_r1_2024"),
        ("prezidentiale_2025_r1", "SIMION",    "simion_r1_2025"),
    ]:
        prefix = election_id.replace("prezidentiale_", "p").replace("_r1", "r1")
        layer = _extract_election(elections, election_id, prefix)
        src_col = f"{prefix}_{candidate}"
        if src_col in layer.columns and layer[src_col].notna().any():
            df = df.merge(layer[["siruta", src_col]].rename(columns={src_col: out_col}),
                          on="siruta", how="left")
        else:
            df[out_col] = np.nan

    return df


def build_targets(df: pd.DataFrame, elections: pd.DataFrame) -> pd.DataFrame:
    """Attach parliamentary 2024 party shares as target columns (prefix t_)."""
    df = df.copy()
    parl = _extract_election(elections, "parlamentare_2024", "t")

    for party in PARLIAMENTARY_MODEL_PARTIES:
        src = f"t_{party}"
        df = df.merge(
            parl[["siruta", src]] if src in parl.columns
            else pd.DataFrame({"siruta": df["siruta"], src: np.nan}),
            on="siruta", how="left",
        )

    for composite in ["ESTABLISHMENT", "ANTI_ESTABLISHMENT"]:
        src = f"t_{composite}"
        if src in parl.columns:
            df = df.merge(parl[["siruta", src]], on="siruta", how="left")
        else:
            df[src] = np.nan

    if "t_turnout" in parl.columns:
        df = df.merge(parl[["siruta", "t_turnout"]], on="siruta", how="left")
    else:
        df["t_turnout"] = np.nan

    return df


def build_validation_layers(df: pd.DataFrame, elections: pd.DataFrame) -> pd.DataFrame:
    """
    Attach presidential R2 results.
    pres19r2_IOHANNIS is also used as a feature (baseline establishment support proxy).
    pres25r2_DAN and pres25r2_SIMION are pure out-of-sample validation.
    pres19r2_DANCILA is included for completeness (anti-Iohannis share).
    """
    df = df.copy()

    for election_id, prefix, candidates in [
        ("prezidentiale_2025_r2", "pres25r2", ["DAN", "SIMION"]),
        ("prezidentiale_2019_r2", "pres19r2", ["IOHANNIS", "DANCILA"]),
    ]:
        layer = _extract_election(elections, election_id, prefix)
        for cand in candidates:
            src = f"{prefix}_{cand}"
            if src in layer.columns:
                df = df.merge(layer[["siruta", src]], on="siruta", how="left")
            else:
                df[src] = np.nan

    return df


def report_missing(df: pd.DataFrame) -> None:
    feature_cols = (EUROSTAT_FEATURES + CENSUS_FEATURES + INS_EXTRA_FEATURES
                    + COMPOSITE_FEATURES + ENGINEERED_FEATURES
                    + SENTIMENT_FEATURES + TARGET_COLS + VALIDATION_COLS)
    present = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"  [WARN] Expected columns not present (data may not be downloaded yet):")
        for c in missing_cols:
            print(f"         {c}")

    print("\nMissing value summary (feature columns):")
    for col in present:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"  {col:45s}: {n_missing}/42 missing")

    complete_cases = df[present].dropna().shape[0]
    print(f"\nComplete cases (all feature columns present): {complete_cases}/42")
    if complete_cases < 30:
        print("  [WARN] Fewer than 30 complete cases — check data downloads before modelling.",
              file=sys.stderr)


def main() -> None:
    for path, label in [(EURO_PATH, "eurostat_clean"), (CENS_PATH, "census_clean"),
                        (ELEC_PATH, "elections_clean")]:
        if not path.exists():
            print(f"Missing input: {path}\nRun the 02_clean stage first.", file=sys.stderr)
            sys.exit(1)

    print("Loading clean data …")
    euro      = pd.read_parquet(EURO_PATH)
    census    = pd.read_parquet(CENS_PATH)
    elections = pd.read_parquet(ELEC_PATH)

    print(f"  Eurostat:  {len(euro)} rows")
    print(f"  Census:    {len(census)} rows")
    print(f"  Elections: {len(elections)} rows, "
          f"{elections['election_id'].nunique()} elections")

    # Base: Eurostat (NUTS 3 geo as anchor)
    df = euro.copy()

    # Join census on siruta — include judet (absent from Eurostat), exclude nuts3_code duplicate
    cens_cols = ["siruta"] + [c for c in census.columns
                              if c not in ("nuts3_code", "siruta")]
    df = df.merge(census[cens_cols], on="siruta", how="left")

    # INS extra features (optional — skip gracefully if not yet ingested)
    if INS_PATH.exists():
        print("Merging INS extra features …")
        ins = pd.read_parquet(INS_PATH)
        df = build_ins_extra_features(df, ins)
        print(f"  INS extra: {len(INS_EXTRA_FEATURES)} derived rate columns added")
    else:
        print(f"  [SKIP] {INS_PATH.name} not found — run scripts/01_ingest/ingest_ins_extra.py",
              file=sys.stderr)
        for col in INS_EXTRA_FEATURES:
            df[col] = np.nan

    # Composite indices
    print("Building composite indices …")
    df = build_composite_indices(df)

    # Sentiment proxies (EP 2024, Georgescu R1 2024, Simion R1 2025)
    print("Building sentiment proxies …")
    df = build_sentiment_proxies(df, elections)

    # Primary targets (parliamentary 2024)
    print("Attaching targets …")
    df = build_targets(df, elections)

    # Presidential validation layers (also provides pres19r2_IOHANNIS feature)
    print("Attaching validation layers …")
    df = build_validation_layers(df, elections)

    # Engineered interaction/fragmentation features (needs sentiment cols to be present)
    print("Building engineered features …")
    df = build_engineered_features(df)

    # Final column order
    id_cols = ["siruta", "judet", "geo_nuts3", "nuts2_code"]
    ordered = (id_cols
               + [c for c in EUROSTAT_FEATURES   if c in df.columns]
               + [c for c in CENSUS_FEATURES     if c in df.columns]
               + [c for c in INS_EXTRA_FEATURES  if c in df.columns]
               + [c for c in COMPOSITE_FEATURES  if c in df.columns]
               + [c for c in ENGINEERED_FEATURES if c in df.columns]
               + [c for c in SENTIMENT_FEATURES  if c in df.columns]
               + [c for c in TARGET_COLS         if c in df.columns]
               + [c for c in VALIDATION_COLS     if c in df.columns])
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    report_missing(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns -> {OUT_PATH}")

    # App export: feature columns only (no targets, no source-level flags)
    app_cols = (id_cols
                + [c for c in EUROSTAT_FEATURES   if c in df.columns]
                + [c for c in CENSUS_FEATURES     if c in df.columns]
                + [c for c in INS_EXTRA_FEATURES  if c in df.columns]
                + [c for c in COMPOSITE_FEATURES  if c in df.columns]
                + [c for c in ENGINEERED_FEATURES if c in df.columns]
                + [c for c in SENTIMENT_FEATURES  if c in df.columns])
    APP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[app_cols].to_json(APP_PATH, orient="records", force_ascii=False, indent=2)
    print(f"App export -> {APP_PATH}")

    print("\nNext step: python scripts/04_model/run_all_models.py")


if __name__ == "__main__":
    main()
