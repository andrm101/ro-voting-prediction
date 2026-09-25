"""
Clean and reshape INS Census 2021 data into a wide feature matrix.

Input:  data/processed/census_raw.parquet
Output: data/processed/census_clean.parquet

One row per județ (42 rows). Columns:
  siruta | judet | nuts3_code
  pct_romani | pct_maghiari | pct_romi | pct_germani | pct_ucraineni
  pct_orthodox | pct_roman_catholic | pct_greco_catholic | pct_reformed
  pct_pentecostal | pct_baptist | pct_adventist | pct_muslim
  pct_urban
  (ethnic_other, religion_other computed as residuals)

Category matching uses fuzzy substring matching to handle INS naming variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import JUDET_SIRUTA, SIRUTA_NUTS3

IN_PATH  = ROOT / "data" / "processed" / "census_raw.parquet"
OUT_PATH = ROOT / "data" / "processed" / "census_clean.parquet"

# ---------------------------------------------------------------------------
# Category → feature name mappings
# Substring match (case-insensitive) against the 'category' column.
# First match wins. Residual = 1 - sum of matched.
# ---------------------------------------------------------------------------
ETHNICITY_MAP: list[tuple[str, str]] = [
    ("rom[aâ]ni",       "pct_romani"),
    ("maghiari",        "pct_maghiari"),
    ("romi",            "pct_romi"),
    ("germani",         "pct_germani"),
    ("ucraineni",       "pct_ucraineni"),
    ("lipoveni",        "pct_lipoveni"),
    ("turci",           "pct_turci"),
    ("t[aă]tari",       "pct_tatari"),
]

RELIGION_MAP: list[tuple[str, str]] = [
    ("ortodoc|ortodox",  "pct_orthodox"),
    ("romano.catolic",   "pct_roman_catholic"),
    ("greco.catolic",    "pct_greco_catholic"),
    ("reformat|calvin",  "pct_reformed"),
    ("penticostal",      "pct_pentecostal"),
    ("baptist",          "pct_baptist"),
    ("adventist",        "pct_adventist"),
    ("musulman|islam",   "pct_muslim"),
    ("unitar",           "pct_unitarian"),
]

URBAN_RURAL_MAP: list[tuple[str, str]] = [
    ("urban",           "pct_urban"),
    ("rural",           "pct_rural"),
]

CATEGORY_TYPE_MAPS: dict[str, list[tuple[str, str]]] = {
    "ethnicity":   ETHNICITY_MAP,
    "religion":    RELIGION_MAP,
    "urban_rural": URBAN_RURAL_MAP,
}


def _match_category(category: str, mapping: list[tuple[str, str]]) -> str | None:
    import re
    for pattern, feature_name in mapping:
        if re.search(pattern, category, re.IGNORECASE):
            return feature_name
    return None


def _pivot_category_type(
    df: pd.DataFrame,
    category_type: str,
    mapping: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    For a given category_type subset, map categories to feature names,
    sum duplicates (e.g. 'Reformați' + 'Calviniști' both → pct_reformed),
    and pivot to wide.
    """
    subset = df[df["category_type"] == category_type].copy()
    if subset.empty:
        print(f"  [WARN] No rows for category_type='{category_type}'", file=sys.stderr)
        return pd.DataFrame()

    subset["feature"] = subset["category"].apply(
        lambda c: _match_category(c, mapping)
    )

    unmapped = subset[subset["feature"].isna()]["category"].unique()
    if len(unmapped):
        print(f"  [INFO] {category_type}: unmapped categories (-> residual): "
              f"{list(unmapped)[:8]}", file=sys.stderr)

    matched = subset.dropna(subset=["feature"])

    # Sum within (siruta, feature) in case two raw categories map to same feature
    wide = (
        matched.groupby(["siruta", "judet", "feature"])["share_of_total"]
        .sum()
        .reset_index()
        .pivot(index=["siruta", "judet"], columns="feature", values="share_of_total")
        .reset_index()
    )
    wide.columns.name = None

    # Residual
    feature_names = [fn for _, fn in mapping]
    present_feats = [f for f in feature_names if f in wide.columns]
    residual_col = f"{category_type}_other"
    wide[residual_col] = (1.0 - wide[present_feats].sum(axis=1, min_count=1)).clip(lower=0)

    return wide


def _validate(df: pd.DataFrame) -> None:
    expected_rows = 42
    if len(df) != expected_rows:
        print(f"  [WARN] Expected {expected_rows} județe, got {len(df)}. "
              "Check for missing or duplicate counties.", file=sys.stderr)

    # UDMR sanity proxy: pct_maghiari > 0.5 in Harghita (SIRUTA=210) and Covasna (SIRUTA=150)
    for siruta, county, threshold in [(210, "Harghita", 0.70), (150, "Covasna", 0.60)]:
        row = df[df["siruta"] == siruta]
        if row.empty:
            print(f"  [WARN] {county} (SIRUTA {siruta}) missing from census output.", file=sys.stderr)
            continue
        val = row["pct_maghiari"].iloc[0] if "pct_maghiari" in df.columns else None
        if val is None or val < threshold:
            print(f"  [WARN] {county}: pct_maghiari={val:.2f} < expected {threshold:.2f}. "
                  "Check ethnicity data.", file=sys.stderr)

    # Pentecostal should be highest in Bihor (RO111) and Satu Mare (RO115)
    if "pct_pentecostal" in df.columns:
        top_pent = df.nlargest(5, "pct_pentecostal")[["judet", "pct_pentecostal"]]
        top_str = top_pent.to_string(index=False).encode("ascii", errors="replace").decode()
        print(f"  [INFO] Top 5 counties by pct_pentecostal:\n{top_str}")


def main() -> None:
    if not IN_PATH.exists():
        print(f"Input not found: {IN_PATH}\nRun scripts/01_ingest/ingest_ins.py first.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading census_raw.parquet ...")
    raw = pd.read_parquet(IN_PATH)
    print(f"  {len(raw):,} rows | category_types: {sorted(raw['category_type'].unique())}")

    frames: list[pd.DataFrame] = []
    for cat_type, mapping in CATEGORY_TYPE_MAPS.items():
        print(f"Processing category_type='{cat_type}' …")
        pivoted = _pivot_category_type(raw, cat_type, mapping)
        if not pivoted.empty:
            frames.append(pivoted)

    if not frames:
        print("No data after pivoting.", file=sys.stderr)
        sys.exit(1)

    # Merge all category types on siruta
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["siruta", "judet"], how="outer")

    # Attach NUTS3 code
    result["nuts3_code"] = result["siruta"].map(SIRUTA_NUTS3)
    unmatched = result[result["nuts3_code"].isna()]["judet"].unique()
    if len(unmatched):
        print(f"  [WARN] No NUTS3 code for: {list(unmatched)}", file=sys.stderr)

    # Sort by siruta for readability
    result = result.sort_values("siruta").reset_index(drop=True)

    print("Validating ...")
    _validate(result)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH, index=False)

    feature_cols = [c for c in result.columns if c.startswith("pct_")]
    print(f"\nSaved {len(result)} rows x {len(result.columns)} columns -> {OUT_PATH}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print("\nMissing value summary:")
    print(result[feature_cols].isna().sum().to_string())


if __name__ == "__main__":
    main()
