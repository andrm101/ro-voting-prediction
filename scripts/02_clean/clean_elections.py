"""
Clean and reshape raw election data.

Input:  data/processed/elections_raw.parquet
Output: data/processed/elections_clean.parquet

Transformations:
  1. Normalise party/candidate names to canonical labels (via constants.py)
  2. Collapse residual parties into "OTHER" per election × județ
  3. Pivot long → wide: one row per (siruta, election_id)
  4. Add composite columns: share_ESTABLISHMENT, share_ANTI_ESTABLISHMENT
  5. Validate row-level share sums ≈ 1.0
  6. Validate UDMR near-zero outside Hungarian-majority counties
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import (
    ANTI_ESTABLISHMENT_PARTIES,
    ESTABLISHMENT_PARTIES,
    PARLIAMENTARY_MODEL_PARTIES,
    SIRUTA_JUDET,
    normalise_party,
)

IN_PATH  = ROOT / "data" / "processed" / "elections_raw.parquet"
OUT_PATH = ROOT / "data" / "processed" / "elections_clean.parquet"

# Counties where UDMR > 5% is expected — majority + significant-minority counties
# Covasna(150), Harghita(210), Mureș(280): Hungarian majority
# Bihor(50), Satu Mare(320), Sălaj(330): 20-35% Hungarian
# Arad(20), Brașov(80), Cluj(130), Maramureș(260): 8-18% Hungarian
HUNGARIAN_COUNTIES_SIRUTA = {150, 210, 280, 50, 320, 330, 20, 80, 130, 260}

# Maximum tolerated share-sum deviation from 1.0 before a warning is raised
SHARE_SUM_TOLERANCE = 0.05


def _normalise_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply canonical party labels, then sum votes for any label that maps to
    the same canonical name within a (siruta, election_id) group.
    """
    df = df.copy()
    df["party"] = df["party_raw"].apply(normalise_party)

    agg = (
        df.groupby(["siruta", "judet", "election_id", "election_type",
                    "year", "round", "party"], dropna=False)
        .agg(
            votes=("votes", "sum"),
            valid_votes=("valid_votes", "first"),
            registered=("registered", "first"),
        )
        .reset_index()
    )
    agg["share"] = agg["votes"] / agg["valid_votes"].replace(0, np.nan)
    agg["turnout"] = agg["valid_votes"] / agg["registered"].replace(0, np.nan)
    return agg


def _pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot party-long → county-wide for each election."""
    index_cols = ["siruta", "judet", "election_id", "election_type",
                  "year", "round"]

    # Fill NaN round (ep, parlamentare are single-round) so pivot_table preserves them
    df = df.copy()
    df["round"] = df["round"].fillna("r1")

    # Turnout: one value per (siruta, election_id)
    turnout = (
        df[index_cols + ["turnout"]]
        .drop_duplicates(subset=["siruta", "election_id"])
        .set_index(["siruta", "election_id"])["turnout"]
        .rename("turnout")
    )

    # Shares pivot
    shares = df.pivot_table(
        index=index_cols,
        columns="party",
        values="share",
        aggfunc="sum",
    )
    shares.columns = [f"share_{c}" for c in shares.columns]
    shares = shares.reset_index()

    wide = shares.merge(
        df[index_cols].drop_duplicates(),
        on=index_cols, how="left"
    )
    wide = wide.merge(
        turnout.reset_index(),
        on=["siruta", "election_id"], how="left"
    )
    return wide


def _add_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Add establishment and anti-establishment composite share columns."""
    df = df.copy()

    estab_cols = [f"share_{p}" for p in ESTABLISHMENT_PARTIES if f"share_{p}" in df.columns]
    anti_cols  = [f"share_{p}" for p in ANTI_ESTABLISHMENT_PARTIES if f"share_{p}" in df.columns]

    if estab_cols:
        df["share_ESTABLISHMENT"] = df[estab_cols].sum(axis=1, min_count=1)
    if anti_cols:
        df["share_ANTI_ESTABLISHMENT"] = df[anti_cols].sum(axis=1, min_count=1)

    return df


def _validate(df: pd.DataFrame) -> None:
    share_cols = [c for c in df.columns if c.startswith("share_") and
                  c not in ("share_ESTABLISHMENT", "share_ANTI_ESTABLISHMENT")]

    # Check per-election share sums
    parl = df[df["election_type"] == "parlamentare"]
    if not parl.empty:
        parl_share_cols = [c for c in share_cols if c in parl.columns]
        row_sums = parl[parl_share_cols].sum(axis=1, min_count=1)
        bad = row_sums[(row_sums - 1.0).abs() > SHARE_SUM_TOLERANCE]
        if not bad.empty:
            print(f"  [WARN] {len(bad)} parliamentary rows with share sum "
                  f"outside ±{SHARE_SUM_TOLERANCE:.0%} of 1.0", file=sys.stderr)
            print(f"         SIRUTAs: {parl.loc[bad.index, 'siruta'].tolist()[:10]}",
                  file=sys.stderr)

    # UDMR outside Hungarian counties should be near-zero
    if "share_UDMR" in df.columns:
        non_hu = df[~df["siruta"].isin(HUNGARIAN_COUNTIES_SIRUTA) & df["election_type"].eq("parlamentare")]
        high_udmr = non_hu[non_hu["share_UDMR"].fillna(0) > 0.05]
        if not high_udmr.empty:
            print(f"  [WARN] Unexpected UDMR share > 5% outside Hungarian counties "
                  f"in {len(high_udmr)} rows: {high_udmr['judet'].unique().tolist()}",
                  file=sys.stderr)


def main() -> None:
    if not IN_PATH.exists():
        print(f"Input not found: {IN_PATH}\nRun scripts/01_ingest/ingest_roaep.py first.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading elections_raw.parquet …")
    raw = pd.read_parquet(IN_PATH)
    print(f"  {len(raw):,} rows | elections: {sorted(raw['election_id'].unique())}")

    print("Normalising party labels …")
    agg = _normalise_and_aggregate(raw)

    # Diagnose unmapped parties using the original raw frame
    raw_with_norm = raw.copy()
    raw_with_norm["party_norm"] = raw_with_norm["party_raw"].apply(normalise_party)
    unknown = raw_with_norm[raw_with_norm["party_norm"] == "OTHER"]["party_raw"].value_counts().head(10)
    if not unknown.empty:
        print("  [INFO] Top unmapped party_raw values -> 'OTHER':")
        for name, cnt in unknown.items():
            print(f"         {name!r:50s} ({cnt} rows)")

    print("Pivoting to wide format …")
    wide = _pivot_to_wide(agg)

    print("Adding composite columns …")
    wide = _add_composites(wide)

    print("Validating …")
    _validate(wide)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(wide):,} rows x {len(wide.columns)} columns -> {OUT_PATH}")
    elections = wide["election_id"].value_counts().sort_index()
    print("Rows per election:")
    for eid, cnt in elections.items():
        print(f"  {eid}: {cnt} counties")
    share_cols = [c for c in wide.columns if c.startswith("share_")]
    print(f"Share columns ({len(share_cols)}): {share_cols}")


if __name__ == "__main__":
    main()
