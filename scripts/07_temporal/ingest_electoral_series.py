"""
Ingest all raw AEP electoral files and produce a single county-level
electoral series parquet: electoral_series_judet.parquet

Sources processed:
  - parlamentare_2020_<county>_cd/s  (precinct-level, per county file)
  - parlamentare_2024_judet           (precinct-level, one national file)
  - ep_2024_judet                     (precinct-level, one national file)
  - prezidentiale_2019_r2_judet       (precinct-level, one national file)
  - prezidentiale_2024_r1_judet       (precinct-level, one national file)
  - prezidentiale_2025_r1_judet       (precinct-level, one national file)
  - prezidentiale_2025_r2_judet       (precinct-level, one national file)

Column conventions (AEP raw data):
  a        = enrolled voters
  b        = votes cast  (b = valid_candidate + null + blank)
  For 2020 parliamentary:
    e = valid candidate votes (sum of all party columns), f = null, g = blank
  For 2024 parliamentary:
    e = valid candidate votes, f = null, g = blank  (same layout)
  For presidential / EP:
    c = valid candidate votes, d = null  (b = c + d)
  For 2019 R2 (older format):
    c = valid candidate votes, d = null, g1/g2 = candidate columns
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "data" / "raw" / "roaep"
OUT  = ROOT / "data" / "processed" / "electoral_series_judet.parquet"

# ---------------------------------------------------------------------------
# Party name → canonical column, for each election year
# ---------------------------------------------------------------------------
PARL_CANONICAL = {
    # 2020 names
    "UNIUNEA DEMOCRATĂ MAGHIARĂ DIN ROMÂNIA":        "t_UDMR",
    "PARTIDUL SOCIAL DEMOCRAT":                       "t_PSD",
    "PARTIDUL NAȚIONAL LIBERAL":                      "t_PNL",
    "ALIANȚA USR PLUS":                               "t_USR",
    "ALIANȚA PENTRU UNIREA ROMÂNILOR":                "t_AUR",
    "PARTIDUL S.O.S. ROMÂNIA":                        "t_SOS",
    "PARTIDUL OAMENILOR TINERI":                      "t_POT",
    # 2024 names (some differ)
    "UNIUNEA SALVAȚI ROMÂNIA":                        "t_USR",
    "ALIANȚA ELECTORALĂ PSD PNL":                     "t_PSD",  # EP alliance only
    # Everything else (PMP, PRO, PPU-SL, minority parties, etc.) → t_OTHER
}

EP_CANONICAL = {
    "UNIUNEA DEMOCRATĂ MAGHIARĂ DIN ROMÂNIA":         "t_UDMR",
    "ALIANȚA ELECTORALĂ PSD PNL":                     "t_PSD_PNL",
    "ALIANȚA DREAPTA UNITĂ USR - PMP - FORȚA DREPTEI":"t_USR",
    "ALIANȚA AUR":                                    "t_AUR",
    "PARTIDUL S.O.S. ROMÂNIA":                        "t_SOS",
}

# Normalise county name to our standard JUDET field used in features parquet
SECTOR_CODES = {44}  # precinct_county_nce for all 6 Bucharest sectors

def _norm_county(raw_name: str) -> str:
    """Collapse Bucharest sectors; strip sector prefix for other counties."""
    if "SECTOR" in raw_name.upper() or "BUCUREŞTI" in raw_name.upper() or "BUCUREȘTI" in raw_name.upper():
        return "MUNICIPIUL BUCUREȘTI"
    return raw_name.strip().upper()


def _strip_voturi(col: str) -> str:
    return col.replace("-voturi", "").strip()


def _read_csv_raw(path: Path) -> pd.DataFrame:
    """Read AEP CSV, stripping BOM and double .csv extension."""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


# ---------------------------------------------------------------------------
# 1.  2020 Parliamentary — per-county files
# ---------------------------------------------------------------------------
def _ingest_2020_parliamentary() -> pd.DataFrame:
    files = sorted(RAW.glob("parlamentare_2020_*_*.csv.csv"))
    records = []
    for f in files:
        # Extract chamber from filename
        stem = f.stem.replace(".csv", "")          # e.g. parlamentare_2020_alba_cd
        parts = stem.split("_")
        chamber = parts[-1].upper()                # CD or S
        if chamber not in ("CD", "S"):
            continue

        df = _read_csv_raw(f)

        # Identify party vote columns
        party_cols = [c for c in df.columns if c.endswith("-voturi")]

        # Aggregate numeric columns to county level
        # valid_votes = column e (index 23 in standard layout); enrolled = a; voted = b
        # Identify by position: a=col12, b=col16, e=col22, f=col23, g=col24
        numeric_cols = ["a", "b", "e", "f", "g"] + party_cols
        # Coerce to numeric (some may be string due to BOM or encoding)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Normalise county name
        df["judet_norm"] = df["precinct_county_name"].apply(_norm_county)

        grp = df.groupby("judet_norm")[numeric_cols].sum().reset_index()
        grp["election"]  = "parlamentare_2020"
        grp["year"]      = 2020
        grp["chamber"]   = chamber
        grp.rename(columns={"judet_norm": "judet",
                             "a": "enrolled",
                             "b": "voted",
                             "e": "valid_votes"}, inplace=True)

        # Map party columns to canonical names
        canonical_sums: dict[str, pd.Series] = {}
        for col in party_cols:
            name = _strip_voturi(col)
            canon = PARL_CANONICAL.get(name, "t_OTHER")
            if canon in canonical_sums:
                canonical_sums[canon] = canonical_sums[canon] + grp[col]
            else:
                canonical_sums[canon] = grp[col].copy()

        for canon, series in canonical_sums.items():
            grp[canon] = series

        # Drop raw party columns
        grp.drop(columns=party_cols + ["f", "g"], errors="ignore", inplace=True)
        records.append(grp)

    if not records:
        raise RuntimeError("No 2020 parliamentary files found")

    out = pd.concat(records, ignore_index=True)

    # Bucharest sectors are already merged via judet_norm groupby
    # Compute shares
    party_cols_canon = [c for c in out.columns if c.startswith("t_")]
    for col in party_cols_canon:
        out[col] = out[col] / out["valid_votes"]

    out["turnout"] = out["voted"] / out["enrolled"]
    return out


# ---------------------------------------------------------------------------
# 2.  2024 Parliamentary — single national file
# ---------------------------------------------------------------------------
def _ingest_2024_parliamentary() -> pd.DataFrame:
    path = RAW / "parlamentare_2024_judet.csv.csv"
    df = _read_csv_raw(path)

    party_cols = [c for c in df.columns if c.endswith("-voturi")]
    numeric_cols = ["a", "b", "e", "f", "g"] + party_cols
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["judet_norm"] = df["precinct_county_name"].apply(_norm_county)

    # Separate by chamber (report_type_code: CD or S)
    records = []
    for chamber, sub in df.groupby("report_type_category_code"):
        if chamber not in ("CD", "S"):
            continue
        grp = sub.groupby("judet_norm")[numeric_cols].sum().reset_index()
        grp["election"] = "parlamentare_2024"
        grp["year"]     = 2024
        grp["chamber"]  = chamber
        grp.rename(columns={"judet_norm": "judet",
                             "a": "enrolled",
                             "b": "voted",
                             "e": "valid_votes"}, inplace=True)

        canonical_sums: dict[str, pd.Series] = {}
        for col in party_cols:
            name = _strip_voturi(col)
            canon = PARL_CANONICAL.get(name, "t_OTHER")
            if canon in canonical_sums:
                canonical_sums[canon] = canonical_sums[canon] + grp[col]
            else:
                canonical_sums[canon] = grp[col].copy()

        for canon, series in canonical_sums.items():
            grp[canon] = series

        grp.drop(columns=party_cols + ["f", "g"], errors="ignore", inplace=True)
        party_cols_canon = [c for c in grp.columns if c.startswith("t_")]
        for col in party_cols_canon:
            grp[col] = grp[col] / grp["valid_votes"]
        grp["turnout"] = grp["voted"] / grp["enrolled"]
        records.append(grp)

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# 3.  Presidential elections — national precinct files
#     b = voted, c = valid candidate votes, d = null  (b = c + d)
#     2019 R2 has different header (g1/g2 for candidates)
# ---------------------------------------------------------------------------
def _ingest_presidential(fname: str, year: int, label: str) -> pd.DataFrame:
    path = RAW / fname
    df = _read_csv_raw(path)

    # 2019 R2 uses legacy column names (Județ, g1, g2)
    is_legacy = "Județ" in df.columns

    county_col     = "Județ" if is_legacy else "precinct_county_name"
    candidate_cols = ["g1", "g2"] if is_legacy else \
                     [c for c in df.columns if c.endswith("-voturi")]

    # valid_votes column: c for all presidential files
    valid_col = "c"

    for col in ["a", "b", valid_col, "d"] + candidate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["judet_norm"] = df[county_col].apply(_norm_county)

    numeric_agg = [c for c in ["a", "b", valid_col, "d"] + candidate_cols
                   if c in df.columns]
    grp = df.groupby("judet_norm")[numeric_agg].sum().reset_index()
    grp["election"] = label
    grp["year"]     = year
    grp["chamber"]  = "PRES"

    grp.rename(columns={"judet_norm": "judet",
                         "a": "enrolled",
                         "b": "voted",
                         valid_col: "valid_votes",
                         "d": "null_votes"}, inplace=True)

    # Normalise candidate columns to short readable names
    rename_map = {}
    for col in candidate_cols:
        if col.endswith("-voturi"):
            clean = _strip_voturi(col)
            # Keep last surname only (after last hyphen-separated component)
            parts = [p for p in re.split(r"[-\s]+", clean) if p]
            short = parts[-1] if parts else clean
            rename_map[col] = "cand_" + short[:20]
        elif col in ("g1", "g2"):
            rename_map[col] = f"cand_{col}"
    grp.rename(columns=rename_map, inplace=True)

    cand_cols_out = list(rename_map.values())
    for col in cand_cols_out:
        grp[col] = grp[col] / grp["valid_votes"]

    grp["turnout"] = grp["voted"] / grp["enrolled"]
    return grp


# ---------------------------------------------------------------------------
# 4.  EP 2024
#     Same layout as presidential (c = valid votes) but has party columns
# ---------------------------------------------------------------------------
def _ingest_ep_2024() -> pd.DataFrame:
    path = RAW / "ep_2024_judet.csv.csv"
    df = _read_csv_raw(path)

    party_cols = [c for c in df.columns if c.endswith("-voturi")]
    for col in ["a", "b", "c", "d"] + party_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["judet_norm"] = df["precinct_county_name"].apply(_norm_county)
    numeric_agg = [c for c in ["a", "b", "c", "d"] + party_cols if c in df.columns]
    grp = df.groupby("judet_norm")[numeric_agg].sum().reset_index()

    grp["election"] = "ep_2024"
    grp["year"]     = 2024
    grp["chamber"]  = "EP"
    grp.rename(columns={"judet_norm": "judet",
                         "a": "enrolled",
                         "b": "voted",
                         "c": "valid_votes",
                         "d": "null_votes"}, inplace=True)

    canonical_sums: dict[str, pd.Series] = {}
    for col in party_cols:
        name = _strip_voturi(col)
        canon = EP_CANONICAL.get(name, "ep_OTHER")
        if canon in canonical_sums:
            canonical_sums[canon] = canonical_sums[canon] + grp[col]
        else:
            canonical_sums[canon] = grp[col].copy()

    for canon, series in canonical_sums.items():
        grp[canon] = series

    grp.drop(columns=party_cols, errors="ignore", inplace=True)
    ep_cols = [c for c in grp.columns if c.startswith(("t_", "ep_"))]
    for col in ep_cols:
        grp[col] = grp[col] / grp["valid_votes"]

    grp["turnout"] = grp["voted"] / grp["enrolled"]
    return grp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("Ingesting AEP electoral series → electoral_series_judet.parquet")
    print("=" * 65)

    frames = []

    print("\n[1/6] 2020 Parliamentary (per-county files, CD + S)...")
    df20 = _ingest_2020_parliamentary()
    print(f"      {len(df20)} county×chamber rows, "
          f"{df20['judet'].nunique()} unique județe")
    frames.append(df20)

    print("[2/6] 2024 Parliamentary...")
    df24 = _ingest_2024_parliamentary()
    print(f"      {len(df24)} county×chamber rows")
    frames.append(df24)

    print("[3/6] EP 2024...")
    dfep = _ingest_ep_2024()
    print(f"      {len(dfep)} county rows")
    frames.append(dfep)

    print("[4/6] Prezidențiale 2019 R2 (Iohannis vs Dăncilă)...")
    df19 = _ingest_presidential(
        "prezidentiale_2019_r2_judet.csv.csv", 2019,
        "prezidentiale_2019_r2"
    )
    print(f"      {len(df19)} county rows, "
          f"candidates: {[c for c in df19.columns if c.startswith('cand_')]}")
    frames.append(df19)

    print("[5/6] Prezidențiale 2024 R1 (Georgescu)...")
    df24r1 = _ingest_presidential(
        "prezidentiale_2024_r1_judet.csv.csv", 2024,
        "prezidentiale_2024_r1"
    )
    print(f"      {len(df24r1)} county rows, "
          f"candidates: {[c for c in df24r1.columns if c.startswith('cand_')]}")
    frames.append(df24r1)

    print("[6/6] Prezidențiale 2025 R1 + R2 (Simion, Dan)...")
    df25r1 = _ingest_presidential(
        "prezidentiale_2025_r1_judet.csv.csv", 2025,
        "prezidentiale_2025_r1"
    )
    df25r2 = _ingest_presidential(
        "prezidentiale_2025_r2_judet.csv.csv", 2025,
        "prezidentiale_2025_r2"
    )
    print(f"      R1: {len(df25r1)} rows | R2: {len(df25r2)} rows")
    print(f"      R2 candidates: {[c for c in df25r2.columns if c.startswith('cand_')]}")
    frames.append(df25r1)
    frames.append(df25r2)

    # Combine all into one long-format table
    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Drop diaspora rows (not mappable to any județ)
    combined = combined[~combined["judet"].str.contains("STRĂIN", na=False)].copy()

    # Re-aggregate any split Bucharest rows (sectors yielded multiple rows
    # per election/chamber after concat — collapse them by summing counts
    # then recomputing shares)
    share_cols = [c for c in combined.columns
                  if c.startswith(("t_", "ep_", "cand_"))]
    count_cols  = ["enrolled", "voted", "valid_votes"]
    id_cols     = ["election", "year", "chamber", "judet"]

    def _reaggregate(sub: pd.DataFrame) -> pd.DataFrame:
        buc_mask = sub["judet"].str.contains("BUCURE", na=False)
        buc = sub[buc_mask]
        rest = sub[~buc_mask]
        if len(buc) <= 1:
            return sub
        # Sum counts, recompute shares
        buc_sum = buc[count_cols + share_cols].copy()
        # Undo shares → counts first
        for col in share_cols:
            if col in buc_sum.columns:
                buc_sum[col] = buc_sum[col] * buc["valid_votes"].values
        buc_agg = buc_sum.sum()
        for col in share_cols:
            if col in buc_agg.index and buc_agg["valid_votes"] > 0:
                buc_agg[col] = buc_agg[col] / buc_agg["valid_votes"]
        row = buc.iloc[[0]][id_cols].copy()
        for col in count_cols + share_cols:
            if col in buc_agg.index:
                row[col] = buc_agg[col]
        return pd.concat([rest, row], ignore_index=True)

    combined = (combined
                .groupby(["election", "year", "chamber"], group_keys=False)
                .apply(_reaggregate)
                .reset_index(drop=True))

    combined = combined.sort_values(["election", "judet", "chamber"]).reset_index(drop=True)

    # Report coverage
    print(f"\nCombined: {len(combined)} rows, {combined['election'].nunique()} elections")
    print(combined.groupby(["election", "chamber"])["judet"].count().to_string())

    # Recompute turnout after reaggregation
    combined["turnout"] = combined["voted"] / combined["enrolled"]

    # Save
    combined.to_parquet(OUT, index=False)
    print(f"\nSaved → {OUT.name}")

    # Quick enrolled summary for ghost electorate preview
    enrolled_2025 = (
        combined[combined["election"] == "prezidentiale_2025_r2"]
        [["judet", "enrolled", "voted", "turnout"]]
        .sort_values("judet")
    )
    print("\nEnrolled voters 2025 R2 (top 10):")
    print(enrolled_2025.head(10).to_string(index=False))
    total_enrolled = enrolled_2025["enrolled"].sum()
    print(f"\nTotal enrolled nationally (2025 R2): {total_enrolled:,.0f}")


if __name__ == "__main__":
    main()
