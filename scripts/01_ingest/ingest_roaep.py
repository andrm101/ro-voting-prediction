"""
Ingest raw election result files from roaep.ro.

All files are precinct-level exports. This script aggregates them to
județ (county) level. Two export formats are handled:

  New format (2024/2025):
    precinct_county_name | ... | <PARTY NAME>-voturi | ...
    Vote columns end with '-voturi'.

  Old format (2019):
    Județ | ... | g1 | g2 | ...
    Numeric candidate columns mapped by position from constants.py.

Output: data/processed/elections_raw.parquet
  siruta | judet | election_id | election_type | year | round
  party_raw | votes | valid_votes | registered | turnout | share
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "roaep"
OUT_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "scripts"))
from constants import JUDET_SIRUTA, PRESIDENTIAL_CANDIDATES

# Diaspora / abroad votes — not a NUTS 3 unit, always excluded
STRAINATATE_PATTERNS = re.compile(r"(?i)(str[aă]in[aă]tate|abroad|diaspora|exterior)")

# Uppercase ASCII → canonical SIRUTA key (generated from JUDET_SIRUTA)
_UPPER_ASCII_MAP: dict[str, str] = {}
for _name in JUDET_SIRUTA:
    _key = unicodedata.normalize("NFKD", _name.upper())
    _key = "".join(c for c in _key if unicodedata.category(c) != "Mn")
    _UPPER_ASCII_MAP[_key] = _name


class ElectionMeta(NamedTuple):
    election_id: str
    election_type: str
    year: int
    round: str | None


def _parse_filename(path: Path) -> ElectionMeta | None:
    # Handle double extension like 'parlamentare_2024_judet.csv.csv'
    stem = path.stem  # may still contain an inner .csv
    stem = stem.replace(".csv", "").replace(".tsv", "")
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    etype = parts[0]
    try:
        year = int(parts[1])
    except ValueError:
        return None
    rnd = parts[2] if len(parts) > 2 and parts[2] in ("r1", "r2") else None
    eid = f"{etype}_{year}" + (f"_{rnd}" if rnd else "")
    return ElectionMeta(election_id=eid, election_type=etype, year=year, round=rnd)


def _strip_diacritics(text: str) -> str:
    """Uppercase, decompose, remove combining marks → ASCII approximation."""
    nfkd = unicodedata.normalize("NFKD", text.upper())
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _normalise_county_name(raw: str) -> str:
    raw = str(raw).strip()
    # Diaspora entries → mark for exclusion
    if STRAINATATE_PATTERNS.search(raw):
        return "__EXCLUDE__"
    # București sectors → canonical (handles "Sector 1", "MUNICIPIUL BUCUREȘTI - SECTOR 2", etc.)
    stripped_upper = _strip_diacritics(raw)
    if re.search(r"(?i)(sector\s*\d|sector\s*[1-6])", stripped_upper):
        return "Municipiul București"
    if re.match(r"(?i)^(mun\.?\s*bucure.ti|bucure.ti)$", stripped_upper):
        return "Municipiul București"
    # Strip diacritics, uppercase, look up canonical
    stripped = _strip_diacritics(raw)
    if stripped in _UPPER_ASCII_MAP:
        return _UPPER_ASCII_MAP[stripped]
    return raw


def _find_county_col(df: pd.DataFrame) -> str | None:
    candidates = ["precinct_county_name", "Județ", "Judet", "judet", "JUDET",
                  "județ", "Județ\n", "cod_judet"]
    for c in df.columns:
        if c.strip().rstrip("\n") in candidates:
            return c
    # Fall back: first column that looks like county names
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().head(5).astype(str).tolist()
            if any(s.upper().replace("Ș","S").replace("Ț","T")
                   .replace("Ă","A").replace("Â","A").replace("Î","I")
                   in _UPPER_ASCII_MAP for s in sample):
                return c
    return None


def _find_vote_cols(df: pd.DataFrame, meta: ElectionMeta) -> dict[str, str]:
    """Return {raw_col: party_label} for all vote columns."""
    # New format: columns ending with '-voturi'
    voturi = {c: c.removesuffix("-voturi").strip()
              for c in df.columns if str(c).endswith("-voturi")}
    if voturi:
        return voturi

    # Old format: positional g1, g2, ... mapped to candidate names
    g_cols = sorted(
        [c for c in df.columns if re.match(r"^g\d+$", str(c))],
        key=lambda x: int(x[1:]),
    )
    if g_cols:
        cand_list = PRESIDENTIAL_CANDIDATES.get(meta.election_id, [])
        if len(cand_list) < len(g_cols):
            print(f"  [WARN] Fewer candidates in constants ({len(cand_list)}) "
                  f"than g-columns ({len(g_cols)}) for {meta.election_id}",
                  file=sys.stderr)
        return {g: cand_list[i] if i < len(cand_list) else f"CAND_{i+1}"
                for i, g in enumerate(g_cols)}

    return {}


def _registered_col(df: pd.DataFrame, etype: str) -> str | None:
    """Return the metadata column holding registered-voter counts."""
    # prezidentiale: 'e' = total registered; parlamentare/ep: 'c' = total registered
    preferred = "e" if etype == "prezidentiale" else "c"
    fallback = "a"
    for col in [preferred, fallback]:
        if col in df.columns:
            return col
    return None


def ingest_file(path: Path) -> pd.DataFrame | None:
    meta = _parse_filename(path)
    if meta is None:
        print(f"  [SKIP] Cannot parse metadata from: {path.name}", file=sys.stderr)
        return None

    # Read — try comma first, then tab, then Excel
    df = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep,
                             encoding="utf-8-sig", dtype=str,
                             keep_default_na=False)
            if df.shape[1] > 5:
                break
        except Exception:
            pass
    if df is None or df.shape[1] <= 5:
        try:
            df = pd.read_excel(path, dtype=str)
        except Exception as exc:
            print(f"  [ERROR] Cannot read {path.name}: {exc}", file=sys.stderr)
            return None

    df.columns = [str(c).strip() for c in df.columns]

    county_col = _find_county_col(df)
    if county_col is None:
        print(f"  [ERROR] {path.name}: cannot identify county column", file=sys.stderr)
        return None

    vote_map = _find_vote_cols(df, meta)
    if not vote_map:
        print(f"  [WARN] {path.name}: no vote columns found — skipping", file=sys.stderr)
        return None

    reg_col = _registered_col(df, meta.election_type)

    # Convert numeric columns
    for col in list(vote_map) + ([reg_col] if reg_col else []):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[county_col] = df[county_col].apply(_normalise_county_name)
    # Drop diaspora / abroad rows
    df = df[df[county_col] != "__EXCLUDE__"].copy()

    # Aggregate to county level
    agg_cols = list(vote_map.keys()) + ([reg_col] if reg_col else [])
    county_df = (
        df.groupby(county_col, sort=False)[agg_cols]
        .sum()
        .reset_index()
        .rename(columns={county_col: "judet"})
    )

    county_df["valid_votes"] = county_df[list(vote_map)].sum(axis=1)
    county_df["registered"] = (
        county_df[reg_col] if reg_col else pd.NA
    )
    county_df["turnout"] = (
        county_df["valid_votes"] / county_df["registered"].replace(0, pd.NA)
        if reg_col else pd.NA
    )
    county_df["siruta"] = county_df["judet"].map(JUDET_SIRUTA)
    unmatched = county_df[county_df["siruta"].isna()]["judet"].unique()
    if len(unmatched):
        print(f"  [WARN] {path.name}: unmatched county names: {list(unmatched)}",
              file=sys.stderr)

    # Melt to long form: one row per party per county
    id_cols = ["siruta", "judet", "valid_votes", "registered", "turnout"]
    long = county_df[id_cols + list(vote_map)].melt(
        id_vars=id_cols, var_name="party_raw", value_name="votes"
    )
    long["party_raw"] = long["party_raw"].map(vote_map)
    long["share"] = long["votes"] / long["valid_votes"].replace(0, pd.NA)
    long["election_id"] = meta.election_id
    long["election_type"] = meta.election_type
    long["year"] = meta.year
    long["round"] = meta.round

    print(f"  [OK] {path.name} -> {county_df['judet'].nunique()} counties, "
          f"{len(vote_map)} parties, {len(long)} rows")
    return long


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        print(
            f"No files found in {RAW_DIR}.\n"
            "Download election results from https://www.roaep.ro/ and place them\n"
            "in data/raw/roaep/ following the naming convention in\n"
            "data/raw/roaep/README.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for f in files:
        print(f"Processing {f.name} …")
        result = ingest_file(f)
        if result is not None:
            frames.append(result)

    if not frames:
        print("No files successfully parsed.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "elections_raw.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"\nSaved {len(combined):,} rows -> {out_path}")
    print(f"Elections: {sorted(combined['election_id'].unique())}")
    siruta_match = combined['siruta'].notna().sum()
    print(f"Rows with SIRUTA match: {siruta_match} / {len(combined)} "
          f"({siruta_match / len(combined):.1%})")


if __name__ == "__main__":
    main()
