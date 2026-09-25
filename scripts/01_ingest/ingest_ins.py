"""
Ingest INS Census 2021 data.

Expected files in data/raw/ins/:
  ins_etnie_religie_judet_2021.csv.xlsx
    Sheet 'Tab 2.5.2': county-level ethnicity × religion cross-tabulation.
    Layout: 25-row blocks per county, first row = county totals (all ethnicities),
    subsequent rows = per-ethnicity breakdowns.

  ins_populatie_2021.csv.xls
    Sheet 'TAB.1.1_RPL2021': total, urban, and rural population per county.
    Three sections separated by 'URBAN' and 'RURAL' row headers.

Output: data/processed/census_raw.parquet
  siruta | judet | category_type | category | count | share_of_total
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ins"
OUT_DIR = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "scripts"))
from constants import JUDET_SIRUTA

# Column indices in Tab 2.5.2 (0-based).
# Index 0 = county/ethnicity label, 1 = total population count.
# Indices 2-24 = religion counts in this order.
RELIGION_COLS: list[tuple[int, str]] = [
    (2,  "Ortodox"),
    (3,  "Romano-Catolic"),
    (4,  "Reformat"),
    (5,  "Penticostal"),
    (6,  "Greco-Catolic"),
    (7,  "Baptist"),
    (8,  "Adventist"),
    (9,  "Musulman"),
    (10, "Unitarian"),
    (11, "Martorii lui Iehova"),
    (20, "Fara religie"),
]

# Ethnicities we extract by label match (case-insensitive substring)
ETHNICITY_PATTERNS: list[tuple[str, str]] = [
    (r"rom[âa]ni",     "Romani"),
    (r"maghiari",      "Maghiari"),
    (r"romi\b",        "Romi"),
    (r"ucraineni",     "Ucraineni"),
    (r"germani",       "Germani"),
    (r"turci",         "Turci"),
    (r"rusi.lipoveni", "Rusi-Lipoveni"),
    (r"t[aă]tari",     "Tatari"),
]
_ETH_RE = [(re.compile(p, re.IGNORECASE), lbl) for p, lbl in ETHNICITY_PATTERNS]

# Counties in the expected order (alphabetical, as they appear in the file)
COUNTY_NAMES_INS: list[str] = [
    "ALBA", "ARAD", "ARGES", "BACAU", "BIHOR", "BISTRITA-NASAUD",
    "BOTOSANI", "BRASOV", "BRAILA", "BUZAU", "CARAS-SEVERIN", "CALARASI",
    "CLUJ", "CONSTANTA", "COVASNA", "DAMBOVITA", "DOLJ", "GALATI",
    "GIURGIU", "GORJ", "HARGHITA", "HUNEDOARA", "IALOMITA", "IASI",
    "ILFOV", "MARAMURES", "MEHEDINTI", "MURES", "NEAMT", "OLT",
    "PRAHOVA", "SATU MARE", "SALAJ", "SIBIU", "SUCEAVA", "TELEORMAN",
    "TIMIS", "TULCEA", "VASLUI", "VALCEA", "VRANCEA", "MUNICIPIUL BUCURESTI",
]

# Map uppercase (no-diacritics) county name → JUDET_SIRUTA key
_UPPER_TO_CANONICAL: dict[str, str] = {}
for _nm in JUDET_SIRUTA:
    _k = (_nm.upper()
          .replace("Ș","S").replace("Ț","T").replace("Ă","A")
          .replace("Â","A").replace("Î","I"))
    _UPPER_TO_CANONICAL[_k] = _nm


def _to_siruta_key(name: str) -> str | None:
    """Map an INS county label to its canonical JUDET_SIRUTA key."""
    clean = name.strip().upper()
    if clean in _UPPER_TO_CANONICAL:
        return _UPPER_TO_CANONICAL[clean]
    if "BUCURESTI" in clean or "BUCHAREST" in clean:
        return "Municipiul București"
    return None


def _safe_int(val) -> int | None:
    """Convert a cell value to int, returning None for missing/flagged entries."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("-", "*", "", "n.a.", "N/A"):
        return None
    try:
        return int(float(s.replace(" ", "").replace(",", ".")))
    except (ValueError, TypeError):
        return None


def parse_ethnicity_religion(path: Path) -> pd.DataFrame:
    """
    Parse Tab 2.5.2 of the ethnicity/religion xlsx.

    Each county block is ~25 rows:
      row 0: county name | total_pop | religion_1_count | ... | religion_N_count
      row 1: "ETNIA"
      row 2+: ethnicity | ethnicity_total | religion_1_for_ethnicity | ...

    Returns long-form DataFrame with category_type in {'ethnicity', 'religion'}.
    """
    df = pd.read_excel(path, sheet_name="Tab 2.5.2", header=None, dtype=object)

    records: list[dict] = []
    i = 0
    n = len(df)

    while i < n:
        cell0 = str(df.iloc[i, 0]).strip()

        # Check if this row is a county header
        canonical = _to_siruta_key(cell0)
        if canonical is None or canonical not in JUDET_SIRUTA:
            i += 1
            continue

        siruta = JUDET_SIRUTA[canonical]
        total_pop = _safe_int(df.iloc[i, 1])
        if total_pop is None or total_pop == 0:
            i += 1
            continue

        # Religion totals from the county header row (all ethnicities combined)
        for col_idx, rel_label in RELIGION_COLS:
            if col_idx < df.shape[1]:
                cnt = _safe_int(df.iloc[i, col_idx])
                if cnt is not None:
                    records.append({
                        "siruta": siruta,
                        "judet": canonical,
                        "category_type": "religion",
                        "category": rel_label,
                        "count": cnt,
                        "share_of_total": cnt / total_pop,
                    })

        # Walk the ethnicity sub-rows (skip the 'ETNIA' label row at i+1)
        j = i + 2
        while j < n and j < i + 30:
            sub0 = str(df.iloc[j, 0]).strip()
            # Stop at the next county or a blank separator
            if sub0 == "" or _to_siruta_key(sub0) is not None:
                break
            # Skip the 'ETNIA' label row if we encounter it
            if sub0.upper() in ("ETNIA", "ETNIE"):
                j += 1
                continue
            # Match ethnicity label
            eth_lbl = None
            for pat, lbl in _ETH_RE:
                if pat.search(sub0):
                    eth_lbl = lbl
                    break
            if eth_lbl is None:
                j += 1
                continue
            cnt = _safe_int(df.iloc[j, 1])
            if cnt is not None:
                records.append({
                    "siruta": siruta,
                    "judet": canonical,
                    "category_type": "ethnicity",
                    "category": eth_lbl,
                    "count": cnt,
                    "share_of_total": cnt / total_pop,
                })
            j += 1

        i += 1

    result = pd.DataFrame(records)
    print(f"  [OK] Ethnicity/religion: {result['siruta'].nunique()} counties, "
          f"{len(result)} rows")
    return result


def parse_urban_rural(path: Path) -> pd.DataFrame:
    """
    Parse the population xls for urban/rural split per county.

    The file has three sections:
      Section 1 (rows after 'ROMANIA'): total population
      Section 2 (rows after 'URBAN'): urban population
      Section 3 (rows after 'RURAL'): rural population

    Returns long-form DataFrame with category_type='urban_rural'.
    """
    df = pd.read_excel(path, header=None, dtype=object)
    year_col = 8  # Column index for 2021 census

    section = "total"
    total: dict[str, int] = {}
    urban: dict[str, int] = {}
    rural: dict[str, int] = {}

    for i in range(len(df)):
        cell = str(df.iloc[i, 0]).strip()
        if cell == "" or cell == "nan":
            continue

        upper = cell.upper().replace("Ș","S").replace("Ț","T").replace("Ă","A").replace("Â","A").replace("Î","I")

        if upper == "URBAN":
            section = "urban"
            continue
        if upper == "RURAL":
            section = "rural"
            continue

        canonical = _to_siruta_key(cell)
        if canonical is None or canonical not in JUDET_SIRUTA:
            # Also try stripping leading spaces
            canonical = _to_siruta_key(cell.strip())
        if canonical is None or canonical not in JUDET_SIRUTA:
            continue

        val = _safe_int(df.iloc[i, year_col])
        if val is None:
            continue

        siruta = JUDET_SIRUTA[canonical]
        key = (siruta, canonical)
        if section == "total":
            total[canonical] = val
        elif section == "urban":
            urban[canonical] = val
        elif section == "rural":
            rural[canonical] = val

    records: list[dict] = []
    for name, tot in total.items():
        if tot == 0:
            continue
        siruta = JUDET_SIRUTA[name]
        urb = urban.get(name, 0) or 0
        rur = rural.get(name, 0) or 0
        # București is entirely urban; rural may be '-' in the source
        if urb == 0 and rur == 0:
            urb = tot
        records.append({
            "siruta": siruta, "judet": name,
            "category_type": "urban_rural",
            "category": "Urban", "count": urb,
            "share_of_total": urb / tot,
        })
        records.append({
            "siruta": siruta, "judet": name,
            "category_type": "urban_rural",
            "category": "Rural", "count": rur,
            "share_of_total": rur / tot,
        })

    result = pd.DataFrame(records)
    print(f"  [OK] Urban/rural: {result['siruta'].nunique()} counties")
    return result


def main() -> None:
    etnie_files = list(RAW_DIR.glob("*etnie*religie*.xlsx")) + \
                  list(RAW_DIR.glob("*etnie*religie*.csv.xlsx"))
    pop_files   = list(RAW_DIR.glob("*populatie*.xls")) + \
                  list(RAW_DIR.glob("*populatie*.csv.xls"))

    if not etnie_files and not pop_files:
        print(
            f"No INS files found in {RAW_DIR}.\n"
            "Expected:\n"
            "  ins_etnie_religie_judet_2021.csv.xlsx\n"
            "  ins_populatie_2021.csv.xls\n"
            "Download from https://insse.ro/cms/ro/content/recensamant-populatie-2021",
            file=sys.stderr,
        )
        sys.exit(1)

    frames: list[pd.DataFrame] = []

    if etnie_files:
        print(f"Processing {etnie_files[0].name} …")
        frames.append(parse_ethnicity_religion(etnie_files[0]))
    else:
        print("  [WARN] No ethnicity/religion file found.", file=sys.stderr)

    if pop_files:
        print(f"Processing {pop_files[0].name} …")
        frames.append(parse_urban_rural(pop_files[0]))
    else:
        print("  [WARN] No population (urban/rural) file found.", file=sys.stderr)

    if not frames:
        print("No data parsed.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "census_raw.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined):,} rows -> {out_path}")
    print(f"Category types: {sorted(combined['category_type'].unique())}")
    print(f"Counties with SIRUTA: {combined['siruta'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
