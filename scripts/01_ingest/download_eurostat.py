"""
One-time download utility for Eurostat bulk TSV files and Romania NUTS 3 GeoJSON.

Run this ONCE after setting up the environment. All files are static bulk
downloads from public Eurostat APIs — no authentication required.

Usage:
  python scripts/01_ingest/download_eurostat.py

Downloads to:
  data/raw/eurostat/<dataset_code>_full.tsv
  app/src/data/geojson/ro_judete.geojson  (filtered to Romania NUTS 3)
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw" / "eurostat"
GEO_DIR  = ROOT / "app" / "src" / "data" / "geojson"

# Eurostat SDMX REST API — returns full TSV for each dataset code.
EUROSTAT_BASE = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
    "/{code}?format=TSV&compressed=false"
)

# Datasets to download (code → human label)
DATASETS: dict[str, str] = {
    "nama_10r_2gdp":     "GDP at current market prices, NUTS 2",
    "lfst_r_lfu3rt":     "Unemployment rate, NUTS 2",
    "edat_lfse_04":      "Educational attainment 25–64, NUTS 2",
    "edat_lfse_14":      "Early school leaving, NUTS 2",
    "ilc_peps13":        "At-risk-of-poverty, NUTS 2",
    "isoc_r_iuse_i":     "Internet use by individuals, NUTS 2",
    "demo_r_d2jan":      "Population density, NUTS 3",
    "demo_r_pjanaggr3":  "Population by age group, NUTS 3",
    "demo_r_gind3":      "Population change / net migration, NUTS 3",
    "nama_10r_3empers":  "Employment by economic activity, NUTS 3",
}

# Eurostat GISCO — NUTS 3 boundaries, 2021 edition, WGS84, 1:1M resolution
GEOJSON_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson"
    "/NUTS_RN_01M_2021_4326_LEVL_3.geojson"
)


def _download(url: str, dest: Path, label: str, retries: int = 3) -> bool:
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {label} …", end=" ", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.write_bytes(data)
            print(f"OK ({len(data) / 1024:.0f} KB → {dest.name})")
            return True
        except Exception as exc:
            print(f"FAIL (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(5)
    return False


def download_eurostat() -> None:
    print("\n-- Eurostat TSV datasets --")
    failed = []
    for code, label in DATASETS.items():
        url  = EUROSTAT_BASE.format(code=code)
        dest = RAW_DIR / f"{code}_full.tsv"
        ok = _download(url, dest, f"{code} ({label})")
        if not ok:
            failed.append(code)
        time.sleep(1)  # be polite to the API

    if failed:
        print(f"\n  [WARN] Failed downloads: {failed}")
        print("  Retry manually or download from:")
        print("  https://ec.europa.eu/eurostat/databrowser/")


def download_geojson() -> None:
    print("\n-- Romania NUTS 3 GeoJSON --")
    raw_geo = GEO_DIR / "NUTS_RN_01M_2021_4326_LEVL_3_raw.geojson"
    ok = _download(GEOJSON_URL, raw_geo, "NUTS 3 boundaries (all Europe)")
    if not ok:
        print("  GeoJSON download failed. Maps will fall back to bar charts.")
        return

    # Filter to Romania only to keep the app-facing file small
    try:
        import json
        print("  Filtering to Romania (RO*) …", end=" ", flush=True)
        with open(raw_geo, encoding="utf-8") as f:
            geojson = json.load(f)
        ro_features = [
            feat for feat in geojson["features"]
            if str(feat.get("properties", {}).get("NUTS_ID", "")).startswith("RO")
            and str(feat.get("properties", {}).get("LEVL_CODE", "")) == "3"
        ]
        ro_geojson = {**geojson, "features": ro_features}
        out_path = GEO_DIR / "ro_judete.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ro_geojson, f, ensure_ascii=False)
        raw_geo.unlink()  # remove the large unfiltered file
        print(f"OK ({len(ro_features)} features → {out_path.name})")
    except Exception as exc:
        print(f"  [ERROR] Could not filter GeoJSON: {exc}", file=sys.stderr)


def main() -> None:
    print("RO-Voting-Prediction — One-time data download")
    print("=" * 55)
    download_eurostat()
    download_geojson()
    print("\n[DONE] Check data/raw/eurostat/ and app/src/data/geojson/")
    print("\nStill required — manual downloads:")
    print("  roaep.ro election CSVs  → data/raw/roaep/")
    print("  INS Census 2021 files   → data/raw/ins/")
    print("  See each folder's README.md for exact instructions.")


if __name__ == "__main__":
    main()
