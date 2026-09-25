"""Download additional Eurostat datasets that were missing from the original download."""
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "eurostat"
BASE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}?format=TSV&compressed=false"

EXTRA = {
    "edat_lfse_16": "early_school_leaving NUTS 2",
    "ilc_peps11":   "at-risk-of-poverty NUTS 2",
}

for code, label in EXTRA.items():
    dest = RAW_DIR / f"{code}_full.tsv"
    if dest.exists():
        print(f"Already exists: {dest.name}")
        continue
    url = BASE.format(code=code)
    print(f"Downloading {code} ({label})...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"OK ({len(data)//1024} KB -> {dest.name})")
    except Exception as e:
        print(f"FAILED: {e}")
    time.sleep(1)

print("Done.")
