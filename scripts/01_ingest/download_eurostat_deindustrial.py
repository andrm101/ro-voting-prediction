"""
Download three new Eurostat datasets to support the deindustrialisation thesis
and NW/Moldova-Wallachia differentiation.

Datasets:
  nama_10r_3gva  — GVA by NACE sector at NUTS3 (MIO_EUR, multi-year)
                   Used to compute: industry_gva_share_2022,
                   gva_deindustrial_index (share_2010 - share_2022)
  lfst_r_lfu2ltu — Long-term unemployment rate (12+ months) at NUTS2
                   Structural joblessness: distinguishes Wallachia ex-industry
                   from Moldova seasonal/frictional unemployment
  nama_10r_2hhinc — Household income per capita (PPS) at NUTS2
                    Welfare proxy beyond GDP: captures transfer-dependent counties
"""
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "eurostat"
BASE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}?format=TSV&compressed=false"

DATASETS = {
    "nama_10r_3gva":   "GVA by sector NUTS3 (deindustrialisation index)",
    "lfst_r_lfu2ltu":  "Long-term unemployment NUTS2",
    "nama_10r_2hhinc": "Household income per capita PPS NUTS2",
}

for code, label in DATASETS.items():
    dest = RAW_DIR / f"{code}_full.tsv"
    if dest.exists():
        print(f"Already exists: {dest.name}  ({dest.stat().st_size // 1024} KB)")
        continue
    url = BASE.format(code=code)
    print(f"Downloading {code} ({label}) ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"OK ({len(data) // 1024} KB -> {dest.name})")
    except Exception as e:
        print(f"FAILED: {e}")
    time.sleep(2)

print("Done.")
