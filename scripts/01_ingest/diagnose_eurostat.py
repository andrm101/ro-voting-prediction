"""Diagnostic: show available dimension values for Romania in key datasets."""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "eurostat"

for fname in ["demo_r_pjanaggr3_full.tsv", "nama_10r_3empers_full.tsv",
              "edat_lfse_16_full.tsv", "ilc_peps11_full.tsv"]:
    path = RAW / fname
    if not path.exists():
        print(f"NOT FOUND: {fname}")
        continue

    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str,
                     keep_default_na=False, nrows=5000)
    first = df.columns[0]
    df["geo"] = df[first].str.split(r"[,\\]").str[-1].str.strip()
    ro = df[df["geo"].str.startswith("RO")].copy()

    print(f"\n=== {fname} ===")
    print(f"Dimensions: {first}")
    print(f"RO rows: {len(ro)}")

    dims = [d.strip() for d in first.split(",")]
    dims[-1] = dims[-1].split("\\")[0].strip()
    dims = [d for d in dims if d and d != "freq"]

    for i, dim in enumerate(dims[:-1]):  # exclude geo
        values = ro[first].str.split(r"[,\\]").str[i+1].value_counts().head(20)
        print(f"  {dim}: {list(values.index)[:10]}")
