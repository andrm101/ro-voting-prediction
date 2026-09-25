"""
Follow-up G — Placebo Transfer Matrix
======================================
The Georgescu→Simion partial correlation (r_partial=0.770) is the paper's
central finding. Without a baseline, it is unclear whether this is unusually
high for Romanian presidential elections or merely normal bloc persistence.

This script computes the FWL partial correlation for a matrix of
origin→destination pairs spanning:
  - Within-bloc transfers (expected high): nationalist→nationalist, reformist→reformist
  - Cross-bloc transfers (expected near-zero or negative)
  - Historical baselines from 2019 R2 (only presidential round available)

Elections available:
  2019 R2: cand_g1 ≈ Iohannis (61.7%), cand_g2 ≈ Dăncilă/PSD (38.3%)
  2024 R1: GEORGESCU, SIMION, LASCONI, CIOLACU, CIUCĂ
  2025 R1: SIMION, ANTONESCU, DAN, PONTA, LASCONI

Transfer pairs:
  [Main finding]
  1. Georgescu 2024 R1  → Simion 2025 R1      (nationalist continuity, post-annulment)
  [Within-bloc baselines]
  2. Simion 2024 R1     → Simion 2025 R1      (same-candidate continuity — upper bound)
  3. Lasconi 2024 R1    → Dan 2025 R1         (reformist continuity)
  4. Iohannis 2019 R2   → Dan 2025 R1         (establishment → reformist, across cycle)
  [Cross-camp controls — expected near-zero or negative]
  5. Georgescu 2024 R1  → Dan 2025 R1         (nationalist → reformist, should be negative)
  6. Iohannis 2019 R2   → Simion 2025 R1      (establishment → nationalist, should be negative)
  [Additional PSD-bloc signal]
  7. Dăncilă 2019 R2    → Simion 2025 R1      (PSD-left populist → nationalist)
  8. Ciolacu 2024 R1    → Simion 2025 R1      (PSD 2024 → nationalist 2025)
  9. Ciolacu 2024 R1    → Dan 2025 R1         (PSD 2024 → reformist 2025)

Outputs:
  figures/13c_placebo_transfer_heatmap.png
  reports/13c_placebo_transfer.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"

feat  = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
feat  = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)

# Pull candidate shares
PULL = {
    # (election, cand_col): new_col_name
    ("prezidentiale_2019_r2", "cand_g1"):          "iohannis_2019r2",
    ("prezidentiale_2019_r2", "cand_g2"):          "dancila_2019r2",
    ("prezidentiale_2024_r1", "cand_GEORGESCU"):   "georgescu_2024r1",
    ("prezidentiale_2024_r1", "cand_SIMION"):      "simion_2024r1",
    ("prezidentiale_2024_r1", "cand_LASCONI"):     "lasconi_2024r1",
    ("prezidentiale_2024_r1", "cand_CIOLACU"):     "ciolacu_2024r1",
    ("prezidentiale_2025_r1", "cand_SIMION"):      "simion_2025r1",
    ("prezidentiale_2025_r1", "cand_DAN"):         "dan_2025r1",
    ("prezidentiale_2025_r1", "cand_LASCONI"):     "lasconi_2025r1",
}

for (elec, cand), new_col in PULL.items():
    sub = series[series.election == elec][["judet", cand]].copy()
    sub["judet_norm"] = sub["judet"].apply(_norm)
    feat = feat.merge(sub[["judet_norm", cand]].rename(columns={cand: new_col}),
                      on="judet_norm", how="left")

CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
]

df = feat.dropna(subset=CONTROLS).copy()
for col in CONTROLS:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

X_ctrl   = df[[f"z_{c}" for c in CONTROLS]].values
X_ctrl_c = sm.add_constant(X_ctrl)

def fwl_partial(origin_col: str, dest_col: str) -> tuple[float, float, int]:
    """Return (r_partial, p_value, n) using FWL on the full df, dropping NAs."""
    valid = df[[origin_col, dest_col]].notna().all(axis=1)
    n = valid.sum()
    if n < 10:
        return np.nan, np.nan, n
    X_c = sm.add_constant(X_ctrl_c[valid.values])
    e_x = (df.loc[valid, origin_col].values
           - sm.OLS(df.loc[valid, origin_col].values, X_c).fit().fittedvalues)
    e_y = (df.loc[valid, dest_col].values
           - sm.OLS(df.loc[valid, dest_col].values, X_c).fit().fittedvalues)
    r, p = pearsonr(e_x, e_y)
    return round(r, 4), round(p, 4), n

# Define transfer pairs
PAIRS = [
    ("georgescu_2024r1", "simion_2025r1",   "Georgescu 2024 R1",  "Simion 2025 R1",   "Nationalist continuity (main)"),
    ("simion_2024r1",    "simion_2025r1",   "Simion 2024 R1",     "Simion 2025 R1",   "Same-candidate continuity"),
    ("lasconi_2024r1",   "dan_2025r1",      "Lasconi 2024 R1",    "Dan 2025 R1",      "Reformist continuity"),
    ("iohannis_2019r2",  "dan_2025r1",      "Iohannis 2019 R2",   "Dan 2025 R1",      "Establishment → reformist (cross-cycle)"),
    ("georgescu_2024r1", "dan_2025r1",      "Georgescu 2024 R1",  "Dan 2025 R1",      "Cross-camp: nationalist → reformist"),
    ("iohannis_2019r2",  "simion_2025r1",   "Iohannis 2019 R2",   "Simion 2025 R1",   "Cross-camp: establishment → nationalist"),
    ("dancila_2019r2",   "simion_2025r1",   "Dăncilă 2019 R2",   "Simion 2025 R1",   "PSD-left populist → nationalist"),
    ("ciolacu_2024r1",   "simion_2025r1",   "Ciolacu 2024 R1",    "Simion 2025 R1",   "PSD 2024 → nationalist 2025"),
    ("ciolacu_2024r1",   "dan_2025r1",      "Ciolacu 2024 R1",    "Dan 2025 R1",      "PSD 2024 → reformist 2025"),
]

print("=" * 75)
print("Follow-up G — Placebo Transfer Matrix")
print("=" * 75)
print(f"\n{'Origin':<22} {'Destination':<18} {'r_partial':>10} {'p':>8}  {'n':>4}  Description")
print("-" * 75)

results = []
for orig_col, dest_col, orig_lbl, dest_lbl, desc in PAIRS:
    r, p, n = fwl_partial(orig_col, dest_col)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
    print(f"  {orig_lbl:<20} {dest_lbl:<18} {r:>10.4f} {p:>8.4f}  {n:>3}  {sig}")
    results.append({
        "origin": orig_lbl, "destination": dest_lbl,
        "r_partial": r, "p_value": p, "n": n, "description": desc
    })

results_df = pd.DataFrame(results)
results_df.to_csv(REP_DIR / "13c_placebo_transfer.csv", index=False, float_format="%.4f")

# Build transfer matrix for heatmap (origins × destinations)
ORIGINS = [
    "Iohannis 2019 R2",
    "Dăncilă 2019 R2",
    "Georgescu 2024 R1",
    "Simion 2024 R1",
    "Lasconi 2024 R1",
    "Ciolacu 2024 R1",
]
DESTINATIONS = [
    "Simion 2025 R1",
    "Dan 2025 R1",
]

matrix = pd.DataFrame(np.nan, index=ORIGINS, columns=DESTINATIONS)
pvals  = pd.DataFrame(np.nan, index=ORIGINS, columns=DESTINATIONS)

for row in results:
    o, d = row["origin"], row["destination"]
    if o in matrix.index and d in matrix.columns:
        matrix.loc[o, d] = row["r_partial"]
        pvals.loc[o, d]  = row["p_value"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix.values.astype(float), cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
plt.colorbar(im, ax=ax, label="Partial correlation $r$")

ax.set_xticks(range(len(DESTINATIONS)))
ax.set_yticks(range(len(ORIGINS)))
ax.set_xticklabels(DESTINATIONS, rotation=20, ha="right", fontsize=9)
ax.set_yticklabels(ORIGINS, fontsize=9)

for i in range(len(ORIGINS)):
    for j in range(len(DESTINATIONS)):
        v = matrix.values[i, j]
        p = pvals.values[i, j]
        if not np.isnan(v):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
            ax.text(j, i, f"{v:.3f}{sig}", ha="center", va="center",
                    fontsize=8.5, color="white" if abs(v) > 0.45 else "black")

ax.set_title("FWL partial correlations: presidential transfer matrix\n"
             "(controlling for 8 structural predictors)\n"
             "*** p<0.001  ** p<0.01  * p<0.05  . p<0.10",
             fontsize=9)
ax.axhline(1.5, color="white", lw=1.5)  # visual separator: 2019 vs 2024
ax.axhline(3.5, color="white", lw=1.5)  # separator: nationalist vs reformist-PSD

plt.tight_layout()
out = FIG_DIR / "13c_placebo_transfer_heatmap.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved -> {out.name}")
print(f"  Saved -> reports/13c_placebo_transfer.csv")

print("\n" + "=" * 75)
print("FOLLOW-UP G COMPLETE")
print("=" * 75)
