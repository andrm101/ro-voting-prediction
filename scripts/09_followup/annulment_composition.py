"""
Follow-up E — Annulment Effect on Vote-Choice Composition
==========================================================
The existing annulment test (09c) asks whether the annulment affected how many
people voted (Simion R1 turnout surprise). This script asks a different question:
did the annulment shift what counties voted for?

Specifically: did counties where Georgescu dominated the 2024 nationalist-populist
bloc show a disproportionately large nationalist uplift in 2025?

Definitions:
  bloc_2024 = georgescu_r1 + simion_r1_2024   (combined nationalist-populist share)
  bloc_2025 = simion_r1_2025                  (combined post-annulment nationalist share)
  nationalist_uplift = bloc_2025 - bloc_2024   (raw share growth of the bloc)

  georgescu_fraction = georgescu_r1 / bloc_2024
    — measures how "Georgescu-coded" vs "Simion-coded" a county's 2024 nationalist vote was
    — if the annulment specifically mobilised Georgescu loyalists, this fraction should
      predict a larger uplift even after structural controls

Models:
  Model A: nationalist_uplift ~ structural_controls (HC3)
           → residuals = structural surprise in the nationalist bloc expansion
  FWL:     georgescu_fraction | controls → nationalist_uplift residuals
  Model B: nationalist_uplift ~ georgescu_fraction + structural_controls

Secondary test:
  Does georgescu_fraction predict Simion R1 share (not uplift)?
  This tests whether the Georgescu-coded counties absorbed the Simion 2025 surge
  disproportionately.

Outputs:
  figures/13b_annulment_composition.png
  reports/13b_annulment_composition.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

# features parquet already contains georgescu_r1_2024 and simion_r1_2025.
# Only merge simion_r1_2024 (Simion's 2024 R1 share, needed for bloc fraction).
pres24 = series[series.election == "prezidentiale_2024_r1"].copy()
pres24["judet_norm"] = pres24["judet"].apply(_norm)
feat = feat.merge(
    pres24[["judet_norm", "cand_SIMION"]]
          .rename(columns={"cand_SIMION": "simion_r1_2024"}),
    on="judet_norm", how="left"
)
# Align column names to unified schema
feat = feat.rename(columns={"georgescu_r1_2024": "georgescu_r1"})

CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
]
CLUSTER_COLS = {
    "Moldova Profunda":   "#e41a1c",
    "Campia Uitata":      "#ff7f00",
    "Insula Capitalei":   "#4daf4a",
    "Centrele Dinamice":  "#377eb8",
    "Arcul Identitar":    "#984ea3",
}

df = feat.dropna(subset=["georgescu_r1", "simion_r1_2024", "simion_r1_2025"] + CONTROLS).copy()

# Construct outcomes
df["bloc_2024"]           = df["georgescu_r1"] + df["simion_r1_2024"]
df["bloc_2025"]           = df["simion_r1_2025"]
df["nationalist_uplift"]  = df["bloc_2025"] - df["bloc_2024"]
# Clamp to avoid division by zero (shouldn't occur with real data)
df["georgescu_fraction"]  = df["georgescu_r1"] / df["bloc_2024"].clip(lower=1e-6)

for col in CONTROLS:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
df["z_georgescu_fraction"] = (df["georgescu_fraction"] - df["georgescu_fraction"].mean()) / \
                              df["georgescu_fraction"].std()

z_ctrls = " + ".join(f"z_{c}" for c in CONTROLS)

print("=" * 65)
print("Follow-up E — Annulment Effect on Vote-Choice Composition")
print("=" * 65)
print(f"\n  n = {len(df)}  counties")
print(f"\n  Mean nationalist uplift (2024→2025): "
      f"{df['nationalist_uplift'].mean()*100:+.2f} pp  "
      f"(SD={df['nationalist_uplift'].std()*100:.2f} pp)")
print(f"  Mean Georgescu fraction of 2024 bloc: "
      f"{df['georgescu_fraction'].mean():.3f}  (SD={df['georgescu_fraction'].std():.3f})")

# Model A: structural controls only
mod_A = smf.ols(f"nationalist_uplift ~ {z_ctrls}", data=df).fit(cov_type="HC3")
df["resid_A"] = mod_A.resid
print(f"\n  Model A (structural only): R²={mod_A.rsquared:.3f}  n={int(mod_A.nobs)}")

# FWL partial correlation: Georgescu fraction | controls → nationalist uplift residuals
X_ctrl   = df[[f"z_{c}" for c in CONTROLS]].values
X_ctrl_c = sm.add_constant(X_ctrl)
e_gf = (df["z_georgescu_fraction"].values
        - sm.OLS(df["z_georgescu_fraction"].values, X_ctrl_c).fit().fittedvalues)
r_partial, p_partial = pearsonr(e_gf, mod_A.resid.values)

print(f"\n  FWL partial corr (Georgescu fraction | controls) → nationalist uplift residuals:")
print(f"    r = {r_partial:.4f}  p = {p_partial:.4f}")

# Model B: add Georgescu fraction
mod_B = smf.ols(f"nationalist_uplift ~ z_georgescu_fraction + {z_ctrls}",
                data=df).fit(cov_type="HC3")
b_gf  = mod_B.params["z_georgescu_fraction"]
se_gf = mod_B.bse["z_georgescu_fraction"]
p_gf  = mod_B.pvalues["z_georgescu_fraction"]
print(f"\n  Model B (+ Georgescu fraction):")
print(f"    R²={mod_B.rsquared:.3f}  adj-R²={mod_B.rsquared_adj:.3f}")
print(f"    β_georgescu_fraction = {b_gf:+.4f}  SE={se_gf:.4f}  p={p_gf:.4f}")
print(f"    95% CI: [{b_gf - 1.96*se_gf:.4f}, {b_gf + 1.96*se_gf:.4f}]")

if p_gf < 0.10:
    print(f"\n  >> Significant (p<0.10): Georgescu-coded counties "
          f"{'gained more' if b_gf > 0 else 'gained less'} nationalist share than expected")
else:
    print(f"\n  >> No significant composition annulment effect (p={p_gf:.3f})")

# Secondary: does Georgescu fraction predict Simion R1 share directly?
mod_C = smf.ols(f"simion_r1_2025 ~ z_georgescu_fraction + {z_ctrls}",
                data=df).fit(cov_type="HC3")
b_gf_c = mod_C.params["z_georgescu_fraction"]
p_gf_c = mod_C.pvalues["z_georgescu_fraction"]
print(f"\n  Secondary — Simion R1 share model:")
print(f"    β_georgescu_fraction = {b_gf_c:+.4f}  p={p_gf_c:.4f}  R²={mod_C.rsquared:.3f}")

# Magnitude
sd_gf = df["georgescu_fraction"].std()
implied_pp = b_gf * sd_gf * 100
print(f"\n  Implied effect on uplift: {implied_pp:+.2f} pp per +1 SD in Georgescu fraction")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: nationalist uplift residual vs Georgescu fraction
ax = axes[0]
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax.scatter(df.loc[m, "georgescu_fraction"],
               df.loc[m, "resid_A"] * 100,
               c=col, s=55, alpha=0.85, label=cl, edgecolors="white", lw=0.4)
xv = df["georgescu_fraction"].values
yv = df["resid_A"].values
sl, ic = np.polyfit(xv, yv, 1)
xs = np.linspace(xv.min(), xv.max(), 100)
ax.plot(xs, (sl * xs + ic) * 100, "k--", lw=1.2, alpha=0.8)
ax.axhline(0, lw=0.7, ls=":", c="grey")
for _, row in df.iterrows():
    if abs(row["resid_A"]) > 0.03 or row["georgescu_fraction"] > 0.7:
        ax.annotate(row["judet"], (row["georgescu_fraction"], row["resid_A"] * 100),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
handles = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
ax.legend(handles=handles, fontsize=6.5, loc="upper left")
ax.set_xlabel("Georgescu fraction of 2024 nationalist bloc", fontsize=10)
ax.set_ylabel("Nationalist uplift residual (pp)\n[actual − structural prediction]", fontsize=9)
ax.set_title(f"Composition annulment effect: r_partial={r_partial:.3f}  p={p_partial:.3f}",
             fontsize=10)

# Panel 2: raw nationalist uplift scatter (no residualisation)
ax2 = axes[1]
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax2.scatter(df.loc[m, "georgescu_fraction"],
                df.loc[m, "nationalist_uplift"] * 100,
                c=col, s=55, alpha=0.85, edgecolors="white", lw=0.4)
sl2, ic2 = np.polyfit(xv, df["nationalist_uplift"].values, 1)
ax2.plot(xs, (sl2 * xs + ic2) * 100, "k--", lw=1.2, alpha=0.8)
ax2.axhline(0, lw=0.7, ls=":", c="grey")
r_raw, p_raw = pearsonr(df["georgescu_fraction"], df["nationalist_uplift"])
ax2.set_xlabel("Georgescu fraction of 2024 nationalist bloc", fontsize=10)
ax2.set_ylabel("Nationalist uplift 2024→2025 (pp)\n[simion_2025 − (georgescu+simion)_2024]", fontsize=9)
ax2.set_title(f"Raw correlation: r={r_raw:.3f}  p={p_raw:.3f}", fontsize=10)

plt.suptitle("Annulment Effect on Nationalist Bloc Vote-Share Composition", fontsize=12, y=1.01)
plt.tight_layout()
out = FIG_DIR / "13b_annulment_composition.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved -> {out.name}")

# Save results
out_df = df[["judet", "cluster_short", "georgescu_r1", "simion_r1_2024",
             "simion_r1_2025", "bloc_2024", "bloc_2025",
             "nationalist_uplift", "georgescu_fraction", "resid_A"]].copy()
out_df["predicted_uplift_A"] = mod_A.fittedvalues.values
out_df = out_df.sort_values("resid_A", ascending=False)
out_df.to_csv(REP_DIR / "13b_annulment_composition.csv", index=False, float_format="%.4f")
print(f"  Saved -> reports/13b_annulment_composition.csv")

print(f"\n  Top 5 counties with highest positive uplift surprise:")
print(out_df[["judet", "georgescu_fraction", "nationalist_uplift", "resid_A"]].head(5)
      .to_string(index=False))
print(f"\n  Top 5 counties with largest negative uplift surprise:")
print(out_df[["judet", "georgescu_fraction", "nationalist_uplift", "resid_A"]].tail(5)
      .to_string(index=False))

print("\n" + "=" * 65)
print("FOLLOW-UP E COMPLETE")
print("=" * 65)
