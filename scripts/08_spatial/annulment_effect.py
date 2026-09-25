"""
Task 9 — Annulment Effect Test
================================
Tests whether counties with high Georgescu 2024 R1 support showed
systematically *different* Simion 2025 R1 turnout than expected from
structural predictors alone, after the Constitutional Court annulment.

Two hypotheses:
  H_indignation: β_Georgescu > 0  — high-Georgescu counties mobilised harder
                                     for Simion in anger at the annulment
  H_abstention:  β_Georgescu < 0  — high-Georgescu voters were demobilised/
                                     disillusioned and stayed home

Method:
  Model A: simion_r1_turnout ~ structural_controls  (HC3 SEs)
           → residuals = structural surprise in Simion turnout
  Model B: simion_r1_turnout ~ georgescu_r1_share + structural_controls
           → β_Georgescu tests the annulment mobilisation hypothesis

  Also: FWL partial correlation of Georgescu share on Simion turnout
  residuals (mirrors Section 5 transfer analysis methodology).

Outputs:
  reports/09c_annulment_effect.csv   — county-level predicted/actual/residuals
  figures/09c_annulment_scatter.png  — scatter of Simion turnout residual vs
                                       Georgescu R1 share
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
feat  = feat.merge(clust[["judet","cluster_short"]], on="judet", how="left")
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)

# Georgescu R1 2024
pres24r1 = series[series.election=="prezidentiale_2024_r1"].copy()
pres24r1["judet_norm"] = pres24r1["judet"].apply(_norm)
feat = feat.merge(pres24r1[["judet_norm","cand_GEORGESCU","turnout"]]
                  .rename(columns={"cand_GEORGESCU":"georgescu_r1","turnout":"turnout_2024r1"}),
                  on="judet_norm", how="left")

# Simion R1 2025 turnout
pres25r1 = series[series.election=="prezidentiale_2025_r1"].copy()
pres25r1["judet_norm"] = pres25r1["judet"].apply(_norm)
feat = feat.merge(pres25r1[["judet_norm","cand_SIMION","turnout"]]
                  .rename(columns={"cand_SIMION":"simion_r1","turnout":"turnout_2025r1"}),
                  on="judet_norm", how="left")

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

df = feat.dropna(subset=["turnout_2025r1","georgescu_r1"] + CONTROLS).copy()
for col in CONTROLS:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
df["z_georgescu"] = (df["georgescu_r1"] - df["georgescu_r1"].mean()) / df["georgescu_r1"].std()

z_ctrls = " + ".join(f"z_{c}" for c in CONTROLS)

print("=" * 65)
print("Task 9 — Annulment Effect Test")
print("=" * 65)

# Model A: structural controls only → residuals = annulment surprise
mod_A = smf.ols(f"turnout_2025r1 ~ {z_ctrls}", data=df).fit(cov_type="HC3")
df["resid_A"] = mod_A.resid

# FWL partial correlation
X_ctrl = df[[f"z_{c}" for c in CONTROLS]].values
X_ctrl_c = sm.add_constant(X_ctrl)
e_G = df["z_georgescu"].values - sm.OLS(df["z_georgescu"].values, X_ctrl_c).fit().fittedvalues
r_partial, p_partial = pearsonr(e_G, mod_A.resid.values)

print(f"\n  Model A (structural only):  R²={mod_A.rsquared:.3f}  n={int(mod_A.nobs)}")
print(f"  Partial corr (Georgescu | controls) on Simion turnout residuals:")
print(f"    r = {r_partial:.4f}  p = {p_partial:.4f}")

# Model B: add Georgescu share directly
mod_B = smf.ols(f"turnout_2025r1 ~ z_georgescu + {z_ctrls}", data=df).fit(cov_type="HC3")
b_georg = mod_B.params["z_georgescu"]
se_georg = mod_B.bse["z_georgescu"]
p_georg  = mod_B.pvalues["z_georgescu"]

print(f"\n  Model B (+ Georgescu share):")
print(f"    R²={mod_B.rsquared:.3f}  adj-R²={mod_B.rsquared_adj:.3f}")
print(f"    β_Georgescu = {b_georg:+.4f}  SE={se_georg:.4f}  p={p_georg:.4f}")

if p_georg < 0.10:
    direction = "INDIGNATION MOBILISATION" if b_georg > 0 else "PROTEST ABSTENTION"
    print(f"\n  >> Significant effect (p<0.10): {direction}")
else:
    print(f"\n  >> No significant annulment effect (p={p_georg:.3f})")
    print(f"     95% CI: [{b_georg - 1.96*se_georg:.4f}, {b_georg + 1.96*se_georg:.4f}]")

# Magnitude: what is the implied pp effect?
sd_georg = df["georgescu_r1"].std()
implied_pp = b_georg * sd_georg * 100
print(f"\n  Implied effect: {implied_pp:+.2f} pp turnout per +1 SD in Georgescu share")
print(f"  (1 SD in Georgescu share ≈ {sd_georg:.3f})")

# ---------------------------------------------------------------------------
# Figure: Simion R1 turnout residual vs Georgescu R1 share
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax.scatter(df.loc[m, "georgescu_r1"]*100, df.loc[m, "resid_A"]*100,
               c=col, s=55, alpha=0.85, label=cl, edgecolors="white", lw=0.4)
# OLS line
x = df["georgescu_r1"].values
y = df["resid_A"].values
slope, intercept = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs*100, (slope*xs+intercept)*100, "k--", lw=1.2, alpha=0.8)
ax.axhline(0, lw=0.7, ls=":", c="grey")
ax.set_xlabel("Georgescu R1 2024 share (%)", fontsize=10)
ax.set_ylabel("Simion R1 2025 turnout residual (pp)\n[actual − structural prediction]", fontsize=9)
ax.set_title(f"Annulment effect: r_partial={r_partial:.3f}  p={p_partial:.3f}", fontsize=10)
for _, row in df.iterrows():
    if abs(row["resid_A"]) > 0.04 or row["georgescu_r1"] > 0.28:
        ax.annotate(row["judet"], (row["georgescu_r1"]*100, row["resid_A"]*100),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
handles = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
ax.legend(handles=handles, fontsize=6.5, loc="upper left")

# Panel 2: Model B coefficient plot (Georgescu + top controls)
ax2 = axes[1]
params = mod_B.params.drop("Intercept")
cis    = mod_B.conf_int().drop("Intercept")
colors = ["#d62728" if n == "z_georgescu" else "#aaaaaa" for n in params.index]
labels = [n.replace("z_", "").replace("_", " ") for n in params.index]
y_pos  = range(len(params))
ax2.barh(y_pos, params.values * 100, color=colors, alpha=0.8, height=0.6)
ax2.errorbar(params.values*100, y_pos,
             xerr=[(params - cis[0]).values*100, (cis[1] - params).values*100],
             fmt="none", color="black", lw=1.0, capsize=3)
ax2.axvline(0, lw=0.8, ls="--", c="black")
ax2.set_yticks(list(y_pos))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel("Coefficient × 100 (effect on Simion R1 turnout)", fontsize=9)
ax2.set_title("Model B coefficients (95% CI)\nRed = Georgescu share", fontsize=10)

plt.suptitle("Annulment Effect on Simion 2025 R1 Turnout", fontsize=12, y=1.01)
plt.tight_layout()
out = FIG_DIR / "09c_annulment_scatter.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved -> {out.name}")

# ---------------------------------------------------------------------------
# Save results table
# ---------------------------------------------------------------------------
out_df = df[["judet","cluster_short","georgescu_r1","turnout_2025r1",
             "resid_A","simion_r1"]].copy()
out_df["predicted_turnout_A"] = mod_A.fittedvalues.values
out_df = out_df.sort_values("resid_A", ascending=False)
out_df.to_csv(REP_DIR / "09c_annulment_effect.csv", index=False, float_format="%.4f")
print(f"  Saved -> reports/09c_annulment_effect.csv")

print(f"\n  Top 5 counties with highest positive turnout surprise:")
print(out_df[["judet","georgescu_r1","turnout_2025r1","resid_A"]].head(5).to_string(index=False))
print(f"\n  Top 5 counties with largest negative turnout surprise:")
print(out_df[["judet","georgescu_r1","turnout_2025r1","resid_A"]].tail(5).to_string(index=False))

print("\n" + "=" * 65)
print("TASK 9 COMPLETE — Annulment Effect")
print("=" * 65)
