"""
Follow-up A — Grievance vs Deindustrialisation Collinearity
============================================================
The grievance index has no significant net effect in the Dirichlet regression
despite being the canonical structural predictor of populism. The most likely
explanation is collinearity with the deindustrialisation index: both load on
the same peripheral-county profile.

This script:
  1. Reports bivariate Pearson r between idx_grievance and gva_deindustrial_index
  2. Produces a scatter plot of the two indices coloured by cluster
  3. Reports VIF of grievance in the M1 Dirichlet predictor matrix
  4. Fits M1 with grievance alone vs deindustrialisation alone (OLS proxy on
     AUR share) to show how much each explains independently

Outputs:
  figures/13a_grievance_deindustrial_scatter.png
  reports/13a_grievance_collinearity.csv
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"

feat  = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
feat  = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")

import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")
parl24 = series[(series.election == "parlamentare_2024") & (series.chamber == "S")].copy()
parl24["judet_norm"] = parl24["judet"].apply(_norm)
feat = feat.merge(parl24[["judet_norm", "t_AUR"]].rename(columns={"t_AUR": "aur_2024"}),
                  on="judet_norm", how="left")

CLUSTER_COLS = {
    "Moldova Profunda":   "#e41a1c",
    "Campia Uitata":      "#ff7f00",
    "Insula Capitalei":   "#4daf4a",
    "Centrele Dinamice":  "#377eb8",
    "Arcul Identitar":    "#984ea3",
}

M1_PREDICTORS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_maghiari", "gva_deindustrial_index",
]

df = feat.dropna(subset=M1_PREDICTORS + ["aur_2024"]).copy()
for col in M1_PREDICTORS:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

print("=" * 65)
print("Follow-up A — Grievance vs Deindustrialisation Collinearity")
print("=" * 65)

# 1. Bivariate correlation
r, p = pearsonr(df["idx_grievance"], df["gva_deindustrial_index"])
print(f"\n  Pearson r(grievance, deindustrialisation) = {r:.4f}  p = {p:.4f}")

# 2. VIF in M1 predictor matrix
X_m1 = sm.add_constant(df[[f"z_{c}" for c in M1_PREDICTORS]])
vif_records = []
for i, col in enumerate(X_m1.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_m1.values, i)
    vif_records.append({"predictor": col.replace("z_", ""), "VIF": round(vif, 3)})

vif_df = pd.DataFrame(vif_records).sort_values("VIF", ascending=False)
print(f"\n  VIF in M1 predictor matrix:")
print(vif_df.to_string(index=False))

# 3. AUR share OLS: grievance alone vs deindustrialisation alone vs both
mod_griev  = smf.ols("aur_2024 ~ z_idx_grievance", data=df).fit()
mod_deindu = smf.ols("aur_2024 ~ z_gva_deindustrial_index", data=df).fit()
mod_both   = smf.ols("aur_2024 ~ z_idx_grievance + z_gva_deindustrial_index", data=df).fit()

print(f"\n  OLS on AUR 2024 share — incremental R²:")
print(f"    Grievance alone:           R²={mod_griev.rsquared:.3f}")
print(f"    Deindustrialisation alone: R²={mod_deindu.rsquared:.3f}")
print(f"    Both together:             R²={mod_both.rsquared:.3f}  "
      f"(increment = {mod_both.rsquared - max(mod_griev.rsquared, mod_deindu.rsquared):+.3f})")
print(f"\n    β_grievance  in joint model: {mod_both.params['z_idx_grievance']:+.4f}  "
      f"p={mod_both.pvalues['z_idx_grievance']:.4f}")
print(f"    β_deindustrial in joint model: {mod_both.params['z_gva_deindustrial_index']:+.4f}  "
      f"p={mod_both.pvalues['z_gva_deindustrial_index']:.4f}")

# Save CSV
result_rows = [
    {"statistic": "Pearson r (grievance vs deindustrialisation)", "value": round(r, 4)},
    {"statistic": "p-value",                                       "value": round(p, 4)},
    {"statistic": "R² grievance alone (AUR OLS)",                 "value": round(mod_griev.rsquared, 4)},
    {"statistic": "R² deindustrialisation alone (AUR OLS)",       "value": round(mod_deindu.rsquared, 4)},
    {"statistic": "R² both (AUR OLS)",                            "value": round(mod_both.rsquared, 4)},
]
pd.DataFrame(result_rows).to_csv(REP_DIR / "13a_grievance_collinearity.csv", index=False)
vif_df.to_csv(REP_DIR / "13a_grievance_vif.csv", index=False)

# 4. Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax.scatter(df.loc[m, "gva_deindustrial_index"],
               df.loc[m, "idx_grievance"],
               c=col, s=55, alpha=0.85, label=cl, edgecolors="white", lw=0.4)
# OLS line
x = df["gva_deindustrial_index"].values
y = df["idx_grievance"].values
slope, intercept = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, slope * xs + intercept, "k--", lw=1.2, alpha=0.8)
for _, row in df.iterrows():
    if abs(row["idx_grievance"]) > 0.6 or row["gva_deindustrial_index"] > 0.12:
        ax.annotate(row["judet"], (row["gva_deindustrial_index"], row["idx_grievance"]),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("Deindustrialisation index", fontsize=10)
ax.set_ylabel("Grievance index", fontsize=10)
ax.set_title(f"r = {r:.3f}  p = {p:.3f}", fontsize=10)
handles = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
ax.legend(handles=handles, fontsize=6.5, loc="upper left")

# Residual plot: grievance net of deindustrialisation
ax2 = axes[1]
X_c = sm.add_constant(df["z_gva_deindustrial_index"].values)
griev_resid = df["z_idx_grievance"].values - sm.OLS(df["z_idx_grievance"].values, X_c).fit().fittedvalues
aur_resid   = df["aur_2024"].values - sm.OLS(df["aur_2024"].values, X_c).fit().fittedvalues
r2, p2 = pearsonr(griev_resid, aur_resid)
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax2.scatter(griev_resid[m.values], aur_resid[m.values],
                c=col, s=55, alpha=0.85, edgecolors="white", lw=0.4)
slope2, intercept2 = np.polyfit(griev_resid, aur_resid, 1)
xs2 = np.linspace(griev_resid.min(), griev_resid.max(), 100)
ax2.plot(xs2, slope2 * xs2 + intercept2, "k--", lw=1.2, alpha=0.8)
ax2.axhline(0, lw=0.7, ls=":", c="grey")
ax2.axvline(0, lw=0.7, ls=":", c="grey")
ax2.set_xlabel("Grievance residual (net of deindustrialisation)", fontsize=9)
ax2.set_ylabel("AUR share residual (net of deindustrialisation)", fontsize=9)
ax2.set_title(f"Grievance → AUR partial corr: r = {r2:.3f}  p = {p2:.3f}", fontsize=10)

plt.suptitle("Grievance vs Deindustrialisation Collinearity", fontsize=12, y=1.01)
plt.tight_layout()
out = FIG_DIR / "13a_grievance_deindustrial_scatter.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved -> {out.name}")
print(f"  Saved -> reports/13a_grievance_collinearity.csv")
print(f"  Saved -> reports/13a_grievance_vif.csv")

print("\n" + "=" * 65)
print("FOLLOW-UP A COMPLETE")
print("=" * 65)
