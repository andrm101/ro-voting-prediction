"""
Conditional Projections (b) — Swing County Ranking
====================================================
Identifies counties where a future Simion vs reformist-candidate presidential
runoff is structurally competitive and the outcome is most uncertain.

Swing is defined in the 2025 R2 frame (the most recent completed runoff):
  predicted_simion_r2_i  = OLS(Simion_R1, structural controls)
  swing_score_i          = |predicted_simion_r2_i − 0.50|  (distance from tipping point)

  lower swing_score  →  more competitive
  bootstrap_width_i  = 90th-percentile minus 10th-percentile of bootstrap
                       predictions  (uncertainty in the prediction)

A county is a "swing county" if it is both competitive (low swing_score)
AND uncertain (wide bootstrap interval). We rank by a composite:
  swing_rank_score = swing_score / bootstrap_width
  lower → more swing

Bootstrap: B=5000 resamplings of the OLS residuals (pairs bootstrap on counties).
The pairs bootstrap resamples (X_i, y_i) pairs jointly, preserving the
design distribution and providing valid uncertainty under heteroscedasticity.

Condition explicitly stated: predictions are conditional on
  (1) Simion as the nationalist candidate
  (2) a single reformist candidate facing him in the runoff
  (3) structural conditions unchanged from 2025

Outputs:
  figures/14b_swing_county_map.png     — choropleth + ranking bar chart
  reports/14b_swing_counties.csv       — full county table with scores
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
import statsmodels.api as sm

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"

feat   = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust  = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
feat   = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)

# Pull Simion R1 and R2 2025
for elec, src_col, new_col in [
    ("prezidentiale_2025_r1", "cand_SIMION", "simion_r1"),
    ("prezidentiale_2025_r2", "cand_SIMION", "simion_r2"),
]:
    sub = series[series.election == elec].copy()
    sub["judet_norm"] = sub["judet"].apply(_norm)
    feat = feat.merge(
        sub[["judet_norm", src_col]].rename(columns={src_col: new_col}),
        on="judet_norm", how="left",
        suffixes=("", "_drop"),
    )
    drop_cols = [c for c in feat.columns if c.endswith("_drop")]
    feat = feat.drop(columns=drop_cols)

# For Simion R2, the features parquet has simion_r1_2025 — use that for R1
# but we need to verify simion_r1 from the direct merge above is consistent
# Use simion_r2 from 2025 R2 and simion_r1 from 2025 R1 merge as our outcome
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

df = feat.dropna(subset=["simion_r1", "simion_r2"] + CONTROLS).copy()
for col in CONTROLS:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

# predictors for the R2 model: Simion R1 + structural controls
PRED_COLS = ["simion_r1"] + [f"z_{c}" for c in CONTROLS]
df["z_simion_r1"] = (df["simion_r1"] - df["simion_r1"].mean()) / df["simion_r1"].std()

X_pred_cols = ["z_simion_r1"] + [f"z_{c}" for c in CONTROLS]
X = sm.add_constant(df[X_pred_cols].values)
y = df["simion_r2"].values
n = len(df)

print("=" * 65)
print("Conditional Projections (b) — Swing County Ranking")
print("=" * 65)
print(f"\n  n={n}  predictors={len(X_pred_cols)}")

# OLS fit
ols = sm.OLS(y, X).fit(cov_type="HC3")
y_hat = ols.fittedvalues
print(f"\n  OLS R²={ols.rsquared:.3f}  adj-R²={ols.rsquared_adj:.3f}")
print(f"  β_simion_r1 = {ols.params[1]:+.4f}  p={ols.pvalues[1]:.4f}")

# Pairs bootstrap for prediction uncertainty (B=5000)
B = 5000
boot_preds = np.zeros((B, n))
rng = np.random.default_rng(42)
for b in range(B):
    idx = rng.integers(0, n, size=n)
    Xb, yb = X[idx], y[idx]
    try:
        params_b = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        boot_preds[b] = X @ params_b
    except np.linalg.LinAlgError:
        boot_preds[b] = y_hat

pred_q10 = np.percentile(boot_preds, 10, axis=0)
pred_q90 = np.percentile(boot_preds, 90, axis=0)
pred_q25 = np.percentile(boot_preds, 25, axis=0)
pred_q75 = np.percentile(boot_preds, 75, axis=0)
pred_med = np.median(boot_preds, axis=0)

# Swing metrics
swing_score     = np.abs(pred_med - 0.50)          # distance from tipping point
bootstrap_width = pred_q90 - pred_q10              # 80% interval width
composite       = swing_score / np.clip(bootstrap_width, 1e-6, None)

df["predicted_simion_r2"] = pred_med
df["pred_q10"] = pred_q10
df["pred_q25"] = pred_q25
df["pred_q75"] = pred_q75
df["pred_q90"] = pred_q90
df["swing_score"]      = swing_score
df["bootstrap_width"]  = bootstrap_width
df["composite_swing"]  = composite
df["swing_rank"]       = df["composite_swing"].rank(method="min").astype(int)

df_ranked = df.sort_values("composite_swing").reset_index(drop=True)

print(f"\n  Top 10 swing counties (most competitive & uncertain):")
print(f"  {'Rank':<5} {'County':<16} {'Cluster':<22} {'Pred R2':>8} {'Obs R2':>8} "
      f"{'Distance':>10} {'80% width':>10}")
print("  " + "-" * 80)
for _, row in df_ranked.head(10).iterrows():
    print(f"  {int(row['swing_rank']):<5} {row['judet']:<16} {row['cluster_short']:<22} "
          f"{row['predicted_simion_r2']*100:>7.1f}%  {row['simion_r2']*100:>7.1f}%  "
          f"{row['swing_score']*100:>9.1f}pp  {row['bootstrap_width']*100:>9.1f}pp")

# Save
out_cols = ["judet", "cluster_short", "simion_r1", "simion_r2",
            "predicted_simion_r2", "pred_q10", "pred_q25", "pred_q75", "pred_q90",
            "swing_score", "bootstrap_width", "composite_swing", "swing_rank"]
df_ranked[out_cols].to_csv(REP_DIR / "14b_swing_counties.csv", index=False, float_format="%.4f")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: predicted R2 with 80% bootstrap interval, ranked by composite swing
ax = axes[0]
top15 = df_ranked.head(15)
y_pos = np.arange(len(top15))
ax.barh(y_pos,
        (top15["pred_q90"] - top15["pred_q10"]) * 100,
        left=top15["pred_q10"] * 100,
        color=[CLUSTER_COLS.get(c, "#888") for c in top15["cluster_short"]],
        alpha=0.6, height=0.7, label="80% bootstrap interval")
ax.scatter(top15["predicted_simion_r2"] * 100, y_pos,
           color="black", s=40, zorder=5, label="Bootstrap median")
ax.scatter(top15["simion_r2"] * 100, y_pos,
           color="red", marker="D", s=35, zorder=6, label="Observed 2025 R2")
ax.axvline(50, lw=1.2, ls="--", color="black", alpha=0.7, label="50% tipping point")
ax.set_yticks(y_pos)
ax.set_yticklabels(top15["judet"].tolist(), fontsize=8)
ax.set_xlabel("Simion predicted R2 share (%)", fontsize=9)
ax.set_title("Top 15 swing counties by composite score\n"
             "(competitive + uncertain, cond. on Simion candidacy)",
             fontsize=9)
handles_cl = [mpatches.Patch(color=v, label=k, alpha=0.7)
              for k, v in CLUSTER_COLS.items()]
ax.legend(fontsize=6.5, loc="lower right")

# Panel 2: scatter of swing_score vs bootstrap_width
ax2 = axes[1]
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax2.scatter(df.loc[m, "bootstrap_width"] * 100,
                df.loc[m, "swing_score"] * 100,
                c=col, s=55, alpha=0.85, label=cl, edgecolors="white", lw=0.4)
# Annotate top 10 swing counties
for _, row in df_ranked.head(10).iterrows():
    ax2.annotate(row["judet"],
                 (row["bootstrap_width"] * 100, row["swing_score"] * 100),
                 fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax2.set_xlabel("Bootstrap 80% interval width (pp)  →  more uncertain", fontsize=9)
ax2.set_ylabel("Distance from 50% tipping point (pp)  →  less competitive", fontsize=9)
ax2.set_title("Swing space: competitiveness vs prediction uncertainty\n"
              "Lower-left = highest swing potential", fontsize=9)
handles_cl2 = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
ax2.legend(handles=handles_cl2, fontsize=6.5, loc="upper right")

plt.suptitle("Conditional Projections: Swing County Ranking\n"
             "(conditional on Simion candidacy, structural status quo)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "14b_swing_county_ranking.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n  Saved -> figures/14b_swing_county_ranking.png")
print(f"  Saved -> reports/14b_swing_counties.csv")
print("\n" + "=" * 65)
print("CONDITIONAL PROJECTIONS (b) COMPLETE")
print("=" * 65)
