"""
Conditional Projections (c) — Presidential Bloc Model
=======================================================
Builds a county-level predictive model for a hypothetical future Simion vs
reformist-candidate runoff, framed explicitly as conditional projections.

Three explicit conditions stated up front:
  C1. George Simion is the nationalist candidate
  C2. A single reformist candidate consolidates the anti-Simion pool (as Dan did)
  C3. Structural conditions remain at 2025 levels

The model is calibrated on the 2025 R2 outcome (Simion vs Dan).

Two-bloc transfer system (estimated by OLS, uncertainty via pairs bootstrap):
  Simion_R2 = α_S + β_S1 · Simion_R1 + β_S2 · X_structural
  Dan_R2    = α_D + β_D1 · Dan_consol · (1 − Simion_R1) + β_D2 · X_structural

Where Dan_consol = Dan_R2 / (1 − Simion_R1) is the consolidation ratio from
the electoral arc analysis (Section 5 / 7 of the paper).

Projection protocol:
  Step 1. Fit transfer model on 2025 actuals → obtain coefficient posterior
  Step 2. Project forward: substitute 2025 R1 shares as the starting point
           and hold structural controls fixed
  Step 3. Bootstrap (B=5000 pairs) → 90% projection intervals
  Step 4. Identify counties where 90% PI straddles 50% → genuine uncertainty

The output is NOT a forecast. It is: "Given the 2025 structural geography and
bloc transfer rates calibrated on 2025 actuals, this is what the same election
fought again under the same conditions would look like, county by county."

Outputs:
  figures/14c_presidential_bloc.png       — transfer scatter + projection map
  reports/14c_presidential_bloc.csv       — county-level projections with PIs
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

feat   = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust  = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
feat   = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)

for elec, src_col, new_col in [
    ("prezidentiale_2025_r1", "cand_SIMION", "simion_r1"),
    ("prezidentiale_2025_r1", "cand_DAN",    "dan_r1"),
    ("prezidentiale_2025_r2", "cand_SIMION", "simion_r2"),
    ("prezidentiale_2025_r2", "cand_DAN",    "dan_r2"),
]:
    sub = series[series.election == elec].copy()
    sub["judet_norm"] = sub["judet"].apply(_norm)
    feat = feat.merge(
        sub[["judet_norm", src_col]].rename(columns={src_col: new_col}),
        on="judet_norm", how="left",
        suffixes=("", "_drop"),
    )
    feat = feat.drop(columns=[c for c in feat.columns if c.endswith("_drop")])

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

df = feat.dropna(subset=["simion_r1", "dan_r1", "simion_r2", "dan_r2"] + CONTROLS).copy()
for col in CONTROLS:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

df["z_simion_r1"] = (df["simion_r1"] - df["simion_r1"].mean()) / df["simion_r1"].std()
df["dan_consol"]  = (df["dan_r2"] / np.clip(1 - df["simion_r1"], 1e-6, None)).clip(upper=1.5)
df["z_dan_consol"] = (df["dan_consol"] - df["dan_consol"].mean()) / df["dan_consol"].std()

n = len(df)
z_ctrls = " + ".join(f"z_{c}" for c in CONTROLS)

print("=" * 65)
print("Conditional Projections (c) — Presidential Bloc Model")
print("=" * 65)
print(f"\n  n={n} counties calibrated on 2025 R2")
print(f"\n  Explicit conditions:")
print(f"    C1. Simion is the nationalist candidate")
print(f"    C2. Single reformist candidate consolidates the anti-Simion pool")
print(f"    C3. Structural conditions held at 2025 values")

# ── Simion transfer model ────────────────────────────────────────────────────
mod_simion = smf.ols(f"simion_r2 ~ z_simion_r1 + {z_ctrls}", data=df).fit(cov_type="HC3")
print(f"\n  Simion transfer model:")
print(f"    R²={mod_simion.rsquared:.3f}  adj-R²={mod_simion.rsquared_adj:.3f}")
print(f"    β_simion_r1 = {mod_simion.params['z_simion_r1']:+.4f}  "
      f"p={mod_simion.pvalues['z_simion_r1']:.4f}")

# ── Dan consolidation model ──────────────────────────────────────────────────
mod_consol = smf.ols(f"dan_consol ~ {z_ctrls}", data=df).fit(cov_type="HC3")
print(f"\n  Dan consolidation model (ratio = Dan R2 / (1 − Simion R1)):")
print(f"    R²={mod_consol.rsquared:.3f}  adj-R²={mod_consol.rsquared_adj:.3f}")
sig_consol = [(CONTROLS[i], mod_consol.params[f"z_{CONTROLS[i]}"],
               mod_consol.pvalues[f"z_{CONTROLS[i]}"])
              for i in range(len(CONTROLS))
              if mod_consol.pvalues[f"z_{CONTROLS[i]}"] < 0.10]
if sig_consol:
    for nm, b, p in sorted(sig_consol, key=lambda x: x[2]):
        print(f"    β_{nm} = {b:+.4f}  p={p:.4f}")
else:
    print("    No predictor significant at p<0.10")

# ── Bootstrap projection ─────────────────────────────────────────────────────
B = 5000
X_simion = sm.add_constant(
    df[["z_simion_r1"] + [f"z_{c}" for c in CONTROLS]].values
)
X_consol = sm.add_constant(df[[f"z_{c}" for c in CONTROLS]].values)

y_simion_r2  = df["simion_r2"].values
y_dan_consol = df["dan_consol"].values
simion_r1    = df["simion_r1"].values

boot_simion_r2  = np.zeros((B, n))
boot_dan_consol = np.zeros((B, n))
rng = np.random.default_rng(42)

for b in range(B):
    idx = rng.integers(0, n, size=n)
    Xs_b = X_simion[idx]; ys_b = y_simion_r2[idx]
    Xd_b = X_consol[idx]; yd_b = y_dan_consol[idx]
    try:
        ps = np.linalg.lstsq(Xs_b, ys_b, rcond=None)[0]
        pd_ = np.linalg.lstsq(Xd_b, yd_b, rcond=None)[0]
        simion_proj    = X_simion @ ps
        consol_proj    = X_consol @ pd_
        # Dan projection: consol_proj × (1 − simion_r1)
        # Normalise to [0,1]
        dan_proj       = np.clip(consol_proj, 0, 1.5) * (1 - simion_r1)
        boot_simion_r2[b]  = np.clip(simion_proj, 0, 1)
        boot_dan_consol[b] = np.clip(dan_proj, 0, 1)
    except np.linalg.LinAlgError:
        boot_simion_r2[b]  = mod_simion.fittedvalues
        boot_dan_consol[b] = np.clip(mod_consol.fittedvalues, 0, 1.5) * (1 - simion_r1)

# Projected Simion R2 statistics
sim_proj_median = np.median(boot_simion_r2, axis=0)
sim_proj_q05    = np.percentile(boot_simion_r2, 5,  axis=0)
sim_proj_q95    = np.percentile(boot_simion_r2, 95, axis=0)
sim_proj_q25    = np.percentile(boot_simion_r2, 25, axis=0)
sim_proj_q75    = np.percentile(boot_simion_r2, 75, axis=0)

uncertain = (sim_proj_q05 < 0.50) & (sim_proj_q95 > 0.50)
n_uncertain = uncertain.sum()

print(f"\n  Bootstrap projection (B={B}, 90% PI):")
print(f"    Counties where 90% PI straddles 50%: {n_uncertain} / {n}")

# National projected share
natl_mean = sim_proj_median.mean()
natl_q05  = sim_proj_q05.mean()
natl_q95  = sim_proj_q95.mean()
print(f"    Mean projected Simion R2 share: {natl_mean*100:.1f}%  "
      f"90% PI [{natl_q05*100:.1f}%, {natl_q95*100:.1f}%]")
print(f"    (Observed 2025 R2: {df['simion_r2'].mean()*100:.1f}%)")

# Cluster-level summary
print(f"\n  Cluster-level projections (median Simion R2 share):")
df["sim_proj_median"] = sim_proj_median
df["sim_proj_q05"]    = sim_proj_q05
df["sim_proj_q25"]    = sim_proj_q25
df["sim_proj_q75"]    = sim_proj_q75
df["sim_proj_q95"]    = sim_proj_q95
df["uncertain"]       = uncertain
cluster_proj = df.groupby("cluster_short").agg(
    obs_r2    = ("simion_r2",     "mean"),
    proj_r2   = ("sim_proj_median","mean"),
    proj_q05  = ("sim_proj_q05",  "mean"),
    proj_q95  = ("sim_proj_q95",  "mean"),
    n_uncert  = ("uncertain",     "sum"),
    n         = ("judet",         "count"),
).round(4)
print(cluster_proj.to_string())

# Save
out_df = df[["judet", "cluster_short", "simion_r1", "dan_r1",
             "simion_r2", "dan_r2", "dan_consol",
             "sim_proj_median", "sim_proj_q05", "sim_proj_q25",
             "sim_proj_q75", "sim_proj_q95", "uncertain"]].copy()
out_df.to_csv(REP_DIR / "14c_presidential_bloc.csv", index=False, float_format="%.4f")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: transfer scatter — observed R1 vs projected R2 with PI
ax = axes[0]
order = np.argsort(simion_r1)
ax.fill_between(simion_r1[order] * 100,
                sim_proj_q05[order] * 100,
                sim_proj_q95[order] * 100,
                alpha=0.18, color="#d62728", label="90% projection interval")
ax.fill_between(simion_r1[order] * 100,
                sim_proj_q25[order] * 100,
                sim_proj_q75[order] * 100,
                alpha=0.35, color="#d62728", label="50% projection interval")
ax.plot(simion_r1[order] * 100, sim_proj_median[order] * 100,
        color="#a50026", lw=1.5, label="Projected median")
for cl, col in CLUSTER_COLS.items():
    m = df["cluster_short"] == cl
    ax.scatter(df.loc[m, "simion_r1"] * 100,
               df.loc[m, "simion_r2"] * 100,
               c=col, s=45, alpha=0.9, zorder=5,
               edgecolors="white", lw=0.4, label=cl)
ax.axhline(50, lw=0.9, ls="--", color="black", alpha=0.6)
ax.set_xlabel("Simion R1 2025 share (%)", fontsize=9)
ax.set_ylabel("Simion R2 2025 share (%) / projected", fontsize=9)
ax.set_title("Transfer model: R1 → R2\n"
             "Dots = observed 2025; bands = bootstrap 50%/90% PI",
             fontsize=9)
handles_ci = [
    plt.Rectangle((0, 0), 1, 1, fc="#d62728", alpha=0.35, label="50% PI"),
    plt.Rectangle((0, 0), 1, 1, fc="#d62728", alpha=0.18, label="90% PI"),
    plt.Line2D([0], [0], color="#a50026", lw=1.5, label="Projected median"),
]
handles_cl = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
ax.legend(handles=handles_ci + handles_cl, fontsize=6.5, ncol=2)

# Panel 2: county-level 90% PI, sorted by projected share
order2 = np.argsort(sim_proj_median)
x_pos  = np.arange(n)
ax2 = axes[1]
ax2.fill_between(x_pos,
                 sim_proj_q05[order2] * 100,
                 sim_proj_q95[order2] * 100,
                 alpha=0.2, color="#d62728")
ax2.fill_between(x_pos,
                 sim_proj_q25[order2] * 100,
                 sim_proj_q75[order2] * 100,
                 alpha=0.4, color="#d62728")
ax2.plot(x_pos, sim_proj_median[order2] * 100,
         color="#a50026", lw=1.2)
for i, idx in enumerate(order2):
    c = CLUSTER_COLS.get(df["cluster_short"].iloc[idx], "#888")
    ax2.scatter(i, df["simion_r2"].iloc[idx] * 100,
                color=c, s=22, zorder=5, edgecolors="white", lw=0.2)
    if uncertain[idx]:
        ax2.axvline(i, color="gold", lw=0.8, alpha=0.6, zorder=1)
ax2.axhline(50, lw=1.0, ls="--", color="black", alpha=0.7, label="50% tipping point")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df["judet"].iloc[order2].tolist(), rotation=90, fontsize=5.0)
ax2.set_ylabel("Projected Simion R2 share (%)", fontsize=9)
ax2.set_title(f"County-level 90% projection intervals\n"
              f"Gold verticals = genuine uncertainty (PI straddles 50%), n={n_uncertain}",
              fontsize=9)

plt.suptitle("Presidential Bloc Model: Conditional Projection\n"
             "(C1: Simion runs  C2: unified reformist  C3: structural status quo)",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "14c_presidential_bloc.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n  Saved -> figures/14c_presidential_bloc.png")
print(f"  Saved -> reports/14c_presidential_bloc.csv")
print("\n" + "=" * 65)
print("CONDITIONAL PROJECTIONS (c) COMPLETE")
print("=" * 65)
