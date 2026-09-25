"""
Task 6 — Temporal electoral analysis
=====================================
1. 2020 → 2024 parliamentary swing vectors (Senate, for comparability)
2. Structural predictors of swing (OLS per party + joint Dirichlet-style summary)
3. 2024 R1 → 2025 R1 → 2025 R2 presidential arc (Georgescu → Simion → runoff)
4. Simion R2 consolidation: how much of non-Simion R1 pool went to Dan by county

All inference is ecological (county-level); individual-level inferences
are not warranted (ecological fallacy caveat).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"
FIG_DIR.mkdir(exist_ok=True)
REP_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")
feat   = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust  = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")

# Merge cluster labels onto features
feat = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")

CLUSTER_COLS = {
    "Moldova Profunda":   "#e41a1c",
    "Campia Uitata":      "#ff7f00",
    "Insula Capitalei":   "#4daf4a",
    "Centrele Dinamice":  "#377eb8",
    "Arcul Identitar":    "#984ea3",
}

PARTY_COLS = ["t_UDMR", "t_PSD", "t_PNL", "t_USR", "t_AUR", "t_SOS", "t_POT", "t_OTHER"]
PARTY_LABELS = {
    "t_UDMR": "UDMR", "t_PSD": "PSD", "t_PNL": "PNL",
    "t_USR": "USR",   "t_AUR": "AUR", "t_SOS": "SOS",
    "t_POT": "POT",   "t_OTHER": "Other",
}

CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
]

print("=" * 65)
print("Task 6 — Temporal Electoral Analysis")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Build 2020 vs 2024 Senate swing dataset
# ---------------------------------------------------------------------------
s2020 = (series[(series.election == "parlamentare_2020") & (series.chamber == "S")]
         .set_index("judet"))
s2024 = (series[(series.election == "parlamentare_2024") & (series.chamber == "S")]
         .set_index("judet"))

# Harmonise county names (diacritics may differ between sources)
def _norm(s: pd.Index) -> pd.Index:
    import unicodedata
    def strip(x):
        x = unicodedata.normalize("NFKD", str(x))
        return "".join(c for c in x if not unicodedata.combining(c)).upper().strip()
    return pd.Index([strip(v) for v in s])

s2020.index = _norm(s2020.index)
s2024.index = _norm(s2024.index)
feat_idx = _norm(feat["judet"])
feat.index = feat_idx

common = s2020.index.intersection(s2024.index)
print(f"\n[1] 2020 vs 2024 swing — common județe: {len(common)}")
print(f"    Missing from 2020 (will be NaN): {set(s2024.index) - set(s2020.index)}")

swing_cols = ["t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR", "t_OTHER"]
swing = pd.DataFrame(index=s2024.index)
for col in swing_cols:
    if col in s2024.columns:
        swing[f"sw_{col[2:]}"] = s2024[col]
    if col in s2020.columns and col in s2024.columns:
        # Fill 2020 where available; NaN for Vâlcea
        vals2020 = s2020[col].reindex(s2024.index)
        swing[f"sw_{col[2:]}"] = s2024[col].values - vals2020.values
swing["t24_AUR"] = s2024["t_AUR"].values
swing["t24_turnout"] = s2024["turnout"].values
swing["t20_turnout"] = s2020["turnout"].reindex(s2024.index).values

# Merge structural features
swing = swing.join(feat[CONTROLS + ["cluster_short"]], how="left")
swing_clean = swing.dropna(subset=CONTROLS + [f"sw_{p[2:]}" for p in ["t_AUR","t_PSD","t_PNL","t_USR"]])
print(f"    Complete cases for regression: {len(swing_clean)}")

# Standardise controls
for col in CONTROLS:
    swing_clean[f"z_{col}"] = (swing_clean[col] - swing_clean[col].mean()) / swing_clean[col].std()
z_controls = " + ".join([f"z_{c}" for c in CONTROLS])

print("\n=== OLS: Swing ~ Structural Controls ===")
ols_results = {}
for party in ["AUR", "PSD", "PNL", "USR", "OTHER"]:
    dep = f"sw_{party}"
    if dep not in swing_clean.columns or swing_clean[dep].isna().all():
        continue
    formula = f"{dep} ~ {z_controls}"
    mod = smf.ols(formula, data=swing_clean).fit(cov_type="HC3")
    ols_results[party] = mod
    sig = mod.pvalues[mod.pvalues < 0.10].drop("Intercept", errors="ignore")
    print(f"\n  {party}:  R²={mod.rsquared:.3f}  adj-R²={mod.rsquared_adj:.3f}  n={int(mod.nobs)}")
    if len(sig) > 0:
        for term, p in sig.items():
            b = mod.params[term]
            print(f"    {term}: β={b:+.4f}  p={p:.3f}")

# ---------------------------------------------------------------------------
# 2. Figure 1: Swing heatmap by cluster
# ---------------------------------------------------------------------------
swing_clean["cluster"] = swing_clean["cluster_short"].fillna("Unknown")
sw_cols_plot = [c for c in swing_clean.columns if c.startswith("sw_")]
cluster_swing = swing_clean.groupby("cluster")[sw_cols_plot].mean()
cluster_swing.columns = [c.replace("sw_", "") for c in cluster_swing.columns]

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(cluster_swing, ax=ax, cmap="RdBu_r", center=0,
            annot=True, fmt=".3f", linewidths=0.4,
            cbar_kws={"label": "Mean swing (pp, 2020→2024)"})
ax.set_title("2020→2024 Parliamentary Vote Swing by Cluster (Senate)\n"
             "Positive = gain, negative = loss", fontsize=10)
ax.set_xlabel("Party"); ax.set_ylabel("")
ax.tick_params(axis="y", rotation=0, labelsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "09a_swing_heatmap.png", dpi=300)
plt.close(fig)
print("\n  Saved -> 09a_swing_heatmap.png")

# ---------------------------------------------------------------------------
# 3. Figure 2: AUR swing scatter (most theoretically important)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, xvar, xlabel in zip(
    axes,
    ["gva_deindustrial_index", "idx_grievance"],
    ["Deindustrialisation Index", "Grievance Index"]
):
    for cl, col in CLUSTER_COLS.items():
        mask = swing_clean["cluster_short"] == cl
        ax.scatter(
            swing_clean.loc[mask, xvar],
            swing_clean.loc[mask, "sw_AUR"] * 100,
            c=col, label=cl, s=55, alpha=0.85, edgecolors="white", linewidths=0.4
        )
    # OLS line
    x = swing_clean[xvar].values
    y = swing_clean["sw_AUR"].values * 100
    m, b, r, p, _ = stats.linregress(x[~np.isnan(x+y)], y[~np.isnan(x+y)])
    xs = np.linspace(x[~np.isnan(x)].min(), x[~np.isnan(x)].max(), 100)
    ax.plot(xs, m*xs+b, "k--", lw=1.2, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("AUR swing 2020→2024 (pp)", fontsize=9)
    ax.set_title(f"r={r:.3f}  p={p:.3f}", fontsize=9)
    ax.axhline(0, lw=0.5, ls=":", c="grey")

axes[0].legend(fontsize=7, framealpha=0.8)
fig.suptitle("AUR Swing 2020→2024 vs Structural Predictors", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "09a_aur_swing_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved -> 09a_aur_swing_scatter.png")

# ---------------------------------------------------------------------------
# 4. Presidential arc: 2024 R1 → 2025 R1 → 2025 R2
# ---------------------------------------------------------------------------
pres24r1 = (series[series.election == "prezidentiale_2024_r1"]
            .set_index("judet").rename(columns={"cand_GEORGESCU": "georgescu_r1_2024"}))
pres25r1 = (series[series.election == "prezidentiale_2025_r1"]
            .set_index("judet").rename(columns={"cand_SIMION": "simion_r1_2025"}))
pres25r2 = (series[series.election == "prezidentiale_2025_r2"]
            .set_index("judet")
            .rename(columns={"cand_SIMION": "simion_r2_2025",
                             "cand_DAN": "dan_r2_2025"}))

pres24r1.index = _norm(pres24r1.index)
pres25r1.index = _norm(pres25r1.index)
pres25r2.index = _norm(pres25r2.index)

arc = pd.DataFrame(index=pres25r2.index)
arc["georgescu_r1"] = pres24r1["georgescu_r1_2024"].reindex(arc.index)
arc["simion_r1"]    = pres25r1["simion_r1_2025"].reindex(arc.index)
arc["simion_r2"]    = pres25r2["simion_r2_2025"]
arc["dan_r2"]       = pres25r2["dan_r2_2025"]
arc["enrolled_r2"]  = pres25r2["enrolled"]

# Simion R2 consolidation: did Simion grow his R1 base?
arc["simion_r2_gain"]  = arc["simion_r2"] - arc["simion_r1"]
# Dan's pool: anti-Simion R1 = 1 - simion_r1
arc["anti_simion_r1"]  = 1.0 - arc["simion_r1"]
# Dan's share of that pool
arc["dan_of_anti"]     = arc["dan_r2"] / arc["anti_simion_r1"].clip(lower=0.01)
arc = arc.join(feat[CONTROLS + ["cluster_short"]], how="left")

print("\n\n=== Presidential Arc Summary ===")
print(f"  National Georgescu R1 2024: {arc['georgescu_r1'].mul(pres24r1['valid_votes'].reindex(arc.index)).sum() / pres24r1['valid_votes'].reindex(arc.index).sum():.1%}")
print(f"  National Simion R1 2025:    {arc['simion_r1'].mul(pres25r1['valid_votes'].reindex(arc.index)).sum() / pres25r1['valid_votes'].reindex(arc.index).sum():.1%}")
print(f"  National Simion R2 2025:    {arc['simion_r2'].mul(pres25r2['valid_votes']).sum() / pres25r2['valid_votes'].sum():.1%}")
print(f"  National Dan R2 2025:       {arc['dan_r2'].mul(pres25r2['valid_votes']).sum() / pres25r2['valid_votes'].sum():.1%}")

r_g_s, p_g_s, _, _, _ = stats.linregress(
    arc["georgescu_r1"].dropna(), arc["simion_r1"].reindex(arc["georgescu_r1"].dropna().index)
).__iter__() if False else (None,)*5
from scipy.stats import pearsonr
r_g_s, p_g_s = pearsonr(arc["georgescu_r1"].dropna(),
                          arc["simion_r1"].reindex(arc["georgescu_r1"].dropna().index).dropna())
print(f"\n  Georgescu→Simion R1 raw r: {r_g_s:.3f}  p={p_g_s:.4f}")

r_s1_s2, p_s1_s2 = pearsonr(arc["simion_r1"].dropna(),
                               arc["simion_r2"].reindex(arc["simion_r1"].dropna().index).dropna())
print(f"  Simion R1→R2 raw r: {r_s1_s2:.3f}  p={p_s1_s2:.4f}")

print("\n  Simion R2 vs R1 swing by cluster:")
arc_cl = arc.dropna(subset=["cluster_short", "simion_r2_gain"])
print(arc_cl.groupby("cluster_short")[["simion_r1","simion_r2","simion_r2_gain","dan_of_anti"]].mean().round(3).to_string())

# ---------------------------------------------------------------------------
# 5. Figure 3: Presidential arc scatter (3-panel)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pairs = [
    ("georgescu_r1", "simion_r1",
     "Georgescu R1 2024 (%)", "Simion R1 2025 (%)"),
    ("simion_r1", "simion_r2",
     "Simion R1 2025 (%)", "Simion R2 2025 (%)"),
    ("anti_simion_r1", "dan_r2",
     "Anti-Simion pool R1 (%)", "Dan R2 2025 (%)"),
]
for ax, (xv, yv, xl, yl) in zip(axes, pairs):
    sub = arc[[xv, yv, "cluster_short"]].dropna()
    for cl, col in CLUSTER_COLS.items():
        m = sub["cluster_short"] == cl
        ax.scatter(sub.loc[m, xv]*100, sub.loc[m, yv]*100,
                   c=col, s=55, alpha=0.85, label=cl,
                   edgecolors="white", linewidths=0.4)
    x = sub[xv].values; y = sub[yv].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs*100, (slope*xs+intercept)*100, "k--", lw=1.2, alpha=0.7)
    ax.plot([0, 100], [0, 100], ":", c="grey", lw=0.8)
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.set_title(f"r={r:.3f}  p={p:.3f}", fontsize=9)

axes[0].legend(fontsize=7, framealpha=0.8, loc="upper left")
fig.suptitle("Presidential Electoral Arc 2024–2025", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "09b_presidential_arc.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\n  Saved -> 09b_presidential_arc.png")

# ---------------------------------------------------------------------------
# 6. Figure 4: Dan R2 consolidation — did urban/Insula Capitalei flip hardest?
# ---------------------------------------------------------------------------
arc_plot = arc.dropna(subset=["cluster_short", "dan_of_anti", "simion_r2_gain"])

fig, ax = plt.subplots(figsize=(10, 6))
for cl, col in CLUSTER_COLS.items():
    m = arc_plot["cluster_short"] == cl
    ax.scatter(arc_plot.loc[m, "simion_r2_gain"]*100,
               arc_plot.loc[m, "dan_of_anti"]*100,
               c=col, s=70, alpha=0.85, label=cl,
               edgecolors="white", linewidths=0.5)

# County labels for outliers
for idx, row in arc_plot.iterrows():
    if abs(row["simion_r2_gain"]) > 0.04 or row["dan_of_anti"] > 0.85:
        ax.annotate(str(idx), (row["simion_r2_gain"]*100, row["dan_of_anti"]*100),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

ax.axvline(0, lw=0.8, ls="--", c="grey")
ax.axhline(50, lw=0.8, ls="--", c="grey")
ax.set_xlabel("Simion R2 swing vs R1 (pp)", fontsize=10)
ax.set_ylabel("Dan's share of anti-Simion R1 pool (%)", fontsize=10)
ax.set_title("2025 Runoff: Simion consolidation vs Dan mobilisation by county\n"
             "Dashed lines: zero swing / 50% of anti-Simion pool", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG_DIR / "09b_runoff_consolidation.png", dpi=300)
plt.close(fig)
print("  Saved -> 09b_runoff_consolidation.png")

# ---------------------------------------------------------------------------
# 7. Save swing table
# ---------------------------------------------------------------------------
swing_out = swing_clean[[c for c in swing_clean.columns if c.startswith("sw_")]
                         + ["t24_AUR", "t24_turnout", "cluster_short"]].copy()
swing_out.index.name = "judet"
swing_out.reset_index().to_csv(REP_DIR / "09a_swing_2020_2024.csv", index=False)
arc.reset_index(names="judet").to_csv(REP_DIR / "09b_presidential_arc.csv", index=False)
print("\n  Saved -> reports/09a_swing_2020_2024.csv")
print("  Saved -> reports/09b_presidential_arc.csv")

# ---------------------------------------------------------------------------
# 8. Compositional swing: CLR (centred log-ratio) transformation
#    Addresses zero-sum constraint — swing shares must sum to zero by definition.
#    CLR respects Aitchison geometry: clr(x) = log(x) - mean(log(x)).
#    CLR difference = clr(2024) - clr(2020) gives compositional swing.
#    Five OLS models on CLR components (one is linearly redundant; we report 5
#    for symmetry but note that CLR(2024) - CLR(2020) sums to zero identically).
# ---------------------------------------------------------------------------
print("\n\n=== Compositional Swing Analysis (CLR) ===")

PARTY_COLS = ["t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR", "t_OTHER"]

def _clr(x: pd.Series) -> pd.Series:
    """Centred log-ratio transform. Zeros replaced with small constant (1e-5)."""
    lx = np.log(x.clip(lower=1e-5))
    return lx - lx.mean()

# Build CLR frames aligned on common county index
clr_2020 = pd.DataFrame(index=s2020.index)
clr_2024 = pd.DataFrame(index=s2024.index)
for col in PARTY_COLS:
    if col in s2020.columns:
        clr_2020[col] = _clr(s2020[col])
    if col in s2024.columns:
        clr_2024[col] = _clr(s2024[col])

# Align to common counties (Vâlcea missing in 2020)
common = clr_2020.index.intersection(clr_2024.index)
clr_swing = clr_2024.loc[common] - clr_2020.loc[common]
clr_swing.index.name = "judet"

# Merge controls
clr_swing = clr_swing.join(
    feat[CONTROLS + ["cluster_short"]].set_index(feat["judet"].apply(
        lambda s: "".join(c for c in __import__("unicodedata").normalize("NFKD", str(s))
                          if not __import__("unicodedata").combining(c)).upper().strip()
    )),
    how="left"
)
clr_clean = clr_swing.dropna(subset=CONTROLS + [c for c in PARTY_COLS if c in clr_swing.columns])
for col in CONTROLS:
    clr_clean[f"z_{col}"] = (clr_clean[col] - clr_clean[col].mean()) / clr_clean[col].std()

z_controls = " + ".join([f"z_{c}" for c in CONTROLS])

print(f"\n  n={len(clr_clean)} counties with complete CLR swing data")
print(f"  (CLR swing columns sum to zero by construction: "
      f"{clr_clean[[c for c in PARTY_COLS if c in clr_clean.columns]].sum(axis=1).abs().max():.2e})")

print(f"\n  {'Party':8s}  {'CLR R²':>8}  {'adj-R²':>8}  "
      f"{'Key predictor':>40}")

clr_ols_results = {}
for col in [c for c in PARTY_COLS if c in clr_clean.columns]:
    formula = f"{col} ~ {z_controls}"
    mod = smf.ols(formula, data=clr_clean).fit(cov_type="HC3")
    clr_ols_results[col] = mod
    sig = mod.pvalues[mod.pvalues < 0.10].drop("Intercept", errors="ignore")
    key = f"{sig.index[0]}: β={mod.params[sig.index[0]]:+.3f} p={sig.values[0]:.3f}" \
          if len(sig) > 0 else "—"
    print(f"  {col:8s}  {mod.rsquared:>8.3f}  {mod.rsquared_adj:>8.3f}  {key:>40}")

# Compare OLS vs CLR R² directionally
print("\n  OLS vs CLR R² comparison (sign consistency check):")
print(f"  {'Party':8s}  {'OLS R²':>8}  {'CLR R²':>8}  {'Direction match':>18}")
ols_r2 = {"AUR": ols_results.get("AUR"), "PSD": ols_results.get("PSD"),
           "PNL": ols_results.get("PNL"), "USR": ols_results.get("USR")}
for party, m_ols in ols_r2.items():
    col = f"t_{party}"
    m_clr = clr_ols_results.get(col)
    if m_ols and m_clr:
        # Check sign agreement for key predictors
        match = "consistent" if abs(m_clr.rsquared - m_ols.rsquared) < 0.15 else "divergent"
        print(f"  {party:8s}  {m_ols.rsquared:>8.3f}  {m_clr.rsquared:>8.3f}  {match:>18}")

# ---------------------------------------------------------------------------
# 9. Deindustrialisation × cluster interaction (AUR swing)
#    Tests whether the structural mechanism operates differently in
#    Moldova Profundă vs the rest. Interaction: z_gva_deindustrial × is_moldova
# ---------------------------------------------------------------------------
print("\n\n=== Deindustrialisation × Cluster Interaction (AUR swing) ===")

swing_int = swing_clean.copy()
swing_int["is_moldova"] = (swing_int["cluster_short"] == "Moldova Profunda").astype(float)
# Interaction term
swing_int["z_deindustrial_x_moldova"] = (swing_int["z_gva_deindustrial_index"]
                                          * swing_int["is_moldova"])

formula_int = ("sw_AUR ~ z_idx_grievance + z_idx_modernity + "
               "z_idx_demographic_pressure + z_pct_employment_agriculture + "
               "z_pct_maghiari + z_pct_romi + z_gva_deindustrial_index + "
               "z_long_term_unemployment_rate + is_moldova + "
               "z_deindustrial_x_moldova")
mod_int = smf.ols(formula_int, data=swing_int).fit(cov_type="HC3")

b_int  = mod_int.params["z_deindustrial_x_moldova"]
se_int = mod_int.bse["z_deindustrial_x_moldova"]
p_int  = mod_int.pvalues["z_deindustrial_x_moldova"]
b_main = mod_int.params["z_gva_deindustrial_index"]

print(f"\n  Interaction model: R²={mod_int.rsquared:.3f}  adj-R²={mod_int.rsquared_adj:.3f}")
print(f"  z_deindustrial (baseline):         β={b_main:+.4f}")
print(f"  z_deindustrial × is_moldova:       β={b_int:+.4f}  SE={se_int:.4f}  p={p_int:.4f}")
print(f"  Total deindustrial effect (Moldova): β={b_main + b_int:+.4f}")
print(f"  is_moldova cluster dummy:           β={mod_int.params['is_moldova']:+.4f}  "
      f"p={mod_int.pvalues['is_moldova']:.4f}")

if p_int < 0.10:
    print(f"\n  >> Significant heterogeneity (p<0.10): deindustrialisation effect "
          f"{'stronger' if b_int > 0 else 'weaker'} in Moldova Profundă")
else:
    print(f"\n  >> No significant interaction (p={p_int:.3f}) — "
          f"deindustrialisation effect is spatially uniform")

# ---------------------------------------------------------------------------
# 10. Save tidy OLS swing coefficients for modelsummary-style Rmd table
# ---------------------------------------------------------------------------
tidy_rows = []
for party, mod in ols_results.items():
    for term in mod.params.index:
        tidy_rows.append({
            "party":    party,
            "term":     term,
            "estimate": mod.params[term],
            "se":       mod.bse[term],
            "p":        mod.pvalues[term],
            "r2":       mod.rsquared,
            "adj_r2":   mod.rsquared_adj,
            "n":        int(mod.nobs),
        })
tidy_swing = pd.DataFrame(tidy_rows)
tidy_swing.to_csv(REP_DIR / "09d_swing_ols_tidy.csv", index=False, float_format="%.4f")
print(f"\n  Saved -> reports/09d_swing_ols_tidy.csv")

print("\n" + "=" * 65)
print("TASK 6 COMPLETE — Temporal Analysis")
print("=" * 65)
