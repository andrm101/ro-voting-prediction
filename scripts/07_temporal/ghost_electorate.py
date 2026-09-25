"""
Task 7 — Ghost Electorate Analysis
====================================
Compares AEP enrolled voter rolls with Eurostat NUTS3 estimated 18+ population
(2021 census-based) to derive a Ghost Electorate Ratio (GER) per county:

    GER = (enrolled − pop_18plus) / enrolled

High GER indicates electoral roll inflation caused by emigration, mortality,
and administrative under-purging — a well-documented phenomenon in CEE states.

Methodology note:
  pop_18plus is estimated from Eurostat demo_r_pjanaggr3 (NUTS3, 2021):
      pop_18plus ≈ Y15_64 × (1 − α) + Y_GE65
  where α = (Y15 + Y16 + Y17) / Y15_64 at national level from demo_r_d2jan
  (α ≈ 0.0501, 2021). This introduces a small imprecision (<1 pp) relative
  to direct 18+ counts and is flagged as a limitation.

Outputs:
  figures/10a_ger_map.png           — county GER choropleth
  figures/10b_ger_cluster.png       — GER by cluster (boxplot)
  figures/10c_ger_correlates.png    — scatter of GER vs emigration + GDP
  reports/10a_ghost_electorate.csv  — county-level GER + covariates
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
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# 1. Estimate 18+ population at NUTS3 from Eurostat demo_r_pjanaggr3
# ---------------------------------------------------------------------------
def _clean_eurostat(s):
    """Strip Eurostat flags (b/e/p) and whitespace, return float."""
    if pd.isna(s):
        return np.nan
    return float(str(s).replace("b","").replace("e","").replace("p","")
                        .replace(" ","").strip())

pj = pd.read_csv(
    ROOT / "data" / "raw" / "eurostat" / "demo_r_pjanaggr3_full.tsv",
    sep="\t", na_values=[":", " "], dtype=str,
)
pj.columns = [c.strip() for c in pj.columns]
key_col = pj.columns[0]
pj[["freq","unit","sex","age","geo"]] = pj[key_col].str.split(",", expand=True)

year_col = [c for c in pj.columns if "2021" in c][0]

# National Y15-17 correction from demo_r_d2jan single-year data (pre-computed)
# Y15=213014, Y16=205867, Y17=205507 from RO national 2021
Y15_17_NATIONAL = 213_014 + 205_867 + 205_507

nat_15_64_row = pj[(pj.sex=="T") & (pj.age=="Y15-64") & (pj.geo=="RO")]
nat_15_64 = _clean_eurostat(nat_15_64_row[year_col].values[0])
ALPHA = Y15_17_NATIONAL / nat_15_64  # ≈ 0.0501

# NUTS3 Romanian counties (3-digit code after RO)
ro3 = pj[(pj.sex=="T") & (pj.geo.str.match(r"^RO\d{3}$"))].copy()
ages = ["TOTAL", "Y15-64", "Y_GE65", "Y_LT15"]
ro3_wide = (ro3[ro3.age.isin(ages)][["geo","age",year_col]]
            .pivot(index="geo", columns="age", values=year_col))

for col in ro3_wide.columns:
    ro3_wide[col] = ro3_wide[col].apply(_clean_eurostat)

ro3_wide["pop_18plus"] = ro3_wide["Y15-64"] * (1.0 - ALPHA) + ro3_wide["Y_GE65"]
ro3_wide = ro3_wide.reset_index().rename(columns={"geo": "geo_nuts3"})

print("=" * 65)
print("Task 7 — Ghost Electorate Analysis")
print("=" * 65)
print(f"\nNational correction α (15-17/15-64): {ALPHA:.4f}")
print(f"NUTS3 counties with 18+ estimate: {(~ro3_wide['pop_18plus'].isna()).sum()}")
print(f"National 18+ sum: {ro3_wide['pop_18plus'].sum():,.0f}")

# ---------------------------------------------------------------------------
# 2. Join NUTS3 → judet via features file
# ---------------------------------------------------------------------------
feat  = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
feat  = feat.merge(clust[["judet","cluster_short"]], on="judet", how="left")

# Normalise judet names for merging
import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)
feat = feat.merge(ro3_wide[["geo_nuts3","pop_18plus","TOTAL"]], on="geo_nuts3", how="left")
print(f"Counties with pop_18plus merged: {(~feat['pop_18plus'].isna()).sum()}/42")

# ---------------------------------------------------------------------------
# 3. Merge enrolled voters (latest election: parlamentare_2024 Senate)
# ---------------------------------------------------------------------------
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

def _norm_idx(df_col):
    return df_col.apply(_norm)

elections_to_analyse = [
    ("parlamentare_2020", "S", "2020 parl"),
    ("parlamentare_2024", "S", "2024 parl"),
    ("prezidentiale_2024_r1", None, "2024 pres R1"),
    ("prezidentiale_2025_r2", None, "2025 pres R2"),
]

enrolled_wide = {}
for elec, chamber, label in elections_to_analyse:
    sub = series[series.election == elec].copy()
    if chamber:
        sub = sub[sub.chamber == chamber]
    sub["judet_norm"] = _norm_idx(sub["judet"])
    enrolled_wide[label] = sub.set_index("judet_norm")["enrolled"]

enroll_df = pd.DataFrame(enrolled_wide)
enroll_df.index.name = "judet_norm"

feat = feat.merge(enroll_df.reset_index(), on="judet_norm", how="left")

# Compute GER for each election
for _, _, label in elections_to_analyse:
    col_enr = label
    feat[f"ger_{label}"] = (feat[col_enr] - feat["pop_18plus"]) / feat[col_enr]

ger_cols = [f"ger_{lbl}" for _, _, lbl in elections_to_analyse]
print("\n[GER summary by election]")
for col in ger_cols:
    vals = feat[col].dropna()
    print(f"  {col.replace('ger_',''):20s}: mean={vals.mean():.3f}  sd={vals.std():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}  n={len(vals)}")

# Use 2024 parliamentary Senate enrolled as primary GER measure
feat["ger"] = feat["ger_2024 parl"]

# ---------------------------------------------------------------------------
# 4. OLS: GER ~ structural predictors (HC3 SEs)
# ---------------------------------------------------------------------------
CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
    "emigrants_permanent_rate",
]
ger_clean = feat.dropna(subset=["ger"] + CONTROLS).copy()
for col in CONTROLS:
    ger_clean[f"z_{col}"] = ((ger_clean[col] - ger_clean[col].mean())
                              / ger_clean[col].std())

z_ctrls = " + ".join(f"z_{c}" for c in CONTROLS)
mod = smf.ols(f"ger ~ {z_ctrls}", data=ger_clean).fit(cov_type="HC3")

print(f"\n[OLS: GER ~ structural controls]")
print(f"  R²={mod.rsquared:.3f}  adj-R²={mod.rsquared_adj:.3f}  n={int(mod.nobs)}")
sig = mod.pvalues[mod.pvalues < 0.10].drop("Intercept", errors="ignore")
if len(sig):
    for term, p in sig.items():
        b = mod.params[term]
        print(f"    {term}: β={b:+.4f}  p={p:.3f}")
else:
    print("  (no predictors significant at p<0.10)")

# ---------------------------------------------------------------------------
# 5. Figure A: GER by cluster (boxplot)
# ---------------------------------------------------------------------------
CLUSTER_COLS = {
    "Moldova Profunda":   "#e41a1c",
    "Campia Uitata":      "#ff7f00",
    "Insula Capitalei":   "#4daf4a",
    "Centrele Dinamice":  "#377eb8",
    "Arcul Identitar":    "#984ea3",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Boxplot by cluster
ax = axes[0]
plot_df = feat.dropna(subset=["ger","cluster_short"]).copy()
order = sorted(plot_df["cluster_short"].unique())
palette = {k: CLUSTER_COLS.get(k, "#aaaaaa") for k in order}
sns.boxplot(data=plot_df, y="cluster_short", x="ger", order=order,
            palette=palette, ax=ax, width=0.5, linewidth=1.2)
ax.axvline(0, color="black", linestyle="--", lw=0.8)
ax.set_xlabel("Ghost Electorate Ratio (2024 Senate)", fontsize=10)
ax.set_ylabel("")
ax.set_title("GER distribution by cluster", fontsize=11)
ax.tick_params(axis="y", labelsize=8)

# GER over time (line per cluster)
ax2 = axes[1]
time_labels = ["2020 parl","2024 parl","2024 pres R1","2025 pres R2"]
cluster_time = (feat.groupby("cluster_short")[[f"ger_{lbl}" for lbl in time_labels]]
                .mean())
cluster_time.columns = time_labels
for cluster, row in cluster_time.iterrows():
    color = CLUSTER_COLS.get(cluster, "#aaaaaa")
    ax2.plot(range(len(time_labels)), row.values, marker="o", color=color,
             label=cluster, linewidth=1.8)
ax2.set_xticks(range(len(time_labels)))
ax2.set_xticklabels(time_labels, rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("Mean GER", fontsize=10)
ax2.set_title("Ghost Electorate Ratio over time by cluster", fontsize=11)
ax2.legend(fontsize=7, loc="upper right")
ax2.axhline(0, color="black", linestyle="--", lw=0.8)

plt.tight_layout()
out = FIG_DIR / "10a_ghost_electorate_cluster.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved -> {out.name}")

# ---------------------------------------------------------------------------
# 6. Figure B: GER vs emigration rate + GDP per capita
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

corr_pairs = [
    ("emigrants_permanent_rate", "Permanent emigration rate", "10b_left"),
    ("gdp_per_capita",           "GDP per capita (PPS)",      "10b_right"),
]

for ax, (xvar, xlabel, _) in zip(axes, corr_pairs):
    sub = feat.dropna(subset=["ger", xvar, "cluster_short"]).copy()
    for cluster in sub["cluster_short"].unique():
        m = sub[sub.cluster_short == cluster]
        color = CLUSTER_COLS.get(cluster, "#aaaaaa")
        ax.scatter(m[xvar], m["ger"], color=color, s=40, alpha=0.85,
                   label=cluster, zorder=3)
    # Regression line
    r, p = stats.pearsonr(sub[xvar], sub["ger"])
    m_ols, b_ols = np.polyfit(sub[xvar], sub["ger"], 1)
    xs = np.linspace(sub[xvar].min(), sub[xvar].max(), 100)
    ax.plot(xs, m_ols * xs + b_ols, "k--", lw=1.2)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("GER (2024 Senate)", fontsize=10)
    ax.set_title(f"r = {r:.3f}  (p = {p:.3f})", fontsize=10)

handles = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
axes[1].legend(handles=handles, fontsize=7, loc="upper right")

plt.suptitle("Ghost Electorate Ratio: correlates", fontsize=12, y=1.01)
plt.tight_layout()
out = FIG_DIR / "10b_ger_correlates.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved -> {out.name}")

# ---------------------------------------------------------------------------
# 7. Figure C: top/bottom counties ranked by GER
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
ranked = feat.dropna(subset=["ger"]).sort_values("ger", ascending=True).copy()
colors = [CLUSTER_COLS.get(c, "#aaaaaa") for c in ranked["cluster_short"]]
bars = ax.barh(ranked["judet"], ranked["ger"], color=colors, edgecolor="none")
ax.axvline(0, color="black", lw=1.0)
ax.axvline(feat["ger"].mean(), color="red", linestyle="--", lw=1.0,
           label=f"Mean GER = {feat['ger'].mean():.3f}")
ax.set_xlabel("Ghost Electorate Ratio (2024 Senate enrolled − 18+ pop) / enrolled",
              fontsize=9)
ax.set_title("Ghost Electorate Ratio by county (2024 parliamentary rolls vs 2021 pop est.)",
             fontsize=10)
ax.legend(fontsize=8)
handles = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
ax.legend(handles=handles, fontsize=7, loc="lower right")
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
out = FIG_DIR / "10c_ger_ranked.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved -> {out.name}")

# ---------------------------------------------------------------------------
# 8. Save report CSV
# ---------------------------------------------------------------------------
report_cols = (
    ["judet", "cluster_short", "pop_18plus", "TOTAL"]
    + [lbl for _, _, lbl in elections_to_analyse]
    + ger_cols
    + ["ger", "emigrants_permanent_rate", "net_migration_rate",
       "gdp_per_capita", "idx_grievance"]
)
report_cols = [c for c in report_cols if c in feat.columns]
report = feat[report_cols].copy()
report = report.sort_values("ger", ascending=False)
out_csv = REP_DIR / "10a_ghost_electorate.csv"
report.to_csv(out_csv, index=False, float_format="%.4f")
print(f"  Saved -> {out_csv.name}")

print("\n[Top 5 highest GER counties]")
print(report[["judet","cluster_short","ger","emigrants_permanent_rate"]].head(5).to_string(index=False))
print("\n[Bottom 5 lowest GER counties]")
print(report[["judet","cluster_short","ger","emigrants_permanent_rate"]].tail(5).to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Formal turnout deflation test
#    H0: GER has no effect on raw reported turnout after controlling for
#        structural characteristics.
#    Model: turnout_rate ~ ger + z_controls  (HC3 SEs, n=42)
#
#    Also computes GER-corrected effective turnout:
#        effective_turnout = voted / pop_18plus
#    and reports how much higher effective turnout is vs. roll-based turnout.
# ---------------------------------------------------------------------------
print("\n\n=== Turnout Deflation Test ===")

# turnout here is voted/enrolled from the electoral series
s2024_turnout = (series[(series.election=="parlamentare_2024") & (series.chamber=="S")]
                 .copy())
s2024_turnout["judet_norm"] = s2024_turnout["judet"].apply(_norm)
feat = feat.merge(
    s2024_turnout[["judet_norm","voted","enrolled","turnout"]]
    .rename(columns={"turnout":"turnout_2024","voted":"voted_2024","enrolled":"enrolled_2024"}),
    on="judet_norm", how="left"
)

feat["effective_turnout"] = feat["voted_2024"] / feat["pop_18plus"]
feat["turnout_deflation"] = feat["effective_turnout"] - feat["turnout_2024"]

td_clean = feat.dropna(subset=["turnout_2024","ger"] + CONTROLS).copy()
for col in CONTROLS:
    if f"z_{col}" not in td_clean.columns:
        td_clean[f"z_{col}"] = ((td_clean[col] - td_clean[col].mean())
                                 / td_clean[col].std())
td_clean["z_ger"] = (td_clean["ger"] - td_clean["ger"].mean()) / td_clean["ger"].std()

formula_td = "turnout_2024 ~ z_ger + " + " + ".join(f"z_{c}" for c in CONTROLS)
mod_td = smf.ols(formula_td, data=td_clean).fit(cov_type="HC3")

print(f"\n  OLS: turnout_2024 ~ GER + structural controls")
print(f"  R²={mod_td.rsquared:.3f}  adj-R²={mod_td.rsquared_adj:.3f}  n={int(mod_td.nobs)}")
b_ger = mod_td.params["z_ger"]
se_ger = mod_td.bse["z_ger"]
p_ger  = mod_td.pvalues["z_ger"]
print(f"\n  GER coefficient: β={b_ger:+.4f}  SE={se_ger:.4f}  p={p_ger:.4f}")
print(f"  Interpretation: 1 SD increase in GER (≈{td_clean['ger'].std():.3f}) "
      f"associates with {b_ger*100:+.2f} pp change in reported turnout")

# Magnitude of effective vs roll-based turnout
print(f"\n  Effective turnout (voted/pop_18plus) vs roll-based (voted/enrolled):")
print(f"  National mean roll-based:      {feat['turnout_2024'].mean():.3f}")
print(f"  National mean effective:       {feat['effective_turnout'].mean():.3f}")
print(f"  Mean deflation (effective−roll): +{feat['turnout_deflation'].mean():.3f}")
print(f"  Range of deflation: [{feat['turnout_deflation'].min():.3f}, "
      f"{feat['turnout_deflation'].max():.3f}]")

# Save extended report
feat["effective_turnout_2024"] = feat["effective_turnout"]
feat["turnout_deflation_2024"] = feat["turnout_deflation"]
report_extended = feat[["judet","cluster_short","ger","turnout_2024",
                          "effective_turnout_2024","turnout_deflation_2024",
                          "emigrants_permanent_rate"]].dropna(subset=["ger"])
report_extended = report_extended.sort_values("turnout_deflation_2024", ascending=False)
out_td = REP_DIR / "10b_turnout_deflation.csv"
report_extended.to_csv(out_td, index=False, float_format="%.4f")
print(f"\n  Saved -> {out_td.name}")

print("\n" + "=" * 65)
print("TASK 7 COMPLETE — Ghost Electorate Analysis")
print("=" * 65)
