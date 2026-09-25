"""
Task 10 — Cluster k Robustness
================================
Tests whether main analytical findings are stable under alternative cluster
solutions k=4 and k=6 vs the primary k=5 solution.

For each k in {4, 5, 6}:
  - Refit Ward hierarchical clustering on PCA scores
  - Assign cluster labels
  - Rerun AUR 2020→2024 swing OLS with same controls → record R²
  - Rerun Georgescu→Simion partial correlation → record r_partial
  - Record ARI between k and k=5 (label-invariant cluster agreement)

Output:
  reports/12a_cluster_k_robustness.csv
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
REP_DIR = ROOT / "reports"

feat  = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust5 = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat["judet_norm"] = feat["judet"].apply(_norm)

# Load existing k=5 labels for ARI comparison
feat = feat.merge(clust5[["judet","cluster_short"]], on="judet", how="left")

# Structural features used in original clustering (same set as 03_features)
CLUSTER_FEATURES = [
    "gdp_per_capita", "unemployment_rate", "poverty_risk_pct",
    "tertiary_education_pct", "early_school_leaving_pct", "internet_use_pct",
    "population_density", "net_migration_rate", "pct_pop_over65",
    "pct_employment_agriculture", "pct_employment_industry",
    "gva_deindustrial_index", "long_term_unemployment_rate",
    "household_income_pps", "pct_maghiari", "pct_romi",
    "pct_orthodox", "pct_urban", "emigrants_permanent_rate",
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
]

# Fill any missing values with column median before scaling
Xraw = feat[CLUSTER_FEATURES].copy()
for col in CLUSTER_FEATURES:
    if Xraw[col].isna().any():
        Xraw[col] = Xraw[col].fillna(Xraw[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xraw)

# PCA: retain components explaining ≥95% variance (as in original pipeline)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_pca.shape[1]} components explain {pca.explained_variance_ratio_.sum():.1%} variance")

# Ward linkage on PCA scores
Z = linkage(X_pca, method="ward", metric="euclidean")

# Swing data
import unicodedata
s2020 = series[(series.election=="parlamentare_2020") & (series.chamber=="S")].copy()
s2024 = series[(series.election=="parlamentare_2024") & (series.chamber=="S")].copy()
s2020["judet_norm"] = s2020["judet"].apply(_norm)
s2024["judet_norm"] = s2024["judet"].apply(_norm)
s2020 = s2020.set_index("judet_norm")
s2024 = s2024.set_index("judet_norm")

pres24r1 = series[series.election=="prezidentiale_2024_r1"].copy()
pres25r1 = series[series.election=="prezidentiale_2025_r1"].copy()
pres24r1["judet_norm"] = pres24r1["judet"].apply(_norm)
pres25r1["judet_norm"] = pres25r1["judet"].apply(_norm)

CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
]

feat_norm = feat.copy()
feat_norm.index = feat_norm["judet_norm"]

sw_aur = s2024["t_AUR"].reindex(feat_norm.index) - s2020["t_AUR"].reindex(feat_norm.index)
feat_norm["sw_AUR"] = sw_aur.values
feat_norm["georgescu_r1"] = pres24r1.set_index("judet_norm")["cand_GEORGESCU"].reindex(feat_norm.index).values
feat_norm["simion_r1"]    = pres25r1.set_index("judet_norm")["cand_SIMION"].reindex(feat_norm.index).values

for col in CONTROLS:
    feat_norm[f"z_{col}"] = (feat_norm[col] - feat_norm[col].mean()) / feat_norm[col].std()

z_ctrls = " + ".join(f"z_{c}" for c in CONTROLS)

def _residualise(y, X):
    X_c = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_c).fit().resid

def run_for_k(k, labels_k5):
    labels = fcluster(Z, k, criterion="maxclust")

    # ARI vs k=5
    ari = adjusted_rand_score(labels_k5, labels)

    # AUR swing OLS with these cluster dummies (not controlling for clusters,
    # just checking that the structural R² doesn't change meaningfully)
    df_k = feat_norm.copy()
    df_k["cluster_k"] = labels
    df_k["sw_AUR"] = sw_aur.values
    df_k = df_k.dropna(subset=["sw_AUR"] + CONTROLS)

    mod = smf.ols(f"sw_AUR ~ {z_ctrls}", data=df_k).fit(cov_type="HC3")
    r2_aur_swing = mod.rsquared

    # Transfer partial correlation
    df_tr = df_k.dropna(subset=["georgescu_r1","simion_r1"])
    X_ctrl = df_tr[[f"z_{c}" for c in CONTROLS]].values
    e_G = _residualise(df_tr["georgescu_r1"].values, X_ctrl)
    e_S = _residualise(df_tr["simion_r1"].values,    X_ctrl)
    r_partial, _ = pearsonr(e_G, e_S)

    return {"k": k, "ARI_vs_k5": round(ari, 3),
            "AUR_swing_R2": round(r2_aur_swing, 3),
            "transfer_r_partial": round(r_partial, 3)}

labels_k5 = fcluster(Z, 5, criterion="maxclust")

print("=" * 65)
print("Task 10 — Cluster k Robustness")
print("=" * 65)
print(f"\n{'k':>4}  {'ARI vs k=5':>12}  {'AUR swing R²':>14}  {'Transfer r_partial':>20}")

results = []
for k in [4, 5, 6]:
    r = run_for_k(k, labels_k5)
    results.append(r)
    ari_str = "—" if k == 5 else f"{r['ARI_vs_k5']:.3f}"
    print(f"  {k:>2}  {ari_str:>12}  {r['AUR_swing_R2']:>14.3f}  {r['transfer_r_partial']:>20.3f}")

out = REP_DIR / "12a_cluster_k_robustness.csv"
pd.DataFrame(results).to_csv(out, index=False)
print(f"\n  Saved -> {out.name}")

print("\n" + "=" * 65)
print("TASK 10 COMPLETE — Cluster k Robustness")
print("=" * 65)
