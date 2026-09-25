"""
Task 1 — Two-Stage Clustering Pipeline: Romanian Județe Voter Profiles
======================================================================
Stage 1: Unsupervised clustering on structural/demographic features only.
         No electoral targets, no composite indices, no interaction terms.
Stage 2: Descriptive vote-distribution overlay per cluster (held-out features).

Outputs
-------
data/processed/cluster_assignments.csv   — siruta, judet, nuts2_code, cluster_id, cluster_name
figures/06a_dendrogram.png
figures/06a_gap_statistic.png
figures/06a_cluster_map.png              — choropleth by cluster
figures/06a_vote_profiles.png            — bar chart of party means per cluster
figures/06a_pca_biplot.png               — PC1 vs PC2 coloured by cluster

References
----------
Tibshirani R, Walther G, Hastie T (2001). Estimating the number of clusters in a
  data set via the gap statistic. JRSS-B 63(2):411-423.
Ward J (1963). Hierarchical grouping to optimize an objective function. JASA 58:236-244.
Hennig C (2007). Cluster-wise assessment of cluster stability. CStatA 47:1170-1189.
  (bootstrap ARI approach)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FEAT_PATH   = ROOT / "data" / "processed" / "features_2022_judet.parquet"
OUT_CSV     = ROOT / "data" / "processed" / "cluster_assignments.csv"
FIG_DIR     = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Feature selection — structural only, no electoral targets/composites
# ---------------------------------------------------------------------------
STRUCTURAL_FEATURES = [
    # Eurostat socio-economic
    "gdp_per_capita",
    "unemployment_rate",
    "poverty_risk_pct",
    "tertiary_education_pct",
    "internet_use_pct",
    "net_migration_rate",
    "population_density",
    # Employment structure (drop pct_employment_services — reference category)
    "pct_employment_agriculture",
    "pct_employment_industry",
    # Age structure
    "pct_pop_over65",
    "pct_pop_under25",
    # Census ethnicity (dominant groups; minority dummies act as geographic proxies)
    "pct_maghiari",
    "pct_romi",
    "pct_reformed",
    "pct_pentecostal",
    # Religion
    "pct_orthodox",
    "pct_greco_catholic",
    "pct_roman_catholic",
    # Settlement
    "pct_urban",
    "avg_household_size",
    # INS structural rates
    "emigrants_permanent_rate",
    "emigrants_temporary_rate",
    "hospitals_per_100k",
    "school_units_per_10k",
    "tourism_units_per_10k",
    "convictions_per_100k",
]

# Held-out electoral features for Stage 2 descriptive overlay
ELECTORAL_OVERLAY = [
    "t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR", "t_SOS", "t_POT",
    "t_turnout", "t_ANTI_ESTABLISHMENT",
    "georgescu_r1_2024", "simion_r1_2025",
]

PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b", "#f4a582", "#1a9850", "#fdae61"]


# ---------------------------------------------------------------------------
# Helper: VIF pruning
# ---------------------------------------------------------------------------
def vif_prune(
    X: pd.DataFrame, threshold: float = 10.0, max_iter: int = 20
) -> list[str]:
    """Iteratively drop the highest-VIF feature until all VIF < threshold."""
    cols = list(X.columns)
    for _ in range(max_iter):
        arr = X[cols].values.astype(float)
        vifs = [variance_inflation_factor(arr, i) for i in range(len(cols))]
        max_vif = max(vifs)
        if max_vif < threshold:
            break
        drop_idx = int(np.argmax(vifs))
        print(f"  VIF prune: drop '{cols[drop_idx]}' (VIF={max_vif:.1f})")
        cols.pop(drop_idx)
    final_arr = X[cols].values.astype(float)
    final_vifs = [variance_inflation_factor(final_arr, i) for i in range(len(cols))]
    print(f"  Remaining {len(cols)} features, max VIF={max(final_vifs):.1f}")
    return cols


# ---------------------------------------------------------------------------
# Gap statistic (Tibshirani 2001)
# ---------------------------------------------------------------------------
def _inertia_ward(Z_data: np.ndarray, k: int) -> float:
    labels = fcluster(
        linkage(Z_data, method="ward"), k, criterion="maxclust"
    )
    total = 0.0
    for c in np.unique(labels):
        members = Z_data[labels == c]
        if len(members) > 1:
            total += np.sum(pdist(members) ** 2) / (2 * len(members))
    return np.log(total + 1e-10)


def gap_statistic(
    Z_data: np.ndarray, k_max: int = 8, n_ref: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (gaps, sk, k_range) where k_range = 1..k_max.
    Optimal k: smallest k such that gap(k) >= gap(k+1) - sk(k+1).
    Reference distribution: uniform over bounding box of PCA-reduced data.
    """
    n, p = Z_data.shape
    k_range = np.arange(1, k_max + 1)
    gaps = np.zeros(k_max)
    sks  = np.zeros(k_max)

    for i, k in enumerate(k_range):
        Wk = _inertia_ward(Z_data, k)

        ref_Wks = []
        for _ in range(n_ref):
            ref = np.column_stack([
                np.random.uniform(Z_data[:, j].min(), Z_data[:, j].max(), n)
                for j in range(p)
            ])
            ref_Wks.append(_inertia_ward(ref, k))

        ref_Wks = np.array(ref_Wks)
        gaps[i] = np.mean(ref_Wks) - Wk
        sks[i]  = np.std(ref_Wks, ddof=1) * np.sqrt(1 + 1 / n_ref)

    return gaps, sks, k_range


def select_k_gap(gaps: np.ndarray, sks: np.ndarray) -> int:
    """Tibshirani criterion: smallest k where gap(k) >= gap(k+1) - sk(k+1)."""
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return int(i + 1)
    return int(np.argmax(gaps) + 1)


# ---------------------------------------------------------------------------
# Bootstrap ARI stability (Hennig 2007)
# ---------------------------------------------------------------------------
def bootstrap_ari(
    Z_data: np.ndarray, k: int, n_boot: int = 500
) -> tuple[float, float]:
    """Mean and std ARI between clustering of bootstrap samples and original."""
    n = Z_data.shape[0]
    original_labels = fcluster(linkage(Z_data, method="ward"), k, criterion="maxclust")
    aris = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        unique_idx = np.unique(idx)
        if len(unique_idx) < k:
            continue
        boot_labels = fcluster(
            linkage(Z_data[unique_idx], method="ward"), k, criterion="maxclust"
        )
        # Map back: for each original point in unique_idx, get boot label
        orig_sub = original_labels[unique_idx]
        aris.append(adjusted_rand_score(orig_sub, boot_labels))
    return float(np.mean(aris)), float(np.std(aris))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_gap(gaps, sks, k_range, k_opt, target_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(k_range, gaps, yerr=sks, fmt="o-", capsize=4,
                color="#2166ac", label="Gap statistic")
    ax.axvline(k_opt, color="#d6604d", ls="--", lw=1.5,
               label=f"Optimal k={k_opt}")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Gap(k)")
    ax.set_title("Gap Statistic for Ward Hierarchical Clustering\n(N=42 Romanian Judete)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {target_path.name}")


def plot_dendrogram(df_meta: pd.DataFrame, Z_link, k_opt: int, target_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    # Colour threshold at the cut level
    n = len(df_meta)
    cut_d = Z_link[-(k_opt - 1), 2] if k_opt > 1 else Z_link[-1, 2] + 1
    dendrogram(
        Z_link,
        labels=df_meta["judet"].values,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=cut_d,
    )
    ax.axhline(cut_d, color="#d6604d", ls="--", lw=1.5,
               label=f"Cut -> k={k_opt}")
    ax.set_title("Ward Hierarchical Clustering of Romanian Judete\n(Structural Features Only)")
    ax.set_ylabel("Ward Linkage Distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {target_path.name}")


def plot_pca_biplot(
    X_pca: np.ndarray, labels: np.ndarray, judet_names: np.ndarray,
    pca: PCA, feature_names: list[str], target_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    k = len(np.unique(labels))
    colors = PALETTE[:k]
    for c in range(1, k + 1):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=colors[c - 1], s=70, zorder=3, label=f"Cluster {c}")
        for name, x, y in zip(judet_names[mask], X_pca[mask, 0], X_pca[mask, 1]):
            ax.annotate(name, (x, y), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")

    # Top loading arrows (top 5 by PC1 loadings)
    loadings = pca.components_[:2].T
    top_idx = np.argsort(np.abs(loadings[:, 0]))[-5:]
    scale = 3.0
    for i in top_idx:
        ax.annotate(
            "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
        )
        ax.text(loadings[i, 0] * scale * 1.08, loadings[i, 1] * scale * 1.08,
                feature_names[i], fontsize=7, color="grey")

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title("PCA Biplot — Structural Features, Coloured by Cluster")
    ax.legend(loc="best", fontsize=8)
    ax.axhline(0, color="black", lw=0.4, ls="--")
    ax.axvline(0, color="black", lw=0.4, ls="--")
    fig.tight_layout()
    fig.savefig(target_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {target_path.name}")


def plot_vote_profiles(
    df: pd.DataFrame, labels: np.ndarray, cluster_names: dict[int, str],
    target_path: Path
) -> None:
    party_cols = [c for c in ELECTORAL_OVERLAY if c.startswith("t_") and c != "t_turnout"
                  and c != "t_ANTI_ESTABLISHMENT" and c != "t_ESTABLISHMENT"]
    df2 = df.copy()
    df2["cluster"] = labels

    rows = []
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c][party_cols]
        means = sub.mean()
        means["cluster"] = c
        means["label"] = cluster_names.get(c, f"Cluster {c}")
        rows.append(means)

    profile_df = pd.DataFrame(rows).set_index("label")
    profile_df = profile_df.drop(columns=["cluster"])
    clean_cols = [c.replace("t_", "") for c in profile_df.columns]
    profile_df.columns = clean_cols

    fig, ax = plt.subplots(figsize=(12, 5))
    profile_df.T.plot(kind="bar", ax=ax, width=0.75, colormap="tab10")
    ax.set_xlabel("Party (2024 Parliamentary)")
    ax.set_ylabel("Mean Vote Share")
    ax.set_title("Vote Share Profile by Cluster (Held-Out — Not Used in Clustering)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(target_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {target_path.name}")


def plot_feature_heatmap(
    X_scaled: np.ndarray, labels: np.ndarray, feature_names: list[str],
    cluster_names: dict[int, str], target_path: Path
) -> None:
    df_h = pd.DataFrame(X_scaled, columns=feature_names)
    df_h["cluster"] = labels
    df_h["label"] = df_h["cluster"].map(
        lambda c: cluster_names.get(c, f"Cluster {c}")
    )
    means = df_h.groupby("label")[feature_names].mean()

    fig, ax = plt.subplots(figsize=(16, len(cluster_names) * 1.2 + 2))
    sns.heatmap(means, ax=ax, cmap="RdBu_r", center=0, linewidths=0.3,
                annot=False, fmt=".2f", cbar_kws={"label": "Z-score (cluster mean)"})
    ax.set_title("Cluster Feature Profiles (Z-scored Structural Features)")
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(target_path, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {target_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Task 1: Structural Clustering Pipeline")
    print("=" * 60)

    # -- Load ------------------------------------------------------------------
    df = pd.read_parquet(FEAT_PATH)
    print(f"Loaded feature matrix: {df.shape[0]} judete x {df.shape[1]} columns")

    # Confirm all structural features present
    missing = [f for f in STRUCTURAL_FEATURES if f not in df.columns]
    if missing:
        print(f"[ERROR] Missing features: {missing}", file=sys.stderr)
        sys.exit(1)

    X_raw = df[STRUCTURAL_FEATURES].copy()
    n_missing = X_raw.isna().sum().sum()
    if n_missing:
        print(f"  [WARN] {n_missing} missing values in structural features — filling with column median")
        X_raw = X_raw.fillna(X_raw.median())

    print(f"Structural feature matrix: {X_raw.shape[0]} x {X_raw.shape[1]}")

    # -- VIF pruning -----------------------------------------------------------
    print("\n[1] VIF pruning (threshold=10) ...")
    retained_cols = vif_prune(X_raw, threshold=10.0)
    X_pruned = X_raw[retained_cols]
    print(f"  After pruning: {len(retained_cols)} features retained")
    print(f"  Retained: {retained_cols}")

    # -- Standardise -----------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pruned)
    X_scaled_df = pd.DataFrame(X_scaled, columns=retained_cols, index=df.index)

    # -- PCA (80% variance) ----------------------------------------------------
    print("\n[2] PCA ...")
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, 0.80) + 1)
    print(f"  Components for 80% variance: {n_components} "
          f"(cumulative: {cumvar[n_components-1]*100:.1f}%)")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA space: {X_pca.shape}")
    for i, ev in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {ev*100:.1f}%")

    # -- Ward linkage on PCA space ---------------------------------------------
    print("\n[3] Ward hierarchical clustering ...")
    Z_link = linkage(X_pca, method="ward")

    # -- Gap statistic (k=1..8) ------------------------------------------------
    print("\n[4] Gap statistic (n_ref=100, k_max=8) ...")
    gaps, sks, k_range = gap_statistic(X_pca, k_max=8, n_ref=100)
    k_opt_gap = select_k_gap(gaps, sks)
    print(f"  Gap statistic optimal k: {k_opt_gap}")
    for i, (g, s) in enumerate(zip(gaps, sks)):
        marker = " <-- optimal" if (i + 1) == k_opt_gap else ""
        print(f"    k={i+1}: gap={g:.4f}  sk={s:.4f}{marker}")

    # Inspect k=2..6 ARI stability before committing
    print("\n[5] Bootstrap ARI stability (n_boot=500) ...")
    ari_results = {}
    for k_test in range(2, 8):
        mean_ari, std_ari = bootstrap_ari(X_pca, k_test, n_boot=500)
        ari_results[k_test] = (mean_ari, std_ari)
        flag = "STRONG" if mean_ari >= 0.85 else ("OK" if mean_ari >= 0.75 else "WEAK")
        print(f"  k={k_test}: mean ARI={mean_ari:.3f} +/- {std_ari:.3f}  [{flag}]")

    # Select k: prefer gap_opt if ARI is strong; else fallback to highest ARI k
    k_final = k_opt_gap
    if ari_results.get(k_opt_gap, (0,))[0] < 0.75:
        # Find highest-stability k
        best_k = max(ari_results, key=lambda k: ari_results[k][0])
        print(f"  [NOTE] ARI at k={k_opt_gap} is weak; falling back to k={best_k} "
              f"(ARI={ari_results[best_k][0]:.3f})")
        k_final = best_k
    print(f"  Final k selected: {k_final}")

    # -- Fit final Ward partition -----------------------------------------------
    ward_labels = fcluster(Z_link, k_final, criterion="maxclust")

    # -- k-means robustness check ----------------------------------------------
    print("\n[6] k-means robustness check ...")
    km = KMeans(n_clusters=k_final, n_init=50, random_state=42)
    km_labels = km.fit_predict(X_pca) + 1  # 1-indexed
    km_ari = adjusted_rand_score(ward_labels, km_labels)
    print(f"  Ward vs k-means ARI: {km_ari:.3f}  "
          f"({'CONSISTENT' if km_ari >= 0.75 else 'DIVERGES -- investigate'})")

    # -- External validation: ARI vs NUTS2 -------------------------------------
    print("\n[7] External validation ...")
    nuts2_labels = pd.Categorical(df["nuts2_code"]).codes
    nuts2_ari = adjusted_rand_score(ward_labels, nuts2_labels)
    print(f"  ARI(Ward clusters vs NUTS2 regions): {nuts2_ari:.3f}")
    if nuts2_ari > 0.6:
        print("  [NOTE] Clusters strongly aligned with NUTS2 administrative regions.")
    elif nuts2_ari > 0.3:
        print("  [NOTE] Partial alignment with NUTS2 — clusters capture sub-regional structure.")
    else:
        print("  [NOTE] Clusters largely independent of NUTS2 boundaries — purely structural.")

    # -- Spatial autocorrelation of cluster labels (Moran's I) -----------------
    # Uses queen-contiguity weights from NUTS3 list (approximate positional order)
    # Full spatial test requires shapefile — skip if geopandas/libpysal not available
    try:
        import libpysal.weights as lw
        import esda
        w_path = ROOT / "data" / "raw" / "eurostat" / "NUTS_RO_weights.gal"
        if w_path.exists():
            w = lw.Queen.from_file(str(w_path))
            mi = esda.Moran(ward_labels, w)
            print(f"  Moran's I on cluster labels: I={mi.I:.3f}, p={mi.p_sim:.3f}")
        else:
            print("  [SKIP] Moran's I: weight matrix file not found (NUTS_RO_weights.gal)")
    except ImportError:
        print("  [SKIP] Moran's I: libpysal/esda not available")

    # -- Cluster descriptions (ANOVA on held-out features) ----------------------
    print("\n[8] Held-out electoral feature ANOVA (F-test, not used in clustering) ...")
    from scipy.stats import f_oneway
    for col in ELECTORAL_OVERLAY:
        if col not in df.columns:
            continue
        groups = [df.loc[ward_labels == c, col].dropna().values
                  for c in range(1, k_final + 1)]
        if any(len(g) < 2 for g in groups):
            continue
        F, p = f_oneway(*groups)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "  "))
        print(f"  {col:<30s}  F={F:6.2f}  p={p:.4f} {sig}")

    # -- Cluster means table ----------------------------------------------------
    print("\n[9] Cluster structural means (z-scored) ...")
    means_df = pd.DataFrame(X_scaled, columns=retained_cols)
    means_df["cluster"] = ward_labels
    cluster_means = means_df.groupby("cluster")[retained_cols].mean()
    print(cluster_means.round(2).to_string())

    # -- Assign provisional names ----------------------------------------------
    # Names derived from dominant structural characteristics
    # Will be refined after visual inspection of heatmap/dendrogram
    CLUSTER_NAMES: dict[int, str] = {}
    for c in range(1, k_final + 1):
        row = cluster_means.loc[c]
        # Heuristic naming based on dominant z-score signals
        signals = []
        if row.get("pct_urban", 0) > 0.6:
            signals.append("Urban")
        elif row.get("pct_urban", 0) < -0.6:
            signals.append("Rural")
        if row.get("gdp_per_capita", 0) > 0.8:
            signals.append("Prosperous")
        elif row.get("poverty_risk_pct", 0) > 0.8:
            signals.append("Deprived")
        if row.get("pct_maghiari", 0) > 1.0:
            signals.append("Magyar-minority")
        if row.get("pct_romi", 0) > 0.8:
            signals.append("High-Roma")
        if row.get("net_migration_rate", 0) < -0.8:
            signals.append("Emigration")
        if row.get("emigrants_temporary_rate", 0) > 0.8:
            signals.append("Labour-export")
        CLUSTER_NAMES[c] = f"C{c}: " + (", ".join(signals) if signals else "Mixed")

    print("\n[10] Provisional cluster names:")
    for c, name in CLUSTER_NAMES.items():
        n_counties = int((ward_labels == c).sum())
        counties = ", ".join(df.loc[ward_labels == c, "judet"].values)
        print(f"  {name}  (n={n_counties}): {counties}")

    # -- Save cluster assignments -----------------------------------------------
    out = df[["siruta", "judet", "nuts2_code", "geo_nuts3"]].copy()
    out["cluster_id"]   = ward_labels
    out["cluster_name"] = out["cluster_id"].map(CLUSTER_NAMES)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved cluster assignments -> {OUT_CSV.name}")

    # -- Figures ----------------------------------------------------------------
    print("\n[11] Generating figures ...")
    plot_gap(gaps, sks, k_range, k_final, FIG_DIR / "06a_gap_statistic.png")
    plot_dendrogram(df, Z_link, k_final, FIG_DIR / "06a_dendrogram.png")
    plot_pca_biplot(X_pca, ward_labels, df["judet"].values, pca,
                    retained_cols, FIG_DIR / "06a_pca_biplot.png")
    plot_feature_heatmap(X_scaled, ward_labels, retained_cols,
                         CLUSTER_NAMES, FIG_DIR / "06a_cluster_heatmap.png")
    plot_vote_profiles(df, ward_labels, CLUSTER_NAMES,
                       FIG_DIR / "06a_vote_profiles.png")

    # -- Summary ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE")
    print("=" * 60)
    print(f"  k selected    : {k_final} (gap statistic + bootstrap ARI)")
    print(f"  Ward vs KMeans ARI : {km_ari:.3f}")
    print(f"  ARI vs NUTS2  : {nuts2_ari:.3f}")
    print(f"  PCA components: {n_components} ({cumvar[n_components-1]*100:.1f}% variance)")
    print(f"  Features used : {len(retained_cols)} (after VIF pruning from {len(STRUCTURAL_FEATURES)})")
    print(f"\nOutputs:")
    print(f"  {OUT_CSV.name}")
    print(f"  figures/06a_*.png  (5 figures)")
    print(f"\nNext: review dendrogram + heatmap, then proceed to Task 3 (Georgescu->Simion transfer)")


if __name__ == "__main__":
    main()
