"""
Approach comparison: VIF-pruned features vs full PCA (all 26 structural features).
Runs both pipelines, generates side-by-side diagnostics, saves a comparison report.

Outputs
-------
figures/06b_gap_fullpca.png
figures/06b_dendrogram_fullpca.png
figures/06b_pca_biplot_fullpca.png
figures/06b_cluster_heatmap_fullpca.png
figures/06b_vote_profiles_fullpca.png
figures/06b_approach_comparison.png   -- head-to-head ARI / ANOVA F comparison
data/processed/cluster_assignments_fullpca.csv
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
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FEAT_PATH = ROOT / "data" / "processed" / "features_2022_judet.parquet"
VIF_CSV   = ROOT / "data" / "processed" / "cluster_assignments.csv"
OUT_CSV   = ROOT / "data" / "processed" / "cluster_assignments_fullpca.csv"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

np.random.seed(42)

STRUCTURAL_FEATURES = [
    # Eurostat core socio-economic
    "gdp_per_capita", "unemployment_rate", "poverty_risk_pct",
    "tertiary_education_pct", "early_school_leaving_pct", "internet_use_pct",
    "net_migration_rate", "population_density",
    # Age structure
    "pct_pop_over65", "pct_pop_under25",
    # Employment — full triad (no regression collinearity concern for clustering)
    "pct_employment_agriculture", "pct_employment_industry", "pct_employment_services",
    # Deindustrialisation thesis (new)
    "industry_gva_share",          # current industrial weight of local economy
    "gva_deindustrial_index",      # lost industrial base since 2010 (positive = deindustrialised)
    "long_term_unemployment_rate", # structural joblessness — ex-industrial towns
    "household_income_pps",        # welfare beyond GDP — captures transfer dependence
    # Census ethnicity — including NW and Bukovina anchors
    "pct_maghiari", "pct_romi", "pct_reformed", "pct_pentecostal",
    "pct_orthodox", "pct_greco_catholic", "pct_roman_catholic",
    "pct_germani",    # Transylvanian Saxon/Swabian counties
    "pct_ucraineni",  # Northern Bukovina (Suceava, Maramures) — NE differentiator
    # Settlement and household
    "pct_urban", "avg_household_size",
    # INS structural rates
    "emigrants_permanent_rate", "emigrants_temporary_rate",
    "hospitals_per_100k", "school_units_per_10k",
    "tourism_units_per_10k", "convictions_per_100k",
]

ELECTORAL_OVERLAY = [
    "t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR", "t_SOS", "t_POT",
    "t_turnout", "t_ANTI_ESTABLISHMENT",
    "georgescu_r1_2024", "simion_r1_2025",
]

PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b", "#f4a582", "#1a9850", "#fdae61"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _inertia_ward(Z_data: np.ndarray, k: int) -> float:
    labels = fcluster(linkage(Z_data, method="ward"), k, criterion="maxclust")
    total = 0.0
    for c in np.unique(labels):
        members = Z_data[labels == c]
        if len(members) > 1:
            total += np.sum(pdist(members) ** 2) / (2 * len(members))
    return np.log(total + 1e-10)


def gap_statistic(Z_data, k_max=8, n_ref=100):
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


def select_k_gap(gaps, sks):
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return int(i + 1)
    return int(np.argmax(gaps) + 1)


def bootstrap_ari(Z_data, k, n_boot=500):
    n = Z_data.shape[0]
    orig = fcluster(linkage(Z_data, method="ward"), k, criterion="maxclust")
    aris = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        uniq = np.unique(idx)
        if len(uniq) < k:
            continue
        boot = fcluster(linkage(Z_data[uniq], method="ward"), k, criterion="maxclust")
        aris.append(adjusted_rand_score(orig[uniq], boot))
    return float(np.mean(aris)), float(np.std(aris))


def vif_prune(X: pd.DataFrame, threshold=10.0):
    cols = list(X.columns)
    for _ in range(30):
        arr = X[cols].values.astype(float)
        vifs = [variance_inflation_factor(arr, i) for i in range(len(cols))]
        if max(vifs) < threshold:
            break
        cols.pop(int(np.argmax(vifs)))
    return cols


def run_pipeline(label: str, X_pca: np.ndarray, df: pd.DataFrame, n_features_in: int):
    """Full gap + ARI + Ward + KMeans pipeline in PCA space. Returns (labels, k_final, ari_results)."""
    print(f"\n--- {label} ---")
    Z_link = linkage(X_pca, method="ward")

    print("  Gap statistic ...")
    gaps, sks, k_range = gap_statistic(X_pca, k_max=8, n_ref=100)
    k_opt_gap = select_k_gap(gaps, sks)
    print(f"  Gap optimal k: {k_opt_gap}")

    print("  Bootstrap ARI ...")
    ari_results = {}
    for k_test in range(2, 8):
        mean_ari, std_ari = bootstrap_ari(X_pca, k_test, n_boot=500)
        ari_results[k_test] = (mean_ari, std_ari)
        flag = "STRONG" if mean_ari >= 0.85 else ("OK" if mean_ari >= 0.75 else "WEAK")
        print(f"    k={k_test}: ARI={mean_ari:.3f} +/- {std_ari:.3f}  [{flag}]")

    k_final = k_opt_gap
    if ari_results.get(k_opt_gap, (0,))[0] < 0.75:
        best_k = max(ari_results, key=lambda k: ari_results[k][0])
        print(f"  ARI weak at k={k_opt_gap}; falling back to k={best_k}")
        k_final = best_k
    print(f"  -> Final k={k_final}")

    ward_labels = fcluster(Z_link, k_final, criterion="maxclust")

    km = KMeans(n_clusters=k_final, n_init=50, random_state=42)
    km_labels = km.fit_predict(X_pca) + 1
    km_ari = adjusted_rand_score(ward_labels, km_labels)
    print(f"  Ward vs k-means ARI: {km_ari:.3f}")

    nuts2_ari = adjusted_rand_score(ward_labels, pd.Categorical(df["nuts2_code"]).codes)
    print(f"  ARI vs NUTS2: {nuts2_ari:.3f}")

    return ward_labels, k_final, ari_results, gaps, sks, k_range, Z_link, km_ari, nuts2_ari


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_gap(gaps, sks, k_range, k_opt, path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(k_range, gaps, yerr=sks, fmt="o-", capsize=4, color="#2166ac")
    ax.axvline(k_opt, color="#d6604d", ls="--", lw=1.5, label=f"k={k_opt}")
    ax.set_xlabel("k"); ax.set_ylabel("Gap(k)")
    ax.set_title(f"Gap Statistic — {title_suffix}")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved -> {path.name}")


def plot_dendrogram(df, Z_link, k_opt, path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(14, 5))
    cut_d = Z_link[-(k_opt - 1), 2] if k_opt > 1 else Z_link[-1, 2] + 1
    dendrogram(Z_link, labels=df["judet"].values, ax=ax,
               leaf_rotation=90, leaf_font_size=8, color_threshold=cut_d)
    ax.axhline(cut_d, color="#d6604d", ls="--", lw=1.5, label=f"k={k_opt}")
    ax.set_title(f"Ward Dendrogram — {title_suffix}")
    ax.set_ylabel("Ward Distance"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved -> {path.name}")


def plot_pca_biplot(X_pca, labels, judet_names, pca, feature_names, path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(9, 7))
    k = len(np.unique(labels))
    for c in range(1, k + 1):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=PALETTE[(c - 1) % len(PALETTE)], s=70, zorder=3, label=f"C{c}")
        for name, x, y in zip(judet_names[mask], X_pca[mask, 0], X_pca[mask, 1]):
            ax.annotate(name, (x, y), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
    loadings = pca.components_[:2].T
    top_idx = np.argsort(np.abs(loadings[:, 0]))[-6:]
    for i in top_idx:
        ax.annotate("", xy=(loadings[i, 0] * 3, loadings[i, 1] * 3), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
        ax.text(loadings[i, 0] * 3.1, loadings[i, 1] * 3.1, feature_names[i],
                fontsize=7, color="grey")
    v1 = pca.explained_variance_ratio_[0] * 100
    v2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({v1:.1f}%)"); ax.set_ylabel(f"PC2 ({v2:.1f}%)")
    ax.set_title(f"PCA Biplot — {title_suffix}")
    ax.legend(fontsize=8); ax.axhline(0, lw=0.4, ls="--", color="k")
    ax.axvline(0, lw=0.4, ls="--", color="k")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved -> {path.name}")


def plot_heatmap(X_scaled, labels, feature_names, cluster_names, path, title_suffix=""):
    df_h = pd.DataFrame(X_scaled, columns=feature_names)
    df_h["label"] = [cluster_names.get(c, f"C{c}") for c in labels]
    means = df_h.groupby("label")[feature_names].mean()
    fig, ax = plt.subplots(figsize=(max(14, len(feature_names) * 0.6), len(cluster_names) * 1.2 + 2))
    sns.heatmap(means, ax=ax, cmap="RdBu_r", center=0, linewidths=0.3,
                cbar_kws={"label": "Z-score"})
    ax.set_title(f"Cluster Feature Profiles — {title_suffix}")
    ax.tick_params(axis="x", rotation=45, labelsize=7); ax.tick_params(axis="y", rotation=0)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved -> {path.name}")


def plot_vote_profiles(df, labels, cluster_names, path, title_suffix=""):
    party_cols = [c for c in ELECTORAL_OVERLAY
                  if c.startswith("t_") and c not in ("t_turnout", "t_ANTI_ESTABLISHMENT", "t_ESTABLISHMENT")]
    df2 = df.copy(); df2["cluster"] = labels
    rows = []
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c][party_cols].mean()
        sub["label"] = cluster_names.get(c, f"C{c}")
        rows.append(sub)
    pf = pd.DataFrame(rows).set_index("label")
    pf.columns = [c.replace("t_", "") for c in pf.columns]
    fig, ax = plt.subplots(figsize=(12, 5))
    pf.T.plot(kind="bar", ax=ax, width=0.75, colormap="tab10")
    ax.set_xlabel("Party"); ax.set_ylabel("Mean Vote Share")
    ax.set_title(f"Vote Profiles by Cluster — {title_suffix}")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  Saved -> {path.name}")


def make_cluster_names(cluster_means, retained_cols):
    names = {}
    for c in cluster_means.index:
        row = cluster_means.loc[c]
        signals = []
        if "pct_urban" in row.index and row["pct_urban"] > 0.8:
            signals.append("Urban")
        elif "pct_urban" in row.index and row["pct_urban"] < -0.6:
            signals.append("Rural")
        if "gdp_per_capita" in row.index and row["gdp_per_capita"] > 0.8:
            signals.append("Prosperous")
        if "poverty_risk_pct" in row.index and row["poverty_risk_pct"] > 0.8:
            signals.append("Deprived")
        if "pct_maghiari" in row.index and row["pct_maghiari"] > 1.0:
            signals.append("Magyar")
        if "pct_romi" in row.index and row["pct_romi"] > 0.8:
            signals.append("High-Roma")
        if "net_migration_rate" in row.index and row["net_migration_rate"] < -0.8:
            signals.append("Emigration")
        if "pct_employment_agriculture" in row.index and row["pct_employment_agriculture"] > 0.8:
            signals.append("Agricultural")
        if "pct_greco_catholic" in row.index and row["pct_greco_catholic"] > 0.8:
            signals.append("Greco-Cath")
        if "pct_pentecostal" in row.index and row["pct_pentecostal"] > 0.8:
            signals.append("Pentecostal")
        names[c] = f"C{c}: " + (", ".join(signals) if signals else "Mixed")
    return names


# ---------------------------------------------------------------------------
# Head-to-head comparison figure
# ---------------------------------------------------------------------------
def plot_comparison(
    df, vif_labels, pca_labels, k_vif, k_pca,
    ari_vif, ari_pca, ward_km_vif, ward_km_pca,
    nuts2_ari_vif, nuts2_ari_pca, path
):
    parties = [c for c in ELECTORAL_OVERLAY
               if c not in ("t_ANTI_ESTABLISHMENT", "t_ESTABLISHMENT")]

    vif_f, pca_f = [], []
    for col in parties:
        if col not in df.columns:
            vif_f.append(np.nan); pca_f.append(np.nan); continue
        def _f(labels):
            groups = [df.loc[labels == c, col].dropna().values
                      for c in range(1, max(labels) + 1) if (labels == c).sum() >= 2]
            if len(groups) < 2:
                return np.nan
            return f_oneway(*groups)[0]
        vif_f.append(_f(vif_labels))
        pca_f.append(_f(pca_labels))

    # Build ARI matrix between the two partitions
    cross_ari = adjusted_rand_score(vif_labels, pca_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ANOVA F-stats
    x = np.arange(len(parties))
    w = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - w/2, vif_f, w, label=f"VIF-pruned (k={k_vif})", color="#2166ac", alpha=0.8)
    bars2 = ax.bar(x + w/2, pca_f, w, label=f"Full-PCA (k={k_pca})", color="#d6604d", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("t_", "").replace("_r1_2024", "\nGeorg.").replace("_r1_2025", "\nSimion")
                        for c in parties], fontsize=8)
    ax.set_ylabel("ANOVA F-statistic")
    ax.set_title("Electoral Separation by Cluster\n(held-out, higher = better-separated)")
    ax.legend(fontsize=8)

    # Right: summary scorecard
    ax2 = axes[1]
    ax2.axis("off")
    metrics = [
        ("k selected",           k_vif,               k_pca),
        ("Ward-KMeans ARI",      f"{ward_km_vif:.3f}", f"{ward_km_pca:.3f}"),
        ("ARI vs NUTS2",         f"{nuts2_ari_vif:.3f}", f"{nuts2_ari_pca:.3f}"),
        ("Best bootstrap ARI",   f"{max(v[0] for v in ari_vif.values()):.3f}",
                                 f"{max(v[0] for v in ari_pca.values()):.3f}"),
        ("Cross-approach ARI",   f"{cross_ari:.3f}", f"{cross_ari:.3f}"),
        ("Features in PCA",      "10 (VIF-pruned)", "26 (all structural)"),
    ]
    col_labels = ["Metric", "VIF-pruned", "Full PCA"]
    tbl = ax2.table(
        cellText=metrics,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)
    ax2.set_title("Head-to-Head Scorecard", pad=12)

    fig.suptitle("VIF-Pruned vs Full-PCA Clustering: Approach Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Approach Comparison: VIF-Pruned vs Full-PCA")
    print("=" * 60)

    df = pd.read_parquet(FEAT_PATH)
    missing = [f for f in STRUCTURAL_FEATURES if f not in df.columns]
    if missing:
        print(f"[ERROR] Missing: {missing}", file=sys.stderr); sys.exit(1)

    X_raw = df[STRUCTURAL_FEATURES].fillna(df[STRUCTURAL_FEATURES].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ------------------------------------------------------------------
    # Approach A: VIF-pruned (reproduce previous results for comparison)
    # ------------------------------------------------------------------
    print("\n[A] VIF-PRUNED APPROACH (reproducing Task 1)")
    retained_vif = vif_prune(pd.DataFrame(X_scaled, columns=STRUCTURAL_FEATURES))
    print(f"  Retained {len(retained_vif)} features: {retained_vif}")
    X_vif = X_scaled[:, [STRUCTURAL_FEATURES.index(c) for c in retained_vif]]

    pca_vif = PCA(random_state=42)
    cumvar_vif = np.cumsum(pca_vif.fit(X_vif).explained_variance_ratio_)
    nc_vif = int(np.searchsorted(cumvar_vif, 0.80) + 1)
    pca_vif = PCA(n_components=nc_vif, random_state=42)
    Xp_vif = pca_vif.fit_transform(X_vif)
    print(f"  PCA: {nc_vif} components ({cumvar_vif[nc_vif-1]*100:.1f}% variance)")

    (vif_labels, k_vif, ari_vif, gaps_vif, sks_vif, kr_vif,
     Zl_vif, wkm_vif, nuts2_vif) = run_pipeline("VIF-pruned", Xp_vif, df, len(retained_vif))

    cm_vif = pd.DataFrame(X_vif, columns=retained_vif)
    cm_vif["cluster"] = vif_labels
    names_vif = make_cluster_names(cm_vif.groupby("cluster")[retained_vif].mean(), retained_vif)

    # ------------------------------------------------------------------
    # Approach B: Full PCA on all 26 structural features
    # ------------------------------------------------------------------
    print("\n[B] FULL-PCA APPROACH (all 26 structural features)")
    pca_full = PCA(random_state=42)
    cumvar_full = np.cumsum(pca_full.fit(X_scaled).explained_variance_ratio_)
    nc_full = int(np.searchsorted(cumvar_full, 0.80) + 1)
    print(f"  PCA: {nc_full} components for 80% variance ({cumvar_full[nc_full-1]*100:.1f}%)")
    for i, ev in enumerate(pca_full.explained_variance_ratio_[:nc_full]):
        print(f"    PC{i+1}: {ev*100:.1f}%")

    pca_b = PCA(n_components=nc_full, random_state=42)
    Xp_full = pca_b.fit_transform(X_scaled)

    (full_labels, k_full, ari_full, gaps_full, sks_full, kr_full,
     Zl_full, wkm_full, nuts2_full) = run_pipeline("Full-PCA", Xp_full, df, 26)

    cm_full = pd.DataFrame(X_scaled, columns=STRUCTURAL_FEATURES)
    cm_full["cluster"] = full_labels
    names_full = make_cluster_names(
        cm_full.groupby("cluster")[STRUCTURAL_FEATURES].mean(), STRUCTURAL_FEATURES
    )

    # ------------------------------------------------------------------
    # Cross-approach ARI
    # ------------------------------------------------------------------
    cross_ari = adjusted_rand_score(vif_labels, full_labels)
    print(f"\nCross-approach ARI (VIF-pruned vs Full-PCA): {cross_ari:.3f}")

    # ------------------------------------------------------------------
    # Print Full-PCA cluster compositions
    # ------------------------------------------------------------------
    print("\n[B] Full-PCA cluster compositions:")
    for c in sorted(np.unique(full_labels)):
        n_c = int((full_labels == c).sum())
        counties = ", ".join(df.loc[full_labels == c, "judet"].values)
        print(f"  {names_full.get(c, f'C{c}')}  (n={n_c}): {counties}")

    print("\n[A] VIF-pruned cluster compositions (for comparison):")
    for c in sorted(np.unique(vif_labels)):
        n_c = int((vif_labels == c).sum())
        counties = ", ".join(df.loc[vif_labels == c, "judet"].values)
        print(f"  {names_vif.get(c, f'C{c}')}  (n={n_c}): {counties}")

    # ------------------------------------------------------------------
    # ANOVA on electoral features for both
    # ------------------------------------------------------------------
    print("\nANOVA F-stats (held-out electoral features):")
    print(f"  {'Feature':<30s}  {'VIF F':>8s}  {'FullPCA F':>10s}  {'Winner':>8s}")
    print("  " + "-" * 65)
    for col in ELECTORAL_OVERLAY:
        if col not in df.columns:
            continue
        def _f(labs):
            groups = [df.loc[labs == c, col].dropna().values
                      for c in range(1, max(labs) + 1) if (labs == c).sum() >= 2]
            return f_oneway(*groups)[0] if len(groups) >= 2 else np.nan
        fv = _f(vif_labels); ff = _f(full_labels)
        winner = "Full-PCA" if (not np.isnan(ff) and (np.isnan(fv) or ff > fv)) else "VIF"
        print(f"  {col:<30s}  {fv:8.2f}  {ff:10.2f}  {winner:>8s}")

    # ------------------------------------------------------------------
    # Save Full-PCA cluster assignments
    # ------------------------------------------------------------------
    out = df[["siruta", "judet", "nuts2_code", "geo_nuts3"]].copy()
    out["cluster_id"]   = full_labels
    out["cluster_name"] = out["cluster_id"].map(names_full)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV.name}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")
    plot_gap(gaps_full, sks_full, kr_full, k_full,
             FIG_DIR / "06b_gap_fullpca.png", "Full PCA (26 features)")
    plot_dendrogram(df, Zl_full, k_full,
                    FIG_DIR / "06b_dendrogram_fullpca.png", "Full PCA")
    plot_pca_biplot(Xp_full, full_labels, df["judet"].values,
                    pca_b, STRUCTURAL_FEATURES,
                    FIG_DIR / "06b_pca_biplot_fullpca.png", "Full PCA")
    plot_heatmap(X_scaled, full_labels, STRUCTURAL_FEATURES, names_full,
                 FIG_DIR / "06b_cluster_heatmap_fullpca.png", "Full PCA (26 features)")
    plot_vote_profiles(df, full_labels, names_full,
                       FIG_DIR / "06b_vote_profiles_fullpca.png", "Full PCA")
    plot_comparison(df, vif_labels, full_labels, k_vif, k_full,
                    ari_vif, ari_full, wkm_vif, wkm_full,
                    nuts2_vif, nuts2_full,
                    FIG_DIR / "06b_approach_comparison.png")

    # ------------------------------------------------------------------
    # Final scorecard
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"  {'Metric':<35s}  {'VIF-pruned':>12s}  {'Full-PCA':>12s}")
    print("  " + "-" * 64)
    print(f"  {'k selected':<35s}  {k_vif:>12d}  {k_full:>12d}")
    print(f"  {'PCA features fed in':<35s}  {len(retained_vif):>12d}  {26:>12d}")
    print(f"  {'Ward vs k-means ARI':<35s}  {wkm_vif:>12.3f}  {wkm_full:>12.3f}")
    print(f"  {'ARI vs NUTS2':<35s}  {nuts2_vif:>12.3f}  {nuts2_full:>12.3f}")
    best_vif = max(v[0] for v in ari_vif.values())
    best_full = max(v[0] for v in ari_full.values())
    print(f"  {'Best bootstrap ARI':<35s}  {best_vif:>12.3f}  {best_full:>12.3f}")
    print(f"  {'Cross-approach ARI':<35s}  {cross_ari:>12.3f}  {cross_ari:>12.3f}")


if __name__ == "__main__":
    main()
