"""
Finalise k=5 Full-PCA cluster partition with Romanian names.
Produces definitive cluster_assignments_final.csv and updated figures.
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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
np.random.seed(42)

FEAT_PATH = ROOT / "data" / "processed" / "features_2022_judet.parquet"
OUT_CSV   = ROOT / "data" / "processed" / "cluster_assignments_final.csv"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

STRUCTURAL_FEATURES = [
    "gdp_per_capita", "unemployment_rate", "poverty_risk_pct",
    "tertiary_education_pct", "early_school_leaving_pct", "internet_use_pct",
    "net_migration_rate", "population_density",
    "pct_pop_over65", "pct_pop_under25",
    "pct_employment_agriculture", "pct_employment_industry", "pct_employment_services",
    "industry_gva_share", "gva_deindustrial_index",
    "long_term_unemployment_rate", "household_income_pps",
    "pct_maghiari", "pct_romi", "pct_reformed", "pct_pentecostal",
    "pct_orthodox", "pct_greco_catholic", "pct_roman_catholic",
    "pct_germani", "pct_ucraineni",
    "pct_urban", "avg_household_size",
    "emigrants_permanent_rate", "emigrants_temporary_rate",
    "hospitals_per_100k", "school_units_per_10k",
    "tourism_units_per_10k", "convictions_per_100k",
]

ELECTORAL_OVERLAY = [
    "t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR", "t_SOS", "t_POT",
    "t_turnout", "t_ANTI_ESTABLISHMENT", "georgescu_r1_2024", "simion_r1_2025",
]

# ---------------------------------------------------------------------------
# Definitive cluster names  (k=5, Full-PCA)
# ---------------------------------------------------------------------------
# id → (short_name, full_label, hex_colour)
CLUSTER_DEFS = {
    1: ("Moldova Profunda",   "C1 — Moldova Profunda: Periferia Agricola si Emigranta",  "#e41a1c"),
    2: ("Campia Uitata",      "C2 — Campia Uitata: Sudul Stagnant",                      "#ff7f00"),
    3: ("Insula Capitalei",   "C3 — Insula Capitalei: Polul de Crestere",                "#4daf4a"),
    4: ("Centrele Dinamice",  "C4 — Centrele Dinamice: Ardelul Motor",                   "#377eb8"),
    5: ("Arcul Identitar",    "C5 — Arcul Identitar: Mozaicul Etnic al NV",              "#984ea3"),
}

def main():
    df = pd.read_parquet(FEAT_PATH)
    X  = df[STRUCTURAL_FEATURES].fillna(df[STRUCTURAL_FEATURES].median())
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=8, random_state=42)
    Xp  = pca.fit_transform(Xs)
    Z   = linkage(Xp, method="ward")
    labels = fcluster(Z, 5, criterion="maxclust")

    short_names = {c: v[0] for c, v in CLUSTER_DEFS.items()}
    full_labels = {c: v[1] for c, v in CLUSTER_DEFS.items()}
    colours     = {c: v[2] for c, v in CLUSTER_DEFS.items()}

    # -----------------------------------------------------------------------
    # Print composition
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("FINAL CLUSTER ASSIGNMENTS  (k=5, Full-PCA, 34 structural features)")
    print("=" * 65)
    Xdf = pd.DataFrame(Xs, columns=STRUCTURAL_FEATURES)
    Xdf["cluster"] = labels
    key_cols = ["gva_deindustrial_index", "pct_employment_agriculture",
                "gdp_per_capita", "pct_urban", "household_income_pps",
                "long_term_unemployment_rate"]
    for c in range(1, 6):
        name  = short_names[c]
        n     = int((labels == c).sum())
        ctys  = ", ".join(df.loc[labels == c, "judet"].values)
        means = Xdf[Xdf["cluster"] == c][key_cols].mean()
        print(f"\n  [{c}] {name}  (n={n})")
        print(f"       {ctys}")
        print(f"       deindustrial={means['gva_deindustrial_index']:+.2f}  "
              f"agri={means['pct_employment_agriculture']:+.2f}  "
              f"gdp={means['gdp_per_capita']:+.2f}  "
              f"urban={means['pct_urban']:+.2f}  "
              f"hhinc={means['household_income_pps']:+.2f}  "
              f"ltu={means['long_term_unemployment_rate']:+.2f}")

    # -----------------------------------------------------------------------
    # ANOVA on held-out electoral features
    # -----------------------------------------------------------------------
    print("\nANOVA (held-out electoral, not used in clustering):")
    for col in ELECTORAL_OVERLAY:
        if col not in df.columns:
            continue
        groups = [df.loc[labels == c, col].dropna().values
                  for c in range(1, 6) if (labels == c).sum() >= 2]
        if len(groups) < 2:
            continue
        F, p = f_oneway(*groups)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "  "))
        print(f"  {col:<32s}  F={F:6.2f}  p={p:.4f} {sig}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    out = df[["siruta", "judet", "nuts2_code", "geo_nuts3"]].copy()
    out["cluster_id"]         = labels
    out["cluster_short"]      = out["cluster_id"].map(short_names)
    out["cluster_label"]      = out["cluster_id"].map(full_labels)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV.name}")

    # -----------------------------------------------------------------------
    # Figure 1: Dendrogram
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 5))
    cut_d = Z[-(5 - 1), 2]
    palette_list = [colours[c] for c in sorted(colours)]
    dendrogram(Z, labels=df["judet"].values, ax=ax,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=cut_d, above_threshold_color="grey")
    ax.axhline(cut_d, color="black", ls="--", lw=1.2, label="k=5 cut")
    ax.set_title("Ward Hierarchical Clustering — 34 Structural Features, k=5\n"
                 "Romanian Judete Voter Profiles", fontsize=11)
    ax.set_ylabel("Ward Linkage Distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06c_dendrogram_final.png", dpi=300)
    plt.close(fig)
    print("  Saved -> 06c_dendrogram_final.png")

    # -----------------------------------------------------------------------
    # Figure 2: PCA biplot with named clusters
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(1, 6):
        mask = labels == c
        ax.scatter(Xp[mask, 0], Xp[mask, 1],
                   color=colours[c], s=80, zorder=3,
                   label=short_names[c], edgecolors="white", linewidths=0.4)
        for name, x, y in zip(df.loc[mask, "judet"].values,
                               Xp[mask, 0], Xp[mask, 1]):
            ax.annotate(name, (x, y), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")

    # Top loadings (PC1)
    loadings = pca.components_[:2].T
    top_idx  = np.argsort(np.abs(loadings[:, 0]))[-6:]
    for i in top_idx:
        ax.annotate("", xy=(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
        ax.text(loadings[i, 0] * 3.4, loadings[i, 1] * 3.4,
                STRUCTURAL_FEATURES[i], fontsize=7, color="#555555")

    v1 = pca.explained_variance_ratio_[0] * 100
    v2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({v1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({v2:.1f}% variance)")
    ax.set_title("PCA Biplot — 34 Structural Features\nk=5 Voter Profile Clusters")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.axhline(0, lw=0.4, ls="--", color="k"); ax.axvline(0, lw=0.4, ls="--", color="k")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06c_pca_biplot_final.png", dpi=300)
    plt.close(fig)
    print("  Saved -> 06c_pca_biplot_final.png")

    # -----------------------------------------------------------------------
    # Figure 3: Feature heatmap
    # -----------------------------------------------------------------------
    display_feats = [
        "gdp_per_capita", "household_income_pps", "poverty_risk_pct",
        "unemployment_rate", "long_term_unemployment_rate",
        "gva_deindustrial_index", "industry_gva_share",
        "pct_employment_agriculture", "pct_employment_industry", "pct_employment_services",
        "pct_urban", "net_migration_rate", "emigrants_temporary_rate",
        "tertiary_education_pct", "early_school_leaving_pct",
        "pct_pop_over65", "pct_pop_under25",
        "pct_orthodox", "pct_reformed", "pct_greco_catholic",
        "pct_maghiari", "pct_romi", "pct_germani", "pct_ucraineni",
    ]
    Xdf["cluster_name"] = Xdf["cluster"].map(short_names)
    means = Xdf.groupby("cluster_name")[display_feats].mean()
    # Reorder rows
    row_order = [short_names[c] for c in range(1, 6)]
    means = means.loc[row_order]

    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(means, ax=ax, cmap="RdBu_r", center=0, linewidths=0.4,
                cbar_kws={"label": "Z-score (cluster mean)", "shrink": 0.7})
    ax.set_title("Cluster Structural Profiles — Z-scored Features\n"
                 "k=5 Romanian Voter Typology", fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06c_heatmap_final.png", dpi=300)
    plt.close(fig)
    print("  Saved -> 06c_heatmap_final.png")

    # -----------------------------------------------------------------------
    # Figure 4: Vote profiles
    # -----------------------------------------------------------------------
    party_cols = ["t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR",
                  "t_SOS", "t_POT", "t_turnout"]
    edf = df[party_cols].copy()
    edf["cluster_name"] = [short_names[c] for c in labels]
    profile = edf.groupby("cluster_name")[party_cols].mean()
    profile = profile.loc[row_order]
    profile.columns = [c.replace("t_", "") for c in profile.columns]

    fig, ax = plt.subplots(figsize=(13, 5))
    profile.T.plot(kind="bar", ax=ax, width=0.78,
                   color=[colours[c] for c in range(1, 6)])
    ax.set_xlabel("Party / Indicator")
    ax.set_ylabel("Mean Value")
    ax.set_title("Electoral Profile by Cluster (Held-Out — Not Used in Clustering)\n"
                 "2024 Parliamentary + Turnout")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06c_vote_profiles_final.png", dpi=300)
    plt.close(fig)
    print("  Saved -> 06c_vote_profiles_final.png")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DONE — definitive k=5 partition locked in")
    print("  cluster_assignments_final.csv")
    print("  figures/06c_*.png  (4 figures)")
    print("  Next: Task 3 — Georgescu->Simion transfer analysis")


if __name__ == "__main__":
    main()
