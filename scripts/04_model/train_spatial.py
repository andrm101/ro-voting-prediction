"""
Spatial autocorrelation diagnostics and spatial lag model.

Tests whether OLS residuals exhibit significant spatial autocorrelation
(Moran's I). If significant, fits a spatial lag model using ML_Lag (PySAL/spreg).
Compares OLS vs spatial model on AIC and pseudo-R².

Spatial weights: k=5 nearest-neighbour based on county capital centroid
coordinates (hardcoded — stable geographic facts, no shapefile dependency).
Row-standardised.

Primary target: t_AUR (the most spatially interesting, given geographic
clustering of nationalist vote in NE Romania and Oltenia).
Also runs on t_PSD and t_ANTI_ESTABLISHMENT.

Outputs:
  data/processed/spatial_diagnostics.parquet
  figures/moran_scatter_<target>.png
  figures/spatial_residuals_<target>.png  (if pysal map available)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import NUTS3_SIRUTA

FEAT_PATH = ROOT / "data" / "processed" / "features_2022_judet.parquet"
EN_PREDS  = ROOT / "data" / "processed" / "predictions_elastic_net.parquet"
OUT_DIAG  = ROOT / "data" / "processed" / "spatial_diagnostics.parquet"
FIG_DIR   = ROOT / "figures"

np.random.seed(42)

# ---------------------------------------------------------------------------
# County capital centroids (lat, lon) keyed by SIRUTA code.
# Source: Romanian administrative geography — stable geographic facts.
# ---------------------------------------------------------------------------
CENTROIDS: dict[int, tuple[float, float]] = {
    10:  (46.07, 23.57),   # Alba Iulia
    20:  (46.17, 21.32),   # Arad
    30:  (44.85, 24.87),   # Pitești (Argeș)
    40:  (46.57, 26.92),   # Bacău
    50:  (47.07, 21.93),   # Oradea (Bihor)
    60:  (47.13, 24.50),   # Bistrița
    70:  (47.75, 26.67),   # Botoșani
    80:  (45.65, 25.60),   # Brașov
    90:  (45.27, 27.97),   # Brăila
    100: (45.15, 26.83),   # Buzău
    110: (45.30, 21.88),   # Reșița (Caraș-Severin)
    120: (44.20, 27.33),   # Călărași
    130: (46.77, 23.60),   # Cluj-Napoca
    140: (44.18, 28.65),   # Constanța
    150: (45.87, 25.78),   # Sfântu Gheorghe (Covasna)
    160: (44.93, 25.45),   # Târgoviște (Dâmbovița)
    170: (44.32, 23.80),   # Craiova (Dolj)
    180: (45.43, 28.05),   # Galați
    190: (43.90, 25.97),   # Giurgiu
    200: (45.03, 23.28),   # Târgu Jiu (Gorj)
    210: (46.37, 25.80),   # Miercurea Ciuc (Harghita)
    220: (45.88, 22.90),   # Deva (Hunedoara)
    230: (44.57, 27.37),   # Slobozia (Ialomița)
    240: (47.17, 27.57),   # Iași
    250: (44.57, 26.15),   # Ilfov
    260: (47.65, 23.57),   # Baia Mare (Maramureș)
    270: (44.63, 22.67),   # Dr.-Tr. Severin (Mehedinți)
    280: (46.55, 24.57),   # Târgu Mureș
    290: (46.92, 26.37),   # Piatra Neamț (Neamț)
    300: (44.43, 24.37),   # Slatina (Olt)
    310: (44.95, 25.97),   # Ploiești (Prahova)
    320: (47.80, 22.88),   # Satu Mare
    330: (47.18, 23.05),   # Zalău (Sălaj)
    340: (45.80, 24.15),   # Sibiu
    350: (47.65, 26.25),   # Suceava
    360: (43.97, 25.32),   # Alexandria (Teleorman)
    370: (45.75, 21.23),   # Timișoara (Timiș)
    380: (45.18, 28.80),   # Tulcea
    390: (46.63, 27.73),   # Vaslui
    400: (45.10, 24.37),   # Râmnicu Vâlcea (Vâlcea)
    410: (45.70, 27.18),   # Focșani (Vrancea)
    179: (44.43, 26.10),   # București
}

SPATIAL_TARGETS = ["t_AUR", "t_PSD", "t_ANTI_ESTABLISHMENT"]

FEATURE_COLS = [
    "gdp_per_capita", "unemployment_rate", "poverty_risk_pct",
    "tertiary_education_pct", "early_school_leaving_pct", "internet_use_pct",
    "population_density", "net_migration_rate",
    "pct_pop_over65", "pct_pop_under25",
    "pct_employment_agriculture",
    "pct_maghiari", "pct_romi",
    "pct_orthodox", "pct_pentecostal", "pct_reformed",
    "pct_urban",
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "ep_2024_anti_estab",
    "georgescu_r1_2024", "simion_r1_2025",
    "pres19r2_IOHANNIS",
    "emigrants_temporary_rate", "hospitals_per_100k",
    "convictions_per_100k", "avg_household_size",
    "ethno_linguistic_frag", "religious_frag",
    "nationalist_consolidation_ratio",
]


def _build_knn_weights(siruta_list: list[int], k: int = 5):
    """Build row-standardised k-NN spatial weights from county centroids."""
    try:
        from libpysal.weights import KNN
    except ImportError:
        print("  [ERROR] libpysal not installed. Run: conda install -c conda-forge libpysal",
              file=sys.stderr)
        return None

    coords = np.array([CENTROIDS[s] for s in siruta_list])
    W = KNN.from_array(coords, k=k)
    W.transform = "r"  # row-standardise
    return W


def run_moran(residuals: np.ndarray, W, target: str, siruta_list: list[int],
              judet_list: list[str]) -> dict:
    """Compute Moran's I and produce scatter plot."""
    try:
        from esda.moran import Moran
    except ImportError:
        print("  [ERROR] esda not installed. Run: conda install -c conda-forge esda",
              file=sys.stderr)
        return {}

    mi = Moran(residuals, W)
    print(f"  Moran's I = {mi.I:.4f}  p = {mi.p_sim:.4f} "
          f"({'SIGNIFICANT' if mi.p_sim < 0.05 else 'not significant'} at 5%)")

    # Moran scatter plot
    lag_residuals = np.array([
        sum(W.weights[i][j] * residuals[W.neighbors[i][j]]
            for j in range(len(W.neighbors[i])))
        for i in range(len(residuals))
    ])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(residuals, lag_residuals, alpha=0.7, edgecolors="black", lw=0.5)
    for i, j in enumerate(judet_list):
        if abs(residuals[i]) > np.percentile(np.abs(residuals), 80):
            ax.annotate(j, (residuals[i], lag_residuals[i]), fontsize=7)
    m_coef = np.polyfit(residuals, lag_residuals, 1)
    x_line = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x_line, np.polyval(m_coef, x_line), "r-", lw=1.5, label=f"slope={m_coef[0]:.3f}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("OLS Residual")
    ax.set_ylabel("Spatial Lag of Residual")
    ax.set_title(f"Moran's I Scatter — {target}\n"
                 f"I={mi.I:.4f}, p={mi.p_sim:.4f} (999 permutations)")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"moran_scatter_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"moran_I": mi.I, "moran_p": mi.p_sim, "moran_z": mi.z_sim}


def run_spatial_lag(y: np.ndarray, X: np.ndarray,
                    feature_names: list[str], W, target: str) -> dict:
    """Fit ML spatial lag model if Moran's I is significant."""
    try:
        from spreg import ML_Lag
    except ImportError:
        print("  [ERROR] spreg not installed. Run: conda install -c conda-forge spreg",
              file=sys.stderr)
        return {}

    Xc = np.column_stack([np.ones(len(y)), X])
    col_names = ["const"] + feature_names

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag_model = ML_Lag(y.reshape(-1, 1), Xc, W,
                           name_y=target, name_x=col_names)

    print(f"\n  Spatial Lag Model — {target}")
    print(f"  Spatial lag coefficient (ρ): {lag_model.rho:.4f}  "
          f"(p={lag_model.z_stat[-1][1]:.4f})")
    print(f"  Pseudo-R²: {lag_model.pr2:.4f}")

    # OLS comparison
    ols_res = sm.OLS(y, sm.add_constant(X)).fit()
    print(f"  OLS R²:    {ols_res.rsquared:.4f}  "
          f"AIC={ols_res.aic:.1f}")
    print(f"  AIC improvement (OLS->SLM): {ols_res.aic - lag_model.aic:.1f}")

    coef_df = pd.DataFrame({
        "feature": col_names + ["rho"],
        "coef":    list(lag_model.betas.ravel()) + [lag_model.rho],
        "z_stat":  [s[0] for s in lag_model.z_stat],
        "p_value": [s[1] for s in lag_model.z_stat],
    })
    print("\n  Coefficients:")
    print(coef_df[["feature", "coef", "p_value"]].to_string(index=False))

    return {
        "rho": lag_model.rho,
        "pseudo_r2": lag_model.pr2,
        "aic_spatial": lag_model.aic,
        "aic_ols": ols_res.aic,
    }


def main() -> None:
    if not FEAT_PATH.exists():
        print(f"Missing: {FEAT_PATH}\nRun build_features.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading feature matrix ...")
    df = pd.read_parquet(FEAT_PATH)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X_df = df[available_features].copy()
    complete_mask = X_df.notna().all(axis=1)
    df_c = df[complete_mask].reset_index(drop=True)
    X_df = X_df[complete_mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    siruta_list = df_c["siruta"].tolist()
    judet_list  = df_c["judet"].tolist()

    # Build weights once
    print(f"\nBuilding k=5 NN spatial weights for {len(siruta_list)} counties ...")
    W = _build_knn_weights(siruta_list, k=5)
    if W is None:
        print("Cannot build spatial weights. Exiting spatial stage.", file=sys.stderr)
        sys.exit(1)

    diagnostics: list[dict] = []

    for target in SPATIAL_TARGETS:
        if target not in df_c.columns or df_c[target].isna().all():
            print(f"\n  [SKIP] {target} not available.")
            continue

        y = df_c[target].values
        print(f"\n== {target} ==================================")

        # OLS residuals
        ols_res = sm.OLS(y, sm.add_constant(X_scaled)).fit()
        residuals = np.asarray(ols_res.resid)
        print(f"  OLS R² = {ols_res.rsquared:.4f}")

        # Moran's I
        moran_stats = run_moran(residuals, W, target, siruta_list, judet_list)

        row = {"target": target, "ols_r2": ols_res.rsquared, **moran_stats}

        # Spatial lag if Moran's I significant at 10%
        if moran_stats.get("moran_p", 1.0) < 0.10:
            print(f"\n  -> Spatial autocorrelation detected (p={moran_stats['moran_p']:.4f}). "
                  f"Fitting spatial lag model ...")
            slm_stats = run_spatial_lag(y, X_scaled, available_features, W, target)
            row.update(slm_stats)
            row["spatial_model_fitted"] = True
        else:
            print(f"  -> No significant spatial autocorrelation. OLS is appropriate.")
            row["spatial_model_fitted"] = False

        diagnostics.append(row)

    if diagnostics:
        diag_df = pd.DataFrame(diagnostics)
        OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
        diag_df.to_parquet(OUT_DIAG, index=False)
        print(f"\nSaved spatial diagnostics -> {OUT_DIAG}")
        print("\nSummary:")
        print(diag_df[["target", "ols_r2", "moran_I", "moran_p",
                        "spatial_model_fitted"]].to_string(index=False))

    print("\nNext step: python scripts/05_evaluate/evaluate.py")


if __name__ == "__main__":
    main()
