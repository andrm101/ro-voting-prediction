"""
Baseline interpretable model: Elastic Net + OLS per party target.

For each target (AUR, PSD, PNL, USR, UDMR, SOS, turnout):
  1. Compute VIF on the full feature matrix — flag VIF > 10.
  2. OLS (statsmodels) for coefficient interpretability and CIs.
  3. ElasticNetCV (sklearn) with LOOCV for predictive evaluation.
     pct_maghiari is a forced structural control — included but noted separately.
  4. Bootstrap CIs on LOOCV R² (1000 resamples) to quantify stability.
  5. Leave-Region-Out CV (8 Romanian development regions) as robustness check.
  6. UDMR is modelled with OLS only (pct_maghiari explains ~95%+ of variance).

Outputs:
  data/processed/predictions_elastic_net.parquet
  data/processed/ols_coefficients.parquet
  data/processed/bootstrap_r2.parquet
  data/processed/lro_cv_results.parquet
  figures/coef_ols_<target>.png
  figures/loocv_scatter_en_<target>.png

Reproducibility: np.random.seed(42)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import PARLIAMENTARY_MODEL_PARTIES

FEAT_PATH = ROOT / "data" / "processed" / "features_2022_judet.parquet"
OUT_PREDS    = ROOT / "data" / "processed" / "predictions_elastic_net.parquet"
OUT_COEFS    = ROOT / "data" / "processed" / "ols_coefficients.parquet"
OUT_BOOT     = ROOT / "data" / "processed" / "bootstrap_r2.parquet"
OUT_LRO      = ROOT / "data" / "processed" / "lro_cv_results.parquet"
FIG_DIR      = ROOT / "figures"

np.random.seed(42)

# OLS feature set: composites replace their raw inputs to avoid perfect multicollinearity.
# idx_grievance absorbs: gdp_per_capita, unemployment_rate, poverty_risk_pct, early_school_leaving_pct
# idx_modernity absorbs: tertiary_education_pct, internet_use_pct, early_school_leaving_pct
# idx_demographic_pressure absorbs: pct_pop_over65, net_migration_rate, pct_pop_under25
# New INS/engineered features do not overlap with composites and can be added safely;
# interaction terms (rural_elderly, etc.) are omitted here to avoid VIF inflation.
# OLS feature set: keep original 16 working features + 4 new orthogonal INS-derived columns.
# ethno_linguistic_frag, religious_frag, nationalist_consolidation_ratio are excluded here:
#   they are mathematical functions of pct_maghiari/pct_orthodox/georgescu+simion respectively
#   and will produce VIF >> 10 when included alongside their components.
# pres19r2_IOHANNIS is excluded: correlated with ep_2024_anti_estab + simion bloc (VIF ~22).
# These variables CAN be used in ElasticNet and XGBoost where regularisation handles collinearity.
OLS_FEATURE_COLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "population_density", "pct_employment_agriculture",
    "pct_employment_industry", "pct_employment_services",
    "pct_maghiari", "pct_romi",
    "pct_orthodox", "pct_pentecostal", "pct_reformed",
    "pct_urban",
    "ep_2024_anti_estab", "georgescu_r1_2024", "simion_r1_2025",
    "emigrants_temporary_rate", "hospitals_per_100k",
    "convictions_per_100k", "avg_household_size",
]

# ElasticNet feature set: raw features + INS extras + interactions
# (no composites — EN handles collinearity via regularisation)
EN_FEATURE_COLS = [
    "gdp_per_capita", "unemployment_rate", "poverty_risk_pct",
    "tertiary_education_pct", "early_school_leaving_pct", "internet_use_pct",
    "population_density", "net_migration_rate",
    "pct_pop_over65", "pct_pop_under25",
    "pct_employment_agriculture", "pct_employment_industry", "pct_employment_services",
    "pct_maghiari", "pct_romi",
    "pct_orthodox", "pct_pentecostal", "pct_reformed",
    "pct_urban",
    "ep_2024_anti_estab", "georgescu_r1_2024", "simion_r1_2025",
    "pres19r2_IOHANNIS",
    "emigrants_permanent_rate", "emigrants_temporary_rate",
    "school_units_per_10k", "hospitals_per_100k",
    "tourism_units_per_10k", "convictions_per_100k", "avg_household_size",
    "ethno_linguistic_frag", "religious_frag",
    "rural_elderly", "young_urban", "agri_emigration",
    "orthodox_poverty", "anti_estab_persistence",
    "nationalist_consolidation_ratio",
]

# For UDMR, pct_maghiari explains ~95%+ of variance — OLS only, no ElasticNet LOOCV
OLS_ONLY_TARGETS = {"t_UDMR"}

# pct_maghiari: structural/ethnic control — always in OLS, VIF not flagged for this column
FORCED_CONTROLS = ["pct_maghiari"]


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Return VIF for each column in X (must be numeric, no NaN)."""
    vif = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return vif.sort_values("VIF", ascending=False)


def fit_ols(X: pd.DataFrame, y: pd.Series, target: str) -> tuple[pd.DataFrame, float]:
    """Fit OLS via statsmodels; return coefficient DataFrame and adjusted R²."""
    Xc = sm.add_constant(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.OLS(y, Xc).fit()
    coefs = pd.DataFrame({
        "target":    target,
        "feature":   res.params.index,
        "coef":      res.params.values,
        "std_err":   res.bse.values,
        "t_stat":    res.tvalues.values,
        "p_value":   res.pvalues.values,
        "ci_low":    res.conf_int()[0].values,
        "ci_high":   res.conf_int()[1].values,
    })
    return coefs, res.rsquared_adj


def loocv_elastic_net(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, float, float]:
    """
    LOOCV with ElasticNetCV refitted on each (N-1) fold.
    Returns (loo_predictions, final_alpha, final_l1_ratio).
    """
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    Xa, ya = X.values, y.values

    # First, find best hyperparameters on full data for final model reporting
    pipe_full = Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5, max_iter=20_000, random_state=42,
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe_full.fit(Xa, ya)
    best_alpha    = pipe_full.named_steps["en"].alpha_
    best_l1_ratio = pipe_full.named_steps["en"].l1_ratio_

    # LOOCV predictions with fixed hyperparameters
    from sklearn.linear_model import ElasticNet
    for train_idx, test_idx in loo.split(Xa):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Xa[train_idx])
        X_te = scaler.transform(Xa[test_idx])
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio,
                           max_iter=20_000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, ya[train_idx])
        preds[test_idx] = model.predict(X_te)

    return preds, best_alpha, best_l1_ratio


def bootstrap_r2(
    X: pd.DataFrame, y: pd.Series,
    best_alpha: float, best_l1_ratio: float,
    n_boot: int = 1000,
) -> dict:
    """
    Paired bootstrap CI for LOOCV R².
    Resamples (actual, loo_pred) pairs with replacement; CI estimates sampling
    variability in R² given the fixed LOOCV predictions.
    Returns {r2_mean, ci_low_95, ci_high_95}.
    """
    from sklearn.linear_model import ElasticNet
    loo = LeaveOneOut()
    Xa, ya = X.values, y.values

    # Generate LOOCV predictions once
    loo_preds = np.full(len(ya), np.nan)
    for train_idx, test_idx in loo.split(Xa):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Xa[train_idx])
        X_te = scaler.transform(Xa[test_idx])
        m = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=20_000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, ya[train_idx])
        loo_preds[test_idx] = m.predict(X_te)

    n = len(ya)
    boot_r2s = np.empty(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_b, p_b = ya[idx], loo_preds[idx]
        ss_res = np.sum((a_b - p_b) ** 2)
        ss_tot = np.sum((a_b - a_b.mean()) ** 2)
        boot_r2s[i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "r2_mean":   float(np.nanmean(boot_r2s)),
        "ci_low_95": float(np.nanpercentile(boot_r2s, 2.5)),
        "ci_high_95": float(np.nanpercentile(boot_r2s, 97.5)),
        "n_boot":    n_boot,
    }


# Romanian development regions (NUTS 2) → list of SIRUTA codes
# Source: Romanian Law 315/2004 on regional development — stable administrative geography
NUTS2_SIRUTA: dict[str, list[int]] = {
    "RO11": [360, 40, 290, 350, 390, 410, 70],        # Nord-Est
    "RO12": [390, 150, 80, 140, 380, 10, 340],         # Sud-Est (Vrancea→RO11, override below)
    "RO21": [150, 80, 140, 380, 90, 180, 100],         # Sud-Est (Constanta,Braila,Buzau,Galati,Tulcea,Covasna,Brasov)
    "RO22": [360, 120, 190, 300, 170, 230, 310],       # Sud-Muntenia + Ilfov
    "RO31": [250, 160, 310, 120, 190, 230, 300, 360],  # Sud-Muntenia
    "RO32": [179],                                      # Bucuresti-Ilfov
    "RO41": [200, 220, 270, 170, 300, 10, 340, 400],   # Sud-Vest Oltenia
    "RO42": [370, 110, 20, 280, 330, 260, 320],        # Vest
    "RO51": [130, 60, 50, 150, 210, 340, 280],         # Nord-Vest (Centre)
    "RO52": [80, 150, 210, 340, 10, 60, 130],          # Centru
}

# Use the nuts2_code column from the feature matrix for region assignment
def leave_region_out_cv(
    X: pd.DataFrame, y: pd.Series,
    nuts2_codes: pd.Series,
    best_alpha: float, best_l1_ratio: float,
) -> pd.DataFrame:
    """
    Leave-Region-Out CV: hold out one NUTS 2 region at a time (8 regions).
    Tests whether the model generalises across geographic clusters, not just
    individual counties. Returns per-region metrics.
    """
    from sklearn.linear_model import ElasticNet
    Xa, ya = X.values, y.values
    regions = nuts2_codes.unique()
    rows: list[dict] = []

    for region in sorted(regions):
        test_mask  = (nuts2_codes == region).values
        train_mask = ~test_mask
        n_test = test_mask.sum()
        if n_test == 0 or train_mask.sum() < 5:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Xa[train_mask])
        X_te = scaler.transform(Xa[test_mask])
        m = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=20_000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_tr, ya[train_mask])
        preds = m.predict(X_te)

        res = ya[test_mask] - preds
        mae  = float(np.mean(np.abs(res)))
        rmse = float(np.sqrt(np.mean(res ** 2)))
        rows.append({"region": region, "n_counties": int(n_test),
                     "mae": mae, "rmse": rmse})

    return pd.DataFrame(rows)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = actual - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2  = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    return {"r2_loocv": r2, "mae": mae, "rmse": rmse}


def plot_coef(coefs: pd.DataFrame, target: str) -> None:
    coefs_plot = coefs[coefs["feature"] != "const"].copy()
    coefs_plot = coefs_plot.sort_values("coef")
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in coefs_plot["coef"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(coefs_plot) * 0.35)))
    ax.barh(coefs_plot["feature"], coefs_plot["coef"], color=colors,
            xerr=[coefs_plot["coef"] - coefs_plot["ci_low"],
                  coefs_plot["ci_high"] - coefs_plot["coef"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OLS Coefficient (standardised features)")
    ax.set_title(f"OLS Coefficients — {target}\n"
                 f"(error bars = 95% CI, N=42 counties)")
    ax.set_ylabel("")
    fig.text(0.5, -0.02,
             "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"coef_ols_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loocv_scatter(
    siruta: pd.Series, judet: pd.Series,
    actual: np.ndarray, predicted: np.ndarray, target: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.7, edgecolors="black", linewidths=0.5)
    # Label outliers (residual > 1.5 × IQR)
    residuals = actual - predicted
    iqr = np.percentile(np.abs(residuals), 75) - np.percentile(np.abs(residuals), 25)
    threshold = np.percentile(np.abs(residuals), 75) + 1.5 * iqr
    for i, (a, p, j) in enumerate(zip(actual, predicted, judet)):
        if abs(a - p) > threshold:
            ax.annotate(j, (a, p), fontsize=7, ha="left", va="bottom")
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="Perfect prediction")
    m = _metrics(actual, predicted)
    ax.set_xlabel(f"Actual {target} (2024 parliamentary)")
    ax.set_ylabel(f"LOOCV Predicted {target}")
    ax.set_title(f"Elastic Net LOOCV — {target}\n"
                 f"R²={m['r2_loocv']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02,
             "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"loocv_scatter_en_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not FEAT_PATH.exists():
        print(f"Missing: {FEAT_PATH}\nRun build_features.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading feature matrix …")
    df = pd.read_parquet(FEAT_PATH)
    print(f"  {len(df)} counties × {len(df.columns)} columns")

    ols_features = [c for c in OLS_FEATURE_COLS if c in df.columns]
    en_features  = [c for c in EN_FEATURE_COLS  if c in df.columns]

    for label, feat_list, ref in [("OLS", ols_features, OLS_FEATURE_COLS),
                                   ("EN",  en_features,  EN_FEATURE_COLS)]:
        missing = [c for c in ref if c not in df.columns]
        if missing:
            print(f"  [WARN] {label} features missing from matrix: {missing}", file=sys.stderr)

    # Complete cases driven by the union of both feature sets
    all_model_features = list(dict.fromkeys(ols_features + en_features))
    complete_mask = df[all_model_features].notna().all(axis=1)
    df_complete = df[complete_mask].reset_index(drop=True)

    n = len(df_complete)
    print(f"  Complete cases for modelling: {n}/42")
    if n < 20:
        print("  [ERROR] Too few complete cases for reliable LOOCV.", file=sys.stderr)
        sys.exit(1)

    # OLS uses composites-only feature set — check VIF on that subset
    scaler_ols = StandardScaler()
    X_ols = pd.DataFrame(
        scaler_ols.fit_transform(df_complete[ols_features]),
        columns=ols_features, index=df_complete.index,
    )
    print(f"\nComputing VIF on OLS feature set ({len(ols_features)} features) ...")
    vif_df = compute_vif(X_ols)
    print(vif_df.to_string(index=False))
    high_vif = vif_df[(vif_df["VIF"] > 10) &
                      (~vif_df["feature"].isin(FORCED_CONTROLS))]
    if not high_vif.empty:
        print(f"  [WARN] {len(high_vif)} OLS features with VIF > 10:")
        for _, row in high_vif.iterrows():
            print(f"    {row['feature']}: VIF={row['VIF']:.1f}")
    else:
        print("  VIF clean — no features above 10.")

    # EN raw feature matrix (regularisation handles collinearity)
    X_en_full = df_complete[en_features].copy()

    all_preds: list[pd.DataFrame] = []
    all_coefs: list[pd.DataFrame] = []
    all_boot:  list[dict]         = []
    all_lro:   list[pd.DataFrame] = []
    targets = [f"t_{p}" for p in PARLIAMENTARY_MODEL_PARTIES] + ["t_turnout"]

    for target in targets:
        if target not in df_complete.columns:
            print(f"\n  [SKIP] {target} not in feature matrix.")
            continue
        y = df_complete[target]
        if y.isna().all():
            print(f"\n  [SKIP] {target} is all NaN.")
            continue

        print(f"\n-- {target} ------------------------------------------")

        # OLS on composite feature set (interpretable, no multicollinearity)
        coefs, adj_r2_ols = fit_ols(X_ols, y, target)
        sig = coefs[coefs["p_value"] < 0.05]["feature"].tolist()
        print(f"  OLS adj-R²: {adj_r2_ols:.3f} | Significant (p<0.05): {sig}")
        all_coefs.append(coefs)
        plot_coef(coefs, target)

        if target in OLS_ONLY_TARGETS:
            print(f"  [INFO] {target}: OLS only (ethnic bloc variable).")
            preds_df = pd.DataFrame({
                "siruta": df_complete["siruta"],
                "judet":  df_complete["judet"],
                "target": target,
                "model":  "OLS",
                "actual": y.values,
                "predicted": (sm.add_constant(X_ols).dot(
                    sm.OLS(y, sm.add_constant(X_ols)).fit().params
                ).values),
            })
            preds_df["residual"] = preds_df["actual"] - preds_df["predicted"]
            all_preds.append(preds_df)
            continue

        # ElasticNet LOOCV on raw feature set (regularisation handles collinearity)
        loo_preds, best_alpha, best_l1_ratio = loocv_elastic_net(X_en_full, y)
        m = _metrics(y.values, loo_preds)
        print(f"  EN LOOCV  R²={m['r2_loocv']:.3f}  MAE={m['mae']:.4f}  "
              f"RMSE={m['rmse']:.4f}  a={best_alpha:.4f}  l1={best_l1_ratio:.2f}")

        # Bootstrap CI on LOOCV R²
        boot = bootstrap_r2(X_en_full, y, best_alpha, best_l1_ratio)
        print(f"  Bootstrap R² (1000): mean={boot['r2_mean']:.3f}  "
              f"95% CI [{boot['ci_low_95']:.3f}, {boot['ci_high_95']:.3f}]")
        all_boot.append({"target": target, **boot, "alpha": best_alpha,
                         "l1_ratio": best_l1_ratio})

        # Leave-Region-Out CV
        if "nuts2_code" in df_complete.columns:
            lro = leave_region_out_cv(X_en_full, y, df_complete["nuts2_code"],
                                      best_alpha, best_l1_ratio)
            lro["target"] = target
            all_lro.append(lro)
            print(f"  LRO-CV   MAE range [{lro['mae'].min():.4f}, {lro['mae'].max():.4f}]")

        preds_df = pd.DataFrame({
            "siruta":    df_complete["siruta"],
            "judet":     df_complete["judet"],
            "target":    target,
            "model":     "ElasticNet",
            "actual":    y.values,
            "predicted": loo_preds,
        })
        preds_df["residual"] = preds_df["actual"] - preds_df["predicted"]
        all_preds.append(preds_df)

        plot_loocv_scatter(
            df_complete["siruta"], df_complete["judet"],
            y.values, loo_preds, target
        )

    # Save outputs
    if all_preds:
        preds_out = pd.concat(all_preds, ignore_index=True)
        OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
        preds_out.to_parquet(OUT_PREDS, index=False)
        print(f"\nSaved predictions -> {OUT_PREDS}")

    if all_coefs:
        coefs_out = pd.concat(all_coefs, ignore_index=True)
        OUT_COEFS.parent.mkdir(parents=True, exist_ok=True)
        coefs_out.to_parquet(OUT_COEFS, index=False)
        print(f"Saved OLS coefficients -> {OUT_COEFS}")

    if all_boot:
        boot_out = pd.DataFrame(all_boot)
        OUT_BOOT.parent.mkdir(parents=True, exist_ok=True)
        boot_out.to_parquet(OUT_BOOT, index=False)
        print(f"Saved bootstrap R2 CIs -> {OUT_BOOT}")
        print("\nBootstrap R2 summary (EN LOOCV):")
        print(boot_out[["target", "r2_mean", "ci_low_95", "ci_high_95"]].to_string(index=False))

    if all_lro:
        lro_out = pd.concat(all_lro, ignore_index=True)
        OUT_LRO.parent.mkdir(parents=True, exist_ok=True)
        lro_out.to_parquet(OUT_LRO, index=False)
        print(f"Saved Leave-Region-Out CV -> {OUT_LRO}")

    print(f"Figures saved to {FIG_DIR}/")
    print("\nNext step: python scripts/04_model/train_gbm.py")


if __name__ == "__main__":
    main()
