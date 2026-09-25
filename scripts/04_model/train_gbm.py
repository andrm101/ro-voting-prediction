"""
XGBoost gradient boosting model with LOOCV evaluation and SHAP attribution.

For each party target:
  1. LOOCV (N-1 train, 1 test) for unbiased performance estimate.
  2. Full model refit on all 42 counties for SHAP value computation.
  3. SHAP TreeExplainer produces per-county, per-feature attribution.
  4. Partial Dependence Plots (PDPs) for top-3 SHAP features per target.

XGBoost hyperparameters are deliberately conservative given N=42:
  max_depth=2, min_child_weight=4, subsample=0.8, n_estimators=150.

Outputs:
  data/processed/predictions_xgboost.parquet
  data/processed/shap_values.parquet
  app/src/data/predictions.json
  app/src/data/shap_values.json
  figures/shap_summary_<target>.png
  figures/loocv_scatter_xgb_<target>.png
  figures/pdp_<target>.png

Reproducibility: np.random.seed(42)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import PARLIAMENTARY_MODEL_PARTIES

FEAT_PATH  = ROOT / "data" / "processed" / "features_2022_judet.parquet"
OUT_PREDS  = ROOT / "data" / "processed" / "predictions_xgboost.parquet"
OUT_SHAP   = ROOT / "data" / "processed" / "shap_values.parquet"
APP_PREDS  = ROOT / "app" / "src" / "data" / "predictions.json"
APP_SHAP   = ROOT / "app" / "src" / "data" / "shap_values.json"
FIG_DIR    = ROOT / "figures"

np.random.seed(42)

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
    "emigrants_permanent_rate", "emigrants_temporary_rate",
    "hospitals_per_100k", "convictions_per_100k",
    "tourism_units_per_10k", "avg_household_size",
    "ethno_linguistic_frag", "religious_frag",
    "rural_elderly", "young_urban", "agri_emigration",
    "orthodox_poverty", "anti_estab_persistence",
    "nationalist_consolidation_ratio",
]

# XGBoost params tuned for N=42.
# max_depth=2 vs original 3: shallower trees reduce variance on small N.
# min_child_weight=4: each leaf needs ≥4 observations — moderately conservative.
# n_estimators=150: slightly fewer trees; each LOOCV fold trains on 41 rows.
# reg_lambda=1.5, reg_alpha=0.2: modest L2/L1 — original was already reasonable.
XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=2,
    learning_rate=0.05,
    min_child_weight=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
)

# UDMR is near-deterministically predicted by pct_maghiari — skip LOOCV, use full-model SHAP only
OLS_ONLY_TARGETS = {"t_UDMR"}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = actual - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {
        "r2_loocv": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mae":  float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
    }


def loocv_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out cross-validation with XGBoost."""
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    scaler = StandardScaler()

    for train_idx, test_idx in loo.split(X):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = XGBRegressor(**XGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y[train_idx])
        preds[test_idx] = model.predict(X_te)

    return preds


def fit_full_model(X: np.ndarray, y: np.ndarray) -> tuple[XGBRegressor, StandardScaler]:
    """Refit on all data for SHAP computation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(**XGB_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)
    return model, scaler


def compute_shap(
    model: XGBRegressor, X_scaled: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Compute SHAP values for all observations."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    return pd.DataFrame(shap_values, columns=feature_names)


def plot_shap_summary(
    shap_df: pd.DataFrame, X_df: pd.DataFrame, target: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, max(5, len(shap_df.columns) * 0.4)))
    # Mean absolute SHAP per feature
    mean_abs = shap_df.abs().mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(mean_abs)))
    ax.barh(mean_abs.index, mean_abs.values, color=colors)
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title(f"XGBoost Feature Importance (SHAP) — {target}\n"
                 f"N=42 counties, LOOCV-validated model")
    ax.set_ylabel("")
    fig.text(0.5, -0.02,
             "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"shap_summary_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pdp(
    model: XGBRegressor,
    scaler: StandardScaler,
    X_df: pd.DataFrame,
    feature_names: list[str],
    shap_df: pd.DataFrame,
    target: str,
    top_n: int = 3,
) -> None:
    """
    Partial Dependence Plots for the top-N features by mean |SHAP| value.
    Uses sklearn's PartialDependenceDisplay with individual ICE lines overlaid.
    Marginalises over all other features (standard PDP assumption).
    """
    from sklearn.inspection import PartialDependenceDisplay

    mean_abs = shap_df[feature_names].abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(top_n).index.tolist()
    feature_indices = [feature_names.index(f) for f in top_features]

    X_scaled = scaler.transform(X_df.values)

    # Wrap XGBoost in a sklearn-compatible estimator with a .predict method
    class _Wrapper:
        def __init__(self, m):
            self._m = m
        def predict(self, X):
            return self._m.predict(X)
        @property
        def estimators_(self):
            return None

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4), sharey=False)
    if top_n == 1:
        axes = [axes]

    for ax, feat_idx, feat_name in zip(axes, feature_indices, top_features):
        grid_vals = np.linspace(X_scaled[:, feat_idx].min(),
                                X_scaled[:, feat_idx].max(), 50)
        # Manual PDP: vary one feature, average over all rows
        pdp_vals = np.empty(len(grid_vals))
        for gi, gv in enumerate(grid_vals):
            X_tmp = X_scaled.copy()
            X_tmp[:, feat_idx] = gv
            pdp_vals[gi] = model.predict(X_tmp).mean()

        # Convert grid back to original scale for axis labels
        orig_vals = (grid_vals * scaler.scale_[feat_idx]
                     + scaler.mean_[feat_idx])
        ax.plot(orig_vals, pdp_vals, lw=2, color="#2c7bb6")
        # ICE lines (individual conditional expectations)
        for i in range(len(X_scaled)):
            ice = np.empty(len(grid_vals))
            for gi, gv in enumerate(grid_vals):
                X_tmp = X_scaled[i:i+1].copy()
                X_tmp[0, feat_idx] = gv
                ice[gi] = model.predict(X_tmp)[0]
            ax.plot(orig_vals, ice, lw=0.3, alpha=0.3, color="grey")

        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel("Predicted vote share" if feat_name == top_features[0] else "")
        ax.set_title(f"PDP: {feat_name}", fontsize=9)

    fig.suptitle(f"Partial Dependence Plots — {target} (top {top_n} SHAP features)",
                 fontsize=10)
    fig.text(0.5, -0.03, "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"pdp_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loocv_scatter(
    judet: pd.Series, actual: np.ndarray, predicted: np.ndarray, target: str
) -> None:
    m = _metrics(actual, predicted)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.7, edgecolors="black", linewidths=0.5,
               zorder=3)
    residuals = actual - predicted
    threshold = np.percentile(np.abs(residuals), 80)
    for a, p, j in zip(actual, predicted, judet):
        if abs(a - p) > threshold:
            ax.annotate(j, (a, p), fontsize=7, ha="left", va="bottom")
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Perfect prediction")
    ax.set_xlabel(f"Actual {target} (2024 parliamentary)")
    ax.set_ylabel(f"XGBoost LOOCV Predicted {target}")
    ax.set_title(f"XGBoost LOOCV — {target}\n"
                 f"R²={m['r2_loocv']:.3f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02,
             "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"loocv_scatter_xgb_{target}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    X = X_df.values
    print(f"  {len(df_c)} complete cases, {len(available_features)} features")

    targets = [f"t_{p}" for p in PARLIAMENTARY_MODEL_PARTIES] + ["t_turnout"]

    all_preds: list[pd.DataFrame] = []
    all_shap:  list[pd.DataFrame] = []
    app_preds_list: list[dict]    = []
    app_shap_dict:  dict          = {}

    for target in targets:
        if target not in df_c.columns or df_c[target].isna().all():
            print(f"  [SKIP] {target} not available.")
            continue

        y = df_c[target].values
        print(f"\n-- {target} -------------------------------------")

        if target in OLS_ONLY_TARGETS:
            print(f"  [INFO] Skipping LOOCV for {target} (near-deterministic ethnic predictor).")
            print(f"         Running full model for SHAP only ...")
            model, scaler_m = fit_full_model(X, y)
            X_scaled_full = scaler_m.transform(X)
            shap_df = compute_shap(model, X_scaled_full, available_features)
            shap_df.insert(0, "siruta", df_c["siruta"].values)
            shap_df.insert(1, "judet",  df_c["judet"].values)
            shap_df.insert(2, "target", target)
            all_shap.append(shap_df)
            mean_abs_shap = shap_df[available_features].abs().mean().to_dict()
            app_shap_dict[target] = {k: round(v, 6) for k, v in
                                      sorted(mean_abs_shap.items(),
                                             key=lambda x: x[1], reverse=True)}
            plot_shap_summary(shap_df[available_features], X_df, target)
            continue

        # LOOCV
        loo_preds = loocv_predict(X, y)
        m = _metrics(y, loo_preds)
        print(f"  LOOCV  R²={m['r2_loocv']:.3f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

        preds_df = pd.DataFrame({
            "siruta":    df_c["siruta"].values,
            "judet":     df_c["judet"].values,
            "target":    target,
            "actual":    y,
            "predicted": loo_preds,
            "residual":  y - loo_preds,
        })
        all_preds.append(preds_df)

        # App predictions payload
        for row in preds_df.to_dict("records"):
            app_preds_list.append({
                "siruta": int(row["siruta"]),
                "judet":  row["judet"],
                "target": row["target"],
                "actual": round(float(row["actual"]), 5),
                "predicted": round(float(row["predicted"]), 5),
            })

        # Full model + SHAP
        print(f"  Fitting full model for SHAP ...")
        model, scaler = fit_full_model(X, y)
        X_scaled_full = scaler.transform(X)
        shap_df = compute_shap(model, X_scaled_full, available_features)
        shap_df.insert(0, "siruta", df_c["siruta"].values)
        shap_df.insert(1, "judet",  df_c["judet"].values)
        shap_df.insert(2, "target", target)
        all_shap.append(shap_df)

        # App SHAP payload: mean |SHAP| per feature per target
        mean_abs_shap = shap_df[available_features].abs().mean().to_dict()
        app_shap_dict[target] = {k: round(v, 6) for k, v in
                                  sorted(mean_abs_shap.items(),
                                         key=lambda x: x[1], reverse=True)}

        plot_shap_summary(shap_df[available_features], X_df, target)
        plot_pdp(model, scaler, X_df, available_features,
                 shap_df[available_features], target)
        plot_loocv_scatter(df_c["judet"], y, loo_preds, target)

    # Save outputs
    if all_preds:
        preds_out = pd.concat(all_preds, ignore_index=True)
        OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
        preds_out.to_parquet(OUT_PREDS, index=False)
        print(f"\nSaved XGBoost predictions -> {OUT_PREDS}")

    if all_shap:
        shap_out = pd.concat(all_shap, ignore_index=True)
        OUT_SHAP.parent.mkdir(parents=True, exist_ok=True)
        shap_out.to_parquet(OUT_SHAP, index=False)
        print(f"Saved SHAP values -> {OUT_SHAP}")

    APP_PREDS.parent.mkdir(parents=True, exist_ok=True)
    with open(APP_PREDS, "w", encoding="utf-8") as f:
        json.dump(app_preds_list, f, ensure_ascii=False, indent=2)
    print(f"App predictions -> {APP_PREDS}")

    with open(APP_SHAP, "w", encoding="utf-8") as f:
        json.dump(app_shap_dict, f, ensure_ascii=False, indent=2)
    print(f"App SHAP -> {APP_SHAP}")

    print(f"Figures -> {FIG_DIR}/")
    print("\nNext step: python scripts/04_model/train_spatial.py")


if __name__ == "__main__":
    main()
