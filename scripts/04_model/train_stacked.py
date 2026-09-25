"""
Stacked ensemble: ElasticNet + XGBoost base learners, Ridge meta-learner.

Motivation: Given N=42 and the distinct modelling philosophies of ElasticNet
(sparse linear) and XGBoost (non-linear interactions), a stacking ensemble can
capture structure that neither alone detects. Meta-learner is kept simple
(Ridge) to avoid overfitting at the meta-level.

Architecture:
  Level 0 (base learners):
    - ElasticNet (StandardScaler + ElasticNetCV, best alpha/l1 from full data)
    - XGBoost    (conservative hyperparameters as in train_gbm.py)
  Level 1 (meta-learner):
    - Ridge with LOOCV-selected alpha
  Evaluation: nested LOOCV — outer loop leaves one county out, inner loop
    fits both base learners on N-1 counties and gets their leave-one-out
    predictions on that held-out county.

Outputs:
  data/processed/predictions_stacked.parquet
  figures/loocv_scatter_stacked_<target>.png

Reproducibility: np.random.seed(42)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import PARLIAMENTARY_MODEL_PARTIES

FEAT_PATH  = ROOT / "data" / "processed" / "features_2022_judet.parquet"
OUT_PREDS  = ROOT / "data" / "processed" / "predictions_stacked.parquet"
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
    "emigrants_temporary_rate", "hospitals_per_100k",
    "convictions_per_100k", "avg_household_size",
    "ethno_linguistic_frag", "religious_frag",
    "nationalist_consolidation_ratio",
]

XGB_PARAMS = dict(
    n_estimators=150, max_depth=2, learning_rate=0.05,
    min_child_weight=4, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.2, reg_lambda=1.5,
    objective="reg:squarederror", random_state=42, verbosity=0,
)

OLS_ONLY_TARGETS = {"t_UDMR"}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residuals = actual - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return {
        "r2_loocv": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mae":  float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
    }


def _get_en_params(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ElasticNetCV on full data to select alpha and l1_ratio."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        en = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5, max_iter=20_000, random_state=42,
        )
        en.fit(Xs, y)
    return en.alpha_, en.l1_ratio_


def stacked_loocv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Nested LOOCV stacking:
      For each hold-out county i:
        1. Fit EN and XGB on the remaining 41 counties.
        2. Predict county i from each base learner.
        3. Stack the two predictions (meta-features) and use a simple
           weighted average (Ridge) trained on the in-fold base predictions.
           Because there is only one test point in LOOCV, the meta-learner
           cannot be re-trained on 1 test point; instead we train it on the
           41 in-fold training predictions via inner LOOCV.
    Returns array of stacked LOOCV predictions.
    """
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)

    # Get EN hyperparameters from full data (for stability across folds)
    alpha, l1_ratio = _get_en_params(X, y)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Base learner 1: ElasticNet
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            en.fit(X_tr_s, y_tr)
        en_pred_test = en.predict(X_te_s)[0]

        # Base learner 2: XGBoost
        xgb = XGBRegressor(**XGB_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb.fit(X_tr_s, y_tr)
        xgb_pred_test = xgb.predict(X_te_s)[0]

        # Meta-learner: trained on inner LOOCV predictions from training fold
        loo_inner = LeaveOneOut()
        en_tr_preds  = np.full(len(y_tr), np.nan)
        xgb_tr_preds = np.full(len(y_tr), np.nan)

        for i_tr, i_te in loo_inner.split(X_tr_s):
            sc2 = StandardScaler()
            X2_tr = sc2.fit_transform(X_tr_s[i_tr])
            X2_te = sc2.transform(X_tr_s[i_te])

            en2 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20_000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                en2.fit(X2_tr, y_tr[i_tr])
            en_tr_preds[i_te] = en2.predict(X2_te)

            xgb2 = XGBRegressor(**XGB_PARAMS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xgb2.fit(X2_tr, y_tr[i_tr])
            xgb_tr_preds[i_te] = xgb2.predict(X2_te)

        # Meta-features: stack base predictions column-wise
        meta_X_tr = np.column_stack([en_tr_preds, xgb_tr_preds])
        meta_X_te = np.array([[en_pred_test, xgb_pred_test]])

        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta.fit(meta_X_tr, y_tr)
        preds[test_idx] = meta.predict(meta_X_te)[0]

    return preds


def plot_loocv_scatter(
    judet: pd.Series, actual: np.ndarray, predicted: np.ndarray, target: str
) -> None:
    m = _metrics(actual, predicted)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)
    residuals = actual - predicted
    threshold = np.percentile(np.abs(residuals), 80)
    for a, p, j in zip(actual, predicted, judet):
        if abs(a - p) > threshold:
            ax.annotate(j, (a, p), fontsize=7, ha="left", va="bottom")
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Perfect prediction")
    ax.set_xlabel(f"Actual {target} (2024 parliamentary)")
    ax.set_ylabel(f"Stacked Ensemble LOOCV Predicted {target}")
    ax.set_title(f"Stacked Ensemble LOOCV -- {target}\n"
                 f"R2={m['r2_loocv']:.3f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024",
             ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"loocv_scatter_stacked_{target}.png", dpi=300, bbox_inches="tight")
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

    for target in targets:
        if target not in df_c.columns or df_c[target].isna().all():
            print(f"  [SKIP] {target} not available.")
            continue
        if target in OLS_ONLY_TARGETS:
            print(f"  [SKIP] {target}: OLS-only target, stacking not applied.")
            continue

        y = df_c[target].values
        print(f"\n-- {target} --")

        loo_preds = stacked_loocv(X, y)
        m = _metrics(y, loo_preds)
        print(f"  Stacked LOOCV  R2={m['r2_loocv']:.3f}  "
              f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

        preds_df = pd.DataFrame({
            "siruta":    df_c["siruta"].values,
            "judet":     df_c["judet"].values,
            "target":    target,
            "actual":    y,
            "predicted": loo_preds,
            "residual":  y - loo_preds,
        })
        all_preds.append(preds_df)
        plot_loocv_scatter(df_c["judet"], y, loo_preds, target)

    if all_preds:
        preds_out = pd.concat(all_preds, ignore_index=True)
        OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)
        preds_out.to_parquet(OUT_PREDS, index=False)
        print(f"\nSaved stacked predictions -> {OUT_PREDS}")

        print("\nModel comparison (stacked LOOCV R2):")
        summary = preds_out.groupby("target").apply(
            lambda g: _metrics(g["actual"].values, g["predicted"].values)["r2_loocv"],
            include_groups=False,
        ).rename("r2_stacked")
        print(summary.to_string())

    print(f"Figures -> {FIG_DIR}/")
    print("\nNext step: python scripts/05_evaluate/evaluate.py")


if __name__ == "__main__":
    main()
