"""
Final evaluation: model comparison, key figures, and app JSON exports.

Loads all prediction outputs and produces:
  1. Model comparison table (Elastic Net vs XGBoost per target)
  2. Georgescu R1 2024 vs Simion R1 2025 correlation scatter (transfer hypothesis)
  3. Swing map: anti-establishment change 2020→2024 by county (bar chart)
  4. Actual vs predicted bar chart per party (both models side-by-side)
  5. Cross-model residual comparison scatter
  6. Summary statistics table saved as CSV for R Markdown import

All figures at 300 DPI. Source attribution on every figure.

Outputs:
  figures/model_comparison.png
  figures/georgescu_simion_transfer.png
  figures/swing_2020_2024.png
  figures/actual_vs_predicted_<target>.png
  data/processed/model_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from constants import PARLIAMENTARY_MODEL_PARTIES

FEAT_PATH    = ROOT / "data" / "processed" / "features_2022_judet.parquet"
EN_PATH      = ROOT / "data" / "processed" / "predictions_elastic_net.parquet"
XGB_PATH     = ROOT / "data" / "processed" / "predictions_xgboost.parquet"
STACKED_PATH = ROOT / "data" / "processed" / "predictions_stacked.parquet"
BOOT_PATH    = ROOT / "data" / "processed" / "bootstrap_r2.parquet"
DIAG_PATH    = ROOT / "data" / "processed" / "spatial_diagnostics.parquet"
OUT_TABLE    = ROOT / "data" / "processed" / "model_comparison.csv"
FIG_DIR      = ROOT / "figures"

SOURCE_NOTE = "Source: Eurostat 2022, INS Census 2021, roaep.ro 2024/2025"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    a, p = actual[mask], predicted[mask]
    if len(a) < 3:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": len(a)}
    residuals = a - p
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return {
        "r2":   float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mae":  float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "n":    int(len(a)),
    }


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    targets = comparison["target"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, len(targets) * 0.5)))
    metrics = ["r2", "mae", "rmse"]
    titles  = ["R² (LOOCV)", "MAE", "RMSE"]

    for ax, metric, title in zip(axes, metrics, titles):
        for model, color, marker in [("ElasticNet", "#2196F3", "o"),
                                      ("XGBoost",    "#FF5722", "s")]:
            sub = comparison[comparison["model"] == model].copy()
            sub = sub.sort_values("target")
            ax.plot(sub[metric], sub["target"], marker=marker, label=model,
                    color=color, markersize=7, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(metric.upper())
        if metric == "r2":
            ax.axvline(0, color="grey", lw=0.5, ls="--")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Model Performance Comparison — LOOCV (N=42 counties)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, -0.02, SOURCE_NOTE, ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/model_comparison.png")


def plot_georgescu_simion_transfer(df: pd.DataFrame) -> None:
    """Scatter: Georgescu R1 2024 share vs Simion R1 2025 share by county."""
    if "georgescu_r1_2024" not in df.columns or "simion_r1_2025" not in df.columns:
        print("  [SKIP] Georgescu/Simion columns not available in feature matrix.")
        return

    sub = df[["judet", "georgescu_r1_2024", "simion_r1_2025"]].dropna()
    if len(sub) < 5:
        print("  [SKIP] Too few counties with both Georgescu and Simion data.")
        return

    x, y = sub["georgescu_r1_2024"].values, sub["simion_r1_2025"].values
    r, p  = stats.pearsonr(x, y)
    slope, intercept, *_ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, alpha=0.8, edgecolors="black", lw=0.5, zorder=3, s=60)

    # Label all counties (small font — key for interpretation)
    for _, row in sub.iterrows():
        ax.annotate(row["judet"], (row["georgescu_r1_2024"], row["simion_r1_2025"]),
                    fontsize=6, ha="left", va="bottom", alpha=0.8)

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5,
            label=f"Linear fit  r={r:.3f}, p={p:.4f}")
    ax.plot([0, max(x.max(), y.max())],
            [0, max(x.max(), y.max())], "k--", lw=0.8, alpha=0.4,
            label="1:1 line (perfect transfer)")

    ax.set_xlabel("Georgescu vote share — Presidential R1, Nov 2024 (annulled)")
    ax.set_ylabel("Simion vote share — Presidential R1, May 2025 (redo)")
    ax.set_title("Georgescu → Simion Vote Transfer by County\n"
                 "Tests geographic stability of the nationalist vote after annulment",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.text(0.5, -0.02, SOURCE_NOTE, ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "georgescu_simion_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/georgescu_simion_transfer.png  (r={r:.3f}, p={p:.4f})")


def plot_swing_chart(df: pd.DataFrame) -> None:
    """Horizontal bar chart of anti-establishment swing 2020→2024, sorted."""
    if "swing_anti_estab_2020_2024" not in df.columns:
        print("  [SKIP] Swing column not available.")
        return

    sub = df[["judet", "swing_anti_estab_2020_2024"]].dropna()
    sub = sub.sort_values("swing_anti_estab_2020_2024")

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in sub["swing_anti_estab_2020_2024"]]
    fig, ax = plt.subplots(figsize=(9, max(6, len(sub) * 0.32)))
    ax.barh(sub["judet"], sub["swing_anti_estab_2020_2024"] * 100, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Change in anti-establishment vote share (pp), 2020→2024")
    ax.set_title("Anti-Establishment Vote Swing by County — Parliamentary 2020→2024\n"
                 "(AUR + SOS + POT combined share)", fontsize=11)
    fig.text(0.5, -0.01, SOURCE_NOTE, ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "swing_2020_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/swing_2020_2024.png")


def plot_actual_vs_predicted(
    en_preds: pd.DataFrame, xgb_preds: pd.DataFrame, target: str
) -> None:
    en  = en_preds[en_preds["target"] == target].sort_values("siruta")
    xgb = xgb_preds[xgb_preds["target"] == target].sort_values("siruta")

    if en.empty and xgb.empty:
        return

    # Use whichever has data as the base
    base = en if not en.empty else xgb
    counties = base["judet"].values
    actual   = base["actual"].values
    x = np.arange(len(counties))

    fig, ax = plt.subplots(figsize=(max(10, len(counties) * 0.25), 5))
    ax.bar(x - 0.25, actual * 100,    0.25, label="Actual",      color="#555555", alpha=0.8)
    if not en.empty:
        ax.bar(x,        en["predicted"].values * 100,  0.25,
               label="Elastic Net", color="#2196F3", alpha=0.8)
    if not xgb.empty:
        ax.bar(x + 0.25, xgb["predicted"].values * 100, 0.25,
               label="XGBoost",     color="#FF5722", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(counties, rotation=90, fontsize=7)
    ax.set_ylabel("Vote share (%)")
    ax.set_title(f"Actual vs Predicted — {target} (2024 Parliamentary)\n"
                 f"LOOCV predictions, N=42 counties")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.text(0.5, -0.02, SOURCE_NOTE, ha="center", fontsize=7, style="italic")
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"actual_vs_predicted_{target}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/actual_vs_predicted_{target}.png")


def build_comparison_table(
    en_preds: pd.DataFrame, xgb_preds: pd.DataFrame,
    stacked_preds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for preds, model_name in [(en_preds, "ElasticNet"), (xgb_preds, "XGBoost"),
                               (stacked_preds, "Stacked")]:
        if preds.empty:
            continue
        for target in preds["target"].unique():
            sub = preds[preds["target"] == target]
            m = _metrics(sub["actual"].values, sub["predicted"].values)
            rows.append({"model": model_name, "target": target, **m})
    return pd.DataFrame(rows).sort_values(["target", "model"])


def main() -> None:
    df = pd.read_parquet(FEAT_PATH) if FEAT_PATH.exists() else pd.DataFrame()

    en_preds      = pd.read_parquet(EN_PATH)      if EN_PATH.exists()      else pd.DataFrame()
    xgb_preds     = pd.read_parquet(XGB_PATH)     if XGB_PATH.exists()     else pd.DataFrame()
    stacked_preds = pd.read_parquet(STACKED_PATH) if STACKED_PATH.exists() else pd.DataFrame()

    if en_preds.empty and xgb_preds.empty:
        print("No prediction outputs found. Run 04_model scripts first.", file=sys.stderr)
        sys.exit(1)

    print("Building model comparison table ...")
    comparison = build_comparison_table(en_preds, xgb_preds, stacked_preds)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_TABLE, index=False)
    print(f"  -> {OUT_TABLE}")
    print("\n" + comparison.to_string(index=False))

    print("\nGenerating figures ...")
    FIG_DIR.mkdir(exist_ok=True)

    if not comparison.empty:
        plot_model_comparison(comparison)

    if not df.empty:
        plot_georgescu_simion_transfer(df)
        plot_swing_chart(df)

    targets = [f"t_{p}" for p in PARLIAMENTARY_MODEL_PARTIES]
    for target in targets:
        plot_actual_vs_predicted(en_preds, xgb_preds, target)

    # Bootstrap CI summary if available
    if BOOT_PATH.exists():
        boot = pd.read_parquet(BOOT_PATH)
        print("\nEN ElasticNet Bootstrap R2 stability (1000 resamples):")
        print(boot[["target", "r2_mean", "ci_low_95", "ci_high_95"]].to_string(index=False))

    # Spatial diagnostics summary if available
    if DIAG_PATH.exists():
        diag = pd.read_parquet(DIAG_PATH)
        print("\nSpatial diagnostics summary:")
        cols = [c for c in ["target", "ols_r2", "moran_I", "moran_p",
                             "spatial_model_fitted", "rho"] if c in diag.columns]
        print(diag[cols].to_string(index=False))

    print(f"\n[EVALUATE COMPLETE] All figures in {FIG_DIR}/")
    print("Model comparison table -> data/processed/model_comparison.csv")
    print("\nFor the academic report: run reports/main.Rmd")


if __name__ == "__main__":
    main()
