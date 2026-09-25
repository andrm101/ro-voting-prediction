"""
Conditional Projections (a) — Bayesian Hierarchical AUR Swing Model
====================================================================
Reformulates the AUR 2020→2024 Senate swing OLS as a Bayesian hierarchical
model with cluster-level random intercepts. The OLS analogue (R²=0.585) treated
all 40 counties as exchangeable; here county predictions are shrunk toward their
cluster posterior mean, reducing overfitting to small within-cluster samples.

Model specification:
  Level 1 (county):
    sw_AUR_i ~ Normal(μ_i, σ)
    μ_i = α_cluster[i] + Σ_k β_k · z_x_{ik}

  Level 2 (cluster):
    α_cluster[j] ~ Normal(α, σ_α)

  Priors:
    α            ~ Normal(0, 0.02)       — centred on zero swing nationally
    β_k          ~ Normal(0, 0.02)       — weakly informative, one SD ≈ 2 pp effect
    σ_α          ~ HalfNormal(0.02)      — cluster-level SD in swing space
    σ            ~ HalfNormal(0.02)      — residual SD

MCMC: 4 chains × 4000 draws (2000 warmup); nutpie backend (XLA-compiled NUTS).
Convergence diagnostics: R-hat < 1.01 and ESS_bulk > 400 required.

NOTE (Windows): pm.sample with cores>1 uses multiprocessing spawn, which
re-imports the module in each worker. All execution-level code must be inside
`if __name__ == "__main__":` to prevent recursive spawning.

Outputs:
  figures/14a_bayes_posterior_coefs.png   — forest plot of β posteriors
  figures/14a_bayes_pred_intervals.png    — county-level 90% predictive intervals
  reports/14a_bayes_summary.csv           — posterior summary (mean, sd, 5%, 95%)
  reports/14a_bayes_predictions.csv       — county-level posterior predictions
"""
from __future__ import annotations
import warnings
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pymc as pm
import arviz as az

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"

CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
]
CONTROL_LABELS = [
    "Grievance", "Modernity", "Dem. pressure", "Agri. employ.",
    "Magyar %", "Roma %", "Deindustrial.", "LT unemployment",
]
CLUSTER_ORDER = [
    "Moldova Profunda", "Campia Uitata", "Insula Capitalei",
    "Centrele Dinamice", "Arcul Identitar",
]
CLUSTER_COLS = {
    "Moldova Profunda":   "#e41a1c",
    "Campia Uitata":      "#ff7f00",
    "Insula Capitalei":   "#4daf4a",
    "Centrele Dinamice":  "#377eb8",
    "Arcul Identitar":    "#984ea3",
}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


def load_data():
    feat   = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
    clust  = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
    feat   = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")
    series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

    feat["judet_norm"] = feat["judet"].apply(_norm)

    s2020 = series[(series.election == "parlamentare_2020") & (series.chamber == "S")].copy()
    s2024 = series[(series.election == "parlamentare_2024") & (series.chamber == "S")].copy()
    s2020["judet_norm"] = s2020["judet"].apply(_norm)
    s2024["judet_norm"] = s2024["judet"].apply(_norm)
    s2020 = s2020.set_index("judet_norm")[["t_AUR"]].rename(columns={"t_AUR": "aur_2020"})
    s2024 = s2024.set_index("judet_norm")[["t_AUR"]].rename(columns={"t_AUR": "aur_2024"})
    feat = feat.join(s2020, on="judet_norm").join(s2024, on="judet_norm")
    feat["sw_AUR"] = feat["aur_2024"] - feat["aur_2020"]

    df = feat.dropna(subset=["sw_AUR"] + CONTROLS).copy()
    for col in CONTROLS:
        df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    return df


if __name__ == "__main__":
    df = load_data()

    cluster_idx, cluster_labels = pd.factorize(df["cluster_short"], sort=True)
    n_clusters = len(cluster_labels)
    X = df[[f"z_{c}" for c in CONTROLS]].values
    y = df["sw_AUR"].values
    n, k = X.shape

    print("=" * 65)
    print("Conditional Projections (a) — Bayesian Hierarchical AUR Swing")
    print("=" * 65)
    print(f"\n  n={n}  k={k} predictors  n_clusters={n_clusters}")

    with pm.Model() as hierarchical_model:
        # Hyperpriors
        alpha_mu  = pm.Normal("alpha_mu",  mu=0.0, sigma=0.02)
        sigma_a   = pm.HalfNormal("sigma_a", sigma=0.02)
        # Cluster random intercepts (non-centred parameterisation)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_clusters)
        alpha_j   = pm.Deterministic("alpha_j", alpha_mu + alpha_raw * sigma_a)
        # Fixed effects
        beta      = pm.Normal("beta", mu=0, sigma=0.02, shape=k)
        # Residual SD
        sigma     = pm.HalfNormal("sigma", sigma=0.02)
        # Linear predictor
        mu        = alpha_j[cluster_idx] + pm.math.dot(X, beta)
        # Likelihood
        obs       = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        # Posterior predictive (in-sample)
        y_hat     = pm.Normal("y_hat", mu=mu, sigma=sigma)

        trace = pm.sample(
            4000, tune=2000, chains=4, cores=4, target_accept=0.9,
            random_seed=42, progressbar=True,
            nuts_sampler="nutpie",
        )
        ppc = pm.sample_posterior_predictive(
            trace, var_names=["y_hat"], random_seed=42, progressbar=False
        )

    # Convergence diagnostics
    rhat   = az.rhat(trace)
    ess    = az.ess(trace)
    beta_rhat = rhat["beta"].values
    beta_ess  = ess["beta"].values.astype(int)
    max_rhat  = float(np.max(beta_rhat))
    min_ess   = int(np.min(beta_ess))
    print(f"\n  Convergence: max R-hat (β) = {max_rhat:.4f}  min ESS (β) = {min_ess}")
    if max_rhat > 1.01:
        print("  WARNING: R-hat > 1.01 — chains may not have converged")
    else:
        print("  Chains converged (R-hat < 1.01)")

    # Extract β posteriors
    beta_post = trace.posterior["beta"].values.reshape(-1, k)
    beta_mean = beta_post.mean(axis=0)
    beta_sd   = beta_post.std(axis=0)
    beta_q05  = np.percentile(beta_post, 5,  axis=0)
    beta_q95  = np.percentile(beta_post, 95, axis=0)

    print(f"\n  Posterior β (mean ± sd)  [90% CI]:")
    for i, lbl in enumerate(CONTROL_LABELS):
        sig = "*" if (beta_q05[i] > 0 or beta_q95[i] < 0) else " "
        print(f"    {lbl:<22} {beta_mean[i]:+.4f} ± {beta_sd[i]:.4f}  "
              f"[{beta_q05[i]:+.4f}, {beta_q95[i]:+.4f}] {sig}")

    # Posterior predictive intervals per county
    y_hat_post = ppc.posterior_predictive["y_hat"].values.reshape(-1, n)
    pp_mean = y_hat_post.mean(axis=0)
    pp_q05  = np.percentile(y_hat_post, 5,  axis=0)
    pp_q95  = np.percentile(y_hat_post, 95, axis=0)
    pp_q25  = np.percentile(y_hat_post, 25, axis=0)
    pp_q75  = np.percentile(y_hat_post, 75, axis=0)

    # Save posterior summary
    summary_rows = []
    for i, lbl in enumerate(CONTROL_LABELS):
        summary_rows.append({
            "parameter": f"beta_{CONTROLS[i]}",
            "mean": round(beta_mean[i], 5),
            "sd":   round(beta_sd[i],   5),
            "q05":  round(beta_q05[i],  5),
            "q95":  round(beta_q95[i],  5),
            "rhat": round(float(beta_rhat[i]), 4),
            "ess":  int(beta_ess[i]),
        })
    alpha_j_post = (trace.posterior["alpha_mu"].values.flatten()[:, None]
                    + trace.posterior["alpha_raw"].values.reshape(-1, n_clusters)
                    * trace.posterior["sigma_a"].values.flatten()[:, None])
    for j, cl in enumerate(cluster_labels):
        summary_rows.append({
            "parameter": f"alpha_{cl}",
            "mean": round(alpha_j_post[:, j].mean(), 5),
            "sd":   round(alpha_j_post[:, j].std(),  5),
            "q05":  round(np.percentile(alpha_j_post[:, j], 5),  5),
            "q95":  round(np.percentile(alpha_j_post[:, j], 95), 5),
            "rhat": None, "ess": None,
        })
    pd.DataFrame(summary_rows).to_csv(REP_DIR / "14a_bayes_summary.csv", index=False)

    # Save county predictions
    pred_df = df[["judet", "cluster_short", "sw_AUR"]].copy()
    pred_df["pp_mean"] = pp_mean
    pred_df["pp_q05"]  = pp_q05
    pred_df["pp_q25"]  = pp_q25
    pred_df["pp_q75"]  = pp_q75
    pred_df["pp_q95"]  = pp_q95
    pred_df["resid"]   = df["sw_AUR"].values - pp_mean
    pred_df.to_csv(REP_DIR / "14a_bayes_predictions.csv", index=False, float_format="%.5f")

    # ── Figure 1: β forest plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(k)
    ax.barh(y_pos, beta_mean * 100, xerr=None,
            color=["#d62728" if (beta_q05[i] > 0 or beta_q95[i] < 0) else "#aaaaaa"
                   for i in range(k)],
            height=0.5, alpha=0.8)
    ax.errorbar(beta_mean * 100, y_pos,
                xerr=[((beta_mean - beta_q05) * 100), ((beta_q95 - beta_mean) * 100)],
                fmt="none", color="black", lw=1.2, capsize=3)
    ax.axvline(0, lw=0.9, ls="--", color="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CONTROL_LABELS, fontsize=9)
    ax.set_xlabel("Posterior β × 100  (effect on AUR swing, pp)", fontsize=9)
    ax.set_title("Bayesian hierarchical AUR swing model\n"
                 "Fixed-effect posteriors with 90% credible intervals\n"
                 "Red = 90% CI excludes zero", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14a_bayes_posterior_coefs.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 2: county-level 90% predictive intervals ─────────────────────
    order = np.argsort(pp_mean)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(n)
    ax.fill_between(x_pos,
                    pp_q05[order] * 100, pp_q95[order] * 100,
                    alpha=0.2, color="#4393c3", label="90% predictive interval")
    ax.fill_between(x_pos,
                    pp_q25[order] * 100, pp_q75[order] * 100,
                    alpha=0.4, color="#4393c3", label="50% predictive interval")
    ax.plot(x_pos, pp_mean[order] * 100, color="#2166ac", lw=1.5, label="Posterior mean")
    for idx_plot, idx_data in enumerate(order):
        c = CLUSTER_COLS.get(df["cluster_short"].iloc[idx_data], "#888888")
        ax.scatter(idx_plot, df["sw_AUR"].iloc[idx_data] * 100,
                   color=c, s=30, zorder=5, edgecolors="white", lw=0.3)
    ax.axhline(0, lw=0.8, ls=":", color="grey")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df["judet"].iloc[order].tolist(), rotation=90, fontsize=5.5)
    ax.set_ylabel("AUR 2020→2024 swing (pp)", fontsize=9)
    ax.set_title("Bayesian hierarchical model: county-level 90% posterior predictive intervals\n"
                 "Dots = observed swing, coloured by cluster", fontsize=9)
    handles_ci = [
        plt.Rectangle((0, 0), 1, 1, fc="#4393c3", alpha=0.4, label="50% PI"),
        plt.Rectangle((0, 0), 1, 1, fc="#4393c3", alpha=0.2, label="90% PI"),
        plt.Line2D([0], [0], color="#2166ac", lw=1.5, label="Posterior mean"),
    ]
    handles_cl = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
    ax.legend(handles=handles_ci + handles_cl, fontsize=6.5, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14a_bayes_pred_intervals.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved -> figures/14a_bayes_posterior_coefs.png")
    print(f"  Saved -> figures/14a_bayes_pred_intervals.png")
    print(f"  Saved -> reports/14a_bayes_summary.csv")
    print(f"  Saved -> reports/14a_bayes_predictions.csv")
    print("\n" + "=" * 65)
    print("CONDITIONAL PROJECTIONS (a) COMPLETE")
    print("=" * 65)
