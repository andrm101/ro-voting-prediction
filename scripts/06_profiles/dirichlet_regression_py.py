"""
Task 2 (Python fallback) — Dirichlet Regression via maximum likelihood.

Used when R / DirichletReg is unavailable. Implements the same "common"
parameterisation as Maier (2014):
  Y_i ~ Dirichlet(alpha_i)
  alpha_ik = mu_ik * phi_i
  log(mu_ik / mu_iK) = X_i @ beta_k   (ALR with last category as reference)
  log(phi_i) = gamma_0                  (common precision, no covariates)

Optimisation: L-BFGS-B via scipy.optimize.minimize on the negative log-likelihood.
Standard errors: inverse observed Fisher information (Hessian at MLE).

References:
  Maier MJ (2014). DirichletReg. WU Vienna Research Report 125.
  Aitchison J (1986). The Statistical Analysis of Compositional Data.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln, digamma
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[2]
DATA_IN  = ROOT / "data" / "processed" / "dirichlet_input.csv"
FIG_DIR  = ROOT / "figures"
REP_DIR  = ROOT / "reports"
FIG_DIR.mkdir(exist_ok=True)
REP_DIR.mkdir(exist_ok=True)

np.random.seed(42)

PARTY_COLS = ["t_AUR","t_PSD","t_PNL","t_USR","t_UDMR","t_SOS","t_POT","t_OTHER"]
PREDICTORS = {
    "M1": ["z_grievance","z_modernity","z_magyari","z_deindustrial"],
    "M2": ["z_grievance","z_modernity","z_magyari","z_deindustrial",
           "z_demog","z_romi","z_ltu"],
}
CLUSTER_COLOURS = {
    "Moldova Profunda":  "#e41a1c",
    "Campia Uitata":     "#ff7f00",
    "Insula Capitalei":  "#4daf4a",
    "Centrele Dinamice": "#377eb8",
    "Arcul Identitar":   "#984ea3",
}


# ---------------------------------------------------------------------------
# Dirichlet log-likelihood
# ---------------------------------------------------------------------------
def dirichlet_nll(params: np.ndarray, Y: np.ndarray, X: np.ndarray,
                  K: int, p: int) -> float:
    """
    Negative log-likelihood for common-precision Dirichlet regression.
    params layout: [beta_1 (p,), beta_2 (p,), ..., beta_{K-1} (p,), log_phi (scalar)]
    Reference category is K (last).
    """
    n = Y.shape[0]
    betas   = params[: (K - 1) * p].reshape(K - 1, p)   # (K-1, p)
    log_phi = params[-1]
    phi     = np.exp(log_phi)

    # ALR -> alpha
    eta = X @ betas.T                    # (n, K-1)
    eta_full = np.hstack([eta, np.zeros((n, 1))])  # add reference category = 0
    # softmax -> mu
    eta_max = eta_full.max(axis=1, keepdims=True)
    exp_eta = np.exp(eta_full - eta_max)
    mu = exp_eta / exp_eta.sum(axis=1, keepdims=True)   # (n, K)
    alpha = mu * phi                                     # (n, K)

    # Dirichlet log-likelihood
    ll = (gammaln(alpha.sum(axis=1)) - gammaln(alpha).sum(axis=1)
          + ((alpha - 1) * np.log(np.clip(Y, 1e-10, None))).sum(axis=1))
    return -ll.sum()


def dirichlet_nll_grad(params, Y, X, K, p):
    """Numerical gradient (for robustness; analytic grad is complex)."""
    eps = 1e-6
    g   = np.zeros_like(params)
    f0  = dirichlet_nll(params, Y, X, K, p)
    for i in range(len(params)):
        p2 = params.copy(); p2[i] += eps
        g[i] = (dirichlet_nll(p2, Y, X, K, p) - f0) / eps
    return g


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_dirichlet(Y: np.ndarray, X: np.ndarray) -> dict:
    n, K = Y.shape
    p    = X.shape[1]
    n_params = (K - 1) * p + 1

    x0 = np.zeros(n_params)
    x0[-1] = np.log(10)  # start with phi=10

    result = minimize(
        dirichlet_nll, x0,
        args=(Y, X, K, p),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )

    if not result.success:
        print(f"  [WARN] Optimiser: {result.message}", file=sys.stderr)

    params = result.x
    nll    = result.fun

    # Hessian-based SEs (finite-difference)
    try:
        from scipy.optimize import approx_fprime
        eps  = 1e-5
        hess = np.zeros((n_params, n_params))
        g0   = approx_fprime(params, dirichlet_nll, eps, Y, X, K, p)
        for i in range(n_params):
            p2  = params.copy(); p2[i] += eps
            g1  = approx_fprime(p2, dirichlet_nll, eps, Y, X, K, p)
            hess[i] = (g1 - g0) / eps
        hess = (hess + hess.T) / 2
        cov  = np.linalg.pinv(hess)
        se   = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        se = np.full(n_params, np.nan)

    return {"params": params, "nll": nll, "se": se,
            "n": n, "K": K, "p": p, "converged": result.success}


def null_nll(Y: np.ndarray) -> float:
    """NLL of null model (intercept only)."""
    x0_null = np.hstack([np.zeros(Y.shape[1] - 1), np.log(10)])
    X_null  = np.ones((Y.shape[0], 1))
    return dirichlet_nll(x0_null, Y, X_null, Y.shape[1], 1)


def pseudo_r2(nll_full: float, nll_null: float, n: int) -> dict:
    ll_f = -nll_full; ll_n = -nll_null
    mcf  = 1 - ll_f / ll_n
    cs   = 1 - np.exp(2 * (ll_n - ll_f) / n)
    nag  = cs / (1 - np.exp(2 * ll_n / n))
    return {"McFadden": mcf, "CoxSnell": cs, "Nagelkerke": nag}


def fitted_values(params, Y, X):
    n, K = Y.shape
    p    = X.shape[1]
    betas   = params[:(K-1)*p].reshape(K-1, p)
    log_phi = params[-1]; phi = np.exp(log_phi)
    eta      = X @ betas.T
    eta_full = np.hstack([eta, np.zeros((n,1))])
    eta_max  = eta_full.max(axis=1, keepdims=True)
    exp_eta  = np.exp(eta_full - eta_max)
    mu       = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    return mu   # fitted composition


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def coef_table(fit: dict, pred_names: list[str], party_names: list[str]) -> pd.DataFrame:
    K = fit["K"]; p = fit["p"]
    betas = fit["params"][:(K-1)*p].reshape(K-1, p)
    ses   = fit["se"][:(K-1)*p].reshape(K-1, p)
    rows  = []
    ref   = party_names[-1]
    for k, party in enumerate(party_names[:-1]):
        for j, pred in enumerate(pred_names):
            est = betas[k, j]; se = ses[k, j]
            z   = est / se if se > 0 else np.nan
            pv  = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            sig = ("***" if pv < 0.001 else ("**" if pv < 0.01
                   else ("*" if pv < 0.05 else ("." if pv < 0.10 else ""))))
            rows.append({
                "category": party, "reference": ref,
                "term": pred, "estimate": est, "se": se,
                "z": z, "p": pv, "sig": sig,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("Task 2 — Dirichlet Regression (Python / scipy MLE)")
    print("=" * 65)

    df = pd.read_csv(DATA_IN)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # Standardise predictors
    def z(col): return (df[col] - df[col].mean()) / df[col].std()
    df["z_grievance"]    = z("idx_grievance")
    df["z_modernity"]    = z("idx_modernity")
    df["z_magyari"]      = z("pct_maghiari")
    df["z_deindustrial"] = z("gva_deindustrial_index")
    df["z_demog"]        = z("idx_demographic_pressure")
    df["z_romi"]         = z("pct_romi")
    df["z_ltu"]          = z("long_term_unemployment_rate")

    Y = df[PARTY_COLS].values.astype(float)
    Y = Y / Y.sum(axis=1, keepdims=True)   # ensure closed

    nll_null = null_nll(Y)
    print(f"\nNull NLL: {nll_null:.2f}")

    results = {}
    for model_name, pred_cols in PREDICTORS.items():
        print(f"\nFitting {model_name}: {pred_cols}")
        X = np.column_stack([np.ones(len(df))] +
                            [df[c].values for c in pred_cols])
        fit = fit_dirichlet(Y, X)
        results[model_name] = {"fit": fit, "X": X, "pred_cols": ["intercept"] + pred_cols}
        aic = 2 * fit["nll"] + 2 * len(fit["params"])
        ps  = pseudo_r2(fit["nll"], nll_null, fit["n"])
        print(f"  NLL={fit['nll']:.2f}  AIC={aic:.2f}  "
              f"Nagelkerke R2={ps['Nagelkerke']:.3f}  "
              f"McFadden R2={ps['McFadden']:.3f}  "
              f"converged={fit['converged']}")

    # Coefficient table for M2
    fit_m2   = results["M2"]["fit"]
    pred_m2  = results["M2"]["pred_cols"]
    coefs_m2 = coef_table(fit_m2, pred_m2, PARTY_COLS)
    print("\n=== M2 Significant predictors (p < 0.10) ===")
    sig = coefs_m2[(coefs_m2["p"] < 0.10) & (coefs_m2["term"] != "intercept")]
    print(sig[["category","term","estimate","se","z","p","sig"]]
          .sort_values("p").to_string(index=False))

    coefs_m2.to_csv(REP_DIR / "08a_dirichlet_coefs_py.csv", index=False)

    # RMSE
    X_m2   = results["M2"]["X"]
    fitted = fitted_values(fit_m2["params"], Y, X_m2)
    rmse   = np.sqrt(((Y - fitted) ** 2).mean(axis=0))
    print("\nRMSE by party:")
    for col, r in zip(PARTY_COLS, rmse):
        print(f"  {col:<12s}  {r:.4f}")

    # --- Figure 1: Coefficient forest plot ---
    plot_df = coefs_m2[
        (coefs_m2["term"] != "intercept") &
        (~coefs_m2["category"].isin(["t_OTHER"]))
    ].copy()
    plot_df["ci_lo"] = plot_df["estimate"] - 1.96 * plot_df["se"]
    plot_df["ci_hi"] = plot_df["estimate"] + 1.96 * plot_df["se"]
    plot_df["party"] = plot_df["category"].str.replace("t_", "")
    plot_df["alpha_val"] = (plot_df["p"] < 0.05).astype(float) * 0.6 + 0.4
    term_map = {
        "intercept": "Intercept", "z_grievance": "Grievance",
        "z_modernity": "Modernity", "z_magyari": "% Magyar",
        "z_deindustrial": "Deindustrialisation",
        "z_demog": "Demographic Pressure",
        "z_romi": "% Roma", "z_ltu": "Long-term Unemployment",
    }
    plot_df["term_lbl"] = plot_df["term"].map(term_map).fillna(plot_df["term"])

    parties = [p for p in PARTY_COLS if p != "t_OTHER"]
    n_parties = len(parties)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    for i, party in enumerate(parties):
        ax  = axes[i]
        sub = plot_df[plot_df["category"] == party].copy()
        if sub.empty:
            ax.axis("off"); continue
        sub = sub.sort_values("estimate")
        colours = ["#e41a1c" if p < 0.05 else "#aaaaaa" for p in sub["p"]]
        ax.barh(sub["term_lbl"], sub["estimate"], color=colours, alpha=0.75)
        ax.errorbar(sub["estimate"], sub["term_lbl"],
                    xerr=1.96 * sub["se"],
                    fmt="none", color="#333333", capsize=3, lw=0.8)
        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.set_title(party.replace("t_",""), fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
    if n_parties < 8:
        for j in range(n_parties, 8): axes[j].axis("off")
    fig.suptitle("Dirichlet Regression M2 — Coefficients by Party\n"
                 "Red = p < 0.05  |  Reference category: t_OTHER",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08a_dirichlet_coefs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved -> 08a_dirichlet_coefs.png")

    # --- Figure 2: Fitted vs observed ---
    fig2, axes2 = plt.subplots(2, 4, figsize=(14, 7))
    axes2 = axes2.flatten()
    for i, party_col in enumerate(parties):
        ax = axes2[i]
        j  = PARTY_COLS.index(party_col)
        obs   = Y[:, j] * 100
        fit_v = fitted[:, j] * 100
        c_vals = [CLUSTER_COLOURS.get(c, "#888888") for c in df["cluster_short"]]
        ax.scatter(obs, fit_v, c=c_vals, s=35, alpha=0.85, zorder=3)
        lims = [min(obs.min(), fit_v.min()) - 1, max(obs.max(), fit_v.max()) + 1]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        r2 = np.corrcoef(obs, fit_v)[0,1] ** 2
        ax.set_title(f"{party_col.replace('t_','')}  R²={r2:.2f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Observed (%)", fontsize=7)
        ax.set_ylabel("Fitted (%)", fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(len(parties), 8): axes2[j].axis("off")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=n) for n, c in CLUSTER_COLOURS.items()]
    fig2.legend(handles=handles, loc="lower right", ncol=1, fontsize=7,
                title="Cluster", title_fontsize=8)
    fig2.suptitle("Dirichlet M2: Fitted vs Observed Vote Shares\n"
                  "Coloured by voter-profile cluster", fontsize=11, y=1.01)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "08a_dirichlet_fitted_obs.png",
                 dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved -> 08a_dirichlet_fitted_obs.png")

    # --- Summary ---
    fit_m1 = results["M1"]["fit"]
    ps_m1  = pseudo_r2(fit_m1["nll"], nll_null, fit_m1["n"])
    ps_m2  = pseudo_r2(fit_m2["nll"], nll_null, fit_m2["n"])
    aic_m1 = 2 * fit_m1["nll"] + 2 * len(fit_m1["params"])
    aic_m2 = 2 * fit_m2["nll"] + 2 * len(fit_m2["params"])
    print("\n" + "=" * 65)
    print("TASK 2 COMPLETE — Dirichlet Regression (Python)")
    print("=" * 65)
    print(f"  M1 AIC: {aic_m1:.2f}  |  M2 AIC: {aic_m2:.2f}")
    print(f"  M2 Nagelkerke pseudo-R2: {ps_m2['Nagelkerke']:.3f}")
    print(f"  M2 McFadden  pseudo-R2: {ps_m2['McFadden']:.3f}")
    print(f"  RMSE: " + "  ".join(
          f"{c.replace('t_','')}={r:.4f}" for c,r in zip(PARTY_COLS[:7], rmse[:7])))
    print("  Figures: figures/08a_*.png")
    print("  Coefficients: reports/08a_dirichlet_coefs_py.csv")


if __name__ == "__main__":
    main()
