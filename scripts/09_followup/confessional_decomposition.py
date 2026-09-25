"""
Confessional Decomposition — Section 4.7
=========================================
Builds a 7-denomination confessional predictor set and refits the Dirichlet M1
replacing pct_maghiari with the expanded confessional set.

Most denomination shares are already in features_2022_judet.parquet.
The only new variable is pct_no_religion (secular bloc: Fara religie + Ateu +
Agnostic + Informatii nedisponibila), extracted from INS RPL2021 Tab 2.5.2.

Denominations used
------------------
  pct_orthodox       — existing in parquet
  pct_roman_catholic — existing in parquet
  pct_reformed       — existing in parquet
  pct_pentecostal    — existing in parquet
  pct_greco_catholic — existing in parquet (col name pct_greco_catholic)
  pct_unitarian      — existing in parquet
  pct_no_religion    — NEW: cols 21+22+23+24 / total from Tab 2.5.2

Hypothesis ladder
-----------------
  H1: UDMR ~ pct_reformed (+) and pct_roman_catholic (+)
  H2: AUR/SOS ~ pct_pentecostal (+)
  H3: USR ~ pct_no_religion (+)

Outputs
-------
  data/processed/religion_judet_2021.csv  — 7-denomination county table
  figures/15a_confessional_heatmap.png    — 7×8 coefficient heatmap
  reports/15a_confessional_coefs.csv      — full coefs with BH-FDR
  reports/15a_confessional_lr_test.csv    — LR test vs original M1
"""
from __future__ import annotations
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2, rankdata

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ins"
PRO_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"

# ── Column indices in Tab 2.5.2 for secular bloc ─────────────────────────────
COL_TOTAL    = 1
SECULAR_COLS = [21, 22, 23, 24]   # Fara religie, Ateu, Agnostic, Nedisponibila

JUDET_NAMES_FILE = {
    "ALBA", "ARAD", "ARGES", "BACAU", "BIHOR", "BISTRITA-NASAUD",
    "BOTOSANI", "BRAILA", "BRASOV", "BUZAU", "CALARASI", "CARAS-SEVERIN",
    "CLUJ", "CONSTANTA", "COVASNA", "DAMBOVITA", "DOLJ", "GALATI",
    "GIURGIU", "GORJ", "HARGHITA", "HUNEDOARA", "IALOMITA", "IASI",
    "ILFOV", "MARAMURES", "MEHEDINTI", "MURES", "NEAMT", "OLT",
    "PRAHOVA", "SALAJ", "SATU MARE", "SIBIU", "SUCEAVA", "TELEORMAN",
    "TIMIS", "TULCEA", "VALCEA", "VASLUI", "VRANCEA",
    "MUNICIPIUL BUCURESTI",
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

def _to_num(x) -> float:
    if pd.isna(x) or str(x).strip() in {"-", "*", ""}:
        return 0.0
    try:
        return float(str(x).replace(" ", "").replace(",", "."))
    except ValueError:
        return 0.0

# ── Extract pct_no_religion from RPL2021 Tab 2.5.2 ───────────────────────────
xl  = pd.ExcelFile(RAW_DIR / "ins_etnie_religie_judet_2021.csv.xlsx")
raw = xl.parse("Tab 2.5.2", header=None)

secular_records = []
for _, row in raw.iterrows():
    val  = str(row[0]).strip() if pd.notna(row[0]) else ""
    norm = _norm(val)
    if norm not in JUDET_NAMES_FILE:
        continue
    total   = _to_num(row[COL_TOTAL])
    if total == 0:
        continue
    secular = sum(_to_num(row[c]) for c in SECULAR_COLS)
    judet_norm = norm  # keep as-is; feat also has MUNICIPIUL BUCURESTI
    secular_records.append({
        "judet_norm":    judet_norm,
        "pct_no_religion": secular / total,
    })

secular_df = pd.DataFrame(secular_records)
print(f"Extracted pct_no_religion for {len(secular_df)} counties")

# ── Load features + clusters ──────────────────────────────────────────────────
feat  = pd.read_parquet(PRO_DIR / "features_2022_judet.parquet")
clust = pd.read_csv(PRO_DIR / "cluster_assignments_final.csv")
feat  = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")
feat["judet_norm"] = feat["judet"].apply(_norm)

feat = feat.merge(secular_df, on="judet_norm", how="left")
print(f"Missing pct_no_religion after merge: {feat['pct_no_religion'].isna().sum()}")

# ── Electoral targets — party shares already in features parquet ──────────────
# Compute t_Other residually so all 42 counties are retained
CORE_PARTIES = ["t_AUR", "t_PSD", "t_PNL", "t_USR", "t_UDMR", "t_SOS", "t_POT"]
feat["t_OTHER"] = 1.0 - feat[CORE_PARTIES].sum(axis=1)
feat["t_OTHER"] = feat["t_OTHER"].clip(lower=1e-6)

PARTIES      = CORE_PARTIES + ["t_OTHER"]
PARTY_LABELS = ["AUR", "PSD", "PNL", "USR", "UDMR", "SOS", "POT", "Other"]
REF_PARTY    = "t_OTHER"
ALR_TARGETS  = CORE_PARTIES
PARTY_LABELS_ALR = PARTY_LABELS[:-1]

# ── Denomination columns (all already in parquet except pct_no_religion) ─────
# Orthodox is the near-universal majority — included as a predictor but its
# z-score absorbs variation in the 'non-Orthodox minority present' signal.
# All 7 denominations are included; Orthodox z-score is permitted to enter
# the model. The denomination shares do NOT sum exactly to 1 because the
# census "Alta religie" (col 20) and "Nedisponibila" (col 24) fractions are
# captured in pct_no_religion, so the constraint is approximate, not exact.
DENOM_COLS = [
    "pct_orthodox", "pct_roman_catholic", "pct_reformed",
    "pct_pentecostal", "pct_greco_catholic", "pct_unitarian",
    "pct_no_religion",
]
DENOM_LABELS = {
    "pct_orthodox":       "Orthodox",
    "pct_roman_catholic": "Roman Catholic",
    "pct_reformed":       "Reformed",
    "pct_pentecostal":    "Pentecostal",
    "pct_greco_catholic": "Greek Catholic",
    "pct_unitarian":      "Unitarian",
    "pct_no_religion":    "No religion / secular",
}

# ── Base controls (same as M1 minus pct_maghiari) ────────────────────────────
BASE_CONTROLS = [
    "idx_grievance", "idx_modernity",
    "pct_employment_agriculture",
    "pct_romi",
    "gva_deindustrial_index",
    "long_term_unemployment_rate",
]
# Original M1 controls (for LR comparison)
ORIG_CONTROLS = BASE_CONTROLS + ["pct_maghiari"]

# Unitarian (and potentially other minority denominations) are NaN where absent
# from the census — treat as 0 share, not missing data
feat[DENOM_COLS] = feat[DENOM_COLS].fillna(0.0)

df = feat.dropna(subset=PARTIES + DENOM_COLS + ORIG_CONTROLS).copy()
print(f"Analytic sample: n={len(df)}")

for col in BASE_CONTROLS + ORIG_CONTROLS + DENOM_COLS:
    if f"z_{col}" not in df.columns:
        df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

DENOM_Z = [f"z_{d}" for d in DENOM_COLS]
BASE_Z  = [f"z_{b}" for b in BASE_CONTROLS]
ORIG_Z  = [f"z_{b}" for b in ORIG_CONTROLS]
CONF_PREDICTORS = BASE_Z + DENOM_Z

# ── Fit confessional and original models for each ALR outcome ─────────────────
print("\n" + "=" * 70)
print("Confessional Decomposition — expanded Dirichlet (ALR, OLS/HC3)")
print("=" * 70)

results_rows = []
loglik_conf  = 0.0
loglik_orig  = 0.0

EPS = 1e-4   # floor for tiny shares before log to avoid -inf
for party_col, party_label in zip(ALR_TARGETS, PARTY_LABELS_ALR):
    y_alr = np.log(
        df[party_col].clip(lower=EPS) / df[REF_PARTY].clip(lower=EPS)
    )

    X_conf = sm.add_constant(df[CONF_PREDICTORS].values)
    X_orig = sm.add_constant(df[ORIG_Z].values)

    mod_c = sm.OLS(y_alr, X_conf).fit(cov_type="HC3")
    mod_o = sm.OLS(y_alr, X_orig).fit(cov_type="HC3")

    loglik_conf += mod_c.llf
    loglik_orig += mod_o.llf

    for j, pred in enumerate(CONF_PREDICTORS):
        results_rows.append({
            "party":     party_label,
            "predictor": pred.replace("z_", ""),
            "estimate":  mod_c.params.iloc[j + 1],
            "se":        mod_c.bse.iloc[j + 1],
            "z":         mod_c.tvalues.iloc[j + 1],
            "p":         mod_c.pvalues.iloc[j + 1],
        })

    print(f"  {party_label:<6}  R²={mod_c.rsquared:.3f}  "
          f"adj-R²={mod_c.rsquared_adj:.3f}")

coef_df = pd.DataFrame(results_rows)

# BH-FDR correction
p_all   = coef_df["p"].values
n_tests = len(p_all)
rank_   = rankdata(p_all, method="ordinal")
q_bh    = np.minimum(p_all * n_tests / rank_, 1.0)
order   = np.argsort(p_all)
q_s     = q_bh[order]
for i in range(len(q_s) - 2, -1, -1):
    q_s[i] = min(q_s[i], q_s[i + 1])
q_bh[order] = q_s
coef_df["q_bh"] = q_bh
coef_df["sig"]  = coef_df["q_bh"].apply(
    lambda q: "***" if q < 0.001 else ("**" if q < 0.01 else
              ("*" if q < 0.05 else ("." if q < 0.10 else "")))
)

# LR test
n_extra  = len(DENOM_Z)
n_eq     = len(ALR_TARGETS)
lr_stat  = 2 * (loglik_conf - loglik_orig)
lr_df    = n_extra * n_eq
lr_p     = 1 - chi2.cdf(lr_stat, df=lr_df)

print(f"\n  LR test (confessional vs original M1):")
print(f"    ΔlogL = {loglik_conf - loglik_orig:.3f}")
print(f"    χ²({lr_df}) = {lr_stat:.3f}   p = {lr_p:.4f}")

lr_out = pd.DataFrame([{
    "model_a":    "M_conf (7 denominations + 6 base controls)",
    "model_b":    "M1_orig (pct_maghiari + 6 base controls)",
    "delta_logL": round(loglik_conf - loglik_orig, 4),
    "chi2":       round(lr_stat, 4),
    "df":         lr_df,
    "p":          round(lr_p, 5),
}])
lr_out.to_csv(REP_DIR / "15a_confessional_lr_test.csv", index=False)

# Hypothesis ladder summary
print(f"\n  Hypothesis ladder:")
for hyp, pred, parties in [
    ("H1 UDMR ~ Reformed",        "pct_reformed",       ["UDMR"]),
    ("H1 UDMR ~ Roman Catholic",  "pct_roman_catholic", ["UDMR"]),
    ("H2 AUR  ~ Pentecostal",     "pct_pentecostal",    ["AUR"]),
    ("H2 SOS  ~ Pentecostal",     "pct_pentecostal",    ["SOS"]),
    ("H3 USR  ~ No religion",     "pct_no_religion",    ["USR"]),
]:
    sub = coef_df[(coef_df["predictor"] == pred) & coef_df["party"].isin(parties)]
    for _, r in sub.iterrows():
        print(f"    {hyp:<36}  β={r['estimate']:+.4f}  "
              f"p={r['p']:.4f}  q={r['q_bh']:.4f} {r['sig']}")

print(f"\n  BH-FDR significant (q < 0.10) — denomination predictors only:")
sig = coef_df[
    coef_df["predictor"].isin(DENOM_COLS) & (coef_df["q_bh"] < 0.10)
].sort_values("q_bh")
if len(sig):
    for _, r in sig.iterrows():
        print(f"    {r['predictor']:<22} → {r['party']:<6}  "
              f"β={r['estimate']:+.4f}  q={r['q_bh']:.4f} {r['sig']}")
else:
    print("    None")

coef_df.to_csv(REP_DIR / "15a_confessional_coefs.csv", index=False, float_format="%.5f")

# Save 7-denomination religion table
relig_out = df[["judet"] + DENOM_COLS].copy()
relig_out.to_csv(PRO_DIR / "religion_judet_2021.csv", index=False, float_format="%.5f")

# ── Figure: 7 × 7 coefficient heatmap ────────────────────────────────────────
heat = coef_df[coef_df["predictor"].isin(DENOM_COLS)].pivot(
    index="predictor", columns="party", values="estimate"
).reindex(index=DENOM_COLS, columns=PARTY_LABELS_ALR)

q_mat = coef_df[coef_df["predictor"].isin(DENOM_COLS)].pivot(
    index="predictor", columns="party", values="q_bh"
).reindex(index=DENOM_COLS, columns=PARTY_LABELS_ALR)

vals   = heat.values[~np.isnan(heat.values)]
vmax   = max(abs(vals)) if len(vals) else 1.0

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax, label="ALR coefficient (z-scored predictor)")

ax.set_xticks(range(len(PARTY_LABELS_ALR)))
ax.set_xticklabels(PARTY_LABELS_ALR, fontsize=9)
ax.set_yticks(range(len(DENOM_COLS)))
ax.set_yticklabels([DENOM_LABELS[d] for d in DENOM_COLS], fontsize=9)

for i, pred in enumerate(DENOM_COLS):
    for j, party in enumerate(PARTY_LABELS_ALR):
        val  = heat.loc[pred, party]
        q    = q_mat.loc[pred, party]
        if np.isnan(val):
            continue
        star  = "*" if q < 0.10 else ""
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center",
                fontsize=6.5, color=color,
                fontweight="bold" if star else "normal")

ax.set_title(
    "Confessional decomposition — Dirichlet ALR coefficients\n"
    "7 denominations × 7 parties  (Other = ALR reference)  |  * q < 0.10 (BH-FDR)",
    fontsize=9,
)
ax.set_xlabel("Party", fontsize=9)
ax.set_ylabel("Denomination (z-scored share)", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "15a_confessional_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n  Saved -> figures/15a_confessional_heatmap.png")
print(f"  Saved -> reports/15a_confessional_coefs.csv")
print(f"  Saved -> reports/15a_confessional_lr_test.csv")
print(f"  Saved -> data/processed/religion_judet_2021.csv")
print("\n" + "=" * 70)
print("CONFESSIONAL DECOMPOSITION COMPLETE")
print("=" * 70)
