"""
Task 8 — Spatial Autocorrelation and Choropleth Maps
=====================================================
Tests whether OLS residuals from the key models exhibit spatial autocorrelation
(Moran's I), and fits spatial lag models as robustness checks where significant.

Models tested:
  1. AUR 2020→2024 Senate swing (primary structural model, R²=0.585)
  2. Ghost Electorate Ratio (R²=0.680)
  3. Georgescu→Simion transfer partial-correlation residuals

Spatial weights: queen contiguity (W row-standardised) from GISCO NUTS-3
boundaries (Eurostat, 2021, 1:10M scale).

Spatial lag model fitted for AUR swing via 2SLS GM estimator (Kelejian &
Prucha 1999) to handle endogeneity of the spatial lag term.

Outputs:
  figures/11a_romania_map.png          — 3-panel choropleth (cluster / GER / AUR swing)
  figures/11b_moran_scatter.png        — Moran scatter plots for each model
  reports/11a_spatial_robustness.csv   — Moran's I statistics + spatial lag results

References:
  Anselin L (1988). Spatial Econometrics: Methods and Models. Kluwer.
  Kelejian HH, Prucha IR (1999). A generalised spatial two-stage LS procedure.
    Journal of Real Estate Finance and Economics 17:99-121.
"""
from __future__ import annotations
import warnings, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from esda.moran import Moran
from libpysal.weights import Queen, w_subset
import spreg

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# 1. Download and cache GISCO NUTS-3 Romania shapefile
# ---------------------------------------------------------------------------
GISCO_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_10M_2021_4326_LEVL_3.geojson"
)
GISCO_LOCAL = RAW_DIR / "eurostat" / "nuts3_2021_10M.geojson"

if not GISCO_LOCAL.exists():
    print(f"Downloading GISCO NUTS-3 shapefile...")
    urllib.request.urlretrieve(GISCO_URL, GISCO_LOCAL)
    print(f"  Saved -> {GISCO_LOCAL}")
else:
    print(f"GISCO shapefile cached at {GISCO_LOCAL.name}")

gdf_all = gpd.read_file(GISCO_LOCAL)
gdf = gdf_all[gdf_all["CNTR_CODE"] == "RO"].copy()
gdf = gdf[["NUTS_ID", "NUTS_NAME", "geometry"]].rename(columns={"NUTS_ID": "geo_nuts3"})
print(f"Romanian NUTS-3 units: {len(gdf)}")

# ---------------------------------------------------------------------------
# 2. Load features and join to geometry
# ---------------------------------------------------------------------------
import unicodedata

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()

feat  = pd.read_parquet(ROOT / "data" / "processed" / "features_2022_judet.parquet")
clust = pd.read_csv(ROOT / "data" / "processed" / "cluster_assignments_final.csv")
feat  = feat.merge(clust[["judet", "cluster_short"]], on="judet", how="left")

series = pd.read_parquet(ROOT / "data" / "processed" / "electoral_series_judet.parquet")

# Swing data: 2020→2024 AUR Senate
s2020 = series[(series.election=="parlamentare_2020") & (series.chamber=="S")].copy()
s2024 = series[(series.election=="parlamentare_2024") & (series.chamber=="S")].copy()
s2020["judet_norm"] = s2020["judet"].apply(_norm)
s2024["judet_norm"] = s2024["judet"].apply(_norm)
s2020 = s2020.set_index("judet_norm")
s2024 = s2024.set_index("judet_norm")
feat["judet_norm"] = feat["judet"].apply(_norm)

sw_aur = (s2024["t_AUR"].reindex(feat["judet_norm"].values)
          - s2020["t_AUR"].reindex(feat["judet_norm"].values))
sw_aur.index = feat["judet_norm"].values
feat["sw_AUR"] = sw_aur.values

# GER
def _clean_eurostat(s):
    if pd.isna(s): return np.nan
    return float(str(s).replace("b","").replace("e","").replace("p","").replace(" ","").strip())

pj = pd.read_csv(
    ROOT / "data" / "raw" / "eurostat" / "demo_r_pjanaggr3_full.tsv",
    sep="\t", na_values=[":", " "], dtype=str,
)
pj.columns = [c.strip() for c in pj.columns]
key_col = pj.columns[0]
pj[["freq","unit","sex","age","geo"]] = pj[key_col].str.split(",", expand=True)
year_col = [c for c in pj.columns if "2021" in c][0]

nat_15_64 = _clean_eurostat(
    pj[(pj.sex=="T") & (pj.age=="Y15-64") & (pj.geo=="RO")][year_col].values[0])
ALPHA = (213_014 + 205_867 + 205_507) / nat_15_64

ro3 = pj[(pj.sex=="T") & (pj.geo.str.match(r"^RO\d{3}$"))].copy()
ro3_wide = (ro3[ro3.age.isin(["Y15-64","Y_GE65"])][["geo","age",year_col]]
            .pivot(index="geo", columns="age", values=year_col))
for col in ro3_wide.columns:
    ro3_wide[col] = ro3_wide[col].apply(_clean_eurostat)
ro3_wide["pop_18plus"] = ro3_wide["Y15-64"] * (1 - ALPHA) + ro3_wide["Y_GE65"]
ro3_wide = ro3_wide.reset_index().rename(columns={"geo": "geo_nuts3"})

feat = feat.merge(ro3_wide[["geo_nuts3","pop_18plus"]], on="geo_nuts3", how="left")

enroll_2024 = (series[(series.election=="parlamentare_2024") & (series.chamber=="S")]
               .copy())
enroll_2024["judet_norm"] = enroll_2024["judet"].apply(_norm)
feat = feat.merge(enroll_2024[["judet_norm","enrolled","turnout"]]
                  .rename(columns={"enrolled":"enrolled_2024","turnout":"turnout_2024"}),
                  on="judet_norm", how="left")
feat["ger"] = (feat["enrolled_2024"] - feat["pop_18plus"]) / feat["enrolled_2024"]

# Georgescu/Simion for transfer residuals
pres24r1 = series[series.election=="prezidentiale_2024_r1"].copy()
pres25r1 = series[series.election=="prezidentiale_2025_r1"].copy()
pres24r1["judet_norm"] = pres24r1["judet"].apply(_norm)
pres25r1["judet_norm"] = pres25r1["judet"].apply(_norm)
feat = feat.merge(pres24r1[["judet_norm","cand_GEORGESCU"]]
                  .rename(columns={"cand_GEORGESCU":"georgescu_r1"}), on="judet_norm", how="left")
feat = feat.merge(pres25r1[["judet_norm","cand_SIMION"]]
                  .rename(columns={"cand_SIMION":"simion_r1"}), on="judet_norm", how="left")

CONTROLS = [
    "idx_grievance", "idx_modernity", "idx_demographic_pressure",
    "pct_employment_agriculture", "pct_maghiari", "pct_romi",
    "gva_deindustrial_index", "long_term_unemployment_rate",
]

# Merge geometry
gdf = gdf.merge(feat[["geo_nuts3","judet","cluster_short","sw_AUR","ger",
                       "georgescu_r1","simion_r1","turnout_2024",
                       "emigrants_permanent_rate"] + CONTROLS],
                on="geo_nuts3", how="left")

print(f"Counties with geometry: {gdf['judet'].notna().sum()}")

# ---------------------------------------------------------------------------
# 3. Build queen contiguity spatial weights
# ---------------------------------------------------------------------------
print("\nBuilding queen contiguity weights...")
gdf_proj = gdf.to_crs("EPSG:3844")  # Romania national CRS (Stereo 70)
W_queen = Queen.from_dataframe(gdf_proj, silence_warnings=True)
W_queen.transform = "r"  # row-standardise
print(f"  {W_queen.n} units,  mean neighbours: {W_queen.mean_neighbors:.2f}")

# Helper: Moran's I with randomisation inference
def moran_test(residuals, W, label):
    valid = ~np.isnan(residuals)
    r = residuals[valid]
    if valid.sum() < W.n:
        # Subset weights to valid observations
        ids = np.where(valid)[0]
        W_sub = w_subset(W, ids.tolist(), silence_warnings=True)
        W_sub.transform = "r"
        mi = Moran(r, W_sub, permutations=999)
    else:
        mi = Moran(r, W, permutations=999)
    print(f"  {label}: I={mi.I:.4f}  p_sim={mi.p_sim:.3f}  "
          f"E[I]={mi.EI:.4f}  z={mi.z_norm:.3f}")
    return mi

# OLS helper for residuals
def ols_resid(y_col, control_cols, df):
    sub = df.dropna(subset=[y_col] + control_cols)
    y = sub[y_col].values
    X = sm.add_constant(sub[control_cols].values, has_constant="add")
    resid = sm.OLS(y, X).fit().resid
    # return aligned to full df index (NaN where missing)
    out = np.full(len(df), np.nan)
    out[df.index.isin(sub.index)] = resid
    return out

print("\n=== Moran's I on OLS residuals ===")
resid_aur    = ols_resid("sw_AUR",        CONTROLS, gdf)
resid_ger    = ols_resid("ger",           CONTROLS + ["emigrants_permanent_rate"], gdf)

# Transfer residuals: FWL
def _residualise(y, X):
    X_c = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_c).fit().resid

ctrl_df = gdf.dropna(subset=["georgescu_r1","simion_r1"] + CONTROLS)
X_ctrl  = ctrl_df[CONTROLS].values
e_G = _residualise(ctrl_df["georgescu_r1"].values, X_ctrl)
e_S = _residualise(ctrl_df["simion_r1"].values, X_ctrl)
resid_transfer_G = np.full(len(gdf), np.nan)
resid_transfer_S = np.full(len(gdf), np.nan)
resid_transfer_G[gdf.index.isin(ctrl_df.index)] = e_G
resid_transfer_S[gdf.index.isin(ctrl_df.index)] = e_S

mi_aur      = moran_test(resid_aur,          W_queen, "AUR swing OLS residuals")
mi_ger      = moran_test(resid_ger,           W_queen, "GER OLS residuals")
mi_georg    = moran_test(resid_transfer_G,    W_queen, "Georgescu transfer residuals (e_G)")
mi_simion   = moran_test(resid_transfer_S,    W_queen, "Simion transfer residuals (e_S)")

# ---------------------------------------------------------------------------
# 4. Spatial lag model for AUR swing (primary robustness check)
# ---------------------------------------------------------------------------
print("\n=== Spatial Lag Model: AUR Swing ===")
sub_aur = gdf.dropna(subset=["sw_AUR"] + CONTROLS).copy()
# Rebuild weights for subset
ids_aur = [i for i, v in enumerate(~gdf[["sw_AUR"] + CONTROLS].isna().any(axis=1)) if v]
W_aur = w_subset(W_queen, ids_aur, silence_warnings=True)
W_aur.transform = "r"

y_aur = sub_aur["sw_AUR"].values
X_std = sub_aur[CONTROLS].copy()
for col in CONTROLS:
    X_std[col] = (X_std[col] - X_std[col].mean()) / X_std[col].std()
X_aur = X_std.values

# OLS baseline (for comparison)
ols_aur = spreg.OLS(y_aur, X_aur,
                    name_y="sw_AUR",
                    name_x=CONTROLS,
                    w=W_aur, spat_diag=True)

# GM spatial lag (Kelejian-Prucha two-stage)
slm_aur = spreg.GM_Lag(y_aur, X_aur,
                        w=W_aur,
                        name_y="sw_AUR",
                        name_x=CONTROLS,
                        name_w="queen")

print(f"\n  OLS:  R²={ols_aur.r2:.3f}  "
      f"LM-lag p={ols_aur.lm_lag[1]:.3f}  LM-error p={ols_aur.lm_error[1]:.3f}")
print(f"  SLM:  pseudo-R²={slm_aur.pr2:.3f}  "
      f"spatial lag ρ={slm_aur.betas[-1][0]:.4f}")

# Build comparison table
ols_params = list(zip(["Intercept"] + CONTROLS, ols_aur.betas.flatten(), ols_aur.std_err))
slm_params = list(zip(["Intercept"] + CONTROLS + ["W·sw_AUR"],
                       slm_aur.betas.flatten(), slm_aur.std_err))

print("\n  Coefficient comparison (OLS vs SLM):")
print(f"  {'Term':35s} {'OLS β':>9} {'SLM β':>9}")
for (n1, b1, _), (n2, b2, _) in zip(ols_params, slm_params):
    print(f"  {n1:35s} {b1:+9.4f} {b2:+9.4f}")
print(f"  {'W·sw_AUR (ρ)':35s} {'—':>9} {slm_aur.betas[-1][0]:+9.4f}")

# ---------------------------------------------------------------------------
# 5. Figure A: Moran scatter plots
# ---------------------------------------------------------------------------
CLUSTER_COLS = {
    "Moldova Profunda":   "#e41a1c",
    "Campia Uitata":      "#ff7f00",
    "Insula Capitalei":   "#4daf4a",
    "Centrele Dinamice":  "#377eb8",
    "Arcul Identitar":    "#984ea3",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
moran_items = [
    (resid_aur,        mi_aur,     "AUR swing OLS residuals"),
    (resid_ger,        mi_ger,     "GER OLS residuals"),
    (resid_transfer_G, mi_georg,   "Georgescu transfer residuals"),
]

for ax, (resid, mi, title) in zip(axes, moran_items):
    valid = ~np.isnan(resid)
    r = resid[valid]
    clusters = gdf["cluster_short"].values[valid]
    for cl, col in CLUSTER_COLS.items():
        m = clusters == cl
        ax.scatter(r[m], mi.w.sparse.dot(r)[m] if mi.w.n == valid.sum()
                   else np.zeros(m.sum()),
                   c=col, s=35, alpha=0.8, label=cl)
    ax.axhline(0, lw=0.7, ls="--", c="grey")
    ax.axvline(0, lw=0.7, ls="--", c="grey")
    # Draw regression line manually
    w_r = mi.w.sparse.dot(r) if mi.w.n == valid.sum() else np.zeros_like(r)
    if mi.w.n == valid.sum():
        slope = np.polyfit(r, w_r, 1)[0]
        xs = np.linspace(r.min(), r.max(), 100)
        ax.plot(xs, slope * xs, "k-", lw=1.2)
    ax.set_xlabel("OLS residual", fontsize=9)
    ax.set_ylabel("Spatial lag of residual (Wr)", fontsize=9)
    ax.set_title(f"{title}\nI = {mi.I:.3f}  p = {mi.p_sim:.3f}", fontsize=9)

handles = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLS.items()]
axes[0].legend(handles=handles, fontsize=6, loc="upper left")
plt.suptitle("Moran Scatter Plots — Spatial Autocorrelation of OLS Residuals",
             fontsize=11, y=1.01)
plt.tight_layout()
out = FIG_DIR / "11b_moran_scatter.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  Saved -> {out.name}")

# ---------------------------------------------------------------------------
# 6. Figure B: 3-panel choropleth map
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Panel 1: cluster
cluster_order = list(CLUSTER_COLS.keys())
cluster_cmap  = mcolors.ListedColormap([CLUSTER_COLS[k] for k in cluster_order])
gdf["cluster_num"] = gdf["cluster_short"].map({k: i for i, k in enumerate(cluster_order)})
gdf.plot(column="cluster_num", cmap=cluster_cmap, linewidth=0.4,
         edgecolor="white", ax=axes[0], legend=False,
         missing_kwds={"color": "#dddddd"})
axes[0].set_title("Voter-Profile Clusters", fontsize=11)
axes[0].axis("off")
handles = [mpatches.Patch(color=CLUSTER_COLS[k], label=k) for k in cluster_order]
axes[0].legend(handles=handles, fontsize=6.5, loc="lower left",
               framealpha=0.85, title="Cluster")

# Panel 2: GER
gdf.plot(column="ger", cmap="Reds", linewidth=0.4, edgecolor="white",
         ax=axes[1], legend=True, vmin=0, vmax=0.35,
         missing_kwds={"color": "#dddddd"},
         legend_kwds={"label": "GER", "shrink": 0.6})
axes[1].set_title("Ghost Electorate Ratio\n(2024 parliamentary rolls vs 2021 pop.)", fontsize=10)
axes[1].axis("off")

# Panel 3: AUR swing
sw_max = gdf["sw_AUR"].abs().max()
gdf.plot(column="sw_AUR", cmap="RdBu_r", linewidth=0.4, edgecolor="white",
         ax=axes[2], legend=True, vmin=-sw_max, vmax=sw_max,
         missing_kwds={"color": "#dddddd"},
         legend_kwds={"label": "Δ share (pp)", "shrink": 0.6})
axes[2].set_title("AUR Vote Share Swing\n2020→2024 Senate", fontsize=10)
axes[2].axis("off")

plt.suptitle("Romanian Electoral Geography — 42 județe",
             fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
out = FIG_DIR / "11a_romania_map.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved -> {out.name}")

# ---------------------------------------------------------------------------
# 7. Save spatial robustness table
# ---------------------------------------------------------------------------
spatial_results = pd.DataFrame([
    {"Model": "AUR swing OLS",             "Moran_I": mi_aur.I,
     "p_sim": mi_aur.p_sim,    "z_norm": mi_aur.z_norm},
    {"Model": "GER OLS",                   "Moran_I": mi_ger.I,
     "p_sim": mi_ger.p_sim,    "z_norm": mi_ger.z_norm},
    {"Model": "Transfer e_Georgescu",      "Moran_I": mi_georg.I,
     "p_sim": mi_georg.p_sim,  "z_norm": mi_georg.z_norm},
    {"Model": "Transfer e_Simion",         "Moran_I": mi_simion.I,
     "p_sim": mi_simion.p_sim, "z_norm": mi_simion.z_norm},
])
spatial_results["SLM_rho"]  = [slm_aur.betas[-1][0], None, None, None]
spatial_results["SLM_pr2"]  = [slm_aur.pr2,          None, None, None]
spatial_results["OLS_LM_lag_p"] = [ols_aur.lm_lag[1], None, None, None]

out_csv = REP_DIR / "11a_spatial_robustness.csv"
spatial_results.to_csv(out_csv, index=False, float_format="%.4f")
print(f"  Saved -> {out_csv.name}")

print("\n" + "=" * 65)
print("TASK 8 COMPLETE — Spatial Robustness Analysis")
print("=" * 65)
