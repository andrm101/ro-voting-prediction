import pandas as pd
import numpy as np

# Load pjanaggr3 for NUTS3 age groups
pj = pd.read_csv(
    "data/raw/eurostat/demo_r_pjanaggr3_full.tsv",
    sep="\t", na_values=[":", " "], dtype=str,
)
pj.columns = [c.strip() for c in pj.columns]
key_col = pj.columns[0]
pj[["freq","unit","sex","age","geo"]] = pj[key_col].str.split(",", expand=True)

year_col = [c for c in pj.columns if "2021" in c][0]

# National-level correction: (Y15+Y16+Y17) from d2jan / Y15-64 from pjanaggr3
nat_15_64 = pj[(pj.sex=="T") & (pj.age=="Y15-64") & (pj.geo=="RO")][year_col].values[0]
print(f"National Y15-64 (2021): {nat_15_64}")
y15_17 = 213014 + 205867 + 205507  # from earlier check
nat_15_64_n = float(str(nat_15_64).replace("b","").replace("e","").replace("p","").strip())
correction = y15_17 / nat_15_64_n
print(f"National correction share (15-17 / 15-64): {correction:.4f}")

# NUTS3 Romania counties
ro3 = pj[(pj.sex=="T") & (pj.geo.str.match(r"^RO\d{3}$"))]
ages_needed = ["TOTAL", "Y15-64", "Y_GE65", "Y_LT15"]
ro3_wide = ro3[ro3.age.isin(ages_needed)][["geo","age", year_col]]
ro3_wide = ro3_wide.pivot(index="geo", columns="age", values=year_col)

def clean(s):
    if pd.isna(s): return np.nan
    return float(str(s).replace("b","").replace("e","").replace("p","").replace(" ","").strip())

for col in ro3_wide.columns:
    ro3_wide[col] = ro3_wide[col].apply(clean)

ro3_wide["pop_18plus"] = ro3_wide["Y15-64"] * (1 - correction) + ro3_wide["Y_GE65"]
print("\nNUTS3 sample:")
print(ro3_wide[["TOTAL","Y_LT15","Y15-64","Y_GE65","pop_18plus"]].head(10).to_string())
print(f"\nNational 18+ sum: {ro3_wide['pop_18plus'].sum():,.0f}")
print(f"National TOTAL sum: {ro3_wide['TOTAL'].sum():,.0f}")

# Load features for NUTS3 → judet mapping
feat = pd.read_parquet("data/processed/features_2022_judet.parquet")
print("\nNUTS3 codes in features:")
print(feat[["judet","geo_nuts3"]].head(10).to_string())
