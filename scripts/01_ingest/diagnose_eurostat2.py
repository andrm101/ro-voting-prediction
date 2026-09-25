import pandas as pd
from pathlib import Path
RAW = Path("data/raw/eurostat")

df = pd.read_csv(RAW / "demo_r_pjanaggr3_full.tsv", sep="\t",
                 encoding="utf-8-sig", dtype=str, keep_default_na=False)
first = df.columns[0]
df["geo"] = df[first].str.split(r"[,\\]").str[-1].str.strip()
ro = df[df["geo"].str.startswith("RO")]
print(f"Total RO rows: {len(ro)}")
ro_sex = ro[first].str.split(r"[,\\]").str[2].value_counts()
ro_age = ro[first].str.split(r"[,\\]").str[3].value_counts()
print("Sex values:", list(ro_sex.index))
print("Age values:", list(ro_age.index))

df2 = pd.read_csv(RAW / "nama_10r_3empers_full.tsv", sep="\t",
                  encoding="utf-8-sig", dtype=str, keep_default_na=False)
first2 = df2.columns[0]
df2["geo"] = df2[first2].str.split(r"[,\\]").str[-1].str.strip()
ro2 = df2[df2["geo"].str.startswith("RO")]
ro2_nace = ro2[first2].str.split(r"[,\\]").str[3].value_counts()
print(f"\nEmployment NACE values (RO): {list(ro2_nace.index)}")
ro2_unit = ro2[first2].str.split(r"[,\\]").str[1].value_counts()
print(f"Employment unit values: {list(ro2_unit.index)}")
ro2_wstatus = ro2[first2].str.split(r"[,\\]").str[2].value_counts()
print(f"Employment wstatus values: {list(ro2_wstatus.index)}")

# Check edat_lfse_16 geo coverage
df3 = pd.read_csv(RAW / "edat_lfse_16_full.tsv", sep="\t",
                  encoding="utf-8-sig", dtype=str, keep_default_na=False)
first3 = df3.columns[0]
df3["geo"] = df3[first3].str.split(r"[,\\]").str[-1].str.strip()
ro3 = df3[df3["geo"].str.startswith("RO")]
print(f"\nESL NUTS 2 geo codes for RO: {sorted(ro3['geo'].unique())}")
ro3_sex = ro3[first3].str.split(r"[,\\]").str[2].value_counts()
print(f"ESL sex: {list(ro3_sex.index)}")

# Check ilc_peps11 geo coverage
df4 = pd.read_csv(RAW / "ilc_peps11_full.tsv", sep="\t",
                  encoding="utf-8-sig", dtype=str, keep_default_na=False)
first4 = df4.columns[0]
df4["geo"] = df4[first4].str.split(r"[,\\]").str[-1].str.strip()
ro4 = df4[df4["geo"].str.startswith("RO")]
print(f"\nPoverty NUTS 2 geo codes: {sorted(ro4['geo'].unique())}")
