import pandas as pd
df = pd.read_parquet("data/processed/electoral_series_judet.parquet")

print("2024 parl chambers:")
print(df[df.election=="parlamentare_2024"][["chamber","judet"]].groupby("chamber").count())

j2020 = set(df[df.election=="parlamentare_2020"]["judet"])
j2024r1 = set(df[df.election=="prezidentiale_2024_r1"]["judet"])
print("\nExtra in 2024 R1 vs 2020:", j2024r1 - j2020)
print("Missing in 2024 R1 vs 2020:", j2020 - j2024r1)

buc = df[(df.election=="parlamentare_2020") & (df.chamber=="CD") & (df.judet.str.contains("BUCUR"))]
print("\n2020 Bucharest:", buc[["judet","enrolled","voted","t_AUR","t_PSD","t_PNL"]].to_string())

print("\n2025 R2 Simion national total:")
r2 = df[df.election=="prezidentiale_2025_r2"]
print(f"  Simion: {r2['cand_SIMION'].mul(r2['valid_votes']).sum():,.0f}")
print(f"  Dan:    {r2['cand_DAN'].mul(r2['valid_votes']).sum():,.0f}")
