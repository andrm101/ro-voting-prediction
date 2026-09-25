import pandas as pd
s = pd.read_parquet("data/processed/electoral_series_judet.parquet")
print(s.columns.tolist())
print(s[s.election=="parlamentare_2024"].head(3)[["judet","chamber","enrolled","voted"]].to_string())
