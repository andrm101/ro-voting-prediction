import pandas as pd

df = pd.read_excel("data/raw/ins/ins_populatie_2021.csv.xls", header=None)
print("Shape:", df.shape)
# Show all rows to understand structure
for i, row in df.iterrows():
    vals = row.tolist()
    if any(v is not None and str(v).strip() not in ('', 'nan') for v in vals):
        print(f"Row {i}: {vals}")
