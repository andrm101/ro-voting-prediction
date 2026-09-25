"""Run all three clean scripts in sequence. Exit on first failure."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "clean_elections.py",
    "clean_census.py",
    "clean_eurostat.py",
]

HERE = Path(__file__).parent

for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)
    result = subprocess.run([sys.executable, str(HERE / script)], check=False)
    if result.returncode != 0:
        print(f"\n[FAIL] {script} exited with code {result.returncode}.", file=sys.stderr)
        print("Fix the issue above before proceeding to feature build.", file=sys.stderr)
        sys.exit(result.returncode)

print("\n[ALL CLEAN COMPLETE] -> data/processed/")
print("Outputs:")
print("  elections_clean.parquet — wide election results, party shares, composites")
print("  census_clean.parquet    — ethnicity, religion, urban/rural shares")
print("  eurostat_clean.parquet  — NUTS 3 socio-economic indicators")
print("\nNext step: python scripts/03_features/build_features.py")
