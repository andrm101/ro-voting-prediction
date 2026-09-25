"""Run all three ingest scripts in sequence. Exit on first failure."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "ingest_roaep.py",
    "ingest_ins.py",
    "ingest_eurostat.py",
]

HERE = Path(__file__).parent

for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)
    result = subprocess.run([sys.executable, str(HERE / script)], check=False)
    if result.returncode != 0:
        print(f"\n[FAIL] {script} exited with code {result.returncode}.", file=sys.stderr)
        print("Fix the issue above before proceeding to clean stage.", file=sys.stderr)
        sys.exit(result.returncode)

print("\n[ALL INGEST COMPLETE] -> data/processed/")
print("Next step: python scripts/02_clean/clean_elections.py")
