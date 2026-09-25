"""Run all model scripts in sequence. Exit on first failure."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_baseline.py",
    "train_gbm.py",
    "train_spatial.py",
    "train_stacked.py",
]

HERE = Path(__file__).parent

for script in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)
    result = subprocess.run([sys.executable, str(HERE / script)], check=False)
    if result.returncode != 0:
        print(f"\n[FAIL] {script} exited with code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)

print("\n[ALL MODELS COMPLETE]")
print("Next step: python scripts/05_evaluate/evaluate.py")
