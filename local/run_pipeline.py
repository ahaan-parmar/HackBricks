"""
run_pipeline.py - LOCAL PIPELINE ORCHESTRATOR
Runs all local steps in order:
  01  Ingest CSV -> Bronze Parquet
  02  Bronze -> Silver Parquet (feature engineering + FSI)

Run: python local/run_pipeline.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))


def run_step(script: Path, label: str):
    print(f"\n[PIPELINE] {label}")
    print(f"  Script: {script}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  Done: {label}")


print("=" * 50)
print("  HackBricks - Local ML Pipeline")
print("=" * 50)

run_step(ROOT / "local" / "01_ingest.py",           "01 - Ingest CSV -> Bronze Parquet")
run_step(ROOT / "local" / "02_transform_silver.py",  "02 - Transform Bronze -> Silver Parquet")

print("\n" + "=" * 50)
print("  Pipeline complete!")
print("  Bronze : data/bronze/student_dropout_bronze.parquet")
print("  Silver : data/silver/student_dropout_silver.parquet")
print("=" * 50)
print("\nNext: start the API server")
print("  python backend/main.py")
