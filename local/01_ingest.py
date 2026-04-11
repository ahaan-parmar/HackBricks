"""
01_ingest.py — DATA INGESTION
Reads UCI Student Dropout CSV → Bronze Parquet (local layer)

Run: python local/01_ingest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datetime import datetime, timezone
from config import RAW_CSV, BRONZE_FILE, BRONZE_DIR


def ingest() -> pd.DataFrame:
    # ── Guard ─────────────────────────────────────────────────────────────────
    if not RAW_CSV.exists():
        print(f"\n[ERROR] Dataset not found at: {RAW_CSV}")
        print("\nDownload steps:")
        print("  1. Go to: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success")
        print("  2. Click 'Download' → extract the zip")
        print(f"  3. Rename the CSV to 'student_dropout.csv' and place it in: {RAW_CSV.parent}")
        sys.exit(1)

    print(f"[01_ingest] Reading: {RAW_CSV}")

    # ── Read ──────────────────────────────────────────────────────────────────
    # Auto-detect delimiter (UCI original uses ";", clean export uses ",")
    import csv as _csv
    with open(RAW_CSV, newline="", encoding="utf-8") as _f:
        _sample = _f.read(2048)
    _sep = ";" if _sample.count(";") > _sample.count(",") else ","
    df = pd.read_csv(RAW_CSV, sep=_sep)

    print(f"  Raw rows    : {len(df)}")
    print(f"  Raw columns : {len(df.columns)}")
    print(f"  Columns     : {list(df.columns)}")

    # ── Add ingestion metadata ────────────────────────────────────────────────
    df["_ingest_timestamp"] = datetime.now(timezone.utc).isoformat()
    df["_source_file"]      = str(RAW_CSV)

    # ── Write Bronze ──────────────────────────────────────────────────────────
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BRONZE_FILE, index=False)

    print(f"  Written to  : {BRONZE_FILE}")
    print(f"  Bronze rows : {len(df)}")
    return df


if __name__ == "__main__":
    df = ingest()
    print("\n[01_ingest] Sample:")
    print(df.head(3).to_string())
    print("\n[01_ingest] DONE")
