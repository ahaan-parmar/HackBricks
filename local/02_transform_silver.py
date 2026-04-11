"""
02_transform_silver.py — SILVER TRANSFORM (local/pandas version)
Bronze Parquet → Silver Parquet with feature engineering + Financial Stress Index

Run: python local/02_transform_silver.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import re
import pandas as pd
from config import BRONZE_FILE, SILVER_DIR, SILVER_FILE

UNEMP_MIN = 7.0
UNEMP_MAX = 16.2


def clean_col_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[\s\(\)/]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # ── Step 1: Rename columns ─────────────────────────────────────────────────
    df.columns = [clean_col_name(c) for c in df.columns]
    print(f"  Columns after rename: {list(df.columns)}")

    # ── Step 2: Drop nulls / bad rows ─────────────────────────────────────────
    df = df.dropna(subset=["target"])
    df = df[df["age_at_enrollment"] > 0]
    df = df[df["curricular_units_1st_sem_enrolled"] >= 0]
    print(f"  Rows after cleaning: {len(df)}")

    # ── Step 3: Binary label ───────────────────────────────────────────────────
    df["target_binary"] = (df["target"] == "Dropout").astype(int)
    dropout_rate = df["target_binary"].mean()
    print(f"  Dropout rate: {dropout_rate:.2%}")

    # ── Step 4: Feature Engineering ───────────────────────────────────────────

    # 4a. Enrollment efficiency
    df["enrollment_efficiency"] = df.apply(
        lambda r: 0.0
        if r["curricular_units_1st_sem_enrolled"] == 0
        else r["curricular_units_1st_sem_approved"] / r["curricular_units_1st_sem_enrolled"],
        axis=1,
    )

    # 4b. Average grade across both semesters
    df["avg_grade"] = (
        df["curricular_units_1st_sem_grade"] + df["curricular_units_2nd_sem_grade"]
    ) / 2.0

    # 4c. Total approved units
    df["total_approved_units"] = (
        df["curricular_units_1st_sem_approved"] + df["curricular_units_2nd_sem_approved"]
    )

    # 4d. Academic disengagement ratio
    df["_disengagement"] = df.apply(
        lambda r: 0.0
        if r["curricular_units_1st_sem_enrolled"] == 0
        else r["curricular_units_1st_sem_without_evaluations"]
        / r["curricular_units_1st_sem_enrolled"],
        axis=1,
    )

    # 4e. Normalised unemployment
    df["_unemp_norm"] = (
        (df["unemployment_rate"] - UNEMP_MIN) / (UNEMP_MAX - UNEMP_MIN)
    ).clip(0, 1)

    # ── Step 5: Financial Stress Index ────────────────────────────────────────
    df["financial_stress_index"] = (
        df["debtor"].astype(float)                         * 0.30
        + (1 - df["tuition_fees_up_to_date"]).astype(float) * 0.25
        + (1 - df["scholarship_holder"]).astype(float)      * 0.15
        + df["displaced"].astype(float)                     * 0.10
        + df["_unemp_norm"]                                 * 0.10
        + df["_disengagement"]                              * 0.10
    )

    # Drop temp columns
    df = df.drop(columns=["_disengagement", "_unemp_norm"])

    # FSI sanity check
    print(f"  FSI  min={df['financial_stress_index'].min():.3f}  "
          f"median={df['financial_stress_index'].median():.3f}  "
          f"max={df['financial_stress_index'].max():.3f}")

    return df


def main() -> pd.DataFrame:
    if not BRONZE_FILE.exists():
        print(f"[ERROR] Bronze file not found: {BRONZE_FILE}")
        print("  Run: python local/01_ingest.py  first")
        sys.exit(1)

    print(f"[02_silver] Reading: {BRONZE_FILE}")
    df = pd.read_parquet(BRONZE_FILE)
    print(f"  Bronze rows: {len(df)}")

    df = transform(df)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER_FILE, index=False)
    print(f"  Written to: {SILVER_FILE}")
    print(f"  Silver rows: {len(df)}")
    return df


if __name__ == "__main__":
    df = main()
    print("\n[02_silver] Sample FSI values:")
    print(df[["target", "financial_stress_index", "avg_grade", "enrollment_efficiency"]].head(5).to_string())
    print("\n[02_silver] DONE")
