"""
00_clean_data.py - DATA CLEANING
Reads raw CSV, cleans column names, handles nulls/outliers,
saves clean parquet to data/processed/

Run: python local/00_clean_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
RAW_CSV    = ROOT / "dataset" / "students_dropout_academic_success.csv"
CLEAN_CSV  = ROOT / "dataset" / "students_dropout_academic_success_clean.csv"
OUT_DIR    = ROOT / "data" / "processed"

# Human-readable column name mapping (raw → clean → display)
COL_RENAME = {
    "Marital Status":                               "marital_status",
    "Application mode":                             "application_mode",
    "Application order":                            "application_order",
    "Course":                                       "course",
    "Daytime/evening attendance":                   "daytime_evening_attendance",
    "Previous qualification":                       "previous_qualification",
    "Previous qualification (grade)":               "previous_qualification_grade",
    "Nacionality":                                  "nacionality",
    "Mother's qualification":                       "mother_s_qualification",
    "Father's qualification":                       "father_s_qualification",
    "Mother's occupation":                          "mother_s_occupation",
    "Father's occupation":                          "father_s_occupation",
    "Admission grade":                              "admission_grade",
    "Displaced":                                    "displaced",
    "Educational special needs":                    "educational_special_needs",
    "Debtor":                                       "debtor",
    "Tuition fees up to date":                      "tuition_fees_up_to_date",
    "Gender":                                       "gender",
    "Scholarship holder":                           "scholarship_holder",
    "Age at enrollment":                            "age_at_enrollment",
    "International":                                "international",
    "Curricular units 1st sem (credited)":          "curricular_units_1st_sem_credited",
    "Curricular units 1st sem (enrolled)":          "curricular_units_1st_sem_enrolled",
    "Curricular units 1st sem (evaluations)":       "curricular_units_1st_sem_evaluations",
    "Curricular units 1st sem (approved)":          "curricular_units_1st_sem_approved",
    "Curricular units 1st sem (grade)":             "curricular_units_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "curricular_units_1st_sem_without_evaluations",
    "Curricular units 2nd sem (credited)":          "curricular_units_2nd_sem_credited",
    "Curricular units 2nd sem (enrolled)":          "curricular_units_2nd_sem_enrolled",
    "Curricular units 2nd sem (evaluations)":       "curricular_units_2nd_sem_evaluations",
    "Curricular units 2nd sem (approved)":          "curricular_units_2nd_sem_approved",
    "Curricular units 2nd sem (grade)":             "curricular_units_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "curricular_units_2nd_sem_without_evaluations",
    "Unemployment rate":                            "unemployment_rate",
    "Inflation rate":                               "inflation_rate",
    "GDP":                                          "gdp",
    "target":                                       "target",
}

DISPLAY_NAMES = {
    "marital_status":                               "Marital Status",
    "application_mode":                             "Application Mode",
    "application_order":                            "Application Order",
    "course":                                       "Course Code",
    "daytime_evening_attendance":                   "Attendance Type",
    "previous_qualification":                       "Previous Qualification",
    "previous_qualification_grade":                 "Prev. Qual. Grade",
    "nacionality":                                  "Nationality",
    "mother_s_qualification":                       "Mother's Qualification",
    "father_s_qualification":                       "Father's Qualification",
    "mother_s_occupation":                          "Mother's Occupation",
    "father_s_occupation":                          "Father's Occupation",
    "admission_grade":                              "Admission Grade",
    "displaced":                                    "Displaced",
    "educational_special_needs":                    "Special Needs",
    "debtor":                                       "Debtor",
    "tuition_fees_up_to_date":                      "Tuition Fees Current",
    "gender":                                       "Gender",
    "scholarship_holder":                           "Scholarship Holder",
    "age_at_enrollment":                            "Age at Enrollment",
    "international":                                "International Student",
    "curricular_units_1st_sem_credited":            "Sem 1 Credited",
    "curricular_units_1st_sem_enrolled":            "Sem 1 Enrolled",
    "curricular_units_1st_sem_evaluations":         "Sem 1 Evaluations",
    "curricular_units_1st_sem_approved":            "Sem 1 Approved",
    "curricular_units_1st_sem_grade":               "Sem 1 Grade",
    "curricular_units_1st_sem_without_evaluations": "Sem 1 Without Eval",
    "curricular_units_2nd_sem_credited":            "Sem 2 Credited",
    "curricular_units_2nd_sem_enrolled":            "Sem 2 Enrolled",
    "curricular_units_2nd_sem_evaluations":         "Sem 2 Evaluations",
    "curricular_units_2nd_sem_approved":            "Sem 2 Approved",
    "curricular_units_2nd_sem_grade":               "Sem 2 Grade",
    "curricular_units_2nd_sem_without_evaluations": "Sem 2 Without Eval",
    "unemployment_rate":                            "Unemployment Rate (%)",
    "inflation_rate":                               "Inflation Rate (%)",
    "gdp":                                          "GDP",
    "target":                                       "Outcome",
    # Engineered
    "avg_grade":                                    "Avg Grade (Both Sems)",
    "enrollment_efficiency":                        "Enrollment Efficiency",
    "total_approved_units":                         "Total Approved Units",
    "financial_stress_index":                       "Financial Stress Index",
    "dropout_prob":                                 "Dropout Probability",
    "risk_score":                                   "Risk Score",
    "risk_tier":                                    "Risk Tier",
    "cluster_label":                                "Segment",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Input  : {df.shape}")

    # Rename if raw column names present
    if "Marital Status" in df.columns:
        df = df.rename(columns=COL_RENAME)

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped duplicates: {before - len(df)}")

    # Drop rows with null target
    df = df.dropna(subset=["target"])

    # Numeric sanity filters
    df = df[df["age_at_enrollment"].between(15, 80)]
    df = df[df["admission_grade"].between(0, 200)]
    df = df[df["curricular_units_1st_sem_enrolled"] >= 0]
    df = df[df["curricular_units_2nd_sem_enrolled"] >= 0]

    # Fill rare nulls in numeric cols with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        nulls = df[c].isnull().sum()
        if nulls > 0:
            df[c] = df[c].fillna(df[c].median())
            print(f"  Filled {nulls} nulls in {c}")

    # Feature engineering
    df["target_binary"] = (df["target"] == "Dropout").astype(int)

    df["enrollment_efficiency"] = df.apply(
        lambda r: 0.0 if r["curricular_units_1st_sem_enrolled"] == 0
        else r["curricular_units_1st_sem_approved"] / r["curricular_units_1st_sem_enrolled"], axis=1)

    df["avg_grade"] = (
        df["curricular_units_1st_sem_grade"] + df["curricular_units_2nd_sem_grade"]) / 2.0

    df["total_approved_units"] = (
        df["curricular_units_1st_sem_approved"] + df["curricular_units_2nd_sem_approved"])

    unemp_norm = ((df["unemployment_rate"] - 7.0) / (16.2 - 7.0)).clip(0, 1)
    diseng = df.apply(lambda r: 0.0 if r["curricular_units_1st_sem_enrolled"] == 0
        else r["curricular_units_1st_sem_without_evaluations"] / r["curricular_units_1st_sem_enrolled"], axis=1)

    df["financial_stress_index"] = (
        df["debtor"].astype(float) * 0.30
        + (1 - df["tuition_fees_up_to_date"]).astype(float) * 0.25
        + (1 - df["scholarship_holder"]).astype(float) * 0.15
        + df["displaced"].astype(float) * 0.10
        + unemp_norm * 0.10 + diseng * 0.10
    )

    print(f"  Output : {df.shape}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


def main():
    print("[00_clean] Loading raw CSV...")
    # Prefer raw, fall back to clean
    src = RAW_CSV if RAW_CSV.exists() else CLEAN_CSV
    df = pd.read_csv(src)
    df = clean(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "students_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"[00_clean] Saved: {out}")

    # Also save display name mapping as JSON
    import json
    with open(OUT_DIR / "display_names.json", "w") as f:
        json.dump(DISPLAY_NAMES, f, indent=2)

    return df

if __name__ == "__main__":
    main()
    print("[00_clean] DONE")
