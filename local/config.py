"""
config.py — Shared configuration for local pipeline execution.
All paths are relative to the HackBricks project root.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent   # HackBricks/
DATA_DIR    = ROOT / "data"

RAW_DIR     = DATA_DIR / "raw"
BRONZE_DIR  = DATA_DIR / "bronze"
SILVER_DIR  = DATA_DIR / "silver"
GOLD_DIR    = DATA_DIR / "gold"
PLOTS_DIR   = DATA_DIR / "plots"

# ── Source file ───────────────────────────────────────────────────────────────
# Download from UCI and place here:
# https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
RAW_CSV     = RAW_DIR / "student_dropout.csv"

# ── Layer files ───────────────────────────────────────────────────────────────
BRONZE_FILE = BRONZE_DIR / "student_dropout_bronze.parquet"
SILVER_FILE = SILVER_DIR / "student_dropout_silver.parquet"

GOLD_PRED_FILE  = GOLD_DIR / "student_dropout_predictions.parquet"
GOLD_SHAP_FILE  = GOLD_DIR / "shap_feature_importance.parquet"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = (ROOT / "mlruns").as_uri()   # file:///C:/... safe on Windows
EXPERIMENT_NAME     = "student_dropout_experiment"
MODEL_NAME          = "student_dropout_classifier"

# ── Financial Stress Index constants ─────────────────────────────────────────
UNEMP_MIN = 7.0
UNEMP_MAX = 16.2

# ── Feature columns (must match across all scripts) ───────────────────────────
FEATURE_COLS = [
    "application_mode", "application_order", "course",
    "daytime_evening_attendance", "previous_qualification_grade",
    "admission_grade", "displaced", "educational_special_needs",
    "debtor", "tuition_fees_up_to_date", "gender",
    "scholarship_holder", "age_at_enrollment", "international",
    "curricular_units_1st_sem_enrolled", "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade", "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_enrolled", "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade", "curricular_units_2nd_sem_evaluations",
    "unemployment_rate", "inflation_rate", "gdp",
    "financial_stress_index",
    "enrollment_efficiency",
    "avg_grade",
    "total_approved_units",
]

LABEL_COL     = "target_binary"
SENSITIVE_COL = "gender"   # 1=male, 0=female
