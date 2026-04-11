# Databricks notebook source
# TITLE: 03 — Train Models
# Logistic Regression vs XGBoost — log all metrics + artifacts to MLflow

# COMMAND ----------
dbutils.widgets.text("catalog",          "main")
dbutils.widgets.text("schema_silver",    "silver")
dbutils.widgets.text("experiment_name",  "/Shared/student_dropout_experiment")
dbutils.widgets.text("model_name",       "student_dropout_classifier")

catalog         = dbutils.widgets.get("catalog")
schema_silver   = dbutils.widgets.get("schema_silver")
experiment_name = dbutils.widgets.get("experiment_name")
model_name      = dbutils.widgets.get("model_name")

silver_table = f"{catalog}.{schema_silver}.student_dropout_silver"
print(f"silver_table    : {silver_table}")
print(f"experiment_name : {experiment_name}")

# COMMAND ----------
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import json

from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics        import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

# COMMAND ----------
# Feature columns — includes the FSI derived feature
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
LABEL_COL = "target_binary"

# COMMAND ----------
# Load Silver → Pandas (UCI dataset ~4400 rows, fits in driver memory)
df_spark = spark.table(silver_table).select(FEATURE_COLS + [LABEL_COL])
df = df_spark.toPandas().dropna()

X = df[FEATURE_COLS]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")
print(f"Dropout %  : {y.mean():.2%}")

# COMMAND ----------
# Set experiment
mlflow.set_experiment(experiment_name)

# COMMAND ----------
# ── Run 1: Logistic Regression ───────────────────────────────────────────────
with mlflow.start_run(run_name="logistic_regression") as lr_run:
    lr_params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "class_weight": "balanced"}

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**lr_params, random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)

    y_pred_lr  = lr_pipeline.predict(X_test)
    y_prob_lr  = lr_pipeline.predict_proba(X_test)[:, 1]

    metrics_lr = {
        "accuracy":   accuracy_score(y_test, y_pred_lr),
        "f1_weighted": f1_score(y_test, y_pred_lr, average="weighted"),
        "roc_auc":    roc_auc_score(y_test, y_prob_lr),
    }

    mlflow.log_params(lr_params)
    mlflow.log_metrics(metrics_lr)
    mlflow.set_tags({"model_type": "logistic_regression", "dataset": "student_dropout"})

    # Log model
    mlflow.sklearn.log_model(lr_pipeline, "model", input_example=X_test.iloc[:5])

    # Log feature importance (coefficients)
    coef = lr_pipeline.named_steps["clf"].coef_[0]
    coef_dict = dict(zip(FEATURE_COLS, coef.tolist()))
    mlflow.log_dict(coef_dict, "feature_coefficients.json")

    # Log classification report
    report = classification_report(y_test, y_pred_lr, target_names=["Not Dropout", "Dropout"])
    mlflow.log_text(report, "classification_report.txt")

    lr_run_id = lr_run.info.run_id
    print(f"LR  run_id : {lr_run_id}")
    print(f"LR  metrics: {metrics_lr}")

# COMMAND ----------
# ── Run 2: XGBoost ───────────────────────────────────────────────────────────
with mlflow.start_run(run_name="xgboost") as xgb_run:
    # scale_pos_weight handles class imbalance
    dropout_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_params = {
        "n_estimators":    200,
        "max_depth":       4,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": dropout_ratio,
        "eval_metric":     "logloss",
        "use_label_encoder": False,
    }

    xgb_model = XGBClassifier(**xgb_params, random_state=42)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    metrics_xgb = {
        "accuracy":    accuracy_score(y_test, y_pred_xgb),
        "f1_weighted": f1_score(y_test, y_pred_xgb, average="weighted"),
        "roc_auc":     roc_auc_score(y_test, y_prob_xgb),
    }

    mlflow.log_params(xgb_params)
    mlflow.log_metrics(metrics_xgb)
    mlflow.set_tags({"model_type": "xgboost", "dataset": "student_dropout"})

    # Log model
    mlflow.xgboost.log_model(xgb_model, "model", input_example=X_test.iloc[:5])

    # Log feature importance
    importance = xgb_model.get_booster().get_score(importance_type="gain")
    mlflow.log_dict(importance, "feature_importance_gain.json")

    report = classification_report(y_test, y_pred_xgb, target_names=["Not Dropout", "Dropout"])
    mlflow.log_text(report, "classification_report.txt")

    xgb_run_id = xgb_run.info.run_id
    print(f"XGB run_id : {xgb_run_id}")
    print(f"XGB metrics: {metrics_xgb}")

# COMMAND ----------
# ── Comparison summary ────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "model":      ["Logistic Regression", "XGBoost"],
    "accuracy":   [metrics_lr["accuracy"],    metrics_xgb["accuracy"]],
    "f1_weighted":[metrics_lr["f1_weighted"], metrics_xgb["f1_weighted"]],
    "roc_auc":    [metrics_lr["roc_auc"],     metrics_xgb["roc_auc"]],
    "run_id":     [lr_run_id,                 xgb_run_id],
})
print("\n===== Model Comparison =====")
print(summary.to_string(index=False))
display(summary)
