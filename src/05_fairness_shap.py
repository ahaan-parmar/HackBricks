# Databricks notebook source
# TITLE: 05 — Fairness Audit + SHAP Explainability
# Loads Champion model → Fairlearn audit by gender → SHAP explainability → staging predictions

# COMMAND ----------
dbutils.widgets.text("catalog",       "main")
dbutils.widgets.text("schema_silver", "silver")
dbutils.widgets.text("model_name",    "student_dropout_classifier")

catalog       = dbutils.widgets.get("catalog")
schema_silver = dbutils.widgets.get("schema_silver")
model_name    = dbutils.widgets.get("model_name")

uc_model_name   = f"{catalog}.default.{model_name}"
silver_table    = f"{catalog}.{schema_silver}.student_dropout_silver"
staging_table   = f"{catalog}.{schema_silver}.predictions_staging"

print(f"Champion model : {uc_model_name}@Champion")
print(f"silver_table   : {silver_table}")
print(f"staging_table  : {staging_table}")

# COMMAND ----------
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics        import accuracy_score
from fairlearn.metrics      import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)

# COMMAND ----------
# Feature columns — must match notebook 03
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
SENSITIVE_COL = "gender"   # 1=male, 0=female in UCI dataset

# COMMAND ----------
# Load Champion model
model_uri = f"models:/{uc_model_name}@Champion"
print(f"Loading model from: {model_uri}")

try:
    model = mlflow.sklearn.load_model(model_uri)
    model_type = "sklearn"
except Exception:
    model = mlflow.xgboost.load_model(model_uri)
    model_type = "xgboost"

print(f"Loaded model type: {model_type}")

# COMMAND ----------
# Load Silver data
df = spark.table(silver_table).select(FEATURE_COLS + [LABEL_COL, SENSITIVE_COL]).toPandas().dropna()

X      = df[FEATURE_COLS]
y_true = df[LABEL_COL]
gender = df[SENSITIVE_COL]

# Predict
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print(f"Rows evaluated: {len(df)}")
print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")

# COMMAND ----------
# ── Fairness Audit ───────────────────────────────────────────────────────────
metrics_dict = {
    "accuracy":       accuracy_score,
    "selection_rate": selection_rate,
}

metric_frame = MetricFrame(
    metrics=metrics_dict,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=gender,
)

dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=gender)
eod = equalized_odds_difference(y_true, y_pred, sensitive_features=gender)

print("\n===== Fairness Report =====")
print(f"Demographic Parity Difference : {dpd:.4f}  (0 = perfect parity)")
print(f"Equalized Odds Difference     : {eod:.4f}  (0 = perfect parity)")
print("\nMetrics by gender (0=female, 1=male):")
display(metric_frame.by_group)

# COMMAND ----------
# Log fairness metrics to MLflow
with mlflow.start_run(run_name="fairness_audit") as fairness_run:
    mlflow.set_tag("run_type", "fairness_audit")
    mlflow.set_tag("model_name", uc_model_name)
    mlflow.log_metric("demographic_parity_difference", round(dpd, 4))
    mlflow.log_metric("equalized_odds_difference",     round(eod, 4))

    # Log per-group metrics
    for col_name in metric_frame.by_group.columns:
        for gender_val, val in metric_frame.by_group[col_name].items():
            mlflow.log_metric(f"{col_name}_gender_{int(gender_val)}", round(val, 4))

    # Log fairness report as text
    report_text = f"""
Fairness Audit Report
=====================
Model       : {uc_model_name}@Champion
Sensitive   : gender (0=female, 1=male)

Demographic Parity Difference : {dpd:.4f}
Equalized Odds Difference     : {eod:.4f}

By Group:
{metric_frame.by_group.to_string()}
"""
    mlflow.log_text(report_text, "fairness_report.txt")
    fairness_run_id = fairness_run.info.run_id
    print(f"\nFairness run_id: {fairness_run_id}")

# COMMAND ----------
# ── SHAP Explainability ──────────────────────────────────────────────────────
print("Computing SHAP values ...")

if model_type == "sklearn":
    # StandardScaler is first step — transform X before SHAP
    X_scaled = model.named_steps["scaler"].transform(X)
    clf      = model.named_steps["clf"]
    explainer    = shap.LinearExplainer(clf, X_scaled, feature_names=FEATURE_COLS)
    shap_values  = explainer.shap_values(X_scaled)
    X_for_plot   = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
else:
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X)
    X_for_plot   = X

# Handle multi-output (binary classifiers sometimes return list)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"SHAP values shape: {shap_values.shape}")

# COMMAND ----------
# SHAP summary plot (beeswarm)
fig_summary, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_for_plot, feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
summary_plot_path = "/tmp/shap_summary.png"
plt.savefig(summary_plot_path, bbox_inches="tight", dpi=150)
plt.close()

# SHAP waterfall for highest-risk student
highest_risk_idx = int(np.argmax(y_prob))
fig_waterfall, ax2 = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[highest_risk_idx],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
        data=X_for_plot.iloc[highest_risk_idx],
        feature_names=FEATURE_COLS,
    ),
    show=False,
)
plt.tight_layout()
waterfall_plot_path = "/tmp/shap_waterfall_highest_risk.png"
plt.savefig(waterfall_plot_path, bbox_inches="tight", dpi=150)
plt.close()

print(f"Plots saved to /tmp/")

# COMMAND ----------
# Log SHAP plots to MLflow fairness run
with mlflow.start_run(run_id=fairness_run_id):
    mlflow.log_artifact(summary_plot_path,   "shap_summary.png")
    mlflow.log_artifact(waterfall_plot_path, "shap_waterfall_highest_risk.png")

# COMMAND ----------
# Mean absolute SHAP importance per feature
shap_importance = pd.DataFrame({
    "feature":          FEATURE_COLS,
    "mean_abs_shap":    np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 most important features by SHAP:")
display(shap_importance.head(10))

# COMMAND ----------
# ── Build staging predictions DataFrame ─────────────────────────────────────
shap_cols_df = pd.DataFrame(
    shap_values,
    columns=[f"shap_{c}" for c in FEATURE_COLS],
)

output_df = df.copy()
output_df["y_pred"]           = y_pred
output_df["y_prob_dropout"]   = y_prob
output_df["fsi_value"]        = df["financial_stress_index"]
output_df = pd.concat([output_df.reset_index(drop=True), shap_cols_df], axis=1)

# Write staging table for notebook 06
spark.createDataFrame(output_df).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(staging_table)

print(f"\nStaging predictions written to: {staging_table}")
print(f"Rows: {len(output_df)}")
