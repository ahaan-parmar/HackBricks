# Databricks notebook source
# TITLE: 06 — Gold Output
# Reads staging predictions → enriches → writes Gold Delta Tables (predictions + SHAP importance)

# COMMAND ----------
dbutils.widgets.text("catalog",       "main")
dbutils.widgets.text("schema_silver", "silver")
dbutils.widgets.text("schema_gold",   "gold")
dbutils.widgets.text("model_name",    "student_dropout_classifier")

catalog       = dbutils.widgets.get("catalog")
schema_silver = dbutils.widgets.get("schema_silver")
schema_gold   = dbutils.widgets.get("schema_gold")
model_name    = dbutils.widgets.get("model_name")

staging_table      = f"{catalog}.{schema_silver}.predictions_staging"
gold_pred_table    = f"{catalog}.{schema_gold}.student_dropout_predictions"
gold_shap_table    = f"{catalog}.{schema_gold}.shap_feature_importance"

print(f"staging_table   : {staging_table}")
print(f"gold_pred_table : {gold_pred_table}")
print(f"gold_shap_table : {gold_shap_table}")

# COMMAND ----------
from pyspark.sql.functions import (
    col, when, current_timestamp, lit,
    round as spark_round, abs as spark_abs
)
from pyspark.sql.types import DoubleType
import pandas as pd

# COMMAND ----------
# Create Gold schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_gold}")

# COMMAND ----------
# Read staging predictions (written by notebook 05)
df = spark.table(staging_table)
print(f"Staging rows: {df.count()}")

# COMMAND ----------
# ── Enrich predictions ───────────────────────────────────────────────────────

# Risk tier based on dropout probability
df = df.withColumn(
    "risk_tier",
    when(col("y_prob_dropout") >= 0.70, lit("HIGH"))
    .when(col("y_prob_dropout") >= 0.40, lit("MEDIUM"))
    .otherwise(lit("LOW"))
)

# Round probability to 4 dp
df = df.withColumn("y_prob_dropout", spark_round(col("y_prob_dropout").cast(DoubleType()), 4))
df = df.withColumn("fsi_value",      spark_round(col("fsi_value").cast(DoubleType()), 4))

# Add pipeline metadata
df = df.withColumn("prediction_timestamp", current_timestamp())
df = df.withColumn("model_name", lit(model_name))

# Try to get job run ID (only available inside a Databricks job)
try:
    run_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get()
except Exception:
    run_id = "interactive"
df = df.withColumn("pipeline_run_id", lit(str(run_id)))

# COMMAND ----------
# ── Write Gold Predictions Table ─────────────────────────────────────────────
(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("risk_tier")
    .saveAsTable(gold_pred_table)
)
print(f"Written: {gold_pred_table}")

# COMMAND ----------
# Optimize for query performance
spark.sql(f"OPTIMIZE {gold_pred_table} ZORDER BY (risk_tier, fsi_value)")
print("OPTIMIZE + ZORDER complete.")

# COMMAND ----------
# Register table comment
spark.sql(f"""
    COMMENT ON TABLE {gold_pred_table} IS
    'Gold layer: Student dropout predictions with risk tiers (HIGH/MEDIUM/LOW),
     dropout probability, Financial Stress Index, and per-feature SHAP values.
     Partitioned by risk_tier. Source: UCI Student Dropout dataset.'
""")

# COMMAND ----------
# ── Write Gold SHAP Importance Table ─────────────────────────────────────────
# Compute mean absolute SHAP per feature from all predictions
shap_col_names = [c for c in df.columns if c.startswith("shap_")]

shap_rows = []
for shap_col in shap_col_names:
    feature_name = shap_col.replace("shap_", "")
    mean_abs = df.select(spark_abs(col(shap_col)).cast(DoubleType())).agg({"abs(shap_" + shap_col.replace("shap_","") + ")": "avg"}).collect()
    # Simpler approach: use pandas
    shap_rows.append(feature_name)

# Use Pandas for aggregation (small table — 30 features)
shap_pd = df.select(shap_col_names).toPandas()
importance_rows = []
for shap_col in shap_col_names:
    importance_rows.append({
        "feature":       shap_col.replace("shap_", ""),
        "mean_abs_shap": round(float(shap_pd[shap_col].abs().mean()), 6),
        "model_name":    model_name,
        "computed_at":   str(pd.Timestamp.now()),
    })

importance_df = pd.DataFrame(importance_rows).sort_values("mean_abs_shap", ascending=False)

(
    spark.createDataFrame(importance_df).write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_shap_table)
)
print(f"Written: {gold_shap_table}")

# COMMAND ----------
# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n===== Gold Output Summary =====")

risk_summary = spark.table(gold_pred_table).groupBy("risk_tier").count().orderBy("risk_tier")
display(risk_summary)

print("\nRisk tier × Gender cross-tab:")
cross_tab = (
    spark.table(gold_pred_table)
    .groupBy("risk_tier",
             when(col("gender") == 1, "male").otherwise("female").alias("gender_label"))
    .count()
    .orderBy("risk_tier", "gender_label")
)
display(cross_tab)

print("\nTop 10 Features by SHAP importance:")
display(spark.table(gold_shap_table).orderBy(col("mean_abs_shap").desc()).limit(10))
