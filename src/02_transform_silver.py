# Databricks notebook source
# TITLE: 02 — Transform Silver
# Bronze → clean Silver Delta Table + Feature Engineering + Financial Stress Index

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema_bronze", "bronze")
dbutils.widgets.text("schema_silver", "silver")

catalog       = dbutils.widgets.get("catalog")
schema_bronze = dbutils.widgets.get("schema_bronze")
schema_silver = dbutils.widgets.get("schema_silver")

bronze_table = f"{catalog}.{schema_bronze}.student_dropout_bronze"
silver_table = f"{catalog}.{schema_silver}.student_dropout_silver"

print(f"bronze_table: {bronze_table}")
print(f"silver_table: {silver_table}")

# COMMAND ----------
from pyspark.sql.functions import (
    col, when, lit, current_timestamp,
    regexp_replace, lower, trim
)
from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------
# Read Bronze
df = spark.table(bronze_table)
print(f"Bronze row count: {df.count()}")

# COMMAND ----------
# ── Step 1: Rename columns ───────────────────────────────────────────────────
# Lowercase, replace spaces/parentheses/slashes with underscores
import re

def clean_col_name(name):
    name = name.lower().strip()
    name = re.sub(r"[\s\(\)/]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name

renamed = {c: clean_col_name(c) for c in df.columns}
for old, new in renamed.items():
    if old != new:
        df = df.withColumnRenamed(old, new)

print("Renamed columns:")
print(df.columns)

# COMMAND ----------
# ── Step 2: Cast types ───────────────────────────────────────────────────────
int_cols = [
    "marital_status", "application_mode", "application_order", "course",
    "daytime_evening_attendance", "previous_qualification",
    "nacionality", "mother_s_qualification", "father_s_qualification",
    "mother_s_occupation", "father_s_occupation", "displaced",
    "educational_special_needs", "debtor", "tuition_fees_up_to_date",
    "gender", "scholarship_holder", "age_at_enrollment", "international",
    "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited", "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_without_evaluations",
]
float_cols = [
    "previous_qualification_grade", "admission_grade",
    "curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade",
    "unemployment_rate", "inflation_rate", "gdp",
]

for c in int_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast(IntegerType()))

for c in float_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast(DoubleType()))

# COMMAND ----------
# ── Step 3: Drop nulls on critical columns, remove bad rows ──────────────────
df = df.dropna(subset=["target"])
df = df.filter(col("age_at_enrollment") > 0)
df = df.filter(col("curricular_units_1st_sem_enrolled") >= 0)

print(f"Row count after cleaning: {df.count()}")

# COMMAND ----------
# ── Step 4: Binary label ─────────────────────────────────────────────────────
# Target: "Dropout"=1, "Graduate"/"Enrolled"=0
df = df.withColumn(
    "target_binary",
    when(col("target") == "Dropout", 1).otherwise(0)
)

dropout_count = df.filter(col("target_binary") == 1).count()
total_count   = df.count()
print(f"Dropout rate: {dropout_count}/{total_count} = {dropout_count/total_count:.2%}")

# COMMAND ----------
# ── Step 5: Feature Engineering ─────────────────────────────────────────────

# 5a. Enrollment efficiency (% of enrolled units that were approved)
df = df.withColumn(
    "enrollment_efficiency",
    when(col("curricular_units_1st_sem_enrolled") == 0, lit(0.0))
    .otherwise(
        col("curricular_units_1st_sem_approved").cast(DoubleType()) /
        col("curricular_units_1st_sem_enrolled").cast(DoubleType())
    )
)

# 5b. Average grade across both semesters
df = df.withColumn(
    "avg_grade",
    (col("curricular_units_1st_sem_grade") + col("curricular_units_2nd_sem_grade")) / 2.0
)

# 5c. Total approved units across both semesters
df = df.withColumn(
    "total_approved_units",
    col("curricular_units_1st_sem_approved") + col("curricular_units_2nd_sem_approved")
)

# 5d. Academic disengagement: proportion of enrolled units with no evaluation
df = df.withColumn(
    "_disengagement",
    when(col("curricular_units_1st_sem_enrolled") == 0, lit(0.0))
    .otherwise(
        col("curricular_units_1st_sem_without_evaluations").cast(DoubleType()) /
        col("curricular_units_1st_sem_enrolled").cast(DoubleType())
    )
)

# 5e. Normalized unemployment (observed UCI range: ~7.0 – 16.2)
UNEMP_MIN = 7.0
UNEMP_MAX = 16.2
df = df.withColumn(
    "_unemp_norm",
    ((col("unemployment_rate") - UNEMP_MIN) / (UNEMP_MAX - UNEMP_MIN))
    .cast(DoubleType())
)
# Clip to [0, 1]
df = df.withColumn("_unemp_norm", when(col("_unemp_norm") < 0, 0.0)
                                  .when(col("_unemp_norm") > 1, 1.0)
                                  .otherwise(col("_unemp_norm")))

# COMMAND ----------
# ── Step 6: Financial Stress Index ───────────────────────────────────────────
#
# Composite weighted score in [0, 1]:
#   Debt burden         (debtor == 1)              × 0.30
#   Payment failure     (tuition_fees == 0)        × 0.25
#   No scholarship      (scholarship_holder == 0)  × 0.15
#   Displacement        (displaced == 1)            × 0.10
#   Macro unemployment  (normalised rate)           × 0.10
#   Academic disengage  (missed eval ratio)         × 0.10
#
# FSI = 0.0–0.3 → Low stress
# FSI = 0.3–0.6 → Moderate stress
# FSI > 0.6     → High stress (strong dropout risk signal)

df = df.withColumn(
    "financial_stress_index",
    (
        col("debtor").cast(DoubleType())                            * 0.30 +
        (1 - col("tuition_fees_up_to_date")).cast(DoubleType())     * 0.25 +
        (1 - col("scholarship_holder")).cast(DoubleType())          * 0.15 +
        col("displaced").cast(DoubleType())                         * 0.10 +
        col("_unemp_norm")                                          * 0.10 +
        col("_disengagement")                                       * 0.10
    )
)

# Drop temp columns
df = df.drop("_disengagement", "_unemp_norm")

# COMMAND ----------
# Sanity check: FSI distribution
print("FSI stats:")
df.select("financial_stress_index").summary("min", "25%", "50%", "75%", "max").show()

fsi_vs_dropout = df.groupBy(
    when(col("financial_stress_index") >= 0.6, "HIGH")
    .when(col("financial_stress_index") >= 0.3, "MEDIUM")
    .otherwise("LOW").alias("fsi_tier"),
    "target_binary"
).count().orderBy("fsi_tier", "target_binary")
display(fsi_vs_dropout)

# COMMAND ----------
# ── Step 7: Write Silver Delta Table ─────────────────────────────────────────
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_silver}")

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("target_binary")
    .saveAsTable(silver_table)
)
print(f"Written to {silver_table}")

# COMMAND ----------
# Register Unity Catalog comments
spark.sql(f"""
    COMMENT ON TABLE {silver_table} IS
    'Silver layer: cleaned UCI Student Dropout data with feature engineering.
     Includes Financial Stress Index (FSI), enrollment efficiency, avg_grade, and binary target label.'
""")

spark.sql(f"""
    ALTER TABLE {silver_table}
    ALTER COLUMN financial_stress_index
    COMMENT 'Composite financial stress score in [0,1]. Weights: debt(0.30), tuition_overdue(0.25),
             no_scholarship(0.15), displaced(0.10), unemployment(0.10), disengagement(0.10).
             >0.6=High stress, 0.3-0.6=Moderate, <0.3=Low.'
""")

# COMMAND ----------
print(f"Silver table row count: {spark.table(silver_table).count()}")
display(spark.table(silver_table).limit(5))
