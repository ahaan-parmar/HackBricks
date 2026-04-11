# Databricks notebook source
# TITLE: 01 — Ingest Bronze
# Reads UCI Student Dropout CSV from Unity Catalog Volume → Bronze Delta Table

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema_bronze", "bronze")

catalog       = dbutils.widgets.get("catalog")
schema_bronze = dbutils.widgets.get("schema_bronze")

bronze_table  = f"{catalog}.{schema_bronze}.student_dropout_bronze"
volume_path   = f"/Volumes/{catalog}/default/raw_data/student_dropout.csv"

print(f"catalog      : {catalog}")
print(f"schema_bronze: {schema_bronze}")
print(f"bronze_table : {bronze_table}")
print(f"source file  : {volume_path}")

# COMMAND ----------
from pyspark.sql.functions import current_timestamp, lit, input_file_name

# COMMAND ----------
# Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_bronze}")
print(f"Schema '{catalog}.{schema_bronze}' ready.")

# COMMAND ----------
# Read UCI CSV
# The dataset uses semicolons as delimiter — DO NOT change
df_raw = (
    spark.read.format("csv")
    .option("header", "true")
    .option("sep", ";")
    .option("inferSchema", "true")
    .option("multiLine", "false")
    .load(volume_path)
)

print(f"Raw row count : {df_raw.count()}")
print(f"Raw columns   : {len(df_raw.columns)}")
df_raw.printSchema()

# COMMAND ----------
# Add ingestion metadata
df_bronze = (
    df_raw
    .withColumn("_ingest_timestamp", current_timestamp())
    .withColumn("_source_file", lit(volume_path))
)

# COMMAND ----------
# Write to Bronze Delta Table (overwrite for idempotency — append if incremental)
(
    df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(bronze_table)
)
print(f"Written to {bronze_table}")

# COMMAND ----------
# Register table comment in Unity Catalog
spark.sql(f"""
    COMMENT ON TABLE {bronze_table} IS
    'Bronze layer: raw UCI Student Dropout dataset. Source: UC Irvine ML Repository.
     Realinho et al. (2022). Ingested as-is with no transformations except metadata columns.'
""")

# COMMAND ----------
# Verify
count = spark.table(bronze_table).count()
print(f"Bronze table row count: {count}")
display(spark.table(bronze_table).limit(5))
