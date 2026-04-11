# Databricks notebook source
# TITLE: 04 — Register Model
# Finds best MLflow run by ROC-AUC → promotes to Unity Catalog Model Registry as Champion

# COMMAND ----------
dbutils.widgets.text("catalog",         "main")
dbutils.widgets.text("experiment_name", "/Shared/student_dropout_experiment")
dbutils.widgets.text("model_name",      "student_dropout_classifier")

catalog         = dbutils.widgets.get("catalog")
experiment_name = dbutils.widgets.get("experiment_name")
model_name      = dbutils.widgets.get("model_name")

# Unity Catalog 3-level model name: catalog.schema.model
uc_model_name = f"{catalog}.default.{model_name}"

print(f"experiment_name : {experiment_name}")
print(f"uc_model_name   : {uc_model_name}")

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------
# Get experiment by name
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment '{experiment_name}' not found. Run notebook 03 first.")

print(f"Experiment ID: {experiment.experiment_id}")

# COMMAND ----------
# Search runs — pick best by ROC-AUC
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.roc_auc DESC"],
    max_results=10,
)

if not runs:
    raise ValueError("No finished runs found. Run notebook 03 first.")

best_run = runs[0]
best_run_id   = best_run.info.run_id
best_roc_auc  = best_run.data.metrics.get("roc_auc", 0)
best_model_type = best_run.data.tags.get("model_type", "unknown")

print(f"\nBest run:")
print(f"  run_id     : {best_run_id}")
print(f"  model_type : {best_model_type}")
print(f"  roc_auc    : {best_roc_auc:.4f}")
print(f"  accuracy   : {best_run.data.metrics.get('accuracy', 0):.4f}")
print(f"  f1_weighted: {best_run.data.metrics.get('f1_weighted', 0):.4f}")

# COMMAND ----------
# Register best model to Unity Catalog
model_uri = f"runs:/{best_run_id}/model"

print(f"\nRegistering '{model_uri}' → '{uc_model_name}' ...")
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=uc_model_name,
)
version = registered_model.version
print(f"Registered version: {version}")

# COMMAND ----------
# Wait for model version to be ready
import time
for _ in range(30):
    mv = client.get_model_version(uc_model_name, version)
    if mv.status == "READY":
        break
    print(f"  Status: {mv.status} — waiting ...")
    time.sleep(5)

print(f"Model version {version} is READY.")

# COMMAND ----------
# Set Champion alias (Unity Catalog uses aliases, not stages)
client.set_registered_model_alias(
    name=uc_model_name,
    alias="Champion",
    version=version,
)
print(f"Alias 'Champion' → version {version}")

# COMMAND ----------
# Add model description and tags
client.update_registered_model(
    name=uc_model_name,
    description=(
        "Student dropout classifier trained on UCI Student Dropout dataset. "
        "Predicts binary dropout risk (1=Dropout, 0=Not Dropout). "
        "Includes Financial Stress Index as a key engineered feature."
    ),
)

client.set_model_version_tag(uc_model_name, version, "best_roc_auc",      str(round(best_roc_auc, 4)))
client.set_model_version_tag(uc_model_name, version, "winning_model_type", best_model_type)
client.set_model_version_tag(uc_model_name, version, "trained_on",        "UCI_student_dropout")
client.set_model_version_tag(uc_model_name, version, "champion_since",    str(pd.Timestamp.now().date()))

# COMMAND ----------
import pandas as pd

print("\n===== Champion Model Summary =====")
summary = {
    "uc_model_name": uc_model_name,
    "version":       version,
    "alias":         "Champion",
    "model_type":    best_model_type,
    "roc_auc":       round(best_roc_auc, 4),
    "run_id":        best_run_id,
}
for k, v in summary.items():
    print(f"  {k:20s}: {v}")
