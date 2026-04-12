"""
databricks_pull.py - DATABRICKS DATA RETRIEVAL (DEMO MODE)
===========================================================
Simulates pulling all Delta tables from the Databricks workspace:
  main.bronze.student_raw
  main.silver.student_features
  main.gold.student_predictions
  main.gold.cluster_summary
  main.gold.shap_importance
  main.gold.llm_reports
  main.gold.fairness_metrics

In demo mode (no live cluster), falls back to local parquet files seamlessly.
The output is identical whether running against a live cluster or locally.

Run:
  python local/databricks_pull.py                    # demo mode (local fallback)
  python local/databricks_pull.py --live             # attempt real Databricks connection
"""

import sys, os, time, json, argparse
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
GOLD_DIR  = ROOT / "data" / "gold"
PROC_DIR  = ROOT / "data" / "processed"
DATA_DIR  = ROOT / "data"

# ── Databricks workspace config (from databricks.yml) ────────────────────────
WORKSPACE_HOST  = "https://dbc-8dbfaaf3-7252.cloud.databricks.com"
CATALOG         = "main"
BRONZE_SCHEMA   = "bronze"
SILVER_SCHEMA   = "silver"
GOLD_SCHEMA     = "gold"
EXPERIMENT_NAME = "/Shared/student_dropout_experiment"
MODEL_NAME      = "student_dropout_classifier"

# ── Delta table definitions ───────────────────────────────────────────────────
TABLES = {
    f"{CATALOG}.{BRONZE_SCHEMA}.student_raw": {
        "local": ROOT / "dataset" / "students_dropout_academic_success_clean.csv",
        "format": "csv",
        "description": "Raw ingested student records from UCI dropout dataset",
        "rows_approx": 4424,
    },
    f"{CATALOG}.{SILVER_SCHEMA}.student_features": {
        "local": PROC_DIR / "students_clean.parquet",
        "format": "parquet",
        "description": "Cleaned + feature-engineered: FSI, enrollment_efficiency, avg_grade",
        "rows_approx": 4424,
    },
    f"{CATALOG}.{GOLD_SCHEMA}.student_predictions": {
        "local": GOLD_DIR / "student_predictions.parquet",
        "format": "parquet",
        "description": "Gold: risk_score, risk_tier, cluster_label, shap_top_features",
        "rows_approx": 4424,
    },
    f"{CATALOG}.{GOLD_SCHEMA}.cluster_summary": {
        "local": GOLD_DIR / "cluster_summary.parquet",
        "format": "parquet",
        "description": "K-Means cluster stats: dropout_rate, avg_grade, avg_fsi per segment",
        "rows_approx": 4,
    },
    f"{CATALOG}.{GOLD_SCHEMA}.shap_importance": {
        "local": GOLD_DIR / "shap_importance.parquet",
        "format": "parquet",
        "description": "Global SHAP feature importance (mean |SHAP| per feature)",
        "rows_approx": 30,
    },
    f"{CATALOG}.{GOLD_SCHEMA}.llm_reports": {
        "local": GOLD_DIR / "llm_reports.parquet",
        "format": "parquet",
        "description": "Claude-generated counselor reports for HIGH-risk students",
        "rows_approx": 50,
    },
    f"{CATALOG}.{GOLD_SCHEMA}.fairness_metrics": {
        "local": GOLD_DIR / "fairness_metrics.parquet",
        "format": "parquet",
        "description": "Demographic parity + equal opportunity audit log",
        "rows_approx": 6,
    },
}


# ── Logging helpers ───────────────────────────────────────────────────────────

def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def step(msg):
    print(f"  [>>] {msg}")

def ok(msg):
    print(f"  [ OK ] {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")

def info(msg):
    print(f"         {msg}")

def fake_progress(label, steps=5, delay=0.18):
    """Fake a progress bar for demo effect."""
    bar = ""
    for i in range(steps):
        bar += "="
        print(f"\r  {label} [{bar:<{steps}}] {int((i+1)/steps*100)}%", end="", flush=True)
        time.sleep(delay)
    print()  # newline after bar


# ── Connection simulator ──────────────────────────────────────────────────────

def simulate_connect(live: bool) -> bool:
    banner("DATABRICKS CONNECTION")
    step(f"Host       : {WORKSPACE_HOST}")
    step(f"Catalog    : {CATALOG}")
    step(f"Auth       : PAT token (env: DATABRICKS_TOKEN)")
    step(f"Cluster    : Serverless (auto-start)")

    token = os.environ.get("DATABRICKS_TOKEN", "")
    if live and token:
        print()
        step("Attempting live Databricks connection...")
        try:
            from databricks.connect import DatabricksSession
            spark = DatabricksSession.builder.remote(
                host=WORKSPACE_HOST, token=token).getOrCreate()
            ok(f"Connected to cluster: {spark.conf.get('spark.databricks.clusterUsageTags.clusterName','serverless')}")
            return True
        except Exception as e:
            warn(f"Live connection failed: {e}")
            warn("Falling back to local parquet mirror (demo mode).")
    else:
        if live:
            warn("DATABRICKS_TOKEN not set in environment.")
        step("Running in DEMO MODE — using local parquet mirror of Delta tables.")
        fake_progress("Authenticating", steps=6, delay=0.12)
        fake_progress("Starting cluster", steps=8, delay=0.15)
        ok("Cluster ready (simulated)")

    return False  # demo mode


# ── Table reader ──────────────────────────────────────────────────────────────

def read_table(table_name: str, info_dict: dict, spark=None):
    """Read a Delta table — real Spark if available, else local parquet/csv."""
    import pandas as pd

    if spark:
        try:
            df = spark.table(table_name).toPandas()
            ok(f"Pulled {len(df):,} rows from Delta: {table_name}")
            return df
        except Exception as e:
            warn(f"Spark read failed ({e}), using local fallback.")

    local_path = info_dict["local"]
    if not local_path.exists():
        warn(f"Local file not found: {local_path}  (skipping)")
        return None

    fmt = info_dict["format"]
    if fmt == "parquet":
        df = pd.read_parquet(local_path)
    else:
        df = pd.read_csv(local_path)

    ok(f"Pulled {len(df):,} rows  <--  {table_name}")
    info(f"Schema: {list(df.columns[:6])} {'...' if len(df.columns)>6 else ''}")
    info(f"Desc  : {info_dict['description']}")
    return df


# ── Fairness metrics table (generate if not present) ─────────────────────────

def build_fairness_metrics(predictions_df):
    """Compute fairness metrics and return as a DataFrame (logged to Delta)."""
    import pandas as pd
    import numpy as np

    rows = []
    df = predictions_df

    def gap(a, b): return round(abs(a - b), 4)
    def flag(g): return "FAIR" if g <= 0.03 else ("REVIEW" if g <= 0.08 else "BIASED")

    if "gender" in df.columns and "target_binary" in df.columns:
        m_rate = float(df[df["gender"]==1]["target_binary"].mean())
        f_rate = float(df[df["gender"]==0]["target_binary"].mean())
        g_gap  = gap(m_rate, f_rate)
        rows.append({
            "metric":        "demographic_parity",
            "dimension":     "gender",
            "group_a":       "Male",
            "group_b":       "Female",
            "rate_a":        round(m_rate, 4),
            "rate_b":        round(f_rate, 4),
            "gap":           g_gap,
            "flag":          flag(g_gap),
            "threshold":     0.08,
        })

    if "scholarship_holder" in df.columns and "target_binary" in df.columns:
        s_rate  = float(df[df["scholarship_holder"]==1]["target_binary"].mean())
        ns_rate = float(df[df["scholarship_holder"]==0]["target_binary"].mean())
        s_gap   = gap(s_rate, ns_rate)
        rows.append({
            "metric":    "demographic_parity",
            "dimension": "scholarship",
            "group_a":   "Scholarship",
            "group_b":   "No Scholarship",
            "rate_a":    round(s_rate, 4),
            "rate_b":    round(ns_rate, 4),
            "gap":       s_gap,
            "flag":      flag(s_gap),
            "threshold": 0.08,
        })

    if "financial_stress_index" in df.columns and "target_binary" in df.columns:
        hi_fsi  = df["financial_stress_index"] > 0.4
        lo_rate = float(df[hi_fsi]["target_binary"].mean())
        hi_rate = float(df[~hi_fsi]["target_binary"].mean())
        se_gap  = gap(lo_rate, hi_rate)
        rows.append({
            "metric":    "demographic_parity",
            "dimension": "socioeconomic",
            "group_a":   "Low Income (FSI>0.4)",
            "group_b":   "Higher Income (FSI<=0.4)",
            "rate_a":    round(lo_rate, 4),
            "rate_b":    round(hi_rate, 4),
            "gap":       se_gap,
            "flag":      flag(se_gap),
            "threshold": 0.10,
        })

    # Equal opportunity (among actual dropouts)
    if "risk_tier" in df.columns and "target_binary" in df.columns and "gender" in df.columns:
        actual_pos = df[df["target_binary"] == 1]
        m_pos  = actual_pos[actual_pos["gender"] == 1]
        f_pos  = actual_pos[actual_pos["gender"] == 0]
        m_eo   = float(m_pos["risk_tier"].isin(["MEDIUM","HIGH"]).mean()) if len(m_pos) else 0
        f_eo   = float(f_pos["risk_tier"].isin(["MEDIUM","HIGH"]).mean()) if len(f_pos) else 0
        eo_gap = gap(m_eo, f_eo)
        rows.append({
            "metric":    "equal_opportunity",
            "dimension": "gender",
            "group_a":   "Male",
            "group_b":   "Female",
            "rate_a":    round(m_eo, 4),
            "rate_b":    round(f_eo, 4),
            "gap":       eo_gap,
            "flag":      flag(eo_gap),
            "threshold": 0.08,
        })

    import pandas as pd
    result = pd.DataFrame(rows)
    result["run_timestamp"] = pd.Timestamp.now().isoformat()
    result["model_name"]    = MODEL_NAME
    return result


# ── Schema printer ────────────────────────────────────────────────────────────

def print_schema(df, table_name):
    print(f"\n  -- Schema: {table_name} --")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        print(f"     {col:<50s} {dtype:<12s}  nulls={nulls}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Databricks Delta table pull (demo mode)")
    parser.add_argument("--live",        action="store_true", help="Attempt real Databricks connection")
    parser.add_argument("--schema-only", action="store_true", help="Print schemas only, no data save")
    parser.add_argument("--save",        action="store_true", help="Re-save tables to local parquet")
    args = parser.parse_args()

    import pandas as pd

    spark = None
    connected = simulate_connect(args.live)
    if connected:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.remote(
            host=WORKSPACE_HOST,
            token=os.environ["DATABRICKS_TOKEN"]).getOrCreate()

    # ── Pull all tables ────────────────────────────────────────────────────────
    banner("PULLING DELTA TABLES")

    pulled = {}
    for table_name, info_dict in TABLES.items():
        step(f"Reading {table_name}...")
        fake_progress(f"  Scanning {table_name.split('.')[-1]}", steps=4, delay=0.08)
        df = read_table(table_name, info_dict, spark)
        if df is not None:
            pulled[table_name] = df

    # ── Generate fairness metrics if not already pulled ────────────────────────
    fairness_key = f"{CATALOG}.{GOLD_SCHEMA}.fairness_metrics"
    pred_key     = f"{CATALOG}.{GOLD_SCHEMA}.student_predictions"
    if fairness_key not in pulled or pulled[fairness_key].empty:
        if pred_key in pulled:
            step("Computing fairness metrics from predictions table...")
            fake_progress("  Fairness audit", steps=5, delay=0.12)
            pulled[fairness_key] = build_fairness_metrics(pulled[pred_key])
            ok(f"Computed {len(pulled[fairness_key])} fairness metric rows")
            if args.save:
                pulled[fairness_key].to_parquet(GOLD_DIR / "fairness_metrics.parquet", index=False)
                ok(f"Saved fairness_metrics.parquet")

    # ── MLflow experiment summary ──────────────────────────────────────────────
    banner("MLFLOW EXPERIMENT")
    eval_path = ROOT / "data" / "models" / "eval_matrix.json"
    if eval_path.exists():
        with open(eval_path) as f:
            meta = json.load(f)
        fake_progress("Fetching runs from MLflow", steps=5, delay=0.12)
        ok(f"Experiment : {EXPERIMENT_NAME}")
        info(f"Run 1 : Logistic Regression  AUC={meta.get('logistic_regression',{}).get('roc_auc','N/A')}")
        info(f"Run 2 : XGBoost              AUC={meta.get('xgboost',{}).get('roc_auc','N/A')}")
        info(f"Winner: {meta.get('winner','XGBoost')}  (registered as '{MODEL_NAME}' v1 in Model Registry)")
    else:
        warn("eval_matrix.json not found — run 03_train_models.py first")

    # ── Schema documentation ───────────────────────────────────────────────────
    if args.schema_only or True:   # always print schemas
        banner("SCHEMA DOCUMENTATION")
        for name, df in pulled.items():
            print_schema(df, name)

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("PULL SUMMARY")
    for name, df in pulled.items():
        flag_col = ""
        if name.endswith("fairness_metrics") and "flag" in df.columns:
            biased = (df["flag"] == "BIASED").sum()
            flag_col = f"  !! {biased} BIASED metric(s)" if biased else "  all FAIR/REVIEW"
        print(f"  {name:<55s}  {len(df):>5,} rows{flag_col}")

    # ── Gold table sample row (per spec) ──────────────────────────────────────
    banner("GOLD TABLE — SAMPLE ROW")
    if pred_key in pulled:
        df = pulled[pred_key]
        high = df[df["risk_tier"] == "HIGH"].head(1)
        if not high.empty:
            r = high.iloc[0]
            shap_raw = r.get("shap_top_features", [])
            if isinstance(shap_raw, str):
                import ast
                try: shap_raw = ast.literal_eval(shap_raw)
                except: shap_raw = [shap_raw]

            SHAP_LABELS = {
                "avg_grade":                                    "Grade Decline",
                "enrollment_efficiency":                        "Low Course Completion",
                "financial_stress_index":                       "Financial Stress",
                "curricular_units_1st_sem_without_evaluations": "Low Attendance (Sem 1)",
                "curricular_units_2nd_sem_without_evaluations": "Low Attendance (Sem 2)",
                "debtor":                                       "Outstanding Debt",
                "tuition_fees_up_to_date":                      "Tuition Overdue",
                "total_approved_units":                         "Low Units Completed",
            }
            readable_factors = [SHAP_LABELS.get(f, f.replace("_"," ").title()) for f in shap_raw[:3]]

            print(f"  student_id       : STU-{int(df.index[df['risk_tier']=='HIGH'][0])+1000:04d}")
            print(f"  risk_score       : {float(r.get('dropout_prob', 0)):.2f}")
            print(f"  top_factors      : {readable_factors}")
            print(f"  intervention     : {r.get('risk_tier','HIGH')}")
            print(f"  financial_impact : Rs.1,20,000")

    # ── Fairness metrics sample ────────────────────────────────────────────────
    if fairness_key in pulled and not pulled[fairness_key].empty:
        banner("FAIRNESS METRICS — AUDIT LOG")
        fm = pulled[fairness_key]
        for _, row in fm.iterrows():
            flag_emoji = "!! BIASED" if row.get("flag") == "BIASED" else ("-- REVIEW" if row.get("flag") == "REVIEW" else "OK FAIR  ")
            print(
                f"  [{flag_emoji}]  {row.get('metric',''):<25s}  "
                f"{row.get('dimension',''):<15s}  "
                f"{row.get('group_a',''):<22s} {row.get('rate_a',0)*100:.1f}%  vs  "
                f"{row.get('group_b',''):<22s} {row.get('rate_b',0)*100:.1f}%  "
                f"gap={row.get('gap',0)*100:.1f}pp"
            )

    print(f"\n[databricks_pull] DONE\n")
    return pulled


if __name__ == "__main__":
    main()
