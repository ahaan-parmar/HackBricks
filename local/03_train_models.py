"""
03_train_models.py - MODEL TRAINING (local)
Two models compete:
  1. Logistic Regression — features pre-selected by Random Forest importance
  2. XGBoost             — all features

Best by ROC-AUC is saved to disk and used by the backend.

Run: python local/03_train_models.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import mlflow, mlflow.sklearn, mlflow.xgboost
import joblib

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost                 import XGBClassifier

from config import (
    SILVER_FILE, MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME, FEATURE_COLS, LABEL_COL,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
TOP_N_FEATURES = 15   # how many RF-selected features LR gets


def load_silver() -> tuple[pd.DataFrame, list[str]]:
    ROOT = Path(__file__).resolve().parent.parent
    csv_path = ROOT / "dataset" / "students_dropout_academic_success_clean.csv"

    if SILVER_FILE.exists():
        print(f"  Source: silver parquet")
        df = pd.read_parquet(SILVER_FILE)
    elif csv_path.exists():
        print(f"  Silver parquet not found — falling back to clean CSV")
        df = pd.read_csv(csv_path)
        # Apply silver transforms inline
        df["target_binary"] = (df["target"] == "Dropout").astype(int)
        df["enrollment_efficiency"] = df.apply(
            lambda r: 0.0 if r["curricular_units_1st_sem_enrolled"] == 0
            else r["curricular_units_1st_sem_approved"] / r["curricular_units_1st_sem_enrolled"], axis=1)
        df["avg_grade"] = (df["curricular_units_1st_sem_grade"] + df["curricular_units_2nd_sem_grade"]) / 2.0
        df["total_approved_units"] = df["curricular_units_1st_sem_approved"] + df["curricular_units_2nd_sem_approved"]
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
    else:
        print("[ERROR] No data source found.")
        sys.exit(1)

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (will skip): {missing}")
    return df[available + [LABEL_COL]].dropna(), available


def metrics_for(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":    round(float(accuracy_score(y_test, y_pred)),  4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "roc_auc":     round(float(roc_auc_score(y_test, y_prob)),   4),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"[03_train] Loading silver: {SILVER_FILE}")
    df, feature_cols = load_silver()

    X = df[feature_cols]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Dropout rate: {y.mean():.2%}")

    dropout_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # ── Step 1: Random Forest feature selection ───────────────────────────────
    print(f"\n[03_train] Step 1 — Random Forest feature selection (top {TOP_N_FEATURES})")
    rf_selector = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_selector.fit(X_train, y_train)

    importances = pd.Series(rf_selector.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(TOP_N_FEATURES).index.tolist()

    print(f"  Top {TOP_N_FEATURES} features by RF importance:")
    for feat in top_features:
        print(f"    {feat:45s} {importances[feat]:.4f}")

    # ── Step 2: Logistic Regression on RF-selected features ───────────────────
    print(f"\n[03_train] Step 2 — Logistic Regression on RF-selected features")
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42
        )),
    ])
    lr_model.fit(X_train[top_features], y_train)
    lr_metrics = metrics_for(lr_model, X_test[top_features], y_test)

    print(f"  Accuracy : {lr_metrics['accuracy']}")
    print(f"  F1       : {lr_metrics['f1_weighted']}")
    print(f"  ROC-AUC  : {lr_metrics['roc_auc']}")

    with mlflow.start_run(run_name="logistic_regression_rf_selected"):
        mlflow.log_metrics(lr_metrics)
        mlflow.log_param("feature_selection", "random_forest")
        mlflow.log_param("n_features", TOP_N_FEATURES)
        mlflow.log_param("top_features", str(top_features))
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.log_text(
            classification_report(y_test, lr_model.predict(X_test[top_features]),
                                  target_names=["Not Dropout", "Dropout"]),
            "classification_report.txt"
        )
        mlflow.sklearn.log_model(lr_model, "model")

    # ── Step 3: XGBoost on all features ──────────────────────────────────────
    print(f"\n[03_train] Step 3 — XGBoost (all {len(feature_cols)} features)")
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=dropout_ratio,
        eval_metric="logloss", random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics = metrics_for(xgb_model, X_test, y_test)

    print(f"  Accuracy : {xgb_metrics['accuracy']}")
    print(f"  F1       : {xgb_metrics['f1_weighted']}")
    print(f"  ROC-AUC  : {xgb_metrics['roc_auc']}")

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_text(
            classification_report(y_test, xgb_model.predict(X_test),
                                  target_names=["Not Dropout", "Dropout"]),
            "classification_report.txt"
        )
        mlflow.xgboost.log_model(xgb_model, "model")

    # ── Step 4: Compare and pick best ────────────────────────────────────────
    print(f"\n[03_train] === Model Comparison ===")
    print(f"  {'Model':40s}  ROC-AUC   F1")
    print(f"  {'LR (RF-selected features)':40s}  {lr_metrics['roc_auc']:.4f}    {lr_metrics['f1_weighted']:.4f}")
    print(f"  {'XGBoost (all features)':40s}  {xgb_metrics['roc_auc']:.4f}    {xgb_metrics['f1_weighted']:.4f}")

    if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        best_name     = "xgboost"
        best_model    = xgb_model
        best_metrics  = xgb_metrics
        best_features = feature_cols
    else:
        best_name     = "logistic_regression_rf_selected"
        best_model    = lr_model
        best_metrics  = lr_metrics
        best_features = top_features

    print(f"\n  Winner: {best_name} (ROC-AUC={best_metrics['roc_auc']})")

    # ── Step 5: Save best model + metadata ───────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    # Always save RF selector for SHAP / feature importance reference
    joblib.dump(rf_selector, MODEL_DIR / "rf_selector.pkl")

    meta = {
        "model_name":    best_name,
        "feature_cols":  best_features,
        "all_features":  feature_cols,
        "top_rf_features": top_features,
        "metrics":       best_metrics,
    }
    with open(MODEL_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved model : data/models/best_model.pkl")
    print(f"  Saved RF    : data/models/rf_selector.pkl")
    print(f"  Saved meta  : data/models/best_model_meta.json")
    print(f"\n[03_train] DONE")

    return best_model, best_name, best_features


if __name__ == "__main__":
    main()
