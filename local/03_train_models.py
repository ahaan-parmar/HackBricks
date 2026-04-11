"""
03_train_models.py - XGBoost vs Logistic Regression
80/20 stratified split, full evaluation matrix, save best model.

Run: python local/03_train_models.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow, mlflow.sklearn, mlflow.xgboost
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"
PLOTS_DIR = ROOT / "data" / "plots"
PROC_DIR  = ROOT / "data" / "processed"

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
    "financial_stress_index", "enrollment_efficiency",
    "avg_grade", "total_approved_units",
]
LABEL_COL = "target_binary"


def load_data():
    clean = PROC_DIR / "students_clean.parquet"
    csv   = ROOT / "dataset" / "students_dropout_academic_success_clean.csv"
    if clean.exists():
        df = pd.read_parquet(clean)
    else:
        df = pd.read_csv(csv)
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
            + unemp_norm * 0.10 + diseng * 0.10)

    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols + [LABEL_COL]].dropna(), cols


def eval_metrics(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model":     name,
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, average="weighted"), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "y_prob":    y_prob,
        "y_pred":    y_pred,
        "cm":        cm,
    }


def plot_confusion_matrix(cm, name, path):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Dropout","Dropout"],
                yticklabels=["Not Dropout","Dropout"])
    ax.set_title(f"Confusion Matrix — {name}", fontsize=10)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc(results, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})", lw=2)
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("student_dropout_experiment")

    print("[03_train] Loading data...")
    df, feature_cols = load_data()
    X = df[feature_cols]
    y = df[LABEL_COL]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"  Train: {len(X_train)} ({y_train.mean():.1%} dropout)")
    print(f"  Test : {len(X_test)}  ({y_test.mean():.1%} dropout)")

    dropout_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    results = []

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n[03_train] Training Logistic Regression...")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                   solver="lbfgs", random_state=42)),
    ])
    lr.fit(X_train, y_train)
    lr_m = eval_metrics(lr, X_test, y_test, "Logistic Regression")
    lr_m["y_test"] = y_test.values
    results.append(lr_m)

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_metrics({k: v for k, v in lr_m.items() if k in ["accuracy","precision","recall","f1","roc_auc"]})
        mlflow.sklearn.log_model(lr, "model")

    print(f"  AUC={lr_m['roc_auc']}  F1={lr_m['f1']}  Acc={lr_m['accuracy']}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n[03_train] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=dropout_ratio,
        eval_metric="logloss", random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_m = eval_metrics(xgb, X_test, y_test, "XGBoost")
    xgb_m["y_test"] = y_test.values
    results.append(xgb_m)

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_metrics({k: v for k, v in xgb_m.items() if k in ["accuracy","precision","recall","f1","roc_auc"]})
        mlflow.xgboost.log_model(xgb, "model")

    print(f"  AUC={xgb_m['roc_auc']}  F1={xgb_m['f1']}  Acc={xgb_m['accuracy']}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n[03_train] ===== Evaluation Matrix =====")
    print(f"  {'Metric':12s}  {'Log Reg':>10s}  {'XGBoost':>10s}")
    for metric in ["accuracy","precision","recall","f1","roc_auc"]:
        print(f"  {metric:12s}  {lr_m[metric]:>10.4f}  {xgb_m[metric]:>10.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(lr_m["cm"],  "Logistic Regression", PLOTS_DIR / "cm_lr.png")
    plot_confusion_matrix(xgb_m["cm"], "XGBoost",             PLOTS_DIR / "cm_xgb.png")
    plot_roc(results, PLOTS_DIR / "roc_comparison.png")
    print(f"  Saved plots to {PLOTS_DIR}")

    # ── Save best model ───────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["roc_auc"])
    best_model = xgb if best["model"] == "XGBoost" else lr
    best_name  = best["model"].lower().replace(" ", "_")

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(xgb,        MODEL_DIR / "xgb_model.pkl")
    joblib.dump(lr,         MODEL_DIR / "lr_model.pkl")

    eval_out = {
        "winner":        best["model"],
        "feature_cols":  feature_cols,
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "logistic_regression": {k: v for k, v in lr_m.items()
                                 if k in ["accuracy","precision","recall","f1","roc_auc","tp","fp","tn","fn"]},
        "xgboost":             {k: v for k, v in xgb_m.items()
                                 if k in ["accuracy","precision","recall","f1","roc_auc","tp","fp","tn","fn"]},
    }
    with open(MODEL_DIR / "eval_matrix.json", "w") as f:
        json.dump(eval_out, f, indent=2)

    # Save XGBoost feature importance
    imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    imp.reset_index().rename(columns={"index":"feature",0:"importance"}).to_parquet(
        ROOT / "data" / "gold" / "feature_importance.parquet" if (ROOT/"data"/"gold").exists()
        else MODEL_DIR / "feature_importance.parquet", index=False)

    print(f"\n  Winner: {best['model']}  (AUC={best['roc_auc']})")
    print(f"  Saved:  data/models/best_model.pkl + eval_matrix.json")
    print("[03_train] DONE")

    return xgb, lr, feature_cols, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
