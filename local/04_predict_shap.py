"""
04_predict_shap.py - PREDICTIONS + SHAP (local)
Loads the best trained model, runs it on the full dataset,
computes SHAP values, and saves a gold predictions parquet.

Run: python local/04_predict_shap.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import shap

from config import SILVER_FILE, GOLD_DIR, FEATURE_COLS, LABEL_COL

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"
GOLD_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, list[str]]:
    csv_path = ROOT / "dataset" / "students_dropout_academic_success_clean.csv"

    if SILVER_FILE.exists():
        df = pd.read_parquet(SILVER_FILE)
    else:
        df = pd.read_csv(csv_path)
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

    available = [c for c in FEATURE_COLS if c in df.columns]
    return df.reset_index(drop=True), available


def main():
    meta_path = MODEL_DIR / "best_model_meta.json"
    model_path = MODEL_DIR / "best_model.pkl"

    if not model_path.exists():
        print("[ERROR] No trained model found. Run: python local/03_train_models.py")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    model_name    = meta["model_name"]
    feature_cols  = meta["feature_cols"]

    print(f"[04_predict] Model    : {model_name}")
    print(f"[04_predict] Features : {len(feature_cols)}")

    model = joblib.load(model_path)

    df, available = load_data()
    X = df[feature_cols].fillna(0)

    # ── Predictions ───────────────────────────────────────────────────────────
    print(f"[04_predict] Running predictions on {len(X)} rows...")
    prob_dropout = model.predict_proba(X)[:, 1]
    df["dropout_prob"]  = prob_dropout.round(4)
    df["risk_score"]    = (prob_dropout * 100).round(1)
    df["risk_tier"]     = pd.cut(
        prob_dropout,
        bins=[-0.001, 0.40, 0.70, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"]
    ).astype(str)

    print(f"  Risk tier counts:")
    print(f"  {df['risk_tier'].value_counts().to_dict()}")

    # ── SHAP values ───────────────────────────────────────────────────────────
    print(f"[04_predict] Computing SHAP values (this takes ~30s)...")

    if model_name == "xgboost":
        # XGBoost 3.x has native SHAP via pred_contribs — avoids library version conflict
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
        # pred_contribs returns shape (n, n_features + 1) — last col is bias
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        sv = contribs[:, :-1]   # drop bias column
    else:
        from sklearn.pipeline import Pipeline as SKPipeline
        if isinstance(model, SKPipeline):
            X_scaled    = model.named_steps["scaler"].transform(X)
            clf         = model.named_steps["clf"]
            explainer   = shap.LinearExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer   = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap_df = pd.DataFrame(sv, columns=[f"shap_{c}" for c in feature_cols])

    # Top-3 SHAP factors per student (feature names + values)
    def top3_shap(row):
        abs_vals = row.abs().nlargest(3)
        return [f.replace("shap_", "") for f in abs_vals.index.tolist()]

    print(f"[04_predict] Building per-student SHAP factor labels...")
    shap_top3 = shap_df.apply(top3_shap, axis=1)
    df["shap_top_features"] = shap_top3.apply(lambda x: x)

    # Global SHAP importance
    mean_abs_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n  Top 10 features by mean |SHAP|:")
    for feat, val in mean_abs_shap.head(10).items():
        print(f"    {feat:45s} {val:.4f}")

    # ── Save gold predictions ─────────────────────────────────────────────────
    gold_cols = (
        [c for c in df.columns if c not in [f"shap_{c}" for c in feature_cols]]
        + list(shap_df.columns)
    )
    out_df = pd.concat([df, shap_df], axis=1)
    out_path = GOLD_DIR / "student_predictions.parquet"
    out_df.to_parquet(out_path, index=False)

    # Save global SHAP importance separately
    shap_imp_path = GOLD_DIR / "shap_importance.parquet"
    mean_abs_shap.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    ).to_parquet(shap_imp_path, index=False)

    print(f"\n[04_predict] Saved predictions : {out_path}")
    print(f"[04_predict] Saved SHAP import : {shap_imp_path}")
    print(f"[04_predict] DONE")


if __name__ == "__main__":
    main()
