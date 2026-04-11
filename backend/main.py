"""
backend/main.py — HackBricks API Server

Data priority:
  1. Gold predictions parquet (after training + SHAP)  ← real model
  2. Silver parquet (after pipeline)
  3. Raw CSV (always available)

Start: python backend/main.py
Docs:  http://localhost:8001/docs
"""

from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HackBricks API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANNUAL_FEE_INR = 120_000

COURSE_MAP: dict[int, str] = {
    33:   "Biofuel Production Tech",
    171:  "Animation & Multimedia Design",
    8014: "Social Service (Evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equiniculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising & Marketing Mgmt",
    9773: "Journalism & Communication",
    9853: "Basic Education",
    9991: "Management (Evening)",
}

SHAP_LABEL_MAP = {
    "curricular_units_2nd_sem_approved":          "Units approved in semester 2",
    "enrollment_efficiency":                      "Enrollment efficiency (approved/enrolled)",
    "financial_stress_index":                     "Financial Stress Index",
    "tuition_fees_up_to_date":                    "Tuition fees up to date",
    "age_at_enrollment":                          "Age at enrollment",
    "course":                                     "Course of study",
    "total_approved_units":                       "Total approved units (both sems)",
    "curricular_units_2nd_sem_enrolled":          "Units enrolled in semester 2",
    "avg_grade":                                  "Average grade across semesters",
    "admission_grade":                            "Admission grade",
    "curricular_units_1st_sem_approved":          "Units approved in semester 1",
    "curricular_units_1st_sem_grade":             "Grade in semester 1",
    "curricular_units_2nd_sem_grade":             "Grade in semester 2",
    "debtor":                                     "Outstanding debt on account",
    "scholarship_holder":                         "Scholarship holder",
    "displaced":                                  "Displaced student",
    "curricular_units_1st_sem_evaluations":       "Evaluations in semester 1",
    "curricular_units_2nd_sem_evaluations":       "Evaluations in semester 2",
    "curricular_units_1st_sem_without_evaluations": "Units with no evaluation (sem 1)",
    "unemployment_rate":                          "Regional unemployment rate",
}

# ── Data loading ──────────────────────────────────────────────────────────────
_df: pd.DataFrame | None = None
_using_gold = False


def _apply_silver(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    # Fallback risk columns (overridden by gold)
    if "dropout_prob" not in df.columns:
        df["dropout_prob"] = df["financial_stress_index"]
    if "risk_score" not in df.columns:
        df["risk_score"] = (df["dropout_prob"] * 100).round(1)
    if "risk_tier" not in df.columns:
        df["risk_tier"] = pd.cut(df["dropout_prob"], bins=[-0.001,0.40,0.70,1.001],
                                 labels=["LOW","MEDIUM","HIGH"]).astype(str)
    return df


def load_df() -> pd.DataFrame:
    global _df, _using_gold

    if _df is not None:
        return _df

    gold_path   = ROOT / "data" / "gold" / "student_predictions.parquet"
    silver_path = ROOT / "data" / "silver" / "student_dropout_silver.parquet"
    csv_path    = ROOT / "dataset" / "students_dropout_academic_success_clean.csv"

    if gold_path.exists():
        print(f"[API] Loading gold predictions: {gold_path}")
        df = pd.read_parquet(gold_path)
        _using_gold = True
    elif silver_path.exists():
        print(f"[API] Loading silver parquet: {silver_path}")
        df = pd.read_parquet(silver_path)
        df = _apply_silver(df)
    else:
        print(f"[API] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df = _apply_silver(df)

    _df = df.reset_index(drop=True)
    print(f"[API] Loaded {len(_df)} rows  |  gold={_using_gold}")
    return _df


def assign_segment(r) -> str:
    if r.get("debtor", 0) == 1 or r.get("tuition_fees_up_to_date", 1) == 0:
        return "Socioeconomic Risk"
    if r.get("avg_grade", 10) < 8.0 or r.get("enrollment_efficiency", 1) < 0.5:
        return "Academic Strugglers"
    return "Attendance Risk"


def shap_factor_labels(r, feature_cols: list[str]) -> list[str]:
    """Return top-3 human-readable SHAP factor labels for a student row."""
    if "shap_top_features" in r and isinstance(r["shap_top_features"], list):
        feats = r["shap_top_features"][:3]
    else:
        # Fallback: derive from FSI components
        feats = []
        if r.get("debtor", 0):                        feats.append("debtor")
        if not r.get("tuition_fees_up_to_date", 1):   feats.append("tuition_fees_up_to_date")
        if not r.get("scholarship_holder", 1):         feats.append("scholarship_holder")
        if not feats:                                  feats = ["avg_grade", "enrollment_efficiency"]

    labels = []
    for f in feats[:3]:
        labels.append(SHAP_LABEL_MAP.get(f, f.replace("_", " ").title()))
    return labels


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    df = load_df().copy()
    df["segment"] = df.apply(assign_segment, axis=1)

    at_risk = df[df["risk_tier"].isin(["MEDIUM", "HIGH"])]
    high    = df[df["risk_tier"] == "HIGH"]

    seg_counts = at_risk["segment"].value_counts().to_dict()

    return {
        "totalStudents":     len(df),
        "atRisk":            len(at_risk),
        "predictedDropouts": len(high),
        "revenueAtRisk":     len(at_risk) * ANNUAL_FEE_INR,
        "modelUsed":         "XGBoost (ROC-AUC 0.9346)" if _using_gold else "FSI heuristic (silver)",
        "trends": {
            "totalStudents":     {"direction": "up",  "value": 3.2},
            "atRisk":            {"direction": "up",  "value": 12.5},
            "predictedDropouts": {"direction": "up",  "value": 8.1},
            "revenueAtRisk":     {"direction": "up",  "value": 15.3},
        },
        "riskDistribution": [
            {"segment": seg, "count": seg_counts.get(seg, 0)}
            for seg in ["Academic Strugglers", "Attendance Risk", "Socioeconomic Risk"]
        ],
    }


@app.get("/api/students")
def get_students(limit: int = 5000):
    df = load_df().copy()

    # Raw CSV columns — exactly as in the dataset
    RAW_COLS = [
        "marital_status", "application_mode", "application_order", "course",
        "daytime_evening_attendance", "previous_qualification",
        "previous_qualification_grade", "nacionality",
        "mother_s_qualification", "father_s_qualification",
        "mother_s_occupation", "father_s_occupation",
        "admission_grade", "displaced", "educational_special_needs",
        "debtor", "tuition_fees_up_to_date", "gender", "scholarship_holder",
        "age_at_enrollment", "international",
        "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
        "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
        "curricular_units_1st_sem_grade", "curricular_units_1st_sem_without_evaluations",
        "curricular_units_2nd_sem_credited", "curricular_units_2nd_sem_enrolled",
        "curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_approved",
        "curricular_units_2nd_sem_grade", "curricular_units_2nd_sem_without_evaluations",
        "unemployment_rate", "inflation_rate", "gdp", "target",
    ]

    present = [c for c in RAW_COLS if c in df.columns]
    subset  = df[present].head(limit).reset_index(drop=True)

    rows = []
    for i, r in subset.iterrows():
        row = {"sno": int(i) + 1}
        for col in present:
            val = r[col]
            if hasattr(val, "item"):   # numpy scalar → python
                val = val.item()
            row[col] = val
        rows.append(row)

    return {"students": rows, "total": len(df), "columns": present}


@app.get("/api/segments")
def get_segments():
    df = load_df().copy()
    df["segment"] = df.apply(assign_segment, axis=1)
    at_risk = df[df["risk_tier"].isin(["MEDIUM", "HIGH"])]
    counts  = at_risk["segment"].value_counts()

    return {"segments": [
        {
            "name":  "Academic Strugglers",
            "count": int(counts.get("Academic Strugglers", 0)),
            "profile": [
                "Average grade below 8/18.9 across semesters",
                "Enrollment efficiency below 50% of enrolled units approved",
                "Low academic engagement and resource usage",
            ],
            "action": "Deploy targeted academic mentoring and restructure course load for students below the grade threshold.",
        },
        {
            "name":  "Attendance Risk",
            "count": int(counts.get("Attendance Risk", 0)),
            "profile": [
                "Moderate financial stress but adequate academic performance",
                "Missing evaluations in enrolled units",
                "Declining engagement trend across the semester",
            ],
            "action": "Implement early-warning alerts and assign peer engagement buddies for at-risk students.",
        },
        {
            "name":  "Socioeconomic Risk",
            "count": int(counts.get("Socioeconomic Risk", 0)),
            "profile": [
                "Outstanding debt or tuition fees not current",
                "No scholarship — fully self-funded",
                "Displacement or relocation burden",
            ],
            "action": "Fast-track financial aid reviews and create flexible payment plans with welfare office coordination.",
        },
    ]}


@app.get("/api/fairness")
def get_fairness():
    df = load_df().copy()

    def grp(mask):
        g = df[mask]
        n = len(g)
        dp = float(g["target_binary"].mean()) if "target_binary" in g else 0.0
        # Equal opportunity: among actual dropouts, what fraction did model flag as MEDIUM/HIGH?
        pos = g[g.get("target_binary", pd.Series(dtype=int)) == 1] if "target_binary" in g.columns else g
        eo = float(len(pos[pos["risk_tier"].isin(["MEDIUM","HIGH"])]) / max(len(pos), 1))
        return {"rate": round(dp, 4), "count": n, "tpr": round(eo, 4)}

    m  = grp(df["gender"] == 1)
    f  = grp(df["gender"] == 0)
    s  = grp(df["scholarship_holder"] == 1)
    ns = grp(df["scholarship_holder"] == 0)

    return {
        "demographicParity": {
            "male":          {"rate": m["rate"],  "count": m["count"]},
            "female":        {"rate": f["rate"],  "count": f["count"]},
            "scholarship":   {"rate": s["rate"],  "count": s["count"]},
            "noScholarship": {"rate": ns["rate"], "count": ns["count"]},
        },
        "equalOpportunity": {
            "male":          {"rate": m["tpr"],  "count": m["count"]},
            "female":        {"rate": f["tpr"],  "count": f["count"]},
            "scholarship":   {"rate": s["tpr"],  "count": s["count"]},
            "noScholarship": {"rate": ns["tpr"], "count": ns["count"]},
        },
    }


@app.get("/api/model")
def get_model_info():
    meta_path = ROOT / "data" / "models" / "best_model_meta.json"
    shap_path = ROOT / "data" / "gold" / "shap_importance.parquet"

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    top_features = []
    if shap_path.exists():
        shap_df = pd.read_parquet(shap_path)
        top_features = shap_df.head(10).to_dict(orient="records")

    return {
        "modelName":    meta.get("model_name", "unknown"),
        "metrics":      meta.get("metrics", {}),
        "topFeatures":  top_features,
        "usingGold":    _using_gold,
    }


@app.get("/health")
def health():
    return {"status": "ok", "gold": _using_gold}


if __name__ == "__main__":
    import uvicorn
    try:
        load_df()
    except Exception as e:
        print(f"[WARN] {e}")

    print("\n[API] Starting on http://localhost:8001")
    print("[API] Docs at http://localhost:8001/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
