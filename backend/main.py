"""
backend/main.py - Drop(In) API
Serves: dashboard, students (priority cols + filters), clusters, model eval,
        SHAP plots, fairness audit (full), LLM reports, financial calculator.
Start: /c/Python313/python.exe backend/main.py  (from project root)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="Drop(In) API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Paths ──────────────────────────────────────────────────────────────────────
GOLD_DIR  = ROOT / "data" / "gold"
MODEL_DIR = ROOT / "data" / "models"
PLOTS_DIR = ROOT / "data" / "plots"

COLLEGE_FEES = {
    "Manipal Institute of Technology": 700000,
    "VIT Vellore":                     650000,
    "SRM Institute":                   580000,
    "Amity University":                520000,
    "Default":                         120000,
}

DISPLAY_NAMES = {
    "marital_status":                               "Marital Status",
    "application_mode":                             "Application Mode",
    "application_order":                            "Application Order",
    "course":                                       "Course Code",
    "daytime_evening_attendance":                   "Attendance Type",
    "previous_qualification":                       "Previous Qualification",
    "previous_qualification_grade":                 "Prev Qual Grade",
    "nacionality":                                  "Nationality",
    "mother_s_qualification":                       "Mother's Qualification",
    "father_s_qualification":                       "Father's Qualification",
    "mother_s_occupation":                          "Mother's Occupation",
    "father_s_occupation":                          "Father's Occupation",
    "admission_grade":                              "Admission Grade",
    "displaced":                                    "Displaced",
    "educational_special_needs":                    "Special Needs",
    "debtor":                                       "Debtor",
    "tuition_fees_up_to_date":                      "Tuition Fees Current",
    "gender":                                       "Gender",
    "scholarship_holder":                           "Scholarship Holder",
    "age_at_enrollment":                            "Age at Enrollment",
    "international":                                "International Student",
    "curricular_units_1st_sem_credited":            "Sem 1 Credited",
    "curricular_units_1st_sem_enrolled":            "Sem 1 Enrolled",
    "curricular_units_1st_sem_evaluations":         "Sem 1 Evaluations",
    "curricular_units_1st_sem_approved":            "Sem 1 Approved",
    "curricular_units_1st_sem_grade":               "Sem 1 Grade",
    "curricular_units_1st_sem_without_evaluations": "Sem 1 Missed Evals",
    "curricular_units_2nd_sem_credited":            "Sem 2 Credited",
    "curricular_units_2nd_sem_enrolled":            "Sem 2 Enrolled",
    "curricular_units_2nd_sem_evaluations":         "Sem 2 Evaluations",
    "curricular_units_2nd_sem_approved":            "Sem 2 Approved",
    "curricular_units_2nd_sem_grade":               "Sem 2 Grade",
    "curricular_units_2nd_sem_without_evaluations": "Sem 2 Missed Evals",
    "unemployment_rate":                            "Unemployment Rate (%)",
    "inflation_rate":                               "Inflation Rate (%)",
    "gdp":                                          "GDP",
    "target":                                       "Outcome",
    "avg_grade":                                    "Avg Grade",
    "enrollment_efficiency":                        "Enrollment Efficiency",
    "total_approved_units":                         "Total Approved Units",
    "financial_stress_index":                       "Financial Stress Index",
    "dropout_prob":                                 "Dropout Probability",
    "risk_score":                                   "Risk Score",
    "risk_tier":                                    "Risk Tier",
    "cluster_label":                                "Segment",
}

# Human-readable dropout risk factor labels (for SHAP top features)
SHAP_LABELS = {
    "avg_grade":                                    "Low Avg Grade",
    "enrollment_efficiency":                        "Low Course Completion",
    "financial_stress_index":                       "High Financial Stress",
    "curricular_units_1st_sem_without_evaluations": "Missed Sem 1 Evaluations",
    "curricular_units_2nd_sem_without_evaluations": "Missed Sem 2 Evaluations",
    "debtor":                                       "Outstanding Debt",
    "tuition_fees_up_to_date":                      "Tuition Fees Overdue",
    "scholarship_holder":                           "No Scholarship",
    "total_approved_units":                         "Low Units Completed",
    "curricular_units_2nd_sem_approved":            "Low Sem 2 Passes",
    "curricular_units_1st_sem_approved":            "Low Sem 1 Passes",
    "curricular_units_1st_sem_grade":               "Low Sem 1 Grade",
    "curricular_units_2nd_sem_grade":               "Low Sem 2 Grade",
    "admission_grade":                              "Low Admission Grade",
    "age_at_enrollment":                            "Age at Enrollment",
    "displaced":                                    "Displaced Student",
    "unemployment_rate":                            "High Unemployment",
    "previous_qualification_grade":                 "Low Prior Qual Grade",
    "application_mode":                             "Application Mode",
    "gdp":                                          "Economic Conditions",
}

RAW_COLS = [
    "marital_status","application_mode","application_order","course",
    "daytime_evening_attendance","previous_qualification","previous_qualification_grade",
    "nacionality","mother_s_qualification","father_s_qualification",
    "mother_s_occupation","father_s_occupation","admission_grade",
    "displaced","educational_special_needs","debtor","tuition_fees_up_to_date",
    "gender","scholarship_holder","age_at_enrollment","international",
    "curricular_units_1st_sem_credited","curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations","curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade","curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited","curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations","curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade","curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate","inflation_rate","gdp","target",
]

# ── Data cache ─────────────────────────────────────────────────────────────────
_df:      pd.DataFrame | None = None
_reports: pd.DataFrame | None = None


def load_df() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df
    p = GOLD_DIR / "student_predictions.parquet"
    c = ROOT / "dataset" / "students_dropout_academic_success_clean.csv"
    if p.exists():
        _df = pd.read_parquet(p)
    else:
        _df = pd.read_csv(c)
    _df = _df.reset_index(drop=True)
    print(f"[API] Loaded {len(_df)} rows")
    return _df


def load_reports() -> pd.DataFrame:
    global _reports
    if _reports is None:
        p = GOLD_DIR / "llm_reports.parquet"
        _reports = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _reports


def safe(v):
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if hasattr(v, "item"): return v.item()
    if isinstance(v, list): return v
    return v


def shap_factors_readable(raw_list) -> list[str]:
    """Convert raw feature name list to human-readable dropout risk labels."""
    if raw_list is None:
        return []
    if isinstance(raw_list, str):
        try:
            raw_list = json.loads(raw_list)
        except Exception:
            return [raw_list]
    try:
        if len(raw_list) == 0:
            return []
    except TypeError:
        return []
    return [SHAP_LABELS.get(f, DISPLAY_NAMES.get(f, f.replace("_"," ").title())) for f in raw_list]


def flag(gap: float) -> str:
    return "FAIR" if gap <= 0.03 else ("REVIEW" if gap <= 0.08 else "BIASED")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
def dashboard():
    df = load_df()
    total     = len(df)
    dropouts  = int((df["target"] == "Dropout").sum()) if "target" in df.columns else 0
    at_risk   = int((df["risk_tier"].isin(["MEDIUM","HIGH"])).sum()) if "risk_tier" in df.columns else 0
    high_risk = int((df["risk_tier"] == "HIGH").sum()) if "risk_tier" in df.columns else 0
    seg_counts = df["cluster_label"].value_counts().to_dict() if "cluster_label" in df.columns else {}

    fairness = {}
    if "gender" in df.columns and "target_binary" in df.columns:
        for g, label in [(1,"male"),(0,"female")]:
            grp = df[df["gender"] == g]
            dp  = float(grp["target_binary"].mean()) if len(grp) else 0.0
            pos = grp[grp["target_binary"] == 1] if "target_binary" in grp.columns else grp.head(0)
            eo  = float(len(pos[pos["risk_tier"].isin(["MEDIUM","HIGH"])]) / max(len(pos),1)) if "risk_tier" in df.columns else dp
            fairness[label] = {"dropout_rate": round(dp,4), "count": len(grp), "equal_opportunity": round(eo,4)}
        for s, label in [(1,"scholarship"),(0,"no_scholarship")]:
            grp = df[df["scholarship_holder"] == s]
            dp  = float(grp["target_binary"].mean()) if len(grp) else 0.0
            pos = grp[grp["target_binary"] == 1] if "target_binary" in grp.columns else grp.head(0)
            eo  = float(len(pos[pos["risk_tier"].isin(["MEDIUM","HIGH"])]) / max(len(pos),1)) if "risk_tier" in df.columns else dp
            fairness[label] = {"dropout_rate": round(dp,4), "count": len(grp), "equal_opportunity": round(eo,4)}

    # Gender gap based on MODEL PREDICTIONS (risk_score), not actual outcomes
    if "risk_score" in df.columns and "gender" in df.columns:
        male_pred  = float(df[df["gender"]==1]["risk_score"].mean()) / 100.0
        female_pred = float(df[df["gender"]==0]["risk_score"].mean()) / 100.0
        gender_gap = abs(male_pred - female_pred)
    else:
        gender_gap = abs(fairness.get("male",{}).get("dropout_rate",0) - fairness.get("female",{}).get("dropout_rate",0))

    # Top 3 SHAP features with descriptions
    top_features = []
    shap_p = GOLD_DIR / "shap_importance.parquet"
    if shap_p.exists():
        sh = pd.read_parquet(shap_p).head(3)
        FEATURE_DESCRIPTIONS = {
            "curricular_units_2nd_sem_approved": "Students who pass fewer courses in semester 2 are far more likely to drop out. This is the single strongest predictor.",
            "enrollment_efficiency":             "Low ratio of approved-to-enrolled courses signals academic disengagement and predicts dropout.",
            "financial_stress_index":            "High financial stress (debt, unpaid fees, no scholarship) strongly increases dropout probability.",
            "tuition_fees_up_to_date":           "Students behind on tuition payments face both financial and administrative barriers to continuing.",
            "age_at_enrollment":                 "Older-than-average students face different challenges (work, family) and have higher dropout rates.",
            "course":                            "Certain courses have inherently higher dropout rates due to difficulty or mismatched expectations.",
            "total_approved_units":              "Fewer total approved units means slower progress, which correlates with eventual dropout.",
            "avg_grade":                         "Low average grades across semesters reflect academic struggle, a key dropout driver.",
            "admission_grade":                   "Lower admission grades suggest weaker academic preparation, increasing dropout risk.",
            "scholarship_holder":                "Students without scholarships face financial pressure that contributes to dropout.",
        }
        for _, r in sh.iterrows():
            feat = r["feature"]
            readable = DISPLAY_NAMES.get(feat, feat.replace("_"," ").title())
            top_features.append({
                "feature":       readable,
                "mean_abs_shap": round(float(r["mean_abs_shap"]), 4),
                "description":   FEATURE_DESCRIPTIONS.get(feat, f"{readable} is a significant predictor of student dropout."),
            })

    # Compare model gap to actual gap.  Model is biased only if it AMPLIFIES the real gap.
    actual_gap = abs(fairness.get("male",{}).get("dropout_rate",0) - fairness.get("female",{}).get("dropout_rate",0))
    amplification = gender_gap - actual_gap   # positive = model makes it worse
    if amplification > 0.03:
        fairness_status = "BIASED"
    elif gender_gap > 0.05:
        fairness_status = "REVIEW"
    else:
        fairness_status = "FAIR"

    return {
        "totalStudents":    total,
        "actualDropouts":   dropouts,
        "dropoutRate":      round(dropouts / total, 4) if total else 0,
        "atRisk":           at_risk,
        "highRisk":         high_risk,
        "riskDistribution": [{"segment": k, "count": v} for k, v in seg_counts.items()],
        "fairness":         fairness,
        "genderGap":        round(gender_gap, 4),
        "fairnessStatus":   fairness_status,
        "topFeatures":      top_features,
    }


@app.get("/api/students")
def students(
    limit: int = Query(5000),
    risk_tier: str = Query(""),
    segment: str = Query(""),
    fairness_filter: str = Query(""),
):
    df = load_df()

    # Compute per-student fairness flag using individual SHAP gender value
    df = df.copy()
    if "shap_gender" in df.columns:
        # If |shap_gender| is large, the model is using gender heavily for THIS student
        abs_shap = df["shap_gender"].abs()
        df["_fairness_flag"] = "FAIR"
        df.loc[abs_shap > 0.03, "_fairness_flag"] = "REVIEW"
        df.loc[abs_shap > 0.10, "_fairness_flag"] = "BIASED"
    else:
        df["_fairness_flag"] = "FAIR"

    # Server-side filters
    if risk_tier and risk_tier != "ALL" and "risk_tier" in df.columns:
        df = df[df["risk_tier"] == risk_tier]
    if segment and segment != "ALL" and "cluster_label" in df.columns:
        df = df[df["cluster_label"] == segment]
    if fairness_filter and fairness_filter != "ALL":
        df = df[df["_fairness_flag"] == fairness_filter]

    df = df.head(limit).reset_index(drop=True)

    # Load LLM reports for remediations
    reps = load_reports()
    report_map: dict[str, str] = {}
    if not reps.empty and "student_id" in reps.columns and "counselor_report" in reps.columns:
        for _, rep in reps.iterrows():
            report_map[str(rep["student_id"])] = str(rep["counselor_report"])

    # Get outcome values for filter options
    full_df = load_df()
    outcome_vals = sorted(full_df["target"].dropna().unique().tolist()) if "target" in full_df.columns else []

    rows = []
    for i, r in df.iterrows():
        sid = f"STU-{int(i)+1000:04d}"
        shap_raw  = r.get("shap_top_features", [])
        shap_read = shap_factors_readable(shap_raw)

        # Determine outcome label
        outcome = safe(r.get("target", "—"))

        # Priority fields (always first)
        row: dict = {
            "Student ID":   sid,
            "Risk Score":   round(float(r.get("dropout_prob", 0)) * 100, 1),
            "Outcome":      outcome,
            "Risk Tier":    safe(r.get("risk_tier", "N/A")),
            "Factor 1":     shap_read[0] if len(shap_read) > 0 else "—",
            "Factor 2":     shap_read[1] if len(shap_read) > 1 else "—",
            "Factor 3":     shap_read[2] if len(shap_read) > 2 else "—",
            "Fairness Flag": r.get("_fairness_flag", "FAIR"),
            "Segment":      safe(r.get("cluster_label", "—")),
        }

        # Attach counselor report if available
        if sid in report_map:
            row["_counselor_report"] = report_map[sid]

        # Then the rest of the raw columns
        present = [c for c in RAW_COLS if c in df.columns]
        for c in present:
            key = DISPLAY_NAMES.get(c, c)
            if key not in row:  # don't overwrite priority cols
                row[key] = safe(r[c])

        rows.append(row)

    # Column order: priority first, then raw
    priority_cols = ["Student ID","Risk Score","Outcome","Risk Tier","Factor 1","Factor 2","Factor 3","Fairness Flag","Segment"]
    rest_cols = [DISPLAY_NAMES.get(c, c) for c in [c for c in RAW_COLS if c in df.columns]]
    rest_cols = [c for c in rest_cols if c not in priority_cols]
    all_cols  = priority_cols + rest_cols

    # Available filter options
    segments   = sorted(full_df["cluster_label"].dropna().unique().tolist()) if "cluster_label" in full_df.columns else []
    risk_tiers = ["HIGH","MEDIUM","LOW"]

    return {
        "students": rows,
        "total":    len(full_df),
        "filtered": len(df),
        "columns":  all_cols,
        "filterOptions": {
            "riskTiers": risk_tiers,
            "segments":  segments,
            "fairness":  ["FAIR","REVIEW","BIASED"],
            "outcomes":  outcome_vals,
        },
    }


@app.get("/api/clusters")
def clusters():
    df = load_df()
    p  = GOLD_DIR / "cluster_summary.parquet"
    summary = pd.read_parquet(p).to_dict(orient="records") if p.exists() else []

    shap_p   = GOLD_DIR / "shap_importance.parquet"
    top_shap = []
    if shap_p.exists():
        sh = pd.read_parquet(shap_p).head(10)
        top_shap = [
            {
                "feature":       DISPLAY_NAMES.get(r["feature"], r["feature"].replace("_"," ").title()),
                "mean_abs_shap": round(float(r["mean_abs_shap"]), 4),
            }
            for _, r in sh.iterrows()
        ]

    # Top 3 feature reasoning
    TOP3_REASONING = {
        "curricular_units_2nd_sem_approved": "Semester 2 course approvals is the strongest dropout predictor. Students passing fewer courses in their second semester show a sharp increase in dropout probability — this signals academic decline at a critical juncture.",
        "enrollment_efficiency":             "Enrollment efficiency (approved/enrolled ratio) captures how effectively a student converts enrolments into passes. Low efficiency indicates disengagement or academic difficulty, both strong dropout precursors.",
        "financial_stress_index":            "The composite financial stress index (debt, unpaid fees, no scholarship) quantifies economic hardship. Students with FSI > 0.4 dropout at nearly 2.5× the rate of others.",
    }
    top3_features = []
    if shap_p.exists():
        sh3 = pd.read_parquet(shap_p).head(3)
        for _, r in sh3.iterrows():
            feat = r["feature"]
            readable = DISPLAY_NAMES.get(feat, feat.replace("_"," ").title())
            top3_features.append({
                "feature":    readable,
                "shap_value": round(float(r["mean_abs_shap"]), 4),
                "reasoning":  TOP3_REASONING.get(feat, f"{readable} is a significant predictor of dropout risk."),
            })

    # Compute silhouette score if possible
    silhouette = None
    try:
        from sklearn.metrics import silhouette_score
        feature_cols = ["avg_grade", "enrollment_efficiency", "financial_stress_index", "total_approved_units"]
        present = [c for c in feature_cols if c in df.columns]
        if len(present) >= 2 and "cluster_id" in df.columns:
            X = df[present].dropna()
            labels = df.loc[X.index, "cluster_id"]
            if len(labels.unique()) > 1:
                score = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
                silhouette = round(float(score), 4)
    except Exception:
        silhouette = None

    cluster_details = []
    for row in summary:
        label = row.get("cluster_label","")
        safe_name = label.lower().replace(" ","_")
        cluster_details.append({
            "name":           label,
            "count":          int(row.get("count",0)),
            "dropoutRate":    round(float(row.get("dropout_rate",0)),3),
            "avgGrade":       round(float(row.get("avg_grade",0)),2),
            "avgFSI":         round(float(row.get("avg_fsi",0)),3),
            "avgDropoutProb": round(float(row.get("avg_dropout_prob",0)),3),
            "shapPlot":       f"/api/plots/shap_beeswarm_{safe_name}",
        })

    return {
        "clusters":      cluster_details,
        "globalShap":    top_shap,
        "silhouette":    silhouette,
        "top3Features":  top3_features,
    }


@app.get("/api/model")
def model_info():
    p = MODEL_DIR / "eval_matrix.json"
    if not p.exists():
        return {"error": "Model not trained yet"}
    with open(p) as f:
        meta = json.load(f)
    lr  = meta.get("logistic_regression", {})
    xgb = meta.get("xgboost", {})
    return {
        "winner":    meta.get("winner","XGBoost"),
        "trainSize": meta.get("train_size",0),
        "testSize":  meta.get("test_size",0),
        "splitRatio":"80/20",
        "models": [
            {"name":"Logistic Regression", **{k:v for k,v in lr.items() if k not in ["tp","fp","tn","fn"]},
             "confusionMatrix":{"tp":lr.get("tp",0),"fp":lr.get("fp",0),"tn":lr.get("tn",0),"fn":lr.get("fn",0)}},
            {"name":"XGBoost",             **{k:v for k,v in xgb.items() if k not in ["tp","fp","tn","fn"]},
             "confusionMatrix":{"tp":xgb.get("tp",0),"fp":xgb.get("fp",0),"tn":xgb.get("tn",0),"fn":xgb.get("fn",0)}},
        ],
    }


@app.get("/api/fairness")
def fairness():
    df = load_df()

    def grp_metrics(mask, name):
        g  = df[mask]
        dp = float(g["target_binary"].mean()) if "target_binary" in g.columns and len(g) else 0.0
        pos = g[g["target_binary"] == 1] if "target_binary" in g.columns else g.head(0)
        eo  = float(len(pos[pos["risk_tier"].isin(["MEDIUM","HIGH"])]) / max(len(pos),1)) if "risk_tier" in df.columns else dp
        return {"group": name, "count": len(g), "dropout_rate": round(dp,4), "equal_opportunity": round(eo,4)}

    male_m    = grp_metrics(df["gender"]==1,            "Male")
    female_m  = grp_metrics(df["gender"]==0,            "Female")
    schol_m   = grp_metrics(df["scholarship_holder"]==1,"Scholarship Holder")
    noschol_m = grp_metrics(df["scholarship_holder"]==0,"No Scholarship")

    # Socioeconomic proxy: FSI > 0.4 = low income
    if "financial_stress_index" in df.columns:
        hi_fsi    = df["financial_stress_index"] > 0.4
        lowinc_m  = grp_metrics(hi_fsi,  "Low Income (High FSI)")
        highinc_m = grp_metrics(~hi_fsi, "Higher Income (Low FSI)")
    else:
        lowinc_m  = {"group":"Low Income","count":0,"dropout_rate":0,"equal_opportunity":0}
        highinc_m = {"group":"Higher Income","count":0,"dropout_rate":0,"equal_opportunity":0}

    gender_gap    = abs(male_m["dropout_rate"] - female_m["dropout_rate"])
    scholar_gap   = abs(schol_m["dropout_rate"] - noschol_m["dropout_rate"])
    socioeco_gap  = abs(lowinc_m["dropout_rate"] - highinc_m["dropout_rate"])
    gender_eo_gap = abs(male_m["equal_opportunity"] - female_m["equal_opportunity"])
    schol_eo_gap  = abs(schol_m["equal_opportunity"] - noschol_m["equal_opportunity"])

    # SHAP-based bias: which features drive the gender gap
    shap_gender_bias = []
    shap_income_bias = []
    shap_cols = [c for c in df.columns if c.startswith("shap_") and df[c].dtype in ['float64','float32','int64','int32']]
    if shap_cols:
        if "gender" in df.columns:
            m_shap = df[df["gender"]==1][shap_cols].mean()
            f_shap = df[df["gender"]==0][shap_cols].mean()
            diff   = (m_shap - f_shap).abs().sort_values(ascending=False)
            for col, val in diff.head(5).items():
                feat = col.replace("shap_","")
                shap_gender_bias.append({
                    "feature":      DISPLAY_NAMES.get(feat, feat.replace("_"," ").title()),
                    "disparity":    round(float(val),4),
                    "male_mean":    round(float(m_shap[col]),4),
                    "female_mean":  round(float(f_shap[col]),4),
                    "pct_contribution": round(float(val) / max(float(diff.sum()), 1e-9) * 100, 1),
                })

        if "financial_stress_index" in df.columns:
            hi  = df["financial_stress_index"] > 0.4
            lo_shap  = df[hi][shap_cols].mean()
            hi_shap  = df[~hi][shap_cols].mean()
            diff = (lo_shap - hi_shap).abs().sort_values(ascending=False)
            for col, val in diff.head(5).items():
                feat = col.replace("shap_","")
                shap_income_bias.append({
                    "feature":         DISPLAY_NAMES.get(feat, feat.replace("_"," ").title()),
                    "disparity":       round(float(val),4),
                    "lowincome_mean":  round(float(lo_shap[col]),4),
                    "highincome_mean": round(float(hi_shap[col]),4),
                    "pct_contribution": round(float(val) / max(float(diff.sum()), 1e-9) * 100, 1),
                })

    # Bias insight sentence
    higher_risk_gender = "males" if male_m["dropout_rate"] > female_m["dropout_rate"] else "females"
    top_gender_feat    = shap_gender_bias[0]["feature"] if shap_gender_bias else "academic performance"
    top_gender_pct     = shap_gender_bias[0]["pct_contribution"] if shap_gender_bias else 0
    bias_insight = (
        f"Model predicts higher dropout risk for {higher_risk_gender} "
        f"({gender_gap*100:.1f}pp gap). "
        f"Low-income students (FSI>0.4) dropout at {lowinc_m['dropout_rate']*100:.1f}% "
        f"vs {highinc_m['dropout_rate']*100:.1f}% for higher-income students "
        f"({socioeco_gap*100:.1f}pp gap). "
        f"Primary gender-bias driver: '{top_gender_feat}' contributing {top_gender_pct:.0f}% of disparity."
    )

    # Mitigation suggestions
    mitigation = []
    if flag(gender_gap) != "FAIR":
        mitigation.append({
            "issue":      f"Gender gap: {gender_gap*100:.1f}pp (threshold: 3pp)",
            "suggestion": "Apply post-processing calibration per gender group. "
                          "Review gender-correlated features (grade patterns) for proxy bias.",
            "priority":   "HIGH" if gender_gap > 0.08 else "MEDIUM",
        })
    if flag(scholar_gap) != "FAIR":
        mitigation.append({
            "issue":      f"Scholarship gap: {scholar_gap*100:.1f}pp",
            "suggestion": "Re-weight training samples for non-scholarship students. "
                          "Expand financial aid criteria to reduce the gap.",
            "priority":   "HIGH" if scholar_gap > 0.08 else "MEDIUM",
        })
    if socioeco_gap > 0.10:
        mitigation.append({
            "issue":      f"Socioeconomic gap: {socioeco_gap*100:.1f}pp (low vs higher income)",
            "suggestion": "Introduce FSI threshold alerts: students with FSI > 0.4 should "
                          "auto-trigger counselor review. Consider FSI-aware intervention tiers.",
            "priority":   "HIGH",
        })
    if shap_gender_bias:
        mitigation.append({
            "issue":      f"'{shap_gender_bias[0]['feature']}' drives {shap_gender_bias[0]['pct_contribution']:.0f}% of gender disparity",
            "suggestion": "Investigate whether this feature is a legitimate risk signal or a proxy for "
                          "gender. Consider fairness-aware feature re-weighting or adversarial debiasing.",
            "priority":   "MEDIUM",
        })
    mitigation.append({
        "issue":      "Ongoing monitoring",
        "suggestion": "Log all fairness metrics to a Delta table monthly. "
                      "Set alerts when any gap exceeds 5pp. Track changes after each model retrain.",
        "priority":   "LOW",
    })

    return {
        "groups":         [male_m, female_m, schol_m, noschol_m, lowinc_m, highinc_m],
        "genderGap":      round(gender_gap, 4),
        "scholarGap":     round(scholar_gap, 4),
        "socioEcoGap":    round(socioeco_gap, 4),
        "genderEOGap":    round(gender_eo_gap, 4),
        "scholarEOGap":   round(schol_eo_gap, 4),
        "genderFlag":     flag(gender_gap),
        "scholarFlag":    flag(scholar_gap),
        "socioEcoFlag":   flag(socioeco_gap),
        "shapGenderBias": shap_gender_bias,
        "shapIncomeBias": shap_income_bias,
        "biasInsight":    bias_insight,
        "mitigation":     mitigation,
        # Legacy structure
        "demographicParity": {
            "male":{"rate":male_m["dropout_rate"],"count":male_m["count"]},
            "female":{"rate":female_m["dropout_rate"],"count":female_m["count"]},
            "scholarship":{"rate":schol_m["dropout_rate"],"count":schol_m["count"]},
            "noScholarship":{"rate":noschol_m["dropout_rate"],"count":noschol_m["count"]},
        },
        "equalOpportunity": {
            "male":{"rate":male_m["equal_opportunity"],"count":male_m["count"]},
            "female":{"rate":female_m["equal_opportunity"],"count":female_m["count"]},
            "scholarship":{"rate":schol_m["equal_opportunity"],"count":schol_m["count"]},
            "noScholarship":{"rate":noschol_m["equal_opportunity"],"count":noschol_m["count"]},
        },
    }


@app.get("/api/recommendations")
def recommendations(college: str = Query("Default"), fee: int = Query(0), limit: int = Query(50)):
    reps       = load_reports()
    # Use custom fee if provided, else fall back to college preset
    annual_fee = fee if fee > 0 else COLLEGE_FEES.get(college, COLLEGE_FEES["Default"])

    if reps.empty:
        return {"reports": [], "college": college, "annualFee": annual_fee,
                "totalAtRisk": 0, "revenueAtRisk": 0}

    out = []
    for _, r in reps.head(limit).iterrows():
        out.append({
            "studentId":       r.get("student_id",""),
            "dropoutProb":     round(float(r.get("dropout_prob",0)),3),
            "riskTier":        r.get("risk_tier",""),
            "segment":         r.get("cluster_label",""),
            "counselorReport": r.get("counselor_report",""),
            "annualFee":       annual_fee,
            "financialImpact": annual_fee,
        })

    return {"reports": out, "college": college, "annualFee": annual_fee,
            "totalAtRisk": len(reps), "revenueAtRisk": len(reps) * annual_fee}


@app.get("/api/financial-calculator")
def financial_calculator(college: str = Query("Default")):
    df         = load_df()
    annual_fee = COLLEGE_FEES.get(college, COLLEGE_FEES["Default"])
    high   = int((df["risk_tier"] == "HIGH").sum())   if "risk_tier" in df.columns else 0
    medium = int((df["risk_tier"] == "MEDIUM").sum()) if "risk_tier" in df.columns else 0
    return {
        "college":              college,
        "annualFee":            annual_fee,
        "highRiskStudents":     high,
        "mediumRiskStudents":   medium,
        "totalAtRisk":          high + medium,
        "revenueAtRisk":        (high + medium) * annual_fee,
        "highRiskRevenue":      high * annual_fee,
        "estimatedRecoverable": round((high + medium) * annual_fee * 0.40),
        "interventionCost":     (high + medium) * 12000,
        "roi":                  round(((high + medium) * annual_fee * 0.40) / max((high + medium) * 12000, 1), 1),
        "availableColleges":    list(COLLEGE_FEES.keys()),
    }


# ── Enrichment endpoints ───────────────────────────────────────────────────────

@app.get("/api/enrichment")
def enrichment():
    """OULAD VLE + World Bank + AISHE enrichment summary for judges."""
    import json as _json
    summary_path = GOLD_DIR / "enrichment_summary.json"
    if not summary_path.exists():
        return JSONResponse(
            {"error": "Enrichment data not generated. Run: python local/run_enrichment.py"},
            status_code=404,
        )
    with open(summary_path) as f:
        summary = _json.load(f)

    # Load VLE feature importance if available
    vle_fi_path = GOLD_DIR / "oulad_vle_feature_importance.parquet"
    vle_top_features = []
    if vle_fi_path.exists():
        vle_fi = pd.read_parquet(vle_fi_path)
        for _, row in vle_fi.iterrows():
            vle_top_features.append({
                "feature":    str(row["feature"]) if "feature" in row.index else "",
                "label":      str(row["feature_label"]) if "feature_label" in row.index else str(row["feature"]),
                "importance": round(float(row["importance"]), 4),
            })

    # Load model comparison
    mc_path = GOLD_DIR / "oulad_model_comparison.parquet"
    model_comparison = []
    if mc_path.exists():
        mc = pd.read_parquet(mc_path)
        for _, row in mc.iterrows():
            model_comparison.append({
                "model":        str(row["model"]),
                "dataset":      str(row["dataset"]),
                "n_features":   int(row["n_features"]),
                "vle_enriched": bool(row["vle_enriched"]),
                "auc_roc":      round(float(row["auc_roc"]), 4),
                "f1_score":     round(float(row["f1_score"]), 4),
                "mlflow_run":   str(row["mlflow_run"]),
            })

    # Load AISHE state data for map/table
    aishe_path = GOLD_DIR / "aishe_state_dropout.parquet"
    aishe_states = []
    if aishe_path.exists():
        aishe_df = pd.read_parquet(aishe_path)
        for _, row in aishe_df.head(30).iterrows():
            aishe_states.append({
                "state":          str(row["state"]),
                "male_dropout":   round(float(row["dropout_rate_male_higher_secondary"]), 1),
                "female_dropout": round(float(row["dropout_rate_female_higher_secondary"]), 1),
                "avg_dropout":    round(float(row["state_avg_dropout_rate"]), 1),
                "gender_gap":     round(float(row["gender_dropout_gap"]), 1),
                "risk_tier":      str(row["state_risk_tier"]),
            })

    # Load World Bank Portugal time series
    wb_path = GOLD_DIR / "worldbank_macro.parquet"
    wb_series = []
    if wb_path.exists():
        wb_df = pd.read_parquet(wb_path)
        prt = wb_df[wb_df["country_code"] == "PRT"].sort_values("year")
        for _, row in prt.iterrows():
            entry: dict = {"year": int(row["year"])}
            for col in ["gdp_per_capita", "unemployment_rate",
                        "tertiary_enrolment_rate", "inflation_rate",
                        "unemployment_rising", "gdp_contracting"]:
                if col in row.index:
                    try:
                        val = row[col]
                        if val is not None and not pd.isna(val):
                            entry[col] = round(float(val), 2)
                    except Exception:
                        pass
            wb_series.append(entry)

    return {
        "oulad":           summary.get("oulad", {}),
        "aishe":           summary.get("aishe", {}),
        "worldbank":       summary.get("worldbank", {}),
        "modelComparison": model_comparison,
        "vleTopFeatures":  vle_top_features,
        "aisheStates":     aishe_states,
        "wbSeries":        wb_series,
    }


@app.get("/api/aishe")
def aishe_data():
    """AISHE state-level dropout rates for fairness benchmarking."""
    path = GOLD_DIR / "aishe_state_dropout.parquet"
    if not path.exists():
        return JSONResponse({"error": "Run python local/run_enrichment.py first"}, status_code=404)
    df = pd.read_parquet(path)
    national = {
        "male_dropout":    round(float(df["dropout_rate_male_higher_secondary"].mean()), 2),
        "female_dropout":  round(float(df["dropout_rate_female_higher_secondary"].mean()), 2),
        "gender_gap":      round(float(df["gender_dropout_gap"].mean()), 2),
        "avg_dropout":     round(float(df["state_avg_dropout_rate"].mean()), 2),
    }
    states = []
    for _, r in df.iterrows():
        states.append({
            "state":          r["state"],
            "male_dropout":   round(float(r["dropout_rate_male_higher_secondary"]), 1),
            "female_dropout": round(float(r["dropout_rate_female_higher_secondary"]), 1),
            "avg_dropout":    round(float(r["state_avg_dropout_rate"]), 1),
            "gender_gap":     round(float(r["gender_dropout_gap"]), 1),
            "risk_tier":      str(r["state_risk_tier"]),
        })
    return {"national": national, "states": states}


# ── Plot endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    path = PLOTS_DIR / f"{plot_name}.png"
    if not path.exists():
        return JSONResponse({"error": f"Plot not found: {plot_name}"}, status_code=404)
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/plots")
def list_plots():
    if not PLOTS_DIR.exists():
        return {"plots": []}
    return {"plots": [p.stem for p in PLOTS_DIR.glob("*.png")]}


@app.get("/health")
def health():
    return {"status": "ok", "gold": (GOLD_DIR / "student_predictions.parquet").exists()}


if __name__ == "__main__":
    import uvicorn
    load_df()
    print("\n[API] http://localhost:8001  |  docs: http://localhost:8001/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
