"""
05_llm_recommendations.py - LLM COUNSELOR REPORTS
Uses Claude to generate plain-English counselor reports for HIGH-risk students.
Includes college-specific financial impact.

Run: python local/05_llm_recommendations.py
"""
import sys, json, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import anthropic

ROOT     = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"

COLLEGE_FEES = {
    "Manipal Institute of Technology": 700000,
    "VIT Vellore":                     650000,
    "SRM Institute":                   580000,
    "Amity University":                520000,
    "Default":                         120000,
}
DEFAULT_COLLEGE = "Default"

FEATURE_DISPLAY = {
    "avg_grade":                                    "Average Grade",
    "enrollment_efficiency":                        "Enrollment Efficiency",
    "financial_stress_index":                       "Financial Stress Index",
    "total_approved_units":                         "Total Approved Units",
    "curricular_units_2nd_sem_approved":            "Sem 2 Approved Units",
    "tuition_fees_up_to_date":                      "Tuition Fees Status",
    "debtor":                                       "Outstanding Debt",
    "scholarship_holder":                           "Scholarship Status",
    "age_at_enrollment":                            "Age at Enrollment",
    "curricular_units_1st_sem_without_evaluations": "Missed Evaluations (Sem 1)",
}


def build_student_context(row: pd.Series) -> str:
    shap_feats = row.get("shap_top_features", [])
    if isinstance(shap_feats, str):
        shap_feats = json.loads(shap_feats)

    top_factors = []
    for f in shap_feats[:3]:
        label = FEATURE_DISPLAY.get(f, f.replace("_", " ").title())
        val   = row.get(f, "N/A")
        top_factors.append(f"  - {label}: {val}")

    return f"""Student Profile:
- Dropout Probability: {row.get('dropout_prob', 0):.1%}
- Risk Tier: {row.get('risk_tier', 'HIGH')}
- Risk Segment: {row.get('cluster_label', 'Unknown')}
- Average Grade: {row.get('avg_grade', 0):.2f}/18.9
- Enrollment Efficiency: {row.get('enrollment_efficiency', 0):.0%}
- Financial Stress Index: {row.get('financial_stress_index', 0):.2f}/1.0
- Debtor: {'Yes' if row.get('debtor') else 'No'}
- Tuition Fees Current: {'Yes' if row.get('tuition_fees_up_to_date') else 'No'}
- Scholarship Holder: {'Yes' if row.get('scholarship_holder') else 'No'}

Top Risk Factors (SHAP):
{chr(10).join(top_factors) if top_factors else '  - Data not available'}"""


def generate_report(client: anthropic.Anthropic, student_context: str, college: str, annual_fee: int) -> str:
    prompt = f"""You are a student counselor at {college}. A student has been flagged as HIGH dropout risk by our ML model.

{student_context}

Annual fee at {college}: ₹{annual_fee:,}

Write a concise counselor report (3-4 sentences) in plain English that:
1. Summarises the key risk factors driving the dropout prediction
2. Recommends 2-3 specific, actionable interventions
3. Mentions the financial impact to the institution if the student drops out

Keep it professional, empathetic, and actionable. No bullet points — flowing prose only."""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


def main(college: str = DEFAULT_COLLEGE, max_students: int = 50):
    pred_path = GOLD_DIR / "student_predictions.parquet"
    if not pred_path.exists():
        print("[ERROR] Run 04_cluster_shap.py first")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[WARN] ANTHROPIC_API_KEY not set — generating mock reports")
        use_mock = True
    else:
        use_mock = False
        client   = anthropic.Anthropic(api_key=api_key)

    annual_fee = COLLEGE_FEES.get(college, COLLEGE_FEES["Default"])

    print(f"[05_llm] College: {college}  |  Annual fee: ₹{annual_fee:,}")

    df = pd.read_parquet(pred_path)
    high_risk = df[df["risk_tier"] == "HIGH"].head(max_students).reset_index(drop=True)
    print(f"[05_llm] Generating reports for {len(high_risk)} HIGH-risk students...")

    reports = []
    for i, row in high_risk.iterrows():
        sid = f"STU-{int(i) + 1000:04d}"
        ctx = build_student_context(row)

        if use_mock:
            report_text = (
                f"This student shows a dropout probability of {row.get('dropout_prob',0):.0%} "
                f"driven primarily by {', '.join(row.get('shap_top_features',['academic performance'])[:2])}. "
                f"Immediate interventions recommended include academic counseling, financial aid review, "
                f"and a structured attendance improvement plan. "
                f"If this student drops out, the institution stands to lose ₹{annual_fee:,} in annual fees."
            )
        else:
            try:
                report_text = generate_report(client, ctx, college, annual_fee)
                time.sleep(0.3)   # rate limit
            except Exception as e:
                print(f"  [WARN] LLM error for {sid}: {e}")
                report_text = f"Counselor report unavailable. Risk factors: {row.get('shap_top_features',[])}."

        reports.append({
            "student_id":       sid,
            "dropout_prob":     round(float(row.get("dropout_prob", 0)), 3),
            "risk_tier":        str(row.get("risk_tier", "HIGH")),
            "cluster_label":    str(row.get("cluster_label", "")),
            "counselor_report": report_text,
            "annual_fee":       annual_fee,
            "college":          college,
            "financial_impact": annual_fee,
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(high_risk)}")

    out = pd.DataFrame(reports)
    out.to_parquet(GOLD_DIR / "llm_reports.parquet", index=False)
    print(f"[05_llm] Saved {len(out)} reports → {GOLD_DIR}/llm_reports.parquet")
    print("[05_llm] DONE")
    return out


if __name__ == "__main__":
    import sys
    college = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COLLEGE
    main(college=college)
