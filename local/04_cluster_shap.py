"""
04_cluster_shap.py - K-MEANS CLUSTERING + SHAP PLOTS
4 clusters: Academic Strugglers, Attendance Risk, Socioeconomic Risk, General Risk
SHAP beeswarm (global + per cluster), feature importance plot.

Run: python local/04_cluster_shap.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.cluster     import KMeans
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models"
PLOTS_DIR = ROOT / "data" / "plots"
GOLD_DIR  = ROOT / "data" / "gold"
PROC_DIR  = ROOT / "data" / "processed"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_NAMES = ["Academic Strugglers", "Attendance Risk", "Socioeconomic Risk", "General Risk"]


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
    return df.reset_index(drop=True)


def assign_cluster_label(cluster_id, centers, scaler, cluster_features):
    """Map numeric cluster id to a meaningful label based on centroid values."""
    # Get centroid in original scale
    c = scaler.inverse_transform([centers[cluster_id]])[0]
    feat_idx = {f: i for i, f in enumerate(cluster_features)}

    grade_idx = feat_idx.get("avg_grade", None)
    eff_idx   = feat_idx.get("enrollment_efficiency", None)
    fsi_idx   = feat_idx.get("financial_stress_index", None)
    att_idx   = feat_idx.get("curricular_units_1st_sem_without_evaluations", None)

    grade = c[grade_idx] if grade_idx is not None else 10
    eff   = c[eff_idx]   if eff_idx   is not None else 0.7
    fsi   = c[fsi_idx]   if fsi_idx   is not None else 0.3
    att   = c[att_idx]   if att_idx   is not None else 0

    if fsi > 0.45:                     return "Socioeconomic Risk"
    if grade < 7.0 or eff < 0.4:      return "Academic Strugglers"
    if att > 1.5:                      return "Attendance Risk"
    return "General Risk"


def plot_global_feature_importance(xgb_model, feature_cols, path):
    imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1a1a2e"] * len(imp)
    imp[::-1].plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_title("Global Feature Importance (XGBoost Gain)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    # Add value labels
    for i, v in enumerate(imp[::-1]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_beeswarm(shap_values, feature_cols, title, path, max_display=15):
    """Manual beeswarm-style SHAP plot using matplotlib."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:max_display]
    top_cols  = [feature_cols[i] for i in top_idx]
    top_shap  = shap_values[:, top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.RdBu_r

    for row_i, feat_i in enumerate(range(len(top_cols))):
        sv   = top_shap[:, feat_i]
        vals = sv
        # jitter y
        y    = np.random.uniform(row_i - 0.3, row_i + 0.3, size=len(sv))
        # color by SHAP value magnitude
        norm = plt.Normalize(vals.min(), vals.max())
        colors = cmap(norm(vals))
        ax.scatter(sv, y, c=colors, alpha=0.4, s=8, linewidths=0)

    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels([c.replace("_", " ").title() for c in top_cols], fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on dropout probability)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="SHAP value")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("[04_cluster_shap] Loading data...")
    df = load_data()

    meta_path = MODEL_DIR / "eval_matrix.json"
    if not (MODEL_DIR / "xgb_model.pkl").exists():
        print("[ERROR] Run 03_train_models.py first")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    X = df[feature_cols].fillna(0)

    # ── XGBoost predictions ───────────────────────────────────────────────────
    print("[04_cluster_shap] Running predictions...")
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    df["dropout_prob"] = xgb_model.predict_proba(X)[:, 1]
    df["risk_score"]   = (df["dropout_prob"] * 100).round(1)
    df["risk_tier"]    = pd.cut(df["dropout_prob"], bins=[-0.001,0.40,0.70,1.001],
                                labels=["LOW","MEDIUM","HIGH"]).astype(str)

    # ── SHAP values (XGBoost native) ──────────────────────────────────────────
    print("[04_cluster_shap] Computing SHAP values...")
    contribs    = xgb_model.get_booster().predict(dmat, pred_contribs=True)
    shap_values = contribs[:, :-1]   # drop bias

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])

    # Top-3 SHAP features per student
    df["shap_top_features"] = [
        shap_df.iloc[i].abs().nlargest(3).index.str.replace("shap_","").tolist()
        for i in range(len(df))
    ]

    # ── K-Means Clustering (4 clusters) ──────────────────────────────────────
    print("[04_cluster_shap] K-Means clustering (k=4)...")
    cluster_features = [
        "avg_grade", "enrollment_efficiency", "financial_stress_index",
        "curricular_units_1st_sem_without_evaluations",
        "curricular_units_2nd_sem_without_evaluations",
        "debtor", "tuition_fees_up_to_date", "scholarship_holder",
        "total_approved_units", "age_at_enrollment",
    ]
    cluster_features = [c for c in cluster_features if c in df.columns]
    Xc = df[cluster_features].fillna(0)

    scaler = StandardScaler()
    Xc_sc  = scaler.fit_transform(Xc)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(Xc_sc)
    df["cluster_id"] = km.labels_

    # Map cluster IDs to 4 distinct meaningful labels by ranking centroids
    centers_orig = scaler.inverse_transform(km.cluster_centers_)
    feat_idx = {f: i for i, f in enumerate(cluster_features)}

    def cval(cid, feat, default=0):
        idx = feat_idx.get(feat)
        return centers_orig[cid][idx] if idx is not None else default

    # Rank clusters on each dimension, assign unique label
    fsi_rank   = sorted(range(4), key=lambda c: cval(c, "financial_stress_index"), reverse=True)
    grade_rank = sorted(range(4), key=lambda c: cval(c, "avg_grade"))   # low grade = bad
    att_rank   = sorted(range(4), key=lambda c: cval(c, "curricular_units_1st_sem_without_evaluations"), reverse=True)

    id_to_label = {}
    id_to_label[fsi_rank[0]]   = "Socioeconomic Risk"   # highest FSI
    remaining = [c for c in range(4) if c != fsi_rank[0]]
    grade_worst = next(c for c in grade_rank if c in remaining)
    id_to_label[grade_worst]   = "Academic Strugglers"  # lowest grade (excluding socio)
    remaining2 = [c for c in remaining if c != grade_worst]
    att_worst  = next(c for c in att_rank if c in remaining2)
    id_to_label[att_worst]     = "Attendance Risk"
    last = [c for c in remaining2 if c != att_worst][0]
    id_to_label[last]          = "General Risk"

    df["cluster_label"] = df["cluster_id"].map(id_to_label)

    print("  Cluster distribution:")
    print(f"  {df['cluster_label'].value_counts().to_dict()}")

    # ── Save gold predictions ─────────────────────────────────────────────────
    out_df = pd.concat([df, shap_df], axis=1)
    out_df.to_parquet(GOLD_DIR / "student_predictions.parquet", index=False)
    print(f"  Saved predictions: {GOLD_DIR}/student_predictions.parquet")

    # Cluster summary
    cluster_summary = (
        df.groupby("cluster_label").agg(
            count=("target_binary","count"),
            dropout_rate=("target_binary","mean"),
            avg_grade=("avg_grade","mean"),
            avg_fsi=("financial_stress_index","mean"),
            avg_dropout_prob=("dropout_prob","mean"),
        ).round(3).reset_index()
    )
    cluster_summary.to_parquet(GOLD_DIR / "cluster_summary.parquet", index=False)

    # Save cluster label map
    with open(MODEL_DIR / "cluster_map.json", "w") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2)

    # ── SHAP Beeswarm — Global ────────────────────────────────────────────────
    print("[04_cluster_shap] Generating SHAP plots...")
    plot_shap_beeswarm(shap_values, feature_cols,
                       "SHAP Beeswarm — All Students",
                       PLOTS_DIR / "shap_beeswarm_global.png")

    # ── SHAP Beeswarm — Per Cluster ───────────────────────────────────────────
    for label in df["cluster_label"].unique():
        idx = df[df["cluster_label"] == label].index
        sv  = shap_values[idx]
        safe_name = label.lower().replace(" ","_").replace("(","").replace(")","")
        plot_shap_beeswarm(sv, feature_cols,
                           f"SHAP Beeswarm — {label}",
                           PLOTS_DIR / f"shap_beeswarm_{safe_name}.png")

    # ── Global Feature Importance ─────────────────────────────────────────────
    plot_global_feature_importance(xgb_model, feature_cols,
                                   PLOTS_DIR / "feature_importance_global.png")

    # ── Mean |SHAP| per feature ───────────────────────────────────────────────
    mean_abs_shap = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap.to_parquet(GOLD_DIR / "shap_importance.parquet", index=False)

    print("[04_cluster_shap] DONE")
    return df, shap_values, feature_cols


if __name__ == "__main__":
    main()
