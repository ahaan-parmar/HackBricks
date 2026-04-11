import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api, DashboardData, ModelInfo, plotUrl } from "@/api/client";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

function FairnessBadge({ status }: { status: "FAIR" | "REVIEW" | "BIASED" }) {
  const styles = {
    FAIR:   "bg-green-100 text-green-800 border-green-300",
    REVIEW: "bg-yellow-100 text-yellow-800 border-yellow-300",
    BIASED: "bg-red-100 text-red-800 border-red-300",
  };
  return (
    <span className={`inline-block border rounded px-2 py-0.5 text-xs font-bold ${styles[status]}`}>
      {status}
    </span>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
      <p className="text-label text-muted-foreground mb-1">{label}</p>
      <p className="text-stat font-bold text-card-foreground">{value}</p>
      {sub && <p className="text-label text-muted-foreground mt-1">{sub}</p>}
    </div>
  );
}

export default function Dashboard() {
  const [data,  setData]  = useState<DashboardData | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.dashboard().then(setData).catch((e) => setError(e.message));
    api.model().then(setModel).catch(() => {});
  }, []);

  if (error) return (
    <div>
      <h1 className="text-heading2 mb-6">Dashboard</h1>
      <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
        Could not reach API: {error}. Make sure <code>/c/Python313/python.exe backend/main.py</code> is running.
      </div>
    </div>
  );

  if (!data) return (
    <div>
      <h1 className="text-heading2 mb-6">Dashboard</h1>
      <p className="text-muted-foreground text-body">Loading…</p>
    </div>
  );

  return (
    <div className="space-y-6">
      <h1 className="text-heading2">Dashboard</h1>

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Students"    value={data.totalStudents.toLocaleString()} />
        <StatCard label="Actual Dropouts"   value={data.actualDropouts.toLocaleString()}
                  sub={`${(data.dropoutRate * 100).toFixed(1)}% dropout rate`} />
        <StatCard label="Predicted At-Risk" value={data.atRisk.toLocaleString()}
                  sub={`${data.highRisk} HIGH risk`} />
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <p className="text-label text-muted-foreground mb-1">Fairness Status</p>
          <div className="mt-1"><FairnessBadge status={data.fairnessStatus} /></div>
          <p className="text-label text-muted-foreground mt-2">
            Gender gap: {(data.genderGap * 100).toFixed(1)}pp
          </p>
        </div>
      </div>

      {/* Risk distribution */}
      <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
        <h2 className="text-heading4 mb-4">Dropout Risk Distribution by Segment</h2>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.riskDistribution} barSize={48}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" vertical={false} />
            <XAxis dataKey="segment" tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={{ stroke: "#E0E0E0" }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={{ stroke: "#E0E0E0" }} tickLine={false} />
            <Tooltip contentStyle={{ fontSize: 12 }} />
            <Bar dataKey="count" fill="#374151" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── SHAP explainability (above model scores) ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <h2 className="text-heading4 mb-1">Global Feature Importance</h2>
          <p className="text-label text-muted-foreground mb-3">XGBoost gain — top predictors of dropout</p>
          <img
            src={plotUrl("feature_importance_global")}
            alt="Global Feature Importance"
            className="w-full rounded"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
          />

          {/* Top 3 features description */}
          {data.topFeatures && data.topFeatures.length > 0 && (
            <div className="mt-4 space-y-3 border-t border-card-border pt-4">
              <p className="text-[13px] font-bold">Top 3 Dropout Drivers</p>
              {data.topFeatures.map((f, i) => (
                <div key={f.feature} className="flex gap-3">
                  <span className="shrink-0 w-5 h-5 rounded-full bg-foreground text-background text-[11px] flex items-center justify-center font-bold mt-0.5">
                    {i + 1}
                  </span>
                  <div>
                    <p className="text-[13px] font-semibold">{f.feature} <span className="font-mono text-muted-foreground font-normal">({f.mean_abs_shap.toFixed(3)})</span></p>
                    <p className="text-[12px] text-muted-foreground leading-relaxed">{f.description}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <h2 className="text-heading4 mb-1">SHAP Beeswarm — All Students</h2>
          <p className="text-label text-muted-foreground mb-3">Each dot = one student; color = feature value</p>
          <img
            src={plotUrl("shap_beeswarm_global")}
            alt="SHAP Beeswarm Global"
            className="w-full rounded"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
          />
        </div>
      </div>

      {/* ── Model evaluation matrix ── */}
      {model && (
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <h2 className="text-heading4 mb-1">Model Evaluation Matrix</h2>
          <p className="text-label text-muted-foreground mb-4">
            Winner: <span className="font-bold text-foreground">{model.winner}</span> &nbsp;|&nbsp;
            {model.splitRatio} stratified split &nbsp;|&nbsp;
            Train: {model.trainSize.toLocaleString()} &nbsp; Test: {model.testSize.toLocaleString()}
          </p>
          <div className="overflow-auto">
            <table className="w-full text-body text-[13px]">
              <thead>
                <tr className="border-b border-card-border">
                  <th className="text-left py-2 pr-6 text-muted-foreground font-medium">Metric</th>
                  {model.models.map(m => (
                    <th key={m.name} className="text-right py-2 px-4 text-muted-foreground font-medium">
                      {m.name} {m.name === model.winner ? "★" : ""}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(["accuracy","precision","recall","f1","roc_auc"] as const).map(metric => (
                  <tr key={metric} className="border-b border-card-border/50">
                    <td className="py-2 pr-6 text-muted-foreground capitalize">{metric.replace("_"," ")}</td>
                    {model.models.map(m => (
                      <td key={m.name} className={`text-right px-4 py-2 font-mono ${m.name === model.winner ? "font-bold" : ""}`}>
                        {(m[metric as keyof typeof m] as number).toFixed(4)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Confusion matrices + ROC */}
      {model && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
            <h2 className="text-[14px] font-bold mb-2">Confusion Matrix — Logistic Regression</h2>
            <img src={`${BASE_URL}/api/plots/cm_lr`} alt="LR Confusion Matrix" className="w-full rounded"
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
          </div>
          <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
            <h2 className="text-[14px] font-bold mb-2">Confusion Matrix — XGBoost</h2>
            <img src={`${BASE_URL}/api/plots/cm_xgb`} alt="XGBoost Confusion Matrix" className="w-full rounded"
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
          </div>
          <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
            <h2 className="text-[14px] font-bold mb-2">ROC Curve Comparison</h2>
            <img src={`${BASE_URL}/api/plots/roc_comparison`} alt="ROC Curve" className="w-full rounded"
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
          </div>
        </div>
      )}
    </div>
  );
}
