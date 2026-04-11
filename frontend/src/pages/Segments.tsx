import { useEffect, useState } from "react";
import { api, ClustersData } from "@/api/client";
import { CheckCircle2 } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

const CLUSTER_COLORS: Record<string, string> = {
  "Academic Strugglers":  "border-l-red-500",
  "Attendance Risk":      "border-l-orange-500",
  "Socioeconomic Risk":   "border-l-purple-500",
  "General Risk":         "border-l-blue-400",
};

const CLUSTER_ACTIONS: Record<string, string> = {
  "Academic Strugglers":  "Recommend tutoring support, peer study groups, and academic counselling sessions.",
  "Attendance Risk":      "Flag for attendance monitoring, send automated alerts, and check on welfare.",
  "Socioeconomic Risk":   "Connect with financial aid office, review scholarship eligibility, defer fees.",
  "General Risk":         "Schedule a general welfare check-in with the student advisor.",
};

function SilhouetteGauge({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(100, (score + 1) / 2 * 100)); // -1 to 1 → 0 to 100
  const label = score >= 0.5 ? "Strong" : score >= 0.25 ? "Moderate" : "Weak";
  const color = score >= 0.5 ? "text-green-700" : score >= 0.25 ? "text-yellow-700" : "text-red-700";
  const barColor = score >= 0.5 ? "bg-green-500" : score >= 0.25 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-4">
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[13px] font-medium">Silhouette Score</span>
          <span className={`text-[14px] font-bold font-mono ${color}`}>{score.toFixed(4)}</span>
        </div>
        <div className="w-full bg-muted h-2.5 rounded overflow-hidden">
          <div className={`h-full rounded ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
        <div className="flex justify-between text-[10px] text-muted-foreground mt-0.5">
          <span>-1 (poor)</span>
          <span>0</span>
          <span>+1 (perfect)</span>
        </div>
      </div>
      <span className={`text-[12px] font-bold ${color} border rounded px-2 py-0.5 ${
        score >= 0.5 ? "bg-green-50 border-green-300" : score >= 0.25 ? "bg-yellow-50 border-yellow-300" : "bg-red-50 border-red-300"
      }`}>
        {label}
      </span>
    </div>
  );
}

export default function Segments() {
  const [data,  setData]  = useState<ClustersData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.clusters().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Segments</h1>
        <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
          Could not reach API: {error}. Make sure <code>python backend/main.py</code> is running.
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Segments</h1>
        <p className="text-muted-foreground text-body">Loading…</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-heading2">Segments</h1>
      <p className="text-body text-muted-foreground -mt-3">
        K-Means clustering (k=4) on academic, financial, and attendance features.
      </p>

      {/* Clustering quality + top 3 feature reasoning */}
      <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
        <h2 className="text-heading4 mb-4">Clustering Quality & Top Feature Reasoning</h2>

        {/* Silhouette score */}
        {data.silhouette !== null && data.silhouette !== undefined && (
          <div className="mb-5">
            <SilhouetteGauge score={data.silhouette} />
            <p className="text-[12px] text-muted-foreground mt-2 italic">
              Silhouette score measures how well-separated clusters are. Higher is better (range: -1 to +1).
              A score above 0.25 indicates meaningful cluster structure.
            </p>
          </div>
        )}

        {/* Top 3 features reasoning */}
        {data.top3Features && data.top3Features.length > 0 && (
          <div className="space-y-3 border-t border-card-border pt-4">
            <p className="text-[13px] font-bold">Top 3 Features Driving Cluster Separation</p>
            {data.top3Features.map((f, i) => (
              <div key={f.feature} className="flex gap-3">
                <span className="shrink-0 w-5 h-5 rounded-full bg-foreground text-background text-[11px] flex items-center justify-center font-bold mt-0.5">
                  {i + 1}
                </span>
                <div>
                  <p className="text-[13px] font-semibold">
                    {f.feature}
                    <span className="font-mono text-muted-foreground font-normal ml-2">(SHAP: {f.shap_value.toFixed(3)})</span>
                  </p>
                  <p className="text-[12px] text-muted-foreground leading-relaxed">{f.reasoning}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Global SHAP importance */}
      <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
        <h2 className="text-heading4 mb-1">Top SHAP Features (Global)</h2>
        <p className="text-label text-muted-foreground mb-4">Mean absolute SHAP value across all students</p>
        <div className="space-y-2">
          {data.globalShap.slice(0, 8).map((f) => {
            const max = data.globalShap[0]?.mean_abs_shap || 1;
            return (
              <div key={f.feature} className="flex items-center gap-3">
                <span className="text-body w-52 shrink-0 text-[13px]">{f.feature}</span>
                <div className="flex-1 bg-muted h-4 rounded overflow-hidden">
                  <div
                    className="bg-foreground/70 h-full rounded"
                    style={{ width: `${(f.mean_abs_shap / max) * 100}%` }}
                  />
                </div>
                <span className="text-label font-mono w-16 text-right">{f.mean_abs_shap.toFixed(4)}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Cluster cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.clusters.map((seg) => (
          <div
            key={seg.name}
            className={`bg-card border border-card-border shadow-card rounded-lg p-5 border-l-4 ${CLUSTER_COLORS[seg.name] ?? "border-l-gray-400"} flex flex-col gap-4`}
          >
            <div>
              <h2 className="text-heading4 font-bold">{seg.name}</h2>
              <p className="text-label text-muted-foreground">{seg.count.toLocaleString()} students</p>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-4 gap-3 text-center">
              <div className="bg-muted/40 rounded p-2">
                <p className="text-[11px] text-muted-foreground">Dropout Rate</p>
                <p className="text-[15px] font-bold">{(seg.dropoutRate * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-muted/40 rounded p-2">
                <p className="text-[11px] text-muted-foreground">Avg Grade</p>
                <p className="text-[15px] font-bold">{seg.avgGrade.toFixed(1)}</p>
              </div>
              <div className="bg-muted/40 rounded p-2">
                <p className="text-[11px] text-muted-foreground">Avg FSI</p>
                <p className="text-[15px] font-bold">{seg.avgFSI.toFixed(2)}</p>
              </div>
              <div className="bg-muted/40 rounded p-2">
                <p className="text-[11px] text-muted-foreground">Avg Risk</p>
                <p className="text-[15px] font-bold">{(seg.avgDropoutProb * 100).toFixed(1)}%</p>
              </div>
            </div>

            {/* Per-cluster SHAP beeswarm */}
            <div>
              <p className="text-label text-muted-foreground mb-2">SHAP Beeswarm — {seg.name}</p>
              <img
                src={`${BASE_URL}${seg.shapPlot}`}
                alt={`SHAP Beeswarm ${seg.name}`}
                className="w-full rounded border border-card-border"
                onError={(e) => {
                  const el = e.target as HTMLImageElement;
                  el.style.display = "none";
                  const p = document.createElement("p");
                  p.className = "text-label text-muted-foreground italic";
                  p.textContent = "Plot not yet generated — run 04_cluster_shap.py";
                  el.parentNode?.appendChild(p);
                }}
              />
            </div>

            {/* Action */}
            <p className="text-body italic text-muted-foreground border-t border-card-border pt-3 text-[13px]">
              {CLUSTER_ACTIONS[seg.name] ?? "Schedule a welfare check-in."}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
