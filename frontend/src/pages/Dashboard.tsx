import { useEffect, useState } from "react";
import { ArrowUp, ArrowDown } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api, DashboardData } from "@/api/client";

function formatRupee(n: number) {
  if (n >= 10_000_000) return `₹${(n / 10_000_000).toFixed(1)}Cr`;
  if (n >= 100_000)    return `₹${(n / 100_000).toFixed(1)}L`;
  return `₹${n.toLocaleString("en-IN")}`;
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.dashboard()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Dashboard</h1>
        <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
          Could not reach API: {error}. Make sure <code>python backend/main.py</code> is running.
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Dashboard</h1>
        <p className="text-muted-foreground text-body">Loading…</p>
      </div>
    );
  }

  const stats = [
    { label: "Total Students",              value: data.totalStudents.toLocaleString(),       trend: data.trends.totalStudents,     bad: false },
    { label: "At Risk",                     value: data.atRisk.toLocaleString(),              trend: data.trends.atRisk,            bad: true  },
    { label: "Predicted Dropouts",          value: data.predictedDropouts.toString(),         trend: data.trends.predictedDropouts, bad: true  },
    { label: "Estimated Revenue at Risk",   value: formatRupee(data.revenueAtRisk),           trend: data.trends.revenueAtRisk,     bad: true  },
  ];

  const highRisk    = data.atRisk;
  const recoverable = Math.round(highRisk * 0.4);   // 40% estimated recoverable

  return (
    <div>
      <h1 className="text-heading2 mb-6">Dashboard</h1>

      <div className="grid grid-cols-4 gap-4 mb-8">
        {stats.map((s) => {
          const isUp       = s.trend.direction === "up";
          const isNegative = s.bad && isUp;
          const isPositive = s.bad && !isUp;
          return (
            <div key={s.label} className="bg-card border border-card-border shadow-card rounded-lg p-5">
              <p className="text-label text-muted-foreground mb-1">{s.label}</p>
              <p className="text-stat font-bold text-card-foreground">{s.value}</p>
              <div className="flex items-center gap-1 mt-2 text-label">
                {isUp ? (
                  <ArrowUp size={14} className={isNegative ? "text-destructive" : "text-success"} />
                ) : (
                  <ArrowDown size={14} className={isPositive ? "text-success" : "text-destructive"} />
                )}
                <span className={isNegative ? "text-destructive" : isPositive ? "text-success" : "text-muted-foreground"}>
                  {s.trend.value}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
        <h2 className="text-heading4 mb-4">Dropout Risk Distribution by Segment</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data.riskDistribution} barSize={48}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E0E0E0" vertical={false} />
            <XAxis dataKey="segment" tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={{ stroke: "#E0E0E0" }} tickLine={false} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={{ stroke: "#E0E0E0" }} tickLine={false} />
            <Tooltip contentStyle={{ fontSize: 12, border: "1px solid #E0E0E0", borderRadius: 4, boxShadow: "none" }} />
            <Bar dataKey="count" fill="#374151" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-card border border-card-border shadow-card rounded-lg p-5 mt-8">
        <h2 className="text-[16px] font-bold text-card-foreground mb-4">Institutional Financial Impact</h2>
        <div className="grid grid-cols-3 gap-6 mb-4">
          <div>
            <p className="text-label text-muted-foreground mb-1">Total Revenue at Risk</p>
            <p className="text-stat font-bold text-card-foreground">{formatRupee(data.revenueAtRisk)}</p>
          </div>
          <div>
            <p className="text-label text-muted-foreground mb-1">High Risk (FSI above 0.6)</p>
            <p className="text-stat font-bold text-card-foreground">
              {formatRupee(data.riskDistribution.reduce((s, d) => s + d.count, 0) * 120_000)}
            </p>
          </div>
          <div>
            <p className="text-label text-muted-foreground mb-1">Estimated Recoverable via Intervention</p>
            <p className="text-stat font-bold text-foreground">{formatRupee(recoverable * 120_000)}</p>
          </div>
        </div>
        <p className="text-label text-muted-foreground">
          Based on average annual fee of ₹1,20,000 per student across {data.atRisk.toLocaleString()} at-risk students.
          <span className="ml-1 italic">(Silver layer — predictions from FSI threshold)</span>
        </p>
      </div>
    </div>
  );
}
