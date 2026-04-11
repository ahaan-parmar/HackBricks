import { useEffect, useState } from "react";
import { api, RecommendationsData } from "@/api/client";

function formatRupee(n: number) {
  if (n >= 10_000_000) return `₹${(n / 10_000_000).toFixed(1)}Cr`;
  if (n >= 100_000)    return `₹${(n / 100_000).toFixed(1)}L`;
  return `₹${n.toLocaleString("en-IN")}`;
}

const TIER_STYLE: Record<string, string> = {
  HIGH:   "bg-red-100 text-red-800 border-red-300",
  MEDIUM: "bg-yellow-100 text-yellow-800 border-yellow-300",
  LOW:    "bg-green-100 text-green-800 border-green-300",
};

const SEGMENT_COLOR: Record<string, string> = {
  "Academic Strugglers": "text-red-600",
  "Attendance Risk":     "text-orange-600",
  "Socioeconomic Risk":  "text-purple-600",
  "General Risk":        "text-blue-600",
};

/** Make parameter names human-readable */
function readableName(s: string): string {
  return s
    .replace(/_/g, " ")
    .replace(/\b\w/g, ch => ch.toUpperCase())
    .replace(/Fsi/g, "FSI")
    .replace(/Sem /g, "Semester ")
    .replace(/Avg /g, "Average ");
}

/** Parse a counselor report blob into an array of bullet strings. */
function parseReport(text: string): string[] {
  if (!text) return [];
  const numbered = text.split(/\n/).filter(l => l.trim());
  if (numbered.length > 1) {
    return numbered.map(l => readableName(l.replace(/^\d+[.\)]\s*/, "").trim())).filter(Boolean);
  }
  const sentences = text.split(/(?<=\.)\s+/).filter(s => s.trim().length > 10);
  if (sentences.length > 1) {
    return sentences.map(s => readableName(s.trim()));
  }
  return [readableName(text.trim())];
}

export default function Reports() {
  const [feeInput, setFeeInput] = useState("120000");
  const [appliedFee, setAppliedFee] = useState(120000);
  const [data,    setData]      = useState<RecommendationsData | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error,   setError]     = useState<string | null>(null);

  const fetchReports = (fee: number) => {
    setLoading(true);
    setError(null);
    api.recommendations(fee)
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  };

  useEffect(() => {
    fetchReports(appliedFee);
  }, [appliedFee]);

  const handleApplyFee = () => {
    const parsed = parseInt(feeInput.replace(/[^\d]/g, ""), 10);
    if (!isNaN(parsed) && parsed > 0) {
      setAppliedFee(parsed);
    }
  };

  // Compute revenue at risk client-side using the applied fee
  const totalAtRisk = data?.totalAtRisk ?? 0;
  const revenueAtRisk = totalAtRisk * appliedFee;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-heading2">Counselor Reports</h1>
          <p className="text-body text-muted-foreground mt-1">
            AI-generated counselor reports for HIGH-risk students.
          </p>
        </div>

        {/* Fee input box */}
        <div className="flex items-center gap-2">
          <label className="text-[13px] text-muted-foreground font-medium">Annual Fee (₹):</label>
          <input
            type="text"
            value={feeInput}
            onChange={(e) => setFeeInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleApplyFee()}
            className="bg-card border border-card-border rounded-md px-3 py-2 text-body text-[13px] w-32 focus:outline-none focus:ring-1 focus:ring-foreground/20 font-mono"
            placeholder="e.g. 120000"
          />
          <button
            onClick={handleApplyFee}
            className="bg-foreground text-background rounded-md px-4 py-2 text-[13px] font-semibold hover:opacity-90 transition-opacity"
          >
            Calculate
          </button>
        </div>
      </div>

      {/* Summary bar */}
      {data && (
        <div className="bg-card border border-card-border shadow-card rounded-lg p-4 grid grid-cols-3 gap-4">
          <div>
            <p className="text-label text-muted-foreground">Annual Fee</p>
            <p className="text-body font-bold">{formatRupee(appliedFee)}</p>
          </div>
          <div>
            <p className="text-label text-muted-foreground">Students at Risk</p>
            <p className="text-body font-bold">{totalAtRisk}</p>
          </div>
          <div>
            <p className="text-label text-muted-foreground">Total Revenue at Risk</p>
            <p className="text-body font-bold text-red-700">{formatRupee(revenueAtRisk)}</p>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
          Could not reach API: {error}. Make sure <code>python backend/main.py</code> is running.
        </div>
      )}

      {loading && <p className="text-muted-foreground text-body">Loading reports…</p>}

      {data && !loading && data.reports.length === 0 && (
        <div className="bg-card border border-card-border rounded-lg p-5 text-muted-foreground text-body">
          No reports available. Run <code>python local/05_llm_recommendations.py</code> to generate them.
        </div>
      )}

      {data && !loading && (
        <div className="space-y-3">
          {data.reports.map((r) => {
            const points = parseReport(r.counselorReport);
            return (
              <div key={r.studentId} className="bg-card border border-card-border shadow-card rounded-lg p-5">
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="font-mono text-[13px] font-bold text-foreground">{r.studentId}</span>
                    <span className={`border rounded px-2 py-0.5 text-xs font-bold ${TIER_STYLE[r.riskTier] ?? ""}`}>
                      {r.riskTier}
                    </span>
                    <span className={`text-[13px] font-semibold ${SEGMENT_COLOR[r.segment] ?? "text-foreground"}`}>
                      {r.segment}
                    </span>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="text-label text-muted-foreground">Dropout Prob.</p>
                    <p className="text-body font-bold">{(r.dropoutProb * 100).toFixed(1)}%</p>
                  </div>
                </div>

                {/* Report as bullet points */}
                <ul className="space-y-1.5 ml-1">
                  {points.map((pt, idx) => (
                    <li key={idx} className="flex gap-2 text-[13px] leading-relaxed">
                      <span className="shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full bg-foreground/50" />
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>

                <div className="mt-3 pt-3 border-t border-card-border flex items-center justify-between">
                  <p className="text-label text-muted-foreground">
                    Financial impact if dropout: <span className="font-semibold text-red-700">{formatRupee(appliedFee)}</span>
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
