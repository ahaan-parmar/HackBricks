import { useEffect, useState, useMemo } from "react";
import { Search, X, AlertTriangle } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

type Row = Record<string, string | number | null>;

interface FilterOptions {
  riskTiers: string[];
  segments:  string[];
  fairness:  string[];
  outcomes:  string[];
}

// Sticky priority columns always shown first
const PRIORITY_COLS = ["Student ID","Risk Score","Outcome","Risk Tier","Factor 1","Factor 2","Factor 3","Fairness Flag","Segment"];

const TIER_STYLE: Record<string, string> = {
  HIGH:   "bg-red-100 text-red-800 font-bold",
  MEDIUM: "bg-yellow-100 text-yellow-800 font-semibold",
  LOW:    "bg-green-100 text-green-800",
};
const FAIRNESS_STYLE: Record<string, string> = {
  FAIR:   "bg-green-100 text-green-800",
  REVIEW: "bg-yellow-100 text-yellow-800",
  BIASED: "bg-red-100 text-red-800 font-bold",
};
const OUTCOME_STYLE: Record<string, string> = {
  Dropout:  "text-red-600 font-semibold",
  Graduate: "text-green-600 font-semibold",
  Enrolled: "text-blue-600",
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

function RemediationModal({ row, onClose }: { row: Row; onClose: () => void }) {
  const segment   = String(row["Segment"] ?? "General Risk");
  const riskScore = Number(row["Risk Score"] ?? 0);
  const counselorReport = String(row["_counselor_report"] ?? "");
  const hasLLMReport = counselorReport.length > 10;
  const reportPoints = hasLLMReport ? parseReport(counselorReport) : [];

  // Fallback segment-based steps
  const REMEDIATION: Record<string, string[]> = {
    "Academic Strugglers": [
      "Schedule immediate academic counselling session.",
      "Enrol student in peer tutoring / study circle programme.",
      "Review grade transcripts with course coordinator.",
      "Set biweekly check-in to track grade recovery.",
    ],
    "Attendance Risk": [
      "Trigger automated attendance alert to student and guardian.",
      "Identify root cause: health, transport, or motivation.",
      "Offer flexible attendance arrangement or make-up sessions.",
      "Assign welfare officer for monthly welfare check.",
    ],
    "Socioeconomic Risk": [
      "Connect with Financial Aid Office for emergency grant review.",
      "Check scholarship eligibility — may qualify for additional aid.",
      "Offer fee deferment or instalment plan.",
      "Refer to student support fund for living expense assistance.",
    ],
    "General Risk": [
      "Schedule general well-being check-in with student advisor.",
      "Explore any undisclosed personal or external stressors.",
      "Provide information on counselling and mental health services.",
      "Monitor progress monthly and escalate if risk increases.",
    ],
  };
  const steps = REMEDIATION[segment] ?? REMEDIATION["General Risk"];

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-card border border-card-border rounded-lg shadow-xl max-w-lg w-full p-6 max-h-[85vh] overflow-auto">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-heading4 font-bold">{String(row["Student ID"])}</h2>
            <p className="text-label text-muted-foreground">
              Risk Score: <span className="font-bold text-red-700">{riskScore.toFixed(1)}%</span>
              &nbsp;| Segment: <span className="font-semibold">{segment}</span>
            </p>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X size={20} />
          </button>
        </div>

        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md flex gap-2">
          <AlertTriangle size={16} className="text-red-600 mt-0.5 shrink-0" />
          <p className="text-[13px] text-red-700">
            This student is flagged as <strong>HIGH</strong> dropout risk.
            Top factors: <em>{String(row["Factor 1"])}, {String(row["Factor 2"])}, {String(row["Factor 3"])}</em>.
          </p>
        </div>

        {/* AI Counselor Report (from LLM) */}
        {hasLLMReport ? (
          <>
            <h3 className="text-[14px] font-bold mb-3">AI Counselor Report</h3>
            <ul className="space-y-1.5 ml-1 mb-4">
              {reportPoints.map((pt, i) => (
                <li key={i} className="flex gap-2 text-[13px] leading-relaxed">
                  <span className="shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full bg-foreground/50" />
                  <span>{pt}</span>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <>
            <h3 className="text-[14px] font-bold mb-3">Recommended Interventions</h3>
            <ol className="space-y-2">
              {steps.map((step, i) => (
                <li key={i} className="flex gap-3 text-[13px]">
                  <span className="shrink-0 w-5 h-5 rounded-full bg-foreground text-background text-[11px] flex items-center justify-center font-bold mt-0.5">
                    {i + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </>
        )}

        <div className="mt-5 pt-4 border-t border-card-border flex justify-between items-center">
          <p className="text-label text-muted-foreground">
            Financial impact if dropout: <span className="font-bold text-red-700">₹1,20,000</span>
          </p>
          <button
            onClick={onClose}
            className="border border-card-border rounded px-4 py-1.5 text-body hover:bg-muted/50"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default function Students() {
  const [rows,     setRows]     = useState<Row[]>([]);
  const [cols,     setCols]     = useState<string[]>([]);
  const [total,    setTotal]    = useState(0);
  const [filtered, setFiltered] = useState(0);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);
  const [search,   setSearch]   = useState("");
  const [riskFilter,     setRiskFilter]     = useState("ALL");
  const [segmentFilter,  setSegmentFilter]  = useState("ALL");
  const [fairnessFilter, setFairnessFilter] = useState("ALL");
  const [outcomeFilter,  setOutcomeFilter]  = useState("ALL");
  const [filterOptions,  setFilterOptions]  = useState<FilterOptions>({ riskTiers: [], segments: [], fairness: [], outcomes: [] });
  const [remRow, setRemRow] = useState<Row | null>(null);

  const fetchData = () => {
    setLoading(true);
    const params = new URLSearchParams({ limit: "5000" });
    if (riskFilter    !== "ALL") params.set("risk_tier",       riskFilter);
    if (segmentFilter !== "ALL") params.set("segment",         segmentFilter);
    if (fairnessFilter!== "ALL") params.set("fairness_filter", fairnessFilter);

    fetch(`${BASE_URL}/api/students?${params}`)
      .then(r => r.json())
      .then(d => {
        setRows(d.students ?? []);
        setCols(d.columns  ?? []);
        setTotal(d.total   ?? 0);
        setFiltered(d.filtered ?? d.total ?? 0);
        if (d.filterOptions) setFilterOptions(d.filterOptions);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  };

  useEffect(() => { fetchData(); }, [riskFilter, segmentFilter, fairnessFilter]);

  const displayed = useMemo(() => {
    let result = rows;
    // Client-side outcome filter
    if (outcomeFilter !== "ALL") {
      result = result.filter(r => String(r["Outcome"]) === outcomeFilter);
    }
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(r =>
        Object.entries(r).some(([k, v]) => !k.startsWith("_") && String(v ?? "").toLowerCase().includes(q))
      );
    }
    return result;
  }, [search, rows, outcomeFilter]);

  // Split columns: priority first, then the rest (exclude hidden fields)
  const priorityCols = cols.filter(c => PRIORITY_COLS.includes(c));
  const restCols     = cols.filter(c => !PRIORITY_COLS.includes(c) && !c.startsWith("_"));
  const allDisplayCols = [...priorityCols, ...restCols];

  if (error) return (
    <div>
      <h1 className="text-heading2 mb-6">Students</h1>
      <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
        Could not reach API: {error}.
      </div>
    </div>
  );

  return (
    <div>
      {remRow && <RemediationModal row={remRow} onClose={() => setRemRow(null)} />}

      <h1 className="text-heading2 mb-4">Students</h1>

      {/* Filters + search row */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        {/* Search */}
        <div className="relative max-w-xs flex-1">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search any value…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full bg-card border border-card-border rounded-md pl-9 pr-3 py-2 text-body text-[13px] placeholder:text-muted-foreground focus:outline-none"
          />
        </div>

        {/* Risk tier filter */}
        <select
          value={riskFilter}
          onChange={e => setRiskFilter(e.target.value)}
          className="bg-card border border-card-border rounded-md px-3 py-2 text-[13px] focus:outline-none"
        >
          <option value="ALL">All Risk Tiers</option>
          {filterOptions.riskTiers.map(t => <option key={t} value={t}>{t}</option>)}
        </select>

        {/* Segment filter */}
        <select
          value={segmentFilter}
          onChange={e => setSegmentFilter(e.target.value)}
          className="bg-card border border-card-border rounded-md px-3 py-2 text-[13px] focus:outline-none"
        >
          <option value="ALL">All Segments</option>
          {filterOptions.segments.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        {/* Outcome filter */}
        <select
          value={outcomeFilter}
          onChange={e => setOutcomeFilter(e.target.value)}
          className="bg-card border border-card-border rounded-md px-3 py-2 text-[13px] focus:outline-none"
        >
          <option value="ALL">All Outcomes</option>
          {filterOptions.outcomes.map(o => <option key={o} value={o}>{o}</option>)}
        </select>

        {/* Fairness filter */}
        <select
          value={fairnessFilter}
          onChange={e => setFairnessFilter(e.target.value)}
          className="bg-card border border-card-border rounded-md px-3 py-2 text-[13px] focus:outline-none"
        >
          <option value="ALL">All Fairness</option>
          {filterOptions.fairness.map(f => <option key={f} value={f}>{f}</option>)}
        </select>

        <span className="text-label text-muted-foreground ml-auto text-[12px]">
          {loading ? "Loading…" : `${displayed.length} shown / ${total} total`}
        </span>
      </div>

      {loading ? (
        <p className="text-muted-foreground text-body">Loading…</p>
      ) : (
        <div className="bg-card border border-card-border shadow-card rounded-lg overflow-auto max-h-[72vh]">
          <table className="w-full text-[12px]" style={{ minWidth: allDisplayCols.length * 110 + 60 }}>
            <thead className="sticky top-0 bg-card z-10">
              <tr className="border-b border-card-border">
                {allDisplayCols.map(c => (
                  <th
                    key={c}
                    className={`text-left text-label text-muted-foreground font-medium px-3 py-2 whitespace-nowrap border-r border-card-border last:border-r-0 ${PRIORITY_COLS.includes(c) ? "bg-muted/30" : ""}`}
                  >
                    {c}
                  </th>
                ))}
                <th className="text-left text-label text-muted-foreground font-medium px-3 py-2 whitespace-nowrap bg-muted/30">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {displayed.map((r, i) => (
                <tr key={i} className={`border-b border-card-border ${i % 2 === 1 ? "bg-row-alt" : ""}`}>
                  {allDisplayCols.map(c => {
                    const val = r[c];
                    const str = val === null || val === undefined ? "—" : String(val);

                    if (c === "Risk Tier") {
                      return (
                        <td key={c} className="px-3 py-1.5 border-r border-card-border">
                          <span className={`rounded px-1.5 py-0.5 text-[11px] ${TIER_STYLE[str] ?? ""}`}>{str}</span>
                        </td>
                      );
                    }
                    if (c === "Fairness Flag") {
                      return (
                        <td key={c} className="px-3 py-1.5 border-r border-card-border">
                          <span className={`rounded px-1.5 py-0.5 text-[11px] ${FAIRNESS_STYLE[str] ?? ""}`}>{str}</span>
                        </td>
                      );
                    }
                    if (c === "Outcome") {
                      return (
                        <td key={c} className={`px-3 py-1.5 border-r border-card-border whitespace-nowrap ${OUTCOME_STYLE[str] ?? ""}`}>
                          {str}
                        </td>
                      );
                    }
                    if (c === "Risk Score") {
                      const n = Number(val);
                      const color = n >= 70 ? "text-red-600 font-bold" : n >= 40 ? "text-yellow-700 font-semibold" : "text-green-700";
                      return (
                        <td key={c} className={`px-3 py-1.5 border-r border-card-border font-mono whitespace-nowrap ${color}`}>
                          {isNaN(n) ? "—" : `${n.toFixed(1)}%`}
                        </td>
                      );
                    }
                    // Convert any remaining raw probabilities to percentages
                    if (c === "Dropout Probability") {
                      const n = Number(val);
                      return (
                        <td key={c} className="px-3 py-1.5 whitespace-nowrap border-r border-card-border font-mono">
                          {isNaN(n) ? "—" : `${(n * 100).toFixed(1)}%`}
                        </td>
                      );
                    }
                    return (
                      <td key={c} className="px-3 py-1.5 whitespace-nowrap border-r border-card-border last:border-r-0">
                        {str}
                      </td>
                    );
                  })}
                  {/* Action cell */}
                  <td className="px-3 py-1.5 whitespace-nowrap">
                    {String(r["Risk Tier"]) === "HIGH" ? (
                      <button
                        onClick={() => setRemRow(r)}
                        className="bg-red-600 hover:bg-red-700 text-white text-[11px] font-semibold rounded px-2 py-1 transition-colors"
                      >
                        Remediation
                      </button>
                    ) : (
                      <span className="text-muted-foreground text-[11px]">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
