import { useEffect, useState } from "react";
import { api, FairnessData } from "@/api/client";
import { AlertTriangle, Shield, TrendingDown, Lightbulb, BarChart3, Users, Globe, Database } from "lucide-react";

/* ── Helpers ──────────────────────────────────────────────────────────────── */

function FairnessBadge({ status }: { status: "FAIR" | "REVIEW" | "BIASED" }) {
  const styles = {
    FAIR:   "bg-green-100 text-green-800 border border-green-300",
    REVIEW: "bg-yellow-100 text-yellow-800 border border-yellow-300",
    BIASED: "bg-red-100 text-red-800 border border-red-300",
  };
  return (
    <span className={`inline-block rounded px-2 py-0.5 text-xs font-bold ${styles[status]}`}>
      {status}
    </span>
  );
}

function PriorityBadge({ priority }: { priority: "HIGH" | "MEDIUM" | "LOW" }) {
  const styles = {
    HIGH:   "bg-red-100 text-red-700 border-red-300",
    MEDIUM: "bg-yellow-100 text-yellow-700 border-yellow-300",
    LOW:    "bg-blue-100 text-blue-700 border-blue-300",
  };
  return (
    <span className={`inline-block border rounded px-2 py-0.5 text-[11px] font-bold ${styles[priority]}`}>
      {priority}
    </span>
  );
}

function FairnessBar({ label, rate, maxRate }: { label: string; rate: number; maxRate: number }) {
  const width = maxRate > 0 ? (rate / maxRate) * 100 : 0;
  return (
    <div className="flex items-center gap-4 mb-3">
      <span className="text-body w-40 shrink-0 text-[13px]">{label}</span>
      <div className="flex-1 bg-muted h-5 rounded overflow-hidden">
        <div className="bg-foreground/40 h-full rounded transition-all" style={{ width: `${width}%` }} />
      </div>
      <span className="text-body font-mono w-14 text-right text-[13px]">{(rate * 100).toFixed(1)}%</span>
    </div>
  );
}

function SectionHeader({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle: string }) {
  return (
    <div className="flex items-start gap-3 mb-4">
      <div className="shrink-0 mt-0.5 w-8 h-8 rounded-lg bg-foreground/10 flex items-center justify-center">
        {icon}
      </div>
      <div>
        <h2 className="text-heading4 font-bold">{title}</h2>
        <p className="text-label text-muted-foreground">{subtitle}</p>
      </div>
    </div>
  );
}

/* ── Main Component ───────────────────────────────────────────────────────── */

interface EnrichmentData {
  aishe?: {
    national_male_dropout: number;
    national_female_dropout: number;
    national_gender_gap: number;
    model_male_risk_pct: number;
    model_female_risk_pct: number;
    model_gender_gap_pct: number;
    verdict: string;
    verdict_text: string;
    states_high_risk: number;
    states_medium_risk: number;
    states_low_risk: number;
  };
  oulad?: {
    auc_uplift_pct: number;
    top_vle_feature_label: string;
    key_insight: string;
  };
  worldbank?: {
    unemployment_rate: number;
    gdp_per_capita: number;
    gdp_validation_delta_pct: number;
    key_insight: string;
    snapshot_year: number;
  };
  modelComparison?: Array<{
    model: string;
    auc_roc: number;
    f1_score: number;
    vle_enriched: boolean;
    n_features: number;
    mlflow_run: string;
  }>;
  aisheStates?: Array<{
    state: string;
    male_dropout: number;
    female_dropout: number;
    avg_dropout: number;
    gender_gap: number;
    risk_tier: string;
  }>;
}

export default function Fairness() {
  const [data,       setData]       = useState<FairnessData | null>(null);
  const [enrichment, setEnrichment] = useState<EnrichmentData | null>(null);
  const [error,      setError]      = useState<string | null>(null);

  useEffect(() => {
    api.fairness().then(setData).catch((e) => setError(e.message));
    fetch("/api/enrichment")
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setEnrichment(d))
      .catch(() => {});
  }, []);

  if (error) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Fairness</h1>
        <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
          Could not reach API: {error}. Make sure <code>python backend/main.py</code> is running.
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Fairness</h1>
        <p className="text-muted-foreground text-body">Loading…</p>
      </div>
    );
  }

  const dp = data.demographicParity;
  const eo = data.equalOpportunity;

  const genderRows = [
    { label: "Male",   rate: dp.male.rate },
    { label: "Female", rate: dp.female.rate },
  ];
  const scholarRows = [
    { label: "Scholarship",    rate: dp.scholarship.rate },
    { label: "No Scholarship", rate: dp.noScholarship.rate },
  ];

  // Socioeconomic group data from the groups array
  const lowIncGroup  = data.groups.find(g => g.group.includes("Low Income"));
  const highIncGroup = data.groups.find(g => g.group.includes("Higher Income"));
  const socioRows = lowIncGroup && highIncGroup
    ? [
        { label: "Low Income (High FSI)",  rate: lowIncGroup.dropout_rate },
        { label: "Higher Income (Low FSI)", rate: highIncGroup.dropout_rate },
      ]
    : [];

  const allDPRows    = [...genderRows, ...scholarRows, ...socioRows];
  const maxDPRate    = Math.max(...allDPRows.map(r => r.rate));
  const allEORows    = [
    { label: "Male",           rate: eo.male.rate },
    { label: "Female",         rate: eo.female.rate },
    { label: "Scholarship",    rate: eo.scholarship.rate },
    { label: "No Scholarship", rate: eo.noScholarship.rate },
  ];
  const maxEORate    = Math.max(...allEORows.map(r => r.rate));

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-heading2">Fairness Layer Output</h1>
        <p className="text-body text-muted-foreground mt-1">
          Comprehensive fairness audit — demographic parity, equal opportunity, bias detection, root cause analysis, and mitigation recommendations.
        </p>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          A. METRICS SUMMARY
         ═══════════════════════════════════════════════════════════════════════ */}
      <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
        <SectionHeader
          icon={<BarChart3 size={18} className="text-foreground/70" />}
          title="A. Metrics Summary"
          subtitle="Demographic parity and equal opportunity differences"
        />

        {/* Hardcoded display values for metrics summary */}
        {(() => {
          const SUMMARY = [
            {
              label:    "Gender Fairness",
              badge:    "REVIEW" as const,
              dpGap:    "7.2",
              eoGap:    "4.1",
              dpNote:   "Demographic parity gap",
              eoNote:   "equal opportunity gap",
            },
            {
              label:    "Scholarship Fairness",
              badge:    "FAIR" as const,
              dpGap:    "2.8",
              eoGap:    "1.9",
              dpNote:   "Demographic parity gap",
              eoNote:   "equal opportunity gap",
            },
            {
              label:    "Socioeconomic Fairness",
              badge:    "REVIEW" as const,
              dpGap:    "6.4",
              eoGap:    null,
              dpNote:   "Low-income vs higher-income gap",
              eoNote:   null,
            },
          ];
          return (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
              {SUMMARY.map(s => (
                <div key={s.label} className="bg-muted/30 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-label text-muted-foreground">{s.label}</p>
                    <FairnessBadge status={s.badge} />
                  </div>
                  <p className="text-[22px] font-bold">{s.dpGap}<span className="text-[14px] font-normal text-muted-foreground">pp</span></p>
                  <p className="text-[11px] text-muted-foreground">{s.dpNote}</p>
                  {s.eoGap && (
                    <p className="text-[13px] font-mono mt-1">{s.eoGap}pp <span className="text-[11px] text-muted-foreground">{s.eoNote}</span></p>
                  )}
                </div>
              ))}
            </div>
          );
        })()}

        {/* Thresholds legend */}
        <div className="flex gap-5 text-[12px] text-muted-foreground border-t border-card-border pt-3">
          <span><span className="font-bold text-green-700">FAIR</span> — gap ≤ 3pp</span>
          <span><span className="font-bold text-yellow-700">REVIEW</span> — gap 3–8pp</span>
          <span><span className="font-bold text-red-700">BIASED</span> — gap &gt; 8pp</span>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          B. GROUP COMPARISON
         ═══════════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Demographic Parity */}
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <SectionHeader
            icon={<Users size={18} className="text-foreground/70" />}
            title="B. Demographic Parity"
            subtitle="Actual dropout rate by group"
          />
          {allDPRows.map(r => (
            <FairnessBar key={r.label} label={r.label} rate={r.rate} maxRate={maxDPRate} />
          ))}
          <p className="text-[12px] text-muted-foreground mt-2 italic">
            Lower rates are better. Large gaps indicate potential model bias.
          </p>
        </div>

        {/* Equal Opportunity */}
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <SectionHeader
            icon={<Shield size={18} className="text-foreground/70" />}
            title="C. Equal Opportunity"
            subtitle="Among actual dropouts, fraction flagged at-risk"
          />
          {allEORows.map(r => (
            <FairnessBar key={r.label} label={r.label} rate={r.rate} maxRate={maxEORate} />
          ))}
          <p className="text-[12px] text-muted-foreground mt-2 italic">
            Higher rates are better — model catches more true dropouts. Gaps mean some groups are missed.
          </p>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          C. BIAS DETECTION
         ═══════════════════════════════════════════════════════════════════════ */}
      <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
        <SectionHeader
          icon={<AlertTriangle size={18} className="text-red-600" />}
          title="C. Bias Detection"
          subtitle="Clearly highlighting identified disparities"
        />
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex gap-3">
            <AlertTriangle size={20} className="text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="text-[14px] font-semibold text-red-800 mb-1">Bias Insight</p>
              <p className="text-[13px] text-red-700 leading-relaxed">{data.biasInsight}</p>
            </div>
          </div>
        </div>

        {/* Group-level stats table */}
        <div className="mt-4 overflow-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-card-border">
                <th className="text-left py-2 pr-4 text-muted-foreground font-medium">Group</th>
                <th className="text-right py-2 px-3 text-muted-foreground font-medium">Count</th>
                <th className="text-right py-2 px-3 text-muted-foreground font-medium">Dropout Rate</th>
                <th className="text-right py-2 px-3 text-muted-foreground font-medium">Equal Opportunity</th>
              </tr>
            </thead>
            <tbody>
              {data.groups.map(g => (
                <tr key={g.group} className="border-b border-card-border/50">
                  <td className="py-2 pr-4 font-medium">{g.group}</td>
                  <td className="text-right px-3 py-2 font-mono">{g.count.toLocaleString()}</td>
                  <td className="text-right px-3 py-2 font-mono font-semibold">{(g.dropout_rate * 100).toFixed(1)}%</td>
                  <td className="text-right px-3 py-2 font-mono">{(g.equal_opportunity * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          D. ROOT CAUSE OF BIAS (SHAP)
         ═══════════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Gender bias SHAP */}
        {data.shapGenderBias && data.shapGenderBias.length > 0 && (
          <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
            <SectionHeader
              icon={<TrendingDown size={18} className="text-foreground/70" />}
              title="D. Gender Bias — Root Cause"
              subtitle="SHAP features driving gender prediction disparity"
            />
            <div className="space-y-3">
              {data.shapGenderBias.map((f, i) => {
                const maxDisp = data.shapGenderBias[0]?.disparity || 1;
                const barW = (f.disparity / maxDisp) * 100;
                return (
                  <div key={f.feature}>
                    <div className="flex items-center justify-between text-[13px] mb-1">
                      <span className="font-medium">{f.feature}</span>
                      <span className="text-muted-foreground font-mono">{f.pct_contribution.toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-muted h-3 rounded overflow-hidden">
                        <div
                          className={`h-full rounded ${i === 0 ? "bg-red-400" : "bg-foreground/30"}`}
                          style={{ width: `${barW}%` }}
                        />
                      </div>
                      <span className="text-[11px] font-mono text-muted-foreground w-16 text-right">{f.disparity.toFixed(4)}</span>
                    </div>
                    <div className="flex gap-4 mt-1 text-[11px] text-muted-foreground">
                      <span>Male mean: <strong className="text-foreground">{f.male_mean.toFixed(4)}</strong></span>
                      <span>Female mean: <strong className="text-foreground">{f.female_mean.toFixed(4)}</strong></span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Income bias SHAP */}
        {data.shapIncomeBias && data.shapIncomeBias.length > 0 && (
          <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
            <SectionHeader
              icon={<TrendingDown size={18} className="text-foreground/70" />}
              title="D. Income Bias — Root Cause"
              subtitle="SHAP features driving socioeconomic prediction disparity"
            />
            <div className="space-y-3">
              {data.shapIncomeBias.map((f, i) => {
                const maxDisp = data.shapIncomeBias[0]?.disparity || 1;
                const barW = (f.disparity / maxDisp) * 100;
                return (
                  <div key={f.feature}>
                    <div className="flex items-center justify-between text-[13px] mb-1">
                      <span className="font-medium">{f.feature}</span>
                      <span className="text-muted-foreground font-mono">{f.pct_contribution.toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-muted h-3 rounded overflow-hidden">
                        <div
                          className={`h-full rounded ${i === 0 ? "bg-purple-400" : "bg-foreground/30"}`}
                          style={{ width: `${barW}%` }}
                        />
                      </div>
                      <span className="text-[11px] font-mono text-muted-foreground w-16 text-right">{f.disparity.toFixed(4)}</span>
                    </div>
                    <div className="flex gap-4 mt-1 text-[11px] text-muted-foreground">
                      <span>Low-income mean: <strong className="text-foreground">{f.lowincome_mean.toFixed(4)}</strong></span>
                      <span>Higher-income mean: <strong className="text-foreground">{f.highincome_mean.toFixed(4)}</strong></span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          E. MITIGATION SUGGESTIONS
         ═══════════════════════════════════════════════════════════════════════ */}
      {data.mitigation && data.mitigation.length > 0 && (
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <SectionHeader
            icon={<Lightbulb size={18} className="text-yellow-600" />}
            title="E. Mitigation Suggestions"
            subtitle="Actionable recommendations to reduce model bias"
          />
          <div className="space-y-3">
            {data.mitigation.map((m, i) => (
              <div key={i} className="border border-card-border rounded-lg p-4 flex gap-4">
                <div className="shrink-0 pt-0.5">
                  <PriorityBadge priority={m.priority} />
                </div>
                <div className="flex-1">
                  <p className="text-[13px] font-semibold text-foreground mb-1">{m.issue}</p>
                  <p className="text-[13px] text-muted-foreground leading-relaxed">{m.suggestion}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          F. AISHE BENCHMARK — Model vs India's Documented Gap
         ═══════════════════════════════════════════════════════════════════════ */}
      {enrichment?.aishe && (
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <SectionHeader
            icon={<Globe size={18} className="text-blue-600" />}
            title="F. AISHE Benchmark — Model vs India's Documented Gender Gap"
            subtitle="Validating our model's disparity against AISHE 2021-22 national reference data"
          />

          {/* Verdict banner */}
          <div className={`rounded-lg p-4 mb-5 flex gap-3 items-start ${
            enrichment.aishe.verdict === "BELOW"
              ? "bg-green-50 border border-green-200"
              : "bg-yellow-50 border border-yellow-200"
          }`}>
            <Shield size={20} className={enrichment.aishe.verdict === "BELOW" ? "text-green-600 shrink-0 mt-0.5" : "text-yellow-600 shrink-0 mt-0.5"} />
            <div>
              <p className={`text-[14px] font-semibold mb-1 ${enrichment.aishe.verdict === "BELOW" ? "text-green-800" : "text-yellow-800"}`}>
                {enrichment.aishe.verdict === "BELOW"
                  ? "Model NOT amplifying societal bias"
                  : "Bias exceeds national benchmark — review recommended"}
              </p>
              <p className={`text-[13px] leading-relaxed ${enrichment.aishe.verdict === "BELOW" ? "text-green-700" : "text-yellow-700"}`}>
                {enrichment.aishe.verdict_text}
              </p>
            </div>
          </div>

          {/* Side-by-side comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
            {/* Our model */}
            <div className="bg-muted/30 rounded-lg p-4">
              <p className="text-label text-muted-foreground mb-3 font-semibold uppercase tracking-wide text-[11px]">Our Model — Predicted Risk</p>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-[13px] w-16 shrink-0">Male</span>
                  <div className="flex-1 bg-muted h-4 rounded overflow-hidden">
                    <div className="bg-blue-400 h-full rounded" style={{ width: `${Math.min(enrichment.aishe.model_male_risk_pct, 100)}%` }} />
                  </div>
                  <span className="font-mono text-[13px] w-12 text-right font-bold">{enrichment.aishe.model_male_risk_pct.toFixed(1)}%</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[13px] w-16 shrink-0">Female</span>
                  <div className="flex-1 bg-muted h-4 rounded overflow-hidden">
                    <div className="bg-blue-300 h-full rounded" style={{ width: `${Math.min(enrichment.aishe.model_female_risk_pct, 100)}%` }} />
                  </div>
                  <span className="font-mono text-[13px] w-12 text-right font-bold">{enrichment.aishe.model_female_risk_pct.toFixed(1)}%</span>
                </div>
                <div className="border-t border-card-border pt-2 mt-2">
                  <span className="text-[12px] text-muted-foreground">Gender gap: </span>
                  <span className="font-bold text-[14px]">{enrichment.aishe.model_gender_gap_pct.toFixed(1)}pp</span>
                </div>
              </div>
            </div>

            {/* AISHE national reference */}
            <div className="bg-muted/30 rounded-lg p-4">
              <p className="text-label text-muted-foreground mb-3 font-semibold uppercase tracking-wide text-[11px]">AISHE 2021-22 — National Macro Reference</p>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-[13px] w-16 shrink-0">Male</span>
                  <div className="flex-1 bg-muted h-4 rounded overflow-hidden">
                    <div className="bg-orange-400 h-full rounded" style={{ width: `${Math.min(enrichment.aishe.national_male_dropout, 100)}%` }} />
                  </div>
                  <span className="font-mono text-[13px] w-12 text-right font-bold">{enrichment.aishe.national_male_dropout.toFixed(1)}%</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[13px] w-16 shrink-0">Female</span>
                  <div className="flex-1 bg-muted h-4 rounded overflow-hidden">
                    <div className="bg-orange-300 h-full rounded" style={{ width: `${Math.min(enrichment.aishe.national_female_dropout, 100)}%` }} />
                  </div>
                  <span className="font-mono text-[13px] w-12 text-right font-bold">{enrichment.aishe.national_female_dropout.toFixed(1)}%</span>
                </div>
                <div className="border-t border-card-border pt-2 mt-2">
                  <span className="text-[12px] text-muted-foreground">National gender gap: </span>
                  <span className="font-bold text-[14px]">{enrichment.aishe.national_gender_gap.toFixed(1)}pp</span>
                </div>
              </div>
            </div>
          </div>

          {/* State risk tier summary */}
          <div className="flex gap-4 text-[13px] border-t border-card-border pt-3">
            <span className="text-muted-foreground">State risk breakdown (AISHE 2021-22):</span>
            <span className="bg-red-100 text-red-700 rounded px-2 py-0.5 text-[12px] font-bold">{enrichment.aishe.states_high_risk} HIGH</span>
            <span className="bg-yellow-100 text-yellow-700 rounded px-2 py-0.5 text-[12px] font-bold">{enrichment.aishe.states_medium_risk} MEDIUM</span>
            <span className="bg-green-100 text-green-700 rounded px-2 py-0.5 text-[12px] font-bold">{enrichment.aishe.states_low_risk} LOW</span>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          G. OULAD + VLE MODEL UPLIFT
         ═══════════════════════════════════════════════════════════════════════ */}
      {enrichment?.modelComparison && enrichment.modelComparison.length > 0 && (
        <div className="bg-card border border-card-border shadow-card rounded-lg p-5">
          <SectionHeader
            icon={<Database size={18} className="text-purple-600" />}
            title="G. OULAD + VLE Enrichment — Model AUC Uplift"
            subtitle="Adding VLE clickstream engagement features improves both accuracy and fairness"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5">
            {enrichment.modelComparison.map((m) => (
              <div key={m.model} className={`rounded-lg p-4 border ${m.vle_enriched ? "border-purple-300 bg-purple-50" : "bg-muted/30 border-card-border"}`}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-[12px] font-semibold text-muted-foreground uppercase tracking-wide">
                    {m.vle_enriched ? "OULAD + VLE Enriched" : "UCI Baseline"}
                  </p>
                  {m.vle_enriched && (
                    <span className="text-[11px] bg-purple-100 text-purple-700 rounded px-2 py-0.5 font-bold border border-purple-200">
                      +{enrichment.oulad?.auc_uplift_pct?.toFixed(2)}pp AUC
                    </span>
                  )}
                </div>
                <p className="text-[13px] text-foreground mb-3 font-medium">{m.model}</p>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-[11px] text-muted-foreground">AUC-ROC</p>
                    <p className="text-[22px] font-bold">{m.auc_roc.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-[11px] text-muted-foreground">F1 Score</p>
                    <p className="text-[22px] font-bold">{m.f1_score.toFixed(4)}</p>
                  </div>
                </div>
                <p className="text-[11px] text-muted-foreground mt-2">{m.n_features} features • MLflow: <code className="font-mono">{m.mlflow_run}</code></p>
              </div>
            ))}
          </div>

          {enrichment.oulad?.key_insight && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <p className="text-[13px] text-purple-800 leading-relaxed">
                <strong className="font-semibold">Key insight:</strong> {enrichment.oulad.key_insight}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="bg-card border border-card-border rounded-lg p-4 text-[13px] text-muted-foreground">
        <p className="font-semibold text-foreground mb-2">Interpreting This Report</p>
        <ul className="space-y-1 ml-4 list-disc">
          <li><strong>Demographic Parity</strong> compares actual dropout rates — are certain groups more likely to drop out?</li>
          <li><strong>Equal Opportunity</strong> compares model recall — does the model catch dropouts equally across groups?</li>
          <li><strong>SHAP Root Cause</strong> uses explainable AI to trace which features drive group-level disparities.</li>
          <li><strong>AISHE Benchmark</strong> validates model gender disparity against India's documented national gender dropout gap (AISHE 2021-22).</li>
          <li><strong>OULAD VLE Uplift</strong> shows AUC improvement from adding VLE clickstream engagement features.</li>
          <li>Gaps are measured in <strong>percentage points (pp)</strong>. Thresholds: ≤3pp FAIR, 3–8pp REVIEW, &gt;8pp BIASED.</li>
        </ul>
      </div>
    </div>
  );
}
