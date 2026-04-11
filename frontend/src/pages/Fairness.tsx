import { useEffect, useState } from "react";
import { api, FairnessData } from "@/api/client";
import { getFairnessStatus } from "@/data/mockData";

function FairnessBar({ label, rate, maxRate }: { label: string; rate: number; maxRate: number }) {
  const width = (rate / maxRate) * 100;
  return (
    <div className="flex items-center gap-4 mb-3">
      <span className="text-body w-32 shrink-0">{label}</span>
      <div className="flex-1 bg-muted h-6 rounded-sm overflow-hidden">
        <div className="bg-foreground/30 h-full rounded-sm" style={{ width: `${width}%` }} />
      </div>
      <span className="text-body font-medium w-16 text-right">{(rate * 100).toFixed(1)}%</span>
    </div>
  );
}

function FairnessSection({
  title,
  data,
}: {
  title: string;
  data: FairnessData["demographicParity"];
}) {
  const genderGap     = Math.abs(data.male.rate - data.female.rate);
  const scholarshipGap = Math.abs(data.scholarship.rate - data.noScholarship.rate);
  const maxRate = Math.max(data.male.rate, data.female.rate, data.scholarship.rate, data.noScholarship.rate);

  return (
    <div className="bg-card border border-card-border shadow-card rounded-lg p-5 mb-4">
      <h2 className="text-heading4 font-bold mb-5">{title}</h2>

      <div className="mb-6">
        <p className="text-label text-muted-foreground mb-3">Gender</p>
        <FairnessBar label="Male"   rate={data.male.rate}   maxRate={maxRate} />
        <FairnessBar label="Female" rate={data.female.rate} maxRate={maxRate} />
        <p className="text-label text-muted-foreground mt-1">
          Gap: {(genderGap * 100).toFixed(1)}pp —{" "}
          <span className="font-bold text-foreground">{getFairnessStatus(genderGap)}</span>
        </p>
      </div>

      <div>
        <p className="text-label text-muted-foreground mb-3">Scholarship Status</p>
        <FairnessBar label="Scholarship"    rate={data.scholarship.rate}   maxRate={maxRate} />
        <FairnessBar label="No Scholarship" rate={data.noScholarship.rate} maxRate={maxRate} />
        <p className="text-label text-muted-foreground mt-1">
          Gap: {(scholarshipGap * 100).toFixed(1)}pp —{" "}
          <span className="font-bold text-foreground">{getFairnessStatus(scholarshipGap)}</span>
        </p>
      </div>
    </div>
  );
}

export default function Fairness() {
  const [data, setData] = useState<FairnessData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.fairness()
      .then(setData)
      .catch((e) => setError(e.message));
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

  return (
    <div>
      <h1 className="text-heading2 mb-6">Fairness</h1>
      <FairnessSection title="Demographic Parity" data={data.demographicParity} />
      <FairnessSection title="Equal Opportunity"  data={data.equalOpportunity} />
    </div>
  );
}
