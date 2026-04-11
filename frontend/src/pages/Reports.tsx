const reports = [
  {
    title: "Semester Risk Summary Report",
    date: "April 2025",
    description: "Full cohort dropout risk analysis across all 2,847 students",
  },
  {
    title: "Segment Intervention Report",
    date: "April 2025",
    description: "Recommended actions per cluster with financial impact breakdown",
  },
  {
    title: "Fairness Audit Report",
    date: "April 2025",
    description: "Demographic parity and equal opportunity metrics logged for compliance",
  },
];

export default function Reports() {
  return (
    <div>
      <h1 className="text-heading2 mb-6">Reports</h1>
      <div className="space-y-4">
        {reports.map((r) => (
          <div
            key={r.title}
            className="bg-card border border-card-border shadow-card rounded-lg p-5 flex items-center justify-between"
          >
            <div>
              <p className="text-body font-bold text-card-foreground">{r.title}</p>
              <p className="text-label text-muted-foreground mt-1">{r.date}</p>
              <p className="text-body text-muted-foreground mt-1">{r.description}</p>
            </div>
            <button className="shrink-0 ml-6 border border-card-border rounded px-4 py-2 text-body text-card-foreground hover:bg-muted/50">
              Download PDF
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
