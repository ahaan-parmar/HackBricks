import { useEffect, useState } from "react";
import { api, SegmentInfo } from "@/api/client";

export default function Segments() {
  const [segments, setSegments] = useState<SegmentInfo[]>([]);
  const [error, setError]       = useState<string | null>(null);

  useEffect(() => {
    api.segments()
      .then((d) => setSegments(d.segments))
      .catch((e) => setError(e.message));
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

  if (!segments.length) {
    return (
      <div>
        <h1 className="text-heading2 mb-6">Segments</h1>
        <p className="text-muted-foreground text-body">Loading…</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-heading2 mb-6">Segments</h1>
      <div className="grid grid-cols-3 gap-4">
        {segments.map((seg) => (
          <div key={seg.name} className="bg-card border border-card-border shadow-card rounded-lg p-5 flex flex-col">
            <h2 className="text-heading4 font-bold mb-1">{seg.name}</h2>
            <p className="text-label text-muted-foreground mb-4">{seg.count} students</p>
            <ul className="space-y-2 mb-6 flex-1">
              {seg.profile.map((p, i) => (
                <li key={i} className="text-body flex gap-2">
                  <span className="text-muted-foreground mt-0.5">•</span>
                  <span>{p}</span>
                </li>
              ))}
            </ul>
            <p className="text-body italic text-muted-foreground border-t border-card-border pt-4">
              {seg.action}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
