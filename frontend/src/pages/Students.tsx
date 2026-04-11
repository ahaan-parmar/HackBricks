import { useEffect, useState, useMemo } from "react";
import { Search } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

type Row = Record<string, string | number | null>;

export default function Students() {
  const [rows, setRows]       = useState<Row[]>([]);
  const [cols, setCols]       = useState<string[]>([]);
  const [total, setTotal]     = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [search, setSearch]   = useState("");

  useEffect(() => {
    fetch(`${BASE_URL}/api/students`)
      .then(r => r.json())
      .then(d => {
        setRows(d.students);
        setCols(["sno", ...d.columns]);
        setTotal(d.total);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const filtered = useMemo(() => {
    if (!search) return rows;
    const q = search.toLowerCase();
    return rows.filter(r =>
      Object.values(r).some(v => String(v ?? "").toLowerCase().includes(q))
    );
  }, [search, rows]);

  const TARGET_STYLE: Record<string, string> = {
    Dropout:  "text-red-600 font-semibold",
    Graduate: "text-green-600 font-semibold",
    Enrolled: "text-blue-600",
  };

  if (error) return (
    <div>
      <h1 className="text-heading2 mb-6">Students</h1>
      <div className="bg-card border border-card-border rounded-lg p-5 text-destructive text-body">
        Could not reach API: {error}. Make sure <code>python backend/main.py</code> is running.
      </div>
    </div>
  );

  return (
    <div>
      <h1 className="text-heading2 mb-6">Students</h1>

      <div className="flex items-center gap-4 mb-4">
        <div className="relative max-w-sm flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search any value…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full bg-card border border-card-border rounded-md pl-9 pr-3 py-2 text-body placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-foreground/20"
          />
        </div>
        <span className="text-label text-muted-foreground ml-auto">
          {loading ? "Loading…" : `${filtered.length} / ${total} rows`}
        </span>
      </div>

      {loading ? (
        <p className="text-muted-foreground text-body">Loading…</p>
      ) : (
        <div className="bg-card border border-card-border shadow-card rounded-lg overflow-auto max-h-[75vh]">
          <table className="w-full text-body text-[12px]" style={{ minWidth: cols.length * 110 }}>
            <thead className="sticky top-0 bg-card z-10">
              <tr className="border-b border-card-border">
                {cols.map(c => (
                  <th key={c} className="text-left text-label text-muted-foreground font-medium px-3 py-2 whitespace-nowrap border-r border-card-border last:border-r-0">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={i} className={`border-b border-card-border ${i % 2 === 1 ? "bg-row-alt" : ""}`}>
                  {cols.map(c => (
                    <td key={c} className={`px-3 py-1.5 whitespace-nowrap border-r border-card-border last:border-r-0 ${c === "target" ? (TARGET_STYLE[String(r[c])] ?? "") : ""}`}>
                      {r[c] === null || r[c] === undefined ? "—" : String(r[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
