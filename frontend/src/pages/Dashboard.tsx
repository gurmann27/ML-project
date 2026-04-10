import { useEffect, useState } from "react";
import { apiGet } from "../lib/api";

type Dashboard = {
  models_loaded: Record<string, boolean>;
  mlops: Record<string, unknown>;
  metrics: Record<string, unknown>;
};

export function DashboardPage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiGet<Dashboard>("/api/v1/monitoring/dashboard")
      .then(setData)
      .catch((e) => setErr(String(e.message)));
  }, []);

  if (err) {
    return (
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-rose-800">
        <p className="font-medium">Could not load dashboard</p>
        <p className="text-sm mt-2 font-mono">{err}</p>
        <p className="text-sm mt-4 text-rose-700">
          Start the API: <code className="bg-black/5 px-1 rounded">uvicorn app.main:app --reload --port 8000</code>
        </p>
      </div>
    );
  }

  if (!data) {
    return <p className="text-slate-500">Loading dashboard…</p>;
  }

  const m = data.metrics as Record<string, number | string | Record<string, number | string | undefined>>;

  return (
    <div className="space-y-8 max-w-5xl">
      <div>
        <h2 className="font-display text-2xl text-ink-950">Operational overview</h2>
        <p className="text-slate-600 mt-1 max-w-2xl">
          Synthetic monitoring metrics and model load status. Values illustrate a typical deployment; live
          predictions use the endpoints in the sidebar.
        </p>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Customers monitored"
          value={String(m.total_customers_monitored ?? "—")}
        />
        <StatCard label="High-risk (churn)" value={String(m.high_risk_churn_count ?? "—")} />
        <StatCard label="30d churn rate" value={formatPct(m.churn_rate_30d)} />
        <StatCard label="Avg health score" value={String(m.avg_health_score ?? "—")} />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="font-semibold text-ink-900 mb-4">Model availability</h3>
          <ul className="space-y-2">
            {Object.entries(data.models_loaded).map(([k, v]) => (
              <li key={k} className="flex justify-between text-sm">
                <span className="text-slate-600 capitalize">{k}</span>
                <span className={v ? "text-emerald-600 font-medium" : "text-slate-400"}>
                  {v ? "ready" : "unavailable"}
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="font-semibold text-ink-900 mb-4">MLOps</h3>
          <pre className="text-xs bg-slate-50 rounded-lg p-4 overflow-auto max-h-48">
            {JSON.stringify(data.mlops, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="text-2xl font-semibold text-ink-950 mt-1">{value}</p>
    </div>
  );
}

function formatPct(v: unknown) {
  if (typeof v === "number") return `${(v * 100).toFixed(1)}%`;
  return "—";
}
