import { NavLink, Outlet } from "react-router-dom";

const nav = [
  { to: "/", label: "Overview" },
  { to: "/churn", label: "Churn prediction" },
  { to: "/segment", label: "Segmentation" },
  { to: "/anomaly", label: "Anomaly detection" },
  { to: "/sentiment", label: "Sentiment (NLP)" },
  { to: "/report", label: "Full report" },
];

export function Layout({
  apiOk,
  onRefreshHealth,
}: {
  apiOk: boolean | null;
  onRefreshHealth: () => void;
}) {
  return (
    <div className="flex min-h-screen">
      <aside className="w-64 shrink-0 bg-ink-900 text-slate-200 flex flex-col border-r border-slate-800">
        <div className="p-6 border-b border-slate-800">
          <p className="font-display text-xl text-white tracking-tight">Monitoring Lab</p>
          <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest">
            ML · NLP · Analytics
          </p>
        </div>
        <nav className="flex-1 p-3 space-y-0.5">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `block rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-slate-800 text-white"
                    : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 text-xs text-slate-600 border-t border-slate-800">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="text-accent-muted hover:text-white"
          >
            OpenAPI docs →
          </a>
        </div>
      </aside>
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-slate-200 bg-white/80 backdrop-blur flex items-center justify-between px-8">
          <h1 className="font-display text-lg text-ink-950">Customer Monitoring System</h1>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-slate-500">API</span>
            {apiOk === null && (
              <span className="rounded-full bg-slate-200 px-2 py-0.5 text-slate-600">checking…</span>
            )}
            {apiOk === true && (
              <span className="rounded-full bg-emerald-100 text-emerald-800 px-2 py-0.5 font-medium">
                connected
              </span>
            )}
            {apiOk === false && (
              <span className="rounded-full bg-rose-100 text-rose-800 px-2 py-0.5 font-medium">
                offline
              </span>
            )}
            <button
              type="button"
              onClick={onRefreshHealth}
              className="text-accent hover:underline ml-2 text-sm"
            >
              Refresh status
            </button>
          </div>
        </header>
        <main className="flex-1 p-8 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
