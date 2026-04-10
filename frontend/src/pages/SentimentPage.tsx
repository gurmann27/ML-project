import { useState } from "react";
import { apiPost } from "../lib/api";

export function SentimentPage() {
  const [customerId, setCustomerId] = useState("CUST-001");
  const [text, setText] = useState(
    "Really love the service, very fast and reliable!",
  );
  const [source, setSource] = useState("feedback");
  const [out, setOut] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setErr(null);
    setOut(null);
    setLoading(true);
    try {
      const res = await apiPost<unknown>("/api/v1/sentiment/analyze", {
        customer_id: customerId,
        text,
        source,
      });
      setOut(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h2 className="font-display text-2xl text-ink-950">Sentiment analysis</h2>
        <p className="text-slate-600 mt-1">
          Transformer-based when available; otherwise keyword-based fallback.
        </p>
      </div>
      <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Customer ID</span>
          <input
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
          />
        </label>
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Source</span>
          <input
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
          />
        </label>
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Text</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-mono"
          />
        </label>
        <button
          type="button"
          disabled={loading || !text.trim()}
          onClick={run}
          className="rounded-lg bg-accent px-5 py-2.5 text-white font-medium shadow hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      {err && (
        <pre className="text-sm text-rose-700 bg-rose-50 p-4 rounded-lg overflow-auto">{err}</pre>
      )}
      {out != null ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="font-semibold mb-3">Response</h3>
          <pre className="text-xs overflow-auto max-h-96">{JSON.stringify(out, null, 2)}</pre>
        </div>
      ) : null}
    </div>
  );
}
