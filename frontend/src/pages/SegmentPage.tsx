import { useState } from "react";
import { apiPost } from "../lib/api";
import { sampleCustomer } from "../lib/sampleCustomer";
import { CustomerJsonPanel } from "../components/CustomerJsonPanel";

export function SegmentPage() {
  const [json, setJson] = useState(JSON.stringify(sampleCustomer, null, 2));
  const [out, setOut] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setErr(null);
    setOut(null);
    let body: unknown;
    try {
      body = JSON.parse(json);
    } catch {
      setErr("Fix JSON syntax before running.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiPost<unknown>("/api/v1/segment/predict", body);
      setOut(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="font-display text-2xl text-ink-950">Customer segmentation</h2>
        <p className="text-slate-600 mt-1">RFM-style segments with upsell recommendations.</p>
      </div>
      <CustomerJsonPanel value={json} onChange={setJson} />
      <button
        type="button"
        disabled={loading}
        onClick={run}
        className="rounded-lg bg-accent px-5 py-2.5 text-white font-medium shadow hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Running…" : "Predict segment"}
      </button>
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
