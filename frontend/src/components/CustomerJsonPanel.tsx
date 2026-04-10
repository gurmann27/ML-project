import { useState } from "react";
import { highRiskCustomer, sampleCustomer } from "../lib/sampleCustomer";

type Props = {
  value: string;
  onChange: (v: string) => void;
};

export function CustomerJsonPanel({ value, onChange }: Props) {
  const [err, setErr] = useState<string | null>(null);

  function validate(json: string) {
    try {
      JSON.parse(json);
      setErr(null);
      return true;
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Invalid JSON");
      return false;
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => {
            onChange(JSON.stringify(sampleCustomer, null, 2));
            setErr(null);
          }}
          className="text-xs font-medium px-3 py-1.5 rounded-md bg-slate-200 hover:bg-slate-300 text-slate-800"
        >
          Load sample profile
        </button>
        <button
          type="button"
          onClick={() => {
            onChange(JSON.stringify(highRiskCustomer, null, 2));
            setErr(null);
          }}
          className="text-xs font-medium px-3 py-1.5 rounded-md bg-amber-100 hover:bg-amber-200 text-amber-900"
        >
          Load high-risk profile
        </button>
      </div>
      <textarea
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          validate(e.target.value);
        }}
        spellCheck={false}
        className="w-full h-72 font-mono text-sm p-4 rounded-xl border border-slate-200 bg-white shadow-sm focus:ring-2 focus:ring-accent/30 focus:border-accent outline-none"
      />
      {err && <p className="text-sm text-rose-600">{err}</p>}
    </div>
  );
}
