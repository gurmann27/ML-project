/**
 * API client for Customer Monitoring System (FastAPI backend).
 * In dev, Vite proxies /api → http://127.0.0.1:8000 (see vite.config.ts).
 */

const base = import.meta.env.VITE_API_URL ?? "";

export async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(`${base}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status}: ${text || r.statusText}`);
  }
  return r.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${base}${path}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status}: ${text || r.statusText}`);
  }
  return r.json() as Promise<T>;
}
