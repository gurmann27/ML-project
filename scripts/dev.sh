#!/usr/bin/env bash
# Start FastAPI on :8000 in the background, then Vite on :5173 (foreground).
# Usage: from project root, run: bash scripts/dev.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required."
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required for the frontend."
  exit 1
fi

echo "Starting API at http://127.0.0.1:8000 (background)..."
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
UV_PID=$!

cleanup() {
  echo ""
  echo "Stopping API (pid $UV_PID)..."
  kill "$UV_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1
echo "Starting UI at http://127.0.0.1:5173 ..."
cd "$ROOT/frontend"
exec npm run dev
