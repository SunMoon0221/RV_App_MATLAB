#!/usr/bin/env bash
# Run API without reloading when .venv packages change.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
  echo "warning: .venv not found — using system python3. Run ./scripts/setup_venv.sh first." >&2
fi

export PYTHONPATH="$ROOT"
exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app
