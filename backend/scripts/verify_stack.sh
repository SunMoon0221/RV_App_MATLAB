#!/usr/bin/env bash
# Quick smoke test: health + mock E2E (requires venv from setup_venv.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./scripts/setup_venv.sh first" >&2
  exit 1
fi

export PYTHONPATH="$ROOT"
.venv/bin/python -m pytest tests/test_e2e_workflow.py tests/test_api.py -q --tb=short

echo ""
echo "If the API is running (./scripts/run_dev.sh), checking health..."
if curl -sf --max-time 2 http://127.0.0.1:8000/api/health >/dev/null; then
  curl -s http://127.0.0.1:8000/api/health
  echo ""
else
  echo "(API not running — start with ./scripts/run_dev.sh)"
fi
