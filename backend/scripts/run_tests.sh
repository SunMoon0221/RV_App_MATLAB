#!/usr/bin/env bash
# Run pytest with the project venv (avoids broken system/python3 -m pytest on synced folders).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Run: ./scripts/setup_venv.sh" >&2
  exit 1
fi

export PYTHONPATH="$ROOT"
exec .venv/bin/python -m pytest tests/ "$@"
