#!/usr/bin/env bash
# Create or repair the backend virtualenv (fixes Mac pytest TimeoutError in iCloud Documents).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python3.11 &>/dev/null; then
  PY="python3.11"
elif command -v python3 &>/dev/null; then
  PY="python3"
else
  echo "error: python3 not found" >&2
  exit 1
fi

echo "Using interpreter: $($PY --version 2>&1)"

# Recreate venv if site-packages is unreadable (common with iCloud-synced Documents).
if [[ -d .venv ]]; then
  if ! .venv/bin/python -c "import site" 2>/dev/null; then
    echo "Removing broken .venv (site module failed)..."
    rm -rf .venv
  fi
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi

PYBIN="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"

"$PYBIN" -m pip install --upgrade pip wheel
"$PIP" install -r requirements.txt

echo ""
echo "Verifying install..."
"$PYBIN" -c "import fastapi, numpy, scipy, cv2, pytest; print('OK:', fastapi.__version__)"

echo ""
echo "Running tests..."
export PYTHONPATH="$ROOT"
"$PYBIN" -m pytest tests/ -q --tb=no

echo ""
echo "Setup complete. Start the API with:"
echo "  ./scripts/run_dev.sh"
