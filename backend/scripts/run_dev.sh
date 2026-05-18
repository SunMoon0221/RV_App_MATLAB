#!/usr/bin/env bash
# Run API without reloading when .venv packages change.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
exec python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app
