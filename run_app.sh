#!/usr/bin/env bash
# Launch mosbot (frame viewer + activity graphs).
set -euo pipefail

PRODUCT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PRODUCT_DIR}:${PYTHONPATH:-}"
export CLOUD_VIEWER_PASSCODE="${CLOUD_VIEWER_PASSCODE:-florawang}"

PORT="${PORT:-8502}"
HOST="${HOST:-127.0.0.1}"

cd "$PRODUCT_DIR"
PYTHON="${PRODUCT_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(cd "$PRODUCT_DIR/.." && pwd)/.venv/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

exec "$PYTHON" -m streamlit run mosquito_lab/lab_app.py \
  --server.port "$PORT" \
  --server.address "$HOST" \
  --server.headless true \
  --browser.gatherUsageStats false
