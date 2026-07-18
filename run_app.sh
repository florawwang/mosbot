#!/usr/bin/env bash
# Launch mosbot (frame viewer + activity graphs).
set -euo pipefail

PRODUCT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PRODUCT_DIR}:${PYTHONPATH:-}"

# Optional local secrets (gitignored)
if [[ -f "${PRODUCT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PRODUCT_DIR}/.env"
  set +a
fi

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

if [[ -z "${CLOUD_VIEWER_PASSCODE:-}" && ! -f "${PRODUCT_DIR}/.streamlit/secrets.toml" ]]; then
  echo "Warning: no CLOUD_VIEWER_PASSCODE and no .streamlit/secrets.toml found."
  echo "Copy .streamlit/secrets.toml.example → .streamlit/secrets.toml and set a passcode."
fi

exec "$PYTHON" -m streamlit run mosquito_lab/lab_app.py \
  --server.port "$PORT" \
  --server.address "$HOST" \
  --server.headless true \
  --browser.gatherUsageStats false
