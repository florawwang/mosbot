#!/usr/bin/env bash
# Helper to launch cloud inference detached on a VM.
#
# Usage:
#   chmod +x run_cloud.sh
#   ./run_cloud.sh local /path/to/images /path/to/labels.csv /path/to/model.pt
#   ./run_cloud.sh drive "https://drive.google.com/drive/folders/..." \
#       "https://drive.google.com/file/d/..." "https://drive.google.com/file/d/..."
#
set -euo pipefail

PRODUCT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PRODUCT_DIR/.." && pwd)"
export PYTHONPATH="${PRODUCT_DIR}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

MODE="${1:-}"
shift || true

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/mosquito-lab-output}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/mosquito-lab-work}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/inference.log}"
VIEWER_PORT="${VIEWER_PORT:-8502}"
PASSCODE="${CLOUD_VIEWER_PASSCODE:-}"
if [[ -z "$PASSCODE" && -f "$PRODUCT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "$PRODUCT_DIR/.env"; set +a
  PASSCODE="${CLOUD_VIEWER_PASSCODE:-}"
fi
if [[ -z "$PASSCODE" ]]; then
  echo "Error: set CLOUD_VIEWER_PASSCODE (env or .env) before launching the viewer."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "local" ]]; then
  IMAGE_FOLDER="${1:?image folder required}"
  LABELS="${2:?labels csv required}"
  MODEL="${3:?model .pt required}"
  CMD=(python -m mosquito_lab.run_inference
    --image-folder "$IMAGE_FOLDER"
    --labels "$LABELS"
    --model "$MODEL"
    --output-dir "$OUTPUT_DIR"
    --work-dir "$WORK_DIR"
    --serve-viewer --viewer-port "$VIEWER_PORT" --passcode "$PASSCODE")
elif [[ "$MODE" == "drive" ]]; then
  IMAGES_URL="${1:?images drive folder url required}"
  LABELS_URL="${2:?labels drive url required}"
  MODEL_URL="${3:?model drive url required}"
  CMD=(python -m mosquito_lab.run_inference
    --images-drive-url "$IMAGES_URL"
    --labels-drive-url "$LABELS_URL"
    --model-drive-url "$MODEL_URL"
    --output-dir "$OUTPUT_DIR"
    --work-dir "$WORK_DIR"
    --serve-viewer --viewer-port "$VIEWER_PORT" --passcode "$PASSCODE")
else
  echo "Usage:"
  echo "  $0 local  IMAGE_FOLDER LABELS_CSV MODEL_PT"
  echo "  $0 drive  IMAGES_DRIVE_URL LABELS_DRIVE_URL MODEL_DRIVE_URL"
  exit 1
fi

echo "Logging to $LOG_FILE"
nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
echo "Started PID $! — tail -f $LOG_FILE"
echo "Viewer port: $VIEWER_PORT (passcode set via CLOUD_VIEWER_PASSCODE)"
