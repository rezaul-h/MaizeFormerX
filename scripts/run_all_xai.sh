set -euo pipefail

DATASET="${1:-dataverse}"
MODEL="${2:-maizeformerx}"
CHECKPOINT="${3:-outputs/checkpoints/manual_train/best.pt}"
DEVICE="${4:-auto}"

echo "[INFO] Running explainability for dataset=${DATASET}, model=${MODEL}, checkpoint=${CHECKPOINT}"
python -m src.cli.explain \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --max-cases 12

echo "[INFO] Explainability generation complete."