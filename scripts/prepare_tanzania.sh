set -euo pipefail

DATASET_ROOT="${1:-./data/raw/tanzania}"

echo "[INFO] Preparing Tanzania dataset from: ${DATASET_ROOT}"
python -m src.cli.prepare_data --dataset tanzania --dataset-root "${DATASET_ROOT}" --seed 42

echo "[INFO] Tanzania preparation complete."