set -euo pipefail

DATASET_ROOT="${1:-./data/raw/dataverse}"

echo "[INFO] Preparing Dataverse dataset from: ${DATASET_ROOT}"
python -m src.main prepare_data --dataset dataverse --seed 42 --debug --config configs/global.yaml --dataset "${DATASET_ROOT}" 2>/dev/null || \
python -m src.cli.prepare_data --dataset dataverse --dataset-root "${DATASET_ROOT}" --seed 42

echo "[INFO] Dataverse preparation complete."