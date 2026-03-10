set -euo pipefail

DATASET_ROOT="${1:-./data/raw/plagues_maiz}"

echo "[INFO] Preparing Plagues Maiz dataset from: ${DATASET_ROOT}"
python -m src.cli.prepare_data --dataset plagues_maiz --dataset-root "${DATASET_ROOT}" --seed 42

echo "[INFO] Plagues Maiz preparation complete."