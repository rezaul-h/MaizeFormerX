set -euo pipefail

DEVICE="${1:-auto}"

echo "[INFO] Running full ablation study on device=${DEVICE}"
python - <<'PY'
from src.experiments.run_ablation import run_ablation_experiment
run_ablation_experiment(device_name="auto")
PY

echo "[INFO] Ablation study complete."