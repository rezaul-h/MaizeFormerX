set -euo pipefail

DEVICE="${1:-auto}"

echo "[INFO] Running efficiency profiling on device=${DEVICE}"
python - <<'PY'
from src.experiments.run_efficiency import run_efficiency_experiment
run_efficiency_experiment(device_name="auto")
PY

echo "[INFO] Efficiency profiling complete."