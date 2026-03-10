set -euo pipefail

DEVICE="${1:-auto}"

echo "[INFO] Running full in-domain benchmark on device=${DEVICE}"
python - <<'PY'
from src.experiments.run_in_domain import run_in_domain_experiment
run_in_domain_experiment(device_name="auto")
PY

echo "[INFO] In-domain benchmark complete."