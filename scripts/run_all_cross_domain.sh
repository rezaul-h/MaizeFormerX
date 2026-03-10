set -euo pipefail

DEVICE="${1:-auto}"

echo "[INFO] Running full cross-domain benchmark on device=${DEVICE}"
python - <<'PY'
from src.experiments.run_cross_domain import run_cross_domain_experiment
run_cross_domain_experiment(device_name="auto")
PY

echo "[INFO] Cross-domain benchmark complete."