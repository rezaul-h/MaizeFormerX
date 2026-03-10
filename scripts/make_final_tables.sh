set -euo pipefail

echo "[INFO] Building final CSV tables from experiment outputs"

python - <<'PY'
from pathlib import Path

from src.analysis.comparison_tables import export_in_domain_comparison_table
from src.analysis.ablation_tables import export_ablation_table
from src.analysis.cross_domain_tables import export_cross_domain_table
from src.analysis.efficiency_tables import export_efficiency_table

tables_dir = Path("outputs/tables")
tables_dir.mkdir(parents=True, exist_ok=True)

in_domain_path = Path("outputs/metrics/in_domain_results.json")
ablation_path = Path("outputs/metrics/ablation_results.json")
cross_domain_path = Path("outputs/metrics/cross_domain_results.json")
efficiency_path = Path("outputs/profiles/efficiency_results.json")

if in_domain_path.exists():
    export_in_domain_comparison_table(in_domain_path, tables_dir / "in_domain_table.csv")

if ablation_path.exists():
    export_ablation_table(ablation_path, tables_dir / "ablation_table.csv")

if cross_domain_path.exists():
    export_cross_domain_table(cross_domain_path, tables_dir / "cross_domain_table.csv")

if efficiency_path.exists():
    export_efficiency_table(efficiency_path, tables_dir / "efficiency_table.csv")

print("[INFO] Final table export complete.")
PY