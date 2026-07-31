#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_runtime
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
if [[ -s "$RUNTIME_RECEIPT" ]]; then
  cpt_require_runtime
  echo "runtime already complete: $RUNTIME_RECEIPT"
  exit 0
fi
[[ ! -e "$RUNTIME_VENV" ]] || { echo "ERROR: incomplete runtime exists: $RUNTIME_VENV" >&2; exit 2; }
uenv run "$CPT_UENV" --view=default -- python3 -m venv --system-site-packages "$RUNTIME_VENV"
uenv run "$CPT_UENV" --view=default -- "$RUNTIME_VENV/bin/python" -m pip install \
  pyarrow==21.0.0 datasets==4.0.0 pytest==8.4.1
uenv run "$CPT_UENV" --view=default -- "$RUNTIME_VENV/bin/python" - "$RUNTIME_RECEIPT" <<'PY'
import importlib.metadata as md,json,platform,sys
from pathlib import Path
out=Path(sys.argv[1])
versions={name:md.version(name) for name in ("datasets","pyarrow","pytest")}
assert versions == {"datasets":"4.0.0","pyarrow":"21.0.0","pytest":"8.4.1"}
value={"schema_version":"greek_cpt25b_runtime_v1","status":"completed","python":platform.python_version(),"machine":platform.machine(),"packages":versions}
out.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
cpt_require_runtime
