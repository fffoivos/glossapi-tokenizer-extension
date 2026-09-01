#!/usr/bin/env bash
#SBATCH --job-name=lr13_runtime
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
runtime="${LR13_BRISTEN_RUNTIME:-/iopsstor/scratch/cscs/fffoivos/python_envs/lr13_bristen_py311}"
receipt="$runtime/runtime_receipt.json"
if [[ -s "$receipt" ]]; then
  "$runtime/bin/python" -c 'import numpy,pyarrow,tokenizers; assert pyarrow.__version__ == "21.0.0"'
  exit 0
fi
[[ ! -e "$runtime" ]] || { echo "ERROR: incomplete runtime exists: $runtime" >&2; exit 2; }
python3.11 -m venv "$runtime"
"$runtime/bin/python" -m pip install --upgrade pip
"$runtime/bin/python" -m pip install numpy==2.2.6 pyarrow==21.0.0 tokenizers==0.21.4
"$runtime/bin/python" - "$receipt" <<'PY'
import importlib.metadata as md,json,platform,sys
from pathlib import Path
versions={n:md.version(n) for n in ("numpy","pyarrow","tokenizers")}
expected={"numpy":"2.2.6","pyarrow":"21.0.0","tokenizers":"0.21.4"}
assert versions == expected,(versions,expected)
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"lr13_bristen_runtime_v1","status":"completed","python":platform.python_version(),"machine":platform.machine(),"packages":versions},indent=2,sort_keys=True)+"\n")
PY

