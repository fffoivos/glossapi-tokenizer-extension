#!/usr/bin/env bash
#SBATCH --job-name=poly-v2-frequencies
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=06:00:00

set -euo pipefail

if [[ "${BIB_UENV_ACTIVE:-0}" != "1" ]]; then
  BIB_UENV_IMAGE="${BIB_UENV_IMAGE:-pytorch/v2.9.1:v2}"
  export BIB_UENV_ACTIVE=1 BIB_UENV_IMAGE
  exec /usr/bin/uenv run "$BIB_UENV_IMAGE" --view=default -- \
    bash "$(readlink -f "$0")"
fi

: "${BIB_FREQUENCY_SCRIPT:?set BIB_FREQUENCY_SCRIPT}"
: "${BIB_TOKEN_PLAN:?set BIB_TOKEN_PLAN}"
: "${BIB_FREQUENCY_RECEIPT_DIR:?set BIB_FREQUENCY_RECEIPT_DIR}"
: "${BIB_FREQUENCY_COUNTS_DIR:?set BIB_FREQUENCY_COUNTS_DIR}"
: "${BIB_PYTHON:?set BIB_PYTHON}"
: "${SLURM_ARRAY_TASK_ID:?requires a Slurm array}"
: "${SLURM_ARRAY_TASK_COUNT:?requires a Slurm array}"

TOKEN_COUNT_SLOTS="${TOKEN_COUNT_SLOTS:-72}"
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export BIB_FREQUENCY_SCRIPT BIB_TOKEN_PLAN BIB_FREQUENCY_RECEIPT_DIR
export BIB_FREQUENCY_COUNTS_DIR BIB_PYTHON

mkdir -p "$BIB_FREQUENCY_RECEIPT_DIR" "$BIB_FREQUENCY_COUNTS_DIR"

"$BIB_PYTHON" -c '
import platform
import numpy
import pyarrow
import tokenizers

assert platform.machine() == "aarch64", platform.machine()
assert pyarrow.__version__ == "21.0.0", pyarrow.__version__
assert tokenizers.__version__ == "0.22.1", tokenizers.__version__
print("runtime", platform.machine(), numpy.__version__, pyarrow.__version__, tokenizers.__version__)
'

mapfile -t task_indexes < <(
  "$BIB_PYTHON" - "$BIB_TOKEN_PLAN" \
    "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" <<'PY'
import json
import sys

plan_path, node_index, node_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(plan_path, encoding="utf-8") as handle:
    plan = json.load(handle)
for task_index in range(len(plan["tasks"])):
    if task_index % node_count == node_index:
        print(task_index)
PY
)

echo "node ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}: ${#task_indexes[@]} tasks, slots=$TOKEN_COUNT_SLOTS"

printf '%s\n' "${task_indexes[@]}" |
  xargs --no-run-if-empty -P "$TOKEN_COUNT_SLOTS" -n 1 bash -c '
    set -euo pipefail
    task_index="$1"
    "$BIB_PYTHON" "$BIB_FREQUENCY_SCRIPT" task \
      --plan "$BIB_TOKEN_PLAN" \
      --task-index "$task_index" \
      --receipt-dir "$BIB_FREQUENCY_RECEIPT_DIR" \
      --counts-dir "$BIB_FREQUENCY_COUNTS_DIR" \
      --batch-size 256
  ' _

echo "node ${SLURM_ARRAY_TASK_ID} complete"
