#!/usr/bin/env bash
#SBATCH --job-name=h2g-nested-uenv-srun
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable experiment code root}"
: "${H2G_CODE_RECEIPT:?set immutable experiment code receipt}"
: "${H2G_NESTED_UENV_SMOKE_RECEIPT:?set new smoke receipt path}"
: "${H2G_UENV_IMAGE:=pytorch/v2.9.1:v2}"
: "${H2G_UENV_VIEW:=default}"
: "${H2G_UENV_DIGEST:?set the runtime-contract uenv digest}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
[[ ! -e "$H2G_NESTED_UENV_SMOKE_RECEIPT" ]] || {
  echo "immutable nested-uenv smoke receipt exists" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

resolved_digest=$(uenv image inspect --format='{sha256}' "$H2G_UENV_IMAGE")
[[ "$resolved_digest" == "$H2G_UENV_DIGEST" ]] || {
  echo "uenv digest drift: expected=$H2G_UENV_DIGEST observed=$resolved_digest" >&2
  exit 2
}

probe=$(uenv run "$H2G_UENV_DIGEST" --view="$H2G_UENV_VIEW" -- \
  srun --uenv="$H2G_UENV_IMAGE" --view="$H2G_UENV_VIEW" --ntasks=1 \
  bash -c 'set -euo pipefail; command -v python3; command -v torchrun; python3 -c "import torch; print(torch.__version__)"')
python_path=$(sed -n '1p' <<<"$probe")
torchrun_path=$(sed -n '2p' <<<"$probe")
torch_version=$(sed -n '3p' <<<"$probe")
[[ "$python_path" == /user-environment/*/python3 ]]
[[ "$torchrun_path" == /user-environment/*/torchrun ]]
[[ "$torch_version" == 2.9.1* ]]

export H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_UENV_IMAGE H2G_UENV_VIEW H2G_UENV_DIGEST
/usr/bin/python3.11 - "$H2G_NESTED_UENV_SMOKE_RECEIPT" "$python_path" "$torchrun_path" "$torch_version" <<'PY'
import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["H2G_CODE_ROOT"]) / "subprojects/08_targeted_8b_cpt_experiments/scripts"))
from contract_utils import executing_code_bundle, write_json_atomic

write_json_atomic(
    Path(sys.argv[1]),
    {
        "schema_version": "apertus_hard_h_to_g_nested_same_uenv_srun_smoke_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ["SLURM_JOB_PARTITION"],
            "nodes": int(os.environ["SLURM_NNODES"]),
        },
        "outer_mount_mode": "uenv_run_by_canonical_worker",
        "inner_mount_mode": "srun_explicit_same_uenv_and_view",
        "uenv": {
            "image": os.environ["H2G_UENV_IMAGE"],
            "view": os.environ["H2G_UENV_VIEW"],
            "digest": os.environ["H2G_UENV_DIGEST"],
        },
        "python_path": sys.argv[2],
        "torchrun_path": sys.argv[3],
        "torch_version": sys.argv[4],
        "executing_code_bundle": executing_code_bundle(),
    },
)
print(sys.argv[1])
PY
