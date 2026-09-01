#!/usr/bin/env bash
#SBATCH --job-name=h2g-uenv10-srun
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_UENV10_SMOKE_RECEIPT:?set new smoke receipt path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
[[ ! -e "$H2G_UENV10_SMOKE_RECEIPT" ]] || { echo "immutable smoke receipt exists" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

probe=$(srun --uenv=pytorch/v2.9.1:v2 --view=default --ntasks=1 \
  bash -c 'set -euo pipefail; command -v python3; command -v torchrun; python3 -c "import torch; print(torch.__version__)"')
python_path=$(sed -n '1p' <<<"$probe")
torchrun_path=$(sed -n '2p' <<<"$probe")
torch_version=$(sed -n '3p' <<<"$probe")
[[ "$python_path" == /user-environment/*/python3 ]]
[[ "$torchrun_path" == /user-environment/*/torchrun ]]
[[ "$torch_version" == 2.9.1* ]]
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 - "$H2G_UENV10_SMOKE_RECEIPT" "$python_path" "$torchrun_path" "$torch_version" <<'PY'
import datetime as dt, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["H2G_CODE_ROOT"]) / "subprojects/08_targeted_8b_cpt_experiments/scripts"))
from contract_utils import executing_code_bundle, write_json_atomic
out=Path(sys.argv[1])
write_json_atomic(out, {
    "schema_version":"apertus_hard_h_to_g_uenv10_srun_smoke_v1",
    "status":"passed",
    "created_at":dt.datetime.now(dt.timezone.utc).isoformat(),
    "slurm":{"job_id":os.environ["SLURM_JOB_ID"],"partition":os.environ["SLURM_JOB_PARTITION"],"nodes":int(os.environ["SLURM_NNODES"])},
    "uenv_image":"pytorch/v2.9.1:v2",
    "view":"default",
    "python_path":sys.argv[2],
    "torchrun_path":sys.argv[3],
    "torch_version":sys.argv[4],
    "srun_mount_mode":"slurm_plugin_explicit_uenv_and_view",
    "executing_code_bundle":executing_code_bundle(),
})
print(out)
PY
