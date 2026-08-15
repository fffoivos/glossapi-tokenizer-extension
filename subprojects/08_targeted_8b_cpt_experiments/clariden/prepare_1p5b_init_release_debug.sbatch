#!/usr/bin/env bash
#SBATCH --job-name=h2g-1p5b-init-ready
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

# One canonical data-stage task owns the corrected-geometry view and exact
# HF -> Megatron -> HF round trip, so they consume one debug allocation.
set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${APERTUS_DATA_STAGE_RUN_ROOT:?run only through the canonical data-stage worker}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "1.5B initialization release preparation requires one debug node" >&2
  exit 2
}

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
export H2G_INIT_SCALE=1p5b
bash "$subproject/clariden/prepare_init_geometry_debug.sbatch"
bash "$subproject/clariden/roundtrip_td_init_debug.sbatch"

completion="$APERTUS_DATA_STAGE_RUN_ROOT/receipts/1p5b-init-release.json"
roundtrip="$H2G_STAGE_ROOT/receipts/1p5b_init_roundtrip.json"
mkdir -p "$(dirname "$completion")"
/usr/bin/python3.11 - "$roundtrip" "$completion" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
value = json.loads(source.read_text(encoding="utf-8"))
if not (
    value.get("schema_version") == "apertus_targeted_init_roundtrip_v1"
    and value.get("status") == "passed"
    and value.get("scale") == "1p5b"
    and value.get("tensor_and_logit_roundtrip_exact") is True
):
    raise SystemExit("1.5B round-trip evidence did not pass")
payload = {
    "schema_version": "apertus_1p5b_init_release_ready_v1",
    "status": "passed",
    "roundtrip_receipt": {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    },
}
temporary = Path(str(output) + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
