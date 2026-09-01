#!/usr/bin/env bash
#SBATCH --job-name=t8b-b-pool-view
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
PARENT_POOL=${TARGET8_PARENT_POOL_RECEIPT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3/inventory/pool_corpus_receipt.json}
PARENT_PACKED_INTEGRITY=${TARGET8_PARENT_PACKED_INTEGRITY:-/iopsstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/_preflight/20260808T023300Z-sanitized-v45/packed_payload_integrity.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "B pool view must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
stage="$TARGET8_STAGE_ROOT/experiment_b"
mkdir -p "$stage/inventory"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_continuation_b_pool_view.py" \
  --parent-pool-receipt "$PARENT_POOL" --hybrid-schedule-manifest "$stage/schedules/schedule_manifest.json" \
  --packed-output "$stage/packed_corpus_receipt.json" \
  --parent-packed-integrity "$PARENT_PACKED_INTEGRITY" \
  --packed-integrity-output "$stage/packed_payload_integrity_receipt.json" \
  --output "$stage/inventory/pool_corpus_receipt.json"
