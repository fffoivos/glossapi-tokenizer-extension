#!/usr/bin/env bash
#SBATCH --job-name=t8b-build-b
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=01:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_PARENT_STAGE_ROOT:?set sanitized parent stage root}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
PARENT_RUN=${TARGET8_PARENT_RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12}
PARENT_CHECKPOINT_RECEIPT=${TARGET8_PARENT_CHECKPOINT_RECEIPT:-$PARENT_RUN/checkpoint_evaluations/iter_0009536/attempt_0/export/source_checkpoint_receipt.json}
PARENT_CHECKPOINT_DIR=${TARGET8_PARENT_CHECKPOINT_DIR:-$PARENT_RUN/checkpoints/iter_0009536}

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "schedule construction must run on debug" >&2
  exit 2
}
/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
output="$TARGET8_STAGE_ROOT/experiment_b/schedules"
[[ ! -e "$output" ]] || { echo "immutable B schedule output already exists" >&2; exit 2; }
mkdir -p "$TARGET8_STAGE_ROOT/experiment_b"

uenv run pytorch/v2.9.1:v2 --view=default -- \
  python3 "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_continuation_b_schedule.py" \
    --parent-schedule-manifest "$TARGET8_PARENT_STAGE_ROOT/schedules/schedule_manifest.json" \
    --parent-checkpoint-receipt "$PARENT_CHECKPOINT_RECEIPT" \
    --parent-checkpoint-dir "$PARENT_CHECKPOINT_DIR" \
    --parent-arm D0_mixed \
    --checkpoint-iteration 9536 \
    --global-batch-sequences 1024 \
    --expected-modern-active-tokens 9123187023 \
    --output-dir "$output"
