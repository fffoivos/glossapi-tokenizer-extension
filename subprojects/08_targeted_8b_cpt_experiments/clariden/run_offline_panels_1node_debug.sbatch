#!/usr/bin/env bash
# Score all thirteen document-local panels on one four-GPU debug node.
# Clariden debug-qos permits at most 90 node-minutes per job, so the four
# groups run sequentially instead of requesting an unschedulable 4x80-minute
# allocation. This changes only evaluation placement, never model/data state.
#SBATCH --job-name=h2g-docval
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable scientific code root}"
: "${H2G_CODE_RECEIPT:?set immutable scientific code receipt}"
: "${H2G_VALIDATION_MANIFEST:?set frozen validation manifest}"
: "${H2G_HF_MODEL:?set verified HF checkpoint}"
: "${H2G_HF_TOKENIZER:?set frozen tokenizer}"
: "${H2G_DOCVAL_OUTPUT:?set unique panel output root}"
: "${H2G_DOCVAL_OUTPUT_FINAL:?set final unique panel output root}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "offline panels must run on debug" >&2; exit 2; }
observed_nodes="${SLURM_STEP_NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}}"
[[ "$observed_nodes" == 1 ]] || {
  echo "offline panel profile requires exactly one debug node" >&2; exit 2;
}
[[ ! -e "$H2G_DOCVAL_OUTPUT" ]] || { echo "immutable panel output exists" >&2; exit 2; }
[[ "$H2G_DOCVAL_OUTPUT" == "$H2G_DOCVAL_OUTPUT_FINAL" ]] || {
  echo "panel output/final identity drift" >&2; exit 2;
}
staging="$H2G_DOCVAL_OUTPUT.partial.${SLURM_JOB_ID:?missing Slurm job id}"
[[ ! -e "$staging" ]] || { echo "panel staging root exists" >&2; exit 2; }
mkdir -p "$staging"
cleanup() {
  status=$?
  if [[ "$status" -ne 0 ]]; then rm -rf -- "$staging"; fi
  exit "$status"
}
trap cleanup EXIT

export H2G_DOCVAL_OUTPUT="$staging"
failure=0
for group in 0 1 2 3; do
  H2G_PANEL_GROUP=$group srun --exact --exclusive --nodes=1 --ntasks=1 \
    --gpus-per-task=4 --cpus-per-task=288 \
    bash "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/run_per_document_panel_group.sh" \
    || failure=1
done
[[ "$failure" == 0 ]] || { echo "one or more panel groups failed" >&2; exit 1; }
mv "$staging" "$H2G_DOCVAL_OUTPUT_FINAL"
trap - EXIT
