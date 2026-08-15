#!/usr/bin/env bash
#SBATCH --job-name=h2g-final-evidence
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_8B_RUN_ROOT \
  H2G_1P5B_RUN_ROOT H2G_STATISTICAL_CONTRACT EVAL_VENV; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "joint evidence finalization requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
output="$H2G_STAGE_ROOT/evaluation/final_matched_evidence"
[[ ! -e "$output" ]] || { echo "immutable final evidence exists: $output" >&2; exit 2; }
cleanup() {
  status=$?
  if [[ "$status" != 0 && -d "$output" ]]; then
    mv "$output" "$H2G_STAGE_ROOT/evaluation/_failed_final_matched_evidence.${SLURM_JOB_ID}"
  fi
  exit "$status"
}
trap cleanup EXIT

uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$EVAL_VENV/bin/python" \
  "$subproject/evaluation/finalize_matched_study_evidence.py" \
  --8b-run-root "$H2G_8B_RUN_ROOT" \
  --1p5b-run-root "$H2G_1P5B_RUN_ROOT" \
  --statistical-contract "$H2G_STATISTICAL_CONTRACT" \
  --output-dir "$output"
trap - EXIT
printf '%s\n' "$output/selection_authorization.json"
