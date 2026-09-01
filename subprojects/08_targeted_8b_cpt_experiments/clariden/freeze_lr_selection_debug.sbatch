#!/usr/bin/env bash
#SBATCH --job-name=h2g-lr-select
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_SCALE:?set 8b or 1p5b}"
: "${H2G_LR_SELECTION_OUTPUT:?set a new immutable LR-selection output path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "LR selection requires one debug node" >&2; exit 2; }
case "$H2G_LR_SELECTION_OUTPUT" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "LR-selection output must be inside the stage receipts root" >&2; exit 2 ;; esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
out="$H2G_LR_SELECTION_OUTPUT"
[[ ! -e "$out" ]] || { echo "immutable LR selection exists: $out" >&2; exit 2; }
extra=()
case "$H2G_SCALE" in
  8b)
    extra+=(--historical-decision "$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/PRODUCTION_LR_DECISION_20260613.md")
    ;;
  1p5b)
    : "${H2G_LR_ARM_RECEIPTS:?set colon-separated three pilot receipts}"
    IFS=: read -r -a paths <<< "$H2G_LR_ARM_RECEIPTS"
    (( ${#paths[@]} == 3 )) || { echo "need exactly three LR-arm receipts" >&2; exit 2; }
    for path in "${paths[@]}"; do [[ -s "$path" ]] || { echo "LR-arm receipt missing: $path" >&2; exit 2; }; extra+=(--arm "$path"); done
    ;;
  *) echo "invalid scale" >&2; exit 2 ;;
esac
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/freeze_lr_selection.py" \
  --scale "$H2G_SCALE" --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  "${extra[@]}" --output "$out"
