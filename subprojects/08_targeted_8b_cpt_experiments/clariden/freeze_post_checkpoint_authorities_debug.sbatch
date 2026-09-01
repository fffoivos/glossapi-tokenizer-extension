#!/usr/bin/env bash
#SBATCH --job-name=h2g-postcp-auth
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_AUTHORITY_KIND \
  H2G_8B_SOURCE_RECEIPT H2G_1P5B_SOURCE_RECEIPT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "post-checkpoint authority freeze requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
script="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_post_checkpoint_authorities.py"
receipts="$H2G_STAGE_ROOT/receipts"
case "$H2G_AUTHORITY_KIND" in
  phase3_entry)
    /usr/bin/python3.11 "$script" --kind phase3_entry \
      --8b-receipt "$H2G_8B_SOURCE_RECEIPT" \
      --1p5b-receipt "$H2G_1P5B_SOURCE_RECEIPT" \
      --output "$receipts/phase_local_cursor_guard.json" \
      --secondary-output "$receipts/constant_floor_scheduler_resume.json"
    ;;
  checkpoint_3456)
    /usr/bin/python3.11 "$script" --kind checkpoint_3456 \
      --8b-receipt "$H2G_8B_SOURCE_RECEIPT" \
      --1p5b-receipt "$H2G_1P5B_SOURCE_RECEIPT" \
      --output "$receipts/checkpoint_pair_3456.json"
    ;;
  resume_3456)
    /usr/bin/python3.11 "$script" --kind resume_3456 \
      --8b-receipt "$H2G_8B_SOURCE_RECEIPT" \
      --1p5b-receipt "$H2G_1P5B_SOURCE_RECEIPT" \
      --output "$receipts/resume_pair_3456.json"
    ;;
  *) echo "invalid H2G_AUTHORITY_KIND=$H2G_AUTHORITY_KIND" >&2; exit 2 ;;
esac
