#!/usr/bin/env bash
#SBATCH --job-name=h2g-prod-plan
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_ARTIFACT_MANIFEST H2G_SCALE \
  H2G_PRODUCER_COMPATIBILITY \
  H2G_TRAINING_RUN_PERMIT H2G_PROFILE_PROMOTION H2G_SUBMISSION_DRY_RUN_OUTPUT \
  H2G_PRODUCTION_TIMING_OUTPUT H2G_ALLOCATION_SCHEDULE_OUTPUT H2G_SUBMISSION_DRY_RUN_LOG; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || { echo "H2G_SCALE must be 8b or 1p5b" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "production timing/allocation freeze requires one debug node" >&2
  exit 2
}
for path in "$H2G_TRAINING_RUN_PERMIT" "$H2G_PROFILE_PROMOTION" \
  "$H2G_SUBMISSION_DRY_RUN_OUTPUT" "$H2G_PRODUCTION_TIMING_OUTPUT" \
  "$H2G_ALLOCATION_SCHEDULE_OUTPUT"; do
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "production-planning receipts must stay inside the stage receipts root: $path" >&2; exit 2 ;; esac
done
case "$H2G_SUBMISSION_DRY_RUN_LOG" in "$H2G_STAGE_ROOT"/logs/*) ;; *) echo "submission dry-run log must stay inside the stage logs root" >&2; exit 2 ;; esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
receipts="$H2G_STAGE_ROOT/receipts"
train_script="$subproject/clariden/train_hard_h_to_g_segment.sbatch"
run_permit="$H2G_TRAINING_RUN_PERMIT"
promotion="$H2G_PROFILE_PROMOTION"
dry_run="$H2G_SUBMISSION_DRY_RUN_OUTPUT"
timing="$H2G_PRODUCTION_TIMING_OUTPUT"
schedule="$H2G_ALLOCATION_SCHEDULE_OUTPUT"
for path in "$dry_run" "$timing" "$schedule"; do
  [[ ! -e "$path" ]] || { echo "immutable production-planning output exists: $path" >&2; exit 2; }
done
manifest_sha_before=$(sha256sum "$H2G_ARTIFACT_MANIFEST" | awk '{print $1}')
mkdir -p "$(dirname "$H2G_SUBMISSION_DRY_RUN_LOG")" "$receipts"
nodes=$(/usr/bin/python3.11 - "$run_permit" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["profile"]["nodes"])
PY
)
log="$H2G_SUBMISSION_DRY_RUN_LOG"
[[ ! -e "$log" ]] || { echo "immutable submission dry-run log exists: $log" >&2; exit 2; }
command=(sbatch --test-only --partition=normal --nodes="$nodes" --dependency="after:${SLURM_JOB_ID}+1" --export=NONE "$train_script")
printf 'COMMAND:' > "$log"
printf ' %q' "${command[@]}" >> "$log"
printf '\n' >> "$log"
set +e
"${command[@]}" >> "$log" 2>&1
rc=$?
set -e
printf 'TEST_ONLY_EXIT_CODE=%s\n' "$rc" >> "$log"
[[ "$rc" == 0 ]] || { cat "$log" >&2; exit "$rc"; }
/usr/bin/python3.11 "$subproject/scripts/freeze_submission_dry_run.py" \
  --scale "$H2G_SCALE" \
  --artifact-manifest "$H2G_ARTIFACT_MANIFEST" --manifest-sha-before "$manifest_sha_before" \
  --run-permit "$run_permit" --log "$log" \
  --train-script "$train_script" --output "$dry_run"
/usr/bin/python3.11 "$subproject/scripts/freeze_production_timing_and_allocation.py" \
  --scale "$H2G_SCALE" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
  --profile-promotion "$promotion" --run-permit "$run_permit" \
  --submission-dry-run-receipt "$dry_run" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --timing-output "$timing" --allocation-output "$schedule"
