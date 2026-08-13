#!/usr/bin/env bash
#SBATCH --job-name=early-wsd-prepare
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${EARLY_CODE_ROOT:?set immutable code root}"
: "${EARLY_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${EARLY_PRELAUNCH_ROOT:?set new prelaunch root}"
: "${EARLY_RUN_ROOT:?set future production run root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "preparation must run on debug" >&2; exit 2; }
[[ ! -e "$EARLY_PRELAUNCH_ROOT" && ! -e "$EARLY_RUN_ROOT" ]] || { echo "prelaunch or run root already exists" >&2; exit 2; }
mkdir -p "$EARLY_PRELAUNCH_ROOT"
SUBPROJECT="$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment"
contract="$SUBPROJECT/configs/experiment_contract.json"
scheduler="$EARLY_PRELAUNCH_ROOT/scheduler_snapshot.json"
recipe="$EARLY_PRELAUNCH_ROOT/branch_recipe.json"
gate="$EARLY_PRELAUNCH_ROOT/launch_gate.json"
test_receipt="$EARLY_PRELAUNCH_ROOT/slurm_test_only.json"
operational="$EARLY_PRELAUNCH_ROOT/operational_gate.json"

/usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$EARLY_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$SUBPROJECT/scripts/capture_scheduler_snapshot.py" --output "$scheduler"
/usr/bin/python3.11 "$SUBPROJECT/scripts/prepare_launch.py" --contract "$contract" \
  --code-bundle-receipt "$EARLY_CODE_BUNDLE_RECEIPT" --scheduler-snapshot "$scheduler" \
  --recipe-output "$recipe" --launch-gate-output "$gate"
common="ALL,EARLY_CODE_ROOT=$EARLY_CODE_ROOT,EARLY_CODE_BUNDLE_RECEIPT=$EARLY_CODE_BUNDLE_RECEIPT,EARLY_CONTRACT=$contract,EARLY_BRANCH_RECIPE=$recipe,EARLY_LAUNCH_GATE=$gate,EARLY_OPERATIONAL_GATE=$operational,EARLY_RUN_ROOT=$EARLY_RUN_ROOT,EARLY_RECOVERY_MODE=0,EARLY_PHASE=branch"
command=(sbatch --uenv-passthrough=ignore --parsable --partition=normal --nodes=16 --time=12:00:00 --switches=1 \
  --job-name=full8_early_wsd --output="$EARLY_RUN_ROOT/logs/%x-%j.out" \
  --error="$EARLY_RUN_ROOT/logs/%x-%j.err" --export="$common" "$SUBPROJECT/clariden/train_and_gate.sbatch")
test_result=$(sbatch --test-only "${command[@]:1}" 2>&1)
printf -v quoted '%q ' "${command[@]}"
/usr/bin/python3.11 "$SUBPROJECT/scripts/freeze_test_only.py" --role training --command "$quoted" --result "$test_result" --output "$test_receipt"
/usr/bin/python3.11 "$SUBPROJECT/scripts/finalize_operational_gate.py" --launch-gate "$gate" --test-only "$test_receipt" --output "$operational"
printf '%s\n' "$operational"
