#!/usr/bin/env bash
set -euo pipefail
: "${EARLY_CODE_ROOT:?set immutable code root}"
: "${EARLY_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${EARLY_RUN_ROOT:?set new run root}"
[[ "${CONFIRM_GPU_LAUNCH:-}" == EARLY_COOLDOWN_C ]] || { echo "set CONFIRM_GPU_LAUNCH=EARLY_COOLDOWN_C" >&2; exit 2; }
receipt="$EARLY_CODE_ROOT.receipt.json"
contract="$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/configs/experiment_contract.json"
recipe="$EARLY_PRELAUNCH_ROOT/branch_recipe.json"
gate="$EARLY_PRELAUNCH_ROOT/launch_gate.json"
operational="$EARLY_PRELAUNCH_ROOT/operational_gate.json"
test_receipt="$EARLY_PRELAUNCH_ROOT/slurm_test_only.json"
SUBPROJECT="$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment"
for path in "$receipt" "$contract" "$recipe" "$gate" "$operational" "$test_receipt"; do [[ -f "$path" ]] || { echo "missing $path" >&2; exit 2; }; done
[[ ! -e "$EARLY_RUN_ROOT" ]] || { echo "run root already exists" >&2; exit 2; }
/usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$receipt" --kind scientific
/usr/bin/python3.11 - "$operational" "$gate" "$test_receipt" <<'PY'
import hashlib,json,sys
from pathlib import Path
d=json.load(open(sys.argv[1]))
if d.get('status')!='passed': raise SystemExit('operational gate failed')
for key,path in [('launch_gate',sys.argv[2]),('slurm_test_only',sys.argv[3])]:
 if d[key]['sha256']!=hashlib.sha256(Path(path).read_bytes()).hexdigest(): raise SystemExit(f'{key} drift')
PY
common="ALL,EARLY_CODE_ROOT=$EARLY_CODE_ROOT,EARLY_CODE_BUNDLE_RECEIPT=$receipt,EARLY_CONTRACT=$contract,EARLY_BRANCH_RECIPE=$recipe,EARLY_LAUNCH_GATE=$gate,EARLY_OPERATIONAL_GATE=$operational,EARLY_RUN_ROOT=$EARLY_RUN_ROOT,EARLY_RECOVERY_MODE=0,EARLY_PHASE=branch"
command=(sbatch --uenv-passthrough=ignore --parsable --partition=normal --nodes=16 --time=12:00:00 --switches=1 \
  --job-name=full8_early_wsd --output="$EARLY_RUN_ROOT/logs/%x-%j.out" \
  --error="$EARLY_RUN_ROOT/logs/%x-%j.err" --export="$common" "$SUBPROJECT/clariden/train_and_gate.sbatch")
printf -v quoted '%q ' "${command[@]}"
/usr/bin/python3.11 - "$test_receipt" "$quoted" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('command')!=sys.argv[2]: raise SystemExit('actual command differs from tested command')
PY
mkdir -p "$EARLY_RUN_ROOT/logs" "$EARLY_RUN_ROOT/orchestration"
training=""; supervisor=""
cleanup() { set +e; [[ -n "$supervisor" ]] && scancel "$supervisor"; [[ -n "$training" ]] && scancel "$training"; }
trap cleanup ERR
training=$("${command[@]}")
supervisor=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=1 --time=00:10:00 \
  --dependency="afterany:$training" --job-name=early_wsd_supervise \
  --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,EARLY_TRAIN_JOB_ID=$training" "$SUBPROJECT/clariden/supervise_after_training_debug.sbatch")
training_audit="$EARLY_RUN_ROOT/orchestration/training_${training}.json"
supervisor_audit="$EARLY_RUN_ROOT/orchestration/supervisor_${supervisor}.json"
/usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" --job-id "$training" --role training --output "$training_audit"
/usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" --job-id "$supervisor" --role branch_supervisor --output "$supervisor_audit"
/usr/bin/python3.11 "$SUBPROJECT/scripts/freeze_launch_graph.py" --training-job "$training" --supervisor-job "$supervisor" \
  --training-audit "$training_audit" --supervisor-audit "$supervisor_audit" --test-only "$test_receipt" \
  --operational-gate "$operational" --output "$EARLY_RUN_ROOT/orchestration/launch_graph.json"
trap - ERR
printf '%s %s\n' "$training" "$supervisor"
