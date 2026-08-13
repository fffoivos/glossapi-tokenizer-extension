#!/usr/bin/env bash
set -euo pipefail
: "${EARLY_CODE_ROOT:?set immutable code root}"
: "${EARLY_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${EARLY_RUN_ROOT:?set new run root}"
: "${EARLY_TRAIN_LEAF_SWITCH:?set pinned leaf switch}"
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
exclude=$("$EARLY_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$EARLY_TRAIN_LEAF_SWITCH" 16)
common="ALL,EARLY_CODE_ROOT=$EARLY_CODE_ROOT,EARLY_CODE_BUNDLE_RECEIPT=$receipt,EARLY_CONTRACT=$contract,EARLY_BRANCH_RECIPE=$recipe,EARLY_LAUNCH_GATE=$gate,EARLY_OPERATIONAL_GATE=$operational,EARLY_RUN_ROOT=$EARLY_RUN_ROOT,EARLY_TRAIN_LEAF_SWITCH=$EARLY_TRAIN_LEAF_SWITCH,EARLY_RECOVERY_MODE=0"
command=(sbatch --uenv-passthrough=ignore --parsable --partition=normal --nodes=16 --time=05:00:00 --switches=1 \
  --exclude="$exclude" --job-name=full8_early_replay --output="$EARLY_RUN_ROOT/logs/%x-%j.out" \
  --error="$EARLY_RUN_ROOT/logs/%x-%j.err" --export="$common,EARLY_PHASE=replay" "$SUBPROJECT/clariden/train_and_gate.sbatch")
printf -v quoted '%q ' "${command[@]}"
/usr/bin/python3.11 - "$test_receipt" "$quoted" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('command')!=sys.argv[2]: raise SystemExit('actual command differs from tested command')
PY
mkdir -p "$EARLY_RUN_ROOT/logs" "$EARLY_RUN_ROOT/orchestration"
replay=""; branch=""; replay_supervisor=""; branch_supervisor=""
cleanup() { set +e; [[ -n "$branch_supervisor" ]] && scancel "$branch_supervisor"; [[ -n "$replay_supervisor" ]] && scancel "$replay_supervisor"; [[ -n "$branch" ]] && scancel "$branch"; [[ -n "$replay" ]] && scancel "$replay"; }
trap cleanup ERR
replay=$("${command[@]}")
branch_command=(sbatch --uenv-passthrough=ignore --parsable --partition=normal --nodes=16 --time=12:00:00 --switches=1 \
  --dependency="after:$replay+200" --exclude="$exclude" --job-name=full8_early_wsd \
  --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,EARLY_PHASE=branch,EARLY_REPLAY_JOB_ID=$replay" "$SUBPROJECT/clariden/train_and_gate.sbatch")
branch_test_result=$(sbatch --test-only "${branch_command[@]:1}" 2>&1)
printf -v branch_quoted '%q ' "${branch_command[@]}"
branch_test="$EARLY_RUN_ROOT/orchestration/branch_holder_test_only.json"
/usr/bin/python3.11 "$SUBPROJECT/scripts/freeze_test_only.py" --role branch_holder --command "$branch_quoted" --result "$branch_test_result" --output "$branch_test"
branch=$("${branch_command[@]}")
replay_supervisor=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=1 --time=00:10:00 \
  --dependency="afterany:$replay" --job-name=early_replay_supervise \
  --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,EARLY_PHASE=replay,EARLY_TRAIN_JOB_ID=$replay,EARLY_BRANCH_JOB_ID=$branch" "$SUBPROJECT/clariden/supervise_after_training_debug.sbatch")
branch_supervisor=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=1 --time=00:10:00 \
  --dependency="afterany:$branch" --job-name=early_wsd_supervise \
  --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,EARLY_PHASE=branch,EARLY_TRAIN_JOB_ID=$branch,EARLY_REPLAY_JOB_ID=$replay" "$SUBPROJECT/clariden/supervise_after_training_debug.sbatch")
replay_audit="$EARLY_RUN_ROOT/orchestration/replay_${replay}.json"
branch_audit="$EARLY_RUN_ROOT/orchestration/branch_holder_${branch}.json"
replay_supervisor_audit="$EARLY_RUN_ROOT/orchestration/replay_supervisor_${replay_supervisor}.json"
branch_supervisor_audit="$EARLY_RUN_ROOT/orchestration/branch_supervisor_${branch_supervisor}.json"
/usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" --job-id "$replay" --role replay --output "$replay_audit"
/usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" --job-id "$branch" --role branch_holder --output "$branch_audit"
/usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" --job-id "$replay_supervisor" --role replay_supervisor --output "$replay_supervisor_audit"
/usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" --job-id "$branch_supervisor" --role branch_supervisor --output "$branch_supervisor_audit"
/usr/bin/python3.11 "$SUBPROJECT/scripts/freeze_launch_graph.py" --replay-job "$replay" --branch-holder-job "$branch" --replay-supervisor-job "$replay_supervisor" --branch-supervisor-job "$branch_supervisor" \
  --replay-audit "$replay_audit" --branch-holder-audit "$branch_audit" --replay-supervisor-audit "$replay_supervisor_audit" --branch-supervisor-audit "$branch_supervisor_audit" --branch-test-only "$branch_test" --operational-gate "$operational" \
  --output "$EARLY_RUN_ROOT/orchestration/launch_graph.json"
trap - ERR
printf '%s %s %s %s\n' "$replay" "$branch" "$replay_supervisor" "$branch_supervisor"
