#!/usr/bin/env bash
# Prepare or explicitly submit the frozen 25B full-corpus Token-Distillation
# probe. Default is a non-submitting dry run; this file is never invoked by the
# CPU dataset chain.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd "$BRIDGE_DIR/../../.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/iopsstor/scratch/cscs/fffoivos}"
BRIDGE_STAGE_ROOT="${BRIDGE_STAGE_ROOT:?set BRIDGE_STAGE_ROOT to a finalized bridge run}"
BRIDGE_MANIFEST="${BRIDGE_MANIFEST:-$BRIDGE_STAGE_ROOT/bridge_manifest.json}"
BRIDGE_DATA_ENV="${BRIDGE_DATA_ENV:-$BRIDGE_STAGE_ROOT/training_data.env}"
TRAINING_ASSETS_RECEIPT="${TRAINING_ASSETS_RECEIPT:-$BRIDGE_STAGE_ROOT/training_assets_receipt.json}"
CONFIG="$HERE/full_corpus_25b.env"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$TRAIN_DIR/bakeoff_train.sbatch}"
VERIFY_SCRIPT="$BRIDGE_DIR/scripts/verify_launch_assets.py"

DRY_RUN="${DRY_RUN:-1}"
CONFIRM_GPU_LAUNCH="${CONFIRM_GPU_LAUNCH:-}"
ACCOUNT="${ACCOUNT:-a0140}"
PARTITION="${PARTITION:-normal}"
NODES="${NODES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-288}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
SEGMENT_ITERS="${SEGMENT_ITERS:-952}"  # 8 * 119
START_ITERATION="${START_ITERATION:-0}"
RESUME_CHECKPOINT_RECEIPT="${RESUME_CHECKPOINT_RECEIPT:-}"
SAVE_INTERVAL="119"
TRAIN_TOKENS="25000000000"
GLOBAL_BATCH_TOKENS="4194304"
TOTAL_ITER="$(( TRAIN_TOKENS / GLOBAL_BATCH_TOKENS ))"  # 5960
RUN_TAG="${RUN_TAG:-full_corpus_td25b_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_probe}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"

for receipt_owned in INIT_CKPT INIT_EVIDENCE MEGATRON_DIR; do
    if [[ -n "${!receipt_owned:-}" ]]; then
        echo "ERROR: unset $receipt_owned; the training-assets receipt owns it" >&2
        exit 2
    fi
done

case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$DRY_RUN" == 0 && "$CONFIRM_GPU_LAUNCH" != "FULL_CORPUS_TD_25B" ]]; then
    echo "ERROR: live launch requires CONFIRM_GPU_LAUNCH=FULL_CORPUS_TD_25B" >&2
    exit 2
fi
if (( NODES < 16 )) && [[ "$DRY_RUN" == 0 ]]; then
    echo "ERROR: the comparable live probe requires at least 16 nodes" >&2
    exit 2
fi
(( SEGMENT_ITERS > 0 && SEGMENT_ITERS % SAVE_INTERVAL == 0 )) || {
    echo "ERROR: SEGMENT_ITERS must be a positive multiple of $SAVE_INTERVAL" >&2; exit 2;
}
(( START_ITERATION >= 0 && START_ITERATION < TOTAL_ITER )) || {
    echo "ERROR: START_ITERATION must be in [0,$TOTAL_ITER)" >&2; exit 2;
}
(( START_ITERATION == 0 || START_ITERATION % SAVE_INTERVAL == 0 )) || {
    echo "ERROR: a resume boundary must be a multiple of $SAVE_INTERVAL" >&2; exit 2;
}

test -s "$BRIDGE_MANIFEST" || { echo "ERROR: finalized bridge missing: $BRIDGE_MANIFEST" >&2; exit 3; }
test -s "$BRIDGE_DATA_ENV" || { echo "ERROR: generated data env missing: $BRIDGE_DATA_ENV" >&2; exit 3; }
test -s "$TRAINING_ASSETS_RECEIPT" || { echo "ERROR: training-assets receipt missing: $TRAINING_ASSETS_RECEIPT" >&2; exit 3; }
test -s "$CONFIG" || { echo "ERROR: frozen probe config missing: $CONFIG" >&2; exit 3; }
verify_args=(--bridge-manifest "$BRIDGE_MANIFEST" --training-data-env "$BRIDGE_DATA_ENV"
    --training-assets-receipt "$TRAINING_ASSETS_RECEIPT" --training-env "$CONFIG"
    --trainer "$TRAIN_SCRIPT" --start-iteration "$START_ITERATION")
if (( START_ITERATION > 0 )); then
    test -s "$OUTPUT_DIR/probe_plan.json" || { echo "ERROR: immutable probe plan missing: $OUTPUT_DIR/probe_plan.json" >&2; exit 3; }
    test -s "$RESUME_CHECKPOINT_RECEIPT" || { echo "ERROR: receipt-bound resume checkpoint required" >&2; exit 3; }
    verify_args+=(--probe-plan "$OUTPUT_DIR/probe_plan.json" --resume-checkpoint-receipt "$RESUME_CHECKPOINT_RECEIPT")
elif [[ -n "$RESUME_CHECKPOINT_RECEIPT" ]]; then
    echo "ERROR: initial segment must not set RESUME_CHECKPOINT_RECEIPT" >&2; exit 3
fi
python3 "$VERIFY_SCRIPT" "${verify_args[@]}"

read -r INIT_CKPT MEGATRON_DIR < <(python3 - "$TRAINING_ASSETS_RECEIPT" "$START_ITERATION" "$RESUME_CHECKPOINT_RECEIPT" <<'PY'
import json,sys
assets=json.load(open(sys.argv[1],encoding="utf-8"))
start=int(sys.argv[2]); resume=sys.argv[3]
load=assets["init_checkpoint"]["tree"]["root"]
if start:
    load=json.load(open(resume,encoding="utf-8"))["checkpoint_tree"]["root"]
print(load, assets["megatron"]["root"])
PY
)

if [[ "$DRY_RUN" == 0 ]]; then
    if (( START_ITERATION == 0 )); then
        [[ ! -e "$OUTPUT_DIR" ]] || { echo "ERROR: output already exists: $OUTPUT_DIR" >&2; exit 3; }
    else
        test -d "$OUTPUT_DIR" || { echo "ERROR: resume output is absent: $OUTPUT_DIR" >&2; exit 3; }
    fi
fi

# No sweep variables or phase-boundary variables are accepted here.
for forbidden in R FOREIGN_REPLAY_R OLD_GREEK_REPLAY_R PHASE1_EXIT_ITER RESET_DATA_INDEX \
                 TRAINER_WRAPPER LR_PEAK LR_FINAL LR_WARMUP_ITERS \
                 ADEMA_BETA2 ADEMA_BETA3 ADEMA_ALPHA; do
    if [[ -n "${!forbidden:-}" ]]; then
        echo "ERROR: unset stale/sweep variable $forbidden; the 25B config owns the frozen recipe" >&2
        exit 4
    fi
done

echo "Frozen probe: TD layer 11 · 25B nominal / 24,998,051,840 effective tokens · 79/20/1 · 5960 iterations"
echo "Bridge: $BRIDGE_MANIFEST"
echo "Dry run: $DRY_RUN"

if [[ "$DRY_RUN" == 0 && "$START_ITERATION" == 0 ]]; then
    mkdir -p "$OUTPUT_DIR"
    python3 - "$OUTPUT_DIR/probe_plan.json" "$BRIDGE_MANIFEST" "$BRIDGE_DATA_ENV" "$CONFIG" "$TRAINING_ASSETS_RECEIPT" "$OUTPUT_DIR" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,bridge,env,config,assets,output_dir=sys.argv[1:]
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()
value={
 "schema_version":"full_cpt_25b_probe_plan_v2","status":"prepared_not_started",
 "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "bridge_manifest":{"path":os.path.realpath(bridge),"sha256":sha(bridge)},
 "training_data_env":{"path":os.path.realpath(env),"sha256":sha(env)},
 "training_config":{"path":os.path.realpath(config),"sha256":sha(config)},
 "training_assets_receipt":{"path":os.path.realpath(assets),"sha256":sha(assets)},
 "output_dir":os.path.realpath(output_dir),
 "nominal_tokens":25000000000,"effective_tokens":24998051840,"nominal_floor_residual":1948160,"iterations":5960,"global_batch_tokens":4194304,
 "segment_policy":"one segment per invocation; every relaunch requires an exact checkpoint-tree receipt",
 "mix":{"new_greek":79,"foreign_replay":20,"old_greek_replay":1},
 "frozen":{"lr_peak":"5.5e-5","lr_final":"5.5e-6","warmup_iterations":400,"cooldown_fraction":"0.2","cooldown":"1-sqrt","beta1":"0.9","beta2":"0.999","beta3":"0.999","alpha":"4","goldfish_k":50,"goldfish_h":50}
}
fd,tmp=tempfile.mkstemp(prefix=".probe_plan.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
fi

current=$START_ITERATION
next=$(( current + SEGMENT_ITERS ))
(( next > TOTAL_ITER )) && next=$TOTAL_ITER
resume=0
(( current == 0 )) || resume=1
submission_receipt="$OUTPUT_DIR/segment_submissions/${current}_${next}.json"
if [[ "$DRY_RUN" == 0 && -e "$submission_receipt" ]]; then
    echo "ERROR: segment already has a submission receipt: $submission_receipt" >&2; exit 5
fi
cmd=(sbatch --parsable
        --job-name="${RUN_TAG}_i${current}_${next}" --account="$ACCOUNT" --partition="$PARTITION"
        --nodes="$NODES" --ntasks-per-node=1 --gpus-per-node="$GPUS_PER_NODE" --gres="gpu:$GPUS_PER_NODE"
        --cpus-per-task="$CPUS_PER_TASK" --mem=460G --time="$TIME_LIMIT"
        --output="$OUTPUT_DIR/%x-%j.out" --error="$OUTPUT_DIR/%x-%j.err"
        --export="ALL,ARM=td,INIT_CKPT=$INIT_CKPT,MEGATRON_DIR=$MEGATRON_DIR,OUTPUT_DIR=$OUTPUT_DIR,SCRIPT_DIR_OVERRIDE=$TRAIN_DIR,TRAIN_CONFIG_OVERRIDE=$CONFIG,BRIDGE_DATA_ENV=$BRIDGE_DATA_ENV,TRAIN_TOKENS=$TRAIN_TOKENS,RESUME_TRAINING=$resume,DISABLE_SAVE=0,SAVE_INTERVAL=$SAVE_INTERVAL,EXIT_INTERVAL=$next,ACCOUNT=$ACCOUNT,PARTITION=$PARTITION,NODES=$NODES,GPUS_PER_NODE=$GPUS_PER_NODE,LAUNCH_MODE=torchrun,TIME_LIMIT=$TIME_LIMIT"
        "$TRAIN_SCRIPT")
printf 'segment: iterations %d..%d resume=%d\n' "$current" "$next" "$resume"
if [[ "$DRY_RUN" == 1 ]]; then
    printf '  DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'
    dependency="DRY"
else
    dependency=$("${cmd[@]}")
    mkdir -p "$(dirname "$submission_receipt")"
    python3 - "$submission_receipt" "$dependency" "$current" "$next" "$BRIDGE_MANIFEST" "$TRAINING_ASSETS_RECEIPT" "${RESUME_CHECKPOINT_RECEIPT:-}" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,job,start,end,bridge,assets,resume=sys.argv[1:]
sha=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest()
value={"schema_version":"full_cpt_segment_submission_v1","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":job,"start_iteration":int(start),"end_iteration":int(end),"bridge_manifest_sha256":sha(bridge),"training_assets_receipt_sha256":sha(assets),"resume_checkpoint_receipt_sha256":sha(resume) if resume else None}
fd,tmp=tempfile.mkstemp(prefix=".segment.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
fi
echo "Submitted segment job: $dependency"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "No jobs submitted. Live launch requires DRY_RUN=0 and CONFIRM_GPU_LAUNCH=FULL_CORPUS_TD_25B."
elif (( next < TOTAL_ITER )); then
    echo "After iteration $next completes, freeze its checkpoint receipt and relaunch with START_ITERATION=$next."
fi
