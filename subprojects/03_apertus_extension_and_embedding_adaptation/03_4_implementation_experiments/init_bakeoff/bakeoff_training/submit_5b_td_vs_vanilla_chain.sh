#!/usr/bin/env bash
# Continue the two live contenders, Vanilla and TD layer11, from 3.5B to 5B.
#
# Shape:
# - two arms run in parallel;
# - each arm is split into two checkpoint-boundary training segments;
# - eval sidecars depend on the checkpoint-producing segment only, so later
#   training keeps moving while conversion/eval jobs run in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_config_common.env"

DRY_RUN="${DRY_RUN:-1}"
CONFIRM_5B_LAUNCH="${CONFIRM_5B_LAUNCH:-0}"
SUBMIT_EVAL_SIDECARS="${SUBMIT_EVAL_SIDECARS:-1}"

PREV_3P5B_RUN_TAG="${PREV_3P5B_RUN_TAG:-continuation_3p5b_20260524T143012Z}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
RUN_TAG="${RUN_TAG:-continuation_5b_td_vs_vanilla_$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="${STATE_DIR:-$OUT_ROOT/${RUN_TAG}_submit_state}"
TRAINING_CHAIN_TSV="$STATE_DIR/training_chain.tsv"
COMMANDS_SH="$STATE_DIR/training_sbatch_commands.sh"

SEGMENT_TIME_LIMIT="${SEGMENT_TIME_LIMIT:-$TIME_LIMIT}"
RUN_SAVE_INTERVAL="${SAVE_INTERVAL:-65}"
RUN_EVAL_INTERVAL="${CONTINUATION_EVAL_INTERVAL:-999999}"
RUN_LOSS_OBJECTIVE="${CONTINUATION_LOSS_OBJECTIVE:-ntp}"
CONT_BASE_DATA_PREFIX="${CONT_BASE_DATA_PREFIX:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_megatron/bulk_mix_text_document}"
CONT_EXT_DATA_PREFIX="${CONT_EXT_DATA_PREFIX:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

case "$DRY_RUN" in
    0|1) ;;
    *) echo "ERROR: DRY_RUN must be 0|1, got: $DRY_RUN" >&2; exit 2 ;;
esac
case "$SUBMIT_EVAL_SIDECARS" in
    0|1) ;;
    *) echo "ERROR: SUBMIT_EVAL_SIDECARS must be 0|1, got: $SUBMIT_EVAL_SIDECARS" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_5B_LAUNCH" != "1" ]; then
    echo "ERROR: live submission requires CONFIRM_5B_LAUNCH=1" >&2
    echo "This launches two chained 4x-GPU training arms plus eval sidecars." >&2
    exit 2
fi

mkdir -p "$STATE_DIR"
: > "$COMMANDS_SH"
chmod +x "$COMMANDS_SH"
printf "run_tag\tsegment\ttarget_iter\ttarget_tokens\tarm\ttrain_arm\toutput_dir\tinit_ckpt\tdependency_job\tjob_id\n" > "$TRAINING_CHAIN_TSV"

log_cmd() {
    printf '%q ' "$@" >> "$COMMANDS_SH"
    printf '\n' >> "$COMMANDS_SH"
}

submit_or_dryrun() {
    local dry_id="$1"
    shift
    log_cmd "$@"
    if [ "$DRY_RUN" = "1" ]; then
        echo "$dry_id"
    else
        "$@"
    fi
}

source_ckpt_for_arm() {
    local arm="$1"
    case "$arm" in
        vanilla|td_layer11)
            printf "%s/%s_%s/checkpoints" "$OUT_ROOT" "$PREV_3P5B_RUN_TAG" "$arm"
            ;;
        *)
            echo "ERROR: unknown arm: $arm" >&2
            exit 2
            ;;
    esac
}

train_arm_for_arm() {
    local arm="$1"
    case "$arm" in
        vanilla) echo "vanilla" ;;
        td_layer11) echo "retok" ;;
        *) echo "ERROR: unknown arm: $arm" >&2; exit 2 ;;
    esac
}

check_source_checkpoint() {
    local arm="$1"
    local ckpt="$2"
    if [ ! -d "$ckpt/iter_0000834" ]; then
        echo "ERROR: missing 3.5B source checkpoint for $arm: $ckpt/iter_0000834" >&2
        exit 3
    fi
}

check_output_dir() {
    local dir="$1"
    if [ -e "$dir" ] && [ "$ALLOW_EXISTING_OUTPUT" != "1" ]; then
        echo "ERROR: output dir already exists: $dir" >&2
        echo "Set ALLOW_EXISTING_OUTPUT=1 only for an intentional resume/resubmit." >&2
        exit 4
    fi
}

check_data_prefix() {
    local label="$1"
    local prefix="$2"
    if [ ! -f "${prefix}.bin" ] || [ ! -f "${prefix}.idx" ]; then
        echo "ERROR: missing $label Megatron data prefix: ${prefix}{.bin,.idx}" >&2
        exit 3
    fi
}

echo "=== submit_5b_td_vs_vanilla_chain.sh ==="
echo "DRY_RUN:              $DRY_RUN"
echo "RUN_TAG:              $RUN_TAG"
echo "OUT_ROOT:             $OUT_ROOT"
echo "EVAL_OUT_ROOT:        $EVAL_OUT_ROOT"
echo "STATE_DIR:            $STATE_DIR"
echo "PREV_3P5B_RUN_TAG:    $PREV_3P5B_RUN_TAG"
echo "ACCOUNT:              $ACCOUNT"
echo "PARTITION:            $PARTITION"
echo "NODES:                $NODES"
echo "GPUS_PER_NODE:        $GPUS_PER_NODE"
echo "LAUNCH_MODE:          $LAUNCH_MODE"
echo "SEGMENT_TIME_LIMIT:   $SEGMENT_TIME_LIMIT"
echo "SAVE_INTERVAL:        $RUN_SAVE_INTERVAL"
echo "EVAL_INTERVAL:        $RUN_EVAL_INTERVAL"
echo "LOSS_OBJECTIVE:       $RUN_LOSS_OBJECTIVE"
echo "BASE_DATA_PREFIX:     $CONT_BASE_DATA_PREFIX"
echo "EXT_DATA_PREFIX:      $CONT_EXT_DATA_PREFIX"
echo "SUBMIT_EVAL_SIDECARS: $SUBMIT_EVAL_SIDECARS"
echo

check_data_prefix "base" "$CONT_BASE_DATA_PREFIX"
check_data_prefix "extended" "$CONT_EXT_DATA_PREFIX"

for arm in vanilla td_layer11; do
    src_ckpt="$(source_ckpt_for_arm "$arm")"
    check_source_checkpoint "$arm" "$src_ckpt"
    check_output_dir "$OUT_ROOT/${RUN_TAG}_${arm}"
done

# target_iter target_tokens. 1013 is ~4.249B tokens; 1192 is ~4.9996B tokens.
segments=(
    "1 1013 4248829952"
    "2 1192 4999610368"
)

for arm in vanilla td_layer11; do
    output_dir="$OUT_ROOT/${RUN_TAG}_${arm}"
    train_arm="$(train_arm_for_arm "$arm")"
    init_ckpt="$(source_ckpt_for_arm "$arm")"
    dependency_job=""
    if [ "$DRY_RUN" = "0" ]; then
        mkdir -p "$output_dir"
    fi

    for segment in "${segments[@]}"; do
        read -r segment_idx target_iter target_tokens <<< "$segment"
        job_name="5b_${arm}_${target_iter}"
        dependency_args=()
        dependency_label="none"
        if [ -n "$dependency_job" ]; then
            dependency_args=(--dependency="afterok:$dependency_job")
            dependency_label="$dependency_job"
            init_ckpt="$output_dir/checkpoints"
        fi

        cmd=(
            sbatch --parsable
            "${dependency_args[@]}"
            --job-name="$job_name"
            --account="$ACCOUNT"
            --partition="$PARTITION"
            --nodes="$NODES"
            --ntasks-per-node="$SBATCH_NTASKS_PER_NODE"
            --gpus-per-node="$GPUS_PER_NODE"
            --gres="gpu:$GPUS_PER_NODE"
            --time="$SEGMENT_TIME_LIMIT"
            --export=ALL,ARM="$train_arm",INIT_CKPT="$init_ckpt",OUTPUT_DIR="$output_dir",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SEGMENT_TIME_LIMIT",TRAIN_TOKENS="$target_tokens",SAVE_INTERVAL="$RUN_SAVE_INTERVAL",EVAL_INTERVAL="$RUN_EVAL_INTERVAL",LOSS_OBJECTIVE="$RUN_LOSS_OBJECTIVE",BASE_DATA_PREFIX="$CONT_BASE_DATA_PREFIX",EXT_DATA_PREFIX="$CONT_EXT_DATA_PREFIX",DISABLE_SAVE=0,RESUME_TRAINING=1
            "$SCRIPT_DIR/bakeoff_train.sbatch"
        )

        job_id="$(submit_or_dryrun "DRYRUN_${arm}_${target_iter}" "${cmd[@]}")"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$RUN_TAG" "$segment_idx" "$target_iter" "$target_tokens" "$arm" "$train_arm" \
            "$output_dir" "$init_ckpt" "$dependency_label" "$job_id" >> "$TRAINING_CHAIN_TSV"
        echo "segment $segment_idx arm=$arm target_iter=$target_iter job=$job_id dependency=$dependency_label"
        dependency_job="$job_id"
    done
done

echo
echo "Training chain manifest: $TRAINING_CHAIN_TSV"
echo "Training sbatch commands: $COMMANDS_SH"

if [ "$SUBMIT_EVAL_SIDECARS" = "1" ]; then
    eval_script="$SCRIPT_DIR/../eval/submit_3p5b_eval_sidecars_incremental.py"
    if [ ! -f "$eval_script" ]; then
        echo "ERROR: missing eval sidecar submitter: $eval_script" >&2
        exit 5
    fi

    eval_submit_log="$STATE_DIR/eval_sidecar_submitter.log"
    eval_submit_job="$(
        submit_or_dryrun "DRYRUN_EVAL_SUBMITTER" \
            sbatch --parsable \
            --job-name="eval_submit_5b" \
            --account="$ACCOUNT" \
            --partition=xfer \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=4G \
            --time=24:00:00 \
            --output="$eval_submit_log" \
            --error="$STATE_DIR/eval_sidecar_submitter.err" \
            --export=ALL,PYTHONUNBUFFERED=1,RUN_TAG="$RUN_TAG",TRAINING_CHAIN_TSV="$TRAINING_CHAIN_TSV",OUT_ROOT="$EVAL_OUT_ROOT",STATE_DIR="$EVAL_OUT_ROOT/${RUN_TAG}_sidecar_eval_incremental",ITER_LIST="1013 1192",TASK_GROUP=full,EVAL_ARMS="vanilla td_layer11",DIAG_ARMS="td_layer11",PACKED_JOB_PREFIX="eval_5b",LOOP=1,MAX_SUBMITTED_JOBS=11,SLEEP_SECONDS=120,SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR/../eval" \
            --wrap="cd '$SCRIPT_DIR/../eval' && python3 -u '$eval_script'"
    )"
    printf "%s\n" "$eval_submit_job" > "$STATE_DIR/eval_submitter_job.txt"
    echo "Eval sidecar submitter job: $eval_submit_job"
    echo "Eval sidecar submitter log: $eval_submit_log"
    echo "Eval sidecar submitter job file: $STATE_DIR/eval_submitter_job.txt"
fi
