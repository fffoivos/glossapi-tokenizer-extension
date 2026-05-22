#!/usr/bin/env bash
# Submit conversion + one packed lm-eval job for several saved bakeoff arms.
#
# Usage:
#   RUN_TAG=<run-tag> bash submit_bakeoff_checkpoint_eval_packed.sh <iter> [task-group] [arms...]
#
# Example:
#   RUN_TAG=bakeoff_1node_chain_20260522_005620 SUBMIT_INTRINSIC=1 \
#     bash submit_bakeoff_checkpoint_eval_packed.sh 390 full vanilla retok centroid

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: RUN_TAG=<run-tag> $0 <iter> [task-group=full] [arms...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITER="$1"
TASK_GROUP="${2:-full}"
shift
if [ $# -gt 0 ]; then
    shift
fi
if [ $# -gt 0 ]; then
    ARMS="$*"
else
    ARMS="${ARMS:-vanilla retok centroid}"
fi
RUN_TAG="${RUN_TAG:?RUN_TAG is required, e.g. bakeoff_1node_chain_20260522_005620}"

case "$TASK_GROUP" in
    full|retention_only|greek_only|safety_only) ;;
    *) echo "ERROR: task-group must be full|retention_only|greek_only|safety_only, got: $TASK_GROUP" >&2; exit 2 ;;
esac
if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: iter must be an integer, got: $ITER" >&2
    exit 2
fi

ITER_PAD="$(printf "%07d" "$ITER")"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
STATE_DIR="${STATE_DIR:-$OUT_ROOT/${RUN_TAG}_packed_submit_iter_${ITER_PAD}_${TASK_GROUP}}"
mkdir -p "$STATE_DIR"
LOCK_FILE="$STATE_DIR/submit.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another packed submitter holds $LOCK_FILE; exiting without duplicate submission."
    exit 0
fi
if [ -s "$STATE_DIR/packed_eval_job.id" ]; then
    existing_job="$(cat "$STATE_DIR/packed_eval_job.id")"
    echo "Packed eval already submitted for this state: $existing_job"
    echo "Remove $STATE_DIR/packed_eval_job.id manually before an intentional resubmit."
    exit 0
fi
SPEC_TSV="$STATE_DIR/eval_spec.tsv"
: > "$SPEC_TSV"

if [ "${SUBMIT_INTRINSIC:-0}" = "1" ]; then
    EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required when SUBMIT_INTRINSIC=1}"
    if [ ! -s "$EVAL_JSONL" ]; then
        echo "ERROR: EVAL_JSONL does not exist or is empty: $EVAL_JSONL" >&2
        exit 4
    fi
fi

echo "Submitting packed bakeoff checkpoint eval"
echo "  RUN_TAG=$RUN_TAG"
echo "  ITER=$ITER"
echo "  TASK_GROUP=$TASK_GROUP"
echo "  ARMS=[$ARMS]"
echo "  RUN_ROOT=$RUN_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  STATE_DIR=$STATE_DIR"
echo

convert_jobs=()
for arm in $ARMS; do
    case "$arm" in
        vanilla|retok|centroid) ;;
        *) echo "ERROR: arm must be vanilla|retok|centroid, got: $arm" >&2; exit 2 ;;
    esac
    megatron_ckpt_root="$RUN_ROOT/${RUN_TAG}_${arm}/checkpoints"
    hf_out_dir="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${ITER_PAD}_hf"
    eval_output_dir="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${ITER_PAD}_${TASK_GROUP}"

    if [ ! -d "$megatron_ckpt_root/iter_$ITER_PAD" ]; then
        echo "ERROR: missing checkpoint for $arm: $megatron_ckpt_root/iter_$ITER_PAD" >&2
        exit 3
    fi
    printf "%s\t%s\t%s\n" "$arm" "$hf_out_dir" "$eval_output_dir" >> "$SPEC_TSV"
    mkdir -p "$OUT_ROOT/${RUN_TAG}_${arm}"

    convert_job="$(sbatch --parsable \
        --export=ALL,RUN_TAG="$RUN_TAG",ARM="$arm",ITER="$ITER",MEGATRON_CKPT_ROOT="$megatron_ckpt_root",HF_OUT_DIR="$hf_out_dir",OUT_ROOT="$OUT_ROOT",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
        --job-name="tohf_${arm}_${ITER}" \
        "$SCRIPT_DIR/convert_bakeoff_checkpoint_to_hf.sbatch")"
    convert_jobs+=("$convert_job")
    echo "Submitted conversion job for $arm: $convert_job"

    if [ "${SUBMIT_INTRINSIC:-0}" = "1" ]; then
        metrics_json="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${ITER_PAD}_tokenizer_fair_metrics.json"
        diag_json="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${ITER_PAD}_new_token_diagnostics.json"

        metrics_job="$(sbatch --parsable \
            --dependency="afterok:$convert_job" \
            --export=ALL,MODEL_PATH="$hf_out_dir",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$metrics_json",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
            --job-name="bpc_${arm}_${ITER}" \
            "$SCRIPT_DIR/run_tokenizer_fair_metrics.sbatch")"

        diag_job="$(sbatch --parsable \
            --dependency="afterok:$convert_job" \
            --export=ALL,MODEL_PATH="$hf_out_dir",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$diag_json",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
            --job-name="diag_${arm}_${ITER}" \
            "$SCRIPT_DIR/run_new_token_diagnostics.sbatch")"

        echo "Submitted intrinsic metrics for $arm: $metrics_job"
        echo "Submitted diagnostics for $arm:       $diag_job"
    fi
done

dependency="$(IFS=:; echo "${convert_jobs[*]}")"
packed_job="$(sbatch --parsable \
    --dependency="afterok:$dependency" \
    --export=ALL,EVAL_SPEC_TSV="$SPEC_TSV",TASK_GROUP="$TASK_GROUP" \
    --job-name="eval_packed_${ITER}_${TASK_GROUP}" \
    "$SCRIPT_DIR/run_eval_packed_arms.sbatch")"

echo "Submitted packed lm-eval job: $packed_job"
echo "$packed_job" > "$STATE_DIR/packed_eval_job.id"
printf "%s\n" "${convert_jobs[@]}" > "$STATE_DIR/convert_jobs.ids"

echo
echo "Monitor:"
echo "  squeue -j $(IFS=,; echo "${convert_jobs[*]}"),$packed_job -o '%.18i %.32j %.2t %.10M %.10l %D %R'"
