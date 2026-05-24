#!/usr/bin/env bash
# Submit checkpoint eval sidecars for the 3.5B continuation bakeoff.
#
# Reads the training_chain.tsv produced by
# ../bakeoff_training/submit_3p5b_continuation_chain.sh. Eval jobs depend on
# the segment that produces their checkpoint; later training segments do not
# depend on eval jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../bakeoff_training/_train_config_common.env"

DRY_RUN="${DRY_RUN:-1}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
TRAINING_CHAIN_TSV="${TRAINING_CHAIN_TSV:?TRAINING_CHAIN_TSV is required}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
STATE_DIR="${STATE_DIR:-$OUT_ROOT/${RUN_TAG}_sidecar_eval_submit}"
COMMANDS_SH="$STATE_DIR/eval_sbatch_commands.sh"
SIDECAR_JOBS_TSV="$STATE_DIR/eval_sidecar_jobs.tsv"
TASK_GROUP="${TASK_GROUP:-full}"
ITER_LIST="${ITER_LIST:-585 715 834}"
EVAL_JSONL="${EVAL_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl}"
EVAL_NICE="${EVAL_NICE:-1000}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-0}"
SUBMIT_INTRINSIC="${SUBMIT_INTRINSIC:-1}"

case "$DRY_RUN" in
    0|1) ;;
    *) echo "ERROR: DRY_RUN must be 0|1, got: $DRY_RUN" >&2; exit 2 ;;
esac
case "$TASK_GROUP" in
    full|retention_only|greek_only|safety_only) ;;
    *) echo "ERROR: TASK_GROUP must be full|retention_only|greek_only|safety_only, got: $TASK_GROUP" >&2; exit 2 ;;
esac
case "$SUBMIT_INTRINSIC" in
    0|1) ;;
    *) echo "ERROR: SUBMIT_INTRINSIC must be 0|1, got: $SUBMIT_INTRINSIC" >&2; exit 2 ;;
esac
if [ ! -f "$TRAINING_CHAIN_TSV" ]; then
    echo "ERROR: missing TRAINING_CHAIN_TSV: $TRAINING_CHAIN_TSV" >&2
    exit 2
fi
if [ "$SUBMIT_INTRINSIC" = "1" ] && [ ! -s "$EVAL_JSONL" ]; then
    echo "ERROR: EVAL_JSONL missing or empty: $EVAL_JSONL" >&2
    exit 3
fi

mkdir -p "$STATE_DIR"
: > "$COMMANDS_SH"
chmod +x "$COMMANDS_SH"
printf "iter\tarm\ttrain_job\tconvert_job\tmetrics_job\tdiagnostics_job\tpacked_eval_job\n" > "$SIDECAR_JOBS_TSV"

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

lookup_row() {
    local iter="$1"
    local arm="$2"
    awk -F '\t' -v iter="$iter" -v arm="$arm" '
        NR > 1 && $3 == iter && $5 == arm { print; found=1 }
        END { if (!found) exit 1 }
    ' "$TRAINING_CHAIN_TSV"
}

field_from_row() {
    local row="$1"
    local idx="$2"
    awk -F '\t' -v idx="$idx" '{ print $idx }' <<< "$row"
}

dependency_arg_for_job() {
    local job_id="$1"
    if [[ "$job_id" == DRYRUN_* ]]; then
        if [ "$DRY_RUN" = "1" ]; then
            printf -- "--dependency=afterok:%s" "$job_id"
        else
            printf ""
        fi
    else
        printf -- "--dependency=afterok:%s" "$job_id"
    fi
}

nice_args=()
if [ -n "$EVAL_NICE" ]; then
    nice_args=(--nice="$EVAL_NICE")
fi

echo "=== submit_3p5b_eval_sidecars.sh ==="
echo "DRY_RUN:            $DRY_RUN"
echo "RUN_TAG:            $RUN_TAG"
echo "TRAINING_CHAIN_TSV: $TRAINING_CHAIN_TSV"
echo "OUT_ROOT:           $OUT_ROOT"
echo "STATE_DIR:          $STATE_DIR"
echo "ITER_LIST:          $ITER_LIST"
echo "TASK_GROUP:         $TASK_GROUP"
echo "EVAL_JSONL:         $EVAL_JSONL"
echo "EVAL_NICE:          ${EVAL_NICE:-<none>}"
echo "SUBMIT_INTRINSIC:   $SUBMIT_INTRINSIC"
echo

for iter in $ITER_LIST; do
    if ! [[ "$iter" =~ ^[0-9]+$ ]]; then
        echo "ERROR: iter must be numeric, got: $iter" >&2
        exit 2
    fi

    iter_pad="$(printf "%07d" "$iter")"
    iter_state_dir="$STATE_DIR/iter_${iter_pad}_${TASK_GROUP}"
    mkdir -p "$iter_state_dir"
    spec_tsv="$iter_state_dir/eval_spec.tsv"
    : > "$spec_tsv"

    convert_jobs=()
    declare -A row_by_arm=()
    declare -A train_job_by_arm=()
    declare -A convert_job_by_arm=()
    declare -A metrics_job_by_arm=()
    declare -A diag_job_by_arm=()

    for arm in vanilla retok td_layer11; do
        if ! row="$(lookup_row "$iter" "$arm")"; then
            echo "ERROR: no row for iter=$iter arm=$arm in $TRAINING_CHAIN_TSV" >&2
            exit 4
        fi
        row_by_arm["$arm"]="$row"
        output_dir="$(field_from_row "$row" 7)"
        train_job="$(field_from_row "$row" 10)"
        train_job_by_arm["$arm"]="$train_job"

        megatron_ckpt_root="$output_dir/checkpoints"
        hf_out_dir="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${iter_pad}_hf"
        eval_output_dir="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${iter_pad}_${TASK_GROUP}"
        printf "%s\t%s\t%s\n" "$arm" "$hf_out_dir" "$eval_output_dir" >> "$spec_tsv"
        if [ "$DRY_RUN" = "0" ]; then
            mkdir -p "$OUT_ROOT/${RUN_TAG}_${arm}"
        fi

        dep="$(dependency_arg_for_job "$train_job")"
        dep_args=()
        if [ -n "$dep" ]; then
            dep_args=("$dep")
        elif [ "$DRY_RUN" = "0" ]; then
            echo "ERROR: live eval submission needs real training job IDs; got $train_job for $arm iter $iter" >&2
            exit 5
        fi

        convert_cmd=(
            sbatch --parsable
            "${dep_args[@]}"
            "${nice_args[@]}"
            --export=ALL,RUN_TAG="$RUN_TAG",ARM="$arm",ITER="$iter",MEGATRON_CKPT_ROOT="$megatron_ckpt_root",HF_OUT_DIR="$hf_out_dir",OUT_ROOT="$OUT_ROOT",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",OVERWRITE="$OVERWRITE_EVAL"
            --job-name="tohf_${arm}_${iter}"
            "$SCRIPT_DIR/convert_bakeoff_checkpoint_to_hf.sbatch"
        )
        convert_job="$(submit_or_dryrun "DRYRUN_CONVERT_${arm}_${iter}" "${convert_cmd[@]}")"
        convert_jobs+=("$convert_job")
        convert_job_by_arm["$arm"]="$convert_job"

        if [ "$SUBMIT_INTRINSIC" = "1" ]; then
            metrics_json="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${iter_pad}_tokenizer_fair_metrics.json"
            metrics_cmd=(
                sbatch --parsable
                --dependency="afterok:$convert_job"
                "${nice_args[@]}"
                --export=ALL,MODEL_PATH="$hf_out_dir",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$metrics_json",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",OVERWRITE="$OVERWRITE_EVAL"
                --job-name="bpc_${arm}_${iter}"
                "$SCRIPT_DIR/run_tokenizer_fair_metrics.sbatch"
            )
            metrics_job="$(submit_or_dryrun "DRYRUN_BPC_${arm}_${iter}" "${metrics_cmd[@]}")"
            metrics_job_by_arm["$arm"]="$metrics_job"

            if [ "$arm" = "retok" ] || [ "$arm" = "td_layer11" ]; then
                diag_json="$OUT_ROOT/${RUN_TAG}_${arm}/iter_${iter_pad}_new_token_diagnostics.json"
                diag_cmd=(
                    sbatch --parsable
                    --dependency="afterok:$convert_job"
                    "${nice_args[@]}"
                    --export=ALL,MODEL_PATH="$hf_out_dir",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$diag_json",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",OVERWRITE="$OVERWRITE_EVAL"
                    --job-name="diag_${arm}_${iter}"
                    "$SCRIPT_DIR/run_new_token_diagnostics.sbatch"
                )
                diag_job="$(submit_or_dryrun "DRYRUN_DIAG_${arm}_${iter}" "${diag_cmd[@]}")"
                diag_job_by_arm["$arm"]="$diag_job"
            else
                diag_job_by_arm["$arm"]="n/a"
            fi
        else
            metrics_job_by_arm["$arm"]="disabled"
            diag_job_by_arm["$arm"]="disabled"
        fi
    done

    dependency="$(IFS=:; echo "${convert_jobs[*]}")"
    packed_cmd=(
        sbatch --parsable
        --dependency="afterok:$dependency"
        "${nice_args[@]}"
        --export=ALL,EVAL_SPEC_TSV="$spec_tsv",TASK_GROUP="$TASK_GROUP"
        --job-name="eval_3p5_${iter}_${TASK_GROUP}"
        "$SCRIPT_DIR/run_eval_packed_arms.sbatch"
    )
    packed_job="$(submit_or_dryrun "DRYRUN_PACKED_${iter}_${TASK_GROUP}" "${packed_cmd[@]}")"

    for arm in vanilla retok td_layer11; do
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$iter" "$arm" "${train_job_by_arm[$arm]}" "${convert_job_by_arm[$arm]}" \
            "${metrics_job_by_arm[$arm]}" "${diag_job_by_arm[$arm]}" "$packed_job" >> "$SIDECAR_JOBS_TSV"
    done

    echo "iter=$iter packed_eval=$packed_job conversions=${convert_jobs[*]}"
done

echo
echo "Eval sidecar manifest: $SIDECAR_JOBS_TSV"
echo "Eval sbatch commands:  $COMMANDS_SH"
