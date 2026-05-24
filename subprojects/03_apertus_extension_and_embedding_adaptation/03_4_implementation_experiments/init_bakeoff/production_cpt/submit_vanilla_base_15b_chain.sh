#!/usr/bin/env bash
# Submit the selected production CPT path:
#   Vanilla Apertus-8B-2509, base tokenizer, NFC-safe bulk corpus, Goldfish loss.
#
# Safety default: DRY_RUN=1 prints the sbatch chain without launching.
# To actually launch:
#   DRY_RUN=0 CONFIRM_PRODUCTION_LAUNCH=1 bash submit_vanilla_base_15b_chain.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAKEOFF_TRAINING_DIR="$SCRIPT_DIR/../bakeoff_training"

TRAIN_TOKENS="${TRAIN_TOKENS:-15000000000}"
BASE_DATA_PREFIX="${BASE_DATA_PREFIX:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document}"
LOSS_OBJECTIVE="${LOSS_OBJECTIVE:-goldfish}"
SAVE_INTERVAL="${SAVE_INTERVAL:-120}"      # ~503M tokens at 4,194,304 tokens/step
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"   # downstream eval is submitted externally from saved checkpoints
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARTITION="${PARTITION:-normal}"
ACCOUNT="${ACCOUNT:-a0140}"
LR_WARMUP_TOKENS="${LR_WARMUP_TOKENS:-$(( TRAIN_TOKENS / 50 ))}" # 2% re-warmup

source "$BAKEOFF_TRAINING_DIR/_train_config_common.env"

INIT_CKPT="${INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/production_cpt}"
RUN_TAG="${RUN_TAG:-vanilla_base_15b_nfc_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$OUT_ROOT/$RUN_TAG"
DRY_RUN="${DRY_RUN:-1}"
CHAIN_JOBS="${CHAIN_JOBS:-14}"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-afterok}"

TRAIN_SAMPLES=$(( TRAIN_TOKENS / SEQ_LENGTH ))
TRAIN_ITERS=$(( TRAIN_SAMPLES / GLOBAL_BATCH_SIZE ))
RUN_ADEMA_WARMUP_STEPS="${ADEMA_WARMUP_STEPS:-$(( (TRAIN_ITERS * 28 + 999) / 1000 ))}" # ceil(2.8% of steps)

case "$LOSS_OBJECTIVE" in
    goldfish) ;;
    *) echo "ERROR: production CPT should run LOSS_OBJECTIVE=goldfish, got $LOSS_OBJECTIVE" >&2; exit 2 ;;
esac
if [ "$NODES" != "1" ]; then
    echo "ERROR: production launcher defaults to the proven one-node path; NODES=$NODES is not enabled here" >&2
    echo "       The prior two-node smoke failed with NCCL/OFI NO_SPACE before iteration 1." >&2
    exit 2
fi
if ! [[ "$CHAIN_JOBS" =~ ^[0-9]+$ ]] || [ "$CHAIN_JOBS" -lt 1 ]; then
    echo "ERROR: CHAIN_JOBS must be a positive integer, got $CHAIN_JOBS" >&2
    exit 2
fi
case "$DEPENDENCY_MODE" in
    afterok|afterany) ;;
    *) echo "ERROR: DEPENDENCY_MODE must be afterok|afterany, got $DEPENDENCY_MODE" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" != "1" ] && [ "${CONFIRM_PRODUCTION_LAUNCH:-0}" != "1" ]; then
    echo "ERROR: refusing live launch without CONFIRM_PRODUCTION_LAUNCH=1" >&2
    exit 2
fi

SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

for path in "$INIT_CKPT/release" "${BASE_DATA_PREFIX}.bin" "${BASE_DATA_PREFIX}.idx" "$BAKEOFF_TRAINING_DIR/bakeoff_train.sbatch"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: required path missing: $path" >&2
        echo "Run this script on the Clariden mirror, after syncing the repo and data artifacts." >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_DIR"

cat > "$OUTPUT_DIR/submission_plan.json" <<META
{
  "run_tag": "$RUN_TAG",
  "output_dir": "$OUTPUT_DIR",
  "selected_path": "vanilla_base_tokenizer",
  "init_ckpt": "$INIT_CKPT",
  "data_prefix": "$BASE_DATA_PREFIX",
  "train_tokens": $TRAIN_TOKENS,
  "train_samples": $TRAIN_SAMPLES,
  "train_iters": $TRAIN_ITERS,
  "global_batch_tokens": $GLOBAL_BATCH_TOKENS,
  "save_interval": $SAVE_INTERVAL,
  "save_interval_tokens": $(( SAVE_INTERVAL * GLOBAL_BATCH_TOKENS )),
  "loss_objective": "$LOSS_OBJECTIVE",
  "goldfish_k": $GOLDFISH_K,
  "goldfish_h": $GOLDFISH_H,
  "lr_warmup_tokens": $LR_WARMUP_TOKENS,
  "ademamix_warmup_steps": $RUN_ADEMA_WARMUP_STEPS,
  "nodes": $NODES,
  "gpus_per_node": $GPUS_PER_NODE,
  "chain_jobs": $CHAIN_JOBS,
  "dependency_mode": "$DEPENDENCY_MODE",
  "time_limit": "$TIME_LIMIT",
  "dry_run": $DRY_RUN
}
META

echo "=== submit_vanilla_base_15b_chain.sh ==="
echo "RUN_TAG:              $RUN_TAG"
echo "OUTPUT_DIR:           $OUTPUT_DIR"
echo "INIT_CKPT:            $INIT_CKPT"
echo "BASE_DATA_PREFIX:     $BASE_DATA_PREFIX"
echo "TRAIN_TOKENS:         $TRAIN_TOKENS"
echo "TRAIN_ITERS:          $TRAIN_ITERS"
echo "LOSS_OBJECTIVE:       $LOSS_OBJECTIVE"
echo "LR_WARMUP_TOKENS:     $LR_WARMUP_TOKENS"
echo "ADEMA_WARMUP_STEPS:   $RUN_ADEMA_WARMUP_STEPS"
echo "SAVE_INTERVAL:        $SAVE_INTERVAL (~$(( SAVE_INTERVAL * GLOBAL_BATCH_TOKENS )) tokens)"
echo "CHAIN_JOBS:           $CHAIN_JOBS"
echo "DEPENDENCY_MODE:      $DEPENDENCY_MODE"
echo "DRY_RUN:              $DRY_RUN"
echo

echo -e "chain_index\tjob_id\tresume\tdependency\tinit_ckpt" > "$OUTPUT_DIR/submission_chain.tsv"

prev_job_id=""
for chain_index in $(seq 0 $(( CHAIN_JOBS - 1 ))); do
    resume_training=0
    load_ckpt="$INIT_CKPT"
    job_name="prod_vanilla_15b"
    dep_args=()
    dep_label=""

    if [ "$chain_index" -gt 0 ]; then
        resume_training=1
        load_ckpt="$OUTPUT_DIR/checkpoints"
        job_name="prod_vanilla_15b_r$(printf "%02d" "$chain_index")"
        dep_args=(--dependency="$DEPENDENCY_MODE:$prev_job_id")
        dep_label="$DEPENDENCY_MODE:$prev_job_id"
    fi

    sbatch_args=(
        "${dep_args[@]}"
        --parsable
        --job-name="$job_name"
        --account="$ACCOUNT"
        --partition="$PARTITION"
        --nodes="$NODES"
        --ntasks-per-node="$SBATCH_NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --gres="gpu:$GPUS_PER_NODE"
        --time="$TIME_LIMIT"
        --export=ALL,ARM=vanilla,INIT_CKPT="$load_ckpt",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$BAKEOFF_TRAINING_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$TIME_LIMIT",TRAIN_TOKENS="$TRAIN_TOKENS",BASE_DATA_PREFIX="$BASE_DATA_PREFIX",LOSS_OBJECTIVE="$LOSS_OBJECTIVE",SAVE_INTERVAL="$SAVE_INTERVAL",EVAL_INTERVAL="$EVAL_INTERVAL",LR_WARMUP_TOKENS="$LR_WARMUP_TOKENS",ADEMA_BETA3_WARMUP_STEPS="$RUN_ADEMA_WARMUP_STEPS",ADEMA_ALPHA_WARMUP_STEPS="$RUN_ADEMA_WARMUP_STEPS",DISABLE_SAVE=0,RESUME_TRAINING="$resume_training"
        "$BAKEOFF_TRAINING_DIR/bakeoff_train.sbatch"
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf 'DRY-RUN chain[%02d]: sbatch' "$chain_index"
        printf ' %q' "${sbatch_args[@]}"
        printf '\n'
        job_id="DRYRUN_$chain_index"
    else
        job_id="$(sbatch "${sbatch_args[@]}")"
        echo "submitted chain[$chain_index]: $job_id"
    fi
    echo -e "$chain_index\t$job_id\t$resume_training\t$dep_label\t$load_ckpt" >> "$OUTPUT_DIR/submission_chain.tsv"
    prev_job_id="$job_id"
done

echo
echo "Plan written to: $OUTPUT_DIR/submission_plan.json"
echo "Chain written to: $OUTPUT_DIR/submission_chain.tsv"
if [ "$DRY_RUN" = "1" ]; then
    echo "No Slurm jobs were submitted."
else
    echo "Monitor with: squeue -u $USER"
fi
