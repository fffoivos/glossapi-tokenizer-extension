#!/usr/bin/env bash
# Submit conversion plus minimal native Greek and BPB smoke evals.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
EVAL_DIR="$LEGACY_INIT_DIR/eval"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
ITER="${ITER:-2}"
ITER_PAD="$(printf "%07d" "$ITER")"
HF_TOKENIZER_DIR="${HF_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
HELDOUT_JSONL="${HELDOUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl}"

if [ -z "${TRAIN_RUN_DIR:-}" ]; then
  latest_train="$(find "$RUN_ROOT" -maxdepth 1 -type d -name '04_vanilla_goldfish_smoke_*' | sort | tail -1)"
  if [ -z "$latest_train" ]; then
    echo "ERROR: set TRAIN_RUN_DIR or run submit_training_smoke.sh first." >&2
    exit 2
  fi
  TRAIN_RUN_DIR="$latest_train"
fi

MEGATRON_CKPT_ROOT="${MEGATRON_CKPT_ROOT:-$TRAIN_RUN_DIR/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/eval_$(basename "$TRAIN_RUN_DIR")}"
HF_OUT_DIR="${HF_OUT_DIR:-$EVAL_ROOT/iter_${ITER_PAD}_hf}"
NATIVE_OUT_DIR="${NATIVE_OUT_DIR:-$EVAL_ROOT/native_mcq_smoke}"
BPB_JSON="${BPB_JSON:-$EVAL_ROOT/heldout_bpb_smoke.json}"

test -d "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD"
test -f "$HELDOUT_JSONL"
mkdir -p "$EVAL_ROOT" "$NATIVE_OUT_DIR"

echo "Submitting 04 eval smoke"
echo "  TRAIN_RUN_DIR=$TRAIN_RUN_DIR"
echo "  MEGATRON_CKPT_ROOT=$MEGATRON_CKPT_ROOT"
echo "  HF_OUT_DIR=$HF_OUT_DIR"
echo "  NATIVE_OUT_DIR=$NATIVE_OUT_DIR"
echo "  BPB_JSON=$BPB_JSON"
echo

convert_job="$(sbatch --parsable \
  --account=a0140 \
  --partition=debug \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=72 \
  --mem=400G \
  --time=01:00:00 \
  --job-name=04van_to_hf \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,RUN_TAG="$(basename "$TRAIN_RUN_DIR")",ARM=vanilla,ITER="$ITER",MEGATRON_CKPT_ROOT="$MEGATRON_CKPT_ROOT",HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR",HF_OUT_DIR="$HF_OUT_DIR",OUT_ROOT="$EVAL_ROOT",SCRIPT_DIR_OVERRIDE="$EVAL_DIR",OVERWRITE=1 \
  "$EVAL_DIR/convert_bakeoff_checkpoint_to_hf.sbatch")"

native_job="$(sbatch --parsable \
  --dependency="afterok:$convert_job" \
  --account=a0140 \
  --partition=debug \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=18 \
  --mem=220G \
  --time=01:00:00 \
  --job-name=04van_native_smoke \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,MODEL_SPEC="VanillaSmoke=$HF_OUT_DIR",OUTPUT_DIR="$NATIVE_OUT_DIR",BENCHMARKS=all,SAMPLE_SIZE=3,CANDIDATE_BATCH_SIZE=4,EXAMPLE_BATCH_SIZE=4,SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
  "$EVAL_DIR/run_native_greek_mcq_eval.sbatch")"

bpb_job="$(sbatch --parsable \
  --dependency="afterok:$convert_job" \
  --account=a0140 \
  --partition=debug \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=18 \
  --mem=220G \
  --time=01:00:00 \
  --job-name=04van_bpb_smoke \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$HELDOUT_JSONL",OUTPUT_JSON="$BPB_JSON",MAX_DOCS=5,SCRIPT_DIR_OVERRIDE="$EVAL_DIR" \
  "$EVAL_DIR/run_tokenizer_fair_metrics.sbatch")"

cat > "$EVAL_ROOT/eval_smoke_jobs.env" <<JOBS
CONVERT_JOB_ID="$convert_job"
NATIVE_JOB_ID="$native_job"
BPB_JOB_ID="$bpb_job"
TRAIN_RUN_DIR="$TRAIN_RUN_DIR"
HF_OUT_DIR="$HF_OUT_DIR"
NATIVE_OUT_DIR="$NATIVE_OUT_DIR"
BPB_JSON="$BPB_JSON"
JOBS

echo "Submitted conversion job: $convert_job"
echo "Submitted native MCQ smoke: $native_job"
echo "Submitted BPB smoke: $bpb_job"
echo "Monitor:"
echo "  squeue -j $convert_job,$native_job,$bpb_job -o '%.18i %.32j %.2t %.10M %.10l %D %R'"
