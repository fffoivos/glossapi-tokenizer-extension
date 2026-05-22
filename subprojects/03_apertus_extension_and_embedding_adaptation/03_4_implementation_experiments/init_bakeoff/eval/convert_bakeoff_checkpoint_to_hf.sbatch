#!/usr/bin/env bash
#SBATCH --job-name=bakeoff_ckpt_to_hf
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Convert one saved bakeoff Megatron torch_dist checkpoint to HF format so the
# lm-eval and intrinsic metric jobs can consume it. Training checkpoints are
# TP=2, so do not pass Megatron's --test-logits path here: loader_core asserts
# TP=1 for that mode.

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/../bakeoff_training/_train_config_common.env"
GUARD="$SCRIPT_DIR/../slurm_cpu_only_guard.sh"
if [ ! -f "$GUARD" ]; then
    GUARD=/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/slurm_cpu_only_guard.sh
fi
source "$GUARD"
require_cpu_only_slurm "bakeoff_ckpt_to_hf"

ARM="${ARM:?ARM is required: vanilla|retok|centroid}"
ITER="${ITER:?ITER is required, e.g. 65}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required, e.g. bakeoff_1node_chain_20260522_005620}"

case "$ARM" in
    vanilla|retok|centroid) ;;
    *) echo "ERROR: ARM must be vanilla|retok|centroid, got: $ARM" >&2; exit 2 ;;
esac

ITER_PAD="$(printf "%07d" "$ITER")"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
MEGATRON_CKPT_ROOT="${MEGATRON_CKPT_ROOT:-$RUN_ROOT/${RUN_TAG}_${ARM}/checkpoints}"
HF_TOKENIZER_DIR="${HF_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/$ARM}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
HF_OUT_DIR="${HF_OUT_DIR:-$OUT_ROOT/${RUN_TAG}_${ARM}/iter_${ITER_PAD}_hf}"
WORK_DIR="${WORK_DIR:-$OUT_ROOT/${RUN_TAG}_${ARM}/iter_${ITER_PAD}_convert_work}"
VIEW_DIR="$WORK_DIR/megatron_view"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/eval "$WORK_DIR"

echo "=== convert_bakeoff_checkpoint_to_hf.sbatch ==="
date -u
echo "host:                $(hostname)"
echo "SCRIPT_DIR:          $SCRIPT_DIR"
echo "RUN_TAG:             $RUN_TAG"
echo "ARM:                 $ARM"
echo "ITER:                $ITER"
echo "MEGATRON_CKPT_ROOT:  $MEGATRON_CKPT_ROOT"
echo "HF_TOKENIZER_DIR:    $HF_TOKENIZER_DIR"
echo "HF_OUT_DIR:          $HF_OUT_DIR"
echo "WORK_DIR:            $WORK_DIR"
echo "MEGATRON_DIR:        $MEGATRON_LM_SWISSAI_DIR"
echo "UENV_IMAGE:          $UENV_IMAGE"
echo "OVERWRITE:           $OVERWRITE"
echo

if [ ! -d "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD" ]; then
    echo "ERROR: missing Megatron checkpoint: $MEGATRON_CKPT_ROOT/iter_$ITER_PAD" >&2
    exit 3
fi
if [ ! -f "$HF_TOKENIZER_DIR/config.json" ] || [ ! -f "$HF_TOKENIZER_DIR/tokenizer.json" ]; then
    echo "ERROR: HF_TOKENIZER_DIR must contain config.json and tokenizer.json: $HF_TOKENIZER_DIR" >&2
    exit 4
fi
if [ -e "$HF_OUT_DIR" ] && [ "$OVERWRITE" != "1" ]; then
    echo "ERROR: HF_OUT_DIR already exists: $HF_OUT_DIR" >&2
    echo "Set OVERWRITE=1 to replace it." >&2
    exit 5
fi

rm -rf "$VIEW_DIR"
mkdir -p "$VIEW_DIR"
ln -s "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD" "$VIEW_DIR/iter_$ITER_PAD"
printf "%s\n" "$ITER" > "$VIEW_DIR/latest_checkpointed_iteration.txt"

if [ "$OVERWRITE" = "1" ]; then
    rm -rf "$HF_OUT_DIR"
fi

export SCRIPT_DIR MEGATRON_LM_SWISSAI_DIR HF_TOKENIZER_DIR HF_OUT_DIR VIEW_DIR WORK_DIR UENV_IMAGE
export CONVERT_FAKE_SHARDING_WORLD_SIZE="${CONVERT_FAKE_SHARDING_WORLD_SIZE:-$TENSOR_MODEL_PARALLEL_SIZE}"

uenv run "$UENV_IMAGE" --view=default -- bash <<'INNER'
set -euo pipefail
export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"
cd "$MEGATRON_LM_SWISSAI_DIR"
echo "CONVERT_FAKE_SHARDING_WORLD_SIZE=$CONVERT_FAKE_SHARDING_WORLD_SIZE"
python3 "$SCRIPT_DIR/run_megatron_convert_with_pg.py" tools/checkpoint/convert.py --model-type GPT \
    --loader core --saver swissai_hf \
    --load-dir "$VIEW_DIR" --save-dir "$HF_OUT_DIR" \
    --hf-tokenizer "$HF_TOKENIZER_DIR" \
    --loader-transformer-impl transformer_engine \
    2>&1 | tee "$WORK_DIR/megatron_to_hf.log"
INNER

cat > "$HF_OUT_DIR/bakeoff_conversion_metadata.json" <<META
{
  "run_tag": "$RUN_TAG",
  "arm": "$ARM",
  "iteration": $ITER,
  "megatron_checkpoint_root": "$MEGATRON_CKPT_ROOT",
  "source_checkpoint": "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD",
  "hf_tokenizer_dir": "$HF_TOKENIZER_DIR",
  "hf_out_dir": "$HF_OUT_DIR",
  "megatron_dir": "$MEGATRON_LM_SWISSAI_DIR",
  "megatron_commit": "$MEGATRON_LM_SWISSAI_COMMIT",
  "uenv_image": "$UENV_IMAGE",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "end_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

echo
echo "=== conversion complete ==="
date -u
echo "HF checkpoint: $HF_OUT_DIR"
