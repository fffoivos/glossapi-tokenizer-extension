#!/usr/bin/env bash
#SBATCH --job-name=convert_init_ckpts
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/init/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/init/%x-%j.err
#
# Convert HF-format init checkpoints into Megatron torch_dist release checkpoints.

set -euo pipefail

# Slurm runs a copied script from /var/spool/slurmd; use the submit directory
# as the stable repo location. submit_init_pipeline.sh submits from arms/.
SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
PATCH_DIR="$SCRIPT_DIR/../megatron_patches"
source "$SCRIPT_DIR/../bakeoff_training/_train_config_common.env"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480}"
ARMS="${ARMS:-vanilla retok centroid}"
MEGATRON_SUBDIR="${MEGATRON_SUBDIR:-megatron}"
TARGET_TENSOR_PARALLEL_SIZE="${TARGET_TENSOR_PARALLEL_SIZE:-1}"
TARGET_PIPELINE_PARALLEL_SIZE="${TARGET_PIPELINE_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
INIT_UENV_IMAGE="${INIT_UENV_IMAGE:-pytorch/v2.9.1:v2}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/init

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/fffoivos/tmp/init_convert_hf_home_${SLURM_JOB_ID:-manual}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export TMPDIR="${TMPDIR:-/iopsstor/scratch/cscs/fffoivos/tmp/init_convert_tmp_${SLURM_JOB_ID:-manual}}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

echo "=== convert_init_checkpoints.sbatch ==="
date -u
echo "host: $(hostname)"
echo "SCRIPT_DIR:      $SCRIPT_DIR"
echo "PATCH_DIR:       $PATCH_DIR"
echo "MEGATRON_DIR:    $MEGATRON_LM_SWISSAI_DIR"
echo "INIT_CKPT_ROOT:  $INIT_CKPT_ROOT"
echo "ARMS:            $ARMS"
echo "MEGATRON_SUBDIR: $MEGATRON_SUBDIR"
echo "TARGET_TP:       $TARGET_TENSOR_PARALLEL_SIZE"
echo "TARGET_PP:       $TARGET_PIPELINE_PARALLEL_SIZE"
echo "OVERWRITE:       $OVERWRITE"
echo "INIT_UENV_IMAGE: $INIT_UENV_IMAGE"
echo "HF_HOME:         $HF_HOME"
echo

bash "$PATCH_DIR/install.sh" "$MEGATRON_LM_SWISSAI_DIR"

export INIT_CKPT_ROOT ARMS MEGATRON_SUBDIR TARGET_TENSOR_PARALLEL_SIZE TARGET_PIPELINE_PARALLEL_SIZE OVERWRITE MEGATRON_LM_SWISSAI_DIR
uenv run "$INIT_UENV_IMAGE" --view=default -- bash <<'INNER'
set -euo pipefail
export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"
cd "$MEGATRON_LM_SWISSAI_DIR"
git rev-parse HEAD

for arm in $ARMS; do
    hf_dir="$INIT_CKPT_ROOT/$arm"
    meg_dir="$INIT_CKPT_ROOT/$arm/$MEGATRON_SUBDIR"
    echo
    echo "=== converting arm: $arm ==="
    echo "HF:       $hf_dir"
    echo "Megatron: $meg_dir"

    if [ ! -f "$hf_dir/config.json" ]; then
        echo "ERROR: missing HF checkpoint config: $hf_dir/config.json" >&2
        exit 2
    fi

    if [ -e "$meg_dir" ]; then
        if [ "$OVERWRITE" = "1" ]; then
            echo "OVERWRITE=1, removing existing Megatron dir: $meg_dir"
            rm -rf "$meg_dir"
        else
            echo "ERROR: Megatron dir already exists: $meg_dir" >&2
            echo "Set OVERWRITE=1 only if you intentionally want to replace it." >&2
            exit 3
        fi
    fi

    python3 tools/checkpoint/convert.py --model-type GPT \
        --loader apertus_hf --saver core \
        --load-dir "$hf_dir" --save-dir "$meg_dir" \
        --tokenizer-model "$hf_dir" --bf16 \
        --loader-transformer-impl transformer_engine \
        --target-tensor-parallel-size "$TARGET_TENSOR_PARALLEL_SIZE" \
        --target-pipeline-parallel-size "$TARGET_PIPELINE_PARALLEL_SIZE"

    if [ -d "$meg_dir/iter_0000000" ]; then
        mv "$meg_dir/iter_0000000" "$meg_dir/release"
    fi
    if [ ! -d "$meg_dir/release" ]; then
        echo "ERROR: expected release checkpoint directory after conversion: $meg_dir/release" >&2
        exit 4
    fi
    echo release > "$meg_dir/latest_checkpointed_iteration.txt"
    ls -la "$meg_dir"
done
INNER

echo
echo "=== convert init checkpoints done ==="
date -u
for arm in $ARMS; do
    test -d "$INIT_CKPT_ROOT/$arm/$MEGATRON_SUBDIR/release"
    echo "$arm: $INIT_CKPT_ROOT/$arm/$MEGATRON_SUBDIR/release"
done
