#!/usr/bin/env bash
#SBATCH --job-name=build_init_ckpts
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
# Build the HF-format init checkpoints for Vanilla / ReTok / Centroid.
# Conversion to Megatron torch_dist is handled by convert_init_checkpoints.sbatch.

set -euo pipefail

# Slurm runs a copied script from /var/spool/slurmd; use the submit directory
# as the stable repo location. submit_init_pipeline.sh submits from arms/.
SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/../bakeoff_training/_train_config_common.env"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480}"
VOCAB_SIZE="${VOCAB_SIZE:-148480}"
ARMS="${ARMS:-vanilla retok centroid}"
INIT_UENV_IMAGE="${INIT_UENV_IMAGE:-pytorch/v2.9.1:v2}"
read -r -a ARM_LIST <<< "$ARMS"

mkdir -p "$INIT_CKPT_ROOT" /capstor/scratch/cscs/fffoivos/runs/init

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/fffoivos/tmp/init_hf_home_${SLURM_JOB_ID:-manual}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export TMPDIR="${TMPDIR:-/iopsstor/scratch/cscs/fffoivos/tmp/init_tmp_${SLURM_JOB_ID:-manual}}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

echo "=== build_init_checkpoints.sbatch ==="
date -u
echo "host: $(hostname)"
echo "SCRIPT_DIR:      $SCRIPT_DIR"
echo "APERTUS_BASE_HF: $APERTUS_BASE_HF"
echo "EXT_TOKENIZER:   $EXT_TOKENIZER_DIR"
echo "INIT_CKPT_ROOT:  $INIT_CKPT_ROOT"
echo "VOCAB_SIZE:      $VOCAB_SIZE"
echo "ARMS:            ${ARM_LIST[*]}"
echo "INIT_UENV_IMAGE: $INIT_UENV_IMAGE"
echo "HF_HOME:         $HF_HOME"
echo

cmd=(
    python3 "$SCRIPT_DIR/build_init_checkpoints.py"
    --apertus-base "$APERTUS_BASE_HF"
    --extended-tokenizer "$EXT_TOKENIZER_DIR"
    --out-root "$INIT_CKPT_ROOT"
    --vocab-size "$VOCAB_SIZE"
    --arms "${ARM_LIST[@]}"
)

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n\n'

uenv run "$INIT_UENV_IMAGE" --view=default -- "${cmd[@]}"

echo
echo "=== build init checkpoints done ==="
date -u
find "$INIT_CKPT_ROOT" -maxdepth 2 -type f \( -name 'config.json' -o -name 'init_build_summary.json' \) | sort
