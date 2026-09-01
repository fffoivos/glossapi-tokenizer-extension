#!/usr/bin/env bash
#SBATCH --job-name=h2g-init-roundtrip
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_INIT_SCALE:?set 8b or 1p5b}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
commit=c92402e39ef3c8e69ea378a59e79059dc14541f4
case "$H2G_INIT_SCALE" in
  8b) tp=2 ;;
  1p5b) tp=1 ;;
  *) echo "invalid H2G_INIT_SCALE=$H2G_INIT_SCALE" >&2; exit 2 ;;
esac
source_hf="$H2G_STAGE_ROOT/assets/init/${H2G_INIT_SCALE}_td_hf_training_geometry"
run="$H2G_STAGE_ROOT/assets/init/${H2G_INIT_SCALE}_roundtrip"
work="$H2G_STAGE_ROOT/work/${H2G_INIT_SCALE}_roundtrip"
converter="$H2G_STAGE_ROOT/tools/megatron_converter_${H2G_INIT_SCALE}"
receipt="$H2G_STAGE_ROOT/receipts/${H2G_INIT_SCALE}_init_roundtrip.json"
geometry_receipt="$H2G_STAGE_ROOT/receipts/${H2G_INIT_SCALE}_td_hf_training_geometry.json"
[[ -s "$source_hf/config.json" && -s "$geometry_receipt" ]] || { echo "corrected-geometry HF init or receipt missing" >&2; exit 2; }
[[ ! -e "$run" && ! -e "$work" && ! -e "$receipt" ]] || { echo "immutable roundtrip output exists" >&2; exit 2; }
mkdir -p "$H2G_STAGE_ROOT/tools" "$H2G_STAGE_ROOT/work"
if [[ ! -d "$converter/.git" ]]; then
  git clone --no-checkout "${H2G_MEGATRON_SOURCE:-https://github.com/swiss-ai/Megatron-LM.git}" "$converter"
  git -C "$converter" checkout --detach "$commit"
fi
[[ "$(git -C "$converter" rev-parse HEAD)" == "$commit" ]] || { echo "Megatron revision drift" >&2; exit 2; }
export SCRIPT_DIR_OVERRIDE="$H2G_CODE_ROOT/frozen_init_tools/megatron_patches"
export MEGATRON_DIR_OVERRIDE="$converter"
export HF_DIR="$source_hf" RUN_ROOT="$run" WORK="$work"
export RAW_MEGATRON_DIR="$run/megatron_tp${tp}_raw"
export PATCHED_MEGATRON_DIR="$run/megatron_tp${tp}_r17patched"
export ROUNDTRIP_HF_DIR="$run/hf_roundtrip"
export TARGET_TENSOR_PARALLEL_SIZE="$tp" TARGET_PIPELINE_PARALLEL_SIZE=1
export INIT_UENV_IMAGE=pytorch/v2.9.1:v2 OVERWRITE=0 LOGITS=1
bash "$SCRIPT_DIR_OVERRIDE/td_layer11_r17_roundtrip.sbatch"
export H2G_CODE_ROOT H2G_CODE_RECEIPT
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_init_roundtrip.py" \
  --scale "$H2G_INIT_SCALE" --reference-hf-root "$source_hf" --geometry-receipt "$geometry_receipt" \
  --roundtrip-hf-root "$ROUNDTRIP_HF_DIR" --megatron-root "$PATCHED_MEGATRON_DIR" \
  --verification-json "$work/verification.json" --megatron-commit "$commit" \
  --target-tensor-parallel "$tp" --output "$receipt"
