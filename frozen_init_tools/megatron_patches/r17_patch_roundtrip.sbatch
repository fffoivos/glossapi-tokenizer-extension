#!/usr/bin/env bash
#SBATCH --job-name=r17_patch_roundtrip
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Patch R17 Apertus extras into a converted Megatron checkpoint, convert back
# to HF, and prove the roundtrip with tensor + small logit diffs.

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/../bakeoff_training/_train_config_common.env"

ARM="${ARM:-vanilla}"
INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480}"
MEGATRON_SUBDIR="${MEGATRON_SUBDIR:-megatron_tp2}"
PATCHED_SUBDIR="${PATCHED_SUBDIR:-${MEGATRON_SUBDIR}_r17patched}"
INIT_UENV_IMAGE="${INIT_UENV_IMAGE:-pytorch/v2.9.1:v2}"
HF_DIR="${HF_DIR:-$INIT_CKPT_ROOT/$ARM}"
MEGATRON_IN_DIR="${MEGATRON_IN_DIR:-$HF_DIR/$MEGATRON_SUBDIR}"
PATCHED_MEGATRON_DIR="${PATCHED_MEGATRON_DIR:-$HF_DIR/$PATCHED_SUBDIR}"
WORK="${WORK:-/capstor/scratch/cscs/fffoivos/runs/r17_patch_roundtrip_${ARM}_${SLURM_JOB_ID:-manual}}"
ROUNDTRIP_HF_DIR="${ROUNDTRIP_HF_DIR:-$WORK/${ARM}_hf_roundtrip_patched}"
OVERWRITE="${OVERWRITE:-0}"
LOGITS="${LOGITS:-1}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/eval "$WORK"

echo "=== r17_patch_roundtrip.sbatch ==="
date -u
echo "host:                  $(hostname)"
echo "SCRIPT_DIR:            $SCRIPT_DIR"
echo "ARM:                   $ARM"
echo "HF_DIR:                $HF_DIR"
echo "MEGATRON_IN_DIR:       $MEGATRON_IN_DIR"
echo "PATCHED_MEGATRON_DIR:  $PATCHED_MEGATRON_DIR"
echo "ROUNDTRIP_HF_DIR:      $ROUNDTRIP_HF_DIR"
echo "MEGATRON_DIR:          $MEGATRON_LM_SWISSAI_DIR"
echo "INIT_UENV_IMAGE:       $INIT_UENV_IMAGE"
echo "OVERWRITE:             $OVERWRITE"
echo "LOGITS:                $LOGITS"
echo "WORK:                  $WORK"
echo

if [ ! -f "$HF_DIR/config.json" ]; then
    echo "ERROR: missing HF config: $HF_DIR/config.json" >&2
    exit 2
fi
if [ ! -d "$MEGATRON_IN_DIR/release" ]; then
    echo "ERROR: missing Megatron release dir: $MEGATRON_IN_DIR/release" >&2
    exit 3
fi
if [ -e "$PATCHED_MEGATRON_DIR" ] && [ "$OVERWRITE" != "1" ]; then
    echo "ERROR: patched output exists: $PATCHED_MEGATRON_DIR" >&2
    echo "Set OVERWRITE=1 if replacing it is intentional." >&2
    exit 4
fi
rm -rf "$ROUNDTRIP_HF_DIR"

export SCRIPT_DIR HF_DIR MEGATRON_IN_DIR PATCHED_MEGATRON_DIR ROUNDTRIP_HF_DIR WORK
export OVERWRITE LOGITS MEGATRON_LM_SWISSAI_DIR

uenv run "$INIT_UENV_IMAGE" --view=default -- bash <<'INNER'
set -euo pipefail
export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"

echo "=== patch Megatron checkpoint ==="
patch_args=(
    --hf-dir "$HF_DIR"
    --megatron-dir "$MEGATRON_IN_DIR"
    --out-dir "$PATCHED_MEGATRON_DIR"
    --max-current-diff 1e-3
)
if [ "$OVERWRITE" = "1" ]; then
    patch_args+=(--overwrite)
fi
python3 "$SCRIPT_DIR/patch_apertus_extras.py" "${patch_args[@]}" | tee "$WORK/patch_apertus_extras.log"

echo
echo "=== Megatron patched -> HF ==="
cd "$MEGATRON_LM_SWISSAI_DIR"
python3 tools/checkpoint/convert.py --model-type GPT \
    --loader core --saver swissai_hf \
    --load-dir "$PATCHED_MEGATRON_DIR" --save-dir "$ROUNDTRIP_HF_DIR" \
    --hf-tokenizer "$HF_DIR" \
    --loader-transformer-impl transformer_engine \
    2>&1 | tee "$WORK/megatron_to_hf.log"

echo
echo "=== verify HF roundtrip ==="
verify_args=(
    --reference-hf-dir "$HF_DIR"
    --roundtrip-hf-dir "$ROUNDTRIP_HF_DIR"
    --tokenizer-dir "$HF_DIR"
    --require-r17-match
    --output-json "$WORK/verification.json"
)
if [ "$LOGITS" = "1" ]; then
    verify_args+=(--logits)
fi
python3 "$SCRIPT_DIR/verify_hf_roundtrip.py" "${verify_args[@]}" | tee "$WORK/verify_hf_roundtrip.log"
INNER

echo
echo "=== R17 patch roundtrip done ==="
date -u
echo "patched Megatron: $PATCHED_MEGATRON_DIR"
echo "roundtrip HF:     $ROUNDTRIP_HF_DIR"
echo "verification:     $WORK/verification.json"
