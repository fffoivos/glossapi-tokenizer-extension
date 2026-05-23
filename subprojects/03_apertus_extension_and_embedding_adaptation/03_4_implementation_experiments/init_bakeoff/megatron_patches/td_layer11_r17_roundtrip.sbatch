#!/usr/bin/env bash
#SBATCH --job-name=td_r17_roundtrip
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
# Convert the selected full-token TD HF checkpoint to Megatron TP=2, patch
# Apertus extras, convert back to HF, and prove the roundtrip.

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/../bakeoff_training/_train_config_common.env"

HF_DIR="${HF_DIR:-/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z/layer11}"
RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_${SLURM_JOB_ID:-manual}}"
RAW_MEGATRON_DIR="${RAW_MEGATRON_DIR:-$RUN_ROOT/megatron_tp2_raw}"
PATCHED_MEGATRON_DIR="${PATCHED_MEGATRON_DIR:-$RUN_ROOT/megatron_tp2_r17patched}"
ROUNDTRIP_HF_DIR="${ROUNDTRIP_HF_DIR:-$RUN_ROOT/hf_roundtrip}"
WORK="${WORK:-/capstor/scratch/cscs/fffoivos/runs/td_r17_roundtrip_${SLURM_JOB_ID:-manual}}"
INIT_UENV_IMAGE="${INIT_UENV_IMAGE:-pytorch/v2.9.1:v2}"
TARGET_TENSOR_PARALLEL_SIZE="${TARGET_TENSOR_PARALLEL_SIZE:-2}"
TARGET_PIPELINE_PARALLEL_SIZE="${TARGET_PIPELINE_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
LOGITS="${LOGITS:-1}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/eval "$WORK" "$RUN_ROOT"

echo "=== td_layer11_r17_roundtrip.sbatch ==="
date -u
echo "host:                  $(hostname)"
echo "SCRIPT_DIR:            $SCRIPT_DIR"
echo "HF_DIR:                $HF_DIR"
echo "RUN_ROOT:              $RUN_ROOT"
echo "RAW_MEGATRON_DIR:      $RAW_MEGATRON_DIR"
echo "PATCHED_MEGATRON_DIR:  $PATCHED_MEGATRON_DIR"
echo "ROUNDTRIP_HF_DIR:      $ROUNDTRIP_HF_DIR"
echo "MEGATRON_DIR:          $MEGATRON_LM_SWISSAI_DIR"
echo "INIT_UENV_IMAGE:       $INIT_UENV_IMAGE"
echo "TARGET_TP:             $TARGET_TENSOR_PARALLEL_SIZE"
echo "TARGET_PP:             $TARGET_PIPELINE_PARALLEL_SIZE"
echo "OVERWRITE:             $OVERWRITE"
echo "LOGITS:                $LOGITS"
echo "WORK:                  $WORK"
echo

if [ ! -f "$HF_DIR/config.json" ]; then
    echo "ERROR: missing HF config: $HF_DIR/config.json" >&2
    exit 2
fi
for path in "$RAW_MEGATRON_DIR" "$PATCHED_MEGATRON_DIR" "$ROUNDTRIP_HF_DIR"; do
    if [ -e "$path" ] && [ "$OVERWRITE" != "1" ]; then
        echo "ERROR: output exists: $path" >&2
        echo "Set OVERWRITE=1 if replacing it is intentional." >&2
        exit 3
    fi
done
if [ "$OVERWRITE" = "1" ]; then
    rm -rf "$RAW_MEGATRON_DIR" "$PATCHED_MEGATRON_DIR" "$ROUNDTRIP_HF_DIR"
fi

bash "$SCRIPT_DIR/install.sh" "$MEGATRON_LM_SWISSAI_DIR"

export SCRIPT_DIR HF_DIR RUN_ROOT RAW_MEGATRON_DIR PATCHED_MEGATRON_DIR ROUNDTRIP_HF_DIR WORK
export INIT_UENV_IMAGE TARGET_TENSOR_PARALLEL_SIZE TARGET_PIPELINE_PARALLEL_SIZE
export OVERWRITE LOGITS MEGATRON_LM_SWISSAI_DIR

uenv run "$INIT_UENV_IMAGE" --view=default -- bash <<'INNER'
set -euo pipefail
export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"

echo "=== HF -> Megatron TP=${TARGET_TENSOR_PARALLEL_SIZE} ==="
cd "$MEGATRON_LM_SWISSAI_DIR"
git rev-parse HEAD | tee "$WORK/megatron_commit.txt"
python3 tools/checkpoint/convert.py --model-type GPT \
    --loader apertus_hf --saver core \
    --load-dir "$HF_DIR" --save-dir "$RAW_MEGATRON_DIR" \
    --tokenizer-model "$HF_DIR" --bf16 \
    --loader-transformer-impl transformer_engine \
    --target-tensor-parallel-size "$TARGET_TENSOR_PARALLEL_SIZE" \
    --target-pipeline-parallel-size "$TARGET_PIPELINE_PARALLEL_SIZE" \
    2>&1 | tee "$WORK/hf_to_megatron.log"

if [ -d "$RAW_MEGATRON_DIR/iter_0000000" ]; then
    mv "$RAW_MEGATRON_DIR/iter_0000000" "$RAW_MEGATRON_DIR/release"
fi
if [ ! -d "$RAW_MEGATRON_DIR/release" ]; then
    echo "ERROR: expected raw Megatron release dir: $RAW_MEGATRON_DIR/release" >&2
    exit 4
fi
echo release > "$RAW_MEGATRON_DIR/latest_checkpointed_iteration.txt"

echo
echo "=== patch Apertus extras ==="
python3 "$SCRIPT_DIR/patch_apertus_extras.py" \
    --hf-dir "$HF_DIR" \
    --megatron-dir "$RAW_MEGATRON_DIR" \
    --out-dir "$PATCHED_MEGATRON_DIR" \
    --max-current-diff 1e-3 \
    --overwrite \
    2>&1 | tee "$WORK/patch_apertus_extras.log"

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
python3 "$SCRIPT_DIR/verify_hf_roundtrip.py" "${verify_args[@]}" \
    2>&1 | tee "$WORK/verify_hf_roundtrip.log"
INNER

cat > "$RUN_ROOT/td_r17_roundtrip_manifest.json" <<MANIFEST
{
  "hf_dir": "$HF_DIR",
  "raw_megatron_dir": "$RAW_MEGATRON_DIR",
  "patched_megatron_dir": "$PATCHED_MEGATRON_DIR",
  "roundtrip_hf_dir": "$ROUNDTRIP_HF_DIR",
  "work": "$WORK",
  "verification_json": "$WORK/verification.json",
  "target_tensor_parallel_size": $TARGET_TENSOR_PARALLEL_SIZE,
  "target_pipeline_parallel_size": $TARGET_PIPELINE_PARALLEL_SIZE,
  "uenv_image": "$INIT_UENV_IMAGE",
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST

echo
echo "=== TD R17 roundtrip done ==="
date -u
echo "raw Megatron:     $RAW_MEGATRON_DIR"
echo "patched Megatron: $PATCHED_MEGATRON_DIR"
echo "roundtrip HF:     $ROUNDTRIP_HF_DIR"
echo "verification:     $WORK/verification.json"
