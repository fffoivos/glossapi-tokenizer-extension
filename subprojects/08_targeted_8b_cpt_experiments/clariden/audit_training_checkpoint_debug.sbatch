#!/usr/bin/env bash
#SBATCH --job-name=h2g-ckpt-audit
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_MODEL_SCALE:?set 8b or 1p5b}"
: "${H2G_SOURCE_PHASE:?set checkpoint-producing phase}"
: "${H2G_CHECKPOINT_UPDATE:?set checkpoint update}"
: "${H2G_CHECKPOINT_ROOT:?set exact iter checkpoint root}"
: "${H2G_SOURCE_PHASE_CACHE_RECEIPT:?set source phase cache receipt}"
: "${H2G_SEGMENT_PREFLIGHT:?set producing segment preflight receipt}"
: "${H2G_TRAINING_LOG:?set producing segment training log}"
: "${H2G_CHECKPOINT_AUDIT_OUTPUT:?set immutable audit output}"
: "${H2G_MEGATRON_DIR:?set pinned patched Megatron root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "checkpoint audit requires debug" >&2; exit 2; }

/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
[[ ! -e "$H2G_CHECKPOINT_AUDIT_OUTPUT" ]] || { echo "checkpoint audit output exists" >&2; exit 2; }

export H2G_CODE_ROOT H2G_CODE_RECEIPT
uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$H2G_MEGATRON_DIR:$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts" \
  python3 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/audit_training_checkpoint.py" \
    --scale "$H2G_MODEL_SCALE" --source-phase "$H2G_SOURCE_PHASE" \
    --update "$H2G_CHECKPOINT_UPDATE" --checkpoint-root "$H2G_CHECKPOINT_ROOT" \
    --source-phase-cache-receipt "$H2G_SOURCE_PHASE_CACHE_RECEIPT" \
    --segment-preflight "$H2G_SEGMENT_PREFLIGHT" --training-log "$H2G_TRAINING_LOG" \
    --output "$H2G_CHECKPOINT_AUDIT_OUTPUT"
