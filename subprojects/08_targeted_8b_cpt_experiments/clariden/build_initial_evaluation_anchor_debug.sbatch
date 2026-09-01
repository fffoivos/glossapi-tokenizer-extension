#!/usr/bin/env bash
#SBATCH --job-name=t8b-freeze-anchor
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_EXPERIMENT:?set A or B}"
: "${TARGET8_VALIDATION_MANIFEST:?set frozen validation manifest}"
: "${TARGET8_PER_DOCUMENT_ROOT:?set exact per-document receipt root}"
: "${TARGET8_GREEKMMLU_RECEIPT:?set exact GreekMMLU receipt}"
: "${TARGET8_CHECKPOINT_RECEIPT:?set exact checkpoint receipt}"
: "${TARGET8_MODEL_EVIDENCE:?set initial-HF or exact-checkpoint export receipt}"
: "${TARGET8_ANCHOR_OUTPUT:?set new immutable anchor output directory}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "anchor freeze must run on debug" >&2; exit 2; }
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

iteration=0
[[ "$TARGET8_EXPERIMENT" == B ]] && iteration=9536
/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_initial_evaluation_anchor.py" \
  --experiment "$TARGET8_EXPERIMENT" --checkpoint-iteration "$iteration" \
  --validation-manifest "$TARGET8_VALIDATION_MANIFEST" \
  --per-document-root "$TARGET8_PER_DOCUMENT_ROOT" \
  --greekmmlu-receipt "$TARGET8_GREEKMMLU_RECEIPT" \
  --checkpoint-receipt "$TARGET8_CHECKPOINT_RECEIPT" \
  --model-evidence "$TARGET8_MODEL_EVIDENCE" \
  --output-dir "$TARGET8_ANCHOR_OUTPUT"
