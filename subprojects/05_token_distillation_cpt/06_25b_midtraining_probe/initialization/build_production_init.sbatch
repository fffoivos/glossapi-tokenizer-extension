#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_poly_init
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=06:00:00

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${BASE_TD_INIT_HF:?set BASE_TD_INIT_HF to the uncpt TokenDistil-Init HF directory}"
: "${TOKENIZER_DIR:?set TOKENIZER_DIR to the frozen 148992 tokenizer}"
: "${TD_COVERAGE_JSONL:?set TD_COVERAGE_JSONL}"
: "${TD_SNIPPETS_JSONL:?set TD_SNIPPETS_JSONL}"
: "${TD_TOKEN_IDS_FILE:?set TD_TOKEN_IDS_FILE}"
: "${CALIBRATION_JSONL:?set CALIBRATION_JSONL}"
: "${OUTPUT_ROOT:?set a new OUTPUT_ROOT}"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "ERROR: output exists: $OUTPUT_ROOT" >&2; exit 2; }

PROBE="$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe"
TD="$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation"
VERIFY="$REPO_ROOT/subprojects/05_token_distillation_cpt/06_25b_midtraining_probe/initialization/verify_production_init.py"
INIT="$OUTPUT_ROOT/merge_chain"
DISTILLED="$OUTPUT_ROOT/distilled"
FINAL="$OUTPUT_ROOT/production"
mkdir -p "$OUTPUT_ROOT"

export TOKENIZERS_PARALLELISM=false PYTHONPATH="$TD/external/token-distillation:${PYTHONPATH:-}"
uenv run pytorch/v2.9.1:v2 --view=default -- bash <<INNER
set -euo pipefail
python3 '$PROBE/build_incremental_checkpoint.py' \
  --base-model '$BASE_TD_INIT_HF' --candidate-tokenizer '$TOKENIZER_DIR' \
  --output-dir '$INIT' --expected-base-vocab 148480 --modern-id-start 131072 --device cuda
python3 '$TD/train_retok_td.py' \
  --student-model '$INIT' --base-tokenizer '$BASE_TD_INIT_HF' --student-tokenizer '$TOKENIZER_DIR' \
  --coverage-jsonl '$TD_COVERAGE_JSONL' --snippets-jsonl '$TD_SNIPPETS_JSONL' \
  --token-ids-file '$TD_TOKEN_IDS_FILE' --output-dir '$DISTILLED' \
  --base-vocab-size 148480 --new-id-start 148480 --new-id-end 148992 \
  --snippets-per-token 25 --min-accepted-snippets-per-token 25 \
  --min-trained-token-fraction 0.90 --epochs 1 --batch-size 8 \
  --learning-rate 1e-4 --target-layer 11 --dtype bfloat16 --device cuda
python3 '$PROBE/calibrate_new_output_rows.py' \
  --model '$DISTILLED' --calibration-jsonl '$CALIBRATION_JSONL' --output-dir '$FINAL' \
  --base-vocab-size 148480 --new-vocab-size 148992 --steps 400 --seq-length 512 \
  --learning-rate 2e-4 --max-grad-norm 1.0 --seed 20260729 --device cuda
cp '$INIT/incremental_init_summary.json' '$FINAL/incremental_init_summary.json'
cp '$DISTILLED/retok_td_manifest.json' '$FINAL/retok_td_manifest.json'
python3 '$VERIFY' --base-model '$BASE_TD_INIT_HF' --final-model '$FINAL' \
  --output '$OUTPUT_ROOT/production_init_verification.json'
INNER

echo "production HF init: $FINAL"
echo "verification: $OUTPUT_ROOT/production_init_verification.json"
