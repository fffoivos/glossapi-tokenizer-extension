#!/usr/bin/env bash
# Run the TokEval layer used by the C3 cutoff sweep on polytonic variants.

set -euo pipefail

REPO=${REPO:-/home/foivos/Projects/glossapi-tokenizer-extension}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the polytonic run directory}
PYTHON_BIN=${PYTHON_BIN:-/home/foivos/venvs/glossapi-corpus-clean/bin/python}
SSP="$REPO/subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
TOKEVAL_ROOT=${TOKEVAL_ROOT:-"$SSP/vendor/tokenizer-intrinsic-evals"}
CONFIG_DIR=${CONFIG_DIR:-"$RUN_ROOT/tokeval_configs"}
OUT=${OUT:-"$RUN_ROOT/tokeval_raw"}

TOKENIZER_CONFIG="$CONFIG_DIR/polytonic_tokenizers.json"
APERTUS55_CONFIG="$CONFIG_DIR/apertus55_lang_config.json"
GREEK_CONFIG="$CONFIG_DIR/greek_only_lang_config.json"

mkdir -p "$OUT/job1_tfg_apertus55" "$OUT/job2_perlang_apertus55_words" "$OUT/job3_greek_only_words"
export PYTHONPATH="$TOKEVAL_ROOT:${PYTHONPATH:-}"

cd "$TOKEVAL_ROOT"

echo "=========================================================="
echo " Job #1 - TFG on Apertus-55 proxy (lines config)"
echo "=========================================================="
"$PYTHON_BIN" -m tokenizer_analysis.cli.run_analysis \
  --tokenizer-config "$TOKENIZER_CONFIG" \
  --language-config "$APERTUS55_CONFIG" \
  --measurement-config "$TOKEVAL_ROOT/configs/text_measurement_config_lines.json" \
  --output-dir "$OUT/job1_tfg_apertus55" \
  --save-full-results \
  --no-plots \
  --verbose

echo
echo "=========================================================="
echo " Job #2 - per-language Apertus-55 metrics (words config)"
echo "=========================================================="
"$PYTHON_BIN" -m tokenizer_analysis.cli.run_analysis \
  --tokenizer-config "$TOKENIZER_CONFIG" \
  --language-config "$APERTUS55_CONFIG" \
  --measurement-config "$TOKEVAL_ROOT/configs/text_measurement_config_words_hf.json" \
  --output-dir "$OUT/job2_perlang_apertus55_words" \
  --save-full-results \
  --per-language-plots \
  --no-global-lines \
  --verbose

echo
echo "=========================================================="
echo " Job #3 - Greek-only FLORES+ deep dive (words config)"
echo "=========================================================="
"$PYTHON_BIN" -m tokenizer_analysis.cli.run_analysis \
  --tokenizer-config "$TOKENIZER_CONFIG" \
  --language-config "$GREEK_CONFIG" \
  --measurement-config "$TOKEVAL_ROOT/configs/text_measurement_config_words_hf.json" \
  --output-dir "$OUT/job3_greek_only_words" \
  --save-full-results \
  --no-plots \
  --verbose

echo
echo "TokEval done under $OUT"
