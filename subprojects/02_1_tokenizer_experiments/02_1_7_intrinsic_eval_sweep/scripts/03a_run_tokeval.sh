#!/usr/bin/env bash
# Run TokEval per the §6.4 job matrix.
#
# Jobs:
#   #1 — TFG on apertus55 (multilingual, single run; emits one TFG number per
#        tokenizer)
#   #2 — Per-language fertility / compression / utilization / UTF-8 / Rényi
#        on apertus55 (per-language)
#   #3 — Per-language Greek metrics on ell_Grek FLORES+ slice
#
# All three jobs read the same tokenizer config (15 tokenizers).
# Output structure: tokeval_raw/<job_id>/<tokenizer_name>/<...>
#
# Parallelism: TokEval runs all tokenizers in one process per invocation.
# Job 1, 2, 3 themselves are independent; we run them serially because
# TokEval already saturates within a single job. If under-utilized,
# Jobs 2 and 3 can be backgrounded with `&`.

set -euo pipefail

REPO=/home/foivos/Projects/glossapi-tokenizer-extension
SSP="$REPO/subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
TOKEVAL="$SSP/vendor/tokenizer-intrinsic-evals"
VENV="$TOKEVAL/.venv/bin"
OUT="$SSP/artifacts/tokeval_raw"
mkdir -p "$OUT/job1_tfg_apertus55" "$OUT/job2_perlang_apertus55" "$OUT/job3_greek_only"

cd "$TOKEVAL"

echo "=========================================================="
echo " Job #1 — TFG on apertus55 (multilingual)"
echo "=========================================================="
"$VENV/tokenizer-analysis" \
    --tokenizer-config "$SSP/configs/cutoff_sweep_tokenizers.json" \
    --language-config "$SSP/configs/apertus55_lang_config.json" \
    --measurement-config "$TOKEVAL/configs/text_measurement_config_lines.json" \
    --output-dir "$OUT/job1_tfg_apertus55" \
    --save-full-results \
    --no-plots \
    --verbose 2>&1 | tail -20

echo
echo "=========================================================="
echo " Job #2 — per-language fertility / compression / utilization on apertus55"
echo "=========================================================="
"$VENV/tokenizer-analysis" \
    --tokenizer-config "$SSP/configs/cutoff_sweep_tokenizers.json" \
    --language-config "$SSP/configs/apertus55_lang_config.json" \
    --measurement-config "$TOKEVAL/configs/text_measurement_config_words_hf.json" \
    --output-dir "$OUT/job2_perlang_apertus55" \
    --save-full-results \
    --per-language-plots \
    --no-global-lines \
    --verbose 2>&1 | tail -20

echo
echo "=========================================================="
echo " Job #3 — Greek-only deep dive on FLORES+ ell_Grek"
echo "=========================================================="
"$VENV/tokenizer-analysis" \
    --tokenizer-config "$SSP/configs/cutoff_sweep_tokenizers.json" \
    --language-config "$SSP/configs/greek_only_lang_config.json" \
    --measurement-config "$TOKEVAL/configs/text_measurement_config_words_hf.json" \
    --output-dir "$OUT/job3_greek_only" \
    --save-full-results \
    --no-plots \
    --verbose 2>&1 | tail -20

echo
echo "TokEval done. Outputs:"
ls -la "$OUT"/*/  | head -30
