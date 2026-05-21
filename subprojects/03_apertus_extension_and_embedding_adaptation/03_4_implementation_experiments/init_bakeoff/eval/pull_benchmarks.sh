#!/usr/bin/env bash
# Stage all benchmark datasets and the eval harness onto Clariden iopsstor.
#
# Per init_bakeoff/eval/EVAL_RECIPE.md: three groups of evals:
#   Group 1: retention (ARC, HellaSwag, WinoGrande, PIQA, MMLU, HumanEval, GSM8K, XNLI, XCOPA)
#   Group 2: Greek (ILSP suite + Belebele Greek)
#   Group 3: safety (HarmBench, PolyglotToxicity, AttaQ Greek, Jailbreak-StrongReject)
#
# Plus: clone swiss-ai/lm-evaluation-harness (Apertus team's fork; primary harness)
#       and EleutherAI/lm-evaluation-harness (upstream, fallback)
#
# Run on Clariden login node. Total wall ~30-60 min depending on HF bandwidth.
# Storage footprint: <5 GB for benchmarks + ~few hundred MB for the harness repos.
#
# Usage:
#   bash pull_benchmarks.sh

set -euo pipefail

STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos}"
BENCH_DIR="$STAGE_ROOT/benchmarks"
CODE_DIR="$STAGE_ROOT/code/eval"

mkdir -p \
  "$BENCH_DIR/retention" \
  "$BENCH_DIR/ilsp_greek" \
  "$BENCH_DIR/other_greek" \
  "$BENCH_DIR/safety" \
  "$CODE_DIR"

echo "=== pull_benchmarks.sh ==="
echo "stage root: $STAGE_ROOT"
echo

# hf_transfer not in pytorch uenv; using plain HTTP

# --- helper: pull-or-warn (don't abort on per-dataset failure) ---
pull() {
    local repo="$1"; local repo_type="${2:-dataset}"; local out="$3"; shift 3
    echo "--- $repo (type=$repo_type) → $out ---"
    huggingface-cli download "$repo" \
        --repo-type "$repo_type" \
        --local-dir "$out" \
        "$@" || echo "  WARN: $repo failed (may be gated or unavailable); continuing"
}

# === Group 1: retention (Apertus-reported benchmarks) ===
echo
echo "=== Group 1: retention benchmarks ==="
pull allenai/ai2_arc        dataset "$BENCH_DIR/retention/ai2_arc"
pull Rowan/hellaswag        dataset "$BENCH_DIR/retention/hellaswag"
pull winogrande              dataset "$BENCH_DIR/retention/winogrande"
pull piqa                    dataset "$BENCH_DIR/retention/piqa"
pull cais/mmlu               dataset "$BENCH_DIR/retention/mmlu"
pull openai/openai_humaneval dataset "$BENCH_DIR/retention/humaneval"
pull openai/gsm8k            dataset "$BENCH_DIR/retention/gsm8k"
pull facebook/xnli           dataset "$BENCH_DIR/retention/xnli"
pull cambridgeltl/xcopa      dataset "$BENCH_DIR/retention/xcopa"

# === Group 2: Greek (ILSP suite + Belebele Greek + native GreekMMLU) ===
echo
echo "=== Group 2: Greek benchmarks (ILSP suite + Belebele) ==="
for ds in arc_greek hellaswag_greek winogrande_greek mmlu_greek MMLU-Pro_greek \
          truthful_qa_greek mgsm_greek medical_mcqa_greek ifeval_greek \
          mt-bench-greek m-ArenaHard_greek; do
    pull "ilsp/$ds" dataset "$BENCH_DIR/ilsp_greek/$ds"
done

# Belebele has all languages in one repo; we pull the Greek subset
pull facebook/belebele dataset "$BENCH_DIR/other_greek/belebele" \
    --include 'ell_Grek/*' --include 'data/ell_Grek/*' --include 'README.md'

# Native-sourced GreekMMLU (Zhang 2026); may be gated
pull dascim/GreekMMLU dataset "$BENCH_DIR/other_greek/dascim_GreekMMLU"

# === Group 3: safety ===
echo
echo "=== Group 3: safety benchmarks ==="
pull swiss-ai/harmbench              dataset "$BENCH_DIR/safety/harmbench"
pull swiss-ai/polyglotoxicityprompts dataset "$BENCH_DIR/safety/polyglotoxicityprompts"
pull swiss-ai/realtoxicityprompts    dataset "$BENCH_DIR/safety/realtoxicityprompts"
pull ilsp/attaq_greek                dataset "$BENCH_DIR/safety/ilsp_attaq_greek"
pull ilsp/Jailbreak-StrongReject-el  dataset "$BENCH_DIR/safety/ilsp_jailbreak_el"

# === Eval harness repos ===
echo
echo "=== eval harness repos ==="
# swiss-ai fork (primary)
if [ ! -d "$CODE_DIR/lm-evaluation-harness-swissai" ]; then
    git clone https://github.com/swiss-ai/lm-evaluation-harness.git \
        "$CODE_DIR/lm-evaluation-harness-swissai"
else
    echo "  $CODE_DIR/lm-evaluation-harness-swissai already exists; pulling..."
    (cd "$CODE_DIR/lm-evaluation-harness-swissai" && git pull --ff-only) || true
fi
# Apertus team's eval invocations (literal task lists Apertus reported)
if [ ! -d "$CODE_DIR/evals-apertus" ]; then
    git clone https://github.com/swiss-ai/evals.git "$CODE_DIR/evals-apertus" || \
        echo "  WARN: swiss-ai/evals clone failed"
fi
# Upstream EleutherAI lm-eval-harness as fallback / for task definitions
if [ ! -d "$CODE_DIR/lm-evaluation-harness-eleuther" ]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git \
        "$CODE_DIR/lm-evaluation-harness-eleuther"
fi
# Inspect AI for custom open-ended evals
echo "  pip install inspect-ai (if not already installed)"
pip install --user inspect-ai 2>&1 | tail -3 || echo "  inspect-ai install may have failed; revisit"

# === sanity check ===
echo
echo "=== sanity check ==="
echo "Benchmarks:"
du -sh "$BENCH_DIR"/* | sort -hr | head -30
echo
echo "Eval harness repos:"
du -sh "$CODE_DIR"/* | head
echo
echo "✓ Benchmarks staged at $BENCH_DIR"
echo "✓ Eval harness staged at $CODE_DIR"
echo "Next: bash run_apertus_baseline.sh (V4 baseline) or run_bakeoff_arm_eval.sh <arm-ckpt-dir>"
