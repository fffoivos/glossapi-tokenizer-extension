#!/usr/bin/env bash
# HAND-OFF DRIVER (planning artifact — the execution agent runs this; the planning agent does not submit).
# Submits one GreekMMLU eval per public peer base model in peer_models_greekmmlu.tsv, using the
# ESTABLISHED runner run_native_greek_mcq_eval.sbatch (dascim/GreekMMLU, full split, log-likelihood MCQ,
# tokenizer-agnostic -> fair across models). Results land as <out>/*_native_mcq_summary.csv, which
# analysis/collect_greekmmlu.py can aggregate next to our Apertus Vanilla/TD points.
#
#   DRY_RUN=1 bash submit_peer_models_greekmmlu.sh     # print the sbatch lines, submit nothing
#   bash submit_peer_models_greekmmlu.sh               # submit one 1-GPU job per model
#
# GATED models (Gemma-2-9B, Llama-3.1-8B): export HF_TOKEN (with the license accepted) before running;
# the token propagates via --export=ALL. Krikri + Qwen3 are open and need no token.
set -euo pipefail
E="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$E/run_native_greek_mcq_eval.sbatch"
MANIFEST="${MANIFEST:-$E/peer_models_greekmmlu.tsv}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval/peer_greekmmlu_$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
test -f "$SBATCH"; test -f "$MANIFEST"
echo "manifest=$MANIFEST  out_root=$OUT_ROOT  dry_run=$DRY_RUN"

while read -r label hf_id gated _rest; do
  [[ -z "${label:-}" || "$label" == \#* ]] && continue
  out="$OUT_ROOT/$label"
  cmd=(sbatch --job-name="gmmlu_${label}"
       --export="ALL,SCRIPT_DIR_OVERRIDE=$E,MODEL_SPEC=${label}=${hf_id},BENCHMARKS=greekmmlu,OUTPUT_DIR=$out"
       "$SBATCH")
  echo "+ $label  ($hf_id)  gated=$gated  -> $out"
  if [ "$DRY_RUN" = "1" ]; then printf '  DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'
  else "${cmd[@]}"; fi
done < "$MANIFEST"

echo "done. After completion: python analysis/collect_greekmmlu.py is for our run-tree; for these peer"
echo "evals read each <out>/*_native_mcq_summary.csv (greekmmlu rows) to get overall acc per model."
