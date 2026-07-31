#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_grmmlu
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
cpt_require_runtime
if [[ -s "$DECONTAM_BINDING" ]]; then
  cpt_python "$CPT_DIR/dataset/freeze_decontamination_binding.py" \
    --queries "$DECONTAM_QUERIES" --benchmark-manifest "$DECONTAM_QUERY_MANIFEST" \
    --output "$DECONTAM_BINDING"
  exit 0
fi
[[ ! -e "$DECONTAM_DIR" ]] || { echo "ERROR: incomplete GreekMMLU freeze exists: $DECONTAM_DIR" >&2; exit 2; }
mkdir -p "$DECONTAM_DIR"
REGISTRY="$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json"
EVAL_DIR=$(dirname "$REGISTRY")
cpt_python "$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py" \
  --registry "$REGISTRY" --eval-dir "$EVAL_DIR" --benchmarks greekmmlu \
  --output-jsonl "$DECONTAM_QUERIES" --summary-json "$DECONTAM_QUERY_SUMMARY"
cpt_python "$REPO_ROOT/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/freeze_greekmmlu_queries.py" \
  --queries-jsonl "$DECONTAM_QUERIES" --output "$DECONTAM_QUERY_MANIFEST" \
  --dataset-revision 6a03aa06b68beb932fb75edff3a34e50b3674649 \
  --required-split test --registry "$REGISTRY" --builder-summary "$DECONTAM_QUERY_SUMMARY"
cpt_python "$CPT_DIR/dataset/freeze_decontamination_binding.py" \
  --queries "$DECONTAM_QUERIES" --benchmark-manifest "$DECONTAM_QUERY_MANIFEST" \
  --output "$DECONTAM_BINDING"
