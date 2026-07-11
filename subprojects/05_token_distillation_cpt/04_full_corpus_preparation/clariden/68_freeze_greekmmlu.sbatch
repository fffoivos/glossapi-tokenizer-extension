#!/usr/bin/env bash
#SBATCH --job-name=cpt4_greekmmlu
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "GreekMMLU query build and freeze"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_init_pipeline_run
phase04_stage_begin "59-greekmmlu-freeze"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0

REGISTRY="$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json"
BUILDER="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py"
FREEZER="$PHASE04_DIR/scripts/freeze_greekmmlu_queries.py"
phase04_require_file "$REGISTRY"
phase04_require_file "$BUILDER"
phase04_require_file "$FREEZER"
phase04_stage_add_input greek_benchmark_registry "$REGISTRY"

REVISION=$(
    "$RUNTIME_VENV/bin/python" - "$REGISTRY" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    registry = json.load(handle)
rows = [row for row in registry["benchmarks"] if row.get("id") == "greekmmlu"]
if len(rows) != 1:
    raise SystemExit("registry must contain exactly one GreekMMLU entry")
print(rows[0]["revision"])
PY
)
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf-cache/greekmmlu}"
mkdir -p "$HF_HOME"
if [[ ! -s "$PHASE04_STAGE_DIR/builder_summary.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$BUILDER" \
        --registry "$REGISTRY" \
        --benchmarks greekmmlu \
        --output-jsonl "$PHASE04_STAGE_DIR/queries.jsonl" \
        --summary-json "$PHASE04_STAGE_DIR/builder_summary.json"
fi
if [[ ! -s "$PHASE04_STAGE_DIR/query_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$FREEZER" \
        --queries-jsonl "$PHASE04_STAGE_DIR/queries.jsonl" \
        --output "$PHASE04_STAGE_DIR/query_manifest.json" \
        --dataset-revision "$REVISION" \
        --required-split test \
        --registry "$REGISTRY" \
        --builder-summary "$PHASE04_STAGE_DIR/builder_summary.json"
fi

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/queries.jsonl" \
    "$PHASE04_STAGE_DIR/builder_summary.json" \
    "$PHASE04_STAGE_DIR/query_manifest.json"
