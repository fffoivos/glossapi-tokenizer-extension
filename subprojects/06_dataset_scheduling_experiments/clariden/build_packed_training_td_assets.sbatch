#!/usr/bin/env bash
#SBATCH --job-name=mini_td_assets
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_STAGE_ROOT:?set frozen packed-corpus stage root}"
: "${MINI_BASE_MODEL_DIR:?set frozen base Mini model/tokenizer}"
: "${MINI_OVERLAY_TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${TD_ASSET_OUTPUT_DIR:?set new TD asset output directory}"
TD_ASSET_PYTHON=${TD_ASSET_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/td_packed_assets_py311/bin/python}
PYTHON_BOOTSTRAP=${PYTHON_BOOTSTRAP:-/usr/bin/python3.11}
TD_HPLT_SEQUENCES=${TD_HPLT_SEQUENCES:-350000}
TD_NON_HPLT_SEQUENCES=${TD_NON_HPLT_SEQUENCES:-150000}
TD_OLD_GREEK_SEQUENCES=${TD_OLD_GREEK_SEQUENCES:--1}
TD_CANDIDATE_SNIPPETS=${TD_CANDIDATE_SNIPPETS:-64}
TD_MIN_OVERALL_ENOUGH25_FRACTION=${TD_MIN_OVERALL_ENOUGH25_FRACTION:-0.90}
TD_MIN_MODERN_ENOUGH25_FRACTION=${TD_MIN_MODERN_ENOUGH25_FRACTION:-0.90}
TD_MIN_POLYTONIC_ENOUGH25_COUNT=${TD_MIN_POLYTONIC_ENOUGH25_COUNT:-256}

[[ ! -e "$TD_ASSET_OUTPUT_DIR" ]] || {
  echo "refusing to reuse TD asset output: $TD_ASSET_OUTPUT_DIR" >&2
  exit 2
}
if [[ ! -x "$TD_ASSET_PYTHON" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$(dirname "$(dirname "$TD_ASSET_PYTHON")")"
fi
if ! "$TD_ASSET_PYTHON" -c 'import numpy,tokenizers' 2>/dev/null; then
  "$TD_ASSET_PYTHON" -m pip install --disable-pip-version-check \
    'numpy==2.3.2' 'tokenizers==0.22.1'
fi

PACKING_PLAN="$MINI_STAGE_ROOT/packing_plan_source_local.json"
PACKED_RECEIPT="$MINI_STAGE_ROOT/packed_corpus_source_local_receipt.json"
INPUT_RECEIPT="$MINI_STAGE_ROOT/input_receipt.json"
HELDOUT_MANIFEST="$MINI_STAGE_ROOT/heldouts/heldout_manifest.json"
readarray -t heldouts < <("$TD_ASSET_PYTHON" - "$HELDOUT_MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
paths={row['name']:row['output']['path'] for row in d['sets']}
for name in ('hplt','non_hplt','historical_polytonic'):
    print(paths[name])
PY
)
HPLT_EVAL=${heldouts[0]}
NON_HPLT_EVAL=${heldouts[1]}
POLYTONIC_EVAL=${heldouts[2]}

"$TD_ASSET_PYTHON" -u \
  "$MINI_CODE_BUNDLE/initialization/build_td_assets_from_packed_training.py" \
  --packing-plan "$PACKING_PLAN" \
  --packed-receipt "$PACKED_RECEIPT" \
  --input-receipt "$INPUT_RECEIPT" \
  --base-tokenizer "$MINI_BASE_MODEL_DIR" \
  --target-tokenizer "$MINI_OVERLAY_TOKENIZER_DIR" \
  --hplt-eval-jsonl "$HPLT_EVAL" \
  --non-hplt-eval-jsonl "$NON_HPLT_EVAL" \
  --polytonic-eval-jsonl "$POLYTONIC_EVAL" \
  --output-dir "$TD_ASSET_OUTPUT_DIR" \
  --hplt-sequences "$TD_HPLT_SEQUENCES" \
  --non-hplt-sequences "$TD_NON_HPLT_SEQUENCES" \
  --old-greek-sequences "$TD_OLD_GREEK_SEQUENCES" \
  --candidate-snippets-per-token "$TD_CANDIDATE_SNIPPETS" \
  --min-overall-enough25-fraction "$TD_MIN_OVERALL_ENOUGH25_FRACTION" \
  --min-modern-enough25-fraction "$TD_MIN_MODERN_ENOUGH25_FRACTION" \
  --min-polytonic-enough25-count "$TD_MIN_POLYTONIC_ENOUGH25_COUNT"

"$TD_ASSET_PYTHON" \
  "$MINI_CODE_BUNDLE/initialization/normalize_td_coverage_geometry.py" \
  --coverage-jsonl "$TD_ASSET_OUTPUT_DIR/td_coverage_prepass.jsonl" \
  --target-tokenizer "$MINI_OVERLAY_TOKENIZER_DIR" \
  --output-jsonl "$TD_ASSET_OUTPUT_DIR/td_coverage_exact_geometry.jsonl" \
  --output-manifest "$TD_ASSET_OUTPUT_DIR/td_coverage_exact_geometry.manifest.json"

sha256sum \
  "$TD_ASSET_OUTPUT_DIR/td_coverage_summary.json" \
  "$TD_ASSET_OUTPUT_DIR/td_coverage_exact_geometry.jsonl" \
  "$TD_ASSET_OUTPUT_DIR/td_snippet_index/snippets.jsonl"
