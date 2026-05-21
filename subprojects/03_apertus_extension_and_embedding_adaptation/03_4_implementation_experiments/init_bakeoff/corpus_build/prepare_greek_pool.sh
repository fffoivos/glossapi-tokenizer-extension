#!/usr/bin/env bash
# Build the post-Apertus-drop + post-internal-dedup Greek pool per the
# runbook at `03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`.
#
# Three-step path (order matters per the runbook):
#   1. Hard-exclude Apertus-overlap docs from the full nanochat parquets.
#   2. Replay nanochat internal dedup with `drop_intra_and_inter`.
#   3. Write a single $SELECTED parquet that downstream mix_builder.py
#      reads via per-source filter_values on source_dataset.
#
# Reviewer round-2 Blocker 3: the previous mix_builder.py applied
# Apertus-drop only on the HPLT source (`bulk.json:L23`) and skipped
# internal-dedup replay entirely — order-wrong AND incomplete. This
# script + bulk.json's `local_parquet: ${SELECTED}` entries close that
# gap so all six Greek source-categories share the same correctly-dedup'd
# pool.
#
# Run on Clariden `xfer` (CPU-only; needs the nanochat parquets + dedup
# metadata staged locally by pull_greek_corpus.sh).
#
# Usage:
#   bash prepare_greek_pool.sh
#
# Output:
#   $WORK/cpt/selected_after_apertus_and_internal_dedup.parquet

set -euo pipefail

# Roots — override via env if running from a different layout.
STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$STAGE_ROOT/nanochat}"
APERTUS_AUDIT_DIR="${APERTUS_AUDIT_DIR:-$STAGE_ROOT/apertus_overlap_overlay}"
WORK="${WORK:-$STAGE_ROOT}"

# Runbook-mandated paths
DEDUP_ROOT="${DEDUP_ROOT:-$NANOCHAT_DIR/dedup_metadata/wave2_20260426_builder_metadata_v2_latest_cleaner_20260507/builder_metadata}"
APERTUS_DROP="${APERTUS_DROP:-$APERTUS_AUDIT_DIR/artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet}"
SELECTED="${SELECTED:-$WORK/cpt/selected_after_apertus_and_internal_dedup.parquet}"

# Repo root — needs the glossapi_corpus_cli on PYTHONPATH
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"

echo "=== prepare_greek_pool.sh ==="
date -u
echo "STAGE_ROOT:    $STAGE_ROOT"
echo "NANOCHAT_DIR:  $NANOCHAT_DIR"
echo "DEDUP_ROOT:    $DEDUP_ROOT"
echo "APERTUS_DROP:  $APERTUS_DROP"
echo "SELECTED:      $SELECTED"
echo "REPO_ROOT:     $REPO_ROOT"
echo

# Sanity: required inputs exist
for required in "$NANOCHAT_DIR" "$DEDUP_ROOT" "$APERTUS_DROP"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: required input not found: $required" >&2
        echo "  Did you run pull_greek_corpus.sh first?" >&2
        exit 2
    fi
done

mkdir -p "$(dirname "$SELECTED")"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Runbook §"Build Fresh Deduped Pool"
python3 -m glossapi_corpus_cli.cli mix-prepare-selected-input \
    --output-root "$NANOCHAT_DIR" \
    --selected-input-path "$SELECTED" \
    --exclude-doc-keys-path "$APERTUS_DROP" \
    --dedup-metadata-root "$DEDUP_ROOT" \
    --dedup-action drop_intra_and_inter \
    --dedup-exact-stage strict_and_relaxed \
    --dedup-similarity-threshold 0.85 \
    --dedup-inter-dataset-policy share_aware

echo
echo "=== done ==="
date -u
ls -la "$SELECTED"
echo
echo "Next: export SELECTED=$SELECTED ; python3 mix_builder.py --recipe recipes/bulk.json ..."
