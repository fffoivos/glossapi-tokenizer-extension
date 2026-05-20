#!/usr/bin/env bash
# Pull our Greek nanochat corpus + the Apertus-overlap drop overlay onto Clariden iopsstor.
#
# Per cpt_plan.md v0.7 section 2: "Old Apertus Greek pretraining data is not replayed."
# Operationalized via the dedup-audit overlay at fffoivos/apertus-c3-dedup-audit-...,
# which lists ~2.22M doc_keys to drop from the nanochat pool. The mix_builder reads
# this overlay and filters the nanochat stream by doc_key.
#
# Run on Clariden login node (no slurm). ~30-60 min depending on HF bandwidth.
# Storage footprint: ~100-150 GB on iopsstor for the full nanochat release + small overlay.
#
# Usage:
#   bash pull_greek_corpus.sh

set -euo pipefail

STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
NANOCHAT_DIR="$STAGE_ROOT/nanochat"
OVERLAY_DIR="$STAGE_ROOT/apertus_overlap_overlay"

mkdir -p "$NANOCHAT_DIR" "$OVERLAY_DIR"

echo "=== pull_greek_corpus.sh ==="
echo "stage root: $STAGE_ROOT"
echo "free space: $(df -h "$STAGE_ROOT" | tail -1)"
echo

# Sanity: are we logged into HF?
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARN: HF_TOKEN not set. Anonymous downloads work for public datasets but may rate-limit."
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "=== 1. nanochat Greek pretraining corpus ==="
# Pulls the full release. Large (~100 GB compressed). The mix_builder streams from
# this locally via load_dataset() with cache_dir pointing here, no Internet during
# the build itself.
huggingface-cli download \
    fffoivos/glossapi-greek-nanochat-pretraining-dataset \
    --repo-type dataset \
    --local-dir "$NANOCHAT_DIR" \
    --include 'data/*.parquet' \
    --include 'dedup_metadata/latest.json' \
    --include 'README.md'

echo
echo "=== 2. Apertus-overlap drop overlay (~few MB) ==="
huggingface-cli download \
    fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z \
    --repo-type dataset \
    --local-dir "$OVERLAY_DIR" \
    --include 'artifacts/dedup_20260519T010924Z/cpt_final_overlay/*.parquet' \
    --include 'artifacts/dedup_20260519T010924Z/REPORT*.md'

echo
echo "=== sanity check ==="
echo "nanochat:"
du -sh "$NANOCHAT_DIR" 2>/dev/null || true
ls "$NANOCHAT_DIR/data/" | head -10
echo
echo "overlay:"
du -sh "$OVERLAY_DIR" 2>/dev/null || true
find "$OVERLAY_DIR" -name '*.parquet' | head -5

echo
echo "✓ Greek corpus + overlay staged at $STAGE_ROOT"
echo "Next: bash pull_replay_datasets.sh ; then run mix_builder.py to assemble the bulk JSON-lines stream."
