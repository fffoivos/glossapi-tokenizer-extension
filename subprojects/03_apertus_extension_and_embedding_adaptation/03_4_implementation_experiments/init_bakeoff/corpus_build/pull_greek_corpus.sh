#!/usr/bin/env bash
# Pull GlossAPI Greek nanochat + dedup metadata + Apertus-overlap drop overlay.
#
# REQUIRES HF_TOKEN: the nanochat dataset is gated.
#   Either set HF_TOKEN env var OR run `huggingface-cli login` on Clariden first.
#
# Fix from round-3 Clariden execution: ONE --include with multiple patterns
# (argparse nargs="*" makes additional --include flags override prior ones).

set -euo pipefail
STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
NANOCHAT_DIR="$STAGE_ROOT/nanochat"
OVERLAY_DIR="$STAGE_ROOT/apertus_overlap_overlay"
mkdir -p "$NANOCHAT_DIR" "$OVERLAY_DIR"
echo "=== pull_greek_corpus.sh ==="; date -u
echo "stage root: $STAGE_ROOT"

if [ -z "${HF_TOKEN:-}" ] && [ ! -f ~/.cache/huggingface/token ]; then
    echo "ERROR: HF_TOKEN unset and no ~/.cache/huggingface/token. Authenticate first:" >&2
    echo "  huggingface-cli login" >&2
    exit 2
fi

echo
echo "=== 1. nanochat Greek pretraining corpus (gated) ==="
huggingface-cli download fffoivos/glossapi-greek-nanochat-pretraining-dataset \
    --repo-type dataset \
    --local-dir "$NANOCHAT_DIR" \
    --include "data/*.parquet" \
              "dedup_metadata/latest.json" \
              "dedup_metadata/wave2_20260426_builder_metadata_v2_latest_cleaner_20260507/builder_metadata/*" \
              "README.md"

echo
echo "=== 2. Apertus-overlap drop overlay (~few MB) ==="
huggingface-cli download fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z \
    --repo-type dataset \
    --local-dir "$OVERLAY_DIR" \
    --include "artifacts/dedup_20260519T010924Z/cpt_final_overlay/*.parquet" \
              "artifacts/dedup_20260519T010924Z/REPORT*.md"

echo
echo "=== sanity check ==="
du -sh "$NANOCHAT_DIR" "$OVERLAY_DIR" 2>/dev/null
echo "nanochat parquets:"; find "$NANOCHAT_DIR" -name "*.parquet" | wc -l
echo "overlay parquets:"; find "$OVERLAY_DIR" -name "*.parquet" | head
echo "✓ Greek corpus + overlay staged at $STAGE_ROOT"
