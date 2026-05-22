#!/usr/bin/env bash
# Operationally satisfy V9 — NFC-normalize the pulled corpus parquets in place.
#
# Per cpt_plan.md v0.7 §8 I1 + V9: Apertus's tokenizer has `normalizer: null`,
# so pre-tokenization NFC at the corpus level is required. Greek nanochat
# delivers NFC upstream (HPLT 500/500 sample-verified; finepdfs-edu 0.07% NFD
# leak remediated). Replay datasets (FineWeb-Edu / FineWeb-2 / FineWeb2-HQ /
# StarCoder / FineMath) are NFC-assumed but not enforced upstream — this
# wrapper makes V9 operationally satisfied by running the idempotent
# normalizer over every parquet shard before mix_builder.py reads them. When
# `prepare_greek_pool.sh` has already materialized the final selected pool,
# the `cpt/` pass below normalizes that selected parquet too.
#
# Uses verify_and_normalize_nfc.py at the 03_3 location (kept there because
# it's also used outside the bakeoff for general corpus health checks).
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]):
#   - Parallelism: --workers W drives xargs -P W over independent parquet
#     files. Each worker invokes verify_and_normalize_nfc.py on one file,
#     writes a sibling temp parquet, then atomically replaces the original.
#   - Saturation: nanochat has 279 parquets + replay has 24+ langs ≈ 300 shards
#     total. Default --workers 64 fits the xfer CPU-only partition reasonably
#     without thrashing — NFC normalization is
#     IO + memcpy-bound, not CPU-bound, so going past ~64 has diminishing
#     return. Override `WORKERS=288` to push further.
#   - Memory: each worker holds one parquet shard at a time (≤ ~1 GB). 64 ×
#     1 GB peak ≈ 64 GB, well under 800 G slot.
#   - Known gaps: NFC is idempotent → re-runs are cheap (cost is just the
#     verify pass on already-NFC files). No per-doc parallelism within a shard.
#
# Usage:
#   bash normalize_nfc.sh
#   WORKERS=64 bash normalize_nfc.sh

set -euo pipefail

STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/../../../03_3_cscs_experiments_kickoff/scripts/verify_and_normalize_nfc.py"
WORKERS="${WORKERS:-64}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: normalizer script not found at $SCRIPT" >&2
    exit 2
fi

echo "=== normalize_nfc.sh ==="
date -u
echo "stage_root: $STAGE_ROOT"
echo "normalizer: $SCRIPT"
echo

file_list="$(mktemp)"
trap 'rm -f "$file_list"' EXIT

# If prepare_greek_pool has materialized the selected CPT pool, normalize that
# final Greek pool rather than the raw nanochat shards and temporary cpt
# intermediates. mix_builder.py reads selected + replay/code/math.
selected="$STAGE_ROOT/cpt/selected_after_apertus_and_internal_dedup.parquet"
if [ -f "$selected" ]; then
    printf '%s\0' "$selected" >> "$file_list"
else
    find "$STAGE_ROOT/nanochat" -type f -name '*.parquet' -print0 >> "$file_list" 2>/dev/null || true
fi

for subdir in replay code math; do
    find "$STAGE_ROOT/$subdir" -type f -name '*.parquet' -print0 >> "$file_list" 2>/dev/null || true
done

file_count="$(tr -cd '\0' < "$file_list" | wc -c | tr -d ' ')"
if [ "$file_count" = "0" ]; then
    echo "ERROR: found no parquet files to normalize under $STAGE_ROOT" >&2
    exit 2
fi

echo "files:   $file_count"
echo "workers: $WORKERS"
echo

normalize_one() {
    local input="$1"
    local tmp="${input}.nfc_tmp_${SLURM_JOB_ID:-manual}_$$"
    rm -f "$tmp"
    python3 "$SCRIPT" normalize "$input" --out "$tmp"
    mv "$tmp" "$input"
}
export SCRIPT
export -f normalize_one

xargs -0 -n 1 -P "$WORKERS" bash -c 'normalize_one "$1"' _ < "$file_list"

echo
echo "✓ V9 satisfied operationally — all parquets under $STAGE_ROOT NFC-normalized."
echo "Next: bash mix_builder.py with recipes/bulk.json"
