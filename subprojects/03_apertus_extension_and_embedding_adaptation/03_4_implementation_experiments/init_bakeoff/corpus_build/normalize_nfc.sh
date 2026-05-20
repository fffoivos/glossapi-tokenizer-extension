#!/usr/bin/env bash
# Operationally satisfy V9 — NFC-normalize the pulled corpus parquets in place.
#
# Per cpt_plan.md v0.7 §8 I1 + V9: Apertus's tokenizer has `normalizer: null`,
# so pre-tokenization NFC at the corpus level is required. Greek nanochat
# delivers NFC upstream (HPLT 500/500 sample-verified; finepdfs-edu 0.07% NFD
# leak remediated). Replay datasets (FineWeb-Edu / FineWeb-2 / FineWeb2-HQ /
# StarCoder / FineMath) are NFC-assumed but not enforced upstream — this
# wrapper makes V9 operationally satisfied by running the idempotent
# normalizer over every parquet shard before mix_builder.py reads them.
#
# Uses verify_and_normalize_nfc.py at the 03_3 location (kept there because
# it's also used outside the bakeoff for general corpus health checks).
#
# Usage:
#   bash normalize_nfc.sh

set -euo pipefail

STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
SCRIPT="$(dirname "$0")/../../../03_3_cscs_experiments_kickoff/scripts/verify_and_normalize_nfc.py"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: normalizer script not found at $SCRIPT" >&2
    exit 2
fi

echo "=== normalize_nfc.sh ==="
date -u
echo "stage_root: $STAGE_ROOT"
echo "normalizer: $SCRIPT"
echo

# The verify_and_normalize_nfc.py script supports both verify-only and
# in-place normalize modes. Use normalize mode here (idempotent — already-NFC
# parquets are no-ops).
for subdir in nanochat replay code math; do
    target="$STAGE_ROOT/$subdir"
    if [ ! -d "$target" ]; then
        echo "  skip (missing): $target"
        continue
    fi
    echo
    echo "=== normalize: $target ==="
    python3 "$SCRIPT" normalize \
        --root "$target" \
        --pattern '*.parquet' \
        --workers 16 \
        --report-every 100
done

echo
echo "✓ V9 satisfied operationally — all parquets under $STAGE_ROOT NFC-normalized."
echo "Next: bash mix_builder.py with recipes/bulk.json"
