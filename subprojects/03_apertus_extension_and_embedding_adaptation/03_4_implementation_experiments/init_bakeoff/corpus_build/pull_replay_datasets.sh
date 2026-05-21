#!/usr/bin/env bash
# Pull non-Greek replay + code + math datasets — single-shard-per-lang variant.
#
# Realized after first attempt: FineWeb2-HQ fra_Latn alone has 436 ~11 GB files
# = 4.8 TB. For a 7B-token mix where each lang gets ~70 M tokens, ONE parquet
# shard (~50 M-200 M tokens) per lang is enough. Pull only 000_00000.parquet
# per lang and we get plenty of material at ~50-500 MB per lang.

set -euo pipefail
STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
REPLAY_DIR="$STAGE_ROOT/replay"
CODE_DIR="$STAGE_ROOT/code"
MATH_DIR="$STAGE_ROOT/math"
mkdir -p "$REPLAY_DIR" "$CODE_DIR" "$MATH_DIR"
echo "=== pull_replay_datasets.sh (single-shard-per-lang) ==="; date -u

echo "=== T1.eng (FineWeb-Edu sample/10BT, single shard) ==="
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --local-dir "$REPLAY_DIR/eng_fineweb_edu" \
    --include "sample/10BT/000_00000.parquet" "README.md"

echo "=== T1: 7 high-resource (FineWeb2-HQ; single shard per lang) ==="
for cfg in fra_Latn deu_Latn ita_Latn spa_Latn rus_Cyrl arb_Arab cmn_Hani; do
    echo "--- T1 $cfg ---"
    huggingface-cli download epfml/FineWeb2-HQ \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2hq" \
        --include "${cfg}/000_00000.parquet" "README.md" || echo "  WARN $cfg failed"
done

echo "=== T2 HQ-langs (FineWeb2-HQ; FW2 fallback) ==="
for cfg in tur_Latn por_Latn pol_Latn nld_Latn pes_Arab jpn_Jpan; do
    echo "--- T2 $cfg ---"
    huggingface-cli download epfml/FineWeb2-HQ \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2hq" \
        --include "${cfg}/000_00000.parquet" "README.md" || {
        echo "  $cfg not in FW2-HQ; falling back to FineWeb-2"
        huggingface-cli download HuggingFaceFW/fineweb-2 \
            --repo-type dataset \
            --local-dir "$REPLAY_DIR/${cfg}_fw2" \
            --include "data/${cfg}/train/000_00000.parquet" "README.md" || echo "  WARN $cfg fallback failed"
    }
done

echo "=== T2 FW2-only (bul/srp/ron/heb/ukr) ==="
for cfg in bul_Cyrl srp_Cyrl ron_Latn heb_Hebr ukr_Cyrl; do
    echo "--- T2 $cfg ---"
    huggingface-cli download HuggingFaceFW/fineweb-2 \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2" \
        --include "data/${cfg}/train/000_00000.parquet" "README.md" || echo "  WARN $cfg failed"
done

echo "=== T3 (FW2; single shard) ==="
for cfg in lat_Latn hye_Armn kat_Geor als_Latn mkd_Cyrl; do
    echo "--- T3 $cfg ---"
    huggingface-cli download HuggingFaceFW/fineweb-2 \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2" \
        --include "data/${cfg}/train/000_00000.parquet" "README.md" || echo "  WARN $cfg failed"
done

echo "=== Code: StarCoder (one shard per major lang) ==="
huggingface-cli download bigcode/starcoderdata \
    --repo-type dataset \
    --local-dir "$CODE_DIR" \
    --include "data/python/train-00000-of-*.parquet" \
              "data/javascript/train-00000-of-*.parquet" \
              "data/rust/train-00000-of-*.parquet" \
              "data/go/train-00000-of-*.parquet" \
              "README.md" \
    || echo "  WARN: starcoderdata pull failed"

echo "=== Math: FineMath-3plus (one shard) ==="
huggingface-cli download HuggingFaceTB/finemath \
    --repo-type dataset \
    --local-dir "$MATH_DIR/finemath" \
    --include "finemath-3plus/train-00000-of-*.parquet" "README.md" \
    || echo "  WARN: FineMath pull failed"

echo
echo "=== summary ==="; date -u
du -sh "$REPLAY_DIR"/* 2>/dev/null | sort -hr | head -25
echo
du -sh "$CODE_DIR" "$MATH_DIR" 2>/dev/null
