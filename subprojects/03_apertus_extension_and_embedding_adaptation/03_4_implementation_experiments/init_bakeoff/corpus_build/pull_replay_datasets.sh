#!/usr/bin/env bash
# Pull non-Greek replay datasets + code corpus onto Clariden iopsstor.
#
# 24 replay languages from cpt_plan.md v0.7 section 4.2:
#   T1 (8): eng (FineWeb-Edu Score-3), fra/deu/ita/spa/rus/arb/cmn (FineWeb-2-HQ)
#   T2 (11): tur/por/pol/nld/pes/jpn (FineWeb-2-HQ where available), bul/srp/ron/heb/ukr (FineWeb-2)
#   T3 (5): lat/hye/kat/als/mkd (FineWeb-2)
# Code: bigcode/starcoderdata
#
# These are huge datasets — we use the `--include 'data/*'` HF download path and
# the mix_builder later streams from the local cache, only consuming what it
# needs to hit the token budget. Plan to use ~200-400 GB on iopsstor depending
# on how aggressively you cap each per-language download.
#
# Storage strategy: rely on HF's incremental file resolution. `huggingface-cli
# download` resumes partials. If we want to cap disk, use `--include` with
# specific shard patterns and stop once per-language has enough material.
#
# Run on Clariden login node (no slurm). Multi-hour download.
#
# Usage:
#   bash pull_replay_datasets.sh

set -euo pipefail

STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
REPLAY_DIR="$STAGE_ROOT/replay"
CODE_DIR="$STAGE_ROOT/code"

mkdir -p "$REPLAY_DIR" "$CODE_DIR"

echo "=== pull_replay_datasets.sh ==="
echo "stage root: $STAGE_ROOT"
echo "free space: $(df -h "$STAGE_ROOT" | tail -1)"
echo

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Sanity: huggingface-cli available?
which huggingface-cli || { echo "ERROR: huggingface-cli not on PATH"; exit 1; }

# === T1: English via FineWeb-Edu Score-3 ===
echo "=== T1.eng (FineWeb-Edu Score-3) ==="
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --local-dir "$REPLAY_DIR/eng_fineweb_edu" \
    --include 'sample/10BT/*' \
    --include 'README.md'
# 10BT is the 10B-token sample; plenty for our ~280M-token English replay share.
# If we want the full score-3 slice we'd drop --include and pull the full repo.

# === T1: other 7 high-resource langs (FineWeb-2-HQ) ===
echo "=== T1: fra/deu/ita/spa/rus/arb/cmn (FineWeb-2-HQ) ==="
for cfg in fra_Latn deu_Latn ita_Latn spa_Latn rus_Cyrl arb_Arab cmn_Hani; do
    echo "--- T1 $cfg ---"
    huggingface-cli download epfml/FineWeb2-HQ \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2hq" \
        --include "${cfg}/*" \
        --include 'README.md' || echo "  WARN $cfg failed; continuing"
done

# === T2: HQ-available languages ===
echo "=== T2 HQ: tur/por/pol/nld/pes/jpn (FineWeb-2-HQ) ==="
for cfg in tur_Latn por_Latn pol_Latn nld_Latn pes_Arab jpn_Jpan; do
    echo "--- T2 $cfg (HQ) ---"
    huggingface-cli download epfml/FineWeb2-HQ \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2hq" \
        --include "${cfg}/*" \
        --include 'README.md' || \
    {
        # FineWeb-2-HQ has only 20 langs; some T2 may be missing → fall back to FW2
        echo "  $cfg not in FW2-HQ; falling back to FineWeb-2"
        huggingface-cli download HuggingFaceFW/fineweb-2 \
            --repo-type dataset \
            --local-dir "$REPLAY_DIR/${cfg}_fw2" \
            --include "data/${cfg}/*" \
            --include 'README.md' || echo "  WARN $cfg failed in both; continuing"
    }
done

# === T2: standard-FW2-only languages ===
echo "=== T2 standard: bul/srp/ron/heb/ukr (FineWeb-2) ==="
for cfg in bul_Cyrl srp_Cyrl ron_Latn heb_Hebr ukr_Cyrl; do
    echo "--- T2 $cfg (FW2) ---"
    huggingface-cli download HuggingFaceFW/fineweb-2 \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2" \
        --include "data/${cfg}/*" \
        --include 'README.md' || echo "  WARN $cfg failed; continuing"
done

# === T3: small languages (FineWeb-2) ===
echo "=== T3: lat/hye/kat/als/mkd (FineWeb-2) ==="
for cfg in lat_Latn hye_Armn kat_Geor als_Latn mkd_Cyrl; do
    echo "--- T3 $cfg ---"
    huggingface-cli download HuggingFaceFW/fineweb-2 \
        --repo-type dataset \
        --local-dir "$REPLAY_DIR/${cfg}_fw2" \
        --include "data/${cfg}/*" \
        --include 'README.md' || echo "  WARN $cfg failed; continuing"
done

# === Code: StarCoder ===
echo "=== code: bigcode/starcoderdata ==="
# starcoderdata is very large; pull a representative subset
huggingface-cli download bigcode/starcoderdata \
    --repo-type dataset \
    --local-dir "$CODE_DIR" \
    --include 'data/python/*' \
    --include 'data/javascript/*' \
    --include 'data/rust/*' \
    --include 'data/go/*' \
    --include 'README.md' || echo "WARN: starcoderdata pull failed; revisit (may need access token for some shards)"

# === Math: FineMath (Apertus stage-1 source) ===
# Per v0.7 §4.4 + submit_apertus_8b.sh:L29 (finemath-3plus-merge).
# FineMath-3plus is the higher-quality subset (3+ rating).
MATH_DIR="$STAGE_ROOT/math"
mkdir -p "$MATH_DIR"
echo "=== math: HuggingFaceTB/finemath (finemath-3plus) ==="
huggingface-cli download HuggingFaceTB/finemath \
    --repo-type dataset \
    --local-dir "$MATH_DIR/finemath" \
    --include 'finemath-3plus/*' \
    --include 'README.md' || echo "  WARN: FineMath pull failed; continuing"

echo
echo "=== summary ==="
du -sh "$REPLAY_DIR"/* 2>/dev/null | sort -hr | head -30
echo
echo "code:"
du -sh "$CODE_DIR" 2>/dev/null || true
echo
echo "✓ Replay + code datasets staged at $STAGE_ROOT"
echo "Next: run mix_builder.py with recipes/bulk.json to assemble the bakeoff stream."
