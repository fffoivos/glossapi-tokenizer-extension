#!/usr/bin/env bash
# (d) PEAK-LR SWEEP — TD + curriculum, at the replay split chosen from (c).
# Vary ONLY LR_PEAK. LR_WARMUP stays 2/(1-beta2)=400 (beta2 fixed).
#
# Selected replay split from PRODUCTION_MIX_DECISION_20260612:
#   total fractions: 79% new Greek, 20% foreign replay, 1% old-Greek replay
#   Megatron blend weights relative to new Greek:
#     FOREIGN_REPLAY_R = 20/79 = 0.253164557
#     OLD_GREEK_REPLAY_R = 1/79 = 0.012658228
#
# Requires dataset/split_replay_final_and_tokenize.sbatch to have created:
#   foreign_replay_only_{base,ext}_text_document
#   old_greek_replay_only_{base,ext}_text_document
set -euo pipefail
source "$(dirname "$0")/../paths.env"
R_STAR="${R_STAR:-0.25}"  # metadata only for this split-replay sweep
FOREIGN_REPLAY_R="${FOREIGN_REPLAY_R:-0.253164557}"
OLD_GREEK_REPLAY_R="${OLD_GREEK_REPLAY_R:-0.012658228}"
LR_GRID="${LR_GRID:-2.75e-5 5.5e-5 8.25e-5 1.1e-4}"

for prefix in foreign_replay_only old_greek_replay_only; do
  for tok in base ext; do
    test -s "$MEGOUT/${prefix}_${tok}_text_document.bin"
    test -s "$MEGOUT/${prefix}_${tok}_text_document.idx"
  done
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for LR in $LR_GRID; do
  RUN_TAG="curr_td_f20_g1_lr${LR}_${STAMP}" \
    ARM=td R="$R_STAR" FOREIGN_REPLAY_R="$FOREIGN_REPLAY_R" OLD_GREEK_REPLAY_R="$OLD_GREEK_REPLAY_R" LR_PEAK="$LR" \
    ADEMA_BETA2=0.995 ADEMA_BETA3=0.999 ADEMA_ALPHA=4 LR_WARMUP_ITERS=400 \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done
echo "peak-LR sweep submitted at split replay: foreign=$FOREIGN_REPLAY_R old_greek=$OLD_GREEK_REPLAY_R; LR_GRID=$LR_GRID"
