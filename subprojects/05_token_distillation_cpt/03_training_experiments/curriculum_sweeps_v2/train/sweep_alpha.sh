#!/usr/bin/env bash
# AdEMAMix alpha sweep at the settled replay split and peak LR.
#
# Settled values:
#   total fractions: 79% new Greek, 20% foreign replay, 1% old-Greek replay
#   LR_PEAK=5.5e-5
# Sweep axis:
#   ADEMA_ALPHA in {0,4,8}
#
# Alpha=0 is the AdamW-like/no-slow-EMA ablation while keeping the same trainer
# path and schedule machinery.
set -euo pipefail
source "$(dirname "$0")/../paths.env"

R_STAR="${R_STAR:-0.25}"  # metadata only for this split-replay sweep
FOREIGN_REPLAY_R="${FOREIGN_REPLAY_R:-0.253164557}"
OLD_GREEK_REPLAY_R="${OLD_GREEK_REPLAY_R:-0.012658228}"
LR_PEAK="${LR_PEAK:-5.5e-5}"
ALPHA_GRID="${ALPHA_GRID:-0 4 8}"

if [ "${SKIP_DATA_CHECK:-0}" != "1" ]; then
  for prefix in foreign_replay_only old_greek_replay_only; do
    for tok in base ext; do
      for suffix in bin idx; do
        path="$MEGOUT/${prefix}_${tok}_text_document.$suffix"
        if [ ! -s "$path" ]; then
          echo "ERROR: missing required split replay binary: $path" >&2
          exit 2
        fi
      done
    done
  done
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for ALPHA in $ALPHA_GRID; do
  SAFE_ALPHA="${ALPHA//./p}"
  RUN_TAG="curr_td_f20_g1_lr5.5e-5_a${SAFE_ALPHA}_${STAMP}" \
    ARM=td R="$R_STAR" \
    FOREIGN_REPLAY_R="$FOREIGN_REPLAY_R" \
    OLD_GREEK_REPLAY_R="$OLD_GREEK_REPLAY_R" \
    LR_PEAK="$LR_PEAK" \
    ADEMA_BETA2=0.995 \
    ADEMA_BETA3=0.999 \
    ADEMA_ALPHA="$ALPHA" \
    LR_WARMUP_ITERS=400 \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done

echo "alpha sweep submitted at split replay: foreign=$FOREIGN_REPLAY_R old_greek=$OLD_GREEK_REPLAY_R LR_PEAK=$LR_PEAK ALPHA_GRID=$ALPHA_GRID"
