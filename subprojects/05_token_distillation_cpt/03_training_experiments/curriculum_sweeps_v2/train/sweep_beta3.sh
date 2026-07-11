#!/usr/bin/env bash
# AdEMAMix β3 (slow-EMA decay) sweep on the EXISTING 13.5 B curriculum_v2 dataset, at the settled HP set.
#
# Settled (held fixed):  LR_PEAK=5.5e-5 · ADEMA_ALPHA=4 · split replay 79/20/1
#                        (FOREIGN_REPLAY_R=20/79, OLD_GREEK_REPLAY_R=1/79).
# Swept:                 ADEMA_BETA3 ∈ {0.99, 0.995}; reuse the completed α=4 run as the β3=0.999 point.
# Scheduler:             KEPT as production (β3/α warmup tied to TRAIN_ITERS=3218). NOT decoupled —
#                        this run answers "under the production fraction-of-run ramp, does the β3 target
#                        matter at the 13.5 B horizon". NOTE: shorter horizon than the shelved 27 B plan,
#                        so the slow-EMA signal is more compressed (the original reason we had considered
#                        2× HPLT); pivoted here for node availability.
# Geometry:              13.5 B, identical to the α/LR sweeps → TOTAL_ITER=3218, PHASE1_EXIT_ITER=2261.
#                        The β3=0.999 arm reproduces the settled α=4 config; don't rerun it by default.
# Data:                  REUSES curriculum_v2's existing decontam'd/anon'd binaries + 9 held-out vals
#                        (no new build). Reads $STAGE/megatron read-only; outputs go to new RUN_TAGs.
#
# Usage (reuses curriculum_v2/megatron; preflight verifies the binaries are present):
#   DRY_RUN=1 bash train/sweep_beta3.sh                  # inspect the 2 active chains
#   DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta3.sh
#   BETA3_GRID="0.999" ...                               # optional reproducibility rerun of the control
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# REUSE curriculum_v2's existing 13.5 B binaries (read-only); no isolated build needed.
export STAGE="${STAGE:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2}"
source "$HERE/../paths.env"   # STAGE → MEGOUT=$STAGE/megatron (the existing binaries)

# --- settled HPs (held fixed) ---
FOREIGN_REPLAY_R="${FOREIGN_REPLAY_R:-0.253164557}"   # 20/79
OLD_GREEK_REPLAY_R="${OLD_GREEK_REPLAY_R:-0.012658228}" # 1/79
LR_PEAK="${LR_PEAK:-5.5e-5}"
ADEMA_ALPHA="${ADEMA_ALPHA:-4.0}"
ADEMA_BETA2="${ADEMA_BETA2:-0.995}"               # exact as-run control value
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-400}"         # exact as-run fixed warmup
# --- swept axis ---
BETA3_GRID="${BETA3_GRID:-0.99 0.995}"
RUN_LABEL="${RUN_LABEL:-13b}"
# --- 13.5 B geometry (identical to the α/LR sweeps; production tied-to-run ramp tied to TRAIN_ITERS≈3218) ---
TRAIN_TOKENS="${TRAIN_TOKENS:-13500000000}"   # → TRAIN_ITERS≈3218 → β3/α ramp endpoints
TOTAL_ITER="${TOTAL_ITER:-3218}"
PHASE1_EXIT_ITER="${PHASE1_EXIT_ITER:-2261}"  # 70/30 boundary (multiple of 119), as the prior sweeps

# --- preflight: the existing 13.5 B training binaries + held-out vals must exist in $MEGOUT ---
if [ "${SKIP_DATA_CHECK:-0}" != "1" ]; then
  for prefix in hplt_only glossapi_only foreign_replay_only old_greek_replay_only; do
    for tok in base ext; do
      for suffix in bin idx; do
        p="$MEGOUT/${prefix}_${tok}_text_document.$suffix"
        [ -s "$p" ] || { echo "ERROR: missing training binary: $p  (expected from the prior sweeps; STAGE=$STAGE)" >&2; exit 2; }
      done
    done
  done
  for v in val_hplt val_openarchives val_greek_phd \
           val_forget_english val_forget_de val_forget_ru val_forget_zh val_forget_code val_forget_old_greek; do
    for tok in base ext; do
      for suffix in bin idx; do
        p="$MEGOUT/${v}_${tok}_text_document.$suffix"
        [ -s "$p" ] || {
          echo "ERROR: missing held-out val binary: $p  (expected in $STAGE/megatron from the prior sweeps)" >&2; exit 2; }
      done
    done
  done
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for B3 in $BETA3_GRID; do
  SAFE="${B3//./p}"
  RUN_TAG="curr_td_b3${SAFE}_${RUN_LABEL}_${STAMP}" \
    ARM=td R=0.25 \
    FOREIGN_REPLAY_R="$FOREIGN_REPLAY_R" OLD_GREEK_REPLAY_R="$OLD_GREEK_REPLAY_R" \
    LR_PEAK="$LR_PEAK" ADEMA_ALPHA="$ADEMA_ALPHA" ADEMA_BETA2="$ADEMA_BETA2" ADEMA_BETA3="$B3" \
    LR_WARMUP_ITERS="$LR_WARMUP_ITERS" \
    TRAIN_TOKENS="$TRAIN_TOKENS" TOTAL_ITER="$TOTAL_ITER" PHASE1_EXIT_ITER="$PHASE1_EXIT_ITER" \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done

echo "β3 sweep submitted ($RUN_LABEL): BETA3_GRID=$BETA3_GRID  STAGE=$STAGE"
echo "  TRAIN_TOKENS=$TRAIN_TOKENS TOTAL_ITER=$TOTAL_ITER PHASE1_EXIT_ITER=$PHASE1_EXIT_ITER  (β2=$ADEMA_BETA2 warmup=$LR_WARMUP_ITERS α=$ADEMA_ALPHA LR=$LR_PEAK, replay 79/20/1)"
