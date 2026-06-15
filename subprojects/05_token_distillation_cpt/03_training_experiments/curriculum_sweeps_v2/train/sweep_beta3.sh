#!/usr/bin/env bash
# AdEMAMix β3 (slow-EMA decay) sweep on the 2× HPLT / ~27 B run, at the settled HP set.
#
# Settled (held fixed):  LR_PEAK=5.5e-5 · ADEMA_ALPHA=4 · split replay 79/20/1
#                        (FOREIGN_REPLAY_R=20/79, OLD_GREEK_REPLAY_R=1/79).
# Swept:                 ADEMA_BETA3 ∈ {0.99, 0.995, 0.999(default)}.
# Scheduler:             KEPT as production (β3/α warmup tied to TRAIN_ITERS, recomputed from the
#                        larger TRAIN_TOKENS). NOT decoupled — this run answers "under the production
#                        fraction-of-run ramp, does the β3 target matter at a ~27 B (2× HPLT) horizon".
# Geometry:              TOTAL_ITER=6436 (~27 B), PHASE1_EXIT_ITER=4522 (=2×2261=38×119, 70/30).
#                        Re-pin PHASE1_EXIT_ITER from the realized ext-tokenizer Stage-B .bin sizes
#                        before launch (BETA3_SWEEP_PLAN §4).
# Data:                  reads from an ISOLATED stage dir so it does not clobber the 13.5 B binaries.
#                        Build them first (BETA3_SWEEP_PLAN §1-3); held-out val binaries are copied in.
#
# Usage (after the 27 B binaries + copied vals exist in $STAGE/megatron):
#   STAGE_BETA3=/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_beta3 \
#     DRY_RUN=1 bash train/sweep_beta3.sh                 # inspect the 3 chains
#   STAGE_BETA3=... DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta3.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# isolate the build/run staging so curriculum_v2's 13.5 B binaries are untouched
export STAGE="${STAGE_BETA3:-${STAGE:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_beta3}}"
source "$HERE/../paths.env"   # STAGE override cascades to MEGOUT=$STAGE/megatron

# --- settled HPs (held fixed) ---
FOREIGN_REPLAY_R="${FOREIGN_REPLAY_R:-0.253164557}"   # 20/79
OLD_GREEK_REPLAY_R="${OLD_GREEK_REPLAY_R:-0.012658228}" # 1/79
LR_PEAK="${LR_PEAK:-5.5e-5}"
ADEMA_ALPHA="${ADEMA_ALPHA:-4.0}"
# --- swept axis ---
BETA3_GRID="${BETA3_GRID:-0.99 0.995 0.999}"
RUN_LABEL="${RUN_LABEL:-27b}"
# --- 27 B / 2× HPLT geometry (production tied-to-run ramp: warmup auto-scales from TRAIN_TOKENS) ---
TRAIN_TOKENS="${TRAIN_TOKENS:-26994540544}"   # 6436 × 4,194,304 ; sets TRAIN_ITERS=6436 → ramp endpoints
TOTAL_ITER="${TOTAL_ITER:-6436}"
PHASE1_EXIT_ITER="${PHASE1_EXIT_ITER:-4522}"  # re-pin from realized ext sizes; must be a multiple of 119

# --- preflight: the 27 B training binaries + the copied held-out vals must exist in $MEGOUT ---
if [ "${SKIP_DATA_CHECK:-0}" != "1" ]; then
  for prefix in hplt_only glossapi_only foreign_replay_only old_greek_replay_only; do
    for tok in base ext; do
      for suffix in bin idx; do
        p="$MEGOUT/${prefix}_${tok}_text_document.$suffix"
        [ -s "$p" ] || { echo "ERROR: missing training binary: $p  (build §1-3 first; STAGE=$STAGE)" >&2; exit 2; }
      done
    done
  done
  for v in val_hplt val_openarchives val_greek_phd \
           val_forget_english val_forget_de val_forget_ru val_forget_zh val_forget_code val_forget_old_greek; do
    for tok in base ext; do
      for suffix in bin idx; do
        p="$MEGOUT/${v}_${tok}_text_document.$suffix"
        [ -s "$p" ] || {
          echo "ERROR: missing held-out val binary: $p  (copy from curriculum_v2/megatron, then rerun)" >&2; exit 2; }
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
    LR_PEAK="$LR_PEAK" ADEMA_ALPHA="$ADEMA_ALPHA" ADEMA_BETA3="$B3" \
    TRAIN_TOKENS="$TRAIN_TOKENS" TOTAL_ITER="$TOTAL_ITER" PHASE1_EXIT_ITER="$PHASE1_EXIT_ITER" \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done

echo "β3 sweep submitted ($RUN_LABEL): BETA3_GRID=$BETA3_GRID  STAGE=$STAGE"
echo "  TRAIN_TOKENS=$TRAIN_TOKENS TOTAL_ITER=$TOTAL_ITER PHASE1_EXIT_ITER=$PHASE1_EXIT_ITER  (α=$ADEMA_ALPHA LR=$LR_PEAK, replay 79/20/1)"
