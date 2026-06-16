#!/usr/bin/env bash
# AdEMAMix β2 (Adam 2nd-moment decay) sweep on the EXISTING 13.5 B curriculum_v2 dataset, settled HP set.
#
# Settled (held fixed):  LR_PEAK=5.5e-5 · ADEMA_BETA3=0.999 · ADEMA_ALPHA=4 · split replay 79/20/1
#                        (FOREIGN_REPLAY_R=20/79, OLD_GREEK_REPLAY_R=1/79).
# Swept:                 ADEMA_BETA2 ∈ {0.99, 0.999} — the 2 REMAINING points. β2=0.995 is the completed
#                        α=4 run (curr_td_f20_g1_lr5.5e-5_a4): production β2=0.995 (common_cpt.env:22), so
#                        a4 IS the middle point — integrate it; don't re-run.
# WARMUP (load-bearing): PINNED LR_WARMUP_ITERS=400 across all arms (NOT the config's coupled 2/(1-β2)).
#                        On 3218 it, coupled would give β2=0.999 a 2000-iter warmup (62% of run) —
#                        pathological + confounds the β2 contrast. 400 = 2/(1-0.995) = the a4 anchor's
#                        warmup, so the 3 β2 points {0.99, 0.995=a4, 0.999} share one warmup → clean.
#                        Set COUPLE_WARMUP=1 to instead let common_cpt.env compute 2/(1-β2).
# Scheduler:             β3/α warmup tied to TRAIN_ITERS=3218 (production), unchanged.
# Geometry:              13.5 B, identical to the α/LR/β3 sweeps → TOTAL_ITER=3218, PHASE1_EXIT_ITER=2261.
# Data:                  REUSES curriculum_v2's existing decontam'd/anon'd binaries + 9 held-out vals.
#
# Usage (reuses curriculum_v2/megatron; preflight verifies the binaries):
#   DRY_RUN=1 bash train/sweep_beta2.sh                  # inspect the 2 chains
#   DRY_RUN=0 CONFIRM_LAUNCH=1 bash train/sweep_beta2.sh
#   COUPLE_WARMUP=1 ...                                  # revert to coupled 2/(1-β2) warmup
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# REUSE curriculum_v2's existing 13.5 B binaries (read-only); no build.
export STAGE="${STAGE:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/curriculum_v2}"
source "$HERE/../paths.env"   # STAGE → MEGOUT=$STAGE/megatron (the existing binaries)

# --- settled HPs (held fixed) ---
FOREIGN_REPLAY_R="${FOREIGN_REPLAY_R:-0.253164557}"   # 20/79
OLD_GREEK_REPLAY_R="${OLD_GREEK_REPLAY_R:-0.012658228}" # 1/79
LR_PEAK="${LR_PEAK:-5.5e-5}"
ADEMA_ALPHA="${ADEMA_ALPHA:-4.0}"
ADEMA_BETA3="${ADEMA_BETA3:-0.999}"
# --- swept axis ---
BETA2_GRID="${BETA2_GRID:-0.99 0.999}"   # 0.995 = the completed α=4 run (the middle point; integrate it)
RUN_LABEL="${RUN_LABEL:-13b}"
# --- warmup policy: PIN fixed (isolate β2) unless COUPLE_WARMUP=1 ---
COUPLE_WARMUP="${COUPLE_WARMUP:-0}"
FIXED_WARMUP="${FIXED_WARMUP:-400}"      # = 2/(1-0.995); matches the α=4 anchor
# --- 13.5 B geometry (identical to the prior sweeps) ---
TRAIN_TOKENS="${TRAIN_TOKENS:-13500000000}"
TOTAL_ITER="${TOTAL_ITER:-3218}"
PHASE1_EXIT_ITER="${PHASE1_EXIT_ITER:-2261}"

# --- preflight: existing 13.5 B training binaries + held-out vals must exist in $MEGOUT ---
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
        [ -s "$p" ] || { echo "ERROR: missing held-out val binary: $p  (expected in $STAGE/megatron from the prior sweeps)" >&2; exit 2; }
      done
    done
  done
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for B2 in $BETA2_GRID; do
  SAFE="${B2//./p}"
  if [ "$COUPLE_WARMUP" = "1" ]; then WU=""; else WU="$FIXED_WARMUP"; fi
  RUN_TAG="curr_td_b2${SAFE}_b3p999_${RUN_LABEL}_${STAMP}" \
    ARM=td R=0.25 \
    FOREIGN_REPLAY_R="$FOREIGN_REPLAY_R" OLD_GREEK_REPLAY_R="$OLD_GREEK_REPLAY_R" \
    LR_PEAK="$LR_PEAK" ADEMA_ALPHA="$ADEMA_ALPHA" ADEMA_BETA3="$ADEMA_BETA3" ADEMA_BETA2="$B2" \
    LR_WARMUP_ITERS="$WU" \
    TRAIN_TOKENS="$TRAIN_TOKENS" TOTAL_ITER="$TOTAL_ITER" PHASE1_EXIT_ITER="$PHASE1_EXIT_ITER" \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done

echo "β2 sweep submitted ($RUN_LABEL): BETA2_GRID=$BETA2_GRID  STAGE=$STAGE"
echo "  warmup: $([ "$COUPLE_WARMUP" = 1 ] && echo "COUPLED 2/(1-β2)" || echo "FIXED ${FIXED_WARMUP} it")  (β3=$ADEMA_BETA3 α=$ADEMA_ALPHA LR=$LR_PEAK, replay 79/20/1)"
echo "  integrate β2=0.995 from the completed α=4 run (same dataset + warmup 400)."
