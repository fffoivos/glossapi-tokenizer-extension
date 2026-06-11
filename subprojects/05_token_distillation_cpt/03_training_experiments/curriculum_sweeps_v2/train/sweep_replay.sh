#!/usr/bin/env bash
# (c) REPLAY SWEEP — TD + curriculum, vary ONLY the replay blend weight R.
# The 3 phase binaries are built once; each R is just a different --data-path weight (no rebuild).
# Read forgetting from the in-training old-data extra-valid loss; adaptation from GreekMMLU.
# IMPORTANT: watch the first chain's HPLT->GlossAPI boundary (phase-2 seg1) for an optimizer
# disturbance before trusting the sweep (see EPISTEMIC_PLAN study (c) ride-along).
set -euo pipefail
source "$(dirname "$0")/../paths.env"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for R in 0.35 0.25 0.15; do
  RUN_TAG="curr_td_replay${R}_${STAMP}" ARM=td R="$R" LR_PEAK=5.5e-5 \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done
echo "replay sweep submitted: R in {0.35(control),0.25,0.15}"
