#!/usr/bin/env bash
# (d) PEAK-LR SWEEP — TD + curriculum, at the replay R* chosen from (c). Vary ONLY LR_PEAK.
# LR_WARMUP stays 2/(1-beta2)=400 (beta2 fixed). 5.5e-5 is the control (= the replay-sweep R* point,
# but re-run here so all 4 LR points share identical RUN_ROOT bookkeeping). Same 3 binaries reused.
# Usage:  R_STAR=0.25 bash sweep_peak_lr.sh
set -euo pipefail
source "$(dirname "$0")/../paths.env"
R_STAR="${R_STAR:?set R_STAR to the replay fraction chosen from (c), e.g. R_STAR=0.25}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for LR in 2.75e-5 5.5e-5 8.25e-5 1.1e-4; do
  RUN_TAG="curr_td_r${R_STAR}_lr${LR}_${STAMP}" ARM=td R="$R_STAR" LR_PEAK="$LR" \
    bash "$V2/train/submit_curriculum_two_phase.sh"
done
echo "peak-LR sweep submitted at R*=$R_STAR: LR in {2.75e-5,5.5e-5(control),8.25e-5,1.1e-4}"
