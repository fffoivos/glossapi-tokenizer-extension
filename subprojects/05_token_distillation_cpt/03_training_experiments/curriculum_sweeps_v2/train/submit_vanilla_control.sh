#!/usr/bin/env bash
# (b) the single VANILLA control: curriculum + baseline replay (R=0.35), base tokenizer (131072).
# Re-anchors the tokenizer reference (TD vs Vanilla) at the new config. One two-phase chain.
# ARM=vanilla makes bakeoff_train select BASE_DATA_PREFIX, INIT_VANILLA, val tokenization=base.
set -euo pipefail
source "$(dirname "$0")/../paths.env"
RUN_TAG="curr_vanilla_r0.35_$(date -u +%Y%m%dT%H%M%SZ)" ARM=vanilla R=0.35 LR_PEAK=5.5e-5 \
  ADEMA_BETA2=0.995 ADEMA_BETA3=0.999 ADEMA_ALPHA=4 LR_WARMUP_ITERS=400 \
  bash "$V2/train/submit_curriculum_two_phase.sh"
