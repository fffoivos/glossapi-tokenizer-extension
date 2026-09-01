#!/usr/bin/env bash
set -euo pipefail

ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
RUN="$ST/runs/h2g_1p5b_2node_v1_retry14_direct_salloc_uenv_20260821"
Q="$RUN/qualification/32fd635f36bea663b3a63a8eabb71d013e7a587b533387d92f31481ba48dccd9/promotion_checkpoint_compatible_v105"
RUNNER=/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260818T211000Z-6b7796a-sequence-range
LAUNCHER="$RUNNER/bin/apertus-campaign"

[[ "${SLURM_JOB_ID:-}" == 3145729 ]] || { echo "expected successor allocation 3145729" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || { echo "normal allocation required" >&2; exit 2; }
[[ "${SLURM_JOB_NUM_NODES:-0}" == 2 ]] || { echo "exact two-node allocation required" >&2; exit 2; }
[[ "${SLURM_NTASKS:-0}" == 8 ]] || { echo "exact eight-task allocation required" >&2; exit 2; }
[[ -s "$ST/receipts/launch_gate_pre_extension.json" ]] || { echo "joint pre-extension gate missing" >&2; exit 2; }
[[ ! -e "$RUN/segments/s4/attempts/attempt_000001" ]] || { echo "1.5B s4 attempt already exists" >&2; exit 2; }

exec uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$LAUNCHER" run-holder-in-allocation \
    --manifest "$Q/manifest-proven.json" --run-root "$RUN" \
    --segment s4 --poll-seconds 5
