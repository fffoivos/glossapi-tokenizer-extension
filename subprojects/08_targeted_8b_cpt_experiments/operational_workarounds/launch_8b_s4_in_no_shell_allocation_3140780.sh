#!/usr/bin/env bash
set -euo pipefail

JOB=3140780
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
RUNNER=/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260816T133000Z-b76b699-contractclosure-v99
VENDOR=/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260816T153500Z-4b1310b-dualabi-v112/src/_vendor/campaign_pydeps
MANIFEST="$ST/canonical/pre_main_data_v9_safepath_5caa614d_v103/compiled_8b_s3_recovery_e9e2921.json"
RUN="$ST/runs/efficiency_bound_proven_8b_v112_recovery_codebinding_v3_staticready"
ADAPTER="$ST/control/retries/run_in_allocation_postprocess_recovery_compat_v3.py"
PERMIT="$RUN/permits/s4/from_s3_attempt_000003_ca9dd64_postprocess_fix.json"
GATE="$ST/receipts/launch_gate_pre_extension.json"

record=$(scontrol show job -o "$JOB")
for token in "UserId=fffoivos(" "Account=a0140" "JobState=RUNNING" \
  "Partition=normal" "NumNodes=16" "NumCPUs=4608" "NumTasks=64" \
  "CPUs/Task=72" "gres/gpu=64"; do
  [[ "$record" == *"$token"* ]] || { echo "allocation $JOB drift: missing $token" >&2; exit 2; }
done
nodelist=$(squeue -h -j "$JOB" -o '%N')
end_iso=$(squeue -h -j "$JOB" -o '%e')
[[ -n "$nodelist" && -n "$end_iso" ]] || { echo "allocation $JOB is not live" >&2; exit 2; }
end_epoch=$(date -d "$end_iso" +%s)
(( end_epoch - $(date +%s) >= 4800 )) || { echo "insufficient 8B allocation time" >&2; exit 2; }

[[ -s "$GATE" ]] || { echo "joint pre-extension gate missing" >&2; exit 2; }
[[ "$(/usr/bin/python3.11 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$GATE")" == launch_ready ]] || {
  echo "joint pre-extension gate is not launch-ready" >&2; exit 2;
}
[[ -s "$PERMIT" && -s "$MANIFEST" && -s "$ADAPTER" ]] || { echo "8B launch binding missing" >&2; exit 2; }
[[ ! -e "$RUN/segments/s4/attempts/attempt_000003" ]] || { echo "8B s4 retry attempt already exists" >&2; exit 2; }

export SLURM_CPUS_ON_NODE=288
export SLURM_CPUS_PER_TASK=288
export SLURM_GPUS_ON_NODE=4
export SLURM_GPUS_PER_NODE=4
export SLURM_JOBID="$JOB"
export SLURM_JOB_ACCOUNT=a0140
export SLURM_JOB_CPUS_PER_NODE='288(x16)'
export SLURM_JOB_END_TIME="$end_epoch"
export SLURM_JOB_ID="$JOB"
export SLURM_JOB_NODELIST="$nodelist"
export SLURM_JOB_NUM_NODES=16
export SLURM_JOB_PARTITION=normal
export SLURM_NNODES=16
export SLURM_NODELIST="$nodelist"
export SLURM_NPROCS=16
export SLURM_NTASKS=16
export SLURM_SUBMIT_DIR=/users/fffoivos
export SLURM_TASKS_PER_NODE='1(x16)'
export PYTHONDONTWRITEBYTECODE=1

exec /usr/bin/python3.11 "$ADAPTER" \
  --runner-root "$RUNNER" --vendor-root "$VENDOR" \
  --manifest "$MANIFEST" --run-root "$RUN" --segment s4 \
  --recovery-permit "$PERMIT"
