#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || exit 2
JOB=3140780
record=$(scontrol show job -o "$JOB")
[[ "$record" == *"JobState=RUNNING"* && "$record" == *"Partition=normal"* && "$record" == *"NumNodes=16"* ]] || exit 2
nodelist=$(squeue -h -j "$JOB" -o '%N')
[[ -n "$nodelist" ]] || exit 2
export SLURM_JOB_ID="$JOB" SLURM_JOBID="$JOB" SLURM_JOB_ACCOUNT=a0140 SLURM_JOB_PARTITION=normal
export SLURM_JOB_NODELIST="$nodelist" SLURM_NODELIST="$nodelist" SLURM_JOB_NUM_NODES=16 SLURM_NNODES=16
export SLURM_NTASKS=16 SLURM_NPROCS=16 SLURM_TASKS_PER_NODE='1(x16)' SLURM_CPUS_PER_TASK=288
export SLURM_GPUS_ON_NODE=4 SLURM_GPUS_PER_NODE=4
exec "$@"
