#!/usr/bin/env bash
# Minimal 2-node torchrun/NCCL smoke. Diagnostic only.
#SBATCH --job-name=allreduce_u291
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=36
#SBATCH --time=00:10:00

set -euo pipefail

SCRIPT="${SCRIPT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/scripts/torchrun_allreduce_smoke.py}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="${MASTER_PORT:-$((33000 + (${SLURM_JOB_ID:-0} % 1000)))}"

export NCCL_NET="${NCCL_NET:-AWS Libfabric}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_PROTO="${NCCL_PROTO:-^LL128}"
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-16384}"
export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
export FI_CXI_RDZV_GET_MIN="${FI_CXI_RDZV_GET_MIN:-0}"
export FI_CXI_RDZV_THRESHOLD="${FI_CXI_RDZV_THRESHOLD:-0}"
export FI_CXI_RDZV_EAGER_SIZE="${FI_CXI_RDZV_EAGER_SIZE:-0}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
unset NCCL_NET_PLUGIN

echo "allreduce smoke: uenv=$UENV_IMAGE script=$SCRIPT"
echo "allreduce smoke: master=$MASTER_ADDR:$MASTER_PORT nodes=${SLURM_NNODES}"
echo "allreduce smoke: NCCL_NET=$NCCL_NET FI_CXI_DEFAULT_CQ_SIZE=$FI_CXI_DEFAULT_CQ_SIZE"

uenv run "$UENV_IMAGE" --view=default -- \
    srun --ntasks-per-node=1 --cpus-per-task "${SLURM_CPUS_PER_TASK:-36}" \
        -lu bash -lc "exec torchrun --nnodes='${SLURM_NNODES}' --nproc_per_node=4 --node_rank=\$SLURM_PROCID --rdzv_backend=c10d --rdzv_endpoint='${MASTER_ADDR}:${MASTER_PORT}' '$SCRIPT'"
