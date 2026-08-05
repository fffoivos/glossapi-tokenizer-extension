#!/usr/bin/env bash
# Run one four-node/16-GPU arm inside the aggregate production allocation.
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SEGMENT_PREFLIGHT:?set passed job-start preflight receipt}"
: "${ARM_ID:?set one exact D0-D4 arm}"
: "${ARM_NODELIST:?set the four-node arm nodelist}"
: "${ARM_OUTPUT_ROOT:?set the arm output root}"
: "${MASTER_ADDR:?set arm rendezvous host}"
: "${MASTER_PORT:?set arm rendezvous port}"

readarray -t contract < <("$HOST_PYTHON" - "$CAMPAIGN_MANIFEST" "$SEGMENT_PREFLIGHT" "$ARM_ID" <<'PY'
import json,sys
campaign=json.load(open(sys.argv[1])); preflight=json.load(open(sys.argv[2])); arm=sys.argv[3]
arms=campaign["scientific_contract"]["arms"]
if arm not in arms: raise SystemExit(f"unknown arm: {arm}")
if preflight.get("status") != "passed": raise SystemExit("segment preflight did not pass")
assets=campaign["assets"]
validation=json.load(open(assets["validation_manifest"]["path"]))
print(assets["megatron_dir"])
print(assets["tokenizer_dir"])
print(preflight.get("runtime_scientific_bundle") or assets["scientific_bundle"])
print(assets["efficiency_bundle"])
print(assets["schedule_manifest"]["path"])
print(preflight["load_roots"][arm])
print(preflight["start_iteration"])
print(preflight["end_iteration"])
print(assets["token_byte_lengths"])
print(campaign["scientific_contract"]["peak_lr"])
print(campaign["scientific_contract"]["minimum_lr"])
for panel in validation["panels"]:
    print(f"VALID\t{panel['name']}\t{panel['megatron_prefix']}")
PY
)
MEGATRON_DIR=${contract[0]}
TOKENIZER_DIR=${contract[1]}
SCIENTIFIC_BUNDLE=${contract[2]}
EFFICIENCY_BUNDLE=${contract[3]}
SCHEDULE_MANIFEST=${contract[4]}
LOAD_ROOT=${contract[5]}
START_ITERATION=${contract[6]}
END_ITERATION=${contract[7]}
TOKEN_BYTE_LENGTHS=${contract[8]}
PEAK_LR=${contract[9]}
MIN_LR=${contract[10]}
VALID_ARGS=()
for row in "${contract[@]:11}"; do
  IFS=$'\t' read -r kind name prefix <<<"$row"
  [[ "$kind" == VALID ]] || { echo "invalid validation contract row" >&2; exit 2; }
  VALID_ARGS+=("$name" "$prefix")
done
[[ "${#VALID_ARGS[@]}" -eq 26 ]] || { echo "expected 13 validation panels" >&2; exit 2; }

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
GPUS_PER_NODE=4
NNODES=4
WORLD_SIZE=16
GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=4
TOTAL_ITERATIONS=38496
TRAIN_SAMPLES=19709952
LR_WARMUP_SAMPLES=409600
LR_WSD_DECAY_SAMPLES=3941990
EVAL_ITERS=${EVAL_ITERS:-1}
[[ "$EVAL_ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_ITERS must be positive" >&2; exit 2; }
mkdir -p "$ARM_OUTPUT_ROOT/checkpoints" "$ARM_OUTPUT_ROOT/triggers"

COMPAT_DIR="$EFFICIENCY_BUNDLE/compat"
TRAIN_ENTRYPOINT="$SCIENTIFIC_BUNDLE/training/pretrain_scheduled_gpt.py"
[[ -f "$COMPAT_DIR/sitecustomize.py" && -f "$TRAIN_ENTRYPOINT" ]] || {
  echo "missing frozen production code" >&2; exit 2;
}
export PYTHONPATH="$COMPAT_DIR:$MEGATRON_DIR:$SCIENTIFIC_BUNDLE/training"
export MINI_SCHEDULE_MANIFEST="$SCHEDULE_MANIFEST"
export MINI_SCHEDULE_ARM="$ARM_ID"
export MINI_SCHEDULE_ALLOW_PREFIX=0
export MINI_TOKEN_BYTE_LENGTHS="$TOKEN_BYTE_LENGTHS"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO="^LL128"
export NCCL_NET_FORCE_FLUSH=0
export NCCL_RAS_ENABLE=0
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
unset NCCL_NET_PLUGIN

mapfile -t SAVE_ITERATIONS < <("$HOST_PYTHON" - "$CAMPAIGN_MANIFEST" "$START_ITERATION" "$END_ITERATION" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); start,end=map(int,sys.argv[2:])
print(*[i for i in c["evaluation"]["checkpoint_iterations"] if start < i <= end],sep="\n")
PY
)
export MINI_SCHEDULE_SAVE_ITERATIONS="$(IFS=,; echo "${SAVE_ITERATIONS[*]}")"
export MINI_SCHEDULE_EVAL_ITERATIONS="$MINI_SCHEDULE_SAVE_ITERATIONS"

TRAINING_CMD=(python3 "$TRAIN_ENTRYPOINT"
  --num-layers 20 --hidden-size 1024 --ffn-hidden-size 6144
  --num-attention-heads 16 --group-query-attention --num-query-groups 4
  --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 500000
  --make-vocab-size-divisible-by 256 --normalization RMSNorm --xielu --qk-layernorm
  --qknorm-impl apex --disable-bias-linear --attention-dropout 0.0 --hidden-dropout 0.0
  --init-method-std 0.02 --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE" --train-samples "$TRAIN_SAMPLES"
  --seq-length 4096 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999
  --ademamix-beta3 0.999 --ademamix-alpha 4.0
  --ademamix-beta3-warmup "$TOTAL_ITERATIONS" --ademamix-alpha-warmup "$TOTAL_ITERATIONS"
  --weight-decay 0.1 --clip-grad 0.1 --lr "$PEAK_LR" --min-lr "$MIN_LR"
  --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt
  --lr-wsd-decay-samples "$LR_WSD_DECAY_SAMPLES" --lr-warmup-samples "$LR_WARMUP_SAMPLES"
  --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --goldfish-loss --goldfish-k 50 --goldfish-h 50
  --reset-attention-mask --reset-position-ids --eod-mask-loss
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_DIR"
  --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4
  --dataloader-type single --seed 20260609 --log-throughput --log-interval 1
  --eval-interval 1000000 --eval-iters "$EVAL_ITERS" --extra-valid-data-path "${VALID_ARGS[@]}"
  --distributed-timeout-minutes 30 --trigger-path "$ARM_OUTPUT_ROOT/triggers"
  --save "$ARM_OUTPUT_ROOT/checkpoints" --save-interval 512 --ckpt-format torch_dist --async-save
  --load "$LOAD_ROOT" --auto-detect-ckpt-format --exit-interval "$END_ITERATION")
if [[ "${INITIAL_VALIDATION_ONLY:-0}" == 1 ]]; then
  TRAINING_CMD+=(--skip-train)
fi
printf '%q ' "${TRAINING_CMD[@]}" >"$ARM_OUTPUT_ROOT/training_command.sh"
printf '\n' >>"$ARM_OUTPUT_ROOT/training_command.sh"

cd "$MEGATRON_DIR"
uenv run "$UENV_IMAGE" --view=default -- \
  bash -c "export PYTHONPATH='$COMPAT_DIR'; exec python3 '$EFFICIENCY_BUNDLE/scripts/check_torch_dist_metadata_compat.py'"
uenv run "$UENV_IMAGE" --view=default -- \
  srun --exclusive --nodes="$NNODES" --nodelist="$ARM_NODELIST" \
  --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task=288 -lu \
  bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' MINI_SCHEDULE_MANIFEST='$MINI_SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM='$MINI_SCHEDULE_ARM' MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_SCHEDULE_SAVE_ITERATIONS='$MINI_SCHEDULE_SAVE_ITERATIONS' MINI_SCHEDULE_EVAL_ITERATIONS='$MINI_SCHEDULE_EVAL_ITERATIONS' MINI_TOKEN_BYTE_LENGTHS='$MINI_TOKEN_BYTE_LENGTHS'; exec torchrun --no-python --nnodes='$NNODES' --nproc_per_node='$GPUS_PER_NODE' --node_rank=\$SLURM_PROCID --master_addr='$MASTER_ADDR' --master_port='$MASTER_PORT' $(printf '%q ' "${TRAINING_CMD[@]}")"
