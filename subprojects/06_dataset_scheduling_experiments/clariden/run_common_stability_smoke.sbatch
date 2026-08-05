#!/usr/bin/env bash
#SBATCH --job-name=mini_lr_smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=04:00:00

set -euo pipefail
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${EFFICIENCY_BUNDLE:?set immutable efficiency bundle}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${INITIAL_CHECKPOINT_ROOT:?set converted tied-TD iteration-0 root}"
: "${PACKED_CORPUS_RECEIPT:?set frozen packed corpus receipt}"
: "${VALIDATION_MANIFEST:?set frozen 13-panel validation manifest}"
: "${TOKEN_BYTE_LENGTHS:?set frozen token-byte table}"
: "${SMOKE_ROOT:?set new common stability-smoke root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
[[ ! -e "$SMOKE_ROOT" ]] || { echo "refusing to replace $SMOKE_ROOT" >&2; exit 2; }
mkdir -p "$SMOKE_ROOT"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCIENTIFIC_BUNDLE/dataset/build_stability_smoke_schedule.py" \
  --packed-receipt "$PACKED_CORPUS_RECEIPT" --output-dir "$SMOKE_ROOT/schedule"
SCHEDULE_MANIFEST="$SMOKE_ROOT/schedule/schedule_manifest.json"
readarray -t validation < <("$HOST_PYTHON" - "$VALIDATION_MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d["status"] == "frozen" and len(d["panels"]) == 13
for row in d["panels"]: print(f"{row['name']}\t{row['megatron_prefix']}")
PY
)
VALID_ARGS=()
for row in "${validation[@]}"; do IFS=$'\t' read -r name prefix <<<"$row"; VALID_ARGS+=("$name" "$prefix"); done

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)
master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
COMPAT_DIR="$EFFICIENCY_BUNDLE/compat"
TRAIN_ENTRYPOINT="$SCIENTIFIC_BUNDLE/training/pretrain_scheduled_gpt.py"
export PYTHONPATH="$COMPAT_DIR:$MEGATRON_DIR:$SCIENTIFIC_BUNDLE/training"
export MINI_SCHEDULE_MANIFEST="$SCHEDULE_MANIFEST" MINI_SCHEDULE_ARM=stability_balanced MINI_SCHEDULE_ALLOW_PREFIX=0
export MINI_TOKEN_BYTE_LENGTHS="$TOKEN_BYTE_LENGTHS" MINI_SCHEDULE_SAVE_ITERATIONS="" MINI_SCHEDULE_EVAL_ITERATIONS=""
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1 NCCL_PROTO="^LL128" NCCL_NET_FORCE_FLUSH=0 NCCL_RAS_ENABLE=0
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=16384 FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_RDZV_GET_MIN=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
unset NCCL_NET_PLUGIN

run_initial_validation() {
  local log="$SMOKE_ROOT/initial_validation.log"
  local cmd=(python3 "$TRAIN_ENTRYPOINT"
    --num-layers 20 --hidden-size 1024 --ffn-hidden-size 6144
    --num-attention-heads 16 --group-query-attention --num-query-groups 4
    --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 500000
    --make-vocab-size-divisible-by 256 --normalization RMSNorm --xielu --qk-layernorm
    --qknorm-impl apex --disable-bias-linear --attention-dropout 0.0 --hidden-dropout 0.0
    --init-method-std 0.02 --micro-batch-size 4 --global-batch-size 512
    --train-samples 19709952 --seq-length 4096
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999 --ademamix-beta3 0.999 --ademamix-alpha 4.0
    --ademamix-beta3-warmup 38496 --ademamix-alpha-warmup 38496
    --weight-decay 0.1 --clip-grad 0.1 --lr 3e-4 --min-lr 3e-5
    --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt --lr-wsd-decay-samples 3941990 --lr-warmup-samples 409600
    --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
    --goldfish-loss --goldfish-k 50 --goldfish-h 50
    --reset-attention-mask --reset-position-ids --eod-mask-loss
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_DIR"
    --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4 --dataloader-type single
    --seed 20260609 --log-throughput --log-interval 1
    --skip-train --eval-iters 1 --extra-valid-data-path "${VALID_ARGS[@]}"
    --distributed-timeout-minutes 30 --load "$INITIAL_CHECKPOINT_ROOT" --auto-detect-ckpt-format)
  printf '%q ' "${cmd[@]}" > "$SMOKE_ROOT/initial_validation_command.sh"
  printf '\n' >> "$SMOKE_ROOT/initial_validation_command.sh"
  cd "$MEGATRON_DIR"
  uenv run "$UENV_IMAGE" --view=default -- srun --exclusive --nodes=4 --nodelist="$nodes" \
    --ntasks=4 --ntasks-per-node=1 --cpus-per-task=288 -lu \
    bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$master' MASTER_PORT=29479 MINI_SCHEDULE_MANIFEST='$SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM=stability_balanced MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$TOKEN_BYTE_LENGTHS'; exec torchrun --no-python --nnodes=4 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='$master' --master_port=29479 $(printf '%q ' "${cmd[@]}")" \
    > "$log" 2>&1
}

run_endpoint_validation() {
  local output=$1
  local peak=$2
  local minimum=$3
  local log="$output/endpoint_validation.log"
  local cmd=(python3 "$TRAIN_ENTRYPOINT"
    --num-layers 20 --hidden-size 1024 --ffn-hidden-size 6144
    --num-attention-heads 16 --group-query-attention --num-query-groups 4
    --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 500000
    --make-vocab-size-divisible-by 256 --normalization RMSNorm --xielu --qk-layernorm
    --qknorm-impl apex --disable-bias-linear --attention-dropout 0.0 --hidden-dropout 0.0
    --init-method-std 0.02 --micro-batch-size 4 --global-batch-size 512
    --train-samples 19709952 --seq-length 4096
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999 --ademamix-beta3 0.999 --ademamix-alpha 4.0
    --ademamix-beta3-warmup 38496 --ademamix-alpha-warmup 38496
    --weight-decay 0.1 --clip-grad 0.1 --lr "$peak" --min-lr "$minimum"
    --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt --lr-wsd-decay-samples 3941990 --lr-warmup-samples 409600
    --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
    --goldfish-loss --goldfish-k 50 --goldfish-h 50
    --reset-attention-mask --reset-position-ids --eod-mask-loss
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_DIR"
    --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4 --dataloader-type single
    --seed 20260609 --log-throughput --log-interval 1
    --skip-train --eval-iters 1 --extra-valid-data-path "${VALID_ARGS[@]}"
    --distributed-timeout-minutes 30 --load "$output/checkpoints" --auto-detect-ckpt-format)
  printf '%q ' "${cmd[@]}" > "$output/endpoint_validation_command.sh"
  printf '\n' >> "$output/endpoint_validation_command.sh"
  cd "$MEGATRON_DIR"
  uenv run "$UENV_IMAGE" --view=default -- srun --exclusive --nodes=4 --nodelist="$nodes" \
    --ntasks=4 --ntasks-per-node=1 --cpus-per-task=288 -lu \
    bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$master' MASTER_PORT=29481 MINI_SCHEDULE_MANIFEST='$SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM=stability_balanced MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$TOKEN_BYTE_LENGTHS'; exec torchrun --no-python --nnodes=4 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='$master' --master_port=29481 $(printf '%q ' "${cmd[@]}")" \
    > "$log" 2>&1
}

run_candidate() {
  local label=$1 peak=$2 minimum=$3
  local output="$SMOKE_ROOT/$label"
  mkdir -p "$output/checkpoints" "$output/triggers"
  local cmd=(python3 "$TRAIN_ENTRYPOINT"
    --num-layers 20 --hidden-size 1024 --ffn-hidden-size 6144
    --num-attention-heads 16 --group-query-attention --num-query-groups 4
    --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 500000
    --make-vocab-size-divisible-by 256 --normalization RMSNorm --xielu --qk-layernorm
    --qknorm-impl apex --disable-bias-linear --attention-dropout 0.0 --hidden-dropout 0.0
    --init-method-std 0.02 --micro-batch-size 4 --global-batch-size 512
    --train-samples 19709952 --seq-length 4096
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999 --ademamix-beta3 0.999 --ademamix-alpha 4.0
    --ademamix-beta3-warmup 38496 --ademamix-alpha-warmup 38496
    --weight-decay 0.1 --clip-grad 0.1 --lr "$peak" --min-lr "$minimum"
    --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt --lr-wsd-decay-samples 3941990 --lr-warmup-samples 409600
    --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
    --goldfish-loss --goldfish-k 50 --goldfish-h 50
    --reset-attention-mask --reset-position-ids --eod-mask-loss
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_DIR"
    --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4 --dataloader-type single
    --seed 20260609 --log-throughput --log-interval 1
    --eval-interval 1000000 --eval-iters 1 --extra-valid-data-path "${VALID_ARGS[@]}"
    --distributed-timeout-minutes 30 --trigger-path "$output/triggers"
    --save "$output/checkpoints" --save-interval 1024 --ckpt-format torch_dist --async-save
    --load "$INITIAL_CHECKPOINT_ROOT" --auto-detect-ckpt-format --exit-interval 1024)
  printf '%q ' "${cmd[@]}" > "$output/training_command.sh"; printf '\n' >> "$output/training_command.sh"
  set +e
  cd "$MEGATRON_DIR"
  uenv run "$UENV_IMAGE" --view=default -- srun --exclusive --nodes=4 --nodelist="$nodes" \
    --ntasks=4 --ntasks-per-node=1 --cpus-per-task=288 -lu \
    bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$master' MASTER_PORT=29480 MINI_SCHEDULE_MANIFEST='$SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM=stability_balanced MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$TOKEN_BYTE_LENGTHS'; exec torchrun --no-python --nnodes=4 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='$master' --master_port=29480 $(printf '%q ' "${cmd[@]}")" \
    > "$output/driver.log" 2>&1
  rc=$?
  set -e
  endpoint_rc=99
  if [[ "$rc" -eq 0 ]]; then
    set +e
    run_endpoint_validation "$output" "$peak" "$minimum"
    endpoint_rc=$?
    set -e
  else
    : > "$output/endpoint_validation.log"
  fi
  effective_rc=$rc
  if [[ "$effective_rc" -eq 0 ]]; then effective_rc=$endpoint_rc; fi
  set +e
  "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/training/evaluate_common_stability_smoke.py" \
    --driver-log "$output/driver.log" \
    --initial-validation-log "$SMOKE_ROOT/initial_validation.log" \
    --endpoint-validation-log "$output/endpoint_validation.log" \
    --validation-manifest "$VALIDATION_MANIFEST" \
    --checkpoint-root "$output/checkpoints" \
    --peak-lr "$peak" --process-exit-code "$effective_rc" --output "$output/receipt.json"
  verdict=$?
  set -e
  return "$verdict"
}

fallback_receipt=()
run_initial_validation
if run_candidate candidate_3e4 3e-4 3e-5; then
  :
else
  run_candidate fallback_1p5e4 1.5e-4 1.5e-5
  fallback_receipt=(--fallback-receipt "$SMOKE_ROOT/fallback_1p5e4/receipt.json")
fi
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/training/finalize_stability_lr_selection.py" \
  --candidate-receipt "$SMOKE_ROOT/candidate_3e4/receipt.json" \
  "${fallback_receipt[@]}" --output "$SMOKE_ROOT/lr_selection.json"
