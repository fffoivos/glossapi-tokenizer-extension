#!/usr/bin/env bash
#SBATCH --job-name=mini_lr_fallback
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
: "${MEGATRON_DIR:?set pinned patched SwissAI Megatron root}"
: "${TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${INITIAL_CHECKPOINT_ROOT:?set converted tied-TD release root}"
: "${VALIDATION_MANIFEST:?set frozen 13-panel validation manifest}"
: "${TOKEN_BYTE_LENGTHS:?set frozen token-byte table}"
: "${SOURCE_SMOKE_ROOT:?set completed 3e-4 candidate smoke root}"
: "${CANDIDATE_ENDPOINT_LOG:?set completed 3e-4 endpoint validation log}"
: "${FALLBACK_ROOT:?set new fallback smoke root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
[[ ! -e "$FALLBACK_ROOT" ]] || { echo "refusing to replace $FALLBACK_ROOT" >&2; exit 2; }
mkdir -p "$FALLBACK_ROOT/fallback_1p5e4/checkpoints" "$FALLBACK_ROOT/fallback_1p5e4/triggers"

SCHEDULE_MANIFEST="$SOURCE_SMOKE_ROOT/schedule/schedule_manifest.json"
CANDIDATE_DRIVER="$SOURCE_SMOKE_ROOT/candidate_3e4/driver.log"
CANDIDATE_CHECKPOINTS="$SOURCE_SMOKE_ROOT/candidate_3e4/checkpoints"
INITIAL_LOG="$SOURCE_SMOKE_ROOT/initial_validation.log"
for required in "$SCHEDULE_MANIFEST" "$CANDIDATE_DRIVER" "$INITIAL_LOG" "$CANDIDATE_ENDPOINT_LOG" "$CANDIDATE_CHECKPOINTS/iter_0001024/.metadata"; do
  [[ -f "$required" ]] || { echo "missing candidate evidence: $required" >&2; exit 3; }
done

set +e
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/training/evaluate_common_stability_smoke.py" \
  --driver-log "$CANDIDATE_DRIVER" \
  --initial-validation-log "$INITIAL_LOG" \
  --endpoint-validation-log "$CANDIDATE_ENDPOINT_LOG" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --checkpoint-root "$CANDIDATE_CHECKPOINTS" \
  --peak-lr 3e-4 --process-exit-code 0 --output "$FALLBACK_ROOT/candidate_3e4_receipt.json"
candidate_verdict=$?
set -e
[[ "$candidate_verdict" -eq 1 ]] || { echo "fallback is permitted only after a semantic candidate failure" >&2; exit 4; }

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

output="$FALLBACK_ROOT/fallback_1p5e4"
train_cmd=(python3 "$TRAIN_ENTRYPOINT"
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
  --weight-decay 0.1 --clip-grad 0.1 --lr 1.5e-4 --min-lr 1.5e-5
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
printf '%q ' "${train_cmd[@]}" > "$output/training_command.sh"; printf '\n' >> "$output/training_command.sh"
cd "$MEGATRON_DIR"
set +e
uenv run "$UENV_IMAGE" --view=default -- srun --exclusive --nodes=4 --nodelist="$nodes" \
  --ntasks=4 --ntasks-per-node=1 --cpus-per-task=288 -lu \
  bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$master' MASTER_PORT=29483 MINI_SCHEDULE_MANIFEST='$SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM=stability_balanced MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$TOKEN_BYTE_LENGTHS'; exec torchrun --no-python --nnodes=4 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='$master' --master_port=29483 $(printf '%q ' "${train_cmd[@]}")" \
  > "$output/driver.log" 2>&1
train_rc=$?
set -e

endpoint_rc=99
if [[ "$train_rc" -eq 0 ]]; then
  endpoint_cmd=(python3 "$TRAIN_ENTRYPOINT"
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
    --weight-decay 0.1 --clip-grad 0.1 --lr 1.5e-4 --min-lr 1.5e-5
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
  printf '%q ' "${endpoint_cmd[@]}" > "$output/endpoint_validation_command.sh"; printf '\n' >> "$output/endpoint_validation_command.sh"
  set +e
  uenv run "$UENV_IMAGE" --view=default -- srun --exclusive --nodes=4 --nodelist="$nodes" \
    --ntasks=4 --ntasks-per-node=1 --cpus-per-task=288 -lu \
    bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MASTER_ADDR='$master' MASTER_PORT=29484 MINI_SCHEDULE_MANIFEST='$SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM=stability_balanced MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$TOKEN_BYTE_LENGTHS'; exec torchrun --no-python --nnodes=4 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='$master' --master_port=29484 $(printf '%q ' "${endpoint_cmd[@]}")" \
    > "$output/endpoint_validation.log" 2>&1
  endpoint_rc=$?
  set -e
else
  : > "$output/endpoint_validation.log"
fi
effective_rc=$train_rc
if [[ "$effective_rc" -eq 0 ]]; then effective_rc=$endpoint_rc; fi
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/training/evaluate_common_stability_smoke.py" \
  --driver-log "$output/driver.log" \
  --initial-validation-log "$INITIAL_LOG" \
  --endpoint-validation-log "$output/endpoint_validation.log" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --checkpoint-root "$output/checkpoints" \
  --peak-lr 1.5e-4 --process-exit-code "$effective_rc" --output "$output/receipt.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/training/finalize_stability_lr_selection.py" \
  --candidate-receipt "$FALLBACK_ROOT/candidate_3e4_receipt.json" \
  --fallback-receipt "$output/receipt.json" \
  --output "$FALLBACK_ROOT/lr_selection.json"
