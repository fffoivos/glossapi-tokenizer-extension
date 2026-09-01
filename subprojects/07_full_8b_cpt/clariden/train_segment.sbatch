#!/usr/bin/env bash
#SBATCH --job-name=full8b-mixed-cpt
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set frozen packed-data root}"
: "${FULL8_RUN_ROOT:?set immutable run root}"
: "${FULL8_START_ITERATION:?set exact segment start}"
: "${FULL8_END_ITERATION:?set exact segment end}"
: "${FULL8_LOAD_CHECKPOINT:?set init or resume checkpoint root}"

RECIPE="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json"
SCHEDULE="$FULL8_STAGE_ROOT/schedules/schedule_manifest.json"
POOL_RECEIPT="$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json"
VALIDATION="$FULL8_STAGE_ROOT/validation/validation_manifest.json"
TOKEN_BYTES="$FULL8_STAGE_ROOT/validation/token_utf8_byte_lengths.npy"
TOKENIZER=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
MEGATRON_RUNTIME_RECEIPT=${FULL8_MEGATRON_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1.receipt.json}
TRAINING_CODE="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/training"
SEGMENT_ROOT="$FULL8_RUN_ROOT/segments/updates_${FULL8_START_ITERATION}_${FULL8_END_ITERATION}"
if [[ "${FULL8_INITIAL_VALIDATION_ONLY:-0}" == 1 ]]; then
  SEGMENT_ROOT="$FULL8_RUN_ROOT/initial_validation"
fi
CHECKPOINT_ROOT="$FULL8_RUN_ROOT/checkpoints"
mkdir -p "$SEGMENT_ROOT" "$CHECKPOINT_ROOT" "$FULL8_RUN_ROOT/triggers"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_recipe.py" \
  --recipe "$RECIPE" --pool-receipt "$POOL_RECEIPT" --schedule-manifest "$SCHEDULE"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_megatron_runtime.py" \
  --receipt "$MEGATRON_RUNTIME_RECEIPT" --megatron-root "$MEGATRON"
[[ -s "$TOKENIZER/tokenizer.json" && -s "$TOKEN_BYTES" && -d "$FULL8_LOAD_CHECKPOINT" ]]

readarray -t contract < <(python3 - "$RECIPE" "$VALIDATION" "$FULL8_START_ITERATION" "$FULL8_END_ITERATION" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); v=json.load(open(sys.argv[2])); start,end=map(int,sys.argv[3:])
b=r["segments"]["boundaries"]
if start not in b or end not in b or b.index(end)!=b.index(start)+1: raise SystemExit("invalid segment boundary")
for i in r["evaluation"]["checkpoint_updates"]:
 if start < i <= end: print(f"SAVE\t{i}")
for p in v["panels"]: print(f"VALID\t{p['name']}\t{p['megatron_prefix']}")
PY
)
SAVE_ITERATIONS=()
VALID_ARGS=()
for row in "${contract[@]}"; do
  IFS=$'\t' read -r kind first second <<<"$row"
  case "$kind" in
    SAVE) SAVE_ITERATIONS+=("$first") ;;
    VALID) VALID_ARGS+=("$first" "$second") ;;
    *) echo "invalid training contract row: $row" >&2; exit 2 ;;
  esac
done
[[ "${#VALID_ARGS[@]}" -eq 26 ]] || { echo "expected exactly 13 validation panels" >&2; exit 2; }
export MINI_SCHEDULE_MANIFEST="$SCHEDULE"
export MINI_SCHEDULE_ARM=D0_mixed
export MINI_SCHEDULE_ALLOW_PREFIX=0
export MINI_TOKEN_BYTE_LENGTHS="$TOKEN_BYTES"
export MINI_SCHEDULE_SAVE_ITERATIONS="$(IFS=,; echo "${SAVE_ITERATIONS[*]}")"
export MINI_SCHEDULE_EVAL_ITERATIONS="$MINI_SCHEDULE_SAVE_ITERATIONS"
export PYTHONPATH="$MEGATRON:$TRAINING_CODE"
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

cmd=(python3 "$TRAINING_CODE/pretrain_scheduled_gpt.py"
  --num-layers 32 --hidden-size 4096 --ffn-hidden-size 21504
  --num-attention-heads 32 --group-query-attention --num-query-groups 8
  --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 500000
  --use-rope-scaling --rope-scaling-factor 8.0
  --make-vocab-size-divisible-by 256 --normalization RMSNorm --xielu --qk-layernorm
  --qknorm-impl apex --disable-bias-linear --untie-embeddings-and-output-weights
  --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.008944
  --micro-batch-size 2 --global-batch-size 1024 --train-samples 19709952
  --seq-length 4096 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999
  --ademamix-beta3 0.999 --ademamix-alpha 4.0
  --ademamix-beta3-warmup 19248 --ademamix-alpha-warmup 19248
  --weight-decay 0.1 --clip-grad 0.1 --lr 5.5e-5 --min-lr 5.5e-6
  --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt
  --lr-wsd-decay-samples 3942400 --lr-warmup-samples 409600 --lr-warmup-init 5.5e-6
  --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --goldfish-loss --goldfish-k 50 --goldfish-h 50
  --reset-attention-mask --reset-position-ids --eod-mask-loss
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER"
  --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4
  --dataloader-type single --seed 20260609 --log-throughput --log-params-norm --log-interval 1
  --eval-interval 25 --eval-iters 1 --extra-valid-data-path "${VALID_ARGS[@]}"
  --manual-gc --manual-gc-interval 500 --exit-signal-handler
  --distributed-timeout-minutes 60 --trigger-path "$FULL8_RUN_ROOT/triggers"
  --save "$CHECKPOINT_ROOT" --save-interval 1000000 --ckpt-format torch_dist --async-save
  --load "$FULL8_LOAD_CHECKPOINT" --auto-detect-ckpt-format
  --ckpt-fully-parallel-load --dist-ckpt-strictness assume_ok_unexpected
  --override-opt_param-scheduler --exit-interval "$FULL8_END_ITERATION")
if [[ "$FULL8_START_ITERATION" == 0 ]]; then cmd+=(--no-load-optim --no-load-rng); fi
if [[ "${FULL8_INITIAL_VALIDATION_ONLY:-0}" == 1 ]]; then cmd+=(--skip-train); fi
printf '%q ' "${cmd[@]}" >"$SEGMENT_ROOT/training_command.sh"; printf '\n' >>"$SEGMENT_ROOT/training_command.sh"

cd "$MEGATRON"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  srun --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=288 -lu \
  bash -c "unset NCCL_NET_PLUGIN; export PYTHONPATH='$PYTHONPATH' MINI_SCHEDULE_MANIFEST='$MINI_SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM='$MINI_SCHEDULE_ARM' MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$MINI_TOKEN_BYTE_LENGTHS' MINI_SCHEDULE_SAVE_ITERATIONS='$MINI_SCHEDULE_SAVE_ITERATIONS' MINI_SCHEDULE_EVAL_ITERATIONS='$MINI_SCHEDULE_EVAL_ITERATIONS'; exec torchrun --no-python --nnodes=16 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)' --master_port=29600 $(printf '%q ' "${cmd[@]}")" \
  2>&1 | tee "$SEGMENT_ROOT/training.log"

if [[ "${FULL8_INITIAL_VALIDATION_ONLY:-0}" == 1 ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/finalize_initial_validation.py" \
    --code-root "$FULL8_CODE_ROOT" --validation-manifest "$VALIDATION" \
    --log "$SEGMENT_ROOT/training.log" \
    --output "$SEGMENT_ROOT/initial_validation_receipt.json"
fi
