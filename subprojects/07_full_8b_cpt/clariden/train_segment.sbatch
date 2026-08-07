#!/usr/bin/env bash
#SBATCH --job-name=full8b-mixed-cpt
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set frozen packed-data root}"
: "${FULL8_RUN_ROOT:?set immutable run root}"
: "${FULL8_START_ITERATION:?set exact segment start}"
: "${FULL8_END_ITERATION:?set exact segment end}"
: "${FULL8_LOAD_CHECKPOINT:?set init or resume checkpoint root}"

RECIPE=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
PROFILES=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
PROFILE_ID=${FULL8_EXECUTION_PROFILE:-dp32_16node}
SCHEDULE="$FULL8_STAGE_ROOT/schedules/schedule_manifest.json"
POOL_RECEIPT="$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json"
VALIDATION="$FULL8_STAGE_ROOT/validation/validation_manifest.json"
TOKEN_BYTES="$FULL8_STAGE_ROOT/validation/token_utf8_byte_lengths.npy"
TOKENIZER=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
MEGATRON_RUNTIME_RECEIPT=${FULL8_MEGATRON_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1.receipt.json}
TRAINING_CODE="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/training"
RUNTIME_COMPAT="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/runtime_compat"
SEGMENT_ROOT="$FULL8_RUN_ROOT/segments/updates_${FULL8_START_ITERATION}_${FULL8_END_ITERATION}"
if [[ -n "${FULL8_SEGMENT_ID:-}" ]]; then
  SEGMENT_ROOT="$FULL8_RUN_ROOT/segments/segment_${FULL8_SEGMENT_ID}/attempt_${FULL8_ATTEMPT:-0}_updates_${FULL8_START_ITERATION}_${FULL8_END_ITERATION}"
fi
if [[ "${FULL8_INITIAL_VALIDATION_ONLY:-0}" == 1 ]]; then
  SEGMENT_ROOT="$FULL8_RUN_ROOT/initial_validation"
fi
if [[ "${FULL8_CHECKPOINT_VALIDATION_ONLY:-0}" == 1 ]]; then
  : "${FULL8_CHECKPOINT_VALIDATION_OUTPUT:?set checkpoint validation output root}"
  SEGMENT_ROOT="$FULL8_CHECKPOINT_VALIDATION_OUTPUT"
fi
[[ "$(( ${FULL8_INITIAL_VALIDATION_ONLY:-0} + ${FULL8_CHECKPOINT_VALIDATION_ONLY:-0} ))" -le 1 ]] || {
  echo "initial and checkpoint validation-only modes are mutually exclusive" >&2; exit 2;
}
CHECKPOINT_ROOT="$FULL8_RUN_ROOT/checkpoints"
DATA_CACHE="$FULL8_RUN_ROOT/dataset_cache"
mkdir -p "$SEGMENT_ROOT" "$CHECKPOINT_ROOT" "$DATA_CACHE" "$FULL8_RUN_ROOT/triggers"
JOB_STARTED_EPOCH=$(date +%s)

# Trigger files are run-global because Megatron polls one directory. Preserve
# evidence from a previous attempt, then start this attempt with a clean edge.
for trigger in save exit; do
  if [[ -e "$FULL8_RUN_ROOT/triggers/$trigger" ]]; then
    mv "$FULL8_RUN_ROOT/triggers/$trigger" "$SEGMENT_ROOT/inherited_trigger_$trigger"
  fi
done

request_graceful_stop() {
  set +e
  touch "$FULL8_RUN_ROOT/triggers/save" "$FULL8_RUN_ROOT/triggers/exit"
  touch "$SEGMENT_ROOT/graceful_stop_requested"
  set -e
}
trap request_graceful_stop USR1 TERM INT

if [[ -n "${FULL8_SEGMENT_ID:-}" && -z "${FULL8_CODE_BUNDLE_RECEIPT:-}" ]]; then
  : "${FULL8_CODE_BUNDLE_RECEIPT:?production requires the executing bundle receipt}"
fi
if [[ -n "${FULL8_CODE_BUNDLE_RECEIPT:-}" ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
    --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
fi

if [[ "$FULL8_START_ITERATION" == 0 ]]; then
  canonical_initial=$(python3 - "$RECIPE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["initialization"]["megatron_tp2_checkpoint"])
PY
)
  [[ "$(realpath "$FULL8_LOAD_CHECKPOINT")" == "$(realpath "$canonical_initial")" ]] || {
    echo "initial checkpoint/recipe drift" >&2
    exit 2
  }
fi

if [[ -n "${FULL8_EXACT_LOAD_ITERATION:-}" ]]; then
  printf -v checkpoint_name 'iter_%07d' "$FULL8_EXACT_LOAD_ITERATION"
  source_checkpoint="$FULL8_LOAD_CHECKPOINT/$checkpoint_name"
  [[ -f "$source_checkpoint/.metadata" ]] || {
    echo "exact benchmark checkpoint is incomplete: $source_checkpoint" >&2; exit 2;
  }
  exact_view="$FULL8_RUN_ROOT/benchmark_load_views/${checkpoint_name}_for_${PROFILE_ID}"
  [[ ! -e "$exact_view" ]] || { echo "exact benchmark load view exists" >&2; exit 2; }
  mkdir -p "$exact_view"
  ln -s "$source_checkpoint" "$exact_view/$checkpoint_name"
  printf '%s\n' "$FULL8_EXACT_LOAD_ITERATION" >"$exact_view/latest_checkpointed_iteration.txt"
  FULL8_LOAD_CHECKPOINT="$exact_view"
fi

if [[ -n "${FULL8_SEGMENT_ID:-}" ]]; then
  : "${FULL8_SELECTED_PROFILE:?production requires selected profile receipt}"
  : "${FULL8_LAUNCH_GATE:?production requires passed launch gate}"
  python3 - "$FULL8_SELECTED_PROFILE" "$FULL8_LAUNCH_GATE" "$PROFILE_ID" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
if p.get("schema_version")!="apertus_full_8b_selected_execution_profile_v1" or p.get("status")!="frozen": raise SystemExit("selected profile receipt drift")
if p["selection"]["profile_id"]!=sys.argv[3]: raise SystemExit("selected profile id drift")
if g.get("schema_version")!="apertus_full_8b_launch_gate_v1" or g.get("status")!="passed": raise SystemExit("launch gate drift")
PY
fi

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_recipe.py" \
  --recipe "$RECIPE" --pool-receipt "$POOL_RECEIPT" --schedule-manifest "$SCHEDULE"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_megatron_runtime.py" \
  --receipt "$MEGATRON_RUNTIME_RECEIPT" --megatron-root "$MEGATRON"
[[ -s "$TOKENIZER/tokenizer.json" && -s "$TOKEN_BYTES" && -d "$FULL8_LOAD_CHECKPOINT" ]]
[[ -f "$RUNTIME_COMPAT/sitecustomize.py" ]] || {
  echo "missing frozen checkpoint compatibility shim" >&2; exit 2;
}

PROFILE_RECEIPT="$SEGMENT_ROOT/execution_profile.json"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_execution_profile.py" \
  --recipe "$RECIPE" --profiles "$PROFILES" --profile-id "$PROFILE_ID" \
  --output "$PROFILE_RECEIPT"
readarray -t profile < <(python3 - "$RECIPE" "$PROFILE_RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); receipt=json.load(open(sys.argv[2]))
if receipt.get("schema_version") != "apertus_full_8b_selected_execution_profile_v1" or receipt.get("status") != "frozen":
 raise SystemExit("execution-profile receipt drift")
v=receipt["selection"]
for key in ("nodes","world_size","data_parallel","gradient_accumulation_steps"):
 print(f"{key}\t{v[key]}")
print("boundaries\t" + ",".join(map(str,v["segment_boundaries"])))
print(f"scientific_digest\t{v['scientific_digest']}")
print(f"training_samples\t{r['batch_and_parallelism']['training_samples']}")
print(f"beta3_warmup_updates\t{r['optimization']['beta3_warmup_updates']}")
print(f"alpha_warmup_updates\t{r['optimization']['alpha_warmup_updates']}")
print(f"lr_decay_samples\t{r['optimization']['learning_rate']['cooldown_updates'] * r['batch_and_parallelism']['global_batch_sequences']}")
print(f"lr_warmup_samples\t{r['optimization']['learning_rate']['warmup_updates'] * r['batch_and_parallelism']['global_batch_sequences']}")
print(f"eval_interval\t{r['evaluation']['source_conditioned']['interval_updates']}")
PY
)
declare -A PROFILE=()
for row in "${profile[@]}"; do IFS=$'\t' read -r key value <<<"$row"; PROFILE[$key]=$value; done
NODES=${PROFILE[nodes]}
WORLD_SIZE=${PROFILE[world_size]}
[[ "$SLURM_NNODES" == "$NODES" ]] || { echo "allocation/profile node drift: $SLURM_NNODES != $NODES" >&2; exit 2; }

readarray -t contract < <(python3 - "$RECIPE" "$VALIDATION" "$FULL8_START_ITERATION" "$FULL8_END_ITERATION" "${PROFILE[boundaries]}" "${FULL8_BENCHMARK_MODE:-0}" "${FULL8_BENCHMARK_SAVE_ITERATIONS:-}" "${FULL8_RECOVERY_MODE:-0}" "${FULL8_CHECKPOINT_VALIDATION_ONLY:-0}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); v=json.load(open(sys.argv[2])); start,end=map(int,sys.argv[3:5])
b=list(map(int,sys.argv[5].split(','))); benchmark=sys.argv[6]=='1'; recovery=sys.argv[8]=='1'; validation_only=sys.argv[9]=='1'
if validation_only:
 if not 0 <= start < end <= max(b): raise SystemExit("invalid checkpoint validation range")
elif benchmark:
 if not (0 <= start < end <= 288): raise SystemExit("invalid benchmark range")
 saves=[int(x) for x in sys.argv[7].split(',') if x]
else:
 if recovery:
  if end not in b: raise SystemExit("recovery end is not a segment boundary")
  previous=b[b.index(end)-1]
  if not previous <= start < end: raise SystemExit("recovery start is outside segment")
 elif start not in b or end not in b or b.index(end)!=b.index(start)+1:
  raise SystemExit("invalid segment boundary")
 saves=r["evaluation"]["checkpoint_updates"]
if not validation_only:
 for i in saves:
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
export PYTHONPATH="$RUNTIME_COMPAT:$MEGATRON:$TRAINING_CODE"
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
  --micro-batch-size 2 --global-batch-size 1024 --train-samples "${PROFILE[training_samples]}"
  --seq-length 4096 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1
  --context-parallel-size 1 --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999
  --ademamix-beta3 0.999 --ademamix-alpha 4.0
  --ademamix-beta3-warmup "${PROFILE[beta3_warmup_updates]}" --ademamix-alpha-warmup "${PROFILE[alpha_warmup_updates]}"
  --weight-decay 0.1 --clip-grad 0.1 --lr 5.5e-5 --min-lr 5.5e-6
  --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt
  --lr-wsd-decay-samples "${PROFILE[lr_decay_samples]}" --lr-warmup-samples "${PROFILE[lr_warmup_samples]}" --lr-warmup-init 5.5e-6
  --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --goldfish-loss --goldfish-k 50 --goldfish-h 50
  --reset-attention-mask --reset-position-ids --eod-mask-loss
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER"
  --data-cache-path "$DATA_CACHE"
  --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4
  --dataloader-type single --seed 20260609 --log-throughput --log-params-norm --log-interval 1
  --eval-interval "${PROFILE[eval_interval]}" --eval-iters 1 --extra-valid-data-path "${VALID_ARGS[@]}"
  --manual-gc --manual-gc-interval 500 --exit-signal-handler
  --distributed-timeout-minutes 60 --trigger-path "$FULL8_RUN_ROOT/triggers"
  --save "$CHECKPOINT_ROOT" --save-interval 1000000 --ckpt-format torch_dist --async-save
  --load "$FULL8_LOAD_CHECKPOINT" --auto-detect-ckpt-format
  --ckpt-fully-parallel-load --dist-ckpt-strictness assume_ok_unexpected
  --override-opt_param-scheduler --exit-interval "$FULL8_END_ITERATION")
if [[ "$FULL8_START_ITERATION" == 0 && "${FULL8_INITIAL_VALIDATION_ONLY:-0}" != 1 && "${FULL8_CHECKPOINT_VALIDATION_ONLY:-0}" != 1 ]]; then
  cmd+=(--no-load-optim --no-load-rng)
fi
if [[ "${FULL8_INITIAL_VALIDATION_ONLY:-0}" == 1 || "${FULL8_CHECKPOINT_VALIDATION_ONLY:-0}" == 1 ]]; then
  cmd+=(--skip-train --no-load-optim --no-load-rng)
fi
printf '%q ' "${cmd[@]}" >"$SEGMENT_ROOT/training_command.sh"; printf '\n' >>"$SEGMENT_ROOT/training_command.sh"

cd "$MEGATRON"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  srun --nodes="$NODES" --ntasks="$WORLD_SIZE" --ntasks-per-node=4 \
  --gpus-per-task=1 --cpus-per-task=72 --cpu-bind=cores -l \
  bash -c 'printf "AFFINITY host=%s rank=%s local_gpu=%s cpu=%s\n" "$(hostname)" "$SLURM_PROCID" "${CUDA_VISIBLE_DEVICES:-unset}" "$(taskset -pc $$ | sed "s/.*: //")"' \
  | tee "$SEGMENT_ROOT/affinity.log"
[[ "$(grep -c 'AFFINITY host=' "$SEGMENT_ROOT/affinity.log")" -eq "$WORLD_SIZE" ]] || { echo "affinity rank count drift" >&2; exit 2; }
(
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task=288 --cpu-bind=cores -lu \
    bash -c "unset NCCL_NET_PLUGIN; export MINI_SCHEDULE_MANIFEST='$MINI_SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM='$MINI_SCHEDULE_ARM' MINI_SCHEDULE_ALLOW_PREFIX=0 MINI_TOKEN_BYTE_LENGTHS='$MINI_TOKEN_BYTE_LENGTHS' MINI_SCHEDULE_SAVE_ITERATIONS='$MINI_SCHEDULE_SAVE_ITERATIONS' MINI_SCHEDULE_EVAL_ITERATIONS='$MINI_SCHEDULE_EVAL_ITERATIONS'; exec uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc 'export PYTHONPATH='\''$PYTHONPATH'\''; exec torchrun --no-python --nnodes=$NODES --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='\''$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)'\'' --master_port=29600 $(printf '%q ' "${cmd[@]}")'" \
    2>&1 | tee "$SEGMENT_ROOT/training.log"
) &
TRAIN_PIPELINE_PID=$!
# Bash defers a trap while synchronously waiting for a foreground process.
# Polling a background pipeline lets B:USR1 create save/exit triggers promptly.
while kill -0 "$TRAIN_PIPELINE_PID" 2>/dev/null; do sleep 2; done
if wait "$TRAIN_PIPELINE_PID"; then
  TRAIN_PIPELINE_RC=0
else
  TRAIN_PIPELINE_RC=$?
fi
[[ "$TRAIN_PIPELINE_RC" -eq 0 ]] || exit "$TRAIN_PIPELINE_RC"

JOB_FINISHED_EPOCH=$(date +%s)
python3 - "$SEGMENT_ROOT/training_job_receipt.json" "$PROFILE_ID" "$FULL8_START_ITERATION" "$FULL8_END_ITERATION" "$JOB_STARTED_EPOCH" "$JOB_FINISHED_EPOCH" "$SLURM_JOB_ID" "$NODES" "$WORLD_SIZE" "${PROFILE[data_parallel]}" "${PROFILE[gradient_accumulation_steps]}" "${PROFILE[scientific_digest]}" "$SEGMENT_ROOT/training.log" "$SEGMENT_ROOT/graceful_stop_requested" <<'PY'
import datetime,json,os,re,sys,tempfile
out,profile,start,end,begun,finished,job,nodes,world,dp,acc,digest,log,marker=sys.argv[1:]
latest=None
pattern=re.compile(r"iteration\s+(\d+)\s*/")
if os.path.isfile(log):
 for line in open(log,errors="replace"):
  match=pattern.search(line)
  if match: latest=int(match.group(1))
graceful=os.path.isfile(marker) and latest is not None and latest<int(end)
payload={"schema_version":"apertus_full_8b_training_job_v1","status":"gracefully_stopped" if graceful else "completed","completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":job,"profile_id":profile,"start_iteration":int(start),"end_iteration":int(end),"requested_end_iteration":int(end),"last_logged_iteration":latest,"completed_to_requested_boundary":not graceful,"elapsed_seconds":int(finished)-int(begun),"nodes":int(nodes),"world_size":int(world),"data_parallel":int(dp),"gradient_accumulation_steps":int(acc),"scientific_digest":digest}
fd,tmp=tempfile.mkstemp(prefix=".training-job.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY

if [[ "${FULL8_INITIAL_VALIDATION_ONLY:-0}" == 1 ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/finalize_initial_validation.py" \
    --code-root "$FULL8_CODE_ROOT" --validation-manifest "$VALIDATION" \
    --log "$SEGMENT_ROOT/training.log" \
    --output "$SEGMENT_ROOT/initial_validation_receipt.json"
fi
if [[ "${FULL8_CHECKPOINT_VALIDATION_ONLY:-0}" == 1 ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/finalize_source_validation_checkpoint.py" \
    --code-root "$FULL8_CODE_ROOT" --validation-manifest "$VALIDATION" \
    --iteration "$FULL8_START_ITERATION" --log "$SEGMENT_ROOT/training.log" \
    --output "$SEGMENT_ROOT/source_validation_receipt.json"
fi
