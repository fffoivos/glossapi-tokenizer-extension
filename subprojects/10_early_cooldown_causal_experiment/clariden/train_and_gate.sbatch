#!/usr/bin/env bash
#SBATCH --job-name=full8-early-wsd
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --switches=1
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
: "${EARLY_CODE_ROOT:?set immutable scientific code root}"
: "${EARLY_CODE_BUNDLE_RECEIPT:?set immutable scientific bundle receipt}"
: "${EARLY_CONTRACT:?set owner-authorized experiment contract}"
: "${EARLY_BRANCH_RECIPE:?set debug-frozen branch recipe}"
: "${EARLY_LAUNCH_GATE:?set passing launch gate}"
: "${EARLY_OPERATIONAL_GATE:?set passing operational gate}"
: "${EARLY_RUN_ROOT:?set one new run root}"
: "${EARLY_PHASE:?set branch}"
: "${EARLY_TRAIN_LEAF_SWITCH:?set hard-selected training leaf}"
[[ "$EARLY_PHASE" == branch ]] || { echo "invalid phase: $EARLY_PHASE" >&2; exit 2; }

SUBPROJECT="$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment"
TRAINING_CODE="$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/training"
RUNTIME_COMPAT="$EARLY_CODE_ROOT/subprojects/07_full_8b_cpt/runtime_compat"
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
MEGATRON_RUNTIME_RECEIPT=${FULL8_MEGATRON_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1.receipt.json}
TOKENIZER=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}

mapfile -t paths < <(/usr/bin/python3.11 - "$EARLY_CONTRACT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d['parent']['run_root'])
print(d['data']['stage_root'])
print(d['data']['schedule_manifest']['path'])
print(d['data']['validation_manifest']['path'])
PY
)
PARENT_RUN=${paths[0]}
SCHEDULE=${paths[2]}
VALIDATION=${paths[3]}
TOKEN_BYTES="$(dirname "$VALIDATION")/token_utf8_byte_lengths.npy"
SOURCE_ITERATION=9536
SOURCE_CHECKPOINT="$PARENT_RUN/checkpoints/iter_0009536"
SOURCE_RECEIPT=$(/usr/bin/python3.11 -c 'import json,sys;print(json.load(open(sys.argv[1]))["parent"]["checkpoint_receipt"]["path"])' "$EARLY_CONTRACT")
CHECKPOINT_ROOT="$EARLY_RUN_ROOT/checkpoints"
DATA_CACHE="$EARLY_RUN_ROOT/dataset_cache"
TRIGGERS="$EARLY_RUN_ROOT/triggers"
CONTROL_ROOT="$EARLY_RUN_ROOT/${EARLY_PHASE}_control"
TRAIN_ROOT="$EARLY_RUN_ROOT/$EARLY_PHASE"
BRANCH_LOG="$TRAIN_ROOT/job_${SLURM_JOB_ID}.log"
RECEIPT="$EARLY_RUN_ROOT/training_receipts/${EARLY_PHASE}_job_${SLURM_JOB_ID}.json"
RECOVERY=${EARLY_RECOVERY_MODE:-0}
[[ "$RECOVERY" == 0 || "$RECOVERY" == 1 ]] || { echo "invalid recovery mode" >&2; exit 2; }
[[ "$SLURM_NNODES" == 16 ]] || { echo "requires exactly 16 nodes" >&2; exit 2; }
mkdir -p "$CHECKPOINT_ROOT" "$DATA_CACHE" "$TRIGGERS" "$TRAIN_ROOT" "$EARLY_RUN_ROOT/training_receipts"
START_EPOCH=$(date +%s)

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$EARLY_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$EARLY_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_megatron_runtime.py" \
  --receipt "$MEGATRON_RUNTIME_RECEIPT" --megatron-root "$MEGATRON"
/usr/bin/python3.11 "$SUBPROJECT/scripts/verify_launch_gate.py" --gate "$EARLY_LAUNCH_GATE"
/usr/bin/python3.11 - "$EARLY_LAUNCH_GATE" "$EARLY_CONTRACT" "$EARLY_BRANCH_RECIPE" "$EARLY_CODE_BUNDLE_RECEIPT" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
gate=json.load(open(sys.argv[1]))
if gate.get('schema_version')!='apertus_full8_early_cooldown_launch_gate_v1' or gate.get('status')!='passed' or not all(gate.get('checks',{}).values()): raise SystemExit('launch gate is not passed')
for key,path in [('contract',sys.argv[2]),('branch_recipe',sys.argv[3]),('code_bundle_receipt',sys.argv[4])]:
 if Path(gate[key]['path']).resolve()!=Path(path).resolve() or gate[key]['sha256']!=sha(path): raise SystemExit(f'{key} binding drift')
PY
/usr/bin/python3.11 - "$EARLY_OPERATIONAL_GATE" "$EARLY_LAUNCH_GATE" <<'PY'
import hashlib,json,sys
from pathlib import Path
d=json.load(open(sys.argv[1])); raw=Path(sys.argv[2]).read_bytes()
if d.get('schema_version')!='apertus_full8_early_cooldown_operational_gate_v1' or d.get('status')!='passed' or d.get('launch_gate',{}).get('sha256')!=hashlib.sha256(raw).hexdigest(): raise SystemExit('operational gate drift')
PY
[[ -f "$SCHEDULE" && -f "$VALIDATION" && -f "$TOKEN_BYTES" && -s "$TOKENIZER/tokenizer.json" ]]
[[ -f "$SOURCE_CHECKPOINT/.metadata" && -f "$SOURCE_CHECKPOINT/common.pt" && -f "$SOURCE_RECEIPT" ]]

# Let Slurm select the eligible topology leaf, then fail closed unless all 16
# nodes are on exactly one level-0 leaf. The leaf identity is evidence, not an
# input to the scientific experiment.
mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
declare -A allocated=()
for node in "${allocated_nodes[@]}"; do allocated["$node"]=1; done
leaves=()
while IFS='|' read -r leaf nodes; do
  hit=0
  while IFS= read -r node; do [[ -n "${allocated[$node]:-}" ]] && hit=1 && break; done < <(scontrol show hostnames "$nodes")
  [[ "$hit" == 1 ]] && leaves+=("$leaf")
done < <(scontrol show topology | awk '$2=="Level=0" {a=$1;sub(/^SwitchName=/,"",a);b=$4;sub(/^Nodes=/,"",b);print a"|"b}')
fail_before_training() {
  local status=$1 rc=$2
  local joined
  joined=$(IFS=:; echo "${leaves[*]:-none}")
  /usr/bin/python3.11 "$SUBPROJECT/scripts/write_training_receipt.py" --output "$RECEIPT" --status "$status" --phase "$EARLY_PHASE" --job-id "$SLURM_JOB_ID" --leaf-switch "$joined" --started-epoch "$START_EPOCH" --finished-epoch "$(date +%s)"
  exit "$rc"
}
[[ ${#leaves[@]} == 1 ]] || {
  echo "allocation is not contained in exactly one leaf: observed=${leaves[*]:-none}" >&2
  fail_before_training topology_rejected 2
}
[[ "${leaves[0]}" == "$EARLY_TRAIN_LEAF_SWITCH" ]] || {
  echo "allocation leaf drift: expected=$EARLY_TRAIN_LEAF_SWITCH observed=${leaves[0]}" >&2
  fail_before_training topology_rejected 2
}

if [[ "$RECOVERY" == 0 ]]; then
  [[ ! -e "$CHECKPOINT_ROOT/iter_0009536" ]] || { echo "source checkpoint view already exists" >&2; exit 2; }
  ln -s "$SOURCE_CHECKPOINT" "$CHECKPOINT_ROOT/iter_0009536"
  [[ "$(realpath "$CHECKPOINT_ROOT/iter_0009536")" == "$(realpath "$SOURCE_CHECKPOINT")" ]]
  printf '9536\n' >"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
  [[ ! -e "$CONTROL_ROOT" && ! -e "$RECEIPT" ]] || { echo "fresh branch artifacts already exist" >&2; exit 2; }
  mkdir -p "$CONTROL_ROOT"
else
  /usr/bin/python3.11 - "$CONTROL_ROOT/sandwich_restart_control_receipt.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get('status')=='passed' and all(d.get('checks',{}).values())
PY
fi

for trigger in save exit; do [[ -e "$TRIGGERS/$trigger" ]] && mv "$TRIGGERS/$trigger" "$TRAIN_ROOT/inherited_${trigger}_$(date +%s)"; done
PHASE=setup
request_graceful_stop() {
  set +e
  touch "$TRAIN_ROOT/slurm_signal_received"
  if [[ "$PHASE" == branch || "$PHASE" == intervention_probe ]]; then touch "$TRIGGERS/save" "$TRIGGERS/exit"; fi
  set -e
}
trap request_graceful_stop USR1 TERM INT

mapfile -t valid_rows < <(/usr/bin/python3.11 - "$VALIDATION" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert len(d['panels'])==13
for p in d['panels']: print(f"{p['name']}\t{p['megatron_prefix']}")
PY
)
VALID_ARGS=()
for row in "${valid_rows[@]}"; do IFS=$'\t' read -r name prefix <<<"$row"; VALID_ARGS+=("$name" "$prefix"); done

export PYTHONPATH="$RUNTIME_COMPAT:$MEGATRON:$TRAINING_CODE"
export CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 NCCL_NET="AWS Libfabric" NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1 NCCL_PROTO="^LL128" NCCL_NET_FORCE_FLUSH=0 NCCL_RAS_ENABLE=0
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=16384 FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
unset NCCL_NET_PLUGIN

common_cmd=(python3 "$TRAINING_CODE/pretrain_scheduled_gpt.py"
  --num-layers 32 --hidden-size 4096 --ffn-hidden-size 21504
  --num-attention-heads 32 --group-query-attention --num-query-groups 8
  --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 500000
  --use-rope-scaling --rope-scaling-factor 8.0
  --make-vocab-size-divisible-by 256 --normalization RMSNorm --xielu --qk-layernorm
  --qknorm-impl apex --disable-bias-linear --untie-embeddings-and-output-weights
  --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.008944
  --micro-batch-size 2 --global-batch-size 1024 --seq-length 4096
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --optimizer ademamix --adam-beta1 0.9 --adam-beta2 0.999 --ademamix-beta3 0.999 --ademamix-alpha 4.0
  --ademamix-beta3-warmup 18284 --ademamix-alpha-warmup 18284
  --weight-decay 0.1 --clip-grad 0.1 --lr 5.5e-5 --min-lr 5.5e-6
  --lr-decay-style WSD --lr-wsd-decay-style 1-sqrt --lr-warmup-samples 409600 --lr-warmup-init 5.5e-6
  --bf16 --main-grads-dtype fp32 --cross-entropy-loss-fusion
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
  --goldfish-loss --goldfish-k 50 --goldfish-h 50
  --reset-attention-mask --reset-position-ids --eod-mask-loss
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER"
  --data-cache-path "$DATA_CACHE" --split 100,0,0 --num-workers 4 --num-dataset-builder-threads 4
  --dataloader-type single --seed 20260609 --log-throughput --log-params-norm --log-interval 1
  --eval-interval 238 --eval-iters 1 --extra-valid-data-path "${VALID_ARGS[@]}"
  --manual-gc --manual-gc-interval 500 --exit-signal-handler --distributed-timeout-minutes 60
  --trigger-path "$TRIGGERS" --save "$CHECKPOINT_ROOT" --ckpt-format torch_dist
  --load "$CHECKPOINT_ROOT" --auto-detect-ckpt-format --ckpt-fully-parallel-load
  --dist-ckpt-strictness assume_ok_unexpected --override-opt_param-scheduler)

launch_megatron() {
  local label=$1 train_samples=$2 prefix_mode=$3 decay_samples=$4 save_iterations=$5 no_save_exit=$6 save_interval=$7 exit_interval=$8 log=$9
  export MINI_SCHEDULE_MANIFEST="$SCHEDULE" MINI_SCHEDULE_ARM=D0_mixed MINI_SCHEDULE_ALLOW_PREFIX="$prefix_mode"
  export MINI_TOKEN_BYTE_LENGTHS="$TOKEN_BYTES" MINI_SCHEDULE_SAVE_ITERATIONS="$save_iterations"
  export MINI_SCHEDULE_EVAL_ITERATIONS="$save_iterations" MINI_SCHEDULE_NO_SAVE_EXIT_ITERATION="$no_save_exit"
  local cmd=("${common_cmd[@]}" --train-samples "$train_samples" --lr-wsd-decay-samples "$decay_samples" --save-interval "$save_interval")
  [[ "$exit_interval" == 0 ]] || cmd+=(--exit-interval "$exit_interval")
  { printf '%q ' "${cmd[@]}"; printf '\n'; } >"$(dirname "$log")/${label}_command.sh"
  cd "$MEGATRON"
  (
    srun --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=288 --cpu-bind=cores -lu \
      bash -c "unset NCCL_NET_PLUGIN; export MINI_SCHEDULE_MANIFEST='$MINI_SCHEDULE_MANIFEST' MINI_SCHEDULE_ARM=D0_mixed MINI_SCHEDULE_ALLOW_PREFIX='$prefix_mode' MINI_TOKEN_BYTE_LENGTHS='$TOKEN_BYTES' MINI_SCHEDULE_SAVE_ITERATIONS='$save_iterations' MINI_SCHEDULE_EVAL_ITERATIONS='$save_iterations' MINI_SCHEDULE_NO_SAVE_EXIT_ITERATION='$no_save_exit'; exec uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc 'export PYTHONPATH='\''$PYTHONPATH'\''; exec torchrun --no-python --nnodes=16 --nproc_per_node=4 --node_rank=\$SLURM_PROCID --master_addr='\''$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)'\'' --master_port=29600 $(printf '%q ' "${cmd[@]}")'" \
      2>&1 | tee "$log"
  ) &
  local pipeline=$!
  while kill -0 "$pipeline" 2>/dev/null; do sleep 2; done
  wait "$pipeline"
}

if [[ "$RECOVERY" == 0 ]]; then
  PHASE=reference_before_probe
  launch_megatron reference_peak_before_probe 18722816 1 3744768 "" 9537 1000000 0 "$CONTROL_ROOT/reference_peak_before.log"
  printf '9536\n' >"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
  PHASE=intervention_probe
  launch_megatron intervention_cooldown_probe 13509632 1 3744768 "" "" 9537 9537 "$CONTROL_ROOT/intervention_cooldown.log"
  [[ -f "$CHECKPOINT_ROOT/iter_0009537/.metadata" && -f "$CHECKPOINT_ROOT/iter_0009537/common.pt" ]]
  intervention_metadata_before=$(/usr/bin/python3.11 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CHECKPOINT_ROOT/iter_0009537/.metadata")
  printf '9536\n' >"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
  PHASE=reference_after_probe
  launch_megatron reference_peak_after_probe 18722816 1 3744768 "" 9537 1000000 0 "$CONTROL_ROOT/reference_peak_after.log"
  [[ "$intervention_metadata_before" == "$(/usr/bin/python3.11 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CHECKPOINT_ROOT/iter_0009537/.metadata")" ]]
  printf '9537\n' >"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
  set +e
  /usr/bin/python3.11 "$SUBPROJECT/scripts/finalize_branch_restart_control.py" \
    --contract "$EARLY_CONTRACT" --reference-before-log "$CONTROL_ROOT/reference_peak_before.log" \
    --intervention-log "$CONTROL_ROOT/intervention_cooldown.log" --intervention-checkpoint-root "$CHECKPOINT_ROOT" \
    --reference-after-log "$CONTROL_ROOT/reference_peak_after.log" \
    --source-checkpoint-receipt "$SOURCE_RECEIPT" --code-bundle-receipt "$EARLY_CODE_BUNDLE_RECEIPT" \
    --output "$CONTROL_ROOT/sandwich_restart_control_receipt.json"
  control_rc=$?
  set -e
  if [[ "$control_rc" -ne 0 ]]; then
    /usr/bin/python3.11 "$SUBPROJECT/scripts/write_training_receipt.py" --output "$RECEIPT" --status sandwich_restart_control_failed --phase branch --job-id "$SLURM_JOB_ID" --log "$CONTROL_ROOT/intervention_cooldown.log" --control-receipt "$CONTROL_ROOT/sandwich_restart_control_receipt.json" --leaf-switch "${leaves[0]}" --started-epoch "$START_EPOCH" --finished-epoch "$(date +%s)"
    exit "$control_rc"
  fi
fi
PHASE=branch
launch_megatron branch 13509632 1 3744768 "10728,11920,13112,13193" "" 1000000 0 "$BRANCH_LOG"
last_iteration=$(grep -Eo 'iteration +[0-9]+/' "$BRANCH_LOG" | tail -1 | tr -cd '0-9')
if [[ "$last_iteration" == 13193 ]]; then final_status=branch_completed; else final_status=branch_gracefully_stopped; fi
final_control="$CONTROL_ROOT/sandwich_restart_control_receipt.json"
PHASE=finished
END_EPOCH=$(date +%s)
/usr/bin/python3.11 "$SUBPROJECT/scripts/write_training_receipt.py" --output "$RECEIPT" --status "$final_status" --phase "$EARLY_PHASE" --job-id "$SLURM_JOB_ID" --log "$BRANCH_LOG" --control-receipt "$final_control" --leaf-switch "${leaves[0]}" --started-epoch "$START_EPOCH" --finished-epoch "$END_EPOCH"
