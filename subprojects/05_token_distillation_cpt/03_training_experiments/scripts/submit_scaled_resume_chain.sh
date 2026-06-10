#!/usr/bin/env bash
# Resume an existing 13.5B CPT arm from its checkpoint directory with a scaled
# Slurm shape. This is for correcting launch geometry without changing the
# scientific config: TP/global batch/hyperparameters stay in the arm env.
set -euo pipefail

ARM_INPUT="${1:?usage: submit_scaled_resume_chain.sh <vanilla|td>}"
case "$ARM_INPUT" in
  vanilla) ARM=vanilla; CONFIG_NAME=arm1_vanilla.env ;;
  td)      ARM=td;      CONFIG_NAME=arm2_modern_greek.env ;;
  *) echo "ERROR: arm must be 'vanilla' or 'td', got '$ARM_INPUT'" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
THIS_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/03_training_experiments"
CONFIG="${TRAIN_CONFIG_OVERRIDE:-$THIS_DIR/configs/$CONFIG_NAME}"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$TRAIN_DIR/bakeoff_train.sbatch}"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required, e.g. cpt13b_vanilla_20260610T025159Z}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
STATE_DIR="${STATE_DIR:-$RUN_ROOT/${RUN_TAG}_scaled_resume_state}"

TRAIN_TOKENS="${TRAIN_TOKENS:-13500000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-119}"
EXIT_INTERVAL_USER_SET="${EXIT_INTERVAL+x}"
N_SEGMENTS_USER_SET="${N_SEGMENTS+x}"
EXIT_INTERVAL="${EXIT_INTERVAL:-238}"
N_SEGMENTS="${N_SEGMENTS:-14}"
SEGMENT_TIME_LIMIT="${SEGMENT_TIME_LIMIT:-12:00:00}"
ACCOUNT="${ACCOUNT:-a0140}"
PARTITION="${PARTITION:-normal}"
NODES="${NODES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-36}"
if [ -z "$EXIT_INTERVAL_USER_SET" ] && [ "${NCCL_NET:-}" = "Socket" ] && [ "$NODES" -ge 16 ]; then
  EXIT_INTERVAL=952
fi
if [ -z "$N_SEGMENTS_USER_SET" ] && [ "${NCCL_NET:-}" = "Socket" ] && [ "$NODES" -ge 16 ]; then
  N_SEGMENTS=4
fi
LAUNCH_MODE="${LAUNCH_MODE:-}"
if [ -z "$LAUNCH_MODE" ]; then
  if [ "$NODES" -gt 1 ]; then
    LAUNCH_MODE="torchrun"
  else
    LAUNCH_MODE="slurm"
  fi
fi
SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
  SBATCH_NTASKS_PER_NODE="1"
fi
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_LAUNCH="${CONFIRM_LAUNCH:-0}"

NETWORK_EXPORT_VARS=(
  NCCL_NET NCCL_SOCKET_IFNAME NCCL_SOCKET_FAMILY NCCL_SOCKET_NTHREADS
  NCCL_NSOCKS_PERTHREAD NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_PROTO
  NCCL_NCHANNELS_PER_NET_PEER NCCL_NET_GDR_C2C NCCL_NET_GDR_READ
  NCCL_P2P_LEVEL NCCL_NET_FORCE_FLUSH NCCL_DEBUG NCCL_DEBUG_SUBSYS
  OFI_NCCL_PROTOCOL FI_CXI_RX_MATCH_MODE FI_CXI_USE_DEFAULT_RDZV
  FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_DEFAULT_RX_SIZE
  FI_CXI_REQ_BUF_SIZE FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_OFLOW_BUF_SIZE
  FI_CXI_OFLOW_BUF_COUNT FI_CXI_REQ_BUF_MAX_CACHED FI_LOG_LEVEL FI_LOG_PROV
)
NETWORK_EXPORT_CSV=""
NETWORK_EXPORT_SUMMARY=()
for v in "${NETWORK_EXPORT_VARS[@]}"; do
  if [ -n "${!v:-}" ]; then
    NETWORK_EXPORT_CSV="${NETWORK_EXPORT_CSV},${v}=${!v}"
    NETWORK_EXPORT_SUMMARY+=("${v}=${!v}")
  fi
done

case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0|1" >&2; exit 2 ;; esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_LAUNCH" != "1" ]; then
  echo "ERROR: live launch requires CONFIRM_LAUNCH=1" >&2
  exit 2
fi
test -f "$TRAIN_SCRIPT" || { echo "ERROR: trainer not found: $TRAIN_SCRIPT" >&2; exit 3; }
test -f "$CONFIG" || { echo "ERROR: config not found: $CONFIG" >&2; exit 3; }
test -d "$OUTPUT_DIR/checkpoints" || { echo "ERROR: checkpoint dir missing: $OUTPUT_DIR/checkpoints" >&2; exit 3; }
test -f "$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt" || {
  echo "ERROR: latest checkpoint marker missing under $OUTPUT_DIR/checkpoints" >&2
  exit 3
}

latest_iter="$(tr -dc '0-9' < "$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt")"
test -n "$latest_iter" || { echo "ERROR: latest checkpoint marker is empty" >&2; exit 3; }
test -f "$OUTPUT_DIR/checkpoints/iter_$(printf "%07d" "$latest_iter")/common.pt" || {
  echo "ERROR: latest checkpoint common.pt missing for iter $latest_iter" >&2
  exit 3
}

for v in INIT_CKPT BASE_DATA_PREFIX EXT_DATA_PREFIX BASE_TOKENIZER_DIR EXT_TOKENIZER_DIR \
         LR_PEAK LR_FINAL LR_WARMUP_INIT LR_WARMUP_ITERS LR_WARMUP_TOKENS \
         LR_WSD_DECAY_SAMPLES ADEMA_BETA2 ADEMA_BETA3 ADEMA_ALPHA \
         ADEMA_BETA3_WARMUP_STEPS ADEMA_ALPHA_WARMUP_STEPS \
         DATA_SEED EVAL_INTERVAL EVAL_ITERS; do
  if [ -n "${!v:-}" ] && [ "${ALLOW_OVERRIDES:-0}" != "1" ]; then
    echo "ERROR: $v is set in this shell and would override the config via --export=ALL. unset it or pass ALLOW_OVERRIDES=1" >&2
    exit 5
  fi
done

mkdir -p "$STATE_DIR"
CHAIN_TSV="$STATE_DIR/scaled_chain_$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf "arm\tsegment\tresume\tinit_ckpt\tdependency\tjob_id\tnodes\tgpus_per_node\tlatest_iter_at_submit\n" > "$CHAIN_TSV"

echo "=== submit_scaled_resume_chain.sh ==="
echo "ARM:                 $ARM ($ARM_INPUT)"
echo "RUN_TAG:             $RUN_TAG"
echo "OUTPUT_DIR:          $OUTPUT_DIR"
echo "CONFIG:              $CONFIG"
echo "TRAIN_SCRIPT:        $TRAIN_SCRIPT"
echo "RESUME checkpoint:   $OUTPUT_DIR/checkpoints (latest iter $latest_iter)"
echo "N_SEGMENTS:          $N_SEGMENTS x $SEGMENT_TIME_LIMIT"
echo "NODES/GPUS_PER_NODE: $NODES / $GPUS_PER_NODE"
echo "LAUNCH_MODE:         $LAUNCH_MODE  sbatch_ntasks_per_node=$SBATCH_NTASKS_PER_NODE"
if [ "${#NETWORK_EXPORT_SUMMARY[@]}" -gt 0 ]; then
  echo "NETWORK_OVERRIDES:   ${NETWORK_EXPORT_SUMMARY[*]}"
else
  echo "NETWORK_OVERRIDES:   <trainer defaults>"
fi
echo "DRY_RUN:             $DRY_RUN"
echo

dependency_job="${FIRST_DEPENDENCY_JOB:-}"
for seg in $(seq 1 "$N_SEGMENTS"); do
  dep_args=()
  dep_label="none"
  if [ -n "$dependency_job" ]; then
    dep_args=(--dependency="afterok:$dependency_job")
    dep_label="$dependency_job"
  fi

  cmd=(
    sbatch --parsable "${dep_args[@]}"
      --job-name="cpt13b_${ARM_INPUT}_16n_s${seg}"
      --account="$ACCOUNT" --partition="$PARTITION"
      --nodes="$NODES" --ntasks-per-node="$SBATCH_NTASKS_PER_NODE"
      --gpus-per-node="$GPUS_PER_NODE" --gres="gpu:$GPUS_PER_NODE"
      --cpus-per-task="$CPUS_PER_TASK" --time="$SEGMENT_TIME_LIMIT"
      --output="$OUTPUT_DIR/%x-%j.out" --error="$OUTPUT_DIR/%x-%j.err"
      --export=ALL,ARM="$ARM",INIT_CKPT="$OUTPUT_DIR/checkpoints",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$TRAIN_DIR",TRAIN_CONFIG_OVERRIDE="$CONFIG",TRAIN_TOKENS="$TRAIN_TOKENS",RESUME_TRAINING=1,DISABLE_SAVE=0,SAVE_INTERVAL="$SAVE_INTERVAL",EXIT_INTERVAL="$EXIT_INTERVAL",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SEGMENT_TIME_LIMIT${NETWORK_EXPORT_CSV}"
      "$TRAIN_SCRIPT"
  )

  if [ "$DRY_RUN" = "1" ]; then
    job_id="DRYRUN_scaled_s${seg}"
    printf '  [dry-run] segment %s resume=1 dep=%s nodes=%s\n' "$seg" "$dep_label" "$NODES"
  else
    job_id="$("${cmd[@]}")"
    echo "  submitted scaled segment $seg job=$job_id dep=$dep_label"
  fi
  printf "%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ARM" "$seg" "$OUTPUT_DIR/checkpoints" "$dep_label" "$job_id" "$NODES" "$GPUS_PER_NODE" "$latest_iter" >> "$CHAIN_TSV"
  dependency_job="$job_id"
done

echo
echo "scaled chain manifest: $CHAIN_TSV"
echo "final scaled segment job: $dependency_job"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN - set DRY_RUN=0 CONFIRM_LAUNCH=1 to submit."
fi
