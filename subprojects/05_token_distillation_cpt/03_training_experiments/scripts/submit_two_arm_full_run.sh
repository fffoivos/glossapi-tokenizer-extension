#!/usr/bin/env bash
# ============================================================================
# Submit ONE arm of the two-arm 13.5B full-run Greek CPT as a walltime-bounded
# Slurm dependency chain. Run it TWICE (arm=vanilla, arm=td) to launch both
# experiments in parallel.
#
#   bash submit_two_arm_full_run.sh vanilla     # arm 1 (base 131072)
#   bash submit_two_arm_full_run.sh td          # arm 2 (modern-greek 148480 + TD)
#
# WHY THIS IS NOT submit_training_5b_chain.sh:
#  (1) FULL-RUN WSD. Every segment passes the SAME full-run TRAIN_TOKENS so the
#      WSD warmup/stable/cooldown is anchored to the whole 13.5B run. The 5B
#      chain passed a per-segment target as TRAIN_TOKENS — fine for its CONSTANT
#      LR, but for WSD it would re-anchor the cooldown each segment (broken).
#  (2) NO STALE OVERRIDES. The 5B chain hardcoded LR_SCHEDULE_STYLE=constant,
#      LR_WARMUP_INIT=1.1e-6, ADEMA_*_WARMUP_STEPS=287 into --export, which would
#      silently beat the new config env. Here we export ONLY per-segment vars and
#      let the arm/common config own all hyperparameters.
#  (3) Segments bounded by --exit-interval (rank-deterministic: all ranks exit
#      at the same absolute iteration — no per-rank SIGUSR2 signal race / NCCL
#      hang). EXIT_INTERVAL is a multiple of SAVE_INTERVAL so a checkpoint lands
#      at the boundary; the next segment resumes from it. The trainer's
#      #SBATCH --signal=SIGUSR2@600 + --exit-signal-handler stays as a walltime
#      BACKSTOP only. (The 5B chain used per-segment train-tokens — wrong for WSD.)
# ============================================================================
set -euo pipefail

ARM_INPUT="${1:?usage: submit_two_arm_full_run.sh <vanilla|td>}"
case "$ARM_INPUT" in
  vanilla) ARM=vanilla; CONFIG_NAME=arm1_vanilla.env ;;
  td)      ARM=td;      CONFIG_NAME=arm2_modern_greek.env ;;
  *) echo "ERROR: arm must be 'vanilla' or 'td', got '$ARM_INPUT'" >&2; exit 2 ;;
esac

# ---- paths (Clariden staging; override for your mirror) ----
REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
THIS_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/03_training_experiments"
CONFIG="${TRAIN_CONFIG_OVERRIDE:-$THIS_DIR/configs/$CONFIG_NAME}"
# The ESTABLISHED trainer (thin Megatron-fork wrapper). Its dir must resolve
# ../megatron_patches/runtime, so SCRIPT_DIR_OVERRIDE points at bakeoff_training.
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$TRAIN_DIR/bakeoff_train.sbatch}"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b}"
RUN_TAG="${RUN_TAG:-cpt13b_${ARM_INPUT}_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
STATE_DIR="${STATE_DIR:-$RUN_ROOT/${RUN_TAG}_submit_state}"

# ---- run shape ----
TRAIN_TOKENS="${TRAIN_TOKENS:-13500000000}"   # FULL run (anchors WSD on every segment)
SAVE_INTERVAL="${SAVE_INTERVAL:-119}"          # ~0.5B tokens
# Primary segment cap = --exit-interval (rank-deterministic: every rank exits at
# the same absolute iteration, so no NCCL-collective mismatch). Must be a
# multiple of SAVE_INTERVAL so a checkpoint exists exactly at the exit boundary.
# Defaults are conservative for generic launches. For validated 16-node
# production launches, use longer chunks to avoid 14 short requeues while
# staying far inside the 12h walltime.
EXIT_INTERVAL_USER_SET="${EXIT_INTERVAL+x}"
N_SEGMENTS_USER_SET="${N_SEGMENTS+x}"
EXIT_INTERVAL="${EXIT_INTERVAL:-238}"
N_SEGMENTS="${N_SEGMENTS:-14}"                # ceil(3218/238); extras no-op once train completes
SEGMENT_TIME_LIMIT="${SEGMENT_TIME_LIMIT:-12:00:00}"
ACCOUNT="${ACCOUNT:-a0140}"
PARTITION="${PARTITION:-normal}"
NODES="${NODES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-36}"
if [ -z "$EXIT_INTERVAL_USER_SET" ] && [ "$NODES" -ge 16 ]; then
  EXIT_INTERVAL=952                           # 8*119; ~2.3h train time/segment on CXI
fi
if [ -z "$N_SEGMENTS_USER_SET" ] && [ "$NODES" -ge 16 ]; then
  N_SEGMENTS=4                                # ceil(3218/952)
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
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

# Optional runtime transport overrides. These are intentionally separated from
# the training-regime config: they change how ranks communicate, not what is
# trained. --export=ALL would forward them anyway; spelling them out makes dry
# runs and Slurm job records auditable.
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

# ---- preflight ----
case "$DRY_RUN" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0|1" >&2; exit 2 ;; esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_LAUNCH" != "1" ]; then
  echo "ERROR: live launch requires CONFIRM_LAUNCH=1" >&2; exit 2
fi
test -f "$TRAIN_SCRIPT"  || { echo "ERROR: trainer not found: $TRAIN_SCRIPT" >&2; exit 3; }
test -f "$CONFIG"        || { echo "ERROR: config not found: $CONFIG" >&2; exit 3; }
if [ -e "$OUTPUT_DIR" ] && [ "$ALLOW_EXISTING_OUTPUT" != "1" ]; then
  echo "ERROR: output dir exists: $OUTPUT_DIR (set ALLOW_EXISTING_OUTPUT=1 to resume)" >&2; exit 3
fi

# Export hygiene: --export=ALL would forward stale shell vars over the config.
# For the production launch, arm configs own checkpoint/data/tokenizer paths and
# common_cpt.env owns the training regime. Explicit overrides require opting in.
for v in INIT_CKPT BASE_DATA_PREFIX EXT_DATA_PREFIX BASE_TOKENIZER_DIR EXT_TOKENIZER_DIR \
         LR_PEAK LR_FINAL LR_WARMUP_INIT LR_WARMUP_ITERS LR_WARMUP_TOKENS \
         LR_WSD_DECAY_SAMPLES ADEMA_BETA2 ADEMA_BETA3 ADEMA_ALPHA \
         ADEMA_BETA3_WARMUP_STEPS ADEMA_ALPHA_WARMUP_STEPS \
         DATA_SEED EVAL_INTERVAL EVAL_ITERS; do
  if [ -n "${!v:-}" ] && [ "${ALLOW_OVERRIDES:-0}" != "1" ]; then
    echo "ERROR: $v is set in this shell and would override the config via --export=ALL. unset it or pass ALLOW_OVERRIDES=1" >&2; exit 5
  fi
done

resolve_config_var() {
  local var="$1"
  bash -c 'source "$1" >/dev/null 2>&1; printf "%s\n" "${!2}"' bash "$CONFIG" "$var"
}

# Resolve INIT_CKPT + the arm data prefix from the config (sourced read-only).
# shellcheck disable=SC1090
( source "$CONFIG" >/dev/null 2>&1
  : "${INIT_CKPT:?INIT_CKPT must be set by the arm config}" )
INIT_CKPT_ARM="$(resolve_config_var INIT_CKPT)"

mkdir -p "$STATE_DIR"
CHAIN_TSV="$STATE_DIR/chain.tsv"
printf "arm\tsegment\tresume\tinit_ckpt\tdependency\tjob_id\n" > "$CHAIN_TSV"

echo "=== submit_two_arm_full_run.sh ==="
echo "ARM:                $ARM   ($ARM_INPUT)"
echo "CONFIG:             $CONFIG"
echo "TRAIN_SCRIPT:       $TRAIN_SCRIPT"
echo "INIT_CKPT (seg 1):  $INIT_CKPT_ARM"
echo "OUTPUT_DIR:         $OUTPUT_DIR"
echo "TRAIN_TOKENS:       $TRAIN_TOKENS  (full-run WSD anchor)"
echo "N_SEGMENTS:         $N_SEGMENTS  x  $SEGMENT_TIME_LIMIT"
echo "LAUNCH_MODE:        $LAUNCH_MODE  sbatch_ntasks_per_node=$SBATCH_NTASKS_PER_NODE"
if [ "${#NETWORK_EXPORT_SUMMARY[@]}" -gt 0 ]; then
  echo "NETWORK_OVERRIDES:  ${NETWORK_EXPORT_SUMMARY[*]}"
else
  echo "NETWORK_OVERRIDES:  <trainer defaults>"
fi
echo "DRY_RUN:            $DRY_RUN"
echo

[ "$DRY_RUN" = "0" ] && mkdir -p "$OUTPUT_DIR"

# Data preflight: the arm binary must exist before any segment is queued.
if [ "$ARM" = vanilla ]; then
  DATA_PREFIX_ARM="$(resolve_config_var BASE_DATA_PREFIX)"
else
  DATA_PREFIX_ARM="$(resolve_config_var EXT_DATA_PREFIX)"
fi
if [ "$DRY_RUN" = "0" ] && [ ! -f "${DATA_PREFIX_ARM}.idx" ]; then
  echo "ERROR: data binary missing: ${DATA_PREFIX_ARM}.idx — run the dataset chain first" >&2; exit 4
fi

dependency_job=""
for seg in $(seq 1 "$N_SEGMENTS"); do
  if [ "$seg" -eq 1 ]; then
    resume=0; init_ckpt="$INIT_CKPT_ARM"; dep_args=(); dep_label="none"
  else
    resume=1; init_ckpt="$OUTPUT_DIR/checkpoints"
    dep_args=(--dependency="afterok:$dependency_job"); dep_label="$dependency_job"
  fi

  # NOTE: we export ONLY per-segment control vars. We deliberately DO NOT export
  # LR_SCHEDULE_STYLE / LR_WARMUP_* / ADEMA_*_WARMUP_STEPS / LR_PEAK etc. — those
  # belong to the config env so the WSD/AdEMAMix policy is single-sourced.
  cmd=(
    sbatch --parsable "${dep_args[@]}"
      --job-name="cpt13b_${ARM_INPUT}_s${seg}"
      --account="$ACCOUNT" --partition="$PARTITION"
      --nodes="$NODES" --ntasks-per-node="$SBATCH_NTASKS_PER_NODE"
      --gpus-per-node="$GPUS_PER_NODE" --gres="gpu:$GPUS_PER_NODE"
      --cpus-per-task="$CPUS_PER_TASK" --time="$SEGMENT_TIME_LIMIT"
      --output="$OUTPUT_DIR/%x-%j.out" --error="$OUTPUT_DIR/%x-%j.err"
      --export=ALL,ARM="$ARM",INIT_CKPT="$init_ckpt",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$TRAIN_DIR",TRAIN_CONFIG_OVERRIDE="$CONFIG",TRAIN_TOKENS="$TRAIN_TOKENS",RESUME_TRAINING="$resume",DISABLE_SAVE=0,SAVE_INTERVAL="$SAVE_INTERVAL",EXIT_INTERVAL="$EXIT_INTERVAL",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SEGMENT_TIME_LIMIT${NETWORK_EXPORT_CSV}"
      "$TRAIN_SCRIPT"
  )

  if [ "$DRY_RUN" = "1" ]; then
    job_id="DRYRUN_s${seg}"
    printf '  [dry-run] segment %s resume=%s dep=%s\n' "$seg" "$resume" "$dep_label"
  else
    job_id="$("${cmd[@]}")"
    echo "  submitted segment $seg job=$job_id resume=$resume dep=$dep_label"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ARM" "$seg" "$resume" "$init_ckpt" "$dep_label" "$job_id" >> "$CHAIN_TSV"
  dependency_job="$job_id"
done

echo
echo "chain manifest: $CHAIN_TSV"
echo "final segment job: $dependency_job"
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN — set DRY_RUN=0 CONFIRM_LAUNCH=1 to submit."
fi
