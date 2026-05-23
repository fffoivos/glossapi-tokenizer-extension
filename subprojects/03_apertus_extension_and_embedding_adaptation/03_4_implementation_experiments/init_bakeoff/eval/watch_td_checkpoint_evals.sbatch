#!/usr/bin/env bash
# Durable CPU-only Slurm wrapper around the TD checkpoint watchers.
#
# This job intentionally runs on xfer with one CPU and no GPU. It may be chained
# with --dependency=afterany:<jobid>; per-checkpoint state directories prevent
# duplicate eval submission when a second watcher window repeats the same specs.

#SBATCH --job-name=td_l11_watch
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=11:50:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/../slurm_cpu_only_guard.sh"
require_cpu_only_slurm "td_l11_eval_watch"

RUN_TAG="${RUN_TAG:-td_full25_layer11_2b_20260523T165038Z}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
EVAL_JSONL="${EVAL_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-200000}"
WATCH_SPECS="${WATCH_SPECS:-65:greek_only:0 130:full:1 260:full:1 390:full:1 455:full:1 476:full:1}"

mkdir -p "$OUT_ROOT"

echo "=== watch_td_checkpoint_evals.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "RUN_TAG: $RUN_TAG"
echo "RUN_ROOT: $RUN_ROOT"
echo "OUT_ROOT: $OUT_ROOT"
echo "EVAL_JSONL: $EVAL_JSONL"
echo "POLL_SECONDS: $POLL_SECONDS"
echo "TIMEOUT_SECONDS: $TIMEOUT_SECONDS"
echo "WATCH_SPECS: $WATCH_SPECS"
echo

pids=()
for spec in $WATCH_SPECS; do
    IFS=: read -r iter task_group submit_intrinsic <<< "$spec"
    log="$OUT_ROOT/watch_td_l11_iter${iter}_${task_group}.log"
    echo "starting watcher iter=$iter task_group=$task_group intrinsic=$submit_intrinsic log=$log"
    (
        RUN_TAG="$RUN_TAG" \
        RUN_ROOT="$RUN_ROOT" \
        OUT_ROOT="$OUT_ROOT" \
        ITER="$iter" \
        TASK_GROUP="$task_group" \
        SUBMIT_INTRINSIC="$submit_intrinsic" \
        EVAL_JSONL="$EVAL_JSONL" \
        POLL_SECONDS="$POLL_SECONDS" \
        TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
        bash "$SCRIPT_DIR/watch_and_submit_td_checkpoint_eval.sh"
    ) >> "$log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo
echo "=== watcher window done status=$status ==="
date -u
exit "$status"
