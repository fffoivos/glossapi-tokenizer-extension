#!/usr/bin/env bash
#SBATCH --job-name=mini_greekmmlu_dtype_wave
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=460G
#SBATCH --time=00:30:00

set -euo pipefail
: "${RUN_ROOT:?set campaign run root}"
: "${SOURCE_ATTEMPT_ROOT:?set source evaluation-attempt root}"
: "${DIAGNOSTIC_ROOT:?set unique diagnostic output root}"
: "${EVAL_DTYPE:?set explicit diagnostic dtype}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${DIAGNOSTIC_RUNNER:?set diagnose_native_greekmmlu_dtype.sbatch path}"
: "${ARMS_COLON:?set one to four colon-separated arms}"

IFS=':' read -r -a arms <<<"$ARMS_COLON"
(( ${#arms[@]} >= 1 && ${#arms[@]} <= 4 )) || {
  echo "dtype wave requires one to four arms" >&2
  exit 2
}
[[ ! -e "$DIAGNOSTIC_ROOT" ]] || {
  echo "refusing to overwrite dtype wave: $DIAGNOSTIC_ROOT" >&2
  exit 2
}
mkdir -p "$DIAGNOSTIC_ROOT"

pids=()
for arm in "${arms[@]}"; do
  model_root="$SOURCE_ATTEMPT_ROOT/tasks/$arm/export/hf"
  output_root="$DIAGNOSTIC_ROOT/$arm/results"
  (
    export HF_MODEL_ROOT="$model_root"
    export OUTPUT_ROOT="$output_root"
    export MODEL_LABEL="${arm}_${EVAL_DTYPE}"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
      --cpus-per-task=18 --mem=100G \
      bash "$DIAGNOSTIC_RUNNER"
  ) >"$DIAGNOSTIC_ROOT/$arm.out" 2>"$DIAGNOSTIC_ROOT/$arm.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "one or more dtype-diagnostic lanes failed" >&2
  exit 1
}
