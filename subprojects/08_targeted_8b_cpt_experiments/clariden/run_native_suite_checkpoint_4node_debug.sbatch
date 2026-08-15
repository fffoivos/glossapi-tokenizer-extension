#!/usr/bin/env bash
# Evaluate one verified checkpoint on the frozen native-Greek suite.
#SBATCH --job-name=h2g-native
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=00:40:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_SCALE H2G_ITERATION \
  H2G_CHECKPOINT_EXPORT H2G_NATIVE_OUTPUT EVAL_CODE_ROOT EVAL_CODE_RECEIPT \
  EVAL_SOURCE_CONTRACT EVAL_SOURCE_MANIFEST EVAL_SOURCE_GATE EVAL_VENV \
  EVAL_EXCLUSIONS; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 4 ]] || {
  echo "native-suite evaluation requires four debug nodes" >&2; exit 2;
}
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || { echo "invalid scale" >&2; exit 2; }
[[ "$H2G_ITERATION" =~ ^[0-9]+$ ]] || { echo "invalid iteration" >&2; exit 2; }
[[ ! -e "$H2G_NATIVE_OUTPUT" ]] || { echo "immutable native-suite output exists" >&2; exit 2; }

/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py" \
  --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_RECEIPT"
observed_exclusions=$(sha256sum "$EVAL_EXCLUSIONS" | awk '{print $1}')
[[ "$observed_exclusions" == 7a8559461b15a308f599faf0ff25cd16c07be0a597078864f779af7f2f1fdd32 ]] || {
  echo "native-suite exclusion payload drift" >&2; exit 2;
}

mkdir -p "$H2G_NATIVE_OUTPUT/logs"
assets="$H2G_NATIVE_OUTPUT/assets"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/evaluation/bind_native_suite_checkpoint.py" \
  --source-contract "$EVAL_SOURCE_CONTRACT" \
  --source-manifest "$EVAL_SOURCE_MANIFEST" \
  --source-execution-gate "$EVAL_SOURCE_GATE" \
  --eval-code-receipt "$EVAL_CODE_RECEIPT" \
  --checkpoint-export "$H2G_CHECKPOINT_EXPORT" \
  --scale "$H2G_SCALE" --iteration "$H2G_ITERATION" --output-dir "$assets"
contract="$assets/contract.json"
manifest="$assets/manifest.json"
model=$(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - "$contract" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint_scope"]
if len(rows) != 1: raise SystemExit("single-checkpoint contract drift")
print(f'{rows[0]["label"]}={rows[0]["model_path"]}')
PY
)
label=${model%%=*}
runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
aggregator="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
rescorer="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/rescore_contamination_filtered.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
for path in "$runner" "$aggregator" "$rescorer" "$native_runner"; do [[ -f "$path" ]] || exit 2; done

shards=(demos other_mcq oyxoy_other oyxoy_wsd{0..9} oyxoy_wic{0..7})
run_task() {
  local shard=$1 benchmarks row_index=0 row_count=1
  case "$shard" in
    demos) benchmarks=demosqa ;;
    other_mcq) benchmarks=medical_mcqa,asep_mcqa,gpcr ;;
    oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
    oyxoy_wsd*) benchmarks=oyxoy_wsd_definition; row_index=${shard#oyxoy_wsd}; row_count=10 ;;
    oyxoy_wic*) benchmarks=oyxoy_wic; row_index=${shard#oyxoy_wic}; row_count=8 ;;
    *) return 2 ;;
  esac
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 --mem=160G \
    --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$H2G_NATIVE_OUTPUT/logs/${label}-${shard}.out" \
    --error="$H2G_NATIVE_OUTPUT/logs/${label}-${shard}.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$runner" \
      --contract "$contract" --manifest "$manifest" --native-runner "$native_runner" \
      --model "$model" --output-dir "$H2G_NATIVE_OUTPUT/$label/$shard" \
      --dtype float32 --scorer-mode legacy --benchmarks "$benchmarks" \
      --candidate-batch-size 1 --example-batch-size 16 --max-examples-per-benchmark 0 \
      --row-shard-index "$row_index" --row-shard-count "$row_count"
}

for wave_start in 0 16; do
  pids=()
  wave_end=$((wave_start + 16))
  (( wave_end > ${#shards[@]} )) && wave_end=${#shards[@]}
  for ((index=wave_start; index<wave_end; index++)); do
    run_task "${shards[$index]}" & pids+=("$!")
  done
  failure=0
  for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
  [[ "$failure" == 0 ]] || { echo "native-suite shard wave failed" >&2; exit 1; }
done

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$aggregator" \
  --root "$H2G_NATIVE_OUTPUT" --contract "$contract" --manifest "$manifest"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$rescorer" \
  --matrix-root "$H2G_NATIVE_OUTPUT" --exclusions-jsonl "$EVAL_EXCLUSIONS" \
  --output-dir "$H2G_NATIVE_OUTPUT/contamination_filtered"
