#!/usr/bin/env bash
#SBATCH --job-name=d0-0p5b-native3cp
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=00:22:30
#SBATCH --no-requeue

set -euo pipefail
: "${D0_WRAPPER_ROOT:?set frozen D0 wrapper root}"
: "${D0_EVAL_OUTPUT_ROOT:?set new output root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$SLURM_NNODES" == 4 ]] || {
  echo "D0 native-Greek matrix requires four debug nodes" >&2
  exit 2
}

EVAL_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$EVAL_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
EVAL_CODE_RECEIPT="$EVAL_CODE_ROOT/bundle_receipt.json"
SOURCE_CONTRACT="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/native_greek_3cp_contract.json"
SOURCE_MANIFEST="$EVAL_ROOT/frozen_examples_v2/manifest.json"
EXECUTION_GATE="$EVAL_ROOT/execution_gate_final_v1/execution_gate.json"
EVAL_VENV=/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026
runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
aggregator="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
# Contamination rescoring was added after the scorer bundle was frozen. Keep
# model scoring on EVAL_CODE_ROOT and bind this post-processing-only script to
# the immutable D0 wrapper bundle.
rescorer="$D0_WRAPPER_ROOT/evaluation/rescore_contamination_filtered.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
bindings="$D0_WRAPPER_ROOT/evaluation/d0_0p5b_checkpoint_bindings.json"
binder="$D0_WRAPPER_ROOT/evaluation/bind_d0_0p5b_native_suite.py"
exclusions=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/recommended_excluded_example_ids.jsonl

[[ ! -e "$D0_EVAL_OUTPUT_ROOT" ]] || { echo "refusing to overwrite $D0_EVAL_OUTPUT_ROOT" >&2; exit 2; }
for path in "$EVAL_CODE_RECEIPT" "$SOURCE_CONTRACT" "$SOURCE_MANIFEST" "$EXECUTION_GATE" \
  "$runner" "$aggregator" "$rescorer" "$native_runner" "$bindings" "$binder" "$exclusions"; do
  [[ -f "$path" ]] || { echo "missing input $path" >&2; exit 2; }
done
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py" \
  --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_RECEIPT"

assets="$D0_EVAL_OUTPUT_ROOT/assets"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$binder" \
  --source-contract "$SOURCE_CONTRACT" --source-manifest "$SOURCE_MANIFEST" \
  --source-execution-gate "$EXECUTION_GATE" --eval-code-receipt "$EVAL_CODE_RECEIPT" \
  --bindings "$bindings" --output-dir "$assets"
contract="$assets/d0_0p5b_native_greek_3cp_contract.json"
manifest="$assets/d0_0p5b_frozen_examples_manifest.json"
mkdir -p "$D0_EVAL_OUTPUT_ROOT/logs"

mapfile -t models < <(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - "$contract" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["checkpoint_scope"]:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
[[ ${#models[@]} -eq 3 ]]
shards=(demos other_mcq oyxoy_other oyxoy_wsd{0..9} oyxoy_wic{0..7})
tasks=()
for model in "${models[@]}"; do
  for shard in "${shards[@]}"; do tasks+=("$model|$shard"); done
done
[[ ${#tasks[@]} -eq 63 ]]

run_task() {
  local task=$1
  local model=${task%%|*}
  local shard=${task#*|}
  local label=${task%%=*}
  local benchmarks row_index=0 row_count=1
  case "$shard" in
    demos) benchmarks=demosqa ;;
    other_mcq) benchmarks=medical_mcqa,asep_mcqa,gpcr ;;
    oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
    oyxoy_wsd*) benchmarks=oyxoy_wsd_definition; row_index=${shard#oyxoy_wsd}; row_count=10 ;;
    oyxoy_wic*) benchmarks=oyxoy_wic; row_index=${shard#oyxoy_wic}; row_count=8 ;;
    *) echo "unknown shard $shard" >&2; return 2 ;;
  esac
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 --mem=160G \
    --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$D0_EVAL_OUTPUT_ROOT/logs/${label}-${shard}.out" \
    --error="$D0_EVAL_OUTPUT_ROOT/logs/${label}-${shard}.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$runner" \
      --contract "$contract" --manifest "$manifest" --native-runner "$native_runner" \
      --model "$model" --output-dir "$D0_EVAL_OUTPUT_ROOT/$label/$shard" \
      --dtype float32 --scorer-mode legacy --benchmarks "$benchmarks" \
      --candidate-batch-size 1 --example-batch-size 16 --max-examples-per-benchmark 0 \
      --row-shard-index "$row_index" --row-shard-count "$row_count"
}

for wave_start in 0 16 32 48; do
  pids=()
  for ((index=wave_start; index<wave_start+16 && index<${#tasks[@]}; index++)); do
    run_task "${tasks[$index]}" &
    pids+=("$!")
  done
  failure=0
  for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
  [[ "$failure" -eq 0 ]] || { echo "one or more D0 native-Greek shards failed" >&2; exit 1; }
done

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$aggregator" \
  --root "$D0_EVAL_OUTPUT_ROOT" --contract "$contract" --manifest "$manifest"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$rescorer" \
  --matrix-root "$D0_EVAL_OUTPUT_ROOT" --exclusions-jsonl "$exclusions" \
  --output-dir "$D0_EVAL_OUTPUT_ROOT/contamination_filtered"
