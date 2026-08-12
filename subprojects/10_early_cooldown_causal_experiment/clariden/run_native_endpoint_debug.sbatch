#!/usr/bin/env bash
#SBATCH --job-name=early-wsd-native
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${EARLY_CODE_ROOT:?set immutable code root}"
: "${EARLY_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${EARLY_RUN_ROOT:?set run root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$SLURM_NNODES" == 4 ]] || { echo "native endpoint requires four debug nodes" >&2; exit 2; }

EVAL_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$EVAL_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
EVAL_CODE_RECEIPT="$EVAL_CODE_ROOT/bundle_receipt.json"
SOURCE_CONTRACT="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/native_greek_3cp_contract.json"
EXECUTION_GATE="$EVAL_ROOT/execution_gate_final_v1/execution_gate.json"
MANIFEST="$EVAL_ROOT/frozen_examples_v2/manifest.json"
EVAL_VENV=/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026
runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
aggregator="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
root="$EARLY_RUN_ROOT/native_greek_endpoint"
contract="$root/endpoint_contract.json"
model="$EARLY_RUN_ROOT/checkpoint_evaluations/iter_0013193/export/hf"
export_receipt="$EARLY_RUN_ROOT/checkpoint_evaluations/iter_0013193/export/checkpoint_eval_export_receipt.json"
[[ ! -e "$root" ]] || { echo "refusing to overwrite native endpoint" >&2; exit 2; }
mkdir -p "$root/logs"

/usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$EARLY_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py" \
  --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_RECEIPT"
/usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/scripts/bind_native_endpoint_contract.py" \
  --source-contract "$SOURCE_CONTRACT" --source-execution-gate "$EXECUTION_GATE" --eval-code-receipt "$EVAL_CODE_RECEIPT" --manifest "$MANIFEST" \
  --model "$model" --export-receipt "$export_receipt" --iteration 13193 --output "$contract"

run_shard() {
  local shard=$1 benchmarks row_index=0 row_count=1
  case "$shard" in
    demos) benchmarks=demosqa ;;
    other_mcq) benchmarks=medical_mcqa,asep_mcqa,gpcr ;;
    oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
    oyxoy_wsd*) benchmarks=oyxoy_wsd_definition; row_index=${shard#oyxoy_wsd}; row_count=10 ;;
    oyxoy_wic*) benchmarks=oyxoy_wic; row_index=${shard#oyxoy_wic}; row_count=8 ;;
  esac
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 --mem=160G \
    --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$root/logs/$shard.out" --error="$root/logs/$shard.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$runner" \
      --contract "$contract" --manifest "$MANIFEST" --native-runner "$native_runner" \
      --model "iter_0013193=$model" --output-dir "$root/iter_0013193/$shard" --dtype float32 \
      --scorer-mode legacy --benchmarks "$benchmarks" --candidate-batch-size 1 --example-batch-size 16 \
      --max-examples-per-benchmark 0 --row-shard-index "$row_index" --row-shard-count "$row_count"
}

shards=(demos other_mcq oyxoy_other oyxoy_wsd{0..9} oyxoy_wic{0..7})
for wave_start in 0 16; do
  pids=()
  for ((i=wave_start; i<wave_start+16 && i<${#shards[@]}; i++)); do run_shard "${shards[$i]}" & pids+=("$!"); done
  failure=0; for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
  [[ "$failure" == 0 ]] || { echo "native endpoint shard failed" >&2; exit 1; }
done
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$aggregator" \
  --root "$root" --contract "$contract" --manifest "$MANIFEST"
/usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/scripts/finalize_native_endpoint.py" \
  --matrix-root "$root" --contract "$contract" --manifest "$MANIFEST" --output "$root/native_endpoint_receipt.json"
