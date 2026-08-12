#!/usr/bin/env bash
#SBATCH --job-name=early-wsd-eval
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=01:29:00
#SBATCH --no-requeue

set -euo pipefail
: "${EARLY_CODE_ROOT:?set immutable code root}"
: "${EARLY_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${EARLY_CONTRACT:?set experiment contract}"
: "${EARLY_RUN_ROOT:?set run root}"
: "${EARLY_ITERATION:?set checkpoint iteration}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$SLURM_NNODES" == 4 ]] || { echo "evaluation requires four debug nodes" >&2; exit 2; }

EVALUATION_BUNDLE="$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments"
NATIVE_ROOT="$EARLY_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval"
SUBPROJECT="$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment"
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
TOKENIZER=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
COMPAT=${FULL8_PYTHON_COMPAT_DIR:-/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260802T230000Z-mini-b2-v8/compat}
CLEAN_SUBSET=${FULL8_GREEKMMLU_CLEAN_SUBSET:-/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T064000Z-static-prelaunch-v2/greekmmlu_clean_subset_manifest.json}
VALIDATION=$(/usr/bin/python3.11 -c 'import json,sys;print(json.load(open(sys.argv[1]))["data"]["validation_manifest"]["path"])' "$EARLY_CONTRACT")

/usr/bin/python3.11 - "$EARLY_CONTRACT" "$EARLY_ITERATION" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); i=int(sys.argv[2])
if i not in d['evaluation']['milestone_iterations']: raise SystemExit('iteration is not a frozen milestone')
PY
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$EARLY_CODE_BUNDLE_RECEIPT" --kind scientific

padded=$(printf '%07d' "$EARLY_ITERATION")
root="$EARLY_RUN_ROOT/checkpoint_evaluations/iter_$padded"
export_root="$root/export"
greek_root="$root/greekmmlu"
doc_root="$root/per_document"
receipt="$root/exact_checkpoint_native_greekmmlu_receipt.json"
[[ ! -e "$root" ]] || { echo "refusing to overwrite $root" >&2; exit 2; }
mkdir -p "$root" "$doc_root" "$EARLY_RUN_ROOT/evaluation_logs"
label="early_wsd_iter${EARLY_ITERATION}"

export EVALUATION_BUNDLE EVALUATION_CODE_BUNDLE_ROOT="$EARLY_CODE_ROOT" EVALUATION_CODE_BUNDLE_RECEIPT="$EARLY_CODE_BUNDLE_RECEIPT"
export MEGATRON_DIR="$MEGATRON" SOURCE_CHECKPOINT_ROOT="$EARLY_RUN_ROOT/checkpoints" SOURCE_ITERATION="$EARLY_ITERATION"
export TOKENIZER_DIR="$TOKENIZER" EXPORT_ROOT="$export_root" PYTHON_COMPAT_DIR="$COMPAT"
export TOKENIZER_SHA=bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b
export EXPORT_MODEL_SCALE=8B FULL8_EVALUATION_ROOT="$EARLY_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation"
bash "$EVALUATION_BUNDLE/clariden/convert_checkpoint_for_native_greekmmlu.sbatch"

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#hosts[@]} == 4 ]]
pids=()
for group in 0 1 2 3; do
  group_gpus=4; group_cpus=288; group_memory=440G
  if [[ "$group" == 3 ]]; then group_gpus=1; group_cpus=72; group_memory=180G; fi
  srun --exclusive --exact --nodes=1 --ntasks=1 --nodelist="${hosts[$group]}" \
    --gpus-per-node="$group_gpus" --gres="gpu:$group_gpus" --cpus-per-task="$group_cpus" --mem="$group_memory" \
    env SLURM_ARRAY_TASK_ID="$group" FULL8_CODE_ROOT="$EARLY_CODE_ROOT" \
      FULL8_CODE_BUNDLE_RECEIPT="$EARLY_CODE_BUNDLE_RECEIPT" FULL8_VALIDATION_MANIFEST="$VALIDATION" \
      FULL8_HF_MODEL="$export_root/hf" FULL8_HF_TOKENIZER="$TOKENIZER" FULL8_DOCVAL_OUTPUT="$doc_root" \
    bash "$SUBPROJECT/clariden/run_per_document_group_in_allocation.sh" \
    >"$root/doc_group_${group}.out" 2>"$root/doc_group_${group}.err" &
  pids+=("$!")
done
export NATIVE_GREEK_EVAL_ROOT="$NATIVE_ROOT" GREEKMMLU_ROOT="$greek_root" MODEL_LABEL="$label"
export EVALUATION_NAMESPACE="$label" EVAL_DTYPE=float32
# GreekMMLU uses the first GPU left available after group 3's three-panel worker.
srun --exclusive --exact --nodes=1 --ntasks=1 --nodelist="${hosts[3]}" \
  --gpus-per-node=1 --gres=gpu:1 --cpus-per-task=72 --mem=220G \
  bash "$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu.sbatch" \
  >"$root/greekmmlu.out" 2>"$root/greekmmlu.err" &
pids+=("$!")
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
[[ "$failure" == 0 ]] || { echo "one or more evaluation workers failed" >&2; exit 1; }

export EXPORT_RECEIPT="$export_root/checkpoint_eval_export_receipt.json" OUTPUT_RECEIPT="$receipt"
export GREEKMMLU_CLEAN_SUBSET="$CLEAN_SUBSET"
bash "$EVALUATION_BUNDLE/clariden/finalize_checkpoint_greekmmlu.sbatch"
/usr/bin/python3.11 "$SUBPROJECT/scripts/finalize_checkpoint_evaluation.py" \
  --contract "$EARLY_CONTRACT" --iteration "$EARLY_ITERATION" --root "$root" \
  --greekmmlu-receipt "$receipt" --per-document-root "$doc_root" \
  --output "$root/evaluation_receipt.json"

# Keep only one successor in the queue, respecting debug MaxSubmitJobsPU=2.
milestones=(10728 11920 13112 13193)
[[ "${milestones[${EARLY_EVAL_INDEX:?set evaluation index}]}" == "$EARLY_ITERATION" ]] || { echo "evaluation index drift" >&2; exit 2; }
common="ALL,EARLY_CODE_ROOT=$EARLY_CODE_ROOT,EARLY_CODE_BUNDLE_RECEIPT=$EARLY_CODE_BUNDLE_RECEIPT,EARLY_CONTRACT=$EARLY_CONTRACT,EARLY_RUN_ROOT=$EARLY_RUN_ROOT,EARLY_TRAIN_JOB_ID=${EARLY_TRAIN_JOB_ID:-unknown}"
if [[ "$EARLY_EVAL_INDEX" -lt 3 ]]; then
  next_index=$((EARLY_EVAL_INDEX + 1)); next_iteration=${milestones[$next_index]}
  child=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=4 --time=01:29:00 \
    --dependency="afterok:$SLURM_JOB_ID" --job-name="early_wsd_eval_$next_iteration" \
    --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
    --export="$common,EARLY_ITERATION=$next_iteration,EARLY_EVAL_INDEX=$next_index" \
    "$SUBPROJECT/clariden/run_checkpoint_evaluation_debug.sbatch")
  role="evaluation_$next_iteration"
else
  child=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=4 --time=01:20:00 \
    --dependency="afterok:$SLURM_JOB_ID" --job-name=early_wsd_native_endpoint \
    --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
    --export="$common" "$SUBPROJECT/clariden/run_native_endpoint_debug.sbatch")
  role=native_endpoint
fi
if ! /usr/bin/python3.11 "$SUBPROJECT/scripts/audit_submitted_job.py" \
  --job-id "$child" --role "$role" --output "$EARLY_RUN_ROOT/orchestration/${role}_${child}.json"; then
  scancel "$child"; exit 1
fi
printf '%s\n' "$child"
