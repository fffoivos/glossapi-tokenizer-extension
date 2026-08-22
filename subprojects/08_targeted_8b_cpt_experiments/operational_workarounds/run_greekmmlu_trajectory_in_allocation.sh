#!/usr/bin/env bash
# Evaluate every saved 1.5B and 8B checkpoint on the full frozen clean panel.
# This is an explicit held-allocation recovery driver.  It does not alter the
# panel, scorer, tokenizer, model weights, or checkpoint identity.

set -euo pipefail

for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_TRAJECTORY_ROOT \
  H2G_CHECKPOINT_SOURCES EVAL_VENV; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == normal && "${SLURM_NNODES:-0}" == 4 ]] || {
  echo "trajectory evaluation requires a held four-node normal allocation" >&2
  exit 2
}

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
export_script="$subproject/clariden/export_checkpoint_for_evaluation_debug.sbatch"
score_script="$subproject/clariden/run_frozen_greekmmlu_4node_debug.sbatch"
clean_examples="$H2G_STAGE_ROOT/evaluation/greekmmlu_sentinels/clean_examples.json"
sentinel_manifest="$H2G_STAGE_ROOT/evaluation/greekmmlu_sentinels/sentinel_manifest.json"
megatron_root="$H2G_STAGE_ROOT/tools/megatron_training_c92402e_extra_valid_helpers_v2"
tokenizer_root="$H2G_STAGE_ROOT/assets/tokenizer_148480"
model_contract="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_replication_v1.json"
converter_overlay="$H2G_STAGE_ROOT/control/retries/export_overlay_v105_lowmem_v2"

/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
[[ -f "$clean_examples" && -f "$sentinel_manifest" ]] || {
  echo "frozen GreekMMLU inputs are missing" >&2; exit 2;
}
[[ -f "$H2G_CHECKPOINT_SOURCES" ]] || { echo "checkpoint source table is missing" >&2; exit 2; }

mkdir -p "$H2G_TRAJECTORY_ROOT/exports" "$H2G_TRAJECTORY_ROOT/results" \
  "$H2G_TRAJECTORY_ROOT/logs"
progress="$H2G_TRAJECTORY_ROOT/progress.tsv"
if [[ ! -e "$progress" ]]; then
  printf 'recorded_at\tscale\tupdate\tstage\tpath\n' > "$progress"
fi

record_progress() {
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" >> "$progress"
}

tail -n +2 "$H2G_CHECKPOINT_SOURCES" | while IFS=$'\t' read -r scale update source_root existing_export existing_result; do
  [[ "$scale" == 1p5b || "$scale" == 8b ]] || { echo "invalid scale in source table" >&2; exit 2; }
  [[ "$update" =~ ^[0-9]+$ ]] || { echo "invalid update in source table" >&2; exit 2; }
  printf -v iteration_dir 'iter_%07d' "$update"
  [[ -f "$source_root/$iteration_dir/.metadata" ]] || {
    echo "checkpoint metadata is missing: $source_root/$iteration_dir/.metadata" >&2
    exit 2
  }

  if [[ "$existing_export" != - ]]; then
    export_receipt="$existing_export"
  else
    export_root="$H2G_TRAJECTORY_ROOT/exports/$scale/$iteration_dir"
    export_receipt="$export_root/checkpoint_export_receipt.json"
    if [[ ! -f "$export_receipt" ]]; then
      mkdir -p "$(dirname "$export_root")"
      export H2G_SCALE="$scale"
      export H2G_SOURCE_CHECKPOINT_ROOT="$source_root"
      export H2G_SOURCE_ITERATION="$update"
      export H2G_MEGATRON_DIR="$megatron_root"
      export H2G_TOKENIZER_DIR="$tokenizer_root"
      export H2G_MODEL_CONTRACT="$model_contract"
      export H2G_CONVERTER_OVERLAY_ROOT="$converter_overlay"
      export H2G_CONVERTER_OVERLAY_RECEIPT="$converter_overlay/converter_overlay_receipt.json"
      export H2G_EXPORT_ROOT="$export_root"
      export H2G_EVAL_PYTHON="${H2G_CONVERSION_PYTHON:-python3}"
      export H2G_ALLOW_TRAJECTORY_PARITY_WARNING=1
      export_completed=false
      for export_attempt in 1 2; do
        if srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 \
          --cpus-per-task=72 --mem=220G \
          --output="$H2G_TRAJECTORY_ROOT/logs/export-${scale}-${update}-a${export_attempt}.out" \
          --error="$H2G_TRAJECTORY_ROOT/logs/export-${scale}-${update}-a${export_attempt}.err" \
          bash "$export_script"; then
          export_completed=true
          break
        fi
        if [[ -e "$export_root" ]]; then
          failed_export="${export_root}.failed-a${export_attempt}-$(date -u +%Y%m%dT%H%M%SZ)"
          mv "$export_root" "$failed_export"
          record_progress "$scale" "$update" export_failed "$failed_export"
        fi
      done
      [[ "$export_completed" == true ]] || {
        echo "checkpoint export failed twice: $scale@$update" >&2; exit 1;
      }
      record_progress "$scale" "$update" exported "$export_receipt"
    fi
  fi
  [[ -f "$export_receipt" ]] || { echo "export receipt is missing: $export_receipt" >&2; exit 2; }

  if [[ "$existing_result" != - ]]; then
    result_root="$existing_result"
  else
    result_root="$H2G_TRAJECTORY_ROOT/results/$scale/$iteration_dir/full_clean"
    if [[ ! -f "$result_root/aggregate/receipt.json" ]]; then
      export H2G_SCALE="$scale"
      export H2G_ITERATION="$update"
      export H2G_CHECKPOINT_EXPORT="$export_receipt"
      export H2G_GREEKMMLU_MODE=full_clean
      export H2G_GREEKMMLU_CLEAN_EXAMPLES="$clean_examples"
      export H2G_GREEKMMLU_SENTINEL_MANIFEST="$sentinel_manifest"
      export H2G_GREEKMMLU_OUTPUT="$result_root"
      score_completed=false
      for score_attempt in 1 2; do
        if bash "$score_script"; then
          score_completed=true
          break
        fi
        record_progress "$scale" "$update" score_failed "attempt_$score_attempt"
      done
      [[ "$score_completed" == true ]] || {
        echo "GreekMMLU scoring failed twice: $scale@$update" >&2; exit 1;
      }
      record_progress "$scale" "$update" scored "$result_root/aggregate/receipt.json"
    fi
  fi
  [[ -f "$result_root/aggregate/receipt.json" ]] || {
    echo "GreekMMLU aggregate receipt is missing: $result_root/aggregate/receipt.json" >&2
    exit 2
  }
  record_progress "$scale" "$update" completed "$result_root/aggregate/receipt.json"
done

printf '%s\n' "$H2G_TRAJECTORY_ROOT"
