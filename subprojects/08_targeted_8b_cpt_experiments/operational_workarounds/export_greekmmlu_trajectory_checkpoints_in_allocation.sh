#!/usr/bin/env bash
# Convert every missing trajectory checkpoint inside one held GPU allocation.

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_TRAJECTORY_ROOT \
  H2G_CHECKPOINT_SOURCES EVAL_VENV; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
case "${SLURM_JOB_PARTITION:-}" in debug|normal) ;; *) exit 2 ;; esac
[[ "${SLURM_NNODES:-0}" == 1 ]] || { echo "export batch requires one allocated node" >&2; exit 2; }

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
export_script="$subproject/clariden/export_checkpoint_for_evaluation_debug.sbatch"
megatron_root="$H2G_STAGE_ROOT/tools/megatron_training_c92402e_extra_valid_helpers_v2"
tokenizer_root="$H2G_STAGE_ROOT/assets/tokenizer_148480"
model_contract="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_replication_v1.json"
converter_overlay="$H2G_STAGE_ROOT/control/retries/export_overlay_v105_lowmem_v2"

/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
mkdir -p "$H2G_TRAJECTORY_ROOT/exports" "$H2G_TRAJECTORY_ROOT/logs"
progress="$H2G_TRAJECTORY_ROOT/progress.tsv"
if [[ ! -e "$progress" ]]; then
  printf 'recorded_at\tscale\tupdate\tstage\tpath\n' > "$progress"
fi
record_progress() {
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" >> "$progress"
}

tail -n +2 "$H2G_CHECKPOINT_SOURCES" | while IFS=$'\t' read -r scale update source_root existing_export _; do
  [[ "$existing_export" == - ]] || continue
  printf -v iteration_dir 'iter_%07d' "$update"
  [[ -f "$source_root/$iteration_dir/.metadata" ]] || {
    echo "checkpoint metadata is missing: $source_root/$iteration_dir/.metadata" >&2; exit 2;
  }
  export_root="$H2G_TRAJECTORY_ROOT/exports/$scale/$iteration_dir"
  export_receipt="$export_root/checkpoint_export_receipt.json"
  [[ ! -f "$export_receipt" ]] || continue
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
  # The native-suite venv is intentionally minimal and currently contains an
  # incomplete dill namespace.  Conversion itself uses only packages already
  # pinned in the uenv, so allow the uenv Python without changing converter
  # code or checkpoint semantics.
  export H2G_EVAL_PYTHON="${H2G_CONVERSION_PYTHON:-$EVAL_VENV/bin/python}"
  export H2G_ALLOW_TRAJECTORY_PARITY_WARNING=1
  completed=false
  for attempt in 1 2; do
    if bash "$export_script" \
      > "$H2G_TRAJECTORY_ROOT/logs/export-${scale}-${update}-a${attempt}.out" \
      2> "$H2G_TRAJECTORY_ROOT/logs/export-${scale}-${update}-a${attempt}.err"; then
      completed=true
      break
    fi
    if [[ -e "$export_root" ]]; then
      failed="${export_root}.failed-a${attempt}-$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$export_root" "$failed"
      record_progress "$scale" "$update" export_failed "$failed"
    fi
  done
  [[ "$completed" == true ]] || { echo "export failed twice: $scale@$update" >&2; exit 1; }
  record_progress "$scale" "$update" exported "$export_receipt"
done
