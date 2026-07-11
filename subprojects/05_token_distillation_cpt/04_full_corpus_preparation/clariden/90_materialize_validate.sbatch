#!/usr/bin/env bash
#SBATCH --job-name=cpt4_materialize
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "release materialization and validation"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$MATERIALIZE_SCRIPT"
phase04_require_file "$VALIDATE_RELEASE_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "80-materialize-validate"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "70-dedup"
phase04_stage_require_upstream "60-greekmmlu-decontam"

DECONTAM_DIR="$PIPELINE_RUN_ROOT/stages/60-greekmmlu-decontam"
DEDUP_DIR="$PIPELINE_RUN_ROOT/stages/70-dedup"
phase04_require_file "$DECONTAM_DIR/final_clean_stage.txt"
FINAL_CLEAN_STAGE=$(<"$DECONTAM_DIR/final_clean_stage.txt")
case "$FINAL_CLEAN_STAGE" in
    50-clean|58-final-clean) ;;
    *) echo "ERROR: invalid recorded final cleaning stage: $FINAL_CLEAN_STAGE" >&2; exit 108 ;;
esac
phase04_stage_require_upstream "$FINAL_CLEAN_STAGE"
CLEAN_DIR="$PIPELINE_RUN_ROOT/stages/$FINAL_CLEAN_STAGE"
CLEANING_MANIFEST="$CLEAN_DIR/cleaning_manifest.json"
DECONTAMINATION_MANIFEST="$DECONTAM_DIR/decontamination_manifest.json"
DEDUP_MANIFEST="$DEDUP_DIR/dedup_manifest.json"
phase04_require_file "$CLEANING_MANIFEST"
phase04_require_file "$DECONTAMINATION_MANIFEST"
phase04_require_file "$DEDUP_MANIFEST"
DECISIONS=$(phase04_contract_python get-json-field \
    --path "$DEDUP_MANIFEST" --field dedup_output.content_bound_decisions.path)
phase04_require_file "$DECISIONS"
phase04_stage_add_input dedup_decisions "$DECISIONS"
phase04_stage_add_input cleaning_manifest "$CLEANING_MANIFEST"
phase04_stage_add_input decontamination_manifest "$DECONTAMINATION_MANIFEST"
phase04_stage_add_input dedup_manifest "$DEDUP_MANIFEST"
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -s "$PHASE04_STAGE_DIR/release_manifest.json" ]]; then
    materialize_resume=()
    if [[ -d "$PHASE04_STAGE_DIR/release" ]] && \
       [[ -n "$(find "$PHASE04_STAGE_DIR/release" -mindepth 1 -print -quit)" ]]; then
        materialize_resume+=(--resume)
    fi
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$MATERIALIZE_SCRIPT" \
        --input "$DECONTAM_DIR/corpus" \
        --cleaning-manifest "$CLEANING_MANIFEST" \
        --decontamination-manifest "$DECONTAMINATION_MANIFEST" \
        --dedup-manifest "$DEDUP_MANIFEST" \
        --dedup-decisions "$DECISIONS" \
        --cleaning-ledger "$CLEAN_DIR/ledger" \
        --decontam-ledger "$DECONTAM_DIR/ledger" \
        --output "$PHASE04_STAGE_DIR/release" \
        --manifest "$PHASE04_STAGE_DIR/release_manifest.json" \
        --token-waterfall "$PHASE04_STAGE_DIR/token_waterfall.json" \
        --temporary-directory "$PHASE04_STAGE_DIR/duckdb-tmp" \
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}" \
        --threads "${DUCKDB_THREADS:-128}" \
        "${materialize_resume[@]}"
fi

if [[ ! -e "$PHASE04_STAGE_DIR/validation.json" ]]; then
    VALIDATION_ATTEMPT_DIR="$PHASE04_STAGE_DIR/validation_attempts"
    mkdir -p "$VALIDATION_ATTEMPT_DIR"
    VALIDATION_ATTEMPT="$VALIDATION_ATTEMPT_DIR/validation-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.json"
    [[ ! -e "$VALIDATION_ATTEMPT" ]] || {
        echo "ERROR: refusing to overwrite immutable validation attempt $VALIDATION_ATTEMPT" >&2
        exit 121
    }
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$VALIDATE_RELEASE_SCRIPT" \
        --release "$PHASE04_STAGE_DIR/release" \
        --manifest "$PHASE04_STAGE_DIR/release_manifest.json" \
        --cleaning-manifest "$CLEANING_MANIFEST" \
        --decontamination-manifest "$DECONTAMINATION_MANIFEST" \
        --dedup-manifest "$DEDUP_MANIFEST" \
        --dedup-decisions "$DECISIONS" \
        --output "$VALIDATION_ATTEMPT" \
        --temporary-directory "$PHASE04_STAGE_DIR/duckdb-tmp" \
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}" \
        --threads "${DUCKDB_THREADS:-128}"
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" - "$VALIDATION_ATTEMPT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if (
    value.get("schema_version") != "full_cpt_release_validation_v1"
    or value.get("integrity_contract_version") != "full_cpt_release_integrity_v1"
    or value.get("status") != "passed"
    or value.get("failed_checks") != []
):
    raise SystemExit(f"ERROR: validation attempt is not promotable: {path}")
PY
    mv "$VALIDATION_ATTEMPT" "$PHASE04_STAGE_DIR/validation.json"
fi

# Never accept the mere existence of a validation receipt.  A failed validator
# deliberately leaves an immutable diagnostic JSON, so resume must inspect all
# three fail-closed fields before the stage can be marked complete.
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" - "$PHASE04_STAGE_DIR/validation.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != "full_cpt_release_validation_v1":
    raise SystemExit(f"ERROR: unsupported release validation schema: {path}")
if value.get("integrity_contract_version") != "full_cpt_release_integrity_v1":
    raise SystemExit(f"ERROR: release validation integrity contract mismatch: {path}")
if value.get("status") != "passed" or value.get("failed_checks") != []:
    raise SystemExit(
        f"ERROR: release validation did not pass: status={value.get('status')!r} "
        f"failed_checks={value.get('failed_checks')!r}"
    )
PY

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/release_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/release_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json" \
    "$PHASE04_STAGE_DIR/token_waterfall.json" \
    "$PHASE04_STAGE_DIR/validation.json"
echo "LOCAL_RELEASE_VALIDATED=$PHASE04_STAGE_DIR/release"
echo "Publication remains a separate, explicitly confirmed stage."
