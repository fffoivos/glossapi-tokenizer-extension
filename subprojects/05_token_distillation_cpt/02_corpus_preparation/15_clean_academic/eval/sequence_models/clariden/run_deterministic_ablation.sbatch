#!/usr/bin/env bash
# Bounded validation-only deterministic/hybrid ToC+BIB ablation.
# Clariden normal allocations reserve a complete four-GH200 node even when no
# GPU is requested. Keep this job short; it deliberately hides accelerators.
# This comparison reads only the physically test-stripped STRUCT-2K import and
# the joint ladder's validation C0 predictions. It cannot mutate corpus data.
#SBATCH --job-name=cpt_det_ablation
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

PHASE04_CLARIDEN_DIR=${PHASE04_CLARIDEN_DIR:?set the exact checked-out Phase-04 clariden directory}
source "$PHASE04_CLARIDEN_DIR/paths.env"
source "$PHASE04_CLARIDEN_DIR/common.sh"

phase04_require_compute_partition "deterministic ToC+BIB validation ablation"
phase04_require_cpu_request
phase04_require_uenv_python "$RUNTIME_VENV/bin/python"
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

SEQUENCE_DIR="$ACADEMIC_DIR/eval/sequence_models"
STRUCT2K_LOCK="$SEQUENCE_DIR/struct2k_handoff_lock.json"
DETERMINISTIC_CONFIG=${DETERMINISTIC_CONFIG:-$SEQUENCE_DIR/joint_config.json}
DETERMINISTIC_ABLATION_RUN_ID=${DETERMINISTIC_ABLATION_RUN_ID:?set an immutable deterministic ablation run ID}
[[ "$DETERMINISTIC_ABLATION_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || {
    echo "ERROR: invalid DETERMINISTIC_ABLATION_RUN_ID=$DETERMINISTIC_ABLATION_RUN_ID" >&2
    exit 101
}

# Root mode derives canonical receipt-bound paths. Exact-path mode supplies the
# receipt as well, from which the immutable root is inferred. Overrides are
# accepted only when they resolve to the artifact named by that receipt.
if [[ -z "${STRUCT2K_SOURCE_ROOT:-}" ]]; then
    DETERMINISTIC_SOURCE_RECEIPT=${DETERMINISTIC_SOURCE_RECEIPT:?set STRUCT2K_SOURCE_ROOT or the exact DETERMINISTIC_SOURCE_RECEIPT}
    STRUCT2K_SOURCE_ROOT=$(dirname "$DETERMINISTIC_SOURCE_RECEIPT")
fi
DETERMINISTIC_SOURCE_RECEIPT=${DETERMINISTIC_SOURCE_RECEIPT:-$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.receipt.json}
DETERMINISTIC_SILVER=${DETERMINISTIC_SILVER:-$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.jsonl}
DETERMINISTIC_SPLIT_MANIFEST=${DETERMINISTIC_SPLIT_MANIFEST:-$STRUCT2K_SOURCE_ROOT/struct2k.LLM_silver.split.json}

if [[ -z "${JOINT_LADDER_RUN_ROOT:-}" ]]; then
    DETERMINISTIC_LADDER_RECEIPT=${DETERMINISTIC_LADDER_RECEIPT:?set JOINT_LADDER_RUN_ROOT or the exact DETERMINISTIC_LADDER_RECEIPT}
    JOINT_LADDER_RUN_ROOT=$(dirname "$DETERMINISTIC_LADDER_RECEIPT")
fi
DETERMINISTIC_LADDER_RECEIPT=${DETERMINISTIC_LADDER_RECEIPT:-$JOINT_LADDER_RUN_ROOT/run.receipt.json}
DETERMINISTIC_BASE_C0=${DETERMINISTIC_BASE_C0:-$JOINT_LADDER_RUN_ROOT/c0.validation.predictions.jsonl}

OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/classifier_research/deterministic_ablations/$DETERMINISTIC_ABLATION_RUN_ID}
for path in \
    "$STRUCT2K_LOCK" \
    "$DETERMINISTIC_CONFIG" \
    "$DETERMINISTIC_SILVER" \
    "$DETERMINISTIC_SPLIT_MANIFEST" \
    "$DETERMINISTIC_SOURCE_RECEIPT" \
    "$DETERMINISTIC_BASE_C0" \
    "$DETERMINISTIC_LADDER_RECEIPT" \
    "$SEQUENCE_DIR/deterministic_ablation_runner.py"; do
    phase04_require_file "$path"
    [[ ! -L "$path" ]] || {
        echo "ERROR: deterministic ablation inputs must not be symlinks: $path" >&2
        exit 102
    }
done
for root in "$STRUCT2K_SOURCE_ROOT" "$JOINT_LADDER_RUN_ROOT"; do
    [[ -d "$root" && ! -L "$root" ]] || {
        echo "ERROR: immutable input root must be a real directory: $root" >&2
        exit 103
    }
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: immutable deterministic ablation output already exists: $OUTPUT_ROOT" >&2
    exit 104
}
case "$(realpath -m "$OUTPUT_ROOT")/" in
    "$(realpath "$REPO_ROOT")/"*)
        echo "ERROR: OUTPUT_ROOT must be outside the exact clean Git worktree" >&2
        exit 104
        ;;
esac

echo "DETERMINISTIC ToC+BIB ABLATION PLAN"
echo "  source_root=$STRUCT2K_SOURCE_ROOT"
echo "  source_receipt=$DETERMINISTIC_SOURCE_RECEIPT"
echo "  silver=$DETERMINISTIC_SILVER"
echo "  split_manifest=$DETERMINISTIC_SPLIT_MANIFEST"
echo "  ladder_root=$JOINT_LADDER_RUN_ROOT"
echo "  ladder_receipt=$DETERMINISTIC_LADDER_RECEIPT"
echo "  base_c0=$DETERMINISTIC_BASE_C0"
echo "  config=$DETERMINISTIC_CONFIG"
echo "  code_commit=$PHASE04_EXPECTED_COMMIT"
echo "  allowed_split=validation_only"
echo "  historical_test_documents_available=0"
echo "  model_fitting=none"
echo "  corpus_mutation=none"
echo "  accelerators=hidden_and_not_requested"
echo "  allocation_note=Clariden_reserves_one_complete_four_GH200_node"
echo "  allocation_limit=00:10:00"
echo "  output_root=$OUTPUT_ROOT"
if [[ "${CONFIRM_DETERMINISTIC_ABLATION:-0}" != "1" ]]; then
    echo "DRY_RUN_ONLY=1"
    echo "Set CONFIRM_DETERMINISTIC_ABLATION=1 to run this bounded validation-only comparison."
    exit 0
fi

LOCK_ROOT="$RUN_ROOT/classifier_research/deterministic_ablation_locks"
mkdir -p "$LOCK_ROOT"
exec {RUN_LOCK_FD}>"$LOCK_ROOT/$DETERMINISTIC_ABLATION_RUN_ID.lock"
flock -n "$RUN_LOCK_FD" || {
    echo "ERROR: deterministic ablation run ID is already active: $DETERMINISTIC_ABLATION_RUN_ID" >&2
    exit 105
}
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "ERROR: deterministic ablation output appeared while acquiring lock: $OUTPUT_ROOT" >&2
    exit 106
}
mkdir -p "$(dirname "$OUTPUT_ROOT")"
PARTIAL="$OUTPUT_ROOT.partial-${SLURM_JOB_ID:?SLURM_JOB_ID is unset}"
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || {
    echo "ERROR: stale deterministic ablation staging directory exists: $PARTIAL" >&2
    exit 107
}

run_python() {
    uenv run "$PHASE04_UENV" --view=default -- \
        env CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void \
        ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
        PYTHONHASHSEED=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        SEQUENCE_UENV="$PHASE04_UENV" \
        "$RUNTIME_VENV/bin/python" "$@"
}

cd "$ACADEMIC_DIR/eval"
run_python -c 'import os; import sequence_models.deterministic_ablation_runner; assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["NVIDIA_VISIBLE_DEVICES"] == "void" and os.environ["ROCR_VISIBLE_DEVICES"] == "" and os.environ["HIP_VISIBLE_DEVICES"] == ""'

# This verification requires the canonical imported source receipt and proves
# that historical-test rows were physically excluded before this process.
run_python -m sequence_models.struct2k_import verify \
    --root "$STRUCT2K_SOURCE_ROOT" \
    --lock "$STRUCT2K_LOCK" \
    --config "$DETERMINISTIC_CONFIG" >/dev/null
run_python -m sequence_models.bib_ladder verify-published-run \
    --run-root "$JOINT_LADDER_RUN_ROOT" >/dev/null

# Validate all ladder artifacts against the immutable run receipt, then require
# the explicitly selected C0 path to be the receipt's validation-only C0 role.
run_python - "$STRUCT2K_SOURCE_ROOT" "$DETERMINISTIC_SOURCE_RECEIPT" \
    "$DETERMINISTIC_SILVER" "$DETERMINISTIC_SPLIT_MANIFEST" \
    "$JOINT_LADDER_RUN_ROOT" "$DETERMINISTIC_LADDER_RECEIPT" \
    "$DETERMINISTIC_BASE_C0" "$DETERMINISTIC_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from sequence_models.contract import canonical_json_sha256, sha256_file

(
    source_root,
    source_receipt_path,
    silver_path,
    split_path,
    ladder_root,
    ladder_receipt_path,
    base_c0_path,
    config_path,
) = map(lambda value: Path(value).resolve(), sys.argv[1:])

source = json.loads(source_receipt_path.read_text(encoding="utf-8"))
artifacts = source.get("materialized_artifacts")
assert source_receipt_path == source_root / "struct2k.LLM_silver.receipt.json"
assert isinstance(artifacts, dict)
assert silver_path == source_root / artifacts["silver_filename"]
assert split_path == source_root / artifacts["split_manifest_filename"]
assert source["status"] == "pass" and source["production_eligible"] is False
assert source["silver_sha256"] == sha256_file(silver_path)
assert source["split_manifest_sha256"] == sha256_file(split_path)
assert source["config_sha256"] == sha256_file(config_path)
exclusion = source["historical_partition_exclusion"]
assert exclusion["historical_test_rows_emitted"] == 0

ladder = json.loads(ladder_receipt_path.read_text(encoding="utf-8"))
assert ladder_receipt_path == ladder_root / "run.receipt.json"
assert ladder["status"] == "passed_cpu_sealed_retrospective_comparison"
assert ladder["active_classes"] == ["BIB", "TOC"]
assert ladder["production_eligible"] is False
assert ladder["config_sha256"] == sha256_file(config_path)
assert ladder["historically_named_test_partition"]["documents_loaded_by_model_or_validation_processes"] == 0
rows = ladder["artifacts"]
assert ladder["artifact_inventory_sha256"] == canonical_json_sha256(rows)
by_role = {}
for row in rows:
    role = row["role"]
    assert role not in by_role
    relative = Path(row["relative_path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    unresolved = ladder_root / relative
    assert unresolved.is_file() and not unresolved.is_symlink()
    path = unresolved.resolve()
    assert path.stat().st_size == row["bytes"]
    assert sha256_file(path) == row["sha256"]
    by_role[role] = path
assert base_c0_path == by_role["c0_validation_predictions"]
PY

run_python -m sequence_models.deterministic_ablation_runner \
    --silver "$DETERMINISTIC_SILVER" \
    --split-manifest "$DETERMINISTIC_SPLIT_MANIFEST" \
    --base-c0 "$DETERMINISTIC_BASE_C0" \
    --config "$DETERMINISTIC_CONFIG" \
    --allowed-split validation \
    --output-dir "$PARTIAL"

verify_ablation_receipt() {
    run_python - "$1" "$DETERMINISTIC_SILVER" "$DETERMINISTIC_SPLIT_MANIFEST" \
        "$DETERMINISTIC_BASE_C0" "$DETERMINISTIC_CONFIG" "$PHASE04_EXPECTED_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

from sequence_models.contract import sha256_file

root, silver, split, base_c0, config = map(lambda value: Path(value).resolve(), sys.argv[1:6])
expected_commit = sys.argv[6]
receipt_path = root / "ablation.report.json"
assert root.is_dir() and not root.is_symlink()
assert receipt_path.is_file() and not receipt_path.is_symlink()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
assert receipt["schema_version"] == "academic-structure-deterministic-ablation-evaluation-v1"
assert receipt["status"] == "pass"
assert receipt["allowed_split"] == "validation"
assert receipt["evidence_tier"] == "LLM_silver"
assert receipt["code_revision"]["clean"] is True
assert receipt["code_revision"]["commit"] == expected_commit
claims = receipt["execution_claims"]
assert claims["model_fitting_performed"] is False
assert claims["data_discovery_performed"] is False
assert claims["corpus_mutation_performed"] is False
assert claims["sealed_or_test_data_accessed"] is False
assert claims["human_gold_used"] is False
assert claims["production_eligible"] is False
assert claims["production_action_authorized"] is False
expected_inputs = {
    "base_c0_predictions": base_c0,
    "config": config,
    "imported_llm_silver": silver,
    "split_manifest": split,
}
for name, path in expected_inputs.items():
    row = receipt["inputs"][name]
    assert Path(row["path"]).resolve() == path
    assert row["sha256"] == sha256_file(path)
expected_modes = {
    "rules-only",
    "base-plus-rules",
    "base-rules-veto",
    "base-plus-rules-veto",
}
assert set(receipt["outputs"]) == expected_modes
assert set(receipt["modes"]) == expected_modes
for name, row in receipt["outputs"].items():
    path = root / row["path"]
    assert path.is_file() and not path.is_symlink()
    assert row["sha256"] == sha256_file(path)
    assert row["document_count"] == receipt["document_count"]
    assert receipt["modes"][name]["output"] == row
PY
}

verify_ablation_receipt "$PARTIAL"
mv --no-clobber -T "$PARTIAL" "$OUTPUT_ROOT"
[[ ! -e "$PARTIAL" && -s "$OUTPUT_ROOT/ablation.report.json" ]] || {
    echo "ERROR: deterministic ablation no-replace publication failed" >&2
    exit 108
}
verify_ablation_receipt "$OUTPUT_ROOT"
echo "DETERMINISTIC_ABLATION_COMPLETE=$OUTPUT_ROOT"
echo "DETERMINISTIC_ABLATION_RECEIPT=$OUTPUT_ROOT/ablation.report.json"
