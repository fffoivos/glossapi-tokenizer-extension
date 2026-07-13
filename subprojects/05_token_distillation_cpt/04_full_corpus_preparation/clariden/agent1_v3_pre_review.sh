#!/usr/bin/env bash
# CPU-only executor for Agent 1's isolated v3 normalize, lineage, and
# deterministic review-packet stages. It never invokes Codex.
set -euo pipefail

HERE=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/agent1_v3_paths.env"
source "$HERE/agent1_v3_common.sh"
export PYTHONDONTWRITEBYTECODE=1

readonly NORMALIZE_STAGE="10-normalize"
readonly LINEAGE_STAGE="20-lineage"
readonly REVIEW_PACKET_STAGE="30-review-packet"
readonly QUALITY_DOCUMENT_SCHEMA="dataset_quality_document_v1"
readonly QUALITY_SUMMARY_SCHEMA="dataset_quality_summary_v1"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

usage() {
    cat >&2 <<'EOF'
usage: agent1_v3_pre_review.sh <normalize|lineage|review-packet>

Run one immutable Agent 1 v3 CPU stage from an allocated Clariden job. New
metadata/manifests are written beneath the contract-created Capstor attempt;
canonical shards, work state, checkpoints, and full-scan roots are written
beneath its paired IOPS bulk-data attempt.
review-packet runs the mandatory GlossAPI full scan before packet selection and
does not invoke a review model.

review-packet requires CODEX_REVIEW_MODEL as frozen review metadata plus the
Phase-0-bound AGENT1_V3_GLOSSAPI_BUILD_RECEIPT and
AGENT1_V3_GLOSSAPI_MODULE_DIR. It validates that v3 runtime before its
mandatory full scan. A v2 runtime/stage root is never accepted.
EOF
}

require_file() {
    local label=$1 path=$2
    [[ -f "$path" && -s "$path" ]] || die "$label is missing or empty: $path"
}

require_directory() {
    local label=$1 path=$2
    [[ -d "$path" ]] || die "$label is missing: $path"
}

require_positive_integer() {
    local label=$1 value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$label must be a positive integer: $value"
}

assert_under() {
    local label=$1 path=$2 root=$3
    case "$path" in
        "$root"/*) ;;
        *) die "$label must remain under $root: $path" ;;
    esac
}

assert_under_one_of() {
    local label=$1 path=$2 first_root=$3 second_root=$4
    case "$path" in
        "$first_root"/*|"$second_root"/*) ;;
        *) die "$label must remain under $first_root or $second_root: $path" ;;
    esac
}

run_python() {
    uenv run "$AGENT1_V3_UENV" --view=default -- \
        "$AGENT1_V3_RUNTIME_VENV/bin/python" "$@"
}

candidate_source_ids() {
    run_python - "$AGENT1_V3_CANDIDATE_ROSTER" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != "agent1_full_corpus_v3_candidate_roster_v1":
    raise SystemExit(f"{path}: unsupported v3 candidate roster")
if value.get("base_source_id") != "nanochat_base":
    raise SystemExit(f"{path}: expected base_source_id nanochat_base")
sources = value.get("candidate_source_ids")
routes = value.get("review_routes")
if not isinstance(sources, list) or not sources or not isinstance(routes, dict):
    raise SystemExit(f"{path}: candidate routes are missing")
if len(sources) != len(set(sources)):
    raise SystemExit(f"{path}: duplicate candidate source_id")
allowed = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
if any(not isinstance(source, str) or not allowed.fullmatch(source) for source in sources):
    raise SystemExit(f"{path}: unsafe candidate source_id")
if set(routes) != set(sources):
    raise SystemExit(f"{path}: review-route coverage mismatch")
if any(route not in {"html_web", "pdf_ocr", "mixed", "structured"} for route in routes.values()):
    raise SystemExit(f"{path}: unsupported review route")
for source in sources:
    print(source)
PY
}

review_seed() {
    run_python - "$AGENT1_V3_REVIEW_POLICY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
review = value.get("review") if isinstance(value, dict) else None
if value.get("schema_version") != "agent1_full_corpus_v3_policy_v1" or not isinstance(review, dict):
    raise SystemExit(f"{path}: unsupported v3 review policy")
seed = review.get("seed")
if not isinstance(seed, str) or not seed:
    raise SystemExit(f"{path}: missing frozen review seed")
if review.get("required_model") != "gpt-5.6-luna":
    raise SystemExit(f"{path}: review model policy drift")
if review.get("model_environment_variable") != "CODEX_REVIEW_MODEL":
    raise SystemExit(f"{path}: review model environment-variable drift")
if review.get("no_model_fallback") is not True:
    raise SystemExit(f"{path}: model fallback must be disabled")
print(seed)
PY
}

stage_attempt_dir() {
    local stage=$1 storage=$2 path expected
    path=$(run_python "$AGENT1_V3_CONTRACT_SCRIPT" get-stage-attempt-dir \
        --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
        --stage "$stage" --storage "$storage")
    case "$storage" in
        metadata) expected="$AGENT1_V3_RUN_ROOT/stages/$stage/attempts" ;;
        data) expected="$AGENT1_V3_DATA_ROOT/stages/$stage/attempts" ;;
        *) die "unsupported stage storage selector: $storage" ;;
    esac
    assert_under "upstream $stage $storage attempt" "$path" "$expected"
    require_directory "upstream $stage $storage attempt" "$path"
    printf '%s\n' "$path"
}

stage_output() {
    local stage=$1 basename=$2 path metadata_root data_root
    path=$(run_python "$AGENT1_V3_CONTRACT_SCRIPT" get-stage-output \
        --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
        --stage "$stage" --basename "$basename")
    metadata_root=$(stage_attempt_dir "$stage" metadata)
    data_root=$(stage_attempt_dir "$stage" data)
    assert_under_one_of "upstream $stage output" "$path" "$metadata_root" "$data_root"
    require_file "upstream $stage output" "$path"
    printf '%s\n' "$path"
}

prepare_stage() {
    local stage=$1 parameters_json=$2
    shift 2
    agent1_v3_begin_stage "$stage" --parameters-json "$parameters_json" "$@"
    require_directory "contract-created metadata attempt directory" "$AGENT1_V3_ATTEMPT_DIR"
    require_directory "contract-created bulk-data attempt directory" "$AGENT1_V3_DATA_ATTEMPT_DIR"
    # begin-stage creates the evidence parent; only then can the CPU gate
    # write its allocation record into this job-unique attempt directory.
    agent1_v3_require_compute_cpu
    agent1_v3_mask_gpu_visibility
    mkdir -p "$AGENT1_V3_DATA_ATTEMPT_DIR/tmp" "$AGENT1_V3_DATA_ATTEMPT_DIR/work"
    export TMPDIR="$AGENT1_V3_DATA_ATTEMPT_DIR/tmp"
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4
}

validate_normalization_roster_coverage() {
    local manifest=$1 output=$2
    assert_under "normalization roster coverage" "$output" "$AGENT1_V3_ATTEMPT_DIR"
    run_python - "$manifest" "$AGENT1_V3_CANDIDATE_ROSTER" "$output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, roster_path, output_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
roster = json.loads(roster_path.read_text(encoding="utf-8"))
if manifest.get("schema_version") != "full_cpt_normalization_manifest_v1":
    raise SystemExit(f"{manifest_path}: unsupported normalization manifest")
if roster.get("schema_version") != "agent1_full_corpus_v3_candidate_roster_v1":
    raise SystemExit(f"{roster_path}: unsupported v3 candidate roster")
if roster.get("base_source_id") != "nanochat_base":
    raise SystemExit(f"{roster_path}: expected base_source_id nanochat_base")
candidates = roster.get("candidate_source_ids")
review_routes = roster.get("review_routes")
if (
    not isinstance(candidates, list)
    or not candidates
    or len(candidates) != len(set(candidates))
    or any(not isinstance(source, str) or not source for source in candidates)
    or not isinstance(review_routes, dict)
):
    raise SystemExit(f"{roster_path}: invalid v3 candidate roster")
allowed_routes = {"html_web", "pdf_ocr", "mixed", "structured"}

def exact_routes(name, fallback):
    value = roster.get(name)
    if value is None:
        value = fallback
    if not isinstance(value, dict):
        raise SystemExit(f"{roster_path}: {name} must be an object")
    missing = sorted(set(candidates) - set(value))
    extra = sorted(set(value) - set(candidates))
    if missing or extra:
        raise SystemExit(
            f"{roster_path}: {name} coverage drift; missing={missing}, extra={extra}"
        )
    invalid = sorted(
        source for source in candidates
        if not isinstance(value[source], str) or value[source] not in allowed_routes
    )
    if invalid:
        raise SystemExit(f"{roster_path}: unsupported {name} routes: {invalid}")
    return {source: value[source] for source in candidates}

review_routes = exact_routes("review_routes", None)
source_routes = exact_routes("source_routes", review_routes)
extraction_routes = exact_routes("extraction_routes", review_routes)
route_declarations = {
    source: {
        "source_route": source_routes[source],
        "review_route": review_routes[source],
        "extraction_route": extraction_routes[source],
    }
    for source in candidates
}
roster_binding = {
    "path": str(roster_path.resolve()),
    "bytes": roster_path.stat().st_size,
    "sha256": hashlib.sha256(roster_path.read_bytes()).hexdigest(),
    "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
    "base_source_id": roster.get("base_source_id"),
    "candidate_source_ids": candidates,
    "review_routes": review_routes,
    "source_routes": source_routes,
    "extraction_routes": extraction_routes,
    "route_declarations": route_declarations,
}
normalizer_roster = manifest.get("candidate_roster")
if not isinstance(normalizer_roster, dict):
    raise SystemExit(f"{manifest_path}: missing v3-bound candidate roster")
for key, expected in roster_binding.items():
    if normalizer_roster.get(key) != expected:
        raise SystemExit(f"{manifest_path}: frozen candidate roster binding drift for {key}")
source_coverage = manifest.get("candidate_roster_source_coverage")
if not isinstance(source_coverage, dict) or source_coverage.get("status") != "passed":
    raise SystemExit(f"{manifest_path}: missing passed candidate roster/source coverage")
if source_coverage.get("candidate_source_ids") != candidates:
    raise SystemExit(f"{manifest_path}: candidate roster/source coverage candidate drift")
expected_normalizable_sources = sorted(["nanochat_base", *candidates])
for field in ("normalizable_registry_source_ids", "acquisition_artifact_source_ids"):
    if source_coverage.get(field) != expected_normalizable_sources:
        raise SystemExit(f"{manifest_path}: candidate roster/source coverage drift for {field}")
canonical_route_coverage = manifest.get("candidate_roster_canonical_route_coverage")
if (
    not isinstance(canonical_route_coverage, dict)
    or canonical_route_coverage.get("schema_version")
    != "agent1_v3_canonical_route_coverage_v1"
    or canonical_route_coverage.get("status") != "passed"
):
    raise SystemExit(f"{manifest_path}: missing passed canonical route provenance coverage")
coverage_rows = canonical_route_coverage.get("sources")
if not isinstance(coverage_rows, list):
    raise SystemExit(f"{manifest_path}: canonical route provenance source rows are missing")
coverage_by_source = {
    row.get("source_id"): row for row in coverage_rows if isinstance(row, dict)
}
if set(coverage_by_source) != set(candidates):
    raise SystemExit(f"{manifest_path}: canonical route provenance source coverage drift")
for source in candidates:
    row = coverage_by_source[source]
    if (
        row.get("status") != "passed"
        or int(row.get("normalized_documents", 0)) < 1
        or any(row.get(field) != route_declarations[source][field] for field in route_declarations[source])
    ):
        raise SystemExit(f"{manifest_path}: canonical route provenance drift for {source}")
sources = {str(row.get("source_id")): row for row in manifest.get("sources", []) if isinstance(row, dict)}
required = ["nanochat_base", *candidates]
missing = [source for source in required if source not in sources]
if missing:
    raise SystemExit(f"{manifest_path}: missing normalized v3 sources: {missing}")
zero = [
    source for source in required
    if int(sources[source].get("counts", {}).get("documents_emitted", 0)) < 1
]
if zero:
    raise SystemExit(f"{manifest_path}: v3 sources emitted zero documents: {zero}")
payload = {
    "schema_version": "agent1_v3_normalization_roster_coverage_v1",
    "normalization_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "candidate_roster": roster_binding,
    "candidate_roster_sha256": roster_binding["sha256"],
    "base_source_id": "nanochat_base",
    "candidate_source_ids": candidates,
    "review_routes": review_routes,
    "source_routes": source_routes,
    "extraction_routes": extraction_routes,
    "route_declarations": route_declarations,
    "canonical_route_coverage_schema": canonical_route_coverage["schema_version"],
    "source_document_counts": {
        source: int(sources[source]["counts"]["documents_emitted"])
        for source in required
    },
    "status": "passed",
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    require_file "normalization roster coverage" "$output"
}

normalization_context() {
    local manifest metadata_attempt data_attempt canonical
    manifest=$(stage_output "$NORMALIZE_STAGE" "normalization_manifest.json")
    metadata_attempt=$(stage_attempt_dir "$NORMALIZE_STAGE" metadata)
    data_attempt=$(stage_attempt_dir "$NORMALIZE_STAGE" data)
    assert_under "normalization manifest" "$manifest" "$metadata_attempt"
    canonical="$data_attempt/canonical"
    assert_under "normalized canonical root" "$canonical" "$data_attempt"
    require_directory "normalized canonical root" "$canonical"
    printf '%s\t%s\n' "$manifest" "$canonical"
}

assert_candidate_canonical_roots() {
    local canonical_root=$1 source
    local -a candidates=()
    mapfile -t candidates < <(candidate_source_ids)
    [[ ${#candidates[@]} -gt 0 ]] || die "candidate roster contains no sources"
    require_directory "normalized nanochat base" "$canonical_root/nanochat_base"
    [[ -n "$(find "$canonical_root/nanochat_base" -type f -name '*.parquet' -print -quit)" ]] || \
        die "normalized nanochat base has no Parquet shards"
    for source in "${candidates[@]}"; do
        require_directory "normalized candidate $source" "$canonical_root/$source"
        [[ -n "$(find "$canonical_root/$source" -type f -name '*.parquet' -print -quit)" ]] || \
            die "normalized candidate $source has no Parquet shards"
    done
}

quality_runtime_paths() {
    local receipt=${AGENT1_V3_GLOSSAPI_BUILD_RECEIPT:-}
    local modules=${AGENT1_V3_GLOSSAPI_MODULE_DIR:-}
    [[ -n "$receipt" ]] || die "AGENT1_V3_GLOSSAPI_BUILD_RECEIPT is required"
    [[ -n "$modules" ]] || die "AGENT1_V3_GLOSSAPI_MODULE_DIR is required"
    # The build happens in Phase 0 and is already an immutable run-contract
    # input.  Do not accept a merely plausible receipt from elsewhere under a
    # v3 scratch root: it must be the exact published runtime and frozen file.
    run_python - \
        "$receipt" "$modules" "$AGENT1_V3_RUN_ROOT" "$AGENT1_V3_DATA_ROOT" \
        "$AGENT1_V3_GLOSSAPI_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_arg, modules_arg, run_root_arg, data_root_arg, commit = sys.argv[1:]
receipt = Path(receipt_arg).resolve()
modules = Path(modules_arg).resolve()
run_root = Path(run_root_arg).resolve()
data_root = Path(data_root_arg).resolve()
expected_root = data_root / "runtime" / f"glossapi-rust-quality-{commit}"
expected_receipt = expected_root / "build_receipt.json"
expected_modules = expected_root / "modules"
if not receipt.is_file() or receipt.stat().st_size < 1:
    raise SystemExit(f"passed v3 GlossAPI build receipt is missing or empty: {receipt}")
if not modules.is_dir():
    raise SystemExit(f"v3 GlossAPI module directory is missing: {modules}")
if receipt != expected_receipt.resolve() or modules != expected_modules.resolve():
    raise SystemExit("GlossAPI runtime path differs from the immutable v3 Phase-0 publication path")

contract_path = run_root / "run_contract.json"
try:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    binding = contract["inputs"]["glossapi_build_receipt"]
    frozen_path = Path(str(binding["path"])).resolve()
    frozen_bytes = int(binding["bytes"])
    frozen_sha256 = str(binding["sha256"])
except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"{contract_path}: missing frozen GlossAPI build receipt binding") from exc
digest = hashlib.sha256()
with receipt.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
if (
    frozen_path != receipt
    or frozen_bytes != receipt.stat().st_size
    or frozen_sha256 != digest.hexdigest()
):
    raise SystemExit("GlossAPI build receipt differs from the v3 frozen run contract")
print(f"{receipt}\t{modules}")
PY
}

run_normalize() {
    local acquisition_receipt=${AGENT1_V3_MERGED_ACQUISITION_RECEIPT:-"$AGENT1_V3_RUN_ROOT/phase0/merged_acquisition_receipt.json"}
    local rows_per_shard=${AGENT1_V3_NORMALIZE_ROWS_PER_SHARD:-50000}
    local cpu_count=${SLURM_CPUS_PER_TASK:-}
    local workers=${AGENT1_V3_NORMALIZE_WORKERS:-${SLURM_CPUS_PER_TASK:-}}
    require_file "merged acquisition receipt" "$acquisition_receipt"
    require_file "source registry" "$AGENT1_V3_SOURCE_CONFIG"
    require_file "source aliases" "$AGENT1_V3_SOURCE_ALIASES"
    require_file "v3 candidate roster" "$AGENT1_V3_CANDIDATE_ROSTER"
    require_positive_integer "SLURM_CPUS_PER_TASK" "$cpu_count"
    require_positive_integer "AGENT1_V3_NORMALIZE_ROWS_PER_SHARD" "$rows_per_shard"
    require_positive_integer "AGENT1_V3_NORMALIZE_WORKERS" "$workers"
    (( workers <= cpu_count )) || die "normalization workers exceed allocated CPUs"

    prepare_stage "$NORMALIZE_STAGE" \
        "{\"executor\":\"agent1_v3_pre_review.sh\",\"action\":\"normalize\",\"rows_per_shard\":$rows_per_shard,\"workers\":$workers}" \
        --input acquisition_receipt "$acquisition_receipt" \
        --input source_registry "$AGENT1_V3_SOURCE_CONFIG" \
        --input source_aliases "$AGENT1_V3_SOURCE_ALIASES" \
        --input candidate_roster "$AGENT1_V3_CANDIDATE_ROSTER"

    # Canonical Parquet and normalizer work space are bulk IOPS products.
    # The two compact manifests remain in the paired Capstor attempt.
    local canonical_root="$AGENT1_V3_DATA_ATTEMPT_DIR/canonical"
    local manifest="$AGENT1_V3_ATTEMPT_DIR/normalization_manifest.json"
    local coverage="$AGENT1_V3_ATTEMPT_DIR/normalization_roster_coverage.json"
    run_python "$PHASE04_DIR/scripts/normalize_sources.py" \
        --sources "$AGENT1_V3_SOURCE_CONFIG" \
        --lineage-aliases "$AGENT1_V3_SOURCE_ALIASES" \
        --acquisition-receipt "$acquisition_receipt" \
        --candidate-roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --output "$canonical_root" \
        --manifest "$manifest" \
        --rows-per-shard "$rows_per_shard" \
        --workers "$workers" \
        --large-task-workers 1 \
        --temporary-directory "$AGENT1_V3_DATA_ATTEMPT_DIR/work/normalize" \
        --duckdb-memory-limit "${AGENT1_V3_NORMALIZE_DUCKDB_MEMORY_LIMIT:-240GB}" \
        --duckdb-threads "$workers"
    require_file "normalization manifest" "$manifest"
    require_directory "canonical output" "$canonical_root"
    validate_normalization_roster_coverage "$manifest" "$coverage"
    agent1_v3_finish_stage "$NORMALIZE_STAGE" \
        --output "$manifest" \
        --output "$coverage"
}

run_lineage() {
    local context normalization_manifest canonical_root
    context=$(normalization_context)
    normalization_manifest=${context%%$'\t'*}
    canonical_root=${context#*$'\t'}
    local initial_roster=${AGENT1_V3_NANOCHAT_INITIAL_ROSTER:-"$PHASE04_DIR/configs/nanochat_initial_roster.json"}
    local cpu_count=${SLURM_CPUS_PER_TASK:-}
    local workers=${AGENT1_V3_LINEAGE_INPUT_WORKERS:-${SLURM_CPUS_PER_TASK:-}}
    require_file "normalization manifest" "$normalization_manifest"
    require_file "source registry" "$AGENT1_V3_SOURCE_CONFIG"
    require_file "source aliases" "$AGENT1_V3_SOURCE_ALIASES"
    require_file "Nanochat initial roster" "$initial_roster"
    require_file "v3 candidate roster" "$AGENT1_V3_CANDIDATE_ROSTER"
    require_positive_integer "SLURM_CPUS_PER_TASK" "$cpu_count"
    require_positive_integer "AGENT1_V3_LINEAGE_INPUT_WORKERS" "$workers"
    (( workers <= cpu_count )) || die "lineage input workers exceed allocated CPUs"
    assert_candidate_canonical_roots "$canonical_root"

    prepare_stage "$LINEAGE_STAGE" \
        "{\"executor\":\"agent1_v3_pre_review.sh\",\"action\":\"lineage\",\"input_workers\":$workers}" \
        --input normalization_manifest "$normalization_manifest" \
        --input source_registry "$AGENT1_V3_SOURCE_CONFIG" \
        --input source_aliases "$AGENT1_V3_SOURCE_ALIASES" \
        --input nanochat_initial_roster "$initial_roster" \
        --input candidate_roster "$AGENT1_V3_CANDIDATE_ROSTER"

    # The action ledger and SQLite/spool checkpoint state can be corpus-scale;
    # source-level summaries are compact contract artifacts.
    local registry="$AGENT1_V3_ATTEMPT_DIR/registry.json"
    local actions="$AGENT1_V3_DATA_ATTEMPT_DIR/document_actions.jsonl"
    local novelty="$AGENT1_V3_ATTEMPT_DIR/source_novelty.json"
    local summary="$AGENT1_V3_ATTEMPT_DIR/summary.json"
    local work_root="$AGENT1_V3_DATA_ATTEMPT_DIR/work/lineage"
    local source
    local -a candidates=()
    local -a candidate_args=()
    mapfile -t candidates < <(candidate_source_ids)
    for source in "${candidates[@]}"; do
        candidate_args+=(--candidate-input "$canonical_root/$source")
    done
    mkdir -p "$work_root/sqlite-tmp"
    run_python "$PHASE04_DIR/scripts/build_source_lineage.py" rows \
        --sources-config "$AGENT1_V3_SOURCE_CONFIG" \
        --roster-config "$initial_roster" \
        --aliases-config "$AGENT1_V3_SOURCE_ALIASES" \
        --base-input "$canonical_root/nanochat_base" \
        "${candidate_args[@]}" \
        --registry-manifest-out "$registry" \
        --actions-out "$actions" \
        --novelty-out "$novelty" \
        --summary-out "$summary" \
        --normalization-manifest "$normalization_manifest" \
        --sqlite-work-path "$work_root/lineage.sqlite" \
        --sqlite-temp-directory "$work_root/sqlite-tmp" \
        --sqlite-cache-mb "${AGENT1_V3_LINEAGE_SQLITE_CACHE_MB:-32768}" \
        --input-workers "$workers" \
        --input-spool-directory "$work_root/input-fragments" \
        --canonical-verification-interval "${AGENT1_V3_LINEAGE_CANONICAL_VERIFY_EVERY:-100000}" \
        --resume
    require_file "lineage registry" "$registry"
    require_file "lineage actions" "$actions"
    require_file "lineage novelty" "$novelty"
    require_file "lineage summary" "$summary"
    agent1_v3_finish_stage "$LINEAGE_STAGE" \
        --output "$registry" \
        --output "$actions" \
        --output "$novelty" \
        --output "$summary"
}

validate_full_scan_evidence() {
    local quality_contract=$1 summary=$2 documents=$3 handoff=$4 route_validation=$5 output=$6
    assert_under "full-scan evidence validation" "$output" "$AGENT1_V3_ATTEMPT_DIR"
    assert_under "full-scan quality contract" "$quality_contract" "$AGENT1_V3_DATA_ATTEMPT_DIR"
    assert_under "full-scan quality summary" "$summary" "$AGENT1_V3_DATA_ATTEMPT_DIR"
    assert_under "full-scan quality documents" "$documents" "$AGENT1_V3_DATA_ATTEMPT_DIR"
    assert_under "full-scan compact handoff" "$handoff" "$AGENT1_V3_ATTEMPT_DIR"
    assert_under "full-scan logical route validation" "$route_validation" "$AGENT1_V3_ATTEMPT_DIR"
    run_python - "$quality_contract" "$summary" "$documents" "$handoff" "$route_validation" "$AGENT1_V3_CANDIDATE_ROSTER" "$output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    contract_path,
    summary_path,
    documents_path,
    handoff_path,
    route_validation_path,
    roster_path,
    output_path,
) = map(Path, sys.argv[1:])
if not contract_path.is_file() or contract_path.stat().st_size < 1:
    raise SystemExit(f"{contract_path}: required full-scan quality contract is missing")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
roster = json.loads(roster_path.read_text(encoding="utf-8"))
if not route_validation_path.is_file() or route_validation_path.stat().st_size < 1:
    raise SystemExit(f"{route_validation_path}: logical route validation is missing")
route_validation = json.loads(route_validation_path.read_text(encoding="utf-8"))
if (
    summary.get("schema_version") != "dataset_quality_summary_v1"
    or summary.get("status") != "passed"
    or summary.get("scan_mode") != "full_scan"
):
    raise SystemExit(f"{summary_path}: not a passed mandatory full scan")
candidates = roster.get("candidate_source_ids")
if not isinstance(candidates, list) or sorted(summary.get("selected_source_ids", [])) != sorted(candidates):
    raise SystemExit(f"{summary_path}: full-scan source coverage differs from the v3 roster")
if not all(isinstance(source, str) and source for source in candidates):
    raise SystemExit(f"{roster_path}: invalid candidate source IDs")
roster_sha256 = hashlib.sha256(roster_path.read_bytes()).hexdigest()
review_routes = roster.get("review_routes")
source_routes = roster.get("source_routes", review_routes)
extraction_routes = roster.get("extraction_routes", review_routes)
for field, expected in (
    ("review_routes", review_routes),
    ("source_routes", source_routes),
    ("extraction_routes", extraction_routes),
):
    if not isinstance(expected, dict) or set(expected) != set(candidates):
        raise SystemExit(f"{roster_path}: invalid {field} coverage")
    if route_validation.get(field) != {source: expected[source] for source in sorted(candidates)}:
        raise SystemExit(f"{route_validation_path}: {field} differs from frozen roster")
if (
    route_validation.get("schema_version")
    != "agent1_v3_candidate_roster_route_validation_v1"
    or route_validation.get("roster_sha256") != roster_sha256
    or route_validation.get("candidate_source_ids") != sorted(candidates)
    or route_validation.get("logical_source_priority")
    != "logical_source_then_observed_extraction"
):
    raise SystemExit(f"{route_validation_path}: logical-source route validation drift")
document_output = summary.get("document_output")
if not isinstance(document_output, dict) or document_output.get("path") != documents_path.name:
    raise SystemExit(f"{summary_path}: consolidated document evidence path drift")
if not documents_path.is_file() or documents_path.stat().st_size < 1:
    raise SystemExit(f"{documents_path}: required full-scan metric Parquet is missing")
contract_output = summary.get("contract")
if not isinstance(contract_output, dict) or contract_output.get("path") != contract_path.name:
    raise SystemExit(f"{summary_path}: full-scan quality contract path drift")
digest = hashlib.sha256()
with documents_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
documents_sha256 = digest.hexdigest()
contract_digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
if (
    int(document_output.get("bytes", -1)) != documents_path.stat().st_size
    or document_output.get("sha256") != documents_sha256
):
    raise SystemExit(f"{summary_path}: consolidated document evidence receipt drift")
if (
    int(contract_output.get("bytes", -1)) != contract_path.stat().st_size
    or contract_output.get("sha256") != contract_digest
):
    raise SystemExit(f"{summary_path}: quality contract receipt drift")
handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
if (
    handoff.get("schema_version") != "dataset_quality_site_handoff_v1"
    or handoff.get("status") != "passed"
    or handoff.get("scan_mode") != "full_scan"
):
    raise SystemExit(f"{handoff_path}: incomplete quality handoff")
payload = {
    "schema_version": "agent1_v3_full_scan_evidence_validation_v1",
    "status": "passed",
    "quality_contract_path": str(contract_path.resolve()),
    "quality_contract_bytes": contract_path.stat().st_size,
    "quality_contract_sha256": contract_digest,
    "quality_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "documents_path": str(documents_path.resolve()),
    "documents_bytes": documents_path.stat().st_size,
    "documents_sha256": documents_sha256,
    "quality_handoff_sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
    "candidate_roster_sha256": roster_sha256,
    "candidate_source_ids": candidates,
    "route_validation_path": str(route_validation_path.resolve()),
    "route_validation_sha256": hashlib.sha256(route_validation_path.read_bytes()).hexdigest(),
    "logical_source_priority": route_validation["logical_source_priority"],
    "source_routes": route_validation["source_routes"],
    "review_routes": route_validation["review_routes"],
    "extraction_routes": route_validation["extraction_routes"],
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    require_file "full-scan evidence validation" "$output"
}

run_review_packet() {
    local context normalization_manifest canonical_root
    context=$(normalization_context)
    normalization_manifest=${context%%$'\t'*}
    canonical_root=${context#*$'\t'}
    local lineage_summary
    lineage_summary=$(stage_output "$LINEAGE_STAGE" "summary.json")
    local runtime_context build_receipt module_dir
    runtime_context=$(quality_runtime_paths)
    build_receipt=${runtime_context%%$'\t'*}
    module_dir=${runtime_context#*$'\t'}
    local model=${CODEX_REVIEW_MODEL:-}
    local seed
    local cpu_count=${SLURM_CPUS_PER_TASK:-}
    local quality_threads=${AGENT1_V3_QUALITY_THREADS:-${SLURM_CPUS_PER_TASK:-}}
    local quality_batch_size=${AGENT1_V3_QUALITY_BATCH_SIZE:-4096}
    local quality_quantile_size=${AGENT1_V3_QUALITY_QUANTILE_SAMPLE_SIZE:-8192}
    require_file "normalization manifest" "$normalization_manifest"
    require_file "lineage summary" "$lineage_summary"
    require_file "v3 candidate roster" "$AGENT1_V3_CANDIDATE_ROSTER"
    require_file "v3 review policy" "$AGENT1_V3_REVIEW_POLICY"
    require_file "v3 review prompt" "$AGENT1_V3_REVIEW_PROMPT"
    require_file "v3 review response schema" "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA"
    [[ -n "$model" ]] || die "CODEX_REVIEW_MODEL is required even though this stage does not invoke Codex"
    require_positive_integer "SLURM_CPUS_PER_TASK" "$cpu_count"
    require_positive_integer "AGENT1_V3_QUALITY_THREADS" "$quality_threads"
    require_positive_integer "AGENT1_V3_QUALITY_BATCH_SIZE" "$quality_batch_size"
    require_positive_integer "AGENT1_V3_QUALITY_QUANTILE_SAMPLE_SIZE" "$quality_quantile_size"
    (( quality_threads <= cpu_count )) || die "quality threads exceed allocated CPUs"
    (( quality_quantile_size >= 100 )) || die "quality quantile sample size must be at least 100"
    seed=$(review_seed)
    [[ -n "$seed" ]] || die "frozen review seed is empty"
    assert_candidate_canonical_roots "$canonical_root"

    prepare_stage "$REVIEW_PACKET_STAGE" \
        "{\"executor\":\"agent1_v3_pre_review.sh\",\"action\":\"review-packet\",\"scan_mode\":\"full_scan\"}" \
        --input normalization_manifest "$normalization_manifest" \
        --input lineage_summary "$lineage_summary" \
        --input candidate_roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --input review_policy "$AGENT1_V3_REVIEW_POLICY" \
        --input review_prompt "$AGENT1_V3_REVIEW_PROMPT" \
        --input review_response_schema "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA" \
        --input glossapi_build_receipt "$build_receipt"

    # The full GlossAPI output includes per-document metrics and resumable
    # batch checkpoints, so it must stay on IOPS.  Its compact validation,
    # handoff, and review packet remain receipt artifacts on Capstor.
    local route_validation="$AGENT1_V3_ATTEMPT_DIR/roster_route_validation.json"
    local quality_root="$AGENT1_V3_DATA_ATTEMPT_DIR/quality-full-scan"
    local quality_contract="$quality_root/contract.json"
    local quality_summary="$quality_root/$QUALITY_SUMMARY_SCHEMA.json"
    local quality_documents="$quality_root/$QUALITY_DOCUMENT_SCHEMA.parquet"
    local quality_handoff="$AGENT1_V3_ATTEMPT_DIR/full_scan_quality_handoff.json"
    local quality_validation="$AGENT1_V3_ATTEMPT_DIR/full_scan_evidence_validation.json"
    local requests="$AGENT1_V3_ATTEMPT_DIR/review_requests.jsonl"
    local packet_manifest="$AGENT1_V3_ATTEMPT_DIR/review_packet_manifest.json"
    local source
    local -a candidates=()
    local -a quality_source_args=()

    run_python "$PHASE04_DIR/scripts/agent1_v3_review.py" validate-roster \
        --roster "$AGENT1_V3_CANDIDATE_ROSTER" --output "$route_validation"
    require_file "v3 route validation" "$route_validation"
    mapfile -t candidates < <(candidate_source_ids)
    for source in "${candidates[@]}"; do
        quality_source_args+=(--source-id "$source")
    done
    (
        export PYTHONPATH="$module_dir${PYTHONPATH:+:$PYTHONPATH}"
        run_python "$PHASE04_DIR/scripts/profile_dataset_quality_rust.py" validate-build-receipt \
            --receipt "$build_receipt" --expected-commit "$AGENT1_V3_GLOSSAPI_COMMIT"
        run_python "$PHASE04_DIR/scripts/profile_dataset_quality_rust.py" run \
            --scan-mode full_scan \
            --normalization-manifest "$normalization_manifest" \
            --canonical-root "$canonical_root" \
            --build-receipt "$build_receipt" \
            --expected-commit "$AGENT1_V3_GLOSSAPI_COMMIT" \
            "${quality_source_args[@]}" \
            --batch-size "$quality_batch_size" \
            --threads "$quality_threads" \
            --quantile-sample-size "$quality_quantile_size" \
            --scratch-dir "$AGENT1_V3_DATA_ATTEMPT_DIR/work/quality-full-scan" \
            --output-dir "$quality_root" \
            --site-handoff "$quality_handoff"
    )
    require_file "GlossAPI full-scan contract" "$quality_contract"
    require_file "GlossAPI full-scan summary" "$quality_summary"
    require_file "GlossAPI full-scan documents" "$quality_documents"
    require_file "GlossAPI full-scan handoff" "$quality_handoff"
    validate_full_scan_evidence \
        "$quality_contract" "$quality_summary" "$quality_documents" "$quality_handoff" \
        "$route_validation" "$quality_validation"

    # This only materializes masked compact review copies and immutable request
    # bindings. It deliberately does not execute `codex` or admit a source.
    run_python "$PHASE04_DIR/scripts/agent1_v3_review_packet.py" \
        --full-scan-evidence "$quality_documents" \
        --canonical-root "$canonical_root" \
        --roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --policy "$AGENT1_V3_REVIEW_POLICY" \
        --prompt "$AGENT1_V3_REVIEW_PROMPT" \
        --response-schema "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA" \
        --seed "$seed" \
        --model "$model" \
        --code-commit "$AGENT1_V3_EXPECTED_COMMIT" \
        --output "$requests" \
        --manifest "$packet_manifest"
    require_file "review requests" "$requests"
    require_file "review packet manifest" "$packet_manifest"
    agent1_v3_finish_stage "$REVIEW_PACKET_STAGE" \
        --output "$route_validation" \
        --output "$quality_root/contract.json" \
        --output "$quality_summary" \
        --output "$quality_documents" \
        --output "$quality_handoff" \
        --output "$quality_validation" \
        --output "$requests" \
        --output "$packet_manifest"
    echo "REVIEW_PACKET_MATERIALIZED=$requests"
    echo "MODEL_INVOCATION=not_run"
}

main() {
    local action=${1:-}
    [[ $# -eq 1 ]] || { usage; exit 2; }
    case "$action" in
        normalize|lineage|review-packet) ;;
        -h|--help|help|'') usage; exit 0 ;;
        *) usage; die "unsupported pre-review action: $action" ;;
    esac
    agent1_v3_init_paths
    agent1_v3_require_clean_commit
    agent1_v3_require_runtime
    case "$action" in
        normalize) run_normalize ;;
        lineage) run_lineage ;;
        review-packet) run_review_packet ;;
    esac
}

main "$@"
