#!/usr/bin/env bash
# CPU-only ordered executor for Agent 1's v3 post-review lane.
#
# The handler deliberately separates a pending admission packet (Stage 40)
# from an approved, user-hash-confirmed admission (Stage 50).  It then applies
# the required order exactly: dedup -> GreekMMLU -> anonymization ->
# verification-only post-mask duplicate scan -> prestructural freeze.  It does
# not invoke Codex, GPUs, structural removal, or publication.
set -euo pipefail

HERE=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/agent1_v3_paths.env"
source "$HERE/agent1_v3_common.sh"
export PYTHONDONTWRITEBYTECODE=1

readonly ADMISSION_STAGE="40-admission"
readonly DEDUP_STAGE="50-dedup"
readonly GREEKMMLU_STAGE="55-greekmmlu-freeze"
readonly DECONTAMINATION_STAGE="60-decontamination"
readonly ANONYMIZATION_STAGE="65-anonymization-sanitization"
readonly PRESTRUCTURAL_STAGE="70-prestructural-freeze"
readonly NORMALIZE_STAGE="10-normalize"
readonly LINEAGE_STAGE="20-lineage"
readonly REVIEW_PACKET_STAGE="30-review-packet"
readonly REVIEW_EVIDENCE_STAGE="35-quality-review-evidence"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

usage() {
    cat >&2 <<'EOF'
usage: agent1_v3_post_admission.sh <admission|dedup|greekmmlu-freeze|decontamination|anonymization-sanitization|prestructural-freeze>

All actions are receipt-bound Clariden CPU stages in the isolated Agent 1 v3
lane.  `admission` creates only a pending packet.  It needs the compact,
evidence-bound AGENT1_V3_ADMISSION_PROPOSAL path and intentionally cannot
confirm it.  `dedup` needs AGENT1_V3_ADMISSION_CONFIRMATION, which must have
been separately created only after explicit user confirmation of the exact
Stage-40 packet SHA-256.

The final two actions never apply a second deduplication: Stage 65 only runs a
verification scan after masking and stops before receipt completion if it finds
any new collision; Stage 70 only freezes a non-publishable prestructural
corpus for a later child run with an Agent 2 handoff.
EOF
}

require_file() {
    local label=$1 path=$2
    [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or a symlink: $path"
}

require_directory() {
    local label=$1 path=$2
    [[ -d "$path" && ! -L "$path" ]] || die "$label is missing or a symlink: $path"
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
    # The allocation record must be written only after the immutable stage
    # contract created the job-unique attempt path.
    agent1_v3_require_compute_cpu
    agent1_v3_mask_gpu_visibility
    mkdir -p "$AGENT1_V3_DATA_ATTEMPT_DIR/tmp" "$AGENT1_V3_DATA_ATTEMPT_DIR/work"
    export TMPDIR="$AGENT1_V3_DATA_ATTEMPT_DIR/tmp"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4
}

normalization_context() {
    local manifest data_attempt canonical
    manifest=$(stage_output "$NORMALIZE_STAGE" "normalization_manifest.json")
    data_attempt=$(stage_attempt_dir "$NORMALIZE_STAGE" data)
    canonical="$data_attempt/canonical"
    assert_under "normalized canonical root" "$canonical" "$data_attempt"
    require_directory "normalized canonical root" "$canonical"
    printf '%s\t%s\n' "$manifest" "$canonical"
}

validate_manifest_parquet_root() {
    local manifest=$1 root=$2 schema=$3 status=$4 receipt_key=$5
    require_file "Parquet-root manifest" "$manifest"
    require_directory "receipt-bound Parquet root" "$root"
    run_python - "$manifest" "$root" "$schema" "$status" "$receipt_key" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

manifest_path, root, schema, status, receipt_key = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
root_path = Path(root).resolve()
if manifest.get("schema_version") != schema or manifest.get("status") != status:
    raise SystemExit(f"{manifest_path}: expected {schema} with status={status}")
rows = manifest.get("files")
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"{manifest_path}: non-empty parquet receipt list is required")
expected = set()
for index, row in enumerate(rows):
    receipt = row if receipt_key == "" else row.get(receipt_key) if isinstance(row, dict) else None
    if not isinstance(receipt, dict):
        raise SystemExit(f"{manifest_path}: missing {receipt_key or 'file'} receipt at index {index}")
    relative = receipt.get("path")
    if not isinstance(relative, str) or not relative:
        raise SystemExit(f"{manifest_path}: invalid receipt path at index {index}")
    candidate = (root_path / relative).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise SystemExit(f"{manifest_path}: receipt escapes expected root: {relative}") from exc
    if candidate in expected or not candidate.is_file() or candidate.is_symlink():
        raise SystemExit(f"{manifest_path}: missing/duplicate/unsafe receipt file: {relative}")
    digestor = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digestor.update(block)
    digest = digestor.hexdigest()
    if candidate.stat().st_size != receipt.get("bytes") or digest != receipt.get("sha256"):
        raise SystemExit(f"{manifest_path}: receipt bytes/SHA-256 drift: {relative}")
    metadata = pq.ParquetFile(candidate).metadata
    if metadata.num_rows != receipt.get("rows") or metadata.num_row_groups != receipt.get("row_groups"):
        raise SystemExit(f"{manifest_path}: Parquet metadata drift: {relative}")
    expected.add(candidate)
actual = {path.resolve() for path in root_path.rglob("*.parquet") if not path.name.startswith(".")}
if actual != expected:
    raise SystemExit(
        f"{manifest_path}: Parquet root receipt coverage drift; expected={len(expected)} actual={len(actual)}"
    )
PY
}

validate_dedup_policy() {
    require_file "frozen v3 policy" "$AGENT1_V3_POLICY"
    run_python - "$AGENT1_V3_POLICY" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
near = value.get("dedup", {}).get("near") if isinstance(value, dict) else None
expected = {
    "token_shingle_size": 5,
    "minhash_permutations": 128,
    "bands": 32,
    "rows_per_band": 4,
    "similarity_threshold": 0.85,
    "oversized_bucket_ceiling": 5000,
    "greek_diacritic_policy": "preserve",
}
if value.get("schema_version") != "agent1_full_corpus_v3_policy_v1" or near != expected:
    raise SystemExit("frozen v3 dedup policy differs from the required production recipe")
if value.get("dedup", {}).get("representative_precedence") != [
    "nanochat_base", "license", "extraction_completeness", "quality", "provenance", "stable_id"
]:
    raise SystemExit("frozen v3 representative precedence drift")
PY
}

generic_dedup_bound_decisions() {
    local manifest=$1 expected_input=$2
    run_python - "$manifest" "$expected_input" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, expected_input = map(Path, sys.argv[1:])
value = json.loads(manifest_path.read_text(encoding="utf-8"))
if value.get("schema_version") != "full_cpt_dedup_wrapper_manifest_v1" or value.get("status") != "completed":
    raise SystemExit(f"{manifest_path}: completed generic dedup manifest required")
if Path(str(value.get("input", ""))).resolve() != expected_input.resolve():
    raise SystemExit(f"{manifest_path}: generic dedup input root drift")
if value.get("recipe", {}).get("mode") != "production":
    raise SystemExit(f"{manifest_path}: generic dedup must use the production recipe")
output = value.get("dedup_output")
if not isinstance(output, dict):
    raise SystemExit(f"{manifest_path}: generic dedup output is missing")
receipt = output.get("content_bound_decisions")
if not isinstance(receipt, dict) or receipt.get("schema_version") != "full_cpt_dedup_decisions_content_bound_v1":
    raise SystemExit(f"{manifest_path}: content-bound decision receipt is missing")
path = Path(str(receipt.get("path", ""))).resolve()
digestor = hashlib.sha256()
if path.is_file():
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digestor.update(block)
digest = digestor.hexdigest() if path.is_file() else ""
if not path.is_file() or path.stat().st_size != receipt.get("bytes") or digest != receipt.get("sha256"):
    raise SystemExit(f"{manifest_path}: content-bound decision receipt drift")
print(path)
PY
}

validate_admission_confirmation_for_run() {
    local confirmation=$1 packet=$2
    require_file "explicit admission confirmation" "$confirmation"
    require_file "pending admission packet" "$packet"
    run_python "$PHASE04_DIR/scripts/agent1_v3_admission.py" validate \
        --confirmation "$confirmation" --roster "$AGENT1_V3_CANDIDATE_ROSTER"
    run_python - "$confirmation" "$packet" "$AGENT1_V3_RUN_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

confirmation_path, packet_path = map(Path, sys.argv[1:3])
run_id = sys.argv[3]
confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
packet = confirmation.get("packet")
if confirmation.get("run_id") != run_id:
    raise SystemExit("admission confirmation belongs to a different v3 run")
if not isinstance(packet, dict) or Path(str(packet.get("path", ""))).resolve() != packet_path.resolve():
    raise SystemExit("admission confirmation does not bind this Stage-40 packet")
actual = hashlib.sha256(packet_path.read_bytes()).hexdigest()
if packet.get("sha256") != actual or confirmation.get("user_confirmed_packet_sha256") != actual:
    raise SystemExit("admission confirmation does not bind the exact Stage-40 packet bytes")
PY
}

assert_zero_decontamination_ambiguity() {
    local manifest=$1 output=$2
    run_python - "$manifest" "$output" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path, output_path = map(Path, sys.argv[1:])
value = json.loads(manifest_path.read_text(encoding="utf-8"))
if value.get("schema_version") != "agent1_full_corpus_v3_decontamination_manifest_v1" or value.get("status") != "passed":
    raise SystemExit("completed v3 decontamination manifest is required")
counts = value.get("counts")
if not isinstance(counts, dict) or not isinstance(counts.get("quarantine"), int):
    raise SystemExit("decontamination manifest lacks an integer ambiguity/quarantine count")
if counts["quarantine"] != 0:
    raise SystemExit(
        f"GreekMMLU ambiguity policy is quarantine_and_stop; {counts['quarantine']} rows require an explicit user decision"
    )
payload = {
    "schema_version": "agent1_full_corpus_v3_decontamination_ambiguity_gate_v1",
    "status": "passed",
    "decontamination_manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
    "ambiguous_quarantine_rows": 0,
    "policy": "quarantine_and_stop",
}
descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

assert_zero_postmask_duplicates() {
    local report=$1
    run_python - "$report" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != "agent1_full_corpus_v3_postmask_duplicate_verification_v1" or value.get("status") != "passed":
    raise SystemExit("completed v3 post-mask duplicate report is required")
if value.get("verification_only") is not True or value.get("second_deduplication_applied") is not False:
    raise SystemExit("post-mask scan does not declare verification-only semantics")
binding = value.get("anonymization_manifest")
inventory = value.get("anonymized_corpus_inventory")
closure = value.get("inventory_closure")
if not isinstance(binding, dict) or not isinstance(inventory, dict) or not isinstance(closure, dict):
    raise SystemExit("post-mask duplicate report lacks anonymization/inventory closure")
if int(inventory.get("rows", -1)) < 1 or any(
    int(closure.get(name, -1)) != 0
    for name in (
        "duplicate_corpus_uids", "duplicate_bound_uids", "corpus_without_decision",
        "decision_outside_corpus", "text_hash_drift", "raw_without_bound",
        "bound_without_raw", "raw_bound_field_drift",
    )
):
    raise SystemExit("post-mask duplicate report inventory closure drift")
count = value.get("material_new_duplicate_count")
if not isinstance(count, int) or count < 0:
    raise SystemExit("post-mask duplicate report has invalid material collision count")
if count:
    raise SystemExit(
        f"post-mask duplicate scan found {count} new collisions; stop for an explicit user post-mask dedup decision"
    )
PY
}

assert_prestructural_run() {
    run_python - "$AGENT1_V3_RUN_ROOT/run_contract.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("prestructural_only") is not True:
    raise SystemExit("Stage 70 must freeze a prestructural-only parent run; structural work belongs in a later child run")
PY
}

run_admission() {
    local proposed=${AGENT1_V3_ADMISSION_PROPOSAL:-}
    [[ -n "$proposed" ]] || die "Stage 40 requires AGENT1_V3_ADMISSION_PROPOSAL (complete compact proposal; it does not confirm it)"
    require_file "admission proposal" "$proposed"

    local requests packet quality_summary lineage_summary novelty
    local responses response_receipt adjudication_receipt closure review_sample_summary review_sample_handoff
    requests=$(stage_output "$REVIEW_PACKET_STAGE" "review_requests.jsonl")
    packet=$(stage_output "$REVIEW_PACKET_STAGE" "review_packet_manifest.json")
    quality_summary=$(stage_output "$REVIEW_PACKET_STAGE" "dataset_quality_summary_v2.json")
    lineage_summary=$(stage_output "$LINEAGE_STAGE" "summary.json")
    novelty=$(stage_output "$LINEAGE_STAGE" "source_novelty.json")
    responses=$(stage_output "$REVIEW_EVIDENCE_STAGE" "responses.jsonl")
    response_receipt=$(stage_output "$REVIEW_EVIDENCE_STAGE" "response_execution_receipt.json")
    adjudication_receipt=$(stage_output "$REVIEW_EVIDENCE_STAGE" "adjudication_execution_receipt.json")
    closure=$(stage_output "$REVIEW_EVIDENCE_STAGE" "quality_review_evidence_closure.json")
    review_sample_summary=$(stage_output "$REVIEW_EVIDENCE_STAGE" "masked-review-sample-quality-summary.json")
    review_sample_handoff=$(stage_output "$REVIEW_EVIDENCE_STAGE" "masked-review-sample-quality-handoff.json")

    prepare_stage "$ADMISSION_STAGE" \
        '{"executor":"agent1_v3_post_admission.sh","action":"admission","decision_state":"pending_explicit_user_hash_confirmation"}' \
        --input candidate_roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --input license_adjudication "$AGENT1_V3_LICENSE_ADJUDICATION" \
        --input review_requests "$requests" \
        --input review_packet_manifest "$packet" \
        --input quality_summary "$quality_summary" \
        --input lineage_summary "$lineage_summary" \
        --input source_novelty "$novelty" \
        --input review_responses "$responses" \
        --input response_execution_receipt "$response_receipt" \
        --input adjudication_execution_receipt "$adjudication_receipt" \
        --input stage35_review_closure "$closure" \
        --input review_sample_quality_summary "$review_sample_summary" \
        --input review_sample_quality_handoff "$review_sample_handoff" \
        --input proposed_decisions "$proposed"

    local aggregate pending_packet packet_build_result
    aggregate="$AGENT1_V3_ATTEMPT_DIR/source_review_aggregate.json"
    pending_packet="$AGENT1_V3_ATTEMPT_DIR/admission_packet.json"
    packet_build_result="$AGENT1_V3_ATTEMPT_DIR/admission_packet_build_result.json"
    run_python "$PHASE04_DIR/scripts/agent1_v3_review_aggregate.py" build \
        --run-id "$AGENT1_V3_RUN_ID" \
        --roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --review-packet "$packet" \
        --review-requests "$requests" \
        --review-responses "$responses" \
        --response-execution-receipt "$response_receipt" \
        --adjudication-execution-receipt "$adjudication_receipt" \
        --stage35-review-closure "$closure" \
        --review-sample-quality-summary "$review_sample_summary" \
        --review-sample-quality-handoff "$review_sample_handoff" \
        --quality-summary "$quality_summary" \
        --lineage-summary "$lineage_summary" \
        --source-novelty "$novelty" \
        --license-adjudication "$AGENT1_V3_LICENSE_ADJUDICATION" \
        --output "$aggregate"
    run_python "$PHASE04_DIR/scripts/agent1_v3_review_aggregate.py" validate \
        --aggregate "$aggregate" --roster "$AGENT1_V3_CANDIDATE_ROSTER"
    run_python "$PHASE04_DIR/scripts/agent1_v3_admission.py" build-packet \
        --run-id "$AGENT1_V3_RUN_ID" \
        --roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --review-aggregate "$aggregate" \
        --proposed-decisions "$proposed" \
        --output "$pending_packet" > "$packet_build_result"
    run_python "$PHASE04_DIR/scripts/agent1_v3_admission.py" validate-packet \
        --packet "$pending_packet" --roster "$AGENT1_V3_CANDIDATE_ROSTER"
    require_file "source review aggregate" "$aggregate"
    require_file "pending admission packet" "$pending_packet"
    require_file "admission packet build result" "$packet_build_result"
    agent1_v3_finish_stage "$ADMISSION_STAGE" \
        --output "$aggregate" \
        --output "$pending_packet" \
        --output "$packet_build_result" \
        --output "$AGENT1_V3_ALLOCATION_EVIDENCE"
    local packet_sha
    packet_sha=$(run_python - "$pending_packet" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
    printf 'PENDING_ADMISSION_PACKET=%s\n' "$pending_packet"
    printf 'PENDING_ADMISSION_PACKET_SHA256=%s\n' "$packet_sha"
    printf 'DESTRUCTIVE_PROGRESSION=blocked_pending_explicit_user_hash_confirmation\n'
}

run_production_near_detector() {
    local input=$1 detector_root=$2 detector_manifest=$3 workers=$4 duckdb_threads=$5
    run_python "$PHASE04_DIR/scripts/run_full_corpus_dedup.py" \
        --input "$input" \
        --staged-input "$detector_root/staged-input" \
        --state-root "$detector_root/state" \
        --run-root "$detector_root/run" \
        --manifest "$detector_manifest" \
        --temporary-directory "$detector_root/tmp" \
        --memory-limit "${AGENT1_V3_DEDUP_MEMORY_LIMIT:-400GB}" \
        --workers "$workers" --duckdb-threads "$duckdb_threads" \
        --greek-diacritic-policy preserve --minhash-threshold 0.85 \
        --num-perm 128 --bands 32 --rows-per-band 4 --shingle-mode token \
        --shingle-size 5 --max-bucket-size 5000
}

run_dedup() {
    local confirmation=${AGENT1_V3_ADMISSION_CONFIRMATION:-}
    [[ -n "$confirmation" ]] || die "Stage 50 requires AGENT1_V3_ADMISSION_CONFIRMATION created after explicit user confirmation of the Stage-40 packet SHA-256"
    local admission_packet normalization_manifest canonical_context canonical_root
    admission_packet=$(stage_output "$ADMISSION_STAGE" "admission_packet.json")
    validate_admission_confirmation_for_run "$confirmation" "$admission_packet"
    normalization_context=$(normalization_context)
    normalization_manifest=${normalization_context%%$'\t'*}
    canonical_root=${normalization_context#*$'\t'}
    validate_dedup_policy
    local cpu_count=${SLURM_CPUS_PER_TASK:-} workers duckdb_threads
    workers=${AGENT1_V3_DEDUP_WORKERS:-$cpu_count}
    require_positive_integer "SLURM_CPUS_PER_TASK" "$cpu_count"
    require_positive_integer "AGENT1_V3_DEDUP_WORKERS" "$workers"
    (( workers <= cpu_count )) || die "dedup workers exceed allocated CPUs"
    duckdb_threads=${AGENT1_V3_DEDUP_DUCKDB_THREADS:-32}
    require_positive_integer "AGENT1_V3_DEDUP_DUCKDB_THREADS" "$duckdb_threads"
    (( duckdb_threads <= workers )) || duckdb_threads=$workers

    prepare_stage "$DEDUP_STAGE" \
        "{\"executor\":\"agent1_v3_post_admission.sh\",\"action\":\"dedup\",\"workers\":$workers,\"duckdb_threads\":$duckdb_threads,\"recipe\":\"greek_cpt_text_dedup_v1\",\"ordered_passes\":[\"exact_content_work_representation\",\"within_source_near\",\"cross_candidate_near\",\"candidate_to_nanochat_near\"],\"admission_mode\":\"explicit_hash_confirmed_user_confirmation\"}" \
        --input normalization_manifest "$normalization_manifest" \
        --input admission_packet "$admission_packet" \
        --input admission_confirmation "$confirmation" \
        --input dedup_policy "$AGENT1_V3_POLICY"

    local pool pool_manifest
    local exact_ledger exact_ledger_manifest exact_materialized exact_materialization_manifest
    local within_partition within_partition_manifest within_source_list within_detector_root
    local within_ledger within_ledger_manifest within_materialized within_materialization_manifest
    local candidate_scope candidate_scope_manifest cross_detector_root cross_scope_ledger cross_scope_ledger_manifest
    local cross_ledger cross_ledger_manifest cross_materialized cross_materialization_manifest
    local candidate_to_nanochat_detector_root candidate_to_nanochat_ledger candidate_to_nanochat_ledger_manifest
    local ledger ledger_manifest materialized materialization_manifest
    local candidate_rows decisions source_root source_name detector_pass_root detector_manifest
    local -a within_detector_manifests within_decisions receipt_outputs near_args
    pool="$AGENT1_V3_DATA_ATTEMPT_DIR/admitted-pool"
    pool_manifest="$AGENT1_V3_ATTEMPT_DIR/admitted_pool_manifest.json"
    exact_ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-exact-identity-ledger.parquet"
    exact_ledger_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_exact_identity_ledger_manifest.json"
    exact_materialized="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-exact-identity-survivors"
    exact_materialization_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_exact_identity_materialization_manifest.json"
    within_partition="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-within-source-inputs"
    within_partition_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_within_source_partition_manifest.json"
    within_source_list="$AGENT1_V3_DATA_ATTEMPT_DIR/work/within-source-inputs.txt"
    within_detector_root="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-within-source-detectors"
    within_ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-within-source-ledger.parquet"
    within_ledger_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_within_source_ledger_manifest.json"
    within_materialized="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-within-source-survivors"
    within_materialization_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_within_source_materialization_manifest.json"
    candidate_scope="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-cross-candidate-input"
    candidate_scope_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_cross_candidate_scope_manifest.json"
    cross_detector_root="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-cross-candidate-detector"
    cross_scope_ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-cross-candidate-scope-ledger.parquet"
    cross_scope_ledger_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_cross_candidate_scope_ledger_manifest.json"
    cross_ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-cross-candidate-ledger.parquet"
    cross_ledger_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_cross_candidate_ledger_manifest.json"
    cross_materialized="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-cross-candidate-survivors"
    cross_materialization_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_cross_candidate_materialization_manifest.json"
    candidate_to_nanochat_detector_root="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-candidate-to-nanochat-detector"
    candidate_to_nanochat_ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-candidate-to-nanochat-ledger.parquet"
    candidate_to_nanochat_ledger_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_candidate_to_nanochat_ledger_manifest.json"
    ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/dedup-ledger.parquet"
    ledger_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_ledger_manifest.json"
    materialized="$AGENT1_V3_DATA_ATTEMPT_DIR/deduplicated"
    materialization_manifest="$AGENT1_V3_ATTEMPT_DIR/dedup_materialization_manifest.json"

    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" prepare-pool \
        --input "$canonical_root" --admission-confirmation "$confirmation" \
        --output "$pool" --manifest "$pool_manifest"

    # Exact canonical identity is a hard boundary before the production
    # detector sees any text: content, work, and representation first.
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" exact-reconcile \
        --pool "$pool" --output-ledger "$exact_ledger" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/exact-identity.sqlite" \
        --manifest "$exact_ledger_manifest"
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" materialize \
        --pool "$pool" --ledger "$exact_ledger" --output "$exact_materialized" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/exact-identity-materialize.sqlite" \
        --manifest "$exact_materialization_manifest"

    # Run the first near pass in isolated source roots.  The combined ledger
    # must cover every exact-identity survivor exactly once.
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" partition-within-source \
        --input "$exact_materialized" --output "$within_partition" \
        --manifest "$within_partition_manifest"
    run_python - "$within_partition_manifest" > "$within_source_list" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version") != "agent1_full_corpus_v3_dedup_within_source_partition_manifest_v1" or value.get("status") != "passed":
    raise SystemExit("within-source partition manifest is not passed")
rows = value.get("sources")
if not isinstance(rows, list) or not rows:
    raise SystemExit("within-source partition manifest has no source roots")
for row in rows:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"]:
        raise SystemExit("within-source partition manifest has an invalid source path")
    print(row["path"])
PY
    mapfile -t within_source_roots < "$within_source_list"
    (( ${#within_source_roots[@]} > 0 )) || die "within-source partition unexpectedly produced no source roots"
    within_detector_manifests=()
    within_decisions=()
    for source_root in "${within_source_roots[@]}"; do
        assert_under "within-source detector input" "$source_root" "$within_partition"
        require_directory "within-source detector input" "$source_root"
        source_name=$(basename -- "$source_root")
        detector_pass_root="$within_detector_root/$source_name"
        detector_manifest="$detector_pass_root/dedup_detector_manifest.json"
        run_production_near_detector "$source_root" "$detector_pass_root" "$detector_manifest" "$workers" "$duckdb_threads"
        decisions=$(generic_dedup_bound_decisions "$detector_manifest" "$source_root")
        assert_under "within-source content-bound decisions" "$decisions" "$detector_pass_root"
        within_detector_manifests+=("$detector_manifest")
        within_decisions+=("$decisions")
    done
    near_args=()
    for decisions in "${within_decisions[@]}"; do
        near_args+=(--raw-decisions "$decisions")
    done
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" near-reconcile \
        --pool "$exact_materialized" --pass-kind within-source "${near_args[@]}" \
        --output-ledger "$within_ledger" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/within-source-reconcile.sqlite" \
        --manifest "$within_ledger_manifest"
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" materialize \
        --pool "$exact_materialized" --ledger "$within_ledger" --output "$within_materialized" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/within-source-materialize.sqlite" \
        --manifest "$within_materialization_manifest"

    # The second near pass admits only additive candidates.  Its scoped ledger
    # is extended with explicit Nanochat-base passthrough rows before the next
    # full-pool materialization boundary.
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" filter-candidates \
        --input "$within_materialized" --output "$candidate_scope" \
        --manifest "$candidate_scope_manifest"
    candidate_rows=$(run_python - "$candidate_scope_manifest" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("schema_version") != "agent1_full_corpus_v3_dedup_scope_manifest_v1" or value.get("status") != "passed":
    raise SystemExit("candidate scope manifest is not passed")
count = value.get("counts", {}).get("candidate_rows")
if not isinstance(count, int) or count < 0:
    raise SystemExit("candidate scope manifest has an invalid candidate_rows count")
print(count)
PY
)
    if (( candidate_rows > 0 )); then
        detector_manifest="$cross_detector_root/dedup_detector_manifest.json"
        run_production_near_detector "$candidate_scope" "$cross_detector_root" "$detector_manifest" "$workers" "$duckdb_threads"
        decisions=$(generic_dedup_bound_decisions "$detector_manifest" "$candidate_scope")
        assert_under "cross-candidate content-bound decisions" "$decisions" "$cross_detector_root"
        run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" near-reconcile \
            --pool "$candidate_scope" --pass-kind cross-candidate --raw-decisions "$decisions" \
            --output-ledger "$cross_scope_ledger" \
            --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/cross-candidate-scope-reconcile.sqlite" \
            --manifest "$cross_scope_ledger_manifest"
        run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" extend-candidate-scope-ledger \
            --pool "$within_materialized" --scope-ledger "$cross_scope_ledger" \
            --output-ledger "$cross_ledger" \
            --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/cross-candidate-extend.sqlite" \
            --manifest "$cross_ledger_manifest"
    else
        run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" passthrough-near \
            --pool "$within_materialized" --pass-kind cross-candidate \
            --reason "no_admitted_candidate_survivors_after_within_source_pass" \
            --output-ledger "$cross_ledger" \
            --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/cross-candidate-passthrough.sqlite" \
            --manifest "$cross_ledger_manifest"
    fi
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" materialize \
        --pool "$within_materialized" --ledger "$cross_ledger" --output "$cross_materialized" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/cross-candidate-materialize.sqlite" \
        --manifest "$cross_materialization_manifest"

    # The final detector sees the post-cross survivors but its reconciler
    # rejects any nontrivial component that is not candidate-to-Nanochat.
    if (( candidate_rows > 0 )); then
        detector_manifest="$candidate_to_nanochat_detector_root/dedup_detector_manifest.json"
        run_production_near_detector "$cross_materialized" "$candidate_to_nanochat_detector_root" "$detector_manifest" "$workers" "$duckdb_threads"
        decisions=$(generic_dedup_bound_decisions "$detector_manifest" "$cross_materialized")
        assert_under "candidate-to-Nanochat content-bound decisions" "$decisions" "$candidate_to_nanochat_detector_root"
        run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" near-reconcile \
            --pool "$cross_materialized" --pass-kind candidate-to-nanochat --raw-decisions "$decisions" \
            --output-ledger "$candidate_to_nanochat_ledger" \
            --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/candidate-to-nanochat-reconcile.sqlite" \
            --manifest "$candidate_to_nanochat_ledger_manifest"
    else
        run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" passthrough-near \
            --pool "$cross_materialized" --pass-kind candidate-to-nanochat \
            --reason "no_admitted_candidate_survivors_for_candidate_to_nanochat_pass" \
            --output-ledger "$candidate_to_nanochat_ledger" \
            --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/candidate-to-nanochat-passthrough.sqlite" \
            --manifest "$candidate_to_nanochat_ledger_manifest"
    fi

    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" compose-ordered-ledgers \
        --pool "$pool" \
        --stage exact_content_work_representation --stage-ledger "$exact_ledger" --stage-manifest "$exact_ledger_manifest" \
        --stage within_source_near --stage-ledger "$within_ledger" --stage-manifest "$within_ledger_manifest" \
        --stage cross_candidate_near --stage-ledger "$cross_ledger" --stage-manifest "$cross_ledger_manifest" \
        --stage candidate_to_nanochat_near --stage-ledger "$candidate_to_nanochat_ledger" --stage-manifest "$candidate_to_nanochat_ledger_manifest" \
        --output-ledger "$ledger" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/compose-ordered-ledgers.sqlite" \
        --manifest "$ledger_manifest"
    run_python "$PHASE04_DIR/scripts/agent1_v3_dedup.py" materialize \
        --pool "$pool" --ledger "$ledger" --output "$materialized" \
        --work-database "$AGENT1_V3_DATA_ATTEMPT_DIR/work/materialize.sqlite" \
        --manifest "$materialization_manifest"

    receipt_outputs=(
        "$pool_manifest"
        "$exact_ledger" "$exact_ledger_manifest" "$exact_materialization_manifest"
        "$within_partition_manifest" "$within_ledger" "$within_ledger_manifest" "$within_materialization_manifest"
        "$candidate_scope_manifest" "$cross_ledger" "$cross_ledger_manifest" "$cross_materialization_manifest"
        "$candidate_to_nanochat_ledger" "$candidate_to_nanochat_ledger_manifest"
        "$ledger" "$ledger_manifest" "$materialization_manifest"
        "${within_detector_manifests[@]}" "${within_decisions[@]}"
    )
    if (( candidate_rows > 0 )); then
        receipt_outputs+=(
            "$cross_detector_root/dedup_detector_manifest.json"
            "$cross_scope_ledger" "$cross_scope_ledger_manifest"
            "$candidate_to_nanochat_detector_root/dedup_detector_manifest.json"
        )
        decisions=$(generic_dedup_bound_decisions "$cross_detector_root/dedup_detector_manifest.json" "$candidate_scope")
        receipt_outputs+=("$decisions")
        decisions=$(generic_dedup_bound_decisions "$candidate_to_nanochat_detector_root/dedup_detector_manifest.json" "$cross_materialized")
        receipt_outputs+=("$decisions")
    fi
    for path in "${receipt_outputs[@]}"; do
        require_file "Stage 50 output" "$path"
    done
    require_directory "deduplicated corpus" "$materialized"
    local -a finish_args=()
    for path in "${receipt_outputs[@]}"; do
        finish_args+=(--output "$path")
    done
    finish_args+=(--output "$AGENT1_V3_ALLOCATION_EVIDENCE")
    agent1_v3_finish_stage "$DEDUP_STAGE" "${finish_args[@]}"
}

run_greekmmlu_freeze() {
    local dedup_manifest dedup_attempt deduplicated
    dedup_manifest=$(stage_output "$DEDUP_STAGE" "dedup_materialization_manifest.json")
    dedup_attempt=$(stage_attempt_dir "$DEDUP_STAGE" data)
    deduplicated="$dedup_attempt/deduplicated"
    require_directory "deduplicated corpus" "$deduplicated"
    validate_manifest_parquet_root "$dedup_manifest" "$deduplicated" \
        "agent1_full_corpus_v3_dedup_materialization_manifest_v1" "passed" ""
    local registry builder freezer eval_dir
    registry=${AGENT1_V3_GREEKMMLU_REGISTRY:-"$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json"}
    builder=${AGENT1_V3_GREEKMMLU_QUERY_BUILDER:-"$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py"}
    freezer="$PHASE04_DIR/scripts/freeze_greekmmlu_queries.py"
    eval_dir=$(cd -- "$(dirname "$registry")" && pwd)
    require_file "Greek benchmark registry" "$registry"
    require_file "GreekMMLU query builder" "$builder"
    require_file "GreekMMLU query freezer" "$freezer"
    local revision
    revision=$(run_python - "$registry" <<'PY'
import json
import re
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = [row for row in value.get("benchmarks", []) if isinstance(row, dict) and row.get("id") == "greekmmlu"]
if len(rows) != 1 or not re.fullmatch(r"[0-9a-f]{40}", str(rows[0].get("revision", ""))):
    raise SystemExit("registry must contain one GreekMMLU row with an immutable 40-hex revision")
if rows[0].get("source") != "dascim/GreekMMLU" or rows[0].get("config") != "All" or rows[0].get("split") != "test":
    raise SystemExit("GreekMMLU registry source/config/split drift")
print(rows[0]["revision"])
PY
)

    prepare_stage "$GREEKMMLU_STAGE" \
        "{\"executor\":\"agent1_v3_post_admission.sh\",\"action\":\"greekmmlu-freeze\",\"dataset_repo_id\":\"dascim/GreekMMLU\",\"dataset_revision\":\"$revision\",\"split\":\"test\"}" \
        --input dedup_materialization_manifest "$dedup_manifest" \
        --input greek_benchmark_registry "$registry" \
        --input greekmmlu_query_builder "$builder" \
        --input greekmmlu_query_freezer "$freezer" \
        --input greekmmlu_policy "$AGENT1_V3_POLICY"

    local queries builder_summary query_manifest
    queries="$AGENT1_V3_DATA_ATTEMPT_DIR/queries.jsonl"
    builder_summary="$AGENT1_V3_ATTEMPT_DIR/greekmmlu_builder_summary.json"
    query_manifest="$AGENT1_V3_ATTEMPT_DIR/greekmmlu_query_manifest.json"
    export HF_HOME="$AGENT1_V3_DATA_ATTEMPT_DIR/hf-cache/greekmmlu"
    mkdir -p "$HF_HOME"
    run_python "$builder" --registry "$registry" --eval-dir "$eval_dir" --benchmarks greekmmlu \
        --output-jsonl "$queries" --summary-json "$builder_summary"
    run_python "$freezer" \
        --queries-jsonl "$queries" --output "$query_manifest" \
        --dataset-revision "$revision" --required-split test \
        --registry "$registry" --builder-summary "$builder_summary"
    for path in "$queries" "$builder_summary" "$query_manifest"; do
        require_file "Stage 55 output" "$path"
    done
    agent1_v3_finish_stage "$GREEKMMLU_STAGE" \
        --output "$queries" \
        --output "$builder_summary" \
        --output "$query_manifest" \
        --output "$AGENT1_V3_ALLOCATION_EVIDENCE"
}

run_decontamination() {
    local dedup_manifest dedup_attempt deduplicated queries query_manifest
    dedup_manifest=$(stage_output "$DEDUP_STAGE" "dedup_materialization_manifest.json")
    dedup_attempt=$(stage_attempt_dir "$DEDUP_STAGE" data)
    deduplicated="$dedup_attempt/deduplicated"
    queries=$(stage_output "$GREEKMMLU_STAGE" "queries.jsonl")
    query_manifest=$(stage_output "$GREEKMMLU_STAGE" "greekmmlu_query_manifest.json")
    require_directory "deduplicated corpus" "$deduplicated"
    validate_manifest_parquet_root "$dedup_manifest" "$deduplicated" \
        "agent1_full_corpus_v3_dedup_materialization_manifest_v1" "passed" ""

    prepare_stage "$DECONTAMINATION_STAGE" \
        '{"executor":"agent1_v3_post_admission.sh","action":"decontamination","workers":1,"ambiguity_policy":"quarantine_and_stop"}' \
        --input dedup_materialization_manifest "$dedup_manifest" \
        --input greekmmlu_queries "$queries" \
        --input greekmmlu_query_manifest "$query_manifest" \
        --input greekmmlu_policy "$AGENT1_V3_POLICY"

    local output dropped quarantine ledger manifest ambiguity_gate
    output="$AGENT1_V3_DATA_ATTEMPT_DIR/decontaminated"
    dropped="$AGENT1_V3_DATA_ATTEMPT_DIR/dropped-greekmmlu"
    quarantine="$AGENT1_V3_DATA_ATTEMPT_DIR/quarantine-greekmmlu"
    ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/greekmmlu-ledger"
    manifest="$AGENT1_V3_ATTEMPT_DIR/decontamination_manifest.json"
    ambiguity_gate="$AGENT1_V3_ATTEMPT_DIR/decontamination_ambiguity_gate.json"
    run_python "$PHASE04_DIR/scripts/agent1_v3_decontaminate.py" \
        --input "$deduplicated" --output "$output" --dropped "$dropped" \
        --quarantine "$quarantine" --ledger "$ledger" --manifest "$manifest" \
        --queries-jsonl "$queries" --benchmark-manifest "$query_manifest" --workers 1
    require_file "decontamination manifest" "$manifest"
    # A non-zero quarantine count is intentionally a hard stop.  The manifest
    # remains available for inspection, but this stage receives no receipt and
    # Stage 65 cannot start until a new explicit decision/run is made.
    assert_zero_decontamination_ambiguity "$manifest" "$ambiguity_gate"
    require_directory "decontaminated corpus" "$output"
    require_file "decontamination ambiguity gate" "$ambiguity_gate"
    agent1_v3_finish_stage "$DECONTAMINATION_STAGE" \
        --output "$manifest" \
        --output "$ambiguity_gate" \
        --output "$AGENT1_V3_ALLOCATION_EVIDENCE"
}

run_anonymization() {
    local decontam_manifest decontam_ambiguity_gate decontam_attempt decontaminated
    decontam_manifest=$(stage_output "$DECONTAMINATION_STAGE" "decontamination_manifest.json")
    decontam_ambiguity_gate=$(stage_output "$DECONTAMINATION_STAGE" "decontamination_ambiguity_gate.json")
    decontam_attempt=$(stage_attempt_dir "$DECONTAMINATION_STAGE" data)
    decontaminated="$decontam_attempt/decontaminated"
    require_directory "decontaminated corpus" "$decontaminated"
    validate_manifest_parquet_root "$decontam_manifest" "$decontaminated" \
        "agent1_full_corpus_v3_decontamination_manifest_v1" "passed" "output"
    validate_dedup_policy
    local cpu_count=${SLURM_CPUS_PER_TASK:-} workers duckdb_threads
    workers=${AGENT1_V3_POSTMASK_DEDUP_WORKERS:-$cpu_count}
    require_positive_integer "SLURM_CPUS_PER_TASK" "$cpu_count"
    require_positive_integer "AGENT1_V3_POSTMASK_DEDUP_WORKERS" "$workers"
    (( workers <= cpu_count )) || die "post-mask dedup workers exceed allocated CPUs"
    duckdb_threads=${AGENT1_V3_POSTMASK_DEDUP_DUCKDB_THREADS:-32}
    require_positive_integer "AGENT1_V3_POSTMASK_DEDUP_DUCKDB_THREADS" "$duckdb_threads"
    (( duckdb_threads <= workers )) || duckdb_threads=$workers

    prepare_stage "$ANONYMIZATION_STAGE" \
        "{\"executor\":\"agent1_v3_post_admission.sh\",\"action\":\"anonymization-sanitization\",\"postmask_scan\":\"verification_only\",\"postmask_dedup_workers\":$workers,\"postmask_duckdb_threads\":$duckdb_threads,\"no_second_deduplication\":true}" \
        --input decontamination_manifest "$decontam_manifest" \
        --input decontamination_ambiguity_gate "$decontam_ambiguity_gate" \
        --input anonymization_policy "$AGENT1_V3_POLICY" \
        --input dedup_policy "$AGENT1_V3_POLICY"

    local output dropped quarantine protected_ledger manifest ledger_closure detector_root detector_manifest report
    output="$AGENT1_V3_DATA_ATTEMPT_DIR/anonymized"
    dropped="$AGENT1_V3_DATA_ATTEMPT_DIR/private-data-dropped"
    quarantine="$AGENT1_V3_DATA_ATTEMPT_DIR/diavgeia-pii-quarantine"
    protected_ledger="$AGENT1_V3_DATA_ATTEMPT_DIR/protected-anonymization-ledger"
    manifest="$AGENT1_V3_ATTEMPT_DIR/anonymization_manifest.json"
    ledger_closure="$AGENT1_V3_ATTEMPT_DIR/protected_anonymization_ledger_closure.json"
    detector_root="$AGENT1_V3_DATA_ATTEMPT_DIR/postmask-duplicate-verification"
    detector_manifest="$AGENT1_V3_ATTEMPT_DIR/postmask_dedup_detector_manifest.json"
    report="$AGENT1_V3_ATTEMPT_DIR/postmask_duplicate_report.json"
    run_python "$PHASE04_DIR/scripts/agent1_v3_anonymize.py" \
        --input "$decontaminated" --output "$output" --dropped "$dropped" \
        --quarantine "$quarantine" --protected-ledger "$protected_ledger" \
        --manifest "$manifest" --policy "$AGENT1_V3_POLICY"
    run_python "$PHASE04_DIR/scripts/agent1_v3_anonymization_ledger_closure.py" \
        --anonymization-manifest "$manifest" --protected-ledger-root "$protected_ledger" \
        --output "$ledger_closure"
    run_python "$PHASE04_DIR/scripts/run_full_corpus_dedup.py" \
        --input "$output" \
        --staged-input "$detector_root/staged-input" \
        --state-root "$detector_root/state" \
        --run-root "$detector_root/run" \
        --manifest "$detector_manifest" \
        --temporary-directory "$detector_root/tmp" \
        --memory-limit "${AGENT1_V3_POSTMASK_DEDUP_MEMORY_LIMIT:-400GB}" \
        --workers "$workers" --duckdb-threads "$duckdb_threads" \
        --greek-diacritic-policy preserve --minhash-threshold 0.85 \
        --num-perm 128 --bands 32 --rows-per-band 4 --shingle-mode token \
        --shingle-size 5 --max-bucket-size 5000
    run_python "$PHASE04_DIR/scripts/agent1_v3_postmask_duplicate_report.py" \
        --dedup-wrapper-manifest "$detector_manifest" \
        --anonymization-manifest "$manifest" \
        --source-corpus "$output" --output "$report"
    for path in "$manifest" "$ledger_closure" "$detector_manifest" "$report"; do
        require_file "Stage 65 output" "$path"
    done
    require_directory "anonymized corpus" "$output"
    # As in Stage 60, write the evidence then stop before receipt completion
    # if the permitted masking transform introduced any collision.
    assert_zero_postmask_duplicates "$report"
    agent1_v3_finish_stage "$ANONYMIZATION_STAGE" \
        --output "$manifest" \
        --output "$ledger_closure" \
        --output "$detector_manifest" \
        --output "$report" \
        --output "$AGENT1_V3_ALLOCATION_EVIDENCE"
}

run_prestructural_freeze() {
    assert_prestructural_run
    local pool_manifest dedup_ledger dedup_ledger_manifest dedup_manifest dedup_attempt pool deduplicated
    local decontam_manifest decontam_attempt decontaminated decontam_dropped decontam_quarantine decontam_ledger
    local anonymization_manifest ledger_closure report anonymization_attempt corpus anonym_dropped anonym_quarantine protected_ledger
    local waterfall_script waterfall_work waterfall
    pool_manifest=$(stage_output "$DEDUP_STAGE" "admitted_pool_manifest.json")
    dedup_ledger=$(stage_output "$DEDUP_STAGE" "dedup-ledger.parquet")
    dedup_ledger_manifest=$(stage_output "$DEDUP_STAGE" "dedup_ledger_manifest.json")
    dedup_manifest=$(stage_output "$DEDUP_STAGE" "dedup_materialization_manifest.json")
    decontam_manifest=$(stage_output "$DECONTAMINATION_STAGE" "decontamination_manifest.json")
    anonymization_manifest=$(stage_output "$ANONYMIZATION_STAGE" "anonymization_manifest.json")
    ledger_closure=$(stage_output "$ANONYMIZATION_STAGE" "protected_anonymization_ledger_closure.json")
    report=$(stage_output "$ANONYMIZATION_STAGE" "postmask_duplicate_report.json")
    dedup_attempt=$(stage_attempt_dir "$DEDUP_STAGE" data)
    pool="$dedup_attempt/admitted-pool"
    deduplicated="$dedup_attempt/deduplicated"
    decontam_attempt=$(stage_attempt_dir "$DECONTAMINATION_STAGE" data)
    decontaminated="$decontam_attempt/decontaminated"
    decontam_dropped="$decontam_attempt/dropped-greekmmlu"
    decontam_quarantine="$decontam_attempt/quarantine-greekmmlu"
    decontam_ledger="$decontam_attempt/greekmmlu-ledger"
    anonymization_attempt=$(stage_attempt_dir "$ANONYMIZATION_STAGE" data)
    corpus="$anonymization_attempt/anonymized"
    anonym_dropped="$anonymization_attempt/private-data-dropped"
    anonym_quarantine="$anonymization_attempt/diavgeia-pii-quarantine"
    protected_ledger="$anonymization_attempt/protected-anonymization-ledger"
    waterfall_script="$PHASE04_DIR/scripts/agent1_v3_transformation_waterfall.py"
    for path in "$pool" "$deduplicated" "$decontaminated" "$decontam_dropped" "$decontam_quarantine" "$decontam_ledger" "$corpus" "$anonym_dropped" "$anonym_quarantine" "$protected_ledger"; do
        require_directory "receipt-bound waterfall corpus/ledger root" "$path"
    done
    require_file "v3 transformation waterfall helper" "$waterfall_script"
    require_file "frozen tokenizer" "$AGENT1_V3_TOKENIZER_JSON"
    require_directory "anonymized prestructural corpus" "$corpus"
    validate_manifest_parquet_root "$anonymization_manifest" "$corpus" \
        "agent1_full_corpus_v3_anonymization_manifest_v1" "completed" "output"
    assert_zero_postmask_duplicates "$report"

    prepare_stage "$PRESTRUCTURAL_STAGE" \
        '{"executor":"agent1_v3_post_admission.sh","action":"prestructural-freeze","publish_permitted":false,"structural_state":"awaiting_agent2_handoff","postmask_duplicate_requirement":"zero","waterfall":"full_corpus_receipt_and_route_closure"}' \
        --input dedup_materialization_manifest "$dedup_manifest" \
        --input admitted_pool_manifest "$pool_manifest" \
        --input identity_dedup_ledger "$dedup_ledger" \
        --input identity_dedup_ledger_manifest "$dedup_ledger_manifest" \
        --input decontamination_manifest "$decontam_manifest" \
        --input anonymization_manifest "$anonymization_manifest" \
        --input protected_anonymization_ledger_closure "$ledger_closure" \
        --input postmask_duplicate_report "$report" \
        --input candidate_roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --input tokenizer "$AGENT1_V3_TOKENIZER_JSON" \
        --input structural_policy "$AGENT1_V3_POLICY"

    local prestructural
    prestructural="$AGENT1_V3_ATTEMPT_DIR/prestructural_manifest.json"
    run_python "$PHASE04_DIR/scripts/agent1_v3_structural_gate.py" freeze \
        --run-id "$AGENT1_V3_RUN_ID" \
        --dedup-manifest "$dedup_manifest" \
        --decontamination-manifest "$decontam_manifest" \
        --anonymization-manifest "$anonymization_manifest" \
        --anonymization-ledger "$ledger_closure" \
        --postmask-duplicate-report "$report" \
        --corpus-root "$corpus" --output "$prestructural"
    require_file "prestructural manifest" "$prestructural"
    run_python - "$prestructural" "$AGENT1_V3_RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

path, run_id = Path(sys.argv[1]), sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
if (
    value.get("schema_version") != "agent1_full_corpus_v3_prestructural_manifest_v1"
    or value.get("status") != "prestructural_frozen"
    or value.get("run_id") != run_id
    or value.get("publish_permitted") is not False
    or value.get("structural_state") != "awaiting_agent2_handoff"
):
    raise SystemExit("prestructural manifest closure drift")
PY
    waterfall_work="$AGENT1_V3_DATA_ATTEMPT_DIR/work/transformation-waterfall.sqlite"
    waterfall="$AGENT1_V3_ATTEMPT_DIR/transformation_waterfall.json"
    local allocated_cpus waterfall_rayon_threads
    allocated_cpus=${SLURM_CPUS_PER_TASK:-}
    require_positive_integer "SLURM_CPUS_PER_TASK" "$allocated_cpus"
    waterfall_rayon_threads=${AGENT1_V3_WATERFALL_RAYON_THREADS:-$allocated_cpus}
    require_positive_integer "AGENT1_V3_WATERFALL_RAYON_THREADS" "$waterfall_rayon_threads"
    (( waterfall_rayon_threads <= allocated_cpus )) || \
        die "waterfall Rayon threads exceed allocated CPUs: $waterfall_rayon_threads > $allocated_cpus"
    export RAYON_NUM_THREADS="$waterfall_rayon_threads"
    run_python "$waterfall_script" \
        --dedup-pool "$pool" --dedup-pool-manifest "$pool_manifest" \
        --dedup-ledger "$dedup_ledger" --dedup-ledger-manifest "$dedup_ledger_manifest" \
        --dedup-materialized "$deduplicated" --dedup-materialization-manifest "$dedup_manifest" \
        --decontamination-input "$deduplicated" --decontamination-output "$decontaminated" \
        --decontamination-dropped "$decontam_dropped" --decontamination-quarantine "$decontam_quarantine" \
        --decontamination-ledger "$decontam_ledger" --decontamination-manifest "$decontam_manifest" \
        --anonymization-input "$decontaminated" --anonymization-output "$corpus" \
        --anonymization-dropped "$anonym_dropped" --anonymization-quarantine "$anonym_quarantine" \
        --protected-ledger-root "$protected_ledger" --anonymization-manifest "$anonymization_manifest" \
        --anonymization-ledger-closure "$ledger_closure" --postmask-duplicate-report "$report" \
        --prestructural-manifest "$prestructural" --candidate-roster "$AGENT1_V3_CANDIDATE_ROSTER" \
        --tokenizer-json "$AGENT1_V3_TOKENIZER_JSON" \
        --work-database "$waterfall_work" --output "$waterfall"
    require_file "transformation waterfall" "$waterfall"
    agent1_v3_finish_stage "$PRESTRUCTURAL_STAGE" \
        --output "$prestructural" \
        --output "$waterfall" \
        --output "$AGENT1_V3_ALLOCATION_EVIDENCE"
    printf 'PRESTRUCTURAL_MANIFEST=%s\n' "$prestructural"
    printf 'TRANSFORMATION_WATERFALL=%s\n' "$waterfall"
    printf 'PUBLISH_PERMITTED=false\n'
    printf 'STRUCTURAL_APPLICATION=blocked_pending_new_child_run_and_agent2_handoff\n'
}

main() {
    local action=${1:-}
    [[ $# -eq 1 ]] || { usage; exit 2; }
    case "$action" in
        admission|dedup|greekmmlu-freeze|decontamination|anonymization-sanitization|prestructural-freeze) ;;
        -h|--help|help|'') usage; exit 0 ;;
        *) usage; die "unsupported post-admission action: $action" ;;
    esac
    agent1_v3_init_paths
    agent1_v3_require_clean_commit
    agent1_v3_require_runtime
    case "$action" in
        admission) run_admission ;;
        dedup) run_dedup ;;
        greekmmlu-freeze) run_greekmmlu_freeze ;;
        decontamination) run_decontamination ;;
        anonymization-sanitization) run_anonymization ;;
        prestructural-freeze) run_prestructural_freeze ;;
    esac
}

main "$@"
