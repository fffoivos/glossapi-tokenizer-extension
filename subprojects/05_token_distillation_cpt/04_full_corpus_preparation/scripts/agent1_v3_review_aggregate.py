#!/usr/bin/env python3
"""Build a closed, source-complete review aggregate for Agent 1 v3.

This is deliberately an evidence reducer, not an admission engine.  It joins
only compact, receipt-bound artifacts from the required v3 boundaries:

* Stage 30's deterministic selection and privacy-preserving request packet;
* Stage 35's schema-validated responses, execution receipt, and closed
  adjudication receipt;
* the mandatory full GlossAPI scan, lineage/novelty report, and licence audit.

It never reads canonical text or makes a source decision.  Its output is the
only supported input to ``agent1_v3_admission.py build-packet``.  This keeps a
later human admission proposal grounded in one complete, immutable evidence
bundle rather than a hand-picked subset of review rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import agent1_v3_review as review  # noqa: E402
import agent1_v3_review_evidence as review_evidence  # noqa: E402
import profile_dataset_quality_rust as quality_runtime  # noqa: E402


AGGREGATE_SCHEMA = "agent1_full_corpus_v3_source_review_aggregate_v1"
SOURCE_EVIDENCE_SCHEMA = "agent1_full_corpus_v3_source_review_evidence_v1"
PACKET_SCHEMA = "agent1_v3_review_packet_manifest_v1"
RESPONSE_RECEIPT_SCHEMA = "agent1_v3_codex_review_response_execution_receipt_v1"
ADJUDICATION_RECEIPT_SCHEMA = "agent1_v3_codex_review_adjudication_execution_receipt_v1"
CALIBRATION_RECEIPT_SCHEMA = "agent1_v3_codex_review_calibration_receipt_v1"
STAGE35_CLOSURE_SCHEMA = "agent1_v3_quality_review_evidence_closure_v1"
REVIEW_SAMPLE_QUALITY_SUMMARY_SCHEMA = "agent1_v3_masked_review_sample_quality_summary_v1"
REVIEW_SAMPLE_QUALITY_HANDOFF_SCHEMA = "agent1_v3_masked_review_sample_quality_handoff_v1"
LINEAGE_SCHEMA = "full_cpt_lineage_summary_v1"
NOVELTY_SCHEMA = "full_cpt_source_novelty_v1"
LICENSE_SCHEMA = "full_cpt_source_license_adjudication_v1"
EXPECTED_REVIEW_MODEL = "gpt-5.6-luna"
SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")

# ``source_route`` is the logical acquisition route and therefore controls the
# primary error model.  These fields preserve a per-document observation about
# the representation that reached the corpus without allowing it to replace
# that logical route.
OBSERVED_ROUTE_FIELDS = (
    "observed_extraction_route",
    "observed_extraction_route_basis",
    "observed_extraction_route_evidence",
    "observed_extraction_route_priority",
)
REQUEST_INVENTORY_FIELDS = (
    "review_id",
    "request_sha256",
    "sample_id",
    "reviewer_slot",
    "source_route",
    "extraction_route",
    *OBSERVED_ROUTE_FIELDS,
)
CALIBRATION_REQUEST_IDENTITY_FIELDS = (
    "review_id",
    "request_sha256",
    "reviewer_slot",
    "sample_id",
    "source_id",
    "source_dataset",
    "source_revision",
    "source_route",
    "extraction_route",
    *OBSERVED_ROUTE_FIELDS,
    "sampling_stratum",
    "original_text_sha256",
    "review_copy_sha256",
    "prompt_sha256",
    "response_schema_sha256",
    "model",
    "code_commit",
    "attempt",
)

# An aggregate is a compact reducer, but it is not a trust boundary on its
# own.  A later admission step must be able to reopen every one of these
# receipt-bound inputs and deterministically reproduce the reducer output.
AGGREGATE_INPUT_NAMES = (
    "candidate_roster",
    "review_packet",
    "review_requests",
    "review_responses",
    "response_execution_receipt",
    "adjudication_execution_receipt",
    "stage35_review_closure",
    "review_sample_quality_summary",
    "review_sample_quality_handoff",
    "quality_summary",
    "lineage_summary",
    "source_novelty",
    "license_adjudication",
)

# The packet materializer creates exactly these compact request fields.  A
# different key would be a protocol change and must not quietly become a
# review input just because it has a plausible request hash.
REQUEST_FIELDS = frozenset(
    {
        "review_id",
        "request_sha256",
        "review_copy",
        "comparison_bundle",
        "schema_version",
        "sample_id",
        "reviewer_slot",
        "source_id",
        "source_dataset",
        "source_revision",
        "source_route",
        "extraction_route",
        *OBSERVED_ROUTE_FIELDS,
        "sampling_stratum",
        "original_text_sha256",
        "review_copy_sha256",
        "prompt_sha256",
        "response_schema_sha256",
        "model",
        "code_commit",
        "attempt",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(label: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(label: str, value: Any, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_fraction(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite fraction")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty JSON file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object row")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: JSONL contains no rows")
    return rows


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _bound_input_path(label: str, value: Any) -> Path:
    """Resolve and byte-verify one aggregate input receipt.

    Aggregate input bindings intentionally contain no optional fields.  This
    prevents a caller from presenting a plausible hash while quietly changing
    the path or adding an alternate selection input used only downstream.
    """

    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label}: expected exact path/bytes/sha256 binding")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}: missing bound path")
    path = Path(raw_path)
    _verify_binding(label, value, path)
    return path.resolve()


def _verify_binding(label: str, expected: Any, actual_path: Path, *, rows: int | None = None) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise ValueError(f"{label}: expected receipt binding object")
    actual = file_binding(actual_path)
    if expected.get("bytes") != actual["bytes"] or expected.get("sha256") != actual["sha256"]:
        raise ValueError(f"{label}: bound file bytes/SHA-256 drift")
    if rows is not None and expected.get("rows") != rows:
        raise ValueError(f"{label}: bound row count drift")
    return actual


def _binding_hash(label: str, value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: expected binding object")
    return _require_sha256(f"{label}.sha256", value.get("sha256"))


def _assert_digest(value: Mapping[str, Any], *, field: str, label: str) -> None:
    supplied = _require_sha256(f"{label}.{field}", value.get(field))
    digest_input = dict(value)
    digest_input.pop(field, None)
    if supplied != sha256_json(digest_input):
        raise ValueError(f"{label}: {field} drift")


def write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable aggregate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        # link(2) fails if another worker won the race; os.replace would not.
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _map_unique(rows: Iterable[Mapping[str, Any]], field: str, *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        value = _require_nonempty_string(f"{label}[{index}].{field}", raw.get(field))
        if value in result:
            raise ValueError(f"{label}: duplicate {field} {value!r}")
        result[value] = dict(raw)
    return result


def _counter_by_stratum(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["sampling_stratum"]) for row in rows)
    return {stratum: int(counts.get(stratum, 0)) for stratum in review.STRATA}


def _assert_counts(label: str, actual: Mapping[str, int], expected: Any) -> None:
    if not isinstance(expected, Mapping):
        raise ValueError(f"{label}: expected stratum-count object")
    projected = {stratum: _require_int(f"{label}.{stratum}", expected.get(stratum, 0)) for stratum in review.STRATA}
    if dict(actual) != projected:
        raise ValueError(f"{label}: stratum count drift; actual={dict(actual)}, expected={projected}")


def _validate_observed_route_context(
    row: Mapping[str, Any],
    *,
    label: str,
    logical_source_route: str,
    declared_extraction_route: str,
    allowed_observed_routes: Sequence[str],
) -> dict[str, str]:
    """Validate one compact observed-route receipt against logical provenance."""

    if logical_source_route not in review.ALLOWED_ROUTES:
        raise ValueError(f"{label}: logical source route is unsupported")
    if declared_extraction_route not in review.ALLOWED_ROUTES:
        raise ValueError(f"{label}: declared extraction route is unsupported")
    allowed = list(allowed_observed_routes)
    if (
        not allowed
        or allowed != sorted(set(allowed))
        or any(route not in review.ALLOWED_ROUTES for route in allowed)
        or logical_source_route not in allowed
    ):
        raise ValueError(f"{label}: allowed observed-route provenance is invalid")
    observed = row.get("observed_extraction_route")
    if observed not in review.ALLOWED_ROUTES:
        raise ValueError(f"{label}: observed extraction route is unsupported")
    if observed not in allowed:
        raise ValueError(
            f"{label}: observed extraction route is not the logical route or a documented secondary exception"
        )
    basis = row.get("observed_extraction_route_basis")
    if basis not in review.OBSERVED_EXTRACTION_ROUTE_BASES:
        raise ValueError(f"{label}: observed extraction route basis is unsupported")
    if basis == "unavailable":
        raise ValueError(
            f"{label}: unavailable observed extraction route cannot carry a route"
        )
    if (
        basis == "declared_extraction_route_fallback"
        and observed != declared_extraction_route
    ):
        raise ValueError(
            f"{label}: declared extraction route fallback differs from frozen extraction route"
        )
    evidence = row.get("observed_extraction_route_evidence")
    if (
        not isinstance(evidence, str)
        or not evidence
        or len(evidence) > 256
        or any(character.isspace() or ord(character) < 0x20 for character in evidence)
    ):
        raise ValueError(f"{label}: observed extraction route evidence is not a bounded text-free code")
    expected_priority = (
        "logical_primary"
        if observed == logical_source_route
        else "secondary_exception_only"
    )
    if row.get("observed_extraction_route_priority") != expected_priority:
        raise ValueError(f"{label}: observed extraction route reverses logical-source priority")
    return {
        "observed_extraction_route": str(observed),
        "observed_extraction_route_basis": str(basis),
        "observed_extraction_route_evidence": evidence,
        "observed_extraction_route_priority": expected_priority,
    }


def _validate_positive_counter(
    value: Any,
    *,
    label: str,
    allowed_keys: set[str],
    expected_total: int,
) -> dict[str, int]:
    """Validate the source-wide route/basis/priority count receipts."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label}: expected a non-empty count object")
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if not isinstance(raw_key, str) or raw_key not in allowed_keys:
            raise ValueError(f"{label}: unsupported count key {raw_key!r}")
        result[raw_key] = _require_int(f"{label}.{raw_key}", raw_count, minimum=1)
    if sum(result.values()) != expected_total:
        raise ValueError(f"{label}: count denominator drift")
    return dict(sorted(result.items()))


def _validate_source_route_provenance(
    row: Mapping[str, Any],
    *,
    label: str,
    logical_source_route: str,
    review_route: str,
    extraction_route: str,
    allowed_observed_routes: Sequence[str],
    eligible_documents: int,
) -> dict[str, Any]:
    """Validate source-level logical/observed route evidence from Stage 30."""

    if row.get("source_route") != logical_source_route:
        raise ValueError(f"{label}: logical source route drift")
    if row.get("review_route") != review_route:
        raise ValueError(f"{label}: review route drift")
    if row.get("extraction_route") != extraction_route:
        raise ValueError(f"{label}: declared extraction route drift")
    allowed = list(allowed_observed_routes)
    if row.get("allowed_observed_extraction_routes") != allowed:
        raise ValueError(f"{label}: allowed observed extraction routes drift")
    route_counts = _validate_positive_counter(
        row.get("observed_extraction_route_counts"),
        label=f"{label}.observed_extraction_route_counts",
        allowed_keys=set(allowed),
        expected_total=eligible_documents,
    )
    basis_counts = _validate_positive_counter(
        row.get("observed_extraction_route_basis_counts"),
        label=f"{label}.observed_extraction_route_basis_counts",
        allowed_keys=set(review.OBSERVED_EXTRACTION_ROUTE_BASES),
        expected_total=eligible_documents,
    )
    priority_counts = _validate_positive_counter(
        row.get("observed_extraction_route_priority_counts"),
        label=f"{label}.observed_extraction_route_priority_counts",
        allowed_keys={"logical_primary", "secondary_exception_only"},
        expected_total=eligible_documents,
    )
    expected_priority_counts: dict[str, int] = {}
    primary_count = route_counts.get(logical_source_route, 0)
    secondary_count = eligible_documents - primary_count
    if primary_count:
        expected_priority_counts["logical_primary"] = primary_count
    if secondary_count:
        expected_priority_counts["secondary_exception_only"] = secondary_count
    if priority_counts != expected_priority_counts:
        raise ValueError(f"{label}: observed route priority counts reverse logical-source priority")
    return {
        "logical_source_priority": review.ROUTE_POLICY_PRIORITY,
        "logical_source_route": logical_source_route,
        "review_route": review_route,
        "declared_extraction_route": extraction_route,
        "allowed_observed_extraction_routes": allowed,
        "eligible_document_observed_extraction_route_counts": route_counts,
        "eligible_document_observed_extraction_route_basis_counts": basis_counts,
        "eligible_document_observed_extraction_route_priority_counts": priority_counts,
    }


def _validate_calibration_request_binding(
    calibration: Mapping[str, Any],
    *,
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind every receipt-safe calibration summary to its Stage-30 request.

    The calibration receipt intentionally omits review-copy text.  Its
    primary and any pre-existing secondary summaries must therefore be a
    byte-for-byte projection of Stage 30; a calibration-only secondary is
    deterministically recreated from that primary.  This makes observed
    extraction provenance auditable without widening the no-corpus-text
    boundary.
    """

    selection = calibration.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Codex calibration receipt lacks selection for request binding")
    pairs = selection.get("selected_pairs")
    if not isinstance(pairs, list):
        raise ValueError("Codex calibration receipt lacks selected pairs for request binding")
    for pair_index, pair in enumerate(pairs, 1):
        if not isinstance(pair, Mapping):
            raise ValueError(f"Codex calibration pair {pair_index} is malformed")
        primary_summary = pair.get("primary_request")
        secondary_summary = pair.get("secondary_request")
        if not isinstance(primary_summary, Mapping) or not isinstance(secondary_summary, Mapping):
            raise ValueError(f"Codex calibration pair {pair_index} is malformed")
        primary_id = primary_summary.get("review_id")
        primary = requests_by_id.get(str(primary_id)) if isinstance(primary_id, str) else None
        if primary is None or primary.get("reviewer_slot") != "primary":
            raise ValueError(f"Codex calibration pair {pair_index}.primary_request is not a Stage-30 primary request")
        expected_primary = {
            field: primary[field]
            for field in CALIBRATION_REQUEST_IDENTITY_FIELDS
        }
        if dict(primary_summary) != expected_primary:
            raise ValueError(
                f"Codex calibration pair {pair_index}.primary_request differs from its exact Stage-30 request"
            )

        secondary_id = secondary_summary.get("review_id")
        secondary = requests_by_id.get(str(secondary_id)) if isinstance(secondary_id, str) else None
        if secondary is None:
            # The runner is allowed to create a deterministic secondary only
            # for calibration when Stage 30 did not sample that document for a
            # full secondary review.  Recreate it from the exact primary
            # packet rather than treating a receipt-only summary as trusted.
            secondary = review.make_review_request(
                {
                    "source_id": primary["source_id"],
                    "source_dataset": primary["source_dataset"],
                    "source_revision": primary["source_revision"],
                    "stable_uid": primary["sample_id"],
                    "source_route": primary["source_route"],
                    "extraction_route": primary["extraction_route"],
                    **{field: primary[field] for field in OBSERVED_ROUTE_FIELDS},
                    "sampling_stratum": primary["sampling_stratum"],
                },
                reviewer_slot="secondary",
                original_text_sha256=str(primary["original_text_sha256"]),
                review_copy_sha256=str(primary["review_copy_sha256"]),
                prompt_sha256=str(primary["prompt_sha256"]),
                response_schema_sha256=str(primary["response_schema_sha256"]),
                model=str(primary["model"]),
                code_commit=str(primary["code_commit"]),
                attempt=int(primary["attempt"]),
                review_copy=str(primary["review_copy"]),
                comparison_bundle=list(primary["comparison_bundle"]),
            )
        if secondary.get("reviewer_slot") != "secondary":
            raise ValueError(f"Codex calibration pair {pair_index}.secondary_request has the wrong reviewer slot")
        expected_secondary = {
            field: secondary[field]
            for field in CALIBRATION_REQUEST_IDENTITY_FIELDS
        }
        if dict(secondary_summary) != expected_secondary:
            raise ValueError(
                f"Codex calibration pair {pair_index}.secondary_request differs from its exact request"
            )


def _packet_digest(packet: Mapping[str, Any]) -> str:
    copy = dict(packet)
    copy.pop("manifest_sha256", None)
    return sha256_json(copy)


def _validate_packet_and_requests(
    *,
    packet: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    roster: Mapping[str, Any],
    roster_path: Path,
    request_path: Path,
) -> dict[str, Any]:
    """Validate Stage 30 selection closure and return its normalized proof."""

    if packet.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("review packet has unsupported schema_version")
    if packet.get("status") != "materialized_no_model_invocation":
        raise ValueError("review packet is not the Stage 30 no-model materialization")
    if packet.get("manifest_sha256") != _packet_digest(packet):
        raise ValueError("review packet manifest hash drift")

    route_validation = review.validate_candidate_roster_routes(roster)
    candidates = list(route_validation["candidate_source_ids"])
    logical_routes = dict(route_validation["source_routes"])
    review_routes = dict(route_validation["review_routes"])
    extraction_routes = dict(route_validation["extraction_routes"])
    observed_route_allowances = dict(route_validation["allowed_observed_extraction_routes"])
    packet_inputs = packet.get("inputs")
    if not isinstance(packet_inputs, Mapping):
        raise ValueError("review packet lacks input bindings")
    _verify_binding("review packet candidate roster", packet_inputs.get("candidate_roster"), roster_path)
    # ``candidate_roster`` is a byte-level file receipt, while the route
    # report deliberately uses canonical JSON to remain stable across harmless
    # whitespace changes.  Both are independently checked below; they are not
    # expected to be equal digests.
    requests_binding = packet.get("requests")
    _verify_binding("review packet requests", requests_binding, request_path)

    execution = packet.get("review_execution")
    if not isinstance(execution, Mapping):
        raise ValueError("review packet lacks review_execution")
    if (
        execution.get("model_environment_variable") != "CODEX_REVIEW_MODEL"
        or execution.get("model") != EXPECTED_REVIEW_MODEL
        or execution.get("no_model_fallback") is not True
        or execution.get("model_invocation") != "not_run"
    ):
        raise ValueError("review packet execution model contract drift")
    for name in ("prompt_sha256", "response_schema_sha256"):
        _require_sha256(f"review packet review_execution.{name}", execution.get(name))
    if execution.get("prompt_sha256") != _binding_hash("review packet prompt", packet_inputs.get("prompt")):
        raise ValueError("review packet prompt binding drift")
    if execution.get("response_schema_sha256") != _binding_hash(
        "review packet response schema", packet_inputs.get("response_schema")
    ):
        raise ValueError("review packet response-schema binding drift")

    selection = packet.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("review packet lacks deterministic selection")
    if selection.get("schema_version") != review.SAMPLE_MANIFEST_SCHEMA:
        raise ValueError("review packet selection schema drift")
    if selection.get("candidate_roster_sha256") != route_validation["roster_sha256"]:
        raise ValueError("review packet selection roster binding drift")
    if selection.get("route_validation") != route_validation:
        raise ValueError("review packet selection route validation drift")
    if selection.get("missing_candidate_sources") != []:
        raise ValueError("review packet selected an incomplete candidate roster")
    if selection.get("manifest_sha256") != sha256_json(
        {key: value for key, value in selection.items() if key != "manifest_sha256"}
    ):
        raise ValueError("review packet selection manifest hash drift")

    source_rows = selection.get("sources")
    selected_rows = selection.get("selected_documents")
    coverage_rows = packet.get("source_review_coverage")
    if not isinstance(source_rows, list) or not isinstance(selected_rows, list) or not isinstance(coverage_rows, list):
        raise ValueError("review packet selection/coverage must be lists")
    selection_sources = _map_unique(source_rows, "source_id", label="selection.sources")
    coverage_sources = _map_unique(coverage_rows, "source_id", label="source_review_coverage")
    if set(selection_sources) != set(candidates) or set(coverage_sources) != set(candidates):
        raise ValueError("review packet source coverage does not equal candidate roster")

    selected_by_uid: dict[str, dict[str, Any]] = {}
    selected_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(selected_rows, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"selection.selected_documents[{index}] must be an object")
        row = dict(raw)
        uid = _require_sha256(f"selection.selected_documents[{index}].stable_uid", row.get("stable_uid"))
        source_id = _require_nonempty_string(f"selection.selected_documents[{index}].source_id", row.get("source_id"))
        if uid in selected_by_uid:
            raise ValueError("review packet selection repeats a stable_uid")
        source = selection_sources.get(source_id)
        if source is None:
            raise ValueError(f"{uid}: review packet selection source/route drift")
        for field in ("source_dataset", "source_revision"):
            if row.get(field) != source.get(field):
                raise ValueError(f"{uid}: review packet selection identity drift: {field}")
        if (
            row.get("source_route") != logical_routes[source_id]
            or row.get("review_route") != review_routes[source_id]
            or row.get("extraction_route") != extraction_routes[source_id]
        ):
            raise ValueError(f"{uid}: review packet selection route provenance drift")
        _validate_observed_route_context(
            row,
            label=f"selection.selected_documents[{index}]",
            logical_source_route=logical_routes[source_id],
            declared_extraction_route=extraction_routes[source_id],
            allowed_observed_routes=observed_route_allowances[source_id],
        )
        if row.get("sampling_stratum") not in review.STRATA:
            raise ValueError(f"{uid}: unsupported selection sampling stratum")
        selected_by_uid[uid] = row
        selected_by_source[source_id].append(row)
    if len(selected_by_uid) != _require_int(
        "selection.selected_document_count", selection.get("selected_document_count"), minimum=1
    ):
        raise ValueError("review packet selected-document denominator drift")

    initial_requests: list[dict[str, Any]] = []
    request_by_id: dict[str, dict[str, Any]] = {}
    request_by_sample_slot: dict[tuple[str, str], dict[str, Any]] = {}
    request_inventory: list[dict[str, str]] = []
    for index, raw in enumerate(requests, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"review request {index} must be an object")
        row = dict(raw)
        extras = sorted(set(row) - REQUEST_FIELDS)
        if extras:
            raise ValueError(f"review request {index} has unsupported fields: {extras}")
        errors = review._validate_request_binding(row)
        if errors:
            raise ValueError(f"review request {index}: {'; '.join(errors)}")
        slot = str(row["reviewer_slot"])
        if slot not in {"primary", "secondary"}:
            raise ValueError("Stage 30 request packet must contain primary/secondary slots only")
        if row.get("model") != EXPECTED_REVIEW_MODEL:
            raise ValueError("review request model drift")
        if row.get("prompt_sha256") != execution.get("prompt_sha256") or row.get("response_schema_sha256") != execution.get("response_schema_sha256"):
            raise ValueError("review request prompt/schema binding drift")
        review_id = str(row["review_id"])
        sample_id = str(row["sample_id"])
        if review_id in request_by_id or (sample_id, slot) in request_by_sample_slot:
            raise ValueError("review packet repeats review identity")
        sample = selected_by_uid.get(sample_id)
        if sample is None:
            raise ValueError("review request is not part of Stage 30 selection")
        for field in (
            "source_id",
            "source_dataset",
            "source_revision",
            "source_route",
            "extraction_route",
            *OBSERVED_ROUTE_FIELDS,
            "sampling_stratum",
        ):
            if row.get(field) != sample.get(field):
                raise ValueError(f"{sample_id}: review request selection identity drift: {field}")
        if not isinstance(row.get("review_copy"), str) or not isinstance(row.get("comparison_bundle"), list):
            raise ValueError(f"{sample_id}: request lacks compact review copy/comparison bundle")
        request_by_id[review_id] = row
        request_by_sample_slot[(sample_id, slot)] = row
        initial_requests.append(row)
        request_inventory.append({field: str(row[field]) for field in REQUEST_INVENTORY_FIELDS})
    if packet.get("request_inventory") != request_inventory:
        raise ValueError("review packet request inventory does not match exact request JSONL")
    request_counts = packet.get("request_counts")
    if not isinstance(request_counts, Mapping):
        raise ValueError("review packet lacks request_counts")
    primary = [row for row in initial_requests if row["reviewer_slot"] == "primary"]
    secondary = [row for row in initial_requests if row["reviewer_slot"] == "secondary"]
    if (
        request_counts.get("primary") != len(primary)
        or request_counts.get("secondary") != len(secondary)
        or request_counts.get("total") != len(initial_requests)
    ):
        raise ValueError("review packet request count closure drift")
    if request_counts.get("primary_by_stratum") != _counter_by_stratum(primary):
        raise ValueError("review packet primary stratum count drift")
    if request_counts.get("secondary_by_stratum") != _counter_by_stratum(secondary):
        raise ValueError("review packet secondary stratum count drift")

    primary_by_sample = {str(row["sample_id"]): row for row in primary}
    if set(primary_by_sample) != set(selected_by_uid) or len(primary_by_sample) != len(selected_by_uid):
        raise ValueError("primary review requests do not exactly cover Stage 30 selection")
    if len({str(row["sample_id"]) for row in secondary}) != len(secondary):
        raise ValueError("secondary review sample identities repeat")

    source_route_provenance: dict[str, dict[str, Any]] = {}
    for source_id in candidates:
        source = selection_sources[source_id]
        coverage = coverage_sources[source_id]
        selected = selected_by_source[source_id]
        denominator = source.get("review_denominator")
        if not isinstance(denominator, Mapping):
            raise ValueError(f"{source_id}: missing review denominator")
        coverage_denominator = coverage.get("review_denominator")
        if coverage_denominator != denominator:
            raise ValueError(f"{source_id}: coverage denominator drift")
        for field in ("source_dataset", "source_revision"):
            if source.get(field) != coverage.get(field):
                raise ValueError(f"{source_id}: coverage identity drift: {field}")
        eligible = _require_int(f"{source_id}.eligible_document_count", denominator.get("eligible_document_count"), minimum=1)
        selected_total = _require_int(f"{source_id}.selected_unique_documents", denominator.get("selected_unique_documents"), minimum=1)
        configured = _require_int(f"{source_id}.configured_review_target", denominator.get("configured_review_target"), minimum=1)
        selection_provenance = _validate_source_route_provenance(
            source,
            label=f"selection.sources[{source_id!r}]",
            logical_source_route=logical_routes[source_id],
            review_route=review_routes[source_id],
            extraction_route=extraction_routes[source_id],
            allowed_observed_routes=observed_route_allowances[source_id],
            eligible_documents=eligible,
        )
        coverage_provenance = _validate_source_route_provenance(
            coverage,
            label=f"source_review_coverage[{source_id!r}]",
            logical_source_route=logical_routes[source_id],
            review_route=review_routes[source_id],
            extraction_route=extraction_routes[source_id],
            allowed_observed_routes=observed_route_allowances[source_id],
            eligible_documents=eligible,
        )
        if coverage_provenance != selection_provenance:
            raise ValueError(f"{source_id}: coverage observed-route provenance drift")
        source_route_provenance[source_id] = selection_provenance
        if selected_total != len(selected) or selected_total != sum(_counter_by_stratum(selected).values()):
            raise ValueError(f"{source_id}: selected document denominator drift")
        if selected_total != min(eligible, configured):
            raise ValueError(f"{source_id}: selection is not exhaustive/minimum according to frozen target")
        status = denominator.get("minimum_requirement_status")
        exhaustive = denominator.get("selection_is_exhaustive")
        if eligible >= review.MINIMUM_ELIGIBLE_DOCUMENTS:
            if status != "met" or selected_total < review.MINIMUM_ELIGIBLE_DOCUMENTS or exhaustive is True:
                raise ValueError(f"{source_id}: invalid >=100 review denominator claim")
        else:
            if (
                status != "unattainable_exhaustive"
                or exhaustive is not True
                or selected_total != eligible
                or denominator.get("denominator_exception") != "eligible_inventory_below_100_all_documents_selected"
            ):
                raise ValueError(f"{source_id}: sub-100 source lacks exhaustive denominator exception")
        primary_rows = [row for row in primary if row["source_id"] == source_id]
        secondary_rows = [row for row in secondary if row["source_id"] == source_id]
        _assert_counts(f"{source_id}.selection.actual_strata", _counter_by_stratum(selected), source.get("actual_strata"))
        _assert_counts(f"{source_id}.coverage.primary", _counter_by_stratum(primary_rows), coverage.get("primary_requests_by_stratum"))
        _assert_counts(f"{source_id}.coverage.secondary", _counter_by_stratum(secondary_rows), coverage.get("secondary_requests_by_stratum"))
        _assert_counts(f"{source_id}.selection.requested", _counter_by_stratum(selected), source.get("requested_strata"))
        if coverage.get("requested_strata") != source.get("requested_strata"):
            raise ValueError(f"{source_id}: requested strata drift")
        if len(primary_rows) != selected_total:
            raise ValueError(f"{source_id}: primary request count does not close selection")

    return {
        "route_validation": route_validation,
        "packet": dict(packet),
        "selection_sources": selection_sources,
        "coverage_sources": coverage_sources,
        "source_route_provenance": source_route_provenance,
        "selected_by_uid": selected_by_uid,
        "selected_by_source": {key: list(value) for key, value in selected_by_source.items()},
        "initial_requests": initial_requests,
        "request_by_id": request_by_id,
        "primary_by_sample": primary_by_sample,
        "primary": primary,
        "secondary": secondary,
    }


def _validate_stage35(
    *,
    proof: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
    response_receipt: Mapping[str, Any],
    response_receipt_path: Path,
    adjudication_receipt: Mapping[str, Any],
    adjudication_receipt_path: Path,
    requests_path: Path,
    responses_path: Path,
) -> dict[str, Any]:
    """Validate exact Stage 35 execution and return resolved document reviews."""

    packet = proof["packet"]
    packet_inputs = packet["inputs"]
    initial_requests = proof["initial_requests"]
    if response_receipt.get("schema_version") != RESPONSE_RECEIPT_SCHEMA or response_receipt.get("status") != "complete":
        raise ValueError("review response execution receipt is incomplete/unsupported")
    _assert_digest(response_receipt, field="receipt_sha256", label="review response receipt")
    response_inputs = response_receipt.get("inputs")
    if not isinstance(response_inputs, Mapping):
        raise ValueError("review response receipt lacks input bindings")
    _verify_binding("review response receipt requests", response_inputs.get("requests"), requests_path)
    # The packet can be copied to a local Codex-review workspace.  Compare its
    # bytes/hash to the response receipt rather than requiring the same path.
    if _binding_hash("review response receipt requests", response_inputs.get("requests")) != _binding_hash(
        "review packet requests", packet.get("requests")
    ):
        raise ValueError("Stage 35 requests do not bind the exact Stage 30 request packet")
    for name, packet_name in (("policy", "review_policy"), ("prompt", "prompt"), ("response_schema", "response_schema")):
        if _binding_hash(f"review response receipt {name}", response_inputs.get(name)) != _binding_hash(
            f"review packet {packet_name}", packet_inputs.get(packet_name)
        ):
            raise ValueError(f"Stage 35 {name} does not bind Stage 30 packet input")
    model = response_receipt.get("model")
    if not isinstance(model, Mapping) or (
        model.get("environment_variable") != "CODEX_REVIEW_MODEL"
        or model.get("required_model") != EXPECTED_REVIEW_MODEL
        or model.get("accepted_model") != EXPECTED_REVIEW_MODEL
        or model.get("no_fallback") is not True
    ):
        raise ValueError("Stage 35 review model receipt drift")
    if response_receipt.get("primary_secondary_sessions_separated") is not True or response_receipt.get("adjudication_sessions_separated") is not True:
        raise ValueError("Stage 35 isolated reviewer-slot execution evidence is missing")
    response_info = response_receipt.get("responses")
    _verify_binding("review response receipt output", response_info, responses_path, rows=len(responses))
    if not isinstance(response_info, Mapping) or response_info.get("slot_counts") is None:
        raise ValueError("review response receipt lacks response slot counts")

    if adjudication_receipt.get("schema_version") != ADJUDICATION_RECEIPT_SCHEMA or adjudication_receipt.get("status") != "complete":
        raise ValueError("adjudication execution receipt is incomplete/unsupported")
    _assert_digest(adjudication_receipt, field="receipt_sha256", label="adjudication receipt")
    if adjudication_receipt.get("model") != EXPECTED_REVIEW_MODEL:
        raise ValueError("adjudication receipt model drift")
    if adjudication_receipt.get("initial_request_rows") != len(initial_requests):
        raise ValueError("adjudication receipt initial request count drift")
    if adjudication_receipt.get("response_rows") != len(responses):
        raise ValueError("adjudication receipt response row count drift")
    _verify_binding("adjudication receipt responses", adjudication_receipt.get("responses"), responses_path)
    if (
        adjudication_receipt.get("primary_secondary_sessions_separated") is not True
        or adjudication_receipt.get("adjudication_sessions_separated") is not True
    ):
        raise ValueError("adjudication receipt lacks session-isolation evidence")
    _verify_binding(
        "review response receipt adjudication receipt",
        response_receipt.get("adjudication_receipt"),
        adjudication_receipt_path,
    )

    response_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(responses, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"review response {index} must be an object")
        row = dict(raw)
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or review_id in response_by_id:
            raise ValueError(f"review response {index} repeats/omits review_id")
        response_by_id[review_id] = row
    final_manifest = review.build_adjudication_manifest(initial_requests, responses)
    review.assert_adjudication_closed(final_manifest)
    receipt_manifest = adjudication_receipt.get("final_adjudication_manifest")
    if receipt_manifest != final_manifest:
        raise ValueError("adjudication receipt final manifest differs from exact review responses")
    if response_receipt.get("responses", {}).get("slot_counts") != dict(
        sorted(Counter(str(row["reviewer_slot"]) for row in responses).items())
    ):
        raise ValueError("review response receipt slot count drift")
    if adjudication_receipt.get("response_slot_counts") != dict(
        sorted(Counter(str(row["reviewer_slot"]) for row in responses).items())
    ):
        raise ValueError("adjudication receipt slot count drift")
    if int(final_manifest["pending_count"]) != 0 or final_manifest.get("status") != "complete":
        raise ValueError("review adjudication remains pending")

    adjudicated_by_sample: dict[str, str] = {}
    for case in final_manifest["cases"]:
        if case.get("status") == "adjudicated":
            sample_id = _require_sha256("adjudicated sample_id", case.get("sample_id"))
            review_id = _require_sha256("adjudicated review id", case.get("adjudicator_review_id"))
            if sample_id in adjudicated_by_sample:
                raise ValueError("adjudication manifest repeats a sample")
            adjudicated_by_sample[sample_id] = review_id
    resolved: dict[str, dict[str, Any]] = {}
    for sample_id, primary in proof["primary_by_sample"].items():
        review_id = adjudicated_by_sample.get(sample_id, str(primary["review_id"]))
        response = response_by_id.get(review_id)
        if response is None:
            raise ValueError(f"{sample_id}: closed adjudication lacks a resolved response")
        if review_id == primary["review_id"]:
            review.assert_valid_review_response(response, primary)
        else:
            # build_adjudication_manifest already verifies the generated
            # adjudicator request against the response.  The identity is then
            # bound again through the final manifest equality above.
            if response.get("reviewer_slot") != "adjudicator":
                raise ValueError(f"{sample_id}: resolved non-primary review is not adjudicator output")
        resolved[sample_id] = response
    if set(resolved) != set(proof["selected_by_uid"]):
        raise ValueError("resolved review documents do not close Stage 30 selection")
    return {
        "responses": response_by_id,
        "resolved": resolved,
        "final_manifest": final_manifest,
        "adjudicated_by_sample": adjudicated_by_sample,
        "response_receipt": dict(response_receipt),
        "adjudication_receipt": dict(adjudication_receipt),
        "response_receipt_binding": file_binding(response_receipt_path),
        "adjudication_receipt_binding": file_binding(adjudication_receipt_path),
        "response_receipt_sha256": str(response_receipt["receipt_sha256"]),
        "adjudication_receipt_sha256": str(adjudication_receipt["receipt_sha256"]),
    }


def _validate_stage35_closure(
    *,
    closure: Mapping[str, Any],
    closure_path: Path,
    proof: Mapping[str, Any],
    stage35: Mapping[str, Any],
    requests_path: Path,
    packet_path: Path,
    responses_path: Path,
    response_receipt_path: Path,
    adjudication_receipt_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Require the receipt published by Stage 35 in addition to recomputation.

    Recomputing closure protects the aggregate from a malformed summary; the
    Stage-35 closure binds the same evidence to its CPU-stage boundary and
    proves that the exact masked review sample diagnostic was allowed to run.
    Both are required.
    """

    if closure.get("schema_version") != STAGE35_CLOSURE_SCHEMA or closure.get("status") != "passed":
        raise ValueError("Stage 35 review-evidence closure is incomplete/unsupported")
    _assert_digest(closure, field="closure_sha256", label="Stage 35 review-evidence closure")
    if closure.get("run_id") != run_id:
        raise ValueError("Stage 35 review-evidence closure run_id drift")
    inputs = closure.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Stage 35 review-evidence closure lacks input bindings")
    for label, binding_name, path in (
        ("requests", "review_requests", requests_path),
        ("packet", "review_packet_manifest", packet_path),
        ("responses", "external_responses", responses_path),
        ("response receipt", "external_response_receipt", response_receipt_path),
        ("adjudication receipt", "external_adjudication_receipt", adjudication_receipt_path),
    ):
        _verify_binding(f"Stage 35 closure {label}", inputs.get(binding_name), path)
    packet = proof["packet"]
    execution = closure.get("review_execution")
    if not isinstance(execution, Mapping) or (
        execution.get("model_environment_variable") != "CODEX_REVIEW_MODEL"
        or execution.get("required_model") != EXPECTED_REVIEW_MODEL
        or execution.get("accepted_model") != EXPECTED_REVIEW_MODEL
        or execution.get("no_model_fallback") is not True
        or execution.get("prompt_sha256") != packet["review_execution"]["prompt_sha256"]
        or execution.get("response_schema_sha256") != packet["review_execution"]["response_schema_sha256"]
    ):
        raise ValueError("Stage 35 review-evidence closure execution binding drift")
    expected_commit = str(proof["initial_requests"][0]["code_commit"])
    if closure.get("code_commit") != expected_commit or execution.get("code_commit") != expected_commit:
        raise ValueError("Stage 35 review-evidence closure code-commit drift")
    calibration_binding = inputs.get("external_calibration_receipt")
    if not isinstance(calibration_binding, Mapping):
        raise ValueError("Stage 35 review-evidence closure lacks the passed calibration receipt")
    calibration_path = Path(str(calibration_binding.get("path", "")))
    _verify_binding("Stage 35 closure calibration receipt", calibration_binding, calibration_path)
    calibration = read_object(calibration_path)
    packet_inputs = packet.get("inputs")
    if not isinstance(packet_inputs, Mapping):  # guarded at Stage 30; explicit for the cross-stage proof.
        raise ValueError("Stage 30 packet lacks inputs for calibration closure")
    # Reuse the Stage-35 validator rather than accepting a bare "passed"
    # flag: this proves the calibration receipt is tied to the same compact
    # request inventory, model/prompt/schema, code commit, and every logical
    # source route represented by the frozen primary review sample.
    review_evidence.validate_calibration_receipt(
        calibration_receipt=calibration,
        requests_path=requests_path,
        policy_binding=packet_inputs.get("review_policy"),
        prompt_binding=packet_inputs.get("prompt"),
        schema_binding=packet_inputs.get("response_schema"),
        model=EXPECTED_REVIEW_MODEL,
        code_commit=expected_commit,
        expected_routes={str(row["source_route"]) for row in proof["primary"]},
    )
    _validate_calibration_request_binding(
        calibration,
        requests_by_id=proof["request_by_id"],
    )
    calibration_receipt_sha = _require_sha256(
        "Stage 35 calibration receipt_sha256", calibration.get("receipt_sha256")
    )
    if (
        execution.get("calibration_receipt_sha256") != calibration_receipt_sha
        or execution.get("calibration_prompt_schema_frozen_for_full_review") is not True
    ):
        raise ValueError("Stage 35 review-evidence closure calibration binding drift")
    _verify_binding(
        "Stage 35 response receipt passed calibration",
        stage35["response_receipt"].get("passed_calibration_receipt"),
        calibration_path,
    )
    _verify_binding(
        "Stage 35 adjudication receipt passed calibration",
        stage35["adjudication_receipt"].get("passed_calibration_receipt"),
        calibration_path,
    )
    packet_summary = closure.get("packet")
    if not isinstance(packet_summary, Mapping) or packet_summary.get("manifest_sha256") != packet.get("manifest_sha256"):
        raise ValueError("Stage 35 review-evidence closure packet binding drift")
    response_closure = closure.get("response_closure")
    if not isinstance(response_closure, Mapping):
        raise ValueError("Stage 35 review-evidence closure lacks response closure")
    if (
        response_closure.get("pending_adjudication_count") != 0
        or response_closure.get("final_adjudication_manifest") != stage35["final_manifest"]
        or response_closure.get("response_execution_receipt_sha256")
        != stage35["response_receipt_sha256"]
        or response_closure.get("adjudication_execution_receipt_sha256")
        != stage35["adjudication_receipt_sha256"]
    ):
        raise ValueError("Stage 35 review-response/adjudication closure drift")
    privacy = closure.get("privacy")
    if not isinstance(privacy, Mapping) or privacy.get("external_bundle_contains_raw_corpus") is not False:
        raise ValueError("Stage 35 closure does not preserve no-raw-corpus evidence boundary")
    if closure.get("admission_decision") != "not_evaluated_in_stage35":
        raise ValueError("Stage 35 closure must not make an admission decision")
    return {
        "closure": file_binding(closure_path),
        "calibration_receipt": file_binding(calibration_path),
        "calibration_receipt_sha256": calibration_receipt_sha,
    }


def _review_sample_quality_evidence(
    *,
    summary: Mapping[str, Any],
    summary_path: Path,
    handoff: Mapping[str, Any],
    handoff_path: Path,
    closure_path: Path,
    source_id: str,
    source_dataset: str,
    selected_primary_documents: int,
    total_primary_documents: int,
) -> dict[str, Any]:
    """Validate the richer Stage-35 masked-sample diagnostic per source."""

    if (
        summary.get("schema_version") != REVIEW_SAMPLE_QUALITY_SUMMARY_SCHEMA
        or summary.get("status") != "passed"
        or summary.get("scan_mode") != "exact_v3_masked_review_sample"
        or summary.get("diagnostic_only") is not True
        or summary.get("admission_decision") != "not_evaluated_in_stage35"
    ):
        raise ValueError("Stage 35 masked-review-sample quality summary is incomplete/unsupported")
    _assert_digest(summary, field="summary_sha256", label="Stage 35 masked-review-sample quality summary")
    sample = summary.get("sample")
    if not isinstance(sample, Mapping) or (
        sample.get("primary_samples") != total_primary_documents
        or sample.get("raw_corpus_included") is not False
        or sample.get("text_variant") != "high_precision_identifier_masked_review_sample"
    ):
        raise ValueError("Stage 35 masked-review-sample quality sample scope drift")
    closure = read_object(closure_path)
    closure_inventory = (
        closure.get("packet", {}).get("primary_sample_inventory_sha256")
        if isinstance(closure.get("packet"), Mapping)
        else None
    )
    _require_sha256("Stage 35 closure primary_sample_inventory_sha256", closure_inventory)
    if sample.get("primary_sample_inventory_sha256") != closure_inventory:
        raise ValueError("Stage 35 masked-review-sample quality inventory differs from closure")
    source_summaries = summary.get("source_summaries")
    if not isinstance(source_summaries, list):
        raise ValueError("Stage 35 masked-review-sample quality lacks source summaries")
    matches = [dict(row) for row in source_summaries if row.get("repo_id") == source_id]
    if len(matches) != 1:
        raise ValueError(f"{source_id}: Stage 35 quality needs exactly one source aggregate")
    source = matches[0]
    if (
        source.get("documents") != selected_primary_documents
        or not isinstance(source.get("source_datasets"), list)
        or set(source["source_datasets"]) != {source_dataset}
    ):
        raise ValueError(f"{source_id}: Stage 35 quality source denominator/identity drift")
    if (
        handoff.get("schema_version") != REVIEW_SAMPLE_QUALITY_HANDOFF_SCHEMA
        or handoff.get("status") != "passed"
        or handoff.get("diagnostic_only") is not True
        or handoff.get("raw_corpus_included") is not False
        or handoff.get("admission_decision") != "not_evaluated_in_stage35"
    ):
        raise ValueError("Stage 35 masked-review-sample quality handoff is incomplete/unsupported")
    _assert_digest(handoff, field="handoff_sha256", label="Stage 35 masked-review-sample quality handoff")
    _verify_binding("Stage 35 quality handoff summary", handoff.get("summary"), summary_path)
    sample_receipt = summary.get("sample", {}).get("receipt") if isinstance(summary.get("sample"), Mapping) else None
    if not isinstance(sample_receipt, Mapping):
        raise ValueError("Stage 35 quality summary lacks masked-sample receipt binding")
    sample_receipt_path = Path(str(sample_receipt.get("path", "")))
    _verify_binding("Stage 35 quality summary masked-sample receipt", sample_receipt, sample_receipt_path)
    sample_receipt_value = read_object(sample_receipt_path)
    if (
        sample_receipt_value.get("schema_version") != "agent1_v3_masked_review_sample_receipt_v1"
        or sample_receipt_value.get("status") != "passed"
        or sample_receipt_value.get("primary_sample_count") != total_primary_documents
        or sample_receipt_value.get("primary_sample_inventory_sha256") != closure_inventory
        or sample_receipt_value.get("raw_corpus_included") is not False
        or sample_receipt_value.get("text_variant") != "high_precision_identifier_masked_review_sample"
        or sample_receipt_value.get("admission_decision") != "not_evaluated_in_stage35"
    ):
        raise ValueError("Stage 35 masked-sample receipt scope/inventory drift")
    _assert_digest(
        sample_receipt_value,
        field="receipt_sha256",
        label="Stage 35 masked-sample receipt",
    )
    receipt_inputs = sample_receipt_value.get("inputs")
    if not isinstance(receipt_inputs, Mapping):
        raise ValueError("Stage 35 masked-sample receipt lacks input bindings")
    _verify_binding(
        "Stage 35 masked-sample receipt closure",
        receipt_inputs.get("quality_review_evidence_closure"),
        closure_path,
    )
    _verify_binding(
        "Stage 35 quality handoff masked-sample receipt",
        handoff.get("sample_receipt"),
        sample_receipt_path,
    )
    return {
        "source_id": source_id,
        "source_dataset": source_dataset,
        "documents": int(source["documents"]),
        "document_rates": dict(source["document_rates"]),
        "distributions": dict(source["distributions"]),
        "template_concentration": dict(source["template_concentration"]),
        "summary_sha256": sha256_file(summary_path),
        "handoff_sha256": sha256_file(handoff_path),
        "stage35_closure_sha256": sha256_file(closure_path),
    }


def _validate_review_sample_quality_scope(
    summary: Mapping[str, Any],
    *,
    candidates: Sequence[str],
    source_datasets: Mapping[str, str],
    total_primary_documents: int,
) -> None:
    """Require one diagnostic aggregate for every and only Stage-30 source."""

    rows = summary.get("source_summaries")
    if not isinstance(rows, list):
        raise ValueError("Stage 35 masked-review-sample quality lacks source summaries")
    by_source = _map_unique(rows, "repo_id", label="Stage 35 masked-review-sample quality sources")
    if set(by_source) != set(candidates) or list(by_source) != sorted(candidates):
        raise ValueError("Stage 35 masked-review-sample quality source coverage differs from Stage 30")
    document_total = 0
    for source_id in candidates:
        row = by_source[source_id]
        datasets = row.get("source_datasets")
        if not isinstance(datasets, list) or set(datasets) != {source_datasets[source_id]}:
            raise ValueError(f"{source_id}: Stage 35 masked quality source-dataset scope drift")
        document_total += _require_int(
            f"{source_id}.Stage 35 masked quality documents", row.get("documents"), minimum=1
        )
    if document_total != total_primary_documents:
        raise ValueError("Stage 35 masked-review-sample quality document denominator differs from Stage 30")


def _quality_evidence(
    quality_summary: Mapping[str, Any], *, source_dataset: str, eligible_documents: int
) -> dict[str, Any]:
    projection = quality_runtime.validate_and_project_quality_summary(quality_summary)
    if projection["scan_mode"] != "full_scan":
        raise ValueError("quality summary must be the mandatory full scan")
    repositories = quality_summary.get("repositories")
    if not isinstance(repositories, list):  # guarded by runtime validator; explicit for typing.
        raise ValueError("quality summary repositories missing")
    matches = [dict(row) for row in repositories if source_dataset in row.get("source_datasets", [])]
    if len(matches) != 1:
        raise ValueError(
            f"{source_dataset}: full quality scan must bind exactly one repository aggregate; found {len(matches)}"
        )
    source = matches[0]
    if source.get("documents") != eligible_documents:
        raise ValueError(
            f"{source_dataset}: quality full-scan document denominator does not equal deterministic review eligibility"
        )
    return {
        "repo_id": str(source["repo_id"]),
        "documents": int(source["documents"]),
        "document_rates": dict(source["document_rates"]),
        "distributions": dict(source["distributions"]),
        "template_concentration": dict(source["template_concentration"]),
    }


def _validate_quality_scope(
    quality_summary: Mapping[str, Any], *, candidates: Sequence[str], source_datasets: Sequence[str]
) -> None:
    # The imported validator checks exact profiler schema, full checkpoint
    # coverage, summaries, and Rust build/normalization receipt closure.
    projection = quality_runtime.validate_and_project_quality_summary(quality_summary)
    if projection["scan_mode"] != "full_scan":
        raise ValueError("quality summary is not a full scan")
    if list(quality_summary.get("selected_source_ids", [])) != sorted(candidates):
        raise ValueError("quality full scan selected sources differ from v3 candidate roster")
    if set(quality_summary.get("excluded_source_ids", [])) & set(candidates):
        raise ValueError("quality full scan excluded a review candidate")
    global_datasets = quality_summary.get("global", {}).get("source_datasets")
    if not isinstance(global_datasets, list) or set(global_datasets) != set(source_datasets):
        raise ValueError("quality full scan source-dataset scope differs from review packet")


def _lineage_and_novelty_evidence(
    *,
    lineage_summary: Mapping[str, Any],
    novelty_summary: Mapping[str, Any],
    lineage_path: Path,
    novelty_path: Path,
    source_id: str,
    source_dataset: str,
    eligible_documents: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if lineage_summary.get("schema_version") != LINEAGE_SCHEMA:
        raise ValueError("lineage summary schema drift")
    if lineage_summary.get("blind_append_allowed") is not False:
        raise ValueError("lineage summary must retain no-blind-append gate")
    bound_novelty = lineage_summary.get("source_novelty")
    if not isinstance(bound_novelty, Mapping) or bound_novelty.get("sha256") != sha256_file(novelty_path):
        raise ValueError("lineage summary source-novelty binding drift")
    if novelty_summary.get("schema_version") != NOVELTY_SCHEMA or novelty_summary.get("near_duplicate_novelty_deferred_to_global_dedup") is not True:
        raise ValueError("lineage novelty summary has unsupported dedup boundary")
    lineage_sources = _map_unique(lineage_summary.get("sources", []), "source_id", label="lineage.sources")
    source = lineage_sources.get(source_id)
    if source is None or source.get("origin") != "candidate":
        raise ValueError(f"{source_id}: lineage has no candidate source record")
    if _require_int(f"{source_id}.lineage.rows", source.get("rows"), minimum=1) != eligible_documents:
        raise ValueError(f"{source_id}: lineage row denominator differs from review eligibility")
    novelty_sources = _map_unique(novelty_summary.get("sources", []), "source_dataset", label="source_novelty.sources")
    novelty = novelty_sources.get(source_dataset)
    if novelty is None:
        raise ValueError(f"{source_id}: novelty summary lacks source_dataset {source_dataset!r}")
    novelty_source_ids = novelty.get("source_ids")
    if not isinstance(novelty_source_ids, list) or source_id not in novelty_source_ids:
        raise ValueError(f"{source_id}: novelty source-id binding drift")
    if _require_int(f"{source_id}.novelty.rows", novelty.get("rows"), minimum=1) != eligible_documents:
        raise ValueError(f"{source_id}: novelty row denominator differs from review eligibility")
    _require_fraction(f"{source_id}.novel_token_fraction", novelty.get("novel_token_fraction"))
    return (
        {
            "rows": int(source["rows"]),
            "distinct_source_dataset_names": int(source["distinct_source_dataset_names"]),
            "distinct_source_families": int(source["distinct_source_families"]),
            "base_candidate_exact_clusters": int(source["base_candidate_exact_clusters"]),
            "base_candidate_work_clusters": int(source["base_candidate_work_clusters"]),
            "blind_append_allowed": False,
            "double_add_hazard_reasons": list(source["double_add_hazard_reasons"]),
        },
        {
            "rows": int(novelty["rows"]),
            "identity_word_tokens": int(novelty["identity_word_tokens"]),
            "exact_unique_rows": int(novelty["exact_unique_rows"]),
            "exact_unique_word_tokens": int(novelty["exact_unique_word_tokens"]),
            "novel_rows_after_lineage_resolution": int(novelty["novel_rows_after_lineage_resolution"]),
            "novel_word_tokens_after_lineage_resolution": int(novelty["novel_word_tokens_after_lineage_resolution"]),
            "novel_token_fraction": float(novelty["novel_token_fraction"]),
            "document_action_counts": dict(novelty.get("document_action_counts", {})),
        },
    )


def _license_evidence(
    license_adjudication: Mapping[str, Any], *, source_id: str, source_revision: str
) -> dict[str, Any]:
    if (
        license_adjudication.get("schema_version") != LICENSE_SCHEMA
        or license_adjudication.get("status") != "technical_audit_complete"
    ):
        raise ValueError("license adjudication schema/status drift")
    rows = _map_unique(license_adjudication.get("sources", []), "source_id", label="license.sources")
    source = rows.get(source_id)
    if source is None:
        raise ValueError(f"{source_id}: license adjudication lacks a source record")
    if source.get("revision") != source_revision:
        raise ValueError(f"{source_id}: review packet revision differs from licence adjudication")
    local = source.get("local_training")
    redistribution = source.get("redistribution")
    if not isinstance(local, Mapping) or not isinstance(redistribution, Mapping):
        raise ValueError(f"{source_id}: license eligibility records are missing")
    if not isinstance(local.get("eligible"), bool) or not isinstance(redistribution.get("eligible"), bool):
        raise ValueError(f"{source_id}: license eligibility booleans are invalid")
    for label, value in (("local_training", local), ("redistribution", redistribution)):
        _require_nonempty_string(f"{source_id}.{label}.status", value.get("status"))
        conditions = value.get("conditions")
        if not isinstance(conditions, list) or any(not isinstance(item, str) or not item for item in conditions):
            raise ValueError(f"{source_id}.{label}.conditions are invalid")
    return {
        "repo_id": _require_nonempty_string(f"{source_id}.repo_id", source.get("repo_id")),
        "revision": source_revision,
        "declared_license": source.get("declared_license"),
        "registry_training_eligibility": source.get("registry_training_eligibility"),
        "local_training": {
            "eligible": bool(local["eligible"]),
            "status": str(local["status"]),
            "conditions": list(local["conditions"]),
        },
        "redistribution": {
            "eligible": bool(redistribution["eligible"]),
            "status": str(redistribution["status"]),
            "conditions": list(redistribution["conditions"]),
        },
    }


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(int(row[field])) for row in rows)
    return {str(score): int(counter.get(str(score), 0)) for score in range(1, 6)}


def _review_evidence(
    *,
    source_id: str,
    selected: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Mapping[str, Any]],
    adjudicated_by_sample: Mapping[str, str],
) -> dict[str, Any]:
    source_primary = [row for row in primary if row["source_id"] == source_id]
    source_secondary = [row for row in secondary if row["source_id"] == source_id]
    source_resolved = [resolved[str(row["stable_uid"])] for row in selected]
    if len(source_primary) != len(selected) or len(source_resolved) != len(selected):
        raise ValueError(f"{source_id}: resolved reviews do not close selected documents")
    recommendation_counts = Counter(str(row["recommendation"]) for row in source_resolved)
    issue_document_counts: Counter[str] = Counter()
    issue_severity_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for response in source_resolved:
        for issue in response["issues"]:
            code = str(issue["code"])
            issue_document_counts[code] += 1
            issue_severity_counts[code][str(int(issue["severity_score"]))] += 1
    resolved_slots = Counter(str(row["reviewer_slot"]) for row in source_resolved)
    return {
        "selected_primary_documents": len(selected),
        "secondary_review_documents": len(source_secondary),
        "resolved_documents": len(source_resolved),
        "adjudicated_documents": sum(str(row["stable_uid"]) in adjudicated_by_sample for row in selected),
        "resolved_reviewer_slot_counts": dict(sorted(resolved_slots.items())),
        "score_distributions": {
            "cleanliness_score": _distribution(source_resolved, "cleanliness_score"),
            "quality_score": _distribution(source_resolved, "quality_score"),
            "diversity_contribution_score": _distribution(source_resolved, "diversity_contribution_score"),
            "confidence_score": _distribution(source_resolved, "confidence_score"),
        },
        "recommendation_counts": {
            name: int(recommendation_counts.get(name, 0)) for name in sorted(review.RECOMMENDATIONS)
        },
        "recommendation_rates": {
            name: round(int(recommendation_counts.get(name, 0)) / len(source_resolved), 12)
            for name in sorted(review.RECOMMENDATIONS)
        },
        "issue_document_counts": dict(sorted(issue_document_counts.items())),
        "issue_severity_distributions": {
            code: {str(score): int(counts.get(str(score), 0)) for score in range(1, 6)}
            for code, counts in sorted(issue_severity_counts.items())
        },
    }


def _cluster_template_evidence(
    selected: Sequence[Mapping[str, Any]], quality: Mapping[str, Any]
) -> dict[str, Any]:
    by_cluster = Counter(str(row["review_cluster_id"]) for row in selected)
    populations = Counter(int(row["review_cluster_size"]) for row in selected)
    cluster_strata = Counter(
        str(row["sampling_stratum"]) for row in selected if row.get("sampling_stratum") == "cluster"
    )
    return {
        "selected_cluster_count": len(by_cluster),
        "selected_cluster_representative_count": int(cluster_strata.get("cluster", 0)),
        "largest_selected_cluster_sample_count": max(by_cluster.values(), default=0),
        "largest_selected_cluster_population": max(populations, default=0),
        "selected_cluster_population_distribution": {
            str(population): int(count) for population, count in sorted(populations.items())
        },
        "quality_template_concentration": dict(quality["template_concentration"]),
    }


def build_aggregate(
    *,
    run_id: str,
    roster_path: Path,
    packet_path: Path,
    requests_path: Path,
    responses_path: Path,
    response_receipt_path: Path,
    adjudication_receipt_path: Path,
    stage35_closure_path: Path,
    review_sample_quality_summary_path: Path,
    review_sample_quality_handoff_path: Path,
    quality_summary_path: Path,
    lineage_summary_path: Path,
    novelty_summary_path: Path,
    license_adjudication_path: Path,
) -> dict[str, Any]:
    """Validate all source-review evidence and return a compact aggregate."""

    _require_nonempty_string("run_id", run_id)
    roster = read_object(roster_path)
    packet = read_object(packet_path)
    requests = read_jsonl(requests_path)
    responses = read_jsonl(responses_path)
    response_receipt = read_object(response_receipt_path)
    adjudication_receipt = read_object(adjudication_receipt_path)
    stage35_closure = read_object(stage35_closure_path)
    review_sample_quality_summary = read_object(review_sample_quality_summary_path)
    review_sample_quality_handoff = read_object(review_sample_quality_handoff_path)
    quality_summary = read_object(quality_summary_path)
    lineage_summary = read_object(lineage_summary_path)
    novelty_summary = read_object(novelty_summary_path)
    license_adjudication = read_object(license_adjudication_path)

    proof = _validate_packet_and_requests(
        packet=packet,
        requests=requests,
        roster=roster,
        roster_path=roster_path,
        request_path=requests_path,
    )
    stage35 = _validate_stage35(
        proof=proof,
        responses=responses,
        response_receipt=response_receipt,
        response_receipt_path=response_receipt_path,
        adjudication_receipt=adjudication_receipt,
        adjudication_receipt_path=adjudication_receipt_path,
        requests_path=requests_path,
        responses_path=responses_path,
    )
    stage35_closure_evidence = _validate_stage35_closure(
        closure=stage35_closure,
        closure_path=stage35_closure_path,
        proof=proof,
        stage35=stage35,
        requests_path=requests_path,
        packet_path=packet_path,
        responses_path=responses_path,
        response_receipt_path=response_receipt_path,
        adjudication_receipt_path=adjudication_receipt_path,
        run_id=run_id,
    )
    candidates = list(proof["route_validation"]["candidate_source_ids"])
    coverage_sources = proof["coverage_sources"]
    source_datasets = [str(coverage_sources[source]["source_dataset"]) for source in candidates]
    if len(source_datasets) != len(set(source_datasets)):
        raise ValueError("review packet maps multiple candidate sources to one source_dataset; admission would be ambiguous")
    _validate_quality_scope(
        quality_summary,
        candidates=candidates,
        source_datasets=source_datasets,
    )
    license_sources = _map_unique(license_adjudication.get("sources", []), "source_id", label="license.sources")
    if set(license_sources) != set(candidates):
        raise ValueError("license adjudication source coverage differs from candidate roster")
    lineage_sources = _map_unique(lineage_summary.get("sources", []), "source_id", label="lineage.sources")
    candidate_lineage = {source_id for source_id, row in lineage_sources.items() if row.get("origin") == "candidate"}
    if candidate_lineage != set(candidates):
        raise ValueError("lineage candidate-source coverage differs from candidate roster")
    _validate_review_sample_quality_scope(
        review_sample_quality_summary,
        candidates=candidates,
        source_datasets={
            source_id: str(coverage_sources[source_id]["source_dataset"])
            for source_id in candidates
        },
        total_primary_documents=len(proof["primary"]),
    )

    source_results: list[dict[str, Any]] = []
    for source_id in candidates:
        coverage = coverage_sources[source_id]
        route_provenance = proof["source_route_provenance"][source_id]
        denominator = dict(coverage["review_denominator"])
        eligible_documents = _require_int(
            f"{source_id}.review_denominator.eligible_document_count",
            denominator.get("eligible_document_count"),
            minimum=1,
        )
        quality = _quality_evidence(
            quality_summary,
            source_dataset=str(coverage["source_dataset"]),
            eligible_documents=eligible_documents,
        )
        lineage, novelty = _lineage_and_novelty_evidence(
            lineage_summary=lineage_summary,
            novelty_summary=novelty_summary,
            lineage_path=lineage_summary_path,
            novelty_path=novelty_summary_path,
            source_id=source_id,
            source_dataset=str(coverage["source_dataset"]),
            eligible_documents=eligible_documents,
        )
        license_evidence = _license_evidence(
            license_adjudication,
            source_id=source_id,
            source_revision=str(coverage["source_revision"]),
        )
        selected = proof["selected_by_source"].get(source_id, [])
        review_evidence = _review_evidence(
            source_id=source_id,
            selected=selected,
            primary=proof["primary"],
            secondary=proof["secondary"],
            resolved=stage35["resolved"],
            adjudicated_by_sample=stage35["adjudicated_by_sample"],
        )
        if review_evidence["selected_primary_documents"] != denominator["selected_unique_documents"]:
            raise ValueError(f"{source_id}: aggregate review count differs from denominator")
        review_sample_quality = _review_sample_quality_evidence(
            summary=review_sample_quality_summary,
            summary_path=review_sample_quality_summary_path,
            handoff=review_sample_quality_handoff,
            handoff_path=review_sample_quality_handoff_path,
            closure_path=stage35_closure_path,
            source_id=source_id,
            source_dataset=str(coverage["source_dataset"]),
            selected_primary_documents=review_evidence["selected_primary_documents"],
            total_primary_documents=len(proof["primary"]),
        )
        row: dict[str, Any] = {
            "schema_version": SOURCE_EVIDENCE_SCHEMA,
            "source_id": source_id,
            "source_dataset": str(coverage["source_dataset"]),
            "source_revision": str(coverage["source_revision"]),
            "source_route": str(coverage["source_route"]),
            # The logical source route remains the primary error model.
            # Source-wide observed-route counts are retained as secondary,
            # receipt-bound representation evidence for later human review.
            "route_provenance": route_provenance,
            "review_denominator": denominator,
            "review_strata": {
                "requested": dict(coverage["requested_strata"]),
                "primary": dict(coverage["primary_requests_by_stratum"]),
                "secondary": dict(coverage["secondary_requests_by_stratum"]),
            },
            "review": review_evidence,
            "cluster_and_template": _cluster_template_evidence(selected, quality),
            "quality": quality,
            "review_sample_quality": review_sample_quality,
            "lineage": lineage,
            "novelty": novelty,
            "license": license_evidence,
        }
        row["source_evidence_sha256"] = sha256_json(row)
        source_results.append(row)

    final_manifest = stage35["final_manifest"]
    payload: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "passed_review_evidence_no_admission_decision",
        "run_id": run_id,
        "inputs": {
            "candidate_roster": file_binding(roster_path),
            "review_packet": file_binding(packet_path),
            "review_requests": file_binding(requests_path),
            "review_responses": file_binding(responses_path),
            "response_execution_receipt": file_binding(response_receipt_path),
            "adjudication_execution_receipt": file_binding(adjudication_receipt_path),
            "stage35_review_closure": file_binding(stage35_closure_path),
            "review_sample_quality_summary": file_binding(review_sample_quality_summary_path),
            "review_sample_quality_handoff": file_binding(review_sample_quality_handoff_path),
            "quality_summary": file_binding(quality_summary_path),
            "lineage_summary": file_binding(lineage_summary_path),
            "source_novelty": file_binding(novelty_summary_path),
            "license_adjudication": file_binding(license_adjudication_path),
        },
        "route_validation": proof["route_validation"],
        "review_closure": {
            "status": "complete",
            "initial_request_count": len(proof["initial_requests"]),
            "response_count": len(responses),
            "response_slot_counts": dict(sorted(Counter(str(row["reviewer_slot"]) for row in responses).items())),
            "adjudication_case_count": int(final_manifest["case_count"]),
            "adjudicated_case_count": sum(case.get("status") == "adjudicated" for case in final_manifest["cases"]),
            "pending_count": int(final_manifest["pending_count"]),
            "adjudication_manifest_sha256": str(final_manifest["manifest_sha256"]),
            "model": EXPECTED_REVIEW_MODEL,
            "no_model_fallback": True,
            "calibration_receipt_sha256": stage35_closure_evidence["calibration_receipt_sha256"],
            "calibration_prompt_schema_frozen_for_full_review": True,
        },
        "source_count": len(source_results),
        "source_ids": candidates,
        "sources": source_results,
    }
    payload["aggregate_sha256"] = sha256_json(payload)
    # Avoid recursive input reopening while constructing the aggregate.  Every
    # public validation path below defaults to a full deterministic rebuild.
    validate_aggregate(payload, roster=roster, verify_evidence=False)
    return payload


def _validate_aggregate_structure(value: Mapping[str, Any], *, roster: Mapping[str, Any] | None = None) -> None:
    """Validate self-hashes and source-complete aggregate shape only."""

    if value.get("schema_version") != AGGREGATE_SCHEMA or value.get("status") != "passed_review_evidence_no_admission_decision":
        raise ValueError("unsupported/incomplete source review aggregate")
    _assert_digest(value, field="aggregate_sha256", label="source review aggregate")
    _require_nonempty_string("aggregate.run_id", value.get("run_id"))
    closure = value.get("review_closure")
    if not isinstance(closure, Mapping) or closure.get("status") != "complete" or closure.get("pending_count") != 0:
        raise ValueError("aggregate review closure remains pending")
    if closure.get("model") != EXPECTED_REVIEW_MODEL or closure.get("no_model_fallback") is not True:
        raise ValueError("aggregate model closure drift")
    _require_sha256("aggregate calibration receipt", closure.get("calibration_receipt_sha256"))
    if closure.get("calibration_prompt_schema_frozen_for_full_review") is not True:
        raise ValueError("aggregate calibration closure drift")
    route_validation = value.get("route_validation")
    if not isinstance(route_validation, Mapping):
        raise ValueError("aggregate lacks route validation")
    candidates = route_validation.get("candidate_source_ids")
    if not isinstance(candidates, list) or not candidates or candidates != sorted(set(candidates)):
        raise ValueError("aggregate candidate route closure is invalid")
    if roster is not None:
        expected = review.validate_candidate_roster_routes(roster)
        if route_validation != expected:
            raise ValueError("aggregate route validation differs from supplied candidate roster")
    if value.get("source_ids") != candidates or value.get("source_count") != len(candidates):
        raise ValueError("aggregate source list/count drift")
    logical_routes = route_validation.get("source_routes")
    review_routes = route_validation.get("review_routes")
    extraction_routes = route_validation.get("extraction_routes")
    observed_route_allowances = route_validation.get("allowed_observed_extraction_routes")
    if not all(
        isinstance(route_map, Mapping)
        and set(route_map) == set(candidates)
        for route_map in (
            logical_routes,
            review_routes,
            extraction_routes,
            observed_route_allowances,
        )
    ):
        raise ValueError("aggregate route provenance maps are incomplete")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise ValueError("aggregate sources must be a list")
    sources = _map_unique(rows, "source_id", label="aggregate.sources")
    if list(sources) != candidates:
        raise ValueError("aggregate sources are not in exact roster order/coverage")
    for source_id in candidates:
        row = sources[source_id]
        if row.get("schema_version") != SOURCE_EVIDENCE_SCHEMA:
            raise ValueError(f"{source_id}: source evidence schema drift")
        source_digest = row.get("source_evidence_sha256")
        copy = dict(row)
        copy.pop("source_evidence_sha256", None)
        if source_digest != sha256_json(copy):
            raise ValueError(f"{source_id}: source evidence hash drift")
        denominator = row.get("review_denominator")
        review_evidence = row.get("review")
        license_evidence = row.get("license")
        if not isinstance(denominator, Mapping) or not isinstance(review_evidence, Mapping) or not isinstance(license_evidence, Mapping):
            raise ValueError(f"{source_id}: aggregate source evidence is incomplete")
        eligible_documents = _require_int(
            f"{source_id}.review_denominator.eligible_document_count",
            denominator.get("eligible_document_count"),
            minimum=1,
        )
        logical_source_route = logical_routes[source_id]
        if row.get("source_route") != logical_source_route:
            raise ValueError(f"{source_id}: aggregate source route is not logical-primary")
        route_provenance = row.get("route_provenance")
        expected_route_provenance_keys = {
            "logical_source_priority",
            "logical_source_route",
            "review_route",
            "declared_extraction_route",
            "allowed_observed_extraction_routes",
            "eligible_document_observed_extraction_route_counts",
            "eligible_document_observed_extraction_route_basis_counts",
            "eligible_document_observed_extraction_route_priority_counts",
        }
        if not isinstance(route_provenance, Mapping) or set(route_provenance) != expected_route_provenance_keys:
            raise ValueError(f"{source_id}: aggregate route provenance is incomplete")
        allowed_observed_routes = observed_route_allowances[source_id]
        if (
            route_provenance.get("logical_source_priority") != review.ROUTE_POLICY_PRIORITY
            or route_provenance.get("logical_source_route") != logical_source_route
            or route_provenance.get("review_route") != review_routes[source_id]
            or route_provenance.get("declared_extraction_route") != extraction_routes[source_id]
            or route_provenance.get("allowed_observed_extraction_routes") != allowed_observed_routes
        ):
            raise ValueError(f"{source_id}: aggregate logical/observed route provenance drift")
        route_counts = _validate_positive_counter(
            route_provenance.get("eligible_document_observed_extraction_route_counts"),
            label=f"{source_id}.aggregate.observed_extraction_route_counts",
            allowed_keys=set(allowed_observed_routes),
            expected_total=eligible_documents,
        )
        _validate_positive_counter(
            route_provenance.get("eligible_document_observed_extraction_route_basis_counts"),
            label=f"{source_id}.aggregate.observed_extraction_route_basis_counts",
            allowed_keys=set(review.OBSERVED_EXTRACTION_ROUTE_BASES),
            expected_total=eligible_documents,
        )
        priority_counts = _validate_positive_counter(
            route_provenance.get("eligible_document_observed_extraction_route_priority_counts"),
            label=f"{source_id}.aggregate.observed_extraction_route_priority_counts",
            allowed_keys={"logical_primary", "secondary_exception_only"},
            expected_total=eligible_documents,
        )
        expected_priority_counts: dict[str, int] = {}
        logical_count = route_counts.get(str(logical_source_route), 0)
        if logical_count:
            expected_priority_counts["logical_primary"] = logical_count
        if eligible_documents - logical_count:
            expected_priority_counts["secondary_exception_only"] = eligible_documents - logical_count
        if priority_counts != expected_priority_counts:
            raise ValueError(f"{source_id}: aggregate observed-route priority counts drift")
        if review_evidence.get("resolved_documents") != denominator.get("selected_unique_documents"):
            raise ValueError(f"{source_id}: aggregate response denominator drift")
        if not isinstance(license_evidence.get("local_training", {}).get("eligible"), bool):
            raise ValueError(f"{source_id}: aggregate lacks local-training license state")


def validate_aggregate(
    value: Mapping[str, Any],
    *,
    roster: Mapping[str, Any] | None = None,
    verify_evidence: bool = True,
) -> None:
    """Fail closed unless the compact aggregate reproduces from its inputs.

    The aggregate's own hashes detect accidental mutation, but an operator who
    can edit JSON could recompute those hashes.  Admission therefore reopens
    every immutable input binding and derives a fresh aggregate.  Structural
    validation is retained only for ``build_aggregate`` itself to avoid
    recursion while it is creating the payload.
    """

    _validate_aggregate_structure(value, roster=roster)
    if not verify_evidence:
        return
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(AGGREGATE_INPUT_NAMES):
        raise ValueError("aggregate input binding set drift")
    paths = {
        name: _bound_input_path(f"aggregate input {name}", inputs.get(name))
        for name in AGGREGATE_INPUT_NAMES
    }
    rebuilt = build_aggregate(
        run_id=str(value["run_id"]),
        roster_path=paths["candidate_roster"],
        packet_path=paths["review_packet"],
        requests_path=paths["review_requests"],
        responses_path=paths["review_responses"],
        response_receipt_path=paths["response_execution_receipt"],
        adjudication_receipt_path=paths["adjudication_execution_receipt"],
        stage35_closure_path=paths["stage35_review_closure"],
        review_sample_quality_summary_path=paths["review_sample_quality_summary"],
        review_sample_quality_handoff_path=paths["review_sample_quality_handoff"],
        quality_summary_path=paths["quality_summary"],
        lineage_summary_path=paths["lineage_summary"],
        novelty_summary_path=paths["source_novelty"],
        license_adjudication_path=paths["license_adjudication"],
    )
    if rebuilt != dict(value):
        raise ValueError("aggregate does not equal deterministic recomputation from receipt-bound evidence")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="validate Stage 30/35 evidence and emit a source-complete aggregate")
    build.add_argument("--run-id", required=True)
    build.add_argument("--roster", type=Path, required=True)
    build.add_argument("--review-packet", type=Path, required=True)
    build.add_argument("--review-requests", type=Path, required=True)
    build.add_argument("--review-responses", type=Path, required=True)
    build.add_argument("--response-execution-receipt", type=Path, required=True)
    build.add_argument("--adjudication-execution-receipt", type=Path, required=True)
    build.add_argument("--stage35-review-closure", type=Path, required=True)
    build.add_argument("--review-sample-quality-summary", type=Path, required=True)
    build.add_argument("--review-sample-quality-handoff", type=Path, required=True)
    build.add_argument("--quality-summary", type=Path, required=True)
    build.add_argument("--lineage-summary", type=Path, required=True)
    build.add_argument("--source-novelty", type=Path, required=True)
    build.add_argument("--license-adjudication", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate a compact aggregate's immutable closure")
    validate.add_argument("--aggregate", type=Path, required=True)
    validate.add_argument("--roster", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite aggregate: {args.output}")
        payload = build_aggregate(
            run_id=args.run_id,
            roster_path=args.roster,
            packet_path=args.review_packet,
            requests_path=args.review_requests,
            responses_path=args.review_responses,
            response_receipt_path=args.response_execution_receipt,
            adjudication_receipt_path=args.adjudication_execution_receipt,
            stage35_closure_path=args.stage35_review_closure,
            review_sample_quality_summary_path=args.review_sample_quality_summary,
            review_sample_quality_handoff_path=args.review_sample_quality_handoff,
            quality_summary_path=args.quality_summary,
            lineage_summary_path=args.lineage_summary,
            novelty_summary_path=args.source_novelty,
            license_adjudication_path=args.license_adjudication,
        )
        write_json_no_replace(args.output, payload)
        print(json.dumps({"ok": True, "aggregate": str(args.output.resolve()), "sources": payload["source_count"]}, sort_keys=True))
        return 0
    aggregate = read_object(args.aggregate)
    roster = read_object(args.roster) if args.roster is not None else None
    validate_aggregate(aggregate, roster=roster)
    print(json.dumps({"ok": True, "aggregate": str(args.aggregate.resolve()), "sources": aggregate["source_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
