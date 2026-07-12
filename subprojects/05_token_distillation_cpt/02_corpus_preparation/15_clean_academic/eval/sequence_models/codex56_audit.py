#!/usr/bin/env python3
"""Build and validate a blinded Codex-5.6 structural-label audit.

The builder uses labels and model predictions only to choose difficult cases.
Those fields are written to a separate key file and never included in the
review requests.  This module invokes no model and has no corpus mutation API.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import GoldDocument, canonical_json_sha256, read_gold, sha256_file
from .deterministic_structure import analyze_bib_line, analyze_toc_line
from .evaluate import read_predictions

REQUEST_SCHEMA = "academic-structure-codex56-audit-request-v1"
KEY_SCHEMA = "academic-structure-codex56-audit-key-v1"
MANIFEST_SCHEMA = "academic-structure-codex56-audit-manifest-v1"
RESPONSE_SCHEMA = "academic-structure-codex56-audit-response-v1"
RECEIPT_SCHEMA = "academic-structure-codex56-audit-receipt-v1"
STRATA = ("toc_high_risk", "bib_high_risk", "model_disagreement", "hard_negative")
LABELS = {"BIB", "TOC", "OTHER", "UNKNOWN"}
SEALED_SPLITS = {
    "test",
    "historical_test",
    "historical-test",
    "sealed_test",
    "sealed-test",
}


@dataclass(frozen=True)
class Candidate:
    stratum: str
    document_id: str
    line_position: int
    risk: tuple[int, int, int, str]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _split_key(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _validate_audit_inputs(
    documents: Sequence[GoldDocument],
    baseline: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
    *,
    allowed_split: str,
) -> None:
    split = _split_key(allowed_split)
    if not split or split in SEALED_SPLITS:
        raise ValueError(f"forbidden audit split {allowed_split!r}")
    if not documents:
        raise ValueError("audit document set is empty")
    identities: set[str] = set()
    for document in documents:
        actual = _split_key(document.split)
        if actual in SEALED_SPLITS:
            raise ValueError(
                f"document {document.document_id!r}: sealed split is forbidden"
            )
        if actual != split:
            raise ValueError(
                f"document {document.document_id!r}: split {document.split!r} "
                f"does not match allowed split {allowed_split!r}"
            )
        if document.document_id in identities:
            raise ValueError(f"duplicate document_id {document.document_id!r}")
        identities.add(document.document_id)
        for name, predictions in (("baseline", baseline), ("candidate", candidate)):
            labels = predictions.get(document.document_id)
            if labels is None or len(labels) != len(document.lines):
                raise ValueError(
                    f"{name} predictions do not exactly cover "
                    f"document {document.document_id!r}"
                )
            if any(label not in {"O", "BIB", "TOC"} for label in labels):
                raise ValueError(
                    f"{name} predictions contain an invalid label for "
                    f"document {document.document_id!r}"
                )
    if set(baseline) != identities or set(candidate) != identities:
        raise ValueError("prediction inventories do not match audit documents")


def _write_new(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row))


def _write_json_new(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(value))


def _candidates(
    documents: Sequence[GoldDocument],
    baseline: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {stratum: [] for stratum in STRATA}
    for document in documents:
        base = baseline[document.document_id]
        current = candidate[document.document_id]
        for position, (line, base_label, current_label) in enumerate(
            zip(document.lines, base, current)
        ):
            toc = analyze_toc_line(line.text, line.abs_idx)
            bib = analyze_bib_line(line.text, line.abs_idx)
            hard = toc.hard_negative or bib.hard_negative
            prose = bool(line.is_running_prose)
            false_against_silver = line.label == "O"
            stable = hashlib.sha256(
                f"{document.document_id}\0{line.line_id}".encode()
            ).hexdigest()
            risk = (
                int(prose),
                int(hard),
                int(false_against_silver) * line.token_count,
                stable,
            )
            if current_label == "TOC":
                pools["toc_high_risk"].append(
                    Candidate("toc_high_risk", document.document_id, position, risk)
                )
            if current_label == "BIB":
                pools["bib_high_risk"].append(
                    Candidate("bib_high_risk", document.document_id, position, risk)
                )
            if current_label != base_label:
                pools["model_disagreement"].append(
                    Candidate(
                        "model_disagreement", document.document_id, position, risk
                    )
                )
            if hard and line.label == "O":
                pools["hard_negative"].append(
                    Candidate("hard_negative", document.document_id, position, risk)
                )
    for pool in pools.values():
        pool.sort(key=lambda item: item.risk, reverse=True)
    return pools


def _source_balanced_selection(
    ordered: Sequence[Candidate],
    by_id: Mapping[str, GoldDocument],
    selected_ids: set[tuple[str, int]],
    limit: int,
) -> list[Candidate]:
    queues: dict[str, list[Candidate]] = collections.defaultdict(list)
    for item in ordered:
        queues[by_id[item.document_id].source].append(item)
    cursors = {source: 0 for source in queues}
    chosen: list[Candidate] = []
    while len(chosen) < limit:
        progressed = False
        for source in sorted(queues):
            queue = queues[source]
            cursor = cursors[source]
            while cursor < len(queue):
                item = queue[cursor]
                cursor += 1
                identity = (item.document_id, item.line_position)
                if identity in selected_ids:
                    continue
                selected_ids.add(identity)
                chosen.append(item)
                progressed = True
                break
            cursors[source] = cursor
            if len(chosen) == limit:
                break
        if not progressed:
            break
    return chosen


def _context_window(
    document: GoldDocument, position: int, radius: int
) -> tuple[Sequence[Any], str]:
    complete_present_inventory = (
        document.coverage == "full_document"
        and document.n_present_lines == len(document.lines)
    )
    maximum_coordinate_step = 3 if complete_present_inventory else 1
    # Annotated windows may omit arbitrary nonblank body lines, while complete
    # documents may omit known blanks. Never join an unknown interval, or a
    # known blank run beyond the decoder's two-line bridge budget.
    left = position
    while (
        left > 0
        and document.lines[left].abs_idx - document.lines[left - 1].abs_idx
        <= maximum_coordinate_step
    ):
        left -= 1
    right = position
    while (
        right + 1 < len(document.lines)
        and document.lines[right + 1].abs_idx - document.lines[right].abs_idx
        <= maximum_coordinate_step
    ):
        right += 1
    start = max(left, position - radius)
    end = min(right + 1, position + radius + 1)
    coverage = (
        "full_document_with_at_most_two_known_blank_lines_per_gap"
        if complete_present_inventory
        else "contiguous_observed_window_only"
    )
    return document.lines[start:end], coverage


def build_audit(
    documents: Sequence[GoldDocument],
    baseline: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
    *,
    per_stratum: int = 50,
    context_radius: int = 15,
    seed: str = "codex56-structure-audit-v1",
    prompt_version: str = "codex56-structure-audit-prompt-v1",
    require_full: bool = True,
    allowed_split: str = "validation",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if per_stratum <= 0 or context_radius <= 0:
        raise ValueError("per_stratum and context_radius must be positive")
    _validate_audit_inputs(
        documents,
        baseline,
        candidate,
        allowed_split=allowed_split,
    )
    by_id = {document.document_id: document for document in documents}
    pools = _candidates(documents, baseline, candidate)
    selected_ids: set[tuple[str, int]] = set()
    requests: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    shortfalls: dict[str, int] = {}
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))

    for stratum in STRATA:
        tied: dict[tuple[int, int, int], list[Candidate]] = collections.defaultdict(
            list
        )
        for item in pools[stratum]:
            tied[item.risk[:3]].append(item)
        ordered: list[Candidate] = []
        for risk in sorted(tied, reverse=True):
            rows = tied[risk]
            rng.shuffle(rows)
            ordered.extend(rows)
        chosen = _source_balanced_selection(ordered, by_id, selected_ids, per_stratum)
        counts[stratum] = len(chosen)
        source_counts[stratum] = dict(
            sorted(
                collections.Counter(
                    by_id[item.document_id].source for item in chosen
                ).items()
            )
        )
        if len(chosen) < per_stratum:
            shortfalls[stratum] = per_stratum - len(chosen)

        for item in chosen:
            document = by_id[item.document_id]
            target = document.lines[item.line_position]
            context, context_coverage = _context_window(
                document, item.line_position, context_radius
            )
            request_identity = {
                "seed": seed,
                "stratum": stratum,
                "document_id": document.document_id,
                "line_id": target.line_id,
                "prompt_version": prompt_version,
            }
            request_id = hashlib.sha256(_canonical_bytes(request_identity)).hexdigest()
            context_rows = [
                {"abs_idx": line.abs_idx, "line_id": line.line_id, "text": line.text}
                for line in context
            ]
            request_body = {
                "schema_version": REQUEST_SCHEMA,
                "request_id": request_id,
                "prompt_version": prompt_version,
                "source": document.source,
                "opaque_document_id": hashlib.sha256(
                    f"{seed}\0{document.document_id}".encode()
                ).hexdigest(),
                "target_abs_idx": target.abs_idx,
                "context_start_abs_idx": context[0].abs_idx,
                "context_end_abs_idx": context[-1].abs_idx,
                "context_coverage": context_coverage,
                "crosses_unrepresented_interval": False,
                "lines": context_rows,
            }
            request_sha = canonical_json_sha256(request_body)
            request_body["request_sha256"] = request_sha
            requests.append(request_body)
            keys.append(
                {
                    "schema_version": KEY_SCHEMA,
                    "request_id": request_id,
                    "request_sha256": request_sha,
                    "stratum": stratum,
                    "document_id": document.document_id,
                    "work_id": document.work_id,
                    "representation_id": document.representation_id,
                    "source": document.source,
                    "line_id": target.line_id,
                    "abs_idx": target.abs_idx,
                    "gold_label": target.label,
                    "baseline_prediction": baseline[document.document_id][
                        item.line_position
                    ],
                    "candidate_prediction": candidate[document.document_id][
                        item.line_position
                    ],
                    "is_running_prose": target.is_running_prose,
                }
            )

    if require_full and shortfalls:
        raise ValueError(f"audit strata cannot reach requested counts: {shortfalls}")
    requests.sort(key=lambda row: row["request_id"])
    keys.sort(key=lambda row: row["request_id"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "seed": seed,
        "prompt_version": prompt_version,
        "allowed_split": allowed_split,
        "per_stratum_requested": per_stratum,
        "context_radius_present_lines": context_radius,
        "counts": counts,
        "source_counts": source_counts,
        "shortfalls": shortfalls,
        "request_count": len(requests),
        "request_set_sha256": canonical_json_sha256(requests),
        "key_set_sha256": canonical_json_sha256(keys),
        "blinding_level": "prompt_blinded_not_access_isolated",
        "blinding": (
            "request payloads omit selection stratum, gold labels, and "
            "baseline/candidate identities/predictions; the Codex CLI is "
            "instructed not to use tools, but read-only sandboxing is not an "
            "access-isolation guarantee"
        ),
        "selection_design": (
            "risk-ranked within source and round-robin source-balanced within "
            "each stratum; audit rates are descriptive of this sample only"
        ),
    }
    return requests, keys, manifest


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                rows.append(row)
    return rows


def preflight_silver_splits(path: str | Path, *, allowed_split: str) -> None:
    allowed = _split_key(allowed_split)
    if not allowed or allowed in SEALED_SPLITS:
        raise ValueError(f"forbidden audit split {allowed_split!r}")
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{number}: expected object")
            split = _split_key(raw.get("split"))
            if split in SEALED_SPLITS:
                raise ValueError(f"{path}:{number}: sealed split is forbidden")
            if split != allowed:
                raise ValueError(
                    f"{path}:{number}: split does not match {allowed_split!r}"
                )
            count += 1
    if count == 0:
        raise ValueError("audit silver input is empty")


def validate_requests(requests: Sequence[Mapping[str, Any]]) -> None:
    """Recompute every request hash and validate its target/context contract."""

    if not requests:
        raise ValueError("audit request set is empty")
    seen: set[str] = set()
    for number, request in enumerate(requests, 1):
        if request.get("schema_version") != REQUEST_SCHEMA:
            raise ValueError(f"request {number}: unsupported schema")
        request_id = request.get("request_id")
        if (
            not isinstance(request_id, str)
            or len(request_id) != 64
            or any(char not in "0123456789abcdef" for char in request_id)
            or request_id in seen
        ):
            raise ValueError(f"request {number}: invalid/duplicate request_id")
        seen.add(request_id)
        claimed_hash = request.get("request_sha256")
        if not isinstance(claimed_hash, str):
            raise ValueError(f"request {number}: missing request hash")
        body = dict(request)
        del body["request_sha256"]
        if canonical_json_sha256(body) != claimed_hash:
            raise ValueError(f"request {number}: request content hash mismatch")
        lines = request.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"request {number}: context lines are required")
        indices: list[int] = []
        line_ids: set[str] = set()
        for line in lines:
            if not isinstance(line, Mapping):
                raise ValueError(f"request {number}: invalid context line")
            abs_idx = line.get("abs_idx")
            line_id = line.get("line_id")
            text = line.get("text")
            if (
                not isinstance(abs_idx, int)
                or isinstance(abs_idx, bool)
                or abs_idx < 0
                or not isinstance(line_id, str)
                or not line_id
                or line_id in line_ids
                or not isinstance(text, str)
            ):
                raise ValueError(f"request {number}: invalid context line")
            indices.append(abs_idx)
            line_ids.add(line_id)
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise ValueError(f"request {number}: context indices are not ordered")
        target = request.get("target_abs_idx")
        if target not in set(indices):
            raise ValueError(f"request {number}: target is outside supplied context")
        if (
            request.get("context_start_abs_idx") != indices[0]
            or request.get("context_end_abs_idx") != indices[-1]
        ):
            raise ValueError(f"request {number}: context bounds mismatch")


def validate_responses(
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    *,
    expected_model: str,
) -> dict[str, Any]:
    validate_requests(requests)
    expected = {str(row["request_id"]): row for row in requests}
    seen: dict[str, Mapping[str, Any]] = {}
    labels: collections.Counter[str] = collections.Counter()
    low_confidence = 0
    for number, response in enumerate(responses, 1):
        if response.get("schema_version") != RESPONSE_SCHEMA:
            raise ValueError(f"response {number}: unsupported schema")
        request_id = str(response.get("request_id", ""))
        if request_id not in expected or request_id in seen:
            raise ValueError(f"response {number}: unknown/duplicate request_id")
        request = expected[request_id]
        if response.get("request_sha256") != request.get("request_sha256"):
            raise ValueError(f"response {number}: request hash mismatch")
        if response.get("reviewer_model") != expected_model:
            raise ValueError(f"response {number}: reviewer model mismatch")
        label = response.get("label")
        if label not in LABELS:
            raise ValueError(f"response {number}: invalid label")
        confidence = response.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"response {number}: invalid confidence")
        should_remove = response.get("should_remove")
        if not isinstance(should_remove, bool):
            raise ValueError(f"response {number}: should_remove must be boolean")
        action_label = label in {"BIB", "TOC"}
        if should_remove != action_label:
            raise ValueError(
                f"response {number}: label/removal decision is inconsistent"
            )
        start = response.get("start_abs_idx")
        end = response.get("end_abs_idx")
        if (start is None) != (end is None):
            raise ValueError(f"response {number}: incomplete span")
        if start is not None:
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start > end
            ):
                raise ValueError(f"response {number}: invalid span")
            if (
                start < request["context_start_abs_idx"]
                or end > request["context_end_abs_idx"]
            ):
                raise ValueError(f"response {number}: span outside supplied context")
        if action_label and start is None:
            raise ValueError(f"response {number}: action label requires a span")
        if not action_label and start is not None:
            raise ValueError(f"response {number}: keep label cannot carry a span")
        evidence = response.get("evidence_abs_indices")
        if not isinstance(evidence, list) or not all(
            isinstance(value, int) for value in evidence
        ):
            raise ValueError(f"response {number}: invalid evidence_abs_indices")
        valid_indices = {line["abs_idx"] for line in request["lines"]}
        if (
            not evidence
            or len(set(evidence)) != len(evidence)
            or not set(evidence).issubset(valid_indices)
        ):
            raise ValueError(f"response {number}: evidence outside supplied context")
        target = request["target_abs_idx"]
        if action_label and (
            start not in valid_indices
            or end not in valid_indices
            or not start <= target <= end
        ):
            raise ValueError(
                f"response {number}: action span must cover the target line"
            )
        cues = response.get("structural_cues")
        if (
            not isinstance(cues, list)
            or not cues
            or not all(isinstance(value, str) and value for value in cues)
        ):
            raise ValueError(f"response {number}: invalid structural_cues")
        seen[request_id] = response
        labels[label] += 1
        low_confidence += int(confidence < 0.6)
    missing = set(expected) - set(seen)
    if missing:
        raise ValueError(f"responses omit {len(missing)} requests")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "reviewer_model": expected_model,
        "request_count": len(expected),
        "response_count": len(seen),
        "label_counts": dict(sorted(labels.items())),
        "low_confidence_count": low_confidence,
        "request_set_sha256": canonical_json_sha256(list(requests)),
        "response_set_sha256": canonical_json_sha256(list(responses)),
    }


def summarize_findings(
    keys: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    *,
    expected_model: str,
) -> dict[str, Any]:
    """Summarize silver disagreement and apply the frozen expansion triggers."""

    validation = validate_responses(requests, responses, expected_model=expected_model)
    key_by_id = {str(row.get("request_id", "")): row for row in keys}
    request_by_id = {str(row.get("request_id", "")): row for row in requests}
    response_by_id = {str(row.get("request_id", "")): row for row in responses}
    if "" in key_by_id or len(key_by_id) != len(keys):
        raise ValueError("audit keys have empty or duplicate request IDs")
    if "" in response_by_id or len(response_by_id) != len(responses):
        raise ValueError("audit responses have empty or duplicate request IDs")
    if set(key_by_id) != set(response_by_id) or set(key_by_id) != set(request_by_id):
        raise ValueError("audit key/request/response request sets differ")
    for request_id, key in key_by_id.items():
        if key.get("request_sha256") != request_by_id[request_id].get("request_sha256"):
            raise ValueError("audit key/request hash mismatch")

    slices: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    global_counts: collections.Counter[str] = collections.Counter()
    examples: list[dict[str, Any]] = []
    for request_id in sorted(key_by_id):
        key = key_by_id[request_id]
        response = response_by_id[request_id]
        expected = {"O": "OTHER", "BIB": "BIB", "TOC": "TOC", "UNKNOWN": "UNKNOWN"}[
            str(key["gold_label"])
        ]
        disagreement = response.get("label") != expected
        high_confidence = disagreement and float(response.get("confidence", 0.0)) >= 0.8
        source = str(key["source"])
        stratum = str(key["stratum"])
        counts = slices[(source, stratum)]
        counts["count"] += 1
        counts["disagreement"] += int(disagreement)
        counts["high_confidence_disagreement"] += int(high_confidence)
        global_counts["count"] += 1
        global_counts["disagreement"] += int(disagreement)
        global_counts["high_confidence_disagreement"] += int(high_confidence)
        if high_confidence:
            examples.append(
                {
                    "request_id": request_id,
                    "source": source,
                    "stratum": stratum,
                    "old_silver_label": key["gold_label"],
                    "audit_label": response["label"],
                    "confidence": response["confidence"],
                }
            )

    slice_rows = []
    affected_sources: set[str] = set()
    for (source, stratum), counts in sorted(slices.items()):
        rate = counts["disagreement"] / counts["count"]
        expand = counts["high_confidence_disagreement"] >= 5 or (
            counts["count"] >= 10 and rate > 0.10
        )
        if expand:
            affected_sources.add(source)
        slice_rows.append(
            {
                "source": source,
                "stratum": stratum,
                "count": counts["count"],
                "disagreement_count": counts["disagreement"],
                "high_confidence_disagreement_count": counts[
                    "high_confidence_disagreement"
                ],
                "disagreement_rate": rate,
                "expand_slice": expand,
            }
        )
    audit_sample_rate = global_counts["disagreement"] / max(global_counts["count"], 1)
    full_reaudit = len(affected_sources) >= 2 or audit_sample_rate > 0.05
    return {
        "schema_version": "academic-structure-codex56-audit-findings-v1",
        "request_count": global_counts["count"],
        "disagreement_count": global_counts["disagreement"],
        "high_confidence_disagreement_count": global_counts[
            "high_confidence_disagreement"
        ],
        "audit_sample_disagreement_rate": audit_sample_rate,
        "slices": slice_rows,
        "affected_sources": sorted(affected_sources),
        "recommend_full_1392_reaudit": full_reaudit,
        "high_confidence_examples": examples,
        "validation_receipt": validation,
        "rate_scope": (
            "risk-ranked audit sample only; not a corpus-wide prevalence estimate"
        ),
        "policy": {
            "slice_trigger": "at_least_5_high_confidence_or_rate_gt_0.10_with_n_ge_10",
            "full_trigger": (
                "at_least_2_affected_sources_or_risk_audit_sample_rate_gt_0.05"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--silver", required=True)
    build.add_argument("--baseline-predictions", required=True)
    build.add_argument("--candidate-predictions", required=True)
    build.add_argument("--requests-out", required=True)
    build.add_argument("--key-out", required=True)
    build.add_argument("--manifest-out", required=True)
    build.add_argument("--per-stratum", type=int, default=50)
    build.add_argument("--context-radius", type=int, default=15)
    build.add_argument("--allow-shortfall", action="store_true")
    build.add_argument("--allowed-split", default="validation")
    validate = sub.add_parser("validate")
    validate.add_argument("--requests", required=True)
    validate.add_argument("--responses", required=True)
    validate.add_argument("--expected-model", required=True)
    validate.add_argument("--receipt-out", required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--key", required=True)
    summarize.add_argument("--requests", required=True)
    summarize.add_argument("--responses", required=True)
    summarize.add_argument("--expected-model", required=True)
    summarize.add_argument("--findings-out", required=True)
    args = parser.parse_args(argv)

    if args.command == "build":
        preflight_silver_splits(args.silver, allowed_split=args.allowed_split)
        documents = read_gold(args.silver)
        baseline = read_predictions(args.baseline_predictions, documents)
        candidate = read_predictions(args.candidate_predictions, documents)
        requests, keys, manifest = build_audit(
            documents,
            baseline,
            candidate,
            per_stratum=args.per_stratum,
            context_radius=args.context_radius,
            require_full=not args.allow_shortfall,
            allowed_split=args.allowed_split,
        )
        manifest.update(
            silver_sha256=sha256_file(args.silver),
            baseline_predictions_sha256=sha256_file(args.baseline_predictions),
            candidate_predictions_sha256=sha256_file(args.candidate_predictions),
        )
        _write_new(args.requests_out, requests)
        _write_new(args.key_out, keys)
        _write_json_new(args.manifest_out, manifest)
        return 0

    if args.command == "validate":
        requests = _read_jsonl(args.requests)
        responses = _read_jsonl(args.responses)
        receipt = validate_responses(
            requests, responses, expected_model=args.expected_model
        )
        receipt.update(
            requests_sha256=sha256_file(args.requests),
            responses_sha256=sha256_file(args.responses),
        )
        _write_json_new(args.receipt_out, receipt)
        return 0

    keys = _read_jsonl(args.key)
    requests = _read_jsonl(args.requests)
    responses = _read_jsonl(args.responses)
    findings = summarize_findings(
        keys,
        requests,
        responses,
        expected_model=args.expected_model,
    )
    findings.update(
        key_sha256=sha256_file(args.key),
        requests_sha256=sha256_file(args.requests),
        responses_sha256=sha256_file(args.responses),
    )
    _write_json_new(args.findings_out, findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
