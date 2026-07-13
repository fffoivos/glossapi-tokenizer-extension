#!/usr/bin/env python3
"""Immutable, explicit source admission for the Agent 1 v3 lane.

The command has intentionally narrow authority:

``build-packet``
    validates the complete Stage 30/35/quality/lineage/licence aggregate and
    freezes a *proposed* terminal decision for every candidate source;
``confirm``
    records an explicit user hash-confirmation of that exact immutable packet;
``validate``
    verifies a confirmation before a later destructive stage consumes it.

It never starts deduplication, deletes corpus rows, or turns a model
recommendation into an admission decision.  A proposal is evidence-bound but
remains pending until a human supplies the packet hash again at confirmation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import agent1_v3_review as review  # noqa: E402
import agent1_v3_review_aggregate as aggregate  # noqa: E402


PACKET_SCHEMA = "agent1_full_corpus_v3_source_admission_packet_v2"
PROPOSAL_SCHEMA = "agent1_full_corpus_v3_source_admission_proposal_v1"
CONFIRMATION_SCHEMA = "agent1_full_corpus_v3_source_admission_confirmation_v1"
DECISIONS = frozenset({"include", "include_after_cleaning", "low_weight", "exclude", "quarantine"})
ADMITTED_DECISIONS = frozenset({"include", "include_after_cleaning", "low_weight"})
SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(label: str, value: Any, *, minimum_length: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum_length:
        raise ValueError(f"{label} must be a non-empty string of at least {minimum_length} characters")
    return value


def _require_fraction_or_none(label: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be null or a fraction")
    number = float(value)
    if not 0.0 <= number <= 1.0:
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
        raise ValueError(f"{path}: expected JSON object")
    return value


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def _verify_binding(label: str, expected: Any, path: Path) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise ValueError(f"{label}: expected binding object")
    actual = binding(path)
    if expected.get("bytes") != actual["bytes"] or expected.get("sha256") != actual["sha256"]:
        raise ValueError(f"{label}: bytes/SHA-256 binding drift")
    return actual


def _digest(value: Mapping[str, Any], field: str) -> str:
    copy = dict(value)
    copy.pop(field, None)
    return sha256_json(copy)


def _assert_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    actual = _require_sha256(f"{label}.{field}", value.get(field))
    if actual != _digest(value, field):
        raise ValueError(f"{label}: {field} drift")


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable admission artifact: {path}")
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
        # link keeps publication no-replace even if two operators race.
        os.link(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _map_sources(rows: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        source_id = _require_nonempty_string(f"{label}[{index}].source_id", raw.get("source_id"))
        if source_id in result:
            raise ValueError(f"{label}: duplicate source_id {source_id!r}")
        result[source_id] = dict(raw)
    return result


def _expected_sources(aggregate_value: Mapping[str, Any], roster: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    aggregate.validate_aggregate(aggregate_value, roster=roster)
    route_validation = review.validate_candidate_roster_routes(roster)
    candidates = list(route_validation["candidate_source_ids"])
    sources = _map_sources(aggregate_value.get("sources"), label="review aggregate sources")
    if list(sources) != candidates:
        raise ValueError("review aggregate sources are not exact roster coverage/order")
    return candidates, sources


def _validate_expected_token_loss(value: Any, *, source_id: str, decision: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"kind", "estimated_fraction", "basis"}:
        raise ValueError(f"{source_id}: expected_token_loss must contain exactly kind/estimated_fraction/basis")
    kind = value.get("kind")
    fraction = _require_fraction_or_none(f"{source_id}.expected_token_loss.estimated_fraction", value.get("estimated_fraction"))
    basis = _require_nonempty_string(f"{source_id}.expected_token_loss.basis", value.get("basis"), minimum_length=12)
    if kind == "not_applicable":
        if fraction is not None:
            raise ValueError(f"{source_id}: non-applicable token loss must be null")
    elif kind == "estimated":
        if fraction is None:
            raise ValueError(f"{source_id}: estimated token loss needs a bounded fraction")
    else:
        raise ValueError(f"{source_id}: unsupported expected token-loss kind")
    if decision == "include_after_cleaning" and kind != "estimated":
        raise ValueError(f"{source_id}: include_after_cleaning needs an expected token-loss estimate")
    return {"kind": str(kind), "estimated_fraction": fraction, "basis": basis}


def _validate_proposal_row(
    raw: Any,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("source admission proposal row must be an object")
    required = {
        "source_id",
        "source_dataset",
        "source_revision",
        "decision",
        "source_evidence_sha256",
        "review_support",
        "manual_override_reason",
        "rationale",
        "required_cleaning",
        "expected_token_loss",
    }
    if set(raw) != required:
        raise ValueError(
            "source admission proposal row key drift; "
            f"missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}"
        )
    source_id = _require_nonempty_string("proposal.source_id", raw.get("source_id"))
    for field in ("source_dataset", "source_revision"):
        if raw.get(field) != source.get(field):
            raise ValueError(f"{source_id}: proposal {field} differs from review aggregate")
    if raw.get("source_evidence_sha256") != source.get("source_evidence_sha256"):
        raise ValueError(f"{source_id}: proposal does not bind exact source review evidence")
    decision = raw.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"{source_id}: unsupported admission decision")
    rationale = _require_nonempty_string(f"{source_id}.rationale", raw.get("rationale"), minimum_length=24)
    if len(rationale) > 2000:
        raise ValueError(f"{source_id}: rationale is too long")
    cleaning = raw.get("required_cleaning")
    if not isinstance(cleaning, list) or any(not isinstance(item, str) or not item.strip() for item in cleaning):
        raise ValueError(f"{source_id}: required_cleaning must be a list of non-empty policy names")
    if len(cleaning) != len(set(cleaning)):
        raise ValueError(f"{source_id}: required_cleaning repeats a policy name")
    if decision == "include_after_cleaning" and not cleaning:
        raise ValueError(f"{source_id}: include_after_cleaning requires named cleaning")
    if decision in {"include", "low_weight"} and cleaning:
        raise ValueError(f"{source_id}: direct admission/low weight cannot smuggle an unstaged cleaning action")
    expected_token_loss = _validate_expected_token_loss(raw.get("expected_token_loss"), source_id=source_id, decision=str(decision))

    review_support = raw.get("review_support")
    override = raw.get("manual_override_reason")
    recommendation_counts = source.get("review", {}).get("recommendation_counts", {})
    if not isinstance(recommendation_counts, Mapping):
        raise ValueError(f"{source_id}: aggregate recommendation distribution is missing")
    if review_support == "supported":
        if override is not None:
            raise ValueError(f"{source_id}: supported proposal must not carry an override reason")
        if int(recommendation_counts.get(decision, 0)) < 1:
            raise ValueError(f"{source_id}: proposed decision is not supported by any resolved review")
    elif review_support == "manual_override":
        _require_nonempty_string(f"{source_id}.manual_override_reason", override, minimum_length=24)
        if len(str(override)) > 2000:
            raise ValueError(f"{source_id}: manual override reason is too long")
    else:
        raise ValueError(f"{source_id}: review_support must be supported or manual_override")

    local_training = source.get("license", {}).get("local_training", {})
    if not isinstance(local_training, Mapping) or not isinstance(local_training.get("eligible"), bool):
        raise ValueError(f"{source_id}: aggregate licence state is incomplete")
    if decision in ADMITTED_DECISIONS and local_training["eligible"] is not True:
        raise ValueError(f"{source_id}: licence excludes local training; proposed admission is forbidden")
    return {
        "source_id": source_id,
        "source_dataset": str(raw["source_dataset"]),
        "source_revision": str(raw["source_revision"]),
        "decision": str(decision),
        "source_evidence_sha256": str(raw["source_evidence_sha256"]),
        "review_support": str(review_support),
        "manual_override_reason": override,
        "rationale": rationale,
        "required_cleaning": sorted(cleaning),
        "expected_token_loss": expected_token_loss,
    }


def validate_proposal(
    proposal: Mapping[str, Any],
    *,
    aggregate_value: Mapping[str, Any],
    aggregate_path: Path,
    roster: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate one complete, evidence-bound source-decision proposal."""

    if set(proposal) != {"schema_version", "review_aggregate", "sources"}:
        raise ValueError("admission proposal must contain exactly schema_version/review_aggregate/sources")
    if proposal.get("schema_version") != PROPOSAL_SCHEMA:
        raise ValueError("unsupported admission proposal schema")
    _verify_binding("proposal review aggregate", proposal.get("review_aggregate"), aggregate_path)
    candidates, source_evidence = _expected_sources(aggregate_value, roster)
    proposal_rows = _map_sources(proposal.get("sources"), label="admission proposal sources")
    if set(proposal_rows) != set(candidates):
        missing = sorted(set(candidates) - set(proposal_rows))
        extra = sorted(set(proposal_rows) - set(candidates))
        raise ValueError(f"admission proposal must decide every roster source; missing={missing}, extra={extra}")
    result = [
        _validate_proposal_row(proposal_rows[source_id], source=source_evidence[source_id])
        for source_id in candidates
    ]
    return result


def _packet_sources(value: Any) -> dict[str, dict[str, Any]]:
    rows = _map_sources(value, label="admission packet sources")
    for source_id, row in rows.items():
        required = {
            "source_id",
            "source_dataset",
            "source_revision",
            "decision",
            "source_evidence_sha256",
            "review_support",
            "manual_override_reason",
            "rationale",
            "required_cleaning",
            "expected_token_loss",
        }
        if set(row) != required:
            raise ValueError(f"{source_id}: admission packet source key drift")
        if row.get("decision") not in DECISIONS:
            raise ValueError(f"{source_id}: admission packet decision drift")
        _require_sha256(f"{source_id}.source_evidence_sha256", row.get("source_evidence_sha256"))
    return rows


def validate_packet(packet: Mapping[str, Any], *, roster: Mapping[str, Any] | None = None) -> None:
    """Validate a pending admission packet without treating it as approved."""

    required_packet_keys = {
        "schema_version",
        "status",
        "created_at",
        "run_id",
        "review_aggregate_sha256",
        "inputs",
        "source_ids",
        "sources",
        "destructive_progression",
        "packet_sha256",
    }
    if set(packet) != required_packet_keys:
        raise ValueError(
            "admission packet key drift; "
            f"missing={sorted(required_packet_keys - set(packet))}, extra={sorted(set(packet) - required_packet_keys)}"
        )
    if packet.get("schema_version") != PACKET_SCHEMA or packet.get("status") != "pending_user_confirmation":
        raise ValueError("admission packet is not a pending v3 packet")
    _assert_digest(packet, "packet_sha256", "admission packet")
    _require_nonempty_string("admission packet run_id", packet.get("run_id"))
    inputs = packet.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("admission packet lacks immutable input bindings")
    required_inputs = {
        "candidate_roster",
        "review_aggregate",
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
        "proposed_decisions",
    }
    if set(inputs) != required_inputs:
        raise ValueError("admission packet input binding set drift")
    for name in sorted(required_inputs):
        bound = inputs[name]
        if not isinstance(bound, Mapping):
            raise ValueError(f"admission packet input {name} is not a binding")
        _require_sha256(f"admission packet input {name}", bound.get("sha256"))
        if not isinstance(bound.get("bytes"), int) or int(bound["bytes"]) < 1:
            raise ValueError(f"admission packet input {name} has invalid bytes")
    if inputs["review_aggregate"].get("sha256") != packet.get("review_aggregate_sha256"):
        raise ValueError("admission packet aggregate hash drift")
    progression = packet.get("destructive_progression")
    if progression != {
        "deduplication_permitted": False,
        "reason": "explicit_hash_confirmed_user_confirmation_required",
    }:
        raise ValueError("pending admission packet destructive-progression guard drift")

    aggregate_path = Path(str(inputs["review_aggregate"].get("path", "")))
    proposal_path = Path(str(inputs["proposed_decisions"].get("path", "")))
    _verify_binding("admission packet review aggregate", inputs["review_aggregate"], aggregate_path)
    _verify_binding("admission packet proposed decisions", inputs["proposed_decisions"], proposal_path)
    aggregate_value = read_object(aggregate_path)
    if roster is None:
        roster_path = Path(str(inputs["candidate_roster"].get("path", "")))
        _verify_binding("admission packet candidate roster", inputs["candidate_roster"], roster_path)
        roster = read_object(roster_path)
    else:
        review.validate_candidate_roster_routes(roster)
        # The caller's roster must still match the packet byte receipt.
        roster_path = Path(str(inputs["candidate_roster"].get("path", "")))
        if roster_path.is_file():
            _verify_binding("admission packet candidate roster", inputs["candidate_roster"], roster_path)
        else:
            raise ValueError("admission packet candidate roster path is unavailable for byte-level validation")
    aggregate.validate_aggregate(aggregate_value, roster=roster)
    if aggregate_value.get("run_id") != packet.get("run_id"):
        raise ValueError("admission packet run_id differs from review aggregate")
    proposal = read_object(proposal_path)
    proposed_sources = validate_proposal(
        proposal,
        aggregate_value=aggregate_value,
        aggregate_path=aggregate_path,
        roster=roster,
    )
    aggregate_inputs = aggregate_value.get("inputs")
    if not isinstance(aggregate_inputs, Mapping):
        raise ValueError("review aggregate lacks input bindings")
    for packet_name, aggregate_name in (
        ("candidate_roster", "candidate_roster"),
        ("review_packet", "review_packet"),
        ("review_requests", "review_requests"),
        ("review_responses", "review_responses"),
        ("response_execution_receipt", "response_execution_receipt"),
        ("adjudication_execution_receipt", "adjudication_execution_receipt"),
        ("stage35_review_closure", "stage35_review_closure"),
        ("review_sample_quality_summary", "review_sample_quality_summary"),
        ("review_sample_quality_handoff", "review_sample_quality_handoff"),
        ("quality_summary", "quality_summary"),
        ("lineage_summary", "lineage_summary"),
        ("source_novelty", "source_novelty"),
        ("license_adjudication", "license_adjudication"),
    ):
        expected = aggregate_inputs.get(aggregate_name)
        actual = inputs.get(packet_name)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping) or (
            expected.get("bytes") != actual.get("bytes") or expected.get("sha256") != actual.get("sha256")
        ):
            raise ValueError(f"admission packet {packet_name} binding differs from complete review aggregate")
    sources = _packet_sources(packet.get("sources"))
    source_ids = packet.get("source_ids")
    if not isinstance(source_ids, list) or source_ids != list(sources):
        raise ValueError("admission packet source list/order drift")
    candidates = list(review.validate_candidate_roster_routes(roster)["candidate_source_ids"])
    if source_ids != candidates:
        raise ValueError("admission packet source coverage differs from roster")
    if packet.get("sources") != proposed_sources:
        raise ValueError("admission packet sources differ from exact evidence-bound proposal")


def cmd_build_packet(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite admission packet: {args.output}")
    roster = read_object(args.roster)
    aggregate_value = read_object(args.review_aggregate)
    candidates, _ = _expected_sources(aggregate_value, roster)
    if aggregate_value.get("run_id") != args.run_id:
        raise ValueError("run_id differs from complete review aggregate")
    proposal = read_object(args.proposed_decisions)
    proposed_sources = validate_proposal(
        proposal,
        aggregate_value=aggregate_value,
        aggregate_path=args.review_aggregate,
        roster=roster,
    )
    aggregate_inputs = aggregate_value.get("inputs")
    if not isinstance(aggregate_inputs, Mapping):  # guarded by aggregate validator; defensive.
        raise ValueError("review aggregate lacks input bindings")
    copied_inputs = {
        "candidate_roster": binding(args.roster),
        "review_aggregate": binding(args.review_aggregate),
        "review_packet": dict(aggregate_inputs["review_packet"]),
        "review_requests": dict(aggregate_inputs["review_requests"]),
        "review_responses": dict(aggregate_inputs["review_responses"]),
        "response_execution_receipt": dict(aggregate_inputs["response_execution_receipt"]),
        "adjudication_execution_receipt": dict(aggregate_inputs["adjudication_execution_receipt"]),
        "stage35_review_closure": dict(aggregate_inputs["stage35_review_closure"]),
        "review_sample_quality_summary": dict(aggregate_inputs["review_sample_quality_summary"]),
        "review_sample_quality_handoff": dict(aggregate_inputs["review_sample_quality_handoff"]),
        "quality_summary": dict(aggregate_inputs["quality_summary"]),
        "lineage_summary": dict(aggregate_inputs["lineage_summary"]),
        "source_novelty": dict(aggregate_inputs["source_novelty"]),
        "license_adjudication": dict(aggregate_inputs["license_adjudication"]),
        "proposed_decisions": binding(args.proposed_decisions),
    }
    payload: dict[str, Any] = {
        "schema_version": PACKET_SCHEMA,
        "status": "pending_user_confirmation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "review_aggregate_sha256": copied_inputs["review_aggregate"]["sha256"],
        "inputs": copied_inputs,
        "source_ids": candidates,
        "sources": proposed_sources,
        "destructive_progression": {
            "deduplication_permitted": False,
            "reason": "explicit_hash_confirmed_user_confirmation_required",
        },
    }
    payload["packet_sha256"] = _digest(payload, "packet_sha256")
    validate_packet(payload, roster=roster)
    write_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "packet": str(args.output.resolve()),
                # Confirmation deliberately uses the byte hash of the exact
                # file the user inspected.  The internal manifest digest is
                # reported separately so it cannot be copied accidentally.
                "packet_file_sha256": sha256(args.output),
                "packet_manifest_sha256": payload["packet_sha256"],
            },
            sort_keys=True,
        )
    )


def cmd_confirm(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite admission confirmation: {args.output}")
    packet = read_object(args.packet)
    roster = read_object(args.roster) if args.roster is not None else None
    validate_packet(packet, roster=roster)
    actual = sha256(args.packet)
    if actual != args.packet_sha256:
        raise ValueError(f"admission packet sha256 mismatch: {actual} != {args.packet_sha256}")
    if args.confirm_user_reviewed_packet_sha256 != actual:
        raise ValueError("explicit user-confirmed packet SHA-256 does not equal packet bytes")
    note = _require_nonempty_string("confirmation note", args.confirmation_note, minimum_length=12)
    payload: dict[str, Any] = {
        "schema_version": CONFIRMATION_SCHEMA,
        "status": "approved",
        "confirmation_mode": "explicit_hash_confirmed_user_confirmation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": packet["run_id"],
        "packet": binding(args.packet),
        "user_confirmed_packet_sha256": actual,
        "confirmation_note": note,
        # Exact packet sources are copied verbatim: confirmation has no flag
        # or secondary decision input through which an operator could alter an
        # evidence-bound proposal after inspecting the hash.
        "sources": packet["sources"],
    }
    payload["confirmation_sha256"] = _digest(payload, "confirmation_sha256")
    validate_confirmation(payload, roster=roster)
    write_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "confirmation": str(args.output.resolve()),
                "confirmed_packet_file_sha256": actual,
            },
            sort_keys=True,
        )
    )


def validate_confirmation(confirmation: Mapping[str, Any], *, roster: Mapping[str, Any] | None = None) -> None:
    required_keys = {
        "schema_version",
        "status",
        "confirmation_mode",
        "created_at",
        "run_id",
        "packet",
        "user_confirmed_packet_sha256",
        "confirmation_note",
        "sources",
        "confirmation_sha256",
    }
    if set(confirmation) != required_keys:
        raise ValueError(
            "admission confirmation key drift; "
            f"missing={sorted(required_keys - set(confirmation))}, extra={sorted(set(confirmation) - required_keys)}"
        )
    if confirmation.get("schema_version") != CONFIRMATION_SCHEMA or confirmation.get("status") != "approved":
        raise ValueError("not an approved v3 admission confirmation")
    if confirmation.get("confirmation_mode") != "explicit_hash_confirmed_user_confirmation":
        raise ValueError("confirmation lacks explicit user hash-confirmation mode")
    _assert_digest(confirmation, "confirmation_sha256", "admission confirmation")
    packet_binding = confirmation.get("packet")
    if not isinstance(packet_binding, Mapping):
        raise ValueError("confirmation lacks packet binding")
    packet_path = Path(str(packet_binding.get("path", "")))
    _verify_binding("confirmed admission packet", packet_binding, packet_path)
    packet = read_object(packet_path)
    validate_packet(packet, roster=roster)
    if confirmation.get("user_confirmed_packet_sha256") != packet_binding.get("sha256"):
        raise ValueError("confirmation user packet hash drift")
    if confirmation.get("run_id") != packet.get("run_id"):
        raise ValueError("confirmation run_id drift")
    _require_nonempty_string("confirmation note", confirmation.get("confirmation_note"), minimum_length=12)
    if confirmation.get("sources") != packet.get("sources"):
        raise ValueError("confirmation sources differ from the exact confirmed proposal")
    _packet_sources(confirmation.get("sources"))


def cmd_validate(args: argparse.Namespace) -> None:
    confirmation = read_object(args.confirmation)
    roster = read_object(args.roster) if args.roster is not None else None
    validate_confirmation(confirmation, roster=roster)
    print(json.dumps({"ok": True, "sources": len(confirmation["sources"]), "confirmation": str(args.confirmation.resolve())}, sort_keys=True))


def cmd_validate_packet(args: argparse.Namespace) -> None:
    packet = read_object(args.packet)
    roster = read_object(args.roster) if args.roster is not None else None
    validate_packet(packet, roster=roster)
    print(json.dumps({"ok": True, "sources": len(packet["sources"]), "packet": str(args.packet.resolve())}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-packet", help="freeze complete evidence-bound proposed decisions; does not admit")
    build.add_argument("--run-id", required=True)
    build.add_argument("--roster", type=Path, required=True)
    build.add_argument("--review-aggregate", type=Path, required=True)
    build.add_argument("--proposed-decisions", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(func=cmd_build_packet)
    confirm = sub.add_parser("confirm", help="record exact user hash-confirmation of a pending packet")
    confirm.add_argument("--packet", type=Path, required=True)
    confirm.add_argument(
        "--packet-sha256",
        required=True,
        help="SHA-256 of the exact packet file bytes (packet_file_sha256 from build-packet)",
    )
    confirm.add_argument("--confirm-user-reviewed-packet-sha256", required=True)
    confirm.add_argument("--confirmation-note", required=True)
    confirm.add_argument("--roster", type=Path)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.set_defaults(func=cmd_confirm)
    validate = sub.add_parser("validate", help="validate approved confirmation and exact pending packet")
    validate.add_argument("--confirmation", type=Path, required=True)
    validate.add_argument("--roster", type=Path)
    validate.set_defaults(func=cmd_validate)
    packet = sub.add_parser("validate-packet", help="validate a pending, not-yet-approved packet")
    packet.add_argument("--packet", type=Path, required=True)
    packet.add_argument("--roster", type=Path)
    packet.set_defaults(func=cmd_validate_packet)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
