#!/usr/bin/env python3
"""Prepare and execute immutable, schema-bound bibliography role reviews."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contract import canonical_json_sha256, sha256_file


PACKET_SCHEMA = "bibliography-role-review-packet-v1"
REVIEW_SCHEMA = "bibliography-role-review-v1"
BATCH_SCHEMA = "bibliography-role-review-execution-batch-v1"
RUN_SCHEMA = "bibliography-role-review-execution-run-v1"
CALIBRATION_SCHEMA = "bibliography-role-calibration-selection-v1"
ROLES = frozenset(
    {
        "ENTRY_ANCHOR",
        "CONTINUATION",
        "FILLER",
        "HEADER",
        "SUBHEADER",
        "NON_BIB",
        "UNKNOWN",
    }
)
BOUNDARIES = frozenset({"NONE", "SOFT_STOP", "HARD_STOP"})
LINE_FIELDS = frozenset({"line_id", "abs_idx", "document_position_percent", "text"})
CASE_FIELDS = frozenset(
    {
        "case_id",
        "block_case_id",
        "chunk_index",
        "chunk_count",
        "document_id",
        "work_id",
        "source",
        "n_physical_lines",
        "lines",
    }
)


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_packet(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("unsupported bibliography review packet")
    blinding = raw.get("blinding")
    required_blinding = {
        "detector_features_hidden",
        "model_predictions_hidden",
        "nomination_strata_hidden",
        "original_region_labels_hidden",
    }
    if not isinstance(blinding, dict) or not required_blinding <= blinding.keys():
        raise ValueError("packet lacks explicit blinding declaration")
    if not all(blinding[key] is True for key in required_blinding):
        raise ValueError("review packet is not fully blinded")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("review packet has no cases")
    seen_cases: set[str] = set()
    for case_number, case in enumerate(cases, 1):
        if not isinstance(case, dict) or frozenset(case) != CASE_FIELDS:
            raise ValueError(f"case {case_number}: unexpected or missing fields")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise ValueError(f"case {case_number}: case_id must be unique")
        seen_cases.add(case_id)
        lines = case.get("lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"case {case_id}: no lines")
        seen_lines: set[str] = set()
        for line in lines:
            if not isinstance(line, dict) or frozenset(line) != LINE_FIELDS:
                raise ValueError(f"case {case_id}: unexpected or missing line fields")
            line_id, abs_idx, text = line.get("line_id"), line.get("abs_idx"), line.get("text")
            if (
                not isinstance(line_id, str)
                or not line_id
                or line_id in seen_lines
                or not isinstance(abs_idx, int)
                or abs_idx < 0
                or not isinstance(text, str)
            ):
                raise ValueError(f"case {case_id}: malformed or repeated line")
            seen_lines.add(line_id)
    return raw


def load_provenance(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") not in {
        "bibliography-role-review-selection-v1",
        CALIBRATION_SCHEMA,
    }:
        raise ValueError("unsupported bibliography review provenance")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("review provenance has no cases")
    return raw


def prepare_calibration(
    packet: Mapping[str, Any], provenance: Mapping[str, Any], *, seed: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose one bounded, single-chunk block per source with complementary stressors."""

    blind_by_block = {str(row["block_case_id"]): row for row in packet["cases"]}
    preferences = {
        "greek_phd": ("exact_header", "long_wrapped"),
        "kallipos": ("conventional_dense", "heterogeneous"),
        "openarchives": ("sparse_internal", "heterogeneous"),
    }
    selected_blind: list[Mapping[str, Any]] = []
    selected_provenance: list[Mapping[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for source, desired in preferences.items():
        candidates = []
        for row in provenance["cases"]:
            block_case_id = str(row.get("block_case_id", ""))
            blind = blind_by_block.get(block_case_id)
            if row.get("source") != source or blind is None or int(blind.get("chunk_count", 0)) != 1:
                continue
            strata = set(str(value) for value in row.get("strata", []))
            preference_score = sum(value in strata for value in desired)
            rank = int.from_bytes(
                hashlib.sha256(f"{seed}\0{source}\0{block_case_id}".encode()).digest(), "big"
            )
            candidates.append((-preference_score, rank, row, blind))
        if not candidates:
            raise ValueError(f"no single-chunk calibration candidate for {source}")
        _, _, chosen_provenance, chosen_blind = min(candidates, key=lambda value: value[:2])
        selected_blind.append(chosen_blind)
        selected_provenance.append(chosen_provenance)
        decisions.append(
            {
                "source": source,
                "block_case_id": chosen_blind["block_case_id"],
                "case_id": chosen_blind["case_id"],
                "desired_strata": list(desired),
                "observed_strata": chosen_provenance["strata"],
                "line_count": len(chosen_blind["lines"]),
            }
        )
    calibration_packet = dict(packet)
    calibration_packet["cases"] = selected_blind
    calibration_packet["selection"] = {
        "purpose": "three-source dual-review calibration",
        "seed": seed,
        "block_count": 3,
        "case_count": 3,
        "source_block_counts": {source: 1 for source in sorted(preferences)},
        "parent_packet_sha256": canonical_json_sha256(packet),
    }
    calibration_provenance = {
        "schema_version": CALIBRATION_SCHEMA,
        "seed": seed,
        "parent_provenance_sha256": canonical_json_sha256(provenance),
        "decisions": decisions,
        "cases": selected_provenance,
        "warning": "Contains immutable source labels and strata; never pass to reviewers.",
    }
    load_packet_from_value(calibration_packet)
    return calibration_packet, calibration_provenance


def load_packet_from_value(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory packet through the same strict inventory checks."""

    with tempfile.TemporaryDirectory(prefix="bib-role-packet-check-") as directory:
        path = Path(directory) / "packet.json"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return load_packet(path)


def _ordered_cases(cases: Sequence[Mapping[str, Any]], pass_id: str) -> list[dict[str, Any]]:
    if pass_id == "pass-a":
        return [dict(row) for row in sorted(cases, key=lambda row: str(row["case_id"]))]
    if pass_id == "pass-b":
        return [
            dict(row)
            for row in sorted(
                cases,
                key=lambda row: hashlib.sha256(
                    f"bibliography-role-pass-b\0{row['case_id']}".encode()
                ).digest(),
                reverse=True,
            )
        ]
    raise ValueError("pass-id must be pass-a or pass-b")


def make_batches(
    cases: Sequence[Mapping[str, Any]], *, pass_id: str, reviewer_id: str,
    model: str, prompt_sha256: str, output_schema_sha256: str, batch_size: int,
) -> list[dict[str, Any]]:
    if not reviewer_id or not model:
        raise ValueError("reviewer_id and model are required")
    if batch_size not in (1, 2):
        raise ValueError("batch_size must be 1 or 2")
    ordered = _ordered_cases(cases, pass_id)
    batches: list[dict[str, Any]] = []
    for start in range(0, len(ordered), batch_size):
        local = ordered[start : start + batch_size]
        contract = {
            "pass_id": pass_id,
            "reviewer_id": reviewer_id,
            "model": model,
            "prompt_sha256": prompt_sha256,
            "output_schema_sha256": output_schema_sha256,
            "case_sha256": [[row["case_id"], canonical_json_sha256(row)] for row in local],
        }
        batches.append(
            {"batch_id": canonical_json_sha256(contract), "contract": contract, "cases": local}
        )
    return batches


def validate_review_payload(
    batch: Mapping[str, Any], payload: Mapping[str, Any], *, reviewer_id: str
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("unsupported review response schema")
    if payload.get("reviewer") != reviewer_id:
        raise ValueError("reviewer identity mismatch")
    expected_cases = {str(row["case_id"]): row for row in batch["cases"]}
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("review response omits or invents cases")
    seen_cases: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "lines", "notes"}:
            raise ValueError("review case has unexpected or missing fields")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen_cases or case_id not in expected_cases:
            raise ValueError("review response repeats or invents a case")
        seen_cases.add(case_id)
        expected_lines = expected_cases[case_id]["lines"]
        lines = case.get("lines")
        if not isinstance(lines, list) or len(lines) != len(expected_lines):
            raise ValueError(f"case {case_id}: review omits or invents lines")
        observed_by_id: dict[str, dict[str, Any]] = {}
        for position, line in enumerate(lines):
            required = {"line_id", "abs_idx", "role", "boundary_flag", "confidence", "reason"}
            if not isinstance(line, dict) or set(line) != required:
                raise ValueError(f"case {case_id}: line {position} has invalid fields")
            line_id = line.get("line_id")
            if not isinstance(line_id, str) or not line_id or line_id in observed_by_id:
                raise ValueError(f"case {case_id}: repeated or empty line identity")
            observed_by_id[line_id] = line
            if line.get("role") not in ROLES or line.get("boundary_flag") not in BOUNDARIES:
                raise ValueError(f"case {case_id}: invalid role or boundary")
            confidence = line.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0 <= confidence <= 1
                or not isinstance(line.get("reason"), str)
                or not line["reason"].strip()
            ):
                raise ValueError(f"case {case_id}: invalid confidence or reason")
        expected_by_id = {str(line["line_id"]): line for line in expected_lines}
        if set(observed_by_id) != set(expected_by_id):
            raise ValueError(f"case {case_id}: review omits or invents line identities")
        canonical_lines = []
        for expected in expected_lines:
            line = observed_by_id[str(expected["line_id"])]
            if line.get("abs_idx") != expected["abs_idx"]:
                raise ValueError(f"case {case_id}: line coordinate mismatch")
            canonical_lines.append(line)
        if not isinstance(case.get("notes"), str):
            raise ValueError(f"case {case_id}: notes must be a string")
        normalized.append({"case_id": case_id, "lines": canonical_lines, "notes": case["notes"]})
    if seen_cases != set(expected_cases):
        raise ValueError("review response case set mismatch")
    normalized.sort(key=lambda row: str(row["case_id"]))
    return {"schema_version": REVIEW_SCHEMA, "reviewer": reviewer_id, "cases": normalized}


def execute_batch(
    batch: Mapping[str, Any], *, pass_id: str, reviewer_id: str, model: str,
    prompt_text: str, output_schema: Path, batch_dir: Path, timeout_seconds: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    final_path = batch_dir / f"{batch['batch_id']}.json"
    if final_path.exists():
        record = json.loads(final_path.read_text(encoding="utf-8"))
        if record.get("contract") != batch["contract"]:
            raise ValueError(f"existing batch contract mismatch: {final_path}")
        validate_review_payload(batch, record.get("review", {}), reviewer_id=reviewer_id)
        return record
    envelope = {
        "pass_id": pass_id,
        "reviewer_id": reviewer_id,
        "independence": "Review only this envelope; no prior pass is supplied.",
        "cases": batch["cases"],
    }
    full_prompt = prompt_text.rstrip() + "\n\n" + json.dumps(envelope, ensure_ascii=False) + "\n"
    with tempfile.TemporaryDirectory(prefix=f"bib-role-{batch['batch_id'][:12]}-") as directory:
        response_path = Path(directory) / "response.json"
        empty_workspace = Path(directory) / "empty-review-workspace"
        empty_workspace.mkdir()
        command = [
            "codex", "exec", "--model", model, "--sandbox", "read-only", "--ephemeral",
            "--skip-git-repo-check", "--cd", str(empty_workspace), "--config",
            f'model_reasoning_effort="{reasoning_effort}"', "--output-schema",
            str(output_schema), "--output-last-message", str(response_path), "-",
        ]
        completed = subprocess.run(
            command, input=full_prompt, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Codex batch {batch['batch_id']} failed ({completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )
        if not response_path.is_file():
            raise RuntimeError(f"Codex batch {batch['batch_id']} produced no final response")
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    try:
        review = validate_review_payload(batch, payload, reviewer_id=reviewer_id)
    except (TypeError, ValueError) as error:
        rejection = {
            "schema_version": "bibliography-role-review-rejection-v1",
            "status": "rejected_not_accepted",
            "batch_id": batch["batch_id"],
            "contract": batch["contract"],
            "validation_error": str(error),
            "payload": payload,
        }
        rejection_path = batch_dir / f"{batch['batch_id']}.rejected-{canonical_json_sha256(payload)[:16]}.json"
        if not rejection_path.exists():
            _write_json_new(rejection_path, rejection)
        raise ValueError(f"rejected response preserved at {rejection_path}: {error}") from error
    record = {
        "schema_version": BATCH_SCHEMA,
        "batch_id": batch["batch_id"],
        "contract": batch["contract"],
        "review": review,
        "review_sha256": canonical_json_sha256(review),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    _write_json_new(final_path, record)
    return record


def _existing_or_write(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"existing immutable output differs: {path}")
        return
    _write_json_new(path, value)


def run_reviews(args: argparse.Namespace) -> int:
    packet_path = Path(args.packet).resolve()
    packet = load_packet(packet_path)
    prompt_path = Path(args.prompt).resolve()
    schema_path = Path(args.output_schema).resolve()
    prompt_sha = sha256_file(prompt_path)
    schema_sha = sha256_file(schema_path)
    batches = make_batches(
        packet["cases"], pass_id=args.pass_id, reviewer_id=args.reviewer_id,
        model=args.model, prompt_sha256=prompt_sha, output_schema_sha256=schema_sha,
        batch_size=args.batch_size,
    )
    if args.maximum_batches is not None:
        if args.maximum_batches <= 0:
            raise ValueError("maximum-batches must be positive")
        batches = batches[: args.maximum_batches]
    batch_dir = Path(args.batch_dir).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "schema_version": "bibliography-role-review-execution-contract-v1",
        "pass_id": args.pass_id,
        "reviewer_id": args.reviewer_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "packet_sha256": sha256_file(packet_path),
        "prompt_sha256": prompt_sha,
        "output_schema_sha256": schema_sha,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "batch_ids": [row["batch_id"] for row in batches],
        "limited_preflight": args.maximum_batches is not None,
        "blinding_level": "prompt_blinded_and_empty_read_only_workspace",
    }
    _existing_or_write(batch_dir / "run.contract.json", run_contract)
    prompt_text = prompt_path.read_text(encoding="utf-8")

    def runner(batch: Mapping[str, Any]) -> dict[str, Any]:
        return execute_batch(
            batch, pass_id=args.pass_id, reviewer_id=args.reviewer_id, model=args.model,
            prompt_text=prompt_text, output_schema=schema_path, batch_dir=batch_dir,
            timeout_seconds=args.timeout_seconds, reasoning_effort=args.reasoning_effort,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(runner, batches))
    cases = [case for record in records for case in record["review"]["cases"]]
    cases.sort(key=lambda row: str(row["case_id"]))
    aggregate = {"schema_version": REVIEW_SCHEMA, "reviewer": args.reviewer_id, "cases": cases}
    responses_out = Path(args.responses_out).resolve()
    _existing_or_write(responses_out, aggregate)
    receipt = {
        "schema_version": RUN_SCHEMA,
        "status": "preflight_passed" if args.maximum_batches is not None else "passed",
        "pass_id": args.pass_id,
        "reviewer_id": args.reviewer_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "case_count": len(cases),
        "line_response_count": sum(len(row["lines"]) for row in cases),
        "batch_count": len(records),
        "packet_sha256": sha256_file(packet_path),
        "run_contract_sha256": sha256_file(batch_dir / "run.contract.json"),
        "responses_sha256": sha256_file(responses_out),
    }
    _existing_or_write(Path(args.receipt_out).resolve(), receipt)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-calibration")
    prepare.add_argument("--packet", required=True)
    prepare.add_argument("--provenance", required=True)
    prepare.add_argument("--packet-out", required=True)
    prepare.add_argument("--provenance-out", required=True)
    prepare.add_argument("--seed", default="bibliography-role-calibration-v1")
    run = subparsers.add_parser("run")
    run.add_argument("--packet", required=True)
    run.add_argument("--pass-id", choices=("pass-a", "pass-b"), required=True)
    run.add_argument("--reviewer-id", required=True)
    run.add_argument("--model", default="gpt-5.6-sol")
    run.add_argument("--reasoning-effort", default="high")
    run.add_argument("--prompt", default=str(Path(__file__).with_name("bibliography_role_review_prompt.md")))
    run.add_argument("--output-schema", default=str(Path(__file__).with_name("bibliography_role_review.schema.json")))
    run.add_argument("--batch-dir", required=True)
    run.add_argument("--responses-out", required=True)
    run.add_argument("--receipt-out", required=True)
    run.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    run.add_argument("--workers", type=int, choices=(1, 2), default=2)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--maximum-batches", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-calibration":
        packet, provenance = prepare_calibration(
            load_packet(Path(args.packet).resolve()),
            load_provenance(Path(args.provenance).resolve()), seed=args.seed,
        )
        _write_json_new(Path(args.packet_out).resolve(), packet)
        _write_json_new(Path(args.provenance_out).resolve(), provenance)
        return 0
    return run_reviews(args)


if __name__ == "__main__":
    raise SystemExit(main())
