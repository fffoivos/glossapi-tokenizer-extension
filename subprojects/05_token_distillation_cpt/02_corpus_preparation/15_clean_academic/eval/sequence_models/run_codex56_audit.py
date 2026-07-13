#!/usr/bin/env python3
"""Execute immutable, schema-bound Codex structural-audit batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .codex56_audit import (
    MANIFEST_SCHEMA,
    REQUEST_SCHEMA,
    _read_jsonl,
    _write_json_new,
    _write_new,
    validate_requests,
    validate_responses,
)
from .contract import canonical_json_sha256, sha256_file

BATCH_SCHEMA = "academic-structure-codex56-audit-execution-batch-v1"
RUN_SCHEMA = "academic-structure-codex56-audit-execution-run-v1"


def make_batches(
    requests: Sequence[Mapping[str, Any]],
    *,
    model: str,
    prompt_sha256: str,
    output_schema_sha256: str,
    batch_size: int = 12,
) -> list[dict[str, Any]]:
    if not model.strip():
        raise ValueError("an exact non-empty reviewer model is required")
    if not 1 <= batch_size <= 12:
        raise ValueError("batch_size must be between 1 and 12")
    seen: set[str] = set()
    ordered = []
    validate_requests(requests)
    for request in requests:
        if request.get("schema_version") != REQUEST_SCHEMA:
            raise ValueError("unsupported request schema")
        request_id = str(request.get("request_id", ""))
        if not request_id or request_id in seen:
            raise ValueError("request IDs must be non-empty and unique")
        seen.add(request_id)
        ordered.append(dict(request))
    if not ordered:
        raise ValueError("audit request set is empty")
    ordered.sort(key=lambda row: row["request_id"])
    batches = []
    for start in range(0, len(ordered), batch_size):
        rows = ordered[start : start + batch_size]
        contract = {
            "model": model,
            "prompt_sha256": prompt_sha256,
            "output_schema_sha256": output_schema_sha256,
            "requests": [[row["request_id"], row["request_sha256"]] for row in rows],
        }
        batches.append(
            {
                "batch_id": canonical_json_sha256(contract),
                "contract": contract,
                "requests": rows,
            }
        )
    return batches


def validate_batch_payload(
    batch: Mapping[str, Any], payload: Mapping[str, Any], *, model: str
) -> list[dict[str, Any]]:
    responses = payload.get("responses")
    if not isinstance(responses, list) or not all(
        isinstance(row, dict) for row in responses
    ):
        raise ValueError("batch output must contain a response-object list")
    validate_responses(batch["requests"], responses, expected_model=model)
    return [dict(row) for row in responses]


def validate_request_manifest(
    requests: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    validate_requests(requests)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported request manifest schema")
    if manifest.get("request_count") != len(requests):
        raise ValueError("request manifest count mismatch")
    if manifest.get("request_set_sha256") != canonical_json_sha256(list(requests)):
        raise ValueError("request manifest set hash mismatch")


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def execute_batch(
    batch: Mapping[str, Any],
    *,
    model: str,
    prompt_text: str,
    output_schema: Path,
    batch_dir: Path,
    timeout_seconds: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    final_path = batch_dir / f"{batch['batch_id']}.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            existing.get("batch_id") != batch["batch_id"]
            or existing.get("model") != model
        ):
            raise ValueError(f"existing batch contract mismatch: {final_path}")
        validate_batch_payload(
            batch, {"responses": existing.get("responses")}, model=model
        )
        return existing

    request_json = json.dumps(
        {"reviewer_model": model, "requests": batch["requests"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    full_prompt = prompt_text.rstrip() + "\n" + request_json + "\n"
    with tempfile.TemporaryDirectory(
        prefix=f"codex56-{batch['batch_id'][:12]}-"
    ) as directory:
        response_path = Path(directory) / "response.json"
        review_directory = Path(directory) / "empty-review-workspace"
        review_directory.mkdir()
        command = [
            "codex",
            "exec",
            "--model",
            model,
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(review_directory),
            "--config",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(output_schema),
            "--output-last-message",
            str(response_path),
            "-",
        ]
        completed = subprocess.run(
            command,
            input=full_prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Codex batch {batch['batch_id']} failed ({completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )
        if not response_path.is_file():
            raise RuntimeError(
                f"Codex batch {batch['batch_id']} produced no final response"
            )
        payload = json.loads(response_path.read_text(encoding="utf-8"))
    try:
        responses = validate_batch_payload(batch, payload, model=model)
    except (TypeError, ValueError) as error:
        payload_hash = canonical_json_sha256(payload)
        rejection_path = batch_dir / (
            f"{batch['batch_id']}.rejected-{payload_hash[:16]}.json"
        )
        rejection = {
            "schema_version": "academic-structure-codex56-audit-rejection-v1",
            "status": "rejected_not_accepted",
            "batch_id": batch["batch_id"],
            "model": model,
            "contract": batch["contract"],
            "validation_error_type": type(error).__name__,
            "validation_error": str(error),
            "payload": payload,
            "payload_sha256": payload_hash,
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        }
        if rejection_path.exists():
            if json.loads(rejection_path.read_text(encoding="utf-8")) != rejection:
                raise ValueError(
                    f"existing rejected response record differs: {rejection_path}"
                ) from error
        else:
            _exclusive_json(rejection_path, rejection)
        raise ValueError(
            f"Codex batch {batch['batch_id']} response rejected; "
            f"preserved at {rejection_path}: {error}"
        ) from error
    record = {
        "schema_version": BATCH_SCHEMA,
        "batch_id": batch["batch_id"],
        "model": model,
        "contract": batch["contract"],
        "responses": responses,
        "response_set_sha256": canonical_json_sha256(responses),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    _exclusive_json(final_path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--request-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-schema", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--responses-out", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--maximum-batches", type=int)
    args = parser.parse_args(argv)
    if args.workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    prompt_path = Path(args.prompt).resolve()
    schema_path = Path(args.output_schema).resolve()
    requests = _read_jsonl(args.requests)
    request_manifest = json.loads(
        Path(args.request_manifest).read_text(encoding="utf-8")
    )
    if not isinstance(request_manifest, dict):
        raise ValueError("request manifest must be an object")
    validate_request_manifest(requests, request_manifest)
    batches = make_batches(
        requests,
        model=args.model,
        prompt_sha256=sha256_file(prompt_path),
        output_schema_sha256=sha256_file(schema_path),
        batch_size=args.batch_size,
    )
    if args.maximum_batches is not None:
        if args.maximum_batches <= 0:
            raise ValueError("maximum-batches must be positive")
        batches = batches[: args.maximum_batches]
    prompt_text = prompt_path.read_text(encoding="utf-8")
    batch_dir = Path(args.batch_dir).resolve()
    run_contract = {
        "schema_version": "academic-structure-codex56-audit-execution-contract-v1",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "requests_sha256": sha256_file(args.requests),
        "request_manifest_sha256": sha256_file(args.request_manifest),
        "blinding_level": "prompt_blinded_not_access_isolated",
        "prompt_sha256": sha256_file(prompt_path),
        "output_schema_sha256": sha256_file(schema_path),
        "batch_ids": [batch["batch_id"] for batch in batches],
        "limited_preflight": args.maximum_batches is not None,
    }
    contract_path = batch_dir / "run.contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise ValueError("existing Codex audit run contract does not match")
    else:
        _exclusive_json(contract_path, run_contract)
    runner = lambda batch: execute_batch(  # noqa: E731
        batch,
        model=args.model,
        prompt_text=prompt_text,
        output_schema=schema_path,
        batch_dir=batch_dir,
        timeout_seconds=args.timeout_seconds,
        reasoning_effort=args.reasoning_effort,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(runner, batches))
    responses = [
        response
        for record in sorted(records, key=lambda row: row["batch_id"])
        for response in record["responses"]
    ]
    responses.sort(key=lambda row: row["request_id"])
    _write_new(args.responses_out, responses)
    receipt = {
        "schema_version": RUN_SCHEMA,
        "status": "preflight_passed" if args.maximum_batches is not None else "passed",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "request_count": sum(len(batch["requests"]) for batch in batches),
        "response_count": len(responses),
        "batch_count": len(records),
        "batch_ids": sorted(record["batch_id"] for record in records),
        "run_contract_sha256": sha256_file(contract_path),
        "requests_sha256": sha256_file(args.requests),
        "prompt_sha256": sha256_file(prompt_path),
        "output_schema_sha256": sha256_file(schema_path),
        "responses_sha256": sha256_file(args.responses_out),
    }
    _write_json_new(args.receipt_out, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
