#!/usr/bin/env python3
"""Immutable contracts for Agent 1's post-Nanochat v3 data lane.

This deliberately does not reuse the v2 pipeline identity.  It is a small,
stdlib-only boundary that can run before the Python runtime is bootstrapped on
Clariden.  The stage jobs use it to bind every v3 artifact to the same run
contract and to reject an existing output rather than overwriting it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


RUN_SCHEMA = "agent1_full_corpus_v3_run_contract_v1"
STAGE_SCHEMA = "agent1_full_corpus_v3_stage_contract_v1"
STAGE_ATTEMPT_SCHEMA = "agent1_full_corpus_v3_stage_attempt_contract_v1"
STAGE_RECEIPT_SCHEMA = "agent1_full_corpus_v3_stage_receipt_v1"
RUN_ID_RE = re.compile(r"^agent1-full-corpus-v3-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{7,40}$")

STAGES = (
    "10-normalize",
    "20-lineage",
    "30-review-packet",
    "35-quality-review-evidence",
    "40-admission",
    "50-dedup",
    "55-greekmmlu-freeze",
    "60-decontamination",
    "65-anonymization-sanitization",
    "70-prestructural-freeze",
    "75-structural-detection-audit",
    "78-structural-apply",
    "80-final-validation",
)

PRESTRUCTURAL_STAGES = STAGES[:10]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def atomic_json(path: Path, value: dict[str, Any], *, no_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if no_replace:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"required non-empty input is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_binding(name: str, binding: Any) -> None:
    if not isinstance(binding, dict):
        raise ValueError(f"invalid binding for {name}")
    path = Path(str(binding.get("path", "")))
    if not path.is_file():
        raise ValueError(f"bound input disappeared: {name}: {path}")
    if path.stat().st_size != binding.get("bytes") or sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"bound input drift: {name}: {path}")


def named_bindings(values: Iterable[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in values:
        if name in result:
            raise ValueError(f"duplicate contract binding {name}")
        result[name] = file_binding(path)
    return result


def run_contract_path(run_root: Path) -> Path:
    return run_root / "run_contract.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether ``path`` is contained by ``root`` without string-prefix bugs."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_storage_boundary(run_root: Path, data_root: Path) -> tuple[Path, Path]:
    """Resolve and reject overlapping metadata and bulk-data namespaces.

    The run root is deliberately Capstor metadata space; the data root is the
    IOPS namespace for canonical shards, checkpoints, and other bulk stage
    products.  They must never be aliases or parents of one another.  The
    caller is responsible for choosing the physical mounts; this contract
    preserves that already-selected boundary exactly and records it immutably.
    """

    resolved_run = run_root.resolve()
    resolved_data = data_root.resolve()
    if resolved_run == resolved_data or _is_relative_to(resolved_run, resolved_data) or _is_relative_to(
        resolved_data, resolved_run
    ):
        raise ValueError(
            "v3 run root (metadata/receipts) and data root (bulk stage data) must be distinct, non-overlapping paths"
        )
    return resolved_run, resolved_data


def configured_data_root(args: argparse.Namespace, run_root: Path) -> Path:
    """Obtain the explicit-or-exported bulk-data root when freezing a run."""

    raw = args.data_root
    if raw is None:
        exported = os.environ.get("AGENT1_V3_DATA_ROOT")
        if not exported:
            raise ValueError("--data-root is required (or export AGENT1_V3_DATA_ROOT before freeze-run)")
        raw = Path(exported)
    _, data_root = validate_storage_boundary(run_root, Path(raw))
    return data_root


def contract_data_root(run_root: Path, contract: dict[str, Any]) -> Path:
    """Load the immutable bulk-data namespace from a validated run contract."""

    raw = contract.get("data_root")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{run_contract_path(run_root)}: missing immutable data_root")
    _, data_root = validate_storage_boundary(run_root, Path(raw))
    return data_root


def metadata_attempt_root(run_root: Path, stage: str, attempt_id: str) -> Path:
    return stage_root(run_root, stage) / "attempts" / attempt_id


def stage_attempt_contract_path(run_root: Path, stage: str, attempt_id: str) -> Path:
    """Return the immutable per-Slurm-job attempt contract path.

    ``stage_contract.json`` remains the single immutable logical contract for
    a stage.  Every actual execution has a separate, never-reused contract
    under its job-unique attempt directory, so a retry never rewrites either
    the original logical contract or the failed attempt's evidence.
    """

    return metadata_attempt_root(run_root, stage, attempt_id) / "attempt_contract.json"


def data_stage_root(data_root: Path, stage: str) -> Path:
    return data_root / "stages" / stage


def data_attempt_root(data_root: Path, stage: str, attempt_id: str) -> Path:
    return data_stage_root(data_root, stage) / "attempts" / attempt_id


def contract_digest(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("created_at", None)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_valid_contract(run_root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    root = run_root.resolve()
    path = run_contract_path(root)
    contract = read_object(path)
    if contract.get("schema_version") != RUN_SCHEMA:
        raise ValueError(f"{path}: unsupported run contract")
    if run_id is not None and contract.get("run_id") != run_id:
        raise ValueError(f"{path}: run_id mismatch")
    if contract.get("run_root") != str(root):
        raise ValueError(f"{path}: run_root drift")
    if contract.get("contract_sha256") != contract_digest(contract):
        raise ValueError(f"{path}: contract hash mismatch")
    contract_data_root(root, contract)
    for name, binding in contract.get("inputs", {}).items():
        verify_binding(str(name), binding)
    return contract


def cmd_freeze_run(args: argparse.Namespace) -> None:
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError(
            "run id must be agent1-full-corpus-v3-<UTC YYYYmmddTHHMMSSZ>-<git short sha>"
        )
    root = args.run_root.resolve()
    data_root = configured_data_root(args, root)
    contract_path = run_contract_path(root)
    # Phase 0 intentionally writes fresh acquisition/runtime evidence before
    # the final merged receipt exists.  Nothing else may pre-exist when the
    # downstream run contract is frozen.
    if root.exists() and any(root.iterdir()):
        permitted = {"phase0"}
        present = {entry.name for entry in root.iterdir()}
        if not present <= permitted:
            raise FileExistsError(f"immutable v3 run root already exists: {root}")
    bindings = named_bindings(
        (
            ("source_registry", args.source_registry),
            ("source_aliases", args.source_aliases),
            ("candidate_roster", args.candidate_roster),
            ("post_cutoff_inventory", args.post_cutoff_inventory),
            ("nanochat_initial_roster", args.nanochat_initial_roster),
            ("acquisition_receipt", args.acquisition_receipt),
            ("tokenizer", args.tokenizer),
            ("review_policy", args.review_policy),
            ("review_prompt", args.review_prompt),
            ("review_response_schema", args.review_response_schema),
            ("glossapi_build_receipt", args.glossapi_build_receipt),
            ("license_adjudication", args.license_adjudication),
            ("training_eligibility_policy", args.training_eligibility_policy),
            ("dedup_policy", args.dedup_policy),
            ("greekmmlu_policy", args.greekmmlu_policy),
            ("anonymization_policy", args.anonymization_policy),
            ("structural_policy", args.structural_policy),
        )
    )
    payload: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "run_id": args.run_id,
        "run_root": str(root),
        "data_root": str(data_root),
        "code_commit": args.code_commit,
        "prestructural_only": bool(args.prestructural_only),
        "stage_graph": list(STAGES),
        "inputs": bindings,
        "created_at": now(),
    }
    payload["contract_sha256"] = contract_digest(payload)
    root.mkdir(parents=True, exist_ok=True)
    try:
        atomic_json(contract_path, payload, no_replace=True)
    except BaseException:
        # The directory is intentionally empty on this path.  Preserve it for
        # operator inspection rather than masking the original failure.
        raise
    print(json.dumps({"ok": True, "contract": str(contract_path), "sha256": payload["contract_sha256"]}))


def expected_upstream(stage: str, *, prestructural_only: bool) -> tuple[str, ...]:
    graph = PRESTRUCTURAL_STAGES if prestructural_only else STAGES
    try:
        index = graph.index(stage)
    except ValueError as exc:
        raise ValueError(f"stage not valid for this run mode: {stage}") from exc
    if stage == "55-greekmmlu-freeze":
        return ("50-dedup",)
    if stage == "75-structural-detection-audit":
        return ("70-prestructural-freeze",)
    if stage == "78-structural-apply":
        return ("75-structural-detection-audit",)
    if stage == "80-final-validation":
        return ("78-structural-apply",)
    return () if index == 0 else (graph[index - 1],)


def stage_root(run_root: Path, stage: str) -> Path:
    return run_root / "stages" / stage


def stage_receipt_path(run_root: Path, stage: str) -> Path:
    return stage_root(run_root, stage) / "stage_receipt.json"


ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_attempt_id(value: str) -> str:
    if not isinstance(value, str) or not ATTEMPT_ID_RE.fullmatch(value):
        raise ValueError("attempt id must use only letters, digits, '.', '_' or '-'")
    return value


def parse_stage_parameters(raw: str) -> dict[str, Any]:
    """Parse a caller-provided parameter object without normalizing its bytes.

    The parsed object is kept for machine-readable inspection, while the raw
    JSON text is frozen separately in the logical contract.  That makes a
    retry fail closed even when a caller changes only whitespace or key order.
    """

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--parameters-json is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--parameters-json must be an object")
    return value


def stage_logical_payload(
    *,
    contract: dict[str, Any],
    stage: str,
    upstream_stages: tuple[str, ...],
    upstream_receipts: dict[str, dict[str, Any]],
    inputs: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    parameters_json: str,
) -> dict[str, Any]:
    """Build the byte-sensitive, attempt-independent part of a stage.

    Attempt IDs and their storage roots intentionally do *not* appear here:
    a retry must get a new job-unique directory.  Everything that could alter
    processing semantics, including the exact submitted JSON parameter bytes,
    does appear here and is compared exactly before a retry is permitted.
    """

    return {
        "schema_version": STAGE_SCHEMA,
        "run_id": contract["run_id"],
        "stage": stage,
        "code_commit": contract["code_commit"],
        "run_contract_sha256": contract["contract_sha256"],
        "upstream_stages": list(upstream_stages),
        "upstream_receipts": upstream_receipts,
        "inputs": inputs,
        "parameters": parameters,
        "parameters_json": parameters_json,
    }


def stage_logical_digest(payload: dict[str, Any]) -> str:
    """Hash an exact logical-stage payload without timestamp exceptions."""

    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stage_storage(
    run_root: Path, data_root: Path, stage: str, attempt_id: str
) -> dict[str, str]:
    """Return the only paired metadata/bulk roots allowed for one attempt."""

    return {
        "metadata_attempt_dir": str(metadata_attempt_root(run_root, stage, attempt_id).resolve()),
        "data_attempt_dir": str(data_attempt_root(data_root, stage, attempt_id).resolve()),
    }


def validate_stage_storage(
    storage: Any, *, run_root: Path, data_root: Path, stage: str, attempt_id: str, path: Path
) -> tuple[Path, Path]:
    """Validate an immutable paired-attempt storage declaration exactly."""

    expected = stage_storage(run_root, data_root, stage, attempt_id)
    if not isinstance(storage, dict) or storage != expected:
        raise ValueError(f"{path}: immutable metadata/data attempt boundary drift")
    return Path(expected["metadata_attempt_dir"]), Path(expected["data_attempt_dir"])


def supplied_or_contract_data_root(
    args: argparse.Namespace, run_root: Path, contract: dict[str, Any]
) -> Path:
    """Reject a caller that tries to switch the frozen bulk-data namespace."""

    expected = contract_data_root(run_root, contract)
    supplied = getattr(args, "data_root", None)
    if supplied is None:
        return expected
    _, received = validate_storage_boundary(run_root, Path(supplied))
    if received != expected:
        raise ValueError(
            f"stage data root differs from the frozen run contract: expected {expected}, received {received}"
        )
    return expected


def load_stage_contract(
    run_root: Path, stage: str, contract: dict[str, Any]
) -> tuple[dict[str, Any], Path, Path]:
    """Load the immutable logical contract plus its initial attempt roots.

    The returned paths belong to the initial attempt recorded in the root
    contract.  They are useful for backward-compatible inspection only; a
    passed receipt is always resolved through its own successful attempt
    contract below.
    """

    root = run_root.resolve()
    path = stage_root(root, stage) / "stage_contract.json"
    stage_contract = read_object(path)
    expected = {
        "schema_version": STAGE_SCHEMA,
        "run_id": contract["run_id"],
        "stage": stage,
        "code_commit": contract["code_commit"],
        "run_contract_sha256": contract["contract_sha256"],
    }
    for key, value in expected.items():
        if stage_contract.get(key) != value:
            raise ValueError(f"{path}: {key} drift")
    if stage_contract.get("contract_sha256") != contract_digest(stage_contract):
        raise ValueError(f"{path}: stage contract hash mismatch")
    attempt_id = validate_attempt_id(stage_contract.get("attempt_id"))
    data_root = contract_data_root(root, contract)
    metadata_root, bulk_root = validate_stage_storage(
        stage_contract.get("storage"),
        run_root=root,
        data_root=data_root,
        stage=stage,
        attempt_id=attempt_id,
        path=path,
    )
    inputs = stage_contract.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{path}: stage inputs must be an object")
    for name, binding in inputs.items():
        verify_binding(f"{stage}:input:{name}", binding)
    parameters_json = stage_contract.get("parameters_json")
    if not isinstance(parameters_json, str):
        raise ValueError(f"{path}: missing exact parameters_json")
    parameters = parse_stage_parameters(parameters_json)
    if stage_contract.get("parameters") != parameters:
        raise ValueError(f"{path}: parsed parameters drift from frozen parameter bytes")
    expected = expected_upstream(stage, prestructural_only=bool(contract["prestructural_only"]))
    if stage_contract.get("upstream_stages") != list(expected):
        raise ValueError(f"{path}: upstream stage declaration drift")
    upstream_receipts = stage_contract.get("upstream_receipts")
    if not isinstance(upstream_receipts, dict) or set(upstream_receipts) != set(expected):
        raise ValueError(f"{path}: upstream receipt binding coverage drift")
    logical = stage_logical_payload(
        contract=contract,
        stage=stage,
        upstream_stages=expected,
        upstream_receipts=upstream_receipts,
        inputs=inputs,
        parameters=parameters,
        parameters_json=parameters_json,
    )
    if stage_contract.get("logical_contract_sha256") != stage_logical_digest(logical):
        raise ValueError(f"{path}: logical stage contract hash mismatch")
    current_upstream = upstream_receipt_bindings(root, expected, contract)
    if upstream_receipts != current_upstream:
        raise ValueError(f"{path}: upstream receipt bindings drift")
    return stage_contract, metadata_root, bulk_root


def output_is_in_stage_attempt(output: Path, metadata_root: Path, bulk_root: Path) -> bool:
    return _is_relative_to(output, metadata_root) or _is_relative_to(output, bulk_root)


def upstream_receipt_bindings(
    run_root: Path, upstream_stages: tuple[str, ...], contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate and bind all required upstream passed receipts exactly."""

    bindings: dict[str, dict[str, Any]] = {}
    for upstream in upstream_stages:
        load_stage_receipt(run_root, upstream, contract)
        bindings[upstream] = file_binding(stage_receipt_path(run_root, upstream))
    return bindings


def load_stage_attempt_contract(
    run_root: Path,
    stage: str,
    contract: dict[str, Any],
    stage_contract: dict[str, Any],
    attempt_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Load one immutable execution attempt bound to a logical stage contract."""

    root = run_root.resolve()
    safe_attempt_id = validate_attempt_id(attempt_id)
    path = stage_attempt_contract_path(root, stage, safe_attempt_id)
    attempt_contract = read_object(path)
    expected = {
        "schema_version": STAGE_ATTEMPT_SCHEMA,
        "run_id": contract["run_id"],
        "stage": stage,
        "code_commit": contract["code_commit"],
        "run_contract_sha256": contract["contract_sha256"],
        "stage_contract_sha256": stage_contract["contract_sha256"],
        "logical_contract_sha256": stage_contract["logical_contract_sha256"],
        "attempt_id": safe_attempt_id,
    }
    for key, value in expected.items():
        if attempt_contract.get(key) != value:
            raise ValueError(f"{path}: {key} drift")
    if attempt_contract.get("contract_sha256") != contract_digest(attempt_contract):
        raise ValueError(f"{path}: attempt contract hash mismatch")
    data_root = contract_data_root(root, contract)
    metadata_root, bulk_root = validate_stage_storage(
        attempt_contract.get("storage"),
        run_root=root,
        data_root=data_root,
        stage=stage,
        attempt_id=safe_attempt_id,
        path=path,
    )
    if not metadata_root.is_dir() or not bulk_root.is_dir():
        raise ValueError(f"{path}: paired attempt directories are missing")
    return attempt_contract, metadata_root, bulk_root


def successful_attempt_binding(
    attempt_contract: dict[str, Any]
) -> dict[str, Any]:
    """Return the receipt payload that pins successful output roots exactly."""

    storage = attempt_contract.get("storage")
    if not isinstance(storage, dict):  # defensive; load_stage_attempt_contract checked it
        raise ValueError("attempt contract lacks immutable storage")
    return {
        "attempt_id": attempt_contract["attempt_id"],
        "attempt_contract_sha256": attempt_contract["contract_sha256"],
        "metadata_attempt_dir": storage["metadata_attempt_dir"],
        "data_attempt_dir": storage["data_attempt_dir"],
    }


def load_stage_receipt(run_root: Path, stage: str, contract: dict[str, Any]) -> dict[str, Any]:
    root = run_root.resolve()
    path = stage_receipt_path(root, stage)
    if not path.is_file():
        raise ValueError(f"upstream stage receipt is missing: {stage}: {path}")
    stage_contract, _, _ = load_stage_contract(root, stage, contract)
    receipt = read_object(path)
    expected = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "status": "passed",
        "run_id": contract["run_id"],
        "code_commit": contract["code_commit"],
        "stage": stage,
        "run_contract_sha256": contract["contract_sha256"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{path}: {key} drift")
    receipt_for_digest = dict(receipt)
    receipt_for_digest.pop("receipt_sha256", None)
    if receipt.get("receipt_sha256") != contract_digest(receipt_for_digest):
        raise ValueError(f"{path}: receipt hash mismatch")
    if receipt.get("stage_contract_sha256") != stage_contract.get("contract_sha256"):
        raise ValueError(f"{path}: receipt does not bind the immutable logical stage contract")
    attempt_id = validate_attempt_id(receipt.get("attempt_id"))
    attempt_contract, metadata_root, bulk_root = load_stage_attempt_contract(
        root, stage, contract, stage_contract, attempt_id
    )
    if receipt.get("attempt_contract_sha256") != attempt_contract.get("contract_sha256"):
        raise ValueError(f"{path}: receipt does not bind its successful attempt contract")
    if receipt.get("successful_attempt") != successful_attempt_binding(attempt_contract):
        raise ValueError(f"{path}: receipt successful-attempt storage binding drift")
    marker = stage_root(root, stage) / "COMPLETED"
    if not marker.is_file():
        raise ValueError(f"{path}: passed receipt lacks immutable COMPLETED marker")
    expected_marker = f"{sha256_file(path)}  stage_receipt.json\n"
    if marker.read_text(encoding="utf-8") != expected_marker:
        raise ValueError(f"{marker}: receipt marker hash drift")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"{path}: outputs must be a non-empty list")
    for output in outputs:
        verify_binding(f"{stage}:output", output)
        output_path = Path(str(output.get("path", ""))).resolve()
        if not output_is_in_stage_attempt(output_path, metadata_root, bulk_root):
            raise ValueError(
                f"{path}: stage output escapes its immutable metadata/data attempt roots: {output_path}"
            )
    return receipt


def validate_incomplete_stage_layout(directory: Path, bulk_stage: Path) -> None:
    """Reject root-level debris rather than silently folding it into a retry."""

    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"incomplete stage root is not a real directory: {directory}")
    allowed_metadata = {"stage_contract.json", "attempts"}
    unexpected_metadata = {entry.name for entry in directory.iterdir()} - allowed_metadata
    if unexpected_metadata:
        raise ValueError(
            f"incomplete stage root has unbound artifacts; inspect instead of resuming: {sorted(unexpected_metadata)}"
        )
    attempts = directory / "attempts"
    if attempts.exists() and (not attempts.is_dir() or attempts.is_symlink()):
        raise ValueError(f"stage attempts namespace is unsafe: {attempts}")
    if not bulk_stage.exists():
        return
    if not bulk_stage.is_dir() or bulk_stage.is_symlink():
        raise ValueError(f"incomplete bulk stage root is not a real directory: {bulk_stage}")
    allowed_bulk = {"attempts"}
    unexpected_bulk = {entry.name for entry in bulk_stage.iterdir()} - allowed_bulk
    if unexpected_bulk:
        raise ValueError(
            f"incomplete bulk stage root has unbound artifacts; inspect instead of resuming: {sorted(unexpected_bulk)}"
        )
    attempts = bulk_stage / "attempts"
    if attempts.exists() and (not attempts.is_dir() or attempts.is_symlink()):
        raise ValueError(f"bulk stage attempts namespace is unsafe: {attempts}")


def cmd_begin_stage(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    contract = load_valid_contract(root, run_id=args.run_id)
    data_root = supplied_or_contract_data_root(args, root, contract)
    attempt_id = validate_attempt_id(args.attempt_id)
    expected = expected_upstream(args.stage, prestructural_only=bool(contract["prestructural_only"]))
    upstream_receipts = upstream_receipt_bindings(root, expected, contract)
    parameters = parse_stage_parameters(args.parameters_json)
    inputs = named_bindings((name, Path(path)) for name, path in args.input)
    requested_logical = stage_logical_payload(
        contract=contract,
        stage=args.stage,
        upstream_stages=expected,
        upstream_receipts=upstream_receipts,
        inputs=inputs,
        parameters=parameters,
        parameters_json=args.parameters_json,
    )
    directory = stage_root(root, args.stage)
    bulk_stage = data_stage_root(data_root, args.stage)
    receipt_path = stage_receipt_path(root, args.stage)
    completed_marker = directory / "COMPLETED"
    if directory.exists():
        # A receipt can be partially published only if a job dies between the
        # no-replace receipt and marker writes.  Treat that ambiguity as
        # terminal: neither a completed nor a partially published stage may
        # ever be resumed or overwritten.
        if receipt_path.exists() or completed_marker.exists():
            raise FileExistsError(
                f"stage is completed or has a published completion artifact and may never resume: {directory}"
            )
        validate_incomplete_stage_layout(directory, bulk_stage)
        stage_contract, _, _ = load_stage_contract(root, args.stage, contract)
        frozen_logical = stage_logical_payload(
            contract=contract,
            stage=args.stage,
            upstream_stages=expected,
            upstream_receipts=stage_contract["upstream_receipts"],
            inputs=stage_contract["inputs"],
            parameters=stage_contract["parameters"],
            parameters_json=stage_contract["parameters_json"],
        )
        if frozen_logical != requested_logical:
            raise ValueError(
                "stage resume rejected: parameters, inputs, or upstream receipt contract is not byte-identical"
            )
        if stage_contract.get("logical_contract_sha256") != stage_logical_digest(requested_logical):
            raise ValueError("stage resume rejected: frozen logical contract digest drift")
        resumed = True
    else:
        if bulk_stage.exists():
            raise FileExistsError(
                f"stage bulk-data directory exists without a metadata stage contract; inspect or use a new run: {bulk_stage}"
            )
        # Claim the metadata stage root first.  The root contract is immutable
        # before any attempt directory exists, so an interrupted first job can
        # later be resumed only via exact logical-contract equality.
        directory.mkdir(parents=True, exist_ok=False)
        initial_storage = stage_storage(root, data_root, args.stage, attempt_id)
        stage_contract = {
            **requested_logical,
            "logical_contract_sha256": stage_logical_digest(requested_logical),
            "attempt_id": attempt_id,
            "storage": initial_storage,
            "created_at": now(),
        }
        stage_contract["contract_sha256"] = contract_digest(stage_contract)
        atomic_json(directory / "stage_contract.json", stage_contract, no_replace=True)
        resumed = False

    attempt = metadata_attempt_root(root, args.stage, attempt_id)
    bulk_attempt = data_attempt_root(data_root, args.stage, attempt_id)
    if attempt.exists() or bulk_attempt.exists():
        raise FileExistsError(
            "attempt id already has immutable artifacts; every retry requires a fresh job-unique attempt id: "
            f"{attempt_id}"
        )
    attempt.mkdir(parents=True, exist_ok=False)
    attempt_contract = {
        "schema_version": STAGE_ATTEMPT_SCHEMA,
        "run_id": contract["run_id"],
        "stage": args.stage,
        "code_commit": contract["code_commit"],
        "run_contract_sha256": contract["contract_sha256"],
        "stage_contract_sha256": stage_contract["contract_sha256"],
        "logical_contract_sha256": stage_contract["logical_contract_sha256"],
        "attempt_id": attempt_id,
        "storage": stage_storage(root, data_root, args.stage, attempt_id),
        "created_at": now(),
    }
    attempt_contract["contract_sha256"] = contract_digest(attempt_contract)
    atomic_json(stage_attempt_contract_path(root, args.stage, attempt_id), attempt_contract, no_replace=True)
    # The attempt contract is immutable before the paired IOPS directory is
    # created.  A later retry gets a different directory instead of touching
    # this one if the job dies at any point after this boundary.
    bulk_attempt.mkdir(parents=True, exist_ok=False)
    print(
        json.dumps(
            {
                "ok": True,
                "resumed": resumed,
                "stage_dir": str(directory),
                "attempt_dir": str(attempt),
                "data_attempt_dir": str(bulk_attempt),
            }
        )
    )


def cmd_finish_stage(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    contract = load_valid_contract(root, run_id=args.run_id)
    supplied_or_contract_data_root(args, root, contract)
    directory = stage_root(root, args.stage)
    receipt_path = stage_receipt_path(root, args.stage)
    marker = directory / "COMPLETED"
    if receipt_path.exists() or marker.exists():
        raise FileExistsError(f"stage already has an immutable completion artifact: {directory}")
    stage_contract, _, _ = load_stage_contract(root, args.stage, contract)
    attempt_id = validate_attempt_id(args.attempt_id)
    attempt_contract, metadata_root, bulk_root = load_stage_attempt_contract(
        root, args.stage, contract, stage_contract, attempt_id
    )
    outputs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for value in args.output:
        output = Path(value).resolve()
        if output in seen:
            raise ValueError(f"duplicate stage output: {output}")
        seen.add(output)
        if not output_is_in_stage_attempt(output, metadata_root, bulk_root):
            raise ValueError(
                "stage output must remain in its job-unique metadata or bulk-data attempt directory: "
                f"{output}"
            )
        outputs.append(file_binding(output))
    if not outputs:
        raise ValueError("a stage receipt requires at least one output")
    receipt = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "status": "passed",
        "run_id": contract["run_id"],
        "stage": args.stage,
        "code_commit": contract["code_commit"],
        "run_contract_sha256": contract["contract_sha256"],
        "stage_contract_sha256": stage_contract.get("contract_sha256"),
        "attempt_id": attempt_id,
        "attempt_contract_sha256": attempt_contract.get("contract_sha256"),
        "successful_attempt": successful_attempt_binding(attempt_contract),
        "outputs": outputs,
        "completed_at": now(),
    }
    receipt["receipt_sha256"] = contract_digest(receipt)
    atomic_json(receipt_path, receipt, no_replace=True)
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(receipt_path)}  stage_receipt.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"ok": True, "receipt": str(receipt_path)}))


def cmd_validate_run(args: argparse.Namespace) -> None:
    contract = load_valid_contract(args.run_root.resolve(), run_id=args.run_id)
    graph = PRESTRUCTURAL_STAGES if contract["prestructural_only"] else STAGES
    for stage in graph:
        receipt = stage_receipt_path(args.run_root.resolve(), stage)
        if not receipt.exists():
            break
        load_stage_receipt(args.run_root.resolve(), stage, contract)
    print(json.dumps({"ok": True, "contract_sha256": contract["contract_sha256"]}))


def cmd_get_stage_output(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    contract = load_valid_contract(root, run_id=args.run_id)
    receipt = load_stage_receipt(root, args.stage, contract)
    matches = [
        str(item["path"])
        for item in receipt.get("outputs", [])
        if Path(str(item.get("path", ""))).name == args.basename
    ]
    if len(matches) != 1:
        raise ValueError(f"{args.stage}: expected exactly one output named {args.basename!r}, found {len(matches)}")
    print(matches[0])


def cmd_get_stage_attempt_dir(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    contract = load_valid_contract(root, run_id=args.run_id)
    # Context is only available for completed stages, so its paths are backed
    # by a receipt as well as the specific attempt that successfully finished
    # the stage.  In particular, never return an earlier failed attempt.
    receipt = load_stage_receipt(root, args.stage, contract)
    stage_contract, _, _ = load_stage_contract(root, args.stage, contract)
    _, metadata_root, bulk_root = load_stage_attempt_contract(
        root, args.stage, contract, stage_contract, str(receipt["attempt_id"])
    )
    print(str(metadata_root if args.storage == "metadata" else bulk_root))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze-run")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument(
        "--data-root",
        type=Path,
        help="immutable IOPS bulk-data root; defaults only from AGENT1_V3_DATA_ROOT",
    )
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--code-commit", required=True)
    freeze.add_argument("--source-registry", type=Path, required=True)
    freeze.add_argument("--source-aliases", type=Path, required=True)
    freeze.add_argument("--candidate-roster", type=Path, required=True)
    freeze.add_argument("--post-cutoff-inventory", type=Path, required=True)
    freeze.add_argument("--nanochat-initial-roster", type=Path, required=True)
    freeze.add_argument("--acquisition-receipt", type=Path, required=True)
    freeze.add_argument("--tokenizer", type=Path, required=True)
    freeze.add_argument("--review-policy", type=Path, required=True)
    freeze.add_argument("--review-prompt", type=Path, required=True)
    freeze.add_argument("--review-response-schema", type=Path, required=True)
    freeze.add_argument("--glossapi-build-receipt", type=Path, required=True)
    freeze.add_argument("--license-adjudication", type=Path, required=True)
    freeze.add_argument("--training-eligibility-policy", type=Path, required=True)
    freeze.add_argument("--dedup-policy", type=Path, required=True)
    freeze.add_argument("--greekmmlu-policy", type=Path, required=True)
    freeze.add_argument("--anonymization-policy", type=Path, required=True)
    freeze.add_argument("--structural-policy", type=Path, required=True)
    freeze.add_argument("--prestructural-only", action="store_true")
    freeze.set_defaults(func=cmd_freeze_run)

    begin = sub.add_parser("begin-stage")
    begin.add_argument("--run-root", type=Path, required=True)
    begin.add_argument("--data-root", type=Path)
    begin.add_argument("--run-id", required=True)
    begin.add_argument("--stage", choices=STAGES, required=True)
    begin.add_argument("--attempt-id", required=True)
    begin.add_argument("--input", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    begin.add_argument("--parameters-json", default="{}")
    begin.set_defaults(func=cmd_begin_stage)

    finish = sub.add_parser("finish-stage")
    finish.add_argument("--run-root", type=Path, required=True)
    finish.add_argument("--data-root", type=Path)
    finish.add_argument("--run-id", required=True)
    finish.add_argument("--stage", choices=STAGES, required=True)
    finish.add_argument("--attempt-id", required=True)
    finish.add_argument("--output", action="append", default=[], required=True)
    finish.set_defaults(func=cmd_finish_stage)

    validate = sub.add_parser("validate-run")
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.set_defaults(func=cmd_validate_run)
    output = sub.add_parser("get-stage-output")
    output.add_argument("--run-root", type=Path, required=True)
    output.add_argument("--run-id", required=True)
    output.add_argument("--stage", choices=STAGES, required=True)
    output.add_argument("--basename", required=True)
    output.set_defaults(func=cmd_get_stage_output)
    context = sub.add_parser("get-stage-attempt-dir")
    context.add_argument("--run-root", type=Path, required=True)
    context.add_argument("--run-id", required=True)
    context.add_argument("--stage", choices=STAGES, required=True)
    context.add_argument("--storage", choices=("metadata", "data"), required=True)
    context.set_defaults(func=cmd_get_stage_attempt_dir)
    return parser


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
