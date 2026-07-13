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


def contract_digest(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("created_at", None)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_valid_contract(run_root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    path = run_contract_path(run_root)
    contract = read_object(path)
    if contract.get("schema_version") != RUN_SCHEMA:
        raise ValueError(f"{path}: unsupported run contract")
    if run_id is not None and contract.get("run_id") != run_id:
        raise ValueError(f"{path}: run_id mismatch")
    if contract.get("contract_sha256") != contract_digest(contract):
        raise ValueError(f"{path}: contract hash mismatch")
    for name, binding in contract.get("inputs", {}).items():
        verify_binding(str(name), binding)
    return contract


def cmd_freeze_run(args: argparse.Namespace) -> None:
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError(
            "run id must be agent1-full-corpus-v3-<UTC YYYYmmddTHHMMSSZ>-<git short sha>"
        )
    root = args.run_root.resolve()
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
            ("acquisition_receipt", args.acquisition_receipt),
            ("tokenizer", args.tokenizer),
            ("review_policy", args.review_policy),
            ("review_prompt", args.review_prompt),
            ("review_response_schema", args.review_response_schema),
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


def load_stage_receipt(run_root: Path, stage: str, contract: dict[str, Any]) -> dict[str, Any]:
    path = stage_receipt_path(run_root, stage)
    if not path.is_file():
        raise ValueError(f"upstream stage receipt is missing: {stage}: {path}")
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
    for output in receipt.get("outputs", []):
        verify_binding(f"{stage}:output", output)
    return receipt


def cmd_begin_stage(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    contract = load_valid_contract(root, run_id=args.run_id)
    expected = expected_upstream(args.stage, prestructural_only=bool(contract["prestructural_only"]))
    for upstream in expected:
        load_stage_receipt(root, upstream, contract)
    directory = stage_root(root, args.stage)
    if directory.exists():
        raise FileExistsError(f"stage directory already exists; inspect or use a new run: {directory}")
    attempt = directory / "attempts" / args.attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise ValueError("--parameters-json must be an object")
    inputs = named_bindings((name, Path(path)) for name, path in args.input)
    stage_contract = {
        "schema_version": STAGE_SCHEMA,
        "run_id": contract["run_id"],
        "stage": args.stage,
        "code_commit": contract["code_commit"],
        "run_contract_sha256": contract["contract_sha256"],
        "attempt_id": args.attempt_id,
        "upstream_stages": list(expected),
        "inputs": inputs,
        "parameters": parameters,
        "created_at": now(),
    }
    stage_contract["contract_sha256"] = contract_digest(stage_contract)
    atomic_json(directory / "stage_contract.json", stage_contract, no_replace=True)
    print(json.dumps({"ok": True, "stage_dir": str(directory), "attempt_dir": str(attempt)}))


def cmd_finish_stage(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    contract = load_valid_contract(root, run_id=args.run_id)
    directory = stage_root(root, args.stage)
    stage_contract = read_object(directory / "stage_contract.json")
    if stage_contract.get("run_contract_sha256") != contract["contract_sha256"]:
        raise ValueError("stage contract does not belong to the frozen run")
    if stage_contract.get("attempt_id") != args.attempt_id:
        raise ValueError("attempt id differs from the frozen stage contract")
    expected_root = (directory / "attempts" / args.attempt_id).resolve()
    outputs: list[dict[str, Any]] = []
    for value in args.output:
        output = Path(value).resolve()
        try:
            output.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError(f"stage output must remain in its job-unique attempt directory: {output}") from exc
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
        "attempt_id": args.attempt_id,
        "outputs": outputs,
        "completed_at": now(),
    }
    receipt["receipt_sha256"] = contract_digest(receipt)
    receipt_path = directory / "stage_receipt.json"
    atomic_json(receipt_path, receipt, no_replace=True)
    marker = directory / "COMPLETED"
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


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze-run")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--code-commit", required=True)
    freeze.add_argument("--source-registry", type=Path, required=True)
    freeze.add_argument("--source-aliases", type=Path, required=True)
    freeze.add_argument("--candidate-roster", type=Path, required=True)
    freeze.add_argument("--acquisition-receipt", type=Path, required=True)
    freeze.add_argument("--tokenizer", type=Path, required=True)
    freeze.add_argument("--review-policy", type=Path, required=True)
    freeze.add_argument("--review-prompt", type=Path, required=True)
    freeze.add_argument("--review-response-schema", type=Path, required=True)
    freeze.add_argument("--dedup-policy", type=Path, required=True)
    freeze.add_argument("--greekmmlu-policy", type=Path, required=True)
    freeze.add_argument("--anonymization-policy", type=Path, required=True)
    freeze.add_argument("--structural-policy", type=Path, required=True)
    freeze.add_argument("--prestructural-only", action="store_true")
    freeze.set_defaults(func=cmd_freeze_run)

    begin = sub.add_parser("begin-stage")
    begin.add_argument("--run-root", type=Path, required=True)
    begin.add_argument("--run-id", required=True)
    begin.add_argument("--stage", choices=STAGES, required=True)
    begin.add_argument("--attempt-id", required=True)
    begin.add_argument("--input", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    begin.add_argument("--parameters-json", default="{}")
    begin.set_defaults(func=cmd_begin_stage)

    finish = sub.add_parser("finish-stage")
    finish.add_argument("--run-root", type=Path, required=True)
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
    return parser


def main() -> int:
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
