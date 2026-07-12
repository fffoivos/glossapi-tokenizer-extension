#!/usr/bin/env python3
"""Small stdlib-only receipt contract for the Phase-04 Clariden DAG."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


RUN_SCHEMA = "full_cpt_pipeline_run_v1"
STAGE_SCHEMA = "full_cpt_pipeline_stage_receipt_v1"
HEX_SHA256 = frozenset("0123456789abcdef")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o644)
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


def cmd_init_run(args: argparse.Namespace) -> None:
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = {
        "schema_version": RUN_SCHEMA,
        "run_id": args.run_id,
        "run_root": str(root),
        "code_commit": args.code_commit,
        "sources": str(args.sources.resolve()),
        "sources_sha256": sha256(args.sources),
        "cleaning_policy": str(args.cleaning_policy.resolve()),
        "cleaning_policy_sha256": sha256(args.cleaning_policy),
        "eligibility_policy": str(args.eligibility_policy.resolve()),
        "eligibility_policy_sha256": sha256(args.eligibility_policy),
        "source_license_adjudication": str(
            args.source_license_adjudication.resolve()
        ),
        "source_license_adjudication_sha256": sha256(
            args.source_license_adjudication
        ),
        "tokenizer_sha256": args.tokenizer_sha256,
    }
    manifest = root / "run_manifest.json"
    if manifest.exists():
        current = read_json(manifest)
        for key, value in expected.items():
            if current.get(key) != value:
                raise ValueError(f"immutable run identity drift for {key}: {current.get(key)!r} != {value!r}")
        return
    value = {**expected, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        write_json_atomic(manifest, value, exclusive=True)
    except FileExistsError:
        cmd_init_run(args)


def stage_identity(args: argparse.Namespace) -> dict[str, str]:
    return {
        "stage": args.stage,
        "run_id": args.run_id,
        "code_commit": args.code_commit,
    }


def cmd_begin_stage(args: argparse.Namespace) -> None:
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    path = args.stage_dir / "stage_attempt.json"
    attempt = {
        "schema_version": "full_cpt_pipeline_stage_attempt_v1",
        **stage_identity(args),
        "job_id": args.job_id,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resume": path.exists(),
    }
    write_json_atomic(path, attempt)


def cmd_add_input(args: argparse.Namespace) -> None:
    inputs_path = args.stage_dir / "stage_inputs.json"
    current = read_json(inputs_path) if inputs_path.exists() else {
        "schema_version": "full_cpt_pipeline_stage_inputs_v1",
        "inputs": {},
        "parameters": {},
    }
    path = args.path.resolve()
    value = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    previous = current["inputs"].get(args.name)
    if previous is not None and previous != value:
        raise ValueError(f"resume input drift for {args.name}: {previous!r} != {value!r}")
    current["inputs"][args.name] = value
    write_json_atomic(inputs_path, current)


def cmd_bind_parameter(args: argparse.Namespace) -> None:
    """Bind a small invocation setting so incomplete-stage resume cannot drift."""

    inputs_path = args.stage_dir / "stage_inputs.json"
    current = read_json(inputs_path) if inputs_path.exists() else {
        "schema_version": "full_cpt_pipeline_stage_inputs_v1",
        "inputs": {},
        "parameters": {},
    }
    if current.get("schema_version") != "full_cpt_pipeline_stage_inputs_v1":
        raise ValueError(f"{inputs_path}: unsupported stage-input schema")
    parameters = current.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{inputs_path}: stage parameters must be an object")
    previous = parameters.get(args.name)
    if previous is not None and previous != args.value:
        raise ValueError(
            f"resume parameter drift for {args.name}: {previous!r} != {args.value!r}"
        )
    parameters[args.name] = args.value
    write_json_atomic(inputs_path, current)


def output_record(stage_dir: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(stage_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"declared output is outside its immutable stage directory: {resolved}") from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"required non-empty stage output missing: {resolved}")
    return {"path": relative.as_posix(), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def cmd_finish_stage(args: argparse.Namespace) -> None:
    receipt_path = args.stage_dir / "stage_receipt.json"
    if receipt_path.exists() or (args.stage_dir / "COMPLETED").exists():
        raise FileExistsError(f"refusing to overwrite completed stage: {args.stage_dir}")
    inputs_path = args.stage_dir / "stage_inputs.json"
    inputs = read_json(inputs_path) if inputs_path.exists() else {"inputs": {}}
    outputs = [output_record(args.stage_dir, path) for path in args.required_output]
    receipt = {
        "schema_version": STAGE_SCHEMA,
        "status": "passed",
        **stage_identity(args),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": inputs.get("inputs", {}),
        "parameters": inputs.get("parameters", {}),
        "outputs": outputs,
    }
    write_json_atomic(receipt_path, receipt, exclusive=True)
    completed = args.stage_dir / "COMPLETED"
    descriptor = os.open(completed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(sha256(receipt_path) + "  stage_receipt.json\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_receipt_outputs(args: argparse.Namespace) -> Path:
    receipt_path = args.stage_dir / "stage_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"stage receipt is missing: {args.stage_dir}")
    receipt = read_json(receipt_path)
    expected = {"schema_version": STAGE_SCHEMA, "status": "passed", **stage_identity(args)}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{receipt_path}: {key} mismatch")
    for name, item in receipt.get("inputs", {}).items():
        path = Path(str(item.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != int(item.get("bytes", -1))
            or sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"completed input drift for {name}: {path}")
    parameters = receipt.get("parameters", {})
    if not isinstance(parameters, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in parameters.items()
    ):
        raise ValueError(f"{receipt_path}: invalid bound stage parameters")
    inventory_cache: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for item in receipt.get("outputs", []):
        path = args.stage_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"completed output drift: {path}")
        if path.suffix == ".json":
            try:
                value = read_json(path)
            except (ValueError, json.JSONDecodeError):
                continue
            schema = value.get("schema_version")
            if schema == "full_cpt_shard_inventory_validation_v1":
                validate_inventory_validation(value, path=path, cache=inventory_cache)
            elif manifest_inventory_entries(value):
                key = str(path.resolve())
                if key not in inventory_cache:
                    inventory_cache[key] = validate_manifest_inventory(value, path=path)
    return receipt_path


def cmd_validate_stage(args: argparse.Namespace) -> None:
    receipt_path = validate_receipt_outputs(args)
    completed = args.stage_dir / "COMPLETED"
    if not completed.is_file():
        raise ValueError(f"stage completion marker is missing: {args.stage_dir}")
    marker = completed.read_text(encoding="utf-8").split()[0]
    if marker != sha256(receipt_path):
        raise ValueError(f"{completed}: receipt hash mismatch")


def cmd_repair_stage_marker(args: argparse.Namespace) -> None:
    receipt_path = validate_receipt_outputs(args)
    completed = args.stage_dir / "COMPLETED"
    if completed.exists():
        raise FileExistsError(f"completion marker already exists: {completed}")
    descriptor = os.open(completed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(sha256(receipt_path) + "  stage_receipt.json\n")
        handle.flush()
        os.fsync(handle.fileno())


def cmd_validate_admission(args: argparse.Namespace) -> None:
    value = read_json(args.path)
    actual_sha256 = sha256(args.path)
    if actual_sha256 != args.expected_sha256:
        raise ValueError(
            f"manual admission confirmation hash mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    schema = value.get("schema_version")
    rows = value.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source admission has no source decisions")
    if schema == "source_quality_review_admission_v1":
        if value.get("pending_adjudications") != 0:
            raise ValueError("source reviews still require adjudication")
        key = "source_dataset"
    elif schema == "full_cpt_source_admission_v1":
        if value.get("status") != "approved":
            raise ValueError("legacy source admission is not explicitly approved")
        key = "source_id"
    else:
        raise ValueError("unsupported source admission schema")
    seen: set[str] = set()
    for row in rows:
        identity = row.get(key) if isinstance(row, dict) else None
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"source admission has invalid/duplicate {key}")
        seen.add(identity)
        if row.get("decision") not in {"include", "include_after_cleaning", "quarantine", "exclude"}:
            raise ValueError(f"invalid source decision for {identity}")
        if args.require_terminal and row.get("decision") not in {"include", "quarantine", "exclude"}:
            raise ValueError(f"final admission is non-terminal for {identity}")


def cmd_validate_cleaning_replay(args: argparse.Namespace) -> None:
    reference_payload = read_json(args.reference_receipt)
    current_payload = read_json(args.current_inputs)
    if reference_payload.get("schema_version") != STAGE_SCHEMA:
        raise ValueError(f"{args.reference_receipt}: unsupported stage receipt schema")
    if reference_payload.get("status") != "passed":
        raise ValueError(f"{args.reference_receipt}: reviewed cleaning stage did not pass")
    if current_payload.get("schema_version") != "full_cpt_pipeline_stage_inputs_v1":
        raise ValueError(f"{args.current_inputs}: unsupported stage-input schema")
    reference = reference_payload.get("inputs", {})
    current = current_payload.get("inputs", {})
    if not isinstance(reference, dict) or not isinstance(current, dict):
        raise ValueError("cleaning replay inputs must be mappings")

    def relevant(values: dict[str, Any]) -> dict[str, Any]:
        fixed = {
            "tokenizer",
            "eligibility_policy",
            "source_config",
            "source_license_adjudication",
            "cleaning_policy",
        }
        prefixes = ("document_actions:", "structural_")
        return {
            key: value
            for key, value in values.items()
            if key in fixed or key.startswith(prefixes)
        }

    finalizer = bool(getattr(args, "finalizer", False))
    required_finalizer_inputs = {
        "tokenizer",
        "eligibility_policy",
        "source_config",
        "source_license_adjudication",
        "cleaning_policy",
    }
    if finalizer:
        missing_reference = required_finalizer_inputs - set(reference)
        missing_current = required_finalizer_inputs - set(current)
        if missing_reference or missing_current:
            raise ValueError(
                "final cleaning replay lacks required reviewed inputs: "
                f"reference_missing={sorted(missing_reference)} "
                f"current_missing={sorted(missing_current)}"
            )
        reference_relevant = {
            key: reference[key] for key in sorted(required_finalizer_inputs)
        }
        current_relevant = {
            key: current[key] for key in sorted(required_finalizer_inputs)
        }
    else:
        reference_relevant = relevant(reference)
        current_relevant = relevant(current)
    if reference_relevant != current_relevant:
        raise ValueError(
            "final cleaning inputs differ from the reviewed cleaning pass: "
            f"reference={sorted(reference_relevant)} current={sorted(current_relevant)}"
        )
    output = getattr(args, "output", None)
    if output is not None:
        result = {
            "schema_version": "full_cpt_cleaning_replay_validation_v1",
            "status": "passed",
            "mode": "structural_last_finalizer" if finalizer else "exact_replay",
            "reference_receipt": {
                "path": str(args.reference_receipt.resolve()),
                "bytes": args.reference_receipt.stat().st_size,
                "sha256": sha256(args.reference_receipt),
            },
            "current_inputs": {
                "path": str(args.current_inputs.resolve()),
                "bytes": args.current_inputs.stat().st_size,
                "sha256": sha256(args.current_inputs),
            },
            "reviewed_inputs": reference_relevant,
        }
        _write_or_validate_immutable_json(
            output,
            result,
            artifact_name="cleaning replay validation",
        )


def cmd_get_input_path(args: argparse.Namespace) -> None:
    receipt = read_json(args.receipt)
    value = receipt.get("inputs", {}).get(args.name)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{args.receipt}: missing recorded input {args.name!r}")
    path = Path(value["path"])
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"recorded input drift: {path}")
    print(path)


def cmd_get_json_field(args: argparse.Namespace) -> None:
    value: Any = read_json(args.path)
    for component in args.field.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"{args.path}: missing JSON field {args.field!r}")
        value = value[component]
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{args.path}: JSON field {args.field!r} is not scalar")
    print(str(value))


def inventory_entries(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            found.append(value)
        for nested in value.values():
            found.extend(inventory_entries(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(inventory_entries(nested))
    return found


def manifest_inventory_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    schema = manifest.get("schema_version")
    if schema == "full_cpt_cleaning_manifest_v1":
        result = []
        roots = {
            "output": Path(str(manifest["output"])),
            "ledger": Path(str(manifest["ledger"])),
            "quarantine": Path(str(manifest["quarantine"])),
        }
        for row in manifest.get("files", []):
            if not isinstance(row, dict):
                raise ValueError("cleaning manifest file row must be an object")
            for key in ("input", "input_ledger", "input_quarantine"):
                if isinstance(row.get(key), dict):
                    result.append(dict(row[key]))
            for key, root in roots.items():
                if not isinstance(row.get(key), dict):
                    raise ValueError(f"cleaning manifest file row lacks {key} receipt")
                receipt = dict(row[key])
                path = Path(str(receipt["path"]))
                receipt["path"] = str(path if path.is_absolute() else root / path)
                result.append(receipt)
        for receipt in manifest.get("document_action_receipts", []):
            if isinstance(receipt, dict):
                result.append(dict(receipt))
        span_inventory = manifest.get("span_inventory")
        if isinstance(span_inventory, dict):
            for receipt in span_inventory.get("span_files", []):
                if isinstance(receipt, dict):
                    result.append(dict(receipt))
        return result
    if schema == "full_cpt_greekmmlu_decontamination_v1":
        result = []
        for row in manifest.get("files", []):
            for key in ("output", "dropped", "ledger"):
                receipt = dict(row[key])
                path = Path(str(receipt["path"]))
                if not path.is_absolute():
                    path = Path(str(manifest[key])) / path
                receipt["path"] = str(path)
                result.append(receipt)
        return result
    if schema == "full_cpt_release_manifest_v1":
        result = []
        root = Path(str(manifest["output"]))
        for row in manifest.get("files", []):
            for key in ("training", "redistribution"):
                receipt = dict(row[key])
                path = Path(str(receipt["path"]))
                receipt["path"] = str(path if path.is_absolute() else root / path)
                result.append(receipt)
        card = manifest.get("dataset_card")
        if not isinstance(card, dict):
            raise ValueError("release manifest lacks dataset-card receipt")
        card_receipt = dict(card)
        card_path = Path(str(card_receipt["path"]))
        card_receipt["path"] = str(card_path if card_path.is_absolute() else root / card_path)
        result.append(card_receipt)
        return result
    if schema == "full_cpt_release_validation_v1":
        result = []
        publication = manifest.get("publication_inventory")
        if not isinstance(publication, dict):
            raise ValueError("release validation lacks publication inventory")
        data_root = Path(str(publication.get("root", "")))
        for row in publication.get("files", []):
            receipt = dict(row)
            relative = Path(str(receipt["path"]))
            receipt["path"] = str(relative if relative.is_absolute() else data_root / relative)
            result.append(receipt)
        release_root = Path(str(manifest.get("release", "")))
        for row in manifest.get("publication_metadata_inventory", []):
            receipt = dict(row)
            relative = Path(str(receipt["path"]))
            receipt["path"] = str(relative if relative.is_absolute() else release_root / relative)
            result.append(receipt)
        return result
    if schema == "full_cpt_publication_receipt_v1":
        result = []
        public_root = Path(str(manifest.get("redistribution_root", "")))
        for row in manifest.get("local_inventory", []):
            if not isinstance(row, dict):
                raise ValueError("publication local inventory row must be an object")
            receipt = dict(row)
            local = Path(str(receipt["path"]))
            if not local.is_absolute():
                if not str(receipt.get("remote_path", "")).startswith("data/"):
                    raise ValueError("relative publication inventory path is not a data shard")
                local = public_root / local
            receipt["path"] = str(local)
            result.append(receipt)
        return result
    if schema == "full_cpt_dedup_wrapper_manifest_v1":
        result = []
        staged_root = Path(str(manifest["staged_input"]))
        for row in manifest.get("files", []):
            receipt = dict(row["staged"])
            path = Path(str(receipt["path"]))
            receipt["path"] = str(path if path.is_absolute() else staged_root / path)
            result.append(receipt)
        output = manifest.get("dedup_output")
        if isinstance(output, dict) and isinstance(output.get("decisions"), dict):
            result.append(dict(output["decisions"]))
        return result
    if schema == "phase04_structural_raw_predictions_v1":
        result = []
        for row in manifest.get("files", []):
            if not isinstance(row, dict):
                raise ValueError("structural raw-prediction file row must be an object")
            for key in ("receipt", "index", "counters", "spans"):
                value = row.get(key)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"structural raw-prediction file row lacks {key} receipt"
                    )
                result.append(dict(value))
        return result
    if schema == "phase04_structural_spans_manifest_v1":
        spans = manifest.get("spans")
        if not isinstance(spans, dict):
            raise ValueError("structural spans manifest lacks spans receipt")
        return [dict(spans)]
    return inventory_entries(manifest)


def validate_inventory_entries(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not entries:
        raise ValueError("manifest has no path/bytes/sha256 inventory entries")
    validated: list[dict[str, Any]] = []
    seen: dict[str, tuple[int, str]] = {}
    for entry in entries:
        path = Path(str(entry["path"])).resolve()
        expected = (int(entry["bytes"]), str(entry["sha256"]))
        previous = seen.get(str(path))
        if previous is not None:
            if previous != expected:
                raise ValueError(f"conflicting inventory entries for {path}")
            continue
        if not path.is_file() or path.stat().st_size != expected[0] or sha256(path) != expected[1]:
            raise ValueError(f"inventory verification failed: {path}")
        seen[str(path)] = expected
        validated.append({"path": str(path), "bytes": expected[0], "sha256": expected[1]})
    validated.sort(key=lambda row: row["path"])
    aggregate = hashlib.sha256(
        json.dumps(validated, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return validated, aggregate


def validate_manifest_inventory(manifest: dict[str, Any], *, path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        result = validate_inventory_entries(manifest_inventory_entries(manifest))
        if manifest.get("schema_version") == "full_cpt_normalization_manifest_v1":
            validate_normalization_parquet_tree(manifest, path=path)
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: attested shard inventory failed: {exc}") from exc


def normalization_shard_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("normalization manifest has no sources")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("normalization source entry must be an object")
        shards = source.get("shards")
        if not isinstance(shards, list):
            raise ValueError(
                f"normalization source {source.get('source_id')!r} has invalid shards"
            )
        for shard in shards:
            if not isinstance(shard, dict) or not {"path", "bytes", "sha256"} <= set(shard):
                raise ValueError("normalization shard lacks path/bytes/sha256")
            entries.append(dict(shard))
    return entries


def validate_normalization_parquet_tree(
    manifest: dict[str, Any],
    *,
    path: Path,
    expected_root: Path | None = None,
) -> tuple[list[dict[str, Any]], str, Path]:
    if manifest.get("schema_version") != "full_cpt_normalization_manifest_v1":
        raise ValueError("exact Parquet-tree validation is only valid for normalization manifests")
    root = Path(str(manifest.get("output", ""))).resolve()
    if expected_root is not None and root != expected_root.resolve():
        raise ValueError(
            f"normalization output root drift: manifest={root}, expected={expected_root.resolve()}"
        )
    if not root.is_dir():
        raise ValueError(f"normalization output root is missing: {root}")

    declared_entries = normalization_shard_entries(manifest)
    declared_paths: list[Path] = []
    for entry in declared_entries:
        shard = Path(str(entry["path"])).resolve()
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"normalization shard escapes output root: {shard}") from exc
        declared_paths.append(shard)
    if len(set(declared_paths)) != len(declared_paths):
        raise ValueError("normalization manifest declares a shard more than once")

    actual_paths = sorted(file.resolve() for file in root.rglob("*.parquet") if file.is_file())
    declared = set(declared_paths)
    actual = set(actual_paths)
    if declared != actual:
        missing = sorted(str(item) for item in declared - actual)
        unexpected = sorted(str(item) for item in actual - declared)
        raise ValueError(
            "normalization Parquet tree differs from its exact shard inventory; "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    validated, aggregate = validate_inventory_entries(declared_entries)
    return validated, aggregate, root


def validate_inventory_validation(
    value: dict[str, Any],
    *,
    path: Path,
    cache: dict[str, tuple[list[dict[str, Any]], str]] | None = None,
) -> None:
    manifest_path = Path(str(value.get("manifest", ""))).resolve()
    if not manifest_path.is_file() or sha256(manifest_path) != value.get("manifest_sha256"):
        raise ValueError(f"{path}: inventory validation manifest drift")
    cached = (cache or {}).get(str(manifest_path))
    if cached is None:
        manifest = read_json(manifest_path)
        validated, aggregate = validate_manifest_inventory(manifest, path=manifest_path)
        if cache is not None:
            cache[str(manifest_path)] = (validated, aggregate)
    else:
        validated, aggregate = cached
    if (
        int(value.get("files", -1)) != len(validated)
        or int(value.get("bytes", -1)) != sum(row["bytes"] for row in validated)
        or value.get("inventory_sha256") != aggregate
    ):
        raise ValueError(f"{path}: inventory validation aggregate drift")
    exact_claim = value.get("exact_parquet_tree")
    if exact_claim is not None:
        if not isinstance(exact_claim, dict):
            raise ValueError(f"{path}: exact Parquet-tree claim must be an object")
        manifest = read_json(manifest_path)
        exact, exact_aggregate, exact_root = validate_normalization_parquet_tree(
            manifest,
            path=manifest_path,
            expected_root=Path(str(exact_claim.get("root", ""))),
        )
        expected_exact = {
            "root": str(exact_root),
            "files": len(exact),
            "bytes": sum(row["bytes"] for row in exact),
            "inventory_sha256": exact_aggregate,
        }
        if exact_claim != expected_exact:
            raise ValueError(f"{path}: exact Parquet-tree aggregate drift")


def cmd_validate_inventory(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    entries = manifest_inventory_entries(manifest)
    validated, aggregate = validate_inventory_entries(entries)
    result = {
        "schema_version": "full_cpt_shard_inventory_validation_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "files": len(validated),
        "bytes": sum(row["bytes"] for row in validated),
        "inventory_sha256": aggregate,
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if args.exact_parquet_root is not None:
        exact, exact_aggregate, exact_root = validate_normalization_parquet_tree(
            manifest,
            path=args.manifest,
            expected_root=args.exact_parquet_root,
        )
        result["exact_parquet_tree"] = {
            "root": str(exact_root),
            "files": len(exact),
            "bytes": sum(row["bytes"] for row in exact),
            "inventory_sha256": exact_aggregate,
        }
    if args.output.exists():
        current = read_json(args.output)
        stable_keys = {key: value for key, value in result.items() if key != "validated_at"}
        current_stable = {key: value for key, value in current.items() if key != "validated_at"}
        if current_stable != stable_keys:
            raise ValueError(f"existing inventory validation drift: {args.output}")
        return
    write_json_atomic(args.output, result, exclusive=True)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - HEX_SHA256)


def _artifact_sha256(receipt: dict[str, Any], name: str) -> str:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("structural model receipt lacks artifacts")
    value = artifacts.get(name)
    if isinstance(value, dict):
        value = value.get("sha256")
    if not valid_sha256(value):
        raise ValueError(f"structural model receipt lacks valid {name} SHA-256")
    return str(value)


def _number(value: Any, *, name: str, lower: float = 0.0, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"structural safety metric {name} is unavailable/non-numeric")
    result = float(value)
    if not lower <= result <= upper:
        raise ValueError(f"structural safety metric {name} is outside [{lower}, {upper}]")
    return result


def _write_or_validate_immutable_json(
    path: Path,
    value: dict[str, Any],
    *,
    artifact_name: str,
    volatile_keys: frozenset[str] = frozenset(),
) -> None:
    """Create an immutable JSON artifact, or prove an existing one is identical.

    Stage resume must never reinterpret an existing decision under new operator
    inputs.  Timestamps are intentionally the only fields that may be ignored
    when validating a previously frozen artifact.
    """

    stable = {key: item for key, item in value.items() if key not in volatile_keys}
    if path.exists():
        current = read_json(path)
        current_stable = {
            key: item for key, item in current.items() if key not in volatile_keys
        }
        if current_stable != stable:
            raise ValueError(
                f"existing {artifact_name} drift on resume: {path}; "
                "use a new PIPELINE_RUN_ID for a different structural choice"
            )
        return
    write_json_atomic(path, value, exclusive=True)


def cmd_freeze_structural_request(args: argparse.Namespace) -> None:
    receipt_path: str | None = None
    receipt_sha256: str | None = None
    if args.requested_mode == "apply":
        if args.receipt is None:
            raise ValueError("requested structural apply requires an exact promoted receipt")
        if not args.receipt.is_file():
            raise ValueError(f"promoted structural receipt is missing: {args.receipt}")
        receipt_path = str(args.receipt.resolve())
        receipt_sha256 = sha256(args.receipt)
    elif args.receipt is not None:
        raise ValueError("requested structural no-op must not carry a model receipt")

    request = {
        "schema_version": "full_cpt_structural_finalization_request_v1",
        "requested_mode": args.requested_mode,
        "apply_structural_requested": args.requested_mode == "apply",
        "model_receipt": receipt_path,
        "model_receipt_sha256": receipt_sha256,
    }
    _write_or_validate_immutable_json(
        args.output,
        request,
        artifact_name="structural finalization request",
    )


def cmd_validate_structural_model(args: argparse.Namespace) -> None:
    stage50 = read_json(args.stage50_cleaning_manifest)
    stage50_sha = sha256(args.stage50_cleaning_manifest)
    expected_stage50 = {
        "schema_version": "full_cpt_cleaning_manifest_v1",
        "status": "completed",
        "cleaning_pass": "post_source_post_pii",
        "structural_applied": False,
    }
    for key, expected in expected_stage50.items():
        if stage50.get(key) != expected:
            raise ValueError(f"Stage 50 structural input has invalid {key}")
    policy = read_json(args.cleaning_policy)
    if policy.get("schema_version") != "full_cpt_cleaning_policy_v1":
        raise ValueError("unsupported cleaning policy schema")

    base = {
        "schema_version": "full_cpt_structural_application_decision_v1",
        "requested_mode": args.requested_mode,
        "apply_structural_requested": args.requested_mode == "apply",
        "stage50_cleaning_manifest": str(args.stage50_cleaning_manifest.resolve()),
        "stage50_cleaning_manifest_sha256": stage50_sha,
        "cleaning_policy": str(args.cleaning_policy.resolve()),
        "cleaning_policy_sha256": sha256(args.cleaning_policy),
    }
    if args.requested_mode == "no_op":
        if args.receipt is not None:
            raise ValueError("requested structural no-op must not carry a model receipt")
        decision = {
            **base,
            "status": "no_op",
            "apply_structural": False,
            "reason": "operator_selected_no_op",
            "model_receipt": None,
            "model_receipt_sha256": None,
            "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _write_or_validate_immutable_json(
            args.output,
            decision,
            artifact_name="structural application decision",
            volatile_keys=frozenset({"validated_at"}),
        )
        return

    if args.receipt is None:
        raise ValueError("requested structural apply requires an exact promoted receipt")

    receipt = read_json(args.receipt)
    receipt_sha = sha256(args.receipt)
    if receipt.get("schema_version") != "academic_structural_model_receipt_v1":
        raise ValueError("unsupported structural model receipt schema")
    if receipt.get("stage50_cleaning_manifest_sha256") != stage50_sha:
        raise ValueError("structural model receipt is not bound to exact Stage 50 text")
    if not isinstance(receipt.get("model_id"), str) or not receipt["model_id"]:
        raise ValueError("structural model receipt lacks model_id")
    artifacts = {
        name: _artifact_sha256(receipt, name)
        for name in ("code", "config", "checkpoint")
    }
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("annotation_status") != "LLM_silver":
        raise ValueError("model-selection evidence must be declared honestly as LLM_silver")
    if not valid_sha256(evidence.get("inventory_sha256")):
        raise ValueError("LLM-silver evidence lacks an inventory SHA-256")
    work_split = evidence.get("work_split")
    if not isinstance(work_split, dict):
        raise ValueError("structural evidence lacks work-level split receipt")
    if (
        work_split.get("leak_free") is not True
        or int(work_split.get("work_overlap_count", -1)) != 0
        or int(work_split.get("exact_text_overlap_count", -1)) != 0
        or not valid_sha256(work_split.get("split_manifest_sha256"))
    ):
        raise ValueError("structural evidence does not prove leak-free work/exact-text splits")
    coverage = evidence.get("task_coverage")
    if not isinstance(coverage, list) or set(coverage) != {"toc", "bibliography"}:
        raise ValueError("structural receipt must cover both ToC and bibliography")
    classifier_selection = evidence.get("classifier_selection_receipt")
    if (
        evidence.get("selected_architecture") != "c0-rust-lr-hysteresis"
        or not isinstance(classifier_selection, dict)
        or not valid_sha256(classifier_selection.get("sha256"))
        or not valid_sha256(evidence.get("joint_ladder_run_receipt_sha256"))
    ):
        raise ValueError(
            "structural receipt lacks a passed post-ladder C0 classifier selection"
        )

    safety = receipt.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("structural model receipt lacks safety section")
    metrics = safety.get("metrics")
    metrics_available = isinstance(metrics, dict) and all(
        metrics.get(name) is not None
        for name in (
            "running_prose_deletion_rate",
            "main_text_retention_rate",
            "catastrophic_document_deletion_rate",
        )
    )
    receipt_status = receipt.get("status")
    promotion_status = receipt.get("promotion_status")
    if receipt_status not in {"passed", "no_op", "failed"} or promotion_status not in {
        "passed",
        "no_op",
        "failed",
    }:
        raise ValueError("structural receipt has invalid status/promotion_status")
    if (receipt_status == "passed") != (promotion_status == "passed"):
        raise ValueError("structural receipt status and promotion_status disagree")
    claims_pass = receipt_status == promotion_status == "passed"
    if claims_pass and not metrics_available:
        raise ValueError("silver-only unavailable safety metrics cannot report a promotion pass")

    reason = "receipt_not_promoted"
    safety_summary: dict[str, Any] = {
        "evidence_status": safety.get("evidence_status"),
        "metrics_available": metrics_available,
    }
    eligible = False
    if metrics_available:
        if safety.get("evidence_status") != "targeted_manual_false_deletion_audit":
            if claims_pass:
                raise ValueError("promotion requires an independent targeted manual false-deletion audit")
            reason = "independent_safety_evidence_unavailable"
        else:
            if not valid_sha256(safety.get("audit_receipt_sha256")):
                raise ValueError("targeted manual safety evidence lacks its audit receipt SHA-256")
            gates = policy.get("structural", {}).get("application_gates")
            if not isinstance(gates, dict):
                raise ValueError("cleaning policy lacks structural.application_gates")
            reviewed = safety.get("reviewed_deletions")
            if not isinstance(reviewed, int) or reviewed < 0:
                raise ValueError("structural safety audit has invalid reviewed_deletions")
            running = _number(metrics["running_prose_deletion_rate"], name="running_prose_deletion_rate")
            retention = _number(metrics["main_text_retention_rate"], name="main_text_retention_rate")
            catastrophic = _number(
                metrics["catastrophic_document_deletion_rate"],
                name="catastrophic_document_deletion_rate",
            )
            metric_gate = (
                reviewed >= int(gates["minimum_reviewed_deletions"])
                and running <= float(gates["maximum_running_prose_deletion_rate"])
                and retention >= float(gates["minimum_main_text_retention_rate"])
                and catastrophic <= float(gates["maximum_catastrophic_document_deletion_rate"])
            )
            declared_gate = safety.get("status") == "passed"
            policy_enabled = (
                policy.get("status") == "approved"
                and policy.get("structural", {}).get("toc", {}).get("enabled_for_materialization") is True
                and policy.get("structural", {}).get("bibliography", {}).get("enabled_for_materialization") is True
            )
            validation = policy.get("validation", {})
            receipt_contract = (
                validation.get("structural_application_receipt_required") is True
                and validation.get("required_model_evidence") == "LLM_silver"
                and validation.get("required_safety_evidence")
                == "targeted_manual_false_deletion_audit"
            )
            eligible = bool(
                metric_gate and declared_gate and claims_pass and policy_enabled and receipt_contract
            )
            if not metric_gate:
                reason = "structural_safety_gate_failed"
            elif not declared_gate or not claims_pass:
                reason = "receipt_not_promoted"
            elif not policy_enabled or not receipt_contract:
                reason = "tracked_cleaning_policy_not_approved"
            else:
                reason = "all_structural_application_gates_passed"
            safety_summary.update(
                {
                    "reviewed_deletions": reviewed,
                    "audit_receipt_sha256": safety["audit_receipt_sha256"],
                    "metrics": {
                        "running_prose_deletion_rate": running,
                        "main_text_retention_rate": retention,
                        "catastrophic_document_deletion_rate": catastrophic,
                    },
                    "policy_gates": gates,
                    "metric_gate_passed": metric_gate,
                }
            )
    elif safety.get("status") == "passed":
        raise ValueError("safety status cannot pass with unavailable metrics")
    else:
        reason = "safety_metrics_unavailable"

    decision = {
        **base,
        "status": "passed" if eligible else "no_op",
        "apply_structural": eligible,
        "reason": reason,
        "model_id": receipt["model_id"],
        "model_receipt": str(args.receipt.resolve()),
        "model_receipt_sha256": receipt_sha,
        "model_selection_evidence": "LLM_silver",
        "artifacts": artifacts,
        "work_split": work_split,
        "safety": safety_summary,
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if not eligible:
        raise ValueError(
            f"requested structural apply is not eligible: {reason}; "
            "select the explicit no-op finalization path or fix/promote the evidence"
        )
    _write_or_validate_immutable_json(
        args.output,
        decision,
        artifact_name="structural application decision",
        volatile_keys=frozenset({"validated_at"}),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-run")
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--code-commit", required=True)
    init.add_argument("--sources", type=Path, required=True)
    init.add_argument("--cleaning-policy", type=Path, required=True)
    init.add_argument("--eligibility-policy", type=Path, required=True)
    init.add_argument("--source-license-adjudication", type=Path, required=True)
    init.add_argument("--tokenizer-sha256", required=True)
    init.set_defaults(func=cmd_init_run)

    for name, function in (
        ("begin-stage", cmd_begin_stage),
        ("finish-stage", cmd_finish_stage),
        ("validate-stage", cmd_validate_stage),
        ("repair-stage-marker", cmd_repair_stage_marker),
    ):
        command = commands.add_parser(name)
        command.add_argument("--stage-dir", type=Path, required=True)
        command.add_argument("--stage", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--code-commit", required=True)
        if name == "begin-stage":
            command.add_argument("--job-id", required=True)
        elif name == "finish-stage":
            command.add_argument("--required-output", action="append", type=Path, default=[])
        command.set_defaults(func=function)

    add_input = commands.add_parser("add-input")
    add_input.add_argument("--stage-dir", type=Path, required=True)
    add_input.add_argument("--name", required=True)
    add_input.add_argument("--path", type=Path, required=True)
    add_input.set_defaults(func=cmd_add_input)

    bind_parameter = commands.add_parser("bind-parameter")
    bind_parameter.add_argument("--stage-dir", type=Path, required=True)
    bind_parameter.add_argument("--name", required=True)
    bind_parameter.add_argument("--value", required=True)
    bind_parameter.set_defaults(func=cmd_bind_parameter)

    admission = commands.add_parser("validate-admission")
    admission.add_argument("--path", type=Path, required=True)
    admission.add_argument("--expected-sha256", required=True)
    admission.add_argument("--require-terminal", action="store_true")
    admission.set_defaults(func=cmd_validate_admission)

    replay = commands.add_parser("validate-cleaning-replay")
    replay.add_argument("--reference-receipt", type=Path, required=True)
    replay.add_argument("--current-inputs", type=Path, required=True)
    replay.add_argument("--finalizer", action="store_true")
    replay.add_argument("--output", type=Path)
    replay.set_defaults(func=cmd_validate_cleaning_replay)

    get_input = commands.add_parser("get-input-path")
    get_input.add_argument("--receipt", type=Path, required=True)
    get_input.add_argument("--name", required=True)
    get_input.set_defaults(func=cmd_get_input_path)

    get_field = commands.add_parser("get-json-field")
    get_field.add_argument("--path", type=Path, required=True)
    get_field.add_argument("--field", required=True)
    get_field.set_defaults(func=cmd_get_json_field)

    inventory = commands.add_parser("validate-inventory")
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--exact-parquet-root", type=Path)
    inventory.set_defaults(func=cmd_validate_inventory)

    structural_request = commands.add_parser("freeze-structural-request")
    structural_request.add_argument(
        "--requested-mode", choices=("no_op", "apply"), required=True
    )
    structural_request.add_argument("--receipt", type=Path)
    structural_request.add_argument("--output", type=Path, required=True)
    structural_request.set_defaults(func=cmd_freeze_structural_request)

    structural = commands.add_parser("validate-structural-model")
    structural.add_argument("--receipt", type=Path)
    structural.add_argument(
        "--requested-mode", choices=("no_op", "apply"), required=True
    )
    structural.add_argument("--stage50-cleaning-manifest", type=Path, required=True)
    structural.add_argument("--cleaning-policy", type=Path, required=True)
    structural.add_argument("--output", type=Path, required=True)
    structural.set_defaults(func=cmd_validate_structural_model)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
