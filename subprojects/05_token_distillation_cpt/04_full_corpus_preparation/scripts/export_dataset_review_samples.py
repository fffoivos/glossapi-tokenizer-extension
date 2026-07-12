#!/usr/bin/env python3
"""Export complete review documents with high-precision identifier masking.

This is the privacy bridge between Clariden and the local static review site.
Only sample IDs already selected by ``build_source_review_packet.py`` are
exported.  Source text remains plain text inside a private JSONL packet and is
never inserted into HTML by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from build_source_review_packet import redact_direct_identifiers
from greek_pii import mask_greek_identifiers
from profile_dataset_quality_rust import (
    lexical_absolute,
    load_normalized_shards,
    load_relative_bytes_nofollow,
    load_relative_json_nofollow,
    metadata_flags,
    normalization_dependency_receipt_paths,
    normalization_identity_closure,
    prepare_secure_directory,
    read_json,
    require_exact_keys,
    require_nonnegative_int,
    require_sha256,
    safe_relative_path,
    secure_directory_under_root,
    sha256_file,
    sha256_json,
    snapshot_inputs,
    strict_json_loads,
    verify_input_snapshots,
    write_json_atomic,
)


SAMPLE_SCHEMA = "dataset_review_complete_sample_v1"
RECEIPT_SCHEMA = "dataset_review_complete_sample_packet_receipt_v1"
CHECKPOINT_SCHEMA = "dataset_review_sample_export_shard_checkpoint_v1"
EXPORT_CONTRACT_SCHEMA = "dataset_review_sample_export_contract_v1"
SITE_ATTESTATION_SCHEMA = "dataset_review_complete_sample_site_attestation_v1"
REDACTION_COUNT_KEYS = frozenset(
    {"email", "ipv4", "iban", "afm", "amka", "phone", "url", "ipv6", "identity"}
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"(?i)(?<![\w@])(?:https?://|www\.)[^\s<>\"']+")
IPV6_RE = re.compile(
    r"(?i)(?<![0-9A-F:])(?:[0-9A-F]{1,4}:){2,7}[0-9A-F]{0,4}(?![0-9A-F:])"
)
IDENTITY_RE = re.compile(
    r"(?i)(?:Α\.?\s*Δ\.?\s*Τ\.?|ΑΔΤ|Δελτί(?:ο|ου)\s+Ταυτότητας|"
    r"αριθμ(?:ός|\.)?\s+(?:δελτίου\s+)?ταυτότητας|passport)\s*[:#-]?\s*"
    r"([A-ZΑ-Ω]{1,3}[\s-]?\d{5,10})\b"
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_redaction_counts(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: redaction counts must be an object")
    result: dict[str, int] = {}
    for name, count in value.items():
        if (
            name not in REDACTION_COUNT_KEYS
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError(f"{context}: invalid redaction count {name!r}={count!r}")
        result[str(name)] = count
    return dict(sorted(result.items()))


def display_document_id(value: str) -> str:
    return hashlib.sha256(
        f"dataset-review-display-id-v1\0{value}".encode("utf-8")
    ).hexdigest()[:16]


def load_primary_requests(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = strict_json_loads(line, context=f"{path}:{line_number}")
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != "source_quality_review_request_v1"
            ):
                raise ValueError(f"{path}:{line_number}: unsupported review request")
            if row.get("reviewer_slot") != "primary":
                continue
            sample_id = str(row.get("sample_id", ""))
            source = row.get("source")
            if not SHA256_RE.fullmatch(sample_id) or not isinstance(source, dict):
                raise ValueError(f"{path}:{line_number}: invalid sample identity")
            if sample_id in result:
                raise ValueError(f"{path}:{line_number}: duplicate primary sample")
            result[sample_id] = {
                "source_id": str(source.get("source_id", "")),
                "source_repo_id": str(source.get("source_repo_id", "")),
                "source_revision": str(source.get("source_revision", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "source_doc_id": str(source.get("source_doc_id", "")),
            }
    if not result:
        raise ValueError(f"{path}: no primary review samples")
    return result


def redact_complete_text(text: str) -> tuple[str, dict[str, int]]:
    # Mask the URL as one span before the generic identifier passes.  In
    # particular, this prevents credentials, query values, or fragments from
    # surviving because an embedded email/phone pattern changed the URL first.
    text, urls = URL_RE.subn("[REDACTED_URL]", text)
    text, first = redact_direct_identifiers(text)
    text, greek = mask_greek_identifiers(text)
    text, ipv6 = IPV6_RE.subn("[REDACTED_IPV6]", text)

    identity_count = 0

    def replace_identity(match: re.Match[str]) -> str:
        nonlocal identity_count
        identity_count += 1
        full = match.group(0)
        value = match.group(1)
        return full.replace(value, "[REDACTED_IDENTITY]")

    text = IDENTITY_RE.sub(replace_identity, text)
    counts: Counter[str] = Counter(first)
    counts.update(greek)
    if urls:
        counts["url"] += urls
    if ipv6:
        counts["ipv6"] += ipv6
    if identity_count:
        counts["identity"] += identity_count
    return text, dict(sorted(counts.items()))


def resolve_receipt_output(receipt_path: Path, value: Any) -> Path:
    declared = Path(str(value))
    if not declared.is_absolute():
        declared = receipt_path.resolve().parent / declared
    return declared.resolve()


def file_output(path: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
    }


def validate_checkpoint(
    directory: Path,
    *,
    checkpoint_root: Path,
    contract_sha256: str,
    shard_receipt: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_root = secure_directory_under_root(
        checkpoint_root,
        root=checkpoint_root,
        context="sample-export checkpoint root",
    )
    directory = secure_directory_under_root(
        directory,
        root=checkpoint_root,
        context="sample-export checkpoint",
    )
    directory_relative = directory.relative_to(checkpoint_root)
    receipt_path, value, _, _ = load_relative_json_nofollow(
        checkpoint_root,
        (directory_relative / "receipt.json").as_posix(),
        context="sample-export checkpoint receipt",
    )
    require_exact_keys(
        value,
        required=(
            "schema_version",
            "status",
            "contract_sha256",
            "input_shard",
            "rows_scanned",
            "redaction_totals",
            "output",
        ),
        context=f"{receipt_path}: checkpoint",
    )
    require_sha256(value["contract_sha256"], context=f"{receipt_path}: contract_sha256")
    input_shard = value["input_shard"]
    if not isinstance(input_shard, Mapping):
        raise ValueError(f"{receipt_path}: input_shard must be an object")
    require_exact_keys(
        input_shard,
        required=("source_id", "path", "bytes", "sha256", "rows"),
        context=f"{receipt_path}: input_shard",
    )
    if not isinstance(input_shard["source_id"], str) or not input_shard["source_id"]:
        raise ValueError(f"{receipt_path}: invalid input_shard source_id")
    safe_relative_path(input_shard["path"], context=f"{receipt_path}: input_shard.path")
    if (
        require_nonnegative_int(
            input_shard["bytes"], context=f"{receipt_path}: input_shard.bytes"
        )
        < 1
        or require_nonnegative_int(
            input_shard["rows"], context=f"{receipt_path}: input_shard.rows"
        )
        < 1
    ):
        raise ValueError(f"{receipt_path}: invalid input_shard size")
    require_sha256(input_shard["sha256"], context=f"{receipt_path}: input_shard.sha256")
    rows_scanned = require_nonnegative_int(
        value["rows_scanned"], context=f"{receipt_path}: rows_scanned"
    )
    redaction_totals = value["redaction_totals"]
    if not isinstance(redaction_totals, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for name, count in redaction_totals.items()
    ):
        raise ValueError(f"{receipt_path}: invalid redaction_totals")
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or value.get("status") != "passed"
        or value.get("contract_sha256") != contract_sha256
        or input_shard != shard_receipt
        or rows_scanned != shard_receipt["rows"]
    ):
        raise ValueError(f"{receipt_path}: sample-export checkpoint drift")
    output = value.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"{receipt_path}: checkpoint output must be an object")
    require_exact_keys(
        output,
        required=("path", "bytes", "sha256", "rows"),
        context=f"{receipt_path}: checkpoint output",
    )
    output_bytes = require_nonnegative_int(
        output["bytes"], context=f"{receipt_path}: output.bytes"
    )
    output_rows = require_nonnegative_int(
        output["rows"], context=f"{receipt_path}: output.rows"
    )
    require_sha256(output["sha256"], context=f"{receipt_path}: output.sha256")
    if output.get("path") != "samples.jsonl":
        raise ValueError(f"{receipt_path}: invalid checkpoint output")
    fragment, fragment_bytes, fragment_size, fragment_sha256 = (
        load_relative_bytes_nofollow(
            checkpoint_root,
            (directory_relative / "samples.jsonl").as_posix(),
            context="sample-export checkpoint output",
        )
    )
    try:
        fragment_text = fragment_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{fragment}: checkpoint output is not UTF-8") from exc
    if (
        output_bytes != fragment_size
        or str(output.get("sha256", "")) != fragment_sha256
        or output_rows != len([line for line in fragment_text.splitlines() if line])
    ):
        raise ValueError(f"{receipt_path}: checkpoint output drift")
    validate_redaction_counts(
        value.get("redaction_totals"), context=f"{receipt_path}.redaction_totals"
    )
    return value


def portable_receipt(
    path: Path,
    *,
    root: Path | None = None,
    label: str | None = None,
    rows: int | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if label is not None:
        relative = label
    elif root is not None:
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"handoff dependency is outside the packet directory: {resolved}"
            ) from exc
    else:
        raise ValueError("portable receipt requires root or label")
    result: dict[str, Any] = {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def build_site_attestation(
    *,
    packet: Path,
    receipt_root: Path,
    review_requests: Path,
    normalization_manifest: Path,
    contract_path: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    checkpoint_root: Path,
    input_shards: list[dict[str, Any]],
    checkpoint_inventory: list[dict[str, Any]],
    primary_sample_ids: set[str],
    redaction_totals: Mapping[str, int],
) -> dict[str, Any]:
    checkpoint_root = secure_directory_under_root(
        checkpoint_root,
        root=checkpoint_root,
        context="sample-export checkpoint root",
    )
    contract_relative = contract_path.relative_to(checkpoint_root).as_posix()
    contract_path, stored_contract, contract_bytes, contract_file_sha256 = (
        load_relative_json_nofollow(
            checkpoint_root,
            contract_relative,
            context="sample-export contract",
        )
    )
    if stored_contract != contract:
        raise ValueError("sample export stored contract drift")
    contract_portable_receipt = {
        "path": contract_path.relative_to(receipt_root).as_posix(),
        "bytes": contract_bytes,
        "sha256": contract_file_sha256,
    }
    normalization = normalization_identity_closure(normalization_manifest)
    normalized_shards = normalization.pop("_normalized_shards")
    normalized_by_identity = {
        (str(row["source_id"]), str(row["path"]), str(row["sha256"])): row
        for row in normalized_shards
    }
    if any(
        normalized_by_identity.get(
            (str(row["source_id"]), str(row["path"]), str(row["sha256"]))
        )
        != row
        for row in input_shards
    ):
        raise ValueError(
            "sample attestation input shard is absent from normalization manifest"
        )
    if contract.get("schema_version") != EXPORT_CONTRACT_SCHEMA:
        raise ValueError("sample export contract schema drift")
    if (
        contract.get("normalization_manifest_sha256")
        != sha256_file(normalization_manifest)
        or contract.get("review_requests_sha256") != sha256_file(review_requests)
        or contract.get("selected_sample_count") != len(primary_sample_ids)
        or contract.get("redaction_pipeline") != "high_precision_identifier_patterns_v1"
        or sha256_json(contract) != contract_sha256
    ):
        raise ValueError("sample export contract dependency drift")

    checkpoint_closure: list[dict[str, Any]] = []
    checkpoint_redactions: Counter[str] = Counter()
    selected_rows = 0
    if len(checkpoint_inventory) != len(input_shards):
        raise ValueError("sample checkpoint/input-shard coverage drift")
    for shard, inventory in zip(input_shards, checkpoint_inventory, strict=True):
        checkpoint_key = hashlib.sha256(
            canonical_json(shard).encode("utf-8")
        ).hexdigest()[:24]
        directory = checkpoint_root / checkpoint_key
        checkpoint = validate_checkpoint(
            directory,
            checkpoint_root=checkpoint_root,
            contract_sha256=contract_sha256,
            shard_receipt=shard,
        )
        _, _, _, checkpoint_receipt_sha256 = load_relative_json_nofollow(
            checkpoint_root,
            (directory.relative_to(checkpoint_root) / "receipt.json").as_posix(),
            context="sample-export checkpoint receipt",
        )
        expected_inventory = {
            "input_shard_sha256": str(shard["sha256"]),
            "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
            "output_sha256": str(checkpoint["output"]["sha256"]),
            "selected_rows": int(checkpoint["output"]["rows"]),
        }
        if inventory != expected_inventory:
            raise ValueError("sample checkpoint inventory semantic drift")
        selected_rows += int(checkpoint["output"]["rows"])
        checkpoint_redactions.update(
            validate_redaction_counts(
                checkpoint["redaction_totals"],
                context=f"{directory / 'receipt.json'}.redaction_totals",
            )
        )
        checkpoint_closure.append(
            {
                "input_shard_sha256": str(shard["sha256"]),
                "receipt_sha256": expected_inventory["checkpoint_receipt_sha256"],
                "output_sha256": expected_inventory["output_sha256"],
                "rows_scanned": int(checkpoint["rows_scanned"]),
                "selected_rows": int(checkpoint["output"]["rows"]),
                "redaction_totals_sha256": sha256_json(checkpoint["redaction_totals"]),
            }
        )
    packet_rows = sum(
        1 for line in packet.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if selected_rows != packet_rows or selected_rows != len(primary_sample_ids):
        raise ValueError("sample checkpoint/packet/request coverage drift")
    validated_redaction_totals = validate_redaction_counts(
        redaction_totals, context="sample site attestation redaction totals"
    )
    if dict(sorted(checkpoint_redactions.items())) != validated_redaction_totals:
        raise ValueError("sample checkpoint redaction totals drift")

    return {
        "schema_version": SITE_ATTESTATION_SCHEMA,
        "status": "passed",
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "packet": portable_receipt(packet, root=receipt_root, rows=packet_rows),
        "review_requests": portable_receipt(
            review_requests, label=review_requests.name
        ),
        "primary_sample_count": len(primary_sample_ids),
        "primary_sample_id_inventory_sha256": sha256_json(sorted(primary_sample_ids)),
        "normalization": {
            **normalization,
            "input_shards": input_shards,
            "input_shard_inventory_sha256": sha256_json(input_shards),
        },
        "export_contract": {
            "receipt": contract_portable_receipt,
            "canonical_sha256": contract_sha256,
            "value": contract,
        },
        "checkpoint_closure": {
            "count": len(checkpoint_closure),
            "selected_rows": selected_rows,
            "inventory_sha256": sha256_json(checkpoint_inventory),
            "receipt_closure_sha256": sha256_json(checkpoint_closure),
            "checkpoint_text_outputs_rehashed_for_attestation": True,
        },
        "masking": {
            "pipeline": "high_precision_identifier_patterns_v1",
            "implementation_sha256": {
                "exporter": str(contract["exporter_script_sha256"]),
                **dict(contract["redaction_dependency_sha256"]),
            },
            "high_precision_identifier_patterns_masked": True,
            "private_data_true_rows": 0,
            "redaction_totals": validated_redaction_totals,
            "redaction_totals_sha256": sha256_json(validated_redaction_totals),
        },
    }


def export_samples(args: argparse.Namespace) -> int:
    receipt_root = prepare_secure_directory(
        args.receipt.parent, context="sample packet directory"
    )
    if (
        args.output.resolve().parent != receipt_root
        or args.site_attestation.resolve().parent != receipt_root
    ):
        raise ValueError(
            "sample packet, attestation, and receipt must share a directory"
        )
    # Snapshot the manifest before parsing it to discover the rest of the
    # receipt closure.  The final verification therefore catches both content
    # drift and a dependency-list swap during discovery.
    input_snapshots = snapshot_inputs(
        (
            args.normalization_manifest,
            args.review_requests,
            Path(__file__).resolve(),
            Path(redact_direct_identifiers.__code__.co_filename).resolve(),
            Path(mask_greek_identifiers.__code__.co_filename).resolve(),
            Path(metadata_flags.__code__.co_filename).resolve(),
        )
    )
    input_snapshots.update(
        snapshot_inputs(
            normalization_dependency_receipt_paths(args.normalization_manifest)
        )
    )
    if args.output.exists() or args.receipt.exists() or args.site_attestation.exists():
        if args.resume and args.output.is_file() and not args.receipt.exists():
            # The packet rename precedes its receipt by a very small window.
            # An unreceipted generated packet is safe to reproduce from the
            # still-immutable canonical inputs and Stage30 request IDs.
            args.output.unlink()
            args.site_attestation.unlink(missing_ok=True)
        elif args.resume and args.receipt.exists() and not args.output.is_file():
            raise ValueError("sample receipt exists without its packet")
    if args.output.exists() or args.receipt.exists() or args.site_attestation.exists():
        if (
            not args.resume
            or not args.output.is_file()
            or not args.receipt.is_file()
            or not args.site_attestation.is_file()
        ):
            raise FileExistsError("sample packet and receipt are immutable")
        input_snapshots.update(
            snapshot_inputs((args.output, args.receipt, args.site_attestation))
        )
        completed = read_json(args.receipt)
        if not isinstance(completed, dict):
            raise ValueError("completed sample packet receipt root must be an object")
        output = completed.get("output", {})
        site_receipt = completed.get("site_attestation", {})
        declared_attestation = resolve_receipt_output(
            args.receipt, site_receipt.get("path", "")
        )
        if (
            completed.get("schema_version") != RECEIPT_SCHEMA
            or completed.get("status") != "passed"
            or completed.get("high_precision_identifier_patterns_masked") is not True
            or resolve_receipt_output(args.receipt, output.get("path", ""))
            != args.output.resolve()
            or int(output.get("bytes", -1)) != args.output.stat().st_size
            or str(output.get("sha256", "")) != sha256_file(args.output)
            or completed.get("normalization_manifest", {}).get("sha256")
            != sha256_file(args.normalization_manifest)
            or completed.get("review_requests", {}).get("sha256")
            != sha256_file(args.review_requests)
            or declared_attestation != args.site_attestation.resolve()
            or int(site_receipt.get("bytes", -1))
            != args.site_attestation.stat().st_size
            or site_receipt.get("sha256") != sha256_file(args.site_attestation)
            or int(output.get("rows", -1))
            != sum(
                1
                for line in args.output.read_text(encoding="utf-8").splitlines()
                if line
            )
        ):
            raise ValueError("completed sample packet resume receipt drift")
        attestation = read_json(args.site_attestation)
        contract = attestation.get("export_contract", {}).get("value", {})
        expected_dependencies = {
            "build_source_review_packet": sha256_file(
                Path(redact_direct_identifiers.__code__.co_filename).resolve()
            ),
            "greek_pii": sha256_file(
                Path(mask_greek_identifiers.__code__.co_filename).resolve()
            ),
            "profile_dataset_quality_rust": sha256_file(
                Path(metadata_flags.__code__.co_filename).resolve()
            ),
        }
        if (
            attestation.get("schema_version") != SITE_ATTESTATION_SCHEMA
            or attestation.get("status") != "passed"
            or attestation.get("packet") != output
            or attestation.get("review_requests", {}).get("sha256")
            != sha256_file(args.review_requests)
            or attestation.get("normalization", {}).get("manifest", {}).get("sha256")
            != sha256_file(args.normalization_manifest)
            or contract.get("exporter_script_sha256")
            != sha256_file(Path(__file__).resolve())
            or contract.get("redaction_dependency_sha256") != expected_dependencies
            or attestation.get("masking", {}).get("implementation_sha256")
            != {
                "exporter": contract.get("exporter_script_sha256"),
                **expected_dependencies,
            }
            or attestation.get("checkpoint_closure", {}).get(
                "checkpoint_text_outputs_rehashed_for_attestation"
            )
            is not True
        ):
            raise ValueError("completed sample packet attestation drift")
        verify_input_snapshots(input_snapshots)
        print(
            canonical_json(
                {
                    "ok": True,
                    "already_complete": True,
                    "samples": int(output["rows"]),
                    "output": str(args.output),
                }
            )
        )
        return 0
    requested = load_primary_requests(args.review_requests)
    _, shards, _ = load_normalized_shards(
        args.normalization_manifest,
        args.canonical_root,
        include_source_ids=set(),
        include_base=False,
    )
    canonical_shard_snapshots = snapshot_inputs(shard.path for shard in shards)
    for shard in shards:
        snapshot = canonical_shard_snapshots[shard.path.resolve()]
        if snapshot.bytes != shard.bytes or snapshot.sha256 != shard.sha256:
            raise ValueError(f"canonical shard receipt drift: {shard.path}")
    input_snapshots.update(canonical_shard_snapshots)
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint_root_value = getattr(args, "checkpoint_dir", None)
    checkpoint_candidate = lexical_absolute(
        Path(checkpoint_root_value)
        if checkpoint_root_value is not None
        else args.output.parent / f".{args.output.name}.checkpoints"
    )
    try:
        checkpoint_candidate.relative_to(receipt_root)
    except ValueError as exc:
        raise ValueError(
            "checkpoint directory must be below the packet directory"
        ) from exc
    checkpoint_root = prepare_secure_directory(
        checkpoint_candidate, context="sample-export checkpoint root"
    )
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"sample-export checkpoints exist; use --resume: {checkpoint_root}"
        )
    checkpoint_root.chmod(0o700)
    contract = {
        "schema_version": EXPORT_CONTRACT_SCHEMA,
        "normalization_manifest_sha256": sha256_file(args.normalization_manifest),
        "review_requests_sha256": sha256_file(args.review_requests),
        "exporter_script_sha256": sha256_file(Path(__file__).resolve()),
        "redaction_dependency_sha256": {
            "build_source_review_packet": sha256_file(
                Path(redact_direct_identifiers.__code__.co_filename).resolve()
            ),
            "greek_pii": sha256_file(
                Path(mask_greek_identifiers.__code__.co_filename).resolve()
            ),
            "profile_dataset_quality_rust": sha256_file(
                Path(metadata_flags.__code__.co_filename).resolve()
            ),
        },
        "redaction_pipeline": "high_precision_identifier_patterns_v1",
        "batch_size": args.batch_size,
        "selected_sample_count": len(requested),
    }
    contract_sha256 = hashlib.sha256(
        canonical_json(contract).encode("utf-8")
    ).hexdigest()
    contract_path = checkpoint_root / "contract.json"
    if contract_path.exists() or contract_path.is_symlink():
        _, current, _, _ = load_relative_json_nofollow(
            checkpoint_root,
            "contract.json",
            context="sample-export contract",
        )
        if current != contract:
            raise ValueError(f"{contract_path}: sample-export resume contract drift")
    else:
        write_json_atomic(contract_path, contract, immutable=True)
        contract_path.chmod(0o600)
    found: set[str] = set()
    input_inventory: list[dict[str, Any]] = []
    checkpoint_inventory: list[dict[str, Any]] = []
    redaction_totals: Counter[str] = Counter()
    samples_by_id: dict[str, str] = {}
    import pyarrow.parquet as pq

    columns = [
        "source_id",
        "stable_uid",
        "source_repo_id",
        "source_revision",
        "source_dataset",
        "source_doc_id",
        "normalized_text_sha256",
        "source_metadata_json",
        "text",
    ]
    for shard in shards:
        shard_receipt = shard.receipt()
        input_inventory.append(shard_receipt)
        checkpoint_key = hashlib.sha256(
            canonical_json(shard_receipt).encode("utf-8")
        ).hexdigest()[:24]
        final = checkpoint_root / checkpoint_key
        if final.exists() or final.is_symlink():
            checkpoint = validate_checkpoint(
                final,
                checkpoint_root=checkpoint_root,
                contract_sha256=contract_sha256,
                shard_receipt=shard_receipt,
            )
        else:
            parquet = pq.ParquetFile(shard.path)
            missing_columns = sorted(set(columns) - set(parquet.schema_arrow.names))
            if missing_columns:
                raise ValueError(
                    f"{shard.path}: missing canonical columns {missing_columns}"
                )
            partial = checkpoint_root / f".{checkpoint_key}.partial-{os.getpid()}"
            if partial.exists() or partial.is_symlink():
                secure_directory_under_root(
                    partial,
                    root=checkpoint_root,
                    context="partial sample-export checkpoint",
                )
                shutil.rmtree(partial)
            prepare_secure_directory(
                partial, context="partial sample-export checkpoint"
            )
            selected_rows: list[dict[str, Any]] = []
            shard_redactions: Counter[str] = Counter()
            row_start = 0
            try:
                for batch in parquet.iter_batches(
                    batch_size=args.batch_size, columns=columns, use_threads=False
                ):
                    values = batch.to_pydict()
                    for index, raw_uid in enumerate(values["stable_uid"]):
                        uid = str(raw_uid)
                        expected = requested.get(uid)
                        if expected is None:
                            continue
                        actual = {
                            "source_id": str(values["source_id"][index]),
                            "source_repo_id": str(values["source_repo_id"][index]),
                            "source_revision": str(values["source_revision"][index]),
                            "source_dataset": str(values["source_dataset"][index]),
                            "source_doc_id": str(values["source_doc_id"][index]),
                        }
                        if actual != expected:
                            raise ValueError(
                                f"{uid}: review request/canonical source identity drift"
                            )
                        private, corrected = metadata_flags(
                            values["source_metadata_json"][index]
                        )
                        if private:
                            raise ValueError(
                                f"{uid}: selected review sample has privateData=true"
                            )
                        text = (
                            ""
                            if values["text"][index] is None
                            else str(values["text"][index])
                        )
                        normalized_sha256 = str(values["normalized_text_sha256"][index])
                        if (
                            not SHA256_RE.fullmatch(normalized_sha256)
                            or hashlib.sha256(text.encode("utf-8")).hexdigest()
                            != normalized_sha256
                        ):
                            raise ValueError(
                                f"{uid}: canonical normalized text hash drift"
                            )
                        redacted, redactions = redact_complete_text(text)
                        shard_redactions.update(redactions)
                        selected_rows.append(
                            {
                                "schema_version": SAMPLE_SCHEMA,
                                "sample_id": uid,
                                **{
                                    key: value
                                    for key, value in actual.items()
                                    if key != "source_doc_id"
                                },
                                "display_document_id": display_document_id(
                                    actual["source_doc_id"]
                                ),
                                "normalized_text_sha256": str(normalized_sha256),
                                "profile_text_sha256": hashlib.sha256(
                                    redacted.encode("utf-8")
                                ).hexdigest(),
                                "profile_text_variant": (
                                    "high_precision_identifier_masked_review_sample"
                                ),
                                "input_shard_path": shard.relative_path,
                                "input_shard_sha256": shard.sha256,
                                "input_row_index": row_start + index,
                                "private_data_true": False,
                                "corrected_version_present": corrected,
                                "high_precision_identifier_patterns_masked": True,
                                "redaction_counts": redactions,
                                "text": redacted,
                            }
                        )
                    row_start += batch.num_rows
                if row_start != shard.rows:
                    raise ValueError(
                        f"{shard.path}: scanned {row_start} rows, receipt declares {shard.rows}"
                    )
                selected_rows.sort(key=lambda row: str(row["sample_id"]))
                fragment = partial / "samples.jsonl"
                fragment.write_text(
                    "".join(canonical_json(row) + "\n" for row in selected_rows),
                    encoding="utf-8",
                )
                fragment.chmod(0o600)
                checkpoint = {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "status": "passed",
                    "contract_sha256": contract_sha256,
                    "input_shard": shard_receipt,
                    "rows_scanned": row_start,
                    "redaction_totals": dict(sorted(shard_redactions.items())),
                    "output": file_output(fragment, rows=len(selected_rows)),
                }
                write_json_atomic(partial / "receipt.json", checkpoint, immutable=True)
                (partial / "receipt.json").chmod(0o600)
                os.replace(partial, final)
                checkpoint = validate_checkpoint(
                    final,
                    checkpoint_root=checkpoint_root,
                    contract_sha256=contract_sha256,
                    shard_receipt=shard_receipt,
                )
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise

        checkpoint_relative = final.relative_to(checkpoint_root)
        fragment, fragment_bytes, _, _ = load_relative_bytes_nofollow(
            checkpoint_root,
            (checkpoint_relative / "samples.jsonl").as_posix(),
            context="sample-export checkpoint output",
        )
        try:
            fragment_text = fragment_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{fragment}: checkpoint output is not UTF-8") from exc
        for line_number, line in enumerate(fragment_text.splitlines(), 1):
            if not line:
                continue
            row = strict_json_loads(line, context=f"{fragment}:{line_number}")
            uid = str(row.get("sample_id", ""))
            expected = requested.get(uid)
            if expected is None or uid in found:
                raise ValueError(
                    f"{fragment}:{line_number}: unknown or duplicate selected sample"
                )
            if (
                row.get("schema_version") != SAMPLE_SCHEMA
                or row.get("high_precision_identifier_patterns_masked") is not True
                or row.get("private_data_true") is not False
                or hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()
                != row.get("profile_text_sha256")
            ):
                raise ValueError(f"{fragment}:{line_number}: invalid checkpoint sample")
            for key in (
                "source_id",
                "source_repo_id",
                "source_revision",
                "source_dataset",
            ):
                if str(row.get(key, "")) != expected[key]:
                    raise ValueError(f"{fragment}:{line_number}: sample identity drift")
            samples_by_id[uid] = canonical_json(row) + "\n"
            found.add(uid)
            redaction_totals.update(
                validate_redaction_counts(
                    row.get("redaction_counts"),
                    context=f"{fragment}:{line_number}.redaction_counts",
                )
            )
        _, _, _, checkpoint_receipt_sha256 = load_relative_json_nofollow(
            checkpoint_root,
            (checkpoint_relative / "receipt.json").as_posix(),
            context="sample-export checkpoint receipt",
        )
        checkpoint_inventory.append(
            {
                "input_shard_sha256": shard.sha256,
                "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
                "output_sha256": str(checkpoint["output"]["sha256"]),
                "selected_rows": int(checkpoint["output"]["rows"]),
            }
        )

    missing = set(requested) - found
    if missing:
        raise ValueError(
            f"canonical corpus lacks {len(missing)} selected samples: {sorted(missing)[:20]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_handle:
            for uid in sorted(found):
                output_handle.write(samples_by_id[uid])
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.chmod(temporary_name, 0o600)
        verify_input_snapshots(input_snapshots)
        os.replace(temporary_name, args.output)
        # The masked packet is now the receipt-bound result of the canonical
        # read.  Attestation generation revalidates checkpoints and the compact
        # normalization closure, so avoid rehashing the full corpus again.
        for path in canonical_shard_snapshots:
            input_snapshots.pop(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    attestation = build_site_attestation(
        packet=args.output,
        receipt_root=receipt_root,
        review_requests=args.review_requests,
        normalization_manifest=args.normalization_manifest,
        contract_path=contract_path,
        contract=contract,
        contract_sha256=contract_sha256,
        checkpoint_root=checkpoint_root,
        input_shards=input_inventory,
        checkpoint_inventory=checkpoint_inventory,
        primary_sample_ids=found,
        redaction_totals=redaction_totals,
    )
    verify_input_snapshots(input_snapshots)
    write_json_atomic(args.site_attestation, attestation, immutable=True)
    args.site_attestation.chmod(0o600)
    _, stored_contract, contract_bytes, contract_file_sha256 = (
        load_relative_json_nofollow(
            checkpoint_root,
            contract_path.relative_to(checkpoint_root).as_posix(),
            context="sample-export contract",
        )
    )
    if stored_contract != contract:
        raise ValueError("sample export stored contract drift")
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "normalization_manifest": {
            "path": "normalization_manifest.json",
            "bytes": args.normalization_manifest.stat().st_size,
            "sha256": sha256_file(args.normalization_manifest),
        },
        "canonical_root": str(args.canonical_root.resolve()),
        "review_requests": {
            "path": args.review_requests.name,
            "bytes": args.review_requests.stat().st_size,
            "sha256": sha256_file(args.review_requests),
        },
        "export_contract": {
            "path": contract_path.relative_to(receipt_root).as_posix(),
            "bytes": contract_bytes,
            "sha256": contract_file_sha256,
            "contract_sha256": contract_sha256,
        },
        "site_attestation": portable_receipt(args.site_attestation, root=receipt_root),
        "input_shards": input_inventory,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_inventory_sha256": hashlib.sha256(
            canonical_json(checkpoint_inventory).encode("utf-8")
        ).hexdigest(),
        "output": file_output(args.output, rows=len(found)),
        "redaction_totals": validate_redaction_counts(
            redaction_totals, context="sample packet redaction totals"
        ),
        "high_precision_identifier_patterns_masked": True,
    }
    verify_input_snapshots(input_snapshots)
    write_json_atomic(args.receipt, payload, immutable=True)
    args.receipt.chmod(0o600)
    print(
        canonical_json({"ok": True, "samples": len(found), "output": str(args.output)})
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization-manifest", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--review-requests", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="persistent per-shard masked export checkpoints (default: beside output)",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--site-attestation", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return export_samples(args)


if __name__ == "__main__":
    raise SystemExit(main())
