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
from pathlib import Path
from typing import Any

from build_source_review_packet import redact_direct_identifiers
from greek_pii import mask_greek_identifiers
from profile_dataset_quality_rust import (
    load_normalized_shards,
    metadata_flags,
    sha256_file,
    validate_file_receipt,
    write_json_atomic,
)


SAMPLE_SCHEMA = "dataset_review_complete_sample_v1"
RECEIPT_SCHEMA = "dataset_review_complete_sample_packet_receipt_v1"
CHECKPOINT_SCHEMA = "dataset_review_sample_export_shard_checkpoint_v1"
EXPORT_CONTRACT_SCHEMA = "dataset_review_sample_export_contract_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IPV6_RE = re.compile(
    r"(?i)(?<![0-9A-F:])(?:[0-9A-F]{1,4}:){2,7}[0-9A-F]{0,4}(?![0-9A-F:])"
)
IDENTITY_RE = re.compile(
    r"(?i)(?:Α\.?\s*Δ\.?\s*Τ\.?|ΑΔΤ|Δελτί(?:ο|ου)\s+Ταυτότητας|"
    r"αριθμ(?:ός|\.)?\s+(?:δελτίου\s+)?ταυτότητας|passport)\s*[:#-]?\s*"
    r"([A-ZΑ-Ω]{1,3}[\s-]?\d{5,10})\b"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
            row = json.loads(line)
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
    contract_sha256: str,
    shard_receipt: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    fragment = directory / "samples.jsonl"
    if not receipt_path.is_file() or not fragment.is_file():
        raise ValueError(f"incomplete sample-export checkpoint: {directory}")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{receipt_path}: checkpoint root must be an object")
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or value.get("status") != "passed"
        or value.get("contract_sha256") != contract_sha256
        or value.get("input_shard") != shard_receipt
        or int(value.get("rows_scanned", -1)) != int(shard_receipt["rows"])
    ):
        raise ValueError(f"{receipt_path}: sample-export checkpoint drift")
    output = value.get("output")
    if not isinstance(output, dict) or output.get("path") != "samples.jsonl":
        raise ValueError(f"{receipt_path}: invalid checkpoint output")
    if (
        int(output.get("bytes", -1)) != fragment.stat().st_size
        or str(output.get("sha256", "")) != sha256_file(fragment)
        or int(output.get("rows", -1))
        != len(
            [line for line in fragment.read_text(encoding="utf-8").splitlines() if line]
        )
    ):
        raise ValueError(f"{receipt_path}: checkpoint output drift")
    return value


def export_samples(args: argparse.Namespace) -> int:
    if args.output.resolve().parent != args.receipt.resolve().parent:
        raise ValueError(
            "sample packet and receipt must share a directory for relocatable receipts"
        )
    if args.output.exists() or args.receipt.exists():
        if args.resume and args.output.is_file() and not args.receipt.exists():
            # The packet rename precedes its receipt by a very small window.
            # An unreceipted generated packet is safe to reproduce from the
            # still-immutable canonical inputs and Stage30 request IDs.
            args.output.unlink()
        elif args.resume and args.receipt.exists() and not args.output.is_file():
            raise ValueError("sample receipt exists without its packet")
    if args.output.exists() or args.receipt.exists():
        if not args.resume or not args.output.is_file() or not args.receipt.is_file():
            raise FileExistsError("sample packet and receipt are immutable")
        completed = json.loads(args.receipt.read_text(encoding="utf-8"))
        if not isinstance(completed, dict):
            raise ValueError("completed sample packet receipt root must be an object")
        output = completed.get("output", {})
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
            or int(output.get("rows", -1))
            != sum(
                1
                for line in args.output.read_text(encoding="utf-8").splitlines()
                if line
            )
        ):
            raise ValueError("completed sample packet resume receipt drift")
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
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint_root_value = getattr(args, "checkpoint_dir", None)
    checkpoint_root = (
        Path(checkpoint_root_value).resolve()
        if checkpoint_root_value is not None
        else (args.output.parent / f".{args.output.name}.checkpoints").resolve()
    )
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"sample-export checkpoints exist; use --resume: {checkpoint_root}"
        )
    checkpoint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
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
    if contract_path.exists():
        current = json.loads(contract_path.read_text(encoding="utf-8"))
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
        validate_file_receipt(shard.path, shard_receipt, rows=shard.rows)
        input_inventory.append(shard_receipt)
        checkpoint_key = hashlib.sha256(
            canonical_json(shard_receipt).encode("utf-8")
        ).hexdigest()[:24]
        final = checkpoint_root / checkpoint_key
        if final.exists():
            checkpoint = validate_checkpoint(
                final,
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
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(mode=0o700)
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
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise

        fragment = final / "samples.jsonl"
        for line_number, line in enumerate(
            fragment.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line:
                continue
            row = json.loads(line)
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
            redaction_totals.update(row.get("redaction_counts", {}))
        checkpoint_inventory.append(
            {
                "input_shard_sha256": shard.sha256,
                "checkpoint_receipt_sha256": sha256_file(final / "receipt.json"),
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
        os.replace(temporary_name, args.output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "normalization_manifest": {
            "path": str(args.normalization_manifest.resolve()),
            "sha256": sha256_file(args.normalization_manifest),
        },
        "canonical_root": str(args.canonical_root.resolve()),
        "review_requests": {
            "path": str(args.review_requests.resolve()),
            "sha256": sha256_file(args.review_requests),
        },
        "export_contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
            "contract_sha256": contract_sha256,
        },
        "input_shards": input_inventory,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_inventory_sha256": hashlib.sha256(
            canonical_json(checkpoint_inventory).encode("utf-8")
        ).hexdigest(),
        "output": file_output(args.output, rows=len(found)),
        "redaction_totals": dict(sorted(redaction_totals.items())),
        "high_precision_identifier_patterns_masked": True,
    }
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return export_samples(args)


if __name__ == "__main__":
    raise SystemExit(main())
