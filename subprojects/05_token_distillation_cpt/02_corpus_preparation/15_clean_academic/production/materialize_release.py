#!/usr/bin/env python3
"""Materialize and independently validate a bibliography-cleaned release candidate."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    APPLY_SUMMARY_SCHEMA,
    RECEIPT_SCHEMA,
    atomic_write_json,
    load_json,
    require_schema,
    sha256_file,
    validate_plan,
)

RECONSTRUCTION_SCHEMA = "bibliography-cleaning-release-reconstruction-v1"
SIZE_COLUMNS = ("chars", "non_whitespace_chars", "utf8_bytes", "approx_word_count")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def _file_receipt(path: Path, *, root: Path, rows: int | None = None) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        receipt["rows"] = rows
    return receipt


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as error:
        raise RuntimeError(
            f"release reconstruction requires same-filesystem hardlinks: {source}"
        ) from error


def _clean_from_spans(text: str, spans: list[dict[str, Any]]) -> str:
    removed_lines: set[int] = set()
    lines = text.split("\n")
    for span in spans:
        start = int(span["line_start"])
        end = int(span["line_end"])
        if start < 0 or end <= start or end > len(lines):
            raise ValueError(f"invalid removal span [{start}, {end})")
        removed_lines.update(range(start, end))
        removed = "\n".join(lines[start:end])
        if (
            hashlib.sha256(removed.encode("utf-8")).hexdigest()
            != span["removed_sha256"]
        ):
            raise ValueError("removal-span content hash mismatch")
    return "\n".join(
        line for index, line in enumerate(lines) if index not in removed_lines
    )


def _expected_size(text: str, column: str) -> int:
    return {
        "chars": len(text),
        "non_whitespace_chars": sum(not char.isspace() for char in text),
        "utf8_bytes": len(text.encode("utf-8")),
        "approx_word_count": len(WORD_RE.findall(text)),
    }[column]


def _validate_fragment(
    source: pq.ParquetFile,
    unit: dict[str, Any],
    receipt: dict[str, Any],
) -> pa.Table:
    output = receipt.get("output")
    ledger = receipt.get("ledger")
    if not isinstance(output, dict) or not isinstance(ledger, dict):
        raise TypeError(f"{unit['unit_id']}: missing output or ledger receipt")
    output_path = Path(output["path"])
    ledger_path = Path(ledger["path"])
    if (
        output_path.is_symlink()
        or ledger_path.is_symlink()
        or sha256_file(output_path) != output["sha256"]
        or sha256_file(ledger_path) != ledger["sha256"]
    ):
        raise ValueError(f"{unit['unit_id']}: fragment or ledger identity failed")

    groups = list(range(unit["row_group_start"], unit["row_group_end"]))
    source_table = pa.concat_tables(
        [source.read_row_group(group) for group in groups], promote_options="none"
    )
    fragment = pq.read_table(output_path)
    ledger_table = pq.read_table(ledger_path)
    if (
        fragment.schema != source.schema_arrow
        or fragment.num_rows != unit["rows"]
        or source_table.num_rows != unit["rows"]
        or ledger_table.num_rows != unit["rows"]
        or int(output["rows"]) != unit["rows"]
        or int(ledger["rows"]) != unit["rows"]
    ):
        raise ValueError(f"{unit['unit_id']}: row/schema closure failed")

    mutable = {"text", *SIZE_COLUMNS}
    for name in source.schema_arrow.names:
        if name not in mutable and not source_table[name].equals(fragment[name]):
            raise ValueError(f"{unit['unit_id']}: nonmutable column changed: {name}")

    source_rows = source_table.select(
        [name for name in ("text", *SIZE_COLUMNS) if name in source_table.column_names]
    ).to_pylist()
    fragment_rows = fragment.select(
        [name for name in ("text", *SIZE_COLUMNS) if name in fragment.column_names]
    ).to_pylist()
    ledger_rows = ledger_table.to_pylist()
    expected_coordinates = [
        (group, row)
        for group in groups
        for row in range(source.metadata.row_group(group).num_rows)
    ]
    for index, (before, after, ledger_row, coordinate) in enumerate(
        zip(source_rows, fragment_rows, ledger_rows, expected_coordinates, strict=True)
    ):
        original = before["text"] or ""
        cleaned = after["text"] or ""
        if (ledger_row["row_group"], ledger_row["row_in_group"]) != coordinate:
            raise ValueError(
                f"{unit['unit_id']}: ledger coordinate drift at row {index}"
            )
        if (
            hashlib.sha256(original.encode("utf-8")).hexdigest()
            != ledger_row["original_sha256"]
        ):
            raise ValueError(
                f"{unit['unit_id']}: original text hash drift at row {index}"
            )
        if _clean_from_spans(original, ledger_row["spans"]) != cleaned:
            raise ValueError(
                f"{unit['unit_id']}: cleaned text is not ledger-derived at row {index}"
            )
        if (
            int(ledger_row["chars_before"]) != len(original)
            or int(ledger_row["chars_after"]) != len(cleaned)
            or int(ledger_row["total_chars_removed"]) != len(original) - len(cleaned)
            or bool(ledger_row["would_empty"])
            or (bool(original.strip()) and not bool(cleaned.strip()))
        ):
            raise ValueError(
                f"{unit['unit_id']}: text accounting failed at row {index}"
            )
        for column in SIZE_COLUMNS:
            if column not in before:
                continue
            if (before[column] is None) != (after[column] is None):
                raise ValueError(
                    f"{unit['unit_id']}: {column} null mask changed at row {index}"
                )
            if before[column] is not None and after[column] != _expected_size(
                cleaned, column
            ):
                raise ValueError(
                    f"{unit['unit_id']}: {column} recomputation failed at row {index}"
                )
    return fragment


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract, plan, contract_sha, plan_sha = validate_plan(args.contract, args.plan)
    if contract["mode"] != "apply":
        raise ValueError("release materialization requires an apply contract")
    summary_path = Path(args.summary).resolve()
    summary = load_json(summary_path)
    require_schema(summary, APPLY_SUMMARY_SCHEMA, summary_path)
    if (
        summary.get("status") != "passed"
        or summary.get("contract_sha256") != contract_sha
        or summary.get("plan_sha256") != plan_sha
        or int(summary["overall"]["would_empty"]) != 0
    ):
        raise ValueError("apply summary is not passed and bound to this contract/plan")

    source_root = Path(contract["release_root"]).resolve()
    source_manifest_path = source_root / "manifests" / "deduplicated_manifest.json"
    source_manifest = load_json(source_manifest_path)
    preflight_manifest = contract["evidence"]["preflight"]
    preflight = load_json(preflight_manifest["path"])
    if (
        preflight["manifest"]["sha256"] != sha256_file(source_manifest_path)
        or len(source_manifest["files"]) != 431
    ):
        raise ValueError("source release manifest differs from the passed preflight")

    output_root = Path(args.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.building-", dir=output_root.parent
        )
    )

    apply_units: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for unit in plan["units"]:
        if unit["apply"]:
            apply_units[int(unit["rank"])].append(unit)
    receipt_dir = Path(contract["run_root"]) / "apply" / "receipts"
    expected_receipts = {
        unit["unit_id"] for units in apply_units.values() for unit in units
    }
    actual_receipts = {path.stem for path in receipt_dir.glob("*.json")}
    if actual_receipts != expected_receipts:
        raise ValueError("apply receipt set changed after aggregation")

    output_files: list[dict[str, Any]] = []
    untouched_ranks: list[int] = []
    transformed_ranks: list[int] = []
    validation = {"documents": 0, "fragments": 0}
    try:
        for source_entry in sorted(
            source_manifest["files"], key=lambda row: int(row["rank"])
        ):
            rank = int(source_entry["rank"])
            source_path = source_root / source_entry["path"]
            if (
                source_path.is_symlink()
                or source_path.stat().st_size != int(source_entry["bytes"])
                or sha256_file(source_path) != source_entry["sha256"]
            ):
                raise ValueError(f"source shard drift: rank {rank}")
            destination = temporary / source_entry["path"]
            units = sorted(
                apply_units.get(rank, []), key=lambda row: row["row_group_start"]
            )
            if not units:
                _hardlink(source_path, destination)
                untouched_ranks.append(rank)
            else:
                parquet = pq.ParquetFile(source_path)
                coverage = [
                    group
                    for unit in units
                    for group in range(unit["row_group_start"], unit["row_group_end"])
                ]
                if coverage != list(range(parquet.metadata.num_row_groups)):
                    raise ValueError(
                        f"rank {rank}: apply units do not cover every row group exactly"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    destination, parquet.schema_arrow, compression="zstd"
                )
                try:
                    for unit in units:
                        receipt_path = receipt_dir / f"{unit['unit_id']}.json"
                        receipt = load_json(receipt_path)
                        require_schema(receipt, RECEIPT_SCHEMA, receipt_path)
                        if (
                            receipt.get("status") != "passed"
                            or receipt.get("mode") != "apply"
                            or receipt.get("contract_sha256") != contract_sha
                            or receipt.get("plan_sha256") != plan_sha
                            or receipt.get("unit_id") != unit["unit_id"]
                            or summary["receipt_sha256"].get(unit["unit_id"])
                            != sha256_file(receipt_path)
                            or summary["ledger_sha256"].get(unit["unit_id"])
                            != receipt["ledger"]["sha256"]
                        ):
                            raise ValueError(
                                f"receipt binding failed: {unit['unit_id']}"
                            )
                        fragment = _validate_fragment(parquet, unit, receipt)
                        writer.write_table(fragment)
                        validation["documents"] += fragment.num_rows
                        validation["fragments"] += 1
                finally:
                    writer.close()
                transformed_ranks.append(rank)
            rows = pq.ParquetFile(destination).metadata.num_rows
            output_files.append(
                {
                    "rank": rank,
                    "origin": source_entry["origin"],
                    **_file_receipt(destination, root=temporary, rows=rows),
                }
            )

        if (
            len(output_files) != 431
            or sum(int(row["rows"]) for row in output_files)
            != int(source_manifest["rows"])
            or validation["documents"] != int(plan["apply_rows"])
            or len(untouched_ranks) + len(transformed_ranks) != 431
        ):
            raise ValueError("whole-release row/file accounting failed")
        source_by_rank = {
            int(row["rank"]): row for row in source_manifest["files"]
        }
        output_by_rank = {int(row["rank"]): row for row in output_files}
        for rank in untouched_ranks:
            old = source_by_rank[rank]
            new = output_by_rank[rank]
            if (new["sha256"], new["bytes"], new["rows"]) != (
                old["sha256"],
                old["bytes"],
                old["rows"],
            ):
                raise ValueError(f"untouched rank {rank} is not checksum-identical")

        decision_source = source_root / source_manifest["decision_ledger"]["path"]
        decision_destination = temporary / "manifests" / "dedup_decision_ledger.parquet"
        _hardlink(decision_source, decision_destination)
        inventory_path = temporary / "manifests" / "deduplicated_inventory.parquet"
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(output_files), inventory_path, compression="zstd"
        )
        reconstruction_path = (
            temporary / "manifests" / "bibliography_reconstruction.json"
        )
        reconstruction = {
            "schema_version": RECONSTRUCTION_SCHEMA,
            "status": "passed",
            "publication_ready": False,
            "source_release": str(source_root),
            "source_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256_file(source_manifest_path),
            },
            "contract": {
                "path": str(Path(args.contract).resolve()),
                "sha256": contract_sha,
            },
            "plan": {"path": str(Path(args.plan).resolve()), "sha256": plan_sha},
            "apply_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "files": len(output_files),
            "rows": sum(int(row["rows"]) for row in output_files),
            "transformed_ranks": transformed_ranks,
            "untouched_ranks": untouched_ranks,
            "transformed_documents": validation["documents"],
            "validated_fragments": validation["fragments"],
        }
        atomic_write_json(reconstruction_path, reconstruction)
        manifest = {
            **source_manifest,
            "created_at": args.created_at,
            "root": str(output_root.resolve()),
            "repository_id": args.repo_id,
            "private_only": False,
            "publication_ready": False,
            "files": output_files,
            "inventory": _file_receipt(inventory_path, root=temporary, rows=431),
            "decision_ledger": _file_receipt(
                decision_destination,
                root=temporary,
                rows=int(source_manifest["decision_ledger"]["rows"]),
            ),
            "bibliography_cleaning": {
                "reconstruction": _file_receipt(reconstruction_path, root=temporary),
                "contract_sha256": contract_sha,
                "plan_sha256": plan_sha,
                "apply_summary_sha256": sha256_file(summary_path),
                "transformed_ranks": transformed_ranks,
                "untouched_ranks": untouched_ranks,
            },
        }
        atomic_write_json(
            temporary / "manifests" / "deduplicated_manifest.json", manifest
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"{output_root}: {len(output_files)} shards, "
        f"{len(transformed_ranks)} transformed, "
        f"{validation['documents']} validated documents"
    )
    return reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--created-at", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
