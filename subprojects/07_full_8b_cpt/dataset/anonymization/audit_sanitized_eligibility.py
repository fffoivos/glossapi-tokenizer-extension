#!/usr/bin/env python3
"""Independently prove the frozen row-eligibility policy and PII token contract."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from anonymization_common import (
    SHARD_SCHEMA,
    absolute_receipt,
    canonical_sha256,
    import_parent_builder,
    load_parent,
    read_json,
    sha256_file,
    utc_now,
    validate_file_receipt,
    validate_overlay,
    validate_task_input,
    write_json_atomic,
)
from bridge_common import task_output_prefix


AUDIT_SCHEMA = "full_cpt_sanitized_eligibility_audit_v1"
PII_TOKEN_IDS = {"<iban-pii>": 36, "<email-pii>": 37, "<ip-pii>": 38}


def _target(row: Mapping[str, Any]) -> bool:
    return row.get("source_dataset") == "openarchives.gr" and row.get("needs_ocr") is True


def _retained_ids(path: Path, targets: set[str]) -> set[str]:
    found: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            doc_id = value.get("doc_id")
            if not isinstance(doc_id, str):
                raise ValueError(f"retained ledger lacks doc_id: {path}:{line_number}")
            if doc_id in targets:
                found.add(doc_id)
    return found


def _token_ids(tokenizer_root: Path) -> dict[str, int]:
    tokenizer_path = tokenizer_root / "tokenizer.json"
    value = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = value.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json lacks model.vocab")
    observed = {token: int(vocab.get(token, -1)) for token in PII_TOKEN_IDS}
    if observed != PII_TOKEN_IDS:
        raise ValueError(f"PII replacement-token id drift: {observed}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    overlay_path = args.overlay.resolve()
    overlay = validate_overlay(overlay_path, Path(__file__))
    parent = load_parent(overlay)
    tasks = overlay["tasks"]
    stage = args.stage_root.resolve()
    output = (args.output or stage / "eligibility_audit.json").resolve()
    if output.exists():
        value = read_json(output)
        if (
            value.get("schema_version") != AUDIT_SCHEMA
            or value.get("status") != "passed"
            or value.get("overlay_sha256") != sha256_file(overlay_path)
        ):
            raise ValueError("existing eligibility audit binding drift")
        print(json.dumps({"ok": True, "resumed": True, "output": str(output)}, sort_keys=True))
        return 0

    builder = import_parent_builder()
    by_path: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for task_index, task in enumerate(tasks):
        if task.get("source_name") == "cleaned_greek_v2":
            validate_task_input(task)
            by_path[str(Path(task["input_path"]).resolve())].append((task_index, task))

    targets_by_task: dict[int, set[str]] = collections.defaultdict(set)
    raw_target_rows = 0
    eligible_target_rows = 0
    for path_string, path_tasks in sorted(by_path.items()):
        path = Path(path_string)
        first = path_tasks[0][1]
        columns = [str(first["text_column"]), "source_dataset", "needs_ocr"]
        for task_index, task in path_tasks:
            for column in task.get("identity_columns", []):
                if str(column) not in columns:
                    columns.append(str(column))
            field = task.get("filter_field")
            if field and str(field) not in columns:
                columns.append(str(field))
        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=4096, use_threads=False):
            data = batch.to_pydict()
            for offset in range(batch.num_rows):
                row = {column: data[column][offset] for column in columns}
                if not _target(row):
                    continue
                raw_target_rows += 1
                text = row.get(str(first["text_column"]))
                if not isinstance(text, str) or not text or not builder._eligible(row, first):
                    continue
                eligible_target_rows += 1
                absolute_row = row_index + offset
                doc_id = builder.document_key(
                    str(first["source_name"]),
                    str(first["input_relative"]),
                    absolute_row,
                    {
                        str(column): row.get(str(column))
                        for column in first.get("identity_columns", [])
                    },
                    identity_scope=str(first["identity_scope"]),
                )
                owners: list[int] = []
                for task_index, task in path_tasks:
                    module = builder._load_phase_module(task)
                    if module is None or builder._in_selected_phase(row, doc_id, task, module):
                        owners.append(task_index)
                if len(owners) != 1:
                    raise ValueError(
                        f"eligibility target has {len(owners)} phase owners: {path}:{absolute_row}"
                    )
                targets_by_task[owners[0]].add(doc_id)
            row_index += batch.num_rows

    required = int(overlay["eligibility"]["required_excluded_rows"])
    if raw_target_rows != required or eligible_target_rows != required:
        raise ValueError(
            f"eligibility target count drift: raw={raw_target_rows} eligible={eligible_target_rows} required={required}"
        )
    if sum(map(len, targets_by_task.values())) != required:
        raise RuntimeError("target identity accounting does not close")

    manifest_excluded = 0
    affected_manifests: list[dict[str, Any]] = []
    retained_collisions: set[str] = set()
    for task_index, target_ids in sorted(targets_by_task.items()):
        task = tasks[task_index]
        manifest_path = Path(str(task_output_prefix(stage, task)) + ".manifest.json")
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") != SHARD_SCHEMA
            or manifest.get("status") != "completed"
            or manifest.get("task_index") != task_index
            or manifest.get("task_sha256") != canonical_sha256(task)
            or manifest.get("anonymization_overlay_sha256") != sha256_file(overlay_path)
        ):
            raise ValueError(f"invalid sanitized task manifest: {manifest_path}")
        observed = int(manifest["counts"].get("policy_excluded_rows", -1))
        if observed != len(target_ids):
            raise ValueError(
                f"policy counter mismatch for task {task_index}: {observed} != {len(target_ids)}"
            )
        manifest_excluded += observed
        retained_path = validate_file_receipt(manifest["outputs"]["retained_ledger"])
        retained_collisions.update(_retained_ids(retained_path, target_ids))
        affected_manifests.append({
            "task_index": task_index,
            "expected_excluded_rows": len(target_ids),
            "manifest": absolute_receipt(manifest_path),
            "retained_ledger": dict(manifest["outputs"]["retained_ledger"]),
        })
    if manifest_excluded != required:
        raise RuntimeError("manifest eligibility counters do not close")
    if retained_collisions:
        raise RuntimeError(
            f"eligibility policy leaked {len(retained_collisions)} documents into retained ledgers"
        )

    token_ids = _token_ids(Path(parent["tokenizer"]["root"]))
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "completed_at": utc_now(),
        "overlay": absolute_receipt(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "policy": overlay["eligibility"],
        "counts": {
            "cleaned_greek_unique_parquet_inputs": len(by_path),
            "raw_matching_rows": raw_target_rows,
            "eligible_matching_rows": eligible_target_rows,
            "phase_owned_matching_rows": sum(map(len, targets_by_task.values())),
            "manifest_policy_excluded_rows": manifest_excluded,
            "retained_matching_rows": 0,
            "affected_tasks": len(targets_by_task),
        },
        "pii_replacement_token_ids": token_ids,
        "affected_task_manifests": affected_manifests,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)
    print(json.dumps({"ok": True, **payload["counts"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
