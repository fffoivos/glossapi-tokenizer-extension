#!/usr/bin/env python3
"""Independently audit a task-specific bibliography consensus-silver artifact."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import sha256_file
from .materialize_consensus_silver import (
    LABEL_SCHEMA,
    TASK_NAMES,
    _document_inventory,
    _iter_jsonl,
    _pass_lines,
    _read_json,
    task_consensus,
)


AUDIT_SCHEMA = "bibliography-task-consensus-silver-audit-v1"


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("document_id") or ""), str(row.get("line_id") or "")


def audit(
    *,
    artifact_dir: Path,
    original_documents_path: Path,
    original_line_key_path: Path,
    pass_a_path: Path,
    pass_b_path: Path,
    output_path: Path,
    expected_document_count: int | None = None,
    expected_line_count: int | None = None,
    expected_comparable_count: int | None = None,
    expected_binary_disagreement_count: int | None = None,
    slurm_job_id: str = "",
) -> dict[str, Any]:
    receipt = _read_json(artifact_dir / "receipt.json")
    output_paths = {
        name: Path(details["path"])
        for name, details in receipt.get("outputs", {}).items()
    }
    required = {"documents", "line_key", "labels", "overlay", "exclusions"}
    if set(output_paths) != required:
        raise ValueError("materialization receipt has an unexpected output inventory")
    for name, path in output_paths.items():
        if path.resolve().parent != artifact_dir.resolve():
            raise ValueError(f"output escapes artifact directory: {name}")
        observed = sha256_file(path)
        if observed != receipt["outputs"][name]["sha256"]:
            raise ValueError(f"output hash mismatch: {name}")

    expected_inputs = {
        "documents": original_documents_path,
        "line_key": original_line_key_path,
        "pass_a": pass_a_path,
        "pass_b": pass_b_path,
    }
    for name, path in expected_inputs.items():
        details = receipt.get("inputs", {}).get(name, {})
        if details.get("sha256") != sha256_file(path):
            raise ValueError(f"source input hash mismatch: {name}")

    original_documents = list(_iter_jsonl(original_documents_path))
    original_keys = list(_iter_jsonl(original_line_key_path))
    original_docs_by_id, original_lines = _document_inventory(original_documents)
    included_documents = list(_iter_jsonl(output_paths["documents"]))
    included_keys = list(_iter_jsonl(output_paths["line_key"]))
    labels = list(_iter_jsonl(output_paths["labels"]))
    overlays = list(_iter_jsonl(output_paths["overlay"]))
    exclusions = _read_json(output_paths["exclusions"])
    excluded_ids = {str(row["document_id"]) for row in exclusions.get("documents", [])}
    if len(excluded_ids) != int(exclusions.get("excluded_document_count", -1)):
        raise ValueError("exclusion manifest repeats document identities")
    if not excluded_ids <= set(original_docs_by_id):
        raise ValueError("exclusion manifest contains absent documents")

    included_ids = [str(document.get("document_id") or "") for document in included_documents]
    if len(included_ids) != len(set(included_ids)) or set(included_ids) & excluded_ids:
        raise ValueError("included document inventory is duplicated or contains exclusions")
    if set(included_ids) | excluded_ids != set(original_docs_by_id):
        raise ValueError("included and excluded documents do not partition the source")
    expected_keys = [
        row for row in original_keys if str(row["document_id"]) not in excluded_ids
    ]
    if included_keys != expected_keys:
        raise ValueError("filtered line key is not an order-preserving source subset")
    if not (len(included_keys) == len(labels) == len(overlays)):
        raise ValueError("line-key, label, and overlay counts differ")

    a_by_alias = _pass_lines(_read_json(pass_a_path))
    b_by_alias = _pass_lines(_read_json(pass_b_path))
    task_counts = {task: collections.Counter() for task in TASK_NAMES}
    comparable = binary_disagreements = exact = unknown_vote = 0
    for key, label, overlay in zip(included_keys, labels, overlays):
        wanted_identity = (str(key["document_id"]), str(key["line_id"]))
        if _identity(label) != wanted_identity or _identity(overlay) != wanted_identity:
            raise ValueError(f"line identity/order mismatch: {wanted_identity}")
        if label.get("schema_version") != LABEL_SCHEMA:
            raise ValueError("unexpected task-label schema")
        source_line = original_lines[wanted_identity]
        if (
            int(label.get("abs_idx", -1)) != int(source_line["abs_idx"])
            or label.get("text_sha256") != overlay.get("text_sha256")
        ):
            raise ValueError(f"line coordinate/hash mismatch: {wanted_identity}")
        alias = str(key["line_alias"])
        role_a, role_b = str(a_by_alias[alias]["role"]), str(b_by_alias[alias]["role"])
        expected_tasks = task_consensus(role_a, role_b)
        if label.get("tasks") != expected_tasks or overlay.get("task_consensus") != expected_tasks:
            raise ValueError(f"task consensus was not reproduced: {wanted_identity}")
        if label.get("pass_a_role") != role_a or label.get("pass_b_role") != role_b:
            raise ValueError(f"stored pass votes differ: {wanted_identity}")
        fine = expected_tasks["fine_role"]
        expected_overlay_role = fine["label"]
        expected_status = "AGREED_REVIEW" if fine["trusted"] else "UNRESOLVED"
        if overlay.get("role") != expected_overlay_role or overlay.get("role_status") != expected_status:
            raise ValueError(f"fine-role overlay differs from consensus: {wanted_identity}")
        for task, decision in expected_tasks.items():
            task_counts[task][str(decision["label"])] += 1
        unknown_vote += int("UNKNOWN" in {role_a, role_b})
        comparable += int("UNKNOWN" not in {role_a, role_b})
        binary_disagreements += int(
            "UNKNOWN" not in {role_a, role_b}
            and not expected_tasks["bibliography_membership"]["trusted"]
        )
        exact += int(fine["trusted"])

    observed = {
        "document_count": len(included_documents),
        "line_count": len(labels),
        "comparable_binary_line_count": comparable,
        "binary_disagreement_count": binary_disagreements,
        "binary_agreed_line_count": comparable - binary_disagreements,
        "exact_role_agreed_line_count": exact,
        "line_with_unknown_vote_count": unknown_vote,
    }
    expected_values = {
        "document_count": expected_document_count,
        "line_count": expected_line_count,
        "comparable_binary_line_count": expected_comparable_count,
        "binary_disagreement_count": expected_binary_disagreement_count,
    }
    for name, wanted in expected_values.items():
        if wanted is not None and observed[name] != wanted:
            raise ValueError(f"expected {name}={wanted}, got {observed[name]}")
    for name, value in observed.items():
        if int(receipt.get(name, -1)) != value:
            raise ValueError(f"materialization receipt count mismatch: {name}")
    normalized_counts = {
        task: dict(sorted(counter.items())) for task, counter in task_counts.items()
    }
    if receipt.get("task_label_counts") != normalized_counts:
        raise ValueError("task-label counts differ from materialization receipt")

    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed_independent_consensus_silver_audit",
        "slurm_job_id": slurm_job_id,
        **observed,
        "excluded_document_count": len(excluded_ids),
        "task_label_counts": normalized_counts,
        "materialization_receipt_sha256": sha256_file(artifact_dir / "receipt.json"),
        "verified_output_sha256": {
            name: sha256_file(path) for name, path in sorted(output_paths.items())
        },
        "checks": {
            "source_hashes_match": True,
            "output_hashes_match": True,
            "included_and_excluded_documents_partition_source": True,
            "excluded_documents_absent_from_outputs": True,
            "line_identity_order_and_coverage_match": True,
            "all_task_labels_recomputed_from_repaired_a_b_votes": True,
            "fine_role_overlay_matches_exact_agreement_only": True,
            "third_pass_used": False,
            "human_gold": False,
        },
    }
    _write_json_new(output_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--original-documents", type=Path, required=True)
    parser.add_argument("--original-line-key", type=Path, required=True)
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-document-count", type=int)
    parser.add_argument("--expected-line-count", type=int)
    parser.add_argument("--expected-comparable-count", type=int)
    parser.add_argument("--expected-binary-disagreement-count", type=int)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(
        artifact_dir=args.artifact_dir.resolve(),
        original_documents_path=args.original_documents.resolve(),
        original_line_key_path=args.original_line_key.resolve(),
        pass_a_path=args.pass_a.resolve(),
        pass_b_path=args.pass_b.resolve(),
        output_path=args.output.resolve(),
        expected_document_count=args.expected_document_count,
        expected_line_count=args.expected_line_count,
        expected_comparable_count=args.expected_comparable_count,
        expected_binary_disagreement_count=args.expected_binary_disagreement_count,
        slurm_job_id=args.slurm_job_id,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
