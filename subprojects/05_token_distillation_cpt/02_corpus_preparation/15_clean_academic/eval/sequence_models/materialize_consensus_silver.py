#!/usr/bin/env python3
"""Materialize task-specific A/B consensus silver after annotation repairs.

The materializer does not adjudicate disagreements.  It preserves both votes,
trusts a target only when the two repaired passes agree after recoding for that
target, and writes excluded documents to a separate provenance manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_role_v2 import ANNOTATION_ROLES, CONTRACT_SCHEMA, OVERLAY_SCHEMA
from .contract import sha256_file


LABEL_SCHEMA = "bibliography-task-consensus-silver-v1"
RECEIPT_SCHEMA = "bibliography-task-consensus-silver-receipt-v1"
EXCLUSION_SCHEMA = "bibliography-task-consensus-silver-exclusions-v1"

BIB_ROLES = frozenset(
    {"ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER"}
)
HEADER_ROLES = frozenset({"BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER"})
CONTEXT_ROLES = frozenset({"CONTINUATION", "FILLER"})
TASK_NAMES = (
    "bibliography_membership",
    "entry_seed",
    "heading_type",
    "context_role",
    "fine_role",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield value


def _write_json_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _role_map(role: str, task: str) -> str:
    if role == "UNKNOWN":
        return "UNKNOWN"
    if task == "bibliography_membership":
        return "BIB" if role in BIB_ROLES else "NON_BIB"
    if task == "entry_seed":
        return "ENTRY" if role == "ENTRY" else "NOT_ENTRY"
    if task == "heading_type":
        return role if role in HEADER_ROLES else "NOT_HEADER"
    if task == "context_role":
        return role if role in CONTEXT_ROLES else "OTHER"
    if task == "fine_role":
        return role
    raise ValueError(f"unknown task: {task}")


def task_consensus(role_a: str, role_b: str) -> dict[str, dict[str, Any]]:
    """Return task labels; UNKNOWN in either pass always fails closed."""

    if role_a not in ANNOTATION_ROLES or role_b not in ANNOTATION_ROLES:
        raise ValueError(f"invalid roles: {role_a!r}, {role_b!r}")
    result: dict[str, dict[str, Any]] = {}
    for task in TASK_NAMES:
        left, right = _role_map(role_a, task), _role_map(role_b, task)
        trusted = left != "UNKNOWN" and left == right
        result[task] = {
            "label": left if trusted else "UNKNOWN",
            "trusted": trusted,
        }
    return result


def _pass_lines(role_pass: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = role_pass.get("lines")
    if not isinstance(raw, list) or not raw:
        raise ValueError("role pass has no lines")
    result: dict[str, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("role-pass line is not an object")
        alias = str(row.get("line_alias") or "")
        role = str(row.get("role") or "")
        confidence = row.get("confidence")
        if (
            not alias
            or alias in result
            or role not in ANNOTATION_ROLES
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"malformed or duplicate role-pass line: {alias!r}")
        result[alias] = dict(row)
    return result


def _document_inventory(
    documents: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    by_document: dict[str, Mapping[str, Any]] = {}
    by_line: dict[tuple[str, str], Mapping[str, Any]] = {}
    for document in documents:
        document_id = str(document.get("document_id") or "")
        lines = document.get("lines")
        if not document_id or document_id in by_document or not isinstance(lines, list):
            raise ValueError(f"malformed or duplicate document: {document_id!r}")
        by_document[document_id] = document
        for line in lines:
            if not isinstance(line, Mapping):
                raise ValueError(f"{document_id}: malformed line")
            line_id = str(line.get("line_id") or "")
            text = line.get("text")
            abs_idx = line.get("abs_idx")
            key = (document_id, line_id)
            if (
                not line_id
                or key in by_line
                or not isinstance(text, str)
                or not isinstance(abs_idx, int)
            ):
                raise ValueError(f"{document_id}: malformed or duplicate line {line_id!r}")
            by_line[key] = line
    return by_document, by_line


def _count_labels(
    counters: dict[str, collections.Counter[str]], tasks: Mapping[str, Mapping[str, Any]],
) -> None:
    for task, decision in tasks.items():
        label = str(decision["label"])
        counters[task][label] += 1


def materialize(
    *,
    documents_path: Path,
    line_key_path: Path,
    pass_a_path: Path,
    pass_b_path: Path,
    excluded_document_ids: Sequence[str],
    output_dir: Path,
    code_commit: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    excluded = frozenset(str(value) for value in excluded_document_ids)
    if "" in excluded or len(excluded) != len(excluded_document_ids):
        raise ValueError("excluded document IDs must be unique and non-empty when supplied")

    documents = list(_iter_jsonl(documents_path))
    line_keys = list(_iter_jsonl(line_key_path))
    pass_a, pass_b = _read_json(pass_a_path), _read_json(pass_b_path)
    a_by_alias, b_by_alias = _pass_lines(pass_a), _pass_lines(pass_b)
    docs_by_id, source_lines = _document_inventory(documents)
    if not excluded <= set(docs_by_id):
        raise ValueError(f"{len(excluded - set(docs_by_id))} excluded documents are absent")

    keys_by_alias: dict[str, dict[str, Any]] = {}
    keyed_lines: set[tuple[str, str]] = set()
    for key in line_keys:
        alias = str(key.get("line_alias") or "")
        document_id = str(key.get("document_id") or "")
        line_id = str(key.get("line_id") or "")
        identity = (document_id, line_id)
        if not alias or alias in keys_by_alias or identity in keyed_lines:
            raise ValueError("line key has empty or duplicate identities")
        if identity not in source_lines:
            raise ValueError(f"line key is absent from documents: {identity}")
        source_line = source_lines[identity]
        if int(key.get("abs_idx", -1)) != int(source_line["abs_idx"]):
            raise ValueError(f"line-key coordinate mismatch: {identity}")
        if str(key.get("source") or "") != str(docs_by_id[document_id].get("source") or ""):
            raise ValueError(f"line-key source mismatch: {identity}")
        keys_by_alias[alias] = key
        keyed_lines.add(identity)
    if keyed_lines != set(source_lines):
        raise ValueError("line key does not exactly cover document lines")
    if set(keys_by_alias) != set(a_by_alias) or set(keys_by_alias) != set(b_by_alias):
        raise ValueError("both passes must exactly cover the line key")

    output_dir.mkdir(parents=True)
    included_documents = [
        document for document in documents
        if str(document["document_id"]) not in excluded
    ]
    included_keys = [
        key for key in line_keys if str(key["document_id"]) not in excluded
    ]
    labels: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    task_counts = {task: collections.Counter() for task in TASK_NAMES}
    task_counts_by_source: dict[str, dict[str, collections.Counter[str]]] = collections.defaultdict(
        lambda: {task: collections.Counter() for task in TASK_NAMES}
    )
    exact_count = binary_disagreement_count = unknown_vote_count = 0
    reviewers = [str(pass_a.get("reviewer") or "pass-a"), str(pass_b.get("reviewer") or "pass-b")]

    for key in included_keys:
        alias = str(key["line_alias"])
        document_id, line_id = str(key["document_id"]), str(key["line_id"])
        line = source_lines[(document_id, line_id)]
        document = docs_by_id[document_id]
        a, b = a_by_alias[alias], b_by_alias[alias]
        if (
            str(a.get("document_alias") or "") != str(key.get("document_alias") or "")
            or str(b.get("document_alias") or "") != str(key.get("document_alias") or "")
            or str(a.get("source") or "") != str(key.get("source") or "")
            or str(b.get("source") or "") != str(key.get("source") or "")
        ):
            raise ValueError(f"pass/line-key provenance mismatch: {alias}")
        role_a, role_b = str(a["role"]), str(b["role"])
        tasks = task_consensus(role_a, role_b)
        source = str(key["source"])
        _count_labels(task_counts, tasks)
        _count_labels(task_counts_by_source[source], tasks)
        exact_count += int(tasks["fine_role"]["trusted"])
        binary_disagreement_count += int(
            role_a != "UNKNOWN" and role_b != "UNKNOWN"
            and not tasks["bibliography_membership"]["trusted"]
        )
        unknown_vote_count += int("UNKNOWN" in {role_a, role_b})
        text_hash = hashlib.sha256(str(line["text"]).encode("utf-8")).hexdigest()
        confidence = min(float(a["confidence"]), float(b["confidence"]))
        row = {
            "schema_version": LABEL_SCHEMA,
            "document_id": document_id,
            "work_id": str(document.get("work_id") or ""),
            "source": source,
            "line_id": line_id,
            "line_alias": alias,
            "abs_idx": int(line["abs_idx"]),
            "text_sha256": text_hash,
            "pass_a_role": role_a,
            "pass_b_role": role_b,
            "pass_a_confidence": float(a["confidence"]),
            "pass_b_confidence": float(b["confidence"]),
            "tasks": tasks,
            "label_origin": "post_repair_dual_agreement",
            "reviewers": reviewers,
            "human_gold": False,
        }
        labels.append(row)
        fine = tasks["fine_role"]
        overlays.append(
            {
                "schema_version": OVERLAY_SCHEMA,
                "role_contract_schema": CONTRACT_SCHEMA,
                "document_id": document_id,
                "work_id": str(document.get("work_id") or ""),
                "source": source,
                "line_id": line_id,
                "line_alias": alias,
                "abs_idx": int(line["abs_idx"]),
                "text_sha256": text_hash,
                "original_region_label": "UNKNOWN",
                "role": str(fine["label"]),
                "role_status": "AGREED_REVIEW" if fine["trusted"] else "UNRESOLVED",
                "role_confidence": confidence if fine["trusted"] else 0.0,
                "boundary_flag": "NONE",
                "boundary_status": "UNRESOLVED",
                "boundary_confidence": 0.0,
                "label_origin": "post_repair_dual_agreement",
                "reviewers": reviewers,
                "review_case_ids": [],
                "raw_role_votes": {reviewers[0]: [role_a], reviewers[1]: [role_b]},
                "raw_boundary_votes": {reviewers[0]: ["NONE"], reviewers[1]: ["NONE"]},
                "task_consensus": tasks,
                "human_gold": False,
            }
        )

    paths = {
        "documents": output_dir / "documents.consensus-silver.jsonl",
        "line_key": output_dir / "line-key.consensus-silver.jsonl",
        "labels": output_dir / "labels.task-consensus.jsonl",
        "overlay": output_dir / "fine-role.overlay-v3.jsonl",
        "exclusions": output_dir / "exclusions.json",
    }
    _write_jsonl_new(paths["documents"], included_documents)
    _write_jsonl_new(paths["line_key"], included_keys)
    _write_jsonl_new(paths["labels"], labels)
    _write_jsonl_new(paths["overlay"], overlays)
    exclusion_rows = []
    for document_id in sorted(excluded):
        document = docs_by_id[document_id]
        exclusion_rows.append(
            {
                "document_id": document_id,
                "source": str(document.get("source") or ""),
                "source_doc_id": str(document.get("source_doc_id") or ""),
                "line_count": len(document["lines"]),
                "reason": "removed_by_user_after_post_repair_disagreement_review",
            }
        )
    _write_json_new(
        paths["exclusions"],
        {
            "schema_version": EXCLUSION_SCHEMA,
            "excluded_document_count": len(exclusion_rows),
            "documents": exclusion_rows,
        },
    )

    comparable = len(labels) - unknown_vote_count
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed_task_specific_consensus_materialization",
        "code_commit": code_commit,
        "slurm_job_id": slurm_job_id,
        "label_semantics": (
            "dual-Codex consensus-silver after deterministic contextual-role and "
            "Markdown-header repairs; disagreements are masked per task; not human gold"
        ),
        "human_gold": False,
        "document_count": len(included_documents),
        "excluded_document_count": len(excluded),
        "line_count": len(labels),
        "comparable_binary_line_count": comparable,
        "binary_disagreement_count": binary_disagreement_count,
        "binary_agreed_line_count": int(sum(task_counts["bibliography_membership"].values()))
        - int(task_counts["bibliography_membership"]["UNKNOWN"]),
        "binary_agreement_rate_on_comparable": (
            (comparable - binary_disagreement_count) / comparable if comparable else 0.0
        ),
        "exact_role_agreed_line_count": exact_count,
        "line_with_unknown_vote_count": unknown_vote_count,
        "task_label_counts": {
            task: dict(sorted(counter.items())) for task, counter in task_counts.items()
        },
        "task_label_counts_by_source": {
            source: {
                task: dict(sorted(counter.items()))
                for task, counter in task_counters.items()
            }
            for source, task_counters in sorted(task_counts_by_source.items())
        },
        "inputs": {
            "documents": {"path": str(documents_path), "sha256": sha256_file(documents_path)},
            "line_key": {"path": str(line_key_path), "sha256": sha256_file(line_key_path)},
            "pass_a": {"path": str(pass_a_path), "sha256": sha256_file(pass_a_path)},
            "pass_b": {"path": str(pass_b_path), "sha256": sha256_file(pass_b_path)},
        },
        "outputs": {},
        "invariants": {
            "original_inputs_mutated": False,
            "excluded_documents_absent_from_all_training_outputs": True,
            "every_included_source_line_emitted_once": True,
            "disagreements_adjudicated": False,
            "third_pass_used": False,
            "task_labels_require_recoded_a_b_agreement": True,
        },
    }
    for name, path in paths.items():
        receipt["outputs"][name] = {
            "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)
        }
    _write_json_new(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--line-key", type=Path, required=True)
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--exclude-document-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = materialize(
        documents_path=args.documents.resolve(),
        line_key_path=args.line_key.resolve(),
        pass_a_path=args.pass_a.resolve(),
        pass_b_path=args.pass_b.resolve(),
        excluded_document_ids=args.exclude_document_id,
        output_dir=args.output_dir.resolve(),
        code_commit=args.code_commit,
        slurm_job_id=args.slurm_job_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
