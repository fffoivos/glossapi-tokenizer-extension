#!/usr/bin/env python3
"""Write a terminal seal for the user-directed consensus-silver cohort.

This is deliberately separate from ``sealed_bibliography_test.freeze``.  The
original 150-document prediction-blind protocol failed its frozen raw A/B
agreement gate and must remain failed.  After that failure, the reviewer chose
to exclude seven systematically problematic documents and retain only repaired
A/B agreements.  This module can seal that derived cohort, but it cannot call
it the original 150-document test or conceal the original blocked receipt.

The unchanged numerical terminal gates are applied to the primary downstream
target, bibliography membership: >=98% overall A/B agreement, >=95% in every
source, <=0.5% unresolved labels, and complete retained-line coverage.  All
auxiliary task-mask coverage is recorded but is not silently promoted to a
terminal gate.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .audit_consensus_silver import AUDIT_SCHEMA
from .contract import sha256_file
from .materialize_consensus_silver import (
    EXCLUSION_SCHEMA,
    LABEL_SCHEMA,
    RECEIPT_SCHEMA,
    TASK_NAMES,
)


FREEZE_SCHEMA = "bibliography-consensus-silver-freeze-v2"
FREEZE_STATUS = "frozen_posthoc_consensus_silver_evaluation_set"
BLOCKED_STATUS = "blocked_consensus_silver_freeze"
PRIMARY_TASK = "bibliography_membership"
ORIGINAL_DOCUMENT_COUNT = 150
OVERALL_AGREEMENT_MINIMUM = 0.98
SOURCE_AGREEMENT_MINIMUM = 0.95
UNRESOLVED_FRACTION_MAXIMUM = 0.005


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
                raise ValueError(f"{path}:{number}: expected JSON object")
            yield value


def _write_json_new(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _expected_output_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "documents": artifact_dir / "documents.consensus-silver.jsonl",
        "line_key": artifact_dir / "line-key.consensus-silver.jsonl",
        "labels": artifact_dir / "labels.task-consensus.jsonl",
        "overlay": artifact_dir / "fine-role.overlay-v3.jsonl",
        "exclusions": artifact_dir / "exclusions.json",
    }


def _parse_expected_source_counts(value: str) -> dict[str, int]:
    parsed = json.loads(value)
    if (
        not isinstance(parsed, dict)
        or not parsed
        or any(
            not isinstance(source, str)
            or not source
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for source, count in parsed.items()
        )
    ):
        raise argparse.ArgumentTypeError("expected a non-empty JSON object of positive counts")
    return dict(sorted(parsed.items()))


def _task_metrics(counts: Mapping[str, int]) -> dict[str, int | float]:
    """Keep inter-annotator agreement distinct from trusted-label coverage."""

    agreement = int(counts.get("agreement", 0))
    disagreement = int(counts.get("disagreement", 0))
    unavailable = int(counts.get("unavailable", 0))
    comparable = agreement + disagreement
    total = comparable + unavailable
    unresolved = disagreement + unavailable
    return {
        "line_count": total,
        "comparable_count": comparable,
        "agreement_count": agreement,
        "disagreement_count": disagreement,
        "unavailable_count": unavailable,
        "agreement_rate_on_comparable": agreement / max(comparable, 1),
        "trusted_coverage_fraction": agreement / max(total, 1),
        "unresolved_count": unresolved,
        "unresolved_fraction": unresolved / max(total, 1),
    }


def freeze_consensus_silver(
    *,
    artifact_dir: Path,
    audit_path: Path,
    original_blocked_receipt_path: Path,
    output_path: Path,
    expected_source_document_counts: Mapping[str, int],
    expected_document_count: int,
    expected_line_count: int,
    code_commit: str,
    slurm_job_id: str = "",
    lock_inputs: bool = False,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    audit_path = audit_path.resolve()
    original_blocked_receipt_path = original_blocked_receipt_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    if output_path.parent != artifact_dir:
        raise ValueError("terminal seal must be written inside the artifact directory")

    receipt_path = artifact_dir / "receipt.json"
    receipt = _read_json(receipt_path)
    audit = _read_json(audit_path)
    original = _read_json(original_blocked_receipt_path)
    paths = _expected_output_paths(artifact_dir)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported materialization receipt")
    if receipt.get("status") != "passed_task_specific_consensus_materialization":
        raise ValueError("materialization did not pass")
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != (
        "passed_independent_consensus_silver_audit"
    ):
        raise ValueError("independent audit did not pass")
    if audit.get("materialization_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("independent audit is not bound to this materialization receipt")
    if original.get("status") != "blocked":
        raise ValueError("original 150-document receipt is not the preserved blocked attempt")
    original_gates = original.get("gates")
    if (
        not isinstance(original_gates, dict)
        or original_gates.get("binary_agreement_overall_gte_0_98") is not False
    ):
        raise ValueError("original blocked receipt does not preserve the observed agreement failure")

    receipt_outputs = receipt.get("outputs")
    audit_hashes = audit.get("verified_output_sha256")
    if not isinstance(receipt_outputs, dict) or set(receipt_outputs) != set(paths):
        raise ValueError("materialization output inventory is invalid")
    if not isinstance(audit_hashes, dict) or set(audit_hashes) != set(paths):
        raise ValueError("audit output inventory is invalid")
    output_hashes: dict[str, str] = {}
    for name, path in paths.items():
        details = receipt_outputs[name]
        if not isinstance(details, dict) or Path(str(details.get("path"))).resolve() != path:
            raise ValueError(f"materialization output path mismatch: {name}")
        digest = sha256_file(path)
        if details.get("sha256") != digest or audit_hashes.get(name) != digest:
            raise ValueError(f"materialization/audit hash mismatch: {name}")
        output_hashes[name] = digest

    documents = list(_iter_jsonl(paths["documents"]))
    line_keys = list(_iter_jsonl(paths["line_key"]))
    labels = list(_iter_jsonl(paths["labels"]))
    exclusions = _read_json(paths["exclusions"])
    if exclusions.get("schema_version") != EXCLUSION_SCHEMA:
        raise ValueError("unsupported exclusion manifest")
    excluded_rows = exclusions.get("documents")
    if not isinstance(excluded_rows, list):
        raise ValueError("exclusion manifest has no document inventory")
    excluded_ids = [str(row.get("document_id") or "") for row in excluded_rows]
    included_ids = [str(row.get("document_id") or "") for row in documents]
    if (
        "" in excluded_ids
        or "" in included_ids
        or len(excluded_ids) != len(set(excluded_ids))
        or len(included_ids) != len(set(included_ids))
        or set(excluded_ids) & set(included_ids)
    ):
        raise ValueError("included/excluded document identities are invalid")

    source_document_counts = dict(
        sorted(collections.Counter(str(row.get("source") or "") for row in documents).items())
    )
    expected_sources = dict(sorted(expected_source_document_counts.items()))
    if source_document_counts != expected_sources:
        raise ValueError(
            f"retained source-document counts differ: {source_document_counts} != {expected_sources}"
        )
    if len(documents) != expected_document_count or len(labels) != expected_line_count:
        raise ValueError("retained document/line count differs from the pinned expectation")
    if len(documents) + len(excluded_ids) != ORIGINAL_DOCUMENT_COUNT:
        raise ValueError("retained and excluded documents do not reconstruct the original 150")
    if not (len(line_keys) == len(labels) == int(receipt.get("line_count", -1))):
        raise ValueError("line-key/label/materialization counts differ")
    if int(receipt.get("document_count", -1)) != len(documents):
        raise ValueError("materialization document count differs")
    if int(receipt.get("excluded_document_count", -1)) != len(excluded_ids):
        raise ValueError("materialization exclusion count differs")

    task_totals = {task: collections.Counter() for task in TASK_NAMES}
    source_task_totals = collections.defaultdict(
        lambda: {task: collections.Counter() for task in TASK_NAMES}
    )
    identities: set[tuple[str, str]] = set()
    aliases: set[str] = set()
    for key, label in zip(line_keys, labels, strict=True):
        identity = (str(key.get("document_id") or ""), str(key.get("line_id") or ""))
        alias = str(key.get("line_alias") or "")
        if (
            identity in identities
            or not all(identity)
            or not alias
            or alias in aliases
            or label.get("schema_version") != LABEL_SCHEMA
            or (str(label.get("document_id") or ""), str(label.get("line_id") or ""))
            != identity
            or str(label.get("line_alias") or "") != alias
            or int(label.get("abs_idx", -1)) != int(key.get("abs_idx", -2))
            or str(label.get("source") or "") != str(key.get("source") or "")
            or label.get("human_gold") is not False
        ):
            raise ValueError(f"line identity/provenance mismatch: {identity}")
        tasks = label.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(TASK_NAMES):
            raise ValueError(f"task inventory mismatch: {identity}")
        raw_vote_unavailable = "UNKNOWN" in {
            str(label.get("pass_a_role") or ""),
            str(label.get("pass_b_role") or ""),
        }
        source = str(label["source"])
        for task in TASK_NAMES:
            decision = tasks[task]
            if (
                not isinstance(decision, dict)
                or set(decision) != {"label", "trusted"}
                or not isinstance(decision.get("label"), str)
                or not isinstance(decision.get("trusted"), bool)
                or decision["trusted"] != (decision["label"] != "UNKNOWN")
            ):
                raise ValueError(f"malformed task decision: {identity} {task}")
            outcome = (
                "unavailable"
                if raw_vote_unavailable
                else ("agreement" if decision["trusted"] else "disagreement")
            )
            task_totals[task][outcome] += 1
            source_task_totals[source][task][outcome] += 1
        identities.add(identity)
        aliases.add(alias)

    task_metrics = {
        task: _task_metrics(counts) for task, counts in task_totals.items()
    }
    task_metrics_by_source = {
        source: {
            task: _task_metrics(counts) for task, counts in tasks.items()
        }
        for source, tasks in sorted(source_task_totals.items())
    }
    overall_primary = float(task_metrics[PRIMARY_TASK]["agreement_rate_on_comparable"])
    unresolved_primary = float(task_metrics[PRIMARY_TASK]["unresolved_fraction"])
    source_primary_metrics = {
        source: tasks[PRIMARY_TASK]
        for source, tasks in task_metrics_by_source.items()
    }
    if set(task_metrics_by_source) != set(expected_sources):
        raise ValueError("primary-task source inventory differs")

    gates = {
        "independent_audit_passed": True,
        "original_failed_150_document_freeze_preserved": True,
        "retained_and_excluded_documents_reconstruct_original_150": (
            len(documents) + len(excluded_ids) == ORIGINAL_DOCUMENT_COUNT
        ),
        "complete_retained_line_coverage": len(line_keys) == len(labels),
        "membership_agreement_overall_gte_0_98": (
            overall_primary >= OVERALL_AGREEMENT_MINIMUM
        ),
        "membership_agreement_each_source_gte_0_95": all(
            details["agreement_rate_on_comparable"] >= SOURCE_AGREEMENT_MINIMUM
            for details in source_primary_metrics.values()
        ),
        "membership_unresolved_fraction_lte_0_005": (
            unresolved_primary <= UNRESOLVED_FRACTION_MAXIMUM
        ),
    }
    status = FREEZE_STATUS if all(gates.values()) else BLOCKED_STATUS
    frozen = {
        "schema_version": FREEZE_SCHEMA,
        "status": status,
        "slurm_job_id": slurm_job_id,
        "code_commit": code_commit,
        "human_gold": False,
        "label_semantics": (
            "post-repair dual-Codex task-specific consensus silver; only exact repaired "
            "A/B task agreement is trusted"
        ),
        "protocol_semantics": (
            "user-directed post-hoc 143-document consensus cohort; this does not convert "
            "the failed original 150-document prediction-blind attempt into a passed freeze"
        ),
        "primary_evaluation_task": PRIMARY_TASK,
        "document_count": len(documents),
        "excluded_document_count": len(excluded_ids),
        "line_count": len(labels),
        "source_document_counts": source_document_counts,
        "source_membership_metrics": source_primary_metrics,
        "task_metrics": task_metrics,
        "task_metrics_by_source": task_metrics_by_source,
        "thresholds": {
            "membership_agreement_overall_minimum": OVERALL_AGREEMENT_MINIMUM,
            "membership_agreement_each_source_minimum": SOURCE_AGREEMENT_MINIMUM,
            "membership_unresolved_fraction_maximum": UNRESOLVED_FRACTION_MAXIMUM,
        },
        "gates": gates,
        "sealed_hashes": {
            **{f"{name}_sha256": digest for name, digest in sorted(output_hashes.items())},
            "materialization_receipt_sha256": sha256_file(receipt_path),
            "independent_audit_sha256": sha256_file(audit_path),
            "original_blocked_consensus_receipt_sha256": sha256_file(
                original_blocked_receipt_path
            ),
        },
        "original_150_document_attempt": {
            "status": "blocked",
            "a_b_binary_agreement_overall": original.get(
                "a_b_binary_agreement_overall"
            ),
            "failed_gate": "binary_agreement_overall_gte_0_98",
            "receipt_path": str(original_blocked_receipt_path),
        },
        "lock_policy": (
            "all derived cohort data, labels, manifests, receipts, audit and seal are mode 0440"
            if lock_inputs
            else "not locked"
        ),
        "code_sha256": sha256_file(Path(__file__)),
    }
    _write_json_new(output_path, frozen, mode=0o440 if lock_inputs else 0o600)
    if status != FREEZE_STATUS:
        raise ValueError(f"consensus-silver gates failed; blocked receipt preserved at {output_path}")
    if lock_inputs:
        for path in (*paths.values(), receipt_path, audit_path, output_path):
            path.chmod(0o440)
    return frozen


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--original-blocked-consensus-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-source-document-counts",
        type=_parse_expected_source_counts,
        required=True,
    )
    parser.add_argument("--expected-document-count", type=int, required=True)
    parser.add_argument("--expected-line-count", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    parser.add_argument("--lock-inputs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = freeze_consensus_silver(
        artifact_dir=args.artifact_dir,
        audit_path=args.audit,
        original_blocked_receipt_path=args.original_blocked_consensus_receipt,
        output_path=args.output,
        expected_source_document_counts=args.expected_source_document_counts,
        expected_document_count=args.expected_document_count,
        expected_line_count=args.expected_line_count,
        code_commit=args.code_commit,
        slurm_job_id=args.slurm_job_id,
        lock_inputs=args.lock_inputs,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
