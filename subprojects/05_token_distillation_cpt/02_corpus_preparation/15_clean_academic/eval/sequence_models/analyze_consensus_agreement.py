#!/usr/bin/env python3
"""Measure repaired A/B agreement on the retained consensus cohort.

Agreement and coverage use different denominators.  Header and contextual-line
metrics also separate category detection from conditional subtype agreement so
the dominant ordinary-line class cannot hide detection misses.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contract import sha256_file
from .materialize_consensus_silver import (
    CONTEXT_ROLES,
    HEADER_ROLES,
    LABEL_SCHEMA,
    TASK_NAMES,
)


ANALYSIS_SCHEMA = "bibliography-consensus-agreement-analysis-v1"


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


def _task_summary(counter: Mapping[str, int]) -> dict[str, int | float]:
    agreement = int(counter.get("agreement", 0))
    disagreement = int(counter.get("disagreement", 0))
    unavailable = int(counter.get("unavailable", 0))
    comparable = agreement + disagreement
    total = comparable + unavailable
    return {
        "line_count": total,
        "comparable_count": comparable,
        "agreement_count": agreement,
        "disagreement_count": disagreement,
        "unavailable_count": unavailable,
        "agreement_rate_on_comparable": agreement / max(comparable, 1),
        "trusted_coverage_fraction": agreement / max(total, 1),
        "unresolved_count": disagreement + unavailable,
        "unresolved_fraction": (disagreement + unavailable) / max(total, 1),
    }


def _detection_summary(counter: Mapping[str, int]) -> dict[str, int | float]:
    union = int(counter.get("union", 0))
    both = int(counter.get("both", 0))
    exact_subtype = int(counter.get("exact_subtype", 0))
    one = int(counter.get("one", 0))
    unavailable = int(counter.get("unavailable", 0))
    return {
        "union_detected_count": union,
        "both_detected_count": both,
        "one_detected_count": one,
        "unavailable_count": unavailable,
        "detection_agreement_on_union": both / max(union, 1),
        "exact_subtype_agreement_count": exact_subtype,
        "subtype_agreement_conditional_on_both": exact_subtype / max(both, 1),
    }


def _add_detection(
    counter: collections.Counter[str], role_a: str, role_b: str, category: frozenset[str]
) -> None:
    if "UNKNOWN" in {role_a, role_b}:
        counter["unavailable"] += 1
        return
    found_a, found_b = role_a in category, role_b in category
    if found_a or found_b:
        counter["union"] += 1
    if found_a and found_b:
        counter["both"] += 1
        counter["exact_subtype"] += int(role_a == role_b)
    elif found_a or found_b:
        counter["one"] += 1


def analyze(
    *,
    labels_path: Path,
    output_path: Path,
    code_commit: str,
    slurm_job_id: str = "",
    lock_output: bool = False,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    task_global = {task: collections.Counter() for task in TASK_NAMES}
    task_source = collections.defaultdict(
        lambda: {task: collections.Counter() for task in TASK_NAMES}
    )
    detection_global = {
        "heading": collections.Counter(),
        "context_line": collections.Counter(),
    }
    detection_source = collections.defaultdict(
        lambda: {
            "heading": collections.Counter(),
            "context_line": collections.Counter(),
        }
    )
    seen: set[tuple[str, str]] = set()
    line_count = 0
    for row in _iter_jsonl(labels_path):
        if row.get("schema_version") != LABEL_SCHEMA:
            raise ValueError("unsupported label row")
        identity = (str(row.get("document_id") or ""), str(row.get("line_id") or ""))
        if not all(identity) or identity in seen:
            raise ValueError(f"invalid/duplicate line identity: {identity}")
        seen.add(identity)
        line_count += 1
        source = str(row.get("source") or "")
        role_a = str(row.get("pass_a_role") or "")
        role_b = str(row.get("pass_b_role") or "")
        unavailable = "UNKNOWN" in {role_a, role_b}
        tasks = row.get("tasks")
        if not source or not isinstance(tasks, dict) or set(tasks) != set(TASK_NAMES):
            raise ValueError(f"invalid source/task inventory: {identity}")
        for task in TASK_NAMES:
            decision = tasks[task]
            if not isinstance(decision, dict) or set(decision) != {"label", "trusted"}:
                raise ValueError(f"invalid task decision: {identity} {task}")
            outcome = (
                "unavailable"
                if unavailable
                else ("agreement" if decision.get("trusted") is True else "disagreement")
            )
            task_global[task][outcome] += 1
            task_source[source][task][outcome] += 1
        for name, category in (("heading", HEADER_ROLES), ("context_line", CONTEXT_ROLES)):
            _add_detection(detection_global[name], role_a, role_b, category)
            _add_detection(detection_source[source][name], role_a, role_b, category)

    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "passed_repaired_retained_agreement_analysis",
        "slurm_job_id": slurm_job_id,
        "code_commit": code_commit,
        "line_count": line_count,
        "labels_sha256": sha256_file(labels_path),
        "denominator_semantics": {
            "task_agreement": "both raw repaired votes are non-UNKNOWN",
            "trusted_coverage": "all retained physical lines",
            "category_detection": "union where either comparable vote detects the category",
            "conditional_subtype": "lines where both comparable votes detect the category",
        },
        "task_metrics": {
            task: _task_summary(counter) for task, counter in task_global.items()
        },
        "category_metrics": {
            name: _detection_summary(counter)
            for name, counter in detection_global.items()
        },
        "by_source": {
            source: {
                "task_metrics": {
                    task: _task_summary(counter) for task, counter in tasks.items()
                },
                "category_metrics": {
                    name: _detection_summary(counter)
                    for name, counter in detection_source[source].items()
                },
            }
            for source, tasks in sorted(task_source.items())
        },
        "code_sha256": sha256_file(Path(__file__)),
    }
    _write_json_new(output_path, result, mode=0o440 if lock_output else 0o600)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    parser.add_argument("--lock-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        labels_path=args.labels.resolve(),
        output_path=args.output.resolve(),
        code_commit=args.code_commit,
        slurm_job_id=args.slurm_job_id,
        lock_output=args.lock_output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
