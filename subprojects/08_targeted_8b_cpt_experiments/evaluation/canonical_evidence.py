"""Fail-closed readers for immutable canonical-campaign evaluation evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require


def completed_result(
    run_root: Path,
    *,
    evaluator_id: str,
    iteration: int,
    schema: str,
    scale: str,
) -> tuple[Path, dict[str, Any]]:
    """Return the unique successful result for one canonical evaluation target."""

    attempts_root = (
        run_root.resolve()
        / "evaluations"
        / evaluator_id
        / f"iter_{iteration:07d}"
        / "attempts"
    )
    require(attempts_root.is_dir(), f"canonical evaluation attempts missing: {attempts_root}")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for attempt_root in sorted(attempts_root.glob("attempt_*")):
        result_path = attempt_root / "result.json"
        execution_path = attempt_root / "evaluation.json"
        if not result_path.is_file() or not execution_path.is_file():
            continue
        result = read_json(result_path)
        execution = read_json(execution_path)
        if not (
            result.get("schema_version") == schema
            and result.get("status") == "completed"
            and result.get("evaluator_id") == evaluator_id
            and int(result.get("iteration", -1)) == iteration
            and result.get("scale") == scale
            and execution.get("schema_version") == "apertus_campaign_evaluation_attempt_v2"
            and execution.get("status") == "completed"
            and execution.get("result") == file_binding(result_path)
            and all(execution.get("result_checks", {}).values())
        ):
            continue
        matches.append((result_path, result))
    require(
        len(matches) == 1,
        f"expected one completed {evaluator_id} result for {scale}@{iteration}; found {len(matches)}",
    )
    return matches[0]

