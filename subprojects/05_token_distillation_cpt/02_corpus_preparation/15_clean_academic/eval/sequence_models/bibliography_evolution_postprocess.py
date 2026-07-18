#!/usr/bin/env python3
"""Single-component postprocessors for the controlled evolution queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask, evaluate_prediction
from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .bibliography_entry_models import load_table
from .bibliography_evolution_headers import (
    HeaderControllerConfig,
    apply_header_controller,
    assert_header_invariants,
    predecoder_walls,
)


SCHEMA_VERSION = "bibliography-evolution-single-postprocess-v1"
OPERATIONS = (
    "header_controller",
    "internal_gap_connection",
    "boundary_trim",
    "outward_edge",
    "weak_unseeded",
    "whole_component_veto",
)
POSTPROCESS_ORDER = (
    "internal_gap_connection",
    "boundary_trim",
    "outward_edge",
    "weak_unseeded",
    "whole_component_veto",
)
TRACE_NAMES = {
    "internal_gap_connection": "internal_gap_connection",
    "boundary_trim": "boundary_trim",
    "outward_edge": "outward_edge_optional",
    "weak_unseeded": "weak_unseeded_optional",
    "whole_component_veto": "whole_component_veto",
}
# Canonical G3 reference pipeline.  Every G3 candidate executes every stage in
# this order and changes only its declared stage.  These are active reference
# settings, not identity/no-op placeholders.
REFERENCE_PARAMETERS = {
    "internal_gap_connection": {"threshold": 0.20, "max_lines": 2},
    "boundary_trim": {"threshold": 0.05, "max_lines": 1},
    "outward_edge": {"threshold": 0.40, "max_lines": 1},
    "weak_unseeded": {"threshold": 0.20, "max_lines": 1},
    "whole_component_veto": {"threshold": 0.02, "max_lines": 1},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_array(path: str | None, length: int, *, dtype: Any) -> np.ndarray:
    if path is None:
        return np.zeros(length, dtype=dtype)
    value = np.load(path, allow_pickle=False)
    if value.shape != (length,):
        raise ValueError(f"array does not align: {path}")
    return value.astype(dtype, copy=False)


def _qualified(table: Any, path: Path) -> tuple[set[int], dict[str, set[int]]]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    ids = packet.get("document_ids")
    if not isinstance(ids, list) or len(ids) != 268 or len(ids) != len(set(ids)):
        raise ValueError("qualified development inventory must contain 268 unique ids")
    index = {str(row["document_id"]): number for number, row in enumerate(table.documents)}
    if not set(ids).issubset(index):
        raise ValueError("qualified inventory is not represented in the table")
    selected = {index[value] for value in ids}
    by_source: dict[str, set[int]] = {}
    for document_index in selected:
        source = str(table.documents[document_index]["source"])
        by_source.setdefault(source, set()).add(document_index)
    return selected, by_source


def _can_cross(
    absolute: np.ndarray,
    hard_wall: np.ndarray,
    upward_stop: np.ndarray,
    downward_stop: np.ndarray,
    left: int,
    right: int,
) -> bool:
    return (
        0 <= left < len(absolute)
        and 0 <= right < len(absolute)
        and not hard_wall[left]
        and not hard_wall[right]
        and not upward_stop[right]
        and not downward_stop[left]
        and int(absolute[right]) - int(absolute[left]) <= MAX_PHYSICAL_GAP
    )


def _postprocess_document(
    prediction: np.ndarray,
    signal: np.ndarray,
    absolute: np.ndarray,
    hard_wall: np.ndarray,
    upward_stop: np.ndarray,
    downward_stop: np.ndarray,
    header_roles: np.ndarray,
    *,
    operation: str,
    threshold: float,
    max_lines: int,
) -> np.ndarray:
    result = prediction.astype(bool, copy=True)
    result[hard_wall] = False
    if operation == "header_controller":
        result = apply_header_controller(
            result,
            header_roles,
            absolute,
            hard_wall_mask=hard_wall,
            config=HeaderControllerConfig(
                attachment_window=max_lines, connector_window=max_lines
            ),
        )
        assert_header_invariants(result, header_roles)
        return result
    if operation in {"internal_gap_connection", "weak_unseeded"}:
        blocks = blocks_from_mask(result, absolute)
        for (_, left_end), (right_start, _) in zip(blocks, blocks[1:]):
            gap_start, gap_end = left_end + 1, right_start
            if gap_end - gap_start > max_lines or gap_end <= gap_start:
                continue
            if any(
                not _can_cross(absolute, hard_wall, upward_stop, downward_stop, index - 1, index)
                for index in range(gap_start, gap_end + 1)
            ):
                continue
            values = signal[gap_start:gap_end]
            score = float(values.mean()) if operation == "internal_gap_connection" else float(values.max(initial=0.0))
            if score >= threshold:
                result[gap_start:gap_end] = True
        return result
    if operation == "boundary_trim":
        for start, end in blocks_from_mask(result, absolute):
            cursor = start
            removed = 0
            while cursor <= end and removed < max_lines and signal[cursor] < threshold:
                result[cursor] = False
                cursor += 1
                removed += 1
            cursor = end
            removed = 0
            while cursor >= start and removed < max_lines and signal[cursor] < threshold:
                result[cursor] = False
                cursor -= 1
                removed += 1
        return result
    if operation == "outward_edge":
        for start, end in blocks_from_mask(result, absolute):
            for distance in range(1, max_lines + 1):
                left = start - distance
                if left < 0 or not _can_cross(absolute, hard_wall, upward_stop, downward_stop, left, left + 1) or signal[left] < threshold:
                    break
                result[left] = True
            for distance in range(1, max_lines + 1):
                right = end + distance
                if right >= len(result) or not _can_cross(absolute, hard_wall, upward_stop, downward_stop, right - 1, right) or signal[right] < threshold:
                    break
                result[right] = True
        return result
    if operation == "whole_component_veto":
        for start, end in blocks_from_mask(result, absolute):
            if float(signal[start : end + 1].mean()) < threshold:
                result[start : end + 1] = False
        return result
    raise ValueError(f"unknown operation: {operation}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="validation")
    n = len(table.targets)
    baseline = _load_array(args.baseline_prediction, n, dtype=bool)
    signal = _load_array(args.signal_probability, n, dtype=np.float32)
    scope = _load_array(args.scope_mask, n, dtype=bool)
    header_roles = _load_array(args.header_roles, n, dtype=np.uint8)
    upward_stop = np.zeros(n, dtype=bool)
    downward_stop = np.zeros(n, dtype=bool)
    if args.barrier_artifact:
        packet = np.load(args.barrier_artifact, allow_pickle=False)
        for name in ("hard_wall", "upward_stop", "downward_stop"):
            if name not in packet or packet[name].shape != (n,):
                raise ValueError("barrier artifact does not align")
        scope |= packet["hard_wall"].astype(bool)
        upward_stop |= packet["upward_stop"].astype(bool)
        downward_stop |= packet["downward_stop"].astype(bool)
    if args.operation == "header_controller":
        local_upward, local_downward = predecoder_walls(header_roles)
        upward_stop |= local_upward
        downward_stop |= local_downward
        # NON_BIB_HEADER remains excluded in every descendant, not merely in
        # this pass.  Persist it as a symmetric hard wall as well as a
        # directional stop so a G3 filler cannot enter or leave the line.
        scope |= local_downward
    selected, by_source = _qualified(table, Path(args.qualified_documents))
    prediction = np.zeros(n, dtype=bool)
    trace: list[dict[str, Any]] = []
    if args.operation == "header_controller":
        stages = ("header_controller",)
    else:
        # Every G3 run executes the same active reference pipeline.  The one
        # declared experimental stage substitutes its swept parameters; all
        # other stages retain their frozen reference parameters.
        stages = POSTPROCESS_ORDER
    current = baseline.astype(bool, copy=True)
    for stage_index, stage in enumerate(stages):
        enabled = stage == args.operation
        parameters = dict(REFERENCE_PARAMETERS.get(stage, {}))
        if enabled:
            parameters = {"threshold": float(args.threshold), "max_lines": int(args.max_lines)}
        next_prediction = np.zeros(n, dtype=bool)
        for document_index in sorted(selected):
            document = table.documents[document_index]
            start, end = int(document["line_start"]), int(document["line_end"])
            next_prediction[start:end] = _postprocess_document(
                current[start:end],
                signal[start:end],
                table.abs_indices[start:end],
                scope[start:end],
                upward_stop[start:end],
                downward_stop[start:end],
                header_roles[start:end],
                operation=stage,
                threshold=float(parameters["threshold"]),
                max_lines=int(parameters["max_lines"]),
            )
        current = next_prediction
        trace.append(
            {
                "position": stage_index,
                "module": TRACE_NAMES.get(stage, stage),
                "operation": stage,
                "status": "enabled_changed_family" if enabled else "executed_fixed_reference",
                "threshold": float(parameters["threshold"]),
                "max_lines": int(parameters["max_lines"]),
            }
        )
    prediction[:] = current
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    prediction_path = output / "prediction.npy"
    with prediction_path.open("xb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    barrier_path = output / "combined_barriers.npz"
    with barrier_path.open("xb") as handle:
        np.savez(
            handle,
            hard_wall=scope.astype(bool),
            upward_stop=upward_stop.astype(bool),
            downward_stop=downward_stop.astype(bool),
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_single_component_development_evaluation",
        "validation_opened": True,
        "final_test_opened": False,
        "operation": args.operation,
        "module_trace": trace,
        "threshold": float(args.threshold),
        "max_lines": int(args.max_lines),
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "baseline_prediction": _sha256(Path(args.baseline_prediction)),
            "signal_probability": _sha256(Path(args.signal_probability)),
            "scope_mask": _sha256(Path(args.scope_mask)),
            "qualified_documents": _sha256(Path(args.qualified_documents)),
            "header_roles": _sha256(Path(args.header_roles)) if args.header_roles else None,
            "barrier_artifact": _sha256(Path(args.barrier_artifact)) if args.barrier_artifact else None,
        },
        "metrics": evaluate_prediction(table, prediction, document_subset=selected),
        "metrics_by_source": {
            source: evaluate_prediction(table, prediction, document_subset=documents)
            for source, documents in sorted(by_source.items())
        },
        "prediction_sha256": _sha256(prediction_path),
    }
    _write_json_new(output / "report.json", result)
    outputs = {
        path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _write_json_new(output / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--baseline-prediction", required=True)
    parser.add_argument("--signal-probability", required=True)
    parser.add_argument("--scope-mask", required=True)
    parser.add_argument("--header-roles")
    parser.add_argument("--barrier-artifact")
    parser.add_argument("--qualified-documents", required=True)
    parser.add_argument("--operation", choices=OPERATIONS, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-lines", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
