#!/usr/bin/env python3
"""Split anchored signal-TCN blocks at explicit structural/prose barriers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import (
    has_auxiliary_scope,
    materialize_auxiliary_headings,
)
from .bibliography_deterministic_roles import ROLE_NAMES, SCHEMA_VERSION as ROLE_SCHEMA
from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    decode_b0_document,
    evaluate_prediction,
)
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import _load_quality_exclusions
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_entry_role_sequence import _validate_role_matrix
from .bibliography_signal_tcn import SCHEMA_VERSION as SIGNAL_SCHEMA


SCHEMA_VERSION = "bibliography-signal-barrier-decode-oof-v1"
BARRIER_ARMS = {
    "none": (),
    "headings": (
        "exact_negative_scope_heading",
        "generic_markdown_heading",
    ),
    "headings_figure": (
        "exact_negative_scope_heading",
        "generic_markdown_heading",
        "figure_caption",
    ),
    "headings_figure_footnote": (
        "exact_negative_scope_heading",
        "generic_markdown_heading",
        "figure_caption",
        "footnote",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def sustained_low_mask(
    probability: np.ndarray, *, threshold: float, minimum_run: int
) -> np.ndarray:
    """Mark complete runs of several very-low contextual scores."""

    result = np.zeros(len(probability), dtype=bool)
    if minimum_run <= 0:
        return result
    low = probability < float(threshold)
    start = 0
    while start < len(low):
        if not low[start]:
            start += 1
            continue
        end = start + 1
        while end < len(low) and low[end]:
            end += 1
        if end - start >= int(minimum_run):
            result[start:end] = True
        start = end
    return result


def build_barrier_mask(
    probability: np.ndarray,
    roles: np.ndarray,
    *,
    barrier_arm: str,
    low_probability: float,
    minimum_low_run: int,
) -> np.ndarray:
    if barrier_arm not in BARRIER_ARMS:
        raise ValueError(f"unknown barrier arm {barrier_arm!r}")
    indices = [ROLE_NAMES.index(name) for name in BARRIER_ARMS[barrier_arm]]
    structural = (
        np.any(roles[:, indices] > 0, axis=1)
        if indices
        else np.zeros(len(probability), dtype=bool)
    )
    return structural | sustained_low_mask(
        probability,
        threshold=float(low_probability),
        minimum_run=int(minimum_low_run),
    )


def _decode_between_barriers(
    probability: np.ndarray,
    abs_indices: np.ndarray,
    barrier: np.ndarray,
    config: BlockConfig,
) -> np.ndarray:
    result = np.zeros(len(probability), dtype=bool)
    start = 0
    while start < len(probability):
        while start < len(probability) and barrier[start]:
            start += 1
        end = start
        while end < len(probability) and not barrier[end]:
            end += 1
        if end > start:
            result[start:end] = decode_b0_document(
                probability[start:end],
                np.zeros(end - start, dtype=np.uint8),
                abs_indices[start:end],
                config,
            )
        start = max(end, start + 1)
    return result


def decode_barrier_blocks(
    table: Any,
    signal_probability: np.ndarray,
    frozen_entry_probability: np.ndarray,
    roles: np.ndarray,
    auxiliary_scope: np.ndarray,
    config: BlockConfig,
    *,
    barrier_arm: str,
    low_probability: float,
    minimum_low_run: int,
    qualified_documents: set[int],
    apply_veto: bool,
) -> tuple[np.ndarray, int, int]:
    prediction = np.zeros(len(table.targets), dtype=bool)
    vetoed = 0
    barrier_lines = 0
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        local_absolute = table.abs_indices[start:end]
        barrier = build_barrier_mask(
            signal_probability[start:end],
            roles[start:end],
            barrier_arm=barrier_arm,
            low_probability=low_probability,
            minimum_low_run=minimum_low_run,
        )
        barrier_lines += int(np.count_nonzero(barrier))
        local = _decode_between_barriers(
            signal_probability[start:end], local_absolute, barrier, config
        )
        if apply_veto:
            local_scope = auxiliary_scope[start:end]
            for block_start, block_end in blocks_from_mask(local, local_absolute):
                if has_auxiliary_scope(
                    local_scope,
                    local_absolute,
                    block_start,
                    end=block_end,
                    window=config.header_window,
                ):
                    local[block_start : block_end + 1] = False
                    vetoed += 1
        local = attach_h0_document(
            local,
            frozen_entry_probability[start:end],
            table.header_kinds[start:end],
            local_absolute,
            config,
        )
        prediction[start:end] = local
    return prediction, vetoed, barrier_lines


def _evaluate_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        signal_path,
        frozen_path,
        role_path,
        scope_path,
        qualified_documents,
        config_payload,
        barrier_arm,
        low_probability,
        minimum_low_run,
    ) = task
    table = load_table(table_dir, expected_split="train")
    signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    frozen = np.load(frozen_path, mmap_mode="r", allow_pickle=False)
    roles = _validate_role_matrix(Path(role_path), len(table.targets))
    scope = np.load(scope_path, mmap_mode="r", allow_pickle=False)
    config = BlockConfig(**config_payload)
    prediction, vetoed, barrier_lines = decode_barrier_blocks(
        table,
        signal,
        frozen,
        roles,
        scope,
        config,
        barrier_arm=barrier_arm,
        low_probability=float(low_probability),
        minimum_low_run=int(minimum_low_run),
        qualified_documents=set(qualified_documents),
        apply_veto=True,
    )
    return {
        "config": asdict(config),
        "barrier_arm": barrier_arm,
        "low_probability": float(low_probability),
        "minimum_low_run": int(minimum_low_run),
        "barrier_line_count": barrier_lines,
        "vetoed_component_count": vetoed,
        "metrics": evaluate_prediction(
            table, prediction, document_subset=set(qualified_documents)
        ),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        -float(row["config"]["anchor_probability"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    if (
        signal_report.get("schema_version") != SIGNAL_SCHEMA
        or signal_report.get("validation_opened") is not False
    ):
        raise ValueError("barrier decode requires validation-isolated TCN scores")
    roles_root = Path(args.deterministic_roles_dir).resolve()
    role_report_path = roles_root / "deterministic_roles_report.json"
    role_report = json.loads(role_report_path.read_text(encoding="utf-8"))
    if (
        role_report.get("schema_version") != ROLE_SCHEMA
        or tuple(role_report.get("role_names", ())) != ROLE_NAMES
        or role_report.get("validation_opened") is not False
    ):
        raise ValueError("barrier decode requires frozen train-only roles")
    role_path = roles_root / "negative_roles.npy"
    _validate_role_matrix(role_path, len(table.targets))
    signal_path = signal_root / "signal_tcn_oof_probability.npy"
    frozen_path = Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    excluded_ids, quality_packet = _load_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    auxiliary_headings, auxiliary_scope = materialize_auxiliary_headings(
        table, Path(args.input).resolve()
    )
    if np.any(
        auxiliary_scope & (table.original_labels == LABEL_TO_ID["BIB"])
    ):
        raise ValueError("audited auxiliary scope now overlaps silver bibliography")
    scope_path = output_dir / "auxiliary_scope_active.npy"
    _save_array(scope_path, auxiliary_scope)

    configs = [
        BlockConfig(
            anchor_probability=float(anchor),
            seed_length_limit=1,
            anchors_required=2,
            anchor_window=16,
            maximum_bridge_gap=16,
            inside_probability=float(inside),
            adjacent_expansion=int(expansion),
            header_window=2,
        )
        for anchor, inside, expansion in itertools.product(
            args.anchor_probabilities,
            args.inside_probabilities,
            args.adjacent_expansions,
        )
        if inside < anchor
    ]
    barrier_settings = [
        (arm, float(low), int(run))
        for arm, (low, run) in itertools.product(
            args.barrier_arms,
            ((0.0, 0), (0.05, 2), (0.10, 2), (0.20, 2), (0.10, 3)),
        )
    ]
    tasks = [
        (
            str(Path(args.table_dir).resolve()),
            str(signal_path),
            str(frozen_path),
            str(role_path),
            str(scope_path),
            tuple(sorted(qualified_documents)),
            asdict(config),
            arm,
            low,
            run,
        )
        for config, (arm, low, run) in itertools.product(configs, barrier_settings)
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        rows = list(executor.map(_evaluate_task, tasks, chunksize=1))
    safe = [row for row in rows if is_safe_candidate(row)]
    selected = max(safe, key=_selection_key) if safe else None
    p95 = [row for row in rows if row["metrics"]["line_precision"] >= 0.95]
    diagnostic_p95 = max(p95, key=_selection_key) if p95 else None
    diagnostic_highest = max(rows, key=_selection_key)
    if selected is not None:
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        frozen = np.load(frozen_path, mmap_mode="r", allow_pickle=False)
        roles = _validate_role_matrix(role_path, len(table.targets))
        prediction, _vetoed, _barrier_lines = decode_barrier_blocks(
            table,
            signal,
            frozen,
            roles,
            auxiliary_scope,
            BlockConfig(**selected["config"]),
            barrier_arm=selected["barrier_arm"],
            low_probability=float(selected["low_probability"]),
            minimum_low_run=int(selected["minimum_low_run"]),
            qualified_documents=qualified_documents,
            apply_veto=True,
        )
        _save_array(output_dir / "selected_oof_prediction.npy", prediction)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "barrier_contract": "explicit stable-negative structural roles and sustained runs of very-low contextual probability split proposal regions; barriers cannot create or expand deletion",
        "barrier_arms": {name: list(values) for name, values in BARRIER_ARMS.items()},
        "grid": {
            "anchor_probabilities": list(args.anchor_probabilities),
            "inside_probabilities": list(args.inside_probabilities),
            "adjacent_expansions": list(args.adjacent_expansions),
            "candidate_count": len(tasks),
        },
        "candidates": rows,
        "safe_candidate_count": len(safe),
        "selected": selected,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_p95,
        "diagnostic_highest_recall_candidate": diagnostic_highest,
        "selection_rule": "retain raw-silver metrics for comparability; require line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document; label-completeness review remains separate",
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "source": _sha256(Path(args.input).resolve()),
            "signal_report": _sha256(signal_report_path),
            "signal_probability": _sha256(signal_path),
            "frozen_probability": _sha256(frozen_path),
            "role_report": _sha256(role_report_path),
            "role_matrix": _sha256(role_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "signal_barrier_decode_oof_report.json", result)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--deterministic-roles-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument(
        "--anchor-probabilities",
        type=float,
        nargs="+",
        default=(0.90, 0.925, 0.95, 0.97, 0.98, 0.99),
    )
    parser.add_argument(
        "--inside-probabilities", type=float, nargs="+", default=(0.20, 0.60)
    )
    parser.add_argument(
        "--adjacent-expansions", type=int, nargs="+", default=(1, 2)
    )
    parser.add_argument(
        "--barrier-arms", nargs="+", choices=tuple(BARRIER_ARMS), default=tuple(BARRIER_ARMS)
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
