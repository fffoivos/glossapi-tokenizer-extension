#!/usr/bin/env python3
"""Decode out-of-fold entry probabilities into bibliography blocks.

B0 establishes blocks only from clusters of normal-length entry anchors.  H0
then attaches exact bibliography headings to already-confirmed blocks.  H0 can
never create a block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from .bibliography_entry_models import ALL_ARMS, Table, load_table


SCHEMA_VERSION = "bibliography-entry-b0-h0-oof-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BlockConfig:
    anchor_probability: float
    seed_length_limit: int
    anchors_required: int
    anchor_window: int
    maximum_bridge_gap: int
    inside_probability: float = 0.25
    adjacent_expansion: int = 1
    header_window: int = 2


def _physical_segments(abs_indices: np.ndarray) -> Iterable[tuple[int, int]]:
    start = 0
    for index in range(1, len(abs_indices)):
        if int(abs_indices[index]) - int(abs_indices[index - 1]) > MAX_PHYSICAL_GAP:
            yield start, index
            start = index
    if len(abs_indices):
        yield start, len(abs_indices)


def decode_b0_document(
    probability: np.ndarray,
    char_lengths: np.ndarray,
    abs_indices: np.ndarray,
    config: BlockConfig,
) -> np.ndarray:
    if not (len(probability) == len(char_lengths) == len(abs_indices)):
        raise ValueError("B0 document arrays have different lengths")
    predicted = np.zeros(len(probability), dtype=bool)
    for segment_start, segment_end in _physical_segments(abs_indices):
        local_probability = probability[segment_start:segment_end]
        local_lengths = char_lengths[segment_start:segment_end]
        anchors = np.flatnonzero(
            (local_probability >= config.anchor_probability)
            & (local_lengths <= config.seed_length_limit)
        )
        seed_spans: list[tuple[int, int]] = []
        count = int(config.anchors_required)
        for offset in range(max(0, len(anchors) - count + 1)):
            group = anchors[offset : offset + count]
            if int(group[-1]) - int(group[0]) <= config.anchor_window:
                seed_spans.append((int(group[0]), int(group[-1])))
        if not seed_spans:
            continue
        merged: list[list[int]] = []
        for start, end in seed_spans:
            if merged and start - merged[-1][1] - 1 <= config.maximum_bridge_gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            # Everything between independently established anchors is inside,
            # including long and locally weak continuation lines.
            predicted[segment_start + start : segment_start + end + 1] = True
            for distance in range(1, config.adjacent_expansion + 1):
                left, right = start - distance, end + distance
                if left >= 0 and local_probability[left] >= config.inside_probability:
                    predicted[segment_start + left] = True
                if right < len(local_probability) and local_probability[right] >= config.inside_probability:
                    predicted[segment_start + right] = True
    return predicted


def attach_h0_document(
    b0_prediction: np.ndarray,
    probability: np.ndarray,
    header_kinds: np.ndarray,
    abs_indices: np.ndarray,
    config: BlockConfig,
) -> np.ndarray:
    """Attach headings to existing blocks without creating new blocks."""

    result = b0_prediction.copy()
    physical_break = np.concatenate(
        (
            [True],
            np.diff(abs_indices.astype(np.int64)) > MAX_PHYSICAL_GAP,
        )
    )
    starts = np.flatnonzero(
        result & (physical_break | ~np.concatenate(([False], result[:-1])))
    )
    for start in starts:
        for distance in range(1, config.header_window + 1):
            candidate = int(start) - distance
            if candidate < 0:
                break
            if int(abs_indices[start]) - int(abs_indices[candidate]) > MAX_PHYSICAL_GAP:
                break
            if header_kinds[candidate] <= 0:
                continue
            between = slice(candidate + 1, int(start))
            if candidate + 1 < start and not np.all(
                (probability[between] >= config.inside_probability)
                | (header_kinds[between] > 0)
            ):
                continue
            result[candidate : int(start)] = True
            break
    return result


def decode_table(
    table: Table, probability: np.ndarray, config: BlockConfig, *, attach_headers: bool
) -> np.ndarray:
    if len(probability) != len(table.targets) or not np.isfinite(probability).all():
        raise ValueError("OOF probability must be finite for every emitted line")
    prediction = np.zeros(len(probability), dtype=bool)
    for document in table.documents:
        start, end = int(document["line_start"]), int(document["line_end"])
        local = decode_b0_document(
            probability[start:end],
            table.char_lengths[start:end],
            table.abs_indices[start:end],
            config,
        )
        if attach_headers:
            local = attach_h0_document(
                local,
                probability[start:end],
                table.header_kinds[start:end],
                table.abs_indices[start:end],
                config,
            )
        prediction[start:end] = local
    return prediction


def blocks_from_mask(mask: np.ndarray, abs_indices: np.ndarray) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(mask):
        gap = index > 0 and int(abs_indices[index]) - int(abs_indices[index - 1]) > MAX_PHYSICAL_GAP
        if enabled and (start is None or gap):
            if start is not None:
                blocks.append((start, index - 1))
            start = index
        elif not enabled and start is not None:
            blocks.append((start, index - 1))
            start = None
    if start is not None:
        blocks.append((start, len(mask) - 1))
    return blocks


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = left[1] - left[0] + right[1] - right[0] + 2 - intersection
    return intersection / union if union else 0.0


def _block_matches(
    gold: Sequence[tuple[int, int]], predicted: Sequence[tuple[int, int]], threshold: float
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (_iou(gold_block, predicted_block), gold_index, predicted_index)
            for gold_index, gold_block in enumerate(gold)
            for predicted_index, predicted_block in enumerate(predicted)
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_prediction: set[int] = set()
    matches = []
    for score, gold_index, predicted_index in candidates:
        if score < threshold:
            break
        if gold_index in used_gold or predicted_index in used_prediction:
            continue
        used_gold.add(gold_index)
        used_prediction.add(predicted_index)
        matches.append((gold_index, predicted_index, score))
    return matches


def _ratio(numerator: int | float, denominator: int | float, *, empty: float) -> float:
    return float(numerator / denominator) if denominator else empty


def evaluate_prediction(
    table: Table, prediction: np.ndarray, *, document_subset: set[int] | None = None
) -> dict[str, Any]:
    gold_mask = table.original_labels == LABEL_TO_ID["BIB"]
    selected_docs = set(range(len(table.documents))) if document_subset is None else document_subset
    line_tp = line_fp = line_fn = 0
    token_tp = token_fp = token_fn = 0
    gold_blocks_total = predicted_blocks_total = 0
    exact_matches = iou_matches = 0
    boundary_errors: list[float] = []
    zero_docs = spurious_zero_blocks = 0
    split_errors = merge_errors = 0
    docs_false_deletion = 0
    longest_false_token_run = 0
    true_long = recovered_true_long = false_long = 0

    for document_index, document in enumerate(table.documents):
        if document_index not in selected_docs:
            continue
        start, end = int(document["line_start"]), int(document["line_end"])
        gold = gold_mask[start:end]
        pred = prediction[start:end]
        tokens = table.token_counts[start:end].astype(np.int64)
        line_tp += int(np.count_nonzero(gold & pred))
        line_fp += int(np.count_nonzero(~gold & pred))
        line_fn += int(np.count_nonzero(gold & ~pred))
        token_tp += int(tokens[gold & pred].sum())
        token_fp += int(tokens[~gold & pred].sum())
        token_fn += int(tokens[gold & ~pred].sum())
        docs_false_deletion += int(np.any(~gold & pred))
        run = 0
        for enabled, weight in zip(~gold & pred, tokens, strict=True):
            run = run + int(weight) if enabled else 0
            longest_false_token_run = max(longest_false_token_run, run)
        long = table.char_lengths[start:end] > 330
        true_long += int(np.count_nonzero(gold & long))
        recovered_true_long += int(np.count_nonzero(gold & long & pred))
        false_long += int(np.count_nonzero(~gold & long & pred))

        gold_blocks = blocks_from_mask(gold, table.abs_indices[start:end])
        predicted_blocks = blocks_from_mask(pred, table.abs_indices[start:end])
        gold_blocks_total += len(gold_blocks)
        predicted_blocks_total += len(predicted_blocks)
        exact_matches += len(_block_matches(gold_blocks, predicted_blocks, 1.0))
        matches = _block_matches(gold_blocks, predicted_blocks, 0.5)
        iou_matches += len(matches)
        for gold_index, predicted_index, _score in matches:
            g, p = gold_blocks[gold_index], predicted_blocks[predicted_index]
            boundary_errors.append((abs(g[0] - p[0]) + abs(g[1] - p[1])) / 2)
        split_errors += sum(
            sum(_iou(gold_block, predicted_block) > 0 for predicted_block in predicted_blocks) > 1
            for gold_block in gold_blocks
        )
        merge_errors += sum(
            sum(_iou(gold_block, predicted_block) > 0 for gold_block in gold_blocks) > 1
            for predicted_block in predicted_blocks
        )
        if not gold_blocks:
            zero_docs += 1
            spurious_zero_blocks += len(predicted_blocks)

    line_precision = _ratio(line_tp, line_tp + line_fp, empty=1.0)
    line_recall = _ratio(line_tp, line_tp + line_fn, empty=0.0)
    token_precision = _ratio(token_tp, token_tp + token_fp, empty=1.0)
    token_recall = _ratio(token_tp, token_tp + token_fn, empty=0.0)
    beta2 = 0.25
    token_f05 = _ratio((1 + beta2) * token_precision * token_recall, beta2 * token_precision + token_recall, empty=0.0)
    return {
        "document_count": len(selected_docs),
        "line_precision": line_precision,
        "line_recall": line_recall,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f0_5": token_f05,
        "line_tp": line_tp,
        "line_fp": line_fp,
        "line_fn": line_fn,
        "token_tp": token_tp,
        "token_fp": token_fp,
        "token_fn": token_fn,
        "gold_block_count": gold_blocks_total,
        "predicted_block_count": predicted_blocks_total,
        "exact_block_precision": _ratio(exact_matches, predicted_blocks_total, empty=1.0),
        "exact_block_recall": _ratio(exact_matches, gold_blocks_total, empty=0.0),
        "iou50_block_precision": _ratio(iou_matches, predicted_blocks_total, empty=1.0),
        "iou50_block_recall": _ratio(iou_matches, gold_blocks_total, empty=0.0),
        "mean_boundary_error_emitted_lines": float(np.mean(boundary_errors)) if boundary_errors else None,
        "spurious_blocks_per_zero_block_document": _ratio(spurious_zero_blocks, zero_docs, empty=0.0),
        "split_error_count": split_errors,
        "merge_error_count": merge_errors,
        "documents_with_false_deletion_fraction": _ratio(docs_false_deletion, len(selected_docs), empty=0.0),
        "longest_consecutive_false_positive_tokens": longest_false_token_run,
        "true_long_line_count": true_long,
        "true_long_line_recall": _ratio(recovered_true_long, true_long, empty=0.0),
        "false_positive_long_line_count": false_long,
    }


def _selection_key(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    eligible = (
        float(metrics["line_precision"]) >= 0.99
        and float(metrics["spurious_blocks_per_zero_block_document"]) <= 0.02
    )
    return (
        int(eligible),
        float(metrics["token_recall"]) if eligible else float(metrics["token_f0_5"]),
        float(metrics["iou50_block_recall"]),
        float(metrics["token_precision"]),
        -int(metrics["predicted_block_count"]),
    )


def _grid() -> list[BlockConfig]:
    return [
        BlockConfig(anchor, length, count, window, bridge)
        for anchor in (0.70, 0.85, 0.95)
        for length in (280, 330, 380)
        for count, window in ((2, 3), (3, 5))
        for bridge in (1, 2)
    ]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _breakdowns(table: Table, prediction: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("source", "coverage"):
        grouped = {}
        values = sorted({str(document[key]) for document in table.documents})
        for value in values:
            subset = {index for index, document in enumerate(table.documents) if str(document[key]) == value}
            grouped[value] = evaluate_prediction(table, prediction, document_subset=subset)
        result[f"by_{key}"] = grouped
    return result


def _h0_attachment_metrics(
    table: Table, b0: np.ndarray, h0: np.ndarray
) -> dict[str, Any]:
    attached = h0 & ~b0
    gold_structural = (table.original_labels == LABEL_TO_ID["BIB"]) & (
        table.header_kinds > 0
    )
    tp = int(np.count_nonzero(attached & gold_structural))
    fp = int(np.count_nonzero(attached & ~gold_structural))
    fn = int(np.count_nonzero(gold_structural & ~h0))
    return {
        "attached_line_count": int(np.count_nonzero(attached)),
        "gold_exact_structural_line_count": int(np.count_nonzero(gold_structural)),
        "precision": _ratio(tp, tp + fp, empty=1.0),
        "recall": _ratio(tp, tp + fn, empty=0.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir)
    line_root = Path(args.line_oof_dir).resolve()
    line_report = json.loads((line_root / "line_oof_report.json").read_text(encoding="utf-8"))
    if line_report.get("validation_opened") is not False:
        raise ValueError("line-model receipt does not prove validation remained unopened")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    output_dir.mkdir(parents=True)

    arm_rows: dict[str, Any] = {}
    for arm in ALL_ARMS:
        probability = np.load(line_root / f"{arm}.oof_probability.npy", mmap_mode="r", allow_pickle=False)
        candidates = []
        best: tuple[tuple[Any, ...], BlockConfig, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]] | None = None
        for config in _grid():
            b0 = decode_table(table, probability, config, attach_headers=False)
            h0 = decode_table(table, probability, config, attach_headers=True)
            b0_metrics = evaluate_prediction(table, b0)
            h0_metrics = evaluate_prediction(table, h0)
            row = {
                "config": asdict(config),
                "b0": b0_metrics,
                "b0_plus_h0": h0_metrics,
                "h0_attachment": _h0_attachment_metrics(table, b0, h0),
            }
            candidates.append(row)
            bundle = (_selection_key(h0_metrics), config, b0, h0, b0_metrics, h0_metrics)
            if best is None or bundle[0] > best[0]:
                best = bundle
        assert best is not None
        _key, config, b0, h0, b0_metrics, h0_metrics = best
        _save_array(output_dir / f"{arm}.b0_prediction.npy", b0)
        _save_array(output_dir / f"{arm}.b0_h0_prediction.npy", h0)
        arm_rows[arm] = {
            "selected_config": asdict(config),
            "selected_b0_metrics": b0_metrics,
            "selected_b0_plus_h0_metrics": h0_metrics,
            "selected_h0_attachment_metrics": _h0_attachment_metrics(table, b0, h0),
            **_breakdowns(table, h0),
            "grid": candidates,
        }

    ranked = sorted(ALL_ARMS, key=lambda arm: _selection_key(arm_rows[arm]["selected_b0_plus_h0_metrics"]), reverse=True)
    best_recall = float(arm_rows[ranked[0]]["selected_b0_plus_h0_metrics"]["token_recall"])
    simplicity = {arm: index for index, arm in enumerate(("L0", "L1", "L3", "L2", "L4", "D1"))}
    near_best = [
        arm
        for arm in ranked
        if best_recall - float(arm_rows[arm]["selected_b0_plus_h0_metrics"]["token_recall"]) <= 0.005
        and float(arm_rows[arm]["selected_b0_plus_h0_metrics"]["line_precision"]) >= 0.99
        and float(arm_rows[arm]["selected_b0_plus_h0_metrics"]["spurious_blocks_per_zero_block_document"]) <= 0.02
    ]
    primary = min(near_best, key=lambda arm: simplicity[arm]) if near_best else ranked[0]
    retained_for_b1 = [primary] + [arm for arm in ranked if arm != primary][:1]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_oof_b0_h0_only_validation_unopened",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "feature_table_manifest_sha256": _sha256(table.root / "manifest.json"),
        "line_oof_report_sha256": _sha256(line_root / "line_oof_report.json"),
        "grid_size_per_arm": len(_grid()),
        "selection_gate": {"minimum_line_precision": 0.99, "maximum_spurious_blocks_per_zero_document": 0.02},
        "primary_arm": primary,
        "retained_for_b1": retained_for_b1,
        "arms": arm_rows,
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "block_oof_report.json", report)
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
