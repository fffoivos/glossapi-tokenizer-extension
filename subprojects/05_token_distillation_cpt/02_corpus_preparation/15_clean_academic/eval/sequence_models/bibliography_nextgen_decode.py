#!/usr/bin/env python3
"""Decode next-generation line probabilities into topology-safe BIB blocks."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask, evaluate_prediction
from .bibliography_auxiliary_scope_veto import has_auxiliary_scope
from .bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from .bibliography_entry_models import load_table
from .bibliography_nextgen_models import SCHEMA_VERSION as MODEL_SCHEMA
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-block-oof-v1"


@dataclass(frozen=True)
class DecoderConfig:
    anchor_probability: float
    inside_probability: float
    anchor_window: int
    maximum_bridge_gap: int
    adjacent_expansion: int
    allow_guarded_single_anchor: bool
    anchors_required: int = 2
    normal_seed_length_limit: int = 330
    single_anchor_probability: float = 0.97
    header_assignment_threshold: float = 0.50
    subheader_support_radius: int = 30
    header_attachment_window: int = 3
    connector_expansion_threshold: float = 0.65
    emit_markdown_headings: bool = True
    apply_auxiliary_scope_veto: bool = False


def _physical_segments(abs_indices: np.ndarray) -> list[tuple[int, int]]:
    starts = [0]
    starts.extend(
        (np.flatnonzero(np.diff(abs_indices.astype(np.int64)) > MAX_PHYSICAL_GAP) + 1)
        .astype(int)
        .tolist()
    )
    starts.append(len(abs_indices))
    return list(zip(starts[:-1], starts[1:], strict=True))


def _near(mask: np.ndarray, index: int, radius: int, direction: int) -> bool:
    if direction < 0:
        return bool(mask[max(0, index - radius) : index].any())
    return bool(mask[index + 1 : min(len(mask), index + radius + 1)].any())


def _heading_topology(
    anchor: np.ndarray,
    markdown: np.ndarray,
    bib_header_probability: np.ndarray,
    bib_subheader_probability: np.ndarray,
    config: DecoderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard barriers, connectable subheaders, and attachable main headers."""

    barriers = np.zeros(len(anchor), dtype=bool)
    connectors = np.zeros(len(anchor), dtype=bool)
    attachable = np.zeros(len(anchor), dtype=bool)
    for index in np.flatnonzero(markdown):
        above = _near(anchor, int(index), config.subheader_support_radius, -1)
        below = _near(anchor, int(index), config.subheader_support_radius, 1)
        if (
            bib_subheader_probability[index] >= config.header_assignment_threshold
            and above
            and below
        ):
            connectors[index] = True
        elif (
            bib_header_probability[index] >= config.header_assignment_threshold
            and below
            and not above
        ):
            attachable[index] = True
            barriers[index] = True
        else:
            barriers[index] = True
    return barriers, connectors, attachable


def decode_document(
    probability: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    char_lengths: np.ndarray,
    abs_indices: np.ndarray,
    config: DecoderConfig,
    auxiliary_scope: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a document while enforcing heading and terminal-tail invariants."""

    if not (
        len(probability) == len(features) == len(char_lengths) == len(abs_indices)
    ):
        raise ValueError("decoder arrays are not aligned")
    if auxiliary_scope is None:
        auxiliary_scope = np.zeros(len(probability), dtype=bool)
    elif auxiliary_scope.shape != probability.shape:
        raise ValueError("auxiliary scope is not line aligned")
    names = {name: index for index, name in enumerate(feature_names)}
    required = {
        "probability:bib_header",
        "probability:bib_subheader",
        "probability:continuation_specialist",
        "presence:numbered_entry_count",
        "structure:markdown_heading",
        "structure:image_marker",
        "structure:table_row",
    }
    missing = required - set(names)
    if missing:
        raise ValueError(f"decoder feature table is missing {sorted(missing)}")
    markdown = features[:, names["structure:markdown_heading"]] > 0
    image = features[:, names["structure:image_marker"]] > 0
    table = features[:, names["structure:table_row"]] > 0
    numbered = features[:, names["presence:numbered_entry_count"]] > 0
    continuation = features[:, names["probability:continuation_specialist"]]
    eligible_length = (char_lengths <= config.normal_seed_length_limit) | table
    raw_anchor = (probability >= config.anchor_probability) & eligible_length & ~image
    barriers, subheaders, attachable_headers = _heading_topology(
        raw_anchor,
        markdown,
        features[:, names["probability:bib_header"]],
        features[:, names["probability:bib_subheader"]],
        config,
    )
    anchor = raw_anchor & ~barriers
    result = np.zeros(len(probability), dtype=bool)
    for physical_start, physical_end in _physical_segments(abs_indices):
        cursor = physical_start
        while cursor < physical_end:
            while cursor < physical_end and barriers[cursor]:
                cursor += 1
            if cursor >= physical_end:
                break
            end = cursor + 1
            while end < physical_end and not barriers[end]:
                end += 1
            local_anchor = np.flatnonzero(anchor[cursor:end])
            seed_spans: list[tuple[int, int]] = []
            count = config.anchors_required
            for offset in range(max(0, len(local_anchor) - count + 1)):
                group = local_anchor[offset : offset + count]
                if int(group[-1]) - int(group[0]) <= config.anchor_window:
                    seed_spans.append((int(group[0]), int(group[-1])))
            if config.allow_guarded_single_anchor:
                for local_index in local_anchor:
                    absolute = cursor + int(local_index)
                    guarded = bool(table[absolute] or numbered[absolute])
                    if not guarded:
                        for distance in range(1, config.header_attachment_window + 1):
                            candidate = absolute - distance
                            if candidate < physical_start or barriers[candidate] and not attachable_headers[candidate]:
                                break
                            guarded |= bool(attachable_headers[candidate])
                    if probability[absolute] >= config.single_anchor_probability and guarded:
                        seed_spans.append((int(local_index), int(local_index)))
            seed_spans.sort()
            merged: list[list[int]] = []
            for start, stop in seed_spans:
                if not merged:
                    merged.append([start, stop])
                    continue
                gap_start, gap_end = merged[-1][1] + 1, start
                gap_probability = probability[cursor + gap_start : cursor + gap_end]
                gap_connector = continuation[cursor + gap_start : cursor + gap_end]
                gap_subheader = subheaders[cursor + gap_start : cursor + gap_end]
                supported_gap = (
                    len(gap_probability) <= config.maximum_bridge_gap
                    and np.all(
                        (gap_probability >= config.inside_probability)
                        | (gap_connector >= config.connector_expansion_threshold)
                        | gap_subheader
                    )
                )
                if supported_gap:
                    merged[-1][1] = max(merged[-1][1], stop)
                else:
                    merged.append([start, stop])
            for start, stop in merged:
                absolute_start, absolute_end = cursor + start, cursor + stop
                result[absolute_start : absolute_end + 1] = True
                left, right = absolute_start, absolute_end
                for _ in range(config.adjacent_expansion):
                    candidate = left - 1
                    if (
                        candidate >= cursor
                        and not image[candidate]
                        and not barriers[candidate]
                        and (
                            probability[candidate] >= config.inside_probability
                            or continuation[candidate] >= config.connector_expansion_threshold
                        )
                    ):
                        result[candidate] = True
                        left = candidate
                    candidate = right + 1
                    if (
                        candidate < end
                        and not image[candidate]
                        and not barriers[candidate]
                        and (
                            probability[candidate] >= config.inside_probability
                            or continuation[candidate] >= config.connector_expansion_threshold
                        )
                    ):
                        result[candidate] = True
                        right = candidate
                for distance in range(1, config.header_attachment_window + 1):
                    candidate = left - distance
                    if candidate < physical_start:
                        break
                    if attachable_headers[candidate]:
                        if not result[candidate + 1 : left].any() and all(
                            not image[index] for index in range(candidate + 1, left)
                        ):
                            result[candidate:left] = True
                        break
                    if barriers[candidate]:
                        break
            cursor = end
    # A subheader is a connector, never a terminal authorization.  Image
    # placeholders likewise survive only when sandwiched inside a block.
    for index in np.flatnonzero(subheaders | image):
        if result[index] and not (
            result[max(0, index - 1) : index].any()
            and result[index + 1 : min(len(result), index + 2)].any()
        ):
            result[index] = False
    if not config.emit_markdown_headings:
        result[markdown] = False
    if config.apply_auxiliary_scope_veto:
        for start, end in blocks_from_mask(result, abs_indices):
            if has_auxiliary_scope(
                auxiliary_scope,
                abs_indices,
                start,
                end=end,
                window=config.header_attachment_window,
            ):
                result[start : end + 1] = False
    return result


def decode_table(
    table: Any,
    probability: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    config: DecoderConfig,
    auxiliary_scope: np.ndarray | None = None,
) -> np.ndarray:
    prediction = np.zeros(len(probability), dtype=bool)
    for document in table.documents:
        start, end = int(document["line_start"]), int(document["line_end"])
        prediction[start:end] = decode_document(
            probability[start:end],
            features[start:end],
            feature_names,
            table.char_lengths[start:end],
            table.abs_indices[start:end],
            config,
            None if auxiliary_scope is None else auxiliary_scope[start:end],
        )
    return prediction


def _char_metrics(table: Any, prediction: np.ndarray) -> dict[str, Any]:
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    trusted = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    lengths = table.char_lengths.astype(np.int64)
    tp = int(lengths[prediction & gold & trusted].sum())
    fp = int(lengths[prediction & ~gold & trusted].sum())
    fn = int(lengths[~prediction & gold & trusted].sum())
    return {
        "char_precision": tp / (tp + fp) if tp + fp else 1.0,
        "char_recall": tp / (tp + fn) if tp + fn else 0.0,
        "char_tp": tp,
        "char_fp": fp,
        "char_fn": fn,
    }


def evaluate(
    table: Any,
    prediction: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    result = {**evaluate_prediction(table, prediction), **_char_metrics(table, prediction)}
    markdown = features[:, feature_names.index("structure:markdown_heading")] > 0
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    result["non_bib_markdown_heading_crossings"] = int(
        np.count_nonzero(prediction & markdown & ~gold)
    )
    by_source = {}
    for source in sorted({str(row["source"]) for row in table.documents}):
        docs = {
            index
            for index, row in enumerate(table.documents)
            if str(row["source"]) == source
        }
        local = evaluate_prediction(table, prediction, document_subset=docs)
        mask = np.isin(table.document_indices, tuple(docs))
        local_prediction = prediction & mask
        local_gold = (table.original_labels == LABEL_TO_ID["BIB"]) & mask
        lengths = table.char_lengths.astype(np.int64)
        char_tp = int(lengths[local_prediction & local_gold].sum())
        char_fp = int(lengths[local_prediction & ~local_gold & mask].sum())
        char_fn = int(lengths[~local_prediction & local_gold].sum())
        local.update(
            char_precision=char_tp / (char_tp + char_fp) if char_tp + char_fp else 1.0,
            char_recall=char_tp / (char_tp + char_fn) if char_tp + char_fn else 0.0,
        )
        by_source[source] = local
    result["by_source"] = by_source
    return result


def _task(payload: tuple[str, str, str, str | None, dict[str, Any]]) -> dict[str, Any]:
    base_root, table_root, probability_path, auxiliary_path, config_payload = payload
    table = load_table(base_root, expected_split="train")
    manifest = json.loads((Path(table_root) / "manifest.json").read_text(encoding="utf-8"))
    features = np.load(Path(table_root) / "features.npy", mmap_mode="r", allow_pickle=False)
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    auxiliary = (
        np.load(auxiliary_path, mmap_mode="r", allow_pickle=False).astype(bool)
        if auxiliary_path
        else None
    )
    config = DecoderConfig(**config_payload)
    prediction = decode_table(
        table, probability, features, manifest["feature_names"], config, auxiliary
    )
    return {"config": config_payload, "metrics": evaluate(table, prediction, features, manifest["feature_names"])}


def _eligible(row: Mapping[str, Any]) -> bool:
    metrics = row["metrics"]
    return (
        metrics["line_precision"] >= 0.98
        and metrics["char_precision"] >= 0.98
        and metrics["non_bib_markdown_heading_crossings"] == 0
        and metrics["spurious_blocks_per_zero_block_document"] <= 0.02
    )


def _deployment_near_miss(row: Mapping[str, Any]) -> bool:
    """Retain the best candidate before the zero-BIB spurious-block gate."""

    metrics = row["metrics"]
    return (
        metrics["line_precision"] >= 0.98
        and metrics["char_precision"] >= 0.98
        and metrics["non_bib_markdown_heading_crossings"] == 0
    )


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        min(metrics["line_recall"], metrics["char_recall"]),
        metrics["char_recall"],
        metrics["line_recall"],
        metrics["line_precision"],
        metrics["char_precision"],
        -metrics["predicted_block_count"],
    )


def _grid() -> list[DecoderConfig]:
    return [
        DecoderConfig(
            anchor,
            inside,
            window,
            bridge,
            expansion,
            single,
            anchors_required=anchors,
            emit_markdown_headings=emit_headings,
            apply_auxiliary_scope_veto=True,
        )
        for anchor, inside, window, bridge, expansion, single, anchors, emit_headings in itertools.product(
            (0.75, 0.90, 0.95, 0.98),
            (0.30, 0.60),
            (8, 16),
            (4, 8),
            (0, 1),
            (False, True),
            (2, 3),
            (False, True),
        )
        if inside < anchor
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root = Path(args.table_dir).resolve()
    base_root = Path(args.base_table_dir).resolve()
    model_root = Path(args.model_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    table_manifest = json.loads((table_root / "manifest.json").read_text(encoding="utf-8"))
    model_report = json.loads((model_root / "report.json").read_text(encoding="utf-8"))
    if table_manifest.get("schema_version") != TABLE_SCHEMA:
        raise ValueError("unsupported nextgen feature table")
    if model_report.get("schema_version") != MODEL_SCHEMA or model_report.get("test_opened") is not False:
        raise ValueError("decoder requires development-only OOF probabilities")
    table = load_table(base_root, expected_split="train")
    probability_path = model_root / "oof_probability.npy"
    auxiliary_path = str(Path(args.auxiliary_scope).resolve()) if args.auxiliary_scope else None
    if auxiliary_path:
        auxiliary = np.load(auxiliary_path, mmap_mode="r", allow_pickle=False)
        if auxiliary.shape != (len(table.targets),):
            raise ValueError("auxiliary scope array is not aligned to the base table")
    configs = _grid()
    tasks = [
        (str(base_root), str(table_root), str(probability_path), auxiliary_path, asdict(config))
        for config in configs
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        rows = list(executor.map(_task, tasks, chunksize=1))
    eligible = [row for row in rows if _eligible(row)]
    near_misses = [row for row in rows if _deployment_near_miss(row)]
    selected = max(eligible, key=_selection_key) if eligible else None
    deployment_near_miss = max(near_misses, key=_selection_key) if near_misses else None
    highest_recall = max(rows, key=_selection_key)
    output.mkdir(parents=True)
    features = np.load(table_root / "features.npy", mmap_mode="r", allow_pickle=False)
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    auxiliary = (
        None
        if auxiliary_path is None
        else np.load(auxiliary_path, mmap_mode="r", allow_pickle=False).astype(bool)
    )
    diagnostic_prediction = decode_table(
        table,
        probability,
        features,
        table_manifest["feature_names"],
        DecoderConfig(**highest_recall["config"]),
        auxiliary,
    )
    with (output / "highest_recall_diagnostic_oof_prediction.npy").open("xb") as handle:
        np.save(handle, diagnostic_prediction, allow_pickle=False)
    if deployment_near_miss is not None:
        if deployment_near_miss["config"] == highest_recall["config"]:
            near_miss_prediction = diagnostic_prediction
        else:
            near_miss_prediction = decode_table(
                table,
                probability,
                features,
                table_manifest["feature_names"],
                DecoderConfig(**deployment_near_miss["config"]),
                auxiliary,
            )
        with (output / "deployment_near_miss_oof_prediction.npy").open("xb") as handle:
            np.save(handle, near_miss_prediction, allow_pickle=False)
    if selected is not None:
        if selected["config"] == highest_recall["config"]:
            prediction = diagnostic_prediction
        else:
            prediction = decode_table(
                table,
                probability,
                features,
                table_manifest["feature_names"],
                DecoderConfig(**selected["config"]),
                auxiliary,
            )
        with (output / "selected_oof_prediction.npy").open("xb") as handle:
            np.save(handle, prediction, allow_pickle=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_recall_first_pareto_gate" if selected else "research_only_no_eligible_decoder",
        "validation_opened": False,
        "test_opened": False,
        "model_kind": model_report["kind"],
        "selection_rule": "maximize min(line_recall,char_recall) subject to line/char precision>=0.98, zero non-BIB Markdown crossings, and <=0.02 spurious blocks per zero-BIB document",
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "selected": selected,
        "deployment_near_miss": deployment_near_miss,
        "highest_recall_diagnostic": highest_recall,
        "candidates": rows,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
            "model_receipt_sha256": sha256_file(model_root / "receipt.json"),
            "probability_sha256": sha256_file(probability_path),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "auxiliary_scope_sha256": sha256_file(Path(auxiliary_path))
            if auxiliary_path
            else None,
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return report


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--auxiliary-scope")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
