#!/usr/bin/env python3
"""Materialize provenance-bound heading and connector expert tables."""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import load_table
from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .bibliography_positional_models import load_positional_table
from .bibliography_role_experts import (
    CONNECTOR_PROBABILITY_COLUMNS, HEADING_PROBABILITY_COLUMNS,
)
from .bibliography_role_features import (
    HEADING_PROBABILITY_NAMES, candidate_window_mask, connector_feature_names,
    connector_feature_row, heading_numeric_features, p0d_matrix,
)
from .bibliography_role_v2 import (
    ID_TO_ROLE, OVERLAY_SCHEMA, ROLE_TO_ID, TRUSTED_STATUSES,
    merge_heading_overrides, migrate_row,
)
from .contract import sha256_file


HEADING_TABLE_SCHEMA = "bibliography-heading-expert-table-v1"
CONNECTOR_TABLE_SCHEMA = "bibliography-connector-expert-table-v1"
BLOCK_TABLE_SCHEMA = "bibliography-role-block-table-v1"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _migrated_rows(path: Path) -> list[dict[str, Any]]:
    return [
        row if row.get("schema_version") == OVERLAY_SCHEMA else migrate_row(row)
        for row in _iter_jsonl(path)
    ]


def _merged_overlay(primary: Path, heading: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _migrated_rows(primary)
    if heading is not None:
        overrides = {
            (str(row["document_id"]), str(row["line_id"])): row
            for row in _iter_jsonl(heading)
        }
        rows = merge_heading_overrides(rows, overrides)
    result = {}
    for row in rows:
        key = (str(row.get("document_id", "")), str(row.get("line_id", "")))
        if not all(key) or key in result:
            raise ValueError("merged overlay has repeated or empty identity")
        result[key] = row
    return result


def _aligned_source(
    source_path: Path, table: Any, split: str,
) -> tuple[list[list[str]], list[list[str]], list[list[int]], dict[tuple[str, str], tuple[int, int, int]]]:
    rows = [row for row in _iter_jsonl(source_path) if row.get("split") == split]
    if len(rows) != len(table.documents):
        raise ValueError("source/base document count mismatch")
    texts: list[list[str]] = []
    line_ids: list[list[str]] = []
    labels: list[list[int]] = []
    index: dict[tuple[str, str], tuple[int, int, int]] = {}
    label_encoding = {"O": 0, "BIB": 1, "TOC": 2, "UNKNOWN": 3}
    for document_index, (source, metadata) in enumerate(zip(rows, table.documents, strict=True)):
        if source.get("document_id") != metadata["document_id"] or source.get("work_id") != metadata["work_id"]:
            raise ValueError(f"source/base identity mismatch at document {document_index}")
        lines = source.get("lines")
        if not isinstance(lines, list) or len(lines) != int(metadata["line_count"]):
            raise ValueError(f"source/base line mismatch at document {document_index}")
        local_texts, local_ids, local_labels = [], [], []
        for offset, line in enumerate(lines):
            line_id = str(line.get("line_id") or f"{metadata['document_id']}:{line['abs_idx']}")
            absolute = int(metadata["line_start"]) + offset
            if int(line["abs_idx"]) != int(table.abs_indices[absolute]):
                raise ValueError(f"source/base coordinate mismatch at {metadata['document_id']}:{offset}")
            local_texts.append(str(line.get("text", "")))
            local_ids.append(line_id)
            local_labels.append(label_encoding[str(line.get("label", "UNKNOWN"))])
            key = (str(metadata["document_id"]), line_id)
            if key in index:
                raise ValueError(f"repeated source line identity: {key}")
            index[key] = (document_index, offset, absolute)
        texts.append(local_texts)
        line_ids.append(local_ids)
        labels.append(local_labels)
    return texts, line_ids, labels, index


def _prepare_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.mkdir(parents=True)


def materialize_heading_table(args: argparse.Namespace) -> dict[str, Any]:
    source, base_root = Path(args.source).resolve(), Path(args.base_table_dir).resolve()
    table = load_table(base_root, expected_split=args.split)
    entry_path = Path(args.entry_oof).resolve()
    entry_probability = np.load(entry_path, mmap_mode="r", allow_pickle=False)
    if entry_probability.shape != (len(table.targets),) or not np.isfinite(entry_probability).all():
        raise ValueError("entry OOF probability is malformed")
    overlay = _merged_overlay(
        Path(args.overlay).resolve(), Path(args.heading_overlay).resolve() if args.heading_overlay else None
    )
    provenance_path = Path(args.inventory_provenance).resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    texts_by_doc, _, _, source_index = _aligned_source(source, table, args.split)
    features, roles, trusted, folds, row_indices, text_rows = [], [], [], [], [], []
    for candidate in provenance["cases"]:
        key = (str(candidate["document_id"]), str(candidate["line_id"]))
        if key not in source_index:
            raise ValueError(f"heading candidate is absent from source: {key}")
        document_index, offset, absolute = source_index[key]
        metadata = table.documents[document_index]
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        text = texts_by_doc[document_index][offset]
        above = entry_probability[max(start, absolute - 30) : absolute]
        below = entry_probability[absolute + 1 : min(end, absolute + 31)]
        previous_blank = offset > 0 and not texts_by_doc[document_index][offset - 1].strip()
        next_blank = offset + 1 < len(texts_by_doc[document_index]) and not texts_by_doc[document_index][offset + 1].strip()
        features.append(heading_numeric_features(
            text, previous_blank=previous_blank, next_blank=next_blank,
            position_fraction=offset / max(1, len(texts_by_doc[document_index]) - 1),
            entry_probabilities_above=above, entry_probabilities_below=below,
        ))
        decision = overlay.get(key)
        role = str(decision.get("role", "UNKNOWN")) if decision else "UNKNOWN"
        status = str(decision.get("role_status", "UNRESOLVED")) if decision else "UNRESOLVED"
        roles.append(ROLE_TO_ID[role])
        trusted.append(int(status in TRUSTED_STATUSES and role != "UNKNOWN"))
        folds.append(int(table.folds[absolute]))
        row_indices.append(absolute)
        text_rows.append({"row_index": absolute, "text": text})
    output = Path(args.output_dir).resolve()
    _prepare_output(output)
    arrays = {
        "features": np.stack(features).astype(np.float32),
        "roles": np.asarray(roles, dtype=np.uint8),
        "trusted": np.asarray(trusted, dtype=np.uint8),
        "folds": np.asarray(folds, dtype=np.uint8),
        "row_indices": np.asarray(row_indices, dtype=np.uint32),
    }
    for name, value in arrays.items():
        _save(output / f"{name}.npy", value)
    with (output / "texts.jsonl").open("x", encoding="utf-8") as handle:
        for row in text_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": HEADING_TABLE_SCHEMA, "status": "passed_heading_table_materialization",
        "split": args.split, "validation_opened": False, "line_count": len(table.targets),
        "candidate_count": len(row_indices), "trusted_candidate_count": int(np.count_nonzero(arrays["trusted"])),
        "role_encoding": ROLE_TO_ID,
        "inputs": {
            "source_sha256": sha256_file(source), "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "entry_oof_sha256": sha256_file(entry_path), "overlay_sha256": sha256_file(Path(args.overlay).resolve()),
            "inventory_provenance_sha256": sha256_file(provenance_path),
        },
    }
    _write_json_new(output / "manifest.json", manifest)
    _write_json_new(output / "receipt.json", {**manifest, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return manifest


def _load_p0d_models(root: Path, n_folds: int) -> list[Any]:
    model_root = root / "models"
    models = []
    for fold in range(n_folds):
        path = model_root / f"P0D.fold{fold}.pkl"
        with path.open("rb") as handle:
            models.append(pickle.load(handle))
    return models


def _connector_supervision(row: Mapping[str, Any] | None) -> dict[str, int]:
    result = {
        "role": ROLE_TO_ID["UNKNOWN"], "trusted": 0,
        "connector_target": 0, "connector_trusted": 0,
        "subtype_target": 0, "subtype_trusted": 0,
        "other_target": 0, "other_trusted": 0,
    }
    if row is None:
        return result
    role, status = str(row.get("role", "UNKNOWN")), str(row.get("role_status", "UNRESOLVED"))
    result["role"] = ROLE_TO_ID[role]
    exact = status in TRUSTED_STATUSES and role != "UNKNOWN"
    result["trusted"] = int(exact)
    if exact:
        result["connector_target"] = int(role in {"CONTINUATION", "FILLER"})
        result["connector_trusted"] = 1
        result["subtype_target"] = int(role == "CONTINUATION")
        result["subtype_trusted"] = int(role in {"CONTINUATION", "FILLER"})
        result["other_target"] = int(role == "OTHER")
        result["other_trusted"] = 1
        return result
    votes = row.get("raw_role_votes")
    if isinstance(votes, Mapping) and len(votes) >= 2:
        normalized = []
        for values in votes.values():
            if not isinstance(values, list) or not values:
                return result
            normalized.append({str(value) for value in values})
        if all(values <= {"CONTINUATION", "FILLER"} and values for values in normalized):
            result["connector_target"] = 1
            result["connector_trusted"] = 1
    return result


def materialize_connector_table(args: argparse.Namespace) -> dict[str, Any]:
    source, base_root = Path(args.source).resolve(), Path(args.base_table_dir).resolve()
    table = load_table(base_root, expected_split=args.split)
    positional = load_positional_table(
        Path(args.positional_table_dir).resolve(), sha256_file(base_root / "manifest.json"), len(table.targets)
    )
    entry_root = Path(args.entry_oof_dir).resolve()
    entry_path = entry_root / "P0D.oof_probability.npy"
    entry_probability = np.load(entry_path, mmap_mode="r", allow_pickle=False)
    heading_root = Path(args.heading_oof_dir).resolve()
    heading_candidate_rows = np.load(heading_root / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    heading_candidate_probability = np.load(heading_root / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
    if heading_candidate_probability.shape != (len(heading_candidate_rows), len(HEADING_PROBABILITY_COLUMNS)):
        raise ValueError("heading OOF probability shape mismatch")
    full_heading = np.zeros((len(table.targets), len(HEADING_PROBABILITY_NAMES)), dtype=np.float32)
    full_heading[heading_candidate_rows] = heading_candidate_probability[:, 1:]
    heading_candidate_mask = np.zeros(len(table.targets), dtype=bool)
    heading_candidate_mask[heading_candidate_rows] = True
    overlay = _merged_overlay(
        Path(args.overlay).resolve(), Path(args.heading_overlay).resolve() if args.heading_overlay else None
    )
    texts_by_doc, ids_by_doc, _, _ = _aligned_source(source, table, args.split)
    models = _load_p0d_models(entry_root, int(table.manifest["n_folds"]))
    feature_rows: list[np.ndarray] = []
    supervision: list[dict[str, int]] = []
    folds: list[int] = []
    row_indices: list[int] = []
    candidate_counts = collections.Counter()
    trusted_connectors_total = trusted_connectors_selected = 0
    for document_index, metadata in enumerate(table.documents):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        texts = texts_by_doc[document_index]
        local_entry = entry_probability[start:end]
        local_headings = heading_candidate_mask[start:end]
        local_abs = table.abs_indices[start:end]
        local_mask = candidate_window_mask(
            local_entry, local_headings, local_abs,
            entry_threshold=float(args.entry_threshold), radius=30,
        )
        for offset, (text, line_id) in enumerate(zip(texts, ids_by_doc[document_index], strict=True)):
            absolute = start + offset
            row = overlay.get((str(metadata["document_id"]), line_id))
            label = _connector_supervision(row)
            trusted_connectors_total += int(label["connector_trusted"] and label["connector_target"])
            if not local_mask[offset] or row is None:
                continue
            trusted_connectors_selected += int(label["connector_trusted"] and label["connector_target"])
            fold = int(table.folds[absolute])
            model = models[fold]

            def score_counts(values: np.ndarray, *, current_model: Any = model) -> float:
                return float(current_model.predict_proba(p0d_matrix(values))[:, 1][0])

            feature = connector_feature_row(
                index=offset, texts=texts, counts=table.counts[start:end],
                gap_summaries=positional.gap_summaries[start:end], abs_indices=local_abs,
                entry_probability=local_entry, heading_probability=full_heading[start:end],
                candidate_mask=local_mask, score_counts=score_counts,
                entry_threshold=float(args.entry_threshold),
            )
            feature_rows.append(feature.values)
            supervision.append(label)
            folds.append(fold)
            row_indices.append(absolute)
            candidate_counts[ID_TO_ROLE[label["role"]]] += 1
    if trusted_connectors_selected != trusted_connectors_total:
        raise ValueError(
            f"candidate windows cover {trusted_connectors_selected}/{trusted_connectors_total} trusted connectors"
        )
    output = Path(args.output_dir).resolve()
    _prepare_output(output)
    arrays = {
        "features": np.stack(feature_rows).astype(np.float32),
        "roles": np.asarray([row["role"] for row in supervision], dtype=np.uint8),
        "trusted": np.asarray([row["trusted"] for row in supervision], dtype=np.uint8),
        "folds": np.asarray(folds, dtype=np.uint8),
        "row_indices": np.asarray(row_indices, dtype=np.uint32),
        "connector_targets": np.asarray([row["connector_target"] for row in supervision], dtype=np.uint8),
        "connector_trusted": np.asarray([row["connector_trusted"] for row in supervision], dtype=np.uint8),
        "subtype_targets": np.asarray([row["subtype_target"] for row in supervision], dtype=np.uint8),
        "subtype_trusted": np.asarray([row["subtype_trusted"] for row in supervision], dtype=np.uint8),
        "other_targets": np.asarray([row["other_target"] for row in supervision], dtype=np.uint8),
        "other_trusted": np.asarray([row["other_trusted"] for row in supervision], dtype=np.uint8),
    }
    for name, value in arrays.items():
        _save(output / f"{name}.npy", value)
    manifest = {
        "schema_version": CONNECTOR_TABLE_SCHEMA, "status": "passed_connector_table_materialization",
        "split": args.split, "validation_opened": False, "line_count": len(table.targets),
        "candidate_count": len(row_indices), "feature_count": len(connector_feature_names()),
        "feature_names": connector_feature_names(), "role_encoding": ROLE_TO_ID,
        "candidate_role_counts": dict(sorted(candidate_counts.items())),
        "trusted_connector_count": trusted_connectors_selected,
        "entry_threshold": float(args.entry_threshold),
        "inputs": {
            "source_sha256": sha256_file(source), "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "positional_manifest_sha256": sha256_file(Path(args.positional_table_dir).resolve() / "manifest.json"),
            "entry_oof_sha256": sha256_file(entry_path),
            "heading_report_sha256": sha256_file(heading_root / "report.json"),
            "overlay_sha256": sha256_file(Path(args.overlay).resolve()),
        },
    }
    _write_json_new(output / "manifest.json", manifest)
    _write_json_new(output / "receipt.json", {**manifest, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return manifest


def _scatter_probabilities(
    *, line_count: int, root: Path, columns: int, default: Sequence[float],
) -> np.ndarray:
    row_indices = np.load(root / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    local = np.load(root / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
    if local.shape != (len(row_indices), columns):
        raise ValueError(f"OOF probability shape mismatch: {root}")
    result = np.tile(np.asarray(default, dtype=np.float32), (line_count, 1))
    if len(np.unique(row_indices)) != len(row_indices) or np.any(row_indices >= line_count):
        raise ValueError(f"OOF row index inventory is invalid: {root}")
    result[row_indices] = local
    return result


def _trusted_role_arrays(
    table: Any, ids_by_doc: Sequence[Sequence[str]], overlay: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roles = np.full(len(table.targets), ROLE_TO_ID["UNKNOWN"], dtype=np.uint8)
    trusted = np.zeros(len(table.targets), dtype=np.uint8)
    hard_stop = np.zeros(len(table.targets), dtype=np.uint8)
    for document_index, metadata in enumerate(table.documents):
        start = int(metadata["line_start"])
        for offset, line_id in enumerate(ids_by_doc[document_index]):
            row = overlay.get((str(metadata["document_id"]), str(line_id)))
            if row is None:
                continue
            role, status = str(row.get("role", "UNKNOWN")), str(row.get("role_status", "UNRESOLVED"))
            index = start + offset
            roles[index] = ROLE_TO_ID[role]
            trusted[index] = int(status in TRUSTED_STATUSES and role != "UNKNOWN")
            hard_stop[index] = int(
                row.get("boundary_status") in TRUSTED_STATUSES
                and row.get("boundary_flag") == "HARD_STOP"
            )
    return roles, trusted, hard_stop


def _trusted_runs(trusted: np.ndarray, abs_indices: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(trusted.astype(bool)):
        physical_break = (
            index > 0
            and int(abs_indices[index]) - int(abs_indices[index - 1]) > MAX_PHYSICAL_GAP
        )
        if enabled and (start is None or physical_break):
            if start is not None:
                result.append((start, index))
            start = index
        elif not enabled and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(trusted)))
    return result


def materialize_block_table(args: argparse.Namespace) -> dict[str, Any]:
    source, base_root = Path(args.source).resolve(), Path(args.base_table_dir).resolve()
    table = load_table(base_root, expected_split=args.split)
    overlay = _merged_overlay(
        Path(args.overlay).resolve(), Path(args.heading_overlay).resolve() if args.heading_overlay else None
    )
    _, ids_by_doc, _, _ = _aligned_source(source, table, args.split)
    entry_path = Path(args.entry_oof_dir).resolve() / "P0D.oof_probability.npy"
    entry = np.load(entry_path, mmap_mode="r", allow_pickle=False)
    if entry.shape != (len(table.targets),):
        raise ValueError("entry probability shape mismatch")
    heading_root, connector_root = Path(args.heading_oof_dir).resolve(), Path(args.connector_oof_dir).resolve()
    heading = _scatter_probabilities(
        line_count=len(table.targets), root=heading_root,
        columns=len(HEADING_PROBABILITY_COLUMNS), default=(0.0, 0.0, 0.0, 0.0),
    )
    connector = _scatter_probabilities(
        line_count=len(table.targets), root=connector_root,
        columns=len(CONNECTOR_PROBABILITY_COLUMNS), default=(0.0, 0.0, 0.0, 1.0),
    )
    gold_roles, trusted, hard_stop = _trusted_role_arrays(table, ids_by_doc, overlay)
    role_probability = np.column_stack(
        (
            entry,
            connector[:, 1],
            connector[:, 2],
            heading[:, 1],
            heading[:, 2],
            heading[:, 3],
            connector[:, 3],
        )
    ).astype(np.float32)
    if not np.isfinite(role_probability).all():
        raise ValueError("role probabilities contain non-finite values")

    output_probability: list[np.ndarray] = []
    output_connector: list[np.ndarray] = []
    output_abs: list[np.ndarray] = []
    output_lengths: list[np.ndarray] = []
    output_roles: list[np.ndarray] = []
    output_trusted: list[np.ndarray] = []
    document_rows: list[dict[str, Any]] = []
    cursor = 0
    inside_ids = np.asarray([
        ROLE_TO_ID[role] for role in ("ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER")
    ])
    gold_blocks = seed_reachable = 0
    by_source = collections.Counter()
    for document_index, metadata in enumerate(table.documents):
        doc_start, doc_end = int(metadata["line_start"]), int(metadata["line_end"])
        local_trusted = trusted[doc_start:doc_end]
        local_roles = gold_roles[doc_start:doc_end]
        local_abs = table.abs_indices[doc_start:doc_end]
        for run_number, (start, end) in enumerate(_trusted_runs(local_trusted, local_abs)):
            if end - start < 3:
                continue
            run_roles = local_roles[start:end]
            run_hard = hard_stop[doc_start + start : doc_start + end]
            run_inside = np.isin(run_roles, inside_ids)
            begins_bounded = (
                start == 0
                or run_roles[0] in {ROLE_TO_ID["OTHER"], ROLE_TO_ID["NON_BIB_HEADER"], ROLE_TO_ID["BIB_HEADER"]}
                or bool(run_hard[0])
            )
            ends_bounded = (
                end == doc_end - doc_start
                or run_roles[-1] in {ROLE_TO_ID["OTHER"], ROLE_TO_ID["NON_BIB_HEADER"]}
                or bool(run_hard[-1])
            )
            if not (begins_bounded and ends_bounded):
                continue
            absolute_start, absolute_end = doc_start + start, doc_start + end
            block_starts = np.flatnonzero(run_inside & ~np.concatenate(([False], run_inside[:-1])))
            block_ends = np.flatnonzero(run_inside & ~np.concatenate((run_inside[1:], [False])))
            for block_start, block_end in zip(block_starts, block_ends, strict=True):
                gold_blocks += 1
                local_entry = entry[absolute_start + block_start : absolute_start + block_end + 1]
                local_length = table.char_lengths[absolute_start + block_start : absolute_start + block_end + 1]
                seed_reachable += int(np.count_nonzero((local_entry >= 0.25) & (local_length <= 330)) >= 2)
            length = absolute_end - absolute_start
            output_probability.append(role_probability[absolute_start:absolute_end])
            output_connector.append(connector[absolute_start:absolute_end, 0])
            output_abs.append(np.asarray(table.abs_indices[absolute_start:absolute_end]))
            output_lengths.append(np.asarray(table.char_lengths[absolute_start:absolute_end]))
            output_roles.append(run_roles)
            output_trusted.append(np.ones(length, dtype=np.uint8))
            document_rows.append({
                "document_id": f"{metadata['document_id']}:review-run-{run_number}",
                "parent_document_id": str(metadata["document_id"]), "work_id": str(metadata["work_id"]),
                "source": str(metadata["source"]), "fold": int(table.folds[absolute_start]),
                "line_start": cursor, "line_end": cursor + length, "line_count": length,
                "gold_inside_line_count": int(np.count_nonzero(run_inside)),
            })
            cursor += length
            by_source[str(metadata["source"])] += 1
    if not document_rows:
        raise ValueError("no fully reviewed, bounded block sequences are available")
    output = Path(args.output_dir).resolve()
    _prepare_output(output)
    arrays = {
        "role_probability": np.concatenate(output_probability),
        "connector_probability": np.concatenate(output_connector),
        "abs_indices": np.concatenate(output_abs),
        "char_lengths": np.concatenate(output_lengths),
        "gold_roles": np.concatenate(output_roles),
        "trusted": np.concatenate(output_trusted),
    }
    for name, value in arrays.items():
        _save(output / f"{name}.npy", value)
    with (output / "documents.jsonl").open("x", encoding="utf-8") as handle:
        for row in document_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": BLOCK_TABLE_SCHEMA, "status": "passed_fully_reviewed_block_table",
        "split": args.split, "validation_opened": False, "n_folds": int(table.manifest["n_folds"]),
        "sequence_count": len(document_rows), "line_count": cursor,
        "sequence_counts_by_source": dict(sorted(by_source.items())),
        "gold_block_count": gold_blocks, "seed_reachable_gold_block_count": seed_reachable,
        "seed_recall_ceiling": seed_reachable / gold_blocks if gold_blocks else 1.0,
        "role_probability_columns": (
            "ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER", "OTHER"
        ),
        "inputs": {
            "source_sha256": sha256_file(source), "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "entry_oof_sha256": sha256_file(entry_path),
            "heading_report_sha256": sha256_file(heading_root / "report.json"),
            "connector_report_sha256": sha256_file(connector_root / "report.json"),
            "overlay_sha256": sha256_file(Path(args.overlay).resolve()),
        },
    }
    _write_json_new(output / "manifest.json", manifest)
    _write_json_new(output / "receipt.json", {**manifest, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    heading = sub.add_parser("heading")
    connector = sub.add_parser("connector")
    block = sub.add_parser("block")
    for command in (heading, connector, block):
        command.add_argument("--source", required=True)
        command.add_argument("--base-table-dir", required=True)
        command.add_argument("--overlay", required=True)
        command.add_argument("--heading-overlay")
        command.add_argument("--split", default="train")
        command.add_argument("--output-dir", required=True)
    heading.add_argument("--entry-oof", required=True)
    heading.add_argument("--inventory-provenance", required=True)
    connector.add_argument("--positional-table-dir", required=True)
    connector.add_argument("--entry-oof-dir", required=True)
    connector.add_argument("--heading-oof-dir", required=True)
    connector.add_argument("--entry-threshold", type=float, default=0.25)
    block.add_argument("--entry-oof-dir", required=True)
    block.add_argument("--heading-oof-dir", required=True)
    block.add_argument("--connector-oof-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "heading":
        materialize_heading_table(args)
    elif args.command == "connector":
        materialize_connector_table(args)
    else:
        materialize_block_table(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
