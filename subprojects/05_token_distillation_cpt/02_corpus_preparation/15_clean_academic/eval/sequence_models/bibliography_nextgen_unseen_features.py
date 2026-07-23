#!/usr/bin/env python3
"""Materialize label-blind nextgen features for a frozen unseen document set."""

from __future__ import annotations

import argparse
import concurrent.futures
import __main__
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from .bibliography_deterministic_roles import ROLE_NAMES as NEGATIVE_ROLE_NAMES
from .bibliography_deterministic_roles import _analyze_document
from .bibliography_entry_dataset import FEATURE_NAMES, MAX_PHYSICAL_GAP
from .bibliography_nextgen_freeze import SCHEMA_VERSION as FREEZE_SCHEMA
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA, feature_names
from .bibliography_positional_features import extract_positional_line
from .bibliography_role_experts import (
    CONNECTOR_PROBABILITY_COLUMNS,
    HEADING_PROBABILITY_COLUMNS,
    ConnectorBundle,
    HeadingBundle,
    HeadingTransform,
)
from .bibliography_role_features import (
    broad_heading_candidate,
    candidate_window_mask,
    connector_feature_row,
    heading_numeric_features,
    line_shape,
    p0d_matrix,
)
from .bibliography_signal_tcn import SignalTCN, build_signal_features
from .bibliography_scope_rules import auxiliary_scope_mask
from .contract import seal_hashes, sha256_file
from .bibliography_nextgen_table import bib_heading_lexicon_match
from .deterministic_structure import BibRole, analyze_bib_line


_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_RULE_LINE = re.compile(r"^[\s_—–\-.]{4,}$")
_IMAGE_MARKER = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    work_id: str
    source: str
    texts: tuple[str, ...]
    line_ids: tuple[str, ...]
    abs_indices: np.ndarray
    counts: np.ndarray
    gaps: np.ndarray
    shapes: np.ndarray
    header_kinds: np.ndarray
    negative_roles: np.ndarray
    char_lengths: np.ndarray


@dataclass
class UnseenTable:
    documents: list[dict[str, Any]]
    abs_indices: np.ndarray
    char_lengths: np.ndarray
    feature_names: tuple[str, ...] = ()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
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


def _extract_document(document: Mapping[str, Any]) -> ExtractedDocument:
    document_id = str(document.get("document_id") or "")
    work_id = str(document.get("work_id") or "")
    source = str(document.get("source") or "")
    lines = document.get("lines")
    if not document_id or not work_id or not source or not isinstance(lines, list) or not lines:
        raise ValueError("unseen document has malformed identity or line inventory")
    texts: list[str] = []
    line_ids: list[str] = []
    absolute: list[int] = []
    counts: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    headings: list[int] = []
    previous = -1
    for offset, line in enumerate(lines):
        text = line.get("text")
        line_id = str(line.get("line_id") or "")
        abs_idx = line.get("abs_idx")
        if (
            not isinstance(text, str)
            or not line_id
            or not isinstance(abs_idx, int)
            or abs_idx <= previous
        ):
            raise ValueError(f"{document_id}: malformed line {offset}")
        previous = abs_idx
        encoding = extract_positional_line(text)
        evidence = analyze_bib_line(text, abs_idx)
        texts.append(text)
        line_ids.append(line_id)
        absolute.append(abs_idx)
        counts.append(encoding.counts)
        gaps.append(encoding.gap_summaries)
        shapes.append(line_shape(text))
        headings.append(
            1 if evidence.role == BibRole.HEADING else 2 if evidence.role == BibRole.SUBHEADING else 0
        )
    _identity, negative, _counts = _analyze_document((document_id, lines))
    return ExtractedDocument(
        document_id=document_id,
        work_id=work_id,
        source=source,
        texts=tuple(texts),
        line_ids=tuple(line_ids),
        abs_indices=np.asarray(absolute, dtype=np.uint32),
        counts=np.stack(counts).astype(np.uint32),
        gaps=np.stack(gaps).astype(np.float32),
        shapes=np.stack(shapes).astype(np.float32),
        header_kinds=np.asarray(headings, dtype=np.uint8),
        negative_roles=negative,
        char_lengths=np.asarray([len(text) for text in texts], dtype=np.uint32),
    )


def _load_pickle(path: Path) -> Any:
    # Historical role-expert artifacts were written with ``python -m`` and are
    # therefore bound to __main__. Rebind only their three known data classes.
    __main__.HeadingTransform = HeadingTransform
    __main__.HeadingBundle = HeadingBundle
    __main__.ConnectorBundle = ConnectorBundle
    with path.open("rb") as handle:
        return pickle.load(handle)


def _batched_predict(models: Sequence[Any], values: np.ndarray, batch: int = 50_000) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    for model in models:
        for start in range(0, len(values), batch):
            result[start : start + batch] += model.predict_proba(
                values[start : start + batch]
            )[:, 1]
    return (result / len(models)).astype(np.float32)


def _physical_segments(abs_indices: np.ndarray) -> list[tuple[int, int]]:
    starts = [0]
    starts.extend(
        (
            np.flatnonzero(np.diff(abs_indices.astype(np.int64)) > MAX_PHYSICAL_GAP)
            + 1
        ).tolist()
    )
    starts.append(len(abs_indices))
    return list(zip(starts[:-1], starts[1:], strict=True))


def _signal_tcn_ensemble(
    entry: np.ndarray,
    documents: Sequence[ExtractedDocument],
    root: Path,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for signal-TCN inference")
    roles = np.concatenate([document.negative_roles for document in documents])
    header = np.concatenate([document.header_kinds for document in documents])
    features = build_signal_features(entry, roles, header)
    output = np.zeros(len(entry), dtype=np.float64)
    document_starts = np.cumsum(
        np.asarray([0, *(len(document.texts) for document in documents)], dtype=np.int64)
    )
    model_paths = sorted((root / "models").glob("fold*.pt"))
    if not model_paths:
        raise ValueError("signal-TCN model inventory is empty")
    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        architecture = checkpoint["architecture"]
        model = SignalTCN(
            features.shape[1],
            hidden_dim=int(architecture["hidden_dim"]),
            dilations=tuple(architecture["dilations"]),
            dropout=float(architecture["dropout"]),
        ).cpu()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        local_output = np.zeros(len(entry), dtype=np.float32)
        with torch.inference_mode():
            for document_index, document in enumerate(documents):
                base = int(document_starts[document_index])
                for segment_start, segment_end in _physical_segments(document.abs_indices):
                    for central_start in range(segment_start, segment_end, 256):
                        central_end = min(segment_end, central_start + 256)
                        input_start = max(segment_start, central_start - 32)
                        input_end = min(segment_end, central_end + 32)
                        x = torch.from_numpy(
                            features[base + input_start : base + input_end]
                        ).unsqueeze(0)
                        mask = torch.ones((1, input_end - input_start), dtype=torch.bool)
                        values = torch.sigmoid(model(x, mask))[0].numpy()
                        left = central_start - input_start
                        right = central_end - input_start
                        local_output[base + central_start : base + central_end] = values[
                            left:right
                        ]
        output += local_output
    return (output / len(model_paths)).astype(np.float32)


def _heading_probabilities(
    documents: Sequence[ExtractedDocument],
    entry: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    total = len(entry)
    candidate = np.zeros(total, dtype=bool)
    numeric: list[np.ndarray] = []
    texts: list[str] = []
    indices: list[int] = []
    for document, row in zip(documents, metadata, strict=True):
        start, end = int(row["line_start"]), int(row["line_end"])
        for offset, text in enumerate(document.texts):
            previous_blank = offset > 0 and not document.texts[offset - 1].strip()
            next_blank = offset + 1 < len(document.texts) and not document.texts[offset + 1].strip()
            if not broad_heading_candidate(
                text, previous_blank=previous_blank, next_blank=next_blank
            ):
                continue
            absolute = start + offset
            candidate[absolute] = True
            indices.append(absolute)
            texts.append(text)
            numeric.append(
                heading_numeric_features(
                    text,
                    previous_blank=previous_blank,
                    next_blank=next_blank,
                    position_fraction=offset / max(1, len(document.texts) - 1),
                    entry_probabilities_above=entry[max(start, absolute - 30) : absolute],
                    entry_probabilities_below=entry[
                        absolute + 1 : min(end, absolute + 31)
                    ],
                )
            )
    probability = np.zeros((total, len(HEADING_PROBABILITY_COLUMNS)), dtype=np.float32)
    if indices:
        matrix = np.stack(numeric).astype(np.float32)
        local = np.zeros((len(indices), probability.shape[1]), dtype=np.float64)
        models = [_load_pickle(path) for path in sorted((root / "models").glob("fold*.pkl"))]
        for model in models:
            for begin in range(0, len(indices), 10_000):
                local[begin : begin + 10_000] += model.predict(
                    texts[begin : begin + 10_000], matrix[begin : begin + 10_000]
                )
        probability[np.asarray(indices)] = (local / len(models)).astype(np.float32)
    return probability, candidate


def _connector_probabilities(
    documents: Sequence[ExtractedDocument],
    metadata: Sequence[Mapping[str, Any]],
    entry: np.ndarray,
    heading: np.ndarray,
    heading_candidate: np.ndarray,
    p0d_models: Sequence[Any],
    root: Path,
) -> tuple[np.ndarray, int]:
    total = len(entry)
    probability = np.tile(
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (total, 1)
    )
    all_indices: list[int] = []
    all_features: list[np.ndarray] = []
    for document, row in zip(documents, metadata, strict=True):
        start, end = int(row["line_start"]), int(row["line_end"])
        local_entry = entry[start:end]
        local_heading_candidate = heading_candidate[start:end]
        mask = candidate_window_mask(
            local_entry,
            local_heading_candidate,
            document.abs_indices,
            entry_threshold=0.25,
            radius=30,
        )
        wanted = np.flatnonzero(mask)
        joined: dict[bytes, np.ndarray] = {}
        for index in wanted:
            for neighbour, left_first in ((index - 1, True), (index + 1, False)):
                if not 0 <= neighbour < len(document.texts):
                    continue
                if abs(int(document.abs_indices[index]) - int(document.abs_indices[neighbour])) > MAX_PHYSICAL_GAP:
                    continue
                text = (
                    document.texts[neighbour].rstrip()
                    + " "
                    + document.texts[index].lstrip()
                    if left_first
                    else document.texts[index].rstrip()
                    + " "
                    + document.texts[neighbour].lstrip()
                )
                values = extract_positional_line(text).counts.reshape(1, -1)
                joined.setdefault(values.tobytes(), values)
        joined_scores: dict[bytes, float] = {}
        if joined:
            keys = list(joined)
            matrix = np.concatenate([joined[key] for key in keys])
            scores = _batched_predict(p0d_models, p0d_matrix(matrix))
            joined_scores = {key: float(value) for key, value in zip(keys, scores, strict=True)}

        def score_counts(values: np.ndarray) -> float:
            return joined_scores[values.astype(np.uint32, copy=False).tobytes()]

        local_heading = heading[start:end, 1:]
        for index in wanted:
            feature = connector_feature_row(
                index=int(index),
                texts=document.texts,
                counts=document.counts,
                gap_summaries=document.gaps,
                abs_indices=document.abs_indices,
                entry_probability=local_entry,
                heading_probability=local_heading,
                candidate_mask=mask,
                score_counts=score_counts,
                entry_threshold=0.25,
            )
            all_indices.append(start + int(index))
            all_features.append(feature.values)
    if all_indices:
        matrix = np.stack(all_features).astype(np.float32)
        models = [_load_pickle(path) for path in sorted((root / "models").glob("fold*.pkl"))]
        local = np.zeros((len(matrix), len(CONNECTOR_PROBABILITY_COLUMNS)), dtype=np.float64)
        for model in models:
            for begin in range(0, len(matrix), 25_000):
                local[begin : begin + 25_000] += model.predict(
                    matrix[begin : begin + 25_000]
                )
        probability[np.asarray(all_indices)] = (local / len(models)).astype(np.float32)
    return probability, len(all_indices)


def run(args: argparse.Namespace) -> dict[str, Any]:
    documents_path = Path(args.documents).resolve()
    freeze_path = Path(args.candidate_freeze).resolve()
    seal_path = Path(args.test_seal).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != "frozen_before_test_open"
        or freeze.get("test_labels_opened") is not False
        or freeze.get("test_seal", {}).get("sha256") != sha256_file(seal_path)
        or sha256_file(documents_path) != seal_hashes(seal)["documents_sha256"]
    ):
        raise ValueError("candidate freeze/test document seal mismatch")
    rows = list(_iter_jsonl(documents_path))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        extracted = list(executor.map(_extract_document, rows, chunksize=1))
    metadata: list[dict[str, Any]] = []
    cursor = 0
    seen: set[str] = set()
    for document in extracted:
        if document.document_id in seen:
            raise ValueError("unseen document IDs are not unique")
        seen.add(document.document_id)
        metadata.append(
            {
                "document_id": document.document_id,
                "work_id": document.work_id,
                "source": document.source,
                "line_start": cursor,
                "line_end": cursor + len(document.texts),
                "line_count": len(document.texts),
            }
        )
        cursor += len(document.texts)
    if len(extracted) != int(seal["document_count"]) or cursor != int(seal["line_count"]):
        raise ValueError("unseen document/line counts differ from test seal")
    counts = np.concatenate([document.counts for document in extracted])
    gaps = np.concatenate([document.gaps for document in extracted])
    shapes = np.concatenate([document.shapes for document in extracted])
    absolute = np.concatenate([document.abs_indices for document in extracted])
    lengths = np.concatenate([document.char_lengths for document in extracted])
    auxiliary_scope = np.concatenate(
        [np.asarray(auxiliary_scope_mask(document.texts), dtype=bool) for document in extracted]
    )
    p0d_root = Path(args.entry_model_dir).resolve()
    p0d_models = [
        _load_pickle(path)
        for path in sorted((p0d_root / "models").glob("P0D.fold*.pkl"))
    ]
    if not p0d_models:
        raise ValueError("P0D fold inventory is empty")
    entry = _batched_predict(p0d_models, p0d_matrix(counts))
    signal = _signal_tcn_ensemble(
        entry, extracted, Path(args.signal_model_dir).resolve()
    )
    heading, heading_candidate = _heading_probabilities(
        extracted, entry, metadata, Path(args.heading_model_dir).resolve()
    )
    connector, connector_count = _connector_probabilities(
        extracted,
        metadata,
        entry,
        heading,
        heading_candidate,
        p0d_models,
        Path(args.connector_model_dir).resolve(),
    )
    markdown = np.asarray(
        [bool(_MARKDOWN_HEADING.match(text)) for document in extracted for text in document.texts],
        dtype=np.float32,
    )
    image = np.asarray(
        [bool(_IMAGE_MARKER.match(text)) for document in extracted for text in document.texts],
        dtype=np.float32,
    )
    table_index = FEATURE_NAMES.index("table_row_count")
    table = (counts[:, table_index] > 0).astype(np.float32)
    rule_line = np.asarray(
        [
            bool(_RULE_LINE.fullmatch(text))
            for document in extracted
            for text in document.texts
        ],
        dtype=np.float32,
    )
    heading_lexicon = np.asarray(
        [
            bool(_MARKDOWN_HEADING.match(text))
            and bib_heading_lexicon_match(text, int(abs_idx))
            for document in extracted
            for text, abs_idx in zip(document.texts, document.abs_indices, strict=True)
        ],
        dtype=np.float32,
    )
    features = np.column_stack(
        (
            entry,
            signal,
            connector[:, 0],
            connector[:, 1],  # specialist has no deployable model; exact fallback
            connector[:, 1],
            connector[:, 2],
            heading[:, 1],
            heading[:, 2],
            heading[:, 3],
            connector[:, 3],
            (counts > 0).astype(np.float32),
            np.log1p(counts.astype(np.float32)),
            shapes,
            gaps,
            markdown,
            image,
            table,
            rule_line,
            heading_lexicon,
        )
    ).astype(np.float32)
    names = feature_names()
    if features.shape != (cursor, len(names)) or not np.isfinite(features).all():
        raise RuntimeError("unseen nextgen feature contract failure")
    output.mkdir(parents=True)
    for name, value in (
        ("features.npy", features),
        ("abs_indices.npy", absolute),
        ("char_lengths.npy", lengths),
        ("auxiliary_scope.npy", auxiliary_scope),
    ):
        with (output / name).open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
    with (output / "documents.jsonl").open("x", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "line_ids.jsonl").open("x", encoding="utf-8") as handle:
        for document in extracted:
            for line_id, abs_idx in zip(
                document.line_ids, document.abs_indices, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "document_id": document.document_id,
                            "line_id": line_id,
                            "abs_idx": int(abs_idx),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    manifest = {
        "schema_version": TABLE_SCHEMA,
        "status": "passed_label_blind_unseen_feature_materialization",
        "split": "unseen_test",
        "test_opened": True,
        "test_labels_opened": False,
        "human_gold": False,
        "document_count": len(metadata),
        "line_count": cursor,
        "feature_count": len(names),
        "feature_names": names,
        "heading_candidate_count": int(np.count_nonzero(heading_candidate)),
        "connector_candidate_count": connector_count,
        "auxiliary_scope_line_count": int(np.count_nonzero(auxiliary_scope)),
        "continuation_specialist_policy": "frozen_connector_continuation_fallback",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "documents_sha256": sha256_file(documents_path),
            "candidate_freeze_sha256": sha256_file(freeze_path),
            "test_seal_sha256": sha256_file(seal_path),
            "entry_receipt_sha256": sha256_file(p0d_root / "receipt.json"),
            "signal_receipt_sha256": sha256_file(
                Path(args.signal_model_dir).resolve() / "receipt.json"
            ),
            "heading_receipt_sha256": sha256_file(
                Path(args.heading_model_dir).resolve() / "receipt.json"
            ),
            "connector_receipt_sha256": sha256_file(
                Path(args.connector_model_dir).resolve() / "receipt.json"
            ),
        },
    }
    _write_json_new(output / "manifest.json", manifest)
    _write_json_new(
        output / "receipt.json",
        {
            **manifest,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--test-seal", required=True)
    parser.add_argument("--entry-model-dir", required=True)
    parser.add_argument("--signal-model-dir", required=True)
    parser.add_argument("--heading-model-dir", required=True)
    parser.add_argument("--connector-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
