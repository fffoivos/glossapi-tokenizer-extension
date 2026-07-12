#!/usr/bin/env python3
"""Dependency-light prediction adapter for deterministic structure ablations.

The adapter reads only explicit GoldDocument-compatible JSONL and existing
``academic-structure-predictions-v1`` files.  It never fits a model, imports a
training implementation, discovers data, or permits a sealed test partition.
Outputs preserve the input ``line_id``/``abs_idx`` identity exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deterministic_structure import (
    BibLineEvidence,
    BibRole,
    DECODER_ID,
    RULES_ID,
    StructureConflict,
    StructureDecision,
    TocLineEvidence,
    TocRole,
    analyze_bib_line,
    analyze_toc_line,
    decode_bib_blocks,
    decode_toc_blocks,
)


INPUT_SCHEMA = "academic-structure-gold-v1"
PREDICTION_SCHEMA = "academic-structure-predictions-v1"
LABELS = frozenset(("O", "BIB", "TOC"))
SEALED_SPLITS = frozenset(
    ("test", "historical_test", "historical-test", "sealed_test", "sealed-test")
)


class AdapterError(ValueError):
    """Raised when an input could invalidate an ablation comparison."""


class SealedPartitionError(AdapterError):
    """Raised before prediction when a sealed partition is presented."""


class AblationMode(str, Enum):
    RULES_ONLY = "rules-only"
    BASE_PLUS_RULES = "base-plus-rules"
    BASE_RULES_VETO = "base-rules-veto"
    BASE_PLUS_RULES_VETO = "base-plus-rules-veto"

    @property
    def uses_base(self) -> bool:
        return self is not AblationMode.RULES_ONLY

    @property
    def adds_rules(self) -> bool:
        return self in {
            AblationMode.RULES_ONLY,
            AblationMode.BASE_PLUS_RULES,
            AblationMode.BASE_PLUS_RULES_VETO,
        }

    @property
    def vetoes_singletons(self) -> bool:
        return self in {
            AblationMode.BASE_RULES_VETO,
            AblationMode.BASE_PLUS_RULES_VETO,
        }


@dataclass(frozen=True)
class PredictionLine:
    line_id: str
    abs_idx: int
    text: str


@dataclass(frozen=True)
class PredictionDocument:
    document_id: str
    work_id: str
    source: str
    split: str
    n_physical_lines: int
    lines: tuple[PredictionLine, ...]
    n_present_lines: int | None = None
    coverage: str | None = None


@dataclass(frozen=True)
class HybridDocumentPrediction:
    labels: tuple[str, ...]
    line_reason_codes: tuple[tuple[str, ...], ...]
    structure_decision: StructureDecision


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _is_sealed_split(value: object) -> bool:
    return isinstance(value, str) and value.strip().casefold() in SEALED_SPLITS


def _document_fields(
    document: object,
) -> tuple[str, str, str, str, int, tuple[object, ...]]:
    try:
        document_id = getattr(document, "document_id")
        work_id = getattr(document, "work_id")
        source = getattr(document, "source")
        split = getattr(document, "split")
        n_physical_lines = getattr(document, "n_physical_lines")
        lines = tuple(getattr(document, "lines"))
    except (AttributeError, TypeError) as error:
        raise AdapterError("document is not GoldDocument-compatible") from error
    _require(
        isinstance(document_id, str) and bool(document_id), "document_id is required"
    )
    _require(
        isinstance(work_id, str) and bool(work_id),
        f"document {document_id!r}: work_id is required",
    )
    _require(
        isinstance(source, str) and bool(source),
        f"document {document_id!r}: source is required",
    )
    _require(
        isinstance(split, str) and bool(split),
        f"document {document_id!r}: split is required",
    )
    if _is_sealed_split(split):
        raise SealedPartitionError(
            f"document {document_id!r}: sealed split {split!r} is forbidden"
        )
    _require(
        isinstance(n_physical_lines, int)
        and not isinstance(n_physical_lines, bool)
        and n_physical_lines > 0,
        f"document {document_id!r}: invalid n_physical_lines",
    )
    _require(bool(lines), f"document {document_id!r}: no represented lines")
    return document_id, work_id, source, split, n_physical_lines, lines


def _validated_lines(document: object) -> tuple[object, ...]:
    document_id, _work_id, _source, _split, n_physical_lines, lines = _document_fields(
        document
    )
    previous = -1
    line_ids: set[str] = set()
    for position, line in enumerate(lines):
        try:
            line_id = getattr(line, "line_id")
            abs_idx = getattr(line, "abs_idx")
            text = getattr(line, "text")
        except AttributeError as error:
            raise AdapterError(
                f"document {document_id!r}, line {position}: incompatible line"
            ) from error
        _require(
            isinstance(line_id, str) and bool(line_id),
            f"document {document_id!r}, line {position}: line_id required",
        )
        _require(
            line_id not in line_ids,
            f"document {document_id!r}: duplicate line_id {line_id!r}",
        )
        line_ids.add(line_id)
        _require(
            isinstance(abs_idx, int)
            and not isinstance(abs_idx, bool)
            and abs_idx > previous,
            f"document {document_id!r}: abs_idx must be strictly increasing",
        )
        previous = abs_idx
        _require(
            isinstance(text, str) and bool(text.strip()),
            f"document {document_id!r}, line {line_id!r}: text must be nonblank",
        )
    _require(
        previous < n_physical_lines,
        f"document {document_id!r}: n_physical_lines does not cover final line",
    )
    return lines


def read_prediction_documents(path: str | Path) -> list[PredictionDocument]:
    """Read only the text/identity fields needed for prediction.

    Annotation labels and annotator metadata, if present, are neither retained
    nor consulted.  Test-like split names are rejected before a document is
    returned.
    """

    documents: list[PredictionDocument] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            _require(
                isinstance(raw, Mapping), f"input row {row_number}: expected object"
            )
            _require(
                raw.get("schema_version") == INPUT_SCHEMA,
                f"input row {row_number}: unsupported schema",
            )
            raw_split = raw.get("split")
            if _is_sealed_split(raw_split):
                raise SealedPartitionError(
                    f"input row {row_number}: sealed split {raw_split!r} is forbidden"
                )
            raw_lines = raw.get("lines")
            _require(
                isinstance(raw_lines, list) and bool(raw_lines),
                f"input row {row_number}: lines required",
            )
            parsed_lines: list[PredictionLine] = []
            for line_number, raw_line in enumerate(raw_lines, 1):
                _require(
                    isinstance(raw_line, Mapping),
                    f"input row {row_number}, line {line_number}: expected object",
                )
                parsed_lines.append(
                    PredictionLine(
                        line_id=raw_line.get("line_id"),  # type: ignore[arg-type]
                        abs_idx=raw_line.get("abs_idx"),  # type: ignore[arg-type]
                        text=raw_line.get("text"),  # type: ignore[arg-type]
                    )
                )
            document = PredictionDocument(
                document_id=raw.get("document_id"),  # type: ignore[arg-type]
                work_id=raw.get("work_id"),  # type: ignore[arg-type]
                source=raw.get("source"),  # type: ignore[arg-type]
                split=raw.get("split"),  # type: ignore[arg-type]
                n_physical_lines=raw.get("n_physical_lines"),  # type: ignore[arg-type]
                lines=tuple(parsed_lines),
                n_present_lines=raw.get("n_present_lines"),  # type: ignore[arg-type]
                coverage=raw.get("coverage"),  # type: ignore[arg-type]
            )
            _validated_lines(document)
            _require(
                document.document_id not in seen,
                f"input row {row_number}: duplicate document_id",
            )
            seen.add(document.document_id)
            documents.append(document)
    _require(bool(documents), "input JSONL is empty")
    return documents


def read_base_predictions(
    path: str | Path, documents: Sequence[object]
) -> dict[str, tuple[str, ...]]:
    """Strictly join an existing prediction-v1 file to exact line identities."""

    expected: dict[str, object] = {}
    expected_lines: dict[str, tuple[object, ...]] = {}
    for document in documents:
        document_id, *_rest = _document_fields(document)
        _require(document_id not in expected, f"duplicate document_id {document_id!r}")
        expected[document_id] = document
        expected_lines[document_id] = _validated_lines(document)
    result: dict[str, tuple[str, ...]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(
                isinstance(row, Mapping),
                f"prediction row {row_number}: expected object",
            )
            _require(
                row.get("schema_version") == PREDICTION_SCHEMA,
                f"prediction row {row_number}: unsupported schema",
            )
            document_id = row.get("document_id")
            _require(
                isinstance(document_id, str)
                and document_id in expected
                and document_id not in result,
                f"prediction row {row_number}: unknown/duplicate document_id",
            )
            if _is_sealed_split(row.get("split")):
                raise SealedPartitionError(
                    f"prediction row {row_number}: sealed split {row.get('split')!r} is forbidden"
                )
            document = expected[document_id]
            for field in ("work_id", "source", "split"):
                _require(
                    row.get(field) == getattr(document, field),
                    f"prediction row {row_number}: {field} identity mismatch",
                )
            raw_lines = row.get("lines")
            _require(
                isinstance(raw_lines, list)
                and len(raw_lines) == len(expected_lines[document_id]),
                f"prediction row {row_number}: incomplete line coverage",
            )
            labels: list[str] = []
            for gold_line, predicted in zip(expected_lines[document_id], raw_lines):
                _require(
                    isinstance(predicted, Mapping),
                    f"prediction row {row_number}: invalid line object",
                )
                _require(
                    predicted.get("line_id") == getattr(gold_line, "line_id")
                    and predicted.get("abs_idx") == getattr(gold_line, "abs_idx"),
                    f"prediction row {row_number}: line identity/order mismatch",
                )
                label = predicted.get("prediction")
                _require(
                    isinstance(label, str) and label in LABELS,
                    f"prediction row {row_number}: invalid prediction {label!r}",
                )
                labels.append(label)
            result[document_id] = tuple(labels)
    missing = set(expected).difference(result)
    _require(not missing, f"base predictions omit {len(missing)} documents")
    return result


def _all_present_lines_are_represented(
    document: object, lines: Sequence[object]
) -> bool:
    n_present_lines = getattr(document, "n_present_lines", None)
    coverage = getattr(document, "coverage", None)
    return (
        coverage == "full_document"
        and isinstance(n_present_lines, int)
        and not isinstance(n_present_lines, bool)
        and n_present_lines == len(lines)
    )


def _gap_evidence(
    previous_index: int,
    current_index: int,
    *,
    all_present_lines_represented: bool,
) -> tuple[list[TocLineEvidence], list[BibLineEvidence]]:
    missing = current_index - previous_index - 1
    if missing <= 0:
        return [], []
    if not all_present_lines_represented:
        barrier_index = previous_index + 1
        return (
            [
                TocLineEvidence(
                    barrier_index,
                    "",
                    TocRole.HARD_OTHER,
                    -4.0,
                    ("TOC_NEGATIVE_UNREPRESENTED_PHYSICAL_GAP",),
                    True,
                    0,
                )
            ],
            [
                BibLineEvidence(
                    barrier_index,
                    "",
                    BibRole.HARD_OTHER,
                    -4.0,
                    ("BIB_NEGATIVE_UNREPRESENTED_PHYSICAL_GAP",),
                    True,
                    0,
                )
            ],
        )

    # Missing physical coordinates are known blanks only when the document
    # represents every present line. Three blanks are sufficient to trip the
    # decoder's two-line bridge limit, so large gaps need not be materialised.
    blank_count = min(missing, 3)
    indexes = range(previous_index + 1, previous_index + blank_count + 1)
    return (
        [analyze_toc_line("", index) for index in indexes],
        [analyze_bib_line("", index) for index in indexes],
    )


def _structure_decision(document: object) -> StructureDecision:
    _document_id, _work_id, _source, _split, n_physical_lines, lines = _document_fields(
        document
    )
    lines = _validated_lines(document)
    complete = _all_present_lines_are_represented(document, lines)
    toc_rows: list[TocLineEvidence] = []
    bib_rows: list[BibLineEvidence] = []
    previous_index: int | None = None
    for line in lines:
        current_index = int(getattr(line, "abs_idx"))
        if previous_index is not None:
            toc_gap, bib_gap = _gap_evidence(
                previous_index,
                current_index,
                all_present_lines_represented=complete,
            )
            toc_rows.extend(toc_gap)
            bib_rows.extend(bib_gap)
        toc_rows.append(analyze_toc_line(getattr(line, "text"), current_index))
        bib_rows.append(analyze_bib_line(getattr(line, "text"), current_index))
        previous_index = current_index
    toc = tuple(toc_rows)
    bib = tuple(bib_rows)
    toc_spans = decode_toc_blocks(toc, n_physical_lines=n_physical_lines)
    bib_spans = decode_bib_blocks(bib)
    conflicts: list[StructureConflict] = []
    conflicted: set[int] = set()
    for toc_span in toc_spans:
        toc_indexes = set(toc_span.line_indices)
        for bib_span in bib_spans:
            overlap = tuple(sorted(toc_indexes.intersection(bib_span.line_indices)))
            if overlap:
                conflicts.append(StructureConflict(toc_span, bib_span, overlap))
                conflicted.update((id(toc_span), id(bib_span)))
    spans = tuple(span for span in toc_spans + bib_spans if id(span) not in conflicted)
    return StructureDecision(toc, bib, spans, tuple(conflicts))


def _mode(value: str | AblationMode) -> AblationMode:
    if isinstance(value, AblationMode):
        return value
    try:
        return AblationMode(value)
    except ValueError as error:
        raise AdapterError(f"unknown ablation mode {value!r}") from error


def _validated_base(
    document: object, mode: AblationMode, base_labels: Sequence[str] | None
) -> tuple[str, ...]:
    lines = _validated_lines(document)
    if not mode.uses_base:
        _require(base_labels is None, "rules-only must not receive base predictions")
        return tuple("O" for _line in lines)
    _require(base_labels is not None, f"{mode.value} requires base predictions")
    labels = tuple(base_labels or ())
    _require(len(labels) == len(lines), "base prediction length mismatch")
    _require(
        all(label in LABELS for label in labels),
        "base predictions contain invalid label",
    )
    return labels


def _runs(labels: Sequence[str]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    start = 0
    while start < len(labels):
        label = labels[start]
        end = start + 1
        while end < len(labels) and labels[end] == label:
            end += 1
        if label != "O":
            runs.append((start, end, label))
        start = end
    return runs


def predict_document(
    document: object,
    mode: str | AblationMode,
    base_labels: Sequence[str] | None = None,
) -> HybridDocumentPrediction:
    """Apply one named deterministic/base ablation to a compatible document."""

    selected = _mode(mode)
    lines = _validated_lines(document)
    original_base = _validated_base(document, selected, base_labels)
    decision = _structure_decision(document)
    labels = list(original_base)
    reasons: list[list[str]] = [
        (["BASE_PREDICTION"] if label != "O" and selected.uses_base else [])
        for label in labels
    ]
    coordinate_to_position = {
        int(getattr(line, "abs_idx")): position for position, line in enumerate(lines)
    }

    rule_labels: dict[int, str] = {}
    independently_addable: set[int] = set()
    for span in decision.spans:
        target = span.kind.value
        support_only = target == "BIB" and span.seed_kind == "bib_headerless_dense_run"
        for coordinate in span.line_indices:
            position = coordinate_to_position.get(coordinate)
            if position is None:
                continue
            previous = rule_labels.get(position)
            if previous is not None and previous != target:
                rule_labels[position] = "CONFLICT"
            else:
                rule_labels[position] = target
            if not support_only:
                independently_addable.add(position)

    for position, rule_label in rule_labels.items():
        if rule_label == "CONFLICT" or (
            labels[position] != "O" and labels[position] != rule_label
        ):
            labels[position] = "O"
            reasons[position].append("RULE_BASE_OVERLAP_FAIL_CLOSED")
        elif selected.adds_rules and position in independently_addable:
            labels[position] = rule_label
            reasons[position].append("DETERMINISTIC_SPAN")
        elif labels[position] == rule_label:
            reasons[position].append("DETERMINISTIC_SAME_TARGET_SUPPORT")
        elif position not in independently_addable:
            reasons[position].append("DETERMINISTIC_HEADERLESS_BIB_SUPPORT_ONLY")

    # A deterministic ToC/BIB ambiguity is always retained, including when the
    # base model predicted one side of it.
    for conflict in decision.conflicts:
        for coordinate in conflict.overlapping_line_indices:
            position = coordinate_to_position.get(coordinate)
            if position is None:
                continue
            labels[position] = "O"
            reasons[position].append("RULE_RULE_OVERLAP_FAIL_CLOSED")

    toc_hard = {item.line_index for item in decision.toc_evidence if item.hard_negative}
    bib_hard = {item.line_index for item in decision.bib_evidence if item.hard_negative}
    for position, line in enumerate(lines):
        coordinate = int(getattr(line, "abs_idx"))
        target = labels[position]
        if (target == "TOC" and coordinate in toc_hard) or (
            target == "BIB" and coordinate in bib_hard
        ):
            labels[position] = "O"
            reasons[position].append(f"{target}_HARD_NEGATIVE_KEEP")

    if selected.vetoes_singletons:
        for start, end, target in _runs(labels):
            if end - start != 1:
                continue
            position = start
            if (
                original_base[position] == target
                and rule_labels.get(position) != target
            ):
                labels[position] = "O"
                reasons[position].append("ISOLATED_ML_SINGLETON_VETO")

    return HybridDocumentPrediction(
        tuple(labels), tuple(tuple(row) for row in reasons), decision
    )


def prediction_row(
    document: object,
    mode: str | AblationMode,
    base_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = _mode(mode)
    document_id, work_id, source, split, _n_physical_lines, lines = _document_fields(
        document
    )
    result = predict_document(document, selected, base_labels)
    return {
        "schema_version": PREDICTION_SCHEMA,
        "model_id": f"deterministic-hybrid-{selected.value}",
        "component_ids": {
            "deterministic_rules": RULES_ID,
            "deterministic_decoder": DECODER_ID,
            "base_model": "bound_by_ablation_run_receipt"
            if selected.uses_base
            else None,
        },
        "document_id": document_id,
        "work_id": work_id,
        "source": source,
        "split": split,
        "ablation": {
            "mode": selected.value,
            "uses_base": selected.uses_base,
            "adds_deterministic_spans": selected.adds_rules,
            "headerless_bibliography_policy": "support_or_veto_only_never_independent_add",
            "vetoes_isolated_ml_singletons": selected.vetoes_singletons,
            "target_specific_hard_negative_keep": True,
            "overlap_policy": "fail_closed_keep",
            "training_performed": False,
            "sealed_partition_access_permitted": False,
        },
        "deterministic_evidence": {
            "span_count": len(result.structure_decision.spans),
            "withheld_conflict_count": len(result.structure_decision.conflicts),
        },
        "lines": [
            {
                "line_id": getattr(line, "line_id"),
                "abs_idx": getattr(line, "abs_idx"),
                "prediction": label,
                "decision_reason_codes": list(reason_codes),
            }
            for line, label, reason_codes in zip(
                lines, result.labels, result.line_reason_codes
            )
        ],
    }


def build_prediction_rows(
    documents: Sequence[object],
    mode: str | AblationMode,
    base_predictions: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    selected = _mode(mode)
    _require(bool(documents), "no documents")
    if selected.uses_base:
        _require(
            base_predictions is not None, f"{selected.value} requires base predictions"
        )
    else:
        _require(
            base_predictions is None, "rules-only must not receive base predictions"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        document_id, *_rest = _document_fields(document)
        _require(document_id not in seen, f"duplicate document_id {document_id!r}")
        seen.add(document_id)
        base = None if base_predictions is None else base_predictions.get(document_id)
        _require(
            not selected.uses_base or base is not None,
            f"base predictions omit document {document_id!r}",
        )
        rows.append(prediction_row(document, selected, base))
    if base_predictions is not None:
        _require(
            set(base_predictions) == seen,
            "base prediction document inventory does not match input documents",
        )
    return rows


def write_prediction_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically create immutable prediction JSONL."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable predictions {output}")
    _require(bool(rows), "no prediction rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--mode", required=True, choices=[mode.value for mode in AblationMode]
    )
    parser.add_argument("--base-predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    mode = AblationMode(args.mode)
    documents = read_prediction_documents(args.input_jsonl)
    if mode.uses_base:
        if not args.base_predictions:
            parser.error(f"--base-predictions is required for {mode.value}")
        base = read_base_predictions(args.base_predictions, documents)
    else:
        if args.base_predictions:
            parser.error("--base-predictions is forbidden for rules-only")
        base = None
    rows = build_prediction_rows(documents, mode, base)
    write_prediction_rows(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
