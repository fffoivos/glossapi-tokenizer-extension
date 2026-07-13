#!/usr/bin/env python3
"""Compare R2 and explicit-feature bibliography proposals on one dev split.

The evaluator refuses test/sealed splits, fits nothing, and reports agreement
with the existing LLM-silver labels.  It is intended for CPU execution and does
not mutate source data.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bibliography_v2 import (
    DECODER_ID as V2_DECODER_ID,
    RULES_ID as V2_RULES_ID,
    BibliographyV2Evidence,
    analyze_bibliography_line_v2,
    decode_bibliography_blocks_v2,
    hard_gap_evidence,
)
from .contract import GoldDocument, read_gold, sha256_file
from .deterministic_adapter import AblationMode, _structure_decision, predict_document
from .deterministic_structure import DECODER_ID as R2_DECODER_ID
from .deterministic_structure import RULES_ID as R2_RULES_ID
from .deterministic_structure import BibRole, StructureKind


REPORT_SCHEMA = "bibliography-explicit-feature-evaluation-v1"
ALLOWED_SPLITS = frozenset(("train", "validation"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _v2_evidence(document: GoldDocument) -> tuple[BibliographyV2Evidence, ...]:
    complete = document.coverage == "full_document" and document.n_present_lines == len(
        document.lines
    )
    rows: list[BibliographyV2Evidence] = []
    previous: int | None = None
    for line in document.lines:
        if previous is not None and line.abs_idx > previous + 1:
            if complete:
                for index in range(previous + 1, min(line.abs_idx, previous + 5)):
                    rows.append(analyze_bibliography_line_v2("", index))
            else:
                rows.append(hard_gap_evidence(previous + 1))
        rows.append(analyze_bibliography_line_v2(line.text, line.abs_idx))
        previous = line.abs_idx
    return tuple(rows)


def _span_labels(document: GoldDocument, spans: Sequence[object]) -> tuple[str, ...]:
    selected = {
        coordinate
        for span in spans
        if getattr(span, "kind") == StructureKind.BIB
        for coordinate in getattr(span, "line_indices")
    }
    return tuple("BIB" if line.abs_idx in selected else "O" for line in document.lines)


def _prediction_sets(
    document: GoldDocument,
) -> tuple[dict[str, tuple[str, ...]], dict[int, BibliographyV2Evidence]]:
    r2_decision = _structure_decision(document)
    r2_proposal = _span_labels(
        document,
        [span for span in r2_decision.spans if span.kind == StructureKind.BIB],
    )
    r2_safe = tuple(
        "BIB" if label == "BIB" else "O"
        for label in predict_document(document, AblationMode.RULES_ONLY).labels
    )

    v2_evidence = _v2_evidence(document)
    by_coordinate = {item.line_index: item for item in v2_evidence if item.text.strip()}
    v2_spans = decode_bibliography_blocks_v2(v2_evidence)
    v2_block = _span_labels(document, v2_spans)
    v2_local = tuple(
        "BIB"
        if by_coordinate[line.abs_idx].role
        in {BibRole.STRONG_ENTRY_START, BibRole.WEAK_ENTRY_START}
        else "O"
        for line in document.lines
    )
    return (
        {
            "r2_safe_action": r2_safe,
            "r2_all_proposals": r2_proposal,
            "v2_local_evidence": v2_local,
            "v2_coherent_blocks": v2_block,
        },
        by_coordinate,
    )


def _runs(document: GoldDocument, labels: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    last: int | None = None
    for line, label in zip(document.lines, labels):
        if label == "BIB":
            if start is None:
                start = line.abs_idx
            last = line.abs_idx
        elif start is not None:
            assert last is not None
            spans.append((start, last))
            start = last = None
    if start is not None:
        assert last is not None
        spans.append((start, last))
    return spans


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return intersection / union


def _empty_counts() -> collections.Counter[str]:
    return collections.Counter()


def _document_counts(
    document: GoldDocument, predicted: Sequence[str]
) -> collections.Counter[str]:
    counts = _empty_counts()
    gold = tuple(
        line.label if line.label != "UNKNOWN" else "O" for line in document.lines
    )
    for line, target, guess in zip(document.lines, gold, predicted):
        if line.label == "UNKNOWN":
            continue
        counts["known"] += 1
        counts["gold_bib"] += int(target == "BIB")
        counts["pred_bib"] += int(guess == "BIB")
        counts["tp"] += int(target == "BIB" and guess == "BIB")
        counts["fp"] += int(target != "BIB" and guess == "BIB")
        counts["fn"] += int(target == "BIB" and guess != "BIB")
        counts["tn"] += int(target != "BIB" and guess != "BIB")
        counts["token_tp"] += line.token_count * int(target == "BIB" and guess == "BIB")
        counts["token_fp"] += line.token_count * int(target != "BIB" and guess == "BIB")
        counts["token_fn"] += line.token_count * int(target == "BIB" and guess != "BIB")
    counts["documents"] = 1
    counts["documents_without_fp"] = int(counts["fp"] == 0)
    gold_spans = _runs(document, gold)
    predicted_spans = _runs(document, predicted)
    counts["gold_spans"] = len(gold_spans)
    counts["pred_spans"] = len(predicted_spans)
    counts["exact_pred"] = sum(span in gold_spans for span in predicted_spans)
    counts["exact_gold"] = sum(span in predicted_spans for span in gold_spans)
    counts["iou50_pred"] = sum(
        any(_iou(p, g) >= 0.5 for g in gold_spans) for p in predicted_spans
    )
    counts["iou50_gold"] = sum(
        any(_iou(g, p) >= 0.5 for p in predicted_spans) for g in gold_spans
    )
    return counts


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(counts: Mapping[str, float]) -> dict[str, Any]:
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    token_precision = _ratio(
        counts["token_tp"], counts["token_tp"] + counts["token_fp"]
    )
    token_recall = _ratio(counts["token_tp"], counts["token_tp"] + counts["token_fn"])
    return {
        "line": {
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        },
        "token": {
            "precision": token_precision,
            "recall": token_recall,
            "f1": _ratio(
                2 * token_precision * token_recall, token_precision + token_recall
            ),
        },
        "span": {
            "exact_precision": _ratio(counts["exact_pred"], counts["pred_spans"]),
            "exact_recall": _ratio(counts["exact_gold"], counts["gold_spans"]),
            "iou50_precision": _ratio(counts["iou50_pred"], counts["pred_spans"]),
            "iou50_recall": _ratio(counts["iou50_gold"], counts["gold_spans"]),
            "predicted": counts["pred_spans"],
            "gold": counts["gold_spans"],
        },
        "document_without_false_positive_fraction": _ratio(
            counts["documents_without_fp"], counts["documents"]
        ),
    }


def _evaluate(
    documents: Sequence[GoldDocument],
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for detector in next(iter(predictions.values())):
        overall = _empty_counts()
        sources: dict[str, collections.Counter[str]] = {}
        for document in documents:
            counts = _document_counts(
                document, predictions[document.document_id][detector]
            )
            overall.update(counts)
            sources.setdefault(document.source, _empty_counts()).update(counts)
        result[detector] = {
            "overall": _metrics(overall),
            "by_source": {
                source: _metrics(counts) for source, counts in sorted(sources.items())
            },
        }
    return result


def _feature_summary(
    documents: Sequence[GoldDocument],
    evidence: Mapping[str, Mapping[int, BibliographyV2Evidence]],
) -> dict[str, Any]:
    labels = {"BIB": _empty_counts(), "O": _empty_counts(), "TOC": _empty_counts()}
    row_counts = collections.Counter()
    for document in documents:
        for line in document.lines:
            if line.label == "UNKNOWN":
                continue
            bucket = labels[line.label]
            row_counts[line.label] += 1
            for name, value in asdict(
                evidence[document.document_id][line.abs_idx].features
            ).items():
                bucket[name] += value
    return {
        label: {
            "line_count": row_counts[label],
            "mean_per_line": {
                name: value / row_counts[label] if row_counts[label] else 0.0
                for name, value in sorted(counts.items())
            },
        }
        for label, counts in labels.items()
    }


def _error_examples(
    documents: Sequence[GoldDocument],
    predictions: Mapping[str, Mapping[str, Sequence[str]]],
    evidence: Mapping[str, Mapping[int, BibliographyV2Evidence]],
    *,
    limit: int = 40,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {"false_positive": [], "false_negative": []}
    for document in documents:
        guesses = predictions[document.document_id]["v2_coherent_blocks"]
        for line, guess in zip(document.lines, guesses):
            kind = (
                "false_positive"
                if line.label != "BIB" and guess == "BIB"
                else "false_negative"
                if line.label == "BIB" and guess != "BIB"
                else None
            )
            if kind is None or len(rows[kind]) >= limit:
                continue
            item = evidence[document.document_id][line.abs_idx]
            rows[kind].append(
                {
                    "document_id": document.document_id,
                    "source": document.source,
                    "abs_idx": line.abs_idx,
                    "text": line.text,
                    "gold": line.label,
                    "predicted": guess,
                    "local_role": item.role.value,
                    "score": item.score,
                    "reason_codes": list(item.reason_codes),
                    "nonzero_features": {
                        name: value
                        for name, value in asdict(item.features).items()
                        if value
                    },
                }
            )
    return rows


def run_evaluation(silver_path: str | Path, *, split: str) -> dict[str, Any]:
    _require(
        split in ALLOWED_SPLITS, "split must be train or validation; test is forbidden"
    )
    all_documents = read_gold(silver_path)
    documents = [document for document in all_documents if document.split == split]
    _require(bool(documents), f"no documents in {split!r}")
    prediction_rows: dict[str, dict[str, tuple[str, ...]]] = {}
    evidence_rows: dict[str, dict[int, BibliographyV2Evidence]] = {}
    for document in documents:
        predictions, evidence = _prediction_sets(document)
        prediction_rows[document.document_id] = predictions
        evidence_rows[document.document_id] = evidence
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "comparison_only",
        "evidence_tier": "LLM_silver",
        "split": split,
        "document_count": len(documents),
        "source_counts": dict(
            sorted(collections.Counter(d.source for d in documents).items())
        ),
        "input": {
            "path": str(Path(silver_path).resolve()),
            "sha256": sha256_file(silver_path),
        },
        "components": {
            "r2": {"rules": R2_RULES_ID, "decoder": R2_DECODER_ID},
            "v2": {"rules": V2_RULES_ID, "decoder": V2_DECODER_ID},
        },
        "metrics": _evaluate(documents, prediction_rows),
        "feature_summary_by_gold_label": _feature_summary(documents, evidence_rows),
        "v2_error_examples": _error_examples(documents, prediction_rows, evidence_rows),
        "claims": {
            "model_fitted": False,
            "test_or_sealed_data_accessed": False,
            "human_gold_used": False,
            "production_eligible": False,
            "corpus_mutated": False,
        },
        "caveats": [
            "Metrics measure agreement with GPT-labelled LLM-silver, not human-gold accuracy.",
            "The train split is exploratory; validation must be run only after rule choices are frozen.",
            "V2 is a research proposal and cannot authorise bibliography removal.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--split", choices=sorted(ALLOWED_SPLITS), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    report = run_evaluation(args.silver, split=args.split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "split": report["split"],
                "document_count": report["document_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
