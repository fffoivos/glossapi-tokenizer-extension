#!/usr/bin/env python3
"""Audit bibliography/ToC spans against canonical Parquet.

The audit tokenizes complete counterfactual document variants with the pinned
ModernGreek-148k tokenizer. It never estimates exact loss by tokenizing isolated
spans, because retained BPE boundaries and overlapping spans are non-additive.

This phase is intentionally audit-only. A later source/model/run-bound
materializer will consume an approved detector-run manifest; this script cannot
write cleaned corpus shards.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import hashlib
import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


BIB_KIND = "bib_span"
TOC_KIND = "toc_span"
KNOWN_KINDS = (BIB_KIND, TOC_KIND)


@dataclass(frozen=True, slots=True)
class Span:
    source: str
    doc_id: str
    row_uid: str
    original_sha256: str
    original_chars: int
    model_id: str
    kind: str
    char_start: int
    char_end: int
    line_start: int | None
    line_end: int | None
    trigger: str
    gated_by: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def load_spans(path: Path) -> tuple[dict[tuple[str, str], list[Span]], dict[str, int]]:
    spans: dict[tuple[str, str], list[Span]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                span = Span(
                    source=str(row["source"]),
                    doc_id=str(row["doc_id"]),
                    row_uid=str(row["row_uid"]),
                    original_sha256=str(row["original_sha256"]),
                    original_chars=int(row["original_chars"]),
                    model_id=str(row["model_id"]),
                    kind=str(row["kind"]),
                    char_start=int(row["char_start"]),
                    char_end=int(row["char_end"]),
                    line_start=int(row["line_start"]) if row.get("line_start") is not None else None,
                    line_end=int(row["line_end"]) if row.get("line_end") is not None else None,
                    trigger=str(row.get("trigger", "")),
                    gated_by=str(row.get("gated_by", "")),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed span: {exc}") from exc
            if span.kind not in KNOWN_KINDS:
                counts[f"ignored_kind:{span.kind}"] += 1
                continue
            expected_uid = hashlib.sha256(f"{span.source}\0{span.doc_id}".encode()).hexdigest()
            if span.row_uid != expected_uid:
                raise ValueError(f"{path}:{line_number}: row_uid does not bind source/doc_id")
            spans[(span.source, span.doc_id)].append(span)
            counts[span.kind] += 1

    for source_doc, doc_spans in spans.items():
        unique = set(doc_spans)
        if len(unique) != len(doc_spans):
            counts["duplicate_span_records"] += len(doc_spans) - len(unique)
            spans[source_doc] = list(unique)
        spans[source_doc].sort(key=lambda s: (s.char_start, s.char_end, s.kind))
    counts["documents"] = len(spans)
    return dict(spans), dict(counts)


def line_of_offset(newlines: Sequence[int], offset: int) -> int:
    """Return zero-based line containing a character offset."""
    return bisect.bisect_left(newlines, offset)


def validate_spans(text: str, spans: Sequence[Span], *, doc_id: str) -> None:
    actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    newlines = [i for i, char in enumerate(text) if char == "\n"]
    for span in spans:
        if span.original_chars != len(text) or span.original_sha256 != actual_sha256:
            raise ValueError(
                f"{doc_id}: immutable text binding mismatch for {span.kind}; "
                "span ledger and input text are not the same document"
            )
        if not 0 <= span.char_start < span.char_end <= len(text):
            raise ValueError(
                f"{doc_id}: invalid {span.kind} offsets [{span.char_start},{span.char_end}) "
                f"for {len(text)} characters"
            )
        if span.line_start is not None:
            actual = line_of_offset(newlines, span.char_start)
            if actual != span.line_start:
                raise ValueError(
                    f"{doc_id}: {span.kind} line_start={span.line_start}, offset implies {actual}; "
                    "span ledger and text are not the same immutable document"
                )
        if span.line_end is not None:
            actual = line_of_offset(newlines, span.char_end - 1)
            if actual != span.line_end:
                raise ValueError(
                    f"{doc_id}: {span.kind} line_end={span.line_end}, offset implies {actual}; "
                    "span ledger and text are not the same immutable document"
                )


def merge_ranges(spans: Sequence[Span], allowed_kinds: set[str]) -> list[tuple[int, int]]:
    ranges = sorted(
        ((span.char_start, span.char_end) for span in spans if span.kind in allowed_kinds),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def overlap_ranges(spans: Sequence[Span]) -> list[tuple[int, int]]:
    bib = merge_ranges(spans, {BIB_KIND})
    toc = merge_ranges(spans, {TOC_KIND})
    intersections: list[tuple[int, int]] = []
    left = right = 0
    while left < len(bib) and right < len(toc):
        start = max(bib[left][0], toc[right][0])
        end = min(bib[left][1], toc[right][1])
        if start < end:
            intersections.append((start, end))
        if bib[left][1] <= toc[right][1]:
            left += 1
        else:
            right += 1
    return intersections


def apply_ranges(text: str, ranges: Sequence[tuple[int, int]]) -> str:
    """Replace each merged range with at most two newlines, preserving all other text."""
    if not ranges:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in ranges:
        parts.append(text[cursor:start])
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < len(text) else ""
        if left and right:
            if left == "\n" and right == "\n":
                replacement = ""
            elif left == "\n" or right == "\n":
                replacement = "\n"
            else:
                replacement = "\n\n"
            parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def apply_spans(text: str, spans: Sequence[Span], allowed_kinds: Iterable[str]) -> str:
    return apply_ranges(text, merge_ranges(spans, set(allowed_kinds)))


def removed_text(text: str, spans: Sequence[Span], allowed_kinds: Iterable[str]) -> str:
    return "\n".join(text[start:end] for start, end in merge_ranges(spans, set(allowed_kinds)))


def script_counts(text: str) -> tuple[int, int, int]:
    greek = latin = polytonic = 0
    for char in text:
        codepoint = ord(char)
        if "A" <= char <= "Z" or "a" <= char <= "z":
            latin += 1
        elif 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
            greek += 1
            if 0x1F00 <= codepoint <= 0x1FFF:
                polytonic += 1
    return greek, latin, polytonic


def expand_inputs(values: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.parquet")))
        elif any(char in value for char in "*?["):
            found.extend(Path(item) for item in sorted(glob.glob(value, recursive=True)))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(value)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        canonical = path.resolve()
        if canonical.suffix != ".parquet":
            continue
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    if not resolved:
        raise ValueError("no Parquet inputs resolved")
    return resolved


class ExactTokenizer:
    def __init__(self, tokenizer_json: Path, expected_sha256: str | None) -> None:
        actual = sha256_file(tokenizer_json)
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(
                f"tokenizer hash mismatch: expected {expected_sha256}, got {actual} ({tokenizer_json})"
            )
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised on Clariden
            raise RuntimeError("install the `tokenizers` package in the Clariden environment") from exc
        self.path = tokenizer_json.resolve()
        self.sha256 = actual
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def counts(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encodings = self.tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class Aggregates:
    NUMERIC_FIELDS = (
        "documents",
        "documents_with_bib",
        "documents_with_toc",
        "documents_with_any",
        "documents_empty_after",
        "characters_before",
        "characters_after",
        "bytes_before",
        "bytes_after",
        "tokens_before",
        "tokens_after_bib",
        "tokens_after_toc",
        "tokens_after_both",
        "tokens_removed_bib",
        "tokens_removed_toc",
        "tokens_removed_both",
        "interaction_tokens",
        "unique_candidate_characters",
        "bib_toc_overlap_characters",
        "documents_with_bib_toc_overlap",
        "bib_spans",
        "toc_spans",
        "bib_removed_greek_chars",
        "bib_removed_latin_chars",
        "bib_removed_polytonic_chars",
        "toc_removed_greek_chars",
        "toc_removed_latin_chars",
        "toc_removed_polytonic_chars",
    )

    def __init__(self) -> None:
        self.by_source: dict[str, dict[str, int | list[int]]] = {}

    def add(self, source: str, row: dict[str, object]) -> None:
        if source not in self.by_source:
            self.by_source[source] = {field: 0 for field in self.NUMERIC_FIELDS}
            self.by_source[source]["document_combined_losses"] = []
        target = self.by_source[source]
        for field in self.NUMERIC_FIELDS:
            target[field] = int(target[field]) + int(row[field])
        losses = target["document_combined_losses"]
        assert isinstance(losses, list)
        if int(row["tokens_removed_both"]) != 0:
            losses.append(int(row["tokens_removed_both"]))

    def summaries(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for source, values in sorted(self.by_source.items()):
            row = {"source": source}
            for field in self.NUMERIC_FIELDS:
                row[field] = int(values[field])
            losses = values["document_combined_losses"]
            assert isinstance(losses, list)
            before = int(values["tokens_before"])
            removed = int(values["tokens_removed_both"])
            row.update(
                {
                    "token_loss_fraction": (removed / before) if before else 0.0,
                    "affected_loss_p50": percentile(losses, 0.50),
                    "affected_loss_p90": percentile(losses, 0.90),
                    "affected_loss_p99": percentile(losses, 0.99),
                    "affected_loss_max": max(losses, default=0),
                    "tokens_before_with_eod": before + int(values["documents"]),
                    "tokens_after_with_eod": int(values["tokens_after_both"]) + int(values["documents"]),
                }
            )
            result.append(row)
        return result


def per_doc_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("source", pa.string()),
            ("doc_id", pa.string()),
            ("input_file", pa.string()),
            ("original_sha256", pa.string()),
            ("cleaned_sha256", pa.string()),
            ("characters_before", pa.int64()),
            ("characters_after", pa.int64()),
            ("bytes_before", pa.int64()),
            ("bytes_after", pa.int64()),
            ("tokens_before", pa.int64()),
            ("tokens_after_bib", pa.int64()),
            ("tokens_after_toc", pa.int64()),
            ("tokens_after_both", pa.int64()),
            ("tokens_removed_bib", pa.int64()),
            ("tokens_removed_toc", pa.int64()),
            ("tokens_removed_both", pa.int64()),
            ("interaction_tokens", pa.int64()),
            ("unique_candidate_characters", pa.int64()),
            ("bib_toc_overlap_characters", pa.int64()),
            ("bib_spans", pa.int32()),
            ("toc_spans", pa.int32()),
            ("bib_removed_greek_chars", pa.int64()),
            ("bib_removed_latin_chars", pa.int64()),
            ("bib_removed_polytonic_chars", pa.int64()),
            ("toc_removed_greek_chars", pa.int64()),
            ("toc_removed_latin_chars", pa.int64()),
            ("toc_removed_polytonic_chars", pa.int64()),
            ("empty_after", pa.bool_()),
        ]
    )


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_source(batch: dict[str, list[object]], index: int, source_column: str | None, source_name: str | None) -> str:
    if source_name:
        return source_name
    if source_column:
        value = batch[source_column][index]
        if value is not None and str(value):
            return str(value)
    raise ValueError("source unavailable: pass --source-column or --source-name")


def validate_detector_manifest(
    path: Path,
    spans_path: Path,
    input_receipt_path: Path,
    source_name: str | None,
    source_regex: str | None,
) -> tuple[dict, dict[tuple[str, str], dict]]:
    manifest = load_json(path)
    input_receipt = load_json(input_receipt_path)
    if manifest.get("schema_version") != "structural_detector_run_v1" or manifest.get("status") != "passed":
        raise ValueError("token audit requires a passed structural_detector_run_v1 manifest")
    if source_name and manifest.get("source") != source_name:
        raise ValueError("detector manifest/source-name mismatch")
    if manifest.get("spans", {}).get("sha256") != sha256_file(spans_path):
        raise ValueError("span ledger hash does not match detector run manifest")
    if (manifest.get("stream_manifest", {}).get("source_regex") or "") != (source_regex or ""):
        raise ValueError("detector manifest/source-regex mismatch")
    if (
        input_receipt.get("schema_version") != "full_cpt_acquisition_receipt_v1"
        or input_receipt.get("status") != "passed"
    ):
        raise ValueError("token audit requires a passed full_cpt_acquisition_receipt_v1")
    if manifest.get("input_receipt_sha256") != sha256_file(input_receipt_path):
        raise ValueError("token audit input receipt differs from the detector run input receipt")
    counters_path = path.parent / manifest.get("counters", {}).get("path", "")
    if not counters_path.is_file() or manifest.get("counters", {}).get("sha256") != sha256_file(counters_path):
        raise ValueError("detector counter ledger is missing or hash-mismatched")
    counters: dict[tuple[str, str], dict] = {}
    with counters_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" in row:
                raise ValueError(f"detector counter ledger contains error at line {line_number}")
            key = (str(row["source"]), str(row["doc_id"]))
            if key in counters:
                raise ValueError(f"duplicate detector counter key: {key}")
            counters[key] = row
    return manifest, counters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Parquet path/dir/glob; repeatable")
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="source_doc_id")
    parser.add_argument("--source-column", default="source_dataset")
    parser.add_argument("--source-name")
    parser.add_argument("--source-regex")
    parser.add_argument("--detector-run-manifest", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--kind", action="append", choices=KNOWN_KINDS, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--review-count", type=int, default=100)
    parser.add_argument("--materialize-dir", type=Path)
    parser.add_argument("--allow-unmatched-spans", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.materialize_dir:
        parser.error(
            "materialization is intentionally disabled in Phase 04; approve a source/model/run-bound "
            "manifest and implement the quarantine/atomic-output materializer in the later phase"
        )
    kinds = set(args.kind or KNOWN_KINDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector_manifest, detector_counters = validate_detector_manifest(
        args.detector_run_manifest,
        args.spans,
        args.input_receipt,
        args.source_name,
        args.source_regex,
    )
    source_pattern = re.compile(args.source_regex) if args.source_regex else None

    inputs = expand_inputs(args.input)
    spans_by_doc, span_inventory = load_spans(args.spans)
    tokenizer = ExactTokenizer(args.tokenizer_json, args.expected_tokenizer_sha256)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on Clariden
        raise RuntimeError("install pyarrow in the Clariden environment") from exc

    per_doc_path = args.output_dir / "structural_token_loss_per_doc.parquet"
    per_doc_writer = pq.ParquetWriter(per_doc_path, per_doc_schema(), compression="zstd")
    aggregates = Aggregates()
    matched_span_docs: set[tuple[str, str]] = set()
    matched_counter_docs: set[tuple[str, str]] = set()
    seen_doc_ids: set[tuple[str, str]] = set()
    top_review: list[tuple[int, str, dict[str, object]]] = []
    overlap_reviews: list[dict[str, object]] = []
    input_inventory: list[dict[str, object]] = []
    rows_processed = 0

    try:
        for input_path in inputs:
            parquet = pq.ParquetFile(input_path)
            names = set(parquet.schema_arrow.names)
            required = {args.text_column, args.id_column}
            if args.source_column and (source_pattern or not args.source_name):
                required.add(args.source_column)
            missing = sorted(required - names)
            if missing:
                if args.source_name and missing == [args.source_column]:
                    required.remove(args.source_column)
                else:
                    raise ValueError(f"{input_path}: missing columns {missing}; has {sorted(names)}")
            columns = [args.id_column, args.text_column]
            if args.source_column in names:
                columns.append(args.source_column)
            input_inventory.append(
                {
                    "path": str(input_path),
                    "bytes": input_path.stat().st_size,
                    "rows": parquet.metadata.num_rows,
                    "row_groups": parquet.num_row_groups,
                }
            )

            try:
                for record_batch in parquet.iter_batches(batch_size=args.batch_size):
                    if args.max_rows and rows_processed >= args.max_rows:
                        break
                    if source_pattern:
                        source_index = record_batch.schema.get_field_index(args.source_column)
                        mask = [
                            bool(source_pattern.search(str(value or "")))
                            for value in record_batch.column(source_index).to_pylist()
                        ]
                        if not any(mask):
                            continue
                        record_batch = record_batch.filter(pa.array(mask))
                    row_count = record_batch.num_rows
                    if args.max_rows:
                        row_count = min(row_count, args.max_rows - rows_processed)
                        record_batch = record_batch.slice(0, row_count)
                    selected = record_batch.select(columns).to_pydict()
                    originals: list[str] = []
                    doc_ids: list[str] = []
                    sources: list[str] = []
                    doc_spans_list: list[list[Span]] = []
                    bib_texts: list[str] = []
                    toc_texts: list[str] = []
                    both_texts: list[str] = []

                    for index in range(row_count):
                        doc_id = str(selected[args.id_column][index])
                        raw_text = selected[args.text_column][index]
                        text = "" if raw_text is None else str(raw_text)
                        source = choose_source(selected, index, args.source_column if args.source_column in selected else None, args.source_name)
                        source_doc = (source, doc_id)
                        if source_doc in seen_doc_ids:
                            raise ValueError(f"duplicate source/document id in audit input: {source_doc}")
                        seen_doc_ids.add(source_doc)
                        counter = detector_counters.get(source_doc)
                        if counter is None:
                            raise ValueError(f"input document was not certified by detector counters: {source_doc}")
                        actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if (
                            counter.get("original_sha256") != actual_sha256
                            or int(counter.get("original_chars", -1)) != len(text)
                        ):
                            raise ValueError(f"detector counter/input text binding mismatch: {source_doc}")
                        matched_counter_docs.add(source_doc)
                        doc_spans = spans_by_doc.get(source_doc, [])
                        if doc_spans:
                            validate_spans(text, doc_spans, doc_id=doc_id)
                            matched_span_docs.add(source_doc)
                        originals.append(text)
                        doc_ids.append(doc_id)
                        sources.append(source)
                        doc_spans_list.append(doc_spans)
                        bib_texts.append(apply_spans(text, doc_spans, {BIB_KIND} & kinds))
                        toc_texts.append(apply_spans(text, doc_spans, {TOC_KIND} & kinds))
                        both_texts.append(apply_spans(text, doc_spans, kinds))

                    original_counts = tokenizer.counts(originals)
                    variant_texts: list[str] = []
                    variant_keys: list[tuple[int, str]] = []
                    after_bib = list(original_counts)
                    after_toc = list(original_counts)
                    after_both = list(original_counts)
                    for index, text in enumerate(originals):
                        has_bib = any(span.kind == BIB_KIND for span in doc_spans_list[index]) and BIB_KIND in kinds
                        has_toc = any(span.kind == TOC_KIND for span in doc_spans_list[index]) and TOC_KIND in kinds
                        if has_bib:
                            variant_keys.append((index, "bib"))
                            variant_texts.append(bib_texts[index])
                        if has_toc:
                            variant_keys.append((index, "toc"))
                            variant_texts.append(toc_texts[index])
                        if has_bib and has_toc:
                            variant_keys.append((index, "both"))
                            variant_texts.append(both_texts[index])
                    for (index, variant), count in zip(variant_keys, tokenizer.counts(variant_texts)):
                        if variant == "bib":
                            after_bib[index] = count
                        elif variant == "toc":
                            after_toc[index] = count
                        else:
                            after_both[index] = count
                    for index, doc_spans in enumerate(doc_spans_list):
                        has_bib = any(span.kind == BIB_KIND for span in doc_spans) and BIB_KIND in kinds
                        has_toc = any(span.kind == TOC_KIND for span in doc_spans) and TOC_KIND in kinds
                        if has_bib and not has_toc:
                            after_both[index] = after_bib[index]
                        elif has_toc and not has_bib:
                            after_both[index] = after_toc[index]

                    audit_rows: list[dict[str, object]] = []
                    for index, text in enumerate(originals):
                        doc_spans = doc_spans_list[index]
                        bib_spans = sum(1 for span in doc_spans if span.kind == BIB_KIND and BIB_KIND in kinds)
                        toc_spans = sum(1 for span in doc_spans if span.kind == TOC_KIND and TOC_KIND in kinds)
                        bib_removed = removed_text(text, doc_spans, {BIB_KIND} & kinds)
                        toc_removed = removed_text(text, doc_spans, {TOC_KIND} & kinds)
                        bib_greek, bib_latin, bib_poly = script_counts(bib_removed)
                        toc_greek, toc_latin, toc_poly = script_counts(toc_removed)
                        cleaned = both_texts[index]
                        union_ranges = merge_ranges(doc_spans, kinds)
                        conflicts = overlap_ranges(doc_spans) if {BIB_KIND, TOC_KIND} <= kinds else []
                        unique_candidate_characters = sum(end - start for start, end in union_ranges)
                        overlap_characters = sum(end - start for start, end in conflicts)
                        delta_bib = original_counts[index] - after_bib[index]
                        delta_toc = original_counts[index] - after_toc[index]
                        delta_both = original_counts[index] - after_both[index]
                        row: dict[str, object] = {
                            "source": sources[index],
                            "doc_id": doc_ids[index],
                            "input_file": str(input_path),
                            "original_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "cleaned_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                            "characters_before": len(text),
                            "characters_after": len(cleaned),
                            "bytes_before": len(text.encode("utf-8")),
                            "bytes_after": len(cleaned.encode("utf-8")),
                            "tokens_before": original_counts[index],
                            "tokens_after_bib": after_bib[index],
                            "tokens_after_toc": after_toc[index],
                            "tokens_after_both": after_both[index],
                            "tokens_removed_bib": delta_bib,
                            "tokens_removed_toc": delta_toc,
                            "tokens_removed_both": delta_both,
                            "interaction_tokens": delta_both - delta_bib - delta_toc,
                            "unique_candidate_characters": unique_candidate_characters,
                            "bib_toc_overlap_characters": overlap_characters,
                            "bib_spans": bib_spans,
                            "toc_spans": toc_spans,
                            "bib_removed_greek_chars": bib_greek,
                            "bib_removed_latin_chars": bib_latin,
                            "bib_removed_polytonic_chars": bib_poly,
                            "toc_removed_greek_chars": toc_greek,
                            "toc_removed_latin_chars": toc_latin,
                            "toc_removed_polytonic_chars": toc_poly,
                            "empty_after": not cleaned.strip(),
                        }
                        row.update(
                            {
                                "documents": 1,
                                "documents_with_bib": int(bib_spans > 0),
                                "documents_with_toc": int(toc_spans > 0),
                                "documents_with_any": int(bib_spans + toc_spans > 0),
                                "documents_empty_after": int(not cleaned.strip()),
                                "documents_with_bib_toc_overlap": int(bool(conflicts)),
                            }
                        )
                        aggregates.add(sources[index], row)
                        audit_rows.append(row)

                        if delta_both > 0 and args.review_count:
                            review = {
                                "source": sources[index],
                                "doc_id": doc_ids[index],
                                "input_file": str(input_path),
                                "tokens_before": original_counts[index],
                                "tokens_removed_bib": delta_bib,
                                "tokens_removed_toc": delta_toc,
                                "tokens_removed_both": delta_both,
                                "removed_bib_preview": bib_removed[:1000],
                                "removed_toc_preview": toc_removed[:1000],
                                "spans": [
                                    {
                                        "kind": span.kind,
                                        "char_start": span.char_start,
                                        "char_end": span.char_end,
                                        "line_start": span.line_start,
                                        "line_end": span.line_end,
                                        "trigger": span.trigger,
                                        "gated_by": span.gated_by,
                                    }
                                    for span in doc_spans
                                    if span.kind in kinds
                                ],
                            }
                            key = f"{sources[index]}::{doc_ids[index]}"
                            item = (delta_both, key, review)
                            if len(top_review) < args.review_count:
                                heapq.heappush(top_review, item)
                            elif item[:2] > top_review[0][:2]:
                                heapq.heapreplace(top_review, item)

                        if conflicts:
                            overlap_reviews.append(
                                {
                                    "source": sources[index],
                                    "doc_id": doc_ids[index],
                                    "original_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                    "overlap_characters": overlap_characters,
                                    "overlap_ranges": conflicts,
                                    "decision": "materialization_blocked_pending_review",
                                    "overlap_previews": [text[start:end][:1000] for start, end in conflicts],
                                }
                            )

                    per_doc_writer.write_table(pa.Table.from_pylist(audit_rows, schema=per_doc_schema()))

                    rows_processed += row_count
                if args.max_rows and rows_processed >= args.max_rows:
                    break
            finally:
                pass
    finally:
        per_doc_writer.close()

    unmatched = sorted(set(spans_by_doc) - matched_span_docs)
    if unmatched and not args.allow_unmatched_spans:
        raise ValueError(
            f"{len(unmatched)} span-ledger documents were not found in inputs; first={unmatched[:5]}"
        )
    unmatched_counters = sorted(set(detector_counters) - matched_counter_docs)
    if unmatched_counters and not args.allow_unmatched_spans:
        raise ValueError(
            f"{len(unmatched_counters)} detector-counter documents were not found in token-audit inputs; "
            f"first={unmatched_counters[:5]}"
        )

    summaries = aggregates.summaries()
    write_summary_csv(args.output_dir / "structural_token_loss_by_source.csv", summaries)
    with (args.output_dir / "structural_token_loss_review.jsonl").open("w", encoding="utf-8") as handle:
        for _loss, _key, review in sorted(top_review, reverse=True):
            handle.write(json.dumps(review, ensure_ascii=False) + "\n")
    overlap_path = args.output_dir / "structural_overlap_conflicts.jsonl"
    with overlap_path.open("w", encoding="utf-8") as handle:
        for review in overlap_reviews:
            handle.write(json.dumps(review, ensure_ascii=False) + "\n")

    total: dict[str, int] = defaultdict(int)
    for row in summaries:
        for field in Aggregates.NUMERIC_FIELDS:
            total[field] += int(row[field])
    report = {
        "schema_version": "structural_token_loss_v1",
        "mode": "audit_only",
        "rows_processed": rows_processed,
        "kinds": sorted(kinds),
        "tokenizer": {
            "path": str(tokenizer.path),
            "sha256": tokenizer.sha256,
            "add_special_tokens": False,
            "eod_policy": "one per retained document; EOD loss is zero unless a document is later excluded",
        },
        "span_ledger": {
            "path": str(args.spans.resolve()),
            "sha256": sha256_file(args.spans),
            "inventory": span_inventory,
            "matched_documents": len(matched_span_docs),
            "unmatched_documents": len(unmatched),
            "unmatched_examples": unmatched[:20],
        },
        "detector_counter_coverage": {
            "certified_documents": len(detector_counters),
            "matched_documents": len(matched_counter_docs),
            "unmatched_documents": len(unmatched_counters),
        },
        "detector_run_manifest": {
            "path": str(args.detector_run_manifest.resolve()),
            "sha256": sha256_file(args.detector_run_manifest),
            "source": detector_manifest.get("source"),
            "binary_sha256": detector_manifest.get("binary_sha256"),
        },
        "input_receipt": {
            "path": str(args.input_receipt.resolve()),
            "sha256": sha256_file(args.input_receipt),
        },
        "inputs": input_inventory,
        "by_source": summaries,
        "total": dict(total),
        "outputs": {
            "per_doc_parquet": str(per_doc_path),
            "by_source_csv": str(args.output_dir / "structural_token_loss_by_source.csv"),
            "review_jsonl": str(args.output_dir / "structural_token_loss_review.jsonl"),
            "overlap_conflicts_jsonl": str(overlap_path),
        },
    }
    report_path = args.output_dir / "structural_token_loss_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"ok": True, "rows": rows_processed, "summary": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
