#!/usr/bin/env python3
"""Shared bounded-memory primitives for the two-pass Phase-04 cleaner.

This module deliberately contains no CLI.  Stage 50 and the structural-last
finalizer share these schemas and exact token-counting semantics without
sharing their transformations: source cleanup/PII may only happen in Stage 50.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from full_corpus_io import canonical_schema, normalize_text, sha256_file, sha256_text


CLEANING_IMPLEMENTATION_VERSION = "full-cpt-two-pass-cleaner-v2"
HEX_SHA256 = frozenset("0123456789abcdef")
STRUCTURAL_KINDS = {"toc": "toc", "toc_span": "toc", "bibliography": "bibliography", "bib_span": "bibliography"}


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - HEX_SHA256)
    )


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_receipt(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = resolved
    if relative_to is not None:
        try:
            display = resolved.relative_to(relative_to.resolve())
        except ValueError:
            display = resolved
    return {
        "path": display.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_receipt(receipt: Mapping[str, Any], *, relative_to: Path | None = None) -> Path:
    path = Path(str(receipt.get("path", "")))
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt.get("bytes", -1))
        or sha256_file(path) != receipt.get("sha256")
    ):
        raise ValueError(f"file receipt verification failed: {path}")
    return path


def cleaned_schema():
    import pyarrow as pa

    return pa.schema(
        [
            *canonical_schema(),
            ("cleaned_text_sha256", pa.string()),
            ("cleaning_trace_json", pa.string()),
            ("pii_by_type_json", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("eligible_for_redistribution", pa.bool_()),
        ]
    )


def ledger_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("action", pa.string()),
            ("reasons_json", pa.string()),
            ("tokens_normalized", pa.int64()),
            ("tokens_source_cleaned", pa.int64()),
            ("tokens_pii_masked", pa.int64()),
            ("tokens_toc_removed", pa.int64()),
            ("tokens_bibliography_removed", pa.int64()),
            ("tokens_structural_union_removed", pa.int64()),
            ("tokens_structural_cleaned", pa.int64()),
            ("tokens_final", pa.int64()),
            ("characters_normalized", pa.int64()),
            ("characters_final", pa.int64()),
            # Hash of the would-be emitted text after this pass.  Kept rows
            # appear in corpus; dropped/quarantined rows retain it for audit.
            ("final_text_sha256", pa.string()),
            ("pii_by_type_json", pa.string()),
            ("source_admission_decision", pa.string()),
            ("training_eligibility_category", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("eligible_for_redistribution", pa.bool_()),
        ]
    )


def encode_counts(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """Count a batch exactly, de-duplicating equal variants within the batch."""

    unique: list[str] = []
    positions: dict[str, int] = {}
    indexes: list[int] = []
    for text in texts:
        index = positions.get(text)
        if index is None:
            index = len(unique)
            positions[text] = index
            unique.append(text)
        indexes.append(index)
    if not unique:
        return []
    encoded = tokenizer.encode_batch(unique, add_special_tokens=False)
    counts = [len(item.ids) for item in encoded]
    return [counts[index] for index in indexes]


def count_stage50_versions(tokenizer: Any, rows: Sequence[dict[str, Any]]) -> None:
    """Populate exact counts while reusing counts for unchanged text versions."""

    variants: list[str] = []
    needed: list[tuple[int, tuple[str, ...]]] = []
    for index, row in enumerate(rows):
        by_text: dict[str, list[str]] = {}
        for field, text_field in (
            ("tokens_normalized", "_normalized"),
            ("tokens_source_cleaned", "_source_cleaned"),
            ("tokens_pii_masked", "_pii_masked"),
        ):
            text = str(row[text_field])
            by_text.setdefault(text, []).append(field)
        for text, fields in by_text.items():
            variants.append(text)
            needed.append((index, tuple(fields)))
    for (index, fields), count in zip(needed, encode_counts(tokenizer, variants), strict=True):
        for field in fields:
            rows[index][field] = count
    for row in rows:
        row["tokens_toc_removed"] = 0
        row["tokens_bibliography_removed"] = 0
        row["tokens_structural_union_removed"] = 0
        row["tokens_structural_cleaned"] = row["tokens_pii_masked"]


def apply_structural_spans(
    text: str, spans: Iterable[Mapping[str, Any]], *, allowed_kinds: set[str] | None = None
) -> tuple[str, list[str], set[str]]:
    """Apply already-validated, non-overlapping spans with stable replacement."""

    allowed = allowed_kinds or {"toc", "bibliography"}
    intervals: list[tuple[int, int, str, str]] = []
    actual_hash = sha256_text(text)
    for span in spans:
        if span.get("input_text_sha256") != actual_hash:
            raise ValueError("structural span is not bound to the exact post-PII input text")
        raw_kind = str(span.get("kind") or span.get("rule_id") or "")
        kind = STRUCTURAL_KINDS.get(raw_kind)
        if kind is None:
            raise ValueError(f"unsupported structural span kind: {raw_kind!r}")
        if kind not in allowed:
            continue
        start, end = int(span["char_start"]), int(span["char_end"])
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"invalid structural span {start}:{end} for text length {len(text)}")
        intervals.append((start, end, kind, str(span.get("rule_id") or raw_kind)))
    intervals.sort(key=lambda value: (value[0], value[1], value[2]))
    previous_end = -1
    for start, end, _, _ in intervals:
        if start < previous_end:
            raise ValueError("structural spans overlap; inventory must provide a disjoint union")
        previous_end = end
    output = text
    for start, end, _, _ in reversed(intervals):
        output = output[:start] + "\n\n" + output[end:]
    return (
        normalize_text(output),
        [reason for _, _, _, reason in intervals],
        {kind for _, _, kind, _ in intervals},
    )


def structural_counterfactuals(
    text: str, spans: Sequence[Mapping[str, Any]]
) -> tuple[str, str, str]:
    toc, _, _ = apply_structural_spans(text, spans, allowed_kinds={"toc"}) if any(
        STRUCTURAL_KINDS.get(str(span.get("kind") or span.get("rule_id") or "")) == "toc"
        for span in spans
    ) else (text, [], set())
    bibliography, _, _ = apply_structural_spans(
        text, spans, allowed_kinds={"bibliography"}
    ) if any(
        STRUCTURAL_KINDS.get(str(span.get("kind") or span.get("rule_id") or "")) == "bibliography"
        for span in spans
    ) else (text, [], set())
    union, _, _ = apply_structural_spans(text, spans)
    return toc, bibliography, union


def per_file_receipt_path(receipt_root: Path, relative: Path) -> Path:
    return receipt_root / relative.parent / f"{relative.name}.json"


def require_exact_parquet_tree(root: Path, relatives: Iterable[Path]) -> None:
    expected = {(root / relative).resolve() for relative in relatives}
    observed = {path.resolve() for path in root.rglob("*.parquet")} if root.exists() else set()
    if observed != expected:
        missing = sorted(str(path) for path in expected - observed)
        extra = sorted(str(path) for path in observed - expected)
        raise ValueError(
            f"Parquet tree inventory mismatch beneath {root}: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )


def reusable_file_receipt(
    receipt_path: Path,
    *,
    input_path: Path,
    config_sha256: str,
    roots: Mapping[str, Path],
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version") != "full_cpt_cleaning_file_receipt_v1"
            or receipt.get("implementation_version") != CLEANING_IMPLEMENTATION_VERSION
            or receipt.get("config_sha256") != config_sha256
            or receipt.get("input", {}).get("sha256") != sha256_file(input_path)
            or receipt.get("input", {}).get("bytes") != input_path.stat().st_size
        ):
            return None
        for name, root in roots.items():
            verify_file_receipt(receipt[name], relative_to=root)
        return receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def write_parquet_batches(path: Path, schema: Any, batches: Iterable[Sequence[Mapping[str, Any]]]) -> None:
    """Atomically stream Python-row batches to one deterministic Parquet file."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        wrote = False
        for rows in batches:
            if rows:
                writer.write_table(pa.Table.from_pylist(list(rows), schema=schema))
                wrote = True
        if not wrote:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
        writer.close()
        os.replace(temporary, path)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
