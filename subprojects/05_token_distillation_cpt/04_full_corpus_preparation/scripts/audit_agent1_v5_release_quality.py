#!/usr/bin/env python3
"""Audit Agent-1 v5 candidate-release quality, structure, and lineage.

The audit is intentionally independent of publication.  It opens every
candidate Parquet shard, checks the canonical schema and compression, validates
the document envelope and JSON metadata, profiles residual extraction
artifacts, and proves that document identity, cleaned text, and lineage fields
survived the GlossAPI -> release boundary byte-for-byte. It also proves that
metadata remained attached to the same row across transform -> GlossAPI. The
compact JSON receipt contains aggregate counters and hashes only, never corpus
text or absolute worker paths.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import heapq
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA = "agent1_v5_release_quality_audit_v2"
TRANSFORM_METADATA_FIELDS = (
    "source_row_uid",
    "source_dataset",
    "source_doc_id_candidate",
    "title",
    "author",
    "source_metadata_json",
)
RELEASE_LINEAGE_FIELDS = (
    "source_dataset",
    "source_doc_id",
    "text_sha256",
    "title",
    "author",
    "source_metadata_json",
)
# Retained as the public name used by the unit-level hash contract.
PRESERVED_FIELDS = RELEASE_LINEAGE_FIELDS
SCAN_COLUMNS = (
    "source_dataset",
    "source_doc_id",
    "text",
    "title",
    "author",
    "source_metadata_json",
    "greek_percentage",
    "latin_percentage",
    "greek_badness_score",
    "mojibake_badness_score",
    "is_empty",
    "filter",
    "quality_method",
    "reevaluated_at",
    "cleaner_chars_before",
    "cleaner_chars_after",
    "chars",
    "non_whitespace_chars",
    "utf8_bytes",
    "approx_word_count",
)
EXPECTED_QUALITY_METHOD = "glossapi_rs_cleaner+glossapi_rs_noise"
GENERATED_IMAGE_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[^\s()<>{}\[\]]*/)?"
    r"[0-9a-f]{32,64}(?:_[0-9]+)+_img\.(?:avif|bmp|gif|jpe?g|png|tiff?|webp)"
    r"(?![0-9A-Za-z])",
    re.IGNORECASE,
)
KNOWN_HTML_TAGS = {
    "a",
    "abbr",
    "acronym",
    "address",
    "article",
    "aside",
    "audio",
    "b",
    "big",
    "blockquote",
    "body",
    "br",
    "button",
    "canvas",
    "caption",
    "center",
    "cite",
    "code",
    "dd",
    "del",
    "details",
    "dfn",
    "dialog",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "figcaption",
    "figure",
    "font",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "li",
    "link",
    "main",
    "mark",
    "math",
    "meta",
    "nav",
    "noscript",
    "object",
    "ol",
    "option",
    "p",
    "picture",
    "pre",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "script",
    "section",
    "select",
    "small",
    "source",
    "span",
    "strike",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "svg",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "track",
    "u",
    "ul",
    "var",
    "video",
    "wbr",
}
HTML_RE = re.compile(
    r"</?(?:"
    + "|".join(sorted(map(re.escape, KNOWN_HTML_TAGS), key=len, reverse=True))
    + r")(?:\s[^<>]*?)?/?>",
    re.IGNORECASE,
)
MOJIBAKE_RE = re.compile(r"(?:\ufffd|Ã.|Â.|Î.|Ï.|Ð.|Ñ.)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPETITION_MARKER = "<!-- repeating-text-removed -->"
IMAGE_DESCRIPTION_MARKER = "<!-- description-of-removed-image:"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_binding(
    binding: Mapping[str, Any], root: Path, *, containment_root: Path | None = None
) -> Path:
    path = Path(str(binding["path"]))
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if containment_root is not None:
        boundary = containment_root.resolve(strict=True)
        if not path.is_relative_to(boundary):
            raise ValueError(f"receipt path escapes its expected root: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise ValueError(f"byte-size drift: {path}")
    expected_sha256 = str(binding.get("sha256") or "")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 drift: {path}")
    return path


def update_stream_hash(
    digest: Any,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str] = PRESERVED_FIELDS,
) -> int:
    count = 0
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value is None:
                digest.update(b"\xff")
                continue
            encoded = str(value).encode("utf-8")
            digest.update(b"\x00")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        count += 1
    return count


def lineage_fingerprint(
    *,
    run_root: Path,
    transform_binding: Mapping[str, Any],
    release_path: Path,
    task_index: int,
    duplicate_keys: set[tuple[str, str]],
    run_contract_sha256: str,
    glossapi_binding: Mapping[str, Any],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    transform_path = resolve_binding(
        transform_binding, run_root, containment_root=run_root
    )
    receipt = read_object(
        run_root / "20-glossapi" / "receipts" / f"task-{task_index:06d}.json"
    )
    if (
        receipt.get("schema_version") != "agent1_v5_glossapi_task_receipt_v1"
        or receipt.get("status") != "passed"
        or int(receipt.get("task_index", -1)) != task_index
        or receipt.get("run_contract_sha256") != run_contract_sha256
        or receipt.get("input", {}).get("sha256") != transform_binding.get("sha256")
        or receipt.get("output", {}).get("sha256") != glossapi_binding.get("sha256")
    ):
        raise ValueError(f"GlossAPI task receipt drift for task {task_index}")
    issue_path = resolve_binding(receipt["issues"], run_root, containment_root=run_root)
    quarantined: set[str] = set()
    with gzip.open(issue_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                uid = row.get("source_row_uid")
                if uid:
                    quarantined.add(str(uid))

    transform_metadata = hashlib.sha256()
    transform_rows = 0
    rewritten_rows = 0
    transform = pq.ParquetFile(transform_path)
    for batch in transform.iter_batches(
        columns=[
            "source_row_uid",
            "source_doc_id_candidate",
            "source_dataset",
            "title",
            "author",
            "source_metadata_json",
        ],
        batch_size=2048,
    ):
        rows = []
        for row in batch.to_pylist():
            uid = str(row.get("source_row_uid"))
            if uid in quarantined:
                continue
            rows.append(
                {
                    "source_row_uid": uid,
                    "source_dataset": str(row["source_dataset"]),
                    "source_doc_id_candidate": str(row["source_doc_id_candidate"]),
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "source_metadata_json": row.get("source_metadata_json"),
                }
            )
        transform_rows += update_stream_hash(
            transform_metadata, rows, TRANSFORM_METADATA_FIELDS
        )

    glossapi_path = resolve_binding(
        receipt["output"], run_root, containment_root=run_root
    )
    glossapi_metadata = hashlib.sha256()
    expected_release = hashlib.sha256()
    glossapi_rows = 0
    for batch in pq.ParquetFile(glossapi_path).iter_batches(
        columns=[*TRANSFORM_METADATA_FIELDS, "text"], batch_size=2048
    ):
        metadata_rows = []
        release_rows = []
        for row in batch.to_pylist():
            uid = str(row["source_row_uid"])
            source = str(row["source_dataset"])
            candidate = str(row["source_doc_id_candidate"])
            document_id = candidate
            if (source, candidate) in duplicate_keys:
                document_id = f"{candidate}#{uid}"
                rewritten_rows += 1
            metadata_rows.append(
                {field: row.get(field) for field in TRANSFORM_METADATA_FIELDS}
            )
            release_rows.append(
                {
                    "source_dataset": source,
                    "source_doc_id": document_id,
                    "text_sha256": hashlib.sha256(
                        str(row.get("text") or "").encode("utf-8")
                    ).hexdigest(),
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "source_metadata_json": row.get("source_metadata_json"),
                }
            )
        glossapi_rows += update_stream_hash(
            glossapi_metadata, metadata_rows, TRANSFORM_METADATA_FIELDS
        )
        update_stream_hash(expected_release, release_rows, RELEASE_LINEAGE_FIELDS)

    actual_release = hashlib.sha256()
    release_rows_count = 0
    release_parquet = pq.ParquetFile(release_path)
    for batch in release_parquet.iter_batches(
        columns=[
            "source_dataset",
            "source_doc_id",
            "text",
            "title",
            "author",
            "source_metadata_json",
        ],
        batch_size=2048,
    ):
        rows = []
        for row in batch.to_pylist():
            rows.append(
                {
                    "source_dataset": row.get("source_dataset"),
                    "source_doc_id": row.get("source_doc_id"),
                    "text_sha256": hashlib.sha256(
                        str(row.get("text") or "").encode("utf-8")
                    ).hexdigest(),
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "source_metadata_json": row.get("source_metadata_json"),
                }
            )
        release_rows_count += update_stream_hash(
            actual_release, rows, RELEASE_LINEAGE_FIELDS
        )
    metadata_passed = (
        transform_rows == glossapi_rows
        and transform_metadata.digest() == glossapi_metadata.digest()
    )
    release_passed = (
        glossapi_rows == release_rows_count
        and expected_release.digest() == actual_release.digest()
    )
    return {
        "task_index": task_index,
        "quarantined_rows": len(quarantined),
        "collision_ids_rewritten": rewritten_rows,
        "transform_rows": transform_rows,
        "glossapi_rows": glossapi_rows,
        "release_rows": release_rows_count,
        "transform_metadata_sha256": transform_metadata.hexdigest(),
        "glossapi_metadata_sha256": glossapi_metadata.hexdigest(),
        "expected_release_lineage_sha256": expected_release.hexdigest(),
        "actual_release_lineage_sha256": actual_release.hexdigest(),
        "metadata_passed": metadata_passed,
        "release_passed": release_passed,
        "passed": metadata_passed and release_passed,
    }


def _push_sample(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]],
    source: str,
    key: int,
    tie: str,
    sample: dict[str, Any],
    limit: int,
) -> None:
    heap = heaps[source]
    entry = (-key, tie, sample)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit_candidate_file(payload: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = Path(str(payload["path"]))
    expected_schema = str(payload["schema"])
    expected_rows = int(payload["rows"])
    sample_size = int(payload["sample_size"])
    parquet = pq.ParquetFile(path)
    schema_ok = str(parquet.schema_arrow) == expected_schema
    codecs = sorted(
        {
            str(parquet.metadata.row_group(group).column(column).compression)
            for group in range(parquet.num_row_groups)
            for column in range(parquet.metadata.row_group(group).num_columns)
        }
    )
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    metric_samples: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    seen_rows = 0
    row_index = 0

    for batch in parquet.iter_batches(columns=list(SCAN_COLUMNS), batch_size=1024):
        for row in batch.to_pylist():
            seen_rows += 1
            row_index += 1
            source = str(row.get("source_dataset") or "")
            document_id = str(row.get("source_doc_id") or "")
            text = str(row.get("text") or "")
            current = counters[source]
            current["rows"] += 1
            current["blank_source_dataset"] += int(not source.strip())
            current["blank_source_doc_id"] += int(not document_id.strip())
            current["blank_text"] += int(not text.strip())
            current["blank_title"] += int(not str(row.get("title") or "").strip())
            current["blank_author"] += int(not str(row.get("author") or "").strip())

            metadata = row.get("source_metadata_json")
            if metadata is None or not str(metadata).strip():
                current["blank_source_metadata_json"] += 1
            else:
                try:
                    parsed = json.loads(str(metadata))
                except json.JSONDecodeError:
                    current["invalid_source_metadata_json"] += 1
                else:
                    current["nonobject_source_metadata_json"] += int(
                        not isinstance(parsed, dict)
                    )
                    current["empty_source_metadata_object"] += int(parsed == {})

            current[f"filter::{str(row.get('filter'))}"] += 1
            current["is_empty_true"] += int(row.get("is_empty") is True)
            current["is_empty_not_false"] += int(row.get("is_empty") is not False)
            current["unexpected_quality_method"] += int(
                row.get("quality_method") != EXPECTED_QUALITY_METHOD
            )
            current["missing_reevaluated_at"] += int(row.get("reevaluated_at") is None)

            chars = int(row.get("chars") or 0)
            non_whitespace = int(row.get("non_whitespace_chars") or 0)
            utf8_bytes = int(row.get("utf8_bytes") or 0)
            words = int(row.get("approx_word_count") or 0)
            current["text_chars"] += len(text)
            current["text_utf8_bytes"] += len(text.encode("utf-8"))
            current["short_lt_100_chars"] += int(len(text) < 100)
            current["short_lt_200_chars"] += int(len(text) < 200)
            current["very_large_gt_1m_chars"] += int(len(text) > 1_000_000)
            current["chars_mismatch"] += int(chars != len(text))
            current["non_whitespace_mismatch"] += int(
                non_whitespace != sum(not character.isspace() for character in text)
            )
            current["utf8_bytes_mismatch"] += int(
                utf8_bytes != len(text.encode("utf-8"))
            )
            current["nonpositive_word_count"] += int(words <= 0)

            greek_badness = _finite(row.get("greek_badness_score"))
            mojibake_badness = _finite(row.get("mojibake_badness_score"))
            greek_percentage = _finite(row.get("greek_percentage"))
            latin_percentage = _finite(row.get("latin_percentage"))
            current["missing_or_nonfinite_greek_badness_score"] += int(
                greek_badness is None
            )
            current["missing_or_nonfinite_mojibake_badness_score"] += int(
                mojibake_badness is None
            )
            current["missing_or_nonfinite_greek_percentage"] += int(
                greek_percentage is None
            )
            current["missing_or_nonfinite_latin_percentage"] += int(
                latin_percentage is None
            )
            current["greek_percentage_out_of_range"] += int(
                greek_percentage is not None and not 0.0 <= greek_percentage <= 100.0
            )
            current["latin_percentage_out_of_range"] += int(
                latin_percentage is not None and not 0.0 <= latin_percentage <= 100.0
            )
            current["negative_greek_badness_score"] += int(
                greek_badness is not None and greek_badness < 0.0
            )
            current["negative_mojibake_badness_score"] += int(
                mojibake_badness is not None and mojibake_badness < 0.0
            )
            expected_filter = None
            if greek_badness is not None:
                expected_filter = "greek>60" if greek_badness > 60.0 else "ok"
            current["filter_score_inconsistent"] += int(
                expected_filter is None or row.get("filter") != expected_filter
            )
            current["greek_badness_gt_60"] += int(
                greek_badness is not None and greek_badness > 60.0
            )
            current["mojibake_badness_gt_0"] += int(
                mojibake_badness is not None and mojibake_badness > 0.0
            )
            current["greek_percentage_lt_20"] += int(
                greek_percentage is not None and greek_percentage < 20.0
            )

            current["residual_generated_image_token"] += int(
                bool(GENERATED_IMAGE_RE.search(text))
            )
            current["residual_recognized_html"] += int(bool(HTML_RE.search(text)))
            current["mojibake_marker"] += int(bool(MOJIBAKE_RE.search(text)))
            current["replacement_character"] += int("\ufffd" in text)
            current["control_character"] += int(bool(CONTROL_RE.search(text)))
            current["odd_markdown_fence_count"] += int(text.count("```") % 2 == 1)
            current["repetition_removal_marker"] += int(REPETITION_MARKER in text)
            current["image_description_marker"] += int(IMAGE_DESCRIPTION_MARKER in text)

            key = int.from_bytes(
                hashlib.sha256(f"{source}\0{document_id}".encode("utf-8")).digest()[:8],
                "big",
            )
            metrics = {
                "key": f"{key:016x}",
                "chars": len(text),
                "words": words,
                "greek_percentage": greek_percentage,
                "greek_badness_score": greek_badness,
                "mojibake_badness_score": mojibake_badness,
            }
            tie = f"{path.name}:{row_index:012d}:{document_id}"
            _push_sample(
                metric_samples, source, key, tie, metrics, max(64, sample_size * 8)
            )

    return {
        "path": str(path),
        "expected_rows": expected_rows,
        "rows": seen_rows,
        "schema_ok": schema_ok,
        "codecs": codecs,
        "source_counters": {key: dict(value) for key, value in counters.items()},
        "metric_samples": {
            source: [sample for _, _, sample in values]
            for source, values in metric_samples.items()
        },
    }


def percentiles(values: Sequence[float]) -> dict[str, float | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"p10": None, "p50": None, "p90": None, "p99": None}

    def pick(fraction: float) -> float:
        return finite[round((len(finite) - 1) * fraction)]

    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99)}


def merge_samples(
    results: Sequence[Mapping[str, Any]], name: str, limit: int
) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for source, rows in result[name].items():
            values[source].extend(rows)
    return {
        source: sorted(rows, key=lambda row: row.get("sample_key", row.get("key", "")))[
            :limit
        ]
        for source, rows in sorted(values.items())
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--samples-per-source", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable audit output exists: {args.output}")
    if args.workers <= 0 or args.samples_per_source <= 0:
        raise ValueError("workers and samples-per-source must be positive")

    run_root = args.run_root.resolve(strict=True)
    contract_path = run_root / "run_contract.json"
    contract = read_object(contract_path)
    combined_path = (
        run_root / "release-pre-dedup" / "manifests" / "combined_manifest.json"
    )
    combined = read_object(combined_path)
    transform = read_object(run_root / "transform_manifest.json")
    if contract.get("status") != "passed" or combined.get("status") != "passed":
        raise ValueError("run contract and combined manifest must be passed")

    candidate_files = [
        row for row in combined["files"] if row.get("origin") == "candidate"
    ]
    if len(candidate_files) != int(transform["task_count"]):
        raise ValueError("candidate file/task count mismatch")
    expected_schema = str(combined["schema"])
    jobs = []
    for row in candidate_files:
        path = resolve_binding(
            row,
            run_root / "release-pre-dedup",
            containment_root=run_root / "release-pre-dedup",
        )
        jobs.append(
            {
                "path": str(path),
                "rows": int(row["rows"]),
                "schema": expected_schema,
                "sample_size": args.samples_per_source,
            }
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(audit_candidate_file, jobs))

    aggregate: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        for source, counters in result["source_counters"].items():
            aggregate[source].update(
                {key: int(value) for key, value in counters.items()}
            )
    metric_samples = merge_samples(results, "metric_samples", 512)
    distributions = {}
    for source, rows in metric_samples.items():
        distributions[source] = {
            field: percentiles(
                [float(row[field]) for row in rows if row.get(field) is not None]
            )
            for field in (
                "chars",
                "words",
                "greek_percentage",
                "greek_badness_score",
                "mojibake_badness_score",
            )
        }

    contract_sha256 = sha256_file(contract_path)
    candidate_manifest_path = run_root / "candidate_manifest.json"
    candidate_manifest = read_object(candidate_manifest_path)
    envelope_plan_path = run_root / "envelope_plan.json"
    envelope_plan = read_object(envelope_plan_path)
    glossapi_manifest_path = run_root / "glossapi_manifest.json"
    glossapi_manifest = read_object(glossapi_manifest_path)
    if (
        contract.get("schema_version") != "agent1_v5_run_contract_v1"
        or contract.get("status") != "passed"
        or combined.get("schema_version") != "agent1_v5_combined_manifest_v1"
        or combined.get("status") != "passed"
        or combined.get("run_id") != contract.get("run_id")
        or combined.get("run_contract_sha256") != contract_sha256
        or transform.get("schema_version") != "agent1_v5_transform_manifest_v1"
        or transform.get("status") != "passed"
        or transform.get("run_contract_sha256") != contract_sha256
        or candidate_manifest.get("schema_version")
        != "agent1_v5_candidate_envelope_manifest_v1"
        or candidate_manifest.get("status") != "passed"
        or candidate_manifest.get("run_contract_sha256") != contract_sha256
        or combined.get("candidate_manifest_sha256")
        != sha256_file(candidate_manifest_path)
        or envelope_plan.get("schema_version") != "agent1_v5_envelope_plan_v1"
        or envelope_plan.get("status") != "passed"
        or envelope_plan.get("run_contract_sha256") != contract_sha256
        or candidate_manifest.get("envelope_plan_sha256")
        != sha256_file(envelope_plan_path)
        or envelope_plan.get("glossapi_manifest_sha256")
        != sha256_file(glossapi_manifest_path)
        or glossapi_manifest.get("schema_version") != "agent1_v5_glossapi_manifest_v1"
        or glossapi_manifest.get("status") != "passed"
        or glossapi_manifest.get("run_contract_sha256") != contract_sha256
        or glossapi_manifest.get("transform_manifest_sha256")
        != sha256_file(run_root / "transform_manifest.json")
    ):
        raise ValueError("run/manifest schema or cross-hash binding drift")
    candidate_sequence = [
        (int(row["rows"]), int(row["bytes"]), str(row["sha256"]))
        for row in candidate_manifest["shards"]
    ]
    combined_candidate_sequence = [
        (int(row["rows"]), int(row["bytes"]), str(row["sha256"]))
        for row in candidate_files
    ]
    if candidate_sequence != combined_candidate_sequence:
        raise ValueError("candidate manifest/combined release shard sequence drift")
    if len(glossapi_manifest.get("shards", [])) != len(candidate_files):
        raise ValueError("GlossAPI/candidate task count drift")
    id_database = resolve_binding(
        envelope_plan["id_database"], run_root, containment_root=run_root
    )
    import sqlite3

    connection = sqlite3.connect(f"file:{id_database}?mode=ro", uri=True)
    duplicate_keys = {
        (str(source), str(candidate))
        for source, candidate in connection.execute(
            "SELECT source_dataset, candidate FROM ids "
            "GROUP BY source_dataset, candidate HAVING COUNT(*) > 1"
        )
    }
    connection.close()

    lineage = []
    for task_index, (binding, glossapi_binding, release) in enumerate(
        zip(transform["shards"], glossapi_manifest["shards"], jobs, strict=True)
    ):
        lineage.append(
            lineage_fingerprint(
                run_root=run_root,
                transform_binding=binding,
                release_path=Path(str(release["path"])),
                task_index=task_index,
                duplicate_keys=duplicate_keys,
                run_contract_sha256=contract_sha256,
                glossapi_binding=glossapi_binding,
            )
        )

    source_ids = set(str(value) for value in contract["source_ids"])
    config_path = resolve_binding(contract["config"], Path("/"))
    config = read_object(config_path)
    expected_repositories = {str(row["repo_id"]) for row in config["sources"].values()}
    actual_repositories = set(aggregate)

    structural_counter_names = (
        "blank_source_dataset",
        "blank_source_doc_id",
        "blank_text",
        "invalid_source_metadata_json",
        "nonobject_source_metadata_json",
        "is_empty_true",
        "is_empty_not_false",
        "unexpected_quality_method",
        "missing_reevaluated_at",
        "chars_mismatch",
        "non_whitespace_mismatch",
        "utf8_bytes_mismatch",
        "nonpositive_word_count",
        "missing_or_nonfinite_greek_badness_score",
        "missing_or_nonfinite_mojibake_badness_score",
        "missing_or_nonfinite_greek_percentage",
        "missing_or_nonfinite_latin_percentage",
        "greek_percentage_out_of_range",
        "latin_percentage_out_of_range",
        "negative_greek_badness_score",
        "negative_mojibake_badness_score",
        "filter_score_inconsistent",
        "residual_generated_image_token",
        "residual_recognized_html",
        "replacement_character",
        "control_character",
    )
    structural_failures = {
        name: sum(counters[name] for counters in aggregate.values())
        for name in structural_counter_names
    }
    filtered_rows = sum(
        value
        for counters in aggregate.values()
        for name, value in counters.items()
        if name.startswith("filter::") and name != "filter::ok"
    )
    blocking_issues = []
    if any(result["rows"] != result["expected_rows"] for result in results):
        blocking_issues.append("candidate_row_count_mismatch")
    if any(not result["schema_ok"] for result in results):
        blocking_issues.append("candidate_schema_mismatch")
    if any(result["codecs"] != ["ZSTD"] for result in results):
        blocking_issues.append("candidate_parquet_not_uniformly_zstd")
    if expected_repositories != actual_repositories or len(source_ids) != len(
        expected_repositories
    ):
        blocking_issues.append("candidate_source_roster_mismatch")
    if any(structural_failures.values()):
        blocking_issues.append("candidate_structural_quality_failure")
    if any(not row["passed"] for row in lineage):
        blocking_issues.append("candidate_metadata_lineage_mismatch")
    if filtered_rows:
        blocking_issues.append("candidate_quality_filter_not_ok")

    total_rows = sum(counters["rows"] for counters in aggregate.values())
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed" if not blocking_issues else "blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": contract["run_id"],
        "audit_scope": "candidate structural/quality/lineage audit of the pre-dedup release; not a final corpus readiness decision",
        "pretraining_ready": False,
        "implementation": {
            "schema_version": AUDIT_SCHEMA,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "workers": args.workers,
            "metric_sample_size_per_source": 512,
        },
        "inputs": {
            "run_contract_sha256": contract_sha256,
            "combined_manifest_sha256": sha256_file(combined_path),
            "transform_manifest_sha256": sha256_file(
                run_root / "transform_manifest.json"
            ),
            "glossapi_manifest_sha256": sha256_file(glossapi_manifest_path),
            "envelope_plan_sha256": sha256_file(envelope_plan_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        },
        "candidate_files": len(candidate_files),
        "candidate_rows": total_rows,
        "expected_candidate_rows": int(combined["candidate_rows"]),
        "expected_source_repositories": sorted(expected_repositories),
        "actual_source_repositories": sorted(actual_repositories),
        "all_candidate_codecs": sorted(
            {codec for result in results for codec in result["codecs"]}
        ),
        "source_counters": {
            source: dict(counters) for source, counters in sorted(aggregate.items())
        },
        "structural_failures": structural_failures,
        "quality_filter_not_ok_rows": filtered_rows,
        "sampled_distributions": distributions,
        "metadata_lineage": {
            "transform_to_glossapi_metadata_fields": list(TRANSFORM_METADATA_FIELDS),
            "glossapi_to_release_fields": list(RELEASE_LINEAGE_FIELDS),
            "tasks": len(lineage),
            "passed_tasks": sum(int(row["passed"]) for row in lineage),
            "duplicate_source_doc_id_keys": len(duplicate_keys),
            "collision_ids_rewritten": sum(
                int(row["collision_ids_rewritten"]) for row in lineage
            ),
            "details": lineage,
        },
        "blocking_issues": blocking_issues,
    }
    if total_rows != int(combined["candidate_rows"]):
        receipt["blocking_issues"].append("candidate_manifest_row_closure_failure")
        receipt["status"] = "blocked"

    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    temporary.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        canonical_json(
            {
                "ok": receipt["status"] == "passed",
                "status": receipt["status"],
                "rows": total_rows,
                "blocking_issues": receipt["blocking_issues"],
            }
        )
    )
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
