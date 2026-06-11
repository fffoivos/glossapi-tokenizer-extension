#!/usr/bin/env python3
"""Scan HPLT text for E001 replacement/control/private-use characters.

This is a non-destructive discovery pass. It streams the selected HPLT parquet,
counts E001-like residue, and writes a compact review pack for matching rows.
It does not edit source rows or emit cleaned text.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import heapq
import json
import math
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - cluster preflight catches this.
    pa = pc = ds = None
    PYARROW_IMPORT_ERROR = exc
else:
    PYARROW_IMPORT_ERROR = None


SOURCE_DATASET_DEFAULT = "HPLT/ell_Grek_ge8_no_mt_clean60"
E001_RE = re.compile(r"[\uFFFD\uE000-\uF8FF\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def release_unused_memory() -> None:
    gc.collect()
    if pa is not None:
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def first_present(row: dict[str, Any], meta: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
        value = meta.get(name)
        if value not in (None, ""):
            return value
    return None


def source_filter(schema_names: set[str], source_dataset: str | None) -> Any:
    if source_dataset and "source_dataset" in schema_names:
        return pc.field("source_dataset") == source_dataset
    return None


def text_columns(schema_names: set[str]) -> list[str]:
    wanted = [
        "source_doc_id",
        "source_dataset",
        "doc_key",
        "text",
        "title",
        "url",
        "host",
        "source_metadata_json",
        "quality_bin",
        "source_mix_chars",
    ]
    return [name for name in wanted if name in schema_names]


def row_identity(row: dict[str, Any], row_number: int) -> str:
    value = compact_text(row.get("source_doc_id") or row.get("doc_key"))
    return value or f"missing-id::{row_number}"


def is_e001_char(ch: str) -> bool:
    code = ord(ch)
    if ch == "\ufffd":
        return True
    if code < 32 and ch not in "\n\r\t":
        return True
    if 127 <= code <= 159:
        return True
    return unicodedata.category(ch) == "Co"


def e001_kind(ch: str) -> str:
    code = ord(ch)
    if ch == "\ufffd":
        return "replacement_char"
    if code < 32 and ch not in "\n\r\t":
        return "c0_control"
    if 127 <= code <= 159:
        return "c1_control"
    if unicodedata.category(ch) == "Co":
        return "private_use"
    return "other"


def visible_char(ch: str) -> str:
    code = ord(ch)
    name = unicodedata.name(ch, "UNNAMED")
    if is_e001_char(ch):
        return f"[U+{code:04X} {name}]"
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    return ch


def visible_context(text: str, center: int, radius: int) -> str:
    start = max(0, center - radius)
    end = min(len(text), center + radius + 1)
    return "".join(visible_char(ch) for ch in text[start:end])


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def meta_fields(row: dict[str, Any]) -> dict[str, Any]:
    meta = parse_json(row.get("source_metadata_json"))
    url = first_present(row, meta, ["url", "source_url", "original_url"])
    host = first_present(row, meta, ["host", "domain", "hostname"])
    if not host and isinstance(url, str) and url:
        host = urlparse(url).netloc
    quality_bin = first_present(row, meta, ["quality_bin", "wds_quality_bin"])
    return {
        "url": url,
        "host": host,
        "quality_bin": quality_bin,
    }


def inspect_text(text: str, context_radius: int, max_contexts: int) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    contexts: list[dict[str, Any]] = []
    for match in E001_RE.finditer(text):
        index = match.start()
        ch = match.group(0)
        kind = e001_kind(ch)
        counts[kind] += 1
        if len(contexts) < max_contexts:
            contexts.append(
                {
                    "char_index": index,
                    "kind": kind,
                    "codepoint": f"U+{ord(ch):04X}",
                    "unicode_name": unicodedata.name(ch, "UNNAMED"),
                    "context": visible_context(text, index, context_radius),
                }
            )
    total = sum(counts.values())
    return {
        "e001_char_count": total,
        "e001_density": float(total) / float(len(text) or 1),
        "by_kind": dict(counts),
        "contexts": contexts,
    }


class Reservoir:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.random = random.Random(seed)
        self.seen = 0
        self.items: list[dict[str, Any]] = []

    def add(self, item: dict[str, Any]) -> None:
        self.seen += 1
        if self.size <= 0:
            return
        if len(self.items) < self.size:
            self.items.append(item)
            return
        index = self.random.randrange(self.seen)
        if index < self.size:
            self.items[index] = item


class TopK:
    def __init__(self, size: int) -> None:
        self.size = size
        self.heap: list[tuple[float, str, dict[str, Any]]] = []

    def add(self, score: float, key: str, item: dict[str, Any]) -> None:
        if self.size <= 0:
            return
        payload = (score, key, item)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, payload)
            return
        if payload > self.heap[0]:
            heapq.heapreplace(self.heap, payload)

    def items_desc(self) -> list[dict[str, Any]]:
        return [item for _score, _key, item in sorted(self.heap, reverse=True)]


def make_candidate(row: dict[str, Any], text: str, row_number: int, inspection: dict[str, Any]) -> dict[str, Any]:
    fields = meta_fields(row)
    source_doc_id = row_identity(row, row_number)
    count = int(inspection["e001_char_count"])
    density = float(inspection["e001_density"])
    if density >= 0.01 or count >= 100:
        action = "quarantine"
    else:
        action = "normalize_or_trim_span"
    return {
        "schema_version": "hplt-e001-control-char-scan-v1",
        "source_doc_id": source_doc_id,
        "parent_source_doc_id": source_doc_id,
        "derived_doc_id": None,
        "is_shadow_record": False,
        "source_dataset": compact_text(row.get("source_dataset")),
        "doc_key": compact_text(row.get("doc_key") or source_doc_id),
        "url": fields["url"],
        "host": fields["host"],
        "quality_bin": fields["quality_bin"],
        "text_sha256_before": sha256_text(text),
        "chars_before": len(text),
        "chars_after": None,
        "chars_removed": None,
        "tokens_before": None,
        "tokens_after": None,
        "tokens_removed": None,
        "candidate_error_type_ids": ["E001"],
        "error_type_ids": ["E001"],
        "candidate_action": action,
        "action": action,
        "action_status": "candidate",
        "confidence": min(1.0, density * 100.0 + min(1.0, count / 100.0)),
        "detector_scores": {
            "e001_char_count": count,
            "e001_density": density,
            "replacement_char_count": inspection["by_kind"].get("replacement_char", 0),
            "c0_control_count": inspection["by_kind"].get("c0_control", 0),
            "c1_control_count": inspection["by_kind"].get("c1_control", 0),
            "private_use_count": inspection["by_kind"].get("private_use", 0),
        },
        "reason_codes": sorted(inspection["by_kind"]),
        "span_ranges": [],
        "split_parts": [],
        "good_text_loss_estimate": None,
        "sample_set": [],
        "evidence_excerpt": inspection["contexts"][0]["context"] if inspection["contexts"] else "",
        "e001_contexts": inspection["contexts"],
        "review_label": None,
        "reviewer": None,
        "review_notes": None,
        "policy_id": "E001_discovery_scan",
        "policy_version": "20260606",
    }


def scan(args: argparse.Namespace) -> dict[str, Any]:
    if PYARROW_IMPORT_ERROR is not None:
        raise RuntimeError(f"pyarrow import failed: {PYARROW_IMPORT_ERROR}")

    dataset = ds.dataset(args.input, format="parquet")
    schema_names = set(dataset.schema.names)
    columns = text_columns(schema_names)
    if "text" not in columns:
        raise RuntimeError("Input does not expose a text column")

    filter_expr = source_filter(schema_names, args.source_dataset)
    total_rows = 0
    total_chars = 0
    match_rows = 0
    char_counts: collections.Counter[str] = collections.Counter()
    host_counts: collections.Counter[str] = collections.Counter()
    qbin_counts: collections.Counter[str] = collections.Counter()

    top_count = TopK(args.top_k)
    top_density = TopK(args.top_k)
    reservoir = Reservoir(args.random_sample_size, args.seed)

    start = time.time()
    next_progress = args.progress_every_rows
    for batch_index, batch in enumerate(dataset.to_batches(columns=columns, filter=filter_expr, batch_size=args.batch_size), 1):
        rows = batch.to_pylist()
        for row in rows:
            total_rows += 1
            if args.max_rows and total_rows > args.max_rows:
                break
            text = compact_text(row.get("text"))
            total_chars += len(text)
            inspection = inspect_text(text, args.context_radius, args.max_contexts_per_doc)
            if not inspection["e001_char_count"]:
                continue
            match_rows += 1
            candidate = make_candidate(row, text, total_rows, inspection)
            for kind, count in inspection["by_kind"].items():
                char_counts[kind] += int(count)
            if candidate.get("host"):
                host_counts[str(candidate["host"])] += 1
            if candidate.get("quality_bin") not in (None, ""):
                qbin_counts[str(candidate["quality_bin"])] += 1
            key = str(candidate["source_doc_id"])
            top_count.add(float(candidate["detector_scores"]["e001_char_count"]), key, dict(candidate, sample_set=["top_e001_char_count"]))
            top_density.add(float(candidate["detector_scores"]["e001_density"]), key, dict(candidate, sample_set=["top_e001_density"]))
            reservoir.add(dict(candidate, sample_set=["e001_random_match"]))
        del rows
        del batch
        if args.release_memory_every_batches and batch_index % args.release_memory_every_batches == 0:
            release_unused_memory()
        if args.max_rows and total_rows >= args.max_rows:
            break
        if args.progress_every_rows and total_rows >= next_progress:
            elapsed = max(1.0, time.time() - start)
            print(
                json.dumps(
                    {
                        "event": "e001_scan_progress",
                        "rows": total_rows,
                        "match_rows": match_rows,
                        "rows_per_sec": round(total_rows / elapsed, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_progress += args.progress_every_rows

    selected: dict[str, dict[str, Any]] = {}
    for item in top_count.items_desc() + top_density.items_desc() + reservoir.items:
        source_doc_id = str(item["source_doc_id"])
        if source_doc_id in selected:
            existing = selected[source_doc_id]
            existing["sample_set"] = sorted(set(existing.get("sample_set") or []) | set(item.get("sample_set") or []))
        else:
            selected[source_doc_id] = item

    rows = [selected[key] for key in sorted(selected)]
    elapsed = time.time() - start
    return {
        "schema_version": "hplt-e001-control-char-scan-v1",
        "created_at_utc": args.timestamp,
        "policy_note": "Discovery only. Source HPLT rows are immutable and no cleaning is applied.",
        "input": str(args.input),
        "source_dataset": args.source_dataset,
        "rows_scanned": total_rows,
        "chars_scanned": total_chars,
        "match_rows": match_rows,
        "match_row_rate": float(match_rows) / float(total_rows or 1),
        "e001_char_counts": dict(char_counts),
        "matched_hosts_top": host_counts.most_common(30),
        "matched_quality_bins": dict(qbin_counts),
        "review_pack_rows": len(rows),
        "review_pack_selection": {
            "top_k": args.top_k,
            "random_sample_size": args.random_sample_size,
            "seed": args.seed,
        },
        "elapsed_sec": round(elapsed, 3),
        "negative_evidence": match_rows == 0,
        "rows": rows,
    }


def write_outputs(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"e001_control_char_scan_{args.timestamp}"
    summary_path = Path(str(base) + "_summary.json")
    pack_jsonl_path = Path(str(base) + "_review_pack.jsonl")
    pack_md_path = Path(str(base) + "_review_pack.md")

    summary = dict(result)
    rows = summary.pop("rows")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with pack_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with pack_md_path.open("w", encoding="utf-8") as out:
        out.write("# E001 Control Character Review Pack\n\n")
        out.write("This pack is non-destructive. It samples rows with replacement/control/private-use characters for manual review.\n\n")
        out.write("- rows scanned: `%s`\n" % summary["rows_scanned"])
        out.write("- matching rows: `%s`\n" % summary["match_rows"])
        out.write("- review rows: `%s`\n" % len(rows))
        out.write("- negative evidence: `%s`\n\n" % summary["negative_evidence"])
        for index, row in enumerate(rows, 1):
            scores = row["detector_scores"]
            out.write("## %03d `%s`\n\n" % (index, row["source_doc_id"]))
            out.write("- action: `%s`\n" % row["candidate_action"])
            out.write("- host: `%s`\n" % (row.get("host") or ""))
            out.write("- url: `%s`\n" % (row.get("url") or ""))
            out.write("- chars_before: `%s`\n" % row["chars_before"])
            out.write("- e001_char_count: `%s`\n" % scores["e001_char_count"])
            out.write("- e001_density: `%.8f`\n" % scores["e001_density"])
            out.write("- reason_codes: `%s`\n\n" % "`, `".join(row.get("reason_codes") or []))
            for context in row.get("e001_contexts") or []:
                out.write("### Context `%s` `%s`\n\n" % (context["char_index"], context["codepoint"]))
                out.write("```text\n%s\n```\n\n" % context["context"])

    return {
        "summary_json": str(summary_path),
        "review_pack_jsonl": str(pack_jsonl_path),
        "review_pack_md": str(pack_md_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet file or dataset directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dataset", default=SOURCE_DATASET_DEFAULT)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no row cap.")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--random-sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--context-radius", type=int, default=180)
    parser.add_argument("--max-contexts-per-doc", type=int, default=5)
    parser.add_argument("--progress-every-rows", type=int, default=100000)
    parser.add_argument("--release-memory-every-batches", type=int, default=50)
    args = parser.parse_args()

    if args.source_dataset == "":
        args.source_dataset = None

    result = scan(args)
    outputs = write_outputs(args, result)
    print(json.dumps(dict(outputs, rows_scanned=result["rows_scanned"], match_rows=result["match_rows"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
