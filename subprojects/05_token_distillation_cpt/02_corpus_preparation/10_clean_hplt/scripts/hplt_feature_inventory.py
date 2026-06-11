#!/usr/bin/env python3
"""Non-destructive HPLT cleaning inventory and review-bundle sampler.

The first pass scans only metadata columns to build statistically meaningful
samples. The second pass reads text only for sampled documents and computes
cleaning-risk features. No corpus text is rewritten by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import html
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - caught in cluster preflight.
    pa = pc = ds = pq = None
    PYARROW_IMPORT_ERROR = exc
else:
    PYARROW_IMPORT_ERROR = None


SOURCE_DATASET_DEFAULT = "HPLT/ell_Grek_ge8_no_mt_clean60"
TEXT_MAX_CHARS_DEFAULT = 200_000
FEATURE_TEXT_MAX_CHARS_DEFAULT = 500_000
ACTION_SCHEMA_VERSION = "hplt-cleaning-action-v1"

MENU_PATTERNS = [
    "cookie",
    "cookies",
    "login",
    "newsletter",
    "privacy",
    "terms",
    "share",
    "follow us",
    "facebook",
    "twitter",
    "instagram",
    "youtube",
    "rss",
    "subscribe",
    "read more",
    "home",
    "menu",
    "search",
    "skip to content",
    "αρχικη",
    "μενου",
    "αναζητηση",
    "σχολια",
    "κοινοποιηση",
    "επικοινωνια",
    "πολιτικη απορρητου",
    "οροι χρησης",
    "διαβαστε περισσοτερα",
    "εγγραφη",
    "συνδεση",
]

SPLIT_PATTERNS = [
    re.compile(r"(?im)^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s*$"),
    re.compile(r"(?im)^\s*(?:posted|published|δημοσιευθηκε|αναρτηθηκε)\b"),
    re.compile(r"(?im)^\s*(?:title|τιτλος)\s*:"),
    re.compile(r"(?im)^-{5,}\s*$"),
]

URL_RE = re.compile(r"https?://|\bwww\.", re.I)
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,16}|#[0-9]{2,7}|#x[0-9a-fA-F]{2,6});")
ESCAPED_UNICODE_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
CSS_JSON_RE = re.compile(r"\b(?:function|var|const|class=|style=|script|DOCTYPE|<div|</|href=|src=|\{|\})\b", re.I)


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def stable_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def release_unused_memory() -> None:
    gc.collect()
    if pa is not None:
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass


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


def first_present(row: dict[str, Any], meta: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
        value = meta.get(name)
        if value not in (None, ""):
            return value
    return None


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return float(value)
        result = float(value)
    except Exception:
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_for_menu(value: str) -> str:
    lowered = value.casefold()
    lowered = lowered.replace("ά", "α").replace("έ", "ε").replace("ή", "η")
    lowered = lowered.replace("ί", "ι").replace("ϊ", "ι").replace("ΐ", "ι")
    lowered = lowered.replace("ό", "ο").replace("ύ", "υ").replace("ϋ", "υ")
    lowered = lowered.replace("ΰ", "υ").replace("ώ", "ω")
    return lowered


def is_greek_char(ch: str) -> bool:
    code = ord(ch)
    return 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF


def is_latin_char(ch: str) -> bool:
    code = ord(ch)
    return (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A)


def is_control_char(ch: str) -> bool:
    code = ord(ch)
    return (code < 32 and ch not in "\n\r\t") or 127 <= code <= 159


def bounded_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


def line_stats(lines: list[str]) -> dict[str, float]:
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return {
            "line_count": 0,
            "short_line_ratio": 0.0,
            "very_short_line_ratio": 0.0,
            "top_bottom_short_line_ratio": 0.0,
        }

    short = sum(1 for line in nonempty if len(line.strip()) <= 40)
    very_short = sum(1 for line in nonempty if len(line.strip()) <= 18)
    edge = nonempty[: min(12, len(nonempty))] + nonempty[max(0, len(nonempty) - 12) :]
    edge_short = sum(1 for line in edge if len(line.strip()) <= 40)
    return {
        "line_count": len(nonempty),
        "short_line_ratio": bounded_ratio(short, len(nonempty)),
        "very_short_line_ratio": bounded_ratio(very_short, len(nonempty)),
        "top_bottom_short_line_ratio": bounded_ratio(edge_short, len(edge)),
    }


def paragraph_repetition(text: str) -> dict[str, float]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if len(line.strip()) >= 40]
    normalized = [" ".join(p.casefold().split()) for p in paragraphs if len(p.strip()) >= 40]
    if not normalized:
        return {"paragraph_count": len(paragraphs), "duplicate_paragraph_ratio": 0.0}
    counts = Counter(normalized)
    duplicate_instances = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "paragraph_count": len(normalized),
        "duplicate_paragraph_ratio": bounded_ratio(duplicate_instances, len(normalized)),
    }


def extract_meta_fields(row: dict[str, Any]) -> dict[str, Any]:
    meta = parse_json(row.get("source_metadata_json"))
    url = first_present(row, meta, ["url", "source_url", "original_url"])
    host = first_present(row, meta, ["host", "domain", "hostname"])
    if not host and isinstance(url, str) and url:
        host = urlparse(url).netloc
    quality_bin = first_present(row, meta, ["quality_bin", "wds_quality_bin"])
    if quality_bin is None:
        scores = meta.get("doc_scores")
        if isinstance(scores, list) and scores:
            quality_bin = scores[0]

    lang = meta.get("lang")
    prob = meta.get("prob")
    primary_lang = None
    primary_lang_prob = None
    if isinstance(lang, list) and lang:
        primary_lang = lang[0]
    elif isinstance(lang, str):
        primary_lang = lang
    if isinstance(prob, list) and prob:
        primary_lang_prob = safe_float(prob[0])
    else:
        primary_lang_prob = safe_float(prob)

    seg_langs = meta.get("seg_langs")
    seg_lang_count = len(seg_langs) if isinstance(seg_langs, list) else 0
    non_greek_seg_count = 0
    if isinstance(seg_langs, list):
        non_greek_seg_count = sum(1 for lang_name in seg_langs if not str(lang_name).startswith("ell"))

    return {
        "url": url,
        "host": host,
        "quality_bin": safe_int(quality_bin),
        "hplt_filter": meta.get("filter"),
        "cluster_size": safe_int(meta.get("cluster_size")),
        "crawl_id": meta.get("crawl_id"),
        "primary_lang": primary_lang,
        "primary_lang_prob": primary_lang_prob,
        "seg_lang_count": seg_lang_count,
        "non_greek_seg_count": non_greek_seg_count,
        "register_level_1": meta.get("register_level_1"),
        "register_level_1_code": meta.get("register_level_1_code"),
        "register_level_2": meta.get("register_level_2"),
        "register_level_2_code": meta.get("register_level_2_code"),
    }


def compute_features(row: dict[str, Any], feature_text_max_chars: int = FEATURE_TEXT_MAX_CHARS_DEFAULT) -> dict[str, Any]:
    text_full = compact_text(row.get("text"))
    text = text_full[:feature_text_max_chars] if feature_text_max_chars > 0 else text_full
    feature_text_truncated = len(text) < len(text_full)
    title = compact_text(row.get("title")).strip()
    meta_fields = extract_meta_fields(row)
    char_count = len(text_full)
    feature_text_chars_analyzed = len(text)
    lowerish = normalize_for_menu(text[:50_000])

    greek_chars = latin_chars = digit_chars = symbol_chars = control_chars = replacement_chars = 0
    whitespace_chars = 0
    for ch in text:
        if is_greek_char(ch):
            greek_chars += 1
        elif is_latin_char(ch):
            latin_chars += 1
        elif ch.isdigit():
            digit_chars += 1
        elif ch.isspace():
            whitespace_chars += 1
        elif is_control_char(ch):
            control_chars += 1
        elif ch == "\ufffd":
            replacement_chars += 1
        elif not ch.isalnum():
            symbol_chars += 1

    html_entity_count = len(HTML_ENTITY_RE.findall(text[:100_000]))
    escaped_unicode_count = len(ESCAPED_UNICODE_RE.findall(text[:100_000]))
    url_count = len(URL_RE.findall(text[:100_000]))
    markup_count = len(CSS_JSON_RE.findall(text[:100_000]))
    menu_hit_count = sum(lowerish.count(pattern) for pattern in MENU_PATTERNS)
    split_marker_count = sum(len(pattern.findall(text[:100_000])) for pattern in SPLIT_PATTERNS)

    lines = text.splitlines()
    line_metrics = line_stats(lines)
    paragraph_metrics = paragraph_repetition(text)

    greek_share = bounded_ratio(greek_chars, max(1, greek_chars + latin_chars + digit_chars + symbol_chars))
    latin_share = bounded_ratio(latin_chars, max(1, greek_chars + latin_chars + digit_chars + symbol_chars))
    symbol_share = bounded_ratio(symbol_chars + control_chars + replacement_chars, max(1, char_count))
    control_share = bounded_ratio(control_chars + replacement_chars, max(1, char_count))

    menu_density = menu_hit_count / max(1.0, char_count / 10_000.0)
    url_density = url_count / max(1.0, char_count / 10_000.0)
    markup_density = markup_count / max(1.0, char_count / 10_000.0)
    entity_density = (html_entity_count + escaped_unicode_count) / max(1.0, char_count / 10_000.0)

    encoding_score = min(1.0, control_share * 80 + entity_density / 12)
    markup_score = min(1.0, markup_density / 18 + url_density / 35)
    symbol_score = min(1.0, symbol_share * 8)
    boilerplate_score = min(
        1.0,
        line_metrics["top_bottom_short_line_ratio"] * 0.45
        + line_metrics["very_short_line_ratio"] * 0.20
        + min(1.0, menu_density / 30) * 0.35,
    )
    internal_repetition_score = min(1.0, paragraph_metrics["duplicate_paragraph_ratio"] * 1.4)
    split_candidate_score = min(
        1.0,
        min(1.0, split_marker_count / 6) * 0.50
        + min(1.0, paragraph_metrics["paragraph_count"] / 80) * 0.25
        + min(1.0, char_count / 80_000) * 0.25,
    )
    lang_drift_score = min(
        1.0,
        max(0.0, 0.82 - greek_share) * 1.8
        + min(1.0, meta_fields["non_greek_seg_count"] / max(1, meta_fields["seg_lang_count"])) * 0.35
        + max(0.0, latin_share - 0.12) * 1.0,
    )
    badness = safe_float(row.get("greek_badness_score"), 0.0) or 0.0
    mojibake = safe_float(row.get("mojibake_badness_score"), 0.0) or 0.0
    badness_score = min(1.0, (badness / 60.0) * 0.75 + (mojibake / 30.0) * 0.25)

    scores = {
        "encoding_score": encoding_score,
        "markup_score": markup_score,
        "symbol_score": symbol_score,
        "boilerplate_score": boilerplate_score,
        "internal_repetition_score": internal_repetition_score,
        "split_candidate_score": split_candidate_score,
        "lang_drift_score": lang_drift_score,
        "badness_score": badness_score,
    }
    overall_score = max(scores.values()) if scores else 0.0
    mean_score = sum(scores.values()) / max(1, len(scores))
    combined_score = min(1.0, overall_score * 0.70 + mean_score * 0.30)

    reason_codes = [name for name, score in scores.items() if score >= 0.35]
    if not reason_codes and combined_score >= 0.25:
        reason_codes = ["borderline_combined_score"]

    source_doc_id = compact_text(row.get("source_doc_id") or row.get("doc_key") or stable_hex(text_full[:200]))
    source_dataset = compact_text(row.get("source_dataset"))
    doc_key = compact_text(row.get("doc_key") or source_doc_id)

    feature_row: dict[str, Any] = {
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "doc_key": doc_key,
        "text_sha256": sha256_text(text_full),
        "title": title or None,
        "url": meta_fields["url"],
        "host": meta_fields["host"],
        "quality_bin": meta_fields["quality_bin"],
        "hplt_filter": meta_fields["hplt_filter"],
        "cluster_size": meta_fields["cluster_size"],
        "crawl_id": meta_fields["crawl_id"],
        "primary_lang": meta_fields["primary_lang"],
        "primary_lang_prob": meta_fields["primary_lang_prob"],
        "seg_lang_count": meta_fields["seg_lang_count"],
        "non_greek_seg_count": meta_fields["non_greek_seg_count"],
        "register_level_1": meta_fields["register_level_1"],
        "register_level_1_code": meta_fields["register_level_1_code"],
        "register_level_2": meta_fields["register_level_2"],
        "register_level_2_code": meta_fields["register_level_2_code"],
        "char_count": char_count,
        "feature_text_chars_analyzed": feature_text_chars_analyzed,
        "line_count": line_metrics["line_count"],
        "paragraph_count": paragraph_metrics["paragraph_count"],
        "greek_chars": greek_chars,
        "latin_chars": latin_chars,
        "digit_chars": digit_chars,
        "symbol_chars": symbol_chars,
        "control_chars": control_chars,
        "replacement_chars": replacement_chars,
        "whitespace_chars": whitespace_chars,
        "greek_share": greek_share,
        "latin_share": latin_share,
        "symbol_share": symbol_share,
        "control_share": control_share,
        "short_line_ratio": line_metrics["short_line_ratio"],
        "very_short_line_ratio": line_metrics["very_short_line_ratio"],
        "top_bottom_short_line_ratio": line_metrics["top_bottom_short_line_ratio"],
        "duplicate_paragraph_ratio": paragraph_metrics["duplicate_paragraph_ratio"],
        "html_entity_count": html_entity_count,
        "escaped_unicode_count": escaped_unicode_count,
        "url_count": url_count,
        "markup_count": markup_count,
        "menu_hit_count": menu_hit_count,
        "split_marker_count": split_marker_count,
        "entity_density_per_10k": entity_density,
        "url_density_per_10k": url_density,
        "markup_density_per_10k": markup_density,
        "menu_density_per_10k": menu_density,
        "greek_badness_score": badness,
        "mojibake_badness_score": mojibake,
        "source_mix_chars": safe_int(row.get("source_mix_chars")),
        "is_historical_or_polytonic": bool(row.get("is_historical_or_polytonic") or False),
        "contains_math": bool(row.get("contains_math") or False),
        "contains_latex": bool(row.get("contains_latex") or False),
        "table_ratio": safe_float(row.get("table_ratio"), 0.0) or 0.0,
        "overall_artifact_score": combined_score,
        "max_artifact_score": overall_score,
        "reason_codes": reason_codes,
        "feature_text_truncated": feature_text_truncated,
        "text_truncated": False,
    }
    feature_row.update(scores)
    return feature_row


@dataclass
class Reservoir:
    name: str
    target_size: int
    rng: random.Random
    seen: int = 0

    def __post_init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, item: dict[str, Any]) -> None:
        self.seen += 1
        if self.target_size <= 0:
            return
        if len(self.items) < self.target_size:
            self.items.append(item)
            return
        replace_at = self.rng.randrange(self.seen)
        if replace_at < self.target_size:
            self.items[replace_at] = item


class TopK:
    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = limit
        self.heap: list[tuple[float, str, dict[str, Any]]] = []

    def add(self, score: float, item: dict[str, Any]) -> None:
        if self.limit <= 0:
            return
        if score <= 0:
            return
        key = stable_hex(str(item.get("source_doc_id", "")))
        payload = (score, key, item)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, payload)
        elif payload > self.heap[0]:
            heapq.heapreplace(self.heap, payload)

    def items_desc(self) -> list[dict[str, Any]]:
        return [
            {**item, "target_score": score, "target_dimension": self.name}
            for score, _key, item in sorted(self.heap, reverse=True)
        ]


def metadata_columns(schema_names: set[str]) -> list[str]:
    wanted = [
        "source_dataset",
        "source_doc_id",
        "source_metadata_json",
        "doc_key",
        "greek_badness_score",
        "mojibake_badness_score",
        "source_mix_chars",
        "filter",
        "quality_method",
        "greek_percentage",
        "latin_percentage",
        "table_ratio",
    ]
    return [name for name in wanted if name in schema_names]


def text_columns(schema_names: set[str]) -> list[str]:
    wanted = [
        "source_dataset",
        "source_doc_id",
        "text",
        "title",
        "source_metadata_json",
        "doc_key",
        "is_historical_or_polytonic",
        "contains_math",
        "contains_latex",
        "greek_percentage",
        "latin_percentage",
        "polytonic_ratio",
        "table_ratio",
        "greek_badness_score",
        "mojibake_badness_score",
        "filter",
        "quality_method",
        "source_mix_chars",
    ]
    return [name for name in wanted if name in schema_names]


def build_source_filter(schema_names: set[str], source_dataset: str | None) -> Any:
    if source_dataset and "source_dataset" in schema_names:
        return pc.field("source_dataset") == source_dataset
    return None


def row_identity(row: dict[str, Any]) -> str:
    return compact_text(row.get("source_doc_id") or row.get("doc_key") or "")


def make_meta_item(row: dict[str, Any], absolute_index: int) -> dict[str, Any]:
    meta_fields = extract_meta_fields(row)
    source_doc_id = row_identity(row)
    return {
        "source_doc_id": source_doc_id,
        "doc_key": compact_text(row.get("doc_key") or source_doc_id),
        "row_index_seen": absolute_index,
        "source_dataset": compact_text(row.get("source_dataset")),
        "quality_bin": meta_fields["quality_bin"],
        "host": meta_fields["host"],
        "url": meta_fields["url"],
        "hplt_filter": meta_fields["hplt_filter"],
        "cluster_size": meta_fields["cluster_size"],
        "crawl_id": meta_fields["crawl_id"],
        "register_level_1_code": meta_fields["register_level_1_code"],
        "register_level_2_code": meta_fields["register_level_2_code"],
        "greek_badness_score": safe_float(row.get("greek_badness_score"), 0.0) or 0.0,
        "mojibake_badness_score": safe_float(row.get("mojibake_badness_score"), 0.0) or 0.0,
        "source_mix_chars": safe_int(row.get("source_mix_chars")),
    }


def scan_metadata(args: argparse.Namespace, dataset: Any, schema_names: set[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    seed = int(args.seed)
    reservoirs: dict[str, Reservoir] = {
        "global_random": Reservoir("global_random", args.global_random_size, random.Random(seed + 1)),
        "feature_screen": Reservoir("feature_screen", args.screen_sample_size, random.Random(seed + 2)),
    }
    per_bin: dict[int, Reservoir] = {
        bin_value: Reservoir(f"wds_bin_{bin_value}", args.per_bin_size, random.Random(seed + 100 + bin_value))
        for bin_value in (8, 9, 10)
    }

    counters: dict[str, Counter[Any]] = {
        "source_dataset": Counter(),
        "quality_bin": Counter(),
        "host": Counter(),
        "hplt_filter": Counter(),
        "crawl_id": Counter(),
        "register_level_1_code": Counter(),
        "register_level_2_code": Counter(),
    }
    missing_ids = 0
    processed = 0
    selected_source_rows = 0
    char_sum = 0
    badness_values: list[float] = []
    length_values: list[int] = []

    flt = build_source_filter(schema_names, args.source_dataset)
    columns = metadata_columns(schema_names)
    if not columns:
        raise RuntimeError("No readable metadata columns found in input parquet")

    start = time.time()
    for batch in dataset.to_batches(columns=columns, filter=flt, batch_size=args.batch_size):
        rows = batch.to_pylist()
        for row in rows:
            processed += 1
            if args.max_metadata_rows and selected_source_rows >= args.max_metadata_rows:
                break
            source_doc_id = row_identity(row)
            if not source_doc_id:
                missing_ids += 1
                source_doc_id = f"missing-id::{processed}"
                row["source_doc_id"] = source_doc_id
            item = make_meta_item(row, processed)
            selected_source_rows += 1
            for key in counters:
                value = item.get(key)
                if value not in (None, ""):
                    counters[key][value] += 1
            source_mix_chars = item.get("source_mix_chars")
            if isinstance(source_mix_chars, int):
                char_sum += source_mix_chars
                length_values.append(source_mix_chars)
            badness_values.append(float(item.get("greek_badness_score") or 0.0))

            reservoirs["global_random"].add(item)
            reservoirs["feature_screen"].add(item)
            qbin = item.get("quality_bin")
            if qbin in per_bin:
                per_bin[int(qbin)].add(item)
        if args.max_metadata_rows and selected_source_rows >= args.max_metadata_rows:
            break
        if args.progress_every_rows and selected_source_rows and selected_source_rows % args.progress_every_rows < len(rows):
            elapsed = max(1.0, time.time() - start)
            print(
                json.dumps(
                    {
                        "event": "metadata_progress",
                        "rows": selected_source_rows,
                        "rows_per_sec": round(selected_source_rows / elapsed, 1),
                    }
                ),
                flush=True,
            )

    all_reservoirs = dict(reservoirs)
    all_reservoirs.update({name: reservoir for name, reservoir in ((r.name, r) for r in per_bin.values())})
    sample_sets = {name: reservoir.items for name, reservoir in all_reservoirs.items()}

    def numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
        return {
            "count": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }

    report = {
        "metadata_scan": {
            "processed_rows_after_filter": selected_source_rows,
            "raw_batches_rows_seen": processed,
            "missing_source_doc_id_rows": missing_ids,
            "approx_source_mix_chars_sum": char_sum,
            "elapsed_sec": round(time.time() - start, 3),
            "max_metadata_rows": args.max_metadata_rows,
        },
        "sample_sizes": {name: len(items) for name, items in sample_sets.items()},
        "sample_seen_counts": {name: reservoir.seen for name, reservoir in all_reservoirs.items()},
        "counters_top": {key: counter.most_common(args.counter_top_n) for key, counter in counters.items()},
        "length_summary_source_mix_chars": numeric_summary(length_values),
        "greek_badness_summary": numeric_summary(badness_values),
    }
    return report, sample_sets


def fetch_text_rows(args: argparse.Namespace, dataset: Any, schema_names: set[str], wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not wanted_ids:
        return {}
    columns = text_columns(schema_names)
    if "text" not in columns:
        raise RuntimeError("Input does not expose a text column")
    if "source_doc_id" not in columns:
        raise RuntimeError("Input does not expose source_doc_id; cannot fetch sampled IDs safely")

    if len(wanted_ids) > args.filtered_fetch_id_limit:
        return stream_fetch_text_rows(args, dataset, schema_names, wanted_ids, columns)

    value_set = pa.array(sorted(wanted_ids))
    filter_expr = pc.is_in(pc.field("source_doc_id"), value_set=value_set)
    source_filter = build_source_filter(schema_names, args.source_dataset)
    if source_filter is not None:
        filter_expr = source_filter & filter_expr

    found: dict[str, dict[str, Any]] = {}
    start = time.time()
    for batch in dataset.to_batches(columns=columns, filter=filter_expr, batch_size=args.batch_size):
        for row in batch.to_pylist():
            source_doc_id = row_identity(row)
            if source_doc_id in wanted_ids:
                found[source_doc_id] = row
        if len(found) >= len(wanted_ids):
            break
    print(
        json.dumps(
            {
                "event": "text_fetch_complete",
                "wanted": len(wanted_ids),
                "found": len(found),
                "elapsed_sec": round(time.time() - start, 3),
            }
        ),
        flush=True,
    )
    return found


def stream_fetch_text_rows(
    args: argparse.Namespace,
    dataset: Any,
    schema_names: set[str],
    wanted_ids: set[str],
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch sampled text by streaming source-filtered batches.

    PyArrow's large `is_in(source_doc_id, 100k_ids)` filter can build a very
    large in-memory selection. Streaming all source-filtered text is more I/O,
    but the memory footprint is bounded by `--text-batch-size` plus the sampled
    rows that we intentionally retain for review.
    """

    source_filter = build_source_filter(schema_names, args.source_dataset)
    found: dict[str, dict[str, Any]] = {}
    scanned_rows = 0
    start = time.time()
    next_progress = args.progress_every_rows
    print(
        json.dumps(
            {
                "event": "text_fetch_streaming",
                "wanted": len(wanted_ids),
                "text_batch_size": args.text_batch_size,
                "reason": "sampled ID count exceeds filtered_fetch_id_limit",
            }
        ),
        flush=True,
    )

    for batch_index, batch in enumerate(dataset.to_batches(columns=columns, filter=source_filter, batch_size=args.text_batch_size), 1):
        rows = batch.to_pylist()
        scanned_rows += len(rows)
        for row in rows:
            source_doc_id = row_identity(row)
            if source_doc_id in wanted_ids:
                found[source_doc_id] = row
        del rows
        if args.release_memory_every_batches and batch_index % args.release_memory_every_batches == 0:
            release_unused_memory()
        if len(found) >= len(wanted_ids):
            break
        if args.progress_every_rows and scanned_rows >= next_progress:
            elapsed = max(1.0, time.time() - start)
            print(
                json.dumps(
                    {
                        "event": "text_fetch_progress",
                        "rows": scanned_rows,
                        "found": len(found),
                        "wanted": len(wanted_ids),
                        "rows_per_sec": round(scanned_rows / elapsed, 1),
                    }
                ),
                flush=True,
            )
            next_progress += args.progress_every_rows

    print(
        json.dumps(
            {
                "event": "text_fetch_complete",
                "mode": "stream_all",
                "wanted": len(wanted_ids),
                "found": len(found),
                "scanned_rows": scanned_rows,
                "elapsed_sec": round(time.time() - start, 3),
            }
        ),
        flush=True,
    )
    return found


def compute_sampled_feature_rows(
    args: argparse.Namespace,
    dataset: Any,
    schema_names: set[str],
    wanted_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute features for sampled rows without retaining full text."""

    if not wanted_ids:
        return {}
    columns = text_columns(schema_names)
    if "text" not in columns:
        raise RuntimeError("Input does not expose a text column")
    if "source_doc_id" not in columns:
        raise RuntimeError("Input does not expose source_doc_id; cannot compute sampled features safely")

    source_filter = build_source_filter(schema_names, args.source_dataset)
    feature_rows: dict[str, dict[str, Any]] = {}
    scanned_rows = 0
    start = time.time()
    next_progress = args.progress_every_rows
    print(
        json.dumps(
            {
                "event": "sampled_feature_streaming",
                "wanted": len(wanted_ids),
                "text_batch_size": args.text_batch_size,
                "retained_full_text": False,
            }
        ),
        flush=True,
    )

    for batch_index, batch in enumerate(dataset.to_batches(columns=columns, filter=source_filter, batch_size=args.text_batch_size), 1):
        rows = batch.to_pylist()
        scanned_rows += len(rows)
        for row in rows:
            source_doc_id = row_identity(row)
            if source_doc_id in wanted_ids:
                feature_rows[source_doc_id] = compute_features(row, args.feature_text_max_chars)
        del rows
        if args.release_memory_every_batches and batch_index % args.release_memory_every_batches == 0:
            release_unused_memory()
        if len(feature_rows) >= len(wanted_ids):
            break
        if args.progress_every_rows and scanned_rows >= next_progress:
            elapsed = max(1.0, time.time() - start)
            print(
                json.dumps(
                    {
                        "event": "sampled_feature_progress",
                        "rows": scanned_rows,
                        "found": len(feature_rows),
                        "wanted": len(wanted_ids),
                        "rows_per_sec": round(scanned_rows / elapsed, 1),
                    }
                ),
                flush=True,
            )
            next_progress += args.progress_every_rows

    print(
        json.dumps(
            {
                "event": "sampled_feature_complete",
                "wanted": len(wanted_ids),
                "found": len(feature_rows),
                "scanned_rows": scanned_rows,
                "elapsed_sec": round(time.time() - start, 3),
            }
        ),
        flush=True,
    )
    return feature_rows


def write_metadata_sample_manifest(path: Path, sample_sets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    membership: dict[str, set[str]] = defaultdict(set)
    items_by_id: dict[str, dict[str, Any]] = {}
    for sample_name, items in sample_sets.items():
        for item in items:
            source_doc_id = item.get("source_doc_id")
            if not source_doc_id:
                continue
            membership[source_doc_id].add(sample_name)
            items_by_id[source_doc_id] = item

    with path.open("w", encoding="utf-8") as out:
        for source_doc_id in sorted(items_by_id):
            payload = dict(items_by_id[source_doc_id])
            payload["sample_set"] = sorted(membership[source_doc_id])
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_metadata_sample_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    sample_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Could not parse {path}:{line_number}: {exc}") from exc
            sample_names = item.pop("sample_set", [])
            if isinstance(sample_names, str):
                sample_names = [sample_names]
            if not isinstance(sample_names, list):
                raise RuntimeError(f"Invalid sample_set at {path}:{line_number}")
            for sample_name in sample_names:
                sample_sets[str(sample_name)].append(dict(item))
    return dict(sample_sets)


def select_review_docs(
    args: argparse.Namespace,
    feature_rows: dict[str, dict[str, Any]],
    sample_sets: dict[str, list[dict[str, Any]]],
) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = defaultdict(set)
    screen_ids = {item["source_doc_id"] for item in sample_sets.get("feature_screen", [])}

    for sample_name in ("global_random", "wds_bin_8", "wds_bin_9", "wds_bin_10"):
        for item in sample_sets.get(sample_name, []):
            source_doc_id = item["source_doc_id"]
            if source_doc_id in feature_rows:
                selected[source_doc_id].add(sample_name)

    top_dimensions = [
        "encoding_score",
        "markup_score",
        "symbol_score",
        "boilerplate_score",
        "internal_repetition_score",
        "split_candidate_score",
        "lang_drift_score",
        "badness_score",
    ]
    topks = {name: TopK(name, args.targeted_per_feature) for name in top_dimensions}
    borderline = Reservoir("borderline_middle_risk", args.borderline_size, random.Random(args.seed + 9_001))

    for source_doc_id in screen_ids:
        features = feature_rows.get(source_doc_id)
        if not features:
            continue
        for name, topk in topks.items():
            topk.add(float(features.get(name) or 0.0), features)
        overall = float(features.get("overall_artifact_score") or 0.0)
        if args.borderline_min <= overall <= args.borderline_max:
            borderline.add(features)

    for name, topk in topks.items():
        for item in topk.items_desc():
            selected[item["source_doc_id"]].add(f"top_{name}")
    for item in borderline.items:
        selected[item["source_doc_id"]].add("borderline_middle_risk")
    return selected


def write_review_bundle(
    args: argparse.Namespace,
    output_dir: Path,
    timestamp: str,
    feature_rows: dict[str, dict[str, Any]],
    text_rows: dict[str, dict[str, Any]],
    review_sets: dict[str, set[str]],
) -> tuple[Path, dict[str, Any]]:
    bundle_dir = output_dir / "experiments" / f"review_bundle_{timestamp}"
    docs_dir = bundle_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / "manifest.jsonl"

    action_counts = Counter()
    sample_counts = Counter()
    rows_written = 0
    with manifest_path.open("w", encoding="utf-8") as out:
        for source_doc_id in sorted(review_sets):
            features = feature_rows.get(source_doc_id)
            row = text_rows.get(source_doc_id)
            if not features or not row:
                continue
            text = compact_text(row.get("text"))
            text_truncated = len(text) > args.review_text_max_chars
            text_for_file = text[: args.review_text_max_chars] if text_truncated else text
            doc_name = f"{stable_hex(source_doc_id)[:16]}.txt"
            doc_path = docs_dir / doc_name
            doc_path.write_text(text_for_file, encoding="utf-8")

            sample_set = sorted(review_sets[source_doc_id])
            for name in sample_set:
                sample_counts[name] += 1
            proposed_action = propose_action(features)
            candidate_action = propose_candidate_action(features)
            error_type_ids = candidate_error_type_ids(features)
            action_counts[proposed_action] += 1
            record = {
                "schema_version": ACTION_SCHEMA_VERSION,
                "source_doc_id": source_doc_id,
                "parent_source_doc_id": source_doc_id,
                "derived_doc_id": None,
                "is_shadow_record": False,
                "doc_key": features.get("doc_key"),
                "source_dataset": features.get("source_dataset"),
                "sample_set": sample_set,
                "proposed_action": proposed_action,
                "action": candidate_action,
                "action_status": "candidate",
                "policy_id": "P0_inventory_candidate",
                "policy_version": "20260605",
                "created_at_utc": timestamp,
                "error_type_ids": error_type_ids,
                "confidence": features.get("max_artifact_score"),
                "review_label": None,
                "reviewer": None,
                "review_notes": None,
                "doc_text_path": str(doc_path),
                "text_truncated": text_truncated,
                "chars_before": features.get("char_count"),
                "chars_after": None,
                "chars_removed": None,
                "tokens_before": None,
                "tokens_after": None,
                "tokens_removed": None,
                "good_text_loss_estimate": None,
                "span_ranges": [],
                "split_parts": [],
                "text_sha256_before": features.get("text_sha256"),
                "text_sha256_after": None,
                "title": features.get("title"),
                "url": features.get("url"),
                "host": features.get("host"),
                "crawl_id": features.get("crawl_id"),
                "quality_bin": features.get("quality_bin"),
                "overall_artifact_score": features.get("overall_artifact_score"),
                "max_artifact_score": features.get("max_artifact_score"),
                "reason_codes": features.get("reason_codes"),
                "evidence_excerpt": None,
                "detector_scores": {
                    "encoding_score": features.get("encoding_score"),
                    "markup_score": features.get("markup_score"),
                    "symbol_score": features.get("symbol_score"),
                    "boilerplate_score": features.get("boilerplate_score"),
                    "internal_repetition_score": features.get("internal_repetition_score"),
                    "split_candidate_score": features.get("split_candidate_score"),
                    "lang_drift_score": features.get("lang_drift_score"),
                    "badness_score": features.get("badness_score"),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
    summary = {
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "doc_count": rows_written,
        "sample_set_counts": dict(sample_counts),
        "proposed_action_counts": dict(action_counts),
    }
    return manifest_path, summary


def candidate_error_type_ids(features: dict[str, Any]) -> list[str]:
    """Map coarse detector features to candidate catalog IDs for review.

    These IDs are retrieval hooks, not final labels. Manual review promotes or
    rejects categories in ERROR_CATALOG.md.
    """
    ids: set[str] = set()
    score = lambda name: float(features.get(name) or 0.0)

    if score("encoding_score") >= 0.35:
        if int(features.get("control_chars") or 0) or int(features.get("replacement_chars") or 0):
            ids.add("E001")
        if int(features.get("html_entity_count") or 0) or int(features.get("escaped_unicode_count") or 0):
            ids.add("E002")
        if not ids:
            ids.add("E005")

    if score("markup_score") >= 0.35:
        if int(features.get("markup_count") or 0):
            ids.add("E003")
        if float(features.get("url_density_per_10k") or 0.0) >= 5.0:
            ids.add("E004")

    if score("symbol_score") >= 0.35:
        ids.add("E005")

    if score("boilerplate_score") >= 0.35:
        if float(features.get("top_bottom_short_line_ratio") or 0.0) >= 0.45:
            ids.update(["E006", "E007"])
        if int(features.get("menu_hit_count") or 0) >= 5:
            ids.update(["E008", "E009"])

    if score("split_candidate_score") >= 0.35:
        ids.add("E011")
        if int(features.get("split_marker_count") or 0) >= 2:
            ids.add("E012")

    if score("internal_repetition_score") >= 0.35:
        ids.add("E013")
        if float(features.get("duplicate_paragraph_ratio") or 0.0) >= 0.20:
            ids.add("E018")

    if score("lang_drift_score") >= 0.35:
        ids.add("E015")

    if score("badness_score") >= 0.35:
        ids.add("E014")

    if float(features.get("table_ratio") or 0.0) >= 0.50:
        ids.add("E017")

    if int(features.get("char_count") or 0) < 500 and float(features.get("greek_share") or 0.0) < 0.70:
        ids.add("E020")

    return sorted(ids)


def propose_candidate_action(features: dict[str, Any]) -> str:
    proposed = propose_action(features)
    if proposed == "review_drop_candidate":
        return "drop_doc"
    if proposed == "review_split_candidate":
        return "split_doc"
    if proposed == "review_trim_candidate":
        if float(features.get("top_bottom_short_line_ratio") or 0.0) >= 0.55:
            return "trim_prefix"
        return "trim_span"
    if proposed == "review_quarantine_candidate":
        return "quarantine"
    return "keep"


def propose_action(features: dict[str, Any]) -> str:
    scores = {name: float(features.get(name) or 0.0) for name in [
        "encoding_score",
        "markup_score",
        "symbol_score",
        "boilerplate_score",
        "internal_repetition_score",
        "split_candidate_score",
        "lang_drift_score",
        "badness_score",
    ]}
    if scores["badness_score"] >= 0.85 or scores["encoding_score"] >= 0.85 or scores["markup_score"] >= 0.85:
        return "review_drop_candidate"
    if scores["split_candidate_score"] >= 0.70:
        return "review_split_candidate"
    if scores["boilerplate_score"] >= 0.60 or scores["internal_repetition_score"] >= 0.60:
        return "review_trim_candidate"
    if float(features.get("overall_artifact_score") or 0.0) >= 0.35:
        return "review_quarantine_candidate"
    return "review_keep_or_minor"


def write_feature_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_rows = []
    for row in rows:
        payload = dict(row)
        payload["reason_codes"] = json.dumps(payload.get("reason_codes") or [], ensure_ascii=False)
        serializable_rows.append(payload)
    table = pa.Table.from_pylist(serializable_rows)
    pq.write_table(table, path, compression="zstd")


def run_parquet(args: argparse.Namespace) -> dict[str, Any]:
    if PYARROW_IMPORT_ERROR is not None:
        raise RuntimeError(f"pyarrow import failed: {PYARROW_IMPORT_ERROR!r}")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    dataset = ds.dataset(str(input_path), format="parquet")
    schema_names = set(dataset.schema.names)
    required = {"source_doc_id", "text"}
    missing = sorted(required - schema_names)
    if missing:
        raise RuntimeError(f"Input missing required columns: {missing}")

    timestamp = args.timestamp or utc_timestamp()
    output_dir = Path(args.output_dir)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "experiments").mkdir(parents=True, exist_ok=True)

    metadata_sample_path = output_dir / "reports" / f"hplt_cleaning_metadata_samples_{timestamp}.jsonl"
    if args.metadata_sample_manifest:
        source_manifest = Path(args.metadata_sample_manifest)
        print(json.dumps({"event": "metadata_sample_manifest_load", "path": str(source_manifest)}), flush=True)
        sample_sets = load_metadata_sample_manifest(source_manifest)
        write_metadata_sample_manifest(metadata_sample_path, sample_sets)
        metadata_report = {
            "metadata_scan": {
                "loaded_from_manifest": str(source_manifest),
                "processed_rows_after_filter": None,
            },
            "sample_sizes": {name: len(items) for name, items in sample_sets.items()},
        }
        print(json.dumps({"event": "metadata_sample_manifest", "path": str(metadata_sample_path)}), flush=True)
    else:
        print(json.dumps({"event": "metadata_scan_start", "input": str(input_path), "source_dataset": args.source_dataset}), flush=True)
        metadata_report, sample_sets = scan_metadata(args, dataset, schema_names)
        write_metadata_sample_manifest(metadata_sample_path, sample_sets)
        print(json.dumps({"event": "metadata_sample_manifest", "path": str(metadata_sample_path)}), flush=True)

    wanted_ids: set[str] = set()
    for items in sample_sets.values():
        wanted_ids.update(item["source_doc_id"] for item in items if item.get("source_doc_id"))
    print(json.dumps({"event": "sampled_feature_start", "sampled_ids": len(wanted_ids)}), flush=True)
    feature_rows = compute_sampled_feature_rows(args, dataset, schema_names, wanted_ids)

    review_sets = select_review_docs(args, feature_rows, sample_sets)
    print(json.dumps({"event": "review_text_fetch_start", "review_ids": len(review_sets)}), flush=True)
    review_text_rows = fetch_text_rows(args, dataset, schema_names, set(review_sets))
    manifest_path, bundle_summary = write_review_bundle(
        args=args,
        output_dir=output_dir,
        timestamp=timestamp,
        feature_rows=feature_rows,
        text_rows=review_text_rows,
        review_sets=review_sets,
    )

    features_path = output_dir / "reports" / f"hplt_cleaning_sampled_features_{timestamp}.parquet"
    write_feature_parquet(features_path, list(feature_rows.values()))

    inventory_path = output_dir / "reports" / f"hplt_cleaning_inventory_{timestamp}.json"
    report = {
        "script": "hplt_feature_inventory.py",
        "timestamp_utc": timestamp,
        "input": str(input_path),
        "source_dataset_filter": args.source_dataset,
        "cpu_only_cluster_target": {
            "cluster": "Clariden",
            "partition": "xfer",
            "gpu_request": "none",
        },
        "arguments": vars(args),
        "schema_columns": sorted(schema_names),
        "metadata_report": metadata_report,
        "metadata_sample_manifest": str(metadata_sample_path),
        "sampled_text": {
            "wanted_ids": len(wanted_ids),
            "feature_rows": len(feature_rows),
            "retained_full_text_rows": 0,
            "review_text_rows": len(review_text_rows),
        },
        "review_bundle": bundle_summary,
        "feature_parquet": str(features_path),
        "manifest_path": str(manifest_path),
    }
    inventory_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "inventory": str(inventory_path), "features": str(features_path), "manifest": str(manifest_path)}), flush=True)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet file or dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and review bundles.")
    parser.add_argument("--source-dataset", default=SOURCE_DATASET_DEFAULT, help="source_dataset value to scan; empty string disables filtering.")
    parser.add_argument("--timestamp", default="", help="Optional fixed timestamp for reproducible test outputs.")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--filtered-fetch-id-limit", type=int, default=5000)
    parser.add_argument("--max-metadata-rows", type=int, default=0, help="Debug cap after source filtering; 0 means full metadata scan.")
    parser.add_argument("--screen-sample-size", type=int, default=100_000, help="Uniform sample used for targeted detector search.")
    parser.add_argument("--global-random-size", type=int, default=1067)
    parser.add_argument("--per-bin-size", type=int, default=200)
    parser.add_argument("--targeted-per-feature", type=int, default=100)
    parser.add_argument("--borderline-size", type=int, default=300)
    parser.add_argument("--borderline-min", type=float, default=0.18)
    parser.add_argument("--borderline-max", type=float, default=0.38)
    parser.add_argument("--review-text-max-chars", type=int, default=TEXT_MAX_CHARS_DEFAULT)
    parser.add_argument("--feature-text-max-chars", type=int, default=FEATURE_TEXT_MAX_CHARS_DEFAULT)
    parser.add_argument("--release-memory-every-batches", type=int, default=1000)
    parser.add_argument("--metadata-sample-manifest", default="", help="Optional metadata sample JSONL to resume from; skips the metadata scan.")
    parser.add_argument("--progress-every-rows", type=int, default=1_000_000)
    parser.add_argument("--counter-top-n", type=int, default=200)
    args = parser.parse_args(argv)
    if args.source_dataset == "":
        args.source_dataset = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = run_parquet(args)
    print(json.dumps({"event": "summary", "review_docs": report["review_bundle"]["doc_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
