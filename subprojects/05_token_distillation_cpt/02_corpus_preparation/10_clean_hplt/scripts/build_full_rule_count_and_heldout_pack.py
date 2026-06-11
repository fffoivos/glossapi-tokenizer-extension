#!/usr/bin/env python3
"""Full HPLT rule count plus fresh held-out review pack.

This is a non-destructive generalization test. It streams the selected HPLT
source rows, applies the same candidate detector logic used by the development
review bundle, counts rule/action hits over the full slice, and samples unseen
rows for held-out review. It never rewrites source rows.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - cluster preflight catches this.
    ds = None
    PYARROW_IMPORT_ERROR = exc
else:
    PYARROW_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import hplt_feature_inventory as inventory  # noqa: E402
import build_review_action_manifest as action_rules  # noqa: E402


SCHEMA_VERSION = "hplt-full-rule-count-generalization-v1"
REVIEW_SCHEMA_VERSION = "hplt-heldout-generalization-review-v1"
SOURCE_DATASET_DEFAULT = "HPLT/ell_Grek_ge8_no_mt_clean60"

BASE_SCORE_NAMES = [
    "encoding_score",
    "markup_score",
    "symbol_score",
    "boilerplate_score",
    "internal_repetition_score",
    "split_candidate_score",
    "lang_drift_score",
    "badness_score",
]

DOC_HOSTS = {
    "docplayer.gr",
    "issuu.com",
    "www.scribd.com",
    "scribd.com",
    "paperzz.com",
    "fdocument.org",
    "slideplayer.gr",
    "manualzz.com",
}


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def stable_hash_int(seed: int, name: str, source_doc_id: str) -> int:
    raw = f"{seed}\t{name}\t{source_doc_id}".encode("utf-8", "replace")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


class HashReservoir:
    """Deterministic bottom-hash sampler with bounded memory."""

    def __init__(self, name: str, limit: int, seed: int) -> None:
        self.name = name
        self.limit = int(limit)
        self.seed = int(seed)
        self.seen = 0
        self.heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, item: dict[str, Any]) -> None:
        if self.limit <= 0:
            return
        source_doc_id = str(item.get("source_doc_id") or "")
        if not source_doc_id:
            return
        self.seen += 1
        priority = stable_hash_int(self.seed, self.name, source_doc_id)
        payload = (-priority, source_doc_id, item)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, payload)
        elif priority < -self.heap[0][0]:
            heapq.heapreplace(self.heap, payload)

    def items(self) -> list[dict[str, Any]]:
        return [item for _priority, _doc_id, item in sorted(self.heap, reverse=True)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Could not parse {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_excluded_ids(patterns: list[str]) -> tuple[set[str], dict[str, Any]]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern))
    ids: set[str] = set()
    file_rows: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        rows = 0
        for row in read_jsonl(path):
            rows += 1
            for key in ("source_doc_id", "parent_source_doc_id"):
                value = row.get(key)
                if value:
                    ids.add(str(value))
        file_rows.append({"path": str(path), "rows": rows})
    return ids, {"files": file_rows, "unique_source_doc_ids": len(ids)}


def load_include_ids(path: str, sample_sets: list[str]) -> tuple[set[str] | None, dict[str, Any]]:
    if not path:
        return None, {"mode": "all_source_rows"}
    include_path = Path(path)
    wanted_sets = set(sample_sets or [])
    ids: set[str] = set()
    rows_read = 0
    rows_selected = 0
    sample_set_counts: collections.Counter[str] = collections.Counter()
    for row in read_jsonl(include_path):
        rows_read += 1
        source_doc_id = row.get("source_doc_id")
        if not source_doc_id:
            continue
        row_sets_raw = row.get("sample_set") or []
        if isinstance(row_sets_raw, str):
            row_sets = {row_sets_raw}
        else:
            row_sets = {str(item) for item in row_sets_raw}
        if wanted_sets and not (row_sets & wanted_sets):
            continue
        ids.add(str(source_doc_id))
        rows_selected += 1
        for sample_set in row_sets:
            sample_set_counts[sample_set] += 1
    return ids, {
        "mode": "include_source_doc_ids",
        "path": str(include_path),
        "sample_sets": sorted(wanted_sets),
        "rows_read": rows_read,
        "rows_selected": rows_selected,
        "unique_source_doc_ids": len(ids),
        "selected_sample_set_counts": counter_to_dict(sample_set_counts),
    }


def source_shard(source_doc_id: str) -> str:
    parts = source_doc_id.split("::")
    return parts[1] if len(parts) >= 3 else "unknown"


def length_bucket(chars: int) -> str:
    if chars < 1_000:
        return "lt_1k"
    if chars < 3_000:
        return "1k_3k"
    if chars < 10_000:
        return "3k_10k"
    if chars < 100_000:
        return "10k_100k"
    return "ge_100k"


def url_router_reasons(url: Any) -> list[str]:
    if not url:
        return []
    parsed = urlparse(str(url))
    path = parsed.path.casefold()
    query = parse_qs(parsed.query)
    reasons: set[str] = set()
    if "/category/" in path or "/categories/" in path:
        reasons.add("path_category")
    if "/tag/" in path or "/tags/" in path:
        reasons.add("path_tag")
    if "/archive" in path or "/archives" in path:
        reasons.add("path_archive")
    if "search" in path or "/label/" in path:
        reasons.add("path_search_or_label")
    if "/page/" in path:
        reasons.add("path_page_number")
    if "itemlist/date" in path:
        reasons.add("path_itemlist_date")
    for key in ("page", "paged", "start", "catid", "tag", "searchword", "q"):
        if key in query:
            reasons.add(f"query_key_{key}")
    return sorted(reasons)


def source_policy_flags(host: Any, url: Any) -> list[str]:
    flags: set[str] = set()
    host_text = str(host or "").casefold()
    url_text = str(url or "").casefold()
    if host_text in DOC_HOSTS:
        flags.add("doc_host")
    if any(term in host_text or term in url_text for term in ("deal", "deals", "coupon", "eshop", "shop")):
        flags.add("product_or_deal_like")
    return sorted(flags)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def candidate_from_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    text_full = inventory.compact_text(row.get("text"))
    text_for_features = text_full[: args.feature_text_max_chars] if args.feature_text_max_chars > 0 else text_full
    features = inventory.compute_features(row, args.feature_text_max_chars)
    text_features = action_rules.compute_text_features(text_for_features)
    scores = {name: float(features.get(name) or 0.0) for name in BASE_SCORE_NAMES}
    scores.update({str(key): float(value or 0.0) for key, value in text_features.items()})

    coarse_ids = inventory.candidate_error_type_ids(features)
    source_doc_id = str(features.get("source_doc_id") or inventory.row_identity(row))
    record = {
        "source_doc_id": source_doc_id,
        "parent_source_doc_id": source_doc_id,
        "doc_key": features.get("doc_key"),
        "source_dataset": features.get("source_dataset") or args.source_dataset,
        "url": features.get("url"),
        "host": features.get("host"),
        "crawl_id": features.get("crawl_id"),
        "quality_bin": features.get("quality_bin"),
        "text_sha256_before": features.get("text_sha256") or text_sha256(text_full),
        "chars_before": features.get("char_count"),
        "error_type_ids": coarse_ids,
        "action": inventory.propose_candidate_action(features),
        "proposed_action": inventory.propose_action(features),
        "detector_scores": scores,
        "reason_codes": features.get("reason_codes") or [],
        "sample_set": [],
        "text_truncated_for_features": len(text_for_features) < len(text_full),
    }
    error_type_ids = action_rules.infer_error_type_ids(
        record,
        args.threshold,
        scores=scores,
        text=text_for_features,
        text_features=text_features,
    )
    action = action_rules.refine_action(
        action_rules.normalize_action(record),
        error_type_ids,
        features.get("char_count"),
        scores=scores,
        text=text_for_features,
    )
    confidence = max(scores.values()) if scores else 0.0
    reasons = url_router_reasons(record.get("url"))
    flags = source_policy_flags(record.get("host"), record.get("url"))
    chars = int(features.get("char_count") or len(text_full))
    candidate = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_doc_id": source_doc_id,
        "parent_source_doc_id": source_doc_id,
        "derived_doc_id": None,
        "is_shadow_record": False,
        "doc_key": record.get("doc_key"),
        "source_dataset": record.get("source_dataset"),
        "candidate_error_type_ids": sorted(error_type_ids),
        "candidate_action": action,
        "action_status": "candidate",
        "confidence": confidence,
        "quality_bin": record.get("quality_bin"),
        "host": record.get("host"),
        "url": record.get("url"),
        "crawl_id": record.get("crawl_id"),
        "source_shard": source_shard(source_doc_id),
        "length_bucket": length_bucket(chars),
        "chars_before": chars,
        "text_sha256_before": record.get("text_sha256_before"),
        "detector_scores": scores,
        "reason_codes": record.get("reason_codes") or [],
        "url_router_reasons": reasons,
        "source_policy_flags": flags,
        "text_truncated_for_features": record.get("text_truncated_for_features"),
        "review_label": None,
        "review_true_error_type_ids": [],
        "review_correct_action": None,
        "review_span_or_split_notes": None,
        "review_good_text_loss_risk": None,
        "review_good_chars_in_removed_span": None,
        "review_removed_chars": None,
        "review_notes": None,
        "_full_text": text_full,
    }
    return candidate, text_for_features


def update_counts(summary: dict[str, Any], candidate: dict[str, Any]) -> None:
    action = str(candidate.get("candidate_action") or "unknown")
    errors = candidate.get("candidate_error_type_ids") or []
    if not errors:
        errors = ["none"]
    host = str(candidate.get("host") or "unknown")
    qbin = str(candidate.get("quality_bin") or "unknown")
    shard = str(candidate.get("source_shard") or "unknown")
    lbucket = str(candidate.get("length_bucket") or "unknown")
    flags = candidate.get("source_policy_flags") or ["none"]
    url_reasons = candidate.get("url_router_reasons") or ["none"]

    summary["by_action"][action] += 1
    summary["by_quality_bin"][qbin] += 1
    summary["by_source_shard"][shard] += 1
    summary["by_length_bucket"][lbucket] += 1
    summary["by_action_quality_bin"][action][qbin] += 1
    summary["by_action_length_bucket"][action][lbucket] += 1
    summary["by_action_source_shard"][action][shard] += 1
    summary["host_by_action"][action][host] += 1
    summary["chars_by_action"][action] += int(candidate.get("chars_before") or 0)
    for error_id in errors:
        summary["by_error_type"][error_id] += 1
        summary["by_error_action"][error_id][action] += 1
    for flag in flags:
        summary["by_source_policy_flag"][flag] += 1
        summary["by_source_policy_flag_action"][flag][action] += 1
    for reason in url_reasons:
        summary["by_url_router_reason"][reason] += 1
        summary["by_url_router_reason_action"][reason][action] += 1


def sample_candidate(
    reservoirs: dict[str, HashReservoir],
    candidate: dict[str, Any],
    excluded: bool,
) -> None:
    if excluded:
        return
    action = str(candidate.get("candidate_action") or "unknown")
    errors = candidate.get("candidate_error_type_ids") or []
    reservoirs["global_random_unseen"].add(candidate)
    if action == "keep" and not errors:
        reservoirs["nohit_keep_control"].add(candidate)
    if action != "keep":
        reservoirs[f"action::{action}"].add(candidate)
    for error_id in errors:
        reservoirs[f"error::{error_id}"].add(candidate)
    if candidate.get("url_router_reasons"):
        reservoirs["url_router_signal"].add(candidate)
    if candidate.get("source_policy_flags"):
        reservoirs["source_policy_signal"].add(candidate)


def counter_to_dict(counter: collections.Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def nested_counter_to_dict(mapping: dict[str, collections.Counter[Any]]) -> dict[str, dict[str, int]]:
    return {str(key): counter_to_dict(counter) for key, counter in sorted(mapping.items())}


def top_nested_hosts(mapping: dict[str, collections.Counter[Any]], n: int) -> dict[str, list[list[Any]]]:
    return {
        str(key): [[host, int(count)] for host, count in counter.most_common(n)]
        for key, counter in sorted(mapping.items())
    }


def make_reservoirs(args: argparse.Namespace) -> dict[str, HashReservoir]:
    reservoirs = {
        "global_random_unseen": HashReservoir("global_random_unseen", args.heldout_global_n, args.seed),
        "nohit_keep_control": HashReservoir("nohit_keep_control", args.heldout_nohit_n, args.seed),
        "url_router_signal": HashReservoir("url_router_signal", args.heldout_url_router_n, args.seed),
        "source_policy_signal": HashReservoir("source_policy_signal", args.heldout_source_policy_n, args.seed),
    }
    for action in ("drop_doc", "normalize_or_trim_span", "quarantine", "split_doc", "trim_prefix", "trim_span", "trim_suffix"):
        reservoirs[f"action::{action}"] = HashReservoir(f"action::{action}", args.heldout_per_action, args.seed)
    for error_id in action_rules.ERROR_NAMES:
        reservoirs[f"error::{error_id}"] = HashReservoir(f"error::{error_id}", args.heldout_per_error, args.seed)
    return reservoirs


def collect_sampled_rows(reservoirs: dict[str, HashReservoir]) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    sample_sets: dict[str, set[str]] = collections.defaultdict(set)
    for sample_name, reservoir in reservoirs.items():
        for item in reservoir.items():
            source_doc_id = str(item.get("source_doc_id") or "")
            if not source_doc_id:
                continue
            if source_doc_id not in rows_by_id:
                rows_by_id[source_doc_id] = dict(item)
            sample_sets[source_doc_id].add(sample_name)
    rows: list[dict[str, Any]] = []
    for source_doc_id in sorted(rows_by_id):
        row = rows_by_id[source_doc_id]
        row["sample_set"] = sorted(sample_sets[source_doc_id])
        rows.append(row)
    return rows


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    timestamp: str,
    raw_summary: dict[str, Any],
    reservoirs: dict[str, HashReservoir],
    excluded_summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = output_dir / f"full_rule_count_{timestamp}_heldout_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    review_pack_path = output_dir / f"full_rule_count_{timestamp}_heldout_review_pack.jsonl"
    template_path = output_dir / f"full_rule_count_{timestamp}_heldout_annotation_template.jsonl"
    summary_path = output_dir / f"full_rule_count_{timestamp}_summary.json"
    md_path = output_dir / f"full_rule_count_{timestamp}.md"

    sampled_rows = collect_sampled_rows(reservoirs)
    with review_pack_path.open("w", encoding="utf-8") as out, template_path.open("w", encoding="utf-8") as tmpl:
        for idx, row in enumerate(sampled_rows, 1):
            full_text = str(row.pop("_full_text", ""))
            truncated = len(full_text) > args.review_text_max_chars
            text_for_review = full_text[: args.review_text_max_chars] if truncated else full_text
            doc_name = f"{idx:05d}_{hashlib.sha1(row['source_doc_id'].encode()).hexdigest()[:12]}.txt"
            doc_path = docs_dir / doc_name
            doc_path.write_text(text_for_review, encoding="utf-8")
            row["doc_text_path"] = str(doc_path)
            row["text_truncated_for_review"] = truncated
            row["created_at_utc"] = timestamp
            row["heldout_review_id"] = f"hg_{idx:05d}_{hashlib.sha1(row['source_doc_id'].encode()).hexdigest()[:10]}"
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            annotation = {
                "heldout_review_id": row["heldout_review_id"],
                "source_doc_id": row["source_doc_id"],
                "sample_set": row.get("sample_set") or [],
                "host": row.get("host"),
                "url": row.get("url"),
                "quality_bin": row.get("quality_bin"),
                "doc_text_path": row.get("doc_text_path"),
                "candidate_action": row["candidate_action"],
                "candidate_error_type_ids": row["candidate_error_type_ids"],
                "review_label": None,
                "review_true_error_type_ids": [],
                "review_correct_action": None,
                "review_good_text_loss_risk": None,
                "review_good_chars_in_removed_span": None,
                "review_removed_chars": None,
                "review_notes": None,
            }
            tmpl.write(json.dumps(annotation, ensure_ascii=False, sort_keys=True) + "\n")

    reservoir_summary = {
        name: {
            "limit": reservoir.limit,
            "seen": reservoir.seen,
            "sampled": len(reservoir.items()),
        }
        for name, reservoir in sorted(reservoirs.items())
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": timestamp,
        "policy_note": "Count-only generalization pass. Source HPLT rows are immutable; no cleaning was applied.",
        "input": str(args.input),
        "source_dataset": args.source_dataset,
        "threshold": args.threshold,
        "max_rows": args.max_rows,
        "feature_text_max_chars": args.feature_text_max_chars,
        "review_text_max_chars": args.review_text_max_chars,
        "excluded_previous_evidence": excluded_summary,
        "included_source_doc_ids": raw_summary["include_summary"],
        "source_rows_seen": raw_summary["source_rows_seen"],
        "include_ids_found": raw_summary["include_ids_found"],
        "rows_scanned": raw_summary["rows_scanned"],
        "rows_excluded_from_heldout_sampling": raw_summary["rows_excluded"],
        "elapsed_sec": raw_summary["elapsed_sec"],
        "rows_per_sec": raw_summary["rows_per_sec"],
        "by_action": counter_to_dict(raw_summary["by_action"]),
        "by_error_type": counter_to_dict(raw_summary["by_error_type"]),
        "by_error_action": nested_counter_to_dict(raw_summary["by_error_action"]),
        "by_quality_bin": counter_to_dict(raw_summary["by_quality_bin"]),
        "by_source_shard": counter_to_dict(raw_summary["by_source_shard"]),
        "by_length_bucket": counter_to_dict(raw_summary["by_length_bucket"]),
        "by_action_quality_bin": nested_counter_to_dict(raw_summary["by_action_quality_bin"]),
        "by_action_length_bucket": nested_counter_to_dict(raw_summary["by_action_length_bucket"]),
        "by_action_source_shard": nested_counter_to_dict(raw_summary["by_action_source_shard"]),
        "top_hosts_by_action": top_nested_hosts(raw_summary["host_by_action"], args.top_hosts),
        "chars_by_action": counter_to_dict(raw_summary["chars_by_action"]),
        "by_source_policy_flag": counter_to_dict(raw_summary["by_source_policy_flag"]),
        "by_source_policy_flag_action": nested_counter_to_dict(raw_summary["by_source_policy_flag_action"]),
        "by_url_router_reason": counter_to_dict(raw_summary["by_url_router_reason"]),
        "by_url_router_reason_action": nested_counter_to_dict(raw_summary["by_url_router_reason_action"]),
        "heldout_sampling": reservoir_summary,
        "heldout_review_pack_rows": len(sampled_rows),
        "heldout_review_pack_jsonl": str(review_pack_path),
        "heldout_annotation_template_jsonl": str(template_path),
        "heldout_docs_dir": str(docs_dir),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, summary)
    return {
        "summary_json": str(summary_path),
        "summary_md": str(md_path),
        "heldout_review_pack_jsonl": str(review_pack_path),
        "heldout_annotation_template_jsonl": str(template_path),
        "heldout_docs_dir": str(docs_dir),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("# Full HPLT Rule Count And Held-Out Pack\n\n")
        out.write("This is count-only generalization evidence. It applies no cleaning and mutates no source rows.\n\n")
        out.write("## Run\n\n")
        out.write(f"- created: `{summary['created_at_utc']}`\n")
        out.write(f"- input: `{summary['input']}`\n")
        out.write(f"- source dataset: `{summary['source_dataset']}`\n")
        out.write(f"- rows scanned: `{summary['rows_scanned']}`\n")
        out.write(f"- elapsed sec: `{summary['elapsed_sec']}`\n")
        out.write(f"- rows/sec: `{summary['rows_per_sec']}`\n")
        out.write(f"- held-out pack rows: `{summary['heldout_review_pack_rows']}`\n")
        out.write(f"- previously seen rows excluded from held-out sampling: `{summary['rows_excluded_from_heldout_sampling']}`\n\n")
        out.write("## Counts By Candidate Action\n\n")
        out.write("| Action | Rows | Candidate row % | Chars before |\n")
        out.write("| --- | ---: | ---: | ---: |\n")
        total = max(1, int(summary["rows_scanned"]))
        chars_by_action = summary.get("chars_by_action") or {}
        for action, count in sorted(summary["by_action"].items(), key=lambda item: (-item[1], item[0])):
            out.write(f"| `{action}` | {count} | {100.0 * count / total:.4f}% | {chars_by_action.get(action, 0)} |\n")
        out.write("\n## Counts By Candidate Error Type\n\n")
        out.write("| Error ID | Rows |\n")
        out.write("| --- | ---: |\n")
        for error_id, count in sorted(summary["by_error_type"].items(), key=lambda item: (-item[1], item[0])):
            out.write(f"| `{error_id}` | {count} |\n")
        out.write("\n## Held-Out Sampling\n\n")
        out.write("| Sample | Seen | Sampled |\n")
        out.write("| --- | ---: | ---: |\n")
        for name, payload in sorted(summary["heldout_sampling"].items()):
            sampled = payload.get("sampled", 0)
            if sampled:
                out.write(f"| `{name}` | {payload.get('seen', 0)} | {sampled} |\n")
        out.write("\n## Top Hosts By Action\n\n")
        for action, hosts in sorted((summary.get("top_hosts_by_action") or {}).items()):
            if not hosts:
                continue
            out.write(f"\n### `{action}`\n\n")
            out.write("| Host | Rows |\n")
            out.write("| --- | ---: |\n")
            for host, count in hosts[:20]:
                out.write(f"| `{host}` | {count} |\n")


def run(args: argparse.Namespace) -> dict[str, str]:
    if PYARROW_IMPORT_ERROR is not None:
        raise RuntimeError(f"pyarrow import failed: {PYARROW_IMPORT_ERROR!r}")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    timestamp = args.timestamp or utc_timestamp()
    output_dir = Path(args.output_dir)
    excluded_ids, excluded_summary = load_excluded_ids(args.exclude_jsonl_glob)
    include_ids, include_summary = load_include_ids(args.include_jsonl, args.include_sample_set)
    reservoirs = make_reservoirs(args)

    dataset = ds.dataset(str(input_path), format="parquet")
    schema_names = set(dataset.schema.names)
    missing = sorted({"source_doc_id", "text"} - schema_names)
    if missing:
        raise RuntimeError(f"Input missing required columns: {missing}")
    columns = inventory.text_columns(schema_names)
    source_filter = inventory.build_source_filter(schema_names, args.source_dataset)

    raw_summary: dict[str, Any] = {
        "rows_scanned": 0,
        "source_rows_seen": 0,
        "include_ids_found": 0,
        "include_summary": include_summary,
        "rows_excluded": 0,
        "by_action": collections.Counter(),
        "by_error_type": collections.Counter(),
        "by_error_action": collections.defaultdict(collections.Counter),
        "by_quality_bin": collections.Counter(),
        "by_source_shard": collections.Counter(),
        "by_length_bucket": collections.Counter(),
        "by_action_quality_bin": collections.defaultdict(collections.Counter),
        "by_action_length_bucket": collections.defaultdict(collections.Counter),
        "by_action_source_shard": collections.defaultdict(collections.Counter),
        "host_by_action": collections.defaultdict(collections.Counter),
        "chars_by_action": collections.Counter(),
        "by_source_policy_flag": collections.Counter(),
        "by_source_policy_flag_action": collections.defaultdict(collections.Counter),
        "by_url_router_reason": collections.Counter(),
        "by_url_router_reason_action": collections.defaultdict(collections.Counter),
    }

    start = time.time()
    next_progress = args.progress_every_rows
    print(
        json.dumps(
            {
                "event": "start",
                "input": str(input_path),
                "source_dataset": args.source_dataset,
                "excluded_ids": len(excluded_ids),
                "include_ids": None if include_ids is None else len(include_ids),
                "include_summary": include_summary,
                "batch_size": args.batch_size,
                "feature_text_max_chars": args.feature_text_max_chars,
            }
        ),
        flush=True,
    )

    for batch_index, batch in enumerate(dataset.to_batches(columns=columns, filter=source_filter, batch_size=args.batch_size), 1):
        rows = batch.to_pylist()
        for row in rows:
            if args.max_rows and raw_summary["rows_scanned"] >= args.max_rows:
                break
            raw_summary["source_rows_seen"] += 1
            source_doc_id = inventory.row_identity(row)
            if include_ids is not None and source_doc_id not in include_ids:
                continue
            if include_ids is not None:
                raw_summary["include_ids_found"] += 1
            candidate, _text = candidate_from_row(row, args)
            source_doc_id = str(candidate.get("source_doc_id") or source_doc_id or "")
            raw_summary["rows_scanned"] += 1
            excluded = source_doc_id in excluded_ids
            if excluded:
                raw_summary["rows_excluded"] += 1
            update_counts(raw_summary, candidate)
            sample_candidate(reservoirs, candidate, excluded)
        del rows
        if args.release_memory_every_batches and batch_index % args.release_memory_every_batches == 0:
            inventory.release_unused_memory()
        if args.max_rows and raw_summary["rows_scanned"] >= args.max_rows:
            break
        if include_ids is not None and raw_summary["include_ids_found"] >= len(include_ids):
            break
        progress_value = raw_summary["source_rows_seen"] if include_ids is not None else raw_summary["rows_scanned"]
        if args.progress_every_rows and progress_value >= next_progress:
            elapsed = max(1.0, time.time() - start)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "rows": raw_summary["rows_scanned"],
                        "source_rows_seen": raw_summary["source_rows_seen"],
                        "include_ids_found": raw_summary["include_ids_found"],
                        "rows_per_sec": round(raw_summary["rows_scanned"] / elapsed, 2),
                        "source_rows_per_sec": round(raw_summary["source_rows_seen"] / elapsed, 2),
                        "by_action": counter_to_dict(raw_summary["by_action"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            next_progress += args.progress_every_rows

    elapsed = max(0.001, time.time() - start)
    raw_summary["elapsed_sec"] = round(elapsed, 3)
    raw_summary["rows_per_sec"] = round(raw_summary["rows_scanned"] / elapsed, 3)
    outputs = write_outputs(args, output_dir, timestamp, raw_summary, reservoirs, excluded_summary)
    print(json.dumps({"event": "complete", **outputs}, ensure_ascii=False), flush=True)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dataset", default=SOURCE_DATASET_DEFAULT)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--feature-text-max-chars", type=int, default=100000)
    parser.add_argument("--review-text-max-chars", type=int, default=120000)
    parser.add_argument("--progress-every-rows", type=int, default=1000000)
    parser.add_argument("--release-memory-every-batches", type=int, default=200)
    parser.add_argument("--heldout-global-n", type=int, default=400)
    parser.add_argument("--heldout-nohit-n", type=int, default=400)
    parser.add_argument("--heldout-per-action", type=int, default=100)
    parser.add_argument("--heldout-per-error", type=int, default=75)
    parser.add_argument("--heldout-url-router-n", type=int, default=250)
    parser.add_argument("--heldout-source-policy-n", type=int, default=150)
    parser.add_argument("--top-hosts", type=int, default=50)
    parser.add_argument("--include-jsonl", default="", help="Optional JSONL with source_doc_id rows to count instead of all source rows.")
    parser.add_argument(
        "--include-sample-set",
        action="append",
        default=[],
        help="When --include-jsonl is set, keep only rows whose sample_set contains this value. May be repeated.",
    )
    parser.add_argument(
        "--exclude-jsonl-glob",
        action="append",
        default=[],
        help="JSONL glob(s) whose source_doc_id/parent_source_doc_id values are excluded from held-out sampling.",
    )
    args = parser.parse_args()
    if not args.exclude_jsonl_glob:
        args.exclude_jsonl_glob = [str(Path("reports") / "*.jsonl")]
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
