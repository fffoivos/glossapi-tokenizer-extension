#!/usr/bin/env python3
"""Map FineWeb-2 documents as possible fragments of our larger documents."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import regex


WORD_RE = regex.compile(r"\p{L}[\p{L}\p{M}\p{N}'’·-]*")


@dataclass
class OurDoc:
    source_dataset: str
    source_doc_id: str
    title: str
    author: str
    token_count: int


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def tokenize(text: str) -> list[str]:
    return [m.group(0).casefold() for m in WORD_RE.finditer(text or "")]


def shingle_key(tokens: list[str], start: int, size: int) -> int:
    return hash(tuple(tokens[start : start + size]))


def host(url: str) -> str:
    if not url:
        return ""
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def build_our_index(
    ours_path: Path,
    *,
    shingle_size: int,
    stride: int,
    noisy_doc_frequency: int,
) -> tuple[list[OurDoc], dict[int, list[int]], dict[str, int]]:
    pf = pq.ParquetFile(ours_path)
    docs: list[OurDoc] = []
    index: dict[int, list[int]] = defaultdict(list)
    stats = {
        "rows": 0,
        "tokens": 0,
        "sampled_anchors_before_noise_filter": 0,
        "removed_noisy_anchors": 0,
    }
    columns = ["source_dataset", "source_doc_id", "title", "author", "text"]
    for batch_i, batch in enumerate(pf.iter_batches(columns=columns, batch_size=128), start=1):
        data = batch.to_pydict()
        for i, text in enumerate(data["text"]):
            tokens = tokenize(text or "")
            doc_idx = len(docs)
            docs.append(
                OurDoc(
                    source_dataset=str(data["source_dataset"][i] or ""),
                    source_doc_id=str(data["source_doc_id"][i] or ""),
                    title=str(data["title"][i] or ""),
                    author=str(data["author"][i] or ""),
                    token_count=len(tokens),
                )
            )
            stats["rows"] += 1
            stats["tokens"] += len(tokens)
            if len(tokens) < shingle_size:
                continue
            seen_for_doc: set[int] = set()
            for start in range(0, len(tokens) - shingle_size + 1, stride):
                key = shingle_key(tokens, start, shingle_size)
                if key not in seen_for_doc:
                    index[key].append(doc_idx)
                    seen_for_doc.add(key)
                    stats["sampled_anchors_before_noise_filter"] += 1
        if batch_i % 20 == 0:
            log(f"indexed ours batches={batch_i} docs={stats['rows']} anchors={stats['sampled_anchors_before_noise_filter']}")

    noisy = [key for key, value in index.items() if len(value) > noisy_doc_frequency]
    for key in noisy:
        del index[key]
    stats["removed_noisy_anchors"] = len(noisy)
    stats["sampled_anchors_after_noise_filter"] = len(index)
    return docs, index, stats


def scan_fineweb(
    fineweb_path: Path,
    our_docs: list[OurDoc],
    index: dict[int, list[int]],
    *,
    shingle_size: int,
    stride: int,
    min_fine_tokens: int,
    min_anchor_hits: int,
    min_containment: float,
) -> dict[str, object]:
    pf = pq.ParquetFile(fineweb_path)
    matches: list[dict[str, object]] = []
    fine_stats = {
        "rows": 0,
        "eligible_rows": 0,
        "tokens": 0,
        "with_any_anchor": 0,
        "matched_rows": 0,
    }
    containment_values: list[float] = []
    by_domain: Counter[str] = Counter()
    by_our_source: Counter[str] = Counter()
    by_threshold = {">=0.25": 0, ">=0.50": 0, ">=0.75": 0, ">=0.90": 0}

    columns = ["id", "url", "text"]
    for batch_i, batch in enumerate(pf.iter_batches(columns=columns, batch_size=256), start=1):
        data = batch.to_pydict()
        for i, text in enumerate(data["text"]):
            fine_stats["rows"] += 1
            tokens = tokenize(text or "")
            fine_stats["tokens"] += len(tokens)
            if len(tokens) < max(min_fine_tokens, shingle_size):
                continue
            fine_stats["eligible_rows"] += 1
            sampled_count = ((len(tokens) - shingle_size) // stride) + 1
            counts: Counter[int] = Counter()
            for start in range(0, len(tokens) - shingle_size + 1, stride):
                for doc_idx in index.get(shingle_key(tokens, start, shingle_size), []):
                    counts[doc_idx] += 1
            if not counts:
                continue
            fine_stats["with_any_anchor"] += 1
            best_doc_idx, best_hits = counts.most_common(1)[0]
            containment = best_hits / sampled_count if sampled_count else 0.0
            containment_values.append(containment)
            if containment >= 0.25:
                by_threshold[">=0.25"] += 1
            if containment >= 0.50:
                by_threshold[">=0.50"] += 1
            if containment >= 0.75:
                by_threshold[">=0.75"] += 1
            if containment >= 0.90:
                by_threshold[">=0.90"] += 1
            if best_hits < min_anchor_hits or containment < min_containment:
                continue
            fine_stats["matched_rows"] += 1
            our = our_docs[best_doc_idx]
            domain = host(str(data["url"][i] or ""))
            by_domain[domain] += 1
            by_our_source[our.source_dataset] += 1
            matches.append(
                {
                    "fineweb_id": str(data["id"][i] or ""),
                    "fineweb_url": str(data["url"][i] or ""),
                    "fineweb_domain": domain,
                    "fineweb_tokens": len(tokens),
                    "our_source_dataset": our.source_dataset,
                    "our_source_doc_id": our.source_doc_id,
                    "our_title": our.title,
                    "our_author": our.author,
                    "our_tokens": our.token_count,
                    "sampled_anchor_hits": best_hits,
                    "sampled_anchor_count": sampled_count,
                    "containment_estimate": containment,
                }
            )
        if batch_i % 20 == 0:
            log(
                "scanned fineweb "
                f"batches={batch_i} rows={fine_stats['rows']} eligible={fine_stats['eligible_rows']} "
                f"matched={fine_stats['matched_rows']}"
            )

    matches.sort(key=lambda row: (-float(row["containment_estimate"]), -int(row["sampled_anchor_hits"])))
    if containment_values:
        containment_summary = {
            "mean": statistics.fmean(containment_values),
            "max": max(containment_values),
            "p50": sorted(containment_values)[len(containment_values) // 2],
            "p90": sorted(containment_values)[int((len(containment_values) - 1) * 0.90)],
            "p95": sorted(containment_values)[int((len(containment_values) - 1) * 0.95)],
        }
    else:
        containment_summary = {}
    return {
        "fineweb_stats": fine_stats,
        "containment_summary_for_docs_with_any_anchor": containment_summary,
        "matched_by_threshold": by_threshold,
        "matched_by_fineweb_domain": dict(by_domain.most_common(50)),
        "matched_by_our_source": dict(by_our_source.most_common()),
        "top_matches": matches[:300],
    }


def write_report(path: Path, result: dict[str, object]) -> None:
    fine = result["fineweb_scan"]
    lines = [
        "# FineWeb-2 Directional Fragment Containment",
        "",
        "This maps FineWeb documents onto our larger documents using sampled",
        "word-shingle anchors. It is directional: FineWeb is treated as the",
        "possible fragment side, our corpus as the possible containing side.",
        "",
        "## Parameters",
        "",
    ]
    for key, value in result["parameters"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            f"- FineWeb rows scanned: `{fine['fineweb_stats']['rows']}`",
            f"- Eligible FineWeb rows: `{fine['fineweb_stats']['eligible_rows']}`",
            f"- FineWeb rows with any anchor in our corpus: `{fine['fineweb_stats']['with_any_anchor']}`",
            f"- FineWeb rows passing the selected containment threshold: `{fine['fineweb_stats']['matched_rows']}`",
            "",
            "Threshold counts:",
            "",
        ]
    )
    for threshold, count in fine["matched_by_threshold"].items():
        lines.append(f"- `{threshold}`: {count:,}")
    lines.extend(["", "Matches by our source:", ""])
    for source, count in fine["matched_by_our_source"].items():
        lines.append(f"- `{source}`: {count:,}")
    lines.extend(["", "Matches by FineWeb domain:", ""])
    for domain, count in list(fine["matched_by_fineweb_domain"].items())[:30]:
        lines.append(f"- `{domain}`: {count:,}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--fineweb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shingle-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--noisy-doc-frequency", type=int, default=25)
    parser.add_argument("--min-fine-tokens", type=int, default=80)
    parser.add_argument("--min-anchor-hits", type=int, default=5)
    parser.add_argument("--min-containment", type=float, default=0.25)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log("building our anchor index")
    our_docs, index, index_stats = build_our_index(
        args.ours,
        shingle_size=args.shingle_size,
        stride=args.stride,
        noisy_doc_frequency=args.noisy_doc_frequency,
    )
    log("scanning fineweb")
    fine_scan = scan_fineweb(
        args.fineweb,
        our_docs,
        index,
        shingle_size=args.shingle_size,
        stride=args.stride,
        min_fine_tokens=args.min_fine_tokens,
        min_anchor_hits=args.min_anchor_hits,
        min_containment=args.min_containment,
    )
    result = {
        "parameters": {
            "shingle_size": args.shingle_size,
            "stride": args.stride,
            "noisy_doc_frequency": args.noisy_doc_frequency,
            "min_fine_tokens": args.min_fine_tokens,
            "min_anchor_hits": args.min_anchor_hits,
            "min_containment": args.min_containment,
        },
        "our_index": index_stats,
        "fineweb_scan": fine_scan,
    }
    (args.out_dir / "fragment_containment_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir / "fragment_containment_report.md", result)
    log("done")
    print(json.dumps({k: result[k] for k in ["parameters", "our_index"]}, ensure_ascii=False))
    print(json.dumps(fine_scan["fineweb_stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
