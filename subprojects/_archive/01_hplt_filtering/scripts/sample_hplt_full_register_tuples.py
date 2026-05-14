#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import os
import re
import shlex
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

from hplt_web_register import (
    category_tuple_labels_from_sub_label,
    label_name,
    top_main_label,
    top_sub_label,
)


DEFAULT_BASE_URL = "https://data.hplt-project.org/three/sorted/ell_Grek/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the full Greek HPLT sorted shards, compute canonical hierarchical register "
            "categories, and select deterministic random samples per category and quality bucket."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for the ell_Grek shard index.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--samples-per-bucket", type=int, default=40, help="Target sample size per category and quality bucket.")
    parser.add_argument("--quality-threshold", type=int, default=8, help="Threshold separating >=N from <N buckets.")
    parser.add_argument("--workers", type=int, default=4, help="Number of shard workers to run in parallel.")
    parser.add_argument("--seed", type=int, default=20260410, help="Deterministic seed for pseudo-random sampling.")
    parser.add_argument(
        "--only-shards",
        nargs="*",
        default=None,
        help="Optional shard filenames to include in the exact order provided.",
    )
    parser.add_argument(
        "--checkpoint-after-each-shard",
        action="store_true",
        help="Rewrite outputs after every completed shard instead of only once at the end.",
    )
    parser.add_argument(
        "--target-category-labels-file",
        type=Path,
        default=None,
        help="Optional newline-delimited list of category tuple labels that must be filled for both buckets.",
    )
    parser.add_argument(
        "--stop-when-targets-full",
        action="store_true",
        help="In quota-fill mode, stop streaming once every target category is filled for both buckets.",
    )
    return parser.parse_args()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return slug.strip("._-") or "category"


def fetch_shard_urls(base_url: str, only_shards: list[str] | None = None) -> list[str]:
    response = requests.get(base_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    discovered: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not href.endswith(".jsonl.zst"):
            continue
        discovered[href] = requests.compat.urljoin(base_url, href)
    if only_shards is not None:
        return [discovered[name] for name in only_shards if name in discovered]
    shard_urls: list[str] = list(discovered.values())
    def shard_key(url: str) -> tuple[int, int]:
        name = Path(urlparse(url).path).name
        left, right = name.replace(".jsonl.zst", "").split("_")
        return int(left), int(right)
    shard_urls.sort(key=shard_key)
    return shard_urls


def quality_bin_from_shard_name(shard_name: str) -> int:
    return int(shard_name.split("_", 1)[0])


def quality_bucket(quality_bin: int, threshold: int) -> str:
    return f"ge{threshold}" if quality_bin >= threshold else f"lt{threshold}"


def priority_for(seed: int, shard_name: str, row_id: str | None, row_index: int) -> float:
    token = f"{seed}|{shard_name}|{row_id or ''}|{row_index}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def category_record(row: dict[str, Any]) -> tuple[str, str, str | None, str | None, str | None, float, str | None, float]:
    web_register = row.get("web-register") or {}
    main_code, main_score = top_main_label(web_register)
    sub_code, sub_score = top_sub_label(web_register)
    main_label, sub_label = category_tuple_labels_from_sub_label(sub_code)
    if main_code and main_label and sub_label and main_label == label_name(main_code):
        return (
            f"{main_label} | {sub_label}",
            main_label,
            sub_label,
            main_code,
            sub_code,
            main_score,
            sub_code,
            sub_score,
        )
    if main_code:
        return (
            f"{label_name(main_code)} | No subcategory",
            label_name(main_code),
            "No subcategory",
            main_code,
            None,
            main_score,
            None,
            0.0,
        )
    return (
        "Null group",
        "Null group",
        "Null group",
        None,
        None,
        0.0,
        None,
        0.0,
    )


def push_candidate(
    heaps: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
    category_label: str,
    bucket: str,
    priority: float,
    record: dict[str, Any],
    limit: int,
) -> None:
    key = (category_label, bucket)
    heap = heaps[key]
    item = (priority, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if priority > heap[0][0]:
        heapq.heapreplace(heap, item)


def stream_shard(url: str, samples_per_bucket: int, quality_threshold: int, seed: int) -> dict[str, Any]:
    shard_name = Path(urlparse(url).path).name
    qbin = quality_bin_from_shard_name(shard_name)
    bucket = quality_bucket(qbin, quality_threshold)
    cmd = f"curl -L --silent {shlex.quote(url)} | zstd -dc"
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None

    tuple_counts: Counter[str] = Counter()
    main_counts: Counter[str] = Counter()
    heaps: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    rows = 0

    try:
        for line in proc.stdout:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            text = row.get("text") or ""
            if not text.strip():
                continue

            category_label, canonical_main, canonical_sub, top_main_code, top_sub_code, top_main_score, canonical_sub_code, top_sub_score = category_record(row)
            main_counts[canonical_main] += 1
            tuple_counts[category_label] += 1

            record = {
                "id": row.get("id"),
                "url": row.get("u"),
                "host": urlparse(row.get("u") or "").netloc or None,
                "content_type": row.get("c"),
                "crawl_id": row.get("crawl_id"),
                "timestamp": row.get("ts"),
                "cluster_size": row.get("cluster_size"),
                "filter": row.get("filter"),
                "quality_bin": str(qbin),
                "quality_bucket": bucket,
                "shard": shard_name,
                "char_count": len(text),
                "top_main_label_code": top_main_code,
                "top_main_label": label_name(top_main_code) if top_main_code else None,
                "top_main_score": top_main_score,
                "top_sub_label_code": top_sub_code,
                "top_sub_label": label_name(top_sub_code) if top_sub_code else None,
                "top_sub_score": top_sub_score,
                "canonical_main_label": canonical_main,
                "canonical_sub_label_code": canonical_sub_code,
                "canonical_sub_label": canonical_sub,
                "category_tuple": [canonical_main, canonical_sub],
                "category_tuple_label": category_label,
                "text": text,
            }
            prio = priority_for(seed, shard_name, row.get("id"), rows)
            push_candidate(heaps, category_label, bucket, prio, record, samples_per_bucket)
    finally:
        proc.kill()
        proc.wait()

    serialized_heaps = {
        f"{category_label}\t{bucket_name}": [
            {"priority": priority, "record": record} for priority, record in heap
        ]
        for (category_label, bucket_name), heap in heaps.items()
    }
    return {
        "url": url,
        "shard": shard_name,
        "quality_bin": qbin,
        "rows_seen": rows,
        "tuple_counts": dict(tuple_counts),
        "main_counts": dict(main_counts),
        "heaps": serialized_heaps,
    }


def load_target_category_labels(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def targets_full(
    *,
    target_labels: list[str],
    merged_heaps: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
    quality_threshold: int,
    samples_per_bucket: int,
) -> bool:
    if not target_labels:
        return False
    left_bucket = f"ge{quality_threshold}"
    right_bucket = f"lt{quality_threshold}"
    for label in target_labels:
        if len(merged_heaps.get((label, left_bucket), [])) < samples_per_bucket:
            return False
        if len(merged_heaps.get((label, right_bucket), [])) < samples_per_bucket:
            return False
    return True


def merge_one_result(
    result: dict[str, Any],
    merged_heaps: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
    tuple_counts_by_bucket: dict[str, Counter[str]],
    main_counts_by_bucket: dict[str, Counter[str]],
    shard_summaries: list[dict[str, Any]],
    samples_per_bucket: int,
    quality_threshold: int,
) -> None:
    bucket = quality_bucket(int(result["quality_bin"]), threshold=quality_threshold)
    shard_summaries.append(
        {
            "shard": result["shard"],
            "quality_bin": result["quality_bin"],
            "rows_seen": result["rows_seen"],
        }
    )
    for label, count in result["tuple_counts"].items():
        tuple_counts_by_bucket[bucket][label] += count
    for label, count in result["main_counts"].items():
        main_counts_by_bucket[bucket][label] += count
    for key, items in result["heaps"].items():
        category_label, bucket_name = key.split("\t", 1)
        heap = merged_heaps[(category_label, bucket_name)]
        for item in items:
            priority = float(item["priority"])
            record = item["record"]
            if len(heap) < samples_per_bucket:
                heapq.heappush(heap, (priority, record))
            elif priority > heap[0][0]:
                heapq.heapreplace(heap, (priority, record))


def stream_shards_until_targets_full(
    *,
    shard_urls: list[str],
    quality_threshold: int,
    samples_per_bucket: int,
    seed: int,
    target_labels: list[str],
) -> tuple[
    dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    list[dict[str, Any]],
]:
    merged_heaps: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    tuple_counts_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    main_counts_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    shard_summaries: list[dict[str, Any]] = []

    stop_all = False
    for url in shard_urls:
        shard_name = Path(urlparse(url).path).name
        qbin = quality_bin_from_shard_name(shard_name)
        bucket = quality_bucket(qbin, quality_threshold)
        cmd = f"curl -L --silent {shlex.quote(url)} | zstd -dc"
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert proc.stdout is not None
        rows_seen = 0
        try:
            for line in proc.stdout:
                if not line.strip():
                    continue
                rows_seen += 1
                row = json.loads(line)
                text = row.get("text") or ""
                if not text.strip():
                    continue

                (
                    category_label,
                    canonical_main,
                    canonical_sub,
                    top_main_code,
                    top_sub_code,
                    top_main_score,
                    canonical_sub_code,
                    top_sub_score,
                ) = category_record(row)
                main_counts_by_bucket[bucket][canonical_main] += 1
                tuple_counts_by_bucket[bucket][category_label] += 1

                record = {
                    "id": row.get("id"),
                    "url": row.get("u"),
                    "host": urlparse(row.get("u") or "").netloc or None,
                    "content_type": row.get("c"),
                    "crawl_id": row.get("crawl_id"),
                    "timestamp": row.get("ts"),
                    "cluster_size": row.get("cluster_size"),
                    "filter": row.get("filter"),
                    "quality_bin": str(qbin),
                    "quality_bucket": bucket,
                    "shard": shard_name,
                    "char_count": len(text),
                    "top_main_label_code": top_main_code,
                    "top_main_label": label_name(top_main_code) if top_main_code else None,
                    "top_main_score": top_main_score,
                    "top_sub_label_code": top_sub_code,
                    "top_sub_label": label_name(top_sub_code) if top_sub_code else None,
                    "top_sub_score": top_sub_score,
                    "canonical_main_label": canonical_main,
                    "canonical_sub_label_code": canonical_sub_code,
                    "canonical_sub_label": canonical_sub,
                    "category_tuple": [canonical_main, canonical_sub],
                    "category_tuple_label": category_label,
                    "text": text,
                }
                prio = priority_for(seed, shard_name, row.get("id"), rows_seen)
                push_candidate(merged_heaps, category_label, bucket, prio, record, samples_per_bucket)

                if targets_full(
                    target_labels=target_labels,
                    merged_heaps=merged_heaps,
                    quality_threshold=quality_threshold,
                    samples_per_bucket=samples_per_bucket,
                ):
                    stop_all = True
                    break
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()

        shard_summaries.append(
            {
                "shard": shard_name,
                "quality_bin": qbin,
                "rows_seen": rows_seen,
            }
        )
        print(
            json.dumps(
                {
                    "event": "shard_complete",
                    "shard": shard_name,
                    "quality_bin": qbin,
                    "rows_seen": rows_seen,
                    "quota_stop": stop_all,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stop_all:
            break

    return merged_heaps, tuple_counts_by_bucket, main_counts_by_bucket, shard_summaries


def plot_horizontal_comparison(
    labels: list[str],
    left_values: list[int],
    right_values: list[int],
    left_label: str,
    right_label: str,
    title: str,
    output_path: Path,
) -> None:
    y = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(13, max(5, 0.38 * len(labels) + 1.5)))
    plt.barh([i - width / 2 for i in y], left_values, height=width, color="#2f6db3", label=left_label)
    plt.barh([i + width / 2 for i in y], right_values, height=width, color="#d48b1f", label=right_label)
    plt.yticks(y, labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Documents")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown_sample(path: Path, rows: list[dict[str, Any]], threshold: int) -> None:
    bucket = rows[0]["quality_bucket"] if rows else "unknown"
    lines = [
        "# HPLT Full-Corpus Register Sample",
        "",
        f"- Quality bucket: `{bucket}`",
        f"- Quality threshold: `{threshold}`",
        f"- Records: `{len(rows)}`",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index}. {row['category_tuple_label']}",
                "",
                f"- id: `{row.get('id')}`",
                f"- quality bin: `{row.get('quality_bin')}`",
                f"- host: `{row.get('host')}`",
                f"- url: `{row.get('url')}`",
                f"- content type: `{row.get('content_type')}`",
                f"- cluster size: `{row.get('cluster_size')}`",
                f"- top main label: `{row.get('top_main_label')}` ({row.get('top_main_score')})",
                f"- top sub label: `{row.get('top_sub_label')}` ({row.get('top_sub_score')})",
                "",
                row.get("text", "")[:1200],
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_txt_docs(base_dir: Path, rows: list[dict[str, Any]]) -> None:
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        doc_id = row.get("id") or f"row_{index:03d}"
        filename = f"{index:03d}_{doc_id}.txt"
        (base_dir / filename).write_text(row.get("text") or "", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    base_url: str,
    shard_urls: list[str],
    quality_threshold: int,
    samples_per_bucket: int,
    merged_heaps: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
    tuple_counts_by_bucket: dict[str, Counter[str]],
    main_counts_by_bucket: dict[str, Counter[str]],
    shard_summaries: list[dict[str, Any]],
) -> None:
    left_bucket = f"ge{quality_threshold}"
    right_bucket = f"lt{quality_threshold}"
    tuple_labels = sorted(
        set(tuple_counts_by_bucket[left_bucket]) | set(tuple_counts_by_bucket[right_bucket]),
        key=lambda label: (-(tuple_counts_by_bucket[left_bucket][label] + tuple_counts_by_bucket[right_bucket][label]), label),
    )
    main_labels = sorted(
        set(main_counts_by_bucket[left_bucket]) | set(main_counts_by_bucket[right_bucket]),
        key=lambda label: (-(main_counts_by_bucket[left_bucket][label] + main_counts_by_bucket[right_bucket][label]), label),
    )

    plot_horizontal_comparison(
        labels=main_labels,
        left_values=[main_counts_by_bucket[left_bucket].get(label, 0) for label in main_labels],
        right_values=[main_counts_by_bucket[right_bucket].get(label, 0) for label in main_labels],
        left_label=f"Quality >= {quality_threshold}",
        right_label=f"Quality < {quality_threshold}",
        title="Canonical First-Level Register Distribution (Full Greek HPLT)",
        output_path=output_dir / "first_level_distribution_canonical_ge_vs_lt.png",
    )
    plot_horizontal_comparison(
        labels=tuple_labels,
        left_values=[tuple_counts_by_bucket[left_bucket].get(label, 0) for label in tuple_labels],
        right_values=[tuple_counts_by_bucket[right_bucket].get(label, 0) for label in tuple_labels],
        left_label=f"Quality >= {quality_threshold}",
        right_label=f"Quality < {quality_threshold}",
        title="Canonical Tuple Register Distribution (Full Greek HPLT)",
        output_path=output_dir / "tuple_distribution_canonical_ge_vs_lt.png",
    )

    samples_root = output_dir / "category_samples"
    summary_records: list[dict[str, Any]] = []

    for tuple_label in tuple_labels:
        category_dir = samples_root / safe_slug(tuple_label)
        category_dir.mkdir(parents=True, exist_ok=True)
        for bucket in (left_bucket, right_bucket):
            heap = merged_heaps.get((tuple_label, bucket), [])
            rows = [record for _, record in sorted(heap, key=lambda item: (-item[0], item[1].get("id") or ""))]
            metadata_rows = []
            for row in rows:
                normalized = dict(row)
                normalized.pop("text", None)
                metadata_rows.append(normalized)
            write_jsonl(category_dir / f"{bucket}.jsonl", metadata_rows)
            write_markdown_sample(category_dir / f"{bucket}.md", rows, quality_threshold)
            write_txt_docs(category_dir / f"{bucket}_txt", rows)
            summary_records.append(
                {
                    "category_tuple_label": tuple_label,
                    "bucket": bucket,
                    "available": tuple_counts_by_bucket[bucket].get(tuple_label, 0),
                    "selected": len(rows),
                    "target": samples_per_bucket,
                    "jsonl": str(category_dir / f"{bucket}.jsonl"),
                    "markdown": str(category_dir / f"{bucket}.md"),
                    "txt_dir": str(category_dir / f"{bucket}_txt"),
                }
            )

    summary = {
        "base_url": base_url,
        "shard_urls": shard_urls,
        "quality_threshold": quality_threshold,
        "samples_per_bucket": samples_per_bucket,
        "canonical_first_level_counts": {
            left_bucket: dict(main_counts_by_bucket[left_bucket].most_common()),
            right_bucket: dict(main_counts_by_bucket[right_bucket].most_common()),
        },
        "canonical_tuple_counts": {
            left_bucket: dict(tuple_counts_by_bucket[left_bucket].most_common()),
            right_bucket: dict(tuple_counts_by_bucket[right_bucket].most_common()),
        },
        "sample_inventory": summary_records,
        "shard_summaries": sorted(shard_summaries, key=lambda item: (item["quality_bin"], item["shard"])),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Full Greek HPLT Register Tuple Sampling",
        "",
        f"- Base URL: `{base_url}`",
        f"- Quality threshold: `{quality_threshold}`",
        f"- Samples per bucket: `{samples_per_bucket}`",
        f"- Shards processed so far: `{len(shard_summaries)}` / `{len(shard_urls)}`",
        "",
        "## Canonical First-Level Counts",
        "",
        f"### Quality >= `{quality_threshold}`",
    ]
    for label, count in main_counts_by_bucket[left_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", f"### Quality < `{quality_threshold}`"])
    for label, count in main_counts_by_bucket[right_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Canonical Tuple Counts", "", f"### Quality >= `{quality_threshold}`"])
    for label, count in tuple_counts_by_bucket[left_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", f"### Quality < `{quality_threshold}`"])
    for label, count in tuple_counts_by_bucket[right_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Sample Inventory"])
    for item in summary_records:
        lines.append(
            f"- `{item['category_tuple_label']}` / `{item['bucket']}`: selected `{item['selected']}` of `{item['available']}` available"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global args
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    only_shards = list(args.only_shards) if args.only_shards else None
    shard_urls = fetch_shard_urls(args.base_url, only_shards=only_shards)
    target_labels = load_target_category_labels(args.target_category_labels_file)
    if args.stop_when_targets_full and target_labels:
        (
            merged_heaps,
            tuple_counts_by_bucket,
            main_counts_by_bucket,
            shard_summaries,
        ) = stream_shards_until_targets_full(
            shard_urls=shard_urls,
            quality_threshold=args.quality_threshold,
            samples_per_bucket=args.samples_per_bucket,
            seed=args.seed,
            target_labels=target_labels,
        )
    else:
        merged_heaps = defaultdict(list)
        tuple_counts_by_bucket = defaultdict(Counter)
        main_counts_by_bucket = defaultdict(Counter)
        shard_summaries = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(stream_shard, url, args.samples_per_bucket, args.quality_threshold, args.seed)
                for url in shard_urls
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                merge_one_result(
                    result=result,
                    merged_heaps=merged_heaps,
                    tuple_counts_by_bucket=tuple_counts_by_bucket,
                    main_counts_by_bucket=main_counts_by_bucket,
                    shard_summaries=shard_summaries,
                    samples_per_bucket=args.samples_per_bucket,
                    quality_threshold=args.quality_threshold,
                )
                print(
                    json.dumps(
                        {
                            "event": "shard_complete",
                            "shard": result["shard"],
                            "quality_bin": result["quality_bin"],
                            "rows_seen": result["rows_seen"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.checkpoint_after_each_shard:
                    write_outputs(
                        output_dir=args.output_dir,
                        base_url=args.base_url,
                        shard_urls=shard_urls,
                        quality_threshold=args.quality_threshold,
                        samples_per_bucket=args.samples_per_bucket,
                        merged_heaps=merged_heaps,
                        tuple_counts_by_bucket=tuple_counts_by_bucket,
                        main_counts_by_bucket=main_counts_by_bucket,
                        shard_summaries=shard_summaries,
                    )

    write_outputs(
        output_dir=args.output_dir,
        base_url=args.base_url,
        shard_urls=shard_urls,
        quality_threshold=args.quality_threshold,
        samples_per_bucket=args.samples_per_bucket,
        merged_heaps=merged_heaps,
        tuple_counts_by_bucket=tuple_counts_by_bucket,
        main_counts_by_bucket=main_counts_by_bucket,
        shard_summaries=shard_summaries,
    )

    print(
        json.dumps(
            {
                "summary_json": str(args.output_dir / "summary.json"),
                "summary_md": str(args.output_dir / "summary.md"),
                "samples_root": str(args.output_dir / "category_samples"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
