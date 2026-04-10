#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import requests

from hplt_web_register import MAIN_LABELS, label_name, top_main_label, top_sub_label


MANIFEST_URL = "https://data.hplt-project.org/three/sorted/manifest.json"


CANONICAL_SECOND_LEVEL = {
    "it": "Interview",
    "os": "Other spoken",
    "ne": "News report",
    "sr": "Sports report",
    "nb": "Narrative blog",
    "on": "Other narrative",
    "re": "Recipe",
    "oh": "Other how-to / instructional",
    "en": "Encyclopedia article",
    "ra": "Research article",
    "dtp": "Description of a thing or person",
    "fi": "FAQ about information",
    "lt": "Legal terms and conditions",
    "oi": "Other informational description",
    "rv": "Review",
    "ob": "Opinion blog",
    "rs": "Denominational religious blog / sermon",
    "av": "Advice",
    "oo": "Other opinion",
    "ds": "Description with intent to sell",
    "ed": "News and opinion blog or editorial",
    "oe": "Other informational persuasion",
    "df": "Discussion forum",
    "of": "Other forum",
    "qa": "Question / answer forum",
    "rr": "Reader / viewer responses",
    "ol": "Other lyrical",
    "po": "Poem",
    "pr": "Prayer",
    "sl": "Song lyrics",
}


# Collapse the overlapping mixed HPLT schema into a non-overlapping second level.
SECOND_LEVEL_COLLAPSE = {
    "it": "it",
    "fs": "it",
    "ta": "os",
    "tv": "os",
    "os": "os",
    "ne": "ne",
    "sr": "sr",
    "nb": "nb",
    "pb": "nb",
    "tb": "nb",
    "on": "on",
    "ha": "on",
    "ma": "on",
    "re": "re",
    "fh": "oh",
    "ht": "oh",
    "ts": "oh",
    "oh": "oh",
    "en": "en",
    "ra": "ra",
    "dtp": "dtp",
    "dp": "dtp",
    "dt": "dtp",
    "fi": "fi",
    "lt": "lt",
    "oi": "oi",
    "cm": "oi",
    "ib": "oi",
    "tr": "oi",
    "rv": "rv",
    "ob": "ob",
    "rs": "rs",
    "av": "av",
    "oo": "oo",
    "ds": "ds",
    "ed": "ed",
    "oe": "oe",
    "ad": "oe",
    "le": "oe",
    "pa": "oe",
    "df": "df",
    "of": "of",
    "qa": "qa",
    "rr": "rr",
    "ol": "ol",
    "po": "po",
    "pr": "pr",
    "sl": "sl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fast approximate 200-doc HPLT Greek review bundle.")
    parser.add_argument("--language", default="ell_Grek")
    parser.add_argument("--per-shard-limit", type=int, default=3000)
    parser.add_argument("--review-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--max-per-host", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_manifest_entry(language: str) -> dict[str, Any]:
    text = requests.get(MANIFEST_URL, timeout=60).text.splitlines()
    for line in text:
        if f'"name": "{language}"' in line:
            return json.loads(line)
    raise ValueError(f"Language {language!r} not found in manifest")


def compact_excerpt(text: str, limit: int = 1200) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def clean_filename(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = slug.strip("._-")
    return slug or "sample"


def canonical_second_level(raw_code: str | None) -> str | None:
    if raw_code is None:
        return None
    return SECOND_LEVEL_COLLAPSE.get(raw_code)


def stream_prefix_rows(url: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cmd = f"curl -L --silent {shlex.quote(url)} | zstd -T0 -dc"
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    try:
        for idx, line in enumerate(proc.stdout, start=1):
            if line.strip():
                rows.append(json.loads(line))
            if idx >= limit:
                break
    finally:
        proc.kill()
        proc.wait()
    return rows


def build_record(row: dict[str, Any], shard_name: str) -> dict[str, Any]:
    url = row.get("u")
    host = urlparse(url).netloc if url else None
    web_register = row.get("web-register") or {}
    main_code, main_score = top_main_label(web_register)
    raw_sub_code, raw_sub_score = top_sub_label(web_register)
    canonical_sub_code = canonical_second_level(raw_sub_code)
    canonical_sub_name = CANONICAL_SECOND_LEVEL.get(canonical_sub_code, label_name(raw_sub_code) if raw_sub_code else None)
    text = row.get("text") or ""
    return {
        "id": row.get("id"),
        "url": url,
        "host": host,
        "content_type": row.get("c"),
        "crawl_id": row.get("crawl_id"),
        "timestamp": row.get("ts"),
        "cluster_size": row.get("cluster_size"),
        "filter": row.get("filter"),
        "shard": shard_name,
        "quality_bin": shard_name.split("_", 1)[0],
        "char_count": len(text),
        "top_main_label_code": main_code,
        "top_main_label": label_name(main_code) if main_code else None,
        "top_main_score": main_score,
        "raw_top_sub_label_code": raw_sub_code,
        "raw_top_sub_label": label_name(raw_sub_code) if raw_sub_code else None,
        "raw_top_sub_score": raw_sub_score,
        "top_second_level_code": canonical_sub_code,
        "top_second_level_label": canonical_sub_name,
        "web_register": web_register,
        "text": text,
        "excerpt": compact_excerpt(text),
    }


def collect_shard_records(url: str, limit: int) -> list[dict[str, Any]]:
    shard_name = Path(urlparse(url).path).name
    rows = stream_prefix_rows(url, limit)
    return [build_record(row, shard_name) for row in rows if row.get("filter") == "keep" and (row.get("text") or "").strip()]


def largest_remainder_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        return {}
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    remaining = target
    for label, count in counts.items():
        raw = target * (count / total)
        base = int(raw)
        quotas[label] = base
        remainders.append((raw - base, label))
        remaining -= base
    for _, label in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        quotas[label] += 1
        remaining -= 1
    return quotas


def select_review_sample(records: list[dict[str, Any]], review_size: int, seed: int, max_per_host: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    main_counts = Counter(record["top_main_label_code"] or "UNLABELED" for record in pool)
    quotas = largest_remainder_quotas(main_counts, review_size)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pool:
        key = record["top_main_label_code"] or "UNLABELED"
        buckets[key].append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    host_counts: Counter[str] = Counter()

    def try_take(record: dict[str, Any]) -> bool:
        rid = str(record["id"])
        host = record.get("host") or "__missing_host__"
        if rid in selected_ids:
            return False
        if host_counts[host] >= max_per_host:
            return False
        selected.append(record)
        selected_ids.add(rid)
        host_counts[host] += 1
        return True

    for label, quota in quotas.items():
        for record in buckets.get(label, []):
            if quota <= 0:
                break
            if try_take(record):
                quota -= 1

    for record in pool:
        if len(selected) >= review_size:
            break
        try_take(record)

    return selected[:review_size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "main_label_counts": dict(Counter(row["top_main_label"] or "Unlabeled" for row in records).most_common()),
        "second_level_counts": dict(Counter(row["top_second_level_label"] or "Unlabeled" for row in records).most_common()),
        "top_hosts": Counter(row["host"] or "missing-host" for row in records).most_common(20),
        "quality_bin_counts": dict(sorted(Counter(row["quality_bin"] for row in records).items())),
    }


def plot_distribution(counter: Counter[str], title: str, output_path: Path) -> None:
    items = counter.most_common()
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    fig_height = max(4, 0.35 * len(labels) + 1.5)
    plt.figure(figsize=(12, fig_height))
    y = list(range(len(labels)))
    plt.barh(y, values, color="#2f6db3")
    plt.yticks(y, labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Documents")
    plt.title(title)
    for idx, value in enumerate(values):
        plt.text(value + 0.3, idx, str(value), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_txt_samples(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        main_slug = clean_filename(record["top_main_label"] or "unlabeled")
        doc_id = clean_filename(str(record["id"])[:40])
        path = output_dir / f"{index:03d}_{main_slug}_{doc_id}.txt"
        header = [
            f"id: {record['id']}",
            f"url: {record.get('url')}",
            f"host: {record.get('host')}",
            f"quality_bin: {record.get('quality_bin')}",
            f"top_main_label: {record.get('top_main_label')}",
            f"top_second_level_label: {record.get('top_second_level_label')}",
            "",
        ]
        path.write_text("\n".join(header) + (record.get("text") or "") + "\n", encoding="utf-8")


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# HPLT Greek Quick Review Bundle",
        "",
        "This is a fast approximate review sample built from bounded multi-shard prefix windows, not the exact full-pass corpus-wide sample.",
        "",
        f"- Review sample size: `{summary['review_summary']['count']}`",
        f"- Per-shard prefix limit: `{summary['per_shard_limit']}`",
        f"- Host cap: `{summary['max_per_host']}`",
        "",
        "## First-Level Distribution",
    ]
    for label, count in summary["review_summary"]["main_label_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Second-Level Distribution"])
    for label, count in summary["review_summary"]["second_level_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Top Hosts"])
    for host, count in summary["review_summary"]["top_hosts"]:
        lines.append(f"- `{host}`: `{count}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entry = load_manifest_entry(args.language)

    all_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_shard_records, url, args.per_shard_limit) for url in manifest_entry["urls"]]
        for future in as_completed(futures):
            records = future.result()
            all_records.extend(records)
            if records:
                print(json.dumps({"completed_shard": records[0]["shard"], "records": len(records)}, ensure_ascii=False), flush=True)

    review_records = select_review_sample(all_records, args.review_size, args.seed, args.max_per_host)

    review_jsonl = args.output_dir / "review_sample_200.jsonl"
    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    first_level_png = args.output_dir / "first_level_distribution.png"
    second_level_png = args.output_dir / "second_level_distribution.png"
    txt_dir = args.output_dir / "sample_txt"

    write_jsonl(review_jsonl, review_records)
    write_txt_samples(review_records, txt_dir)

    review_summary = summarize(review_records)
    plot_distribution(Counter(review_summary["main_label_counts"]), "HPLT Greek Review Sample: First-Level Labels", first_level_png)
    plot_distribution(Counter(review_summary["second_level_counts"]), "HPLT Greek Review Sample: Second-Level Labels", second_level_png)

    summary = {
        "language": args.language,
        "per_shard_limit": args.per_shard_limit,
        "review_size": args.review_size,
        "max_per_host": args.max_per_host,
        "review_summary": review_summary,
        "review_jsonl": str(review_jsonl),
        "sample_txt_dir": str(txt_dir),
        "first_level_distribution_png": str(first_level_png),
        "second_level_distribution_png": str(second_level_png),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md)
    print(
        json.dumps(
            {
                "review_jsonl": str(review_jsonl),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "first_level_png": str(first_level_png),
                "second_level_png": str(second_level_png),
                "sample_txt_dir": str(txt_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
