#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


DEFAULT_INPUT = "https://data.hplt-project.org/three/sorted/ell_Grek/10_1.jsonl.zst"
TRACKED_FIELDS = [
    "u",
    "c",
    "ts",
    "crawl_id",
    "lang",
    "prob",
    "cluster_size",
    "seg_langs",
    "filter",
    "doc_scores",
    "web-register",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect HPLT Greek shard metadata and domain concentration.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Local .jsonl/.jsonl.zst path or URL.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of rows to inspect.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where JSON and Markdown summaries will be written.")
    return parser.parse_args()


def build_stream_command(input_value: str) -> str:
    quoted = shlex.quote(input_value)
    if input_value.startswith(("http://", "https://")):
        return f"curl -L --silent {quoted} | zstd -dc"
    if input_value.endswith(".zst"):
        return f"zstd -dc {quoted}"
    return f"cat {quoted}"


def stream_rows(input_value: str) -> Iterator[dict]:
    cmd = build_stream_command(input_value)
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if line.strip():
                yield json.loads(line)
    finally:
        proc.kill()
        proc.wait()


def normalize_html_lang(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def summarize(input_value: str, sample_size: int) -> dict:
    key_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    content_type_counts: Counter[str | None] = Counter()
    html_lang_counts: Counter[str | None] = Counter()
    filter_counts: Counter[str | None] = Counter()
    host_counts: Counter[str] = Counter()
    web_register_key_counts: Counter[str] = Counter()
    cluster_sizes: list[int] = []
    examples: list[dict] = []

    rows = 0
    for row in stream_rows(input_value):
        rows += 1
        key_counts.update(row.keys())
        content_type_counts[row.get("c")] += 1
        html_lang_counts[normalize_html_lang(row.get("html_lang"))] += 1
        filter_counts[row.get("filter")] += 1

        url = row.get("u")
        if url:
            host_counts[urlparse(url).netloc] += 1

        if row.get("cluster_size") is not None:
            cluster_sizes.append(int(row["cluster_size"]))

        web_register = row.get("web-register") or {}
        web_register_key_counts.update(web_register.keys())

        for field in TRACKED_FIELDS:
            if field not in row or row[field] in (None, [], {}, ""):
                missing_counts[field] += 1

        if len(examples) < 5:
            examples.append(
                {
                    "id": row.get("id"),
                    "host": urlparse(url).netloc if url else None,
                    "content_type": row.get("c"),
                    "crawl_id": row.get("crawl_id"),
                    "lang": row.get("lang"),
                    "prob": row.get("prob"),
                    "cluster_size": row.get("cluster_size"),
                    "filter": row.get("filter"),
                }
            )

        if rows >= sample_size:
            break

    cluster_summary = None
    if cluster_sizes:
        sorted_sizes = sorted(cluster_sizes)
        cluster_summary = {
            "min": sorted_sizes[0],
            "median": statistics.median(sorted_sizes),
            "p90": sorted_sizes[int(len(sorted_sizes) * 0.9)],
            "max": sorted_sizes[-1],
        }

    return {
        "input": input_value,
        "sample_size": sample_size,
        "rows_inspected": rows,
        "keys": sorted(key_counts.keys()),
        "missing_counts": dict(sorted(missing_counts.items())),
        "content_type_counts": content_type_counts.most_common(20),
        "html_lang_counts": html_lang_counts.most_common(20),
        "filter_counts": filter_counts.most_common(20),
        "host_top_20": host_counts.most_common(20),
        "web_register_keys": sorted(web_register_key_counts.keys()),
        "cluster_size_summary": cluster_summary,
        "examples": examples,
    }


def write_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# HPLT Greek Metadata Probe",
        "",
        f"- Input: `{summary['input']}`",
        f"- Rows inspected: `{summary['rows_inspected']}`",
        f"- Keys: `{', '.join(summary['keys'])}`",
        "",
        "## Content Types",
    ]
    for value, count in summary["content_type_counts"]:
        lines.append(f"- `{value}`: `{count}`")

    lines.extend(["", "## Filters"])
    for value, count in summary["filter_counts"]:
        lines.append(f"- `{value}`: `{count}`")

    lines.extend(["", "## HTML Lang"])
    for value, count in summary["html_lang_counts"][:10]:
        lines.append(f"- `{value}`: `{count}`")

    lines.extend(["", "## Top Hosts"])
    for value, count in summary["host_top_20"]:
        lines.append(f"- `{value}`: `{count}`")

    lines.extend(["", "## Cluster Size"])
    cluster = summary["cluster_size_summary"]
    if cluster is None:
        lines.append("- No cluster sizes found")
    else:
        lines.append(
            f"- min `{cluster['min']}`, median `{cluster['median']}`, p90 `{cluster['p90']}`, max `{cluster['max']}`"
        )

    lines.extend(["", "## Missing Counts"])
    if summary["missing_counts"]:
        for key, count in summary["missing_counts"].items():
            lines.append(f"- `{key}`: `{count}`")
    else:
        lines.append("- None across tracked fields")

    lines.extend(["", "## Web Register Keys"])
    lines.append(f"- `{', '.join(summary['web_register_keys'])}`")
    lines.extend(["", "## Examples", "```json", json.dumps(summary["examples"], ensure_ascii=False, indent=2), "```"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.input, args.sample_size)

    json_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
