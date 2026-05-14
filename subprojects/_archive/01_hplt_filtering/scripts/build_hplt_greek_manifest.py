#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight manifest from an HPLT Greek .jsonl.zst shard.")
    parser.add_argument("--input", required=True, help="Local .jsonl/.jsonl.zst path or URL.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL manifest path.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Output summary JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    parser.add_argument("--include-web-register", action="store_true", help="Include raw web-register scores in each manifest row.")
    return parser.parse_args()


def build_stream_command(input_value: str) -> str:
    quoted = shlex.quote(input_value)
    if input_value.startswith(("http://", "https://")):
        return f"curl -L --silent {quoted} | zstd -dc"
    if input_value.endswith(".zst"):
        return f"zstd -dc {quoted}"
    return f"cat {quoted}"


def normalize_html_lang(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def infer_shard(input_value: str) -> str:
    parsed = urlparse(input_value)
    if parsed.scheme and parsed.netloc:
        return Path(parsed.path).name
    return Path(input_value).name


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_stream_command(args.input)
    proc = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None

    shard_name = infer_shard(args.input)
    rows = 0
    host_counts: Counter[str] = Counter()
    content_type_counts: Counter[str | None] = Counter()
    filter_counts: Counter[str | None] = Counter()
    with args.output_jsonl.open("w", encoding="utf-8") as out_handle:
        try:
            for line in proc.stdout:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1

                url = row.get("u")
                host = urlparse(url).netloc if url else None
                content_type = row.get("c")
                filter_value = row.get("filter")

                host_counts[host] += 1
                content_type_counts[content_type] += 1
                filter_counts[filter_value] += 1

                manifest_row = {
                    "source": "hplt_v3_sorted_ell_Grek",
                    "shard": shard_name,
                    "row_index": rows,
                    "id": row.get("id"),
                    "url": url,
                    "host": host,
                    "content_type": content_type,
                    "timestamp": row.get("ts"),
                    "crawl_id": row.get("crawl_id"),
                    "filter": filter_value,
                    "cluster_size": row.get("cluster_size"),
                    "html_lang": normalize_html_lang(row.get("html_lang")),
                    "lang": row.get("lang"),
                    "prob": row.get("prob"),
                    "char_count": len(row.get("text") or ""),
                    "byte_count": len((row.get("text") or "").encode("utf-8")),
                    "seg_langs_count": len(row.get("seg_langs") or []),
                    "doc_scores_count": len(row.get("doc_scores") or []),
                }
                if args.include_web_register:
                    manifest_row["web_register"] = row.get("web-register") or {}

                out_handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

                if args.limit is not None and rows >= args.limit:
                    break
        finally:
            proc.kill()
            proc.wait()

    summary = {
        "input": args.input,
        "shard": shard_name,
        "rows_written": rows,
        "content_type_counts": content_type_counts.most_common(20),
        "filter_counts": filter_counts.most_common(20),
        "host_top_20": host_counts.most_common(20),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "summary_json": str(args.summary_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
