#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from hplt_web_register import MAIN_LABELS, label_name, top_main_label, top_sub_label


MANIFEST_URL = "https://data.hplt-project.org/three/sorted/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a representative HPLT Greek review sample.")
    parser.add_argument("--language", default="ell_Grek", help="Language-script code in HPLT manifest.")
    parser.add_argument("--candidate-target", type=int, default=2400, help="Approximate random candidate count.")
    parser.add_argument("--review-size", type=int, default=200, help="Final review sample size.")
    parser.add_argument("--seed", type=int, default=20260409, help="Base random seed.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel shard workers.")
    parser.add_argument("--max-per-host", type=int, default=8, help="Soft host cap for the final review sample.")
    parser.add_argument("--excerpt-chars", type=int, default=1200, help="Excerpt length to save for review.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser.parse_args()


def load_manifest_entry(language: str) -> dict[str, Any]:
    text = requests.get(MANIFEST_URL, timeout=60).text.splitlines()
    for line in text:
        if f'"name": "{language}"' in line:
            return json.loads(line)
    raise ValueError(f"Language {language!r} not found in manifest")


def shard_sampling_probability(total_docs: int, candidate_target: int) -> float:
    return min(1.0, candidate_target / max(total_docs, 1))


def run_shard_sample(url: str, probability: float, seed: int, temp_dir: Path) -> dict[str, Any]:
    shard_name = Path(urlparse(url).path).name
    out_path = temp_dir / f"{shard_name}.sampled.jsonl"
    stats_path = temp_dir / f"{shard_name}.stats.json"
    awk_script = (
        "BEGIN { srand(seed); total=0; kept=0; } "
        "{ total++; if (rand() < p) { print; kept++; } } "
        "END { printf(\"{\\\"total\\\": %d, \\\"kept\\\": %d}\\n\", total, kept) > stats }"
    )
    cmd = (
        f"curl -L --silent {shlex.quote(url)} | zstd -T0 -dc | "
        f"awk -v seed={seed} -v p={probability:.12f} -v stats={shlex.quote(str(stats_path))} "
        f"'{awk_script}' > {shlex.quote(str(out_path))}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return {"shard": shard_name, "url": url, "sampled_path": str(out_path), "total": stats["total"], "kept": stats["kept"]}


def compact_excerpt(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def build_record(row: dict[str, Any], shard_name: str, excerpt_chars: int) -> dict[str, Any]:
    url = row.get("u")
    host = urlparse(url).netloc if url else None
    web_register = row.get("web-register") or {}
    main_code, main_score = top_main_label(web_register)
    sub_code, sub_score = top_sub_label(web_register)
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
        "top_sub_label_code": sub_code,
        "top_sub_label": label_name(sub_code) if sub_code else None,
        "top_sub_score": sub_score,
        "web_register": web_register,
        "excerpt": compact_excerpt(text, excerpt_chars),
    }


def load_candidate_records(sampled_paths: list[dict[str, Any]], excerpt_chars: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in sampled_paths:
        shard_name = str(item["shard"])
        path = Path(str(item["sampled_path"]))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(build_record(row, shard_name, excerpt_chars))
    return records


def largest_remainder_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    total = sum(counts.values())
    labels = [label for label, count in counts.items() if count > 0]
    if total == 0 or not labels:
        return {}
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    remaining = target
    for label in labels:
        raw = target * (counts[label] / total)
        base = math.floor(raw)
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
    host_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()

    def try_take(record: dict[str, Any]) -> bool:
        record_id = str(record["id"])
        host = record.get("host") or "__missing_host__"
        if record_id in selected_ids:
            return False
        if host_counts[host] >= max_per_host:
            return False
        selected.append(record)
        selected_ids.add(record_id)
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

    if len(selected) < review_size:
        for record in pool:
            if len(selected) >= review_size:
                break
            record_id = str(record["id"])
            if record_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record_id)

    return selected[:review_size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    main_counts = Counter(record["top_main_label"] or "Unlabeled" for record in records)
    sub_counts = Counter(record["top_sub_label"] or "Unlabeled" for record in records)
    host_counts = Counter(record["host"] or "missing-host" for record in records)
    quality_counts = Counter(record["quality_bin"] for record in records)
    return {
        "count": len(records),
        "main_label_counts": dict(main_counts.most_common()),
        "sub_label_counts": dict(sub_counts.most_common(20)),
        "top_hosts": host_counts.most_common(20),
        "quality_bin_counts": dict(sorted(quality_counts.items())),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HPLT Greek Review Sample",
        "",
        f"- Language: `{summary['language']}`",
        f"- Total docs in manifest: `{summary['manifest_documents']}`",
        f"- Sampling probability: `{summary['sampling_probability']:.8f}`",
        f"- Candidate sample size: `{summary['candidate_summary']['count']}`",
        f"- Final review size: `{summary['review_summary']['count']}`",
        f"- Host cap in final review sample: `{summary['max_per_host']}`",
        "",
        "## Candidate Main Labels",
    ]
    for label, count in summary["candidate_summary"]["main_label_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Final Review Main Labels"])
    for label, count in summary["review_summary"]["main_label_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Final Review Top Hosts"])
    for host, count in summary["review_summary"]["top_hosts"]:
        lines.append(f"- `{host}`: `{count}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entry = load_manifest_entry(args.language)
    probability = shard_sampling_probability(int(manifest_entry["documents"]), args.candidate_target)

    with tempfile.TemporaryDirectory(prefix="hplt_review_sample_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        sampled_outputs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for index, url in enumerate(manifest_entry["urls"]):
                futures.append(executor.submit(run_shard_sample, url, probability, args.seed + index, temp_dir))
            for future in as_completed(futures):
                result = future.result()
                sampled_outputs.append(result)
                print(
                    json.dumps(
                        {
                            "completed_shard": result["shard"],
                            "streamed_rows": result["total"],
                            "sampled_rows": result["kept"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        sampled_outputs.sort(key=lambda item: str(item["shard"]))
        candidate_records = load_candidate_records(sampled_outputs, args.excerpt_chars)
        print(json.dumps({"candidate_records": len(candidate_records)}, ensure_ascii=False), flush=True)
        if len(candidate_records) < args.review_size:
            raise RuntimeError(
                f"Only sampled {len(candidate_records)} candidates, which is smaller than requested review size {args.review_size}. "
                "Increase --candidate-target and rerun."
            )
        review_records = select_review_sample(candidate_records, args.review_size, args.seed, args.max_per_host)
        print(json.dumps({"review_records": len(review_records)}, ensure_ascii=False), flush=True)

    candidate_path = args.output_dir / "candidate_sample.jsonl"
    review_path = args.output_dir / "review_sample_200.jsonl"
    summary_json_path = args.output_dir / "summary.json"
    summary_md_path = args.output_dir / "summary.md"
    write_jsonl(candidate_path, candidate_records)
    write_jsonl(review_path, review_records)

    summary = {
        "language": args.language,
        "manifest_documents": int(manifest_entry["documents"]),
        "manifest_bytes": int(manifest_entry["bytes"]),
        "sampling_probability": probability,
        "candidate_target": args.candidate_target,
        "review_size": args.review_size,
        "max_per_host": args.max_per_host,
        "workers": args.workers,
        "shard_stats": sampled_outputs,
        "candidate_summary": summarize_records(candidate_records),
        "review_summary": summarize_records(review_records),
        "candidate_jsonl": str(candidate_path),
        "review_jsonl": str(review_path),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary_md_path, summary)
    print(
        json.dumps(
            {
                "candidate_jsonl": str(candidate_path),
                "review_jsonl": str(review_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
