#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from hplt_web_register import category_tuple_codes_from_sub_label, category_tuple_labels_from_sub_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze HPLT sample rows using canonical hierarchical register tuples and emit quality-bucket samples."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Input HPLT sample JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--samples-per-bucket", type=int, default=40, help="Target random sample size per category and quality bucket.")
    parser.add_argument("--quality-threshold", type=int, default=8, help="Threshold separating >=N from <N buckets.")
    parser.add_argument("--seed", type=int, default=20260410, help="Random seed.")
    return parser.parse_args()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return slug.strip("._-") or "category"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def quality_bucket(quality_bin: int, threshold: int) -> str:
    return f"ge{threshold}" if quality_bin >= threshold else f"lt{threshold}"


def add_tuple_fields(rows: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        raw_sub_code = row.get("top_sub_label_code")
        main_code, sub_code = category_tuple_codes_from_sub_label(raw_sub_code)
        main_label, sub_label = category_tuple_labels_from_sub_label(raw_sub_code)
        record = dict(row)
        record["canonical_main_label_code"] = main_code
        record["canonical_sub_label_code"] = sub_code
        record["canonical_main_label"] = main_label
        record["canonical_sub_label"] = sub_label
        record["category_tuple"] = [main_label, sub_label]
        record["category_tuple_label"] = f"{main_label} | {sub_label}" if main_label and sub_label else "Null group"
        record["quality_bucket"] = quality_bucket(int(record["quality_bin"]), threshold)
        enriched.append(record)
    return enriched


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: value for key, value in counter.most_common()}


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
        f"# HPLT Register Tuple Sample",
        "",
        f"- Quality bucket: `{bucket}`",
        f"- Quality threshold: `{threshold}`",
        f"- Records: `{len(rows)}`",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        tuple_label = row["category_tuple_label"]
        lines.extend(
            [
                f"## {index}. {tuple_label}",
                "",
                f"- id: `{row.get('id')}`",
                f"- quality bin: `{row.get('quality_bin')}`",
                f"- host: `{row.get('host')}`",
                f"- url: `{row.get('url')}`",
                f"- content type: `{row.get('content_type')}`",
                f"- cluster size: `{row.get('cluster_size')}`",
                "",
                row.get("excerpt") or "",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = add_tuple_fields(load_rows(args.input_jsonl), args.quality_threshold)

    tuple_mismatch_count = sum(
        1
        for row in rows
        if row.get("top_main_label_code") != row.get("canonical_main_label_code")
    )
    null_group_count = sum(1 for row in rows if row["category_tuple_label"] == "Null group")

    main_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    tuple_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        bucket = row["quality_bucket"]
        main_by_bucket[bucket][row["canonical_main_label"] or "Null group"] += 1
        tuple_by_bucket[bucket][row["category_tuple_label"]] += 1
        grouped_rows[(row["category_tuple_label"], bucket)].append(row)

    left_bucket = f"ge{args.quality_threshold}"
    right_bucket = f"lt{args.quality_threshold}"

    main_labels = sorted(
        set(main_by_bucket[left_bucket]) | set(main_by_bucket[right_bucket]),
        key=lambda label: (-(main_by_bucket[left_bucket][label] + main_by_bucket[right_bucket][label]), label),
    )
    tuple_labels = sorted(
        set(tuple_by_bucket[left_bucket]) | set(tuple_by_bucket[right_bucket]),
        key=lambda label: (-(tuple_by_bucket[left_bucket][label] + tuple_by_bucket[right_bucket][label]), label),
    )

    plot_horizontal_comparison(
        labels=main_labels,
        left_values=[main_by_bucket[left_bucket].get(label, 0) for label in main_labels],
        right_values=[main_by_bucket[right_bucket].get(label, 0) for label in main_labels],
        left_label=f"Quality >= {args.quality_threshold}",
        right_label=f"Quality < {args.quality_threshold}",
        title="Canonical First-Level Register Distribution",
        output_path=args.output_dir / "first_level_distribution_canonical_ge_vs_lt.png",
    )
    plot_horizontal_comparison(
        labels=tuple_labels,
        left_values=[tuple_by_bucket[left_bucket].get(label, 0) for label in tuple_labels],
        right_values=[tuple_by_bucket[right_bucket].get(label, 0) for label in tuple_labels],
        left_label=f"Quality >= {args.quality_threshold}",
        right_label=f"Quality < {args.quality_threshold}",
        title="Canonical Tuple Register Distribution",
        output_path=args.output_dir / "tuple_distribution_canonical_ge_vs_lt.png",
    )

    rng = random.Random(args.seed)
    samples_root = args.output_dir / "category_samples"
    summary_records: list[dict[str, Any]] = []

    for tuple_label in tuple_labels:
        category_slug = safe_slug(tuple_label)
        category_dir = samples_root / category_slug
        category_dir.mkdir(parents=True, exist_ok=True)
        for bucket in (left_bucket, right_bucket):
            candidates = list(grouped_rows.get((tuple_label, bucket), []))
            rng.shuffle(candidates)
            chosen = candidates[: args.samples_per_bucket]
            normalized_rows = []
            for row in chosen:
                normalized = dict(row)
                normalized.pop("web_register", None)
                normalized_rows.append(normalized)
            write_jsonl(category_dir / f"{bucket}.jsonl", normalized_rows)
            write_markdown_sample(category_dir / f"{bucket}.md", normalized_rows, args.quality_threshold)
            summary_records.append(
                {
                    "category_tuple_label": tuple_label,
                    "bucket": bucket,
                    "available": len(candidates),
                    "selected": len(normalized_rows),
                    "target": args.samples_per_bucket,
                    "jsonl": str(category_dir / f"{bucket}.jsonl"),
                    "markdown": str(category_dir / f"{bucket}.md"),
                }
            )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "quality_threshold": args.quality_threshold,
        "rows": len(rows),
        "tuple_mismatch_count": tuple_mismatch_count,
        "tuple_mismatch_share": tuple_mismatch_count / len(rows) if rows else 0.0,
        "null_group_count": null_group_count,
        "canonical_first_level_counts": {
            left_bucket: counter_to_dict(main_by_bucket[left_bucket]),
            right_bucket: counter_to_dict(main_by_bucket[right_bucket]),
        },
        "canonical_tuple_counts": {
            left_bucket: counter_to_dict(tuple_by_bucket[left_bucket]),
            right_bucket: counter_to_dict(tuple_by_bucket[right_bucket]),
        },
        "sample_inventory": summary_records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Canonical HPLT Register Tuple Analysis",
        "",
        f"- Input rows: `{len(rows)}`",
        f"- Quality threshold: `{args.quality_threshold}`",
        f"- Docs whose independent top-main label disagreed with the second-level parent: `{tuple_mismatch_count}`",
        f"- Disagreement share: `{100.0 * summary['tuple_mismatch_share']:.2f}%`",
        f"- Null-group docs: `{null_group_count}`",
        "",
        "## Canonical First-Level Counts",
        "",
        f"### Quality >= `{args.quality_threshold}`",
    ]
    for label, count in main_by_bucket[left_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", f"### Quality < `{args.quality_threshold}`"])
    for label, count in main_by_bucket[right_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Canonical Tuple Counts", "", f"### Quality >= `{args.quality_threshold}`"])
    for label, count in tuple_by_bucket[left_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", f"### Quality < `{args.quality_threshold}`"])
    for label, count in tuple_by_bucket[right_bucket].most_common():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Sample Inventory"])
    for item in summary_records:
        lines.append(
            f"- `{item['category_tuple_label']}` / `{item['bucket']}`: selected `{item['selected']}` of `{item['available']}` available"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_json": str(args.output_dir / "summary.json"),
                "summary_md": str(args.output_dir / "summary.md"),
                "samples_root": str(samples_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
