#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HPLT quality-bin distribution and >=8 filter skew.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Review-sample JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--quality-threshold", type=int, default=8, help="Minimum quality bin to keep.")
    parser.add_argument(
        "--second-level-top-k",
        type=int,
        default=12,
        help="How many second-level labels to show in the comparison plot.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def counter_to_sorted_dict(counter: Counter, sort_numeric: bool = False) -> dict:
    if sort_numeric:
        items = sorted(counter.items(), key=lambda item: int(item[0]))
    else:
        items = counter.most_common()
    return {key: value for key, value in items}


def percentage(counter: Counter, total: int, keys: list[str]) -> list[float]:
    if total <= 0:
        return [0.0 for _ in keys]
    return [100.0 * counter.get(key, 0) / total for key in keys]


def plot_quality_bins(counter: Counter, output_path: Path) -> None:
    labels = [str(i) for i in range(5, 11)]
    values = [counter.get(label, 0) for label in labels]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color="#2f6db3")
    plt.xlabel("HPLT quality bin")
    plt.ylabel("Documents")
    plt.title("HPLT Review Sample: Quality-Bin Distribution")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.3, str(value), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_comparison(
    title: str,
    all_counter: Counter,
    filtered_counter: Counter,
    all_total: int,
    filtered_total: int,
    labels: list[str],
    threshold: int,
    output_path: Path,
) -> None:
    all_pct = percentage(all_counter, all_total, labels)
    filtered_pct = percentage(filtered_counter, filtered_total, labels)
    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(12, max(4.5, 0.35 * len(labels) + 1.5)))
    plt.barh([i - width / 2 for i in x], all_pct, height=width, label="All review docs", color="#9ebcda")
    plt.barh([i + width / 2 for i in x], filtered_pct, height=width, label=f"Only quality >= {threshold}", color="#2f6db3")
    plt.yticks(x, labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Percent of documents")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_markdown(summary: dict, output_path: Path, threshold: int) -> None:
    lines = [
        "# HPLT Quality Filter Analysis",
        "",
        f"- Input sample size: `{summary['total_docs']}`",
        f"- Docs with quality >= `{threshold}`: `{summary['kept_docs']}`",
        f"- Share with quality >= `{threshold}`: `{summary['kept_share_percent']:.2f}%`",
        "",
        "## Quality Bin Counts",
    ]
    for label, count in summary["quality_bin_counts"].items():
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(["", f"## First-Level Labels After Quality >= `{threshold}`"])
    for label, count in summary["kept_main_label_counts"].items():
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(["", f"## Second-Level Labels After Quality >= `{threshold}`"])
    for label, count in summary["kept_second_level_counts"].items():
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(["", "## Top First-Level Shifts"])
    for item in summary["main_label_shift_points"]:
        lines.append(
            f"- `{item['label']}`: `{item['all_percent']:.2f}%` -> `{item['kept_percent']:.2f}%` "
            f"(`{item['delta_points']:+.2f}` pts)"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_shift_points(all_counter: Counter, kept_counter: Counter, all_total: int, kept_total: int) -> list[dict]:
    labels = sorted(set(all_counter) | set(kept_counter))
    items: list[dict] = []
    for label in labels:
        all_pct = 100.0 * all_counter.get(label, 0) / all_total if all_total else 0.0
        kept_pct = 100.0 * kept_counter.get(label, 0) / kept_total if kept_total else 0.0
        items.append(
            {
                "label": label,
                "all_percent": all_pct,
                "kept_percent": kept_pct,
                "delta_points": kept_pct - all_pct,
            }
        )
    items.sort(key=lambda item: (-abs(item["delta_points"]), item["label"]))
    return items


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input_jsonl)
    kept = [row for row in rows if int(row["quality_bin"]) >= args.quality_threshold]

    quality_bin_counts = Counter(str(row["quality_bin"]) for row in rows)
    all_main = Counter(row["top_main_label"] or "Unlabeled" for row in rows)
    kept_main = Counter(row["top_main_label"] or "Unlabeled" for row in kept)
    all_second = Counter(row["top_second_level_label"] or "Unlabeled" for row in rows)
    kept_second = Counter(row["top_second_level_label"] or "Unlabeled" for row in kept)

    # Use the most common labels across both populations to keep the chart readable.
    second_level_keys = []
    for label, _ in (all_second + kept_second).most_common(args.second_level_top_k):
        if label not in second_level_keys:
            second_level_keys.append(label)

    plot_quality_bins(quality_bin_counts, args.output_dir / "quality_bin_distribution.png")
    plot_comparison(
        "First-Level Label Distribution: All vs Quality >= 8",
        all_main,
        kept_main,
        len(rows),
        len(kept),
        list(all_main.keys()),
        args.quality_threshold,
        args.output_dir / "first_level_distribution_all_vs_ge8.png",
    )
    plot_comparison(
        "Second-Level Label Distribution: All vs Quality >= 8",
        all_second,
        kept_second,
        len(rows),
        len(kept),
        second_level_keys,
        args.quality_threshold,
        args.output_dir / "second_level_distribution_all_vs_ge8.png",
    )

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "quality_threshold": args.quality_threshold,
        "total_docs": len(rows),
        "kept_docs": len(kept),
        "kept_share_percent": (100.0 * len(kept) / len(rows)) if rows else 0.0,
        "quality_bin_counts": counter_to_sorted_dict(quality_bin_counts, sort_numeric=True),
        "all_main_label_counts": counter_to_sorted_dict(all_main),
        "kept_main_label_counts": counter_to_sorted_dict(kept_main),
        "all_second_level_counts": counter_to_sorted_dict(all_second),
        "kept_second_level_counts": counter_to_sorted_dict(kept_second),
        "main_label_shift_points": build_shift_points(all_main, kept_main, len(rows), len(kept)),
    }

    (args.output_dir / "quality_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, args.output_dir / "quality_filter_summary.md", args.quality_threshold)
