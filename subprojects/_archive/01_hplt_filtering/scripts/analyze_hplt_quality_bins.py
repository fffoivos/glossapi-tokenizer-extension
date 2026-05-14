#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import requests


MANIFEST_URL = "https://data.hplt-project.org/three/sorted/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HPLT quality-bin distribution for a language.")
    parser.add_argument("--language", default="ell_Grek", help="HPLT language-script code.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs.")
    return parser.parse_args()


def load_manifest_entry(language: str) -> dict:
    lines = requests.get(MANIFEST_URL, timeout=60).text.splitlines()
    for line in lines:
        if f'"name": "{language}"' in line:
            return json.loads(line)
    raise ValueError(f"Language {language!r} not found in manifest")


def head_content_length(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    if value is None:
        raise ValueError(f"Missing Content-Length for {url}")
    return int(value)


def shard_bin(url: str) -> int:
    name = Path(urlparse(url).path).name
    return int(name.split("_", 1)[0])


def gib(value: int) -> float:
    return value / (1024**3)


def pct(value: float) -> float:
    return round(value * 100.0, 2)


def plot_bar(labels: list[str], values: list[float], title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color="#2f6db3")
    plt.title(title)
    plt.xlabel("Quality bin")
    plt.ylabel(xlabel)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# HPLT Quality-Bin Analysis",
        "",
        f"- Language: `{summary['language']}`",
        f"- Total documents in language manifest: `{summary['total_documents_manifest']}`",
        f"- Total compressed bytes across shards: `{summary['total_compressed_bytes']}`",
        f"- Total compressed GiB across shards: `{summary['total_compressed_gib']}`",
        "",
        "Important note:",
        "- HPLT does not publish per-bin document counts directly in the manifest.",
        "- This report uses exact published shard sizes (`Content-Length`) as a whole-corpus volume measure for the quality bins.",
        "- If exact per-bin document counts are needed, that requires a full line-count pass over all Greek shards.",
        "",
        "## Overall Distribution By Quality Bin",
    ]
    for row in summary["bin_rows"]:
        lines.append(
            f"- bin `{row['quality_bin']}`: `{row['compressed_gib']}` GiB, `{row['compressed_share_pct']}`% of compressed corpus volume"
        )
    lines.extend(
        [
            "",
            "## Keep Threshold `>=8`",
            f"- retained compressed volume: `{summary['keep_ge_8']['compressed_gib']}` GiB",
            f"- retained share of total compressed volume: `{summary['keep_ge_8']['share_of_total_pct']}`%",
            f"- dropped share (`5-7`): `{summary['drop_lt_8']['share_of_total_pct']}`%",
            "",
            "## New Distribution After Filtering To `>=8`",
        ]
    )
    for row in summary["keep_ge_8"]["normalized_rows"]:
        lines.append(
            f"- bin `{row['quality_bin']}` within kept set: `{row['share_within_kept_pct']}`%"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_entry(args.language)
    per_bin_bytes: dict[int, int] = defaultdict(int)
    shard_rows: list[dict] = []
    for url in manifest["urls"]:
        size = head_content_length(url)
        bin_value = shard_bin(url)
        per_bin_bytes[bin_value] += size
        shard_rows.append(
            {
                "url": url,
                "shard": Path(urlparse(url).path).name,
                "quality_bin": bin_value,
                "compressed_bytes": size,
                "compressed_gib": round(gib(size), 3),
            }
        )

    total_bytes = sum(per_bin_bytes.values())
    bin_rows = []
    for bin_value in sorted(per_bin_bytes):
        size = per_bin_bytes[bin_value]
        bin_rows.append(
            {
                "quality_bin": bin_value,
                "compressed_bytes": size,
                "compressed_gib": round(gib(size), 3),
                "compressed_share_pct": pct(size / total_bytes),
            }
        )

    kept_rows = [row for row in bin_rows if row["quality_bin"] >= 8]
    kept_bytes = sum(row["compressed_bytes"] for row in kept_rows)
    normalized_rows = []
    for row in kept_rows:
        normalized_rows.append(
            {
                "quality_bin": row["quality_bin"],
                "compressed_bytes": row["compressed_bytes"],
                "compressed_gib": row["compressed_gib"],
                "share_within_kept_pct": pct(row["compressed_bytes"] / kept_bytes),
            }
        )

    summary = {
        "language": args.language,
        "total_documents_manifest": int(manifest["documents"]),
        "total_compressed_bytes": total_bytes,
        "total_compressed_gib": round(gib(total_bytes), 3),
        "shard_rows": shard_rows,
        "bin_rows": bin_rows,
        "keep_ge_8": {
            "compressed_bytes": kept_bytes,
            "compressed_gib": round(gib(kept_bytes), 3),
            "share_of_total_pct": pct(kept_bytes / total_bytes),
            "normalized_rows": normalized_rows,
        },
        "drop_lt_8": {
            "compressed_bytes": total_bytes - kept_bytes,
            "compressed_gib": round(gib(total_bytes - kept_bytes), 3),
            "share_of_total_pct": pct((total_bytes - kept_bytes) / total_bytes),
        },
    }

    summary_json = args.output_dir / "summary.json"
    summary_md = args.output_dir / "summary.md"
    overall_png = args.output_dir / "quality_bin_distribution_all.png"
    kept_png = args.output_dir / "quality_bin_distribution_ge8.png"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md)
    plot_bar(
        [str(row["quality_bin"]) for row in bin_rows],
        [row["compressed_share_pct"] for row in bin_rows],
        f"HPLT {args.language}: compressed-volume share by quality bin",
        "Share of compressed corpus volume (%)",
        overall_png,
    )
    plot_bar(
        [str(row["quality_bin"]) for row in normalized_rows],
        [row["share_within_kept_pct"] for row in normalized_rows],
        f"HPLT {args.language}: distribution after keeping only quality bins >=8",
        "Share within kept set (%)",
        kept_png,
    )

    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "quality_bin_distribution_all": str(overall_png),
                "quality_bin_distribution_ge8": str(kept_png),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
