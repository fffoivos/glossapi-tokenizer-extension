#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from glossapi import Corpus


PREFIXES = ("x_60_", "x_10_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark HPLT sample txt files by Greek badness buckets using GlossAPI Corpus.clean()."
    )
    parser.add_argument(
        "--category-samples-root",
        type=Path,
        required=True,
        help="Path to the category_samples directory.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Directory where per-bucket clean runs and summaries will be stored.",
    )
    return parser.parse_args()


def normalize_prefixed_names(txt_dir: Path) -> None:
    for path in sorted(txt_dir.glob("*.txt")):
        name = path.name
        base_name = name
        changed = True
        while changed:
            changed = False
            for prefix in PREFIXES:
                if base_name.startswith(prefix):
                    base_name = base_name[len(prefix):]
                    changed = True
        if base_name != name:
            target = txt_dir / base_name
            if target.exists():
                target.unlink()
            path.rename(target)


def stage_txt_as_md(txt_dir: Path, input_md_dir: Path) -> None:
    input_md_dir.mkdir(parents=True, exist_ok=True)
    for txt_path in sorted(txt_dir.glob("*.txt")):
        md_path = input_md_dir / f"{txt_path.stem}.md"
        md_path.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")


def apply_prefixes_from_scores(txt_dir: Path, parquet_path: Path) -> dict[str, int]:
    df = pd.read_parquet(parquet_path)
    df["score"] = pd.to_numeric(df.get("greek_badness_score"), errors="coerce")

    gt60 = set(
        df.loc[df["score"] > 60, "filename"]
        .astype(str)
        .str.replace(r"\.pdf$", ".txt", regex=True)
        .tolist()
    )
    gt10 = set(
        df.loc[(df["score"] > 10) & (df["score"] <= 60), "filename"]
        .astype(str)
        .str.replace(r"\.pdf$", ".txt", regex=True)
        .tolist()
    )

    counts = {"x_60": 0, "x_10": 0, "ok": 0}
    for txt_path in sorted(txt_dir.glob("*.txt")):
        base_name = txt_path.name
        if base_name in gt60:
            target = txt_dir / f"x_60_{base_name}"
            txt_path.rename(target)
            counts["x_60"] += 1
        elif base_name in gt10:
            target = txt_dir / f"x_10_{base_name}"
            txt_path.rename(target)
            counts["x_10"] += 1
        else:
            counts["ok"] += 1
    return counts


def process_bucket(txt_dir: Path, analysis_root: Path) -> dict[str, object]:
    normalize_prefixed_names(txt_dir)

    bucket_key = f"{txt_dir.parent.name}__{txt_dir.name}"
    bucket_root = analysis_root / bucket_key
    if bucket_root.exists():
        shutil.rmtree(bucket_root)
    input_md_dir = bucket_root / "input_md"
    run_dir = bucket_root / "run"
    stage_txt_as_md(txt_dir, input_md_dir)

    corpus = Corpus(input_dir=input_md_dir, output_dir=run_dir)
    corpus.clean(input_dir=input_md_dir, drop_bad=False, threshold=0.10, write_cleaned_files=True)

    parquet_path = run_dir / "download_results" / "download_results.parquet"
    counts = apply_prefixes_from_scores(txt_dir, parquet_path)
    report = {
        "category": txt_dir.parent.name,
        "bucket": txt_dir.name.replace("_txt", ""),
        "txt_dir": str(txt_dir),
        "parquet": str(parquet_path),
        "total": counts["x_60"] + counts["x_10"] + counts["ok"],
        "x_60_count": counts["x_60"],
        "x_10_count": counts["x_10"],
        "ok_count": counts["ok"],
    }
    (bucket_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for category_dir in sorted(path for path in args.category_samples_root.iterdir() if path.is_dir()):
        for txt_dir in sorted(path for path in category_dir.glob("*_txt") if path.is_dir()):
            reports.append(process_bucket(txt_dir, args.analysis_root))

    summary = {
        "category_samples_root": str(args.category_samples_root),
        "analysis_root": str(args.analysis_root),
        "bucket_count": len(reports),
        "reports": reports,
    }
    (args.analysis_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Full Sample Greek Badness Marking",
        "",
        f"- Bucket count: `{len(reports)}`",
        "",
        "## Buckets",
    ]
    for report in reports:
        lines.append(
            f"- `{report['category']}` / `{report['bucket']}`: "
            f"`x_60={report['x_60_count']}`, "
            f"`x_10={report['x_10_count']}`, "
            f"`ok={report['ok_count']}`"
        )
    (args.analysis_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
