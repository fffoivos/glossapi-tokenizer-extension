#!/usr/bin/env python3
"""Count continuous silver BIB blocks per annotated document.

The audit is label-only: it does not run a classifier or inspect line text.
Every document is preserved as one JSONL/CSV row with its BIB block count and
the exact present-line coordinate spans that produced that count.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "bibliography-silver-block-audit-v1"
LABELS = {0: "O", 1: "BIB", 2: "TOC"}
KNOWN_LABELS = frozenset(LABELS.values())
DEFAULT_MAX_PHYSICAL_GAP = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_identity(raw: Any, *, document_id: str) -> tuple[int, str]:
    if isinstance(raw, Mapping):
        abs_idx = raw.get("abs_idx")
        label = raw.get("label")
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) < 3:
            raise ValueError(f"{document_id}: legacy line has fewer than 3 fields")
        abs_idx, label = raw[0], raw[2]
    else:
        raise ValueError(f"{document_id}: unsupported line row")
    if not isinstance(abs_idx, int) or abs_idx < 0:
        raise ValueError(f"{document_id}: invalid abs_idx {abs_idx!r}")
    if isinstance(label, int) and not isinstance(label, bool):
        label = LABELS.get(label)
    if label not in KNOWN_LABELS:
        raise ValueError(f"{document_id}: invalid label {label!r} at {abs_idx}")
    return abs_idx, str(label)


def audit_document(
    row: Mapping[str, Any], *, max_physical_gap: int = DEFAULT_MAX_PHYSICAL_GAP
) -> dict[str, Any]:
    """Count uninterrupted BIB runs in emitted order for one document.

    Blank physical lines are absent from the silver corpus, so small coordinate
    gaps do not split a block. A gap larger than ``max_physical_gap`` does split
    it, preventing the front and tail of a windowed document from being joined.
    """

    document_id = str(row.get("document_id") or row.get("doc_id") or "")
    if not document_id:
        raise ValueError("document has no document_id/doc_id")
    raw_lines = row.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError(f"{document_id}: missing lines")

    blocks: list[dict[str, int]] = []
    active: dict[str, int] | None = None
    previous_idx: int | None = None
    previous_label: str | None = None
    bib_line_count = 0
    coverage_gap_splits = 0
    present_labels = collections.Counter[str]()

    for raw_line in raw_lines:
        abs_idx, label = _line_identity(raw_line, document_id=document_id)
        if previous_idx is not None and abs_idx <= previous_idx:
            raise ValueError(
                f"{document_id}: coordinates are not strictly increasing: "
                f"{previous_idx}, {abs_idx}"
            )
        present_labels[label] += 1
        coordinate_gap = abs_idx - previous_idx if previous_idx is not None else 1
        gap_split = coordinate_gap > max_physical_gap
        if label == "BIB":
            bib_line_count += 1
            if active is None or previous_label != "BIB" or gap_split:
                if active is not None:
                    blocks.append(active)
                if gap_split and previous_label == "BIB":
                    coverage_gap_splits += 1
                active = {
                    "block_index": len(blocks),
                    "start_abs_idx": abs_idx,
                    "end_abs_idx": abs_idx,
                    "present_line_count": 1,
                }
            else:
                active["end_abs_idx"] = abs_idx
                active["present_line_count"] += 1
        elif active is not None:
            blocks.append(active)
            active = None
        previous_idx, previous_label = abs_idx, label
    if active is not None:
        blocks.append(active)

    for index, block in enumerate(blocks):
        block["block_index"] = index
        block["physical_span_line_count"] = (
            block["end_abs_idx"] - block["start_abs_idx"] + 1
        )

    annotation = row.get("annotation")
    annotation_status = (
        str(annotation.get("status", "")) if isinstance(annotation, Mapping) else ""
    )
    return {
        "schema_version": "bibliography-silver-block-document-v1",
        "document_id": document_id,
        "work_id": str(row.get("work_id", "")),
        "source": str(row.get("source", "")),
        "split": str(row.get("split", "")),
        "coverage": str(row.get("coverage", "")),
        "historical_split": str(row.get("historical_split", "")),
        "historical_mode": str(row.get("historical_mode", row.get("mode", ""))),
        "annotation_status": annotation_status,
        "n_physical_lines": int(row.get("n_physical_lines", row.get("n_lines", 0))),
        "n_present_lines": len(raw_lines),
        "label_counts": {label: present_labels[label] for label in sorted(KNOWN_LABELS)},
        "bib_line_count": bib_line_count,
        "bib_block_count": len(blocks),
        "coverage_gap_splits": coverage_gap_splits,
        "blocks": blocks,
    }


def _iter_documents(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {row_number}: {error}") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {row_number} is not an object")
            yield row


def _percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _group_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(int(row["bib_block_count"]))
    result: dict[str, Any] = {}
    for name, values in sorted(grouped.items()):
        result[name or "(missing)"] = {
            "document_count": len(values),
            "mean": round(statistics.fmean(values), 6),
            "median": statistics.median(values),
            "maximum": max(values),
        }
    return result


def summarize(rows: Sequence[Mapping[str, Any]], *, top_n: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("no documents were audited")
    values = [int(row["bib_block_count"]) for row in rows]
    histogram = collections.Counter(values)
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row["bib_block_count"]),
            -int(row["bib_line_count"]),
            str(row["document_id"]),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "score_definition": "number of continuous emitted BIB-label runs per document",
        "continuity": {
            "ordinary_rule": "a block starts on an O/TOC-to-BIB transition and ends before the next non-BIB emitted line",
            "blank_lines": "absent blank physical lines do not split a block",
            "window_seam_guard": "a large absolute-coordinate gap splits adjacent BIB rows",
        },
        "document_count": len(rows),
        "documents_with_no_bib_block": sum(value == 0 for value in values),
        "documents_with_one_bib_block": sum(value == 1 for value in values),
        "documents_with_multiple_bib_blocks": sum(value > 1 for value in values),
        "mean": round(statistics.fmean(values), 6),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values),
        "histogram": [
            {"bib_block_count": value, "document_count": histogram[value]}
            for value in range(max(values) + 1)
        ],
        "by_source": _group_summary(rows, "source"),
        "by_split": _group_summary(rows, "split"),
        "by_coverage": _group_summary(rows, "coverage"),
        "top_documents": [
            {
                "rank": index + 1,
                "document_id": str(row["document_id"]),
                "work_id": str(row["work_id"]),
                "source": str(row["source"]),
                "split": str(row["split"]),
                "coverage": str(row["coverage"]),
                "bib_block_count": int(row["bib_block_count"]),
                "bib_line_count": int(row["bib_line_count"]),
                "blocks": row["blocks"],
            }
            for index, row in enumerate(ranked[:top_n])
        ],
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "document_id",
        "work_id",
        "source",
        "split",
        "coverage",
        "historical_split",
        "historical_mode",
        "n_physical_lines",
        "n_present_lines",
        "bib_line_count",
        "bib_block_count",
        "coverage_gap_splits",
        "block_spans",
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field, "") for field in fields}
            flat["block_spans"] = ";".join(
                f"{block['start_abs_idx']}-{block['end_abs_idx']}"
                for block in row["blocks"]
            )
            writer.writerow(flat)


def _plot_svg(summary: Mapping[str, Any]) -> str:
    width, height = 1200, 760
    left, right = 78, 32
    histogram_top, histogram_bottom = 78, 430
    top_top, top_bottom = 520, 720
    histogram = list(summary["histogram"])
    if len(histogram) > 16:
        shown = histogram[:15]
        shown.append(
            {
                "bib_block_count": "15+",
                "document_count": sum(item["document_count"] for item in histogram[15:]),
            }
        )
    else:
        shown = histogram
    plot_width = width - left - right
    bar_gap = 8
    bar_width = max(8, (plot_width - bar_gap * max(0, len(shown) - 1)) / len(shown))
    maximum_count = max(item["document_count"] for item in shown)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Silver bibliography block distribution</title>",
        f'<desc id="desc">Histogram for {summary["document_count"]} documents and the fifteen documents with the most continuous BIB blocks.</desc>',
        "<style>:root{color-scheme:light dark}text{font-family:system-ui,sans-serif;fill:#17201c}.axis{stroke:#6c756f;stroke-width:1}.grid{stroke:#d8d1c3;stroke-width:1}.bar{fill:#527963}.tail{fill:#a15e50}.small{font-size:12px}.label{font-size:14px}.heading{font-size:20px;font-weight:600}@media(prefers-color-scheme:dark){text{fill:#ecefe9}.axis{stroke:#a8b0aa}.grid{stroke:#46504a}.bar{fill:#74a88e}.tail{fill:#d08a78}}</style>",
        '<text class="heading" x="78" y="34">Continuous silver BIB blocks per document</text>',
        f'<text class="label" x="78" y="57">n={summary["document_count"]} · mean={summary["mean"]:.2f} · median={summary["median"]} · p95={summary["p95"]} · max={summary["maximum"]}</text>',
    ]
    for tick in range(5):
        value = maximum_count * tick / 4
        y = histogram_bottom - (histogram_bottom - histogram_top) * tick / 4
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.0f}</text>')
    for index, item in enumerate(shown):
        x = left + index * (bar_width + bar_gap)
        bar_height = (histogram_bottom - histogram_top) * item["document_count"] / maximum_count
        y = histogram_bottom - bar_height
        class_name = "tail" if item["bib_block_count"] == "15+" else "bar"
        parts.append(f'<rect class="{class_name}" x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3"/>')
        parts.append(f'<text class="small" x="{x+bar_width/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{item["document_count"]}</text>')
        parts.append(f'<text class="small" x="{x+bar_width/2:.1f}" y="{histogram_bottom+19}" text-anchor="middle">{item["bib_block_count"]}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{histogram_bottom}" x2="{width-right}" y2="{histogram_bottom}"/>',
            f'<text class="label" x="{(left+width-right)/2:.1f}" y="{histogram_bottom+48}" text-anchor="middle">BIB block count</text>',
            f'<text class="heading" x="{left}" y="{top_top-24}">Highest-scoring documents</text>',
        ]
    )
    top = list(summary["top_documents"][:15])
    maximum_blocks = max(int(item["bib_block_count"]) for item in top)
    label_width = 238
    available = width - left - right - label_width
    row_height = (top_bottom - top_top) / len(top)
    for index, item in enumerate(top):
        y = top_top + index * row_height
        blocks = int(item["bib_block_count"])
        bar_length = available * blocks / maximum_blocks
        short = html.escape(str(item["document_id"])[:18])
        parts.append(f'<text class="small" x="{left}" y="{y+11:.1f}">{index+1}. {short}</text>')
        parts.append(f'<rect class="tail" x="{left+label_width}" y="{y:.1f}" width="{bar_length:.1f}" height="{max(5,row_height-4):.1f}" rx="2"/>')
        parts.append(f'<text class="small" x="{left+label_width+bar_length+7:.1f}" y="{y+11:.1f}">{blocks}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _report(summary: Mapping[str, Any]) -> str:
    rows = [
        "# Silver bibliography block audit",
        "",
        "A document's score is the number of continuous emitted `BIB`-label runs.",
        "The audit does not run a classifier and does not reinterpret bibliography",
        "headers; it measures the existing silver section labels exactly.",
        "",
        "## Distribution",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f'| Documents | {summary["document_count"]:,} |',
        f'| Mean BIB blocks/document | {summary["mean"]:.3f} |',
        f'| Median | {summary["median"]} |',
        f'| 90th percentile | {summary["p90"]} |',
        f'| 95th percentile | {summary["p95"]} |',
        f'| 99th percentile | {summary["p99"]} |',
        f'| Maximum | {summary["maximum"]} |',
        f'| No BIB block | {summary["documents_with_no_bib_block"]:,} |',
        f'| Exactly one BIB block | {summary["documents_with_one_bib_block"]:,} |',
        f'| Multiple BIB blocks | {summary["documents_with_multiple_bib_blocks"]:,} |',
        "",
        "## Highest scores",
        "",
        "| Rank | Document | Source | Split | Coverage | Blocks | BIB lines |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for item in summary["top_documents"]:
        rows.append(
            f'| {item["rank"]} | `{item["document_id"]}` | {item["source"]} | '
            f'{item["split"]} | {item["coverage"]} | {item["bib_block_count"]} | '
            f'{item["bib_line_count"]} |'
        )
    rows.extend(
        [
            "",
            "## Preserved outputs",
            "",
            "- `documents.jsonl`: every document, exact block count, and block spans",
            "- `documents.csv`: flattened per-document review table",
            "- `summary.json`: distribution, group summaries, and ranked tail",
            "- `distribution.svg`: histogram and highest-scoring documents",
            "- `receipt.json`: hashes and run provenance",
            "",
        ]
    )
    return "\n".join(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError(f"input must be a regular file: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        audit_document(row, max_physical_gap=args.max_physical_gap)
        for row in _iter_documents(input_path)
    ]
    if len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate document_id in input")
    summary = summarize(rows, top_n=args.top_n)
    summary["input"] = {"path": str(input_path), "sha256": _sha256(input_path)}
    summary["max_physical_gap"] = args.max_physical_gap

    paths = {
        "documents_jsonl": output_dir / "documents.jsonl",
        "documents_csv": output_dir / "documents.csv",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "plot": output_dir / "distribution.svg",
    }
    _write_jsonl(paths["documents_jsonl"], rows)
    _write_csv(paths["documents_csv"], rows)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(_report(summary), encoding="utf-8")
    paths["plot"].write_text(_plot_svg(summary), encoding="utf-8")

    receipt = {
        "schema_version": "bibliography-silver-block-audit-receipt-v1",
        "status": "passed",
        "input": summary["input"],
        "document_count": len(rows),
        "known_line_count": sum(int(row["n_present_lines"]) for row in rows),
        "max_physical_gap": args.max_physical_gap,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="silver JSONL")
    parser.add_argument("--output-dir", required=True, help="new or empty output directory")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--max-physical-gap", type=int, default=DEFAULT_MAX_PHYSICAL_GAP)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = run(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

