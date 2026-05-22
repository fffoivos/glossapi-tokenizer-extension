#!/usr/bin/env python3
"""Render a compact Markdown report for the TD coverage prepass."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            yield row


def as_int(row: Dict[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def quantile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return int(round(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac))


def pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{100.0 * part / total:.2f}%"


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def token_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| id | status | firings | docs | snip100 | base_len | token |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {status} | {firings} | {docs} | {snip100} | {base_len} | `{token}` |".format(
                id=row.get("new_token_id"),
                status=md_escape(row.get("status")),
                firings=as_int(row, "extended_firings"),
                docs=as_int(row, "docs_with_firing"),
                snip100=as_int(row, "usable_snippets_100"),
                base_len=as_int(row, "base_subtoken_len"),
                token=md_escape(row.get("token_string")),
            )
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(args.coverage_jsonl))
    if not rows:
        raise SystemExit(f"empty coverage JSONL: {args.coverage_jsonl}")

    status_counts = Counter(str(row.get("status")) for row in rows)
    action_counts = Counter(str(row.get("recommended_action")) for row in rows)
    firings = sorted(as_int(row, "extended_firings") for row in rows)
    docs = sorted(as_int(row, "docs_with_firing") for row in rows)
    base_len_counts = Counter(as_int(row, "base_subtoken_len") for row in rows)

    by_len = defaultdict(list)
    for row in rows:
        by_len[as_int(row, "base_subtoken_len")].append(as_int(row, "extended_firings"))

    low_rows = sorted(
        [row for row in rows if str(row.get("status")) not in {"enough_100", "enough_25"}],
        key=lambda row: (
            as_int(row, "usable_snippets_100"),
            as_int(row, "extended_firings"),
            as_int(row, "docs_with_firing"),
        ),
    )[: args.top_n]
    top_rows = sorted(
        rows,
        key=lambda row: (as_int(row, "extended_firings"), as_int(row, "docs_with_firing")),
        reverse=True,
    )[: args.top_n]

    n = len(rows)
    lines = []
    lines.append("# Token Distillation Coverage Summary")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- recommended next step: `{summary.get('recommended_next_step')}`")
    lines.append(f"- tokens scanned: `{summary.get('tokens_scanned')}` / `{summary.get('target_extended_tokens')}`")
    lines.append(f"- docs used: `{summary.get('docs_used')}`")
    lines.append(f"- stopped on budget: `{summary.get('stopped_on_budget')}`")
    lines.append(f"- NFC required: `{summary.get('require_nfc')}`; non-NFC docs: `{summary.get('non_nfc_docs')}`")
    lines.append(f"- enough_100 fraction: `{summary.get('enough_100_fraction')}`")
    lines.append(f"- enough_25 fraction: `{summary.get('enough_25_fraction')}`")
    lines.append(f"- low_lt25 count: `{summary.get('low_lt25_count')}`")
    lines.append("")

    lines.append("## Status Counts")
    lines.append("")
    lines.append("| status | count | share |")
    lines.append("|---|---:|---:|")
    for status, count in sorted(status_counts.items()):
        lines.append(f"| `{status}` | {count} | {pct(count, n)} |")
    lines.append("")

    lines.append("## Action Counts")
    lines.append("")
    lines.append("| action | count | share |")
    lines.append("|---|---:|---:|")
    for action, count in sorted(action_counts.items()):
        lines.append(f"| `{action}` | {count} | {pct(count, n)} |")
    lines.append("")

    lines.append("## Firing Quantiles")
    lines.append("")
    lines.append("| metric | p0 | p5 | p25 | p50 | p75 | p95 | p100 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| extended firings | {p0} | {p5} | {p25} | {p50} | {p75} | {p95} | {p100} |".format(
            p0=quantile(firings, 0.00),
            p5=quantile(firings, 0.05),
            p25=quantile(firings, 0.25),
            p50=quantile(firings, 0.50),
            p75=quantile(firings, 0.75),
            p95=quantile(firings, 0.95),
            p100=quantile(firings, 1.00),
        )
    )
    lines.append(
        "| docs with firing | {p0} | {p5} | {p25} | {p50} | {p75} | {p95} | {p100} |".format(
            p0=quantile(docs, 0.00),
            p5=quantile(docs, 0.05),
            p25=quantile(docs, 0.25),
            p50=quantile(docs, 0.50),
            p75=quantile(docs, 0.75),
            p95=quantile(docs, 0.95),
            p100=quantile(docs, 1.00),
        )
    )
    lines.append("")

    lines.append("## Base-Subtoken Length")
    lines.append("")
    lines.append("| base_subtoken_len | tokens | median firings | p5 firings | p95 firings |")
    lines.append("|---:|---:|---:|---:|---:|")
    for base_len in sorted(base_len_counts):
        values = sorted(by_len[base_len])
        lines.append(
            f"| {base_len} | {base_len_counts[base_len]} | {quantile(values, 0.50)} | "
            f"{quantile(values, 0.05)} | {quantile(values, 0.95)} |"
        )
    lines.append("")

    lines.append(f"## Top {args.top_n} Tokens By Firing")
    lines.append("")
    lines.extend(token_table(top_rows))
    lines.append("")

    lines.append(f"## Lowest-Coverage {args.top_n} Tokens")
    lines.append("")
    lines.extend(token_table(low_rows))
    lines.append("")

    lines.append("## Artifact Paths")
    lines.append("")
    artifacts = summary.get("artifacts") or {}
    for key in sorted(artifacts):
        lines.append(f"- `{key}`: `{artifacts[key]}`")
    lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
