#!/usr/bin/env python3
"""Build a deterministic validation slice from a held-out rule-count pack.

This is review preparation only. It reads an existing held-out review pack,
selects rows by sample-set quotas, and writes a smaller review pack plus an
annotation template. It does not infer labels and does not mutate source data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_HELDOUT_PACK = Path("reports/full_rule_count_20260606T105018Z_heldout_review_pack.jsonl")
DEFAULT_LOCAL_DOC_DIR = Path("reports/full_rule_count_20260606T105018Z_heldout_docs")
DEFAULT_OUTPUT_PREFIX = Path("reports/heldout_validation_slice")
DEFAULT_QUOTAS = {
    "action::drop_doc": 60,
    "action::quarantine": 60,
    "action::trim_prefix": 17,
    "global_random_unseen": 60,
    "nohit_keep_control": 60,
    "url_router_signal": 40,
    "source_policy_signal": 40,
}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def row_key(row: dict[str, Any], sample_set: str, seed: str) -> tuple[str, str]:
    source_doc_id = str(row.get("source_doc_id") or "")
    digest = hashlib.sha256(f"{seed}\t{sample_set}\t{source_doc_id}".encode()).hexdigest()
    return (digest, source_doc_id)


def parse_quota(values: list[str]) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Quota must be sample_set=N, got {value!r}")
        key, raw_n = value.split("=", 1)
        quotas[key.strip()] = int(raw_n)
    return quotas


def localize_doc_path(row: dict[str, Any], local_doc_dir: Path | None) -> dict[str, Any]:
    out = dict(row)
    if local_doc_dir:
        name = Path(str(row.get("doc_text_path") or "")).name
        if name:
            local_path = local_doc_dir / name
            if local_path.exists():
                out["doc_text_path"] = str(local_path)
    return out


def compact_text(text: str, head_chars: int, tail_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= head_chars + tail_chars + 80:
        return text
    return text[:head_chars] + "\n\n[...snip...]\n\n" + text[-tail_chars:]


def read_doc(row: dict[str, Any], head_chars: int, tail_chars: int) -> str:
    path = Path(str(row.get("doc_text_path") or ""))
    if not path.exists():
        return f"[missing doc_text_path: {path}]"
    return compact_text(path.read_text(encoding="utf-8", errors="replace"), head_chars, tail_chars)


def select_rows(rows: list[dict[str, Any]], quotas: dict[str, int], seed: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    selected_by_set: dict[str, list[str]] = {}
    for sample_set, limit in quotas.items():
        candidates = [
            row
            for row in rows
            if sample_set in (row.get("sample_set") or [])
        ]
        candidates.sort(key=lambda row: row_key(row, sample_set, seed))
        chosen: list[str] = []
        for row in candidates[: max(0, limit)]:
            source_doc_id = str(row.get("source_doc_id") or "")
            if not source_doc_id:
                continue
            by_id.setdefault(source_doc_id, row)
            chosen.append(source_doc_id)
        selected_by_set[sample_set] = chosen
    selected = [by_id[key] for key in sorted(by_id)]
    summary = {
        "input_rows": len(rows),
        "quota": quotas,
        "selected_rows": len(selected),
        "selected_unique_source_docs": len(by_id),
        "selected_by_set": {key: len(value) for key, value in sorted(selected_by_set.items())},
    }
    return selected, summary


def annotation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "heldout_review_id": row.get("heldout_review_id"),
        "source_doc_id": row.get("source_doc_id"),
        "sample_set": row.get("sample_set") or [],
        "host": row.get("host"),
        "url": row.get("url"),
        "quality_bin": row.get("quality_bin"),
        "doc_text_path": row.get("doc_text_path"),
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "review_label": None,
        "review_true_error_type_ids": [],
        "review_false_negative_error_ids": [],
        "review_correct_action": None,
        "review_severity": None,
        "review_good_text_loss_risk": None,
        "review_good_chars_in_removed_span": None,
        "review_removed_chars": None,
        "review_notes": None,
        "reviewer": None,
        "review_basis": "heldout_generalization_fulltext",
    }


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_prefix: Path,
    timestamp: str,
    head_chars: int,
    tail_chars: int,
) -> dict[str, str]:
    jsonl_path = output_prefix.with_name(f"{output_prefix.name}_{timestamp}.jsonl")
    template_path = output_prefix.with_name(f"{output_prefix.name}_{timestamp}_annotation_template.jsonl")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_{timestamp}_summary.json")
    md_path = output_prefix.with_name(f"{output_prefix.name}_{timestamp}.md")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as out, template_path.open("w", encoding="utf-8") as tmpl:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            tmpl.write(json.dumps(annotation_row(row), ensure_ascii=False, sort_keys=True) + "\n")

    summary = dict(summary)
    summary.update(
        {
            "created_at_utc": timestamp,
            "review_pack_jsonl": str(jsonl_path),
            "annotation_template_jsonl": str(template_path),
            "markdown": str(md_path),
        }
    )
    write_json(summary_path, summary)

    with md_path.open("w", encoding="utf-8") as out:
        out.write("# Held-Out Validation Slice\n\n")
        out.write("This is a deterministic review slice. Fill the annotation template; do not edit source HPLT rows.\n\n")
        out.write(f"- rows: `{len(rows)}`\n")
        out.write(f"- annotation template: `{template_path}`\n")
        out.write(f"- summary: `{summary_path}`\n\n")
        for idx, row in enumerate(rows, 1):
            out.write(f"## {idx}. {row.get('heldout_review_id') or row.get('source_doc_id')}\n\n")
            out.write(f"- source_doc_id: `{row.get('source_doc_id')}`\n")
            out.write(f"- sample_set: `{', '.join(row.get('sample_set') or [])}`\n")
            out.write(f"- candidate_action: `{row.get('candidate_action')}`\n")
            out.write(f"- candidate_error_type_ids: `{', '.join(row.get('candidate_error_type_ids') or [])}`\n")
            out.write(f"- host: `{row.get('host')}`\n")
            out.write(f"- url: `{row.get('url')}`\n")
            out.write(f"- doc_text_path: `{row.get('doc_text_path')}`\n\n")
            out.write("```text\n")
            out.write(read_doc(row, head_chars, tail_chars))
            out.write("\n```\n\n")

    return {
        "review_pack_jsonl": str(jsonl_path),
        "annotation_template_jsonl": str(template_path),
        "summary_json": str(summary_path),
        "markdown": str(md_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-pack", type=Path, default=DEFAULT_HELDOUT_PACK)
    parser.add_argument("--local-doc-dir", type=Path, default=DEFAULT_LOCAL_DOC_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--seed", default="20260606-heldout-validation")
    parser.add_argument("--quota", action="append", default=[], help="Sample-set quota as sample_set=N. Repeatable.")
    parser.add_argument("--head-chars", type=int, default=1400)
    parser.add_argument("--tail-chars", type=int, default=900)
    args = parser.parse_args()

    rows = [localize_doc_path(row, args.local_doc_dir) for row in read_jsonl(args.heldout_pack)]
    selected, summary = select_rows(rows, parse_quota(args.quota), args.seed)
    outputs = write_outputs(selected, summary, args.output_prefix, args.timestamp, args.head_chars, args.tail_chars)
    print(json.dumps({"summary": summary, "outputs": outputs}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
