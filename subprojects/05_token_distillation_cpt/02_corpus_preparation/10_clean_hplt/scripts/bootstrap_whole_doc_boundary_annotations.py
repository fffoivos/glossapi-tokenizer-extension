#!/usr/bin/env python3
"""Bootstrap whole-document boundary annotations from prior reviews.

This script is conservative: it only promotes boundary-pack rows whose reviewed
action is whole-document (`drop_doc` or `quarantine`) and whose boundary pack
already carries prior review notes plus a prior good-text-loss risk. It copies
that reviewed evidence into a new annotation file and leaves all other rows
pending. It does not inspect text semantically, apply a policy, or mutate source
HPLT rows.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hplt-whole-doc-boundary-bootstrap-v1"
DEFAULT_REVIEW_PACK = "reports/boundary_spec_review_pack_20260606T002713Z.jsonl"
DEFAULT_TEMPLATE = "reports/boundary_spec_review_pack_20260606T002713Z_annotation_template.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/boundary_whole_doc_bootstrap_annotations"
WHOLE_DOC_ACTIONS = {"drop_doc", "quarantine"}
TIMESTAMP_SUFFIX_RE = re.compile(r"\d{8}T\d{6}Z$")


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def has_prior_good_text_loss(value: Any) -> bool:
    text = compact_text(value).strip().lower()
    return bool(text) and text not in {"none", "null", "unknown", "not_estimated"}


def row_notes(row: dict[str, Any]) -> str:
    return compact_text(row.get("review_span_or_split_notes") or row.get("review_notes")).strip()


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    review_rows = read_jsonl(Path(args.review_pack))
    template_rows = read_jsonl(Path(args.template))
    review_by_id = {compact_text(row.get("source_doc_id")): row for row in review_rows}

    output_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    skipped_reasons: collections.Counter[str] = collections.Counter()

    for template in template_rows:
        source_doc_id = compact_text(template.get("source_doc_id"))
        review = review_by_id.get(source_doc_id)
        row = dict(template)
        action = compact_text(row.get("boundary_review_action") or row.get("reviewed_action"))
        notes = row_notes(review or {})
        good_text_loss = (review or {}).get("review_good_text_loss_risk")

        if review is None:
            skipped_reasons["missing_review_pack_row"] += 1
        elif action not in WHOLE_DOC_ACTIONS:
            skipped_reasons["not_whole_doc_action"] += 1
        elif not notes:
            skipped_reasons["missing_prior_review_notes"] += 1
        elif not has_prior_good_text_loss(good_text_loss):
            skipped_reasons["missing_prior_good_text_loss_risk"] += 1
        else:
            row["schema_version"] = "hplt-boundary-spec-annotation-template-v1"
            row["boundary_review_status"] = "reviewed_accept"
            row["boundary_review_action"] = action
            row["span_ranges"] = []
            row["split_parts"] = []
            row["dropped_span_ranges"] = []
            row["good_text_loss_estimate"] = compact_text(good_text_loss)
            row["boundary_notes"] = (
                "Bootstrapped from prior manual review for whole-document "
                f"{action}: {notes}"
            )
            row["reviewer"] = args.reviewer
            row["reviewed_at_utc"] = args.timestamp
            row["bootstrap_schema_version"] = SCHEMA_VERSION
            row["bootstrap_source_review_label"] = review.get("review_label")
            row["bootstrap_source_review_notes"] = notes
            row["bootstrap_source_review_good_text_loss_risk"] = good_text_loss
            row["bootstrap_policy_note"] = (
                "Whole-document action copied from prior reviewed annotation; "
                "no span or split boundary was inferred."
            )
            selected_rows.append(row)
        output_rows.append(row)

    output_prefix = Path(args.output_prefix)
    if not TIMESTAMP_SUFFIX_RE.search(output_prefix.name):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)
    annotations_path = Path(str(output_prefix) + ".jsonl")
    summary_path = Path(str(output_prefix) + "_summary.json")
    md_path = Path(str(output_prefix) + ".md")

    write_jsonl(annotations_path, output_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": args.timestamp,
        "policy_note": "Bootstrap only. This copies prior reviewed whole-document decisions into a boundary-annotation file and does not apply a cleaning policy.",
        "review_pack": args.review_pack,
        "template": args.template,
        "annotations_jsonl": str(annotations_path),
        "markdown": str(md_path),
        "template_rows": len(template_rows),
        "selected_rows": len(selected_rows),
        "pending_rows": len(output_rows) - len(selected_rows),
        "selected_by_action": dict(collections.Counter(row.get("boundary_review_action") for row in selected_rows).most_common()),
        "selected_by_good_text_loss": dict(collections.Counter(compact_text(row.get("good_text_loss_estimate")) for row in selected_rows).most_common()),
        "skipped_reasons": dict(skipped_reasons.most_common()),
    }
    write_json(summary_path, summary)
    write_markdown(md_path, summary, selected_rows)
    return summary


def write_markdown(path: Path, summary: dict[str, Any], selected_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Whole-Document Boundary Bootstrap\n\n")
        out.write("This is a conservative bootstrap from prior manual reviews. It selects only `drop_doc` and `quarantine` rows that already have prior review notes and a prior good-text-loss risk. It does not infer spans, splits, or new semantic labels.\n\n")
        out.write("## Summary\n\n")
        for key in ["template_rows", "selected_rows", "pending_rows"]:
            out.write(f"- {key}: `{summary.get(key)}`\n")
        out.write("\n## Selected By Action\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in summary["selected_by_action"].items():
            out.write(f"| `{action}` | {count} |\n")
        out.write("\n## Skipped Reasons\n\n| Reason | Rows |\n| --- | ---: |\n")
        for reason, count in summary["skipped_reasons"].items():
            out.write(f"| `{reason}` | {count} |\n")
        out.write("\n## Selected Rows\n\n")
        for row in selected_rows[:100]:
            out.write(
                "- `%s` action=`%s` loss=`%s` host=`%s`\n"
                % (
                    row.get("source_doc_id"),
                    row.get("boundary_review_action"),
                    row.get("good_text_loss_estimate"),
                    row.get("host"),
                )
            )
        if len(selected_rows) > 100:
            out.write(f"- ... {len(selected_rows) - 100} more selected rows\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", default=DEFAULT_REVIEW_PACK)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--reviewer", default="codex_prior_review_bootstrap")
    args = parser.parse_args()

    summary = bootstrap(args)
    print(
        json.dumps(
            {
                "annotations_jsonl": summary["annotations_jsonl"],
                "selected_rows": summary["selected_rows"],
                "pending_rows": summary["pending_rows"],
                "summary": summary["annotations_jsonl"].removesuffix(".jsonl") + "_summary.json",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
