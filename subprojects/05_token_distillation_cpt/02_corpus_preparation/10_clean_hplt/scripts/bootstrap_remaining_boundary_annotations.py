#!/usr/bin/env python3
"""Create reviewed annotations for the final 9 exact-boundary HPLT rows.

This is a manual boundary map, not a detector. It consumes the 9-row
boundary-spec review pack and writes accepted annotations only where an exact
literal marker or bounded shortcode span is defensible from the full text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hplt-boundary-spec-annotation-template-v1"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
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


def must_find(text: str, marker: str, source_doc_id: str) -> int:
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"marker not found for {source_doc_id}: {marker!r}")
    if text.find(marker, start + 1) >= 0:
        raise RuntimeError(f"marker is not unique for {source_doc_id}: {marker!r}")
    return start


def base_annotation(row: dict[str, Any], action: str, reviewed_at: str, notes: str, good_text_loss: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewed_at_utc": reviewed_at,
        "reviewer": "codex",
        "source_doc_id": row["source_doc_id"],
        "parent_source_doc_id": row["source_doc_id"],
        "source_annotation_files": row.get("source_annotation_files") or [],
        "source_review_rows_count": row.get("source_review_rows_count"),
        "text_sha256_before": row["text_sha256_before"],
        "url": row.get("url"),
        "host": row.get("host"),
        "quality_bin": row.get("quality_bin"),
        "error_type_ids": row.get("error_type_ids") or row.get("review_true_error_type_ids") or [],
        "reviewed_action": action,
        "boundary_review_action": action,
        "boundary_review_status": "reviewed_accept",
        "boundary_notes": notes,
        "boundary_instructions": [
            "Reviewed exact boundary decision; source rows remain immutable and changed text must be emitted only as duplicate/shadow records."
        ],
        "good_text_loss_estimate": good_text_loss,
        "span_ranges": [],
        "split_parts": [],
        "dropped_span_ranges": [],
        "chars_before": row.get("chars_before"),
        "chars_after": None,
        "chars_removed": None,
        "tokens_before": row.get("tokens_before"),
        "tokens_after": None,
        "tokens_removed": None,
    }


def split_annotation(
    row: dict[str, Any],
    reviewed_at: str,
    first_part_id: str,
    split_starts: list[tuple[str, str]],
    notes: str,
    final_end_marker: str | None = None,
) -> dict[str, Any]:
    text = row["full_text"]
    starts = [(first_part_id, 0)]
    for part_id, marker in split_starts:
        starts.append((part_id, must_find(text, marker, row["source_doc_id"])))
    final_end = len(text)
    dropped: list[dict[str, Any]] = []
    if final_end_marker is not None:
        marker_start = must_find(text, final_end_marker, row["source_doc_id"])
        final_end = marker_start
        while final_end > 0 and text[final_end - 1].isspace():
            final_end -= 1
        dropped.append(
            {
                "start": final_end,
                "end": len(text),
                "replacement": "",
                "review_marker": f"trailing related/read-more suffix at {final_end_marker!r}",
            }
        )
    parts = []
    for index, (part_id, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else final_end
        while end > start and text[end - 1].isspace():
            end -= 1
        parts.append(
            {
                "part_id": part_id,
                "start": start,
                "end": end,
                "review_marker": "exact topic-reset boundary" if start else "first attached article",
            }
        )
    annotation = base_annotation(row, "split_doc", reviewed_at, notes, "low")
    annotation["split_parts"] = parts
    annotation["dropped_span_ranges"] = dropped
    return annotation


def span_annotation(
    row: dict[str, Any],
    action: str,
    reviewed_at: str,
    spans: list[dict[str, Any]],
    notes: str,
    good_text_loss: Any = 0.0,
) -> dict[str, Any]:
    annotation = base_annotation(row, action, reviewed_at, notes, good_text_loss)
    annotation["span_ranges"] = spans
    return annotation


SHORTCODE_RE = re.compile(r"\[(?P<body>/?(?:nextpage|one_half|one_half_last|custom_list)[^\]]*)\]")
NEXTPAGE_TITLE_RE = re.compile(r"nextpage\s+title=.(?P<title>[^”\"']+).")


def shortcode_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = row["full_text"]
    suffix_start = must_find(text, "[/nextpage]\n[nextpage title=”Φωτογραφίες” ]", row["source_doc_id"])
    spans: list[dict[str, Any]] = []
    for match in SHORTCODE_RE.finditer(text):
        if match.start() >= suffix_start:
            continue
        body = match.group("body")
        replacement = ""
        marker = "CMS shortcode residue"
        if body.startswith("nextpage"):
            title_match = NEXTPAGE_TITLE_RE.search(body)
            title = title_match.group("title").strip() if title_match else ""
            replacement = (title + "\n") if match.start() == 0 else ("\n" + title + "\n")
            marker = "WordPress nextpage shortcode normalized to section title"
        spans.append(
            {
                "start": match.start(),
                "end": match.end(),
                "replacement": replacement,
                "review_marker": marker,
            }
        )
    spans.append(
        {
            "start": suffix_start,
            "end": len(text),
            "replacement": "",
            "review_marker": "empty trailing nextpage shortcode-only sections",
        }
    )
    spans.sort(key=lambda item: item["start"])
    return spans


def build_annotations(rows: list[dict[str, Any]], reviewed_at: str) -> list[dict[str, Any]]:
    by_id = {row["source_doc_id"]: row for row in rows}
    annotations: list[dict[str, Any]] = []

    annotations.append(
        split_annotation(
            by_id["hplt::9_1.jsonl.zst::e63a08bddef7a8ad459a85eda2b722f7"],
            reviewed_at,
            "ancient_greek_anarchism",
            [("anarchist_feminism", "Είναι αρκετά κοινό στις μέρες μας")],
            "Split at the exact topic reset from ancient Greek/anarchism essay to separate anarchist-feminism argument.",
        )
    )
    annotations.append(
        split_annotation(
            by_id["hplt::8_1.jsonl.zst::0ae7d8f6da5b3cbe4fe8838c5012a255"],
            reviewed_at,
            "milky_way_mass",
            [("sunlight_covid_mortality", "Οι περιοχές με περισσότερη λιακάδα")],
            "Split Milky Way mass article from attached sunlight/Covid mortality article; drop only the trailing read-more stub.",
            final_end_marker="ΔΙΑΒΑΣΤΕ ΕΠΙΣΗΣ",
        )
    )
    annotations.append(
        split_annotation(
            by_id["hplt::8_2.jsonl.zst::602020562af1a07a76f56d16bb091ca3"],
            reviewed_at,
            "shark_catch_south_crete",
            [("pensions_political_report", "Στο πλευρό των συνταξιούχων")],
            "Split shark catch article from attached pensions/political report; drop only the trailing read-more stub.",
            final_end_marker="ΔΙΑΒΑΣΤΕ ΕΠΙΣΗΣ",
        )
    )
    annotations.append(
        split_annotation(
            by_id["hplt::8_2.jsonl.zst::d57351d3fd730c12768bb29dd8aaaf87"],
            reviewed_at,
            "panelladikes_results",
            [("kart_accident_recovery", "Βελτιώνεται καθημερινά")],
            "Split Panelladikes results article from attached child/kart-accident recovery report; drop only the trailing read-more stub.",
            final_end_marker="Διαβάστε επίσης:",
        )
    )

    row = by_id["hplt::8_1.jsonl.zst::9d395ed79eb881f19515ab0c86d914a8"]
    start = must_find(row["full_text"], "4 σχόλια :", row["source_doc_id"])
    annotations.append(
        span_annotation(
            row,
            "trim_suffix",
            reviewed_at,
            [{"start": start, "end": len(row["full_text"]), "replacement": "", "review_marker": "Blogspot comment thread and comment-policy suffix"}],
            "Trim suffix at literal `4 σχόλια :`, preserving the coherent KTEL pet-transport article and removing extracted comments/form policy.",
            0.0,
        )
    )

    row = by_id["hplt::8_1.jsonl.zst::f8b508b024d90aef8450c532d3f0640e"]
    end = len("iSpeech.org ")
    annotations.append(
        span_annotation(
            row,
            "trim_prefix",
            reviewed_at,
            [{"start": 0, "end": end, "replacement": "", "review_marker": "leading iSpeech.org widget/source residue"}],
            "Trim only the leading `iSpeech.org ` prefix; preserve the coherent Easter-bonus article.",
            0.0,
        )
    )

    row = by_id["hplt::8_1.jsonl.zst::ecec64f371b7dd456beffe514fab99b0"]
    duplicate_sentence = "Αυτή είναι άλλη μια θαυμαστή αλήθεια που όμως έχει παρεξηγηθεί, διαστρεβλωθεί και διδαχτεί λάθος από αμόρφωτους «δασκάλους». "
    first = row["full_text"].find(duplicate_sentence)
    second = row["full_text"].find(duplicate_sentence, first + 1)
    if first < 0 or second < 0:
        raise RuntimeError("duplicate sentence not found twice for epistoligr row")
    annotations.append(
        span_annotation(
            row,
            "trim_span",
            reviewed_at,
            [{"start": second, "end": second + len(duplicate_sentence), "replacement": "", "review_marker": "second copy of duplicated opening sentence"}],
            "Remove only the second copy of the duplicated opening sentence; preserve the theology article body.",
            0.0,
        )
    )

    row = by_id["hplt::8_2.jsonl.zst::2ced04301afcca4d18d88d4113a6fd57"]
    start = must_find(row["full_text"], "Σχετικά άρθρα:", row["source_doc_id"])
    end = must_find(row["full_text"], "Συστατικά", row["source_doc_id"])
    annotations.append(
        span_annotation(
            row,
            "trim_span",
            reviewed_at,
            [{"start": start, "end": end, "replacement": "", "review_marker": "inline related-article link block before ingredients"}],
            "Remove the bounded `Σχετικά άρθρα` related-link block before the recipe ingredients; preserve the recipe text.",
            0.0,
        )
    )

    row = by_id["hplt::9_1.jsonl.zst::01b125fe9c6c5ca695127c0bff963a5d"]
    annotations.append(
        span_annotation(
            row,
            "normalize_or_trim_span",
            reviewed_at,
            shortcode_spans(row),
            "Normalize/remove WordPress shortcode residue (`nextpage`, column, and custom_list tags), preserving useful section titles and article text; remove empty trailing shortcode-only sections.",
            0.0,
        )
    )

    return annotations


def write_markdown(path: Path, annotations: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Remaining Boundary Annotations\n\n")
        out.write("Manual exact-boundary annotations for the final 9 rows from the boundary-spec pack. This is not a detector and does not mutate source HPLT rows.\n\n")
        out.write("## Summary\n\n")
        for key in ["review_pack", "annotation_rows", "by_action", "created_at_utc"]:
            out.write(f"- {key}: `{summary[key]}`\n")
        out.write("\n## Decisions\n\n")
        for row in annotations:
            out.write(f"- `{row['source_doc_id']}` `{row['boundary_review_action']}`: {row['boundary_notes']}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--timestamp", default=utc_timestamp())
    args = parser.parse_args()

    rows = read_jsonl(Path(args.review_pack))
    annotations = build_annotations(rows, args.timestamp)
    by_action: dict[str, int] = {}
    for row in annotations:
        by_action[row["boundary_review_action"]] = by_action.get(row["boundary_review_action"], 0) + 1

    output_prefix = Path(args.output_prefix)
    annotation_path = Path(str(output_prefix) + "_annotations.jsonl")
    summary_path = Path(str(output_prefix) + "_summary.json")
    markdown_path = Path(str(output_prefix) + ".md")
    summary = {
        "created_at_utc": args.timestamp,
        "schema_version": "hplt-remaining-boundary-bootstrap-v1",
        "policy_note": "Manual exact-boundary annotations only; source rows remain immutable.",
        "review_pack": args.review_pack,
        "annotations": str(annotation_path),
        "markdown": str(markdown_path),
        "annotation_rows": len(annotations),
        "by_action": by_action,
    }
    write_jsonl(annotation_path, annotations)
    write_json(summary_path, summary)
    write_markdown(markdown_path, annotations, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
