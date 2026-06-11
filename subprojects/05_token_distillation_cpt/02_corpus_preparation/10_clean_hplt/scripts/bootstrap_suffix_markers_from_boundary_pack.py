#!/usr/bin/env python3
"""Bootstrap exact suffix-trim annotations from reviewed marker decisions.

This script is intentionally a manual marker map. It does not discover suffixes
or infer document semantics; it only converts reviewed rows with literal,
auditable suffix-start markers into boundary annotations.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hplt-boundary-spec-annotation-template-v1"
DEFAULT_REVIEW_PACK = "reports/boundary_spec_review_pack_20260606T051032Z.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/boundary_manual_suffix_markers_20260606T051032Z"


SUFFIX_MARKERS: dict[str, dict[str, str]] = {
    "hplt::9_1.jsonl.zst::d8b382c04cea0a4ea2504fa24fa64f98": {"marker": "ΑΛΛΕΣ ΣΥΖΗΤΗΣΕΙΣ", "occurrence": "first"},
    "hplt::9_2.jsonl.zst::2a4526a1c9795ab16fad4c3f83697c5d": {"marker": "Μου αρέσει - Δεν μου αρέσει", "occurrence": "first"},
    "hplt::9_1.jsonl.zst::4199d635714d8937ee734a8eb5233108": {"marker": "- Για τα πάρτι της", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::10a91564da98cda8ed8ca6117fbd0b16": {"marker": "Πηγή Φωτογραφίας", "occurrence": "first"},
    "hplt::9_2.jsonl.zst::7136cbc846a202db44646a26bcab5008": {"marker": "Social Buttons", "occurrence": "first"},
    "hplt::9_1.jsonl.zst::d986070c6d5c85860d370778dfec602a": {"marker": "Leave a Reply", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::163744199c59a465209a7aa087f21c9b": {"marker": "POPULAR TAGS", "occurrence": "first"},
    "hplt::9_1.jsonl.zst::5fd226a84d1d14f39f9259c03303e372": {"marker": "ΑπάντησηΔιαγραφή", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::9d17327de19729e0204e5fe48dd97f7e": {"marker": "Διαβάστε ακόμη:", "occurrence": "first"},
    "hplt::9_1.jsonl.zst::e85adc18f92058619f80e21e1ef516bf": {"marker": "Tweet 0 σχόλια: Δημοσίευση σχολίου", "occurrence": "first"},
    "hplt::9_1.jsonl.zst::4ecd0d765430f2ca0c25c8ce47cc868b": {"marker": "Σχετικά Links:", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::d0becc728a880442df9e5e9c070d524e": {"marker": "Επιπρόσθετες Πληροφορίες", "occurrence": "last"},
    "hplt::8_1.jsonl.zst::a81a1521a14007557bf9dac92068610f": {"marker": "Δεν υπάρχουν σχόλια:", "occurrence": "first"},
    "hplt::9_2.jsonl.zst::21e5bc59c8529cf28426cf212456d44f": {"marker": "More Articles", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::34f4c3984e0dae4d50d0e925ba3b8a0f": {"marker": "Σχολιάστε", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::1d149e43ff6b51f76b72c0535e736f1d": {"marker": "Αφήστε ένα σχόλιο", "occurrence": "first"},
    "hplt::9_1.jsonl.zst::e8700b2eaf5e357c78ee890d5236e915": {"marker": "Σχολιάστε", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::11784c9c5e57f3ca60c0e5a955d1391e": {"marker": "Latest posts by", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::62b54117401cc554436bd88912b4f139": {"marker": "Share this!", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::19aa4d00f57655e276ef98b960a3c48d": {"marker": "Δεν υπάρχουν σχόλια", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::b6fdc123e8bb914b626112c49df534c5": {"marker": "Σχόλια Η efsyn.gr", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::f7142c25bb5a45baf5f503cc5cefce45": {"marker": "0 σχόλια: Speak up your mind", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::4d130d654435632f650f04808fd88452": {"marker": "Δεν υπάρχουν σχόλια: Δημοσίευση σχολίου", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::48ab28110e8d0d6a513a0aa5e6851c66": {"marker": "Δείτε περισσότερα άρθρα σχετικά", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::fd10b9b67daa393b6a7cdb6ef0db8f5f": {"marker": "Δεν υπάρχουν σχόλια: Δημοσίευση σχολίου", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::5c6ffdd77b92104e5a44901cadc1ebc8": {"marker": "Θα παρακαλούσαμε να μην χρησιμοποιείτε greeklish", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::2d3d1d19a582fc8e977734dbc7b2736c": {"marker": "0 Σχόλια", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::8cec0834e6cf296df61e7d031a09300d": {"marker": "Για όλες τις αγορές", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::238ce7aeb88a1f98a9e2d38ddbc3cc28": {"marker": "Leave a Reply", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::19754d5ed77c3ee82ceee0fcbca218d8": {"marker": "Ακολουθήστε την ΠΤΗΣΗ", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::4a9d8d59e9902239b19cfa6f2199cc21": {"marker": "Tο pronews.gr δημοσιεύει", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::a5d6bb0fa07ff92e4ac92b456ff5e611": {"marker": "Δεν υπάρχουν σχόλια: Δημοσίευση σχολίου", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::692fc5ec1b074e531b7dbfdbc993d406": {"marker": "Περιγραφή Αξιολογήσεις", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::31f811b43544d9686d6fb450642d526a": {"marker": "0 ΣΧΟΛΙΑ", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::3cc9e7882af71750f5f7437c07d04095": {"marker": "Δεν υπάρχουν σχόλια", "occurrence": "first"},
    "hplt::8_1.jsonl.zst::d61f1430e5e25aa4d41151f96917a3d3": {"marker": "premature signs of skin aging", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::b75c593357dfb84599c49a60b37fa87b": {"marker": "ΔΙΑΒΑΣΤΕ ΕΠΙΣΗΣ", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::ba1a5cb5ee6e48b7ff83e69681d255c2": {"marker": "Read more", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::e102ab041d27909bb737bbcc4e728378": {"marker": "Δείτε ακόμα", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::5900940f8d4dbe72d48731e906cf27f3": {"marker": "Το παρόν διαδικτυακό μέσο", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::589ecb9c1ec18c291b62fb4d06b4f0ee": {"marker": "- Επιλογές κράτησης:", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::d5d9bc450b9afc8c16f4c2ddf5f634f8": {"marker": "Δεν υπάρχουν σχόλια: Δημοσίευση σχολίου", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::e5f73f03a8a185d6986e12932601517d": {"marker": "Ακολουθείστε το UFight.gr", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::f1dcddcfb7888bfd8d181f211ca71eac": {"marker": "Ο έντυπος «πολίτης»", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::b150d816ef1f79edcee0d74566cb1697": {"marker": "Το pitsirikos.net χρειάζεται", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::a5ec9d5e76c1503a628869c840776422": {"marker": "Το παρόν διαδικτυακό μέσο", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::ee14bdd12bb75a73bc2731c517cc9b42": {"marker": "ΠΡΟΗΓΟΥΜΕΝΟ", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::f49d00e7eea02b8a9ff56d3cbd77763c": {"marker": "Και μη ξεχνάτε!", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::a78e73eb82683f88e9feaa93b32c56c9": {"marker": "Δεν υπάρχουν σχόλια: Δημοσίευση σχολίου", "occurrence": "first"},
    "hplt::8_2.jsonl.zst::b85d22f8423421d28f1785e0aadba559": {"marker": "Facebook Comments", "occurrence": "first"},
}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def marker_start(text: str, marker: str, occurrence: str) -> tuple[int, int]:
    positions: list[int] = []
    cursor = 0
    while True:
        index = text.find(marker, cursor)
        if index < 0:
            break
        positions.append(index)
        cursor = index + 1
    if not positions:
        raise ValueError(f"marker not found: {marker!r}")
    if occurrence == "first":
        return positions[0], len(positions)
    if occurrence == "last":
        return positions[-1], len(positions)
    raise ValueError(f"unknown occurrence: {occurrence!r}")


def annotation_row(source: dict[str, Any], marker: str, occurrence: str, timestamp: str) -> tuple[dict[str, Any], dict[str, Any]]:
    text = compact_text(source.get("full_text"))
    start, marker_count = marker_start(text, marker, occurrence)
    end = len(text)
    before_chars = int(source.get("chars_before") or len(text))
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "source_doc_id": source.get("source_doc_id"),
            "parent_source_doc_id": source.get("parent_source_doc_id") or source.get("source_doc_id"),
            "source_review_rows_count": source.get("source_review_rows_count"),
            "source_annotation_files": source.get("source_annotation_files") or [],
            "host": source.get("host"),
            "url": source.get("url"),
            "quality_bin": source.get("quality_bin"),
            "text_sha256_before": source.get("full_text_sha256") or source.get("text_sha256_before"),
            "chars_before": before_chars,
            "tokens_before": source.get("tokens_before"),
            "error_type_ids": source.get("error_type_ids") or [],
            "reviewed_action": "trim_suffix",
            "boundary_review_action": "trim_suffix",
            "boundary_review_status": "reviewed_accept",
            "span_ranges": [
                {
                    "start": start,
                    "end": end,
                    "replacement": "",
                    "review_marker": marker,
                    "marker_occurrence": occurrence,
                    "marker_count": marker_count,
                }
            ],
            "split_parts": [],
            "dropped_span_ranges": [],
            "chars_after": start,
            "chars_removed": end - start,
            "tokens_after": None,
            "tokens_removed": None,
            "good_text_loss_estimate": 0.0,
            "boundary_notes": (
                "Exact suffix trim at reviewed literal marker "
                f"{marker!r} ({occurrence} occurrence). Prior review: "
                + compact_text(source.get("review_span_or_split_notes") or source.get("review_notes"))
            ),
            "reviewer": "codex",
            "reviewed_at_utc": timestamp,
            "boundary_instructions": [
                "Reviewed exact suffix marker; source rows remain immutable.",
                "Changed text must be emitted only as duplicate/shadow records.",
            ],
        },
        {
            "source_doc_id": source.get("source_doc_id"),
            "host": source.get("host"),
            "marker": marker,
            "occurrence": occurrence,
            "marker_count": marker_count,
            "start": start,
            "end": end,
            "chars_removed": end - start,
            "start_percent": 100.0 * start / max(1, end),
            "prefix_window": text[max(0, start - 120):start],
            "suffix_start_window": text[start:start + 220],
        },
    )


def write_markdown(path: Path, summary: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Exact Suffix-Marker Boundary Bootstrap\n\n")
        out.write("This report records reviewed suffix trims created from explicit literal markers. It is not a detector and does not infer rows outside the manual map.\n\n")
        out.write("## Summary\n\n")
        for key in ["review_pack", "selected_rows", "skipped_rows", "created_at_utc"]:
            out.write(f"- {key}: `{summary.get(key)}`\n")
        out.write("\n## Evidence\n\n")
        for item in evidence:
            out.write(f"### `{item['source_doc_id']}`\n\n")
            out.write(f"- host: `{item.get('host')}`\n")
            out.write(f"- marker: `{item['marker']}` ({item['occurrence']}, count `{item['marker_count']}`)\n")
            out.write(f"- start/end/chars_removed: `{item['start']}` / `{item['end']}` / `{item['chars_removed']}`\n")
            out.write(f"- start_percent: `{item['start_percent']:.2f}`\n\n")
            out.write("```text\n")
            out.write((item["prefix_window"] + ">>>" + item["suffix_start_window"]).replace("\r", ""))
            out.write("\n```\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", default=DEFAULT_REVIEW_PACK)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    args = parser.parse_args()

    review_pack = Path(args.review_pack)
    source_rows = {compact_text(row.get("source_doc_id")): row for row in read_jsonl(review_pack)}
    missing_sources = sorted(set(SUFFIX_MARKERS) - set(source_rows))
    if missing_sources:
        raise RuntimeError("Marker map has source IDs not present in review pack: " + ", ".join(missing_sources))

    annotations: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for source_doc_id, spec in SUFFIX_MARKERS.items():
        row = source_rows[source_doc_id]
        if compact_text(row.get("action")) != "trim_suffix":
            raise RuntimeError(f"{source_doc_id} is not a trim_suffix row")
        annotation, item = annotation_row(row, spec["marker"], spec["occurrence"], args.timestamp)
        annotations.append(annotation)
        evidence.append(item)

    prefix = Path(args.output_prefix)
    annotation_path = Path(str(prefix) + "_annotations.jsonl")
    summary_path = Path(str(prefix) + "_summary.json")
    markdown_path = Path(str(prefix) + ".md")
    evidence_path = Path(str(prefix) + "_evidence.jsonl")

    write_jsonl(annotation_path, annotations)
    write_jsonl(evidence_path, evidence)
    summary = {
        "schema_version": "hplt-suffix-marker-boundary-bootstrap-v1",
        "created_at_utc": args.timestamp,
        "review_pack": str(review_pack),
        "annotation_jsonl": str(annotation_path),
        "evidence_jsonl": str(evidence_path),
        "summary_json": str(summary_path),
        "markdown": str(markdown_path),
        "selected_rows": len(annotations),
        "skipped_rows": len([row for row in source_rows.values() if compact_text(row.get("action")) == "trim_suffix"]) - len(annotations),
        "selected_by_action": dict(collections.Counter(row.get("boundary_review_action") for row in annotations).most_common()),
        "chars_removed_total": sum(int(row["span_ranges"][0]["end"] - row["span_ranges"][0]["start"]) for row in annotations),
        "policy_note": "Exact suffix-marker annotation bootstrap only. Source HPLT rows stay immutable; materialization must run separately on Clariden CPU-only xfer.",
    }
    write_json(summary_path, summary)
    write_markdown(markdown_path, summary, evidence)
    print(json.dumps({"annotations": str(annotation_path), "summary": str(summary_path), "evidence": str(evidence_path), "selected_rows": len(annotations)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
