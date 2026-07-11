#!/usr/bin/env python3
"""Source-aware, non-destructive quality/PII/template profiler for Parquet inputs.

Workers open Parquet row groups themselves, so large text is not copied from a
reader process through multiprocessing queues. The output is an audit, never a
cleaned corpus. Exact/near deduplication remains a separate authoritative phase.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Sequence


EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IBAN = re.compile(r"(?i)\bGR\s*\d{2}(?:[\s-]*\d){23}\b")
IPV4 = re.compile(r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)")
IPV6 = re.compile(r"(?i)(?<![0-9A-F:])(?:[0-9A-F]{1,4}:){2,7}[0-9A-F]{0,4}(?![0-9A-F:])")
PHONE = re.compile(r"(?<!\d)(?:\+?30[\s.-]*)?(?:2\d{9}|69\d{8})(?!\d)")
AFM_LABEL = re.compile(r"(?i)(?:Α\.?\s*Φ\.?\s*Μ\.?|ΑΦΜ|tax\s*id)\s*[:#-]?\s*\d{9}\b")
AMKA_LABEL = re.compile(r"(?i)\bΑ\.?\s*Μ\.?\s*Κ\.?\s*Α\.?\s*[:#-]?\s*\d{11}\b")
IDENTITY_LABEL = re.compile(
    r"(?i)(?:αριθ(?:μός|μ\.)?\s*(?:δελτίου\s*)?(?:ταυτότητας|διαβατηρίου)|"
    r"Α\.?\s*Δ\.?\s*Τ\.?|identity\s*(?:card|number)|passport)\s*[:#-]?\s*[A-ZΑ-Ω]{0,3}\s*\d{5,10}"
)
PERSONNEL_CUE = re.compile(
    r"(?i)(?:πατρώνυμο|μητρώνυμο|αριθμός\s+ταυτότητας|Α\.?Δ\.?Τ\.?|πίνακας\s+(?:κατάταξης|υποψηφίων|προσληπτέων))"
)
BIB_HEADER = re.compile(r"(?im)^\s{0,3}#{0,6}\s*(?:βιβλιογραφ(?:ία|ικές)|αναφορές|references|bibliography)\s*$")
TOC_HEADER = re.compile(r"(?im)^\s{0,3}#{0,6}\s*(?:περιεχόμενα|περιεχομενα|table\s+of\s+contents|contents)\s*$")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DIGITAL_GOVERNANCE = re.compile(
    r"(?is)(?:Ministry\s+of\s+Digital\s+Governance|Digitally\s+signed\s+by\s+Ministry\s+of\s+Digital\s+Governance)"
)
ADA = re.compile(r"(?i)\b[0-9A-ZΑ-Ω]{4,12}-[0-9A-ZΑ-Ω]{3,12}\b")
ADA_LINE = re.compile(r"(?i)^\s*ΑΔΑ:\s*[0-9A-ZΑ-Ω]{4,12}-[0-9A-ZΑ-Ω]{3,12}\s*$")
URL = re.compile(r"(?i)https?://\S+|www\.\S+")
VARIABLE_NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
WHITESPACE = re.compile(r"\s+")

PII_PATTERNS = {
    "email": EMAIL,
    "iban": IBAN,
    "ipv4": IPV4,
    "ipv6": IPV6,
    "phone": PHONE,
    "afm_labelled": AFM_LABEL,
    "amka_labelled": AMKA_LABEL,
    "identity_labelled": IDENTITY_LABEL,
}

LENGTH_BINS = (0, 100, 500, 2_000, 10_000, 50_000, 250_000, 1_000_000, 5_000_000)


def expand_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(path.rglob("*.parquet"))
        elif any(char in value for char in "*?["):
            paths.extend(Path(item) for item in glob.glob(value, recursive=True))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(value)
    unique = sorted({path.resolve() for path in paths if path.suffix == ".parquet"})
    if not unique:
        raise ValueError("no Parquet inputs resolved")
    return unique


def length_bin(length: int) -> str:
    for lower, upper in zip(LENGTH_BINS, LENGTH_BINS[1:]):
        if lower <= length < upper:
            return f"{lower}-{upper - 1}"
    return f">={LENGTH_BINS[-1]}"


def parse_metadata(value: object) -> tuple[dict, bool]:
    if value is None:
        return {}, False
    if isinstance(value, dict):
        return value, False
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return (parsed, False) if isinstance(parsed, dict) else ({}, True)
        except json.JSONDecodeError:
            return {}, True
    return {}, True


def normalized_template(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected = lines[:2] + (lines[-5:] if len(lines) > 2 else [])
    value = "\n".join(selected).lower()
    value = URL.sub("<url>", value)
    value = EMAIL.sub("<email>", value)
    value = ADA.sub("<ada>", value)
    value = VARIABLE_NUMBER.sub("<n>", value)
    return WHITESPACE.sub(" ", value).strip()[:12_000]


def line_quality(text: str) -> tuple[float, float, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0, 0.0, 0
    unique_fraction = len(set(lines)) / len(lines)
    one_token_fraction = sum(1 for line in lines if len(line.split()) <= 1) / len(lines)
    markdown_table_lines = sum(1 for line in lines if line.startswith("|") and line.endswith("|"))
    return unique_fraction, one_token_fraction, markdown_table_lines


def diavgeia_overlay_spans(text: str, doc_id: str, source: str = "diavgeia") -> list[dict[str, object]]:
    """Return conservative, reversible Diavgeia boilerplate candidates.

    Offsets use Python Unicode character indices, which match the Rust detector's
    Unicode-scalar offsets. The signing block is accepted only when the complete
    Ministry/Digital Governance signature template is present in a short window.
    """
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    spans: list[dict[str, object]] = []
    line_index = 0
    while line_index < len(lines):
        stripped = lines[line_index].strip()
        folded = stripped.casefold()
        if folded == "ministry of" or folded.startswith("ministry of digital governance"):
            for end_index in range(line_index, min(len(lines), line_index + 14)):
                if not lines[end_index].strip().casefold().startswith("location: athens"):
                    continue
                normalized = WHITESPACE.sub(" ", " ".join(lines[line_index : end_index + 1])).casefold()
                required = (
                    "ministry of digital governance",
                    "digitally signed by ministry of digital governance",
                    "date:",
                    "reason:",
                    "location: athens",
                )
                if all(value in normalized for value in required):
                    spans.append(
                        {
                            "schema_version": "cleaning_span_v1",
                            "source": source,
                            "doc_id": doc_id,
                            "kind": "diavgeia_signing_block_span",
                            "rule_id": "diavgeia.digital_signature_footer.v1",
                            "char_start": offsets[line_index],
                            "char_end": offsets[end_index] + len(lines[end_index]),
                            "line_start": line_index,
                            "line_end": end_index,
                            "trigger": "exact_digital_governance_signature_block",
                            "gated_by": "complete_template_window",
                        }
                    )
                    line_index = end_index
                    break
        line_index += 1

    for index, line in enumerate(lines):
        if ADA_LINE.fullmatch(line.strip()):
            spans.append(
                {
                    "schema_version": "cleaning_span_v1",
                    "source": source,
                    "doc_id": doc_id,
                    "kind": "diavgeia_ada_stamp_span",
                    "rule_id": "diavgeia.ada_watermark.v1",
                    "char_start": offsets[index],
                    "char_end": offsets[index] + len(line),
                    "line_start": index,
                    "line_end": index,
                    "trigger": "isolated_ada_stamp_line",
                    "gated_by": "full_line_match",
                }
            )
    spans.sort(key=lambda row: (int(row["char_start"]), int(row["char_end"]), str(row["kind"])))
    return spans


def profile_row_group(task: tuple[str, int, str, str, str | None, int, str]) -> dict:
    path_string, row_group, text_column, id_column, metadata_column, template_limit, source = task
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path_string)
    columns = [text_column, id_column]
    if metadata_column:
        columns.append(metadata_column)
    table = parquet.read_row_group(row_group, columns=columns)
    values = table.to_pydict()
    metrics: Counter[str] = Counter()
    pii_documents: Counter[str] = Counter()
    pii_matches: Counter[str] = Counter()
    decision_types: Counter[str] = Counter()
    organizations: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    template_examples: dict[str, dict[str, str]] = {}
    candidate_spans: list[dict[str, object]] = []
    candidate_actions: list[dict[str, object]] = []

    for index, raw_text in enumerate(values[text_column]):
        doc_id = str(values[id_column][index])
        text = "" if raw_text is None else str(raw_text)
        metadata, metadata_error = parse_metadata(values[metadata_column][index] if metadata_column else None)
        if metadata_error:
            metrics["metadata_parse_errors"] += 1
        metrics["rows"] += 1
        metrics["characters"] += len(text)
        metrics["bytes_utf8"] += len(text.encode("utf-8"))
        metrics[f"length:{length_bin(len(text))}"] += 1
        if not text.strip():
            metrics["empty_documents"] += 1
            continue
        metrics["nonempty_documents"] += 1
        replacement_count = text.count("\ufffd")
        control_count = len(CONTROL.findall(text))
        if replacement_count:
            metrics["replacement_char_documents"] += 1
            metrics["replacement_chars"] += replacement_count
        if control_count:
            metrics["control_char_documents"] += 1
            metrics["control_chars"] += control_count
        if DIGITAL_GOVERNANCE.search(text):
            metrics["digital_governance_footer_documents"] += 1
        if BIB_HEADER.search(text):
            metrics["bibliography_header_documents"] += 1
        if TOC_HEADER.search(text):
            metrics["toc_header_documents"] += 1
        if PERSONNEL_CUE.search(text):
            metrics["personnel_cue_documents"] += 1
        unique_fraction, one_token_fraction, table_lines = line_quality(text)
        if unique_fraction < 0.50:
            metrics["low_unique_line_fraction_documents"] += 1
        if one_token_fraction > 0.50:
            metrics["one_token_per_line_documents"] += 1
        if table_lines >= 20:
            metrics["large_markdown_table_documents"] += 1

        for name, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                pii_documents[name] += 1
                pii_matches[name] += len(matches)

        private_data = metadata.get("privateData")
        private_data_true = private_data is True or (
            isinstance(private_data, str) and private_data.lower() == "true"
        )
        if private_data_true:
            metrics["private_data_true_documents"] += 1
        corrected_version = metadata.get("correctedVersionId") not in (None, "", 0, False)
        if corrected_version:
            metrics["corrected_version_documents"] += 1
        decision_type = metadata.get("decisionTypeId")
        organization = metadata.get("organizationId")
        if decision_type not in (None, ""):
            decision_types[str(decision_type)] += 1
        if organization not in (None, ""):
            organizations[str(organization)] += 1

        template = normalized_template(text)
        if template:
            signature = hashlib.sha256(template.encode("utf-8")).hexdigest()
            templates[signature] += 1
            template_examples.setdefault(
                signature,
                {"doc_id": doc_id, "template": template[:2000], "input_file": path_string},
            )

        spans = diavgeia_overlay_spans(text, doc_id, source) if source == "diavgeia" else []
        if spans:
            candidate_spans.extend(spans)
            metrics["diavgeia_candidate_spans"] += len(spans)
            for span in spans:
                metrics[f"candidate_rule:{span['rule_id']}"] += 1

        action_flags = {
            "private_data_true": private_data_true,
            "personnel_table_cue": bool(PERSONNEL_CUE.search(text)),
            "corrected_version": corrected_version,
            "structured_pii": sorted(name for name, pattern in PII_PATTERNS.items() if pattern.search(text)),
        }
        if source == "diavgeia" and any(
            value if isinstance(value, bool) else bool(value)
            for value in action_flags.values()
        ):
            candidate_actions.append(
                {
                    "schema_version": "document_action_candidate_v1",
                    "source": source,
                    "doc_id": doc_id,
                    "input_file": path_string,
                    **action_flags,
                    "decision": "audit_only",
                }
            )

    keep = {signature for signature, _count in templates.most_common(template_limit)}
    return {
        "path": path_string,
        "row_group": row_group,
        "metrics": dict(metrics),
        "pii_documents": dict(pii_documents),
        "pii_matches": dict(pii_matches),
        "decision_types": dict(decision_types),
        "organizations": dict(organizations),
        "templates": {signature: templates[signature] for signature in keep},
        "template_examples": {signature: template_examples[signature] for signature in keep},
        "candidate_spans": candidate_spans,
        "candidate_actions": candidate_actions,
    }


def merge_counter(target: Counter[str], values: dict[str, int]) -> None:
    target.update({str(key): int(value) for key, value in values.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--metadata-column")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--max-row-groups", type=int, default=0)
    parser.add_argument("--template-candidates-per-row-group", type=int, default=256)
    parser.add_argument("--top-templates", type=int, default=200)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on Clariden
        raise RuntimeError("install pyarrow in the Clariden environment") from exc

    inputs = expand_inputs(args.input)
    tasks: list[tuple[str, int, str, str, str | None, int, str]] = []
    inventory: list[dict[str, object]] = []
    for path in inputs:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        required = {args.text_column, args.id_column}
        if args.metadata_column:
            required.add(args.metadata_column)
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; has {sorted(names)}")
        inventory.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
            }
        )
        for row_group in range(parquet.num_row_groups):
            tasks.append(
                (
                    str(path),
                    row_group,
                    args.text_column,
                    args.id_column,
                    args.metadata_column,
                    args.template_candidates_per_row_group,
                    args.source,
                )
            )
    if args.max_row_groups:
        tasks = tasks[: args.max_row_groups]

    metrics: Counter[str] = Counter()
    pii_documents: Counter[str] = Counter()
    pii_matches: Counter[str] = Counter()
    decision_types: Counter[str] = Counter()
    organizations: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    template_examples: dict[str, dict[str, str]] = {}
    candidate_spans_path = args.output_dir / "cleaning_span_candidates.jsonl"
    candidate_actions_path = args.output_dir / "document_action_candidates.jsonl"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    span_handle = candidate_spans_path.open("w", encoding="utf-8")
    action_handle = candidate_actions_path.open("w", encoding="utf-8")

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            for result in executor.map(profile_row_group, tasks, chunksize=1):
                merge_counter(metrics, result["metrics"])
                merge_counter(pii_documents, result["pii_documents"])
                merge_counter(pii_matches, result["pii_matches"])
                merge_counter(decision_types, result["decision_types"])
                merge_counter(organizations, result["organizations"])
                merge_counter(templates, result["templates"])
                for signature, example in result["template_examples"].items():
                    template_examples.setdefault(signature, example)
                for row in result["candidate_spans"]:
                    span_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                for row in result["candidate_actions"]:
                    action_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        span_handle.close()
        action_handle.close()

    top_templates = templates.most_common(args.top_templates)
    template_path = args.output_dir / "source_quality_top_templates.csv"
    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signature", "count", "doc_id", "input_file", "template"])
        writer.writeheader()
        for signature, count in top_templates:
            writer.writerow({"signature": signature, "count": count, **template_examples[signature]})

    report = {
        "schema_version": "source_quality_profile_v1",
        "source": args.source,
        "mode": "audit_only",
        "inputs": inventory,
        "row_groups_profiled": len(tasks),
        "workers": min(args.workers, len(tasks)),
        "metrics": dict(sorted(metrics.items())),
        "pii_document_counts": dict(sorted(pii_documents.items())),
        "pii_match_counts": dict(sorted(pii_matches.items())),
        "decision_types_top": decision_types.most_common(100),
        "organizations_top": organizations.most_common(100),
        "template_note": (
            "Template candidates are row-group-local top signatures merged globally; use the authoritative "
            "dedup/template phase for complete cluster membership."
        ),
        "top_templates": [
            {"signature": signature, "count": count, **template_examples[signature]}
            for signature, count in top_templates[:20]
        ],
        "outputs": {
            "top_templates_csv": str(template_path),
            "cleaning_span_candidates_jsonl": str(candidate_spans_path),
            "document_action_candidates_jsonl": str(candidate_actions_path),
        },
    }
    report_path = args.output_dir / "source_quality_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"ok": True, "source": args.source, "summary": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
