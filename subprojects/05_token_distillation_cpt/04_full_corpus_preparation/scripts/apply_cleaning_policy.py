#!/usr/bin/env python3
"""Apply approved source cleaning, document actions, PII and optional spans.

The materializer is fail-closed: candidate sources require an approved source
admission record, and structural spans are ignored unless explicitly requested
and the tracked policy enables them.  Every mutation is recorded in per-document
and aggregate token ledgers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from full_corpus_io import canonical_schema, normalize_text, read_json_object, sha256_text, strip_html_markup, token_count
from greek_pii import mask_greek_identifiers


HERE = Path(__file__).resolve().parents[1]
PII_DIR = HERE.parent / "02_corpus_preparation" / "40_anonymize" / "scripts"
if str(PII_DIR) not in sys.path:
    sys.path.insert(0, str(PII_DIR))
from pii_masker import mask as mask_apertus_pii  # type: ignore  # noqa: E402


WEB_PROFILES = {"news", "web_articles"}
PERSONNEL = re.compile(
    r"(?i)(?:πατρώνυμο|μητρώνυμο|αριθμός\s+ταυτότητας|Α\.?Δ\.?Τ\.?|"
    r"πίνακας\s+(?:κατάταξης|υποψηφίων|προσληπτέων))"
)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def cleaned_schema():
    import pyarrow as pa

    return pa.schema(
        [
            *canonical_schema(),
            ("cleaned_text_sha256", pa.string()),
            ("cleaning_trace_json", pa.string()),
            ("pii_by_type_json", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("eligible_for_redistribution", pa.bool_()),
        ]
    )


def ledger_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("action", pa.string()),
            ("reasons_json", pa.string()),
            ("tokens_normalized", pa.int64()),
            ("tokens_source_cleaned", pa.int64()),
            ("tokens_pii_masked", pa.int64()),
            ("tokens_structural_cleaned", pa.int64()),
            ("tokens_final", pa.int64()),
            ("characters_normalized", pa.int64()),
            ("characters_final", pa.int64()),
            ("pii_by_type_json", pa.string()),
        ]
    )


def load_admission(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    payload = read_json_object(path)
    schema = payload.get("schema_version")
    rows = payload.get("sources", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path}: sources must be a list")
    if schema == "source_quality_review_admission_v1":
        key = "source_dataset"
        pending = int(payload.get("pending_adjudications", 0))
        unresolved = any(row.get("decision") == "pending_adjudication" for row in rows)
        status = "pending" if pending or unresolved else "approved"
    elif schema == "full_cpt_source_admission_v1":
        # Backward-compatible bounded-smoke contract. Production review output
        # uses exact source_dataset keys through source_quality_review_admission_v1.
        key = "source_id"
        status = str(payload.get("status", "pending"))
    else:
        raise ValueError(f"{path}: unsupported source-admission schema")
    mapping = {str(row[key]): dict(row) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{path}: duplicate {key} values")
    return status, mapping


def load_eligibility_policy(path: Path) -> dict[str, dict[str, bool]]:
    payload = read_json_object(path)
    if payload.get("schema_version") != "full_cpt_training_eligibility_policy_v1":
        raise ValueError(f"{path}: unsupported training-eligibility policy")
    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"{path}: categories must be a non-empty mapping")
    result: dict[str, dict[str, bool]] = {}
    for name, row in categories.items():
        if not isinstance(row, dict) or set(row) < {"training_eligible", "redistribution_eligible"}:
            raise ValueError(f"{path}: invalid eligibility category {name!r}")
        result[str(name)] = {
            "training_eligible": bool(row["training_eligible"]),
            "redistribution_eligible": bool(row["redistribution_eligible"]),
        }
    return result


def load_document_actions(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stable_uid = str(row.get("stable_uid", ""))
                if not stable_uid:
                    raise ValueError(f"{path}:{line_number}: action lacks stable_uid")
                if row.get("action") not in {"drop", "quarantine"}:
                    raise ValueError(f"{path}:{line_number}: unsupported action")
                actions[stable_uid].append(row)
    return actions


def load_spans(paths: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    spans: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source = str(row.get("acquisition_source_id") or row.get("source") or "")
                doc_id = str(row.get("stable_uid") or row.get("doc_id") or row.get("id") or "")
                if not source or not doc_id:
                    raise ValueError(f"{path}:{line_number}: span lacks source/document identity")
                spans[(source, doc_id)].append(row)
    return spans


def apply_spans(text: str, spans: Iterable[dict[str, Any]]) -> tuple[str, list[str]]:
    intervals: list[tuple[int, int, str]] = []
    for span in spans:
        expected_hash = span.get("input_text_sha256")
        if expected_hash != sha256_text(text):
            raise ValueError("structural span is not bound to the exact post-PII input text")
        start, end = int(span["char_start"]), int(span["char_end"])
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"invalid structural span {start}:{end} for text length {len(text)}")
        intervals.append((start, end, str(span.get("rule_id") or span.get("kind") or "structural_span")))
    intervals.sort()
    merged: list[tuple[int, int, list[str]]] = []
    for start, end, reason in intervals:
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end, [*merged[-1][2], reason])
            else:
                merged[-1][2].append(reason)
        else:
            merged.append((start, end, [reason]))
    output = text
    for start, end, _ in reversed(merged):
        output = output[:start] + "\n\n" + output[end:]
    return normalize_text(output), [reason for _, _, reasons in merged for reason in reasons]


def recursive_private_data(metadata: Any) -> bool:
    if isinstance(metadata, str):
        try:
            return recursive_private_data(json.loads(metadata))
        except (json.JSONDecodeError, TypeError):
            return False
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if str(key).casefold() == "privatedata" and value in {True, 1, "true", "True", "1"}:
                return True
            if recursive_private_data(value):
                return True
    if isinstance(metadata, list):
        return any(recursive_private_data(item) for item in metadata)
    return False


def admission_for(
    source_id: str,
    source_dataset: str,
    source_role: str,
    admissions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if source_id == "nanochat_base" or source_role == "base":
        return {"decision": "include"}
    # Production admissions are grouped by the exact upstream source_dataset.
    # The source-id fallback exists only for the legacy smoke-test schema.
    if source_dataset in admissions:
        return admissions[source_dataset]
    return admissions.get(
        source_id,
        {"decision": "pending"},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument(
        "--eligibility-policy",
        type=Path,
        default=HERE / "configs" / "training_eligibility_policy.json",
    )
    parser.add_argument("--cleaning-policy", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--document-actions", action="append", type=Path, default=[])
    parser.add_argument("--structural-spans", action="append", type=Path, default=[])
    parser.add_argument("--apply-structural", action="store_true")
    parser.add_argument("--allow-pending-admission", action="store_true", help="bounded smoke only")
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    status, admissions = load_admission(args.source_admission)
    if status != "approved" and not args.allow_pending_admission:
        raise ValueError("source admission must be approved for materialization")
    policy = read_json_object(args.cleaning_policy)
    eligibility_policy = load_eligibility_policy(args.eligibility_policy)
    structural_enabled = (
        bool(args.apply_structural)
        and policy.get("status") == "approved"
        and bool(policy.get("structural", {}).get("bibliography", {}).get("enabled_for_materialization"))
        and bool(policy.get("structural", {}).get("toc", {}).get("enabled_for_materialization"))
    )
    if args.apply_structural and not structural_enabled:
        raise ValueError("structural cleaning requested but the tracked policy is not approved/enabled")
    from tokenizers import Tokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    actions = load_document_actions(args.document_actions)
    spans = load_spans(args.structural_spans) if structural_enabled else {}
    counters: Counter[str] = Counter()
    source_counters: dict[str, Counter[str]] = defaultdict(Counter)
    output_files: list[dict[str, Any]] = []
    for input_path in sorted(args.input.rglob("*.parquet")):
        relative = input_path.relative_to(args.input)
        table = pq.read_table(input_path)
        cleaned_rows: list[dict[str, Any]] = []
        quarantine_rows: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        for row in table.to_pylist():
            source_id = str(row["acquisition_source_id"])
            admission = admission_for(
                source_id,
                str(row["source_dataset"]),
                str(row["source_role"]),
                admissions,
            )
            decision = str(admission.get("decision", "pending"))
            eligibility_name = str(row.get("training_eligibility") or "inherited_base")
            if eligibility_name not in eligibility_policy:
                raise ValueError(f"unreviewed training_eligibility category: {eligibility_name!r}")
            eligibility = eligibility_policy[eligibility_name]
            action = "keep"
            reasons: list[str] = []
            if decision not in {"include", "include_after_cleaning"}:
                action = "drop"
                reasons.append(f"source_admission:{decision}")
            for document_action in actions.get(str(row["stable_uid"]), []):
                candidate = str(document_action["action"])
                if candidate == "drop":
                    action = "drop"
                elif candidate == "quarantine" and action == "keep":
                    action = "quarantine"
                reasons.append(str(document_action.get("reason") or document_action.get("rule_id") or candidate))
            normalized = str(row["text"] or "")
            if not normalized:
                action = "drop"
                reasons.append("empty_document")
            if row.get("greek_badness_score") is not None and float(row["greek_badness_score"]) > 60:
                action = "drop"
                reasons.append("greek_badness_score_gt_60")
            if bool(row.get("needs_ocr")) and str(row.get("source_family_id")) == "openarchives":
                action = "drop"
                reasons.append("openarchives_needs_ocr")
            try:
                metadata = json.loads(str(row.get("source_metadata_json") or "{}"))
            except json.JSONDecodeError:
                metadata = {}
            if source_id == "diavgeia" and recursive_private_data(metadata):
                action = "drop"
                reasons.append("diavgeia_privateData_true")
            source_cleaned = normalized
            trace: list[dict[str, Any]] = []
            if str(row.get("cleaning_profile")) in WEB_PROFILES and "<" in source_cleaned:
                candidate, removed_tags = strip_html_markup(source_cleaned)
                if removed_tags and candidate:
                    trace.append({"rule": "html_markup_v1", "matches": removed_tags})
                    source_cleaned = candidate
            generic_masked, generic_counts = mask_apertus_pii(source_cleaned)
            pii_masked, greek_counts = mask_greek_identifiers(generic_masked)
            pii_counts = {**generic_counts}
            for key, value in greek_counts.items():
                pii_counts[key] = pii_counts.get(key, 0) + value
            if sum(pii_counts.values()):
                trace.append({"rule": "high_confidence_direct_pii_v1", "matches": sum(pii_counts.values())})
            source_span_key = (source_id, str(row["stable_uid"]))
            doc_span_key = (source_id, str(row["source_doc_id"]))
            selected_spans = spans.get(source_span_key, spans.get(doc_span_key, []))
            structural_cleaned = pii_masked
            if selected_spans:
                structural_cleaned, structural_rules = apply_spans(pii_masked, selected_spans)
                trace.extend({"rule": rule, "matches": 1} for rule in structural_rules)
            if source_id == "diavgeia" and PERSONNEL.search(normalized) and sum(pii_counts.values()) >= 3:
                action = "quarantine"
                reasons.append("diavgeia_pii_heavy_personnel_table")
            final_text = normalize_text(structural_cleaned)
            if not final_text and action == "keep":
                action = "quarantine"
                reasons.append("cleaning_emptied_document")
            counts = {
                "tokens_normalized": token_count(tokenizer, normalized),
                "tokens_source_cleaned": token_count(tokenizer, source_cleaned),
                "tokens_pii_masked": token_count(tokenizer, pii_masked),
                "tokens_structural_cleaned": token_count(tokenizer, structural_cleaned),
                "tokens_final": token_count(tokenizer, final_text) if action == "keep" else 0,
            }
            ledger_rows.append(
                {
                    "stable_uid": str(row["stable_uid"]),
                    "acquisition_source_id": source_id,
                    "source_dataset": str(row["source_dataset"]),
                    "source_doc_id": str(row["source_doc_id"]),
                    "action": action,
                    "reasons_json": json.dumps(sorted(set(reasons)), ensure_ascii=False),
                    **counts,
                    "characters_normalized": len(normalized),
                    "characters_final": len(final_text) if action == "keep" else 0,
                    "pii_by_type_json": json.dumps(pii_counts, sort_keys=True),
                }
            )
            counters[f"action:{action}"] += 1
            source_counters[source_id][f"action:{action}"] += 1
            for key, value in counts.items():
                counters[key] += value
                source_counters[source_id][key] += value
            final_row = {
                **row,
                "text": final_text,
                "cleaned_text_sha256": sha256_text(final_text),
                "cleaning_trace_json": json.dumps(trace, ensure_ascii=False, sort_keys=True),
                "pii_by_type_json": json.dumps(pii_counts, sort_keys=True),
                # include_after_cleaning is deliberately not promotable until a
                # fresh post-clean packet turns it into an include decision.
                "eligible_for_training": bool(
                    action == "keep"
                    and decision == "include"
                    and eligibility["training_eligible"]
                ),
                "eligible_for_redistribution": bool(
                    action == "keep"
                    and decision == "include"
                    and eligibility["redistribution_eligible"]
                ),
            }
            if action == "keep":
                cleaned_rows.append(final_row)
            elif action == "quarantine":
                quarantine_rows.append(final_row)
        output_path = args.output / relative
        ledger_path = args.ledger / relative
        quarantine_path = args.quarantine / relative
        for destination, rows, schema in (
            (output_path, cleaned_rows, cleaned_schema()),
            (ledger_path, ledger_rows, ledger_schema()),
            (quarantine_path, quarantine_rows, cleaned_schema()),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
            os.replace(temporary, destination)
        output_files.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "ledger": str(ledger_path),
                "quarantine": str(quarantine_path),
                "input_rows": table.num_rows,
                "kept_rows": len(cleaned_rows),
                "quarantined_rows": len(quarantine_rows),
            }
        )
    payload = {
        "schema_version": "full_cpt_cleaning_manifest_v1",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "tokenizer_json": str(args.tokenizer_json.resolve()),
        "tokenizer_sha256": sha256_text(args.tokenizer_json.read_text(encoding="utf-8")),
        "source_admission": str(args.source_admission.resolve()),
        "eligibility_policy": str(args.eligibility_policy.resolve()),
        "cleaning_policy": str(args.cleaning_policy.resolve()),
        "structural_applied": structural_enabled,
        "counts": dict(counters),
        "per_source": {key: dict(value) for key, value in sorted(source_counters.items())},
        "files": output_files,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(counters)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
