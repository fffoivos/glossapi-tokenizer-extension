#!/usr/bin/env python3
"""Build the immutable post-source-clean/post-PII Stage 50 corpus.

The implementation is deliberately structural-no-op.  It streams Parquet row
groups, tokenizes changed text variants in batches, and processes a bounded
number of files in parallel on CPU nodes.  A per-file receipt makes a reviewed
resume practical without weakening input/config/output identity checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cleaning_runtime import (
    CLEANING_IMPLEMENTATION_VERSION,
    apply_structural_spans,
    canonical_json_sha256,
    cleaned_schema,
    count_stage50_versions,
    file_receipt,
    ledger_schema,
    per_file_receipt_path,
    require_exact_parquet_tree,
    reusable_file_receipt,
    write_json_atomic,
)
from full_corpus_io import (
    canonical_schema as _canonical_schema,
    normalize_text,
    read_json_object,
    sha256_file,
    sha256_text,
    strip_html_markup,
)
from source_license import decision_for as license_decision_for
from source_license import load_adjudication as load_license_adjudication
from greek_pii import mask_greek_identifiers


HERE = Path(__file__).resolve().parents[1]
PII_DIR = HERE.parent / "02_corpus_preparation" / "40_anonymize" / "scripts"
if str(PII_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(PII_DIR))
from pii_masker import mask as mask_apertus_pii  # type: ignore  # noqa: E402


WEB_PROFILES = {"news", "web_articles"}
PERSONNEL = re.compile(
    r"(?i)(?:πατρώνυμο|μητρώνυμο|αριθμός\s+ταυτότητας|Α\.?Δ\.?Τ\.?|"
    r"πίνακας\s+(?:κατάταξης|υποψηφίων|προσληπτέων))"
)
_WORKER: dict[str, Any] = {}
# Public compatibility export used by existing fixtures/callers.
canonical_schema = _canonical_schema


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
        required = {"training_eligible", "redistribution_eligible"}
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"{path}: invalid eligibility category {name!r}")
        result[str(name)] = {
            "training_eligible": bool(row["training_eligible"]),
            "redistribution_eligible": bool(row["redistribution_eligible"]),
        }
    return result


def _canonical_action(row: Mapping[str, Any], *, path: Path, line_number: int) -> dict[str, Any]:
    stable_uid = str(row.get("stable_uid", ""))
    if len(stable_uid) != 64 or any(char not in "0123456789abcdef" for char in stable_uid):
        raise ValueError(f"{path}:{line_number}: action lacks a valid stable_uid")
    action = str(row.get("action", ""))
    if action not in {"drop", "quarantine"}:
        raise ValueError(f"{path}:{line_number}: unsupported action {action!r}")
    input_hash = str(row.get("input_text_sha256", ""))
    if len(input_hash) != 64 or any(char not in "0123456789abcdef" for char in input_hash):
        raise ValueError(f"{path}:{line_number}: action lacks a valid input_text_sha256")
    reason = str(row.get("reason") or row.get("rule_id") or "")
    if not reason:
        raise ValueError(f"{path}:{line_number}: action lacks reason/rule_id")
    return {
        "stable_uid": stable_uid,
        "input_text_sha256": input_hash,
        "action": action,
        "reason": reason,
    }


def load_document_actions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load JSONL/Parquet content-bound actions, allowing exact duplicates only."""

    actions: dict[str, dict[str, Any]] = {}
    for root in paths:
        files = (
            sorted([*root.rglob("*.parquet"), *root.rglob("*.jsonl")])
            if root.is_dir()
            else [root]
        )
        if not files:
            raise ValueError(f"{root}: no Parquet/JSONL document-action shards")
        for path in files:
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq

                raw_rows: Iterable[tuple[int, Mapping[str, Any]]] = (
                    (row_number, row)
                    for row_number, row in enumerate(
                        (
                            row
                            for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536)
                            for row in batch.to_pylist()
                        ),
                        1,
                    )
                )
            elif path.suffix == ".jsonl":
                def jsonl_rows() -> Iterable[tuple[int, Mapping[str, Any]]]:
                    with path.open(encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if not line.strip():
                                continue
                            raw = json.loads(line)
                            if not isinstance(raw, dict):
                                raise ValueError(f"{path}:{line_number}: action must be an object")
                            yield line_number, raw

                raw_rows = jsonl_rows()
            else:
                raise ValueError(f"{path}: document actions must be JSONL or Parquet")
            for line_number, raw in raw_rows:
                row = _canonical_action(raw, path=path, line_number=line_number)
                stable_uid = row["stable_uid"]
                previous = actions.get(stable_uid)
                if previous is not None:
                    if (
                        previous["input_text_sha256"] != row["input_text_sha256"]
                        or previous["action"] != row["action"]
                    ):
                        raise ValueError(f"conflicting duplicate actions for stable_uid {stable_uid}")
                    previous["reasons"] = sorted(
                        {*previous.get("reasons", [previous["reason"]]), row["reason"]}
                    )
                else:
                    actions[stable_uid] = {**row, "reasons": [row["reason"]]}
    return actions


def document_action_receipts(paths: Sequence[Path]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for root in paths:
        files = (
            sorted([*root.rglob("*.parquet"), *root.rglob("*.jsonl")])
            if root.is_dir()
            else [root]
        )
        receipts.extend(file_receipt(path) for path in files)
    return receipts


def build_document_action_index(paths: Sequence[Path], database: Path) -> dict[str, Any]:
    """Create a bounded-memory, conflict-checked lookup for multi-million actions."""

    inputs = document_action_receipts(paths)
    inputs_sha256 = canonical_json_sha256(inputs)
    index_receipt_path = database.with_suffix(database.suffix + ".receipt.json")
    if database.is_file() and index_receipt_path.is_file():
        try:
            prior = read_json_object(index_receipt_path)
            if (
                prior.get("schema_version") == "full_cpt_document_action_index_v1"
                and prior.get("inputs_sha256") == inputs_sha256
                and prior.get("database", {}).get("bytes") == database.stat().st_size
                and prior.get("database", {}).get("sha256") == sha256_file(database)
            ):
                return {
                    "database": str(database.resolve()),
                    "input_rows": int(prior["input_rows"]),
                    "distinct_actions": int(prior["distinct_actions"]),
                    "exact_duplicate_rows": int(prior["exact_duplicate_rows"]),
                    "files": inputs,
                    "resumed": True,
                }
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE action_rows (
          stable_uid TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          action TEXT NOT NULL,
          reason TEXT NOT NULL
        )
        """
    )
    input_rows = 0
    buffer: list[tuple[str, str, str, str]] = []

    def flush() -> None:
        if buffer:
            connection.executemany("INSERT INTO action_rows VALUES (?, ?, ?, ?)", buffer)
            buffer.clear()

    try:
        for root in paths:
            files = (
                sorted([*root.rglob("*.parquet"), *root.rglob("*.jsonl")])
                if root.is_dir()
                else [root]
            )
            if not files:
                raise ValueError(f"{root}: no Parquet/JSONL document-action shards")
            for path in files:
                if path.suffix == ".parquet":
                    import pyarrow.parquet as pq

                    rows: Iterable[tuple[int, Mapping[str, Any]]] = (
                        (row_number, row)
                        for row_number, row in enumerate(
                            (
                                row
                                for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536)
                                for row in batch.to_pylist()
                            ),
                            1,
                        )
                    )
                elif path.suffix == ".jsonl":
                    def jsonl_rows() -> Iterable[tuple[int, Mapping[str, Any]]]:
                        with path.open(encoding="utf-8") as handle:
                            for line_number, line in enumerate(handle, 1):
                                if line.strip():
                                    raw = json.loads(line)
                                    if not isinstance(raw, dict):
                                        raise ValueError(
                                            f"{path}:{line_number}: action must be an object"
                                        )
                                    yield line_number, raw

                    rows = jsonl_rows()
                else:
                    raise ValueError(f"{path}: document actions must be JSONL or Parquet")
                for line_number, raw in rows:
                    row = _canonical_action(raw, path=path, line_number=line_number)
                    input_rows += 1
                    buffer.append(
                        (
                            row["stable_uid"],
                            row["input_text_sha256"],
                            row["action"],
                            row["reason"],
                        )
                    )
                    if len(buffer) >= 65_536:
                        flush()
                flush()
        connection.commit()
        conflict = connection.execute(
            """
            SELECT stable_uid FROM action_rows GROUP BY stable_uid
            HAVING min(input_text_sha256) <> max(input_text_sha256)
                OR min(action) <> max(action)
            LIMIT 1
            """
        ).fetchone()
        if conflict is not None:
            raise ValueError(f"conflicting duplicate actions for stable_uid {conflict[0]}")
        connection.execute(
            """
            CREATE TABLE actions (
              stable_uid TEXT PRIMARY KEY,
              input_text_sha256 TEXT NOT NULL,
              action TEXT NOT NULL,
              reasons_hex TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            INSERT INTO actions
            SELECT stable_uid, min(input_text_sha256), min(action),
                   group_concat(hex(reason), ',')
            FROM (
              SELECT DISTINCT stable_uid, input_text_sha256, action, reason
              FROM action_rows ORDER BY stable_uid, reason
            )
            GROUP BY stable_uid
            """
        )
        distinct = int(connection.execute("SELECT count(*) FROM actions").fetchone()[0])
        distinct_records = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT DISTINCT stable_uid, input_text_sha256, action, reason FROM action_rows)"
            ).fetchone()[0]
        )
        exact_duplicates = input_rows - distinct_records
        connection.execute("DROP TABLE action_rows")
        connection.commit()
    finally:
        connection.close()
    result = {
        "database": str(database.resolve()),
        "input_rows": input_rows,
        "distinct_actions": distinct,
        "exact_duplicate_rows": exact_duplicates,
        "files": inputs,
        "resumed": False,
    }
    write_json_atomic(
        index_receipt_path,
        {
            "schema_version": "full_cpt_document_action_index_v1",
            "inputs_sha256": inputs_sha256,
            "database": file_receipt(database),
            "input_rows": input_rows,
            "distinct_actions": distinct,
            "exact_duplicate_rows": exact_duplicates,
        },
    )
    return result


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
    admissions: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if source_id == "nanochat_base" or source_role == "base":
        return {"decision": "include"}
    if source_dataset in admissions:
        return admissions[source_dataset]
    return admissions.get(source_id, {"decision": "pending"})


def apply_spans(text: str, spans: Iterable[dict[str, Any]]) -> tuple[str, list[str]]:
    """Compatibility wrapper used by focused tests; Stage 50 never calls it."""

    normalized: list[dict[str, Any]] = []
    for span in spans:
        row = dict(span)
        if not row.get("kind"):
            rule = str(row.get("rule_id") or "")
            if "bibliograph" in rule.casefold():
                row["kind"] = "bibliography"
            elif "toc" in rule.casefold() or "table_of_contents" in rule.casefold():
                row["kind"] = "toc"
        normalized.append(row)
    output, reasons, _ = apply_structural_spans(text, normalized)
    return output, reasons


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER
    from tokenizers import Tokenizer

    _WORKER = dict(config)
    _WORKER["tokenizer"] = Tokenizer.from_file(config["tokenizer_json"])
    uri = f"file:{config['action_database']}?mode=ro"
    _WORKER["action_connection"] = sqlite3.connect(uri, uri=True)


def _actions_for_uids(stable_uids: Sequence[str]) -> dict[str, dict[str, Any]]:
    connection = _WORKER["action_connection"]
    found: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(stable_uids), 500):
        chunk = stable_uids[offset : offset + 500]
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        for uid, input_hash, action, reasons_hex in connection.execute(
            f"SELECT stable_uid, input_text_sha256, action, reasons_hex FROM actions "
            f"WHERE stable_uid IN ({placeholders})",
            list(chunk),
        ):
            found[str(uid)] = {
                "stable_uid": str(uid),
                "input_text_sha256": str(input_hash),
                "action": str(action),
                "reasons": [
                    bytes.fromhex(value).decode("utf-8")
                    for value in str(reasons_hex).split(",")
                ],
            }
    return found


def _transform_batch(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str], dict[str, Counter[str]], int]:
    config = _WORKER
    actions = _actions_for_uids([str(row["stable_uid"]) for row in rows])
    admissions = config["admissions"]
    eligibility_policy = config["eligibility_policy"]
    license_adjudication = config["license_adjudication"]
    prepared: list[dict[str, Any]] = []
    applied_action_count = 0

    for row in rows:
        stable_uid = str(row["stable_uid"])
        source_id = str(row["acquisition_source_id"])
        normalized = str(row["text"] or "")
        normalized_hash = sha256_text(normalized)
        if row.get("normalized_text_sha256") != normalized_hash:
            raise ValueError(f"{stable_uid}: normalized_text_sha256 does not match text")
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
        category_ceiling = eligibility_policy[eligibility_name]
        eligibility = license_decision_for(
            source_id, eligibility_name, license_adjudication
        )
        for key in ("training_eligible", "redistribution_eligible"):
            if eligibility[key] and not category_ceiling[key]:
                raise ValueError(
                    f"{source_id}: source adjudication exceeds {eligibility_name!r} {key} ceiling"
                )
        action = "keep"
        reasons: list[str] = []
        if decision not in {"include", "include_after_cleaning"}:
            action = "drop"
            reasons.append(f"source_admission:{decision}")

        document_action = actions.get(stable_uid)
        if document_action is not None:
            if document_action["input_text_sha256"] != normalized_hash:
                raise ValueError(
                    f"{stable_uid}: document action input hash differs from normalized text"
                )
            candidate = document_action["action"]
            if candidate == "drop":
                action = "drop"
            elif candidate == "quarantine" and action == "keep":
                action = "quarantine"
            reasons.extend(document_action["reasons"])
            applied_action_count += 1

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
                source_cleaned = normalize_text(candidate)
        generic_masked, generic_counts = mask_apertus_pii(source_cleaned)
        pii_masked, greek_counts = mask_greek_identifiers(generic_masked)
        pii_masked = normalize_text(pii_masked)
        pii_counts = dict(generic_counts)
        for key, value in greek_counts.items():
            pii_counts[key] = pii_counts.get(key, 0) + value
        if sum(pii_counts.values()):
            trace.append({"rule": "high_confidence_direct_pii_v1", "matches": sum(pii_counts.values())})
        if source_id == "diavgeia" and PERSONNEL.search(normalized) and sum(pii_counts.values()) >= 3:
            action = "quarantine"
            reasons.append("diavgeia_pii_heavy_personnel_table")
        if not pii_masked and action == "keep":
            action = "quarantine"
            reasons.append("cleaning_emptied_document")

        row_training_eligible = bool(
            action == "keep" and decision == "include" and eligibility["training_eligible"]
        )
        row_redistribution_eligible = bool(
            row_training_eligible and eligibility["redistribution_eligible"]
        )
        if action == "keep" and not row_training_eligible:
            if decision == "include_after_cleaning":
                reasons.append("post_clean_review_required")
            else:
                reasons.append(f"source_license_not_approved:{source_id}")
        prepared.append(
            {
                "_row": row,
                "_normalized": normalized,
                "_source_cleaned": source_cleaned,
                "_pii_masked": pii_masked,
                "_trace": trace,
                "_pii_counts": pii_counts,
                "_action": action,
                "_reasons": reasons,
                "_decision": decision,
                "_eligibility_name": eligibility_name,
                "_eligible_training": row_training_eligible,
                "_eligible_redistribution": row_redistribution_eligible,
            }
        )

    count_stage50_versions(config["tokenizer"], prepared)
    cleaned_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    source_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for item in prepared:
        row = item["_row"]
        source_id = str(row["acquisition_source_id"])
        stable_uid = str(row["stable_uid"])
        action = item["_action"]
        counts = {
            key: int(item[key])
            for key in (
                "tokens_normalized",
                "tokens_source_cleaned",
                "tokens_pii_masked",
                "tokens_toc_removed",
                "tokens_bibliography_removed",
                "tokens_structural_union_removed",
                "tokens_structural_cleaned",
            )
        }
        counts["tokens_final"] = counts["tokens_pii_masked"] if item["_eligible_training"] else 0
        ledger_rows.append(
            {
                "stable_uid": stable_uid,
                "acquisition_source_id": source_id,
                "source_dataset": str(row["source_dataset"]),
                "source_doc_id": str(row["source_doc_id"]),
                "action": action,
                "reasons_json": json.dumps(sorted(set(item["_reasons"])), ensure_ascii=False),
                **counts,
                "characters_normalized": len(item["_normalized"]),
                "characters_final": len(item["_pii_masked"]) if action == "keep" else 0,
                "final_text_sha256": sha256_text(item["_pii_masked"]),
                "pii_by_type_json": json.dumps(item["_pii_counts"], sort_keys=True),
                "source_admission_decision": item["_decision"],
                "training_eligibility_category": item["_eligibility_name"],
                "eligible_for_training": item["_eligible_training"],
                "eligible_for_redistribution": item["_eligible_redistribution"],
            }
        )
        counters[f"action:{action}"] += 1
        source_counters[source_id][f"action:{action}"] += 1
        for key, value in counts.items():
            counters[key] += value
            source_counters[source_id][key] += value
        final_row = {
            **row,
            "text": item["_pii_masked"],
            "cleaned_text_sha256": sha256_text(item["_pii_masked"]),
            "cleaning_trace_json": json.dumps(item["_trace"], ensure_ascii=False, sort_keys=True),
            "pii_by_type_json": json.dumps(item["_pii_counts"], sort_keys=True),
            "eligible_for_training": item["_eligible_training"],
            "eligible_for_redistribution": item["_eligible_redistribution"],
        }
        if action == "keep":
            cleaned_rows.append(final_row)
        elif action == "quarantine":
            quarantine_rows.append(final_row)
    return cleaned_rows, quarantine_rows, ledger_rows, counters, source_counters, applied_action_count


def _process_file(job: dict[str, str]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_path = Path(job["input"])
    output_path = Path(job["output"])
    quarantine_path = Path(job["quarantine"])
    ledger_path = Path(job["ledger"])
    relative = Path(job["relative"])
    roots = {name: Path(path) for name, path in _WORKER["roots"].items()}
    for path in (output_path, quarantine_path, ledger_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)
    writers = {
        "output": pq.ParquetWriter(output_path.with_suffix(output_path.suffix + ".partial"), cleaned_schema(), compression="zstd"),
        "quarantine": pq.ParquetWriter(quarantine_path.with_suffix(quarantine_path.suffix + ".partial"), cleaned_schema(), compression="zstd"),
        "ledger": pq.ParquetWriter(ledger_path.with_suffix(ledger_path.suffix + ".partial"), ledger_schema(), compression="zstd"),
    }
    counters: Counter[str] = Counter()
    source_counters: dict[str, Counter[str]] = defaultdict(Counter)
    applied_action_count = 0
    input_rows = kept_rows = quarantined_rows = ledger_count = 0
    parquet = pq.ParquetFile(input_path)
    try:
        for row_group in range(parquet.num_row_groups):
            for batch in parquet.iter_batches(
                row_groups=[row_group], batch_size=int(_WORKER["batch_rows"])
            ):
                raw_rows = batch.to_pylist()
                cleaned, quarantined, ledger, batch_counts, batch_sources, batch_actions = _transform_batch(raw_rows)
                input_rows += len(raw_rows)
                kept_rows += len(cleaned)
                quarantined_rows += len(quarantined)
                ledger_count += len(ledger)
                counters.update(batch_counts)
                applied_action_count += batch_actions
                for source, values in batch_sources.items():
                    source_counters[source].update(values)
                if cleaned:
                    writers["output"].write_table(pa.Table.from_pylist(cleaned, schema=cleaned_schema()))
                if quarantined:
                    writers["quarantine"].write_table(pa.Table.from_pylist(quarantined, schema=cleaned_schema()))
                if ledger:
                    writers["ledger"].write_table(pa.Table.from_pylist(ledger, schema=ledger_schema()))
        for writer in writers.values():
            writer.close()
        os.replace(output_path.with_suffix(output_path.suffix + ".partial"), output_path)
        os.replace(quarantine_path.with_suffix(quarantine_path.suffix + ".partial"), quarantine_path)
        os.replace(ledger_path.with_suffix(ledger_path.suffix + ".partial"), ledger_path)
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for path in (output_path, quarantine_path, ledger_path):
            path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)
        raise
    if input_rows != ledger_count:
        raise RuntimeError(f"{input_path}: cleaning ledger coverage mismatch")
    receipt = {
        "schema_version": "full_cpt_cleaning_file_receipt_v1",
        "implementation_version": CLEANING_IMPLEMENTATION_VERSION,
        "cleaning_pass": "post_source_post_pii",
        "config_sha256": _WORKER["config_sha256"],
        "relative_path": relative.as_posix(),
        "input": file_receipt(input_path),
        "output": file_receipt(output_path, relative_to=roots["output"]),
        "quarantine": file_receipt(quarantine_path, relative_to=roots["quarantine"]),
        "ledger": file_receipt(ledger_path, relative_to=roots["ledger"]),
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "quarantined_rows": quarantined_rows,
        "counters": dict(sorted(counters.items())),
        "per_source": {key: dict(sorted(value.items())) for key, value in sorted(source_counters.items())},
        "applied_action_count": applied_action_count,
    }
    write_json_atomic(Path(job["receipt"]), receipt)
    return receipt


def _auto_workers(value: int) -> int:
    if value > 0:
        return value
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, min(16, cpus // 4 or 1))


def _aggregate(receipts: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, Counter[str]], int]:
    counters: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    action_uses = 0
    for receipt in receipts:
        counters.update(receipt["counters"])
        for source, values in receipt["per_source"].items():
            per_source[source].update(values)
        action_uses += int(receipt["applied_action_count"])
    return counters, per_source, action_uses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=HERE / "configs" / "sources.json")
    parser.add_argument(
        "--license-adjudication",
        type=Path,
        default=HERE / "configs" / "source_license_adjudication.json",
    )
    parser.add_argument(
        "--eligibility-policy", type=Path, default=HERE / "configs" / "training_eligibility_policy.json"
    )
    parser.add_argument("--cleaning-policy", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--document-actions", action="append", type=Path, default=[])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--workers", type=int, default=0, help="0 chooses a bounded CPU-node default")
    parser.add_argument("--tokenizer-threads", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--allow-pending-admission", action="store_true", help="bounded smoke only")
    # Old flags are retained only to produce an actionable hard failure.
    parser.add_argument("--structural-spans", action="append", type=Path, default=[])
    parser.add_argument("--apply-structural", action="store_true")
    args = parser.parse_args()
    if args.apply_structural or args.structural_spans:
        raise ValueError("Stage 50 is structural-no-op; use finalize_structural_cleaning.py")
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    status, admissions = load_admission(args.source_admission)
    if status != "approved" and not args.allow_pending_admission:
        raise ValueError("source admission must be approved for materialization")
    eligibility_policy = load_eligibility_policy(args.eligibility_policy)
    license_adjudication = load_license_adjudication(
        args.license_adjudication, source_registry_path=args.source_config
    )
    policy = read_json_object(args.cleaning_policy)
    if policy.get("schema_version") not in {None, "full_cpt_cleaning_policy_v1"}:
        raise ValueError("unsupported cleaning policy")
    inputs = sorted(args.input.rglob("*.parquet"))
    if not inputs:
        raise ValueError(f"no Parquet inputs beneath {args.input}")
    roots = {
        "output": args.output.resolve(),
        "quarantine": args.quarantine.resolve(),
        "ledger": args.ledger.resolve(),
    }
    workers = _auto_workers(args.workers)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    tokenizer_threads = (
        args.tokenizer_threads
        if args.tokenizer_threads > 0
        else max(1, allocated_cpus // workers // 2)
    )
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    config_receipts = {
        "source_admission": file_receipt(args.source_admission),
        "source_config": file_receipt(args.source_config),
        "license_adjudication": file_receipt(args.license_adjudication),
        "eligibility_policy": file_receipt(args.eligibility_policy),
        "cleaning_policy": file_receipt(args.cleaning_policy),
        "tokenizer": file_receipt(args.tokenizer_json),
        "document_actions": document_action_receipts(args.document_actions),
        "batch_rows": args.batch_rows,
        "workers": workers,
        "tokenizer_threads_per_worker": tokenizer_threads,
        "implementation_version": CLEANING_IMPLEMENTATION_VERSION,
        "cleaning_pass": "post_source_post_pii",
    }
    config_sha256 = canonical_json_sha256(config_receipts)
    work_dir = args.work_dir or (args.manifest.parent / "work")
    action_index = build_document_action_index(
        args.document_actions, work_dir / "document-actions.sqlite"
    )
    receipt_root = args.manifest.parent / "file_receipts"
    jobs: list[dict[str, str]] = []
    receipts_by_relative: dict[str, dict[str, Any]] = {}
    for input_path in inputs:
        relative = input_path.relative_to(args.input)
        receipt_path = per_file_receipt_path(receipt_root, relative)
        existing = reusable_file_receipt(
            receipt_path,
            input_path=input_path,
            config_sha256=config_sha256,
            roots=roots,
        )
        if existing is not None:
            receipts_by_relative[relative.as_posix()] = existing
            continue
        jobs.append(
            {
                "input": str(input_path),
                "relative": relative.as_posix(),
                "output": str(args.output / relative),
                "quarantine": str(args.quarantine / relative),
                "ledger": str(args.ledger / relative),
                "receipt": str(receipt_path),
            }
        )

    worker_config = {
        "tokenizer_json": str(args.tokenizer_json),
        "admissions": admissions,
        "eligibility_policy": eligibility_policy,
        "license_adjudication": license_adjudication,
        "action_database": action_index["database"],
        "batch_rows": args.batch_rows,
        "config_sha256": config_sha256,
        "roots": {key: str(value) for key, value in roots.items()},
    }
    if jobs and workers == 1:
        _init_worker(worker_config)
        for job in jobs:
            receipt = _process_file(job)
            receipts_by_relative[receipt["relative_path"]] = receipt
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(worker_config,)) as executor:
            futures = {executor.submit(_process_file, job): job for job in jobs}
            for future in as_completed(futures):
                receipt = future.result()
                receipts_by_relative[receipt["relative_path"]] = receipt

    receipts = [receipts_by_relative[path] for path in sorted(receipts_by_relative)]
    if len(receipts) != len(inputs):
        raise RuntimeError("not every cleaning input produced a verified per-file receipt")
    relatives = [Path(receipt["relative_path"]) for receipt in receipts]
    for root in roots.values():
        require_exact_parquet_tree(root, relatives)
    counters, source_counters, action_uses = _aggregate(receipts)
    if action_uses != int(action_index["distinct_actions"]):
        raise ValueError(
            "document action coverage mismatch: "
            f"indexed={action_index['distinct_actions']} applied={action_uses}"
        )
    payload = {
        "schema_version": "full_cpt_cleaning_manifest_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "implementation_version": CLEANING_IMPLEMENTATION_VERSION,
        "cleaning_pass": "post_source_post_pii",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "quarantine": str(args.quarantine.resolve()),
        "ledger": str(args.ledger.resolve()),
        "tokenizer_json": str(args.tokenizer_json.resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer_json),
        "source_admission": str(args.source_admission.resolve()),
        "source_admission_sha256": sha256_file(args.source_admission),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "license_adjudication": str(args.license_adjudication.resolve()),
        "license_adjudication_sha256": sha256_file(args.license_adjudication),
        "eligibility_policy": str(args.eligibility_policy.resolve()),
        "eligibility_policy_sha256": sha256_file(args.eligibility_policy),
        "cleaning_policy": str(args.cleaning_policy.resolve()),
        "cleaning_policy_sha256": sha256_file(args.cleaning_policy),
        "document_action_receipts": config_receipts["document_actions"],
        "document_action_index": {
            key: value for key, value in action_index.items() if key != "database"
        },
        "config_sha256": config_sha256,
        "structural_applied": False,
        "structural_semantics": "deterministic_no_op",
        "workers": workers,
        "tokenizer_threads_per_worker": tokenizer_threads,
        "batch_rows": args.batch_rows,
        "resumed_files": len(inputs) - len(jobs),
        "counts": dict(sorted(counters.items())),
        "per_source": {key: dict(sorted(value.items())) for key, value in sorted(source_counters.items())},
        "files": receipts,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(counters)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
