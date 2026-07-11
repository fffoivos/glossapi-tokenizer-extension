#!/usr/bin/env python3
"""Finalize source admission and apply optional structural spans last.

This stage consumes the exact Stage 50 post-source/post-PII shards.  It never
reruns HTML cleanup, document actions, source filters, or PII masking.  A
validated no-op decision copies the exact text while updating only terminal
admission/eligibility.  A passing structural decision additionally applies
content-bound, disjoint ToC/bibliography spans and records exact counterfactual
token loss for each kind and their union.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from apply_cleaning_policy import admission_for, load_admission, load_eligibility_policy
from cleaning_runtime import (
    STRUCTURAL_KINDS,
    apply_structural_spans,
    canonical_json_sha256,
    cleaned_schema,
    encode_counts,
    file_receipt,
    ledger_schema,
    per_file_receipt_path,
    require_exact_parquet_tree,
    structural_counterfactuals,
    valid_sha256,
    verify_file_receipt,
    write_json_atomic,
)
from full_corpus_io import read_json_object, sha256_file, sha256_text
from source_license import decision_for as license_decision_for
from source_license import load_adjudication as load_license_adjudication


FINALIZER_VERSION = "full-cpt-structural-last-finalizer-v1"
_WORKER: dict[str, Any] = {}


def _require_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json_object(path)
    expected = {
        "schema_version": "full_cpt_cleaning_manifest_v1",
        "status": "completed",
        "cleaning_pass": "post_source_post_pii",
        "structural_applied": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path}: Stage 50 manifest {key} mismatch")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError(f"{path}: Stage 50 manifest has no files")
    return manifest


def _validate_stage50_replay_inputs(
    manifest: Mapping[str, Any],
    *,
    tokenizer: Path,
    source_config: Path,
    license_adjudication: Path,
    eligibility_policy: Path,
    cleaning_policy: Path,
) -> None:
    """Bind Stage 58 policy inputs to the reviewed Stage 50 pass.

    Terminal source admission may change after post-clean review. The policy
    ceiling and deterministic cleaning inputs may not: changing any of them
    would turn Stage 58 into a new, unreviewed cleaning pass.
    """

    bindings = {
        "tokenizer": ("tokenizer_sha256", tokenizer),
        "source registry": ("source_config_sha256", source_config),
        "license adjudication": (
            "license_adjudication_sha256",
            license_adjudication,
        ),
        "eligibility policy": ("eligibility_policy_sha256", eligibility_policy),
        "cleaning policy": ("cleaning_policy_sha256", cleaning_policy),
    }
    for label, (manifest_field, current_path) in bindings.items():
        reviewed_sha256 = manifest.get(manifest_field)
        current_sha256 = sha256_file(current_path)
        if reviewed_sha256 != current_sha256:
            raise ValueError(
                f"finalizer {label} differs from reviewed Stage 50 input: "
                f"{current_sha256} != {reviewed_sha256}"
            )


def _load_decision(
    path: Path,
    *,
    stage50_sha256: str,
    cleaning_policy_sha256: str,
    cleaning_policy: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json_object(path)
    if value.get("schema_version") != "full_cpt_structural_application_decision_v1":
        raise ValueError(f"{path}: unsupported structural decision schema")
    if value.get("stage50_cleaning_manifest_sha256") != stage50_sha256:
        raise ValueError(f"{path}: structural decision is not bound to Stage 50")
    if value.get("cleaning_policy_sha256") != cleaning_policy_sha256:
        raise ValueError(f"{path}: structural decision is not bound to cleaning policy")
    apply = value.get("apply_structural")
    if not isinstance(apply, bool):
        raise ValueError(f"{path}: apply_structural must be boolean")
    requested_mode = value.get("requested_mode")
    requested_apply = value.get("apply_structural_requested")
    if requested_mode not in {"no_op", "apply"} or not isinstance(requested_apply, bool):
        raise ValueError(f"{path}: structural requested mode is missing/invalid")
    if requested_apply != (requested_mode == "apply") or apply != requested_apply:
        raise ValueError(f"{path}: structural request and application decision disagree")
    if apply and value.get("status") != "passed":
        raise ValueError(f"{path}: only a passed decision may apply structural spans")
    if not apply and value.get("status") != "no_op":
        raise ValueError(f"{path}: non-applying decision must be no_op")
    if apply:
        if value.get("model_receipt_sha256") is None:
            raise ValueError(f"{path}: requested apply lacks an exact model receipt binding")
        policy_enabled = (
            cleaning_policy.get("status") == "approved"
            and cleaning_policy.get("structural", {})
            .get("toc", {})
            .get("enabled_for_materialization")
            is True
            and cleaning_policy.get("structural", {})
            .get("bibliography", {})
            .get("enabled_for_materialization")
            is True
        )
        if not policy_enabled:
            raise ValueError(f"{path}: tracked structural policy is not approved/enabled")
        if value.get("model_selection_evidence") != "LLM_silver":
            raise ValueError(f"{path}: model evidence is not declared LLM_silver")
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict) or any(
            not valid_sha256(artifacts.get(name)) for name in ("code", "config", "checkpoint")
        ):
            raise ValueError(f"{path}: structural artifact hashes are incomplete")
        work_split = value.get("work_split")
        if (
            not isinstance(work_split, dict)
            or work_split.get("leak_free") is not True
            or int(work_split.get("work_overlap_count", -1)) != 0
            or int(work_split.get("exact_text_overlap_count", -1)) != 0
        ):
            raise ValueError(f"{path}: structural work split is not leak-free")
        safety = value.get("safety")
        if (
            not isinstance(safety, dict)
            or safety.get("evidence_status") != "targeted_manual_false_deletion_audit"
            or safety.get("metric_gate_passed") is not True
            or not valid_sha256(safety.get("audit_receipt_sha256"))
        ):
            raise ValueError(f"{path}: targeted manual structural safety gate is incomplete")
        model_receipt = Path(str(value.get("model_receipt", "")))
        if (
            not model_receipt.is_file()
            or sha256_file(model_receipt) != value.get("model_receipt_sha256")
        ):
            raise ValueError(f"{path}: structural model receipt file/hash drift")
    elif value.get("model_receipt") is not None or value.get("model_receipt_sha256") is not None:
        raise ValueError(f"{path}: requested no-op must not carry a model receipt")
    return value


def _build_span_index(
    paths: Sequence[Path],
    *,
    database: Path,
    model_receipt_sha256: str,
    stage50_manifest_sha256: str,
) -> dict[str, Any]:
    database.parent.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE spans (
          stable_uid TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          char_start INTEGER NOT NULL,
          char_end INTEGER NOT NULL,
          kind TEXT NOT NULL,
          rule_id TEXT NOT NULL,
          PRIMARY KEY (stable_uid, char_start, char_end, kind, rule_id)
        ) WITHOUT ROWID
        """
    )
    rows = 0
    try:
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError(f"{path}:{line_number}: span must be an object")
                    stable_uid = str(raw.get("stable_uid", ""))
                    input_hash = str(raw.get("input_text_sha256", ""))
                    if not valid_sha256(stable_uid) or not valid_sha256(input_hash):
                        raise ValueError(f"{path}:{line_number}: invalid stable_uid/input hash")
                    if raw.get("model_receipt_sha256") != model_receipt_sha256:
                        raise ValueError(f"{path}:{line_number}: model receipt binding mismatch")
                    bound_cleaning = raw.get("stage50_cleaning_manifest_sha256")
                    if bound_cleaning is None:
                        bound_cleaning = raw.get("cleaning_manifest_sha256")
                    if bound_cleaning != stage50_manifest_sha256:
                        raise ValueError(f"{path}:{line_number}: Stage 50 manifest binding mismatch")
                    raw_kind = str(raw.get("kind", ""))
                    kind = STRUCTURAL_KINDS.get(raw_kind)
                    if kind is None:
                        raise ValueError(f"{path}:{line_number}: unsupported kind {raw_kind!r}")
                    start, end = int(raw["char_start"]), int(raw["char_end"])
                    if start < 0 or end <= start:
                        raise ValueError(f"{path}:{line_number}: invalid offsets {start}:{end}")
                    rule_id = str(raw.get("rule_id") or raw_kind)
                    try:
                        connection.execute(
                            "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                            (stable_uid, input_hash, start, end, kind, rule_id),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"{path}:{line_number}: duplicate structural span") from exc
                    rows += 1
        connection.commit()
        previous_uid = ""
        previous_hash = ""
        previous_end = -1
        documents = 0
        for uid, input_hash, start, end in connection.execute(
            "SELECT stable_uid, input_text_sha256, char_start, char_end FROM spans "
            "ORDER BY stable_uid, char_start, char_end"
        ):
            if uid != previous_uid:
                previous_uid, previous_hash, previous_end = uid, input_hash, -1
                documents += 1
            elif input_hash != previous_hash:
                raise ValueError(f"{uid}: spans disagree on input_text_sha256")
            if int(start) < previous_end:
                raise ValueError(f"{uid}: structural span inventory contains overlaps")
            previous_end = int(end)
        connection.execute("CREATE INDEX spans_uid ON spans(stable_uid)")
        connection.commit()
    finally:
        connection.close()
    return {
        "rows": rows,
        "documents": documents if rows else 0,
        "database": str(database.resolve()),
        "span_files": [file_receipt(path) for path in paths],
        "span_inventory_sha256": canonical_json_sha256([file_receipt(path) for path in paths]),
    }


def _iter_rows(path: Path, batch_rows: int) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for row_group in range(parquet.num_row_groups):
        for batch in parquet.iter_batches(row_groups=[row_group], batch_size=batch_rows):
            yield from batch.to_pylist()


def _write_rows(writer: Any, rows: list[dict[str, Any]], schema: Any) -> None:
    if not rows:
        return
    import pyarrow as pa

    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER
    from tokenizers import Tokenizer

    _WORKER = dict(config)
    _WORKER["tokenizer"] = Tokenizer.from_file(config["tokenizer_json"])
    if config.get("span_database"):
        uri = f"file:{config['span_database']}?mode=ro"
        _WORKER["span_connection"] = sqlite3.connect(uri, uri=True)


def _spans_for(stable_uid: str) -> list[dict[str, Any]]:
    connection = _WORKER.get("span_connection")
    if connection is None:
        return []
    return [
        {
            "stable_uid": stable_uid,
            "input_text_sha256": row[0],
            "char_start": row[1],
            "char_end": row[2],
            "kind": row[3],
            "rule_id": row[4],
        }
        for row in connection.execute(
            "SELECT input_text_sha256, char_start, char_end, kind, rule_id "
            "FROM spans WHERE stable_uid=? ORDER BY char_start, char_end",
            (stable_uid,),
        )
    ]


def _finalize_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], dict[str, Counter[str]], int]:
    tokenizer = _WORKER["tokenizer"]
    admissions = _WORKER["admissions"]
    eligibility_policy = _WORKER["eligibility_policy"]
    license_adjudication = _WORKER["license_adjudication"]
    structural = bool(_WORKER["apply_structural"])
    allowed_profiles = set(_WORKER["allowed_profiles"])
    prepared: list[dict[str, Any]] = []
    variants: list[str] = []
    needed: list[tuple[int, str]] = []
    seen_spans = 0

    for row, ledger in pairs:
        stable_uid = str(row["stable_uid"])
        text = str(row["text"] or "")
        text_hash = sha256_text(text)
        if row.get("cleaned_text_sha256") != text_hash:
            raise ValueError(f"{stable_uid}: Stage 50 cleaned_text_sha256 mismatch")
        if ledger.get("final_text_sha256") != text_hash:
            raise ValueError(f"{stable_uid}: Stage 50 corpus/ledger text hash mismatch")
        if (
            int(ledger.get("tokens_toc_removed", -1)) != 0
            or int(ledger.get("tokens_bibliography_removed", -1)) != 0
            or int(ledger.get("tokens_structural_union_removed", -1)) != 0
            or int(ledger.get("tokens_structural_cleaned", -1))
            != int(ledger.get("tokens_pii_masked", -2))
        ):
            raise ValueError(f"{stable_uid}: Stage 50 is not a structural no-op")
        spans = _spans_for(stable_uid) if structural else []
        seen_spans += len(spans)
        if spans and (
            str(row.get("cleaning_profile")) not in allowed_profiles
            or str(row.get("structural_policy")) != "apply_after_review"
        ):
            raise ValueError(
                f"{stable_uid}: structural spans target a row outside the approved "
                "profile/policy route"
            )
        for span in spans:
            if span["input_text_sha256"] != text_hash or int(span["char_end"]) > len(text):
                raise ValueError(f"{stable_uid}: span inventory is out of bounds or text-unbound")

        admission = admission_for(
            str(row["acquisition_source_id"]),
            str(row["source_dataset"]),
            str(row["source_role"]),
            admissions,
        )
        decision = str(admission.get("decision", "pending"))
        if decision not in {"include", "exclude", "quarantine"}:
            raise ValueError(f"{stable_uid}: final source admission is non-terminal: {decision!r}")
        eligibility_name = str(row.get("training_eligibility") or "inherited_base")
        if eligibility_name not in eligibility_policy:
            raise ValueError(f"{stable_uid}: unreviewed eligibility category {eligibility_name!r}")
        source_id = str(row["acquisition_source_id"])
        category_ceiling = eligibility_policy[eligibility_name]
        source_eligibility = license_decision_for(
            source_id, eligibility_name, license_adjudication
        )
        for key in ("training_eligible", "redistribution_eligible"):
            if source_eligibility[key] and not category_ceiling[key]:
                raise ValueError(
                    f"{stable_uid}: {source_id} adjudication exceeds {eligibility_name!r} {key} ceiling"
                )
        action = "keep" if decision == "include" else ("quarantine" if decision == "quarantine" else "drop")
        raw_reasons = json.loads(str(ledger.get("reasons_json") or "[]"))
        if not isinstance(raw_reasons, list) or not all(
            isinstance(reason, str) for reason in raw_reasons
        ):
            raise ValueError(f"{stable_uid}: reasons_json is not a string list")
        reasons = set(raw_reasons)
        reasons.discard("post_clean_review_required")
        if decision != "include":
            reasons.add(f"source_admission:{decision}")

        toc_text, bibliography_text, union_text = (
            structural_counterfactuals(text, spans) if spans else (text, text, text)
        )
        trace = json.loads(str(row.get("cleaning_trace_json") or "[]"))
        if not isinstance(trace, list):
            raise ValueError(f"{stable_uid}: cleaning_trace_json is not a list")
        _, structural_rules, _ = apply_structural_spans(text, spans) if spans else (text, [], set())
        trace.extend({"rule": rule, "matches": 1, "stage": "structural_last"} for rule in structural_rules)
        item = {
            "row": row,
            "ledger": ledger,
            "text": text,
            "final_text": union_text,
            "toc_text": toc_text,
            "bibliography_text": bibliography_text,
            "has_toc": toc_text != text,
            "has_bibliography": bibliography_text != text,
            "has_union": union_text != text,
            "action": action,
            "decision": decision,
            "reasons": reasons,
            "eligibility_name": eligibility_name,
            "eligibility": source_eligibility,
            "trace": trace,
        }
        prepared.append(item)
        index = len(prepared) - 1
        for field, candidate, needed_change in (
            ("toc_after", toc_text, item["has_toc"]),
            ("bibliography_after", bibliography_text, item["has_bibliography"]),
            ("union_after", union_text, item["has_union"]),
        ):
            if needed_change:
                variants.append(candidate)
                needed.append((index, field))

    for (index, field), count in zip(needed, encode_counts(tokenizer, variants), strict=True):
        prepared[index][field] = count

    corpus: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    for item in prepared:
        row, ledger = item["row"], item["ledger"]
        before = int(ledger["tokens_pii_masked"])
        toc_after = int(item.get("toc_after", before))
        bibliography_after = int(item.get("bibliography_after", before))
        union_after = int(item.get("union_after", before))
        action = item["action"]
        if action == "keep" and not item["final_text"]:
            action = "quarantine"
            item["reasons"].add("structural_cleaning_emptied_document")
        eligible_training = bool(
            action == "keep" and item["decision"] == "include" and item["eligibility"]["training_eligible"]
        )
        eligible_redistribution = bool(
            eligible_training and item["eligibility"]["redistribution_eligible"]
        )
        final_hash = sha256_text(item["final_text"])
        ledger.update(
            {
                "action": action,
                "reasons_json": json.dumps(sorted(item["reasons"]), ensure_ascii=False),
                "tokens_toc_removed": before - toc_after,
                "tokens_bibliography_removed": before - bibliography_after,
                "tokens_structural_union_removed": before - union_after,
                "tokens_structural_cleaned": union_after,
                "tokens_final": union_after if eligible_training else 0,
                "characters_final": len(item["final_text"]) if action == "keep" else 0,
                "final_text_sha256": final_hash,
                "source_admission_decision": item["decision"],
                "training_eligibility_category": item["eligibility_name"],
                "eligible_for_training": eligible_training,
                "eligible_for_redistribution": eligible_redistribution,
            }
        )
        row.update(
            {
                "text": item["final_text"],
                "cleaned_text_sha256": final_hash,
                "cleaning_trace_json": json.dumps(item["trace"], ensure_ascii=False, sort_keys=True),
                "eligible_for_training": eligible_training,
                "eligible_for_redistribution": eligible_redistribution,
            }
        )
        if action == "keep":
            corpus.append(row)
        elif action == "quarantine":
            quarantine.append(row)
        source = str(row["acquisition_source_id"])
        counters[f"action:{action}"] += 1
        per_source[source][f"action:{action}"] += 1
        for field in (
            "tokens_normalized",
            "tokens_source_cleaned",
            "tokens_pii_masked",
            "tokens_toc_removed",
            "tokens_bibliography_removed",
            "tokens_structural_union_removed",
            "tokens_structural_cleaned",
            "tokens_final",
        ):
            value = int(ledger[field])
            counters[field] += value
            per_source[source][field] += value
    return corpus, quarantine, counters, per_source, seen_spans


def _update_noncorpus_ledger(row: dict[str, Any]) -> None:
    admissions = _WORKER["admissions"]
    decision = str(
        admission_for(
            str(row["acquisition_source_id"]),
            str(row["source_dataset"]),
            "",
            admissions,
        ).get("decision", "pending")
    )
    if decision not in {"include", "exclude", "quarantine"}:
        raise ValueError(f"{row['stable_uid']}: final source admission is non-terminal")
    row["source_admission_decision"] = decision
    row["eligible_for_training"] = False
    row["eligible_for_redistribution"] = False
    row["tokens_final"] = 0


def _process_file(job: dict[str, str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    corpus_input = Path(job["input"])
    ledger_input = Path(job["input_ledger"])
    quarantine_input = Path(job["input_quarantine"])
    output = Path(job["output"])
    ledger_output = Path(job["ledger"])
    quarantine_output = Path(job["quarantine"])
    relative = Path(job["relative"])
    roots = {key: Path(value) for key, value in _WORKER["roots"].items()}
    destinations = {"output": output, "ledger": ledger_output, "quarantine": quarantine_output}
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)
    writers = {
        "output": pq.ParquetWriter(output.with_suffix(output.suffix + ".partial"), cleaned_schema(), compression="zstd"),
        "ledger": pq.ParquetWriter(ledger_output.with_suffix(ledger_output.suffix + ".partial"), ledger_schema(), compression="zstd"),
        "quarantine": pq.ParquetWriter(quarantine_output.with_suffix(quarantine_output.suffix + ".partial"), cleaned_schema(), compression="zstd"),
    }
    batch_rows = int(_WORKER["batch_rows"])
    ledger_iterator = _iter_rows(ledger_input, batch_rows)
    current_ledger = next(ledger_iterator, None)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending_ledger: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    seen_spans = 0
    corpus_rows = quarantine_rows = ledger_rows = 0

    def flush() -> None:
        nonlocal pairs, pending_ledger, seen_spans, corpus_rows, quarantine_rows, ledger_rows
        if not pairs and not pending_ledger:
            return
        corpus, newly_quarantined, batch_counts, sources, span_count = _finalize_pairs(pairs)
        seen_spans += span_count
        counters.update(batch_counts)
        for source, values in sources.items():
            per_source[source].update(values)
        corpus_count = len(corpus)
        new_quarantine_count = len(newly_quarantined)
        pending_ledger_count = len(pending_ledger)
        _write_rows(writers["output"], corpus, cleaned_schema())
        _write_rows(writers["quarantine"], newly_quarantined, cleaned_schema())
        _write_rows(writers["ledger"], pending_ledger, ledger_schema())
        corpus_rows += corpus_count
        quarantine_rows += new_quarantine_count
        ledger_rows += pending_ledger_count
        pairs = []
        pending_ledger = []

    try:
        for corpus_row in _iter_rows(corpus_input, batch_rows):
            uid = str(corpus_row["stable_uid"])
            while current_ledger is not None and str(current_ledger["stable_uid"]) != uid:
                if current_ledger["action"] == "keep":
                    raise ValueError(f"{relative}: Stage 50 corpus/ledger order mismatch at {uid}")
                _update_noncorpus_ledger(current_ledger)
                source = str(current_ledger["acquisition_source_id"])
                counters[f"action:{current_ledger['action']}"] += 1
                per_source[source][f"action:{current_ledger['action']}"] += 1
                for field in (
                    "tokens_normalized",
                    "tokens_source_cleaned",
                    "tokens_pii_masked",
                    "tokens_toc_removed",
                    "tokens_bibliography_removed",
                    "tokens_structural_union_removed",
                    "tokens_structural_cleaned",
                    "tokens_final",
                ):
                    value = int(current_ledger[field])
                    counters[field] += value
                    per_source[source][field] += value
                pending_ledger.append(current_ledger)
                current_ledger = next(ledger_iterator, None)
            if current_ledger is None:
                raise ValueError(f"{relative}: Stage 50 ledger ended before corpus")
            pending_ledger.append(current_ledger)
            pairs.append((corpus_row, current_ledger))
            current_ledger = next(ledger_iterator, None)
            if len(pairs) >= batch_rows:
                flush()
        while current_ledger is not None:
            if current_ledger["action"] == "keep":
                raise ValueError(f"{relative}: Stage 50 ledger has an unmatched keep row")
            _update_noncorpus_ledger(current_ledger)
            source = str(current_ledger["acquisition_source_id"])
            counters[f"action:{current_ledger['action']}"] += 1
            per_source[source][f"action:{current_ledger['action']}"] += 1
            for field in (
                "tokens_normalized",
                "tokens_source_cleaned",
                "tokens_pii_masked",
                "tokens_toc_removed",
                "tokens_bibliography_removed",
                "tokens_structural_union_removed",
                "tokens_structural_cleaned",
                "tokens_final",
            ):
                value = int(current_ledger[field])
                counters[field] += value
                per_source[source][field] += value
            pending_ledger.append(current_ledger)
            current_ledger = next(ledger_iterator, None)
            if len(pending_ledger) >= batch_rows:
                flush()
        flush()

        # Previously quarantined Stage 50 rows remain quarantined.  They are
        # not structurally transformed because spans are generated on corpus.
        old_quarantine: list[dict[str, Any]] = []
        for row in _iter_rows(quarantine_input, batch_rows):
            row["eligible_for_training"] = False
            row["eligible_for_redistribution"] = False
            old_quarantine.append(row)
            if len(old_quarantine) >= batch_rows:
                quarantine_rows += len(old_quarantine)
                _write_rows(writers["quarantine"], old_quarantine, cleaned_schema())
        quarantine_rows += len(old_quarantine)
        _write_rows(writers["quarantine"], old_quarantine, cleaned_schema())
        for writer in writers.values():
            writer.close()
        for name, path in destinations.items():
            os.replace(path.with_suffix(path.suffix + ".partial"), path)
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for path in destinations.values():
            path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)
        raise

    receipt = {
        "schema_version": "full_cpt_final_cleaning_file_receipt_v1",
        "implementation_version": FINALIZER_VERSION,
        "config_sha256": _WORKER["config_sha256"],
        "relative_path": relative.as_posix(),
        "input": file_receipt(corpus_input),
        "input_ledger": file_receipt(ledger_input),
        "input_quarantine": file_receipt(quarantine_input),
        "output": file_receipt(output, relative_to=roots["output"]),
        "ledger": file_receipt(ledger_output, relative_to=roots["ledger"]),
        "quarantine": file_receipt(quarantine_output, relative_to=roots["quarantine"]),
        "corpus_rows": corpus_rows,
        "quarantine_rows": quarantine_rows,
        "ledger_rows": ledger_rows,
        "spans_seen": seen_spans,
        "counters": dict(sorted(counters.items())),
        "per_source": {key: dict(sorted(values.items())) for key, values in sorted(per_source.items())},
    }
    write_json_atomic(Path(job["receipt"]), receipt)
    return receipt


def _reusable_final_receipt(
    receipt_path: Path,
    *,
    config_sha256: str,
    inputs: Mapping[str, Path],
    roots: Mapping[str, Path],
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    try:
        value = read_json_object(receipt_path)
        if (
            value.get("schema_version") != "full_cpt_final_cleaning_file_receipt_v1"
            or value.get("implementation_version") != FINALIZER_VERSION
            or value.get("config_sha256") != config_sha256
        ):
            return None
        for name, path in inputs.items():
            receipt = value[name]
            if receipt.get("sha256") != sha256_file(path) or receipt.get("bytes") != path.stat().st_size:
                return None
        for name, root in roots.items():
            verify_file_receipt(value[name], relative_to=root)
        return value
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _auto_workers(value: int) -> int:
    if value > 0:
        return value
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, min(16, cpus // 4 or 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-ledger", type=Path, required=True)
    parser.add_argument("--input-quarantine", type=Path, required=True)
    parser.add_argument("--input-cleaning-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--license-adjudication", type=Path, required=True)
    parser.add_argument("--eligibility-policy", type=Path, required=True)
    parser.add_argument("--cleaning-policy", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--structural-decision", type=Path, required=True)
    parser.add_argument("--structural-spans", action="append", type=Path, default=[])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tokenizer-threads", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=2048)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")

    stage50 = _require_manifest(args.input_cleaning_manifest)
    stage50_sha256 = sha256_file(args.input_cleaning_manifest)
    if Path(str(stage50["output"])).resolve() != args.input.resolve():
        raise ValueError("Stage 50 manifest output differs from finalizer input")
    if Path(str(stage50["ledger"])).resolve() != args.input_ledger.resolve():
        raise ValueError("Stage 50 manifest ledger differs from finalizer input ledger")
    if Path(str(stage50["quarantine"])).resolve() != args.input_quarantine.resolve():
        raise ValueError("Stage 50 manifest quarantine differs from finalizer input quarantine")
    _validate_stage50_replay_inputs(
        stage50,
        tokenizer=args.tokenizer_json,
        source_config=args.source_config,
        license_adjudication=args.license_adjudication,
        eligibility_policy=args.eligibility_policy,
        cleaning_policy=args.cleaning_policy,
    )
    status, admissions = load_admission(args.source_admission)
    if status != "approved":
        raise ValueError("final source admission must be approved")
    if any(row.get("decision") not in {"include", "exclude", "quarantine"} for row in admissions.values()):
        raise ValueError("final source admission contains non-terminal decisions")
    eligibility = load_eligibility_policy(args.eligibility_policy)
    license_adjudication = load_license_adjudication(
        args.license_adjudication, source_registry_path=args.source_config
    )
    policy = read_json_object(args.cleaning_policy)
    decision = _load_decision(
        args.structural_decision,
        stage50_sha256=stage50_sha256,
        cleaning_policy_sha256=sha256_file(args.cleaning_policy),
        cleaning_policy=policy,
    )
    apply_structural = bool(decision["apply_structural"])
    if apply_structural and not args.structural_spans:
        raise ValueError("passing structural decision requires span inventory")
    if not apply_structural and args.structural_spans:
        raise ValueError("no-op structural decision cannot consume spans")

    model_receipt_sha = str(decision.get("model_receipt_sha256") or "")
    span_index = None
    if apply_structural:
        if not valid_sha256(model_receipt_sha):
            raise ValueError("passing structural decision lacks model receipt SHA-256")
        span_index = _build_span_index(
            args.structural_spans,
            database=args.work_dir / "structural-spans.sqlite",
            model_receipt_sha256=model_receipt_sha,
            stage50_manifest_sha256=stage50_sha256,
        )
        if int(span_index["rows"]) == 0:
            raise ValueError("passing structural decision requires at least one bound span")

    roots = {
        "output": args.output.resolve(),
        "ledger": args.ledger.resolve(),
        "quarantine": args.quarantine.resolve(),
    }
    workers = _auto_workers(args.workers)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    tokenizer_threads = (
        args.tokenizer_threads
        if args.tokenizer_threads > 0
        else max(1, allocated_cpus // workers // 2)
    )
    os.environ["RAYON_NUM_THREADS"] = str(tokenizer_threads)
    config = {
        "implementation_version": FINALIZER_VERSION,
        "stage50_cleaning_manifest_sha256": stage50_sha256,
        "source_admission": file_receipt(args.source_admission),
        "source_config": file_receipt(args.source_config),
        "license_adjudication": file_receipt(args.license_adjudication),
        "eligibility_policy": file_receipt(args.eligibility_policy),
        "cleaning_policy": file_receipt(args.cleaning_policy),
        "tokenizer": file_receipt(args.tokenizer_json),
        "structural_decision": file_receipt(args.structural_decision),
        "span_files": [file_receipt(path) for path in args.structural_spans],
        "batch_rows": args.batch_rows,
        "workers": workers,
        "tokenizer_threads_per_worker": tokenizer_threads,
    }
    config_sha256 = canonical_json_sha256(config)
    receipt_root = args.manifest.parent / "file_receipts"
    jobs: list[dict[str, str]] = []
    receipts: dict[str, dict[str, Any]] = {}
    stage50_output = Path(str(stage50["output"]))
    stage50_ledger = Path(str(stage50["ledger"]))
    stage50_quarantine = Path(str(stage50["quarantine"]))
    for file_row in stage50["files"]:
        relative = Path(str(file_row["relative_path"]))
        inputs = {
            "input": verify_file_receipt(file_row["output"], relative_to=stage50_output),
            "input_ledger": verify_file_receipt(file_row["ledger"], relative_to=stage50_ledger),
            "input_quarantine": verify_file_receipt(file_row["quarantine"], relative_to=stage50_quarantine),
        }
        receipt_path = per_file_receipt_path(receipt_root, relative)
        prior = _reusable_final_receipt(
            receipt_path, config_sha256=config_sha256, inputs=inputs, roots=roots
        )
        if prior is not None:
            receipts[relative.as_posix()] = prior
            continue
        jobs.append(
            {
                **{key: str(value) for key, value in inputs.items()},
                "relative": relative.as_posix(),
                "output": str(args.output / relative),
                "ledger": str(args.ledger / relative),
                "quarantine": str(args.quarantine / relative),
                "receipt": str(receipt_path),
            }
        )

    allowed_profiles = policy.get("structural", {}).get("allowed_apply_profiles", [])
    worker_config = {
        "tokenizer_json": str(args.tokenizer_json),
        "admissions": admissions,
        "eligibility_policy": eligibility,
        "license_adjudication": license_adjudication,
        "apply_structural": apply_structural,
        "allowed_profiles": allowed_profiles,
        "span_database": span_index["database"] if span_index else None,
        "batch_rows": args.batch_rows,
        "config_sha256": config_sha256,
        "roots": {key: str(value) for key, value in roots.items()},
    }
    if jobs and workers == 1:
        _init_worker(worker_config)
        for job in jobs:
            value = _process_file(job)
            receipts[value["relative_path"]] = value
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(worker_config,)) as executor:
            pending = {executor.submit(_process_file, job): job for job in jobs}
            for future in as_completed(pending):
                value = future.result()
                receipts[value["relative_path"]] = value

    ordered = [receipts[key] for key in sorted(receipts)]
    if len(ordered) != len(stage50["files"]):
        raise RuntimeError("not every Stage 50 shard produced a final cleaning receipt")
    relatives = [Path(receipt["relative_path"]) for receipt in ordered]
    for root in roots.values():
        require_exact_parquet_tree(root, relatives)
    seen_spans = sum(int(row["spans_seen"]) for row in ordered)
    expected_spans = int(span_index["rows"]) if span_index else 0
    if seen_spans != expected_spans:
        raise ValueError(
            f"structural inventory coverage mismatch: expected {expected_spans}, saw {seen_spans}"
        )
    counters: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in ordered:
        counters.update(row["counters"])
        for source, values in row["per_source"].items():
            per_source[source].update(values)
    payload = {
        "schema_version": "full_cpt_cleaning_manifest_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "implementation_version": FINALIZER_VERSION,
        "cleaning_pass": "structural_last_final",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "ledger": str(args.ledger.resolve()),
        "quarantine": str(args.quarantine.resolve()),
        "stage50_cleaning_manifest": str(args.input_cleaning_manifest.resolve()),
        "stage50_cleaning_manifest_sha256": stage50_sha256,
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
        "structural_decision": str(args.structural_decision.resolve()),
        "structural_decision_sha256": sha256_file(args.structural_decision),
        "structural_model_receipt_sha256": model_receipt_sha if apply_structural else None,
        "structural_applied": apply_structural,
        "structural_semantics": "validated_spans" if apply_structural else "deterministic_no_op",
        "span_inventory": span_index,
        "config_sha256": config_sha256,
        "workers": workers,
        "tokenizer_threads_per_worker": tokenizer_threads,
        "batch_rows": args.batch_rows,
        "resumed_files": len(stage50["files"]) - len(jobs),
        "counts": dict(sorted(counters.items())),
        "per_source": {key: dict(sorted(values.items())) for key, values in sorted(per_source.items())},
        "token_loss_semantics": {
            "tokens_toc_removed": "pre-structural tokens minus ToC-only counterfactual tokens",
            "tokens_bibliography_removed": "pre-structural tokens minus bibliography-only counterfactual tokens",
            "tokens_structural_union_removed": "pre-structural tokens minus actual union-cleaned tokens",
            "non_additivity": "kind counterfactuals need not sum to the union because tokenizer boundaries change",
            "final_text_sha256": "SHA256 of emitted text for keep rows; would-be final text for dropped/quarantined audit rows",
        },
        "files": ordered,
    }
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "structural_applied": apply_structural}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
