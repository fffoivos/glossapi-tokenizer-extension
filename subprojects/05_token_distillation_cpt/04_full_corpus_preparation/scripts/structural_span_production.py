#!/usr/bin/env python3
"""Receipt-bound production path for Phase-04 academic structural spans.

The Rust detector emits *raw* predictions against the immutable Stage-50 text.
Those predictions deliberately do not contain the final model-receipt hash: the
receipt depends on a later, independent manual safety audit.  Only ``rebind``
adds that hash after the audit and model receipt have passed.  This ordering
avoids a receipt/prediction hash cycle while keeping every deletion bound to the
exact text that was scored.

All corpus operations are streaming and CPU-only.  Detection checkpoints one
Stage-50 Parquet shard at a time, so an interrupted Clariden allocation resumes
from verified shard receipts instead of rescoring completed shards.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from cleaning_runtime import (
    canonical_json_sha256,
    encode_counts,
    file_receipt,
    valid_sha256,
    verify_file_receipt,
    write_json_atomic,
)
from full_corpus_io import read_json_object, sha256_file, sha256_text
from full_corpus_io import normalize_text
from structural_classifier_selection import validate_selection as validate_classifier_selection


RAW_SCHEMA = "phase04_structural_raw_predictions_v1"
RAW_SHARD_SCHEMA = "phase04_structural_raw_prediction_shard_v1"
PACKET_SCHEMA = "academic_structural_false_deletion_review_packet_v1"
PACKET_ROW_SCHEMA = "academic_structural_false_deletion_review_case_v1"
MANUAL_AUDIT_SCHEMA = "academic_structural_manual_audit_receipt_v1"
AUDIT_VALIDATION_SCHEMA = "academic_structural_audit_validation_v1"
MODEL_RECEIPT_SCHEMA = "academic_structural_model_receipt_v1"
SPAN_MANIFEST_SCHEMA = "phase04_structural_spans_manifest_v1"
DETECTOR_SOURCE = "phase04_stage50"
REQUIRED_REVIEW_CASES = 100
REQUIRED_REVIEW_CASES_PER_HEAD = 50
IMPLEMENTATION_VERSION = "phase04-structural-span-production-v1"
ELIGIBLE_STRUCTURAL_POLICY = "apply_after_review"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def _load(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    value = read_json_object(path)
    if schema is not None and value.get("schema_version") != schema:
        raise ValueError(
            f"{path}: expected {schema}, got {value.get('schema_version')!r}"
        )
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def _resolve_receipt(receipt: Mapping[str, Any], root: Path) -> Path:
    return verify_file_receipt(receipt, relative_to=root)


def _artifact_bundle(files: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    if not files:
        raise ValueError("artifact bundle cannot be empty")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for name, path in files:
        if not name or name in names:
            raise ValueError(f"duplicate/empty artifact name: {name!r}")
        names.add(name)
        receipt = file_receipt(path)
        rows.append({"name": name, **receipt})
    identity = [
        {"name": row["name"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in rows
    ]
    return {"sha256": canonical_json_sha256(identity), "files": rows}


def build_artifacts(
    *,
    code_files: Sequence[Path],
    sequence_config: Path,
    smoother: Path,
    bib_model: Path,
    toc_model: Path,
) -> dict[str, Any]:
    return {
        "code": _artifact_bundle([(f"code:{path.name}", path) for path in code_files]),
        "config": _artifact_bundle(
            [("sequence_config", sequence_config), ("structural_smoother", smoother)]
        ),
        "checkpoint": _artifact_bundle(
            [("bibliography_line_model", bib_model), ("toc_line_model", toc_model)]
        ),
    }


def _validate_artifacts(artifacts: Mapping[str, Any]) -> None:
    for category in ("code", "config", "checkpoint"):
        bundle = artifacts.get(category)
        if not isinstance(bundle, dict) or not valid_sha256(bundle.get("sha256")):
            raise ValueError(f"invalid structural {category} artifact bundle")
        files = bundle.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"structural {category} artifact bundle has no files")
        identity = []
        for row in files:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise ValueError(f"invalid structural {category} artifact row")
            path = verify_file_receipt(row)
            identity.append(
                {
                    "name": row["name"],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if canonical_json_sha256(identity) != bundle["sha256"]:
            raise ValueError(f"structural {category} artifact bundle identity drift")


def _artifact_path(artifacts: Mapping[str, Any], *, category: str, name: str) -> Path:
    bundle = artifacts.get(category)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("files"), list):
        raise ValueError(f"structural {category} artifact bundle is invalid")
    matches = [row for row in bundle["files"] if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            f"structural artifact {category}/{name} is absent or ambiguous"
        )
    return verify_file_receipt(matches[0])


def _stage50_shards(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load(manifest_path, schema="full_cpt_cleaning_manifest_v1")
    expected = {
        "status": "completed",
        "cleaning_pass": "post_source_post_pii",
        "structural_applied": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{manifest_path}: Stage50 {key} mismatch")
    root = Path(str(manifest.get("output", ""))).resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{manifest_path}: Stage50 manifest has no output files")
    shards: list[dict[str, Any]] = []
    observed: set[Path] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("output"), dict):
            raise ValueError(f"{manifest_path}: invalid Stage50 file receipt")
        relative = str(item.get("relative_path", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(
                f"{manifest_path}: invalid Stage50 relative path {relative!r}"
            )
        path = _resolve_receipt(item["output"], root)
        if path in observed:
            raise ValueError(f"{manifest_path}: duplicate Stage50 output path {path}")
        observed.add(path)
        shards.append(
            {
                "relative_path": relative,
                "path": path,
                "receipt": dict(item["output"]),
                "expected_rows": int(item.get("kept_rows", -1)),
            }
        )
    actual = {path.resolve() for path in root.rglob("*.parquet")}
    if actual != observed:
        raise ValueError(
            f"{manifest_path}: Stage50 output tree drift; missing={len(observed - actual)} "
            f"extra={len(actual - observed)}"
        )
    return manifest, sorted(shards, key=lambda row: row["relative_path"])


def _build_receipt_matches(
    build_receipt_path: Path,
    detector_binary: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    receipt = _load(build_receipt_path, schema="full_cpt_detector_build_receipt_v1")
    if receipt.get("status") != "passed":
        raise ValueError("detector build receipt is not passed")
    binary = receipt.get("binary")
    if not isinstance(binary, dict):
        raise ValueError("detector build receipt lacks binary")
    if binary.get("sha256") != sha256_file(detector_binary):
        raise ValueError("detector binary hash differs from build receipt")
    if int(binary.get("size", -1)) != detector_binary.stat().st_size:
        raise ValueError("detector binary size differs from build receipt")
    if Path(str(binary.get("path", ""))).resolve() != detector_binary.resolve():
        raise ValueError("detector binary path differs from build receipt")
    if expected_commit is not None and receipt.get("code_commit") != expected_commit:
        raise ValueError(
            "detector build receipt commit differs from the execution commit"
        )
    return receipt


def _validate_parity(
    parity_path: Path,
    *,
    detector_binary: Path,
    bib_model: Path,
    toc_model: Path,
    smoother: Path,
    silver_receipt: Mapping[str, Any] | None = None,
    silver_receipt_sha256: str | None = None,
    cleaning_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parity = _load(parity_path, schema="struct_rust_parity_receipt_v1")
    if (
        parity.get("status") != "passed"
        or parity.get("evidence_status") != "LLM_silver"
    ):
        raise ValueError("Rust parity receipt is not passed LLM-silver evidence")
    if parity.get("binary_sha256") != sha256_file(detector_binary):
        raise ValueError("Rust parity receipt does not bind the detector binary")
    expected = {
        "bib": sha256_file(bib_model),
        "toc": sha256_file(toc_model),
        "smoother": sha256_file(smoother),
    }
    if parity.get("model_sha256") != expected:
        raise ValueError("Rust parity receipt does not bind the exact model artifacts")
    if (silver_receipt is None) != (cleaning_policy is None) or (
        silver_receipt is None
    ) != (silver_receipt_sha256 is None):
        raise ValueError(
            "strict parity validation requires the silver receipt/hash and cleaning policy"
        )
    if silver_receipt is not None and cleaning_policy is not None:
        silver_sha = silver_receipt.get("silver_sha256")
        if not valid_sha256(silver_sha) or parity.get("corpus_sha256") != silver_sha:
            raise ValueError(
                "Rust parity corpus differs from the validated joint LLM-silver corpus"
            )
        if (
            parity.get("source_receipt_sha256") != silver_receipt_sha256
            or parity.get("evaluation_partition") != "validation"
            or parity.get("partition_semantics")
            != "derived_historical_train_validation_runtime_parity_not_quality_holdout"
            or parity.get("historical_test_documents_loaded") != 0
        ):
            raise ValueError(
                "Rust parity is not bound to the imported validation-only joint source"
            )
        validation = cleaning_policy.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(
                "cleaning policy lacks structural parity validation settings"
            )
        required_documents = int(validation.get("required_parity_documents", -1))
        if int(parity.get("heldout_documents", -1)) != required_documents:
            raise ValueError(
                f"Rust parity covers {parity.get('heldout_documents')} documents; "
                f"{required_documents} are required"
            )
        if any(
            int(parity.get("positive_document_counts", {}).get(head, 0)) <= 0
            for head in ("bib", "toc")
        ):
            raise ValueError(
                "Rust parity lacks positive held-out coverage for one or both heads"
            )
        maximum_delta = float(validation.get("maximum_probability_delta", -1.0))
        for head in ("bib", "toc"):
            result = parity.get("heads", {}).get(head, {})
            if int(result.get("span_mismatches", -1)) != 0:
                raise ValueError(f"Rust parity has decoded span mismatches for {head}")
            if float(result.get("max_probability_difference", 2.0)) > maximum_delta:
                raise ValueError(
                    f"Rust probability parity exceeds the policy delta for {head}"
                )
        if float(parity.get("tolerance", 2.0)) > maximum_delta:
            raise ValueError("Rust parity tolerance is looser than cleaning policy")
    return parity


def _raw_identity(
    *,
    stage50_manifest_sha256: str,
    detector_binary_sha256: str,
    detector_build_receipt_sha256: str,
    parity_receipt_sha256: str,
    cleaning_policy_sha256: str,
    allowed_apply_profiles: Sequence[str],
    eligible_structural_policy: str,
    artifacts: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "implementation_version": IMPLEMENTATION_VERSION,
            "stage50_cleaning_manifest_sha256": stage50_manifest_sha256,
            "detector_binary_sha256": detector_binary_sha256,
            "detector_build_receipt_sha256": detector_build_receipt_sha256,
            "parity_receipt_sha256": parity_receipt_sha256,
            "cleaning_policy_sha256": cleaning_policy_sha256,
            "allowed_apply_profiles": sorted(allowed_apply_profiles),
            "eligible_structural_policy": eligible_structural_policy,
            "artifact_sha256": {
                key: artifacts[key]["sha256"] for key in sorted(artifacts)
            },
        }
    )


def _row_uid(stable_uid: str) -> str:
    return hashlib.sha256(f"{DETECTOR_SOURCE}\0{stable_uid}".encode()).hexdigest()


def _shard_output_dir(root: Path, relative: str) -> Path:
    return root / "shards" / Path(relative).parent / f"{Path(relative).name}.structural"


def _shard_paths(directory: Path) -> dict[str, Path]:
    return {
        "index": directory / "input_text_index.jsonl",
        "counters": directory / "raw_counters.jsonl",
        "spans": directory / "raw_spans.jsonl",
        "receipt": directory / "shard_receipt.json",
    }


def _validate_shard_outputs(
    *,
    directory: Path,
    input_path: Path,
    input_receipt: Mapping[str, Any],
    expected_stage50_rows: int,
    allowed_apply_profiles: set[str],
    run_identity_sha256: str,
) -> dict[str, Any]:
    paths = _shard_paths(directory)
    receipt = _load(paths["receipt"], schema=RAW_SHARD_SCHEMA)
    if (
        receipt.get("status") != "completed"
        or receipt.get("run_identity_sha256") != run_identity_sha256
    ):
        raise ValueError(f"{directory}: raw shard run identity/status drift")
    if receipt.get("input") != dict(input_receipt):
        raise ValueError(f"{directory}: raw shard input receipt drift")
    if (
        not input_path.is_file()
        or input_path.stat().st_size != int(receipt["input"].get("bytes", -1))
        or sha256_file(input_path) != receipt["input"].get("sha256")
    ):
        raise ValueError(f"{directory}: raw shard Stage50 input bytes/hash drift")
    for name in ("index", "counters", "spans"):
        path = verify_file_receipt(receipt[name], relative_to=directory)
        if path != paths[name].resolve():
            raise ValueError(f"{directory}: raw shard {name} path drift")

    scanned_rows, eligible_rows = _validate_index_against_stage50(
        input_path=input_path,
        index_path=paths["index"],
        allowed_apply_profiles=allowed_apply_profiles,
    )
    if expected_stage50_rows >= 0 and scanned_rows != expected_stage50_rows:
        raise ValueError(
            f"{directory}: scanned Stage50 rows {scanned_rows} != kept_rows {expected_stage50_rows}"
        )
    if (
        scanned_rows != int(receipt.get("stage50_rows_scanned", -1))
        or eligible_rows != int(receipt.get("input_rows", -1))
        or scanned_rows - eligible_rows
        != int(receipt.get("excluded_nonacademic_rows", -1))
        or sorted(allowed_apply_profiles) != receipt.get("allowed_apply_profiles")
        or receipt.get("eligible_structural_policy") != ELIGIBLE_STRUCTURAL_POLICY
    ):
        raise ValueError(f"{directory}: academic-profile routing receipt drift")

    index_iter = _iter_jsonl(paths["index"])
    counter_iter = _iter_jsonl(paths["counters"])
    rows = 0
    models: dict[str, str] = {}
    for rows, (indexed, counter) in enumerate(
        zip(index_iter, counter_iter, strict=True), 1
    ):
        stable_uid = str(indexed.get("stable_uid", ""))
        text_hash = str(indexed.get("input_text_sha256", ""))
        if not valid_sha256(stable_uid) or not valid_sha256(text_hash):
            raise ValueError(f"{directory}: invalid indexed stable_uid/text hash")
        if (
            counter.get("doc_id") != stable_uid
            or counter.get("original_sha256") != text_hash
        ):
            raise ValueError(
                f"{directory}: Rust counter is not bound to indexed Stage50 text"
            )
        if counter.get("source") != DETECTOR_SOURCE or counter.get(
            "row_uid"
        ) != _row_uid(stable_uid):
            raise ValueError(f"{directory}: Rust counter identity mismatch")
        if int(counter.get("original_chars", -1)) != int(indexed.get("text_chars", -2)):
            raise ValueError(f"{directory}: Rust counter character count mismatch")
        for key in ("bib_model_id", "bib_decoder_id", "toc_model_id", "toc_decoder_id"):
            value = str(counter.get(key, ""))
            if not value:
                raise ValueError(f"{directory}: Rust counter lacks {key}")
            previous = models.setdefault(key, value)
            if previous != value:
                raise ValueError(
                    f"{directory}: Rust counter model identity drift for {key}"
                )
    if rows != eligible_rows:
        raise ValueError(
            f"{directory}: counter/index rows {rows} != eligible Stage50 rows {eligible_rows}"
        )
    if rows != int(receipt.get("input_rows", -1)):
        raise ValueError(f"{directory}: raw shard row count drift")

    temporary_root = Path(os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    temporary_root.mkdir(parents=True, exist_ok=True)
    descriptor, index_name = tempfile.mkstemp(
        prefix="phase04-structural-validate-", suffix=".sqlite", dir=temporary_root
    )
    os.close(descriptor)
    index_db = Path(index_name)
    index_db.unlink()
    connection = sqlite3.connect(index_db)
    connection.execute(
        "CREATE TABLE docs (stable_uid TEXT PRIMARY KEY, text_sha256 TEXT, text_chars INTEGER) WITHOUT ROWID"
    )
    try:
        for indexed in _iter_jsonl(paths["index"]):
            try:
                connection.execute(
                    "INSERT INTO docs VALUES (?, ?, ?)",
                    (
                        indexed["stable_uid"],
                        indexed["input_text_sha256"],
                        int(indexed["text_chars"]),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"{directory}: duplicate stable_uid in Stage50 shard"
                ) from exc
        connection.commit()
        span_rows = 0
        for span_rows, span in enumerate(_iter_jsonl(paths["spans"]), 1):
            stable_uid = str(span.get("doc_id", ""))
            found = connection.execute(
                "SELECT text_sha256, text_chars FROM docs WHERE stable_uid=?",
                (stable_uid,),
            ).fetchone()
            if found is None:
                raise ValueError(
                    f"{directory}: raw prediction references an unknown stable_uid"
                )
            if span.get("source") != DETECTOR_SOURCE or span.get("row_uid") != _row_uid(
                stable_uid
            ):
                raise ValueError(f"{directory}: raw prediction identity mismatch")
            if (
                span.get("original_sha256") != found[0]
                or int(span.get("original_chars", -1)) != found[1]
            ):
                raise ValueError(f"{directory}: raw prediction text binding mismatch")
            start, end = int(span.get("char_start", -1)), int(span.get("char_end", -1))
            if start < 0 or end <= start or end > found[1]:
                raise ValueError(
                    f"{directory}: raw prediction has invalid character offsets"
                )
            if span.get("kind") not in {"bib_span", "toc_span"}:
                raise ValueError(f"{directory}: unsupported raw prediction kind")
            expected_model = (
                f"{models.get('bib_model_id', '')}:{models.get('bib_decoder_id', '')}"
                if span["kind"] == "bib_span"
                else f"{models.get('toc_model_id', '')}:{models.get('toc_decoder_id', '')}"
            )
            if span.get("model_id") != expected_model:
                raise ValueError(f"{directory}: raw prediction model identity mismatch")
        if span_rows != int(receipt.get("span_rows", -1)):
            raise ValueError(f"{directory}: raw prediction row count drift")
    finally:
        connection.close()
        index_db.unlink(missing_ok=True)
    if receipt.get("models") != models:
        raise ValueError(f"{directory}: model identity receipt drift")
    return receipt


def _validate_index_against_stage50(
    *,
    input_path: Path,
    index_path: Path,
    allowed_apply_profiles: set[str],
    batch_rows: int = 4096,
) -> tuple[int, int]:
    """Recompute the exact academic-profile route and compare its ordered index."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(input_path)
    required = {
        "stable_uid",
        "source_dataset",
        "text",
        "cleaning_profile",
        "structural_policy",
        "eligible_for_training",
    }
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{input_path}: missing Stage50 detector columns {missing}")
    indexed = iter(_iter_jsonl(index_path))
    scanned, eligible = parquet.metadata.num_rows, 0
    for (
        row_number,
        stable_uid,
        source_dataset,
        text,
        profile,
        structural_policy,
        _eligible_for_training,
    ) in _iter_routed_rows(
        parquet,
        allowed_apply_profiles=allowed_apply_profiles,
        batch_rows=batch_rows,
    ):
        eligible += 1
        try:
            row = next(indexed)
        except StopIteration as exc:
            raise ValueError(
                f"{index_path}: missing eligible Stage50 index row"
            ) from exc
        expected = {
            "stable_uid": stable_uid,
            "source_dataset": source_dataset,
            "input_text_sha256": sha256_text(text),
            "text_chars": len(text),
            "cleaning_profile": profile,
            "structural_policy": structural_policy,
            "row_number": row_number,
        }
        if row != expected:
            raise ValueError(
                f"{index_path}: eligible Stage50 index/content drift at row {row_number}"
            )
    try:
        extra = next(indexed)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError(
            f"{index_path}: contains rows outside the eligible Stage50 subset"
        )
    return scanned, eligible


def _iter_routed_rows(
    parquet: Any, *, allowed_apply_profiles: set[str], batch_rows: int
) -> Iterator[tuple[int, str, str, str, str, str, bool]]:
    """Read text only for row groups containing an explicitly eligible row.

    Stage50 can be tens of billions of tokens. Routing columns are tiny, so we
    inspect them first and avoid decompressing the text column for wholly
    nonacademic HPLT/news/legal row groups.
    """

    full_columns = [
        "stable_uid",
        "source_dataset",
        "text",
        "cleaning_profile",
        "structural_policy",
        "eligible_for_training",
    ]
    row_offset = 0
    for row_group in range(parquet.num_row_groups):
        route = parquet.read_row_group(
            row_group, columns=["cleaning_profile", "structural_policy"]
        ).to_pydict()
        row_group_rows = len(route["cleaning_profile"])
        has_eligible = any(
            str(profile or "") in allowed_apply_profiles
            and str(policy or "") == ELIGIBLE_STRUCTURAL_POLICY
            for profile, policy in zip(
                route["cleaning_profile"], route["structural_policy"], strict=True
            )
        )
        if not has_eligible:
            row_offset += row_group_rows
            continue
        local_offset = 0
        for batch in parquet.iter_batches(
            row_groups=[row_group], batch_size=batch_rows, columns=full_columns
        ):
            values = batch.to_pydict()
            for local_index, (
                stable_uid,
                source_dataset,
                raw_text,
                raw_profile,
                raw_structural_policy,
                eligible_for_training,
            ) in enumerate(
                zip(
                    values["stable_uid"],
                    values["source_dataset"],
                    values["text"],
                    values["cleaning_profile"],
                    values["structural_policy"],
                    values["eligible_for_training"],
                    strict=True,
                )
            ):
                profile = str(raw_profile or "")
                structural_policy = str(raw_structural_policy or "")
                if (
                    profile not in allowed_apply_profiles
                    or structural_policy != ELIGIBLE_STRUCTURAL_POLICY
                ):
                    continue
                yield (
                    row_offset + local_offset + local_index,
                    str(stable_uid or ""),
                    str(source_dataset or ""),
                    str(raw_text or ""),
                    profile,
                    structural_policy,
                    bool(eligible_for_training),
                )
            local_offset += batch.num_rows
        if local_offset != row_group_rows:
            raise ValueError("Parquet row-group routing/read count drift")
        row_offset += row_group_rows
    if row_offset != parquet.metadata.num_rows:
        raise ValueError("Parquet routing coverage differs from metadata row count")


def _stream_shard_to_detector(
    *,
    input_path: Path,
    input_receipt: Mapping[str, Any],
    expected_rows: int,
    output_dir: Path,
    detector_binary: Path,
    run_identity_sha256: str,
    batch_rows: int,
    rayon_threads: int,
    allowed_apply_profiles: set[str],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    partial = output_dir.with_name(output_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    paths = _shard_paths(partial)
    environment = dict(os.environ)
    environment.update(
        {
            "RAYON_NUM_THREADS": str(rayon_threads),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        str(detector_binary),
        "--mode",
        "structure-spans",
        "--source",
        DETECTOR_SOURCE,
        "--input",
        "-",
        "--text-field",
        "text",
        "--id-field",
        "id",
        "--out-spans",
        str(paths["spans"]),
        "--out-counters",
        str(paths["counters"]),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, text=True, encoding="utf-8", env=environment
    )
    if process.stdin is None:  # pragma: no cover - subprocess invariant
        raise RuntimeError("detector stdin pipe was not created")
    rows = scanned_rows = 0
    try:
        parquet = pq.ParquetFile(input_path)
        required = {
            "stable_uid",
            "source_dataset",
            "text",
            "cleaning_profile",
            "structural_policy",
            "eligible_for_training",
        }
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(
                f"{input_path}: missing Stage50 detector columns {missing}"
            )
        scanned_rows = parquet.metadata.num_rows
        with paths["index"].open("w", encoding="utf-8") as index_handle:
            for (
                row_number,
                uid,
                source_dataset,
                text,
                profile,
                structural_policy,
                _eligible_for_training,
            ) in _iter_routed_rows(
                parquet,
                allowed_apply_profiles=allowed_apply_profiles,
                batch_rows=batch_rows,
            ):
                rows += 1
                if not valid_sha256(uid):
                    raise ValueError(
                        f"{input_path}: row {row_number} has invalid stable_uid"
                    )
                if not source_dataset:
                    raise ValueError(
                        f"{input_path}: row {row_number} has empty source_dataset"
                    )
                text_hash = sha256_text(text)
                index_handle.write(
                    _json_line(
                        {
                            "stable_uid": uid,
                            "source_dataset": source_dataset,
                            "input_text_sha256": text_hash,
                            "text_chars": len(text),
                            "cleaning_profile": profile,
                            "structural_policy": structural_policy,
                            "row_number": row_number,
                        }
                    )
                )
                process.stdin.write(_json_line({"id": uid, "text": text}))
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"reference_detect exited with status {return_code}")
        for path in (paths["index"], paths["counters"], paths["spans"]):
            if not path.is_file():
                raise ValueError(f"reference_detect output missing: {path}")
        # First-pass counters establish the immutable model IDs for this shard.
        model_sets: dict[str, set[str]] = collections.defaultdict(set)
        counter_rows = 0
        for counter_rows, counter in enumerate(_iter_jsonl(paths["counters"]), 1):
            if "error" in counter:
                raise ValueError(f"{input_path}: detector emitted an error counter")
            for key in (
                "bib_model_id",
                "bib_decoder_id",
                "toc_model_id",
                "toc_decoder_id",
            ):
                model_sets[key].add(str(counter.get(key, "")))
        models = {
            key: next(iter(values))
            for key, values in model_sets.items()
            if len(values) == 1 and "" not in values
        }
        if counter_rows and len(models) != 4:
            raise ValueError(
                f"{input_path}: detector model IDs are missing or inconsistent"
            )
        span_rows = sum(1 for _ in _iter_jsonl(paths["spans"]))
        receipt = {
            "schema_version": RAW_SHARD_SCHEMA,
            "status": "completed",
            "created_at": _utc_now(),
            "implementation_version": IMPLEMENTATION_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "input": dict(input_receipt),
            "stage50_rows_scanned": scanned_rows,
            "input_rows": rows,
            "excluded_nonacademic_rows": scanned_rows - rows,
            "allowed_apply_profiles": sorted(allowed_apply_profiles),
            "eligible_structural_policy": ELIGIBLE_STRUCTURAL_POLICY,
            "span_rows": span_rows,
            "models": models,
            "index": file_receipt(paths["index"], relative_to=partial),
            "counters": file_receipt(paths["counters"], relative_to=partial),
            "spans": file_receipt(paths["spans"], relative_to=partial),
        }
        write_json_atomic(paths["receipt"], receipt)
        _validate_shard_outputs(
            directory=partial,
            input_path=input_path,
            input_receipt=input_receipt,
            expected_stage50_rows=expected_rows,
            allowed_apply_profiles=allowed_apply_profiles,
            run_identity_sha256=run_identity_sha256,
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to replace completed raw shard: {output_dir}"
            )
        os.replace(partial, output_dir)
        return _validate_shard_outputs(
            directory=output_dir,
            input_path=input_path,
            input_receipt=input_receipt,
            expected_stage50_rows=expected_rows,
            allowed_apply_profiles=allowed_apply_profiles,
            run_identity_sha256=run_identity_sha256,
        )
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _aggregate_raw(
    *,
    output_root: Path,
    shards: Sequence[dict[str, Any]],
    run_identity_sha256: str,
    allowed_apply_profiles: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str], int]:
    global_index = output_root / ".global-input-index.sqlite"
    global_index.unlink(missing_ok=True)
    connection = sqlite3.connect(global_index)
    connection.execute(
        "CREATE TABLE docs (stable_uid TEXT PRIMARY KEY, text_sha256 TEXT NOT NULL, source_dataset TEXT NOT NULL) WITHOUT ROWID"
    )
    records: list[dict[str, Any]] = []
    totals = collections.Counter()
    models: dict[str, str] = {}
    overlap_pairs = 0
    try:
        for shard in shards:
            directory = _shard_output_dir(output_root, shard["relative_path"])
            receipt = _validate_shard_outputs(
                directory=directory,
                input_path=shard["path"],
                input_receipt=shard["receipt"],
                expected_stage50_rows=shard["expected_rows"],
                allowed_apply_profiles=allowed_apply_profiles,
                run_identity_sha256=run_identity_sha256,
            )
            for key, value in receipt.get("models", {}).items():
                previous = models.setdefault(key, value)
                if previous != value:
                    raise ValueError(
                        f"detector model identity drifts across Stage50 shards: {key}"
                    )
            for indexed in _iter_jsonl(_shard_paths(directory)["index"]):
                try:
                    connection.execute(
                        "INSERT INTO docs VALUES (?, ?, ?)",
                        (
                            indexed["stable_uid"],
                            indexed["input_text_sha256"],
                            indexed["source_dataset"],
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"duplicate stable_uid across Stage50 shards: {indexed['stable_uid']}"
                    ) from exc
            connection.commit()
            counters_path = _shard_paths(directory)["counters"]
            for counter in _iter_jsonl(counters_path):
                totals["documents"] += 1
                totals["bibliography_spans"] += int(counter.get("bib_spans", 0))
                totals["toc_spans"] += int(counter.get("toc_spans", 0))
                totals["bibliography_lines"] += int(counter.get("bib_lines", 0))
                totals["toc_lines"] += int(counter.get("toc_lines", 0))
                overlap_pairs += int(counter.get("overlap_pairs", 0))
            totals["prediction_rows"] += int(receipt["span_rows"])
            totals["stage50_rows_scanned"] += int(receipt["stage50_rows_scanned"])
            totals["excluded_nonacademic_rows"] += int(
                receipt["excluded_nonacademic_rows"]
            )
            records.append(
                {
                    "relative_path": shard["relative_path"],
                    "input": dict(shard["receipt"]),
                    # These top-level manifest receipts stay absolute because
                    # the generic stage contract revalidates them without
                    # schema-specific path-root inference.
                    "receipt": file_receipt(_shard_paths(directory)["receipt"]),
                    "index": file_receipt(_shard_paths(directory)["index"]),
                    "counters": file_receipt(_shard_paths(directory)["counters"]),
                    "spans": file_receipt(_shard_paths(directory)["spans"]),
                    "input_rows": int(receipt["input_rows"]),
                    "stage50_rows_scanned": int(receipt["stage50_rows_scanned"]),
                    "excluded_nonacademic_rows": int(
                        receipt["excluded_nonacademic_rows"]
                    ),
                    "span_rows": int(receipt["span_rows"]),
                }
            )
    finally:
        connection.close()
        global_index.unlink(missing_ok=True)
    if totals["documents"] <= 0 or len(models) != 4:
        raise ValueError(
            "Stage50 structural detector produced no document/model identity coverage"
        )
    return records, dict(sorted(totals.items())), models, overlap_pairs


def detect(args: argparse.Namespace) -> int:
    if args.batch_rows < 1 or args.rayon_threads < 1:
        raise ValueError("batch rows and Rayon threads must be positive")
    stage50, shards = _stage50_shards(args.stage50_cleaning_manifest)
    stage50_sha = sha256_file(args.stage50_cleaning_manifest)
    build = _build_receipt_matches(
        args.detector_build_receipt,
        args.detector_binary,
        expected_commit=args.code_commit,
    )
    parity_receipt_sha256 = "unavailable"
    if args.parity_receipt is not None:
        _validate_parity(
            args.parity_receipt,
            detector_binary=args.detector_binary,
            bib_model=args.bib_model,
            toc_model=args.toc_model,
            smoother=args.smoother,
        )
        parity_receipt_sha256 = sha256_file(args.parity_receipt)
    artifacts = build_artifacts(
        code_files=args.model_code,
        sequence_config=args.sequence_config,
        smoother=args.smoother,
        bib_model=args.bib_model,
        toc_model=args.toc_model,
    )
    _validate_artifacts(artifacts)
    cleaning_policy = _load(args.cleaning_policy, schema="full_cpt_cleaning_policy_v1")
    raw_profiles = cleaning_policy.get("structural", {}).get("allowed_apply_profiles")
    if (
        not isinstance(raw_profiles, list)
        or not raw_profiles
        or any(not isinstance(value, str) or not value for value in raw_profiles)
    ):
        raise ValueError(
            "cleaning policy has no valid structural.allowed_apply_profiles"
        )
    allowed_apply_profiles = set(raw_profiles)
    identity = _raw_identity(
        stage50_manifest_sha256=stage50_sha,
        detector_binary_sha256=sha256_file(args.detector_binary),
        detector_build_receipt_sha256=sha256_file(args.detector_build_receipt),
        parity_receipt_sha256=parity_receipt_sha256,
        cleaning_policy_sha256=sha256_file(args.cleaning_policy),
        allowed_apply_profiles=sorted(allowed_apply_profiles),
        eligible_structural_policy=ELIGIBLE_STRUCTURAL_POLICY,
        artifacts=artifacts,
    )
    if args.manifest.exists():
        validate_raw_manifest(args.manifest)
        existing = _load(args.manifest, schema=RAW_SCHEMA)
        if existing.get("run_identity_sha256") != identity:
            raise ValueError(
                "completed raw detector manifest has a different run identity"
            )
        print(json.dumps({"ok": True, "resumed": True, "manifest": str(args.manifest)}))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resumed = 0
    for shard in shards:
        directory = _shard_output_dir(args.output_dir, shard["relative_path"])
        if directory.exists():
            _validate_shard_outputs(
                directory=directory,
                input_path=shard["path"],
                input_receipt=shard["receipt"],
                expected_stage50_rows=shard["expected_rows"],
                allowed_apply_profiles=allowed_apply_profiles,
                run_identity_sha256=identity,
            )
            resumed += 1
            continue
        _stream_shard_to_detector(
            input_path=shard["path"],
            input_receipt=shard["receipt"],
            expected_rows=shard["expected_rows"],
            output_dir=directory,
            detector_binary=args.detector_binary,
            run_identity_sha256=identity,
            batch_rows=args.batch_rows,
            rayon_threads=args.rayon_threads,
            allowed_apply_profiles=allowed_apply_profiles,
        )
    records, totals, models, overlap_pairs = _aggregate_raw(
        output_root=args.output_dir,
        shards=shards,
        run_identity_sha256=identity,
        allowed_apply_profiles=allowed_apply_profiles,
    )
    payload = {
        "schema_version": RAW_SCHEMA,
        "status": "completed",
        "created_at": _utc_now(),
        "implementation_version": IMPLEMENTATION_VERSION,
        "mode": "structure-spans",
        "detector_source": DETECTOR_SOURCE,
        "character_offset_unit": "Unicode scalar values / Python code points",
        "code_commit": args.code_commit,
        "run_identity_sha256": identity,
        "stage50_cleaning_manifest": file_receipt(args.stage50_cleaning_manifest),
        "stage50_cleaning_manifest_sha256": stage50_sha,
        "stage50_corpus_root": str(Path(stage50["output"]).resolve()),
        "cleaning_policy": file_receipt(args.cleaning_policy),
        "cleaning_policy_sha256": sha256_file(args.cleaning_policy),
        "allowed_apply_profiles": sorted(allowed_apply_profiles),
        "eligible_structural_policy": ELIGIBLE_STRUCTURAL_POLICY,
        "output_root": str(args.output_dir.resolve()),
        "detector": {
            "binary": file_receipt(args.detector_binary),
            "build_receipt": file_receipt(args.detector_build_receipt),
            "parity_receipt": (
                file_receipt(args.parity_receipt)
                if args.parity_receipt is not None
                else None
            ),
            "parity_status": "passed"
            if args.parity_receipt is not None
            else "unavailable",
            "build_code_commit": build["code_commit"],
        },
        "artifacts": artifacts,
        "models": models,
        "input_text_inventory_sha256": canonical_json_sha256(
            [
                {
                    "relative_path": row["relative_path"],
                    "rows": row["input_rows"],
                    "sha256": row["index"]["sha256"],
                }
                for row in records
            ]
        ),
        "raw_prediction_inventory_sha256": canonical_json_sha256(
            [
                {
                    "relative_path": row["relative_path"],
                    "rows": row["span_rows"],
                    "sha256": row["spans"]["sha256"],
                }
                for row in records
            ]
        ),
        "counts": totals,
        "conflicts": {
            "overlap_pairs": overlap_pairs,
            "strict_rebind_eligible": overlap_pairs == 0,
        },
        "files": records,
        "resumed_shards": resumed,
    }
    write_json_atomic(args.manifest, payload)
    validate_raw_manifest(args.manifest)
    print(
        json.dumps(
            {"ok": True, "manifest": str(args.manifest), "counts": totals},
            sort_keys=True,
        )
    )
    return 0


def validate_raw_manifest(path: Path) -> dict[str, Any]:
    manifest = _load(path, schema=RAW_SCHEMA)
    if (
        manifest.get("status") != "completed"
        or manifest.get("mode") != "structure-spans"
    ):
        raise ValueError(
            f"{path}: raw detector manifest is not completed structure-spans"
        )
    stage50_receipt = manifest.get("stage50_cleaning_manifest")
    if not isinstance(stage50_receipt, dict):
        raise ValueError(f"{path}: raw detector manifest lacks Stage50 receipt")
    stage50_path = verify_file_receipt(stage50_receipt)
    if sha256_file(stage50_path) != manifest.get("stage50_cleaning_manifest_sha256"):
        raise ValueError(f"{path}: Stage50 manifest hash drift")
    _stage50, stage50_shards = _stage50_shards(stage50_path)
    stage50_by_relative = {row["relative_path"]: row for row in stage50_shards}
    policy_path = verify_file_receipt(manifest["cleaning_policy"])
    policy = _load(policy_path, schema="full_cpt_cleaning_policy_v1")
    profiles = policy.get("structural", {}).get("allowed_apply_profiles")
    if (
        sha256_file(policy_path) != manifest.get("cleaning_policy_sha256")
        or not isinstance(profiles, list)
        or sorted(profiles) != manifest.get("allowed_apply_profiles")
        or manifest.get("eligible_structural_policy") != ELIGIBLE_STRUCTURAL_POLICY
    ):
        raise ValueError(f"{path}: structural cleaning-profile policy drift")
    allowed_apply_profiles = set(profiles)
    detector = manifest.get("detector")
    if not isinstance(detector, dict):
        raise ValueError(f"{path}: raw detector identity missing")
    binary = verify_file_receipt(detector["binary"])
    build_path = verify_file_receipt(detector["build_receipt"])
    parity_record = detector.get("parity_receipt")
    parity_path = (
        verify_file_receipt(parity_record) if isinstance(parity_record, dict) else None
    )
    if (parity_path is None) != (detector.get("parity_status") == "unavailable"):
        raise ValueError(f"{path}: raw detector parity status/receipt disagree")
    build = _build_receipt_matches(
        build_path, binary, expected_commit=manifest.get("code_commit")
    )
    if build.get("code_commit") != detector.get("build_code_commit"):
        raise ValueError(f"{path}: detector build commit drift")
    _validate_artifacts(manifest.get("artifacts", {}))
    if parity_path is not None:
        _validate_parity(
            parity_path,
            detector_binary=binary,
            bib_model=_artifact_path(
                manifest["artifacts"],
                category="checkpoint",
                name="bibliography_line_model",
            ),
            toc_model=_artifact_path(
                manifest["artifacts"], category="checkpoint", name="toc_line_model"
            ),
            smoother=_artifact_path(
                manifest["artifacts"],
                category="config",
                name="structural_smoother",
            ),
        )
    files = manifest.get("files")
    root = Path(str(manifest.get("output_root", "")))
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path}: raw detector manifest has no shard files")
    totals = collections.Counter()
    for row in files:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: invalid raw detector shard row")
        relative = str(row.get("relative_path", ""))
        stage50_shard = stage50_by_relative.get(relative)
        if stage50_shard is None:
            raise ValueError(
                f"{path}: raw detector references unknown Stage50 shard {relative!r}"
            )
        for name in ("receipt", "index", "counters", "spans"):
            _resolve_receipt(row[name], root)
        receipt_path = _resolve_receipt(row["receipt"], root)
        _validate_shard_outputs(
            directory=receipt_path.parent,
            input_path=stage50_shard["path"],
            input_receipt=stage50_shard["receipt"],
            expected_stage50_rows=stage50_shard["expected_rows"],
            allowed_apply_profiles=allowed_apply_profiles,
            run_identity_sha256=str(manifest.get("run_identity_sha256", "")),
        )
        totals["documents"] += int(row.get("input_rows", -1))
        totals["prediction_rows"] += int(row.get("span_rows", -1))
    if set(stage50_by_relative) != {str(row.get("relative_path", "")) for row in files}:
        raise ValueError(f"{path}: raw detector shard coverage differs from Stage50")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or totals["documents"] != int(
        counts.get("documents", -1)
    ):
        raise ValueError(f"{path}: raw detector document count drift")
    if totals["prediction_rows"] != int(counts.get("prediction_rows", -1)):
        raise ValueError(f"{path}: raw detector prediction count drift")
    identity = _raw_identity(
        stage50_manifest_sha256=manifest["stage50_cleaning_manifest_sha256"],
        detector_binary_sha256=sha256_file(binary),
        detector_build_receipt_sha256=sha256_file(build_path),
        parity_receipt_sha256=(
            sha256_file(parity_path) if parity_path is not None else "unavailable"
        ),
        cleaning_policy_sha256=sha256_file(policy_path),
        allowed_apply_profiles=sorted(allowed_apply_profiles),
        eligible_structural_policy=ELIGIBLE_STRUCTURAL_POLICY,
        artifacts=manifest["artifacts"],
    )
    if identity != manifest.get("run_identity_sha256"):
        raise ValueError(f"{path}: raw detector run identity drift")
    return manifest


def _prediction_id(span: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "stable_uid": span["doc_id"],
            "input_text_sha256": span["original_sha256"],
            "kind": span["kind"],
            "char_start": int(span["char_start"]),
            "char_end": int(span["char_end"]),
            "model_id": span["model_id"],
        }
    )


_PROBABILITY = re.compile(r"\bp=([01](?:\.\d+)?)\b")
_STRUCTURAL_MARKER = re.compile(
    r"(?i)(βιβλιογραφ|αναφορ|πηγ|references|bibliograph|περιεχ[όο]μενα|table of contents|"
    r"\.{4,}|_{4,}|https?://|doi\.|\b(?:19|20)\d{2}\b)"
)


def _risk(
    span: Mapping[str, Any], text: str, *, overlap: bool
) -> tuple[float, dict[str, Any]]:
    start, end = int(span["char_start"]), int(span["char_end"])
    excerpt = text[start:end]
    chars = max(1, len(excerpt))
    letters = sum(character.isalpha() for character in excerpt)
    lines = [line for line in excerpt.splitlines() if line.strip()]
    long_lines = sum(len(line.strip()) >= 120 for line in lines)
    probability_match = _PROBABILITY.search(str(span.get("gated_by", "")))
    probability = float(probability_match.group(1)) if probability_match else None
    deletion_fraction = chars / max(1, len(text))
    alphabetic_fraction = letters / chars
    long_line_fraction = long_lines / max(1, len(lines))
    marker_absent = not bool(_STRUCTURAL_MARKER.search(excerpt))
    # This is solely a deterministic prioritisation heuristic.  It is never a
    # safety metric or an automatic label.
    score = (
        4.0 * min(1.0, deletion_fraction)
        + 1.5 * alphabetic_fraction
        + 2.0 * long_line_fraction
        + (1.0 if marker_absent else 0.0)
        + (2.0 if overlap else 0.0)
        + (1.0 - probability if probability is not None else 0.5)
    )
    return score, {
        "deletion_fraction": deletion_fraction,
        "alphabetic_fraction": alphabetic_fraction,
        "long_line_fraction": long_line_fraction,
        "structural_marker_absent": marker_absent,
        "overlapping_prediction": overlap,
        "opening_probability": probability,
    }


def _verify_python_codepoint_span(
    span: Mapping[str, Any], text: str
) -> tuple[int, int]:
    """Prove that Rust ``char`` offsets index the same Python text slice.

    Rust's detector counts Unicode scalar values with ``.chars()``. Python 3
    indexes ``str`` by Unicode code point, including one position for astral
    characters. The line-boundary and trigger checks make an accidental UTF-8
    byte offset fail even when it happens to remain within ``len(text)``.
    """

    start, end = int(span["char_start"]), int(span["char_end"])
    if start < 0 or end <= start or end > len(text):
        raise ValueError("raw prediction offsets are invalid on Stage50 text")
    if int(span.get("original_chars", -1)) != len(text):
        raise ValueError("Rust/Python document code-point count mismatch")
    if (start and text[start - 1] != "\n") or (end < len(text) and text[end] != "\n"):
        raise ValueError(
            "Rust structural offsets are not Python code-point line boundaries"
        )
    expected_trigger = text[start:end].split("\n", 1)[0][:40]
    if span.get("trigger") != expected_trigger:
        raise ValueError("Rust trigger does not match Python code-point slicing")
    return start, end


def _raw_span_database(raw: Mapping[str, Any], database: Path) -> sqlite3.Connection:
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE spans (
          prediction_id TEXT PRIMARY KEY,
          stable_uid TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          kind TEXT NOT NULL,
          char_start INTEGER NOT NULL,
          char_end INTEGER NOT NULL,
          payload TEXT NOT NULL,
          overlaps INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID
        """
    )
    root = Path(raw["output_root"])
    for shard in raw["files"]:
        spans_path = _resolve_receipt(shard["spans"], root)
        for span in _iter_jsonl(spans_path):
            prediction_id = _prediction_id(span)
            try:
                connection.execute(
                    "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        prediction_id,
                        span["doc_id"],
                        span["original_sha256"],
                        span["kind"],
                        int(span["char_start"]),
                        int(span["char_end"]),
                        json.dumps(span, ensure_ascii=False, sort_keys=True),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"duplicate raw structural prediction {prediction_id}"
                ) from exc
    connection.execute(
        "CREATE INDEX spans_uid ON spans(stable_uid, char_start, char_end)"
    )
    connection.execute(
        """
        UPDATE spans SET overlaps=1 WHERE EXISTS (
          SELECT 1 FROM spans AS other
          WHERE other.stable_uid=spans.stable_uid
            AND other.prediction_id<>spans.prediction_id
            AND other.char_start < spans.char_end
            AND spans.char_start < other.char_end
        )
        """
    )
    connection.commit()
    return connection


def _balanced_rows(
    connection: sqlite3.Connection, *, limit: int, kind: str | None = None
) -> list[dict[str, Any]]:
    where, parameters = ("WHERE kind=?", (kind,)) if kind is not None else ("", ())
    sources = [
        row[0]
        for row in connection.execute(
            f"SELECT source_dataset FROM candidates {where} GROUP BY source_dataset "
            "ORDER BY MAX(risk_score) DESC, source_dataset",
            parameters,
        )
    ]
    queues: dict[str, Iterator[sqlite3.Row]] = {
        source: iter(
            connection.execute(
                "SELECT * FROM candidates WHERE source_dataset=? "
                + ("AND kind=? " if kind is not None else "")
                + "ORDER BY risk_score DESC, prediction_id",
                (source, kind) if kind is not None else (source,),
            ).fetchall()
        )
        for source in sources
    }
    selected: list[dict[str, Any]] = []
    active = list(sources)
    while active and len(selected) < limit:
        next_active: list[str] = []
        for source in active:
            try:
                selected.append(dict(next(queues[source])))
                next_active.append(source)
                if len(selected) == limit:
                    break
            except StopIteration:
                pass
        active = next_active
    return selected


def _balanced_select(
    candidates_db: Path, limit: int, *, require_balanced_heads: bool = False
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(candidates_db)
    connection.row_factory = sqlite3.Row
    try:
        if not require_balanced_heads:
            return _balanced_rows(connection, limit=limit)
        if limit != REQUIRED_REVIEW_CASES:
            raise ValueError(
                "two-head production audit requires the exact 100-case packet"
            )
        per_head = {
            kind: _balanced_rows(
                connection, limit=REQUIRED_REVIEW_CASES_PER_HEAD, kind=kind
            )
            for kind in ("bib_span", "toc_span")
        }
        for kind, rows in per_head.items():
            if len(rows) != REQUIRED_REVIEW_CASES_PER_HEAD:
                raise ValueError(
                    f"only {len(rows)} {kind} predictions exist; "
                    f"{REQUIRED_REVIEW_CASES_PER_HEAD} are required for two-head safety review"
                )
        # Alternate heads so packet order cannot visually bury either task.
        selected = []
        for index in range(REQUIRED_REVIEW_CASES_PER_HEAD):
            selected.append(per_head["bib_span"][index])
            selected.append(per_head["toc_span"][index])
        return selected
    finally:
        connection.close()


def _counterfactual_text(
    text: str, spans: Sequence[Mapping[str, Any]], kinds: set[str]
) -> str:
    ranges = sorted(
        (int(span["char_start"]), int(span["char_end"]))
        for span in spans
        if str(span["kind"]) in kinds
    )
    merged: list[list[int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    output = text
    for start, end in reversed(merged):
        output = output[:start] + "\n\n" + output[end:]
    return normalize_text(output)


def _loss_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("source_dataset", pa.string()),
            ("cleaning_profile", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("input_text_sha256", pa.string()),
            ("bibliography_spans", pa.int64()),
            ("toc_spans", pa.int64()),
            ("tokens_before", pa.int64()),
            ("tokens_after_bibliography", pa.int64()),
            ("tokens_after_toc", pa.int64()),
            ("tokens_after_union", pa.int64()),
            ("tokens_removed_bibliography", pa.int64()),
            ("tokens_removed_toc", pa.int64()),
            ("tokens_removed_union", pa.int64()),
            ("token_interaction", pa.int64()),
        ]
    )


_LOSS_FIELDS = (
    "documents",
    "documents_with_bibliography",
    "documents_with_toc",
    "documents_with_any",
    "tokens_before",
    "tokens_after_bibliography",
    "tokens_after_toc",
    "tokens_after_union",
    "tokens_removed_bibliography",
    "tokens_removed_toc",
    "tokens_removed_union",
    "token_interaction",
)


def _add_loss(target: collections.Counter[str], row: Mapping[str, Any]) -> None:
    target["documents"] += 1
    target["documents_with_bibliography"] += int(row["bibliography_spans"] > 0)
    target["documents_with_toc"] += int(row["toc_spans"] > 0)
    target["documents_with_any"] += int(
        row["bibliography_spans"] > 0 or row["toc_spans"] > 0
    )
    for field in _LOSS_FIELDS[4:]:
        target[field] += int(row[field])


def _loss_summary(counter: Mapping[str, int]) -> dict[str, Any]:
    result = {field: int(counter.get(field, 0)) for field in _LOSS_FIELDS}
    before = result["tokens_before"]
    for name in ("bibliography", "toc", "union"):
        result[f"{name}_loss_fraction"] = (
            result[f"tokens_removed_{name}"] / before if before else 0.0
        )
    return result


def token_loss(args: argparse.Namespace) -> int:
    """Compute exact whole-document token counterfactuals without mutation."""

    if args.report.exists() or args.per_document.exists():
        raise FileExistsError("refusing to overwrite structural token-loss outputs")
    if args.batch_rows < 1 or args.tokenizer_batch_docs < 1:
        raise ValueError("token-loss batch sizes must be positive")
    raw = validate_raw_manifest(args.raw_manifest)
    stage50_path = Path(raw["stage50_cleaning_manifest"]["path"])
    stage50, shards = _stage50_shards(stage50_path)
    tokenizer_sha = sha256_file(args.tokenizer_json)
    if tokenizer_sha != stage50.get("tokenizer_sha256"):
        raise ValueError(
            "token-loss tokenizer differs from the exact Stage50 tokenizer"
        )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    allowed_profiles = set(raw["allowed_apply_profiles"])
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    spans = _raw_span_database(raw, work / "token-loss-spans.sqlite")
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.per_document.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.per_document.with_suffix(args.per_document.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temporary, _loss_schema(), compression="zstd")
    overall: dict[str, collections.Counter[str]] = {
        "all_routed": collections.Counter(),
        "training_eligible": collections.Counter(),
    }
    by_source: dict[str, dict[str, collections.Counter[str]]] = collections.defaultdict(
        lambda: {
            "all_routed": collections.Counter(),
            "training_eligible": collections.Counter(),
        }
    )
    pending: list[tuple[str, str, str, str, bool, list[dict[str, Any]]]] = []
    documents = matched_predictions = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        variants: list[str] = []
        for _uid, _source, text, _profile, _training, doc_spans in pending:
            variants.extend(
                [
                    text,
                    _counterfactual_text(text, doc_spans, {"bib_span"}),
                    _counterfactual_text(text, doc_spans, {"toc_span"}),
                    _counterfactual_text(text, doc_spans, {"bib_span", "toc_span"}),
                ]
            )
        counts = encode_counts(tokenizer, variants)
        output_rows: list[dict[str, Any]] = []
        for index, (uid, source, text, profile, training, doc_spans) in enumerate(
            pending
        ):
            before, after_bib, after_toc, after_union = counts[
                index * 4 : index * 4 + 4
            ]
            bib_spans = sum(span["kind"] == "bib_span" for span in doc_spans)
            toc_spans = sum(span["kind"] == "toc_span" for span in doc_spans)
            row = {
                "stable_uid": uid,
                "source_dataset": source,
                "cleaning_profile": profile,
                "eligible_for_training": training,
                "input_text_sha256": sha256_text(text),
                "bibliography_spans": bib_spans,
                "toc_spans": toc_spans,
                "tokens_before": before,
                "tokens_after_bibliography": after_bib,
                "tokens_after_toc": after_toc,
                "tokens_after_union": after_union,
                "tokens_removed_bibliography": before - after_bib,
                "tokens_removed_toc": before - after_toc,
                "tokens_removed_union": before - after_union,
                "token_interaction": (before - after_bib)
                + (before - after_toc)
                - (before - after_union),
            }
            output_rows.append(row)
            _add_loss(overall["all_routed"], row)
            _add_loss(by_source[source]["all_routed"], row)
            if training:
                _add_loss(overall["training_eligible"], row)
                _add_loss(by_source[source]["training_eligible"], row)
        writer.write_table(pa.Table.from_pylist(output_rows, schema=_loss_schema()))
        pending = []

    try:
        for shard in shards:
            parquet = pq.ParquetFile(shard["path"])
            for (
                _row_number,
                uid,
                source,
                text,
                profile,
                _structural_policy,
                eligible_for_training,
            ) in _iter_routed_rows(
                parquet,
                allowed_apply_profiles=allowed_profiles,
                batch_rows=args.batch_rows,
            ):
                rows = spans.execute(
                    "SELECT payload FROM spans WHERE stable_uid=? "
                    "ORDER BY char_start, char_end, prediction_id",
                    (uid,),
                ).fetchall()
                doc_spans = [json.loads(row[0]) for row in rows]
                for span in doc_spans:
                    if span["original_sha256"] != sha256_text(text):
                        raise ValueError(f"{uid}: token-loss span/text hash mismatch")
                    _verify_python_codepoint_span(span, text)
                matched_predictions += len(doc_spans)
                documents += 1
                pending.append(
                    (uid, source, text, profile, eligible_for_training, doc_spans)
                )
                if len(pending) >= args.tokenizer_batch_docs:
                    flush()
        flush()
        writer.close()
        if documents != int(raw["counts"]["documents"]):
            raise ValueError(
                "token-loss routed document coverage differs from raw detection"
            )
        if matched_predictions != int(raw["counts"]["prediction_rows"]):
            raise ValueError("token-loss raw prediction coverage mismatch")
        os.replace(temporary, args.per_document)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        spans.close()

    full_training_tokens = int(stage50.get("counts", {}).get("tokens_final", -1))
    if full_training_tokens < 0:
        raise ValueError("Stage50 manifest lacks exact training token total")
    training = _loss_summary(overall["training_eligible"])
    report = {
        "schema_version": "phase04_structural_token_loss_v1",
        "status": "completed",
        "created_at": _utc_now(),
        "semantics": "exact_whole_document_counterfactual_no_corpus_mutation",
        "raw_manifest": file_receipt(args.raw_manifest),
        "raw_manifest_sha256": sha256_file(args.raw_manifest),
        "stage50_cleaning_manifest_sha256": raw["stage50_cleaning_manifest_sha256"],
        "tokenizer": file_receipt(args.tokenizer_json),
        "allowed_apply_profiles": sorted(allowed_profiles),
        "eligible_structural_policy": ELIGIBLE_STRUCTURAL_POLICY,
        "per_document": file_receipt(args.per_document),
        "full_stage50_training_tokens_before": full_training_tokens,
        "all_routed": _loss_summary(overall["all_routed"]),
        "training_eligible": {
            **training,
            "bibliography_loss_fraction_of_full_stage50_training": (
                training["tokens_removed_bibliography"] / full_training_tokens
                if full_training_tokens
                else 0.0
            ),
            "toc_loss_fraction_of_full_stage50_training": (
                training["tokens_removed_toc"] / full_training_tokens
                if full_training_tokens
                else 0.0
            ),
            "union_loss_fraction_of_full_stage50_training": (
                training["tokens_removed_union"] / full_training_tokens
                if full_training_tokens
                else 0.0
            ),
        },
        "per_source": {
            source: {
                scope: _loss_summary(counter)
                for scope, counter in sorted(scopes.items())
            }
            for source, scopes in sorted(by_source.items())
        },
        "conflicts": raw["conflicts"],
        "prediction_rows": matched_predictions,
    }
    write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(args.report),
                "training_tokens_removed_union": training["tokens_removed_union"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_review_packet(args: argparse.Namespace) -> int:
    if args.cases != REQUIRED_REVIEW_CASES:
        raise ValueError(
            f"production structural audit requires exactly {REQUIRED_REVIEW_CASES} cases"
        )
    if args.packet.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite structural review packet/manifest")
    raw = validate_raw_manifest(args.raw_manifest)
    stage50_path = Path(raw["stage50_cleaning_manifest"]["path"])
    _stage50, shards = _stage50_shards(stage50_path)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    span_connection = _raw_span_database(raw, work / "raw-spans.sqlite")
    candidates_path = work / "review-candidates.sqlite"
    candidates_path.unlink(missing_ok=True)
    candidates = sqlite3.connect(candidates_path)
    candidates.execute(
        """
        CREATE TABLE candidates (
          prediction_id TEXT PRIMARY KEY,
          stable_uid TEXT NOT NULL,
          source_dataset TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          kind TEXT NOT NULL,
          char_start INTEGER NOT NULL,
          char_end INTEGER NOT NULL,
          predicted_deletion_chars INTEGER NOT NULL,
          document_chars INTEGER NOT NULL,
          context_start INTEGER NOT NULL,
          context_end INTEGER NOT NULL,
          predicted_text TEXT NOT NULL,
          context_text TEXT NOT NULL,
          risk_score REAL NOT NULL,
          risk_factors_json TEXT NOT NULL,
          raw_prediction_json TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    found_predictions = 0
    try:
        import pyarrow.parquet as pq

        allowed_apply_profiles = set(raw["allowed_apply_profiles"])
        for shard in shards:
            parquet = pq.ParquetFile(shard["path"])
            for (
                _row_number,
                uid,
                source_dataset,
                text,
                _profile,
                _structural_policy,
                _eligible_for_training,
            ) in _iter_routed_rows(
                parquet,
                allowed_apply_profiles=allowed_apply_profiles,
                batch_rows=args.batch_rows,
            ):
                text_hash = sha256_text(text)
                rows = span_connection.execute(
                    "SELECT prediction_id, input_text_sha256, payload, overlaps FROM spans "
                    "WHERE stable_uid=? ORDER BY char_start, char_end, prediction_id",
                    (uid,),
                ).fetchall()
                for prediction_id, expected_hash, payload, overlaps in rows:
                    found_predictions += 1
                    if expected_hash != text_hash:
                        raise ValueError(
                            f"{uid}: raw prediction differs from exact Stage50 text"
                        )
                    span = json.loads(payload)
                    try:
                        start, end = _verify_python_codepoint_span(span, text)
                    except ValueError as exc:
                        raise ValueError(f"{uid}: {exc}") from exc
                    context_start = max(0, start - args.context_chars)
                    context_end = min(len(text), end + args.context_chars)
                    score, factors = _risk(span, text, overlap=bool(overlaps))
                    candidates.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            prediction_id,
                            uid,
                            source_dataset,
                            text_hash,
                            span["kind"],
                            start,
                            end,
                            end - start,
                            len(text),
                            context_start,
                            context_end,
                            text[start:end],
                            text[context_start:context_end],
                            score,
                            json.dumps(factors, sort_keys=True),
                            json.dumps(span, ensure_ascii=False, sort_keys=True),
                        ),
                    )
            candidates.commit()
        expected_predictions = int(raw["counts"]["prediction_rows"])
        if found_predictions != expected_predictions:
            raise ValueError(
                f"review candidate coverage {found_predictions} != raw predictions {expected_predictions}"
            )
        selected = _balanced_select(
            candidates_path, args.cases, require_balanced_heads=True
        )
        if len(selected) != args.cases:
            raise ValueError(
                f"only {len(selected)} predicted deletion cases exist; {args.cases} are required"
            )
        args.packet.parent.mkdir(parents=True, exist_ok=True)
        sources = collections.Counter()
        kinds = collections.Counter()
        sources_by_kind: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        with args.packet.open("x", encoding="utf-8") as handle:
            for rank, row in enumerate(selected, 1):
                raw_prediction = json.loads(row.pop("raw_prediction_json"))
                risk_factors = json.loads(row.pop("risk_factors_json"))
                row["schema_version"] = PACKET_ROW_SCHEMA
                row["case_id"] = canonical_json_sha256(
                    {
                        "packet_source_manifest_sha256": sha256_file(args.raw_manifest),
                        "prediction_id": row["prediction_id"],
                    }
                )
                row["selection_rank"] = rank
                row["kind"] = "bibliography" if row["kind"] == "bib_span" else "toc"
                row["risk_factors"] = risk_factors
                row["raw_model_id"] = raw_prediction["model_id"]
                row["review_context_sha256"] = canonical_json_sha256(
                    {
                        "stable_uid": row["stable_uid"],
                        "input_text_sha256": row["input_text_sha256"],
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                        "context_start": row["context_start"],
                        "context_end": row["context_end"],
                        "predicted_text": row["predicted_text"],
                        "context_text": row["context_text"],
                    }
                )
                sources[row["source_dataset"]] += 1
                kinds[row["kind"]] += 1
                sources_by_kind[row["kind"]][row["source_dataset"]] += 1
                handle.write(_json_line(row))
        packet_receipt = file_receipt(args.packet)
        payload = {
            "schema_version": PACKET_SCHEMA,
            "status": "awaiting_manual_review",
            "created_at": _utc_now(),
            "selection": {
                "algorithm": "source-balanced-round-robin-highest-risk-v1",
                "risk_semantics": "prioritisation_only_not_a_label_or_safety_metric",
                "required_cases": REQUIRED_REVIEW_CASES,
                "actual_cases": len(selected),
                "source_counts": dict(sorted(sources.items())),
                "source_counts_by_kind": {
                    kind: dict(sorted(counts.items()))
                    for kind, counts in sorted(sources_by_kind.items())
                },
                "kind_counts": dict(sorted(kinds.items())),
                "minimum_cases_per_head": REQUIRED_REVIEW_CASES_PER_HEAD,
                "automatic_adjudication": False,
            },
            "raw_manifest": file_receipt(args.raw_manifest),
            "raw_manifest_sha256": sha256_file(args.raw_manifest),
            "stage50_cleaning_manifest_sha256": raw["stage50_cleaning_manifest_sha256"],
            "character_offset_verification": {
                "rust_unit": "Unicode scalar values",
                "python_unit": "code points",
                "all_predictions_trigger_and_line_boundaries_verified": True,
            },
            "packet": packet_receipt,
        }
        write_json_atomic(args.manifest, payload)
        print(
            json.dumps(
                {"ok": True, "packet": str(args.packet), "sources": dict(sources)},
                sort_keys=True,
            )
        )
        return 0
    finally:
        span_connection.close()
        candidates.close()


def _load_packet(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load(manifest_path, schema=PACKET_SCHEMA)
    if manifest.get("status") != "awaiting_manual_review":
        raise ValueError("structural review packet has an invalid status")
    selection = manifest.get("selection")
    if (
        not isinstance(selection, dict)
        or int(selection.get("required_cases", -1)) != REQUIRED_REVIEW_CASES
        or int(selection.get("actual_cases", -1)) != REQUIRED_REVIEW_CASES
        or selection.get("automatic_adjudication") is not False
        or selection.get("kind_counts")
        != {
            "bibliography": REQUIRED_REVIEW_CASES_PER_HEAD,
            "toc": REQUIRED_REVIEW_CASES_PER_HEAD,
        }
        or int(selection.get("minimum_cases_per_head", -1))
        != REQUIRED_REVIEW_CASES_PER_HEAD
    ):
        raise ValueError(
            "structural review packet does not enforce the manual 100-case contract"
        )
    raw_path = verify_file_receipt(manifest["raw_manifest"])
    if sha256_file(raw_path) != manifest.get("raw_manifest_sha256"):
        raise ValueError("structural review packet raw-manifest binding drift")
    validate_raw_manifest(raw_path)
    packet_path = verify_file_receipt(manifest["packet"])
    rows = list(_iter_jsonl(packet_path))
    if len(rows) != REQUIRED_REVIEW_CASES:
        raise ValueError("structural review packet row count drift")
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if len(set(case_ids)) != len(case_ids) or any(
        not valid_sha256(value) for value in case_ids
    ):
        raise ValueError("structural review packet has invalid/duplicate case IDs")
    kind_counts = collections.Counter(str(row.get("kind", "")) for row in rows)
    if kind_counts != {
        "bibliography": REQUIRED_REVIEW_CASES_PER_HEAD,
        "toc": REQUIRED_REVIEW_CASES_PER_HEAD,
    }:
        raise ValueError(
            "structural review packet is not exactly balanced across both heads"
        )
    source_by_kind: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        source_by_kind[str(row["kind"])][str(row["source_dataset"])] += 1
    if {
        kind: dict(sorted(counts.items()))
        for kind, counts in sorted(source_by_kind.items())
    } != selection.get("source_counts_by_kind"):
        raise ValueError("structural review packet source/head balance receipt drift")
    return manifest, rows


def validate_audit(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit validation: {args.output}")
    packet_manifest, cases = _load_packet(args.packet_manifest)
    manual = _load(args.manual_receipt, schema=MANUAL_AUDIT_SCHEMA)
    if (
        manual.get("status") != "completed"
        or manual.get("annotation_method") != "manual"
    ):
        raise ValueError("audit receipt is not a completed manual audit")
    if manual.get("automatic_adjudication_used") is not False:
        raise ValueError(
            "automatic adjudication is forbidden for the structural safety audit"
        )
    if (
        not isinstance(manual.get("reviewer_id"), str)
        or not manual["reviewer_id"].strip()
    ):
        raise ValueError("manual structural audit receipt lacks reviewer_id")
    if manual.get("packet_manifest_sha256") != sha256_file(args.packet_manifest):
        raise ValueError(
            "manual audit receipt is not bound to the review packet manifest"
        )
    annotation_receipt = manual.get("annotations")
    if not isinstance(annotation_receipt, dict):
        raise ValueError("manual structural audit receipt lacks annotations receipt")
    annotation_path = verify_file_receipt(annotation_receipt)
    if (
        args.annotations is not None
        and annotation_path.resolve() != args.annotations.resolve()
    ):
        raise ValueError(
            "manual audit receipt annotations path differs from --annotations"
        )
    annotations = list(_iter_jsonl(annotation_path))
    if len(annotations) != REQUIRED_REVIEW_CASES or int(
        annotation_receipt.get("rows", -1)
    ) != len(annotations):
        raise ValueError(
            "manual structural audit must contain exactly 100 adjudications"
        )
    case_by_id = {row["case_id"]: row for row in cases}
    annotation_ids: set[str] = set()
    running_chars = main_text_chars = predicted_chars = 0
    catastrophic_documents: set[str] = set()
    reviewed_documents: dict[str, int] = {}
    decisions = collections.Counter()
    for row in annotations:
        case_id = str(row.get("case_id", ""))
        case = case_by_id.get(case_id)
        if case is None or case_id in annotation_ids:
            raise ValueError(f"manual audit has unknown/duplicate case_id {case_id!r}")
        annotation_ids.add(case_id)
        if row.get("review_context_sha256") != case["review_context_sha256"]:
            raise ValueError(
                f"{case_id}: manual decision is not bound to the exact review context"
            )
        decision = row.get("decision")
        if decision not in {
            "structural_only",
            "mixed_with_running_prose",
            "running_prose",
            "catastrophic_document_deletion",
        }:
            raise ValueError(
                f"{case_id}: unsupported manual audit decision {decision!r}"
            )
        running = row.get("running_prose_chars_removed")
        main = row.get("main_text_chars_removed")
        catastrophic = row.get("catastrophic_document_deletion")
        if isinstance(running, bool) or not isinstance(running, int) or running < 0:
            raise ValueError(f"{case_id}: invalid running_prose_chars_removed")
        if isinstance(main, bool) or not isinstance(main, int) or main < 0:
            raise ValueError(f"{case_id}: invalid main_text_chars_removed")
        span_chars = int(case["predicted_deletion_chars"])
        if running > span_chars or main > span_chars:
            raise ValueError(
                f"{case_id}: manual character counts exceed predicted deletion"
            )
        if not isinstance(catastrophic, bool):
            raise ValueError(f"{case_id}: catastrophic flag must be boolean")
        if decision == "structural_only" and (running or main or catastrophic):
            raise ValueError(
                f"{case_id}: structural_only contradicts false-deletion fields"
            )
        if decision in {"mixed_with_running_prose", "running_prose"} and running == 0:
            raise ValueError(
                f"{case_id}: running-prose decision requires a positive character count"
            )
        if decision == "catastrophic_document_deletion" and not catastrophic:
            raise ValueError(
                f"{case_id}: catastrophic decision requires catastrophic=true"
            )
        if (running or main or catastrophic) and not str(
            row.get("reviewer_notes", "")
        ).strip():
            raise ValueError(
                f"{case_id}: unsafe/mixed decision requires reviewer_notes"
            )
        running_chars += running
        main_text_chars += main
        predicted_chars += span_chars
        stable_uid = case["stable_uid"]
        existing_chars = reviewed_documents.setdefault(
            stable_uid, int(case["document_chars"])
        )
        if existing_chars != int(case["document_chars"]):
            raise ValueError(f"{case_id}: document character count drift")
        if catastrophic:
            catastrophic_documents.add(stable_uid)
        decisions[decision] += 1
    if set(case_by_id) != annotation_ids:
        raise ValueError(
            "manual structural audit does not cover the exact packet case set"
        )

    policy = _load(args.cleaning_policy, schema="full_cpt_cleaning_policy_v1")
    gates = policy.get("structural", {}).get("application_gates")
    if not isinstance(gates, dict):
        raise ValueError("cleaning policy lacks structural application gates")
    running_rate = running_chars / max(1, predicted_chars)
    document_chars = sum(reviewed_documents.values())
    retention = 1.0 - (main_text_chars / max(1, document_chars))
    catastrophic_rate = len(catastrophic_documents) / max(1, len(reviewed_documents))
    metrics = {
        "running_prose_deletion_rate": running_rate,
        "main_text_retention_rate": retention,
        "catastrophic_document_deletion_rate": catastrophic_rate,
    }
    passed = bool(
        len(annotations) >= int(gates["minimum_reviewed_deletions"])
        and running_rate <= float(gates["maximum_running_prose_deletion_rate"])
        and retention >= float(gates["minimum_main_text_retention_rate"])
        and catastrophic_rate
        <= float(gates["maximum_catastrophic_document_deletion_rate"])
    )
    result = {
        "schema_version": AUDIT_VALIDATION_SCHEMA,
        "status": "passed" if passed else "failed",
        "validated_at": _utc_now(),
        "evidence_status": "targeted_manual_false_deletion_audit",
        "reviewed_deletions": len(annotations),
        "reviewed_documents": len(reviewed_documents),
        "packet_manifest": file_receipt(args.packet_manifest),
        "packet_manifest_sha256": sha256_file(args.packet_manifest),
        "raw_manifest_sha256": packet_manifest["raw_manifest_sha256"],
        "manual_receipt": file_receipt(args.manual_receipt),
        "manual_receipt_sha256": sha256_file(args.manual_receipt),
        "annotations": file_receipt(annotation_path),
        "cleaning_policy": file_receipt(args.cleaning_policy),
        "cleaning_policy_sha256": sha256_file(args.cleaning_policy),
        "decisions": dict(sorted(decisions.items())),
        "character_totals": {
            "predicted_deletion_chars": predicted_chars,
            "running_prose_chars_removed": running_chars,
            "main_text_chars_removed": main_text_chars,
            "unique_reviewed_document_chars": document_chars,
        },
        "metrics": metrics,
        "policy_gates": gates,
        "metric_gate_passed": passed,
        "automatic_adjudication_used": False,
    }
    write_json_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "metrics": metrics}, sort_keys=True))
    return 0 if passed else 2


def _validate_silver_evidence(receipt_path: Path, split_path: Path) -> dict[str, Any]:
    receipt = _load(
        receipt_path, schema="academic-structure-silver-contract-receipt-v1"
    )
    if (
        receipt.get("status") != "pass"
        or receipt.get("evidence_tier") != "LLM_silver"
        or receipt.get("production_eligible") is not False
    ):
        raise ValueError("model-selection evidence is not honest, validated LLM_silver")
    if receipt.get("split_manifest_sha256") != sha256_file(split_path):
        raise ValueError("LLM-silver receipt does not bind the supplied split manifest")
    split = _load(split_path, schema="academic-structure-split-v1")
    if split.get("inventory_sha256") != receipt.get("inventory_sha256"):
        raise ValueError("LLM-silver split/evidence inventory hash mismatch")
    scopes = receipt.get("task_scope_counts")
    if (
        not isinstance(scopes, dict)
        or int(scopes.get("bibliography_toc_windows", 0)) <= 0
    ):
        raise ValueError(
            "joint bibliography+ToC LLM-silver evidence is absent from the supplied receipt; "
            "supply the passed locked STRUCT_2K import receipt before constructing a two-head "
            "production receipt"
        )
    if not valid_sha256(receipt.get("inventory_sha256")):
        raise ValueError("LLM-silver evidence inventory hash is invalid")
    return receipt


def _validate_audit_validation(
    path: Path, *, raw_manifest_sha256: str
) -> dict[str, Any]:
    value = _load(path, schema=AUDIT_VALIDATION_SCHEMA)
    if (
        value.get("status") != "passed"
        or value.get("evidence_status") != "targeted_manual_false_deletion_audit"
        or value.get("metric_gate_passed") is not True
        or int(value.get("reviewed_deletions", -1)) != REQUIRED_REVIEW_CASES
        or value.get("automatic_adjudication_used") is not False
    ):
        raise ValueError("targeted manual structural audit validation did not pass")
    if value.get("raw_manifest_sha256") != raw_manifest_sha256:
        raise ValueError(
            "targeted manual audit is bound to a different raw detector run"
        )
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or any(
        isinstance(metrics.get(name), bool)
        or not isinstance(metrics.get(name), (int, float))
        for name in (
            "running_prose_deletion_rate",
            "main_text_retention_rate",
            "catastrophic_document_deletion_rate",
        )
    ):
        raise ValueError("targeted manual audit metrics are missing")
    verify_file_receipt(value["manual_receipt"])
    verify_file_receipt(value["annotations"])
    policy_path = verify_file_receipt(value["cleaning_policy"])
    if sha256_file(policy_path) != value.get("cleaning_policy_sha256"):
        raise ValueError("targeted manual audit cleaning-policy hash drift")
    return value


def build_model_receipt(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite model receipt: {args.output}")
    raw = validate_raw_manifest(args.raw_manifest)
    if int(raw.get("conflicts", {}).get("overlap_pairs", -1)) != 0:
        raise ValueError(
            "raw ToC/bibliography predictions overlap; strict production rebind is blocked"
        )
    raw_sha = sha256_file(args.raw_manifest)
    audit = _validate_audit_validation(
        args.audit_validation, raw_manifest_sha256=raw_sha
    )
    silver = _validate_silver_evidence(args.silver_receipt, args.silver_split_manifest)
    classifier_selection = validate_classifier_selection(
        args.classifier_selection_receipt,
        source_receipt=args.silver_receipt,
        source_split_manifest=args.silver_split_manifest,
        sequence_config=_artifact_path(
            raw["artifacts"], category="config", name="sequence_config"
        ),
        bib_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="bibliography_line_model"
        ),
        toc_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="toc_line_model"
        ),
        smoother=_artifact_path(
            raw["artifacts"], category="config", name="structural_smoother"
        ),
    )
    audit_policy = _load(
        verify_file_receipt(audit["cleaning_policy"]),
        schema="full_cpt_cleaning_policy_v1",
    )
    detector_binary = verify_file_receipt(raw["detector"]["binary"])
    _validate_parity(
        args.parity_receipt,
        detector_binary=detector_binary,
        bib_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="bibliography_line_model"
        ),
        toc_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="toc_line_model"
        ),
        smoother=_artifact_path(
            raw["artifacts"], category="config", name="structural_smoother"
        ),
        silver_receipt=silver,
        silver_receipt_sha256=sha256_file(args.silver_receipt),
        cleaning_policy=audit_policy,
    )
    embedded_parity = raw["detector"].get("parity_receipt")
    if isinstance(embedded_parity, dict) and embedded_parity.get(
        "sha256"
    ) != sha256_file(args.parity_receipt):
        raise ValueError(
            "promotion parity receipt differs from the one bound at detection time"
        )
    models = raw["models"]
    model_id = (
        f"bib={models['bib_model_id']}:{models['bib_decoder_id']};"
        f"toc={models['toc_model_id']}:{models['toc_decoder_id']}"
    )
    metrics = {key: float(value) for key, value in audit["metrics"].items()}
    receipt = {
        "schema_version": MODEL_RECEIPT_SCHEMA,
        "status": "passed",
        "promotion_status": "passed",
        "created_at": _utc_now(),
        "model_id": model_id,
        "stage50_cleaning_manifest_sha256": raw["stage50_cleaning_manifest_sha256"],
        "raw_detector_run": file_receipt(args.raw_manifest),
        "raw_detector_run_sha256": raw_sha,
        "artifacts": raw["artifacts"],
        "evidence": {
            "annotation_status": "LLM_silver",
            "inventory_sha256": silver["inventory_sha256"],
            "task_coverage": ["toc", "bibliography"],
            "selected_architecture": classifier_selection["selected_architecture"],
            "classifier_selection_receipt": file_receipt(
                args.classifier_selection_receipt
            ),
            "joint_ladder_run_receipt_sha256": classifier_selection["joint_ladder"][
                "run_receipt_sha256"
            ],
            "silver_receipt": file_receipt(args.silver_receipt),
            "silver_split_manifest": file_receipt(args.silver_split_manifest),
            "runtime_parity_receipt": file_receipt(args.parity_receipt),
            "work_split": {
                "leak_free": True,
                "work_overlap_count": 0,
                "exact_text_overlap_count": 0,
                "split_manifest_sha256": sha256_file(args.silver_split_manifest),
                "validation_contract": "academic-structure-silver-contract-receipt-v1",
            },
        },
        "safety": {
            "status": "passed",
            "evidence_status": "targeted_manual_false_deletion_audit",
            "reviewed_deletions": REQUIRED_REVIEW_CASES,
            "audit_receipt_sha256": sha256_file(args.audit_validation),
            "audit_validation": file_receipt(args.audit_validation),
            "manual_audit_receipt_sha256": audit["manual_receipt_sha256"],
            "cleaning_policy_sha256": audit["cleaning_policy_sha256"],
            "metrics": metrics,
        },
    }
    write_json_atomic(args.output, receipt)
    validate_model_receipt(args.output, raw_manifest=args.raw_manifest)
    print(
        json.dumps(
            {"ok": True, "receipt": str(args.output), "model_id": model_id},
            sort_keys=True,
        )
    )
    return 0


def validate_model_receipt(
    path: Path, *, raw_manifest: Path | None = None
) -> dict[str, Any]:
    receipt = _load(path, schema=MODEL_RECEIPT_SCHEMA)
    if receipt.get("status") != "passed" or receipt.get("promotion_status") != "passed":
        raise ValueError("structural model receipt is not passed")
    raw_path = verify_file_receipt(receipt["raw_detector_run"])
    if raw_manifest is not None and raw_path.resolve() != raw_manifest.resolve():
        raise ValueError(
            "structural model receipt references a different raw detector manifest"
        )
    raw = validate_raw_manifest(raw_path)
    if sha256_file(raw_path) != receipt.get("raw_detector_run_sha256"):
        raise ValueError("structural model receipt raw detector hash drift")
    if (
        receipt.get("stage50_cleaning_manifest_sha256")
        != raw["stage50_cleaning_manifest_sha256"]
    ):
        raise ValueError("structural model receipt Stage50 binding drift")
    if receipt.get("artifacts") != raw["artifacts"]:
        raise ValueError("structural model receipt artifact binding drift")
    _validate_artifacts(receipt["artifacts"])
    evidence = receipt.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("annotation_status") != "LLM_silver"
        or evidence.get("selected_architecture") != "c0-rust-lr-hysteresis"
        or not isinstance(evidence.get("classifier_selection_receipt"), dict)
        or not valid_sha256(evidence.get("joint_ladder_run_receipt_sha256"))
    ):
        raise ValueError(
            "structural model receipt lacks honest joint-ladder C0 selection evidence"
        )
    if set(evidence.get("task_coverage", [])) != {"toc", "bibliography"}:
        raise ValueError("structural model receipt lacks two-head task coverage")
    silver_path = verify_file_receipt(evidence["silver_receipt"])
    split_path = verify_file_receipt(evidence["silver_split_manifest"])
    silver = _validate_silver_evidence(silver_path, split_path)
    if silver["inventory_sha256"] != evidence.get("inventory_sha256"):
        raise ValueError("structural model receipt silver inventory drift")
    safety = receipt.get("safety")
    if (
        not isinstance(safety, dict)
        or safety.get("evidence_status") != "targeted_manual_false_deletion_audit"
    ):
        raise ValueError(
            "structural model receipt lacks targeted manual safety evidence"
        )
    validation_path = verify_file_receipt(safety["audit_validation"])
    if sha256_file(validation_path) != safety.get("audit_receipt_sha256"):
        raise ValueError("structural model receipt audit validation hash drift")
    validation = _validate_audit_validation(
        validation_path, raw_manifest_sha256=sha256_file(raw_path)
    )
    classifier_selection_path = verify_file_receipt(
        evidence["classifier_selection_receipt"]
    )
    classifier_selection = validate_classifier_selection(
        classifier_selection_path,
        source_receipt=silver_path,
        source_split_manifest=split_path,
        sequence_config=_artifact_path(
            raw["artifacts"], category="config", name="sequence_config"
        ),
        bib_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="bibliography_line_model"
        ),
        toc_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="toc_line_model"
        ),
        smoother=_artifact_path(
            raw["artifacts"], category="config", name="structural_smoother"
        ),
    )
    if (
        evidence.get("selected_architecture")
        != classifier_selection["selected_architecture"]
        or evidence.get("joint_ladder_run_receipt_sha256")
        != classifier_selection["joint_ladder"]["run_receipt_sha256"]
    ):
        raise ValueError("structural model receipt classifier-selection binding drift")
    audit_policy = _load(
        verify_file_receipt(validation["cleaning_policy"]),
        schema="full_cpt_cleaning_policy_v1",
    )
    parity_path = verify_file_receipt(evidence["runtime_parity_receipt"])
    _validate_parity(
        parity_path,
        detector_binary=verify_file_receipt(raw["detector"]["binary"]),
        bib_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="bibliography_line_model"
        ),
        toc_model=_artifact_path(
            raw["artifacts"], category="checkpoint", name="toc_line_model"
        ),
        smoother=_artifact_path(
            raw["artifacts"], category="config", name="structural_smoother"
        ),
        silver_receipt=silver,
        silver_receipt_sha256=sha256_file(silver_path),
        cleaning_policy=audit_policy,
    )
    split = evidence.get("work_split")
    if (
        not isinstance(split, dict)
        or split.get("leak_free") is not True
        or int(split.get("work_overlap_count", -1)) != 0
        or int(split.get("exact_text_overlap_count", -1)) != 0
        or split.get("split_manifest_sha256") != sha256_file(split_path)
    ):
        raise ValueError("structural model receipt work split is not leak-free/bound")
    if validation["metrics"] != safety.get("metrics"):
        raise ValueError("structural model receipt safety metrics drift")
    return receipt


def validate_model_receipt_command(args: argparse.Namespace) -> int:
    receipt = validate_model_receipt(args.receipt, raw_manifest=args.raw_manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "receipt": str(args.receipt.resolve()),
                "receipt_sha256": sha256_file(args.receipt),
                "model_id": receipt["model_id"],
                "evidence": "LLM_silver_plus_targeted_manual_false_deletion_audit",
            },
            sort_keys=True,
        )
    )
    return 0


def rebind(args: argparse.Namespace) -> int:
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite final structural spans/manifest")
    raw = validate_raw_manifest(args.raw_manifest)
    validate_model_receipt(args.model_receipt, raw_manifest=args.raw_manifest)
    if int(raw.get("conflicts", {}).get("overlap_pairs", -1)) != 0:
        raise ValueError(
            "raw structural predictions overlap; strict span conversion is blocked"
        )
    model_sha = sha256_file(args.model_receipt)
    stage50_sha = raw["stage50_cleaning_manifest_sha256"]
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    database = args.work_dir / "rebind-spans.sqlite"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE spans (
          stable_uid TEXT NOT NULL,
          input_text_sha256 TEXT NOT NULL,
          char_start INTEGER NOT NULL,
          char_end INTEGER NOT NULL,
          kind TEXT NOT NULL,
          rule_id TEXT NOT NULL,
          raw_prediction_id TEXT NOT NULL UNIQUE,
          PRIMARY KEY (stable_uid, char_start, char_end, kind, rule_id)
        ) WITHOUT ROWID
        """
    )
    rows = 0
    root = Path(raw["output_root"])
    try:
        for shard in raw["files"]:
            for span in _iter_jsonl(_resolve_receipt(shard["spans"], root)):
                rows += 1
                kind = "bibliography" if span["kind"] == "bib_span" else "toc"
                record = (
                    span["doc_id"],
                    span["original_sha256"],
                    int(span["char_start"]),
                    int(span["char_end"]),
                    kind,
                    span["model_id"],
                    _prediction_id(span),
                )
                try:
                    connection.execute(
                        "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?)", record
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "duplicate raw prediction cannot be rebound"
                    ) from exc
        connection.commit()
        previous_uid = ""
        previous_hash = ""
        previous_end = -1
        with temporary.open("x", encoding="utf-8") as handle:
            for (
                uid,
                text_hash,
                start,
                end,
                kind,
                rule_id,
                prediction_id,
            ) in connection.execute(
                "SELECT stable_uid, input_text_sha256, char_start, char_end, kind, rule_id, "
                "raw_prediction_id FROM spans ORDER BY stable_uid, char_start, char_end, kind, rule_id"
            ):
                if uid != previous_uid:
                    previous_uid, previous_hash, previous_end = uid, text_hash, -1
                elif text_hash != previous_hash:
                    raise ValueError(
                        f"{uid}: raw predictions disagree on exact input text"
                    )
                if int(start) < previous_end:
                    raise ValueError(
                        f"{uid}: raw predictions overlap; strict final schema requires disjoint spans"
                    )
                previous_end = int(end)
                output = {
                    "stable_uid": uid,
                    "input_text_sha256": text_hash,
                    "stage50_cleaning_manifest_sha256": stage50_sha,
                    "model_receipt_sha256": model_sha,
                    "kind": kind,
                    "char_start": int(start),
                    "char_end": int(end),
                    "rule_id": rule_id,
                    "raw_prediction_id": prediction_id,
                    "raw_detector_run_sha256": sha256_file(args.raw_manifest),
                }
                # Inline validation mirrors structural_span.schema.json and is
                # intentionally stricter about the final canonical kind names.
                if (
                    not valid_sha256(output["stable_uid"])
                    or not valid_sha256(output["input_text_sha256"])
                    or not valid_sha256(output["stage50_cleaning_manifest_sha256"])
                    or not valid_sha256(output["model_receipt_sha256"])
                    or output["kind"] not in {"toc", "bibliography"}
                    or output["char_start"] < 0
                    or output["char_end"] <= output["char_start"]
                    or not output["rule_id"]
                ):
                    raise ValueError("strict final structural-span validation failed")
                handle.write(_json_line(output))
        if rows != int(raw["counts"]["prediction_rows"]):
            raise ValueError(
                "final structural span coverage differs from raw predictions"
            )
        os.replace(temporary, args.output)
        payload = {
            "schema_version": SPAN_MANIFEST_SCHEMA,
            "status": "completed",
            "created_at": _utc_now(),
            "raw_manifest": file_receipt(args.raw_manifest),
            "raw_manifest_sha256": sha256_file(args.raw_manifest),
            "model_receipt": file_receipt(args.model_receipt),
            "model_receipt_sha256": model_sha,
            "stage50_cleaning_manifest_sha256": stage50_sha,
            "spans": {**file_receipt(args.output), "rows": rows},
            "receipt_cycle_avoided": True,
            "binding_order": [
                "raw_predictions",
                "manual_audit",
                "model_receipt",
                "final_spans",
            ],
        }
        write_json_atomic(args.manifest, payload)
        print(
            json.dumps(
                {"ok": True, "spans": rows, "manifest": str(args.manifest)},
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def _add_detection_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage50-cleaning-manifest", type=Path, required=True)
    parser.add_argument("--cleaning-policy", type=Path, required=True)
    parser.add_argument("--detector-binary", type=Path, required=True)
    parser.add_argument("--detector-build-receipt", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--model-code", action="append", type=Path, required=True)
    parser.add_argument("--sequence-config", type=Path, required=True)
    parser.add_argument("--smoother", type=Path, required=True)
    parser.add_argument("--bib-model", type=Path, required=True)
    parser.add_argument("--toc-model", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    detector = commands.add_parser("detect")
    _add_detection_inputs(detector)
    detector.add_argument("--code-commit", required=True)
    detector.add_argument("--output-dir", type=Path, required=True)
    detector.add_argument("--manifest", type=Path, required=True)
    detector.add_argument("--batch-rows", type=int, default=2048)
    detector.add_argument(
        "--rayon-threads", type=int, default=max(1, os.cpu_count() or 1)
    )
    detector.set_defaults(func=detect)

    packet = commands.add_parser("build-review-packet")
    packet.add_argument("--raw-manifest", type=Path, required=True)
    packet.add_argument("--packet", type=Path, required=True)
    packet.add_argument("--manifest", type=Path, required=True)
    packet.add_argument("--work-dir", type=Path, required=True)
    packet.add_argument("--cases", type=int, default=REQUIRED_REVIEW_CASES)
    packet.add_argument("--context-chars", type=int, default=600)
    packet.add_argument("--batch-rows", type=int, default=2048)
    packet.set_defaults(func=build_review_packet)

    loss = commands.add_parser("token-loss")
    loss.add_argument("--raw-manifest", type=Path, required=True)
    loss.add_argument("--tokenizer-json", type=Path, required=True)
    loss.add_argument("--per-document", type=Path, required=True)
    loss.add_argument("--report", type=Path, required=True)
    loss.add_argument("--work-dir", type=Path, required=True)
    loss.add_argument("--batch-rows", type=int, default=2048)
    loss.add_argument("--tokenizer-batch-docs", type=int, default=64)
    loss.set_defaults(func=token_loss)

    audit = commands.add_parser("validate-audit")
    audit.add_argument("--packet-manifest", type=Path, required=True)
    audit.add_argument("--manual-receipt", type=Path, required=True)
    audit.add_argument("--annotations", type=Path)
    audit.add_argument("--cleaning-policy", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(func=validate_audit)

    model = commands.add_parser("build-model-receipt")
    model.add_argument("--raw-manifest", type=Path, required=True)
    model.add_argument("--audit-validation", type=Path, required=True)
    model.add_argument("--silver-receipt", type=Path, required=True)
    model.add_argument("--silver-split-manifest", type=Path, required=True)
    model.add_argument("--classifier-selection-receipt", type=Path, required=True)
    model.add_argument("--parity-receipt", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.set_defaults(func=build_model_receipt)

    validate_model = commands.add_parser("validate-model-receipt")
    validate_model.add_argument("--receipt", type=Path, required=True)
    validate_model.add_argument("--raw-manifest", type=Path, required=True)
    validate_model.set_defaults(func=validate_model_receipt_command)

    final = commands.add_parser("rebind")
    final.add_argument("--raw-manifest", type=Path, required=True)
    final.add_argument("--model-receipt", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--work-dir", type=Path, required=True)
    final.set_defaults(func=rebind)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
