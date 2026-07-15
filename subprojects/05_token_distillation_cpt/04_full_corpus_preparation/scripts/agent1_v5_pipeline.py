#!/usr/bin/env python3
"""Receipt-bound Eiger CPU pipeline for the Agent-1 v5 candidate corpus.

The command is intentionally stage oriented.  Expensive work is performed by
independent Slurm array elements, each of which publishes an immutable Parquet
shard plus a checksum-bound receipt.  Merge commands validate complete row and
task closure before a dependent stage may start.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import read_json_object, sha256_json, sha256_text  # noqa: E402
from full_corpus_io import jsonable, sha256_file  # noqa: E402
import prototype_agent1_v4_gfm_normalization as gfm  # noqa: E402


CONFIG_SCHEMA = "agent1_v5_eiger_pipeline_config_v1"
CONTRACT_SCHEMA = "agent1_v5_run_contract_v1"
TASK_MANIFEST_SCHEMA = "agent1_v5_transform_task_manifest_v1"
TRANSFORM_RECEIPT_SCHEMA = "agent1_v5_transform_task_receipt_v1"
TRANSFORM_MANIFEST_SCHEMA = "agent1_v5_transform_manifest_v1"
GLOSSAPI_RECEIPT_SCHEMA = "agent1_v5_glossapi_task_receipt_v1"
GLOSSAPI_MANIFEST_SCHEMA = "agent1_v5_glossapi_manifest_v1"
ENVELOPE_PLAN_SCHEMA = "agent1_v5_envelope_plan_v1"
ENVELOPE_RECEIPT_SCHEMA = "agent1_v5_envelope_task_receipt_v1"
BASE_PLAN_SCHEMA = "agent1_v5_base_plan_v1"
BASE_RECEIPT_SCHEMA = "agent1_v5_base_cast_receipt_v1"
COMBINED_MANIFEST_SCHEMA = "agent1_v5_combined_manifest_v1"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,95}$")
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
LATEX_RE = re.compile(
    r"(?:\\(?:begin|end)\{[^}]+\}|\\(?:frac|sqrt|sum|int|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|phi|omega)\b|\$\$|(?<!\\)\$(?!\s))",
    re.IGNORECASE,
)
MATH_RE = re.compile(r"[∑∫√∞≈≠≤≥±×÷∂∇∈∉∩∪⊂⊆⊕⊗≃≅∀∃]")
RECOGNIZED_HTML_RE = gfm.KNOWN_HTML_TAG_RE
GENERATED_IMAGE_RE = gfm.GENERATED_IMAGE_TOKEN_RE


ENVELOPE_FIELDS = (
    "source_dataset",
    "source_doc_id",
    "text",
    "title",
    "author",
    "source_metadata_json",
)
QUALITY_FIELDS = (
    "is_historical_or_polytonic",
    "contains_math",
    "contains_latex",
    "greek_percentage",
    "latin_percentage",
    "polytonic_ratio",
    "table_ratio",
    "greek_badness_score",
    "len_greek",
    "mojibake_badness_score",
    "needs_ocr",
    "is_empty",
    "filter",
    "ocr_success",
    "quality_method",
    "reevaluated_at",
)
CLEANER_FIELDS = (
    "content_chars_kept",
    "chars_dropped_by_line_drop",
    "chars_dropped_by_normalization",
    "chars_dropped_by_per_char_filter",
    "lines_dropped_by_cleaner",
    "marker_chars_passthrough",
    "marker_chars_added",
    "charset_greek_ratio",
    "charset_moji_ratio",
    "charset_punct_ratio",
    "mojibake_noise_ratio",
    "rule_a_match_count",
    "rule_b_match_count",
    "residue_line_drop_count",
    "phase_a_fallback_reason",
    "phase_a_dialect_ambiguous_input",
    "cleaner_chars_before",
    "cleaner_chars_after",
)
OPTIONAL_TEXT_STAT_FIELDS = (
    "chars",
    "non_whitespace_chars",
    "utf8_bytes",
    "approx_word_count",
)
CANONICAL_FIELD_NAMES = ENVELOPE_FIELDS + QUALITY_FIELDS + CLEANER_FIELDS + OPTIONAL_TEXT_STAT_FIELDS


def canonical_schema():
    import pyarrow as pa

    large_string = {
        "source_dataset",
        "source_doc_id",
        "text",
        "title",
        "author",
        "source_metadata_json",
        "filter",
        "quality_method",
    }
    float64 = {
        "greek_percentage",
        "latin_percentage",
        "polytonic_ratio",
        "table_ratio",
        "greek_badness_score",
        "mojibake_badness_score",
        "charset_greek_ratio",
        "charset_moji_ratio",
        "charset_punct_ratio",
        "mojibake_noise_ratio",
    }
    bools = {
        "is_historical_or_polytonic",
        "contains_math",
        "contains_latex",
        "needs_ocr",
        "is_empty",
        "ocr_success",
        "phase_a_dialect_ambiguous_input",
    }
    uint64 = {
        "content_chars_kept",
        "chars_dropped_by_line_drop",
        "chars_dropped_by_normalization",
        "chars_dropped_by_per_char_filter",
        "lines_dropped_by_cleaner",
        "marker_chars_passthrough",
        "marker_chars_added",
        "rule_a_match_count",
        "rule_b_match_count",
        "residue_line_drop_count",
        "cleaner_chars_before",
        "cleaner_chars_after",
    }
    int64 = {"len_greek", *OPTIONAL_TEXT_STAT_FIELDS}
    fields = []
    for name in CANONICAL_FIELD_NAMES:
        if name in large_string or name == "phase_a_fallback_reason":
            arrow_type = pa.large_string()
        elif name in float64:
            arrow_type = pa.float64()
        elif name in bools:
            arrow_type = pa.bool_()
        elif name in uint64:
            arrow_type = pa.uint64()
        elif name in int64:
            arrow_type = pa.int64()
        elif name == "reevaluated_at":
            arrow_type = pa.timestamp("us", tz="UTC")
        else:  # pragma: no cover - schema closure guard
            raise AssertionError(name)
        fields.append(pa.field(name, arrow_type, nullable=True))
    return pa.schema(fields)


def transform_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("source_id", pa.string()),
            ("source_repo_id", pa.string()),
            ("source_revision", pa.string()),
            ("source_dataset", pa.large_string()),
            ("source_doc_id_candidate", pa.large_string()),
            ("source_doc_id_was_synthetic", pa.bool_()),
            ("source_artifact_path", pa.large_string()),
            ("source_row_index", pa.int64()),
            ("source_row_uid", pa.string()),
            ("transformed_text", pa.large_string()),
            ("title", pa.large_string()),
            ("author", pa.large_string()),
            ("source_metadata_json", pa.large_string()),
            ("original_text_sha256", pa.string()),
            ("transformed_text_sha256", pa.string()),
            ("transform_metrics_json", pa.large_string()),
        ]
    )


def glossapi_schema():
    import pyarrow as pa

    prefix = list(transform_schema())
    canonical_tail = [field for field in canonical_schema() if field.name not in ENVELOPE_FIELDS]
    return pa.schema(prefix + [pa.field("text", pa.large_string())] + canonical_tail)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_atomic(path: Path, value: Mapping[str, object], *, replace: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and not replace:
        raise FileExistsError(f"immutable output exists: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_parquet_atomic(path: Path, schema: Any, batches: Iterable[Sequence[Mapping[str, Any]]]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True, write_statistics=True)
    rows = 0
    try:
        for batch in batches:
            if not batch:
                continue
            table = pa.Table.from_pylist(list(batch), schema=schema)
            writer.write_table(table)
            rows += table.num_rows
        writer.close()
        temporary.replace(path)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return rows


def file_receipt(path: Path, *, root: Path | None = None, rows: int | None = None) -> dict[str, object]:
    path = path.resolve()
    display = path
    if root is not None:
        display = path.relative_to(root.resolve())
    result: dict[str, object] = {
        "path": display.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def validate_file_receipt(binding: Mapping[str, Any], *, root: Path | None = None) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path.is_absolute() and root is not None:
        path = root / path
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(binding.get("bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise ValueError(f"file receipt mismatch: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"{path}: unsupported config")
    sources = value.get("sources")
    if not isinstance(sources, Mapping) or len(sources) != 18:
        raise ValueError("v5 config must contain exactly 18 sources")
    if value.get("release", {}).get("private_only") is not True:
        raise ValueError("v5 publisher must remain private-only")
    dedup = value.get("dedup", {})
    if int(dedup.get("num_buckets", 0)) * int(dedup.get("hashes_per_bucket", 0)) != 128:
        raise ValueError("MinHash configuration must contain exactly 128 permutations")
    return value


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def contract(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    if value.get("schema_version") != CONTRACT_SCHEMA or value.get("status") != "passed":
        raise ValueError(f"{path}: invalid run contract")
    return value


def validate_acquired_file_identity(row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("local_path", ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"acquired file is missing or is a symlink: {path}")
    stat_result = path.stat()
    expected = {
        "size": stat_result.st_size,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }
    for name, observed in expected.items():
        if int(row.get(name, -1)) != int(observed):
            raise ValueError(f"acquired file identity drift ({name}): {path}")
    digest = row.get("expected_hash")
    if row.get("hash_kind") not in {"lfs_sha256", "sha256"} or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError(f"acquired file lacks a supported SHA-256 binding: {path}")
    return path


def freeze_contract(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    acquisition = read_json_object(args.acquisition_receipt)
    if acquisition.get("schema_version") != "full_cpt_acquisition_receipt_v1" or acquisition.get("status") != "passed":
        raise ValueError("acquisition receipt is not passed")
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError("unsafe run id")
    code_commit = git_output(Path(__file__).resolve().parents[1], "rev-parse", "HEAD")
    if git_output(Path(__file__).resolve().parents[1], "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("pipeline checkout must be clean before freezing a production contract")
    glossapi_root = args.glossapi_root.resolve()
    expected_glossapi = str(config["pins"]["glossapi_commit"])
    if git_output(glossapi_root, "rev-parse", "HEAD") != expected_glossapi:
        raise ValueError("GlossAPI checkout is not at the pinned commit")
    if git_output(glossapi_root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("GlossAPI checkout must be clean")
    acquired = {str(row.get("source_id")): row for row in acquisition.get("sources", []) if isinstance(row, Mapping)}
    for source_id, source in config["sources"].items():
        row = acquired.get(source_id)
        if row is None or row.get("repo_id") != source["repo_id"] or row.get("revision") != source["revision"]:
            raise ValueError(f"{source_id}: config/acquisition identity drift")
        for file_row in row.get("files", []):
            validate_acquired_file_identity(file_row)
    base = acquired.get("nanochat_base")
    if (
        base is None
        or base.get("repo_id") != config["pins"]["nanochat_repo_id"]
        or base.get("revision") != config["pins"]["nanochat_revision"]
    ):
        raise ValueError("Nanochat base identity drift")
    for file_row in base.get("files", []):
        validate_acquired_file_identity(file_row)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    created_at = utc_now()
    pre_repo = f"{config['release']['owner']}/{config['release']['pre_dedup_name_prefix']}-{args.run_id}"
    dedup_repo = f"{config['release']['owner']}/{config['release']['dedup_name_prefix']}-{args.run_id}"
    result: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "passed",
        "run_id": args.run_id,
        "created_at": created_at,
        "run_root": str(run_root),
        "code_commit": code_commit,
        "config": file_receipt(args.config),
        "acquisition_receipt": file_receipt(args.acquisition_receipt),
        "glossapi": {"root": str(glossapi_root), "commit": expected_glossapi},
        "pins": config["pins"],
        "source_ids": list(config["sources"]),
        "license_override": config["release"]["license_override"],
        "private_repositories": {"pre_dedup": pre_repo, "deduplicated": dedup_repo},
        "canonical_columns": list(CANONICAL_FIELD_NAMES),
        "canonical_schema": str(canonical_schema()),
        "dedup": config["dedup"],
    }
    write_json_atomic(run_root / "run_contract.json", result)
    write_json_atomic(
        run_root / "license_override_receipt.json",
        {
            "schema_version": "agent1_v5_private_license_override_v1",
            "status": "passed",
            "run_id": args.run_id,
            "created_at": created_at,
            "source_ids": list(config["sources"]),
            **config["release"]["license_override"],
            "private_repositories": result["private_repositories"],
        },
    )
    print(canonical_json({"ok": True, "run_root": str(run_root), "code_commit": code_commit}))
    return 0


def plan_transform_tasks(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    cfg = load_config(args.config)
    run = contract(args.contract)
    acquisition_path = validate_file_receipt(run["acquisition_receipt"])
    acquisition = read_json_object(acquisition_path)
    acquired = {str(row.get("source_id")): row for row in acquisition.get("sources", []) if isinstance(row, Mapping)}
    target = int(cfg["execution"]["target_task_bytes"])
    tasks: list[dict[str, object]] = []
    for source_id in cfg["sources"]:
        source = acquired[source_id]
        for file_row in sorted(source.get("files", []), key=lambda row: str(row.get("path", ""))):
            path = Path(str(file_row["local_path"])).resolve()
            if path.suffix.casefold() != ".parquet":
                continue
            parquet = pq.ParquetFile(path)
            row_start = 0
            group: list[int] = []
            group_bytes = 0
            group_rows = 0
            group_row_start = 0
            for rg in range(parquet.num_row_groups):
                metadata = parquet.metadata.row_group(rg)
                size = int(metadata.total_byte_size)
                rows = int(metadata.num_rows)
                if group and group_bytes + size > target:
                    tasks.append(
                        {
                            "task_index": len(tasks),
                            "source_id": source_id,
                            "repo_id": source["repo_id"],
                            "revision": source["revision"],
                            "artifact_path": str(file_row["path"]),
                            "input_path": str(path),
                            "input_expected_hash": file_row.get("expected_hash"),
                            "input_hash_kind": file_row.get("hash_kind"),
                            "row_groups": group,
                            "row_start": group_row_start,
                            "rows": group_rows,
                            "uncompressed_bytes": group_bytes,
                        }
                    )
                    group = []
                    group_bytes = 0
                    group_rows = 0
                    group_row_start = row_start
                if not group:
                    group_row_start = row_start
                group.append(rg)
                group_bytes += size
                group_rows += rows
                row_start += rows
            if group:
                tasks.append(
                    {
                        "task_index": len(tasks),
                        "source_id": source_id,
                        "repo_id": source["repo_id"],
                        "revision": source["revision"],
                        "artifact_path": str(file_row["path"]),
                        "input_path": str(path),
                        "input_expected_hash": file_row.get("expected_hash"),
                        "input_hash_kind": file_row.get("hash_kind"),
                        "row_groups": group,
                        "row_start": group_row_start,
                        "rows": group_rows,
                        "uncompressed_bytes": group_bytes,
                    }
                )
    if not tasks:
        raise ValueError("no candidate Parquet tasks were planned")
    result: dict[str, object] = {
        "schema_version": TASK_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "config_sha256": sha256_file(args.config),
        "task_count": len(tasks),
        "input_rows": sum(int(row["rows"]) for row in tasks),
        "tasks": tasks,
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "tasks": len(tasks), "rows": result["input_rows"]}))
    return 0


def _parse_json_container(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def read_path(row: Mapping[str, Any], path: str | None) -> Any:
    if path is None:
        return None
    value: Any = row
    for part in path.split("."):
        value = _parse_json_container(value)
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def optional_text(value: Any, *, author: bool = False) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [optional_text(item, author=author) for item in value]
        values = [item for item in values if item]
        return ("; " if author else " - ").join(values) or None
    value = str(value).strip()
    return value or None


def _remove_metadata_path(payload: dict[str, Any], path: str | None) -> None:
    if path is None:
        return
    parts = path.split(".")
    current: Any = payload
    for index, part in enumerate(parts):
        if not isinstance(current, dict) or part not in current:
            return
        if index == len(parts) - 1:
            current.pop(part, None)
            return
        parsed = _parse_json_container(current[part])
        if parsed is not current[part]:
            current[part] = jsonable(parsed)
        current = current[part]


def metadata_json(row: Mapping[str, Any], mapping: Mapping[str, Any]) -> str | None:
    payload = {str(key): jsonable(value) for key, value in row.items() if value is not None}
    for name in ("text_path", "title_path", "author_path"):
        _remove_metadata_path(payload, mapping.get(name))
    return canonical_json(payload) if payload else None


def source_doc_id(
    row: Mapping[str, Any], mapping: Mapping[str, Any], *, source_id: str,
    revision: str, artifact_path: str, row_index: int, text_sha256: str,
) -> tuple[str, bool]:
    contract_value = mapping["source_doc_id"]
    paths = list(contract_value["paths"])
    values = [optional_text(read_path(row, path)) for path in paths]
    values = [value for value in values if value]
    mode = str(contract_value["mode"])
    if mode == "first_nonblank" and values:
        return values[0], False
    if mode == "join_nonblank" and len(values) == len(paths):
        return "|".join(values), False
    if mode not in {"first_nonblank", "join_nonblank"}:
        raise ValueError(f"{source_id}: unsupported source_doc_id mode")
    synthetic = sha256_json(
        {
            "namespace": "agent1_v5_synthetic_source_doc_id_v1",
            "source_id": source_id,
            "revision": revision,
            "artifact_path": artifact_path,
            "row_index": row_index,
            "text_sha256": text_sha256,
        }
    )
    return f"synthetic:{synthetic}", True


def compact_transform_metrics(result: Mapping[str, Any], before: str, after: str) -> dict[str, object]:
    repetition = result["repetition_metrics"]
    images = result["generated_image_metrics"]
    markup = result["markup_metrics"]
    return {
        "chars_before": len(before),
        "chars_after": len(after),
        "complex_repetition_replacements": int(repetition["complex_repetition_replacements"]),
        "complex_repetition_characters_removed": int(repetition["complex_repetition_characters_removed"]),
        "complex_repetition_rule_counts": repetition["complex_repetition_rule_counts"],
        "generated_image_artifact_count": int(images["generated_image_artifact_count"]),
        "generated_image_characters_removed": int(images["generated_image_characters_removed"]),
        "image_description_comments_emitted": int(images["image_description_comments_emitted"]),
        "recognized_html_start_tags": int(sum(markup["tag_counts"].values())),
        "html_tag_counts": markup["tag_counts"],
        "transformations": markup["transformations"],
        "residual_recognized_html_tags": 0,
    }


def _task_output_paths(run_root: Path, stage: str, task_index: int) -> tuple[Path, Path]:
    directory = run_root / stage
    return directory / "shards" / f"task-{task_index:06d}.parquet", directory / "receipts" / f"task-{task_index:06d}.json"


def transform_task(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    cfg = load_config(args.config)
    run = contract(args.contract)
    tasks = read_json_object(args.tasks)
    if tasks.get("schema_version") != TASK_MANIFEST_SCHEMA:
        raise ValueError("unsupported transform task manifest")
    task_index = int(args.task_index)
    task = tasks["tasks"][task_index]
    if int(task["task_index"]) != task_index:
        raise ValueError("transform task index drift")
    source_id = str(task["source_id"])
    mapping = cfg["sources"][source_id]
    repetition, repetition_path = gfm._load_repetition_module(Path(run["glossapi"]["root"]))
    run_root = Path(str(run["run_root"]))
    output, receipt_path = _task_output_paths(run_root, "10-transform", task_index)
    if receipt_path.exists():
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") == TRANSFORM_RECEIPT_SCHEMA:
            validate_file_receipt(receipt["output"], root=run_root)
            print(canonical_json({"ok": True, "reused": True, "task": task_index}))
            return 0
        raise ValueError(f"invalid pre-existing receipt: {receipt_path}")

    audit_path = run_root / "10-transform" / "audits" / f"task-{task_index:06d}.jsonl.gz"
    issue_path = run_root / "10-transform" / "issues" / f"task-{task_index:06d}.jsonl.gz"
    audit_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    issue_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    audit_tmp = audit_path.with_name(f".{audit_path.name}.partial-{os.getpid()}")
    issue_tmp = issue_path.with_name(f".{issue_path.name}.partial-{os.getpid()}")
    parquet = pq.ParquetFile(Path(str(task["input_path"])))
    counters: Counter[str] = Counter()

    def rows() -> Iterable[list[dict[str, Any]]]:
        output_rows: list[dict[str, Any]] = []
        current_index = int(task["row_start"])
        with gzip.open(audit_tmp, "wt", encoding="utf-8") as audit, gzip.open(issue_tmp, "wt", encoding="utf-8") as issues:
            for row_group in task["row_groups"]:
                table = parquet.read_row_group(int(row_group))
                for row in table.to_pylist():
                    row_index = current_index
                    current_index += 1
                    counters["input_rows"] += 1
                    raw_text = optional_text(read_path(row, mapping["text_path"]))
                    if raw_text is None:
                        counters["quarantined_blank_text"] += 1
                        issues.write(canonical_json({
                            "source_id": source_id,
                            "artifact_path": task["artifact_path"],
                            "row_index": row_index,
                            "reason": "blank_selected_text",
                            "text_path": mapping["text_path"],
                        }) + "\n")
                        continue
                    original_sha = sha256_text(raw_text)
                    result = gfm.clean_then_normalize_to_gfm(
                        raw_text, repetition_cleaner=repetition.replace_complex_repetitions
                    )
                    normalized = str(result["normalized_markdown"])
                    if RECOGNIZED_HTML_RE.search(normalized) or GENERATED_IMAGE_RE.search(normalized):
                        raise RuntimeError(f"{source_id}:{row_index}: transformation postcondition failed")
                    if not normalized.strip():
                        counters["quarantined_empty_after_transform"] += 1
                        issues.write(canonical_json({
                            "source_id": source_id,
                            "artifact_path": task["artifact_path"],
                            "row_index": row_index,
                            "reason": "empty_after_transform",
                            "original_text_sha256": original_sha,
                        }) + "\n")
                        continue
                    transformed_sha = sha256_text(normalized)
                    candidate_id, synthetic = source_doc_id(
                        row,
                        mapping,
                        source_id=source_id,
                        revision=str(task["revision"]),
                        artifact_path=str(task["artifact_path"]),
                        row_index=row_index,
                        text_sha256=original_sha,
                    )
                    uid = sha256_json({
                        "namespace": "agent1_v5_source_row_uid_v1",
                        "source_id": source_id,
                        "revision": task["revision"],
                        "artifact_path": task["artifact_path"],
                        "row_index": row_index,
                        "original_text_sha256": original_sha,
                    })
                    compact = compact_transform_metrics(result, raw_text, normalized)
                    output_rows.append({
                        "source_id": source_id,
                        "source_repo_id": task["repo_id"],
                        "source_revision": task["revision"],
                        "source_dataset": task["repo_id"],
                        "source_doc_id_candidate": candidate_id,
                        "source_doc_id_was_synthetic": synthetic,
                        "source_artifact_path": task["artifact_path"],
                        "source_row_index": row_index,
                        "source_row_uid": uid,
                        "transformed_text": normalized,
                        "title": optional_text(read_path(row, mapping.get("title_path"))),
                        "author": optional_text(read_path(row, mapping.get("author_path")), author=True),
                        "source_metadata_json": metadata_json(row, mapping),
                        "original_text_sha256": original_sha,
                        "transformed_text_sha256": transformed_sha,
                        "transform_metrics_json": canonical_json(compact),
                    })
                    counters["output_rows"] += 1
                    counters["synthetic_source_doc_ids"] += int(synthetic)
                    if original_sha != transformed_sha:
                        counters["changed_rows"] += 1
                    if any((
                        compact["complex_repetition_replacements"],
                        compact["generated_image_artifact_count"],
                        compact["recognized_html_start_tags"],
                    )):
                        audit.write(canonical_json({
                            "source_row_uid": uid,
                            "source_id": source_id,
                            "source_doc_id_candidate": candidate_id,
                            "original_text_sha256": original_sha,
                            "transformed_text_sha256": transformed_sha,
                            "repetition_metrics": result["repetition_metrics"],
                            "generated_image_metrics": result["generated_image_metrics"],
                            "markup_metrics": result["markup_metrics"],
                        }) + "\n")
                        counters["audited_changed_rows"] += 1
                    if len(output_rows) >= int(cfg["execution"]["transform_batch_rows"]):
                        yield output_rows
                        output_rows = []
            if current_index != int(task["row_start"]) + int(task["rows"]):
                raise ValueError("row-group row count drift")
            if output_rows:
                yield output_rows

    output_rows = write_parquet_atomic(output, transform_schema(), rows())
    audit_tmp.replace(audit_path)
    issue_tmp.replace(issue_path)
    if output_rows != counters["output_rows"] or counters["input_rows"] != int(task["rows"]):
        raise ValueError("transform row closure failed")
    receipt: dict[str, object] = {
        "schema_version": TRANSFORM_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": task_index,
        "task_sha256": sha256_json(task),
        "run_contract_sha256": sha256_file(args.contract),
        "repetition_module": file_receipt(repetition_path),
        "input": {key: task[key] for key in ("source_id", "input_path", "artifact_path", "row_groups", "row_start", "rows")},
        "output": file_receipt(output, root=run_root, rows=output_rows),
        "audit": file_receipt(audit_path, root=run_root),
        "issues": file_receipt(issue_path, root=run_root),
        "counters": dict(counters),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": task_index, "rows": output_rows, "issues": counters["quarantined_blank_text"] + counters["quarantined_empty_after_transform"]}))
    return 0


def merge_transform(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    run = contract(args.contract)
    tasks = read_json_object(args.tasks)
    run_root = Path(str(run["run_root"]))
    receipts: list[dict[str, Any]] = []
    source_counts: dict[str, Counter[str]] = {source_id: Counter() for source_id in cfg["sources"]}
    for task in tasks["tasks"]:
        index = int(task["task_index"])
        _, receipt_path = _task_output_paths(run_root, "10-transform", index)
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") != TRANSFORM_RECEIPT_SCHEMA or receipt.get("task_sha256") != sha256_json(task):
            raise ValueError(f"transform receipt drift for task {index}")
        validate_file_receipt(receipt["output"], root=run_root)
        validate_file_receipt(receipt["audit"], root=run_root)
        validate_file_receipt(receipt["issues"], root=run_root)
        receipts.append(receipt)
        source_counts[str(task["source_id"])].update({str(key): int(value) for key, value in receipt["counters"].items()})
    if len(receipts) != int(tasks["task_count"]):
        raise ValueError("transform task coverage mismatch")
    blocking = []
    for source_id, counts in source_counts.items():
        if counts["output_rows"] == 0:
            blocking.append({"source_id": source_id, "reason": "source_has_zero_usable_text_rows"})
        missing = counts["quarantined_blank_text"] + counts["quarantined_empty_after_transform"]
        if missing:
            blocking.append({"source_id": source_id, "reason": "missing_or_empty_text_rows_require_user_review", "rows": missing})
        if counts["input_rows"] != counts["output_rows"] + missing:
            raise ValueError(f"{source_id}: transform row waterfall does not close")
    result: dict[str, object] = {
        "schema_version": TRANSFORM_MANIFEST_SCHEMA,
        "status": "blocked" if blocking else "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "task_manifest_sha256": sha256_file(args.tasks),
        "task_count": len(receipts),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "input_rows": sum(value["input_rows"] for value in source_counts.values()),
        "output_rows": sum(value["output_rows"] for value in source_counts.values()),
        "blocking_issues": blocking,
        "shards": [receipt["output"] for receipt in receipts],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": not blocking, "rows": result["output_rows"], "blocking_issues": len(blocking)}))
    return 2 if blocking else 0


def build_glossapi_runtime_receipt(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = args.glossapi_root.resolve()
    expected = str(cfg["pins"]["glossapi_commit"])
    if git_output(root, "rev-parse", "HEAD") != expected:
        raise ValueError("GlossAPI runtime source is not at the pinned commit")
    if git_output(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("GlossAPI runtime source checkout is dirty")
    modules = []
    for name in ("glossapi_rs_cleaner", "glossapi_rs_noise"):
        module = importlib.import_module(name)
        path = Path(str(module.__file__)).resolve()
        modules.append({"name": name, **file_receipt(path)})
    result: dict[str, object] = {
        "schema_version": "agent1_v5_glossapi_runtime_receipt_v1",
        "status": "passed",
        "created_at": utc_now(),
        "source_root": str(root),
        "source_commit": expected,
        "python": sys.version,
        "modules": modules,
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "commit": expected, "modules": [row["name"] for row in modules]}))
    return 0


def validate_glossapi_runtime(path: Path, expected_commit: str) -> tuple[Any, Any]:
    receipt = read_json_object(path)
    if (
        receipt.get("schema_version") != "agent1_v5_glossapi_runtime_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("source_commit") != expected_commit
    ):
        raise ValueError("unsupported GlossAPI runtime receipt")
    root = Path(str(receipt["source_root"])).resolve()
    if git_output(root, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("GlossAPI runtime checkout commit drift")
    if git_output(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("GlossAPI runtime checkout became dirty")
    declared = {str(row["name"]): row for row in receipt["modules"]}
    modules = []
    for name in ("glossapi_rs_cleaner", "glossapi_rs_noise"):
        module = importlib.import_module(name)
        imported = Path(str(module.__file__)).resolve()
        expected = validate_file_receipt(declared[name])
        if imported != expected:
            raise ValueError(f"{name}: imported module differs from receipt")
        modules.append(module)
    return modules[0], modules[1]


def _contains_latex(text: str) -> bool:
    return bool(LATEX_RE.search(text))


def _contains_math(text: str) -> bool:
    return _contains_latex(text) or bool(MATH_RE.search(text))


def _final_quality_row(
    *, source: Mapping[str, Any], text: str, canonical_text: str,
    noise: Mapping[str, Any], cleaner: Mapping[str, Any], reevaluated_at: dt.datetime,
) -> dict[str, Any]:
    greek_badness = noise.get("rust_noise_badness_score")
    polytonic_ratio = noise.get("rust_noise_polytonic_ratio")
    is_empty = bool(cleaner["cleaner_is_empty"]) or not text.strip()
    if is_empty:
        filter_value = "empty"
    elif greek_badness is not None and float(greek_badness) > 60:
        filter_value = "greek>60"
    else:
        filter_value = "ok"
    content_chars = int(cleaner["cleaner_characters_no_comments"])
    return {
        "is_historical_or_polytonic": bool(polytonic_ratio is not None and float(polytonic_ratio) >= 0.02),
        "contains_math": _contains_math(text),
        "contains_latex": _contains_latex(text),
        "greek_percentage": cleaner.get("cleaner_greek_percentage"),
        "latin_percentage": cleaner.get("cleaner_latin_percentage"),
        "polytonic_ratio": polytonic_ratio,
        "table_ratio": noise.get("rust_noise_table_ratio"),
        "greek_badness_score": greek_badness,
        "len_greek": noise.get("rust_noise_greek_characters"),
        "mojibake_badness_score": cleaner.get("cleaner_badness_score"),
        "needs_ocr": None,
        "is_empty": is_empty,
        "filter": filter_value,
        "ocr_success": None,
        "quality_method": "glossapi_rs_cleaner+glossapi_rs_noise",
        "reevaluated_at": reevaluated_at,
        "content_chars_kept": content_chars,
        "chars_dropped_by_line_drop": None,
        "chars_dropped_by_normalization": None,
        "chars_dropped_by_per_char_filter": None,
        "lines_dropped_by_cleaner": None,
        "marker_chars_passthrough": None,
        "marker_chars_added": None,
        "charset_greek_ratio": (float(cleaner["cleaner_greek_percentage"]) / 100.0 if cleaner.get("cleaner_greek_percentage") is not None else None),
        "charset_moji_ratio": None,
        "charset_punct_ratio": None,
        "mojibake_noise_ratio": None,
        "rule_a_match_count": None,
        "rule_b_match_count": None,
        "residue_line_drop_count": None,
        "phase_a_fallback_reason": None,
        "phase_a_dialect_ambiguous_input": None,
        "cleaner_chars_before": len(canonical_text),
        "cleaner_chars_after": len(text),
        "chars": len(text),
        "non_whitespace_chars": sum(not char.isspace() for char in text),
        "utf8_bytes": len(text.encode("utf-8")),
        "approx_word_count": len(WORD_RE.findall(text)),
    }


def glossapi_task(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    cfg = load_config(args.config)
    run = contract(args.contract)
    manifest = read_json_object(args.transform_manifest)
    if manifest.get("schema_version") != TRANSFORM_MANIFEST_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("transform manifest is not passed")
    cleaner_module, noise_module = validate_glossapi_runtime(
        args.runtime_receipt, str(cfg["pins"]["glossapi_commit"])
    )
    glossapi_root = Path(str(run["glossapi"]["root"]))
    if str(glossapi_root / "src") not in sys.path:
        sys.path.insert(0, str(glossapi_root / "src"))
    from glossapi.ocr.utils.cleaning import canonicalize_markdown
    from profile_dataset_quality_rust import parse_cleaner_report, parse_noise_rows

    task_index = int(args.task_index)
    run_root = Path(str(run["run_root"]))
    input_binding = manifest["shards"][task_index]
    input_path = validate_file_receipt(input_binding, root=run_root)
    output, receipt_path = _task_output_paths(run_root, "20-glossapi", task_index)
    if receipt_path.exists():
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") == GLOSSAPI_RECEIPT_SCHEMA:
            validate_file_receipt(receipt["output"], root=run_root)
            print(canonical_json({"ok": True, "reused": True, "task": task_index}))
            return 0
        raise ValueError(f"invalid pre-existing receipt: {receipt_path}")
    issue_path = run_root / "20-glossapi" / "issues" / f"task-{task_index:06d}.jsonl.gz"
    issue_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    issue_tmp = issue_path.with_name(f".{issue_path.name}.partial-{os.getpid()}")
    parquet = pq.ParquetFile(input_path)
    counters: Counter[str] = Counter()
    reevaluated = dt.datetime.fromisoformat(str(run["created_at"]))
    scratch_root = args.scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def process_batch(batch_rows: list[dict[str, Any]], issues: Any) -> list[dict[str, Any]]:
        if not batch_rows:
            return []
        with tempfile.TemporaryDirectory(prefix=f"agent1-v5-gloss-{task_index:06d}-", dir=scratch_root) as temp_name:
            temporary = Path(temp_name)
            raw_dir = temporary / "input"
            clean_dir = temporary / "cleaned"
            report_path = temporary / "cleaner.parquet"
            raw_dir.mkdir()
            expected: set[str] = set()
            canonical_by_key: dict[str, str] = {}
            source_by_key: dict[str, dict[str, Any]] = {}
            for index, source in enumerate(batch_rows):
                key = f"d{index:07d}"
                canonical = canonicalize_markdown(str(source["transformed_text"]))
                (raw_dir / f"{key}.md").write_text(canonical, encoding="utf-8")
                expected.add(key)
                canonical_by_key[key] = canonical
                source_by_key[key] = source
            noise_rows = noise_module.score_markdown_directory_detailed(str(raw_dir), int(args.threads))
            noise = parse_noise_rows(noise_rows, expected)
            cleaner_module.run_complete_pipeline(
                str(raw_dir),
                str(clean_dir),
                str(report_path),
                ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"],
                int(args.threads),
                True,
            )
            cleaner = parse_cleaner_report(report_path, expected)
            output_rows = []
            for key in sorted(expected):
                source = source_by_key[key]
                cleaned_path = clean_dir / f"{key}.md"
                if not cleaned_path.is_file():
                    raise ValueError(f"GlossAPI did not emit cleaned text for {key}")
                text = cleaned_path.read_text(encoding="utf-8")
                quality = _final_quality_row(
                    source=source,
                    text=text,
                    canonical_text=canonical_by_key[key],
                    noise=noise[key],
                    cleaner=cleaner[key],
                    reevaluated_at=reevaluated,
                )
                counters["input_rows"] += 1
                if quality["is_empty"]:
                    counters["quarantined_empty_after_glossapi"] += 1
                    issues.write(canonical_json({
                        "source_id": source["source_id"],
                        "source_row_uid": source["source_row_uid"],
                        "reason": "empty_after_glossapi",
                        "transformed_text_sha256": source["transformed_text_sha256"],
                    }) + "\n")
                    continue
                output_rows.append({**source, "text": text, **quality})
                counters["output_rows"] += 1
            return output_rows

    def output_batches() -> Iterable[list[dict[str, Any]]]:
        with gzip.open(issue_tmp, "wt", encoding="utf-8") as issues:
            pending: list[dict[str, Any]] = []
            for batch in parquet.iter_batches(batch_size=int(cfg["execution"]["glossapi_batch_rows"])):
                pending.extend(batch.to_pylist())
                if pending:
                    yield process_batch(pending, issues)
                    pending = []

    rows_written = write_parquet_atomic(output, glossapi_schema(), output_batches())
    issue_tmp.replace(issue_path)
    if counters["input_rows"] != int(input_binding["rows"]):
        raise ValueError("GlossAPI input row closure failed")
    if counters["input_rows"] != counters["output_rows"] + counters["quarantined_empty_after_glossapi"]:
        raise ValueError("GlossAPI output row closure failed")
    receipt: dict[str, object] = {
        "schema_version": GLOSSAPI_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": task_index,
        "run_contract_sha256": sha256_file(args.contract),
        "runtime_receipt_sha256": sha256_file(args.runtime_receipt),
        "input": input_binding,
        "output": file_receipt(output, root=run_root, rows=rows_written),
        "issues": file_receipt(issue_path, root=run_root),
        "counters": dict(counters),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": task_index, "rows": rows_written}))
    return 0


def merge_glossapi(args: argparse.Namespace) -> int:
    run = contract(args.contract)
    transform = read_json_object(args.transform_manifest)
    if transform.get("schema_version") != TRANSFORM_MANIFEST_SCHEMA or transform.get("status") != "passed":
        raise ValueError("transform manifest is not passed")
    run_root = Path(str(run["run_root"]))
    receipts = []
    counters: Counter[str] = Counter()
    for index, input_binding in enumerate(transform["shards"]):
        _, receipt_path = _task_output_paths(run_root, "20-glossapi", index)
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") != GLOSSAPI_RECEIPT_SCHEMA or receipt.get("input", {}).get("sha256") != input_binding["sha256"]:
            raise ValueError(f"GlossAPI receipt drift for task {index}")
        output_path = validate_file_receipt(receipt["output"], root=run_root)
        validate_file_receipt(receipt["issues"], root=run_root)
        if str(output_path.suffix) != ".parquet":
            raise ValueError("GlossAPI output is not Parquet")
        receipts.append(receipt)
        counters.update({str(key): int(value) for key, value in receipt["counters"].items()})
    if counters["input_rows"] != int(transform["output_rows"]):
        raise ValueError("GlossAPI manifest input closure failed")
    blocking = []
    if counters["quarantined_empty_after_glossapi"]:
        blocking.append({
            "reason": "empty_after_glossapi_rows_require_user_review",
            "rows": counters["quarantined_empty_after_glossapi"],
        })
    result: dict[str, object] = {
        "schema_version": GLOSSAPI_MANIFEST_SCHEMA,
        "status": "blocked" if blocking else "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "transform_manifest_sha256": sha256_file(args.transform_manifest),
        "task_count": len(receipts),
        "input_rows": counters["input_rows"],
        "output_rows": counters["output_rows"],
        "counters": dict(counters),
        "blocking_issues": blocking,
        "shards": [receipt["output"] for receipt in receipts],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": not blocking, "rows": result["output_rows"], "blocking_issues": len(blocking)}))
    return 2 if blocking else 0


def plan_envelope(args: argparse.Namespace) -> int:
    import sqlite3
    import pyarrow.parquet as pq

    run = contract(args.contract)
    manifest = read_json_object(args.glossapi_manifest)
    if manifest.get("schema_version") != GLOSSAPI_MANIFEST_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("GlossAPI manifest is not passed")
    run_root = Path(str(run["run_root"]))
    database = run_root / "30-envelope" / "source_doc_ids.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if database.exists():
        raise FileExistsError(f"immutable output exists: {database}")
    connection = sqlite3.connect(database)
    # Keep the immutable lookup self-contained in one file. A WAL sidecar
    # would not be covered by the database receipt consumed by array tasks.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("CREATE TABLE ids(source_dataset TEXT NOT NULL, candidate TEXT NOT NULL, uid TEXT NOT NULL PRIMARY KEY)")
    connection.execute("CREATE INDEX ids_key ON ids(source_dataset, candidate)")
    rows = 0
    for binding in manifest["shards"]:
        path = validate_file_receipt(binding, root=run_root)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["source_dataset", "source_doc_id_candidate", "source_row_uid"], batch_size=8192):
            values = [
                (str(row["source_dataset"]), str(row["source_doc_id_candidate"]), str(row["source_row_uid"]))
                for row in batch.to_pylist()
            ]
            connection.executemany("INSERT INTO ids VALUES (?, ?, ?)", values)
            rows += len(values)
    connection.commit()
    duplicate_keys = int(connection.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM ids GROUP BY source_dataset, candidate HAVING COUNT(*) > 1)"
    ).fetchone()[0])
    connection.close()
    if rows != int(manifest["output_rows"]):
        raise ValueError("envelope ID scan row closure failed")
    result: dict[str, object] = {
        "schema_version": ENVELOPE_PLAN_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "glossapi_manifest_sha256": sha256_file(args.glossapi_manifest),
        "task_count": len(manifest["shards"]),
        "rows": rows,
        "duplicate_source_doc_id_keys": duplicate_keys,
        "id_database": file_receipt(database, root=run_root),
        "input_shards": manifest["shards"],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "tasks": result["task_count"], "duplicate_keys": duplicate_keys}))
    return 0


def envelope_task(args: argparse.Namespace) -> int:
    import sqlite3
    import pyarrow.parquet as pq

    run = contract(args.contract)
    plan = read_json_object(args.envelope_plan)
    if plan.get("schema_version") != ENVELOPE_PLAN_SCHEMA or plan.get("status") != "passed":
        raise ValueError("envelope plan is not passed")
    run_root = Path(str(run["run_root"]))
    task_index = int(args.task_index)
    input_binding = plan["input_shards"][task_index]
    input_path = validate_file_receipt(input_binding, root=run_root)
    database = validate_file_receipt(plan["id_database"], root=run_root)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    duplicate_keys = {
        (str(row[0]), str(row[1]))
        for row in connection.execute("SELECT source_dataset, candidate FROM ids GROUP BY source_dataset, candidate HAVING COUNT(*) > 1")
    }
    connection.close()
    output, receipt_path = _task_output_paths(run_root, "30-envelope", task_index)
    if receipt_path.exists():
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") == ENVELOPE_RECEIPT_SCHEMA:
            validate_file_receipt(receipt["output"], root=run_root)
            print(canonical_json({"ok": True, "reused": True, "task": task_index}))
            return 0
        raise ValueError(f"invalid pre-existing receipt: {receipt_path}")
    parquet = pq.ParquetFile(input_path)
    counters: Counter[str] = Counter()

    def batches() -> Iterable[list[dict[str, Any]]]:
        for batch in parquet.iter_batches(batch_size=2048):
            output_rows = []
            for row in batch.to_pylist():
                source_dataset = str(row["source_dataset"])
                candidate = str(row["source_doc_id_candidate"])
                source_doc_id = candidate
                if (source_dataset, candidate) in duplicate_keys:
                    source_doc_id = f"{candidate}#{str(row['source_row_uid'])}"
                    counters["collision_ids_rewritten"] += 1
                payload = {name: row.get(name) for name in CANONICAL_FIELD_NAMES}
                payload.update({
                    "source_dataset": source_dataset,
                    "source_doc_id": source_doc_id,
                    "text": row["text"],
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "source_metadata_json": row.get("source_metadata_json"),
                })
                output_rows.append(payload)
                counters["rows"] += 1
            yield output_rows

    rows_written = write_parquet_atomic(output, canonical_schema(), batches())
    if rows_written != int(input_binding["rows"]):
        raise ValueError("envelope task row closure failed")
    receipt: dict[str, object] = {
        "schema_version": ENVELOPE_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": task_index,
        "run_contract_sha256": sha256_file(args.contract),
        "envelope_plan_sha256": sha256_file(args.envelope_plan),
        "input": input_binding,
        "output": file_receipt(output, root=run_root, rows=rows_written),
        "counters": dict(counters),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": task_index, "rows": rows_written}))
    return 0


def merge_envelope(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    run = contract(args.contract)
    plan = read_json_object(args.envelope_plan)
    if plan.get("schema_version") != ENVELOPE_PLAN_SCHEMA or plan.get("status") != "passed":
        raise ValueError("envelope plan is not passed")
    run_root = Path(str(run["run_root"]))
    receipts = []
    rows = 0
    rewritten = 0
    expected_schema = canonical_schema()
    for index, input_binding in enumerate(plan["input_shards"]):
        _, receipt_path = _task_output_paths(run_root, "30-envelope", index)
        receipt = read_json_object(receipt_path)
        if (
            receipt.get("schema_version") != ENVELOPE_RECEIPT_SCHEMA
            or receipt.get("input", {}).get("sha256") != input_binding["sha256"]
        ):
            raise ValueError(f"envelope receipt drift for task {index}")
        path = validate_file_receipt(receipt["output"], root=run_root)
        if not pq.ParquetFile(path).schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError(f"canonical schema drift: {path}")
        rows += int(receipt["output"]["rows"])
        rewritten += int(receipt["counters"].get("collision_ids_rewritten", 0))
        receipts.append(receipt)
    if rows != int(plan["rows"]):
        raise ValueError("candidate envelope row closure failed")
    result: dict[str, object] = {
        "schema_version": "agent1_v5_candidate_envelope_manifest_v1",
        "status": "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "envelope_plan_sha256": sha256_file(args.envelope_plan),
        "rows": rows,
        "task_count": len(receipts),
        "collision_ids_rewritten": rewritten,
        "schema": str(expected_schema),
        "shards": [receipt["output"] for receipt in receipts],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "rows": rows, "tasks": len(receipts)}))
    return 0


def plan_base(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    run = contract(args.contract)
    acquisition = read_json_object(validate_file_receipt(run["acquisition_receipt"]))
    base_rows = [row for row in acquisition["sources"] if row.get("source_id") == "nanochat_base"]
    if len(base_rows) != 1:
        raise ValueError("acquisition receipt does not have exactly one Nanochat base")
    base = base_rows[0]
    if base.get("repo_id") != run["pins"]["nanochat_repo_id"] or base.get("revision") != run["pins"]["nanochat_revision"]:
        raise ValueError("Nanochat base identity drift")
    tasks = []
    for row in sorted(base["files"], key=lambda item: str(item.get("path", ""))):
        path = Path(str(row["local_path"])).resolve()
        if path.suffix.casefold() != ".parquet":
            continue
        parquet = pq.ParquetFile(path)
        tasks.append({
            "task_index": len(tasks),
            "source_id": "nanochat_base",
            "repo_id": base["repo_id"],
            "revision": base["revision"],
            "artifact_path": row["path"],
            "input_path": str(path),
            "input_expected_hash": row.get("expected_hash"),
            "input_hash_kind": row.get("hash_kind"),
            "rows": int(parquet.metadata.num_rows),
            "row_groups": parquet.num_row_groups,
        })
    if not tasks:
        raise ValueError("Nanochat base has no Parquet files")
    result: dict[str, object] = {
        "schema_version": BASE_PLAN_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "task_count": len(tasks),
        "rows": sum(int(row["rows"]) for row in tasks),
        "tasks": tasks,
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "tasks": len(tasks), "rows": result["rows"]}))
    return 0


def _update_text_stream_hash(hasher: Any, values: Sequence[Any]) -> None:
    for value in values:
        text = "" if value is None else str(value)
        encoded = text.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)


def cast_base_task(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    run = contract(args.contract)
    plan = read_json_object(args.base_plan)
    if plan.get("schema_version") != BASE_PLAN_SCHEMA or plan.get("status") != "passed":
        raise ValueError("base plan is not passed")
    task_index = int(args.task_index)
    task = plan["tasks"][task_index]
    run_root = Path(str(run["run_root"]))
    output, receipt_path = _task_output_paths(run_root, "35-base-cast", task_index)
    if receipt_path.exists():
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") == BASE_RECEIPT_SCHEMA:
            validate_file_receipt(receipt["output"], root=run_root)
            print(canonical_json({"ok": True, "reused": True, "task": task_index}))
            return 0
        raise ValueError(f"invalid pre-existing receipt: {receipt_path}")
    source = pq.ParquetFile(Path(str(task["input_path"])))
    target = canonical_schema()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    writer = pq.ParquetWriter(temporary, target, compression="zstd", use_dictionary=True, write_statistics=True)
    input_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=int(args.batch_rows)):
            if "text" not in batch.schema.names:
                raise ValueError(f"Nanochat base shard lacks text: {task['input_path']}")
            original_text = batch.column(batch.schema.get_field_index("text")).to_pylist()
            _update_text_stream_hash(input_hash, original_text)
            arrays = []
            for field in target:
                if field.name in batch.schema.names:
                    source_array = batch.column(batch.schema.get_field_index(field.name))
                    try:
                        array = pc.cast(source_array, field.type, safe=False)
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                        raise ValueError(f"{task['artifact_path']}:{field.name}: cannot cast {source_array.type} to {field.type}") from exc
                else:
                    array = pa.nulls(batch.num_rows, type=field.type)
                arrays.append(array)
            output_batch = pa.RecordBatch.from_arrays(arrays, schema=target)
            _update_text_stream_hash(output_hash, output_batch.column(target.get_field_index("text")).to_pylist())
            writer.write_batch(output_batch)
            rows += batch.num_rows
        writer.close()
        if rows != int(task["rows"]):
            raise ValueError("Nanochat base cast row closure failed")
        if input_hash.hexdigest() != output_hash.hexdigest():
            raise ValueError("Nanochat base text changed during schema cast")
        temporary.replace(output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    receipt: dict[str, object] = {
        "schema_version": BASE_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": task_index,
        "run_contract_sha256": sha256_file(args.contract),
        "base_plan_sha256": sha256_file(args.base_plan),
        "input": task,
        "text_stream_sha256": input_hash.hexdigest(),
        "output": file_receipt(output, root=run_root, rows=rows),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": task_index, "rows": rows}))
    return 0


def merge_base(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    run = contract(args.contract)
    plan = read_json_object(args.base_plan)
    if plan.get("schema_version") != BASE_PLAN_SCHEMA or plan.get("status") != "passed":
        raise ValueError("base plan is not passed")
    run_root = Path(str(run["run_root"]))
    expected_schema = canonical_schema()
    receipts = []
    rows = 0
    for task in plan["tasks"]:
        index = int(task["task_index"])
        _, receipt_path = _task_output_paths(run_root, "35-base-cast", index)
        receipt = read_json_object(receipt_path)
        if receipt.get("schema_version") != BASE_RECEIPT_SCHEMA or receipt.get("input", {}).get("input_path") != task["input_path"]:
            raise ValueError(f"base receipt drift for task {index}")
        path = validate_file_receipt(receipt["output"], root=run_root)
        if not pq.ParquetFile(path).schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError(f"base canonical schema drift: {path}")
        rows += int(receipt["output"]["rows"])
        receipts.append(receipt)
    if rows != int(plan["rows"]):
        raise ValueError("Nanochat base merged row closure failed")
    result: dict[str, object] = {
        "schema_version": "agent1_v5_base_manifest_v1",
        "status": "passed",
        "created_at": utc_now(),
        "run_contract_sha256": sha256_file(args.contract),
        "base_plan_sha256": sha256_file(args.base_plan),
        "rows": rows,
        "task_count": len(receipts),
        "shards": [receipt["output"] for receipt in receipts],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "rows": rows, "tasks": len(receipts)}))
    return 0


def _hardlink_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable combined output exists: {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OSError(
            f"combined release must be on the same filesystem as stage outputs; cannot hardlink {source} to {destination}"
        ) from exc


def combine_release(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = contract(args.contract)
    base = read_json_object(args.base_manifest)
    candidates = read_json_object(args.candidate_manifest)
    if base.get("schema_version") != "agent1_v5_base_manifest_v1" or base.get("status") != "passed":
        raise ValueError("base manifest is not passed")
    if candidates.get("schema_version") != "agent1_v5_candidate_envelope_manifest_v1" or candidates.get("status") != "passed":
        raise ValueError("candidate manifest is not passed")
    run_root = Path(str(run["run_root"]))
    release_root = args.output.resolve()
    release_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    inventory_rows = []
    rank = 0
    try:
        for origin, manifest in (("nanochat_base", base), ("candidate", candidates)):
            for binding in manifest["shards"]:
                source = validate_file_receipt(binding, root=run_root)
                relative = Path("data") / f"{rank:06d}.parquet"
                destination = release_root / relative
                _hardlink_immutable(source, destination)
                if not pq.ParquetFile(destination).schema_arrow.equals(canonical_schema(), check_metadata=False):
                    raise ValueError(f"combined schema drift: {destination}")
                inventory_rows.append({
                    "rank": rank,
                    "origin": origin,
                    "path": relative.as_posix(),
                    "rows": int(binding["rows"]),
                    "bytes": int(binding["bytes"]),
                    "sha256": str(binding["sha256"]),
                })
                rank += 1
        inventory_schema = pa.schema([
            ("rank", pa.int64()),
            ("origin", pa.string()),
            ("path", pa.string()),
            ("rows", pa.int64()),
            ("bytes", pa.int64()),
            ("sha256", pa.string()),
        ])
        write_parquet_atomic(release_root / "manifests" / "dedup_input_inventory.parquet", inventory_schema, [inventory_rows])
        license_source = run_root / "license_override_receipt.json"
        shutil.copy2(license_source, release_root / "manifests" / "license_override_receipt.json")
        card = (
            "---\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files: data/*.parquet\n"
            "---\n\n"
            f"# Greek Nanochat plus new sources — pre-dedup `{run['run_id']}`\n\n"
            "Private, receipt-bound intermediate release. The Nanochat base is schema-cast without text changes; "
            "candidate documents have passed extraction-artifact cleaning, HTML-to-GFM conversion, and GlossAPI.\n"
        )
        (release_root / "README.md").write_text(card, encoding="utf-8")
        result: dict[str, object] = {
            "schema_version": COMBINED_MANIFEST_SCHEMA,
            "status": "passed",
            "created_at": utc_now(),
            "run_id": run["run_id"],
            "run_contract_sha256": sha256_file(args.contract),
            "base_manifest_sha256": sha256_file(args.base_manifest),
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "repository_id": run["private_repositories"]["pre_dedup"],
            "private_only": True,
            "root": str(release_root),
            "schema": str(canonical_schema()),
            "base_rows": int(base["rows"]),
            "candidate_rows": int(candidates["rows"]),
            "rows": int(base["rows"]) + int(candidates["rows"]),
            "files": inventory_rows,
            "inventory": file_receipt(release_root / "manifests" / "dedup_input_inventory.parquet", root=release_root, rows=len(inventory_rows)),
        }
        write_json_atomic(release_root / "manifests" / "combined_manifest.json", result)
    except BaseException:
        shutil.rmtree(release_root, ignore_errors=True)
        raise
    print(canonical_json({"ok": True, "rows": result["rows"], "files": len(inventory_rows), "root": str(release_root)}))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("freeze-contract")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--acquisition-receipt", type=Path, required=True)
    command.add_argument("--glossapi-root", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--run-id", required=True)
    command.set_defaults(func=freeze_contract)

    command = subparsers.add_parser("plan-transform")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=plan_transform_tasks)

    command = subparsers.add_parser("transform-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=transform_task)

    command = subparsers.add_parser("merge-transform")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_transform)

    command = subparsers.add_parser("build-glossapi-runtime-receipt")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--glossapi-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=build_glossapi_runtime_receipt)

    command = subparsers.add_parser("glossapi-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--transform-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.add_argument("--scratch-root", type=Path, required=True)
    command.add_argument("--threads", type=int, default=8)
    command.set_defaults(func=glossapi_task)

    command = subparsers.add_parser("merge-glossapi")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--transform-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_glossapi)

    command = subparsers.add_parser("plan-envelope")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--glossapi-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=plan_envelope)

    command = subparsers.add_parser("envelope-task")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--envelope-plan", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=envelope_task)

    command = subparsers.add_parser("merge-envelope")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--envelope-plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_envelope)

    command = subparsers.add_parser("plan-base")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=plan_base)

    command = subparsers.add_parser("cast-base-task")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--base-plan", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.add_argument("--batch-rows", type=int, default=1024)
    command.set_defaults(func=cast_base_task)

    command = subparsers.add_parser("merge-base")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--base-plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_base)

    command = subparsers.add_parser("combine-release")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--base-manifest", type=Path, required=True)
    command.add_argument("--candidate-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=combine_release)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
