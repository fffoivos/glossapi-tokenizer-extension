#!/usr/bin/env python3
"""Derive a SPAN rehydration source receipt from a passed Phase-04 acquisition."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .span_rehydration import RehydrationError, _open_jsonl_text, sha256_file


HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
PHASE04_DIR = HERE.parents[3] / "04_full_corpus_preparation"
DEFAULT_SOURCES = PHASE04_DIR / "configs" / "sources.json"
DEFAULT_MANIFEST = EVAL_DIR / "units" / "SPAN_manifest.jsonl"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_LOGICAL_SOURCES = {"greek_phd", "openarchives", "kallipos"}


@dataclass(frozen=True)
class Route:
    logical_source: str
    acquisition_source_id: str
    path_patterns: tuple[str, ...]
    format: str
    fields: Mapping[str, Any]
    historical_source_relation: str
    document_id_alignment: str


def _load_object(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RehydrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise RehydrationError(f"{path}: expected schema_version {schema}")
    return value


def _unique_rows(rows: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RehydrationError(f"{name}: expected a list")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("source_id"):
            raise RehydrationError(f"{name}:{index}: invalid source row")
        source_id = str(row["source_id"])
        if source_id in result:
            raise RehydrationError(f"{name}: duplicate source_id {source_id!r}")
        result[source_id] = row
    return result


def _config_sources(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {
        "nanochat_base": config.get("base", {}),
        "apertus_overlap_overlay": config.get("apertus_overlap_overlay", {}),
        "modern_greek_148k_tokenizer": config.get("tokenizer", {}),
    }
    for row in config.get("sources", []):
        if isinstance(row, dict) and row.get("source_id"):
            result[str(row["source_id"])] = row
    return result


def _under(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_phase04_bindings(
    acquisition_path: Path,
    lock_path: Path,
    sources_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    acquisition = _load_object(acquisition_path, "full_cpt_acquisition_receipt_v1")
    lock = _load_object(lock_path, "full_cpt_sources_lock_v1")
    sources = _load_object(sources_path, "full_cpt_sources_v1")
    if acquisition.get("status") != "passed":
        raise RehydrationError(f"{acquisition_path}: acquisition status is not passed")
    lock_sha = sha256_file(lock_path)
    sources_sha = sha256_file(sources_path)
    if acquisition.get("source_lock_sha256") != lock_sha:
        raise RehydrationError(
            "acquisition receipt is not bound to the supplied source lock"
        )
    if acquisition.get("sources_config_sha256") != sources_sha:
        raise RehydrationError(
            "acquisition receipt is not bound to the supplied sources.json"
        )
    if lock.get("sources_config_sha256") != sources_sha:
        raise RehydrationError("source lock is not bound to the supplied sources.json")
    recorded_lock = Path(str(acquisition.get("source_lock", ""))).resolve()
    if recorded_lock != lock_path.resolve():
        raise RehydrationError(
            f"acquisition receipt names a different source lock: {recorded_lock}"
        )
    locked = _unique_rows(lock.get("sources"), "source lock")
    acquired = _unique_rows(acquisition.get("sources"), "acquisition receipt")
    if set(locked) != set(acquired):
        raise RehydrationError("acquisition and lock source inventories differ")
    configured = _config_sources(sources)
    for source_id, lock_row in locked.items():
        receipt_row = acquired[source_id]
        config_row = configured.get(source_id)
        if config_row is None:
            raise RehydrationError(f"{source_id}: absent from sources.json")
        for field, configured_value in (
            ("repo_id", config_row.get("repo_id")),
            ("repo_type", config_row.get("repo_type", "dataset")),
            ("revision", config_row.get("revision")),
        ):
            if (
                lock_row.get(field) != configured_value
                or receipt_row.get(field) != configured_value
            ):
                raise RehydrationError(
                    f"{source_id}: {field} differs between config, lock, and acquisition receipt"
                )
        locked_files = {
            str(row.get("path")): row
            for row in lock_row.get("selected_files", [])
            if isinstance(row, dict) and row.get("path")
        }
        receipt_files = {
            str(row.get("path")): row
            for row in receipt_row.get("files", [])
            if isinstance(row, dict) and row.get("path")
        }
        if len(locked_files) != len(lock_row.get("selected_files", [])):
            raise RehydrationError(
                f"{source_id}: lock contains duplicate/invalid files"
            )
        if len(receipt_files) != len(receipt_row.get("files", [])):
            raise RehydrationError(
                f"{source_id}: acquisition contains duplicate/invalid files"
            )
        if set(locked_files) != set(receipt_files):
            raise RehydrationError(
                f"{source_id}: acquisition file inventory differs from lock"
            )
        local_root = Path(str(receipt_row.get("local_root", ""))).resolve()
        for relative, locked_file in locked_files.items():
            receipt_file = receipt_files[relative]
            lfs_sha = locked_file.get("lfs_sha256")
            expected_hash = lfs_sha or locked_file.get("blob_id")
            expected_kind = "lfs_sha256" if lfs_sha else "git_blob_id"
            if not isinstance(expected_hash, str) or not expected_hash:
                raise RehydrationError(
                    f"{source_id}:{relative}: lock has no content identity"
                )
            if (
                receipt_file.get("hash_kind") != expected_kind
                or receipt_file.get("expected_hash") != expected_hash
            ):
                raise RehydrationError(
                    f"{source_id}:{relative}: acquisition hash does not match the LFS lock"
                )
            local_path = Path(str(receipt_file.get("local_path", ""))).resolve()
            expected_local = (local_root / relative).resolve()
            if local_path != expected_local or not _under(local_root, local_path):
                raise RehydrationError(
                    f"{source_id}:{relative}: unsafe/inconsistent local path"
                )
            if not local_path.is_file():
                raise RehydrationError(
                    f"{source_id}:{relative}: acquired file is missing"
                )
            stat = local_path.stat()
            if stat.st_size != int(locked_file.get("size", -1)):
                raise RehydrationError(f"{source_id}:{relative}: acquired size drift")
            for field, actual in (
                ("device", stat.st_dev),
                ("inode", stat.st_ino),
                ("mtime_ns", stat.st_mtime_ns),
                ("ctime_ns", stat.st_ctime_ns),
            ):
                if int(receipt_file.get(field, -1)) != int(actual):
                    raise RehydrationError(
                        f"{source_id}:{relative}: acquired {field} drift"
                    )
    return acquisition, locked, acquired, configured


def _manifest_source_ids(path: Path, source: str) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RehydrationError(
                    f"{path}:{line_number}: manifest row is not an object"
                )
            if row.get("source") == source:
                result.add(str(row.get("doc_id", "")))
    if not result or "" in result:
        raise RehydrationError(f"{path}: no valid {source} document identities")
    return result


def _routes(args: argparse.Namespace, config: Mapping[str, Any]) -> list[Route]:
    if args.greek_phd_route == "nanochat_base":
        if args.greek_phd_document_id_column or args.greek_phd_text_column:
            raise RehydrationError(
                "Greek PhD Nanochat route has pinned source_doc_id/text fields; do not override them"
            )
        greek = Route(
            logical_source="greek_phd",
            acquisition_source_id="nanochat_base",
            path_patterns=(
                "data/greek_phd.part-00000.parquet",
                "data/greek_phd.part-00001.parquet",
            ),
            format="parquet_documents",
            fields={
                "document_id": "source_doc_id",
                "text_precedence": ["text"],
                "row_filter": {"column": "source_dataset", "equals": "greek_phd"},
            },
            historical_source_relation=(
                "Nanochat processed representation of Greek PhD; closer identifier domain than v2, "
                "but not the historical raw Mozilla JSONL snapshot"
            ),
            document_id_alignment="hash_domain_compatible_unverified",
        )
    else:
        configured = _config_sources(config)["greek_phd_v2"]
        id_column = args.greek_phd_document_id_column
        if not id_column:
            raise RehydrationError(
                "greek_phd_v2 does not declare the historical hash doc_id; pass an explicit "
                "--greek-phd-document-id-column and --allow-unverified-greek-phd-id-domain"
            )
        text_precedence = args.greek_phd_text_column or [
            *configured.get("text_columns", []),
            *configured.get("alternate_text_columns", []),
        ]
        if not args.allow_unverified_greek_phd_id_domain:
            raise RehydrationError(
                "greek_phd_v2 identifier alignment is unverified; explicit "
                "--allow-unverified-greek-phd-id-domain is required"
            )
        greek = Route(
            logical_source="greek_phd",
            acquisition_source_id="greek_phd_v2",
            path_patterns=("Greek PhD Theses Corpus v2.0.parquet",),
            format="parquet_documents",
            fields={"document_id": id_column, "text_precedence": text_precedence},
            historical_source_relation=(
                "newer Greek PhD extraction; historical labels used raw Mozilla JSONL and this "
                "route must not be treated as equivalent"
            ),
            document_id_alignment="unverified_nonhistorical_identifier_requires_mapping",
        )
    if args.kallipos_route == "kallipos_sections":
        kallipos = Route(
            logical_source="kallipos",
            acquisition_source_id="kallipos_sections",
            path_patterns=("Dataset_Kallipos.parquet",),
            format="parquet_sections",
            fields={"filename": "filename", "order": "id", "section": "section"},
            historical_source_relation=(
                "same repository artifact family used by the historical section-grouping builder; "
                "revision equivalence remains unverified"
            ),
            document_id_alignment="filename_domain_compatible_unverified",
        )
    else:
        kallipos = Route(
            logical_source="kallipos",
            acquisition_source_id="nanochat_base",
            path_patterns=("data/Apothetirio_Kallipos.parquet",),
            format="parquet_documents",
            fields={
                "document_id": "source_doc_id",
                "text_precedence": ["text"],
                "row_filter": {
                    "column": "source_dataset",
                    "equals": "Apothetirio_Kallipos",
                },
            },
            historical_source_relation=(
                "Nanochat processed document representation; historical labels used grouped raw "
                "Kallipos sections"
            ),
            document_id_alignment="paper_id_domain_compatible_unverified",
        )
    openarchives = Route(
        logical_source="openarchives",
        acquisition_source_id="openarchives_current",
        path_patterns=("data/openarchives/**/*.jsonl.zst",),
        format="jsonl_documents",
        fields={
            "document_id": "doc_id",
            "text_precedence": ["text", "document", "content"],
        },
        historical_source_relation=(
            "current pinned raw OpenArchives JSONL shard family; historical snapshot revision "
            "equivalence remains unverified"
        ),
        document_id_alignment="doc_id_domain_compatible_unverified",
    )
    return [greek, openarchives, kallipos]


def _select_files(
    route: Route,
    lock_row: Mapping[str, Any],
    receipt_row: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    locked = {str(row["path"]): row for row in lock_row["selected_files"]}
    acquired = {str(row["path"]): row for row in receipt_row["files"]}
    selected_paths = sorted(
        path
        for path in locked
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in route.path_patterns)
    )
    if not selected_paths:
        raise RehydrationError(
            f"{route.logical_source}: route selected no {route.acquisition_source_id} artifacts"
        )
    if route.logical_source in {"greek_phd", "kallipos"} and len(selected_paths) != len(
        route.path_patterns
    ):
        raise RehydrationError(
            f"{route.logical_source}: expected {len(route.path_patterns)} exact artifacts, "
            f"selected {selected_paths}"
        )
    return [(locked[path], acquired[path]) for path in selected_paths]


def _inspect_parquet_route(route: Route, paths: Sequence[Path]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RehydrationError(
            "source-receipt building requires the pinned pyarrow runtime"
        ) from exc
    reports: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    if route.format == "parquet_sections":
        required = set(route.fields.values())
    else:
        required = {str(route.fields["document_id"])}
        required.update(str(item) for item in route.fields["text_precedence"])
        if route.fields.get("row_filter"):
            required.add(str(route.fields["row_filter"]["column"]))
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = sorted(required - columns)
        if missing:
            raise RehydrationError(
                f"{path}: route-required Parquet columns are absent: {missing}"
            )
        if route.format == "parquet_documents":
            id_column = str(route.fields["document_id"])
            for batch in parquet.iter_batches(
                batch_size=32, columns=[id_column], use_threads=False
            ):
                sample_ids.extend(
                    str(value)
                    for value in batch.column(0).to_pylist()
                    if value is not None
                )
                break
        reports.append(
            {
                "path": str(path.resolve()),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "columns": sorted(columns),
            }
        )
    return {
        "format": route.format,
        "files": reports,
        "sample_document_ids": sample_ids[:64],
    }


def _inspect_jsonl_route(route: Route, paths: Sequence[Path]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    id_field = str(route.fields["document_id"])
    text_fields = list(map(str, route.fields["text_precedence"]))
    for path in paths:
        first: dict[str, Any] | None = None
        with _open_jsonl_text(path) as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RehydrationError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise RehydrationError(
                        f"{path}:{line_number}: row must be an object"
                    )
                first = value
                break
        if first is None:
            raise RehydrationError(f"{path}: empty JSONL artifact")
        if not isinstance(first.get(id_field), str) or not any(
            isinstance(first.get(field), str) for field in text_fields
        ):
            raise RehydrationError(
                f"{path}: first row lacks string {id_field!r} or all text precedence fields"
            )
        reports.append(
            {
                "path": str(path.resolve()),
                "first_row_fields": sorted(first),
                "first_document_id_sha256": hashlib.sha256(
                    str(first[id_field]).encode("utf-8")
                ).hexdigest(),
            }
        )
    return {"format": route.format, "files": reports}


def build_span_source_receipt(args: argparse.Namespace) -> dict[str, Any]:
    acquisition_path = Path(args.acquisition_receipt).resolve()
    lock_path = Path(args.source_lock).resolve()
    sources_path = Path(args.sources_config).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    acquisition, locked, acquired, config_rows = _validate_phase04_bindings(
        acquisition_path, lock_path, sources_path
    )
    config = _load_object(sources_path, "full_cpt_sources_v1")
    routes = _routes(args, config)
    if {route.logical_source for route in routes} != REQUIRED_LOGICAL_SOURCES:
        raise RehydrationError(
            "builder did not resolve exactly the three SPAN logical sources"
        )
    logical_sources: dict[str, Any] = {}
    schema_reports: dict[str, Any] = {}
    for route in routes:
        source_id = route.acquisition_source_id
        if (
            source_id not in locked
            or source_id not in acquired
            or source_id not in config_rows
        ):
            raise RehydrationError(
                f"{route.logical_source}: acquisition lacks required source {source_id!r}"
            )
        selected = _select_files(route, locked[source_id], acquired[source_id])
        paths = [
            Path(str(receipt_file["local_path"])).resolve()
            for _, receipt_file in selected
        ]
        report = (
            _inspect_jsonl_route(route, paths)
            if route.format == "jsonl_documents"
            else _inspect_parquet_route(route, paths)
        )
        if route.logical_source == "greek_phd" and route.format == "parquet_documents":
            manifest_ids = _manifest_source_ids(manifest_path, "greek_phd")
            if not all(HEX64_RE.fullmatch(item) for item in manifest_ids):
                raise RehydrationError(
                    "tracked Greek PhD manifest IDs are not the expected hash domain"
                )
            samples = report.get("sample_document_ids", [])
            sample_hash_compatible = bool(samples) and all(
                HEX64_RE.fullmatch(item) for item in samples
            )
            if (
                route.acquisition_source_id == "nanochat_base"
                and not sample_hash_compatible
            ):
                raise RehydrationError(
                    "Nanochat Greek PhD source_doc_id samples are not hash-domain compatible"
                )
            report["manifest_document_id_count"] = len(manifest_ids)
            report["manifest_document_ids_sha256"] = hashlib.sha256(
                json.dumps(sorted(manifest_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            report["sample_id_domain_hash_compatible"] = sample_hash_compatible
        source_row = acquired[source_id]
        artifacts = []
        for locked_file, receipt_file in selected:
            lfs_sha = locked_file.get("lfs_sha256")
            if not isinstance(lfs_sha, str) or not HEX64_RE.fullmatch(lfs_sha):
                raise RehydrationError(
                    f"{source_id}:{locked_file.get('path')}: selected SPAN artifact lacks LFS SHA-256"
                )
            artifacts.append(
                {
                    "path": str(Path(str(receipt_file["local_path"])).resolve()),
                    "repository_path": str(locked_file["path"]),
                    "sha256": lfs_sha,
                    "bytes": int(locked_file["size"]),
                    "acquisition_hash_kind": "lfs_sha256",
                }
            )
        logical_sources[route.logical_source] = {
            "repo_type": str(source_row["repo_type"]),
            "repo_id": str(source_row["repo_id"]),
            "revision": str(source_row["revision"]),
            "format": route.format,
            "fields": dict(route.fields),
            "acquisition_source_id": source_id,
            "selection_globs": list(route.path_patterns),
            "historical_source_relation": route.historical_source_relation,
            "label_text_equivalence": "unverified_without_expected_snapshot_artifact_sha256",
            "document_id_alignment": route.document_id_alignment,
            "artifacts": artifacts,
        }
        schema_reports[route.logical_source] = report
    receipt = {
        "schema_version": "span-source-artifacts-v1",
        "snapshot_equivalence_status": "rehydrated_unverified_snapshot",
        "labels_read_created_or_inferred": False,
        "derivation": {
            "schema_version": "span-source-artifact-derivation-v1",
            "acquisition_receipt": str(acquisition_path),
            "acquisition_receipt_sha256": sha256_file(acquisition_path),
            "source_lock": str(lock_path),
            "source_lock_sha256": sha256_file(lock_path),
            "sources_config": str(sources_path),
            "sources_config_sha256": sha256_file(sources_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "acquisition_code_commit": acquisition.get("code_commit"),
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "execution_code_commit": os.environ.get("PHASE04_EXPECTED_COMMIT"),
            },
            "route_choices": {
                "greek_phd": args.greek_phd_route,
                "openarchives": "openarchives_current",
                "kallipos": args.kallipos_route,
            },
            "schema_reports": schema_reports,
        },
        "sources": logical_sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-receipt", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--sources-config", default=str(DEFAULT_SOURCES))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--greek-phd-route",
        required=True,
        choices=("nanochat_base", "greek_phd_v2"),
        help="explicitly choose the historical candidate; newer v2 is never preferred implicitly",
    )
    parser.add_argument(
        "--kallipos-route",
        required=True,
        choices=("kallipos_sections", "nanochat_base"),
    )
    parser.add_argument("--greek-phd-document-id-column")
    parser.add_argument("--greek-phd-text-column", action="append")
    parser.add_argument("--allow-unverified-greek-phd-id-domain", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build_span_source_receipt(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "sources": sorted(receipt["sources"]),
                "snapshot_equivalence_status": receipt["snapshot_equivalence_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
