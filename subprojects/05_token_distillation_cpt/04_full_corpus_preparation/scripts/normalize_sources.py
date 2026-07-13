#!/usr/bin/env python3
"""Normalize receipt-bound Phase-04 sources into resumable canonical shards."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from full_corpus_io import (
    SourceArtifact,
    artifact_relative_path,
    artifacts_from_receipt,
    base_family_map,
    canonical_row,
    canonical_schema,
    expand_nested_row,
    iter_artifact_rows,
    iter_grouped_section_rows,
    sha256_file,
    sha256_text,
    source_config_map,
)
from source_lineage import canonical_json, load_json
from validate_agent1_v3_candidate_roster import (
    validate_observed_extraction_route,
    validate_roster as validate_route_basis_roster,
)


NORMALIZATION_MANIFEST_SCHEMA = "full_cpt_normalization_manifest_v1"
SOURCE_RECEIPT_SCHEMA = "full_cpt_normalization_source_receipt_v1"
FILE_RECEIPT_SCHEMA = "full_cpt_normalization_file_receipt_v1"
SHARD_RECEIPT_SCHEMA = "full_cpt_normalization_shard_receipt_v1"
UNIQUENESS_RECEIPT_SCHEMA = "full_cpt_normalization_uid_uniqueness_v1"
DEFAULT_LARGE_TASK_BYTE_THRESHOLD = 2 * 1024**3
DEFAULT_LARGE_TASK_WORKERS = 2
V3_CANDIDATE_ROSTER_SCHEMA = "agent1_full_corpus_v3_candidate_roster_v1"
V3_ROUTE_FIELDS = ("source_route", "review_route", "extraction_route")
V3_OBSERVED_EXTRACTION_FIELDS = (
    "observed_extraction_route",
    "observed_extraction_route_basis",
    "observed_extraction_route_evidence",
    "observed_extraction_route_priority",
)
V3_ALLOWED_ROUTES = frozenset({"html_web", "pdf_ocr", "mixed", "structured"})
V3_OBSERVED_EXTRACTION_BASES = frozenset(
    {
        "explicit_row_route",
        "row_representation_metadata",
        "declared_extraction_route_fallback",
        "unavailable",
    }
)
V3_OBSERVED_EXTRACTION_PRIORITIES = frozenset(
    {"logical_primary", "secondary_exception_only"}
)


def write_json_atomic(
    path: Path, value: dict[str, Any], *, immutable: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def file_binding(path: Path) -> dict[str, Any]:
    """Return the immutable binding used for a v3 candidate-roster input."""

    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_v3_route_map(
    *,
    roster_path: Path,
    field: str,
    value: object,
    candidates: list[str],
    fallback: dict[str, str] | None = None,
) -> dict[str, str]:
    """Validate one exact candidate-to-route map from the frozen roster."""

    if value is None:
        if fallback is None:
            raise ValueError(f"{roster_path}: missing required {field}")
        return dict(fallback)
    if not isinstance(value, dict):
        raise ValueError(f"{roster_path}: {field} must be an object")
    candidate_set = set(candidates)
    keys = set(value)
    missing = sorted(candidate_set - keys)
    extra = sorted(keys - candidate_set)
    if missing or extra:
        raise ValueError(
            f"{roster_path}: {field} coverage drift; missing={missing}, extra={extra}"
        )
    result: dict[str, str] = {}
    for source_id in candidates:
        route = value[source_id]
        if not isinstance(route, str) or route not in V3_ALLOWED_ROUTES:
            raise ValueError(
                f"{roster_path}: unsupported {field}[{source_id!r}]={route!r}"
            )
        result[source_id] = route
    return result


def _observed_route_allowances(
    roster: Mapping[str, Any],
    *,
    candidates: list[str],
    source_routes: Mapping[str, str],
    extraction_routes: Mapping[str, str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Return per-source observed-route allowances from frozen route basis.

    V3 production rosters must carry the route-basis declaration.  A narrow
    compatibility fallback supports historical unit fixtures without treating
    a source-level extraction fallback as an unbounded document-level route.
    """

    if roster.get("route_basis") is None:
        return (
            {
                source_id: sorted({source_routes[source_id], extraction_routes[source_id]})
                for source_id in candidates
            },
            {
                source_id: {
                    "logical_acquisition_type": source_routes[source_id],
                    "declared_extraction_route_fallback": extraction_routes[source_id],
                    "allowed_observed_extraction_routes": sorted(
                        {source_routes[source_id], extraction_routes[source_id]}
                    ),
                    "observed_route_priorities": {
                        route: (
                            "logical_primary"
                            if route == source_routes[source_id]
                            else "secondary_exception_only"
                        )
                        for route in sorted(
                            {source_routes[source_id], extraction_routes[source_id]}
                        )
                    },
                }
                for source_id in candidates
            },
        )
    report = validate_route_basis_roster(roster)
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("candidate roster route-basis validation produced no source map")
    allowances: dict[str, list[str]] = {}
    route_metadata: dict[str, dict[str, Any]] = {}
    for source_id in candidates:
        entry = sources.get(source_id)
        if not isinstance(entry, Mapping):
            raise ValueError(f"candidate roster route-basis lacks {source_id!r}")
        allowed = entry.get("allowed_observed_extraction_routes")
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(not isinstance(route, str) or route not in V3_ALLOWED_ROUTES for route in allowed)
        ):
            raise ValueError(f"candidate roster route-basis has invalid observed-route allowance for {source_id!r}")
        # Exercise the validator itself for all declared allowances, so this
        # consumer cannot accidentally reinterpret an exception as primary.
        priorities = {
            route: validate_observed_extraction_route(
                report, source_id=source_id, observed_extraction_route=route
            )["observed_route_priority"]
            for route in allowed
        }
        allowances[source_id] = sorted(allowed)
        route_metadata[source_id] = {
            "logical_acquisition_type": str(entry["logical_acquisition_type"]),
            "declared_extraction_route_fallback": str(entry["declared_extraction_route_fallback"]),
            "allowed_observed_extraction_routes": sorted(allowed),
            "observed_route_priorities": priorities,
        }
    return allowances, route_metadata


def load_v3_candidate_roster(path: Path) -> dict[str, Any]:
    """Load the route declaration that v3 canonicalization must preserve.

    The frozen v3 roster declares ``source_routes`` (logical provenance),
    ``review_routes`` (review policy), and ``extraction_routes`` (the
    source-level declared observed-route fallback).  Every canonical row then
    records its own observed extraction route, basis, and evidence.  Legacy
    fixtures may omit the latter two distinctions, in which case they default
    to the review route.  This preserves the fact that a Parquet transport can
    still contain text logically sourced from a PDF/OCR extraction or an HTML
    scrape.
    """

    roster_path = path.resolve()
    roster = read_json(roster_path)
    if roster.get("schema_version") != V3_CANDIDATE_ROSTER_SCHEMA:
        raise ValueError(
            f"{roster_path}: unsupported candidate roster schema "
            f"{roster.get('schema_version')!r}"
        )
    base_source_id = roster.get("base_source_id")
    if not isinstance(base_source_id, str) or not base_source_id:
        raise ValueError(f"{roster_path}: base_source_id must be a non-empty string")
    if base_source_id != "nanochat_base":
        raise ValueError(f"{roster_path}: expected base_source_id 'nanochat_base'")
    candidates = roster.get("candidate_source_ids")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(source, str) or not source for source in candidates)
        or len(candidates) != len(set(candidates))
        or base_source_id in candidates
    ):
        raise ValueError(
            f"{roster_path}: candidate_source_ids must be unique non-empty strings "
            "and exclude base_source_id"
        )
    candidate_ids = list(candidates)
    review_routes = _validate_v3_route_map(
        roster_path=roster_path,
        field="review_routes",
        value=roster.get("review_routes"),
        candidates=candidate_ids,
    )
    source_routes = _validate_v3_route_map(
        roster_path=roster_path,
        field="source_routes",
        value=roster.get("source_routes"),
        candidates=candidate_ids,
        fallback=review_routes,
    )
    extraction_routes = _validate_v3_route_map(
        roster_path=roster_path,
        field="extraction_routes",
        value=roster.get("extraction_routes"),
        candidates=candidate_ids,
        fallback=review_routes,
    )
    observed_route_allowances, route_basis_metadata = _observed_route_allowances(
        roster,
        candidates=candidate_ids,
        source_routes=source_routes,
        extraction_routes=extraction_routes,
    )
    binding = file_binding(roster_path)
    return {
        **binding,
        "schema_version": V3_CANDIDATE_ROSTER_SCHEMA,
        "base_source_id": base_source_id,
        "candidate_source_ids": candidate_ids,
        "review_routes": review_routes,
        "source_routes": source_routes,
        "extraction_routes": extraction_routes,
        "allowed_observed_extraction_routes": observed_route_allowances,
        "route_basis_metadata": route_basis_metadata,
        "route_declarations": {
            source_id: {
                "source_route": source_routes[source_id],
                "review_route": review_routes[source_id],
                "extraction_route": extraction_routes[source_id],
                **route_basis_metadata[source_id],
            }
            for source_id in candidate_ids
        },
    }


def validate_v3_candidate_roster_source_coverage(
    *,
    roster_binding: dict[str, Any],
    source_registry: dict[str, Any],
    artifacts: list[SourceArtifact],
) -> dict[str, Any]:
    """Fail closed if frozen v3 candidates drift from registry or receipt.

    Candidate omission is unsafe because it could make a later review packet
    look complete even though its original source never reached canonical
    normalization.  Extra normalizable sources are unsafe for the converse
    reason: they would enter the pool without an assigned review route.
    """

    base_source_id = str(roster_binding["base_source_id"])
    candidates = [str(source) for source in roster_binding["candidate_source_ids"]]
    expected_ids = {base_source_id, *candidates}
    registry_by_source = source_config_map(source_registry)
    configured_ids = set(registry_by_source)
    missing_registry = sorted(expected_ids - configured_ids)
    extra_registry = sorted(configured_ids - expected_ids)
    if missing_registry or extra_registry:
        raise ValueError(
            "v3 candidate roster/source registry coverage drift; "
            f"missing={missing_registry}, extra={extra_registry}"
        )
    if str(registry_by_source[base_source_id].get("role", "base")) != "base":
        raise ValueError(
            f"v3 base source {base_source_id!r} must retain role='base' in sources registry"
        )
    invalid_candidate_roles = {
        source_id: registry_by_source[source_id].get("role")
        for source_id in candidates
        if str(registry_by_source[source_id].get("role", "")) in {"", "base", "base_overlay"}
    }
    if invalid_candidate_roles:
        raise ValueError(
            "v3 candidate roster includes non-candidate source registry entries: "
            f"{invalid_candidate_roles}"
        )
    artifact_ids = [artifact.source_id for artifact in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("normalization artifacts repeat a source identity")
    actual_artifact_ids = set(artifact_ids)
    missing_artifacts = sorted(expected_ids - actual_artifact_ids)
    extra_artifacts = sorted(actual_artifact_ids - expected_ids)
    if missing_artifacts or extra_artifacts:
        raise ValueError(
            "v3 candidate roster/acquisition coverage drift; "
            f"missing={missing_artifacts}, extra={extra_artifacts}"
        )
    return {
        "schema_version": "agent1_v3_normalization_source_coverage_v1",
        "status": "passed",
        "candidate_roster": {
            key: roster_binding[key]
            for key in ("path", "bytes", "sha256", "schema_version", "base_source_id")
        },
        "candidate_source_ids": candidates,
        "normalizable_registry_source_ids": sorted(configured_ids),
        "acquisition_artifact_source_ids": sorted(actual_artifact_ids),
    }


def file_receipt_path(output: Path, source_id: str, file_index: int) -> Path:
    return output / ".receipts" / "files" / source_id / f"file-{file_index:05d}.json"


def source_receipt_path(output: Path, source_id: str) -> Path:
    return output / ".receipts" / "sources" / f"{source_id}.json"


def receipt_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_inventory_entry(entry: dict[str, Any], *, context: str) -> Path:
    path = Path(str(entry.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(entry.get("bytes", -1))
        or sha256_file(path) != str(entry.get("sha256", ""))
    ):
        raise ValueError(f"{context}: immutable inventory drift for {path}")
    return path


def shard_directory(
    output: Path, source_id: str, file_index: int, shard_index: int
) -> Path:
    return output / source_id / f"shard-{file_index:05d}-{shard_index:05d}"


def validate_shard_receipt(
    directory: Path,
    *,
    contract_sha256: str,
    source_id: str,
    file_index: int,
    shard_index: int,
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    if not directory.is_dir() or not receipt_path.is_file():
        raise ValueError(f"incomplete normalized shard directory: {directory}")
    receipt = read_json(receipt_path)
    expected = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "source_id": source_id,
        "file_index": file_index,
        "shard_index": shard_index,
        "input": input_binding,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{receipt_path}: resume contract drift for {key}")
    data = receipt.get("output")
    if not isinstance(data, dict):
        raise ValueError(f"{receipt_path}: missing output inventory")
    validate_inventory_entry(data, context=str(receipt_path))
    if Path(str(data["path"])).resolve().parent != directory.resolve():
        raise ValueError(
            f"{receipt_path}: output path escapes its immutable shard directory"
        )
    start = int(receipt.get("unit_start", -1))
    end = int(receipt.get("unit_end_exclusive", -1))
    if start < 0 or end <= start or int(data.get("rows", 0)) < 1:
        raise ValueError(f"{receipt_path}: invalid unit or row range")
    result = dict(receipt)
    result["receipt"] = receipt_entry(receipt_path)
    return result


class ShardWriter:
    """Commit each shard as one atomic directory containing data and receipt."""

    def __init__(
        self,
        *,
        output: Path,
        source_id: str,
        file_index: int,
        rows_per_shard: int,
        contract_sha256: str,
        input_binding: dict[str, Any],
    ) -> None:
        self.output = output
        self.source_id = source_id
        self.file_index = file_index
        self.rows_per_shard = rows_per_shard
        self.contract_sha256 = contract_sha256
        self.input_binding = input_binding
        self.schema = canonical_schema()
        self.rows: list[dict[str, Any]] = []
        self.pending_counts: Counter[str] = Counter()
        self.pending_source_names: Counter[str] = Counter()
        self.pending_unit_start: int | None = None
        self.pending_unit_end = 0
        self.shards = self._load_existing()
        self.shard_index = len(self.shards)
        self.resume_unit = self.shards[-1]["unit_end_exclusive"] if self.shards else 0
        self.total_counts: Counter[str] = Counter()
        self.total_source_names: Counter[str] = Counter()
        for shard in self.shards:
            self.total_counts.update(shard["counts"])
            self.total_source_names.update(shard["exact_source_dataset_counts"])

    def _load_existing(self) -> list[dict[str, Any]]:
        root = self.output / self.source_id
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"shard-{self.file_index:05d}-"
        for partial in root.glob(f".{prefix}*.partial-*"):
            if partial.is_dir():
                shutil.rmtree(partial)
            else:
                partial.unlink()
        directories = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
        result: list[dict[str, Any]] = []
        expected_unit = 0
        for shard_index, directory in enumerate(directories):
            expected_name = f"{prefix}{shard_index:05d}"
            if directory.name != expected_name:
                raise ValueError(
                    f"{self.source_id}: non-contiguous resumed shard {directory.name}; "
                    f"expected {expected_name}"
                )
            receipt = validate_shard_receipt(
                directory,
                contract_sha256=self.contract_sha256,
                source_id=self.source_id,
                file_index=self.file_index,
                shard_index=shard_index,
                input_binding=self.input_binding,
            )
            if int(receipt["unit_start"]) != expected_unit:
                raise ValueError(f"{directory}: non-contiguous normalized unit range")
            expected_unit = int(receipt["unit_end_exclusive"])
            result.append(receipt)
        return result

    @property
    def documents_emitted(self) -> int:
        return int(
            self.total_counts["documents_emitted"]
            + self.pending_counts["documents_emitted"]
        )

    def add_unit(
        self,
        unit_index: int,
        rows: list[dict[str, Any]],
        counts: Counter[str],
        source_names: Counter[str],
    ) -> None:
        if self.pending_unit_start is None:
            self.pending_unit_start = unit_index
        if unit_index != self.pending_unit_end and self.pending_unit_end != 0:
            raise ValueError(
                "normalization units were not presented in deterministic order"
            )
        self.pending_unit_end = unit_index + 1
        self.rows.extend(rows)
        self.pending_counts.update(counts)
        self.pending_source_names.update(source_names)
        if len(self.rows) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        assert self.pending_unit_start is not None
        final = shard_directory(
            self.output, self.source_id, self.file_index, self.shard_index
        )
        if final.exists():
            raise FileExistsError(f"refusing to overwrite immutable shard: {final}")
        temporary = final.parent / f".{final.name}.partial-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        data_name = f"part-{self.file_index:05d}-{self.shard_index:05d}.parquet"
        temporary_data = temporary / data_name
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        pq.write_table(
            table,
            temporary_data,
            compression="zstd",
            row_group_size=min(8192, len(self.rows)),
            use_dictionary=True,
        )
        final_data = final / data_name
        output_entry = {
            "path": str(final_data),
            "bytes": temporary_data.stat().st_size,
            "sha256": sha256_file(temporary_data),
            "rows": len(self.rows),
        }
        receipt = {
            "schema_version": SHARD_RECEIPT_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "source_id": self.source_id,
            "file_index": self.file_index,
            "shard_index": self.shard_index,
            "input": self.input_binding,
            "unit_start": self.pending_unit_start,
            "unit_end_exclusive": self.pending_unit_end,
            "counts": dict(sorted(self.pending_counts.items())),
            "exact_source_dataset_counts": dict(
                sorted(self.pending_source_names.items())
            ),
            "output": output_entry,
        }
        write_json_atomic(temporary / "receipt.json", receipt, immutable=True)
        os.replace(temporary, final)
        committed = dict(receipt)
        committed["receipt"] = receipt_entry(final / "receipt.json")
        self.shards.append(committed)
        self.total_counts.update(self.pending_counts)
        self.total_source_names.update(self.pending_source_names)
        self.rows = []
        self.pending_counts = Counter()
        self.pending_source_names = Counter()
        self.pending_unit_start = None
        self.shard_index += 1

    def close(self) -> None:
        self.flush()
        if self.pending_unit_start is not None:
            # A suffix of empty/no-text input units has no Parquet payload to
            # commit, but still belongs in the completed file accounting.
            self.total_counts.update(self.pending_counts)
            self.total_source_names.update(self.pending_source_names)
            self.pending_counts = Counter()
            self.pending_source_names = Counter()
            self.pending_unit_start = None


def source_rows(
    artifact: SourceArtifact,
    path: Path,
    *,
    grouped_sections: bool,
    temporary_root: Path,
) -> Iterable[tuple[int, dict[str, Any], list[tuple[str, str, str]]]]:
    if grouped_sections:
        yield from (
            (row_index, raw_row, [("0", text_field, raw_text)])
            for row_index, raw_row, text_field, raw_text in iter_grouped_section_rows(
                artifact, path, temporary_root=temporary_root
            )
        )
        return
    yield from (
        (row_index, raw_row, list(expand_nested_row(raw_row, artifact)))
        for row_index, raw_row in iter_artifact_rows(path)
    )


def validate_file_receipt(
    path: Path,
    *,
    output: Path,
    artifact: SourceArtifact,
    file_index: int,
    contract_sha256: str,
) -> dict[str, Any]:
    receipt = read_json(path)
    expected = {
        "schema_version": FILE_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "source_id": artifact.source_id,
        "file_index": file_index,
        "input": dict(artifact.file_bindings[file_index]),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{path}: resume contract drift for {key}")
    shards = receipt.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"{path}: shards must be a list")
    for shard_index, declared in enumerate(shards):
        actual = validate_shard_receipt(
            shard_directory(output, artifact.source_id, file_index, shard_index),
            contract_sha256=contract_sha256,
            source_id=artifact.source_id,
            file_index=file_index,
            shard_index=shard_index,
            input_binding=dict(artifact.file_bindings[file_index]),
        )
        for key in ("unit_start", "unit_end_exclusive", "counts", "output"):
            if declared.get(key) != actual.get(key):
                raise ValueError(f"{path}: shard receipt drift for {key}")
        if declared.get("receipt") != actual.get("receipt"):
            raise ValueError(f"{path}: shard sidecar receipt drift")
    result = dict(receipt)
    result["receipt"] = receipt_entry(path)
    return result


def normalize_file(
    *,
    artifact: SourceArtifact,
    file_index: int,
    output: Path,
    temporary_root: Path,
    rows_per_shard: int,
    contract_sha256: str,
    lineage_aliases: dict[str, Any],
    base_families: dict[str, str],
    embedded_structural_routes: list[dict[str, Any]],
    max_documents: int,
    progress_every: int,
    declared_routes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt_path = file_receipt_path(output, artifact.source_id, file_index)
    if receipt_path.exists():
        return validate_file_receipt(
            receipt_path,
            output=output,
            artifact=artifact,
            file_index=file_index,
            contract_sha256=contract_sha256,
        )
    source_path = artifact.files[file_index]
    grouped = "group_sections_to_work" in str(artifact.config.get("merge_policy", ""))
    writer = ShardWriter(
        output=output,
        source_id=artifact.source_id,
        file_index=file_index,
        rows_per_shard=rows_per_shard,
        contract_sha256=contract_sha256,
        input_binding=dict(artifact.file_bindings[file_index]),
    )
    # A resumed shard range is already immutable.  Reiterate only enough input
    # structure to reach the next unit; do not normalize or hash those texts.
    total_units = writer.resume_unit
    trailing_counts: Counter[str] = Counter()
    trailing_source_names: Counter[str] = Counter()
    truncated = False
    iterator = source_rows(
        artifact,
        source_path,
        grouped_sections=grouped,
        temporary_root=temporary_root / artifact.source_id,
    )
    for unit_index, (artifact_row_index, raw_row, representations) in enumerate(
        iterator
    ):
        if unit_index < writer.resume_unit:
            continue
        if max_documents and writer.documents_emitted >= max_documents:
            truncated = True
            break
        counts: Counter[str] = Counter(rows_scanned=1)
        names: Counter[str] = Counter()
        emitted: list[dict[str, Any]] = []
        for suffix, text_field, raw_text in representations:
            if (
                max_documents
                and writer.documents_emitted + len(emitted) >= max_documents
            ):
                truncated = True
                break
            row = canonical_row(
                source=artifact,
                artifact_path=source_path,
                artifact_row_index=artifact_row_index,
                raw_row=raw_row,
                representation_suffix=suffix,
                text_field=text_field,
                raw_text=raw_text,
                lineage_aliases=lineage_aliases,
                base_families=base_families,
                embedded_structural_routes=embedded_structural_routes,
                declared_routes=declared_routes,
            )
            emitted.append(row)
            counts["documents_emitted"] += 1
            counts["characters"] += len(row["text"])
            counts["bytes_utf8"] += len(row["text"].encode("utf-8"))
            counts["empty_documents"] += int(not row["text"])
            names[row["source_dataset"]] += 1
            if row["source_id"] != artifact.source_id:
                counts[f"embedded_route:{row['source_id']}"] += 1
        if not emitted:
            counts["rows_without_text"] += 1
        writer.add_unit(unit_index, emitted, counts, names)
        total_units = unit_index + 1
        if progress_every and total_units % progress_every == 0:
            print(
                f"normalize_sources: source={artifact.source_id} file={file_index} "
                f"units={total_units:,} emitted={writer.documents_emitted:,}",
                flush=True,
            )
        if truncated:
            break
    # Preserve final no-output units in the file-level accounting.  Shards are
    # allowed to cover only ranges that emitted at least one document.
    trailing_counts.update(writer.pending_counts)
    trailing_source_names.update(writer.pending_source_names)
    writer.close()
    counts = Counter(writer.total_counts)
    # writer.close folds pending counters into total_counts; variables above
    # are intentionally retained only to make the no-output contract explicit.
    del trailing_counts, trailing_source_names
    payload = {
        "schema_version": FILE_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "source_id": artifact.source_id,
        "file_index": file_index,
        "input": dict(artifact.file_bindings[file_index]),
        "grouped_sections": grouped,
        "units_scanned": total_units,
        "truncated": truncated,
        "counts": dict(sorted(counts.items())),
        "exact_source_dataset_counts": dict(sorted(writer.total_source_names.items())),
        "shards": [
            {
                key: value
                for key, value in shard.items()
                if key
                in {
                    "unit_start",
                    "unit_end_exclusive",
                    "counts",
                    "exact_source_dataset_counts",
                    "output",
                    "receipt",
                }
            }
            for shard in writer.shards
        ],
    }
    write_json_atomic(receipt_path, payload, immutable=True)
    payload["receipt"] = receipt_entry(receipt_path)
    return payload


def normalize_task(payload: dict[str, Any]) -> list[dict[str, Any]]:
    artifact: SourceArtifact = payload["artifact"]
    results: list[dict[str, Any]] = []
    emitted = 0
    maximum = int(payload["max_documents"])
    for file_index in payload["file_indices"]:
        remaining = max(0, maximum - emitted) if maximum else 0
        if maximum and remaining == 0:
            break
        receipt = normalize_file(
            artifact=artifact,
            file_index=file_index,
            output=payload["output"],
            temporary_root=payload["temporary_root"],
            rows_per_shard=payload["rows_per_shard"],
            contract_sha256=payload["contract_sha256"],
            lineage_aliases=payload["lineage_aliases"],
            base_families=payload["base_families"],
            embedded_structural_routes=payload["embedded_structural_routes"],
            declared_routes=payload.get("declared_routes"),
            max_documents=remaining,
            progress_every=payload["progress_every"],
        )
        results.append(receipt)
        emitted += int(receipt["counts"].get("documents_emitted", 0))
    return results


def normalization_task_input_bytes(payload: dict[str, Any]) -> int:
    """Return receipt-bound input bytes for one independently scheduled task."""

    artifact: SourceArtifact = payload["artifact"]
    return sum(
        int(artifact.file_bindings[file_index]["bytes"])
        for file_index in payload["file_indices"]
    )


def partition_normalization_tasks(
    tasks: list[dict[str, Any]], *, large_task_byte_threshold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stably split ordinary tasks from high-memory, large-input tasks."""

    if large_task_byte_threshold < 1:
        raise ValueError("large-task byte threshold must be positive")
    ordinary: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []
    for task in tasks:
        target = (
            large
            if normalization_task_input_bytes(task) >= large_task_byte_threshold
            else ordinary
        )
        target.append(task)
    return ordinary, large


def collect_task_receipts(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
    pool_name: str,
    file_receipts: dict[str, list[dict[str, Any]]],
) -> None:
    """Run one bounded process pool and collect its immutable file receipts."""

    if not tasks:
        return
    pool_workers = min(workers, len(tasks))
    print(
        f"normalize_sources: starting pool={pool_name} "
        f"tasks={len(tasks)} workers={pool_workers}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=pool_workers) as executor:
        futures = {executor.submit(normalize_task, task): task for task in tasks}
        try:
            for future in as_completed(futures):
                task = futures[future]
                artifact: SourceArtifact = task["artifact"]
                for receipt in future.result():
                    file_receipts[artifact.source_id].append(receipt)
                print(
                    f"normalize_sources: completed task source={artifact.source_id} "
                    f"files={task['file_indices']}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def duckdb_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def validate_candidate_canonical_route_coverage(
    connection: Any,
    *,
    source_relation: str,
    declared_routes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify logical route declarations and per-document extraction evidence.

    This runs inside the existing global DuckDB pass, so it does not require a
    second corpus-scale Parquet scan.  ``acquisition_source_id`` is used as the
    grouping key: a candidate must not silently become an embedded/base route
    or a differently named source before it reaches quality profiling.
    Observed extraction may differ from the declared source fallback, but must
    always have a controlled route plus auditable basis and evidence code.
    """

    if not declared_routes:
        raise ValueError("candidate route coverage requires at least one declaration")
    candidates = sorted(declared_routes)
    route_fields = V3_ROUTE_FIELDS
    allowed_observed_by_source: dict[str, set[str]] = {}
    for source_id in candidates:
        declaration = declared_routes[source_id]
        if not isinstance(declaration, Mapping):
            continue
        allowed = declaration.get("allowed_observed_extraction_routes")
        if not isinstance(allowed, list):
            continue
        allowed_observed_by_source[source_id] = {
            route for route in allowed if isinstance(route, str)
        }
    malformed = {
        source_id: declaration
        for source_id, declaration in declared_routes.items()
        if not isinstance(declaration, Mapping)
        or any(
            not isinstance(declaration.get(field), str)
            or declaration.get(field) not in V3_ALLOWED_ROUTES
            for field in route_fields
        )
        or not allowed_observed_by_source.get(source_id)
        or not allowed_observed_by_source[source_id] <= V3_ALLOWED_ROUTES
        or declaration.get("source_route") not in allowed_observed_by_source[source_id]
    }
    if malformed:
        raise ValueError(f"invalid frozen candidate route declarations: {malformed}")
    candidate_sql = ",".join(duckdb_literal(source_id) for source_id in candidates)
    try:
        observed_rows = connection.execute(
            f"""
            SELECT acquisition_source_id, source_id,
                   source_route, review_route, extraction_route,
                   observed_extraction_route,
                   observed_extraction_route_basis,
                   observed_extraction_route_evidence,
                   observed_extraction_route_priority,
                   COUNT(*) AS documents
            FROM {source_relation}
            WHERE acquisition_source_id IN ({candidate_sql})
            GROUP BY acquisition_source_id, source_id,
                     source_route, review_route, extraction_route,
                     observed_extraction_route,
                     observed_extraction_route_basis,
                     observed_extraction_route_evidence,
                     observed_extraction_route_priority
            ORDER BY acquisition_source_id, source_id,
                     source_route, review_route, extraction_route,
                     observed_extraction_route,
                     observed_extraction_route_basis,
                     observed_extraction_route_evidence,
                     observed_extraction_route_priority
            """
        ).fetchall()
    except Exception as exc:
        raise ValueError(
            "canonical candidate route provenance fields are missing or unreadable; "
            "v3 normalization cannot continue"
        ) from exc

    observed: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in candidates}
    for (
        acquisition_source_id,
        source_id,
        source_route,
        review_route,
        extraction_route,
        observed_extraction_route,
        observed_extraction_route_basis,
        observed_extraction_route_evidence,
        observed_extraction_route_priority,
        documents,
    ) in observed_rows:
        acquisition = str(acquisition_source_id)
        observed.setdefault(acquisition, []).append(
            {
                "canonical_source_id": str(source_id),
                "source_route": source_route,
                "review_route": review_route,
                "extraction_route": extraction_route,
                "observed_extraction_route": observed_extraction_route,
                "observed_extraction_route_basis": observed_extraction_route_basis,
                "observed_extraction_route_evidence": observed_extraction_route_evidence,
                "observed_extraction_route_priority": observed_extraction_route_priority,
                "observed_route_priority": (
                    "logical_primary"
                    if observed_extraction_route
                    == declared_routes[acquisition]["source_route"]
                    else "secondary_exception_only"
                ),
                "documents": int(documents),
            }
        )

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for source_id in candidates:
        expected = dict(declared_routes[source_id])
        rows = observed[source_id]
        documents = sum(int(row["documents"]) for row in rows)
        valid = bool(rows) and all(
            row["canonical_source_id"] == source_id
            and all(row[field] == expected[field] for field in route_fields)
            and row["observed_extraction_route"] in allowed_observed_by_source[source_id]
            and row["observed_extraction_route_basis"] in V3_OBSERVED_EXTRACTION_BASES
            and isinstance(row["observed_extraction_route_evidence"], str)
            and bool(row["observed_extraction_route_evidence"].strip())
            and row["observed_extraction_route_priority"]
            in V3_OBSERVED_EXTRACTION_PRIORITIES
            and row["observed_extraction_route_priority"]
            == row["observed_route_priority"]
            for row in rows
        )
        observed_route_counts = dict(
            sorted(
                (
                    str(route),
                    sum(
                        int(row["documents"])
                        for row in rows
                        if row["observed_extraction_route"] == route
                    ),
                )
                for route in {
                    row["observed_extraction_route"]
                    for row in rows
                    if row["observed_extraction_route"] in V3_ALLOWED_ROUTES
                }
            )
        )
        result = {
            "source_id": source_id,
            **expected,
            "allowed_observed_extraction_routes": sorted(
                allowed_observed_by_source[source_id]
            ),
            "normalized_documents": documents,
            "observed_extraction_route_counts": observed_route_counts,
            "observed": rows,
            "status": "passed" if valid else "failed",
        }
        results.append(result)
        if not valid:
            failures.append(source_id)
    if failures:
        raise ValueError(
            "canonical candidate route provenance drift for "
            f"{failures}; expected frozen roster declarations for every row"
        )
    return {
        "schema_version": "agent1_v3_canonical_route_coverage_v1",
        "status": "passed",
        "candidate_source_ids": candidates,
        "sources": results,
    }


def validate_global_canonical_inventory(
    paths: list[Path],
    *,
    declared_inventory: list[dict[str, Any]],
    output: Path,
    contract_sha256: str,
    memory_limit: str,
    temporary_directory: Path,
    threads: int,
    candidate_route_declarations: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    import duckdb

    if not paths:
        raise ValueError("normalization emitted no canonical Parquet shards")
    temporary_directory.mkdir(parents=True, exist_ok=True)
    inventory = sorted(
        [
            {
                "path": str(Path(str(row["path"])).resolve()),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in declared_inventory
        ],
        key=lambda row: row["path"],
    )
    if {row["path"] for row in inventory} != {str(path.resolve()) for path in paths}:
        raise ValueError(
            "declared canonical inventory path set differs from shard receipts"
        )
    for row in inventory:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(
                f"canonical shard size drift before global UID pass: {path}"
            )
    inventory_sha256 = sha256_text(canonical_json(inventory))
    receipt_path = output / ".receipts" / "uid_uniqueness.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema_version") != UNIQUENESS_RECEIPT_SCHEMA
            or receipt.get("contract_sha256") != contract_sha256
            or receipt.get("canonical_inventory_sha256") != inventory_sha256
        ):
            raise ValueError(f"{receipt_path}: uniqueness resume contract drift")
        if candidate_route_declarations is not None:
            route_coverage = receipt.get("candidate_roster_route_coverage")
            if (
                not isinstance(route_coverage, dict)
                or route_coverage.get("schema_version")
                != "agent1_v3_canonical_route_coverage_v1"
                or route_coverage.get("status") != "passed"
            ):
                raise ValueError(
                    f"{receipt_path}: missing passed v3 candidate route coverage"
                )
        result = dict(receipt)
        result["receipt"] = receipt_entry(receipt_path)
        return dict(receipt["work_statistics"]), result

    sql_paths = (
        "["
        + ",".join(duckdb_literal(str(path.resolve())) for path in sorted(paths))
        + "]"
    )
    candidate_route_coverage: dict[str, Any] | None = None
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit={duckdb_literal(memory_limit)}")
        connection.execute(
            f"SET temp_directory={duckdb_literal(str(temporary_directory.resolve()))}"
        )
        connection.execute(f"SET threads={max(1, threads)}")
        connection.execute("SET preserve_insertion_order=false")
        source = f"read_parquet({sql_paths}, union_by_name=true)"
        total_rows, unique_uids = connection.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT stable_uid) FROM {source}"
        ).fetchone()
        if int(total_rows) != int(unique_uids):
            duplicate = connection.execute(
                f"""
                SELECT stable_uid, COUNT(*) AS members
                FROM {source}
                GROUP BY stable_uid HAVING COUNT(*) > 1
                ORDER BY stable_uid LIMIT 1
                """
            ).fetchone()
            raise ValueError(
                f"duplicate stable_uid after spillable global pass: {duplicate}"
            )
        rows = connection.execute(
            f"""
            SELECT source_id, COUNT(*) AS unique_work_keys,
                   SUM(CASE WHEN members > 1 THEN 1 ELSE 0 END) AS multi_work_keys
            FROM (
                SELECT source_id, work_key, COUNT(*) AS members
                FROM {source} GROUP BY source_id, work_key
            ) grouped
            GROUP BY source_id ORDER BY source_id
            """
        ).fetchall()
        if candidate_route_declarations is not None:
            candidate_route_coverage = validate_candidate_canonical_route_coverage(
                connection,
                source_relation=source,
                declared_routes=candidate_route_declarations,
            )
    finally:
        connection.close()
    work_stats = {
        str(source_id): {
            "unique_work_keys": int(unique_work_keys),
            "multi_representation_work_keys": int(multi_work_keys or 0),
        }
        for source_id, unique_work_keys, multi_work_keys in rows
    }
    receipt = {
        "schema_version": UNIQUENESS_RECEIPT_SCHEMA,
        "contract_sha256": contract_sha256,
        "canonical_inventory_sha256": inventory_sha256,
        "canonical_shards": len(inventory),
        "canonical_bytes": sum(item["bytes"] for item in inventory),
        "rows_checked": int(total_rows),
        "unique_stable_uids": int(unique_uids),
        "duplicates": 0,
        "engine": "duckdb_exact_distinct_spillable",
        "work_statistics": work_stats,
    }
    if candidate_route_coverage is not None:
        receipt["candidate_roster_route_coverage"] = candidate_route_coverage
    write_json_atomic(receipt_path, receipt, immutable=True)
    receipt["receipt"] = receipt_entry(receipt_path)
    return work_stats, receipt


def build_contract(
    args: argparse.Namespace,
    artifacts: list[SourceArtifact],
    embedded_structural_routes: list[dict[str, Any]],
    candidate_roster: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    value = {
        "schema_version": (
            "full_cpt_normalization_contract_v3"
            if candidate_roster is not None
            else "full_cpt_normalization_contract_v2"
        ),
        "sources_config_sha256": sha256_file(args.sources),
        "lineage_aliases_sha256": sha256_file(args.lineage_aliases),
        "acquisition_receipt_sha256": sha256_file(args.acquisition_receipt),
        "rows_per_shard": args.rows_per_shard,
        "max_rows_per_source": args.max_rows_per_source,
        "embedded_structural_routes": embedded_structural_routes,
        # Bind the canonical schema-affecting provenance extension into the
        # immutable resume contract.  A pre-observation shard must never be
        # accepted as a resumed v3 run merely because the source inputs match.
        "canonical_observed_extraction_contract": {
            "schema_version": "agent1_v3_per_document_observed_extraction_v2",
            "fields": list(V3_OBSERVED_EXTRACTION_FIELDS),
            "allowed_routes": sorted(V3_ALLOWED_ROUTES),
            "allowed_bases": sorted(V3_OBSERVED_EXTRACTION_BASES),
            "allowed_priorities": sorted(V3_OBSERVED_EXTRACTION_PRIORITIES),
            "logical_source_route_primary": True,
        },
        "sources": [
            {
                "source_id": artifact.source_id,
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "role": artifact.role,
                "source_family_id": artifact.source_family_id,
                "config": artifact.config,
                "files": list(artifact.file_bindings),
            }
            for artifact in artifacts
        ],
    }
    if candidate_roster is not None:
        value["candidate_roster"] = {
            key: candidate_roster[key]
            for key in (
                "path",
                "bytes",
                "sha256",
                "schema_version",
                "base_source_id",
                "candidate_source_ids",
                "review_routes",
                "source_routes",
                "extraction_routes",
                "allowed_observed_extraction_routes",
                "route_basis_metadata",
                "route_declarations",
            )
        }
    return value, sha256_text(canonical_json(value))


def validate_embedded_route_coverage(
    routes: list[dict[str, Any]],
    artifacts: list[SourceArtifact],
    summaries: list[dict[str, Any]],
    *,
    bounded_smoke: bool,
) -> dict[str, Any]:
    """Require positive routed rows for every in-scope configured route.

    A source-selected run evaluates only routes owned by selected acquisition
    sources. A bounded smoke may stop before a later matching artifact, so it
    records (but does not pretend to satisfy) the row-count postcondition.
    Artifact-glob coverage is deterministic and is enforced in both modes.
    """

    artifacts_by_source = {artifact.source_id: artifact for artifact in artifacts}
    summaries_by_source = {
        str(summary["source_id"]): summary for summary in summaries
    }
    results: list[dict[str, Any]] = []
    for route in routes:
        route_id = str(route.get("source_id", ""))
        acquisition_source_id = str(route.get("acquisition_source_id", ""))
        coverage = route.get("coverage_contract")
        if not isinstance(coverage, dict):
            raise ValueError(
                f"embedded route {route_id!r} lacks a coverage_contract; "
                "run scripts/validate_configs.py"
            )
        minimum_rows = coverage.get("minimum_normalized_rows")
        if (
            not isinstance(minimum_rows, int)
            or isinstance(minimum_rows, bool)
            or minimum_rows < 1
        ):
            raise ValueError(
                f"embedded route {route_id!r} has invalid minimum_normalized_rows"
            )
        artifact = artifacts_by_source.get(acquisition_source_id)
        if artifact is None:
            results.append(
                {
                    "source_id": route_id,
                    "acquisition_source_id": acquisition_source_id,
                    "expected_source_dataset": coverage.get(
                        "expected_source_dataset"
                    ),
                    "minimum_normalized_rows": minimum_rows,
                    "matched_acquisition_artifacts": [],
                    "normalized_rows": 0,
                    "postcondition_enforced": False,
                    "status": "not_selected",
                }
            )
            continue
        globs = [str(pattern) for pattern in route.get("acquisition_include_globs", [])]
        available_artifacts = [
            artifact_relative_path(artifact, path) for path in artifact.files
        ]
        matched_artifacts = sorted(
            relative
            for relative in available_artifacts
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in globs)
        )
        if not matched_artifacts:
            raise ValueError(
                f"embedded route {route_id!r} matched no acquired artifacts for "
                f"{acquisition_source_id!r}; globs={globs}, "
                f"available_artifacts={available_artifacts}. Fix sources.json or "
                "the acquisition inventory before normalization."
            )
        summary = summaries_by_source.get(acquisition_source_id, {})
        counts = summary.get("counts", {})
        normalized_rows = int(counts.get(f"embedded_route:{route_id}", 0))
        enforced = not bounded_smoke
        if enforced and normalized_rows < minimum_rows:
            raise ValueError(
                f"embedded route {route_id!r} matched acquired artifacts "
                f"{matched_artifacts} but routed {normalized_rows} rows; required at "
                f"least {minimum_rows}. Inspect source_column={route.get('source_column')!r}, "
                f"source_regex={route.get('source_regex')!r}, and the upstream "
                "source_dataset values."
            )
        status = (
            "passed"
            if enforced
            else "observed_positive_bounded_smoke"
            if normalized_rows >= minimum_rows
            else "not_enforced_bounded_smoke"
        )
        results.append(
            {
                "source_id": route_id,
                "acquisition_source_id": acquisition_source_id,
                "expected_source_dataset": coverage.get("expected_source_dataset"),
                "minimum_normalized_rows": minimum_rows,
                "matched_acquisition_artifacts": matched_artifacts,
                "normalized_rows": normalized_rows,
                "postcondition_enforced": enforced,
                "status": status,
            }
        )
    return {
        "schema_version": "full_cpt_embedded_route_coverage_v1",
        "bounded_smoke": bounded_smoke,
        "all_enforced_routes_passed": all(
            row["status"] in {"passed", "not_selected"} for row in results
        )
        if not bounded_smoke
        else None,
        "routes": results,
    }


def validate_selected_source_coverage(
    artifacts: list[SourceArtifact],
    summaries: list[dict[str, Any]],
    *,
    bounded_smoke: bool,
) -> dict[str, Any]:
    """Fail closed when a production-selected text source silently emits no documents."""

    summaries_by_source = {
        str(summary["source_id"]): summary for summary in summaries
    }
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for artifact in artifacts:
        configured_text = sorted(
            {
                str(value)
                for value in (
                    list(artifact.config.get("text_columns", []))
                    + list(artifact.config.get("alternate_text_columns", []))
                )
                if str(value)
            }
        )
        if not configured_text:
            continue
        emitted = int(
            summaries_by_source.get(artifact.source_id, {})
            .get("counts", {})
            .get("documents_emitted", 0)
        )
        enforced = not bounded_smoke
        status = (
            "passed"
            if emitted > 0
            else "not_enforced_bounded_smoke"
            if bounded_smoke
            else "failed_zero_documents"
        )
        rows.append(
            {
                "source_id": artifact.source_id,
                "candidate_text_columns": configured_text,
                "documents_emitted": emitted,
                "postcondition_enforced": enforced,
                "status": status,
            }
        )
        if enforced and emitted < 1:
            failures.append(artifact.source_id)
    if failures:
        raise ValueError(
            "production normalization emitted zero documents for selected text-bearing "
            f"sources: {sorted(failures)}"
        )
    return {
        "schema_version": "full_cpt_selected_source_coverage_v1",
        "bounded_smoke": bounded_smoke,
        "all_enforced_sources_passed": not failures if not bounded_smoke else None,
        "sources": rows,
    }


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources", type=Path, default=here / "configs" / "sources.json"
    )
    parser.add_argument(
        "--lineage-aliases",
        type=Path,
        default=here / "configs" / "source_lineage_aliases.json",
    )
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument(
        "--candidate-roster",
        type=Path,
        help=(
            "immutable Agent 1 v3 roster; binds canonical source/review/extraction "
            "routes and requires exact base/candidate source coverage"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--large-task-byte-threshold",
        type=int,
        default=DEFAULT_LARGE_TASK_BYTE_THRESHOLD,
        help=(
            "receipt-bound bytes at or above which a file task uses the "
            "separate high-memory pool"
        ),
    )
    parser.add_argument(
        "--large-task-workers",
        type=int,
        default=DEFAULT_LARGE_TASK_WORKERS,
        help="maximum workers in the separate high-memory task pool",
    )
    parser.add_argument("--temporary-directory", type=Path)
    parser.add_argument("--duckdb-memory-limit", default="32GB")
    parser.add_argument(
        "--duckdb-threads", type=int, default=min(32, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    if args.rows_per_shard < 1:
        parser.error("--rows-per-shard must be positive")
    if (
        args.workers < 1
        or args.large_task_workers < 1
        or args.large_task_byte_threshold < 1
        or args.duckdb_threads < 1
    ):
        parser.error("worker, byte-threshold and DuckDB settings must be positive")
    if args.manifest.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable manifest: {args.manifest}"
        )
    selected = set(args.source or []) or None
    source_registry = load_json(args.sources)
    candidate_roster = (
        load_v3_candidate_roster(args.candidate_roster)
        if args.candidate_roster is not None
        else None
    )
    artifacts = artifacts_from_receipt(args.sources, args.acquisition_receipt, selected)
    candidate_roster_source_coverage = (
        validate_v3_candidate_roster_source_coverage(
            roster_binding=candidate_roster,
            source_registry=source_registry,
            artifacts=artifacts,
        )
        if candidate_roster is not None
        else None
    )
    lineage_aliases = load_json(args.lineage_aliases)
    base_families = base_family_map(source_registry, lineage_aliases)
    args.output.mkdir(parents=True, exist_ok=True)
    temporary_root = (
        args.temporary_directory.resolve()
        if args.temporary_directory
        else (args.output / ".work").resolve()
    )
    temporary_root.mkdir(parents=True, exist_ok=True)
    embedded_structural_routes = [
        dict(route) for route in source_registry.get("embedded_structural_routes", [])
    ]
    contract, contract_sha256 = build_contract(
        args,
        artifacts,
        embedded_structural_routes,
        candidate_roster=candidate_roster,
    )
    contract_path = args.output / ".receipts" / "normalization_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError(f"{contract_path}: normalization resume contract drift")
    else:
        write_json_atomic(contract_path, contract, immutable=True)

    common = {
        "output": args.output.resolve(),
        "temporary_root": temporary_root,
        "rows_per_shard": args.rows_per_shard,
        "contract_sha256": contract_sha256,
        "lineage_aliases": lineage_aliases,
        "base_families": base_families,
        "embedded_structural_routes": embedded_structural_routes,
        "declared_routes": (
            candidate_roster["route_declarations"]
            if candidate_roster is not None
            else None
        ),
        "max_documents": args.max_rows_per_source,
        "progress_every": args.progress_every,
    }
    tasks: list[dict[str, Any]] = []
    for artifact in artifacts:
        file_indices = list(range(len(artifact.files)))
        grouped = "group_sections_to_work" in str(
            artifact.config.get("merge_policy", "")
        )
        if grouped or args.max_rows_per_source:
            tasks.append({**common, "artifact": artifact, "file_indices": file_indices})
        else:
            tasks.extend(
                {**common, "artifact": artifact, "file_indices": [file_index]}
                for file_index in file_indices
            )

    ordinary_tasks, large_tasks = partition_normalization_tasks(
        tasks, large_task_byte_threshold=args.large_task_byte_threshold
    )
    file_receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # These execution-only limits intentionally do not enter the immutable
    # normalization contract: changing concurrency cannot change canonical
    # bytes, receipt identities or the validity of an incomplete-stage resume.
    collect_task_receipts(
        ordinary_tasks,
        workers=args.workers,
        pool_name="ordinary",
        file_receipts=file_receipts,
    )
    collect_task_receipts(
        large_tasks,
        workers=args.large_task_workers,
        pool_name="large",
        file_receipts=file_receipts,
    )

    all_shards: list[Path] = []
    declared_shards: list[dict[str, Any]] = []
    for receipts in file_receipts.values():
        for receipt in receipts:
            for shard in receipt["shards"]:
                all_shards.append(Path(str(shard["output"]["path"])).resolve())
                declared_shards.append(dict(shard["output"]))
    found_shards = sorted(args.output.glob("*/shard-*/*.parquet"))
    if set(all_shards) != {path.resolve() for path in found_shards}:
        raise ValueError(
            "canonical output contains unreceipted or missing Parquet shards"
        )
    work_stats, uniqueness = validate_global_canonical_inventory(
        sorted(all_shards),
        declared_inventory=declared_shards,
        output=args.output,
        contract_sha256=contract_sha256,
        memory_limit=args.duckdb_memory_limit,
        temporary_directory=temporary_root / "duckdb",
        threads=args.duckdb_threads,
        candidate_route_declarations=(
            candidate_roster["route_declarations"]
            if candidate_roster is not None
            else None
        ),
    )

    summaries: list[dict[str, Any]] = []
    for artifact in artifacts:
        receipts = sorted(
            file_receipts[artifact.source_id], key=lambda row: row["file_index"]
        )
        counts: Counter[str] = Counter()
        names: Counter[str] = Counter()
        shards: list[dict[str, Any]] = []
        for receipt in receipts:
            counts.update(receipt["counts"])
            names.update(receipt["exact_source_dataset_counts"])
            for shard in receipt["shards"]:
                shards.append({**shard["output"], "receipt": shard["receipt"]})
        stats = work_stats.get(
            artifact.source_id,
            {"unique_work_keys": 0, "multi_representation_work_keys": 0},
        )
        source_payload = {
            "schema_version": SOURCE_RECEIPT_SCHEMA,
            "contract_sha256": contract_sha256,
            "source_id": artifact.source_id,
            "repo_id": artifact.repo_id,
            "revision": artifact.revision,
            "source_family_id": artifact.source_family_id,
            "role": artifact.role,
            "counts": dict(sorted(counts.items())),
            "exact_source_dataset_counts": dict(sorted(names.items())),
            **stats,
            "files": [receipt["receipt"] for receipt in receipts],
            "shards": sorted(shards, key=lambda row: row["path"]),
        }
        if candidate_roster is not None:
            source_payload.update(
                candidate_roster["route_declarations"].get(
                    artifact.source_id,
                    {
                        "source_route": None,
                        "review_route": None,
                        "extraction_route": None,
                    },
                )
            )
        receipt_path = source_receipt_path(args.output, artifact.source_id)
        if receipt_path.exists():
            if read_json(receipt_path) != source_payload:
                raise ValueError(f"{receipt_path}: immutable source receipt drift")
        else:
            write_json_atomic(receipt_path, source_payload, immutable=True)
        summaries.append({**source_payload, "receipt": receipt_entry(receipt_path)})

    route_coverage = validate_embedded_route_coverage(
        embedded_structural_routes,
        artifacts,
        summaries,
        bounded_smoke=bool(args.max_rows_per_source),
    )
    selected_source_coverage = validate_selected_source_coverage(
        artifacts,
        summaries,
        bounded_smoke=bool(args.max_rows_per_source),
    )

    payload: dict[str, Any] = {
        "schema_version": NORMALIZATION_MANIFEST_SCHEMA,
        "contract": receipt_entry(contract_path),
        "contract_sha256": contract_sha256,
        "sources_config": str(args.sources.resolve()),
        "sources_config_sha256": sha256_file(args.sources),
        "lineage_aliases": str(args.lineage_aliases.resolve()),
        "lineage_aliases_sha256": sha256_file(args.lineage_aliases),
        "acquisition_receipt": str(args.acquisition_receipt.resolve()),
        "acquisition_receipt_sha256": sha256_file(args.acquisition_receipt),
        "output": str(args.output.resolve()),
        "bounded_smoke": bool(args.max_rows_per_source),
        "embedded_structural_route_coverage": route_coverage,
        "selected_source_coverage": selected_source_coverage,
        "uid_uniqueness": uniqueness,
        "sources": summaries,
        "total_documents": sum(
            row["counts"].get("documents_emitted", 0) for row in summaries
        ),
    }
    if candidate_roster is not None:
        payload.update(
            {
                "candidate_roster": {
                    key: candidate_roster[key]
                    for key in (
                        "path",
                        "bytes",
                        "sha256",
                        "schema_version",
                        "base_source_id",
                        "candidate_source_ids",
                        "review_routes",
                        "source_routes",
                        "extraction_routes",
                        "allowed_observed_extraction_routes",
                        "route_basis_metadata",
                        "route_declarations",
                    )
                },
                "candidate_roster_source_coverage": candidate_roster_source_coverage,
                "candidate_roster_canonical_route_coverage": uniqueness[
                    "candidate_roster_route_coverage"
                ],
            }
        )
    write_json_atomic(args.manifest, payload, immutable=True)
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
