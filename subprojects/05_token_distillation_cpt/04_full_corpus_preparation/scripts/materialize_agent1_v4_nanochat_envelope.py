#!/usr/bin/env python3
"""Materialize admitted v4 candidates in the exact six-column Nanochat envelope.

This runs only after a passed human gate and an explicitly approved, profile-
bound mapping file.  It never projects or re-cleans the existing Nanochat base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import file_binding, read_json_object, sha256_json, sha256_text, write_json_no_replace  # noqa: E402
from full_corpus_io import artifact_relative_path, artifacts_from_receipt, iter_parquet_rows, jsonable  # noqa: E402
from profile_agent1_v4_fields import PROFILE_SCHEMA  # noqa: E402
from validate_agent1_v4_human_decisions import RECEIPT_SCHEMA  # noqa: E402


MAPPING_SCHEMA = "agent1_v4_field_mapping_v1"
MANIFEST_SCHEMA = "agent1_v4_nanochat_envelope_manifest_v1"
SIX_COLUMNS = ("source_dataset", "source_doc_id", "text", "title", "author", "source_metadata_json")


def _read_path(row: Mapping[str, Any], path: str | None) -> Any:
    if path is None:
        return None
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _optional_text(value: Any, *, author: bool = False) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        pieces = [_optional_text(item, author=author) for item in value]
        pieces = [piece for piece in pieces if piece]
        return ("; " if author else " - ").join(pieces) or None
    text = str(value).strip()
    return text or None


def _mapping(path: Path, *, profile_path: Path, human_gate_path: Path, admitted: list[str]) -> dict[str, dict[str, object]]:
    value = read_json_object(path)
    if value.get("schema_version") != MAPPING_SCHEMA:
        raise ValueError("unsupported field mapping schema")
    if value.get("field_profile_sha256") != hashlib.sha256(profile_path.read_bytes()).hexdigest():
        raise ValueError("field mapping is bound to a different field profile")
    if value.get("human_gate_receipt_sha256") != hashlib.sha256(human_gate_path.read_bytes()).hexdigest():
        raise ValueError("field mapping is bound to a different human gate receipt")
    if not isinstance(value.get("approval"), Mapping):
        raise ValueError("field mapping requires an explicit approval object")
    mappings = value.get("mappings")
    if not isinstance(mappings, Mapping) or set(mappings) != set(admitted):
        raise ValueError("field mapping does not close admitted sources")
    result: dict[str, dict[str, object]] = {}
    required = {"text_path", "title_path", "author_path", "source_dataset_path", "source_doc_id_paths"}
    for source_id in admitted:
        row = mappings[source_id]
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError(f"field mapping keys drift for {source_id}")
        if not isinstance(row.get("text_path"), str) or not str(row["text_path"]).strip():
            raise ValueError(f"field mapping lacks text_path for {source_id}")
        for key in ("title_path", "author_path", "source_dataset_path"):
            if row[key] is not None and (not isinstance(row[key], str) or not str(row[key]).strip()):
                raise ValueError(f"field mapping has invalid {key} for {source_id}")
        if not isinstance(row.get("source_doc_id_paths"), list) or any(not isinstance(item, str) or not item.strip() for item in row["source_doc_id_paths"]):
            raise ValueError(f"field mapping has invalid source_doc_id_paths for {source_id}")
        result[source_id] = dict(row)
    return result


def _admitted(human_gate_path: Path) -> list[str]:
    receipt = read_json_object(human_gate_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("human gate receipt is not passed")
    admitted = receipt.get("admitted_source_ids")
    if not isinstance(admitted, list) or any(not isinstance(value, str) for value in admitted):
        raise ValueError("human gate receipt has invalid admitted sources")
    return list(admitted)


def _metadata(row: Mapping[str, Any], source: Any, mapping: Mapping[str, object]) -> str | None:
    excluded = set(source.config.get("text_columns", [])) | set(source.config.get("alternate_text_columns", []) or [])
    for key in ("text_path", "title_path", "author_path"):
        path = mapping.get(key)
        if isinstance(path, str):
            excluded.add(path.split(".", 1)[0])
    payload = {str(key): jsonable(value) for key, value in row.items() if key not in excluded and value is not None}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if payload else None


def _synthetic_id(source: Any, artifact_path: str, row_index: int) -> str:
    return "synthetic:" + sha256_json({"namespace": "agent1_v4_envelope_source_doc_id_v1", "source_id": source.source_id, "revision": source.revision, "artifact_path": artifact_path, "row_index": row_index, "representation_suffix": "0"})


def _packet_binding(root: Path, path: Path) -> dict[str, object]:
    binding = file_binding(path)
    binding["path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return binding


def materialize_envelope(
    *, sources_path: Path, acquisition_receipt: Path, human_gate_receipt: Path,
    field_profile: Path, mapping_path: Path, output: Path,
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"envelope output already exists: {output}")
    admitted = _admitted(human_gate_receipt)
    profile = read_json_object(field_profile)
    if profile.get("schema_version") != PROFILE_SCHEMA or profile.get("status") != "passed" or profile.get("admitted_source_ids") != admitted:
        raise ValueError("field profile is not a passed closure for admitted sources")
    mappings = _mapping(mapping_path, profile_path=field_profile, human_gate_path=human_gate_receipt, admitted=admitted)
    source_artifacts = artifacts_from_receipt(sources_path, acquisition_receipt, set(admitted)) if admitted else []
    source_by_id = {source.source_id: source for source in source_artifacts}
    if set(source_by_id) != set(admitted):
        raise ValueError("admitted source/acquisition closure drift")
    schema = pa.schema([pa.field(column, pa.large_string(), nullable=True) for column in SIX_COLUMNS])
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        os.chmod(staging, 0o700)
        source_counts: dict[str, dict[str, int]] = {}
        ledgers: list[dict[str, object]] = []
        for source_id in admitted:
            source = source_by_id[source_id]
            mapping = mappings[source_id]
            shard = staging / "candidates" / f"{source_id}.parquet"
            shard.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            writer = pq.ParquetWriter(shard, schema, compression="zstd")
            batch: list[dict[str, str | None]] = []
            seen_doc_ids: set[tuple[str, str]] = set()
            counts: Counter[str] = Counter()
            ledger_path = staging / "sidecars" / f"{source_id}.jsonl"
            ledger_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with ledger_path.open("x", encoding="utf-8") as ledger:
                for artifact in sorted(source.files):
                    relative = artifact_relative_path(source, artifact)
                    for row_index, row in iter_parquet_rows(artifact):
                        counts["input"] += 1
                        text = _optional_text(_read_path(row, str(mapping["text_path"])))
                        if text is None:
                            counts["quarantined_blank_text"] += 1
                            ledger.write(json.dumps({"source_id": source_id, "artifact_path": relative, "row_index": row_index, "action": "quarantine", "reason": "blank_selected_text"}, ensure_ascii=False, sort_keys=True) + "\n")
                            continue
                        source_dataset = _optional_text(_read_path(row, mapping["source_dataset_path"])) or source.repo_id
                        identifier_values = [_optional_text(_read_path(row, path)) for path in mapping["source_doc_id_paths"]]
                        identifier_values = [value for value in identifier_values if value]
                        source_doc_id = "|".join(identifier_values) if identifier_values else _synthetic_id(source, relative, row_index)
                        key = (source_dataset, source_doc_id)
                        collision = False
                        if key in seen_doc_ids:
                            collision = True
                            source_doc_id += "#" + sha256_json({"source_id": source_id, "artifact_path": relative, "row_index": row_index})[:16]
                            key = (source_dataset, source_doc_id)
                        seen_doc_ids.add(key)
                        title = _optional_text(_read_path(row, mapping["title_path"]))
                        author = _optional_text(_read_path(row, mapping["author_path"]), author=True)
                        metadata = _metadata(row, source, mapping)
                        payload = {"source_dataset": source_dataset, "source_doc_id": source_doc_id, "text": text, "title": title, "author": author, "source_metadata_json": metadata}
                        batch.append(payload)
                        counts["materialized"] += 1
                        counts["source_doc_id_collision"] += int(collision)
                        ledger_row = {
                            "source_id": source_id, "repo_id": source.repo_id, "revision": source.revision,
                            "artifact_path": relative, "row_index": row_index, "selected_text_path": mapping["text_path"],
                            "source_dataset": source_dataset, "source_doc_id": source_doc_id, "original_text_sha256": sha256_text(text),
                            "stable_uid": sha256_json({"namespace": "agent1_v4_envelope_stable_uid_v1", "source_id": source_id, "revision": source.revision, "source_dataset": source_dataset, "source_doc_id": source_doc_id, "text_sha256": sha256_text(text)}),
                            "action": "materialize", "collision_resolved": collision,
                        }
                        ledger.write(json.dumps(ledger_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                        if len(batch) >= 4096:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                            batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            writer.close()
            if counts["materialized"] == 0:
                raise ValueError(f"{source_id}: no nonblank rows materialized from approved text mapping")
            source_counts[source_id] = dict(counts)
            ledgers.append({"source_id": source_id, "path": f"sidecars/{source_id}.jsonl", "rows": counts["input"]})
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "passed",
            "six_column_schema": list(SIX_COLUMNS),
            "sources": file_binding(sources_path), "acquisition_receipt": file_binding(acquisition_receipt),
            "human_gate_receipt": file_binding(human_gate_receipt), "field_profile": file_binding(field_profile), "field_mapping": file_binding(mapping_path),
            "admitted_source_ids": admitted, "source_counts": source_counts, "sidecars": ledgers,
            "candidate_shards": [_packet_binding(staging, staging / "candidates" / f"{source_id}.parquet") for source_id in admitted],
        }
        write_json_no_replace(staging / "envelope_manifest.json", manifest)
        os.rename(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / "configs" / "sources.json")
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--human-gate-receipt", type=Path, required=True)
    parser.add_argument("--field-profile", type=Path, required=True)
    parser.add_argument("--field-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = materialize_envelope(
        sources_path=args.sources, acquisition_receipt=args.acquisition_receipt, human_gate_receipt=args.human_gate_receipt,
        field_profile=args.field_profile, mapping_path=args.field_mapping, output=args.output,
    )
    print(json.dumps({"ok": True, "sources": len(manifest["admitted_source_ids"])}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
