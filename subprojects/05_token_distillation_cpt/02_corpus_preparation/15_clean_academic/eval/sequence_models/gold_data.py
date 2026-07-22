#!/usr/bin/env python3
"""Streaming, resume-safe human-gold packet builder and adjudicated-gold importer.

No model predictions are read or written. Sampling ignores labels and is bound
to canonical Phase-04 Parquet plus its passed acquisition receipt.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import (
    build_split_manifest,
    canonical_json_sha256,
    read_gold,
    sha256_file,
    validate_gold,
)

HERE = Path(__file__).resolve().parent
PHASE04_DIR = HERE.parents[3] / "04_full_corpus_preparation"
LABELS = {"O", "BIB", "TOC"}


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _load_phase04_module(name: str, filename: str) -> Any:
    path = PHASE04_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Phase-04 helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hash_id(namespace: str, *parts: object) -> str:
    payload = "\0".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"resume collision: existing file differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json_once(path: Path, value: Any) -> None:
    _write_once(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


@dataclass(frozen=True)
class Identity:
    document_id: str
    work_id: str
    source: str


def _phase04_routes(sources: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(route["source_id"]): route for route in sources["embedded_structural_routes"]}


def _parse_source_inputs(values: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = collections.defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source-input must be route_id=path_or_glob, got {value!r}")
        route_id, path = value.split("=", 1)
        if not route_id or not path:
            raise ValueError(f"invalid --source-input {value!r}")
        result[route_id].append(path)
    return dict(result)


def _artifact_map(receipt: Mapping[str, Any], acquisition_source_id: str) -> tuple[str, dict[Path, str]]:
    rows = [row for row in receipt.get("sources", []) if row.get("source_id") == acquisition_source_id]
    if len(rows) != 1:
        raise ValueError(f"receipt must contain exactly one {acquisition_source_id!r} entry")
    row = rows[0]
    mapping: dict[Path, str] = {}
    for item in row.get("files", []):
        local = item.get("local_path")
        relative = item.get("path")
        if isinstance(local, str) and isinstance(relative, str):
            mapping[Path(local).resolve()] = relative
    return str(row["revision"]), mapping


def _open_state(path: Path, bindings: Mapping[str, str]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints (
          route_id TEXT NOT NULL, path TEXT NOT NULL, row_group INTEGER NOT NULL,
          PRIMARY KEY(route_id, path, row_group)
        );
        CREATE TABLE IF NOT EXISTS seen_documents (
          document_id TEXT PRIMARY KEY, gold_source TEXT NOT NULL, upstream_document_id TEXT NOT NULL,
          text_sha256 TEXT NOT NULL, canonical_work_key TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
          gold_source TEXT NOT NULL, sample_priority TEXT NOT NULL, document_id TEXT PRIMARY KEY,
          work_id TEXT NOT NULL, representation_id TEXT NOT NULL,
          upstream_document_id TEXT NOT NULL, canonical_work_key TEXT NOT NULL,
          text_sha256 TEXT NOT NULL, text TEXT NOT NULL, source_dataset TEXT NOT NULL,
          route_id TEXT NOT NULL, route_revision TEXT NOT NULL, artifact_path TEXT NOT NULL,
          row_group INTEGER NOT NULL, row_offset INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS candidates_order ON candidates(gold_source, sample_priority, document_id);
        """
    )
    existing = dict(connection.execute("SELECT key, value FROM meta"))
    if existing:
        if existing != dict(bindings):
            raise ValueError("resume state bindings differ from this invocation; use a new state path")
    else:
        connection.executemany("INSERT INTO meta(key,value) VALUES (?,?)", sorted(bindings.items()))
        connection.commit()
    return connection


def _prune_candidates(connection: sqlite3.Connection, source: str, limit: int) -> None:
    connection.execute(
        """DELETE FROM candidates WHERE gold_source=? AND document_id NOT IN (
             SELECT document_id FROM candidates WHERE gold_source=?
             ORDER BY sample_priority, document_id LIMIT ?
           )""",
        (source, source, limit),
    )


def _scan_route(
    *,
    connection: sqlite3.Connection,
    route_id: str,
    gold_source: str,
    route: Mapping[str, Any],
    revision: str,
    artifact_paths: Mapping[Path, str],
    inputs: Sequence[Path],
    seed: str,
    candidate_limit: int,
    batch_size: int,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - Clariden runtime path
        raise RuntimeError("packet creation requires the pinned Phase-04 pyarrow runtime") from exc
    pattern = re.compile(str(route["source_regex"]))
    text_column = str(route["text_columns"][0])
    id_column = str(route["id_columns"][0])
    source_column = str(route["source_column"])
    for path in inputs:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        required = {text_column, id_column, source_column}
        missing = required - columns
        if missing:
            raise ValueError(f"{path}: missing canonical columns {sorted(missing)}")
        work_column = "work_key" if "work_key" in columns else None
        selected_columns = [id_column, text_column, source_column]
        if work_column:
            selected_columns.append(work_column)
        artifact_path = artifact_paths.get(path.resolve())
        if artifact_path is None:
            raise ValueError(f"{path}: not bound to the acquisition receipt")
        for row_group in range(parquet.num_row_groups):
            checkpoint = connection.execute(
                "SELECT 1 FROM checkpoints WHERE route_id=? AND path=? AND row_group=?",
                (route_id, str(path.resolve()), row_group),
            ).fetchone()
            if checkpoint:
                continue
            row_offset = 0
            connection.execute("BEGIN IMMEDIATE")
            try:
                for batch in parquet.iter_batches(
                    batch_size=batch_size, row_groups=[row_group], columns=selected_columns
                ):
                    values = batch.to_pydict()
                    for index, raw_id in enumerate(values[id_column]):
                        current_offset = row_offset + index
                        upstream_id = str(raw_id or "")
                        source_dataset = str(values[source_column][index] or "")
                        if not pattern.search(source_dataset):
                            continue
                        if not upstream_id:
                            raise ValueError(f"{path}: empty {id_column} at row group {row_group}")
                        text = str(values[text_column][index] or "")
                        if not text.strip():
                            continue
                        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        raw_work = str(values[work_column][index] or "") if work_column else upstream_id
                        if not raw_work:
                            raw_work = upstream_id
                        canonical_work_key = f"{gold_source}\0{raw_work}"
                        document_id = _hash_id("academic-structure-document-v1", gold_source, upstream_id)
                        work_id = _hash_id("academic-structure-work-v1", canonical_work_key)
                        representation_id = _hash_id(
                            "academic-structure-representation-v1", revision, artifact_path,
                            row_group, current_offset, text_sha256,
                        )
                        priority = _hash_id(seed, gold_source, upstream_id, text_sha256)
                        try:
                            connection.execute(
                                "INSERT INTO seen_documents VALUES (?,?,?,?,?)",
                                (document_id, gold_source, upstream_id, text_sha256, canonical_work_key),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                f"duplicate canonical document identity for {gold_source}:{upstream_id}"
                            ) from exc
                        connection.execute(
                            "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                gold_source, priority, document_id, work_id, representation_id,
                                upstream_id, canonical_work_key, text_sha256, text, source_dataset,
                                route_id, revision, artifact_path, row_group, current_offset,
                            ),
                        )
                    row_offset += batch.num_rows
                _prune_candidates(connection, gold_source, candidate_limit)
                connection.execute(
                    "INSERT INTO checkpoints VALUES (?,?,?)",
                    (route_id, str(path.resolve()), row_group),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise


def _select_candidates(
    connection: sqlite3.Connection,
    gold_sources: Sequence[str],
    documents_per_source: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_work: set[str] = set()
    used_text: set[str] = set()
    columns = [row[1] for row in connection.execute("PRAGMA table_info(candidates)")]
    for source in gold_sources:
        count = 0
        cursor = connection.execute(
            "SELECT * FROM candidates WHERE gold_source=? ORDER BY sample_priority, document_id",
            (source,),
        )
        for values in cursor:
            row = dict(zip(columns, values))
            if row["work_id"] in used_work or row["text_sha256"] in used_text:
                continue
            used_work.add(row["work_id"])
            used_text.add(row["text_sha256"])
            selected.append(row)
            count += 1
            if count == documents_per_source:
                break
        if count != documents_per_source:
            raise ValueError(
                f"{source}: only {count} unique work/text candidates; need {documents_per_source}. "
                "Increase candidate_reserve_multiplier and rebuild the state."
            )
    return selected


def _packet(row: Mapping[str, Any], config: Mapping[str, Any], split: str) -> dict[str, Any]:
    physical_lines = str(row["text"]).split("\n")
    return {
        "schema_version": config["packet_schema_version"],
        "packet_id": row["document_id"],
        "document_id": row["document_id"],
        "work_id": row["work_id"],
        "representation_id": row["representation_id"],
        "source": row["gold_source"],
        "split": split,
        "text_sha256": row["text_sha256"],
        "n_physical_lines": len(physical_lines),
        "n_present_lines": sum(bool(line.strip()) for line in physical_lines),
        "identity": {
            "phase04_route_id": row["route_id"],
            "route_revision": row["route_revision"],
            "source_dataset": row["source_dataset"],
            "source_doc_id": row["upstream_document_id"],
            "canonical_work_key": row["canonical_work_key"],
            "source_artifact_path": row["artifact_path"],
            "row_group": row["row_group"],
            "row_offset": row["row_offset"],
        },
        "instructions": {
            "human_only": config["human_attestation"],
            "blind_to_models": "Do not consult detector/model/LLM labels. They are not human evidence.",
            "coverage": "Cover every physical line exactly once with contiguous judgments; do not omit blanks.",
            "labels": {
                "O": "retain: not a bibliography or table-of-contents line",
                "BIB": "bibliography/reference-list line, including wrapped entries",
                "TOC": "table-of-contents navigation line",
            },
            "is_running_prose": (
                "Judge independently from O/BIB/TOC. True only for substantive running main-text prose; "
                "false for bibliography, ToC, blank lines, headings, captions, tables, front matter, and apparatus."
            ),
            "uncertainty": "Escalate uncertainty to adjudication; never fill it with a model guess.",
        },
        "annotation_output": {
            "schema_version": config["annotation_schema_version"],
            "required_fields": [
                "packet_id", "document_id", "text_sha256", "annotation_kind",
                "annotator_id", "human_attestation", "judgments",
            ],
            "judgment_fields": ["start_line", "end_line", "label", "is_running_prose"],
        },
        "physical_lines": [
            {"abs_idx": index, "text": text} for index, text in enumerate(physical_lines)
        ],
    }


def _double_ids(
    rows: Sequence[Mapping[str, Any]], source: str, fraction: float, seed: str
) -> set[str]:
    test = [row for row in rows if row["gold_source"] == source and row["split"] == "test"]
    wanted = math.ceil(len(test) * fraction)
    ordered = sorted(
        test,
        key=lambda row: _hash_id(seed, "double-annotation", row["document_id"]),
    )
    return {str(row["document_id"]) for row in ordered[:wanted]}


def sample_packets(args: argparse.Namespace) -> int:
    pipeline = _load_object(args.pipeline_config)
    eval_config = _load_object(args.eval_config)
    sources = _load_object(args.phase04_sources)
    receipt = _load_object(args.input_receipt)
    routes = _phase04_routes(sources)
    configured = {row["phase04_route_id"]: row for row in pipeline["routes"]}
    source_inputs = _parse_source_inputs(args.source_input)
    if set(source_inputs) != set(configured):
        raise ValueError(
            f"source inputs must cover exactly {sorted(configured)}; got {sorted(source_inputs)}"
        )
    validate_input = _load_phase04_module(
        "phase04_validate_input_receipt", "validate_input_receipt.py"
    )
    route_checks: dict[str, Any] = {}
    resolved_inputs: dict[str, list[Path]] = {}
    revisions: dict[str, str] = {}
    artifact_maps: dict[str, dict[Path, str]] = {}
    for route_id, mapping in configured.items():
        route = routes[route_id]
        check = validate_input.validate_input(
            receipt_path=Path(args.input_receipt),
            sources_path=Path(args.phase04_sources),
            source_id=route_id,
            input_values=source_inputs[route_id],
            text_column=str(route["text_columns"][0]),
            id_column=str(route["id_columns"][0]),
            source_column=str(route["source_column"]),
        )
        route_checks[route_id] = check
        resolved_inputs[route_id] = [Path(path).resolve() for path in check["paths"]]
        revision, artifacts = _artifact_map(receipt, str(route["acquisition_source_id"]))
        revisions[route_id] = revision
        artifact_maps[route_id] = artifacts

    bindings = {
        "gold_data_sha256": sha256_file(__file__),
        "phase04_receipt_validator_sha256": sha256_file(
            PHASE04_DIR / "scripts" / "validate_input_receipt.py"
        ),
        "pipeline_config_sha256": sha256_file(args.pipeline_config),
        "eval_config_sha256": sha256_file(args.eval_config),
        "phase04_sources_sha256": sha256_file(args.phase04_sources),
        "input_receipt_sha256": sha256_file(args.input_receipt),
        "route_checks_sha256": canonical_json_sha256(route_checks),
    }
    output_dir = Path(args.output_dir)
    completed = output_dir / "COMPLETED.json"
    if completed.exists():
        marker = _load_object(completed)
        if marker.get("bindings") != bindings:
            raise ValueError("completed packet set is bound to different inputs/configuration")
        manifest_path = output_dir / "packet_manifest.json"
        split_path = output_dir / "split_manifest.json"
        if (
            not manifest_path.is_file()
            or marker.get("packet_manifest_sha256") != sha256_file(manifest_path)
            or not split_path.is_file()
            or marker.get("split_manifest_sha256") != sha256_file(split_path)
        ):
            raise ValueError("completed packet set has missing or drifted manifests")
        existing_manifest = _load_object(manifest_path)
        assignment_path = output_dir / existing_manifest["assignment_ledger_path"]
        if (
            not assignment_path.is_file()
            or sha256_file(assignment_path) != existing_manifest["assignment_ledger_sha256"]
        ):
            raise ValueError("completed packet set has a missing or drifted assignment ledger")
        for entry in existing_manifest.get("packets", []):
            packet_path = output_dir / entry["packet_path"]
            if not packet_path.is_file() or sha256_file(packet_path) != entry["packet_sha256"]:
                raise ValueError(f"completed packet set has missing or drifted packet: {packet_path}")
        print(json.dumps(marker, sort_keys=True))
        return 0
    connection = _open_state(Path(args.state), bindings)
    documents_per_source = int(pipeline["documents_per_source"])
    candidate_limit = documents_per_source * int(pipeline["candidate_reserve_multiplier"])
    for route_id, mapping in configured.items():
        _scan_route(
            connection=connection,
            route_id=route_id,
            gold_source=str(mapping["gold_source"]),
            route=routes[route_id],
            revision=revisions[route_id],
            artifact_paths=artifact_maps[route_id],
            inputs=resolved_inputs[route_id],
            seed=str(pipeline["seed"]),
            candidate_limit=candidate_limit,
            batch_size=args.batch_size,
        )
    gold_sources = [str(row["gold_source"]) for row in pipeline["routes"]]
    selected = _select_candidates(connection, gold_sources, documents_per_source)
    identities = [Identity(row["document_id"], row["work_id"], row["gold_source"]) for row in selected]
    split_manifest = build_split_manifest(identities, eval_config["split"])
    for row in selected:
        row["split"] = split_manifest["assignments"][row["document_id"]]
    double: set[str] = set()
    for source in gold_sources:
        double.update(_double_ids(
            selected, source, float(pipeline["minimum_double_annotated_test_fraction"]),
            str(pipeline["seed"]),
        ))
    packet_rows: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: (item["gold_source"], item["document_id"])):
        packet = _packet(row, pipeline, row["split"])
        packet_path = output_dir / "packets" / row["gold_source"] / f"{row['document_id']}.json"
        _write_json_once(packet_path, packet)
        packet_rows.append({
            "packet_id": row["document_id"],
            "document_id": row["document_id"],
            "work_id": row["work_id"],
            "representation_id": row["representation_id"],
            "source": row["gold_source"],
            "split": row["split"],
            "text_sha256": row["text_sha256"],
            "packet_path": str(packet_path.relative_to(output_dir)),
            "packet_sha256": sha256_file(packet_path),
            "independent_annotations_required": 2 if row["document_id"] in double else 1,
            "human_adjudication_required": row["split"] == "test" or row["document_id"] in double,
        })
    counts = collections.Counter((row["source"], row["split"]) for row in packet_rows)
    for source in gold_sources:
        if counts[(source, "test")] < int(pipeline["minimum_test_documents_per_source"]):
            raise ValueError(f"{source}: locked test count is below the configured minimum")
    manifest = {
        "schema_version": "academic-structure-human-packet-manifest-v1",
        "status": "locked",
        "bindings": bindings,
        "identity_contract": pipeline["identity_contract"],
        "sampling": {
            "seed": pipeline["seed"],
            "algorithm": "streaming-sqlite-sha256-priority-v1",
            "documents_per_source": documents_per_source,
            "candidate_reserve_multiplier": pipeline["candidate_reserve_multiplier"],
            "labels_or_model_outputs_consulted": False,
        },
        "annotation_policy": {
            "human_attestation": pipeline["human_attestation"],
            "minimum_double_annotated_test_fraction": pipeline["minimum_double_annotated_test_fraction"],
            "test_requires_human_adjudication": True,
            "automatic_adjudication": False,
        },
        "route_checks": route_checks,
        "counts_by_source_split": {
            source: {split: counts[(source, split)] for split in ("train", "validation", "test")}
            for source in gold_sources
        },
        "packets": packet_rows,
    }
    split_path = output_dir / "split_manifest.json"
    assignment_path = output_dir / "annotation_assignments.jsonl"
    manifest_path = output_dir / "packet_manifest.json"
    assignment_rows: list[dict[str, Any]] = []
    for entry in packet_rows:
        slots = [f"independent-{index + 1}" for index in range(entry["independent_annotations_required"])]
        for slot in slots:
            assignment_rows.append({
                "schema_version": "academic-structure-human-assignment-v1",
                "packet_id": entry["packet_id"],
                "source": entry["source"],
                "split": entry["split"],
                "role": "independent_annotation",
                "slot": slot,
                "packet_path": entry["packet_path"],
                "must_be_human": True,
                "model_labels_allowed": False,
            })
        if entry["human_adjudication_required"]:
            assignment_rows.append({
                "schema_version": "academic-structure-human-assignment-v1",
                "packet_id": entry["packet_id"],
                "source": entry["source"],
                "split": entry["split"],
                "role": "human_adjudication",
                "depends_on_slots": slots,
                "packet_path": entry["packet_path"],
                "must_be_human": True,
                "model_labels_allowed": False,
            })
    assignment_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in assignment_rows
    ).encode("utf-8")
    _write_once(assignment_path, assignment_payload)
    manifest["assignment_ledger_path"] = str(assignment_path.relative_to(output_dir))
    manifest["assignment_ledger_sha256"] = sha256_file(assignment_path)
    manifest["assignment_count"] = len(assignment_rows)
    _write_json_once(split_path, split_manifest)
    manifest["split_manifest_sha256"] = sha256_file(split_path)
    _write_json_once(manifest_path, manifest)
    marker = {
        "schema_version": "academic-structure-human-packets-completed-v1",
        "bindings": bindings,
        "packet_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "packet_count": len(packet_rows),
    }
    _write_json_once(completed, marker)
    print(json.dumps(marker, sort_keys=True))
    return 0


def _expand_judgments(
    judgments: object, physical_lines: Sequence[Mapping[str, Any]], *, label: str
) -> list[tuple[str, bool]]:
    if not isinstance(judgments, list) or not judgments:
        raise ValueError(f"{label}: judgments must be a non-empty list")
    expanded: list[tuple[str, bool]] = []
    expected = 0
    for index, judgment in enumerate(judgments):
        if not isinstance(judgment, Mapping):
            raise ValueError(f"{label}: judgment {index} is not an object")
        start, end = judgment.get("start_line"), judgment.get("end_line")
        value, prose = judgment.get("label"), judgment.get("is_running_prose")
        if not isinstance(start, int) or not isinstance(end, int) or start != expected or end < start:
            raise ValueError(f"{label}: judgments must be ordered, gap-free, and non-overlapping")
        if value not in LABELS or not isinstance(prose, bool):
            raise ValueError(f"{label}: invalid label/prose judgment")
        if prose and value != "O":
            raise ValueError(f"{label}: running prose must have label O")
        for line_number in range(start, end + 1):
            if line_number >= len(physical_lines):
                raise ValueError(f"{label}: judgment exceeds physical line count")
            text = str(physical_lines[line_number]["text"])
            if not text.strip() and (value != "O" or prose):
                raise ValueError(f"{label}: blank lines must be O and not running prose")
            expanded.append((str(value), prose))
        expected = end + 1
    if expected != len(physical_lines):
        raise ValueError(f"{label}: judgments do not cover every physical line")
    return expanded


def _validate_human_file(
    value: Mapping[str, Any], packet: Mapping[str, Any], config: Mapping[str, Any], *, adjudication: bool
) -> list[tuple[str, bool]]:
    schema_key = "adjudication_schema_version" if adjudication else "annotation_schema_version"
    expected_kind = "human_adjudication" if adjudication else "human_independent"
    if value.get("schema_version") != config[schema_key] or value.get("annotation_kind") != expected_kind:
        raise ValueError(f"{packet['packet_id']}: invalid human file schema/kind")
    for key in ("packet_id", "document_id", "text_sha256"):
        if value.get(key) != packet.get(key):
            raise ValueError(f"{packet['packet_id']}: human file {key} mismatch")
    if value.get("human_attestation") != config["human_attestation"]:
        raise ValueError(f"{packet['packet_id']}: exact human attestation is required")
    person_key = "adjudicator_id" if adjudication else "annotator_id"
    if not isinstance(value.get(person_key), str) or not value[person_key].strip():
        raise ValueError(f"{packet['packet_id']}: {person_key} is required")
    return _expand_judgments(
        value.get("judgments"), packet["physical_lines"], label=f"{packet['packet_id']}:{person_key}"
    )


def _annotation_files(directory: Path, packet_id: str) -> list[Path]:
    return sorted(directory.glob(f"{packet_id}.*.json"))


def import_gold(args: argparse.Namespace) -> int:
    pipeline = _load_object(args.pipeline_config)
    eval_config = _load_object(args.eval_config)
    packet_root = Path(args.packet_root)
    manifest_path = packet_root / "packet_manifest.json"
    split_path = packet_root / "split_manifest.json"
    completed_path = packet_root / "COMPLETED.json"
    manifest = _load_object(manifest_path)
    split_manifest = _load_object(split_path)
    completed = _load_object(completed_path)
    if completed.get("packet_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("packet COMPLETED marker does not bind the current manifest")
    if manifest.get("status") != "locked" or manifest.get("split_manifest_sha256") != sha256_file(split_path):
        raise ValueError("packet manifest/split lock is incomplete or stale")
    if manifest.get("annotation_policy", {}).get("automatic_adjudication") is not False:
        raise ValueError("packet manifest must explicitly forbid automatic adjudication")
    structural = _load_phase04_module("phase04_structural_token_loss", "structural_token_loss.py")
    tokenizer_config = pipeline["tokenizer"]
    tokenizer = structural.ExactTokenizer(
        Path(args.tokenizer_json), str(tokenizer_config["tokenizer_json_sha256"])
    )
    output_path = Path(args.output)
    receipt_path = Path(args.receipt)
    if receipt_path.exists():
        existing = _load_object(receipt_path)
        if not output_path.is_file() or existing.get("gold_sha256") != sha256_file(output_path):
            raise ValueError("existing import receipt/gold binding is incomplete")
        print(json.dumps(existing, ensure_ascii=False, sort_keys=True))
        return 0
    rows: list[dict[str, Any]] = []
    annotation_inventory: list[dict[str, Any]] = []
    for entry in manifest["packets"]:
        packet_path = packet_root / entry["packet_path"]
        if sha256_file(packet_path) != entry["packet_sha256"]:
            raise ValueError(f"packet hash mismatch: {packet_path}")
        packet = _load_object(packet_path)
        reconstructed = "\n".join(str(line["text"]) for line in packet["physical_lines"])
        if hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() != packet["text_sha256"]:
            raise ValueError(f"{entry['packet_id']}: packet physical lines do not reconstruct locked text")
        annotation_paths = _annotation_files(Path(args.annotations), entry["packet_id"])
        required = int(entry["independent_annotations_required"])
        if len(annotation_paths) != required:
            raise ValueError(
                f"{entry['packet_id']}: need exactly {required} independent annotations, "
                f"found {len(annotation_paths)}"
            )
        annotations = [_load_object(path) for path in annotation_paths]
        expanded_annotations = [
            _validate_human_file(value, packet, pipeline, adjudication=False)
            for value in annotations
        ]
        annotator_ids = [str(value["annotator_id"]) for value in annotations]
        if len(set(annotator_ids)) != len(annotator_ids):
            raise ValueError(f"{entry['packet_id']}: independent annotator IDs must be distinct")
        annotation_hashes = [sha256_file(path) for path in annotation_paths]
        adjudication_path = Path(args.adjudications) / f"{entry['packet_id']}.json"
        adjudication_required = bool(entry["human_adjudication_required"])
        if adjudication_required:
            if not adjudication_path.is_file():
                raise ValueError(f"{entry['packet_id']}: human adjudication is required")
            adjudication = _load_object(adjudication_path)
            final = _validate_human_file(adjudication, packet, pipeline, adjudication=True)
            if sorted(adjudication.get("input_annotation_sha256", [])) != sorted(annotation_hashes):
                raise ValueError(f"{entry['packet_id']}: adjudication does not cite exact annotations")
            adjudicator_id = str(adjudication["adjudicator_id"])
            if adjudicator_id in annotator_ids:
                raise ValueError(f"{entry['packet_id']}: adjudicator must be independent")
            status = "human_adjudicated"
            adjudication_sha256 = sha256_file(adjudication_path)
        else:
            if adjudication_path.exists():
                raise ValueError(f"{entry['packet_id']}: unexpected adjudication outside locked assignment")
            if len(expanded_annotations) != 1:
                raise ValueError(f"{entry['packet_id']}: multiple annotations require adjudication")
            final = expanded_annotations[0]
            adjudicator_id = None
            adjudication_sha256 = None
            status = "human_single"
        present = [
            (line, decision) for line, decision in zip(packet["physical_lines"], final)
            if str(line["text"]).strip()
        ]
        token_counts: list[int] = []
        for start in range(0, len(present), 512):
            token_counts.extend(tokenizer.counts([str(line["text"]) for line, _ in present[start:start + 512]]))
        gold_lines = []
        for (line, (label, prose)), token_count in zip(present, token_counts):
            abs_idx = int(line["abs_idx"])
            text = str(line["text"])
            gold_lines.append({
                "line_id": _hash_id("academic-structure-line-v1", packet["document_id"], abs_idx, text),
                "abs_idx": abs_idx,
                "text": text,
                "label": label,
                "token_count": token_count,
                "is_running_prose": prose,
            })
        rows.append({
            "schema_version": "academic-structure-gold-v1",
            "document_id": packet["document_id"],
            "work_id": packet["work_id"],
            "representation_id": packet["representation_id"],
            "source": packet["source"],
            "split": packet["split"],
            "coverage": "full_document",
            "n_physical_lines": packet["n_physical_lines"],
            "n_present_lines": packet["n_present_lines"],
            "annotation": {
                "status": status,
                "annotator_ids": annotator_ids,
                "adjudicator_id": adjudicator_id,
                "annotation_sha256": annotation_hashes,
                "adjudication_sha256": adjudication_sha256,
                "automatic_adjudication": False,
            },
            "tokenizer": {
                "id": tokenizer_config["id"],
                "revision": tokenizer_config["revision"],
                "tokenizer_json_sha256": tokenizer_config["tokenizer_json_sha256"],
            },
            "text_sha256": packet["text_sha256"],
            "lines": gold_lines,
        })
        annotation_inventory.append({
            "packet_id": entry["packet_id"],
            "annotation_sha256": annotation_hashes,
            "adjudication_sha256": adjudication_sha256,
        })
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: row["document_id"])
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".validation", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        documents = read_gold(temporary_name)
        contract_receipt = validate_gold(
            documents, eval_config["gold_contract"], split_manifest=split_manifest, for_promotion=True
        )
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    _write_once(output_path, payload)
    receipt = {
        "schema_version": "academic-structure-human-gold-import-v1",
        "status": "passed",
        "gold_sha256": sha256_file(output_path),
        "packet_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "pipeline_config_sha256": sha256_file(args.pipeline_config),
        "eval_config_sha256": sha256_file(args.eval_config),
        "tokenizer_json_sha256": tokenizer.sha256,
        "contract": contract_receipt,
        "annotation_inventory_sha256": canonical_json_sha256(annotation_inventory),
        "automatic_adjudication": False,
    }
    _write_json_once(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample", help="stream canonical Parquet and create locked packets")
    sample.add_argument("--pipeline-config", default=str(HERE / "gold_pipeline.json"))
    sample.add_argument("--eval-config", default=str(HERE / "config.json"))
    sample.add_argument("--phase04-sources", default=str(PHASE04_DIR / "configs" / "sources.json"))
    sample.add_argument("--input-receipt", required=True)
    sample.add_argument("--source-input", action="append", required=True)
    sample.add_argument("--state", required=True)
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--batch-size", type=int, default=1024)
    imported = subparsers.add_parser("import-gold", help="validate human work and emit pinned gold")
    imported.add_argument("--pipeline-config", default=str(HERE / "gold_pipeline.json"))
    imported.add_argument("--eval-config", default=str(HERE / "config.json"))
    imported.add_argument("--packet-root", required=True)
    imported.add_argument("--annotations", required=True)
    imported.add_argument("--adjudications", required=True)
    imported.add_argument("--tokenizer-json", required=True)
    imported.add_argument("--output", required=True)
    imported.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        return sample_packets(args)
    return import_gold(args)


if __name__ == "__main__":
    raise SystemExit(main())
