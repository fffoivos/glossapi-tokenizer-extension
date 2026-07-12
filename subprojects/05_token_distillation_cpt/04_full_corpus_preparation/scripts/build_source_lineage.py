#!/usr/bin/env python3
"""Build deterministic source- and row-level lineage manifests.

Inputs are canonical JSONL envelopes.  This command performs no source cleaning
and writes no corpus text; it records identity and representation relationships
needed to stop replacement/resegmentation datasets from being double-added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from source_lineage import (
    build_registry_manifest,
    canonical_json,
    canonicalize_row,
    iter_lineage_rows,
    load_json,
    resolve_canonical_inputs,
    sha256_parts,
    write_json,
)


RELATIONSHIP_SCHEMA = "full_cpt_lineage_relationship_membership_v1"
SUMMARY_SCHEMA = "full_cpt_lineage_summary_v1"
ACTION_SCHEMA = "full_cpt_lineage_document_action_v1"
NOVELTY_SCHEMA = "full_cpt_source_novelty_v1"
CANDIDATE_ANALYSIS_TABLE = "lineage_candidate_analysis_rows"
CANDIDATE_REPRESENTATIVES_TABLE = "lineage_candidate_representatives"


def config_paths(parser: argparse.ArgumentParser, here: Path) -> None:
    parser.add_argument(
        "--sources-config", type=Path, default=here / "configs" / "sources.json"
    )
    parser.add_argument(
        "--roster-config",
        type=Path,
        default=here / "configs" / "nanochat_initial_roster.json",
    )
    parser.add_argument(
        "--aliases-config",
        type=Path,
        default=here / "configs" / "source_lineage_aliases.json",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def open_database(
    path: Path,
    *,
    resume: bool,
    contract_sha256: str,
    cache_mb: int,
) -> sqlite3.Connection:
    existed = path.exists()
    if existed and not resume:
        raise FileExistsError(f"refusing to overwrite SQLite work file: {path}")
    connection = sqlite3.connect(path)
    connection.executescript(
        f"""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-{max(16, cache_mb) * 1024};
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingested_inputs (
            path TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            binding_sha256 TEXT NOT NULL,
            rows INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rows (
            stable_uid TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_family_id TEXT NOT NULL,
            normalized_text_sha256 TEXT NOT NULL,
            work_key TEXT NOT NULL,
            representation_generation TEXT NOT NULL,
            identity_word_tokens INTEGER NOT NULL,
            content_relation TEXT,
            merge_policy TEXT,
            record_json TEXT NOT NULL
        );
        """
    )
    existing_contract = connection.execute(
        "SELECT value FROM metadata WHERE key = 'contract_sha256'"
    ).fetchone()
    if existing_contract is None:
        if existed and connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]:
            raise ValueError(
                f"{path}: existing lineage database has no resume contract"
            )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('contract_sha256', ?)",
            (contract_sha256,),
        )
        connection.commit()
    elif str(existing_contract[0]) != contract_sha256:
        connection.close()
        raise ValueError(f"{path}: lineage resume contract drift")
    return connection


def ensure_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS rows_exact_idx
            ON rows(normalized_text_sha256, stable_uid);
        CREATE INDEX IF NOT EXISTS rows_work_idx ON rows(work_key, stable_uid);
        CREATE INDEX IF NOT EXISTS rows_source_idx ON rows(source_id, stable_uid);
        """
    )
    connection.commit()


ROW_COLUMNS = (
    "stable_uid",
    "origin",
    "source_id",
    "source_dataset",
    "source_family_id",
    "normalized_text_sha256",
    "work_key",
    "representation_generation",
    "identity_word_tokens",
    "content_relation",
    "merge_policy",
    "record_json",
)


def fragment_directory(root: Path, origin: str, path: Path) -> Path:
    key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]
    return root / f"{origin}-{key}"


def validate_fragment_receipt(
    directory: Path,
    *,
    origin: str,
    path: Path,
    binding_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"incomplete lineage input fragment: {directory}")
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "full_cpt_lineage_input_fragment_v1",
        "origin": origin,
        "input_path": str(path.resolve()),
        "input_binding_sha256": binding_sha256,
        "lineage_contract_sha256": contract_sha256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"{receipt_path}: lineage fragment resume drift for {key}")
    fragment = directory / "fragment.sqlite"
    if (
        not fragment.is_file()
        or fragment.stat().st_size != int(value.get("fragment_bytes", -1))
        or sha256_file(fragment) != str(value.get("fragment_sha256", ""))
    ):
        raise ValueError(f"{receipt_path}: lineage fragment payload drift")
    return value


def prepare_input_fragment(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(payload["path"]).resolve()
    origin = str(payload["origin"])
    root = Path(payload["spool_root"]).resolve()
    final = fragment_directory(root, origin, path)
    binding_sha256 = str(payload["binding_sha256"])
    contract_sha256 = str(payload["contract_sha256"])
    if final.exists():
        return validate_fragment_receipt(
            final,
            origin=origin,
            path=path,
            binding_sha256=binding_sha256,
            contract_sha256=contract_sha256,
        )
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob(f".{final.name}.partial-*"):
        shutil.rmtree(stale, ignore_errors=True)
    temporary = root / f".{final.name}.partial-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    fragment = temporary / "fragment.sqlite"
    connection = sqlite3.connect(fragment)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE prepared (
            stable_uid TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_family_id TEXT NOT NULL,
            normalized_text_sha256 TEXT NOT NULL,
            work_key TEXT NOT NULL,
            representation_generation TEXT NOT NULL,
            identity_word_tokens INTEGER NOT NULL,
            content_relation TEXT,
            merge_policy TEXT,
            record_json TEXT NOT NULL
        );
        """
    )
    sources = payload["sources"]
    roster = payload["roster"]
    aliases = payload["aliases"]
    source_by_id = {
        str(row["source_id"]): row
        for row in sources.get("sources", [])
        if isinstance(row, dict) and row.get("source_id")
    }
    appearance_cache: dict[str, dict[str, Any]] = {}
    alias_cache: dict[tuple[str, str, str], str | None] = {}
    binding = payload.get("binding")
    bound_inputs = {path: binding} if isinstance(binding, dict) else {}
    pending: list[tuple[Any, ...]] = []
    count = 0
    line_number = 0
    try:
        for _, line_number, row, canonical_bound in iter_lineage_rows(
            [path], origin=origin, bound_inputs=bound_inputs
        ):
            interval = int(payload["verification_interval"])
            record = canonicalize_row(
                row,
                origin=origin,
                sources=sources,
                roster=roster,
                aliases=aliases,
                canonical_bound=canonical_bound,
                verify_bound=(
                    canonical_bound
                    and (
                        line_number == 1
                        or (interval > 0 and line_number % interval == 0)
                    )
                ),
                source_by_id=source_by_id,
                first_appearance_cache=appearance_cache,
                alias_cache=alias_cache,
            )
            pending.append(
                (
                    record["stable_uid"],
                    record["origin"],
                    record["source_id"],
                    record["source_dataset"],
                    record["source_family_id"],
                    record["normalized_text_sha256"],
                    record["work_key"],
                    record["representation_generation"],
                    int(record["identity_word_tokens"]),
                    record.get("content_relation"),
                    record.get("merge_policy"),
                    canonical_json(record),
                )
            )
            count += 1
            if len(pending) >= 8192:
                connection.executemany(
                    "INSERT INTO prepared VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    pending,
                )
                pending = []
        if pending:
            connection.executemany(
                "INSERT INTO prepared VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                pending,
            )
        connection.commit()
    except (ValueError, sqlite3.IntegrityError) as exc:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
    connection.close()
    value = {
        "schema_version": "full_cpt_lineage_input_fragment_v1",
        "origin": origin,
        "input_path": str(path),
        "input_binding_sha256": binding_sha256,
        "lineage_contract_sha256": contract_sha256,
        "rows": count,
        "fragment_bytes": fragment.stat().st_size,
        "fragment_sha256": sha256_file(fragment),
        "fragment_path": str(final / "fragment.sqlite"),
    }
    (temporary / "receipt.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, final)
    return value


def merge_fragment(
    connection: sqlite3.Connection,
    receipt: dict[str, Any],
) -> None:
    path = str(receipt["input_path"])
    alias = "fragment_db"
    fragment_path = str(receipt["fragment_path"])
    connection.execute(f"ATTACH DATABASE ? AS {alias}", (fragment_path,))
    try:
        connection.execute("BEGIN IMMEDIATE")
        columns = ", ".join(ROW_COLUMNS)
        connection.execute(
            f"INSERT INTO rows({columns}) SELECT {columns} FROM {alias}.prepared"
        )
        connection.execute(
            "INSERT INTO ingested_inputs(path, origin, binding_sha256, rows) "
            "VALUES (?, ?, ?, ?)",
            (
                path,
                receipt["origin"],
                receipt["input_binding_sha256"],
                int(receipt["rows"]),
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.execute(f"DETACH DATABASE {alias}")


def insert_inputs(
    connection: sqlite3.Connection,
    *,
    origin: str,
    paths: list[Path],
    sources: dict,
    roster: dict,
    aliases: dict,
    bound_inputs: dict[Path, dict[str, Any]],
    verification_interval: int,
    input_workers: int,
    spool_root: Path,
    contract_sha256: str,
) -> int:
    if not paths:
        return int(
            connection.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM ingested_inputs WHERE origin = ?",
                (origin,),
            ).fetchone()[0]
        )
    resolved = resolve_canonical_inputs(paths)
    payloads: list[dict[str, Any]] = []
    for path in resolved:
        binding = bound_inputs.get(path.resolve())
        binding_sha256 = (
            str(binding["sha256"]) if binding is not None else sha256_file(path)
        )
        previous = connection.execute(
            "SELECT origin, binding_sha256, rows FROM ingested_inputs WHERE path = ?",
            (str(path.resolve()),),
        ).fetchone()
        if previous is not None:
            if (str(previous[0]), str(previous[1])) != (origin, binding_sha256):
                raise ValueError(f"{path}: resume input binding drift")
            continue
        payloads.append(
            {
                "path": str(path.resolve()),
                "origin": origin,
                "binding": binding,
                "binding_sha256": binding_sha256,
                "contract_sha256": contract_sha256,
                "verification_interval": verification_interval,
                "sources": sources,
                "roster": roster,
                "aliases": aliases,
                "spool_root": str(spool_root.resolve()),
            }
        )

    def merge(receipt: dict[str, Any]) -> None:
        try:
            merge_fragment(connection, receipt)
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"{receipt['input_path']}: duplicate stable_uid while merging lineage"
            ) from exc
        shutil.rmtree(Path(str(receipt["fragment_path"])).parent)
        print(
            f"build_source_lineage: ingested origin={origin} "
            f"file={Path(receipt['input_path']).name} rows={int(receipt['rows']):,}",
            flush=True,
        )

    if input_workers == 1:
        for payload in payloads:
            merge(prepare_input_fragment(payload))
    elif payloads:
        with ProcessPoolExecutor(
            max_workers=min(input_workers, len(payloads))
        ) as executor:
            futures = [
                executor.submit(prepare_input_fragment, payload) for payload in payloads
            ]
            try:
                for future in as_completed(futures):
                    merge(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    return int(
        connection.execute(
            "SELECT COALESCE(SUM(rows), 0) FROM ingested_inputs WHERE origin = ?",
            (origin,),
        ).fetchone()[0]
    )


def write_rows(connection: sqlite3.Connection, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for (record_json,) in connection.execute(
            "SELECT record_json FROM rows ORDER BY stable_uid"
        ):
            line = record_json + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def relationship_groups(
    connection: sqlite3.Connection, kind: str
) -> Iterable[tuple[Any, ...]]:
    if kind == "exact":
        return connection.execute(
            """
            SELECT normalized_text_sha256, COUNT(*),
                   SUM(origin = 'base'), SUM(origin = 'candidate'),
                   COUNT(DISTINCT source_id), COUNT(DISTINCT work_key),
                   COUNT(DISTINCT representation_generation)
            FROM rows
            GROUP BY normalized_text_sha256
            HAVING COUNT(*) > 1
            ORDER BY normalized_text_sha256
            """
        )
    return connection.execute(
        """
        SELECT work_key, COUNT(*),
               SUM(origin = 'base'), SUM(origin = 'candidate'),
               COUNT(DISTINCT source_id), COUNT(DISTINCT normalized_text_sha256),
               COUNT(DISTINCT representation_generation)
        FROM rows
        GROUP BY work_key
        HAVING COUNT(*) > 1
        ORDER BY work_key
        """
    )


def relationship_type(
    kind: str,
    *,
    base_members: int,
    candidate_members: int,
    distinct_sources: int,
    distinct_content: int,
    distinct_generations: int,
) -> str:
    if kind == "exact":
        if base_members and candidate_members:
            return "base_candidate_exact_text"
        if distinct_sources > 1:
            return "cross_candidate_exact_text"
        return "within_representation_exact_text"
    if base_members and candidate_members:
        return "base_candidate_same_work_representation"
    if distinct_generations > 1 or distinct_content > 1:
        return "same_work_alternate_representation"
    return "within_representation_duplicate_work"


def write_relationships(connection: sqlite3.Connection, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    memberships = 0
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for kind in ("exact", "work"):
            key_field = "normalized_text_sha256" if kind == "exact" else "work_key"
            for (
                key,
                member_count,
                base_members,
                candidate_members,
                distinct_sources,
                distinct_content,
                distinct_generations,
            ) in relationship_groups(connection, kind):
                rel_type = relationship_type(
                    kind,
                    base_members=int(base_members),
                    candidate_members=int(candidate_members),
                    distinct_sources=int(distinct_sources),
                    distinct_content=int(distinct_content),
                    distinct_generations=int(distinct_generations),
                )
                relationship_id = sha256_parts(
                    "full_cpt_relationship_v1", rel_type, key
                )
                member_rows = connection.execute(
                    f"""
                    SELECT stable_uid, origin, source_id, source_dataset,
                           source_family_id, representation_generation
                    FROM rows WHERE {key_field} = ? ORDER BY stable_uid
                    """,
                    (key,),
                )
                for member in member_rows:
                    record = {
                        "schema_version": RELATIONSHIP_SCHEMA,
                        "relationship_id": relationship_id,
                        "relationship_type": rel_type,
                        "relationship_key": key,
                        "member_count": int(member_count),
                        "base_member_count": int(base_members),
                        "candidate_member_count": int(candidate_members),
                        "distinct_source_count": int(distinct_sources),
                        "distinct_content_count": int(distinct_content),
                        "distinct_representation_generation_count": int(
                            distinct_generations
                        ),
                        "stable_uid": member[0],
                        "origin": member[1],
                        "source_id": member[2],
                        "source_dataset": member[3],
                        "source_family_id": member[4],
                        "representation_generation": member[5],
                    }
                    line = canonical_json(record) + "\n"
                    handle.write(line)
                    digest.update(line.encode("utf-8"))
                    memberships += 1
                counts[rel_type] += 1
    return {
        "cluster_counts": dict(sorted(counts.items())),
        "membership_rows": memberships,
        "sha256": digest.hexdigest(),
    }


def drop_candidate_analysis_rows(connection: sqlite3.Connection) -> None:
    """Remove bounded TEMP state used by the post-ingest lineage reports."""

    connection.execute(
        f"DROP TABLE IF EXISTS temp.{CANDIDATE_REPRESENTATIVES_TABLE}"
    )
    connection.execute(f"DROP TABLE IF EXISTS temp.{CANDIDATE_ANALYSIS_TABLE}")


def prepare_candidate_analysis_rows(connection: sqlite3.Connection) -> None:
    """Project candidate report columns once instead of rescanning full rows.

    The production ``rows`` table carries the canonical JSON record and is much
    larger than the columns needed for novelty and source summaries.  Keeping a
    candidate-only TEMP projection makes all per-source aggregation bounded by
    candidate rows, while the original persistent database remains unchanged.
    """

    drop_candidate_analysis_rows(connection)
    print(
        "build_source_lineage: phase=candidate_analysis_projection status=started",
        flush=True,
    )
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE {CANDIDATE_ANALYSIS_TABLE} AS
            SELECT stable_uid, source_id, source_dataset, source_family_id,
                   normalized_text_sha256, work_key, identity_word_tokens
            FROM rows NOT INDEXED
            WHERE origin = 'candidate'
            """
        )
        print(
            "build_source_lineage: "
            "phase=candidate_analysis_projection status=rows_materialized",
            flush=True,
        )
        connection.execute(
            f"""
            CREATE UNIQUE INDEX temp.lineage_candidate_analysis_uid_idx
            ON {CANDIDATE_ANALYSIS_TABLE}(stable_uid)
            """
        )
        connection.execute(
            f"""
            CREATE INDEX temp.lineage_candidate_analysis_dataset_exact_idx
            ON {CANDIDATE_ANALYSIS_TABLE}(
                source_dataset, normalized_text_sha256, stable_uid
            )
            """
        )
        connection.execute(
            f"""
            CREATE INDEX temp.lineage_candidate_analysis_source_exact_idx
            ON {CANDIDATE_ANALYSIS_TABLE}(source_id, normalized_text_sha256)
            """
        )
        connection.execute(
            f"""
            CREATE INDEX temp.lineage_candidate_analysis_source_work_idx
            ON {CANDIDATE_ANALYSIS_TABLE}(source_id, work_key)
            """
        )
        print(
            "build_source_lineage: "
            "phase=candidate_analysis_projection status=indexes_materialized",
            flush=True,
        )
    except BaseException:
        drop_candidate_analysis_rows(connection)
        print(
            "build_source_lineage: "
            "phase=candidate_analysis_projection status=failed",
            flush=True,
        )
        raise
    print(
        "build_source_lineage: phase=candidate_analysis_projection status=completed",
        flush=True,
    )


def source_summaries(
    connection: sqlite3.Connection,
    registry: dict,
    *,
    candidate_analysis_ready: bool = False,
) -> list[dict[str, Any]]:
    """Summarize sources with a constant number of corpus-wide SQL queries."""

    owns_candidate_analysis = not candidate_analysis_ready
    if owns_candidate_analysis:
        prepare_candidate_analysis_rows(connection)
    print(
        "build_source_lineage: phase=source_summaries status=started",
        flush=True,
    )
    route_by_id = {
        entry["source_id"]: entry for entry in registry.get("candidates", [])
    }
    try:
        grouped_rows = list(
            connection.execute(
                """
                SELECT source_id, origin, COUNT(*),
                       COUNT(DISTINCT source_dataset),
                       COUNT(DISTINCT source_family_id)
                FROM rows NOT INDEXED
                WHERE origin = 'base'
                GROUP BY source_id, origin
                """
            )
        )
        grouped_rows.extend(
            connection.execute(
                f"""
                SELECT source_id, 'candidate', COUNT(*),
                       COUNT(DISTINCT source_dataset),
                       COUNT(DISTINCT source_family_id)
                FROM {CANDIDATE_ANALYSIS_TABLE}
                GROUP BY source_id
                """
            )
        )
        grouped_rows.sort(key=lambda row: (str(row[1]), str(row[0])))

        cross_exact_by_source = {
            str(source_id): int(clusters)
            for source_id, clusters in connection.execute(
                f"""
                SELECT candidate.source_id,
                       COUNT(DISTINCT candidate.normalized_text_sha256)
                FROM {CANDIDATE_ANALYSIS_TABLE} candidate
                WHERE EXISTS (
                    SELECT 1 FROM rows base
                    WHERE base.origin = 'base'
                      AND base.normalized_text_sha256 =
                          candidate.normalized_text_sha256
                )
                GROUP BY candidate.source_id
                ORDER BY candidate.source_id
                """
            )
        }
        print(
            "build_source_lineage: phase=source_summaries "
            "status=exact_overlap_completed",
            flush=True,
        )
        cross_work_by_source = {
            str(source_id): int(clusters)
            for source_id, clusters in connection.execute(
                f"""
                SELECT candidate.source_id,
                       COUNT(DISTINCT candidate.work_key)
                FROM {CANDIDATE_ANALYSIS_TABLE} candidate
                WHERE EXISTS (
                    SELECT 1 FROM rows base
                    WHERE base.origin = 'base'
                      AND base.work_key = candidate.work_key
                )
                GROUP BY candidate.source_id
                ORDER BY candidate.source_id
                """
            )
        }
        print(
            "build_source_lineage: phase=source_summaries "
            "status=work_overlap_completed",
            flush=True,
        )

        result: list[dict[str, Any]] = []
        for source_id, origin, rows, names, families in grouped_rows:
            source_id = str(source_id)
            origin = str(origin)
            cross_exact = cross_exact_by_source.get(source_id, 0)
            cross_work = cross_work_by_source.get(source_id, 0)
            route = route_by_id.get(source_id, {})
            reasons: list[str] = []
            if route.get("requires_base_identity_audit"):
                reasons.append("registry_requires_base_identity_audit")
            if route.get("requires_family_internal_dedup"):
                reasons.append("registry_requires_family_internal_dedup")
            if cross_exact:
                reasons.append("observed_base_candidate_exact_text")
            if cross_work:
                reasons.append("observed_base_candidate_work_match")
            if origin == "candidate" and not reasons:
                reasons.append("new_family_still_requires_global_exact_near_dedup")
            result.append(
                {
                    "source_id": source_id,
                    "origin": origin,
                    "rows": int(rows),
                    "distinct_source_dataset_names": int(names),
                    "distinct_source_families": int(families),
                    "base_candidate_exact_clusters": int(cross_exact),
                    "base_candidate_work_clusters": int(cross_work),
                    "blind_append_allowed": (
                        False if origin == "candidate" else None
                    ),
                    "double_add_hazard_reasons": reasons,
                }
            )
        print(
            "build_source_lineage: phase=source_summaries "
            f"status=completed sources={len(result):,}",
            flush=True,
        )
        return result
    finally:
        if owns_candidate_analysis:
            drop_candidate_analysis_rows(connection)


def resolve_document_actions(
    connection: sqlite3.Connection,
) -> dict[str, int]:
    """Resolve only relationships whose tracked policy is unambiguous.

    Exact candidate copies of Nanochat are dropped.  A reviewed same-source
    replacement/resegmentation also yields to the base representation when its
    work key matches.  Any remaining same-work group with different content is
    quarantined rather than silently selecting an arbitrary representation.
    """

    connection.execute("DROP TABLE IF EXISTS resolved_actions")
    connection.execute(
        """
        CREATE TEMP TABLE resolved_actions (
            stable_uid TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            input_text_sha256 TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            relationship_key TEXT NOT NULL,
            resolution_policy TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO resolved_actions
        SELECT candidate.stable_uid, candidate.source_id, candidate.source_dataset,
               candidate.normalized_text_sha256, 'drop',
               'lineage_exact_text_already_in_nanochat', candidate.work_key,
               'base_wins_exact_v1'
        FROM rows candidate
        WHERE candidate.origin = 'candidate'
          AND EXISTS (
              SELECT 1 FROM rows base
              WHERE base.origin = 'base'
                AND base.normalized_text_sha256 = candidate.normalized_text_sha256
          )
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO resolved_actions
        SELECT candidate.stable_uid, candidate.source_id, candidate.source_dataset,
               candidate.normalized_text_sha256, 'drop',
               'lineage_reviewed_replacement_work_already_in_nanochat',
               candidate.work_key, 'same_source_base_wins_work_v1'
        FROM rows candidate
        WHERE candidate.origin = 'candidate'
          AND candidate.content_relation LIKE 'same_source%'
          AND EXISTS (
              SELECT 1 FROM rows base
              WHERE base.origin = 'base' AND base.work_key = candidate.work_key
          )
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO resolved_actions
        SELECT candidate.stable_uid, candidate.source_id, candidate.source_dataset,
               candidate.normalized_text_sha256, 'quarantine',
               CASE WHEN EXISTS (
                   SELECT 1 FROM rows base
                   WHERE base.work_key = candidate.work_key AND base.origin = 'base'
               ) THEN 'lineage_unresolved_base_candidate_same_work'
                 ELSE 'lineage_unresolved_cross_candidate_same_work' END,
               candidate.work_key, 'fail_closed_same_work_alternate_content_v1'
        FROM rows candidate
        WHERE candidate.origin = 'candidate'
          AND EXISTS (
              SELECT 1 FROM rows sibling
              WHERE sibling.work_key = candidate.work_key
                AND sibling.normalized_text_sha256 != candidate.normalized_text_sha256
          )
        """
    )
    connection.execute(
        "CREATE INDEX resolved_actions_action ON resolved_actions(action)"
    )
    counts = Counter(
        {
            f"{action}:{reason}": int(rows)
            for action, reason, rows in connection.execute(
                "SELECT action, reason, COUNT(*) FROM resolved_actions "
                "GROUP BY action, reason ORDER BY action, reason"
            )
        }
    )
    counts["total_actions"] = int(
        connection.execute("SELECT COUNT(*) FROM resolved_actions").fetchone()[0]
    )
    counts["unresolved_quarantines"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM resolved_actions WHERE action = 'quarantine'"
        ).fetchone()[0]
    )
    return dict(sorted(counts.items()))


def write_actions(connection: sqlite3.Connection, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for values in connection.execute(
            """
            SELECT stable_uid, source_id, source_dataset, input_text_sha256,
                   action, reason, relationship_key, resolution_policy
            FROM resolved_actions ORDER BY stable_uid
            """
        ):
            row = {
                "schema_version": ACTION_SCHEMA,
                "stable_uid": values[0],
                "source_id": values[1],
                "source_dataset": values[2],
                "input_text_sha256": values[3],
                "action": values[4],
                "reason": values[5],
                "relationship_key": values[6],
                "resolution_policy": values[7],
            }
            line = canonical_json(row) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def source_novelty(
    connection: sqlite3.Connection,
    *,
    candidate_analysis_ready: bool = False,
) -> list[dict[str, Any]]:
    """Measure exact-lineage novelty with a deterministic word-token proxy.

    The numerator deduplicates exact text within each exact source_dataset and
    excludes every base-overlap or unresolved work action.  Near-duplicate
    removal remains the later global MinHash stage and is deliberately not
    overstated here.
    """

    owns_candidate_analysis = not candidate_analysis_ready
    if owns_candidate_analysis:
        prepare_candidate_analysis_rows(connection)
    connection.execute(
        f"DROP TABLE IF EXISTS temp.{CANDIDATE_REPRESENTATIVES_TABLE}"
    )
    print(
        "build_source_lineage: phase=source_novelty status=started",
        flush=True,
    )
    try:
        totals = list(
            connection.execute(
                f"""
                SELECT source_dataset, COUNT(*), SUM(identity_word_tokens),
                       GROUP_CONCAT(DISTINCT source_id)
                FROM {CANDIDATE_ANALYSIS_TABLE}
                GROUP BY source_dataset
                ORDER BY source_dataset
                """
            )
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE {CANDIDATE_REPRESENTATIVES_TABLE} AS
            WITH representative_ids AS (
                SELECT source_dataset, normalized_text_sha256,
                       MIN(stable_uid) AS stable_uid
                FROM {CANDIDATE_ANALYSIS_TABLE}
                GROUP BY source_dataset, normalized_text_sha256
            )
            SELECT representative.source_dataset,
                   representative.normalized_text_sha256,
                   representative.stable_uid,
                   candidate.identity_word_tokens
            FROM representative_ids representative
            JOIN {CANDIDATE_ANALYSIS_TABLE} candidate
              ON candidate.source_dataset = representative.source_dataset
             AND candidate.normalized_text_sha256 =
                 representative.normalized_text_sha256
             AND candidate.stable_uid = representative.stable_uid
            """
        )
        print(
            "build_source_lineage: phase=source_novelty "
            "status=representatives_materialized",
            flush=True,
        )
        unique_by_dataset = {
            str(source_dataset): (
                int(unique_rows),
                int(unique_tokens or 0),
                int(novel_rows or 0),
                int(novel_tokens or 0),
            )
            for (
                source_dataset,
                unique_rows,
                unique_tokens,
                novel_rows,
                novel_tokens,
            ) in connection.execute(
                f"""
                SELECT representative.source_dataset,
                       COUNT(*),
                       COALESCE(SUM(representative.identity_word_tokens), 0),
                       COALESCE(SUM(
                           CASE WHEN action.stable_uid IS NULL THEN 1 ELSE 0 END
                       ), 0),
                       COALESCE(SUM(
                           CASE WHEN action.stable_uid IS NULL
                                THEN representative.identity_word_tokens ELSE 0 END
                       ), 0)
                FROM {CANDIDATE_REPRESENTATIVES_TABLE} representative
                LEFT JOIN resolved_actions action
                  ON action.stable_uid = representative.stable_uid
                GROUP BY representative.source_dataset
                ORDER BY representative.source_dataset
                """
            )
        }
        actions_by_dataset: dict[str, dict[str, int]] = {}
        for source_dataset, action, rows in connection.execute(
            f"""
            SELECT candidate.source_dataset, action.action, COUNT(*)
            FROM resolved_actions action
            JOIN {CANDIDATE_ANALYSIS_TABLE} candidate
              ON candidate.stable_uid = action.stable_uid
            GROUP BY candidate.source_dataset, action.action
            ORDER BY candidate.source_dataset, action.action
            """
        ):
            actions_by_dataset.setdefault(str(source_dataset), {})[str(action)] = int(
                rows
            )

        result: list[dict[str, Any]] = []
        for source_dataset, rows, total_tokens, source_ids in totals:
            source_dataset = str(source_dataset)
            if source_dataset not in unique_by_dataset:
                raise ValueError(
                    f"candidate novelty representatives missing {source_dataset!r}"
                )
            unique_rows, unique_tokens, novel_rows, novel_tokens = unique_by_dataset[
                source_dataset
            ]
            total_tokens = int(total_tokens or 0)
            result.append(
                {
                    "source_dataset": source_dataset,
                    "source_ids": sorted(str(source_ids or "").split(",")),
                    "rows": int(rows),
                    "identity_word_tokens": total_tokens,
                    "exact_unique_rows": unique_rows,
                    "exact_unique_word_tokens": unique_tokens,
                    "novel_rows_after_lineage_resolution": novel_rows,
                    "novel_word_tokens_after_lineage_resolution": novel_tokens,
                    "novel_token_fraction": (
                        round(novel_tokens / total_tokens, 8)
                        if total_tokens
                        else 0.0
                    ),
                    "document_action_counts": actions_by_dataset.get(
                        source_dataset, {}
                    ),
                }
            )
        print(
            "build_source_lineage: phase=source_novelty status=completed "
            f"source_datasets={len(result):,} "
            f"candidate_rows={sum(int(row[1]) for row in totals):,} "
            f"representatives={sum(row[0] for row in unique_by_dataset.values()):,}",
            flush=True,
        )
        return result
    finally:
        connection.execute(
            f"DROP TABLE IF EXISTS temp.{CANDIDATE_REPRESENTATIVES_TABLE}"
        )
        if owns_candidate_analysis:
            drop_candidate_analysis_rows(connection)


def normalization_bindings(path: Path | None) -> dict[Path, dict[str, Any]]:
    if path is None:
        return {}
    manifest = load_json(path)
    if manifest.get("schema_version") != "full_cpt_normalization_manifest_v1":
        raise ValueError(f"{path}: unsupported normalization manifest schema")
    result: dict[Path, dict[str, Any]] = {}
    for source in manifest.get("sources", []):
        if not isinstance(source, dict):
            raise ValueError(f"{path}: normalization source entry must be an object")
        source_id = str(source.get("source_id") or "")
        for shard in source.get("shards", []):
            if not isinstance(shard, dict):
                raise ValueError(f"{path}: normalization shard entry must be an object")
            resolved = Path(str(shard.get("path", ""))).resolve()
            entry = {
                "path": str(resolved),
                "bytes": int(shard.get("bytes", -1)),
                "sha256": str(shard.get("sha256", "")),
                "rows": int(shard.get("rows", -1)),
                "source_id": source_id,
            }
            if (
                not resolved.is_file()
                or resolved.stat().st_size != entry["bytes"]
                or len(entry["sha256"]) != 64
                or entry["rows"] < 1
            ):
                raise ValueError(
                    f"{path}: invalid or drifted normalized shard {resolved}"
                )
            if resolved in result:
                raise ValueError(f"{path}: duplicate normalized shard {resolved}")
            result[resolved] = entry
    if not result:
        raise ValueError(f"{path}: normalization manifest has no canonical shards")
    return result


def validate_normalization_inventory_receipt(
    manifest_path: Path | None, receipt_path: Path | None
) -> dict[str, Any] | None:
    if manifest_path is None:
        if receipt_path is not None:
            raise ValueError("normalization inventory validation requires a manifest")
        return None
    if receipt_path is None:
        raise ValueError(
            "receipt-bound canonical fast path requires --normalization-inventory-validation"
        )
    value = load_json(receipt_path)
    if value.get("schema_version") != "full_cpt_shard_inventory_validation_v1":
        raise ValueError(f"{receipt_path}: unsupported inventory validation schema")
    if Path(str(value.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise ValueError(f"{receipt_path}: normalization manifest path drift")
    if str(value.get("manifest_sha256")) != sha256_file(manifest_path):
        raise ValueError(f"{receipt_path}: normalization manifest checksum drift")
    if (
        int(value.get("files", 0)) < 1
        or len(str(value.get("inventory_sha256", ""))) != 64
    ):
        raise ValueError(f"{receipt_path}: invalid normalized inventory attestation")
    return value


def lineage_contract(
    args: argparse.Namespace,
    *,
    base_paths: list[Path],
    candidate_paths: list[Path],
    bindings: dict[Path, dict[str, Any]],
) -> str:
    inputs = []
    for origin, paths in (("base", base_paths), ("candidate", candidate_paths)):
        for path in paths:
            binding = bindings.get(path.resolve())
            inputs.append(
                {
                    "origin": origin,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": str(binding["sha256"]) if binding else sha256_file(path),
                    "rows": int(binding["rows"]) if binding else None,
                }
            )
    value = {
        "schema_version": "full_cpt_lineage_build_contract_v3",
        "sources_config_sha256": sha256_file(args.sources_config),
        "roster_config_sha256": sha256_file(args.roster_config),
        "aliases_config_sha256": sha256_file(args.aliases_config),
        "normalization_manifest_sha256": (
            sha256_file(args.normalization_manifest)
            if args.normalization_manifest
            else None
        ),
        "normalization_inventory_validation_sha256": (
            sha256_file(args.normalization_inventory_validation)
            if args.normalization_inventory_validation
            else None
        ),
        "inputs": sorted(inputs, key=lambda row: (row["origin"], row["path"])),
        "canonical_verification_interval": args.canonical_verification_interval,
        "debug_exports": {
            "rows": args.rows_out is not None,
            "relationships": args.relationships_out is not None,
        },
    }
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_rows(
    args: argparse.Namespace, sources: dict, roster: dict, aliases: dict
) -> int:
    if not args.base_jsonl and not args.candidate_jsonl:
        raise ValueError("at least one --base-jsonl or --candidate-jsonl is required")
    registry = build_registry_manifest(sources, roster, aliases)
    write_json(args.registry_manifest_out, registry)

    base_paths = resolve_canonical_inputs(args.base_jsonl) if args.base_jsonl else []
    candidate_paths = (
        resolve_canonical_inputs(args.candidate_jsonl) if args.candidate_jsonl else []
    )
    bindings = normalization_bindings(args.normalization_manifest)
    inventory_validation = validate_normalization_inventory_receipt(
        args.normalization_manifest, args.normalization_inventory_validation
    )
    supplied_paths = {path.resolve() for path in [*base_paths, *candidate_paths]}
    if bindings and supplied_paths != set(bindings):
        missing = sorted(str(path) for path in set(bindings) - supplied_paths)
        unexpected = sorted(str(path) for path in supplied_paths - set(bindings))
        raise ValueError(
            "lineage canonical inputs differ from normalization manifest; "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    for path in base_paths:
        if bindings and bindings[path.resolve()]["source_id"] != "nanochat_base":
            raise ValueError(f"{path}: bound base input is not nanochat_base")
    for path in candidate_paths:
        if bindings and bindings[path.resolve()]["source_id"] == "nanochat_base":
            raise ValueError(f"{path}: bound candidate input is nanochat_base")
    contract_sha256 = lineage_contract(
        args,
        base_paths=base_paths,
        candidate_paths=candidate_paths,
        bindings=bindings,
    )

    temporary = args.sqlite_work_path is None
    if temporary:
        fd, raw_path = tempfile.mkstemp(prefix="full-cpt-lineage-", suffix=".sqlite")
        os.close(fd)
        database_path = Path(raw_path)
        database_path.unlink()
    else:
        database_path = args.sqlite_work_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
    if args.sqlite_temp_directory:
        args.sqlite_temp_directory.mkdir(parents=True, exist_ok=True)
        os.environ["SQLITE_TMPDIR"] = str(args.sqlite_temp_directory.resolve())
    connection = open_database(
        database_path,
        resume=args.resume,
        contract_sha256=contract_sha256,
        cache_mb=args.sqlite_cache_mb,
    )
    spool_root = (
        args.input_spool_directory.resolve()
        if args.input_spool_directory
        else database_path.parent / "lineage-input-fragments"
    )
    try:
        base_rows = insert_inputs(
            connection,
            origin="base",
            paths=base_paths,
            sources=sources,
            roster=roster,
            aliases=aliases,
            bound_inputs=bindings,
            verification_interval=args.canonical_verification_interval,
            input_workers=args.input_workers,
            spool_root=spool_root,
            contract_sha256=contract_sha256,
        )
        candidate_rows = insert_inputs(
            connection,
            origin="candidate",
            paths=candidate_paths,
            sources=sources,
            roster=roster,
            aliases=aliases,
            bound_inputs=bindings,
            verification_interval=args.canonical_verification_interval,
            input_workers=args.input_workers,
            spool_root=spool_root,
            contract_sha256=contract_sha256,
        )
        ensure_indexes(connection)
        row_manifest = None
        if args.rows_out is not None:
            rows_sha256 = write_rows(connection, args.rows_out)
            row_manifest = {"path": str(args.rows_out), "sha256": rows_sha256}
        relationship_manifest = None
        if args.relationships_out is not None:
            relationships = write_relationships(connection, args.relationships_out)
            relationship_manifest = {
                "path": str(args.relationships_out),
                **relationships,
            }
        action_counts = resolve_document_actions(connection)
        actions_sha256 = write_actions(connection, args.actions_out)
        print(
            "build_source_lineage: phase=document_actions status=completed "
            f"rows={action_counts.get('total_actions', 0):,}",
            flush=True,
        )
        prepare_candidate_analysis_rows(connection)
        try:
            novelty_sources = source_novelty(
                connection, candidate_analysis_ready=True
            )
            summary_sources = source_summaries(
                connection, registry, candidate_analysis_ready=True
            )
        finally:
            drop_candidate_analysis_rows(connection)
        novelty_payload = {
            "schema_version": NOVELTY_SCHEMA,
            "method": "exact_normalized_text_plus_reviewed_same_work_word_token_proxy_v1",
            "near_duplicate_novelty_deferred_to_global_dedup": True,
            "document_actions": {
                "path": str(args.actions_out),
                "sha256": actions_sha256,
                "counts": action_counts,
            },
            "sources": novelty_sources,
        }
        write_json(args.novelty_out, novelty_payload)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "nanochat_anchor_revision": roster.get("repository", {}).get(
                "first_data_revision"
            ),
            "base_rows": base_rows,
            "candidate_rows": candidate_rows,
            "total_rows": base_rows + candidate_rows,
            "debug_exports_enabled": bool(
                args.rows_out is not None or args.relationships_out is not None
            ),
            "row_manifest": row_manifest,
            "relationship_manifest": relationship_manifest,
            "sources": summary_sources,
            "document_actions": {
                "path": str(args.actions_out),
                "sha256": actions_sha256,
                "counts": action_counts,
            },
            "source_novelty": {
                "path": str(args.novelty_out),
                "sha256": hashlib.sha256(args.novelty_out.read_bytes()).hexdigest(),
                "method": novelty_payload["method"],
            },
            "unresolved_relationships_quarantined": action_counts.get(
                "unresolved_quarantines", 0
            ),
            "blind_append_allowed": False,
            "lineage_contract_sha256": contract_sha256,
            "normalization_manifest": (
                {
                    "path": str(args.normalization_manifest.resolve()),
                    "sha256": sha256_file(args.normalization_manifest),
                }
                if args.normalization_manifest
                else None
            ),
            "normalization_inventory_validation": (
                {
                    "path": str(args.normalization_inventory_validation.resolve()),
                    "sha256": sha256_file(args.normalization_inventory_validation),
                    "inventory_sha256": inventory_validation["inventory_sha256"],
                }
                if inventory_validation is not None
                else None
            ),
        }
        write_json(args.summary_out, summary)
        print(
            "build_source_lineage: phase=lineage_reports status=completed",
            flush=True,
        )
    finally:
        connection.close()
        if temporary:
            for suffix in ("", "-wal", "-shm"):
                Path(str(database_path) + suffix).unlink(missing_ok=True)
    return 0


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry_parser = subparsers.add_parser("registry")
    config_paths(registry_parser, here)
    registry_parser.add_argument("--output", type=Path, required=True)

    rows_parser = subparsers.add_parser("rows")
    config_paths(rows_parser, here)
    rows_parser.add_argument(
        "--base-jsonl",
        "--base-input",
        dest="base_jsonl",
        action="append",
        type=Path,
        default=[],
        help="canonical JSONL, Parquet file, or sharded Parquet root (repeatable)",
    )
    rows_parser.add_argument(
        "--candidate-jsonl",
        "--candidate-input",
        dest="candidate_jsonl",
        action="append",
        type=Path,
        default=[],
        help="canonical JSONL, Parquet file, or sharded Parquet root (repeatable)",
    )
    rows_parser.add_argument("--registry-manifest-out", type=Path, required=True)
    rows_parser.add_argument("--rows-out", type=Path)
    rows_parser.add_argument("--relationships-out", type=Path)
    rows_parser.add_argument("--actions-out", type=Path, required=True)
    rows_parser.add_argument("--novelty-out", type=Path, required=True)
    rows_parser.add_argument("--summary-out", type=Path, required=True)
    rows_parser.add_argument("--sqlite-work-path", type=Path)
    rows_parser.add_argument("--sqlite-temp-directory", type=Path)
    rows_parser.add_argument("--sqlite-cache-mb", type=int, default=4096)
    rows_parser.add_argument(
        "--input-workers", type=int, default=min(8, os.cpu_count() or 1)
    )
    rows_parser.add_argument("--input-spool-directory", type=Path)
    rows_parser.add_argument("--normalization-manifest", type=Path)
    rows_parser.add_argument("--normalization-inventory-validation", type=Path)
    rows_parser.add_argument(
        "--canonical-verification-interval", type=int, default=100_000
    )
    rows_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.command == "rows":
        if args.sqlite_cache_mb < 16:
            parser.error("--sqlite-cache-mb must be at least 16")
        if args.input_workers < 1:
            parser.error("--input-workers must be positive")
        if args.canonical_verification_interval < 0:
            parser.error("--canonical-verification-interval must be non-negative")

    sources = load_json(args.sources_config)
    roster = load_json(args.roster_config)
    aliases = load_json(args.aliases_config)
    if args.command == "registry":
        write_json(args.output, build_registry_manifest(sources, roster, aliases))
        return 0
    return build_rows(args, sources, roster, aliases)


if __name__ == "__main__":
    raise SystemExit(main())
