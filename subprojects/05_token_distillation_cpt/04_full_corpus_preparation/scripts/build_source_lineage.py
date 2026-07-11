#!/usr/bin/env python3
"""Build deterministic source- and row-level lineage manifests.

Inputs are canonical JSONL envelopes.  This command performs no source cleaning
and writes no corpus text; it records identity and representation relationships
needed to stop replacement/resegmentation datasets from being double-added.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from source_lineage import (
    build_registry_manifest,
    canonical_json,
    canonicalize_row,
    iter_jsonl,
    load_json,
    sha256_parts,
    write_json,
)


RELATIONSHIP_SCHEMA = "full_cpt_lineage_relationship_membership_v1"
SUMMARY_SCHEMA = "full_cpt_lineage_summary_v1"


def config_paths(parser: argparse.ArgumentParser, here: Path) -> None:
    parser.add_argument("--sources-config", type=Path, default=here / "configs" / "sources.json")
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


def open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE rows (
            stable_uid TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_family_id TEXT NOT NULL,
            normalized_text_sha256 TEXT NOT NULL,
            work_key TEXT NOT NULL,
            representation_generation TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE INDEX rows_exact_idx ON rows(normalized_text_sha256, stable_uid);
        CREATE INDEX rows_work_idx ON rows(work_key, stable_uid);
        CREATE INDEX rows_source_idx ON rows(source_id, stable_uid);
        """
    )
    return connection


def insert_inputs(
    connection: sqlite3.Connection,
    *,
    origin: str,
    paths: list[Path],
    sources: dict,
    roster: dict,
    aliases: dict,
) -> int:
    count = 0
    for path, line_number, row in iter_jsonl(paths):
        try:
            record = canonicalize_row(
                row,
                origin=origin,
                sources=sources,
                roster=roster,
                aliases=aliases,
            )
            connection.execute(
                """
                INSERT INTO rows(
                    stable_uid, origin, source_id, source_dataset, source_family_id,
                    normalized_text_sha256, work_key, representation_generation, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["stable_uid"],
                    record["origin"],
                    record["source_id"],
                    record["source_dataset"],
                    record["source_family_id"],
                    record["normalized_text_sha256"],
                    record["work_key"],
                    record["representation_generation"],
                    canonical_json(record),
                ),
            )
        except (ValueError, sqlite3.IntegrityError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        count += 1
        if count % 100_000 == 0:
            connection.commit()
    connection.commit()
    return count


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


def relationship_groups(connection: sqlite3.Connection, kind: str) -> Iterable[tuple[Any, ...]]:
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
                relationship_id = sha256_parts("full_cpt_relationship_v1", rel_type, key)
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
                        "distinct_representation_generation_count": int(distinct_generations),
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


def source_summaries(connection: sqlite3.Connection, registry: dict) -> list[dict[str, Any]]:
    route_by_id = {
        entry["source_id"]: entry for entry in registry.get("candidates", [])
    }
    result: list[dict[str, Any]] = []
    for source_id, origin, rows, names, families in connection.execute(
        """
        SELECT source_id, origin, COUNT(*), COUNT(DISTINCT source_dataset),
               COUNT(DISTINCT source_family_id)
        FROM rows GROUP BY source_id, origin ORDER BY origin, source_id
        """
    ):
        cross_exact = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT DISTINCT candidate.normalized_text_sha256
                FROM rows candidate
                WHERE candidate.source_id = ? AND candidate.origin = 'candidate'
                  AND EXISTS (
                      SELECT 1 FROM rows base
                      WHERE base.origin = 'base'
                        AND base.normalized_text_sha256 = candidate.normalized_text_sha256
                  )
            )
            """,
            (source_id,),
        ).fetchone()[0]
        cross_work = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT DISTINCT candidate.work_key
                FROM rows candidate
                WHERE candidate.source_id = ? AND candidate.origin = 'candidate'
                  AND EXISTS (
                      SELECT 1 FROM rows base
                      WHERE base.origin = 'base' AND base.work_key = candidate.work_key
                  )
            )
            """,
            (source_id,),
        ).fetchone()[0]
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
                "blind_append_allowed": False if origin == "candidate" else None,
                "double_add_hazard_reasons": reasons,
            }
        )
    return result


def build_rows(args: argparse.Namespace, sources: dict, roster: dict, aliases: dict) -> int:
    if not args.base_jsonl and not args.candidate_jsonl:
        raise ValueError("at least one --base-jsonl or --candidate-jsonl is required")
    registry = build_registry_manifest(sources, roster, aliases)
    write_json(args.registry_manifest_out, registry)

    temporary = args.sqlite_work_path is None
    if temporary:
        fd, raw_path = tempfile.mkstemp(prefix="full-cpt-lineage-", suffix=".sqlite")
        os.close(fd)
        database_path = Path(raw_path)
    else:
        database_path = args.sqlite_work_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        if database_path.exists():
            raise FileExistsError(f"refusing to overwrite SQLite work file: {database_path}")
    connection = open_database(database_path)
    try:
        base_rows = insert_inputs(
            connection,
            origin="base",
            paths=args.base_jsonl,
            sources=sources,
            roster=roster,
            aliases=aliases,
        )
        candidate_rows = insert_inputs(
            connection,
            origin="candidate",
            paths=args.candidate_jsonl,
            sources=sources,
            roster=roster,
            aliases=aliases,
        )
        rows_sha256 = write_rows(connection, args.rows_out)
        relationships = write_relationships(connection, args.relationships_out)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "nanochat_anchor_revision": roster.get("repository", {}).get("first_data_revision"),
            "base_rows": base_rows,
            "candidate_rows": candidate_rows,
            "total_rows": base_rows + candidate_rows,
            "row_manifest": {"path": str(args.rows_out), "sha256": rows_sha256},
            "relationship_manifest": {
                "path": str(args.relationships_out),
                **relationships,
            },
            "sources": source_summaries(connection, registry),
            "blind_append_allowed": False,
        }
        write_json(args.summary_out, summary)
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
    rows_parser.add_argument("--base-jsonl", action="append", type=Path, default=[])
    rows_parser.add_argument("--candidate-jsonl", action="append", type=Path, default=[])
    rows_parser.add_argument("--registry-manifest-out", type=Path, required=True)
    rows_parser.add_argument("--rows-out", type=Path, required=True)
    rows_parser.add_argument("--relationships-out", type=Path, required=True)
    rows_parser.add_argument("--summary-out", type=Path, required=True)
    rows_parser.add_argument("--sqlite-work-path", type=Path)
    args = parser.parse_args()

    sources = load_json(args.sources_config)
    roster = load_json(args.roster_config)
    aliases = load_json(args.aliases_config)
    if args.command == "registry":
        write_json(args.output, build_registry_manifest(sources, roster, aliases))
        return 0
    return build_rows(args, sources, roster, aliases)


if __name__ == "__main__":
    raise SystemExit(main())
