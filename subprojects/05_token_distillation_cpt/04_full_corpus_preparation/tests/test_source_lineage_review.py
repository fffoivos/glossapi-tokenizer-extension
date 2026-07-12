from __future__ import annotations

import json
import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_source_reviews as AGGREGATE  # noqa: E402
import build_source_lineage as BUILD  # noqa: E402
import full_corpus_io as CORPUS_IO  # noqa: E402
import source_lineage as LINEAGE  # noqa: E402


def configs() -> tuple[dict, dict, dict, dict]:
    return (
        LINEAGE.load_json(HERE / "configs" / "sources.json"),
        LINEAGE.load_json(HERE / "configs" / "nanochat_initial_roster.json"),
        LINEAGE.load_json(HERE / "configs" / "source_lineage_aliases.json"),
        LINEAGE.load_json(HERE / "configs" / "source_review_policy.json"),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def test_canonical_iterator_streams_sharded_parquet_root(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "canonical"
    root.mkdir()
    pq.write_table(
        pa.table({"source_id": ["a", "b"], "text": ["ένα", "δύο"]}),
        root / "part.parquet",
    )
    rows = list(LINEAGE.iter_jsonl([root]))
    assert [row[2]["source_id"] for row in rows] == ["a", "b"]
    assert [row[1] for row in rows] == [1, 2]


def test_first_appearance_is_anchored_to_first_data_revision() -> None:
    _, roster, _, _ = configs()
    initial = LINEAGE.first_appearance("greek_phd", roster)
    later = LINEAGE.first_appearance("HPLT/ell_Grek_ge8_no_mt_clean60", roster)
    new = LINEAGE.first_appearance("glossAPI/diavgeia", roster)
    assert initial == {
        "cohort": "nanochat_first_data_revision",
        "revision": "500b8bf577e1e70f4902b77edce2cda02a2559cb",
        "committed_at": "2026-03-16T16:40:36.000Z",
        "anchor_revision": "500b8bf577e1e70f4902b77edce2cda02a2559cb",
    }
    assert later["cohort"] == "nanochat_later_source_name_addition"
    assert new["cohort"] == "not_present_at_nanochat_anchor"


def test_registry_manifest_never_authorizes_blind_append() -> None:
    sources, roster, aliases, _ = configs()
    manifest = LINEAGE.build_registry_manifest(sources, roster, aliases)
    routes = {entry["source_id"]: entry for entry in manifest["candidates"]}
    assert len(routes) == 23
    assert all(route["blind_append_allowed"] is False for route in routes.values())
    assert routes["opengov_deliberations_v2"]["requires_base_identity_audit"] is True
    assert (
        routes["school_books_new_editions"]["reviewed_aliases"][0]["alias_kind"]
        == "hybrid"
    )
    assert routes["diavgeia"]["fallback_first_appearance"]["cohort"] == (
        "not_present_at_nanochat_anchor"
    )


def test_canonical_lineage_preserves_exact_name_and_stable_identity_across_cleaning() -> (
    None
):
    sources, roster, aliases, _ = configs()
    row = {
        "source_id": "archetai",
        "source_dataset": "Exact Upstream Name / Do Not Normalize",
        "source_artifact_path": "part-000.parquet",
        "source_row_id": "17",
        "source_doc_id": "https://EXAMPLE.org/work/17/#fragment",
        "text": "Κείμενο με κενά   \r\n",
    }
    first = LINEAGE.canonicalize_row(
        row, origin="candidate", sources=sources, roster=roster, aliases=aliases
    )
    changed = LINEAGE.canonicalize_row(
        {**row, "text": "Καθαρισμένο κείμενο"},
        origin="candidate",
        sources=sources,
        roster=roster,
        aliases=aliases,
    )
    assert first["source_dataset"] == row["source_dataset"]
    assert first["source_dataset_origin"] == "preserved_upstream_value"
    assert first["stable_uid"] == changed["stable_uid"]
    assert first["work_key"] == changed["work_key"]
    assert first["original_text_sha256"] != changed["original_text_sha256"]
    assert first["first_appearance"]["cohort"] == "not_present_at_nanochat_anchor"


def test_missing_source_name_uses_pinned_repo_and_resegmentation_requires_work_id() -> (
    None
):
    sources, roster, aliases, _ = configs()
    fallback = LINEAGE.canonicalize_row(
        {
            "source_id": "archetai",
            "source_artifact_path": "data.parquet",
            "source_row_id": "1",
            "source_doc_id": "doc-1",
            "text": "κείμενο",
        },
        origin="candidate",
        sources=sources,
        roster=roster,
        aliases=aliases,
    )
    assert fallback["source_dataset"] == "glossAPI/archetai"
    assert fallback["source_dataset_origin"] == "pinned_repo_fallback"

    with pytest.raises(ValueError, match="require an explicit work_id"):
        LINEAGE.canonicalize_row(
            {
                "source_id": "pergamos_sections",
                "source_artifact_path": "sections.parquet",
                "source_row_id": "1",
                "source_doc_id": "work-1-section-1",
                "text": "ενότητα",
            },
            origin="candidate",
            sources=sources,
            roster=roster,
            aliases=aliases,
        )


def test_lineage_cli_detects_base_candidate_exact_and_work_relationships(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        base,
        [
            {
                "source_dataset": "opengov.gr-diaboyleuseis",
                "source_family_id": "opengov_deliberations",
                "source_artifact_path": "data/opengov.parquet",
                "source_row_id": "base-1",
                "source_doc_id": "consultation-1",
                "work_id": "consultation-1",
                "text": "Ίδιο κείμενο\n",
            }
        ],
    )
    write_jsonl(
        candidate,
        [
            {
                "source_id": "opengov_deliberations_v2",
                "source_dataset": "opengov.gr-diaboyleuseis",
                "source_artifact_path": "opengov_v2.parquet",
                "source_row_id": "candidate-1",
                "source_doc_id": "consultation-1",
                "work_id": "consultation-1",
                "text": "Ίδιο κείμενο",
            }
        ],
    )
    registry = tmp_path / "registry.json"
    rows = tmp_path / "rows.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    actions = tmp_path / "actions.jsonl"
    novelty = tmp_path / "novelty.json"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_source_lineage.py"),
            "rows",
            "--base-jsonl",
            str(base),
            "--candidate-jsonl",
            str(candidate),
            "--registry-manifest-out",
            str(registry),
            "--rows-out",
            str(rows),
            "--relationships-out",
            str(relationships),
            "--actions-out",
            str(actions),
            "--novelty-out",
            str(novelty),
            "--summary-out",
            str(summary),
        ],
        check=True,
    )
    relations = [json.loads(line) for line in relationships.read_text().splitlines()]
    assert {record["relationship_type"] for record in relations} == {
        "base_candidate_exact_text",
        "base_candidate_same_work_representation",
    }
    candidate_summary = next(
        entry
        for entry in json.loads(summary.read_text())["sources"]
        if entry["source_id"] == "opengov_deliberations_v2"
    )
    assert candidate_summary["base_candidate_exact_clusters"] == 1
    assert candidate_summary["base_candidate_work_clusters"] == 1
    assert (
        "observed_base_candidate_exact_text"
        in candidate_summary["double_add_hazard_reasons"]
    )
    action = json.loads(actions.read_text())
    assert action["action"] == "drop"
    assert action["reason"] == "lineage_exact_text_already_in_nanochat"
    novelty_row = json.loads(novelty.read_text())["sources"][0]
    assert novelty_row["novel_token_fraction"] == 0.0


def test_lineage_cli_does_not_emit_large_debug_exports_by_default(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(
        candidate,
        [
            {
                "source_id": "archetai",
                "source_dataset": "glossAPI/archetai",
                "source_artifact_path": "candidate.jsonl",
                "source_row_id": "1",
                "source_doc_id": "doc-1",
                "text": "Ένα αυτόνομο κείμενο.",
            }
        ],
    )
    outputs = {
        "registry": tmp_path / "registry.json",
        "actions": tmp_path / "actions.jsonl",
        "novelty": tmp_path / "novelty.json",
        "summary": tmp_path / "summary.json",
    }
    command = [
        sys.executable,
        str(SCRIPTS / "build_source_lineage.py"),
        "rows",
        "--candidate-jsonl",
        str(candidate),
        "--registry-manifest-out",
        str(outputs["registry"]),
        "--actions-out",
        str(outputs["actions"]),
        "--novelty-out",
        str(outputs["novelty"]),
        "--summary-out",
        str(outputs["summary"]),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["debug_exports_enabled"] is False
    assert summary["row_manifest"] is None
    assert summary["relationship_manifest"] is None
    assert not (tmp_path / "rows.jsonl").exists()
    assert not (tmp_path / "relationships.jsonl").exists()


def _legacy_source_novelty(connection: sqlite3.Connection) -> list[dict]:
    """Frozen pre-optimization query shape used as a semantic oracle."""

    result: list[dict] = []
    for source_dataset, rows, total_tokens, source_ids in connection.execute(
        """
        SELECT source_dataset, COUNT(*), SUM(identity_word_tokens),
               GROUP_CONCAT(DISTINCT source_id)
        FROM rows WHERE origin = 'candidate'
        GROUP BY source_dataset ORDER BY source_dataset
        """
    ):
        unique_rows, unique_tokens = connection.execute(
            """
            WITH representatives AS (
                SELECT normalized_text_sha256, MIN(stable_uid) AS stable_uid
                FROM rows
                WHERE origin = 'candidate' AND source_dataset = ?
                GROUP BY normalized_text_sha256
            )
            SELECT COUNT(*), COALESCE(SUM(row.identity_word_tokens), 0)
            FROM representatives representative
            JOIN rows row ON row.stable_uid = representative.stable_uid
            """,
            (source_dataset,),
        ).fetchone()
        novel_rows, novel_tokens = connection.execute(
            """
            WITH representatives AS (
                SELECT normalized_text_sha256, MIN(stable_uid) AS stable_uid
                FROM rows
                WHERE origin = 'candidate' AND source_dataset = ?
                GROUP BY normalized_text_sha256
            )
            SELECT COUNT(*), COALESCE(SUM(row.identity_word_tokens), 0)
            FROM representatives representative
            JOIN rows row ON row.stable_uid = representative.stable_uid
            LEFT JOIN resolved_actions action ON action.stable_uid = row.stable_uid
            WHERE action.stable_uid IS NULL
            """,
            (source_dataset,),
        ).fetchone()
        action_counts = dict(
            connection.execute(
                """
                SELECT action.action, COUNT(*)
                FROM rows row
                JOIN resolved_actions action ON action.stable_uid = row.stable_uid
                WHERE row.origin = 'candidate' AND row.source_dataset = ?
                GROUP BY action.action ORDER BY action.action
                """,
                (source_dataset,),
            ).fetchall()
        )
        total_tokens = int(total_tokens or 0)
        novel_tokens = int(novel_tokens or 0)
        result.append(
            {
                "source_dataset": str(source_dataset),
                "source_ids": sorted(str(source_ids or "").split(",")),
                "rows": int(rows),
                "identity_word_tokens": total_tokens,
                "exact_unique_rows": int(unique_rows),
                "exact_unique_word_tokens": int(unique_tokens),
                "novel_rows_after_lineage_resolution": int(novel_rows),
                "novel_word_tokens_after_lineage_resolution": novel_tokens,
                "novel_token_fraction": (
                    round(novel_tokens / total_tokens, 8) if total_tokens else 0.0
                ),
                "document_action_counts": action_counts,
            }
        )
    return result


def _legacy_source_summaries(
    connection: sqlite3.Connection, registry: dict
) -> list[dict]:
    """Frozen pre-optimization summary queries used as a semantic oracle."""

    route_by_id = {
        entry["source_id"]: entry for entry in registry.get("candidates", [])
    }
    result: list[dict] = []
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
                        AND base.normalized_text_sha256 =
                            candidate.normalized_text_sha256
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
                      WHERE base.origin = 'base'
                        AND base.work_key = candidate.work_key
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


def _reporting_fixture() -> tuple[sqlite3.Connection, dict]:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE rows (
            stable_uid TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_family_id TEXT NOT NULL,
            normalized_text_sha256 TEXT NOT NULL,
            work_key TEXT NOT NULL,
            identity_word_tokens INTEGER NOT NULL
        );
        CREATE INDEX rows_exact_idx
            ON rows(normalized_text_sha256, stable_uid);
        CREATE INDEX rows_work_idx ON rows(work_key, stable_uid);
        CREATE INDEX rows_source_idx ON rows(source_id, stable_uid);
        CREATE TEMP TABLE resolved_actions (
            stable_uid TEXT PRIMARY KEY,
            source_dataset TEXT NOT NULL,
            action TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "base-shared",
                "base",
                "shared",
                "base-dataset",
                "base-family",
                "exact-overlap",
                "work-overlap",
                0,
            ),
            (
                "base-only",
                "base",
                "base-only",
                "second-base-dataset",
                "second-base-family",
                "base-only-hash",
                "base-only-work",
                0,
            ),
            (
                "candidate-a-1",
                "candidate",
                "shared",
                "dataset-a",
                "candidate-family",
                "exact-overlap",
                "work-overlap",
                7,
            ),
            (
                "candidate-a-2",
                "candidate",
                "shared",
                "dataset-a",
                "candidate-family",
                "exact-overlap",
                "work-overlap",
                99,
            ),
            (
                "candidate-a-3",
                "candidate",
                "shared",
                "dataset-a",
                "candidate-family",
                "candidate-only-hash",
                "candidate-only-work",
                3,
            ),
            (
                "candidate-a-4",
                "candidate",
                "new-source",
                "dataset-a",
                "second-candidate-family",
                "second-candidate-hash",
                "second-candidate-work",
                2,
            ),
            (
                "candidate-zero",
                "candidate",
                "new-source",
                "dataset-zero",
                "second-candidate-family",
                "zero-token-hash",
                "zero-token-work",
                0,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO resolved_actions VALUES (?, ?, ?)",
        [
            ("candidate-a-2", "dataset-a", "drop"),
            ("candidate-a-3", "dataset-a", "quarantine"),
        ],
    )
    registry = {
        "candidates": [
            {
                "source_id": "shared",
                "requires_base_identity_audit": True,
                "requires_family_internal_dedup": False,
            },
            {
                "source_id": "new-source",
                "requires_base_identity_audit": False,
                "requires_family_internal_dedup": False,
            },
        ]
    }
    return connection, registry


def test_set_based_lineage_reports_match_legacy_without_per_source_scans(
    capsys: pytest.CaptureFixture[str],
) -> None:
    connection, registry = _reporting_fixture()
    try:
        expected_novelty = _legacy_source_novelty(connection)
        expected_summaries = _legacy_source_summaries(connection, registry)
        traced: list[str] = []
        connection.set_trace_callback(traced.append)
        BUILD.prepare_candidate_analysis_rows(connection)
        try:
            observed_novelty = BUILD.source_novelty(
                connection, candidate_analysis_ready=True
            )
            observed_summaries = BUILD.source_summaries(
                connection, registry, candidate_analysis_ready=True
            )
        finally:
            BUILD.drop_candidate_analysis_rows(connection)
        connection.set_trace_callback(None)

        assert observed_novelty == expected_novelty
        assert observed_summaries == expected_summaries
        assert observed_novelty == [
            {
                "source_dataset": "dataset-a",
                "source_ids": ["new-source", "shared"],
                "rows": 4,
                "identity_word_tokens": 111,
                "exact_unique_rows": 3,
                "exact_unique_word_tokens": 12,
                "novel_rows_after_lineage_resolution": 2,
                "novel_word_tokens_after_lineage_resolution": 9,
                "novel_token_fraction": 0.08108108,
                "document_action_counts": {"drop": 1, "quarantine": 1},
            },
            {
                "source_dataset": "dataset-zero",
                "source_ids": ["new-source"],
                "rows": 1,
                "identity_word_tokens": 0,
                "exact_unique_rows": 1,
                "exact_unique_word_tokens": 0,
                "novel_rows_after_lineage_resolution": 1,
                "novel_word_tokens_after_lineage_resolution": 0,
                "novel_token_fraction": 0.0,
                "document_action_counts": {},
            },
        ]

        normalized_sql = [" ".join(statement.lower().split()) for statement in traced]
        statements_reading_persistent_rows = [
            statement
            for statement in normalized_sql
            if " from rows" in statement
        ]
        # One candidate projection, one base rollup, and two set-based overlap
        # maps: this count is independent of the number of source datasets.
        assert len(statements_reading_persistent_rows) == 4
        assert sum(
            "from rows not indexed" in statement
            for statement in statements_reading_persistent_rows
        ) == 2
        assert not any(
            "where candidate.source_id =" in statement
            or (
                "where row.origin = 'candidate'" in statement
                and "row.source_dataset =" in statement
            )
            for statement in normalized_sql
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_temp_master "
            "WHERE name LIKE 'lineage_candidate_%'"
        ).fetchone() == (0,)
        progress = capsys.readouterr().out
        assert "phase=candidate_analysis_projection status=completed" in progress
        assert "phase=source_novelty status=completed" in progress
        assert "phase=source_summaries status=completed" in progress
    finally:
        connection.close()


def test_lineage_resume_contract_binds_debug_export_mode(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text("{}\n", encoding="utf-8")
    configs = []
    for name in ("sources.json", "roster.json", "aliases.json"):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        configs.append(path)
    common = {
        "sources_config": configs[0],
        "roster_config": configs[1],
        "aliases_config": configs[2],
        "normalization_manifest": None,
        "normalization_inventory_validation": None,
        "canonical_verification_interval": 100_000,
    }
    normal = BUILD.lineage_contract(
        SimpleNamespace(
            **common,
            rows_out=None,
            relationships_out=None,
        ),
        base_paths=[],
        candidate_paths=[candidate],
        bindings={},
    )
    debug = BUILD.lineage_contract(
        SimpleNamespace(
            **common,
            rows_out=tmp_path / "rows.jsonl",
            relationships_out=tmp_path / "relationships.jsonl",
        ),
        base_paths=[],
        candidate_paths=[candidate],
        bindings={},
    )
    assert normal != debug


def _canonical_fixture_row(
    *,
    source_id: str,
    source_dataset: str,
    source_family_id: str,
    repo_id: str,
    revision: str,
    source_doc_id: str,
    text: str,
) -> dict:
    artifact = f"{source_id}/part.parquet"
    source_row_id = f"{artifact}:0:0"
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    work_id = source_doc_id
    return {
        "source_id": source_id,
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "text": text,
        "title": None,
        "author": None,
        "greek_badness_score": None,
        "mojibake_badness_score": None,
        "needs_ocr": None,
        "is_empty": False,
        "ocr_success": None,
        "is_historical_or_polytonic": None,
        "source_family_id": source_family_id,
        "acquisition_source_id": source_id,
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_artifact_path": artifact,
        "source_row_id": source_row_id,
        "source_text_field": "text",
        "original_text_sha256": text_hash,
        "normalized_text_sha256": text_hash,
        "stable_uid": LINEAGE.sha256_parts(
            "full_cpt_stable_uid_v1",
            repo_id,
            revision,
            artifact,
            source_row_id,
            source_dataset,
            "text",
        ),
        "work_key": LINEAGE.sha256_parts(
            "full_cpt_work_key_v1", source_family_id, work_id
        ),
        "work_id": work_id,
        "representation_generation": (
            "nanochat_pinned_release"
            if source_id == "nanochat_base"
            else "candidate_first_representation"
        ),
        "lineage_alias_id": None,
        "source_metadata_json": "{}",
        "cleaning_profile": "base_canonical",
        "structural_policy": "source_routed",
        "training_eligibility": "review",
        "source_role": "base" if source_id == "nanochat_base" else "additive_candidate",
    }


def test_bound_lineage_omits_base_text_and_resumes_per_canonical_shard(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    base_root = tmp_path / "canonical" / "nanochat_base"
    candidate_root = tmp_path / "canonical" / "candidate"
    base_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    base_path = base_root / "base.parquet"
    candidate_path = candidate_root / "candidate.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _canonical_fixture_row(
                    source_id="nanochat_base",
                    source_dataset="base-name",
                    source_family_id="base-name",
                    repo_id="owner/base",
                    revision="base-rev",
                    source_doc_id="base-1",
                    text="κείμενο βάσης",
                )
            ],
            schema=CORPUS_IO.canonical_schema(),
        ),
        base_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _canonical_fixture_row(
                    source_id="candidate",
                    source_dataset="candidate-name",
                    source_family_id="candidate-family",
                    repo_id="owner/candidate",
                    revision="candidate-rev",
                    source_doc_id="candidate-1",
                    text="καινούριο ελληνικό κείμενο",
                )
            ],
            schema=CORPUS_IO.canonical_schema(),
        ),
        candidate_path,
    )
    config = {
        "base": {
            "repo_id": "owner/base",
            "revision": "base-rev",
            "role": "base",
            "text_columns": ["text"],
        },
        "sources": [
            {
                "source_id": "candidate",
                "repo_id": "owner/candidate",
                "revision": "candidate-rev",
                "role": "additive_candidate",
                "source_family_id": "candidate-family",
                "content_relation": "new_family",
                "merge_policy": "append_after_review_and_global_dedup",
                "text_columns": ["text"],
            }
        ],
    }
    sources = tmp_path / "sources.json"
    roster = tmp_path / "roster.json"
    aliases = tmp_path / "aliases.json"
    sources.write_text(json.dumps(config))
    roster.write_text(json.dumps({"repository": {}, "sources": []}))
    aliases.write_text(json.dumps({"aliases": []}))

    def inventory(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 1,
        }

    manifest = tmp_path / "normalization.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_normalization_manifest_v1",
                "sources": [
                    {"source_id": "nanochat_base", "shards": [inventory(base_path)]},
                    {"source_id": "candidate", "shards": [inventory(candidate_path)]},
                ],
            }
        )
    )
    validation = tmp_path / "inventory-validation.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_shard_inventory_validation_v1",
                "manifest": str(manifest.resolve()),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "files": 2,
                "bytes": base_path.stat().st_size + candidate_path.stat().st_size,
                "inventory_sha256": "a" * 64,
            }
        )
    )
    bindings = BUILD.normalization_bindings(manifest)
    base_rows = list(
        LINEAGE.iter_lineage_rows([base_root], origin="base", bound_inputs=bindings)
    )
    assert len(base_rows) == 1
    assert "text" not in base_rows[0][2]
    verified = LINEAGE.canonicalize_row(
        base_rows[0][2],
        origin="base",
        sources=config,
        roster={"repository": {}, "sources": []},
        aliases={"aliases": []},
        canonical_bound=True,
        verify_bound=True,
    )
    assert verified["identity_word_tokens"] == 0

    outputs = {
        "registry": tmp_path / "registry.json",
        "rows": tmp_path / "rows.jsonl",
        "relationships": tmp_path / "relationships.jsonl",
        "actions": tmp_path / "actions.jsonl",
        "novelty": tmp_path / "novelty.json",
        "summary": tmp_path / "summary.json",
    }
    database = tmp_path / "work" / "lineage.sqlite"
    command = [
        sys.executable,
        str(SCRIPTS / "build_source_lineage.py"),
        "rows",
        "--sources-config",
        str(sources),
        "--roster-config",
        str(roster),
        "--aliases-config",
        str(aliases),
        "--base-input",
        str(base_root),
        "--candidate-input",
        str(candidate_root),
        "--normalization-manifest",
        str(manifest),
        "--normalization-inventory-validation",
        str(validation),
        "--registry-manifest-out",
        str(outputs["registry"]),
        "--rows-out",
        str(outputs["rows"]),
        "--relationships-out",
        str(outputs["relationships"]),
        "--actions-out",
        str(outputs["actions"]),
        "--novelty-out",
        str(outputs["novelty"]),
        "--summary-out",
        str(outputs["summary"]),
        "--sqlite-work-path",
        str(database),
        "--input-spool-directory",
        str(tmp_path / "fragments"),
        "--input-workers",
        "2",
        "--resume",
    ]
    first = subprocess.run(command, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM ingested_inputs").fetchone()[0]
            == 2
        )
    checksums = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in outputs.items()
    }
    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert "ingested origin=" not in second.stdout
    assert checksums == {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in outputs.items()
    }


def candidate_rows(source_id: str, source_dataset: str, count: int) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        dirty = index < max(25, count // 4)
        text = (
            f"<div>Άρθρο {index}</div> test{index}@example.gr Ã© "
            if dirty
            else f"Καθαρό και χρήσιμο ελληνικό κείμενο για το έγγραφο {index}. " * 8
        )
        rows.append(
            {
                "source_id": source_id,
                "source_dataset": source_dataset,
                "source_artifact_path": "fixture.parquet",
                "source_row_id": str(index),
                "source_doc_id": f"doc-{index}",
                "review_cluster_id": f"template-{index % 60}",
                "text": text,
            }
        )
    return rows


def run_review_packet(
    input_path: Path, output_dir: Path, extra_args: list[str] | None = None
) -> tuple[Path, Path]:
    requests = output_dir / "requests.jsonl"
    summary = output_dir / "summary.json"
    output_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_source_review_packet.py"),
            "--candidate-jsonl",
            str(input_path),
            "--requests-out",
            str(requests),
            "--summary-out",
            str(summary),
            *(extra_args or []),
        ],
        check=True,
    )
    return requests, summary


def test_postclean_packet_filters_exact_admission_decision(tmp_path: Path) -> None:
    rows = candidate_rows("archetai", "keep-after-clean", 105) + candidate_rows(
        "diavgeia", "already-included", 105
    )
    input_path = tmp_path / "candidates.jsonl"
    write_jsonl(input_path, rows)
    admission = {
        "schema_version": "source_quality_review_admission_v1",
        "pending_adjudications": 0,
        "sources": [
            {
                "source_dataset": "keep-after-clean",
                "decision": "include_after_cleaning",
            },
            {"source_dataset": "already-included", "decision": "include"},
        ],
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission))
    _, summary_path = run_review_packet(
        input_path,
        tmp_path / "filtered",
        [
            "--source-admission",
            str(admission_path),
            "--decision",
            "include_after_cleaning",
            "--review-phase",
            "post_clean",
        ],
    )
    summary = json.loads(summary_path.read_text())
    assert summary["review_phase"] == "post_clean"
    assert [row["source_dataset"] for row in summary["sources"]] == ["keep-after-clean"]


def test_review_packet_uses_exact_100_200_strata_and_is_order_independent(
    tmp_path: Path,
) -> None:
    default_rows = candidate_rows("archetai", "Exact Archetai Source", 121)
    default_rows[-1]["privateData"] = True
    large_rows = candidate_rows("diavgeia", "Exact Diavgeia Source", 220)
    combined = default_rows + large_rows
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    write_jsonl(first_input, combined)
    write_jsonl(second_input, list(reversed(combined)))

    first_requests, first_summary = run_review_packet(first_input, tmp_path / "first")
    second_requests, _ = run_review_packet(second_input, tmp_path / "second")
    first_records = [
        json.loads(line) for line in first_requests.read_text().splitlines()
    ]
    second_records = [
        json.loads(line) for line in second_requests.read_text().splitlines()
    ]
    assert [record["review_id"] for record in first_records] == [
        record["review_id"] for record in second_records
    ]
    assert "@example.gr" not in first_requests.read_text()
    assert "[REDACTED_EMAIL]" in first_requests.read_text()

    summary = json.loads(first_summary.read_text())
    reports = {entry["source_dataset"]: entry for entry in summary["sources"]}
    default = reports["Exact Archetai Source"]
    assert default["unique_sampled_documents"] == 100
    assert default["sampling_strata"] == {"cluster": 20, "random": 60, "risk": 20}
    assert default["double_review_documents"] == 10
    assert default["request_rows"] == 110
    assert default["private_data_documents_excluded"] == 1
    large = reports["Exact Diavgeia Source"]
    assert large["unique_sampled_documents"] == 200
    assert large["sampling_strata"] == {"cluster": 50, "random": 100, "risk": 50}
    assert large["double_review_documents"] == 20
    assert large["request_rows"] == 220
    assert summary["unique_sampled_documents"] == 300


def review_response(
    request: dict, action: str = "include", confidence: str = "high"
) -> dict:
    return {
        "schema_version": "source_quality_review_response_v1",
        "review_id": request["review_id"],
        "sample_id": request["sample_id"],
        "reviewer_slot": request["reviewer_slot"],
        "source_dataset": request["source_dataset"],
        "substantive_training_value": "high",
        "quality_score": 4,
        "language_register": "modern_greek",
        "defects": {key: "none" for key in AGGREGATE.DEFECT_KEYS},
        "variability": {"template_similarity": "low", "substantive_variation": "high"},
        "action": action,
        "defects_deterministically_repairable": action == "include_after_cleaning",
        "safety_or_license_blocker": False,
        "confidence": confidence,
        "evidence": "Substantive, clean Greek prose.",
    }


def test_review_aggregation_requires_adjudication_and_applies_cleaning_gate(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, candidate_rows("archetai", "Exact Archetai Source", 120))
    requests_path, summary_path = run_review_packet(input_path, tmp_path / "packet")
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    cleaning_samples = {
        request["sample_id"]
        for request in requests
        if request["reviewer_slot"] == "primary"
    }
    cleaning_samples = set(sorted(cleaning_samples)[:5])
    responses = [
        review_response(
            request,
            action=(
                "include_after_cleaning"
                if request["sample_id"] in cleaning_samples
                else "include"
            ),
        )
        for request in requests
    ]
    reviews_path = tmp_path / "reviews.jsonl"
    write_jsonl(reviews_path, responses)
    output = tmp_path / "admission.json"
    novelty_path = tmp_path / "novelty.json"
    novelty_path.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_source_novelty_v1",
                "near_duplicate_novelty_deferred_to_global_dedup": True,
                "sources": [
                    {
                        "source_dataset": "Exact Archetai Source",
                        "novel_token_fraction": 1.0,
                    }
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "aggregate_source_reviews.py"),
            "--requests",
            str(requests_path),
            "--packet-summary",
            str(summary_path),
            "--reviews",
            str(reviews_path),
            "--novelty-summary",
            str(novelty_path),
            "--output",
            str(output),
        ]
    )
    assert completed.returncode == 0
    source = json.loads(output.read_text())["sources"][0]
    assert source["decision"] == "include_after_cleaning"
    assert source["post_clean_review_required"] is True

    secondary = next(
        request for request in requests if request["reviewer_slot"] == "secondary"
    )
    for response in responses:
        if response["review_id"] == secondary["review_id"]:
            response["action"] = "exclude"
            response["defects_deterministically_repairable"] = False
    write_jsonl(reviews_path, responses)
    pending = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "aggregate_source_reviews.py"),
            "--requests",
            str(requests_path),
            "--packet-summary",
            str(summary_path),
            "--reviews",
            str(reviews_path),
            "--novelty-summary",
            str(novelty_path),
            "--output",
            str(output),
        ]
    )
    assert pending.returncode == 2
    result = json.loads(output.read_text())
    assert result["pending_adjudications"] == 1
    assert result["sources"][0]["decision"] == "pending_adjudication"
