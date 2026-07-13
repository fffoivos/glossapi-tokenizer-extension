from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_lineage_module():
    path = HERE / "scripts" / "source_lineage.py"
    spec = importlib.util.spec_from_file_location("phase04_normalize_lineage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_io_module():
    path = HERE / "scripts" / "full_corpus_io.py"
    scripts = str(path.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("phase04_normalize_io", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def load_normalize_module():
    path = HERE / "scripts" / "normalize_sources.py"
    scripts = str(path.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("phase04_normalizer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def test_base_family_map_prefers_replacement_over_hybrid_membership() -> None:
    io = load_io_module()
    config = {
        "sources": [
            {"repo_id": "owner/replacement", "source_family_id": "greek_phd"},
            {"repo_id": "owner/hybrid", "source_family_id": "openarchives"},
        ]
    }
    aliases = {
        "aliases": [
            {
                "alias_kind": "hybrid",
                "current_repo_id": "owner/hybrid",
                "initial_source_datasets": ["openarchives.gr", "greek_phd"],
            },
            {
                "alias_kind": "replacement",
                "current_repo_id": "owner/replacement",
                "initial_source_datasets": ["greek_phd"],
            },
        ]
    }
    assert io.base_family_map(config, aliases) == {
        "greek_phd": "greek_phd",
        "openarchives.gr": "openarchives",
    }


def test_acquisition_receipt_must_match_current_sources_config(
    tmp_path: Path,
) -> None:
    io = load_io_module()
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"base": {}, "sources": []}), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "sources_config_sha256": "0" * 64,
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="current sources.json"):
        io.artifacts_from_receipt(config, receipt)


def test_nanochat_embedded_academic_route_sets_structural_policy(
    tmp_path: Path,
) -> None:
    io = load_io_module()
    revision = "a" * 40
    artifact_path = tmp_path / revision / "data" / "greek_phd-000.parquet"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"fixture")
    source = io.SourceArtifact(
        source_id="nanochat_base",
        repo_id="owner/base",
        revision=revision,
        role="base",
        source_family_id="nanochat_base",
        files=(artifact_path,),
        file_bindings=({},),
        config={"id_columns": ["source_doc_id"], "source_column": "source_dataset"},
    )
    route = {
        "source_id": "greek_phd",
        "acquisition_source_id": "nanochat_base",
        "acquisition_include_globs": ["data/greek_phd*.parquet"],
        "source_regex": "^greek_phd",
        "cleaning_profile": "academic_ocr",
        "structural_policy": "apply_after_review",
    }
    row = io.canonical_row(
        source=source,
        artifact_path=artifact_path,
        artifact_row_index=0,
        raw_row={"source_doc_id": "doc-1", "source_dataset": "greek_phd"},
        representation_suffix="0",
        text_field="text",
        raw_text="Ακαδημαϊκό κείμενο",
        lineage_aliases={"aliases": []},
        base_families={},
        embedded_structural_routes=[route],
    )
    assert row["source_id"] == "greek_phd"
    assert row["acquisition_source_id"] == "nanochat_base"
    assert row["cleaning_profile"] == "academic_ocr"
    assert row["structural_policy"] == "apply_after_review"


def test_embedded_structural_routes_fail_on_ambiguous_match(tmp_path: Path) -> None:
    io = load_io_module()
    revision = "b" * 40
    artifact_path = tmp_path / revision / "data" / "greek_phd.parquet"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"fixture")
    source = io.SourceArtifact(
        source_id="nanochat_base",
        repo_id="owner/base",
        revision=revision,
        role="base",
        source_family_id="nanochat_base",
        files=(artifact_path,),
        file_bindings=({},),
        config={"id_columns": ["source_doc_id"], "source_column": "source_dataset"},
    )
    route = {
        "source_id": "route-a",
        "acquisition_source_id": "nanochat_base",
        "acquisition_include_globs": ["data/*.parquet"],
        "source_regex": "^greek_phd$",
        "cleaning_profile": "academic_ocr",
        "structural_policy": "apply_after_review",
    }
    with pytest.raises(ValueError, match="multiple embedded structural routes"):
        io.canonical_row(
            source=source,
            artifact_path=artifact_path,
            artifact_row_index=0,
            raw_row={"source_doc_id": "doc-1", "source_dataset": "greek_phd"},
            representation_suffix="0",
            text_field="text",
            raw_text="Ακαδημαϊκό κείμενο",
            lineage_aliases={"aliases": []},
            base_families={},
            embedded_structural_routes=[route, {**route, "source_id": "route-b"}],
        )


def test_embedded_route_coverage_requires_positive_rows_without_smoke_false_positive(
    tmp_path: Path,
) -> None:
    normalize = load_normalize_module()
    io = load_io_module()
    revision = "c" * 40
    artifact_path = tmp_path / revision / "data" / "greek_phd.part-00000.parquet"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"fixture")
    artifact = io.SourceArtifact(
        source_id="nanochat_base",
        repo_id="owner/base",
        revision=revision,
        role="base",
        source_family_id="nanochat_base",
        files=(artifact_path,),
        file_bindings=({},),
        config={},
    )
    route = {
        "source_id": "greek_phd",
        "acquisition_source_id": "nanochat_base",
        "acquisition_include_globs": ["data/greek_phd*.parquet"],
        "source_regex": "^greek_phd$",
        "source_column": "source_dataset",
        "coverage_contract": {
            "expected_source_dataset": "greek_phd",
            "minimum_normalized_rows": 1,
            "enforcement_scope": "unbounded_normalization",
        },
    }
    summaries = [
        {
            "source_id": "nanochat_base",
            "counts": {"embedded_route:greek_phd": 12},
        }
    ]
    coverage = normalize.validate_embedded_route_coverage(
        [route], [artifact], summaries, bounded_smoke=False
    )
    assert coverage["all_enforced_routes_passed"] is True
    assert coverage["routes"][0]["status"] == "passed"
    assert coverage["routes"][0]["normalized_rows"] == 12

    with pytest.raises(ValueError, match="matched acquired artifacts.*routed 0 rows"):
        normalize.validate_embedded_route_coverage(
            [route],
            [artifact],
            [{"source_id": "nanochat_base", "counts": {}}],
            bounded_smoke=False,
        )
    smoke = normalize.validate_embedded_route_coverage(
        [route],
        [artifact],
        [{"source_id": "nanochat_base", "counts": {}}],
        bounded_smoke=True,
    )
    assert smoke["routes"][0]["status"] == "not_enforced_bounded_smoke"
    assert smoke["routes"][0]["postcondition_enforced"] is False


def test_large_normalization_tasks_use_receipt_bound_byte_sum() -> None:
    normalize = load_normalize_module()
    artifact = SimpleNamespace(file_bindings=({"bytes": 7}, {"bytes": 5}, {"bytes": 3}))
    tasks = [
        {"artifact": artifact, "file_indices": [0]},
        {"artifact": artifact, "file_indices": [1, 2]},
        {"artifact": artifact, "file_indices": [2]},
    ]

    ordinary, large = normalize.partition_normalization_tasks(
        tasks, large_task_byte_threshold=8
    )

    assert ordinary == [tasks[0], tasks[2]]
    assert large == [tasks[1]]
    assert normalize.normalization_task_input_bytes(tasks[1]) == 8
    with pytest.raises(ValueError, match="threshold must be positive"):
        normalize.partition_normalization_tasks(tasks, large_task_byte_threshold=0)


def test_production_selected_text_source_must_emit_documents(tmp_path: Path) -> None:
    normalize = load_normalize_module()
    io = load_io_module()
    payload = tmp_path / "source.parquet"
    payload.write_bytes(b"fixture")
    artifact = io.SourceArtifact(
        source_id="selected-source",
        repo_id="owner/source",
        revision="d" * 40,
        role="additive_candidate",
        source_family_id="selected-source",
        files=(payload,),
        file_bindings=({},),
        config={"text_columns": ["text"]},
    )
    with pytest.raises(ValueError, match="zero documents.*selected-source"):
        normalize.validate_selected_source_coverage(
            [artifact],
            [{"source_id": "selected-source", "counts": {}}],
            bounded_smoke=False,
        )
    smoke = normalize.validate_selected_source_coverage(
        [artifact],
        [{"source_id": "selected-source", "counts": {}}],
        bounded_smoke=True,
    )
    assert smoke["sources"][0]["status"] == "not_enforced_bounded_smoke"
    passed = normalize.validate_selected_source_coverage(
        [artifact],
        [
            {
                "source_id": "selected-source",
                "counts": {"documents_emitted": 1},
            }
        ],
        bounded_smoke=False,
    )
    assert passed["all_enforced_sources_passed"] is True


def test_normalizer_preserves_source_names_and_groups_sectioned_works(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    base_path = tmp_path / "hf" / "nanochat_base" / "base-rev" / "data" / "base.parquet"
    base_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "source_dataset": ["greek_phd", "openarchives.gr"],
                "source_doc_id": ["doc-1", "doc-2"],
                "text": ["Κείμενο\r\nΑ", "Κείμενο Β\u200b"],
                "title": ["Α", "Β"],
            }
        ),
        base_path,
    )
    section_path = tmp_path / "hf" / "sections" / "section-rev" / "sections.parquet"
    section_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": [2, 1, 3],
                "filename": ["work-b", "work-a", "work-b"],
                "section": ["Βήτα 1", "Άλφα", "Βήτα 2"],
                "title": ["Β", "Α", "Β"],
            }
        ),
        section_path,
    )
    config = {
        "base": {
            "repo_id": "owner/base",
            "revision": "base-rev",
            "role": "base",
            "text_columns": ["text"],
            "id_columns": ["source_doc_id"],
            "source_column": "source_dataset",
        },
        "sources": [
            {
                "source_id": "sections",
                "repo_id": "owner/sections",
                "revision": "section-rev",
                "role": "replacement_candidate",
                "source_family_id": "academic_sections",
                "content_relation": "same_source_resegmentation",
                "merge_policy": "group_sections_to_work_then_keyed_replacement_audit",
                "text_columns": ["section"],
                "id_columns": ["id"],
                "work_id_columns": ["filename"],
                "section_order_columns": ["id"],
                "cleaning_profile": "academic_sectioned",
                "structural_policy": "apply_after_review",
                "training_eligibility": "review",
            }
        ],
    }
    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps(config))
    receipt = {
        "schema_version": "full_cpt_acquisition_receipt_v1",
        "status": "passed",
        "sources_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sources": [
            {
                "source_id": "nanochat_base",
                "revision": "base-rev",
                "files": [{"local_path": str(base_path)}],
            },
            {
                "source_id": "sections",
                "revision": "section-rev",
                "files": [{"local_path": str(section_path)}],
            },
        ],
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    output = tmp_path / "normalized"
    manifest = tmp_path / "normalization.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "normalize_sources.py"),
            "--sources",
            str(config_path),
            "--acquisition-receipt",
            str(receipt_path),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--rows-per-shard",
            "2",
        ],
        check=True,
    )

    rows = []
    for path in sorted(output.rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    assert len(rows) == 4
    base = [row for row in rows if row["source_repo_id"] == "owner/base"]
    assert {row["source_dataset"] for row in base} == {"greek_phd", "openarchives.gr"}
    assert "\r" not in base[0]["text"]
    assert "\u200b" not in "".join(row["text"] for row in base)

    sections = sorted(
        (row for row in rows if row["source_repo_id"] == "owner/sections"),
        key=lambda row: row["source_doc_id"],
    )
    assert [row["source_doc_id"] for row in sections] == ["work-a", "work-b"]
    assert sections[1]["text"] == "Βήτα 1\n\nΒήτα 2"
    assert sections[1]["source_text_field"] == "section_grouped"
    assert sections[1]["source_id"] == "sections"
    assert sections[1]["acquisition_source_id"] == "sections"
    assert sections[1]["work_id"] == "work-b"
    assert json.loads(sections[1]["source_metadata_json"])["_section_count"] == 2
    assert len({row["stable_uid"] for row in rows}) == 4

    lineage = load_lineage_module()
    recomputed = lineage.canonicalize_row(
        sections[1],
        origin="candidate",
        sources=config,
        roster={"repository": {}},
        aliases={"aliases": []},
    )
    assert recomputed["stable_uid"] == sections[1]["stable_uid"]
    assert recomputed["work_key"] == sections[1]["work_key"]

    payload = json.loads(manifest.read_text())
    assert payload["total_documents"] == 4
    assert payload["bounded_smoke"] is False


def test_v3_normalizer_binds_roster_and_preserves_logical_route_provenance(
    tmp_path: Path,
) -> None:
    """Diavgeia/OpenGov stay mixed even though their transport is Parquet."""

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")

    def write_source(source_id: str, revision: str, *, text_column: str) -> Path:
        path = tmp_path / "raw" / source_id / revision / f"{source_id}.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "source_dataset": [source_id],
                    "id": [f"{source_id}-1"],
                    text_column: [f"κείμενο {source_id}"],
                }
            ),
            path,
        )
        return path

    base_path = write_source("nanochat_base", "base-rev", text_column="text")
    diavgeia_path = write_source("diavgeia", "diavgeia-rev", text_column="markdown_text")
    opengov_path = write_source(
        "opengov_deliberations_v2", "opengov-rev", text_column="articles"
    )
    config = {
        "base": {
            "repo_id": "owner/base",
            "revision": "base-rev",
            "role": "base",
            "text_columns": ["text"],
            "id_columns": ["id"],
            "source_column": "source_dataset",
        },
        "sources": [
            {
                "source_id": "diavgeia",
                "repo_id": "owner/diavgeia",
                "revision": "diavgeia-rev",
                "role": "additive_candidate",
                "source_family_id": "diavgeia",
                "text_columns": ["markdown_text"],
                "id_columns": ["id"],
                # A stale physical/registry hint must not override v3's
                # frozen logical source route.
                "source_route": "structured",
                "extraction_route": "structured",
            },
            {
                "source_id": "opengov_deliberations_v2",
                "repo_id": "owner/opengov",
                "revision": "opengov-rev",
                "role": "replacement_candidate",
                "source_family_id": "opengov",
                "text_columns": ["articles"],
                "id_columns": ["id"],
            },
        ],
    }
    config_path = tmp_path / "sources.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"aliases": []}), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "sources_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "sources": [
                    {
                        "source_id": "nanochat_base",
                        "revision": "base-rev",
                        "files": [{"local_path": str(base_path)}],
                    },
                    {
                        "source_id": "diavgeia",
                        "revision": "diavgeia-rev",
                        "files": [{"local_path": str(diavgeia_path)}],
                    },
                    {
                        "source_id": "opengov_deliberations_v2",
                        "revision": "opengov-rev",
                        "files": [{"local_path": str(opengov_path)}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        json.dumps(
            {
                "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
                "base_source_id": "nanochat_base",
                "candidate_source_ids": ["diavgeia", "opengov_deliberations_v2"],
                "review_routes": {
                    "diavgeia": "mixed",
                    "opengov_deliberations_v2": "mixed",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "canonical"
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "normalize_sources.py"),
            "--sources",
            str(config_path),
            "--lineage-aliases",
            str(aliases),
            "--acquisition-receipt",
            str(receipt_path),
            "--candidate-roster",
            str(roster_path),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--rows-per-shard",
            "2",
            "--workers",
            "1",
            "--duckdb-memory-limit",
            "256MB",
            "--duckdb-threads",
            "1",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    rows = []
    for path in sorted(output.rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    base_rows = [row for row in rows if row["acquisition_source_id"] == "nanochat_base"]
    candidate_rows = [row for row in rows if row["acquisition_source_id"] != "nanochat_base"]
    assert len(base_rows) == 1
    assert {
        field: base_rows[0][field]
        for field in (
            "source_route",
            "review_route",
            "extraction_route",
            "observed_extraction_route",
            "observed_extraction_route_basis",
            "observed_extraction_route_evidence",
            "observed_extraction_route_priority",
        )
    } == {
        "source_route": None,
        "review_route": None,
        "extraction_route": None,
        "observed_extraction_route": None,
        "observed_extraction_route_basis": "unavailable",
        "observed_extraction_route_evidence": "none",
        "observed_extraction_route_priority": None,
    }
    assert {row["acquisition_source_id"] for row in candidate_rows} == {
        "diavgeia",
        "opengov_deliberations_v2",
    }
    assert all(
        row["source_route"] == row["review_route"] == row["extraction_route"] == "mixed"
        for row in candidate_rows
    )
    assert all(
        row["observed_extraction_route"] == "mixed"
        and row["observed_extraction_route_basis"]
        == "declared_extraction_route_fallback"
        and row["observed_extraction_route_evidence"] == "roster:extraction_route"
        and row["observed_extraction_route_priority"] == "logical_primary"
        for row in candidate_rows
    )

    normalized_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert normalized_manifest["candidate_roster"]["sha256"] == hashlib.sha256(
        roster_path.read_bytes()
    ).hexdigest()
    route_coverage = normalized_manifest["candidate_roster_canonical_route_coverage"]
    assert route_coverage["status"] == "passed"
    assert {
        row["source_id"]: row["source_route"] for row in route_coverage["sources"]
    } == {"diavgeia": "mixed", "opengov_deliberations_v2": "mixed"}
    assert all(
        row["allowed_observed_extraction_routes"] == ["mixed"]
        and row["observed_extraction_route_counts"] == {"mixed": 1}
        for row in route_coverage["sources"]
    )

    drift_roster = tmp_path / "drift-roster.json"
    drift_roster.write_text(
        json.dumps(
            {
                "schema_version": "agent1_full_corpus_v3_candidate_roster_v1",
                "base_source_id": "nanochat_base",
                "candidate_source_ids": ["diavgeia"],
                "review_routes": {"diavgeia": "mixed"},
            }
        ),
        encoding="utf-8",
    )
    drift = subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "normalize_sources.py"),
            "--sources",
            str(config_path),
            "--lineage-aliases",
            str(aliases),
            "--acquisition-receipt",
            str(receipt_path),
            "--candidate-roster",
            str(drift_roster),
            "--output",
            str(tmp_path / "drift-canonical"),
            "--manifest",
            str(tmp_path / "drift-manifest.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert drift.returncode != 0
    assert "candidate roster/source registry coverage drift" in drift.stderr


def test_observed_extraction_is_per_document_and_canonical_coverage_requires_exception(
    tmp_path: Path,
) -> None:
    io = load_io_module()
    normalize = load_normalize_module()
    duckdb = pytest.importorskip("duckdb")

    artifact_path = tmp_path / "raw" / "source-a" / "rev" / "rows.parquet"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.touch()
    artifact = io.SourceArtifact(
        source_id="source-a",
        repo_id="owner/source-a",
        revision="rev",
        role="additive_candidate",
        source_family_id="source-a",
        files=(artifact_path,),
        file_bindings=(),
        config={"id_columns": ["id"], "text_columns": ["text"]},
    )
    row = io.canonical_row(
        source=artifact,
        artifact_path=artifact_path,
        artifact_row_index=0,
        raw_row={"id": "one", "content_type": "text/html"},
        representation_suffix="0",
        text_field="text",
        raw_text="κείμενο",
        lineage_aliases={"aliases": []},
        base_families={},
        declared_routes={
            "source-a": {
                "source_route": "pdf_ocr",
                "review_route": "pdf_ocr",
                "extraction_route": "pdf_ocr",
            }
        },
    )
    assert row["source_route"] == "pdf_ocr"
    assert row["extraction_route"] == "pdf_ocr"
    assert row["observed_extraction_route"] == "html_web"
    assert row["observed_extraction_route_basis"] == "row_representation_metadata"
    assert row["observed_extraction_route_evidence"] == "raw_metadata:content_type=text_html"
    assert row["observed_extraction_route_priority"] == "secondary_exception_only"

    with pytest.raises(ValueError, match="declared extraction route fallback must equal"):
        io.validate_observed_extraction_route_basis(
            observed_extraction_route="html_web",
            observed_extraction_route_basis="declared_extraction_route_fallback",
            declared_extraction_route="pdf_ocr",
            context="fixture",
        )
    with pytest.raises(ValueError, match="unavailable observed extraction route cannot carry"):
        io.validate_observed_extraction_route_basis(
            observed_extraction_route="pdf_ocr",
            observed_extraction_route_basis="unavailable",
            declared_extraction_route="pdf_ocr",
            context="fixture",
        )

    connection = duckdb.connect()
    try:
        connection.execute(
            """
            CREATE TABLE canonical_routes (
                acquisition_source_id VARCHAR,
                source_id VARCHAR,
                source_route VARCHAR,
                review_route VARCHAR,
                extraction_route VARCHAR,
                observed_extraction_route VARCHAR,
                observed_extraction_route_basis VARCHAR,
                observed_extraction_route_evidence VARCHAR,
                observed_extraction_route_priority VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO canonical_routes VALUES
            ('source-a', 'source-a', 'pdf_ocr', 'pdf_ocr', 'pdf_ocr',
             'html_web', 'row_representation_metadata', 'raw_metadata:content_type=text_html',
             'secondary_exception_only')
            """
        )
        declared = {
            "source-a": {
                "source_route": "pdf_ocr",
                "review_route": "pdf_ocr",
                "extraction_route": "pdf_ocr",
                "allowed_observed_extraction_routes": ["pdf_ocr", "html_web"],
            }
        }
        coverage = normalize.validate_candidate_canonical_route_coverage(
            connection,
            source_relation="canonical_routes",
            declared_routes=declared,
        )
        assert coverage["sources"][0]["observed_extraction_route_counts"] == {
            "html_web": 1
        }
        assert coverage["sources"][0]["observed"][0]["observed_route_priority"] == (
            "secondary_exception_only"
        )

        connection.execute("DELETE FROM canonical_routes")
        connection.execute(
            """
            INSERT INTO canonical_routes VALUES
            ('source-a', 'source-a', 'pdf_ocr', 'pdf_ocr', 'pdf_ocr',
             'html_web', 'row_representation_metadata', 'raw_metadata:content_type=text_html',
             'logical_primary')
            """
        )
        with pytest.raises(ValueError, match="canonical candidate route provenance drift"):
            normalize.validate_candidate_canonical_route_coverage(
                connection,
                source_relation="canonical_routes",
                declared_routes=declared,
            )

        connection.execute("DELETE FROM canonical_routes")
        connection.execute(
            """
            INSERT INTO canonical_routes VALUES
            ('source-a', 'source-a', 'pdf_ocr', 'pdf_ocr', 'pdf_ocr',
             'structured', 'row_representation_metadata', 'raw_metadata:format=json',
             'secondary_exception_only')
            """
        )
        with pytest.raises(ValueError, match="canonical candidate route provenance drift"):
            normalize.validate_candidate_canonical_route_coverage(
                connection,
                source_relation="canonical_routes",
                declared_routes=declared,
            )

        connection.execute("DELETE FROM canonical_routes")
        connection.execute(
            """
            INSERT INTO canonical_routes VALUES
            ('source-a', 'source-a', 'pdf_ocr', 'pdf_ocr', 'pdf_ocr',
             'html_web', 'declared_extraction_route_fallback', 'roster:extraction_route',
             'secondary_exception_only')
            """
        )
        with pytest.raises(ValueError, match="canonical candidate route provenance drift"):
            normalize.validate_candidate_canonical_route_coverage(
                connection,
                source_relation="canonical_routes",
                declared_routes=declared,
            )

        connection.execute("DELETE FROM canonical_routes")
        connection.execute(
            """
            INSERT INTO canonical_routes VALUES
            ('source-a', 'source-a', 'pdf_ocr', 'pdf_ocr', 'pdf_ocr',
             'pdf_ocr', 'unavailable', 'none', 'logical_primary')
            """
        )
        with pytest.raises(ValueError, match="canonical candidate route provenance drift"):
            normalize.validate_candidate_canonical_route_coverage(
                connection,
                source_relation="canonical_routes",
                declared_routes=declared,
            )
    finally:
        connection.close()


def _run_normalizer(
    config: Path,
    receipt: Path,
    output: Path,
    manifest: Path,
    *,
    workers: int,
    large_task_byte_threshold: int | None = None,
    large_task_workers: int = 1,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(HERE / "scripts" / "normalize_sources.py"),
        "--sources",
        str(config),
        "--acquisition-receipt",
        str(receipt),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        "--rows-per-shard",
        "2",
        "--workers",
        str(workers),
        "--large-task-workers",
        str(large_task_workers),
        "--duckdb-threads",
        "2",
        "--duckdb-memory-limit",
        "256MB",
    ]
    if large_task_byte_threshold is not None:
        command.extend(["--large-task-byte-threshold", str(large_task_byte_threshold)])
    return subprocess.run(command, text=True, capture_output=True)


def test_normalizer_parallel_outputs_are_deterministic_and_receipt_resumable(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")
    revision = "base-rev"
    raw_root = tmp_path / "hf" / revision
    raw_root.mkdir(parents=True)
    inputs = []
    for file_index in range(2):
        path = raw_root / f"part-{file_index}.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_dataset": ["dataset-a"] * 3,
                    "source_doc_id": [f"{file_index}-{row}" for row in range(3)],
                    "text": [f"κείμενο {file_index}-{row}" for row in range(3)],
                }
            ),
            path,
        )
        inputs.append(path)
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            {
                "base": {
                    "repo_id": "owner/base",
                    "revision": revision,
                    "role": "base",
                    "text_columns": ["text"],
                    "id_columns": ["source_doc_id"],
                    "source_column": "source_dataset",
                },
                "sources": [],
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "sources_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "sources": [
                    {
                        "source_id": "nanochat_base",
                        "revision": revision,
                        "files": [{"local_path": str(path)} for path in inputs],
                    }
                ],
            }
        )
    )
    first_output = tmp_path / "one-worker"
    second_output = tmp_path / "two-workers"
    first_manifest = tmp_path / "one.json"
    second_manifest = tmp_path / "two.json"
    assert (
        _run_normalizer(
            config, receipt, first_output, first_manifest, workers=1
        ).returncode
        == 0
    )
    large_pool_run = _run_normalizer(
        config,
        receipt,
        second_output,
        second_manifest,
        workers=2,
        large_task_byte_threshold=1,
        large_task_workers=1,
    )
    assert large_pool_run.returncode == 0, large_pool_run.stderr
    assert "starting pool=large tasks=2 workers=1" in large_pool_run.stdout

    def shard_contract(path: Path) -> list[tuple[int, str]]:
        value = json.loads(path.read_text())
        return [
            (int(shard["rows"]), str(shard["sha256"]))
            for source in value["sources"]
            for shard in source["shards"]
        ]

    assert shard_contract(first_manifest) == shard_contract(second_manifest)
    first = json.loads(first_manifest.read_text())
    assert first["uid_uniqueness"]["duplicates"] == 0
    assert first["uid_uniqueness"]["rows_checked"] == 6
    assert all("receipt" in shard for shard in first["sources"][0]["shards"])

    mtimes = {path: path.stat().st_mtime_ns for path in first_output.rglob("*.parquet")}
    manifest_sha256 = hashlib.sha256(first_manifest.read_bytes()).hexdigest()
    first_manifest.unlink()
    resumed = _run_normalizer(
        config,
        receipt,
        first_output,
        first_manifest,
        workers=2,
        large_task_byte_threshold=1,
        large_task_workers=1,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert mtimes == {
        path: path.stat().st_mtime_ns for path in first_output.rglob("*.parquet")
    }
    assert hashlib.sha256(first_manifest.read_bytes()).hexdigest() == manifest_sha256

    # The tiny legacy receipt has no upstream content ID, so the normalizer
    # computes one. Changing an input therefore fails the persisted contract.
    pq.write_table(
        pa.table(
            {
                "source_dataset": ["dataset-a"],
                "source_doc_id": ["changed"],
                "text": ["άλλο"],
            }
        ),
        inputs[0],
    )
    first_manifest.unlink()
    drift = _run_normalizer(config, receipt, first_output, first_manifest, workers=1)
    assert drift.returncode != 0
    assert "resume contract drift" in drift.stderr


def test_global_uid_pass_rejects_cross_file_identity_collision(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")
    inputs = []
    for parent in (tmp_path / "a" / "rev", tmp_path / "b" / "rev"):
        parent.mkdir(parents=True)
        path = parent / "same.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_dataset": ["same-source"],
                    "source_doc_id": ["same-doc"],
                    "text": ["ίδιο"],
                }
            ),
            path,
        )
        inputs.append(path)
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            {
                "base": {
                    "repo_id": "owner/base",
                    "revision": "rev",
                    "role": "base",
                    "text_columns": ["text"],
                    "id_columns": ["source_doc_id"],
                    "source_column": "source_dataset",
                },
                "sources": [],
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "sources_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "sources": [
                    {
                        "source_id": "nanochat_base",
                        "revision": "rev",
                        "files": [{"local_path": str(path)} for path in inputs],
                    }
                ],
            }
        )
    )
    completed = _run_normalizer(
        config,
        receipt,
        tmp_path / "normalized",
        tmp_path / "manifest.json",
        workers=2,
    )
    assert completed.returncode != 0
    assert "duplicate stable_uid after spillable global pass" in completed.stderr
