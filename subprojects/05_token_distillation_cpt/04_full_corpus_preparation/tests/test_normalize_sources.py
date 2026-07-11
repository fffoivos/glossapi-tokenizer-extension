from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_lineage_module():
    path = HERE / "scripts" / "source_lineage.py"
    spec = importlib.util.spec_from_file_location("phase04_normalize_lineage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalizer_preserves_source_names_and_groups_sectioned_works(tmp_path: Path) -> None:
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
                "id": ["work-b", "work-a", "work-b"],
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
