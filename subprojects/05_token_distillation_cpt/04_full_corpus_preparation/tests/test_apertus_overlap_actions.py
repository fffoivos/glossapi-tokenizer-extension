from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "build_apertus_overlap_actions.py"


def test_apertus_overlay_is_natural_key_joined_and_text_hash_bound(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")

    canonical = tmp_path / "canonical"
    base = canonical / "nanochat_base" / "part.parquet"
    base.parent.mkdir(parents=True)
    texts = ["Κείμενο Α", "Κείμενο Β"]
    pq.write_table(
        pa.table(
            {
                "stable_uid": ["a" * 64, "b" * 64],
                "source_dataset": ["demo", "demo"],
                "source_doc_id": ["doc-a", "doc-b"],
                "normalized_text_sha256": [
                    hashlib.sha256(value.encode()).hexdigest() for value in texts
                ],
            }
        ),
        base,
    )
    overlay = tmp_path / "apertus_overlap_drop_docs.parquet"
    pq.write_table(
        pa.table(
            {
                "doc_key": ["c" * 64],
                "source_dataset": ["demo"],
                "source_doc_id": ["doc-b"],
                "best_overlap_stage": ["near_duplicate"],
                "best_estimated_jaccard": [0.97],
            }
        ),
        overlay,
    )
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {
                "apertus_overlap_overlay": {
                    "repo_id": "owner/overlay",
                    "revision": "d" * 40,
                }
            }
        )
    )
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "sources": [
                    {
                        "source_id": "apertus_overlap_overlay",
                        "repo_id": "owner/overlay",
                        "revision": "d" * 40,
                        "files": [{"local_path": str(overlay)}],
                    }
                ],
            }
        )
    )
    normalization = tmp_path / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_normalization_manifest_v1",
                "output": str(canonical.resolve()),
            }
        )
    )
    output = tmp_path / "actions.parquet"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--canonical-root",
            str(canonical),
            "--normalization-manifest",
            str(normalization),
            "--acquisition-receipt",
            str(acquisition),
            "--sources",
            str(sources),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--temporary-directory",
            str(tmp_path / "duckdb"),
            "--memory-limit",
            "1GB",
            "--threads",
            "2",
        ],
        check=True,
    )
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 1
    assert rows[0]["stable_uid"] == "b" * 64
    assert rows[0]["action"] == "drop"
    assert rows[0]["reason"] == "apertus_pretraining_overlap"
    assert rows[0]["input_text_sha256"] == hashlib.sha256(texts[1].encode()).hexdigest()
    receipt = json.loads(manifest.read_text())
    assert receipt["status"] == "completed"
    assert receipt["counts"]["unmatched_natural_keys"] == 0
    assert receipt["counts"]["action_rows"] == 1
