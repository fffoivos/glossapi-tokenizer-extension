from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PHASE = Path(__file__).resolve().parents[1]
SCRIPTS = PHASE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_token_waterfall import build_waterfall  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _corpus_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("text", pa.string()),
            ("title", pa.string()),
            ("author", pa.string()),
            ("greek_badness_score", pa.float64()),
            ("mojibake_badness_score", pa.float64()),
            ("needs_ocr", pa.bool_()),
            ("is_empty", pa.bool_()),
            ("ocr_success", pa.bool_()),
            ("is_historical_or_polytonic", pa.bool_()),
            ("source_family_id", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_repo_id", pa.string()),
            ("source_revision", pa.string()),
            ("source_artifact_path", pa.string()),
            ("source_row_id", pa.string()),
            ("stable_uid", pa.string()),
            ("cleaned_text_sha256", pa.string()),
            ("source_metadata_json", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("eligible_for_redistribution", pa.bool_()),
        ]
    )


def _corpus_row(uid: str, text: str, *, redistributable: bool) -> dict[str, object]:
    return {
        "source_dataset": "demo_source",
        "source_doc_id": f"upstream-{uid}",
        "text": text,
        "title": "Δοκιμή",
        "author": "Author withheld from public output",
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": False,
        "is_empty": False,
        "ocr_success": True,
        "is_historical_or_polytonic": False,
        "source_family_id": "demo",
        "acquisition_source_id": "demo_acquisition",
        "source_repo_id": "org/demo",
        "source_revision": "0123456789abcdef",
        "source_artifact_path": "data/train-00000.parquet",
        "source_row_id": uid,
        "stable_uid": uid,
        "cleaned_text_sha256": _sha(text),
        "source_metadata_json": json.dumps({"private": "not public"}),
        "eligible_for_training": True,
        "eligible_for_redistribution": redistributable,
    }


def _write_parquet(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _decision_schema() -> pa.Schema:
    return pa.schema(
        [
            ("doc_key", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("decision", pa.string()),
            ("decision_stage", pa.string()),
            ("cluster_id", pa.string()),
            ("kept_doc_key", pa.string()),
            ("reason", pa.string()),
            ("exact_strict_version", pa.string()),
            ("exact_relaxed_version", pa.string()),
            ("near_norm_version", pa.string()),
            ("tokenization_version", pa.string()),
            ("shingle_version", pa.string()),
            ("minhash_version", pa.string()),
            ("lsh_version", pa.string()),
            ("selection_version", pa.string()),
        ]
    )


def _decision(uid: str, decision: str, stage: str, kept: str) -> dict[str, object]:
    return {
        "doc_key": f"key-{uid}",
        "source_dataset": "demo_source",
        "source_doc_id": uid,
        "decision": decision,
        "decision_stage": stage,
        "cluster_id": None if decision == "keep" else "strict:duplicate",
        "kept_doc_key": f"key-{kept}",
        "reason": "kept" if decision == "keep" else stage,
        "exact_strict_version": "strict-v1",
        "exact_relaxed_version": "relaxed-v1",
        "near_norm_version": "near-v1",
        "tokenization_version": "tokens-v1",
        "shingle_version": "shingles-v1",
        "minhash_version": "minhash-v1",
        "lsh_version": "lsh-v1",
        "selection_version": "selection-v1",
    }


def _ledger_schema() -> pa.Schema:
    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("action", pa.string()),
            ("reasons_json", pa.string()),
            ("tokens_normalized", pa.int64()),
            ("tokens_source_cleaned", pa.int64()),
            ("tokens_pii_masked", pa.int64()),
            ("tokens_structural_cleaned", pa.int64()),
            ("tokens_final", pa.int64()),
        ]
    )


def _decontam_schema() -> pa.Schema:
    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("action", pa.string()),
            ("reason", pa.string()),
            ("benchmark_matches_json", pa.string()),
        ]
    )


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_dedup_staging_materialization_waterfall_and_validation(tmp_path: Path) -> None:
    corpus = tmp_path / "decontaminated"
    duplicate_text = "Αυτό είναι ένα πλήρες ελληνικό έγγραφο για δοκιμή."
    rows = [
        _corpus_row("uid-a", duplicate_text, redistributable=True),
        _corpus_row("uid-b", duplicate_text, redistributable=True),
        _corpus_row("uid-c", "Ένα δεύτερο και μοναδικό ελληνικό έγγραφο.", redistributable=False),
    ]
    _write_parquet(corpus / "demo" / "part-00000.parquet", rows, _corpus_schema())

    staged = tmp_path / "dedup-staged"
    stage_manifest = tmp_path / "dedup-stage.json"
    _run(
        "run_full_corpus_dedup.py",
        "--input",
        str(corpus),
        "--staged-input",
        str(staged),
        "--state-root",
        str(tmp_path / "state"),
        "--run-root",
        str(tmp_path / "dedup-run"),
        "--manifest",
        str(stage_manifest),
        "--temporary-directory",
        str(tmp_path / "duck-stage"),
        "--workers",
        "2",
        "--stage-only",
    )
    staged_rows = pq.read_table(staged / "demo" / "part-00000.parquet").to_pylist()
    assert [row["source_doc_id"] for row in staged_rows] == ["uid-a", "uid-b", "uid-c"]
    assert [row["upstream_source_doc_id"] for row in staged_rows] == [
        "upstream-uid-a",
        "upstream-uid-b",
        "upstream-uid-c",
    ]
    assert json.loads(stage_manifest.read_text())["dedup_implementation"]["reimplemented"] is False

    decisions = tmp_path / "dedup-run" / "final" / "dedup_decisions.parquet"
    _write_parquet(
        decisions,
        [
            _decision("uid-a", "keep", "kept_after_exact", "uid-a"),
            _decision("uid-b", "drop", "strict_exact", "uid-a"),
            _decision("uid-c", "keep", "kept_after_exact", "uid-c"),
        ],
        _decision_schema(),
    )

    cleaning = tmp_path / "cleaning-ledger"
    cleaning_rows = []
    for uid in ("uid-a", "uid-b", "uid-c", "uid-d"):
        cleaning_rows.append(
            {
                "stable_uid": uid,
                "acquisition_source_id": "demo_acquisition",
                "source_dataset": "demo_source",
                "source_doc_id": f"upstream-{uid}",
                "action": "keep",
                "reasons_json": "[]",
                "tokens_normalized": 12,
                "tokens_source_cleaned": 11,
                "tokens_pii_masked": 11,
                "tokens_structural_cleaned": 10,
                "tokens_final": 10,
            }
        )
    _write_parquet(cleaning / "part.parquet", cleaning_rows, _ledger_schema())
    decontam = tmp_path / "decontam-ledger"
    decontam_rows = []
    for uid in ("uid-a", "uid-b", "uid-c", "uid-d"):
        decontam_rows.append(
            {
                "stable_uid": uid,
                "acquisition_source_id": "demo_acquisition",
                "source_dataset": "demo_source",
                "source_doc_id": f"upstream-{uid}",
                "input_text_sha256": "hash",
                "action": "drop" if uid == "uid-d" else "keep",
                "reason": "greekmmlu_exact_prompt" if uid == "uid-d" else "no_high_confidence_match",
                "benchmark_matches_json": "[]",
            }
        )
    _write_parquet(decontam / "part.parquet", decontam_rows, _decontam_schema())

    waterfall = tmp_path / "token-waterfall.json"
    payload = build_waterfall(
        cleaning_ledger=cleaning,
        decontam_ledger=decontam,
        dedup_decisions=decisions,
        output=waterfall,
        temporary_directory=tmp_path / "duck-waterfall",
        memory_limit="1GB",
        threads=2,
    )
    assert payload["invariants"]["final_tokens"] == 20
    assert payload["invariants"]["reconciled"] is True
    assert any(row["stage"] == "toc_bib" and row["tokens_removed"] == 4 for row in payload["events_global"])
    assert any(row["stage"] == "greekmmlu_decontamination" and row["tokens_removed"] == 10 for row in payload["events_global"])
    assert any(row["stage"] == "strict_exact" and row["tokens_removed"] == 10 for row in payload["events_global"])

    release = tmp_path / "release"
    release_manifest = tmp_path / "release-manifest.json"
    _run(
        "materialize_release.py",
        "--input",
        str(corpus),
        "--dedup-decisions",
        str(decisions),
        "--output",
        str(release),
        "--manifest",
        str(release_manifest),
        "--token-waterfall",
        str(waterfall),
        "--temporary-directory",
        str(tmp_path / "duck-materialize"),
        "--memory-limit",
        "1GB",
        "--threads",
        "2",
    )
    training = pq.read_table(release / "training" / "data" / "demo" / "part-00000.parquet")
    redistribution = pq.read_table(release / "redistribution" / "data" / "demo" / "part-00000.parquet")
    assert set(training["stable_uid"].to_pylist()) == {"uid-a", "uid-c"}
    assert redistribution["stable_uid"].to_pylist() == ["uid-a"]
    assert "source_metadata_json" not in redistribution.schema.names
    assert "author" not in redistribution.schema.names

    receipt = tmp_path / "validation.json"
    _run(
        "validate_release.py",
        "--release",
        str(release),
        "--manifest",
        str(release_manifest),
        "--dedup-decisions",
        str(decisions),
        "--output",
        str(receipt),
        "--temporary-directory",
        str(tmp_path / "duck-validate"),
        "--memory-limit",
        "1GB",
        "--threads",
        "2",
    )
    validation = json.loads(receipt.read_text())
    assert validation["status"] == "passed"
    assert not validation["failed_checks"]
