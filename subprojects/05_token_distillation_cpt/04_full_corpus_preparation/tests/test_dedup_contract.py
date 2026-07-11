from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PHASE = Path(__file__).resolve().parents[1]
SCRIPTS = PHASE / "scripts"
REPO_ROOT = PHASE.parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glossapi_corpus_cli import text_dedup  # noqa: E402
from run_full_corpus_dedup import build_content_bound_decisions  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("stable_uid", pa.string()),
            ("text", pa.string()),
            ("cleaned_text_sha256", pa.string()),
            ("eligible_for_training", pa.bool_()),
            ("title", pa.string()),
            ("author", pa.string()),
            ("greek_badness_score", pa.float64()),
            ("mojibake_badness_score", pa.float64()),
            ("needs_ocr", pa.bool_()),
            ("is_empty", pa.bool_()),
            ("ocr_success", pa.bool_()),
            ("is_historical_or_polytonic", pa.bool_()),
        ]
    )


def _canonical_row(uid: str, text: str) -> dict[str, object]:
    return {
        "source_dataset": "demo",
        "source_doc_id": f"upstream-{uid}",
        "stable_uid": uid,
        "text": text,
        "cleaned_text_sha256": _sha(text),
        "eligible_for_training": True,
        "title": "title",
        "author": "author",
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": False,
        "is_empty": False,
        "ocr_success": True,
        "is_historical_or_polytonic": False,
    }


def _write(path: Path, rows: list[dict[str, object]], schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _wrapper_args(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "run_full_corpus_dedup.py"),
        "--input",
        str(tmp_path / "input"),
        "--staged-input",
        str(tmp_path / "staged"),
        "--state-root",
        str(tmp_path / "state"),
        "--run-root",
        str(tmp_path / "run"),
        "--manifest",
        str(tmp_path / "dedup.json"),
        "--temporary-directory",
        str(tmp_path / "duckdb"),
        "--memory-limit",
        "128MB",
        "--workers",
        "2",
        "--duckdb-threads",
        "1",
        "--staging-workers",
        "2",
        "--stage-only",
    ]


def test_stage_receipt_allows_verified_resume_with_nonempty_staging(tmp_path: Path) -> None:
    _write(
        tmp_path / "input" / "one.parquet",
        [_canonical_row("uid-1", "Πρώτο ελληνικό κείμενο")],
        _canonical_schema(),
    )
    _write(
        tmp_path / "input" / "nested" / "two.parquet",
        [_canonical_row("uid-2", "Δεύτερο ελληνικό κείμενο")],
        _canonical_schema(),
    )
    command = _wrapper_args(tmp_path)
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    manifest_path = tmp_path / "dedup.json"
    manifest_before = manifest_path.read_bytes()

    resumed = subprocess.run(
        [*command, "--resume"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(first.stdout)["status"] == "staged"
    assert json.loads(resumed.stdout)["status"] == "staged"
    assert manifest_path.read_bytes() == manifest_before
    manifest = json.loads(manifest_before)
    assert manifest["staging_contract"]["requested_workers"] == 2
    assert manifest["staging_contract"]["effective_workers"] == 2
    assert manifest["staging_contract"]["max_workers"] == 16
    assert manifest["counts"]["text_hash_verified_rows"] == 2
    assert manifest["recipe"]["mode"] == "production"
    assert Path(manifest["command"][1]).name == "invoke_text_dedup.py"


def test_resume_rejects_staged_content_that_no_longer_matches_receipt(tmp_path: Path) -> None:
    _write(
        tmp_path / "input" / "one.parquet",
        [_canonical_row("uid-1", "Ελληνικό κείμενο")],
        _canonical_schema(),
    )
    command = _wrapper_args(tmp_path)
    subprocess.run(command, check=True, capture_output=True, text=True)
    staged = tmp_path / "staged" / "one.parquet"
    staged.write_bytes(staged.read_bytes() + b"drift")

    result = subprocess.run([*command, "--resume"], capture_output=True, text=True)

    assert result.returncode != 0
    assert "staging receipt byte-size mismatch" in result.stderr


def test_partial_staging_progress_recovers_without_deleting_verified_outputs(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "input" / "one.parquet",
        [_canonical_row("uid-1", "Ελληνικό κείμενο επανεκκίνησης")],
        _canonical_schema(),
    )
    command = _wrapper_args(tmp_path)
    subprocess.run(command, check=True, capture_output=True, text=True)
    staged = tmp_path / "staged" / "one.parquet"
    staged_sha256 = hashlib.sha256(staged.read_bytes()).hexdigest()
    (tmp_path / "dedup.json").unlink()

    resumed = subprocess.run(command, check=True, capture_output=True, text=True)

    assert json.loads(resumed.stdout)["status"] == "staged"
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == staged_sha256
    assert (tmp_path / "dedup.json").is_file()


def test_production_recipe_override_requires_explicit_experimental_mode(tmp_path: Path) -> None:
    command = [*_wrapper_args(tmp_path), "--minhash-threshold", "0.80"]
    rejected = subprocess.run(command, capture_output=True, text=True)

    assert rejected.returncode != 0
    assert "production dedup recipe is immutable" in rejected.stderr


def test_content_bound_decisions_cover_every_staged_uid(tmp_path: Path) -> None:
    staged = tmp_path / "staged.parquet"
    _write(
        staged,
        [
            {
                "source_doc_id": "uid-1",
                "stable_uid": "uid-1",
                "source_dataset": "demo",
                "cleaned_text_sha256": _sha("one"),
            },
            {
                "source_doc_id": "uid-2",
                "stable_uid": "uid-2",
                "source_dataset": "demo",
                "cleaned_text_sha256": _sha("two"),
            },
        ],
    )
    raw = tmp_path / "raw.parquet"
    _write(
        raw,
        [
            {
                "doc_key": "key-1",
                "source_dataset": "demo",
                "source_doc_id": "uid-1",
                "decision": "keep",
            },
            {
                "doc_key": "key-2",
                "source_dataset": "demo",
                "source_doc_id": "uid-2",
                "decision": "drop",
            },
        ],
    )
    output = tmp_path / "bound.parquet"

    receipt, coverage = build_content_bound_decisions(
        staged_files=[staged],
        raw_decisions=raw,
        output=output,
        temporary_directory=tmp_path / "duckdb",
        memory_limit="128MB",
        threads=1,
    )

    rows = pq.read_table(output).to_pylist()
    assert [row["stable_uid"] for row in rows] == ["uid-1", "uid-2"]
    assert [row["input_text_sha256"] for row in rows] == [_sha("one"), _sha("two")]
    assert coverage == {
        "staged_rows": 2,
        "decision_rows": 2,
        "duplicate_decisions": 0,
        "missing_decisions": 0,
        "decisions_without_input": 0,
        "source_dataset_drift": 0,
        "identity_drift": 0,
    }
    assert receipt["schema_version"] == "full_cpt_dedup_decisions_content_bound_v1"
    assert receipt["sha256"]

    resumed_receipt, resumed_coverage = build_content_bound_decisions(
        staged_files=[staged],
        raw_decisions=raw,
        output=output,
        temporary_directory=tmp_path / "duckdb",
        memory_limit="128MB",
        threads=1,
    )
    assert resumed_receipt == receipt
    assert resumed_coverage == coverage


def test_wrapper_completion_uses_lightweight_invoker_and_emits_bound_manifest(tmp_path: Path) -> None:
    duplicate_text = "Αυτό είναι ένα επαρκώς μεγάλο ελληνικό κείμενο για έλεγχο διπλοτύπων."
    _write(
        tmp_path / "input" / "one.parquet",
        [
            _canonical_row("uid-1", duplicate_text),
            _canonical_row("uid-2", duplicate_text),
        ],
        _canonical_schema(),
    )
    command = _wrapper_args(tmp_path)
    subprocess.run(command, check=True, capture_output=True, text=True)
    command.remove("--stage-only")
    command.append("--reuse-staged")

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert json.loads(result.stdout)["status"] == "completed"
    manifest = json.loads((tmp_path / "dedup.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["recipe"]["mode"] == "production"
    content_bound = manifest["dedup_output"]["content_bound_decisions"]
    assert content_bound["schema_version"] == "full_cpt_dedup_decisions_content_bound_v1"
    assert content_bound == manifest["dedup_output"]["decisions"]
    bound_path = Path(content_bound["path"])
    assert bound_path.is_file()
    assert hashlib.sha256(bound_path.read_bytes()).hexdigest() == content_bound["sha256"]
    assert pq.read_table(bound_path).num_rows == 2
    assert manifest["dedup_output"]["content_binding"]["missing_decisions"] == 0
    assert manifest["dedup_output"]["summary_sha256"]
    assert manifest["staging_manifest_sha256"]


def test_content_bound_decisions_fail_closed_on_incomplete_decision_set(tmp_path: Path) -> None:
    staged = tmp_path / "staged.parquet"
    _write(
        staged,
        [
            {
                "source_doc_id": "uid-1",
                "stable_uid": "uid-1",
                "source_dataset": "demo",
                "cleaned_text_sha256": _sha("one"),
            }
        ],
    )
    raw = tmp_path / "raw.parquet"
    _write(
        raw,
        [],
        pa.schema(
            [
                ("doc_key", pa.string()),
                ("source_dataset", pa.string()),
                ("source_doc_id", pa.string()),
                ("decision", pa.string()),
            ]
        ),
    )

    with pytest.raises(ValueError, match="decision coverage/content-binding gate failed"):
        build_content_bound_decisions(
            staged_files=[staged],
            raw_decisions=raw,
            output=tmp_path / "bound.parquet",
            temporary_directory=tmp_path / "duckdb",
            memory_limit="128MB",
            threads=1,
        )


def test_duckdb_environment_contract_reaches_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_directory = tmp_path / "duckdb-with-'quote"
    monkeypatch.setenv(text_dedup.DUCKDB_TEMP_DIRECTORY_ENV, str(temporary_directory))
    monkeypatch.setenv(text_dedup.DUCKDB_MEMORY_LIMIT_ENV, "64MB")
    monkeypatch.setenv(text_dedup.DUCKDB_THREADS_ENV, "2")

    connection = text_dedup.connect_duckdb()
    try:
        threads, memory_limit, temp_directory = connection.execute(
            "SELECT current_setting('threads'), current_setting('memory_limit'), "
            "current_setting('temp_directory')"
        ).fetchone()
    finally:
        connection.close()

    assert int(threads) == 2
    assert "MiB" in str(memory_limit)
    assert Path(str(temp_directory)) == temporary_directory.resolve()
    assert temporary_directory.is_dir()


def test_lightweight_invoker_does_not_import_broad_cli() -> None:
    source = (SCRIPTS / "invoke_text_dedup.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "glossapi_corpus_cli" in imported_modules
    assert "glossapi_corpus_cli.cli" not in imported_modules
    assert "typer" not in imported_modules
