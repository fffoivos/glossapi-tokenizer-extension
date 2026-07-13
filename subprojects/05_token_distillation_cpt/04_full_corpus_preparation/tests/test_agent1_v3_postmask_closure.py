from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PHASE = Path(__file__).resolve().parents[1]
LEDGER_CLOSURE = PHASE / "scripts" / "agent1_v3_anonymization_ledger_closure.py"
POSTMASK_REPORT = PHASE / "scripts" / "agent1_v3_postmask_duplicate_report.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _run(script: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args], text=True, capture_output=True, check=check
    )


def _completed_anonymization_manifest(
    tmp_path: Path, protected_root: Path, shards: list[Path]
) -> Path:
    manifest = tmp_path / "anonymization.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "agent1_full_corpus_v3_anonymization_manifest_v1",
                "status": "completed",
                "protected_ledger": {
                    "path": str(protected_root.resolve()),
                    "contains_raw_span_values": True,
                    "public_training_output": False,
                    "directory_mode": "0700",
                    "file_mode": "0600",
                },
                "counts": {
                    "protected_ledger_rows": sum(int(_receipt(shard)["rows"]) for shard in shards)
                },
                "files": [
                    {"protected_ledger": _receipt(shard, relative_to=protected_root)}
                    for shard in shards
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest


def _run_ledger_closure(
    *, manifest: Path, protected_root: Path, output: Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return _run(
        LEDGER_CLOSURE,
        "--anonymization-manifest",
        str(manifest),
        "--protected-ledger-root",
        str(protected_root),
        "--output",
        str(output),
        check=check,
    )


def test_protected_ledger_closure_binds_every_private_shard_without_span_values(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    protected_root = tmp_path / "protected"
    shard = protected_root / "source-a" / "part.parquet"
    raw_email = "private.person@example.gr"
    _write_parquet(
        shard,
        [
            {
                "stable_uid": "uid-1",
                "action": "keep",
                "protected_spans_json": json.dumps([{ "raw_value": raw_email }]),
            }
        ],
    )
    protected_root.chmod(0o700)
    shard.chmod(0o600)
    manifest = _completed_anonymization_manifest(tmp_path, protected_root, [shard])
    output = tmp_path / "ledger-closure.json"
    _run_ledger_closure(manifest=manifest, protected_root=protected_root, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "agent1_full_corpus_v3_protected_anonymization_ledger_closure_v1"
    assert payload["status"] == "passed"
    assert payload["counts"] == {"protected_ledger_rows": 1, "shards": 1}
    assert payload["protected_ledger"]["public_training_output"] is False
    assert payload["files"][0]["sha256"] == _sha256(shard)
    assert raw_email not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("unsafe_target", "mode", "expected_error"),
    [
        ("root", 0o755, "must have mode 0700"),
        ("shard", 0o640, "must have mode 0600"),
    ],
)
def test_protected_ledger_closure_rejects_unsafe_filesystem_modes(
    tmp_path: Path, unsafe_target: str, mode: int, expected_error: str
) -> None:
    pytest.importorskip("pyarrow")
    protected_root = tmp_path / "protected"
    shard = protected_root / "source-a" / "part.parquet"
    _write_parquet(shard, [{"stable_uid": "uid-1", "protected_spans_json": "[]"}])
    protected_root.chmod(0o700)
    shard.chmod(0o600)
    manifest = _completed_anonymization_manifest(tmp_path, protected_root, [shard])
    (protected_root if unsafe_target == "root" else shard).chmod(mode)

    output = tmp_path / "ledger-closure.json"
    result = _run_ledger_closure(
        manifest=manifest,
        protected_root=protected_root,
        output=output,
        check=False,
    )
    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not output.exists()


def test_protected_ledger_closure_rejects_unreceipted_parquet_shards(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    protected_root = tmp_path / "protected"
    expected = protected_root / "source-a" / "part.parquet"
    extra = protected_root / "source-b" / "unreceipted.parquet"
    _write_parquet(expected, [{"stable_uid": "uid-1", "protected_spans_json": "[]"}])
    _write_parquet(extra, [{"stable_uid": "uid-2", "protected_spans_json": "[]"}])
    protected_root.chmod(0o700)
    expected.chmod(0o600)
    extra.chmod(0o600)
    manifest = _completed_anonymization_manifest(tmp_path, protected_root, [expected])

    output = tmp_path / "ledger-closure.json"
    result = _run_ledger_closure(
        manifest=manifest,
        protected_root=protected_root,
        output=output,
        check=False,
    )
    assert result.returncode != 0
    assert "inventory does not exactly match" in result.stderr
    assert not output.exists()


def test_protected_ledger_closure_rejects_symlinked_shards(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    protected_root = tmp_path / "protected"
    target = protected_root / "source-a" / "actual.parquet"
    shard = protected_root / "source-a" / "part.parquet"
    _write_parquet(target, [{"stable_uid": "uid-1", "protected_spans_json": "[]"}])
    protected_root.chmod(0o700)
    target.chmod(0o600)
    shard.symlink_to(target.name)
    manifest = _completed_anonymization_manifest(tmp_path, protected_root, [shard])

    output = tmp_path / "ledger-closure.json"
    result = _run_ledger_closure(
        manifest=manifest,
        protected_root=protected_root,
        output=output,
        check=False,
    )
    assert result.returncode != 0
    assert "contains a symlink" in result.stderr
    assert not output.exists()


def test_protected_ledger_closure_rejects_symlinked_root(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    protected_root = tmp_path / "protected"
    shard = protected_root / "source-a" / "part.parquet"
    _write_parquet(shard, [{"stable_uid": "uid-1", "protected_spans_json": "[]"}])
    protected_root.chmod(0o700)
    shard.chmod(0o600)
    manifest = _completed_anonymization_manifest(tmp_path, protected_root, [shard])
    symlinked_root = tmp_path / "protected-link"
    symlinked_root.symlink_to(protected_root, target_is_directory=True)

    output = tmp_path / "ledger-closure.json"
    result = _run_ledger_closure(
        manifest=manifest,
        protected_root=symlinked_root,
        output=output,
        check=False,
    )
    assert result.returncode != 0
    assert "missing/unsafe" in result.stderr
    assert not output.exists()


PRODUCTION_PARAMETERS = {
    "greek_diacritic_policy": "preserve",
    "minhash_threshold": 0.85,
    "num_perm": 128,
    "bands": 32,
    "rows_per_band": 4,
    "shingle_mode": "token",
    "shingle_size": 5,
    "max_bucket_size": 5000,
}


def _postmask_fixture(tmp_path: Path, decisions: list[str]) -> tuple[Path, Path, Path, Path]:
    """Build a fully receipted anonymized inventory and detector result."""

    corpus = tmp_path / "anonymized"
    corpus_shard = corpus / "source" / "part.parquet"
    corpus_rows = [
        {
            "stable_uid": f"uid-{index}",
            "text": f"masked-{index}",
            "cleaned_text_sha256": hashlib.sha256(f"masked-{index}".encode("utf-8")).hexdigest(),
        }
        for index in range(len(decisions))
    ]
    _write_parquet(corpus_shard, corpus_rows)
    anonymization = tmp_path / "anonymization.json"
    anonymization.write_text(
        json.dumps(
            {
                "schema_version": "agent1_full_corpus_v3_anonymization_manifest_v1",
                "status": "completed",
                "output": str(corpus.resolve()),
                "files": [
                    {
                        "relative_path": "source/part.parquet",
                        "output": _receipt(corpus_shard, relative_to=corpus),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    raw = tmp_path / "detector" / "final" / "dedup_decisions.parquet"
    rows = [
        {
            "doc_key": f"doc-{index}",
            "source_doc_id": f"uid-{index}",
            "decision": decision,
            "kept_doc_key": f"doc-{index}" if decision == "keep" else "doc-0",
            "decision_stage": "exact" if decision == "drop" else "identity",
        }
        for index, decision in enumerate(decisions)
    ]
    _write_parquet(raw, rows)
    bound = tmp_path / "detector" / "final" / "dedup_decisions_content_bound.parquet"
    _write_parquet(
        bound,
        [
            {
                **row,
                "stable_uid": row["source_doc_id"],
                "input_text_sha256": corpus_rows[index]["cleaned_text_sha256"],
            }
            for index, row in enumerate(rows)
        ],
    )
    wrapper = tmp_path / "postmask-wrapper.json"
    wrapper.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_dedup_wrapper_manifest_v1",
                "status": "completed",
                "input": str(corpus.resolve()),
                "identity_contract": {
                    "input_text_sha256": "verified canonical cleaned_text_sha256",
                },
                "recipe": {
                    "id": "greek_cpt_text_dedup_v1",
                    "mode": "production",
                    "approved_production_parameters": PRODUCTION_PARAMETERS,
                },
                "dedup_parameters": PRODUCTION_PARAMETERS,
                "dedup_output": {
                    "raw_decisions": _receipt(raw),
                    "content_bound_decisions": {
                        **_receipt(bound),
                        "schema_version": "full_cpt_dedup_decisions_content_bound_v1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return corpus, anonymization, wrapper, bound


@pytest.mark.parametrize("decisions, expected_count", [(["keep", "keep"], 0), (["keep", "drop"], 1)])
def test_postmask_report_is_verification_only_and_counts_new_collisions(
    tmp_path: Path, decisions: list[str], expected_count: int
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    corpus, anonymization, wrapper, _ = _postmask_fixture(tmp_path, decisions)
    output = tmp_path / "postmask-report.json"
    _run(
        POSTMASK_REPORT,
        "--dedup-wrapper-manifest",
        str(wrapper),
        "--anonymization-manifest",
        str(anonymization),
        "--source-corpus",
        str(corpus),
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verification_only"] is True
    assert report["materialization_performed"] is False
    assert report["second_deduplication_applied"] is False
    assert report["material_new_duplicate_count"] == expected_count
    assert report["requires_explicit_user_decision"] is (expected_count > 0)
    assert report["decision_counts"] == {"drop": expected_count, "keep": len(decisions) - expected_count}
    assert report["inventory_closure"]["corpus_rows"] == len(decisions)
    assert report["inventory_closure"]["text_hash_drift"] == 0


def test_postmask_report_rejects_missing_detector_decision(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    corpus, anonymization, wrapper, bound = _postmask_fixture(tmp_path, ["keep", "keep"])
    import pyarrow.parquet as pq

    # Keep the manifest receipt valid while deliberately making the bound
    # detector inventory incomplete; the report must not manufacture a zero.
    one_row = pq.read_table(bound).slice(0, 1).to_pylist()
    _write_parquet(bound, one_row)
    payload = json.loads(wrapper.read_text(encoding="utf-8"))
    payload["dedup_output"]["content_bound_decisions"] = {
        **_receipt(bound),
        "schema_version": "full_cpt_dedup_decisions_content_bound_v1",
    }
    wrapper.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "postmask-report.json"
    result = _run(
        POSTMASK_REPORT,
        "--dedup-wrapper-manifest",
        str(wrapper),
        "--anonymization-manifest",
        str(anonymization),
        "--source-corpus",
        str(corpus),
        "--output",
        str(output),
        check=False,
    )
    assert result.returncode != 0
    assert "inventory" in result.stderr
    assert not output.exists()
