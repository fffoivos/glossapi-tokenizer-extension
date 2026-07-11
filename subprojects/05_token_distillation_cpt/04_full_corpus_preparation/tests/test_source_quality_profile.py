from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROFILE = load_module("phase04_source_quality", HERE / "scripts" / "profile_source_quality.py")
VERIFY = load_module("phase04_verify_staged_schemas", HERE / "scripts" / "verify_staged_schemas.py")


def test_template_fingerprint_abstracts_variable_identifiers() -> None:
    first = "Απόφαση ΑΔΑ ΨΓΠ9469ΗΡ8-ΧΚ5\nΠοσό 1.234,56\nDigitally signed by Ministry of Digital Governance 2026-01-01"
    second = "Απόφαση ΑΔΑ ΑΒΓ1234ΔΕΖ-Ω99\nΠοσό 9.876,00\nDigitally signed by Ministry of Digital Governance 2025-12-31"
    assert PROFILE.normalized_template(first) == PROFILE.normalized_template(second)


def test_diavgeia_overlay_spans_are_exact_and_reversible() -> None:
    text = (
        "Ουσιαστικό κείμενο\n"
        "ΑΔΑ: ΨΓΠ9469ΗΡ8-ΧΚ5\n"
        "Ministry of\n"
        "Digital\n"
        "Governance\n"
        "Digitally signed by Ministry\n"
        "of Digital Governance\n"
        "Date: 2025.12.24\n"
        "11:38:56 EET\n"
        "Reason:\n"
        "Location: Athens\n"
    )
    spans = PROFILE.diavgeia_overlay_spans(text, "doc-1")
    assert [span["kind"] for span in spans] == [
        "diavgeia_ada_stamp_span",
        "diavgeia_signing_block_span",
    ]
    for span in spans:
        removed = text[span["char_start"] : span["char_end"]]
        assert removed.strip()
    assert text[spans[0]["char_start"] : spans[0]["char_end"]].strip().startswith("ΑΔΑ:")
    assert "Location: Athens" in text[spans[1]["char_start"] : spans[1]["char_end"]]


def test_quality_profile_cli_diavgeia_fixture(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    text = (
        "ΑΠΟΦΑΣΗ\nΑ.Φ.Μ.: 123456789\nemail@example.gr\n"
        "Digitally signed by Ministry of Digital Governance\n"
        "Ministry of Digital Governance\n"
    )
    input_path = tmp_path / "diavgeia.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["ADA-1", "ADA-2"],
                "markdown_text": [text, ""],
                "metadata_json": [
                    json.dumps(
                        {
                            "privateData": True,
                            "decisionTypeId": "B.1",
                            "organizationId": "ORG",
                            "correctedVersionId": "old",
                        }
                    ),
                    "{}",
                ],
            }
        ),
        input_path,
        row_group_size=1,
    )
    output_dir = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "profile_source_quality.py"),
            "--input",
            str(input_path),
            "--source",
            "diavgeia",
            "--text-column",
            "markdown_text",
            "--id-column",
            "id",
            "--metadata-column",
            "metadata_json",
            "--workers",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    report = json.loads((output_dir / "source_quality_summary.json").read_text(encoding="utf-8"))
    assert report["metrics"]["rows"] == 2
    assert report["metrics"]["empty_documents"] == 1
    assert report["metrics"]["digital_governance_footer_documents"] == 1
    assert report["metrics"]["private_data_true_documents"] == 1
    assert report["metrics"]["corrected_version_documents"] == 1
    assert report["pii_document_counts"]["email"] == 1
    assert report["pii_document_counts"]["afm_labelled"] == 1
    actions = [
        json.loads(line)
        for line in (output_dir / "document_action_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(actions) == 1
    assert actions[0]["private_data_true"] is True
    assert set(actions[0]["structured_pii"]) == {"afm_labelled", "email"}


def test_staged_schema_verifier_checks_text_and_candidate_id_columns(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    revision = "a" * 40
    local_dir = tmp_path / "fixture" / revision
    local_dir.mkdir(parents=True)
    path = local_dir / "data.parquet"
    pq.write_table(pa.table({"doc_id": ["1"], "text": ["κείμενο"]}), path)
    source = {"text_columns": ["text"], "id_columns": ["source_doc_id", "doc_id"]}
    locked = {
        "source_id": "fixture",
        "revision": revision,
        "selected_files": [{"path": "data.parquet"}],
    }
    report, errors = VERIFY.verify_source(source, locked, tmp_path)
    assert errors == []
    assert report["status"] == "ok"
    assert report["rows"] == 1

    bad_report, bad_errors = VERIFY.verify_source(
        {"text_columns": ["missing"], "id_columns": ["also_missing"]}, locked, tmp_path
    )
    assert bad_report["status"] == "error"
    assert len(bad_errors) == 2

    second = local_dir / "new-edition.parquet"
    pq.write_table(
        pa.table({"identifier": ["2"], "text_content": ["νεότερο κείμενο"]}), second
    )
    heterogeneous_lock = {
        "source_id": "fixture",
        "revision": revision,
        "selected_files": [{"path": "data.parquet"}, {"path": "new-edition.parquet"}],
    }
    heterogeneous, heterogeneous_errors = VERIFY.verify_source(
        {
            "text_columns": ["text", "plain_text", "text_content"],
            "alternate_text_columns": ["markdown", "markdown_content"],
            "id_columns": ["doc_id", "identifier"],
        },
        heterogeneous_lock,
        tmp_path,
    )
    assert heterogeneous_errors == []
    assert heterogeneous["status"] == "ok"
    assert [row["text_columns"] for row in heterogeneous["files"]] == [
        ["text"],
        ["text_content"],
    ]

    _, required_errors = VERIFY.verify_source(
        {
            "text_columns": ["text", "documents"],
            "required_text_columns": ["text", "documents"],
            "id_columns": ["doc_id", "identifier"],
        },
        heterogeneous_lock,
        tmp_path,
    )
    assert any("missing required text columns" in error for error in required_errors)


def test_structural_route_gate_blocks_diavgeia_and_requires_exact_embedded_filter() -> None:
    script = HERE / "scripts" / "check_source_route.py"
    blocked = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            "diavgeia",
            "--input-scope",
            "single_source",
            "--text-column",
            "markdown_text",
            "--id-column",
            "id",
        ],
        text=True,
        capture_output=True,
    )
    assert blocked.returncode != 0
    assert "disabled" in blocked.stderr

    allowed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source",
            "openarchives",
            "--input-scope",
            "canonical_mixed",
            "--source-regex",
            r"^openarchives\.gr$",
            "--text-column",
            "text",
            "--id-column",
            "source_doc_id",
            "--source-column",
            "source_dataset",
        ],
        text=True,
        capture_output=True,
    )
    assert allowed.returncode == 0, allowed.stderr


def test_detector_run_validator_checks_coverage_and_binary_receipt(tmp_path: Path) -> None:
    source = "fixture"
    doc_id = "doc-1"
    uid = hashlib.sha256(f"{source}\0{doc_id}".encode()).hexdigest()
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"binary")
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    parity = tmp_path / "parity.json"
    parity.write_text(
        json.dumps(
            {
                "schema_version": "struct_rust_parity_receipt_v1",
                "status": "passed",
                "binary_sha256": binary_sha,
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256()
    encoded = doc_id.encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    stream = tmp_path / "stream.json"
    stream.write_text(
        json.dumps(
            {
                "source": source,
                "rows_emitted": 1,
                "id_sequence_sha256": digest.hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    original_sha = hashlib.sha256("text".encode()).hexdigest()
    counters = tmp_path / "counters.jsonl"
    counters.write_text(
        json.dumps(
            {
                "source": source,
                "doc_id": doc_id,
                "row_uid": uid,
                "original_sha256": original_sha,
                "original_chars": 4,
                "overlap_pairs": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spans = tmp_path / "spans.jsonl"
    spans.write_text(
        json.dumps(
            {
                "source": source,
                "doc_id": doc_id,
                "row_uid": uid,
                "original_sha256": original_sha,
                "original_chars": 4,
                "model_id": "fixture",
                "kind": "bib_span",
                "char_start": 0,
                "char_end": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sources = tmp_path / "sources.json"
    policy = tmp_path / "policy.json"
    sources.write_text("{}", encoding="utf-8")
    policy.write_text("{}", encoding="utf-8")
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"fixture parquet")
    input_receipt = tmp_path / "input_receipt.json"
    input_receipt.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_acquisition_receipt_v1",
                "status": "passed",
                "code_commit": "0" * 40,
                "sources_config_sha256": hashlib.sha256(sources.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    input_check = tmp_path / "input_check.json"
    input_check.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_input_receipt_check_v1",
                "ok": True,
                "source": source,
                "input_receipt_sha256": hashlib.sha256(input_receipt.read_bytes()).hexdigest(),
                "paths": [str(input_path.resolve())],
                "inputs": [{"path": str(input_path.resolve()), "bytes": input_path.stat().st_size}],
                "bytes": input_path.stat().st_size,
                "text_column": "text",
                "id_column": "source_doc_id",
                "source_column": "source_dataset",
            }
        ),
        encoding="utf-8",
    )
    detector_build = tmp_path / "detector_build.json"
    detector_build.write_text(
        json.dumps(
            {
                "schema_version": "full_cpt_detector_build_receipt_v1",
                "status": "passed",
                "code_commit": "a" * 40,
                "binary": {"path": str(binary.resolve()), "sha256": binary_sha},
            }
        ),
        encoding="utf-8",
    )
    stream.write_text(
        json.dumps(
            {
                "schema_version": "detector_input_stream_v1",
                "source": source,
                "inputs": [{"path": str(input_path.resolve()), "bytes": input_path.stat().st_size}],
                "text_column": "text",
                "id_column": "source_doc_id",
                "source_column": "source_dataset",
                "rows_emitted": 1,
                "id_sequence_sha256": digest.hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run.json"
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "validate_detector_run.py"),
            "--source",
            source,
            "--stream-manifest",
            str(stream),
            "--counters",
            str(counters),
            "--spans",
            str(spans),
            "--binary",
            str(binary),
            "--parity-receipt",
            str(parity),
            "--input-receipt",
            str(input_receipt),
            "--input-receipt-check",
            str(input_check),
            "--detector-build-receipt",
            str(detector_build),
            "--sources-config",
            str(sources),
            "--cleaning-policy",
            str(policy),
            "--code-commit",
            "a" * 40,
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
