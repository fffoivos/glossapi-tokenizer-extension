from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "agent1_v3_contract.py"


def _write(path: Path, payload: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _freeze_args(tmp_path: Path) -> list[str]:
    files = {name: _write(tmp_path / "inputs" / f"{name}.json") for name in (
        "source_registry",
        "source_aliases",
        "candidate_roster",
        "post_cutoff_inventory",
        "nanochat_initial_roster",
        "acquisition_receipt",
        "tokenizer",
        "review_policy",
        "review_prompt",
        "review_response_schema",
        "glossapi_build_receipt",
        "license_adjudication",
        "training_eligibility_policy",
        "policy",
    )}
    return [
        sys.executable,
        str(SCRIPT),
        "freeze-run",
        "--run-root",
        str(tmp_path / "run"),
        "--data-root",
        str(tmp_path / "data" / _run_id()),
        "--run-id",
        "agent1-full-corpus-v3-20260712T123456Z-abcdef0",
        "--code-commit",
        "abcdef0123456789",
        "--source-registry",
        str(files["source_registry"]),
        "--source-aliases",
        str(files["source_aliases"]),
        "--candidate-roster",
        str(files["candidate_roster"]),
        "--post-cutoff-inventory",
        str(files["post_cutoff_inventory"]),
        "--nanochat-initial-roster",
        str(files["nanochat_initial_roster"]),
        "--acquisition-receipt",
        str(files["acquisition_receipt"]),
        "--tokenizer",
        str(files["tokenizer"]),
        "--review-policy",
        str(files["review_policy"]),
        "--review-prompt",
        str(files["review_prompt"]),
        "--review-response-schema",
        str(files["review_response_schema"]),
        "--glossapi-build-receipt",
        str(files["glossapi_build_receipt"]),
        "--license-adjudication",
        str(files["license_adjudication"]),
        "--training-eligibility-policy",
        str(files["training_eligibility_policy"]),
        "--dedup-policy",
        str(files["policy"]),
        "--greekmmlu-policy",
        str(files["policy"]),
        "--anonymization-policy",
        str(files["policy"]),
        "--structural-policy",
        str(files["policy"]),
        "--prestructural-only",
    ]


def _run_id() -> str:
    return "agent1-full-corpus-v3-20260712T123456Z-abcdef0"


def test_contract_allows_phase0_evidence_then_freezes_once(tmp_path: Path) -> None:
    phase0 = tmp_path / "run" / "phase0"
    _write(phase0 / "merged_acquisition_receipt.json", "merged\n")
    result = subprocess.run(_freeze_args(tmp_path), check=True, capture_output=True, text=True)
    contract = json.loads((tmp_path / "run" / "run_contract.json").read_text())

    assert json.loads(result.stdout)["ok"] is True
    assert contract["prestructural_only"] is True
    assert contract["stage_graph"][0] == "10-normalize"
    assert contract["contract_sha256"]

    second = subprocess.run(_freeze_args(tmp_path), capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_contract_rejects_out_of_order_stage_and_completes_attempt(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    common = ["--run-root", str(root), "--run-id", _run_id()]

    early = subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "20-lineage", "--attempt-id", "1"],
        capture_output=True,
        text=True,
    )
    assert early.returncode != 0
    assert "stage receipt" in early.stderr

    subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "10-normalize", "--attempt-id", "1"],
        check=True,
    )
    output = _write(root / "stages" / "10-normalize" / "attempts" / "1" / "normalization_manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert (root / "stages" / "10-normalize" / "COMPLETED").is_file()

    subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "20-lineage", "--attempt-id", "2"],
        check=True,
    )


def test_contract_detects_bound_input_drift(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    source = tmp_path / "inputs" / "source_registry.json"
    source.write_text("drift\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "validate-run", "--run-root", str(tmp_path / "run"), "--run-id", _run_id()],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "bound input drift" in result.stderr


def test_outputs_must_be_attempt_local(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    common = ["--run-root", str(root), "--run-id", _run_id()]
    subprocess.run(
        [sys.executable, str(SCRIPT), "begin-stage", *common, "--stage", "10-normalize", "--attempt-id", "1"],
        check=True,
    )
    output = _write(tmp_path / "elsewhere.json")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "job-unique metadata or bulk-data attempt" in result.stderr


def test_contract_binds_paired_attempt_roots_and_receipt_closure(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    data_root = tmp_path / "data" / _run_id()
    common = ["--run-root", str(root), "--run-id", _run_id()]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "begin-stage",
            *common,
            "--data-root",
            str(data_root),
            "--stage",
            "10-normalize",
            "--attempt-id",
            "attempt-1",
        ],
        check=True,
    )
    metadata_attempt = root / "stages" / "10-normalize" / "attempts" / "attempt-1"
    data_attempt = data_root / "stages" / "10-normalize" / "attempts" / "attempt-1"
    metadata_output = _write(metadata_attempt / "normalization_manifest.json")
    bulk_output = _write(data_attempt / "canonical" / "source-a" / "part.parquet", "parquet\n")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--data-root",
            str(data_root),
            "--stage",
            "10-normalize",
            "--attempt-id",
            "attempt-1",
            "--output",
            str(metadata_output),
            "--output",
            str(bulk_output),
        ],
        check=True,
    )

    frozen = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    stage = json.loads((root / "stages" / "10-normalize" / "stage_contract.json").read_text(encoding="utf-8"))
    assert frozen["data_root"] == str(data_root.resolve())
    assert stage["storage"] == {
        "metadata_attempt_dir": str(metadata_attempt.resolve()),
        "data_attempt_dir": str(data_attempt.resolve()),
    }
    for storage, expected in (("metadata", metadata_attempt), ("data", data_attempt)):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "get-stage-attempt-dir",
                *common,
                "--stage",
                "10-normalize",
                "--storage",
                storage,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == str(expected.resolve())

    subprocess.run([sys.executable, str(SCRIPT), "validate-run", *common], check=True)
    marker = root / "stages" / "10-normalize" / "COMPLETED"
    marker.unlink()
    missing_marker = subprocess.run(
        [sys.executable, str(SCRIPT), "validate-run", *common], capture_output=True, text=True
    )
    assert missing_marker.returncode != 0
    assert "COMPLETED marker" in missing_marker.stderr

    receipt = root / "stages" / "10-normalize" / "stage_receipt.json"
    marker.write_text(f"{hashlib.sha256(receipt.read_bytes()).hexdigest()}  stage_receipt.json\n", encoding="utf-8")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["completed_at"] = "tampered"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tampered = subprocess.run(
        [sys.executable, str(SCRIPT), "validate-run", *common], capture_output=True, text=True
    )
    assert tampered.returncode != 0
    assert "receipt hash mismatch" in tampered.stderr


def test_identical_incomplete_stage_resumes_in_new_attempt_and_receipt_pins_it(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    data_root = tmp_path / "data" / _run_id()
    common = ["--run-root", str(root), "--run-id", _run_id(), "--data-root", str(data_root)]
    input_path = _write(tmp_path / "inputs" / "resume-input.json", "fixed input\n")
    parameters = '{"batch_size":128,"recipe":"fixed"}'

    first = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "begin-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "failed-job",
            "--parameters-json",
            parameters,
            "--input",
            "source",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(first.stdout)["resumed"] is False
    failed_attempt = root / "stages" / "10-normalize" / "attempts" / "failed-job"
    failed_attempt_contract = failed_attempt / "attempt_contract.json"
    failed_output = _write(failed_attempt / "partial-output.json", "incomplete\n")
    root_contract = root / "stages" / "10-normalize" / "stage_contract.json"
    root_contract_bytes = root_contract.read_bytes()
    failed_attempt_contract_bytes = failed_attempt_contract.read_bytes()

    second = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "begin-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "retry-job",
            "--parameters-json",
            parameters,
            "--input",
            "source",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(second.stdout)["resumed"] is True
    retry_attempt = root / "stages" / "10-normalize" / "attempts" / "retry-job"
    retry_data_attempt = data_root / "stages" / "10-normalize" / "attempts" / "retry-job"
    assert retry_attempt.is_dir()
    assert retry_data_attempt.is_dir()
    assert root_contract.read_bytes() == root_contract_bytes
    assert failed_attempt_contract.read_bytes() == failed_attempt_contract_bytes

    success_output = _write(retry_attempt / "normalization_manifest.json", "passed\n")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "retry-job",
            "--output",
            str(success_output),
        ],
        check=True,
    )
    receipt = json.loads((root / "stages" / "10-normalize" / "stage_receipt.json").read_text())
    retry_attempt_contract = json.loads((retry_attempt / "attempt_contract.json").read_text())
    assert receipt["attempt_id"] == "retry-job"
    assert receipt["attempt_contract_sha256"] == retry_attempt_contract["contract_sha256"]
    assert receipt["successful_attempt"] == {
        "attempt_id": "retry-job",
        "attempt_contract_sha256": retry_attempt_contract["contract_sha256"],
        "metadata_attempt_dir": str(retry_attempt.resolve()),
        "data_attempt_dir": str(retry_data_attempt.resolve()),
    }
    assert [item["path"] for item in receipt["outputs"]] == [str(success_output.resolve())]
    assert str(failed_output.resolve()) not in {item["path"] for item in receipt["outputs"]}

    for storage, expected in (("metadata", retry_attempt), ("data", retry_data_attempt)):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "get-stage-attempt-dir",
                "--run-root",
                str(root),
                "--run-id",
                _run_id(),
                "--stage",
                "10-normalize",
                "--storage",
                storage,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == str(expected.resolve())


def test_resume_revalidates_and_rebinds_completed_upstream_receipt(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    data_root = tmp_path / "data" / _run_id()
    common = ["--run-root", str(root), "--run-id", _run_id(), "--data-root", str(data_root)]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "begin-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "normalize-job",
        ],
        check=True,
    )
    normalized = _write(root / "stages" / "10-normalize" / "attempts" / "normalize-job" / "manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "normalize-job",
            "--output",
            str(normalized),
        ],
        check=True,
    )
    lineage_input = _write(tmp_path / "inputs" / "lineage-input.json", "fixed input\n")
    lineage_begin = [
        sys.executable,
        str(SCRIPT),
        "begin-stage",
        *common,
        "--stage",
        "20-lineage",
        "--parameters-json",
        '{"lineage":"fixed"}',
        "--input",
        "normalized",
        str(lineage_input),
    ]
    subprocess.run([*lineage_begin, "--attempt-id", "failed-lineage"], check=True)
    stage_contract = json.loads((root / "stages" / "20-lineage" / "stage_contract.json").read_text())
    upstream = stage_contract["upstream_receipts"]["10-normalize"]
    upstream_receipt = root / "stages" / "10-normalize" / "stage_receipt.json"
    assert upstream == {
        "path": str(upstream_receipt.resolve()),
        "bytes": upstream_receipt.stat().st_size,
        "sha256": hashlib.sha256(upstream_receipt.read_bytes()).hexdigest(),
    }
    resumed = subprocess.run(
        [*lineage_begin, "--attempt-id", "retry-lineage"], check=True, capture_output=True, text=True
    )
    assert json.loads(resumed.stdout)["resumed"] is True


@pytest.mark.parametrize(
    ("retry_parameters", "retry_input_name"),
    [
        ('{ "batch_size" : 128 }', "source"),  # Same JSON object, different frozen bytes.
        ('{"batch_size":129}', "source"),
        ('{"batch_size":128}', "replacement"),
    ],
)
def test_resume_rejects_changed_parameter_or_input_contract(
    tmp_path: Path, retry_parameters: str, retry_input_name: str
) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    data_root = tmp_path / "data" / _run_id()
    common = ["--run-root", str(root), "--run-id", _run_id(), "--data-root", str(data_root)]
    source = _write(tmp_path / "inputs" / "source.json", "fixed input\n")
    replacement = _write(tmp_path / "inputs" / "replacement.json", "fixed input\n")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "begin-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "first-job",
            "--parameters-json",
            '{"batch_size":128}',
            "--input",
            "source",
            str(source),
        ],
        check=True,
    )
    retry_input = replacement if retry_input_name == "replacement" else source
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "begin-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "retry-job",
            "--parameters-json",
            retry_parameters,
            "--input",
            retry_input_name,
            str(retry_input),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not byte-identical" in result.stderr
    assert not (root / "stages" / "10-normalize" / "attempts" / "retry-job").exists()


def test_completed_stage_never_resumes_even_with_identical_contract(tmp_path: Path) -> None:
    subprocess.run(_freeze_args(tmp_path), check=True)
    root = tmp_path / "run"
    data_root = tmp_path / "data" / _run_id()
    common = ["--run-root", str(root), "--run-id", _run_id(), "--data-root", str(data_root)]
    source = _write(tmp_path / "inputs" / "source.json", "fixed input\n")
    begin = [
        sys.executable,
        str(SCRIPT),
        "begin-stage",
        *common,
        "--stage",
        "10-normalize",
        "--attempt-id",
        "first-job",
        "--parameters-json",
        '{"batch_size":128}',
        "--input",
        "source",
        str(source),
    ]
    subprocess.run(begin, check=True)
    output = _write(root / "stages" / "10-normalize" / "attempts" / "first-job" / "manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finish-stage",
            *common,
            "--stage",
            "10-normalize",
            "--attempt-id",
            "first-job",
            "--output",
            str(output),
        ],
        check=True,
    )
    before = (root / "stages" / "10-normalize" / "stage_contract.json").read_bytes()
    retry = begin.copy()
    retry[retry.index("first-job")] = "retry-job"
    result = subprocess.run(retry, capture_output=True, text=True)
    assert result.returncode != 0
    assert "may never resume" in result.stderr
    assert (root / "stages" / "10-normalize" / "stage_contract.json").read_bytes() == before
    assert not (root / "stages" / "10-normalize" / "attempts" / "retry-job").exists()
