from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


EVAL = (
    Path(__file__).resolve().parents[2]
    / "02_corpus_preparation"
    / "15_clean_academic"
    / "eval"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


PARITY = load_module("phase04_rust_parity_struct", EVAL / "rust_parity_struct.py")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _modern_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    corpus = root / "struct2k.LLM_silver.jsonl"
    rows = []
    for index, label in enumerate(("BIB", "TOC")):
        rows.append(
            {
                "schema_version": "academic-structure-gold-v1",
                "document_id": f"doc-{index}",
                "source": "greek_phd" if index == 0 else "kallipos",
                "split": "train" if index == 0 else "validation",
                "historical_split": "train",
                "n_physical_lines": 3,
                "lines": [
                    {"abs_idx": 0, "text": "κείμενο", "label": "O"},
                    {"abs_idx": 2, "text": f"δομή {index}", "label": label},
                ],
            }
        )
    write_jsonl(corpus, rows)
    split = root / "struct2k.LLM_silver.split.json"
    inventory = "1" * 64
    write_json(
        split,
        {
            "schema_version": "academic-structure-split-v1",
            "inventory_sha256": inventory,
            "assignments": {row["document_id"]: row["split"] for row in rows},
        },
    )
    receipt = root / "struct2k.LLM_silver.receipt.json"
    write_json(
        receipt,
        {
            "schema_version": "academic-structure-silver-contract-receipt-v1",
            "status": "pass",
            "evidence_tier": "LLM_silver",
            "production_eligible": False,
            "silver_sha256": sha(corpus),
            "split_manifest_sha256": sha(split),
            "inventory_sha256": inventory,
            "document_count": 2,
            "split_counts": {"train": 1, "validation": 1},
            "task_scope_counts": {"bibliography_toc_windows": 2},
            "materialized_artifacts": {
                "silver_filename": corpus.name,
                "split_manifest_filename": split.name,
            },
            "historical_partition_exclusion": {
                "eligible_historical_train_documents": 2,
                "historical_test_documents_excluded": 608,
                "historical_test_rows_emitted": 0,
                "historical_test_predictions_permitted": False,
            },
        },
    )
    return corpus, receipt, split


def test_modern_importer_coordinates_use_zero_based_exclusive_n_physical_boundary(
    tmp_path: Path,
) -> None:
    corpus, receipt, split = _modern_source(tmp_path)
    data = PARITY.load_materialized(corpus)
    assert data.docs["doc-0"]["lines"] == [(0, "κείμενο"), (2, "δομή 0")]
    assert data.docs["doc-0"]["N"] == 3
    assert PARITY.validate_source_receipt(receipt, corpus, split)["document_count"] == 2

    rows = [
        json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["lines"][-1]["abs_idx"] = 3
    write_jsonl(corpus, rows)
    with pytest.raises(ValueError, match="coordinate boundary"):
        PARITY.load_materialized(corpus)


def test_modern_all_partition_derives_document_count_and_receipt_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, source_receipt, source_split = _modern_source(tmp_path)
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"fixture binary")
    output = tmp_path / "nested" / "parity.json"
    monkeypatch.setattr(
        PARITY,
        "compare_head",
        lambda _binary, docs, _data, **_kwargs: {
            "documents": len(docs),
            "max_probability_difference": 0.0,
            "span_mismatches": 0,
        },
    )
    argv = [
        "rust_parity_struct.py",
        "--binary",
        str(binary),
        "--corpus",
        str(corpus),
        "--source-receipt",
        str(source_receipt),
        "--source-split-manifest",
        str(source_split),
        "--partition",
        "all",
        "--expected-corpus-sha256",
        sha(corpus),
        "--receipt",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert PARITY.main() == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["heldout_documents"] == 2
    assert value["evaluation_partition"] == "all"
    assert value["historical_test_documents_loaded"] == 0
    assert value["source_split_manifest_sha256"] == sha(source_split)
    assert value["input_snapshot_method"] == PARITY.SNAPSHOT_METHOD
    assert value["inputs_rehashed_before_publication"] is True
    assert output.parent.is_dir()
    with pytest.raises(FileExistsError, match="overwrite"):
        PARITY.main()


def _modern_argv(
    *,
    corpus: Path,
    source_receipt: Path,
    source_split: Path,
    binary: Path,
    output: Path,
    tolerance: str = "0.001",
) -> list[str]:
    return [
        "rust_parity_struct.py",
        "--binary",
        str(binary),
        "--corpus",
        str(corpus),
        "--source-receipt",
        str(source_receipt),
        "--source-split-manifest",
        str(source_split),
        "--partition",
        "all",
        "--expected-corpus-sha256",
        sha(corpus),
        "--tolerance",
        tolerance,
        "--receipt",
        str(output),
    ]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1])
def test_runner_rejects_bool_nonfinite_or_negative_numbers(value: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        PARITY._finite_nonnegative(value, label="fixture")


@pytest.mark.parametrize("tolerance", ["nan", "inf", "-0.1"])
def test_runner_rejects_invalid_cli_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tolerance: str
) -> None:
    corpus, source_receipt, source_split = _modern_source(tmp_path)
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"fixture binary")
    output = tmp_path / "parity.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _modern_argv(
            corpus=corpus,
            source_receipt=source_receipt,
            source_split=source_split,
            binary=binary,
            output=output,
            tolerance=tolerance,
        ),
    )
    with pytest.raises(ValueError, match="parity tolerance"):
        PARITY.main()
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_probability_difference", True, "finite non-negative"),
        ("max_probability_difference", float("nan"), "finite non-negative"),
        ("max_probability_difference", -0.1, "finite non-negative"),
        ("max_probability_difference", 0.002, "exceeds parity tolerance"),
        ("documents", 1, "document coverage drift"),
        ("documents", None, "positive integer"),
    ],
)
def test_runner_rejects_invalid_head_delta_or_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    corpus, source_receipt, source_split = _modern_source(tmp_path)
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"fixture binary")
    output = tmp_path / "parity.json"

    def result(_binary, docs, _data, **_kwargs):
        row = {
            "documents": len(docs),
            "max_probability_difference": 0.0,
            "span_mismatches": 0,
        }
        if value is None:
            row.pop(field)
        else:
            row[field] = value
        return row

    monkeypatch.setattr(PARITY, "compare_head", result)
    monkeypatch.setattr(
        sys,
        "argv",
        _modern_argv(
            corpus=corpus,
            source_receipt=source_receipt,
            source_split=source_split,
            binary=binary,
            output=output,
        ),
    )
    with pytest.raises(ValueError, match=message):
        PARITY.main()
    assert not output.exists()


def test_runner_rejects_symlink_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, source_receipt, source_split = _modern_source(tmp_path)
    symlink = tmp_path / corpus.name
    symlink.symlink_to(corpus)
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"fixture binary")
    output = tmp_path / "parity.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _modern_argv(
            corpus=symlink,
            source_receipt=source_receipt,
            source_split=source_split,
            binary=binary,
            output=output,
        ),
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        PARITY.main()
    assert not output.exists()


def test_runner_rehashes_original_inputs_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, source_receipt, source_split = _modern_source(tmp_path)
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"fixture binary")
    output = tmp_path / "parity.json"
    mutated = False

    def mutate_after_snapshot(_binary, docs, _data, **_kwargs):
        nonlocal mutated
        if not mutated:
            corpus.write_text(
                corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            mutated = True
        return {
            "documents": len(docs),
            "max_probability_difference": 0.0,
            "span_mismatches": 0,
        }

    monkeypatch.setattr(PARITY, "compare_head", mutate_after_snapshot)
    monkeypatch.setattr(
        sys,
        "argv",
        _modern_argv(
            corpus=corpus,
            source_receipt=source_receipt,
            source_split=source_split,
            binary=binary,
            output=output,
        ),
    )
    with pytest.raises(ValueError, match="changed after the parity input snapshot"):
        PARITY.main()
    assert not output.exists()
