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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _modern_source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    corpus = root / "struct2k.LLM_silver.jsonl"
    rows = []
    for index, label in enumerate(("BIB", "TOC")):
        rows.append(
            {
                "schema_version": "academic-structure-gold-v1",
                "document_id": f"doc-{index}",
                "source": "greek_phd" if index == 0 else "kallipos",
                "split": "validation",
                "historical_split": "train",
                "n_physical_lines": 3,
                "lines": [
                    {"abs_idx": 0, "text": "κείμενο", "label": "O"},
                    {"abs_idx": 2, "text": f"δομή {index}", "label": label},
                ],
            }
        )
    write_jsonl(corpus, rows)
    receipt = root / "struct2k.LLM_silver.receipt.json"
    write_json(
        receipt,
        {
            "schema_version": "academic-structure-silver-contract-receipt-v1",
            "status": "pass",
            "evidence_tier": "LLM_silver",
            "production_eligible": False,
            "silver_sha256": sha(corpus),
            "document_count": 2,
            "split_counts": {"validation": 2},
            "task_scope_counts": {"bibliography_toc_windows": 2},
            "materialized_artifacts": {"silver_filename": corpus.name},
            "historical_partition_exclusion": {
                "eligible_historical_train_documents": 2,
                "historical_test_documents_excluded": 608,
                "historical_test_rows_emitted": 0,
                "historical_test_predictions_permitted": False,
            },
        },
    )
    return corpus, receipt


def test_modern_importer_coordinates_use_zero_based_exclusive_n_physical_boundary(
    tmp_path: Path,
) -> None:
    corpus, receipt = _modern_source(tmp_path)
    data = PARITY.load_materialized(corpus)
    assert data.docs["doc-0"]["lines"] == [(0, "κείμενο"), (2, "δομή 0")]
    assert data.docs["doc-0"]["N"] == 3
    assert PARITY.validate_source_receipt(receipt, corpus)["document_count"] == 2

    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    rows[0]["lines"][-1]["abs_idx"] = 3
    write_jsonl(corpus, rows)
    with pytest.raises(ValueError, match="coordinate boundary"):
        PARITY.load_materialized(corpus)


def test_modern_all_partition_derives_document_count_and_receipt_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, source_receipt = _modern_source(tmp_path)
    binary = tmp_path / "reference_detect"
    binary.write_bytes(b"fixture binary")
    output = tmp_path / "nested" / "parity.json"
    monkeypatch.setattr(
        PARITY,
        "compare_head",
        lambda _binary, docs, _data, *, head, work, tolerance: {
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
    assert output.parent.is_dir()
    with pytest.raises(FileExistsError, match="overwrite"):
        PARITY.main()
