from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models import deterministic_ablation_runner as runner  # noqa: E402
from sequence_models.contract import canonical_json_sha256, sha256_file  # noqa: E402


def _line(document_id: str, index: int, text: str, label: str) -> dict[str, object]:
    return {
        "line_id": hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest(),
        "abs_idx": index,
        "text": text,
        "label": label,
        "token_count": max(1, len(text.split())),
        "is_running_prose": None,
    }


def _document(
    document_id: str,
    *,
    split: str,
    texts: list[str],
    labels: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "academic-structure-gold-v1",
        "document_id": document_id,
        "work_id": f"work-{document_id}",
        "representation_id": f"representation-{document_id}",
        "source": "synthetic",
        "split": split,
        "coverage": "full_document",
        "n_physical_lines": len(texts),
        "n_present_lines": len(texts),
        "annotation": {
            "status": "LLM_silver",
            "annotator_ids": ["synthetic-llm"],
            "adjudicator_id": None,
            "engine": "synthetic-engine",
            "task_scope": "bibliography_toc_windows",
        },
        "tokenizer": {"id": "synthetic-tokenizer", "revision": "v1"},
        "lines": [
            _line(document_id, index, text, label)
            for index, (text, label) in enumerate(zip(texts, labels))
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _base_row(document: dict[str, object], labels: list[str]) -> dict[str, object]:
    lines = document["lines"]
    assert isinstance(lines, list)
    return {
        "schema_version": "academic-structure-predictions-v1",
        "model_id": "c0-synthetic",
        "document_id": document["document_id"],
        "work_id": document["work_id"],
        "source": document["source"],
        "split": document["split"],
        "lines": [
            {
                "line_id": line["line_id"],
                "abs_idx": line["abs_idx"],
                "prediction": label,
            }
            for line, label in zip(lines, labels)
        ],
    }


def _bundle(tmp_path: Path) -> dict[str, Path]:
    train = _document(
        "train-doc",
        split="train",
        texts=["Ordinary training line"],
        labels=["O"],
    )
    toc = _document(
        "toc-doc",
        split="validation",
        texts=[
            "ΠΕΡΙΕΧΟΜΕΝΑ",
            "1. Εισαγωγή ........ 7",
            "2. Μέθοδος ........ 15",
        ],
        labels=["TOC", "TOC", "TOC"],
    )
    bib = _document(
        "bib-doc",
        split="validation",
        texts=[
            "ΒΙΒΛΙΟΓΡΑΦΙΑ",
            "Smith, J. (2019). First title. London: Press.",
            "Brown, K. (2020). Second title. London: Press.",
        ],
        labels=["BIB", "BIB", "BIB"],
    )
    silver = tmp_path / "silver.jsonl"
    _write_jsonl(silver, [train, toc, bib])

    assignments = {
        str(row["document_id"]): str(row["split"]) for row in (train, toc, bib)
    }
    inventory = sorted(
        (str(row["document_id"]), str(row["work_id"]), str(row["source"]))
        for row in (train, toc, bib)
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "academic-structure-split-v1",
                "assignments": assignments,
                "inventory_sha256": canonical_json_sha256(inventory),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    base = tmp_path / "c0.jsonl"
    _write_jsonl(
        base,
        [
            _base_row(toc, ["O", "TOC", "O"]),
            _base_row(bib, ["O", "BIB", "O"]),
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "silver_contract": {
                    "annotation_status": "LLM_silver",
                    "allowed_task_scopes": ["bibliography_toc_windows"],
                    "comparison_only": True,
                    "production_eligible": False,
                },
                "deployment_gates": {
                    "maximum_false_deletion_fraction_per_document": 0.01,
                    "maximum_contiguous_false_deletion_tokens": 256,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "silver": silver,
        "split_manifest": split_manifest,
        "base_c0": base,
        "config": config,
    }


def test_runner_writes_four_immutable_predictions_and_one_receipt(
    tmp_path: Path,
) -> None:
    paths = _bundle(tmp_path)
    output = tmp_path / "run"
    report = runner.run_ablation_evaluation(
        silver_path=paths["silver"],
        split_manifest_path=paths["split_manifest"],
        base_c0_path=paths["base_c0"],
        config_path=paths["config"],
        output_dir=output,
    )

    assert report["schema_version"] == runner.REPORT_SCHEMA
    assert report["evidence_tier"] == "LLM_silver"
    assert report["allowed_split"] == "validation"
    assert report["document_count"] == 2
    assert set(report["modes"]) == {mode.value for mode in runner.RUN_MODES}
    assert report["execution_claims"] == {
        "model_fitting_performed": False,
        "data_discovery_performed": False,
        "corpus_mutation_performed": False,
        "sealed_or_test_data_accessed": False,
        "human_gold_used": False,
        "production_eligible": False,
        "production_action_authorized": False,
    }
    assert (
        report["base_c0"]["metrics"]["metric_availability"][
            "independent_running_prose_safety_metrics"
        ]
        is False
    )

    for mode in runner.RUN_MODES:
        receipt = report["outputs"][mode.value]
        prediction_path = output / receipt["path"]
        assert prediction_path.is_file()
        assert receipt["sha256"] == sha256_file(prediction_path)
        assert receipt["document_count"] == 2
        assert report["modes"][mode.value]["evidence_tier"] == "LLM_silver"
        assert (
            report["modes"][mode.value]["metrics"]["coverage"][
                "represented_present_line_fraction"
            ]
            == 1.0
        )

    persisted = json.loads((output / "ablation.report.json").read_text())
    assert persisted == report
    assert "commit" in report["code_revision"]
    assert all("sha256" in row for row in report["inputs"].values())

    with pytest.raises(FileExistsError, match="immutable output directory"):
        runner.run_ablation_evaluation(
            silver_path=paths["silver"],
            split_manifest_path=paths["split_manifest"],
            base_c0_path=paths["base_c0"],
            config_path=paths["config"],
            output_dir=output,
        )


@pytest.mark.parametrize(
    "split", ["test", "historical-test", "SEALED_test", "held-out-test"]
)
def test_forbidden_allowed_split_fails_before_any_input_read(
    tmp_path: Path, split: str
) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(runner.SealedDataError, match="forbidden"):
        runner.run_ablation_evaluation(
            silver_path=missing,
            split_manifest_path=missing,
            base_c0_path=missing,
            config_path=missing,
            output_dir=tmp_path / "run",
            allowed_split=split,
        )


def test_silver_test_alias_fails_before_gold_line_materialisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver.jsonl"
    _write_jsonl(
        silver,
        [
            {
                "schema_version": "academic-structure-gold-v1",
                "document_id": "sealed",
                "split": "historical_test",
                "lines": [{"this": "must never become a GoldLine"}],
            }
        ],
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read_gold must not be called")

    monkeypatch.setattr(runner, "read_gold", fail_if_called)
    missing = tmp_path / "missing"
    with pytest.raises(runner.SealedDataError, match="historical_test"):
        runner.run_ablation_evaluation(
            silver_path=silver,
            split_manifest_path=missing,
            base_c0_path=missing,
            config_path=missing,
            output_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


def test_manifest_test_alias_fails_before_gold_line_materialisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver.jsonl"
    _write_jsonl(silver, [{"split": "validation"}])
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps({"assignments": {"doc": "sealed-test"}}), encoding="utf-8"
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read_gold must not be called")

    monkeypatch.setattr(runner, "read_gold", fail_if_called)
    missing = tmp_path / "missing"
    with pytest.raises(runner.SealedDataError, match="sealed-test"):
        runner.run_ablation_evaluation(
            silver_path=silver,
            split_manifest_path=manifest,
            base_c0_path=missing,
            config_path=missing,
            output_dir=tmp_path / "run",
        )


def test_config_must_pin_comparison_only_nonproduction(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    config = json.loads(paths["config"].read_text())
    config["silver_contract"]["production_eligible"] = True
    paths["config"].write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(runner.AblationRunError, match="comparison-only"):
        runner.run_ablation_evaluation(
            silver_path=paths["silver"],
            split_manifest_path=paths["split_manifest"],
            base_c0_path=paths["base_c0"],
            config_path=paths["config"],
            output_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()
