from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models import bib_ladder, struct2k_import  # noqa: E402
from sequence_models.bib_ladder import (  # noqa: E402
    prepare_selection,
    select_shared_calibration,
    verify_selection_bundle,
)
from sequence_models.contract import read_gold, sha256_file  # noqa: E402
from sequence_models.contract import parse_gold_rows  # noqa: E402
from sequence_models.feature_crf import LinearChainCRF  # noqa: E402
from sequence_models.features import TAGS  # noqa: E402


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _historical_split(value: str) -> str:
    digest = hashlib.md5(value.encode(), usedforsecurity=False).hexdigest()
    return "test" if int(digest, 16) % 10 < 3 else "train"


def _ids(split: str, count: int) -> list[str]:
    result: list[str] = []
    index = 0
    while len(result) < count:
        candidate = f"fixture-{split}-{index}"
        if _historical_split(candidate) == split:
            result.append(candidate)
        index += 1
    return result


def _build_handoff(root: Path) -> tuple[Path, dict[str, object]]:
    struct = root / "STRUCT_2K"
    struct.mkdir(parents=True)
    ids = _ids("train", 4) + _ids("test", 2)
    sources = ["greek_phd", "openarchives"] * 3
    gold: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    corrections: list[dict[str, object]] = []
    label_counts = {0: 0, 1: 0, 2: 0}
    for index, (document_id, source) in enumerate(zip(ids, sources)):
        split = _historical_split(document_id)
        texts = [
            f"Κύριο κείμενο {index}",
            f"Βιβλιογραφία {index}",
            f"Τελικό κείμενο {index}",
        ]
        kind = "table_of_contents" if index == 1 else "bibliography"
        start, end = (0, 0) if kind == "table_of_contents" else (1, 1)
        if index == 0:
            end = 99
            corrections.append(
                {
                    "annotation_file": "STRUCT_2K/ann_00000.json",
                    "document_id": document_id,
                    "kind": "bibliography",
                    "original": [1, 99],
                    "corrected": [1, 1],
                    "evidence": "fixture duplicated coordinate",
                }
            )
        labels = []
        for coordinate in range(3):
            labels.append(
                2
                if kind == "table_of_contents" and start <= coordinate <= end
                else 1
                if kind == "bibliography" and start <= coordinate <= end
                else 0
            )
        for label in labels:
            label_counts[label] += 1
        lines = [
            [coordinate, text, label]
            for coordinate, (text, label) in enumerate(zip(texts, labels))
        ]
        mode = "whole"
        gold.append(
            {
                "doc_id": document_id,
                "source": source,
                "split": split,
                "n_lines": 3,
                "mode": mode,
                "lines": lines,
            }
        )
        manifests.append(
            {
                "i": index,
                "doc_id": document_id,
                "source": source,
                "split": split,
                "n_lines": 3,
                "mode": mode,
                "chars": 30,
                "badness": 0,
            }
        )
        numbered = "\n".join(
            f"L{coordinate:05d}: {text}" for coordinate, text in enumerate(texts)
        )
        _json(
            struct / f"batch_{index:05d}.json",
            [
                {
                    "doc_id": document_id,
                    "source": source,
                    "split": split,
                    "n_lines": 3,
                    "mode": mode,
                    "badness": 0,
                    "text_numbered": numbered,
                }
            ],
        )
        _json(
            struct / f"ann_{index:05d}.json",
            {
                "doc_id": document_id,
                "source": source,
                "split": split,
                "n_lines": 3,
                "mode": mode,
                "doc_type": "book",
                "sections": [
                    {
                        "kind": kind,
                        "start_line": start,
                        "end_line": end,
                        "confidence": "high",
                    }
                ],
                "_engine": {"model": "gpt-5.5", "effort": "medium"},
            },
        )
    _jsonl(root / "STRUCT_2K_gold.jsonl", gold)
    _jsonl(struct / "manifest.jsonl", manifests)
    (root / "SOURCE_STATE.txt").write_text("fixture\n", encoding="utf-8")
    (root / "STRUCT_linemat.npz").write_bytes(b"fixture-cache-not-trusted")
    listed = sorted(path for path in root.rglob("*") if path.is_file())
    entries = {str(path.relative_to(root)): sha256_file(path) for path in listed}
    inventory = root / "INVENTORY.sha256"
    inventory.write_text(
        "".join(f"{digest}  ./{relative}\n" for relative, digest in entries.items()),
        encoding="utf-8",
    )
    source_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    for document_id in ids:
        split = _historical_split(document_id)
        split_counts[split] = split_counts.get(split, 0) + 1
    lock = {
        "schema_version": "struct2k-handoff-lock-v1",
        "source": {
            "host": "fixture",
            "repo": "/fixture",
            "commit": "a" * 40,
        },
        "inventory": {
            "filename": "INVENTORY.sha256",
            "sha256": sha256_file(inventory),
            "entry_count": len(entries),
            "required_files": {
                name: entries[name]
                for name in (
                    "SOURCE_STATE.txt",
                    "STRUCT_2K_gold.jsonl",
                    "STRUCT_2K/manifest.jsonl",
                    "STRUCT_linemat.npz",
                )
            },
        },
        "legacy_contract": {
            "document_count": len(ids),
            "present_line_count": 3 * len(ids),
            "label_counts": {str(key): value for key, value in label_counts.items()},
            "source_counts": dict(sorted(source_counts.items())),
            "historical_split_counts": dict(sorted(split_counts.items())),
            "historical_split_algorithm": "md5(document_id)%10<3_is_test",
            "annotation_engine": "gpt-5.5",
            "annotation_effort": "medium",
            "annotation_status": "LLM_silver",
        },
        "coordinate_corrections": corrections,
    }
    lock_path = root.parent / "lock.json"
    _json(lock_path, lock)
    return lock_path, lock


def _joint_config(path: Path) -> None:
    _json(
        path,
        {
            "schema_version": "academic-structure-sequence-eval-v1",
            "active_classes": ["BIB", "TOC"],
            "historical_partition_usage": {
                "historical_test_document_count": 2,
            },
            "silver_contract": {
                "annotation_status": "LLM_silver",
                "allowed_task_scopes": ["bibliography_toc_windows"],
            },
            "split": {
                "seed": "fixture-joint",
                "train_fraction": 0.5,
                "validation_fraction": 0.5,
                "test_fraction": 0.0,
            },
        },
    )


class _Tokenizer:
    def __init__(self, _path: Path) -> None:
        pass

    def counts(self, texts: list[str]) -> list[int]:
        return [max(1, len(text.split())) for text in texts]


def test_handoff_audit_replays_lineage_and_applies_only_locked_correction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "handoff"
    lock_path, _lock = _build_handoff(root)
    receipt, documents, inventory = struct2k_import.audit_handoff(root, lock_path)
    assert receipt["annotation_status"] == "LLM_silver"
    assert receipt["human_gold"] is False
    assert receipt["document_count"] == 6
    assert receipt["coordinate_corrections_applied"] == 1
    assert receipt["corrected_documents"] == 1
    assert receipt["corrected_present_lines"] == 1
    assert len(documents) == 6
    assert len(inventory) == 16


def test_handoff_audit_fails_closed_on_inventory_or_unlocked_coordinate_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "handoff"
    lock_path, lock = _build_handoff(root)
    (root / "STRUCT_2K" / "ann_00001.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(struct2k_import.Struct2KImportError, match="hash mismatch"):
        struct2k_import.audit_handoff(root, lock_path)
    root2 = tmp_path / "handoff2"
    lock_path2, lock2 = _build_handoff(root2)
    lock2["coordinate_corrections"] = []
    _json(lock_path2, lock2)
    with pytest.raises(
        struct2k_import.Struct2KImportError, match="invalid uncorrected section"
    ):
        struct2k_import.audit_handoff(root2, lock_path2)


def test_joint_materialization_and_selection_physically_exclude_historical_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "handoff"
    lock_path, _lock = _build_handoff(root)
    config = tmp_path / "joint.json"
    _joint_config(config)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(struct2k_import, "TOKENIZER_SHA256", sha256_file(tokenizer))
    output = tmp_path / "materialized"
    output.mkdir()
    paths = {
        "silver": output / "struct2k.LLM_silver.jsonl",
        "split": output / "struct2k.LLM_silver.split.json",
        "snapshot": output / "struct2k.handoff.audit.receipt.json",
        "receipt": output / "struct2k.LLM_silver.receipt.json",
    }
    monkeypatch.setattr(struct2k_import, "ExactTokenizer", _Tokenizer)
    source_receipt = struct2k_import.materialize(
        handoff_root=root,
        lock_path=lock_path,
        tokenizer_path=tokenizer,
        config_path=config,
        silver_path=paths["silver"],
        split_manifest_path=paths["split"],
        snapshot_receipt_path=paths["snapshot"],
        receipt_path=paths["receipt"],
    )
    assert (
        struct2k_import.verify_materialized_source(
            root=output, lock_path=lock_path, config_path=config
        )
        == source_receipt
    )
    documents = read_gold(paths["silver"])
    assert len(documents) == 4
    assert {document.split for document in documents} == {"train", "validation"}
    assert all(
        document.task_scope == "bibliography_toc_windows" for document in documents
    )
    materialized_rows = [
        json.loads(line)
        for line in paths["silver"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {row["historical_split"] for row in materialized_rows} == {"train"}
    assert {row["upstream_document_id"] for row in materialized_rows}.isdisjoint(
        _ids("test", 2)
    )
    assert source_receipt["historical_partition_exclusion"] == {
        **source_receipt["historical_partition_exclusion"],
        "historical_test_documents_excluded": 2,
        "historical_test_rows_emitted": 0,
        "historical_test_predictions_permitted": False,
    }
    corrected = next(
        document
        for document in documents
        if document.annotation_engine == "gpt-5.5 medium STRUCT-2K annotation workflow"
        and document.lines[1].label == "BIB"
        and document.lines[2].label == "O"
    )
    assert corrected.lines[2].label == "O"

    selection = tmp_path / "selection"
    selection.mkdir()
    monkeypatch.setattr(bib_ladder, "STRUCT2K_LOCK_PATH", lock_path)
    prepare_selection(
        silver_path=paths["silver"],
        split_manifest_path=paths["split"],
        rehydration_receipt_path=paths["receipt"],
        config_path=config,
        selection_silver_path=selection / "selection.train-validation.jsonl",
        selection_manifest_path=selection / "selection.train-validation.split.json",
        validation_silver_path=selection / "selection.validation.jsonl",
        receipt_path=selection / "selection.receipt.json",
    )
    selected, validation, receipt = verify_selection_bundle(
        selection_silver_path=selection / "selection.train-validation.jsonl",
        selection_manifest_path=selection / "selection.train-validation.split.json",
        validation_silver_path=selection / "selection.validation.jsonl",
        selection_receipt_path=selection / "selection.receipt.json",
        config_path=config,
    )
    assert len(selected) == 4
    assert validation
    assert receipt["active_classes"] == ["BIB", "TOC"]
    assert receipt["counts"]["historically_named_test_documents_excluded"] == 2
    assert all(document.split != "test" for document in selected)

    tampered = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    tampered["historical_partition_exclusion"]["historical_test_documents_excluded"] = 1
    _json(paths["receipt"], tampered)
    rejected = tmp_path / "rejected-selection"
    rejected.mkdir()
    with pytest.raises(
        bib_ladder.LadderError,
        match="historical-test exclusion",
    ):
        prepare_selection(
            silver_path=paths["silver"],
            split_manifest_path=paths["split"],
            rehydration_receipt_path=paths["receipt"],
            config_path=config,
            selection_silver_path=rejected / "selection.train-validation.jsonl",
            selection_manifest_path=rejected / "selection.train-validation.split.json",
            validation_silver_path=rejected / "selection.validation.jsonl",
            receipt_path=rejected / "selection.receipt.json",
        )


def test_joint_calibration_uses_action_recall_and_keeps_both_head_metrics() -> None:
    receipt = select_shared_calibration(
        [
            {
                "deletion_bias": 0.0,
                "action_precision": 0.9,
                "action_recall": 0.8,
                "bib_recall": 0.9,
                "toc_recall": 0.4,
                "predicted_action_tokens": 20,
            },
            {
                "deletion_bias": 1.0,
                "action_precision": 0.95,
                "action_recall": 0.7,
                "bib_recall": 0.6,
                "toc_recall": 0.9,
                "predicted_action_tokens": 15,
            },
        ],
        reference_action_precision=0.9,
        active_classes=("BIB", "TOC"),
    )
    assert receipt["active_classes"] == ["BIB", "TOC"]
    assert receipt["selected"]["action_recall"] == 0.8
    assert receipt["selected"]["toc_recall"] == 0.4


def test_c0_and_feature_crf_retain_joint_toc_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sequence_models import baseline

    raw = {
        "schema_version": "academic-structure-gold-v1",
        "document_id": "joint-document",
        "work_id": "joint-work",
        "representation_id": "joint-representation",
        "source": "greek_phd",
        "split": "validation",
        "coverage": "full_document",
        "n_physical_lines": 10,
        "n_present_lines": 3,
        "annotation": {
            "status": "LLM_silver",
            "engine": "fixture",
            "task_scope": "bibliography_toc_windows",
            "annotator_ids": ["LLM:fixture"],
            "adjudicator_id": None,
        },
        "tokenizer": {"id": "fixture", "revision": "fixture"},
        "lines": [
            {
                "line_id": f"line-{index}",
                "abs_idx": index,
                "text": text,
                "label": label,
                "token_count": 1,
                "is_running_prose": None,
            }
            for index, (text, label) in enumerate(
                (("ΠΕΡΙΕΧΟΜΕΝΑ", "TOC"), ("κείμενο", "O"), ("Author (2020)", "BIB"))
            )
        ],
    }
    document = parse_gold_rows([raw])[0]
    monkeypatch.setattr(
        baseline,
        "_scores",
        lambda _rows, model: list(model["scores"]),
    )
    decoder = {
        "bib": {"theta_hi": 0.5, "theta_lo": 0.2, "gap": 0, "lmin": 1},
        "toc": {"theta_hi": 0.5, "theta_lo": 0.2, "gap": 0, "lmin": 1},
    }
    predictions, conflicts = baseline.predict_document(
        document,
        {"scores": [0.0, 0.0, 0.9]},
        {"scores": [0.9, 0.0, 0.0]},
        decoder,
        active_classes=("BIB", "TOC"),
    )
    assert predictions == ["TOC", "O", "BIB"]
    assert conflicts == [False, False, False]

    model = LinearChainCRF(1, seed=3, active_classes=("BIB", "TOC"))
    model.emission_bias[:] = -100.0
    model.emission_bias[TAGS.index("S-TOC")] = 100.0
    decoded = model.viterbi([{0: 0.0}])
    assert TAGS[int(decoded[0])] == "S-TOC"


def test_joint_n1_checkpoint_preserves_active_mask(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sequence_models.char_tcn_crf import _make_model, load_n1_checkpoint
    from sequence_models.features import FeatureEncoder

    config_path = EVAL_DIR / "sequence_models" / "joint_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = next(
        row
        for row in config["architecture_ladder"]
        if row["id"] == "n1-bytecnn-tcn-masked-crf"
    )
    encoder = FeatureEncoder(char_hash_dim=0)
    model = _make_model(architecture, encoder.n_features, ("BIB", "TOC"))
    checkpoint = {
        "schema_version": "n1-checkpoint-v2",
        "architecture_id": "n1-bytecnn-tcn-masked-crf",
        "architecture": architecture,
        "engineered_dim": encoder.n_features,
        "feature_encoder": encoder.metadata(),
        "active_classes": ["BIB", "TOC"],
        "deletion_bias": config["calibration"]["deletion_bias_grid"][0],
        "state_dict": model.state_dict(),
        "inputs": {"config_sha256": sha256_file(config_path)},
        "seed": config["execution"]["seed"],
        "production_eligible": False,
    }
    path = tmp_path / "joint-n1.pt"
    torch.save(checkpoint, path)
    loaded, metadata = load_n1_checkpoint(path, config)
    assert metadata["active_classes"] == ["BIB", "TOC"]
    assert bool(loaded.crf.active_tag_mask[TAGS.index("S-TOC")])
