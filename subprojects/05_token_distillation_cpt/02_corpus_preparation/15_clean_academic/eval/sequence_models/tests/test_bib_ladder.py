from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bib_ladder import (  # noqa: E402
    EXPECTED_ARTIFACT_ROLES,
    LadderError,
    evaluate_validation,
    finalize_run,
    prepare_selection,
    select_shared_calibration,
    verify_selection_bundle,
    verify_published_run,
)
from sequence_models.baseline import (  # noqa: E402
    EVAL_DIR as BASELINE_EVAL_DIR,
    _load_json as load_baseline_json,
    main as baseline_main,
    predict_document,
)
from sequence_models.char_tcn_crf import (  # noqa: E402
    CharTCNCRF,
    _cache_emissions,
    _collate,
    _make_model,
    _predictions_from_emissions,
    _scrub_silver_safety as scrub_n1_safety,
    calibrate_n1,
    count_neural_sequences,
    load_n1_checkpoint,
    make_neural_examples,
    write_n1_predictions,
)
from sequence_models.contract import (  # noqa: E402
    build_split_manifest,
    read_gold,
    sha256_file,
)
from sequence_models.features import TAGS, FeatureEncoder  # noqa: E402
from sequence_models.feature_crf import (  # noqa: E402
    LinearChainCRF,
    calibrate_deletion_bias,
    make_examples,
    write_predictions as write_feature_predictions,
)
from sequence_models.evaluate import evaluate, read_predictions  # noqa: E402

CONFIG = EVAL_DIR / "sequence_models" / "config.json"


def _row(index: int, split: str) -> dict[str, object]:
    document_id = f"document-{index}"
    lines = []
    for line_index, (text, label) in enumerate(
        (
            (f"Κανονικό κείμενο {index}", "O"),
            (f"Author {index} (2020). Title.", "BIB"),
            (f"Τέλος {index}", "O"),
        )
    ):
        lines.append(
            {
                "line_id": f"{document_id}:{line_index}",
                "abs_idx": line_index,
                "text": text,
                "label": label,
                "token_count": 4,
                "is_running_prose": None,
            }
        )
    return {
        "schema_version": "academic-structure-gold-v1",
        "document_id": document_id,
        "work_id": f"work-{index}",
        "representation_id": f"representation-{index}",
        "source": "greek_phd",
        "split": split,
        "coverage": "annotated_windows",
        "n_physical_lines": 3,
        "n_present_lines": 3,
        "annotation": {
            "status": "LLM_silver",
            "engine": "fixture LLM workflow",
            "task_scope": "bibliography_binary_windows",
            "annotator_ids": ["LLM:fixture"],
            "adjudicator_id": None,
        },
        "tokenizer": {"id": "fixture-tokenizer", "revision": "fixture-revision"},
        "lines": lines,
    }


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _source_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    rows = [_row(index, "train") for index in range(12)]
    silver = root / "span.LLM_silver.jsonl"
    split = root / "span.LLM_silver.split.json"
    receipt = root / "span.LLM_silver.receipt.json"
    _jsonl(silver, rows)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = build_split_manifest(read_gold(silver), config["split"])
    for row in rows:
        row["split"] = manifest["assignments"][str(row["document_id"])]
    _jsonl(silver, rows)
    assert build_split_manifest(read_gold(silver), config["split"]) == manifest
    _json(split, manifest)
    unit_receipt = root / "SPAN-rehydrated.receipt.json"
    _json(
        unit_receipt,
        {
            "schema_version": "span-unit-rehydration-receipt-v1",
            "operation": "text_payload_rehydration_only",
            "labels_read_created_or_inferred": False,
            "research_fit_eligible": True,
            "research_evidence_scope": "LLM_silver_comparison_only",
            "promotion_eligible": False,
        },
    )
    source_receipt = {
        "schema_version": "academic-structure-silver-contract-receipt-v1",
        "inventory_sha256": manifest["inventory_sha256"],
        "silver_sha256": sha256_file(silver),
        "split_manifest_sha256": sha256_file(split),
        "config_sha256": sha256_file(CONFIG),
        "sequence_fit_eligible": True,
        "sequence_evidence_scope": "LLM_silver_comparison_only",
        "production_eligible": False,
        "source_unit_snapshot": {
            "receipt_path": str(unit_receipt.resolve()),
            "receipt_sha256": sha256_file(unit_receipt),
            "snapshot_equivalence_status": "rehydrated_unverified_snapshot",
            "research_fit_eligible": True,
            "research_evidence_scope": "LLM_silver_comparison_only",
            "production_eligible": False,
        },
    }
    _json(receipt, source_receipt)
    return {
        "silver": silver,
        "split": split,
        "receipt": receipt,
        "unit_receipt": unit_receipt,
    }


def _prepare(root: Path) -> dict[str, Path]:
    source = _source_fixture(root / "source")
    run = root / "run"
    run.mkdir()
    paths = {
        **source,
        "run": run,
        "selection": run / "selection.train-validation.jsonl",
        "selection_manifest": run / "selection.train-validation.split.json",
        "validation": run / "selection.validation.jsonl",
        "selection_receipt": run / "selection.receipt.json",
    }
    prepare_selection(
        silver_path=source["silver"],
        split_manifest_path=source["split"],
        rehydration_receipt_path=source["receipt"],
        config_path=CONFIG,
        selection_silver_path=paths["selection"],
        selection_manifest_path=paths["selection_manifest"],
        validation_silver_path=paths["validation"],
        receipt_path=paths["selection_receipt"],
    )
    return paths


def _predictions(path: Path, model_id: str, validation: Path) -> None:
    rows = []
    for document in read_gold(validation):
        rows.append(
            {
                "schema_version": "academic-structure-predictions-v1",
                "model_id": model_id,
                "document_id": document.document_id,
                "work_id": document.work_id,
                "source": document.source,
                "split": document.split,
                "lines": [
                    {
                        "line_id": line.line_id,
                        "abs_idx": line.abs_idx,
                        "prediction": line.label,
                    }
                    for line in document.lines
                ],
            }
        )
    _jsonl(path, rows)


def _arm_receipts(
    run: Path,
    baseline: Path,
    candidates: dict[str, Path],
) -> dict[str, Path]:
    result = {"c0-rust-lr-hysteresis": run / "c0.receipt.json"}
    _json(
        result["c0-rust-lr-hysteresis"],
        {
            "schema_version": "academic-structure-c0-reference-v2",
            "status": "passed_descriptive_reference_prediction",
            "comparison_role": "historical_reference_only",
            "target": "BIB",
            "active_classes": ["BIB"],
            "overlap_caveat": "fixture overlap cannot be excluded",
            "production_eligible": False,
            "outputs": {"validation_predictions_sha256": sha256_file(baseline)},
        },
    )
    calibration = select_shared_calibration(
        [
            {
                "deletion_bias": 0.0,
                "action_precision": 1.0,
                "action_recall": 1.0,
                "bib_recall": 1.0,
                "predicted_action_tokens": 8,
            }
        ],
        reference_action_precision=1.0,
    )
    for architecture_id, predictions in candidates.items():
        path = run / f"{architecture_id}.receipt.json"
        result[architecture_id] = path
        _json(
            path,
            {
                "schema_version": (
                    "academic-structure-n1-training-v2"
                    if architecture_id == "n1-bytecnn-tcn-masked-crf"
                    else "academic-structure-feature-crf-training-v2"
                ),
                "architecture_id": architecture_id,
                "target": "BIB",
                "active_classes": ["BIB"],
                "production_eligible": False,
                "calibration": calibration,
                "outputs": {"validation_predictions_sha256": sha256_file(predictions)},
            },
        )
    return result


def test_prepare_physically_excludes_locked_test_and_binds_outputs(
    tmp_path: Path,
) -> None:
    paths = _prepare(tmp_path)
    documents, validation, receipt = verify_selection_bundle(
        selection_silver_path=paths["selection"],
        selection_manifest_path=paths["selection_manifest"],
        validation_silver_path=paths["validation"],
        selection_receipt_path=paths["selection_receipt"],
        config_path=CONFIG,
    )
    assert {document.split for document in documents} == {"train", "validation"}
    assert {document.split for document in validation} == {"validation"}
    assert receipt["counts"]["historically_named_test_documents_excluded"] > 0
    assert (
        receipt["architecture_access_contract"]["historically_named_test_rows_emitted"]
        == 0
    )
    assert (
        "not an unbiased"
        in receipt["architecture_access_contract"]["partition_semantics"]
    )
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_selection(
            silver_path=paths["silver"],
            split_manifest_path=paths["split"],
            rehydration_receipt_path=paths["receipt"],
            config_path=CONFIG,
            selection_silver_path=paths["selection"],
            selection_manifest_path=paths["selection_manifest"],
            validation_silver_path=paths["validation"],
            receipt_path=paths["selection_receipt"],
        )


def test_selection_verifier_rejects_drift(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    paths["validation"].write_text(
        paths["validation"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(LadderError, match="bytes differ"):
        verify_selection_bundle(
            selection_silver_path=paths["selection"],
            selection_manifest_path=paths["selection_manifest"],
            validation_silver_path=paths["validation"],
            selection_receipt_path=paths["selection_receipt"],
            config_path=CONFIG,
        )


def test_selection_verifier_rejects_contract_receipt_tampering(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    receipt = json.loads(paths["selection_receipt"].read_text(encoding="utf-8"))
    receipt["selection_contract"]["inventory_sha256"] = "0" * 64
    _json(paths["selection_receipt"], receipt)
    with pytest.raises(LadderError, match="receipt contract"):
        verify_selection_bundle(
            selection_silver_path=paths["selection"],
            selection_manifest_path=paths["selection_manifest"],
            validation_silver_path=paths["validation"],
            selection_receipt_path=paths["selection_receipt"],
            config_path=CONFIG,
        )


def test_prepare_rejects_config_and_split_manifest_drift(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path / "source")
    changed_config = tmp_path / "changed-config.json"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["split"]["seed"] = "different-seed"
    _json(changed_config, config)
    outputs = [
        tmp_path / name for name in ("selection", "manifest", "validation", "receipt")
    ]
    with pytest.raises(LadderError, match="different sequence config"):
        prepare_selection(
            silver_path=source["silver"],
            split_manifest_path=source["split"],
            rehydration_receipt_path=source["receipt"],
            config_path=changed_config,
            selection_silver_path=outputs[0],
            selection_manifest_path=outputs[1],
            validation_silver_path=outputs[2],
            receipt_path=outputs[3],
        )

    manifest = json.loads(source["split"].read_text(encoding="utf-8"))
    manifest["algorithm"] = "operator-authored-substitute"
    _json(source["split"], manifest)
    receipt = json.loads(source["receipt"].read_text(encoding="utf-8"))
    receipt["split_manifest_sha256"] = sha256_file(source["split"])
    _json(source["receipt"], receipt)
    with pytest.raises(LadderError, match="exact config-derived"):
        prepare_selection(
            silver_path=source["silver"],
            split_manifest_path=source["split"],
            rehydration_receipt_path=source["receipt"],
            config_path=CONFIG,
            selection_silver_path=outputs[0],
            selection_manifest_path=outputs[1],
            validation_silver_path=outputs[2],
            receipt_path=outputs[3],
        )


def test_prepare_rejects_source_unit_receipt_outside_hydration_root(
    tmp_path: Path,
) -> None:
    source = _source_fixture(tmp_path / "source")
    outside = tmp_path / "outside-unit-receipt.json"
    outside.write_bytes(source["unit_receipt"].read_bytes())
    receipt = json.loads(source["receipt"].read_text(encoding="utf-8"))
    receipt["source_unit_snapshot"]["receipt_path"] = str(outside.resolve())
    receipt["source_unit_snapshot"]["receipt_sha256"] = sha256_file(outside)
    _json(source["receipt"], receipt)
    with pytest.raises(LadderError, match="outside the immutable hydration root"):
        prepare_selection(
            silver_path=source["silver"],
            split_manifest_path=source["split"],
            rehydration_receipt_path=source["receipt"],
            config_path=CONFIG,
            selection_silver_path=tmp_path / "selection",
            selection_manifest_path=tmp_path / "manifest",
            validation_silver_path=tmp_path / "validation",
            receipt_path=tmp_path / "selection-receipt",
        )


def test_prepare_rejects_noncanonical_split_bytes_even_when_receipted(
    tmp_path: Path,
) -> None:
    source = _source_fixture(tmp_path / "source")
    source["split"].write_text(
        source["split"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    receipt = json.loads(source["receipt"].read_text(encoding="utf-8"))
    receipt["split_manifest_sha256"] = sha256_file(source["split"])
    _json(source["receipt"], receipt)
    with pytest.raises(LadderError, match="bytes are not canonical"):
        prepare_selection(
            silver_path=source["silver"],
            split_manifest_path=source["split"],
            rehydration_receipt_path=source["receipt"],
            config_path=CONFIG,
            selection_silver_path=tmp_path / "selection",
            selection_manifest_path=tmp_path / "manifest",
            validation_silver_path=tmp_path / "validation",
            receipt_path=tmp_path / "selection-receipt",
        )


def test_validation_report_uses_only_validation_view(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    baseline = paths["run"] / "c0.validation.predictions.jsonl"
    _predictions(baseline, "c0-rust-lr-hysteresis-python-bib-head", paths["validation"])
    candidates = {}
    for architecture_id, short in (
        ("c1-feature-bioes-crf", "c1"),
        ("c2-char-ngram-feature-bioes-crf", "c2"),
        ("n1-bytecnn-tcn-masked-crf", "n1"),
    ):
        candidate = paths["run"] / f"{short}.validation.predictions.jsonl"
        _predictions(candidate, architecture_id, paths["validation"])
        candidates[architecture_id] = candidate
    arm_receipts = _arm_receipts(paths["run"], baseline, candidates)
    output = paths["run"] / "validation.comparison.json"
    report = evaluate_validation(
        selection_silver_path=paths["selection"],
        selection_manifest_path=paths["selection_manifest"],
        validation_silver_path=paths["validation"],
        selection_receipt_path=paths["selection_receipt"],
        config_path=CONFIG,
        baseline_path=baseline,
        candidate_paths=candidates,
        arm_receipt_paths=arm_receipts,
        output_path=output,
    )
    assert report["historically_named_test_partition"] == {
        "documents_loaded_by_model_or_validation_processes": 0,
        "predictions_written": 0,
        "semantics": "sealed_retrospective_comparison_not_unbiased_never_seen_test",
    }
    assert report["validation_document_count"] == len(read_gold(paths["validation"]))
    assert report["selection_decision"]["automated"] is False
    assert report["baseline"]["metrics"]["token"]["toc_recall"] is None


def test_c0_bib_projection_never_emits_unsupervised_toc(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    bib, toc, decoder = map(
        load_baseline_json,
        (
            BASELINE_EVAL_DIR / "span_line_lr_struct_model.json",
            BASELINE_EVAL_DIR / "toc_line_lr_model.json",
            BASELINE_EVAL_DIR / "struct_smooth_params.json",
        ),
    )
    for document in read_gold(paths["validation"]):
        predictions, _conflicts = predict_document(
            document, bib, toc, decoder, active_classes=("BIB",)
        )
        assert set(predictions) <= {"O", "BIB"}


def test_n1_example_builder_segments_unknown_without_torch(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    document = read_gold(paths["selection"])[0]
    examples = make_neural_examples(
        [document], FeatureEncoder(char_hash_dim=0), max_bytes=32
    )
    assert len(examples) == 1
    assert examples[0].line_indices == (0, 1, 2)
    assert len(examples[0].engineered[0]) == FeatureEncoder(char_hash_dim=0).n_features
    assert max(len(row) for row in examples[0].byte_ids) <= 32


def test_n1_forward_loss_and_bib_mask_when_torch_is_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    paths = _prepare(tmp_path)
    document = read_gold(paths["selection"])[0]
    encoder = FeatureEncoder(char_hash_dim=0)
    examples = make_neural_examples([document], encoder, max_bytes=32)
    byte_ids, engineered, line_mask, tags = _collate(examples)
    model = CharTCNCRF(
        engineered_dim=encoder.n_features,
        byte_embedding_dim=4,
        char_channels_per_kernel=3,
        char_kernels=(3,),
        hidden_dim=8,
        tcn_dilations=(1,),
        dropout=0.0,
        target_classes=("BIB",),
    )
    loss = model.loss(byte_ids, engineered, line_mask, tags)
    assert bool(torch.isfinite(loss))
    decoded = model.decode(byte_ids, engineered, line_mask, deletion_bias=0.25)
    assert len(decoded) == 1
    assert all(
        not bool(model.crf.active_tag_mask[index])
        for index, tag in enumerate(TAGS)
        if tag.endswith("-TOC")
    )
    assert not any(TAGS[tag_id].endswith("-TOC") for tag_id in decoded[0])


def test_strict_feature_checkpoint_rejects_metadata_only_npz(tmp_path: Path) -> None:
    malformed = tmp_path / "metadata-only.npz"
    np.savez_compressed(
        malformed, metadata=np.asarray(json.dumps({"active_classes": ["BIB"]}))
    )
    with pytest.raises(ValueError, match="array inventory"):
        LinearChainCRF.load(malformed)


def test_n1_safety_scrub_is_recursive_over_sources_and_document_fields() -> None:
    row = {
        "token": {
            "prose_contamination": 0.0,
            "true_main_text_retention": 1.0,
            "toc_recall": 0.0,
        },
        "line": {"toc_recall": 0.0},
        "span": {"toc": {"exact_precision": 1.0, "exact_recall": 0.0}},
        "document": {
            "catastrophic_prose_deletions": 0,
            "maximum_contiguous_false_deletion_tokens": 0,
        },
    }
    metrics = {**json.loads(json.dumps(row)), "by_source": {"source": row}}
    scrub_n1_safety(metrics)
    for current in (metrics, metrics["by_source"]["source"]):
        assert current["token"]["prose_contamination"] is None
        assert current["token"]["true_main_text_retention"] is None
        assert current["document"]["catastrophic_prose_deletions"] is None
        assert current["document"]["maximum_contiguous_false_deletion_tokens"] is None
        assert all(value is None for value in current["span"]["toc"].values())


def test_strict_n1_checkpoint_rejects_malformed_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    architecture = next(
        row
        for row in config["architecture_ladder"]
        if row["id"] == "n1-bytecnn-tcn-masked-crf"
    )
    encoder = FeatureEncoder(char_hash_dim=0)
    model = _make_model(architecture, encoder.n_features)
    state = model.state_dict()
    first = next(iter(state))
    state[first] = torch.zeros((1,), dtype=state[first].dtype)
    checkpoint = {
        "schema_version": "n1-checkpoint-v2",
        "architecture_id": "n1-bytecnn-tcn-masked-crf",
        "architecture": architecture,
        "engineered_dim": encoder.n_features,
        "feature_encoder": encoder.metadata(),
        "active_classes": ["BIB"],
        "deletion_bias": config["calibration"]["deletion_bias_grid"][0],
        "state_dict": state,
        "inputs": {"config_sha256": sha256_file(CONFIG)},
        "seed": config["execution"]["seed"],
        "production_eligible": False,
    }
    path = tmp_path / "malformed.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="tensor"):
        load_n1_checkpoint(path, config)


def test_strict_n1_checkpoint_rejects_tampered_derived_masks(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    architecture = next(
        row
        for row in config["architecture_ladder"]
        if row["id"] == "n1-bytecnn-tcn-masked-crf"
    )
    encoder = FeatureEncoder(char_hash_dim=0)
    model = _make_model(architecture, encoder.n_features)
    state = model.state_dict()
    state["crf.active_tag_mask"] = torch.ones_like(state["crf.active_tag_mask"])
    checkpoint = {
        "schema_version": "n1-checkpoint-v2",
        "architecture_id": "n1-bytecnn-tcn-masked-crf",
        "architecture": architecture,
        "engineered_dim": encoder.n_features,
        "feature_encoder": encoder.metadata(),
        "active_classes": ["BIB"],
        "deletion_bias": config["calibration"]["deletion_bias_grid"][0],
        "state_dict": state,
        "inputs": {"config_sha256": sha256_file(CONFIG)},
        "seed": config["execution"]["seed"],
        "production_eligible": False,
    }
    path = tmp_path / "tampered-mask.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="derived mask"):
        load_n1_checkpoint(path, config)


def test_publication_verifier_rejects_empty_artifact_receipt(tmp_path: Path) -> None:
    run = tmp_path / "published"
    run.mkdir()
    rows: list[dict[str, object]] = []
    _json(
        run / "run.receipt.json",
        {
            "schema_version": "academic-structure-bib-ladder-run-v1",
            "status": "passed_cpu_sealed_retrospective_comparison",
            "architecture_ids": [
                "c0-rust-lr-hysteresis",
                "c1-feature-bioes-crf",
                "c2-char-ngram-feature-bioes-crf",
                "n1-bytecnn-tcn-masked-crf",
            ],
            "target": "BIB",
            "active_classes": ["BIB"],
            "production_eligible": False,
            "artifacts": rows,
            "artifact_inventory_sha256": "0" * 64,
            "decision": "LLM_silver_replay_only_no_automatic_selection_no_production_change",
            "human_annotation": {
                "required": False,
                "campaign_planned": False,
                "future_independent_option": "newly sampled LLM-silver documents",
            },
        },
    )
    with pytest.raises(LadderError, match="artifact inventory"):
        verify_published_run(run)


def _build_real_receipted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], dict[str, Path]]:
    torch = pytest.importorskip("torch")
    for key, value in {
        "SEQUENCE_UENV": "fixture-uenv",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "SLURM_CPUS_PER_TASK": "16",
    }.items():
        monkeypatch.setenv(key, value)
    paths = _prepare(tmp_path)
    run = paths["run"]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed = int(config["execution"]["seed"])
    baseline = run / "c0.validation.predictions.jsonl"
    c0_receipt = run / "c0.receipt.json"
    baseline_main(
        [
            "--selection-silver",
            str(paths["selection"]),
            "--selection-manifest",
            str(paths["selection_manifest"]),
            "--validation-silver",
            str(paths["validation"]),
            "--selection-receipt",
            str(paths["selection_receipt"]),
            "--config",
            str(CONFIG),
            "--uenv",
            "fixture-uenv",
            "--output",
            str(baseline),
            "--receipt-out",
            str(c0_receipt),
        ]
    )
    runtime = json.loads(c0_receipt.read_text(encoding="utf-8"))["execution"]
    common = {
        "selection_silver_sha256": sha256_file(paths["selection"]),
        "selection_manifest_sha256": sha256_file(paths["selection_manifest"]),
        "validation_silver_sha256": sha256_file(paths["validation"]),
        "selection_receipt_sha256": sha256_file(paths["selection_receipt"]),
        "config_sha256": sha256_file(CONFIG),
    }
    selection_documents = read_gold(paths["selection"])
    validation = read_gold(paths["validation"])
    c0_values = read_predictions(baseline, validation)
    c0_metrics, _ = evaluate(validation, c0_values, split="validation")
    reference_precision = c0_metrics["token"]["action_precision"]
    models: dict[str, Path] = {}
    candidates: dict[str, Path] = {}
    arm_receipts: dict[str, Path] = {"c0-rust-lr-hysteresis": c0_receipt}
    for architecture_id, short in (
        ("c1-feature-bioes-crf", "c1"),
        ("c2-char-ngram-feature-bioes-crf", "c2"),
    ):
        architecture = next(
            row for row in config["architecture_ladder"] if row["id"] == architecture_id
        )
        encoder = FeatureEncoder(
            char_hash_dim=int(architecture.get("char_hash_dim", 0)),
            char_ngram_min=int(architecture.get("char_ngram_min", 2)),
            char_ngram_max=int(architecture.get("char_ngram_max", 5)),
        )
        model = LinearChainCRF(encoder.n_features, seed=seed, active_classes=("BIB",))
        calibration = calibrate_deletion_bias(
            make_examples(validation, encoder),
            model,
            config["calibration"]["deletion_bias_grid"],
            reference_action_precision=reference_precision,
        )
        deletion_bias = float(calibration["selected"]["deletion_bias"])
        model_path = run / f"{short}.model.npz"
        prediction_path = run / f"{short}.validation.predictions.jsonl"
        model.save(
            model_path,
            {
                "architecture_id": architecture_id,
                "config_sha256": common["config_sha256"],
                "silver_sha256": common["selection_silver_sha256"],
                "split_manifest_sha256": common["selection_manifest_sha256"],
                "validation_silver_sha256": common["validation_silver_sha256"],
                "selection_receipt_sha256": common["selection_receipt_sha256"],
                "reference_predictions_sha256": sha256_file(baseline),
                "active_classes": ["BIB"],
                "calibration": calibration,
                "deletion_bias": deletion_bias,
                "production_eligible": False,
                "test_used_for_training_or_calibration": False,
            },
        )
        write_feature_predictions(
            prediction_path,
            validation,
            encoder,
            model,
            model_id=architecture_id,
            deletion_bias=deletion_bias,
        )
        receipt_path = run / f"{short}.receipt.json"
        _json(
            receipt_path,
            {
                "schema_version": "academic-structure-feature-crf-training-v2",
                "status": "passed_cpu_fit_checkpoint_reload_and_validation_prediction",
                "architecture_id": architecture_id,
                "target": "BIB",
                "active_classes": ["BIB"],
                "production_eligible": False,
                "effective_seed": seed,
                "inputs": {
                    **common,
                    "reference_predictions_sha256": sha256_file(baseline),
                },
                "execution": {**runtime, "effective_seed": seed},
                "calibration": calibration,
                "outputs": {
                    "model_sha256": sha256_file(model_path),
                    "validation_predictions_sha256": sha256_file(prediction_path),
                },
            },
        )
        models[short] = model_path
        candidates[architecture_id] = prediction_path
        arm_receipts[architecture_id] = receipt_path

    source = json.loads(paths["selection_receipt"].read_text(encoding="utf-8"))[
        "source"
    ]
    profile = run / "n1.profile.receipt.json"
    _json(
        profile,
        {
            "schema_version": "academic-structure-n1-profile-v1",
            "status": "passed_one_epoch_profile_and_determinism_smoke",
            "architecture_id": "n1-bytecnn-tcn-masked-crf",
            "target": "BIB",
            "active_classes": ["BIB"],
            "production_eligible": False,
            "inputs": {
                **common,
                "source_rehydration_receipt_sha256": source[
                    "rehydration_receipt_sha256"
                ],
            },
            "execution": {
                **runtime,
                "effective_seed": seed,
                "code_commit": "a" * 40,
            },
            "effective_seed": seed,
            "counts": {
                "train_documents": sum(
                    document.split == "train" for document in selection_documents
                ),
                "validation_documents_contract_checked_not_scored_by_profile": len(
                    validation
                ),
                "train_sequences": count_neural_sequences(
                    [
                        document
                        for document in selection_documents
                        if document.split == "train"
                    ]
                ),
            },
            "determinism_smoke": {
                "status": "pass",
                "state_sha256": "b" * 64,
                "losses": [1.0, 0.5],
                "replicas": 2,
                "steps_per_replica": 2,
                "active_classes": ["BIB"],
            },
            "one_epoch_seconds": 1.0,
            "projected_full_fit_seconds_with_15pct_margin": 9.2,
            "maximum_full_fit_seconds": 32400.0,
            "within_full_fit_budget": True,
            "peak_rss_bytes": 1,
        },
    )
    n1_architecture = next(
        row
        for row in config["architecture_ladder"]
        if row["id"] == "n1-bytecnn-tcn-masked-crf"
    )
    n1_encoder = FeatureEncoder(char_hash_dim=0)
    torch.manual_seed(seed)
    n1 = _make_model(n1_architecture, n1_encoder.n_features)
    n1_model = run / "n1.model.pt"
    n1_inputs = {
        **common,
        "reference_predictions_sha256": sha256_file(baseline),
        "profile_receipt_sha256": sha256_file(profile),
    }
    n1_examples = make_neural_examples(
        validation,
        n1_encoder,
        max_bytes=int(n1_architecture["max_utf8_bytes_per_line"]),
    )
    n1_emissions = _cache_emissions(
        n1, n1_examples, batch_size=int(n1_architecture["batch_size"])
    )
    n1_calibration = calibrate_n1(
        validation,
        n1_examples,
        n1_emissions,
        n1,
        config["calibration"]["deletion_bias_grid"],
        reference_action_precision=reference_precision,
    )
    n1_deletion_bias = float(n1_calibration["selected"]["deletion_bias"])
    torch.save(
        {
            "schema_version": "n1-checkpoint-v2",
            "architecture_id": "n1-bytecnn-tcn-masked-crf",
            "architecture": n1_architecture,
            "engineered_dim": n1_encoder.n_features,
            "feature_encoder": n1_encoder.metadata(),
            "active_classes": ["BIB"],
            "deletion_bias": n1_deletion_bias,
            "state_dict": n1.state_dict(),
            "inputs": n1_inputs,
            "seed": seed,
            "production_eligible": False,
        },
        n1_model,
    )
    n1_values = _predictions_from_emissions(
        validation, n1_examples, n1_emissions, n1, deletion_bias=n1_deletion_bias
    )
    n1_predictions = run / "n1.validation.predictions.jsonl"
    write_n1_predictions(n1_predictions, validation, n1_values)
    n1_receipt = run / "n1.training.receipt.json"
    _json(
        n1_receipt,
        {
            "schema_version": "academic-structure-n1-training-v2",
            "status": "passed_cpu_fit_checkpoint_reload_and_validation_prediction",
            "architecture_id": "n1-bytecnn-tcn-masked-crf",
            "target": "BIB",
            "active_classes": ["BIB"],
            "production_eligible": False,
            "effective_seed": seed,
            "inputs": n1_inputs,
            "execution": {
                **runtime,
                "effective_seed": seed,
                "code_commit": "a" * 40,
            },
            "calibration": n1_calibration,
            "deletion_bias": n1_deletion_bias,
            "outputs": {
                "model_sha256": sha256_file(n1_model),
                "validation_predictions_sha256": sha256_file(n1_predictions),
            },
            "historically_named_test_partition": {"documents_loaded": 0},
        },
    )
    candidates["n1-bytecnn-tcn-masked-crf"] = n1_predictions
    arm_receipts["n1-bytecnn-tcn-masked-crf"] = n1_receipt
    report = run / "validation.comparison.json"
    evaluate_validation(
        selection_silver_path=paths["selection"],
        selection_manifest_path=paths["selection_manifest"],
        validation_silver_path=paths["validation"],
        selection_receipt_path=paths["selection_receipt"],
        config_path=CONFIG,
        baseline_path=baseline,
        candidate_paths=candidates,
        arm_receipt_paths=arm_receipts,
        output_path=report,
    )
    artifacts = {
        "selection_silver": paths["selection"],
        "selection_manifest": paths["selection_manifest"],
        "validation_silver": paths["validation"],
        "selection_receipt": paths["selection_receipt"],
        "c0_validation_predictions": baseline,
        "c0_receipt": c0_receipt,
        "c1_model": models["c1"],
        "c1_validation_predictions": candidates["c1-feature-bioes-crf"],
        "c1_receipt": arm_receipts["c1-feature-bioes-crf"],
        "c2_model": models["c2"],
        "c2_validation_predictions": candidates["c2-char-ngram-feature-bioes-crf"],
        "c2_receipt": arm_receipts["c2-char-ngram-feature-bioes-crf"],
        "n1_profile_receipt": profile,
        "n1_model": n1_model,
        "n1_train_receipt": n1_receipt,
        "n1_validation_predictions": n1_predictions,
        "validation_report": report,
    }
    assert set(artifacts) == EXPECTED_ARTIFACT_ROLES
    return paths, artifacts


def test_final_receipt_rejects_unlisted_files_and_binds_real_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, artifacts = _build_real_receipted_run(tmp_path, monkeypatch)
    run = paths["run"]
    rogue = run / "test.predictions.jsonl"
    rogue.write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(LadderError, match="unreceipted"):
        finalize_run(
            run_root=run,
            artifacts=artifacts,
            config_path=CONFIG,
            source_rehydration_receipt_path=paths["receipt"],
            uenv="fixture-uenv",
            code_commit="a" * 40,
            job_id="123",
            output_path=run / "run.receipt.json",
        )
    rogue.unlink()
    receipt = finalize_run(
        run_root=run,
        artifacts=artifacts,
        config_path=CONFIG,
        source_rehydration_receipt_path=paths["receipt"],
        uenv="fixture-uenv",
        code_commit="a" * 40,
        job_id="123",
        output_path=run / "run.receipt.json",
    )
    assert receipt["status"] == "passed_cpu_sealed_retrospective_comparison"
    assert len(receipt["artifacts"]) == len(EXPECTED_ARTIFACT_ROLES)
    assert receipt["human_annotation"]["required"] is False
    assert verify_published_run(run)["status"] == "pass"


def test_finalizer_rejects_model_prediction_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, artifacts = _build_real_receipted_run(tmp_path, monkeypatch)
    prediction_path = artifacts["c1_validation_predictions"]
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    current = rows[0]["lines"][0]["prediction"]
    rows[0]["lines"][0]["prediction"] = "BIB" if current == "O" else "O"
    _jsonl(prediction_path, rows)
    c1_receipt = json.loads(artifacts["c1_receipt"].read_text(encoding="utf-8"))
    c1_receipt["outputs"]["validation_predictions_sha256"] = sha256_file(
        prediction_path
    )
    artifacts["c1_receipt"].unlink()
    _json(artifacts["c1_receipt"], c1_receipt)
    report = json.loads(artifacts["validation_report"].read_text(encoding="utf-8"))
    report["prediction_sha256"]["c1-feature-bioes-crf"] = sha256_file(prediction_path)
    report["arm_receipt_sha256"]["c1-feature-bioes-crf"] = sha256_file(
        artifacts["c1_receipt"]
    )
    artifacts["validation_report"].unlink()
    _json(artifacts["validation_report"], report)
    with pytest.raises(LadderError, match="do not reproduce"):
        finalize_run(
            run_root=paths["run"],
            artifacts=artifacts,
            config_path=CONFIG,
            source_rehydration_receipt_path=paths["receipt"],
            uenv="fixture-uenv",
            code_commit="a" * 40,
            job_id="123",
            output_path=paths["run"] / "run.receipt.json",
        )


def test_finalizer_rejects_report_status_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, artifacts = _build_real_receipted_run(tmp_path, monkeypatch)
    report = json.loads(artifacts["validation_report"].read_text(encoding="utf-8"))
    report["status"] = "operator_declared_pass"
    artifacts["validation_report"].unlink()
    _json(artifacts["validation_report"], report)
    with pytest.raises(LadderError, match="report status/provenance"):
        finalize_run(
            run_root=paths["run"],
            artifacts=artifacts,
            config_path=CONFIG,
            source_rehydration_receipt_path=paths["receipt"],
            uenv="fixture-uenv",
            code_commit="a" * 40,
            job_id="123",
            output_path=paths["run"] / "run.receipt.json",
        )


def test_clariden_runner_is_cpu_only_and_has_no_test_prediction_flag() -> None:
    script = (
        EVAL_DIR / "sequence_models" / "clariden" / "run_bib_ladder.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --gres" not in script
    assert "phase04_require_cpu_request" in script
    assert 'CUDA_VISIBLE_DEVICES=""' in script
    assert "--test-predictions" not in script
    assert "prepare-selection" in script
    assert "selection.train-validation.jsonl" in script
    assert "finalize-run" in script
    assert "flock -n" in script
    assert "mv --no-clobber -T" in script
    assert "verify-published-run" in script
    assert "N1_PROFILE_RECEIPT" in script
    assert "SEQUENCE_UENV" in script
    assert "CONFIRM_CLASSIFIER_COMPARISON" in script

    profile = (
        EVAL_DIR / "sequence_models" / "clariden" / "profile_n1.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --time=12:00:00" in profile
    assert "char_tcn_crf profile" in profile
    assert "flock -n" in profile
    assert "mv --no-clobber -T" in profile

    feature_cli = (EVAL_DIR / "sequence_models" / "feature_crf.py").read_text(
        encoding="utf-8"
    )
    assert "--test-predictions" not in feature_cli
    assert 'parser.add_argument("--seed"' not in feature_cli
    assert "verify_selection_bundle" in feature_cli
    assert 'parser.add_argument("--receipt-out", required=True)' in feature_cli
