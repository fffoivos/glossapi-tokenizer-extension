from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"


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


SELECTION = load_module(
    "phase04_structural_classifier_selection",
    SCRIPTS / "structural_classifier_selection.py",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    config = tmp_path / "joint_config.json"
    write_json(
        config,
        {
            "schema_version": "academic-structure-sequence-eval-v1",
            "active_classes": ["BIB", "TOC"],
            "split": {"test_fraction": 0.0},
            "historical_partition_usage": {
                "historical_train_document_count": 4,
                "historical_test_document_count": 2,
            }
        },
    )
    split = tmp_path / "source" / "struct2k.LLM_silver.split.json"
    inventory = "1" * 64
    write_json(
        split,
        {
            "schema_version": "academic-structure-split-v1",
            "inventory_sha256": inventory,
            "assignments": {
                "d0": "train",
                "d1": "train",
                "d2": "validation",
                "d3": "validation",
            },
        },
    )
    source = tmp_path / "source" / "struct2k.LLM_silver.receipt.json"
    write_json(
        source,
        {
            "schema_version": "academic-structure-silver-contract-receipt-v1",
            "status": "pass",
            "evidence_tier": "LLM_silver",
            "production_eligible": False,
            "sequence_fit_eligible": True,
            "config_sha256": SELECTION.sha256_file(config),
            "silver_sha256": "2" * 64,
            "split_manifest_sha256": SELECTION.sha256_file(split),
            "inventory_sha256": inventory,
            "document_count": 4,
            "split_counts": {"train": 2, "validation": 2},
            "task_scope_counts": {"bibliography_toc_windows": 4},
            "materialized_artifacts": {
                "silver_filename": "struct2k.LLM_silver.jsonl",
                "split_manifest_filename": split.name,
            },
            "historical_partition_exclusion": {
                "status": "passed_before_materialization",
                "eligible_historical_train_documents": 4,
                "historical_test_documents_excluded": 2,
                "historical_test_rows_emitted": 0,
                "historical_test_predictions_permitted": False,
            },
        },
    )
    bib = tmp_path / "span_line_lr_struct_model.json"
    toc = tmp_path / "toc_line_lr_model.json"
    smoother = tmp_path / "struct_smooth_params.json"
    for path, value in ((bib, {"bib": 1}), (toc, {"toc": 1}), (smoother, {"smooth": 1})):
        write_json(path, value)

    run_root = tmp_path / "ladder"
    run_root.mkdir()
    paths = {}
    for role in sorted(SELECTION.EXPECTED_ARTIFACT_ROLES):
        path = run_root / f"{role}.artifact"
        if role == "c0_receipt":
            write_json(
                path,
                {
                    "schema_version": "academic-structure-c0-reference-v2",
                    "status": "passed_descriptive_reference_prediction",
                    "architecture_id": SELECTION.C0,
                    "target": "BIB+TOC",
                    "active_classes": ["BIB", "TOC"],
                    "comparison_role": "historical_reference_only",
                    "production_eligible": False,
                    "model_artifacts": {
                        bib.name: SELECTION.sha256_file(bib),
                        toc.name: SELECTION.sha256_file(toc),
                        smoother.name: SELECTION.sha256_file(smoother),
                    },
                },
            )
        else:
            path.write_text(f"{role}\n", encoding="utf-8")
        paths[role] = path
    artifacts = [
        {
            "role": role,
            "relative_path": path.name,
            "sha256": SELECTION.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for role, path in sorted(paths.items())
    ]
    write_json(
        run_root / "run.receipt.json",
        {
            "schema_version": "academic-structure-bib-ladder-run-v1",
            "status": "passed_cpu_sealed_retrospective_comparison",
            "target": "BIB+TOC",
            "active_classes": ["BIB", "TOC"],
            "architecture_ids": SELECTION.EXPECTED_ARCHITECTURES,
            "config_sha256": SELECTION.sha256_file(config),
            "production_eligible": False,
            "decision": "LLM_silver_replay_only_no_automatic_selection_no_production_change",
            "source": {
                "silver_sha256": "2" * 64,
                "split_manifest_sha256": SELECTION.sha256_file(split),
                "rehydration_receipt_sha256": SELECTION.sha256_file(source),
            },
            "historically_named_test_partition": {
                "documents_loaded_by_model_or_validation_processes": 0,
                "predictions_written": 0,
            },
            "artifacts": artifacts,
            "artifact_inventory_sha256": SELECTION.canonical_json_sha256(artifacts),
        },
    )
    return argparse.Namespace(
        ladder_run_root=run_root,
        source_receipt=source,
        source_split_manifest=split,
        sequence_config=config,
        bib_model=bib,
        toc_model=toc,
        smoother=smoother,
        selected_architecture=SELECTION.C0,
        selection_rationale="C0 is the only arm with an existing Rust implementation.",
        output=tmp_path / "nested" / "selection.json",
    )


def test_c0_selection_binds_joint_ladder_source_and_deployment_artifacts(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    receipt = SELECTION.build_selection(args)
    assert receipt["selected_architecture"] == SELECTION.C0
    assert receipt["automatic_winner_selection"] is False
    assert receipt["structural_application_eligible"] is False
    assert args.output.is_file(), "atomic writer must create the requested parent"
    with pytest.raises(FileExistsError, match="overwrite"):
        SELECTION.build_selection(args)

    args.bib_model.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact|model hash|file receipt"):
        SELECTION.validate_selection(
            args.output,
            source_receipt=args.source_receipt,
            source_split_manifest=args.source_split_manifest,
            sequence_config=args.sequence_config,
            bib_model=args.bib_model,
            toc_model=args.toc_model,
            smoother=args.smoother,
        )


@pytest.mark.parametrize("architecture", sorted(SELECTION.RESEARCH_ONLY))
def test_non_c0_research_arms_require_a_separate_rust_port_package(
    tmp_path: Path, architecture: str
) -> None:
    args = _fixture(tmp_path)
    args.selected_architecture = architecture
    with pytest.raises(ValueError, match="Python research arm|Rust.*port"):
        SELECTION.build_selection(args)
