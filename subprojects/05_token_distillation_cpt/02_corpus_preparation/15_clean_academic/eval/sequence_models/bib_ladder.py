#!/usr/bin/env python3
"""Receipt-bound preparation and reporting for the BIB-only and joint ladders.

The SPAN artifact contains a locked retrospective test split; the recovered
STRUCT-2K source is materialized only after its historical test is excluded.
This module is the narrow gate that emits a receipt-bound train+validation
selection. Model processes receive only that selection and never open the
historically named sealed partition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import resource
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contract import (
    GoldDocument,
    build_split_manifest,
    canonical_json_sha256,
    read_gold,
    sha256_file,
    validate_silver,
)
from .evaluate import evaluate, read_predictions

SELECTION_SCHEMA = "academic-structure-bib-selection-v1"
VALIDATION_REPORT_SCHEMA = "academic-structure-bib-validation-comparison-v1"
RUN_RECEIPT_SCHEMA = "academic-structure-bib-ladder-run-v1"
EXPECTED_CANDIDATES = (
    "c1-feature-bioes-crf",
    "c2-char-ngram-feature-bioes-crf",
    "n1-bytecnn-tcn-masked-crf",
)
EXPECTED_ARTIFACT_ROLES = {
    "selection_silver",
    "selection_manifest",
    "validation_silver",
    "selection_receipt",
    "c0_validation_predictions",
    "c0_receipt",
    "c1_model",
    "c1_validation_predictions",
    "c1_receipt",
    "c2_model",
    "c2_validation_predictions",
    "c2_receipt",
    "n1_profile_receipt",
    "n1_model",
    "n1_train_receipt",
    "n1_validation_predictions",
    "validation_report",
}
TASK_SCOPE_CLASSES = {
    "bibliography_binary_windows": ("BIB",),
    "bibliography_toc_windows": ("BIB", "TOC"),
}
STRUCT2K_LOCK_PATH = Path(__file__).with_name("struct2k_handoff_lock.json")


class LadderError(ValueError):
    """Raised when evidence or output state would invalidate the comparison."""


def active_classes_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("active_classes", ["BIB"])
    if raw not in (["BIB"], ["BIB", "TOC"]):
        raise LadderError(f"unsupported active_classes contract: {raw!r}")
    return tuple(raw)


def active_classes_for_documents(
    documents: Sequence[GoldDocument], config: Mapping[str, Any]
) -> tuple[str, ...]:
    scopes = {document.task_scope for document in documents}
    if len(scopes) != 1 or next(iter(scopes)) not in TASK_SCOPE_CLASSES:
        raise LadderError(
            f"unsupported or mixed task scopes: {sorted(map(str, scopes))!r}"
        )
    observed = TASK_SCOPE_CLASSES[next(iter(scopes))]
    configured = active_classes_from_config(config)
    if observed != configured:
        raise LadderError(
            f"task scope activates {observed!r}, but config activates {configured!r}"
        )
    if configured == ("BIB",) and any(
        line.label == "TOC" for document in documents for line in document.lines
    ):
        raise LadderError("BIB-only evidence contains observed ToC labels")
    return configured


def target_name(active_classes: Sequence[str]) -> str:
    return "+".join(active_classes)


def mark_silver_safety_unavailable(
    metrics: dict[str, Any], active_classes: Sequence[str]
) -> None:
    joint = tuple(active_classes) == ("BIB", "TOC")
    for row in [metrics, *metrics.get("by_source", {}).values()]:
        row["token"]["prose_contamination"] = None
        row["token"]["true_main_text_retention"] = None
        row["document"]["catastrophic_prose_deletions"] = None
        row["document"]["maximum_contiguous_false_deletion_tokens"] = None
        if not joint:
            row["token"]["toc_recall"] = None
            row["line"]["toc_recall"] = None
            row["span"]["toc"] = {key: None for key in row["span"]["toc"]}
    metrics["metric_availability"] = {
        "silver_agreement_metrics": True,
        "bib_metrics": True,
        "toc_metrics": joint,
        "toc_reason": None if joint else "SPAN supervision is BIB-only",
        "independent_running_prose_safety_metrics": False,
    }


def configure_runtime(
    config: Mapping[str, Any], *, uenv: str, effective_seed: int | None
) -> dict[str, Any]:
    """Configure and describe the exact CPU Python runtime used by an arm."""
    if not uenv or os.environ.get("SEQUENCE_UENV") != uenv:
        raise LadderError("declared uenv differs from SEQUENCE_UENV")
    expected_hash_seed = str(config["execution"]["python_hash_seed"])
    if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise LadderError("PYTHONHASHSEED differs from the sequence config")
    import torch

    threads = int(config["execution"]["torch_num_threads"])
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(
        bool(config["execution"]["deterministic_algorithms"])
    )
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    return {
        "device": "cpu",
        "accelerator_used": False,
        "uenv": uenv,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "python_hash_seed": expected_hash_seed,
        "effective_seed": effective_seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "executable": sys.executable,
    }


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def select_shared_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_action_precision: float,
    active_classes: Sequence[str] = ("BIB",),
) -> dict[str, Any]:
    """Apply one auditable validation-silver operating-point rule to every arm.

    C0 precision is only a historical descriptive operating point. Its old
    STRUCT-2K overlap with SPAN cannot be excluded, and joint validation is
    known to be a subset of C0's historical training partition. It is a
    matching rule, never a held-out safety floor or gain estimate.
    """
    active = tuple(active_classes)
    if active not in (("BIB",), ("BIB", "TOC")):
        raise LadderError(f"unsupported calibration active classes: {active!r}")
    recall_key = "bib_recall" if active == ("BIB",) else "action_recall"
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = {
            "deletion_bias": float(raw["deletion_bias"]),
            "action_precision": float(raw["action_precision"]),
            "action_recall": float(raw["action_recall"]),
            "bib_recall": float(raw["bib_recall"]),
            "predicted_action_tokens": int(raw["predicted_action_tokens"]),
        }
        if active == ("BIB", "TOC"):
            row["toc_recall"] = float(raw["toc_recall"])
        normalized.append(row)
    if not normalized:
        raise LadderError("calibration grid is empty")
    biases = [row["deletion_bias"] for row in normalized]
    if len(set(biases)) != len(biases):
        raise LadderError("calibration grid contains duplicate deletion biases")
    frontier = [
        row
        for row in normalized
        if not any(
            other["action_precision"] >= row["action_precision"]
            and other[recall_key] >= row[recall_key]
            and (
                other["action_precision"] > row["action_precision"]
                or other[recall_key] > row[recall_key]
            )
            for other in normalized
        )
    ]
    frontier.sort(
        key=lambda row: (
            row["action_precision"],
            row[recall_key],
            -row["deletion_bias"],
        )
    )
    eligible = [
        row for row in frontier if row["action_precision"] >= reference_action_precision
    ]
    if not eligible:
        raise LadderError(
            "no calibration point matches the descriptive C0 action precision"
        )
    selected = max(
        eligible,
        key=lambda row: (
            row[recall_key],
            row["action_precision"],
            -row["deletion_bias"],
        ),
    )
    return {
        "schema_version": "academic-structure-shared-calibration-v1",
        "rule": (
            "maximize_BIB_recall_on_PR_frontier_at_or_above_descriptive_C0_action_precision"
            if active == ("BIB",)
            else "maximize_joint_action_recall_on_PR_frontier_at_or_above_descriptive_C0_action_precision"
        ),
        "active_classes": list(active),
        "reference": {
            "architecture_id": "c0-rust-lr-hysteresis",
            "action_precision": float(reference_action_precision),
            "role": "historical_reference_only_not_independent_heldout_safety_floor",
        },
        "grid": normalized,
        "pareto_frontier": frontier,
        "selected": selected,
        "evidence_semantics": "LLM_silver_agreement_only",
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LadderError(f"{path}: expected a JSON object")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: str | Path, value: Any) -> None:
    _atomic_write(path, _json_bytes(value))


def _atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    _atomic_write(path, payload)


def _require_new_outputs(*paths: str | Path) -> None:
    existing = [
        str(path) for path in map(Path, paths) if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(f"refusing immutable output overwrite: {existing}")


def _manifest_for_subset(
    documents: Sequence[GoldDocument], source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    inventory = sorted((doc.document_id, doc.work_id, doc.source) for doc in documents)
    assignments = {doc.document_id: doc.split for doc in documents}
    return {
        "schema_version": "academic-structure-split-v1",
        "seed": source_manifest.get("seed"),
        "algorithm": "locked-source-manifest-subset-v1",
        "source_inventory_sha256": source_manifest.get("inventory_sha256"),
        "inventory_sha256": canonical_json_sha256(inventory),
        "assignments": dict(sorted(assignments.items())),
        "permitted_splits": ["train", "validation"],
    }


def _validate_rehydration_receipt(
    receipt: Mapping[str, Any],
    *,
    silver_path: str | Path,
    split_manifest_path: str | Path,
    config_path: str | Path,
) -> None:
    if receipt.get("schema_version") != "academic-structure-silver-contract-receipt-v1":
        raise LadderError("source receipt is not a hydrated silver contract receipt")
    if receipt.get("silver_sha256") != sha256_file(silver_path):
        raise LadderError("source silver bytes differ from their rehydration receipt")
    if receipt.get("split_manifest_sha256") != sha256_file(split_manifest_path):
        raise LadderError("source split manifest differs from its rehydration receipt")
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise LadderError(
            "source silver was hydrated under a different sequence config"
        )
    if receipt.get("sequence_fit_eligible") is not True:
        raise LadderError(
            "rehydration receipt does not authorize silver research fitting"
        )
    if receipt.get("sequence_evidence_scope") != "LLM_silver_comparison_only":
        raise LadderError(
            "rehydration receipt has an unexpected research evidence scope"
        )
    if receipt.get("production_eligible") is not False:
        raise LadderError("LLM silver must remain ineligible for production")
    snapshot = receipt.get("source_unit_snapshot")
    if not isinstance(snapshot, Mapping):
        raise LadderError("rehydration receipt lacks its source-unit snapshot binding")
    if (
        snapshot.get("research_fit_eligible") is not True
        or snapshot.get("research_evidence_scope") != "LLM_silver_comparison_only"
        or snapshot.get("production_eligible") is not False
    ):
        raise LadderError(
            "source-unit snapshot does not authorize comparison-only fitting"
        )
    digest = snapshot.get("receipt_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise LadderError("source-unit snapshot receipt SHA-256 is missing")
    receipt_path = snapshot.get("receipt_path")
    if not isinstance(receipt_path, str) or not receipt_path:
        raise LadderError("source-unit snapshot receipt path is missing")
    unit_receipt = Path(receipt_path)
    if not unit_receipt.is_absolute():
        unit_receipt = Path(silver_path).resolve().parent / unit_receipt
    if unit_receipt.is_symlink() or not unit_receipt.is_file():
        raise LadderError("source-unit snapshot receipt is not a regular file")
    if unit_receipt.resolve().parent != Path(silver_path).resolve().parent:
        raise LadderError(
            "source-unit snapshot receipt is outside the immutable hydration root"
        )
    if sha256_file(unit_receipt) != digest:
        raise LadderError(
            "source-unit snapshot receipt bytes differ from the silver receipt"
        )
    unit_value = _load_json(unit_receipt)
    if unit_value.get("schema_version") == "span-unit-rehydration-receipt-v1":
        valid_snapshot = (
            unit_value.get("operation") == "text_payload_rehydration_only"
            and unit_value.get("labels_read_created_or_inferred") is False
            and unit_value.get("research_fit_eligible") is True
            and unit_value.get("research_evidence_scope")
            == "LLM_silver_comparison_only"
            and unit_value.get("promotion_eligible") is False
        )
    elif unit_value.get("schema_version") == "struct2k-handoff-audit-receipt-v1":
        lock_path = STRUCT2K_LOCK_PATH
        lock = _load_json(lock_path)
        handoff = unit_value.get("handoff", {})
        historical = unit_value.get("historical_partition", {})
        valid_snapshot = (
            unit_value.get("status")
            == "passed_inventory_and_legacy_replay_with_locked_coordinate_corrections"
            and unit_value.get("operation")
            == "audit_and_transcode_existing_joint_LLM_silver"
            and unit_value.get("annotation_status") == "LLM_silver"
            and unit_value.get("human_gold") is False
            and unit_value.get("new_semantic_annotations_created") is False
            and unit_value.get("coordinate_corrections_applied")
            == len(lock.get("coordinate_corrections", []))
            and unit_value.get("coordinate_correction_lock_sha256")
            == sha256_file(lock_path)
            and handoff.get("inventory_sha256") == lock["inventory"]["sha256"]
            and handoff.get("legacy_silver_sha256")
            == lock["inventory"]["required_files"]["STRUCT_2K_gold.jsonl"]
            and handoff.get("source_commit") == lock["source"]["commit"]
            and historical.get("eligible_for_new_split") == "train_only"
            and historical.get("test_documents_available_to_model_processes") == 0
            and unit_value.get("research_fit_eligible") is True
            and unit_value.get("research_evidence_scope")
            == "LLM_silver_comparison_only"
            and unit_value.get("production_eligible") is False
        )
    else:
        valid_snapshot = False
    if not valid_snapshot:
        raise LadderError("source-unit snapshot receipt semantics are invalid")


def _source_artifact_paths(
    receipt_path: str | Path, receipt: Mapping[str, Any]
) -> tuple[Path, Path]:
    root = Path(receipt_path).resolve().parent
    artifacts = receipt.get("materialized_artifacts")
    if artifacts is None:
        silver_name = "span.LLM_silver.jsonl"
        split_name = "span.LLM_silver.split.json"
    else:
        if not isinstance(artifacts, Mapping):
            raise LadderError("source receipt materialized_artifacts is malformed")
        silver_name = artifacts.get("silver_filename")
        split_name = artifacts.get("split_manifest_filename")
    paths: list[Path] = []
    for name in (silver_name, split_name):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or Path(name).is_absolute()
        ):
            raise LadderError("source receipt contains an unsafe artifact filename")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise LadderError(f"source artifact is not a regular file: {name}")
        paths.append(path)
    if receipt.get("silver_sha256") != sha256_file(paths[0]):
        raise LadderError("resolved source silver differs from its receipt")
    if receipt.get("split_manifest_sha256") != sha256_file(paths[1]):
        raise LadderError("resolved source split differs from its receipt")
    return paths[0], paths[1]


def prepare_selection(
    *,
    silver_path: str | Path,
    split_manifest_path: str | Path,
    rehydration_receipt_path: str | Path,
    config_path: str | Path,
    selection_silver_path: str | Path,
    selection_manifest_path: str | Path,
    validation_silver_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Validate full evidence once and exclude the historical comparison partition."""
    _require_new_outputs(
        selection_silver_path,
        selection_manifest_path,
        validation_silver_path,
        receipt_path,
    )
    config = _load_json(config_path)
    if config.get("schema_version") != "academic-structure-sequence-eval-v1":
        raise LadderError("unsupported sequence-model config")
    source_receipt = _load_json(rehydration_receipt_path)
    _validate_rehydration_receipt(
        source_receipt,
        silver_path=silver_path,
        split_manifest_path=split_manifest_path,
        config_path=config_path,
    )
    source_manifest = _load_json(split_manifest_path)
    documents = read_gold(silver_path)
    recomputed_manifest = build_split_manifest(documents, config["split"])
    if source_manifest != recomputed_manifest:
        raise LadderError(
            "source split manifest is not the exact config-derived manifest"
        )
    if Path(split_manifest_path).read_bytes() != _json_bytes(recomputed_manifest):
        raise LadderError(
            "source split manifest bytes are not canonical config-derived JSON"
        )
    contract = validate_silver(
        documents, config["silver_contract"], split_manifest=source_manifest
    )
    if contract["inventory_sha256"] != source_receipt.get("inventory_sha256"):
        raise LadderError(
            "source receipt contract inventory differs from current silver"
        )
    active_classes = active_classes_for_documents(documents, config)
    snapshot_contract = source_receipt.get("source_unit_snapshot", {})
    if (
        active_classes == ("BIB", "TOC")
        and snapshot_contract.get("snapshot_schema_version")
        != "struct2k-handoff-audit-receipt-v1"
    ):
        raise LadderError(
            "joint evidence must originate from the locked STRUCT-2K handoff audit"
        )

    source_rows: list[dict[str, Any]] = []
    with Path(silver_path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise LadderError(f"source row {row_number}: expected an object")
            source_rows.append(row)
    rows_by_id = {str(row.get("document_id")): row for row in source_rows}
    if len(rows_by_id) != len(documents):
        raise LadderError("source JSONL document identity inventory is not unique")

    selected_documents = [
        doc for doc in documents if doc.split in ("train", "validation")
    ]
    validation_documents = [doc for doc in documents if doc.split == "validation"]
    test_documents = [doc for doc in documents if doc.split == "test"]
    exclusion = source_receipt.get("historical_partition_exclusion")
    if active_classes == ("BIB",):
        if not selected_documents or not validation_documents or not test_documents:
            raise LadderError(
                "SPAN evidence requires non-empty train, validation, and test splits"
            )
        excluded_count = len(test_documents)
    else:
        expected_historical_test = int(
            config.get("historical_partition_usage", {}).get(
                "historical_test_document_count", -1
            )
        )
        if not selected_documents or not validation_documents or test_documents:
            raise LadderError(
                "joint materialization must contain train/validation only after upstream exclusion"
            )
        if (
            not isinstance(exclusion, Mapping)
            or exclusion.get("status") != "passed_before_materialization"
            or expected_historical_test <= 0
            or exclusion.get("historical_test_documents_excluded")
            != expected_historical_test
            or exclusion.get("historical_test_rows_emitted") != 0
            or exclusion.get("historical_test_predictions_permitted") is not False
            or exclusion.get("eligible_historical_train_documents") != len(documents)
        ):
            raise LadderError(
                "joint source receipt does not prove the configured historical-test exclusion"
            )
        excluded_count = expected_historical_test
    selection_manifest = _manifest_for_subset(selected_documents, source_manifest)
    selected_rows = [rows_by_id[doc.document_id] for doc in selected_documents]
    validation_rows = [rows_by_id[doc.document_id] for doc in validation_documents]

    _atomic_jsonl(selection_silver_path, selected_rows)
    _atomic_json(selection_manifest_path, selection_manifest)
    _atomic_jsonl(validation_silver_path, validation_rows)
    selection_contract = validate_silver(
        read_gold(selection_silver_path),
        config["silver_contract"],
        split_manifest=selection_manifest,
    )
    receipt = {
        "schema_version": SELECTION_SCHEMA,
        "status": "passed_historical_partition_physically_excluded",
        "target": target_name(active_classes),
        "active_classes": list(active_classes),
        "evidence_tier": "LLM_silver",
        "production_eligible": False,
        "source": {
            "silver_sha256": sha256_file(silver_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "rehydration_receipt_sha256": sha256_file(rehydration_receipt_path),
            "source_snapshot_receipt_sha256": source_receipt["source_unit_snapshot"][
                "receipt_sha256"
            ],
            "snapshot_equivalence_status": source_receipt["source_unit_snapshot"].get(
                "snapshot_equivalence_status"
            ),
            "sequence_evidence_scope": "LLM_silver_comparison_only",
        },
        "config_sha256": sha256_file(config_path),
        "outputs": {
            "selection_silver_sha256": sha256_file(selection_silver_path),
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "validation_silver_sha256": sha256_file(validation_silver_path),
        },
        "counts": {
            "source_documents": len(documents),
            "train_documents": sum(doc.split == "train" for doc in selected_documents),
            "validation_documents": len(validation_documents),
            "historically_named_test_documents_excluded": excluded_count,
            "source_materialized_documents": len(documents),
        },
        "selection_contract": selection_contract,
        "architecture_access_contract": {
            "permitted_splits": ["train", "validation"],
            "historically_named_test_rows_emitted": 0,
            "historical_partition_predictions_permitted": False,
            "selection_rule": "locked split assignment only; labels never select rows",
            "partition_semantics": (
                "sealed_retrospective_comparison; not an unbiased never-seen test"
            ),
            "note": (
                "Only this preparation gate validates the full rehydrated inventory. "
                "Architecture fitting and validation consume the emitted subset files. "
                "No human annotation is required; a future independent evaluation may use "
                "newly sampled LLM-silver documents."
            ),
        },
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def verify_selection_bundle(
    *,
    selection_silver_path: str | Path,
    selection_manifest_path: str | Path,
    validation_silver_path: str | Path,
    selection_receipt_path: str | Path,
    config_path: str | Path,
) -> tuple[list[GoldDocument], list[GoldDocument], dict[str, Any]]:
    receipt = _load_json(selection_receipt_path)
    if receipt.get("schema_version") != SELECTION_SCHEMA:
        raise LadderError("selection receipt has an unsupported schema")
    if receipt.get("status") != "passed_historical_partition_physically_excluded":
        raise LadderError(
            "selection receipt did not pass historical-partition exclusion"
        )
    config = _load_json(config_path)
    expected_active = active_classes_from_config(config)
    if (
        receipt.get("target") != target_name(expected_active)
        or receipt.get("active_classes") != list(expected_active)
        or receipt.get("production_eligible") is not False
    ):
        raise LadderError("selection receipt task differs from the active config")
    expected = {
        "selection_silver_sha256": sha256_file(selection_silver_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "validation_silver_sha256": sha256_file(validation_silver_path),
    }
    if receipt.get("outputs") != expected:
        raise LadderError("selection bundle bytes differ from their immutable receipt")
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise LadderError("selection bundle was prepared under a different config")
    access = receipt.get("architecture_access_contract")
    if not isinstance(access, Mapping) or (
        access.get("permitted_splits") != ["train", "validation"]
        or access.get("historically_named_test_rows_emitted") != 0
        or access.get("historical_partition_predictions_permitted") is not False
        or access.get("partition_semantics")
        != "sealed_retrospective_comparison; not an unbiased never-seen test"
    ):
        raise LadderError(
            "selection receipt does not prove historical-partition exclusion"
        )

    manifest = _load_json(selection_manifest_path)
    documents = read_gold(selection_silver_path)
    active_classes_for_documents(documents, config)
    recomputed_contract = validate_silver(
        documents, config["silver_contract"], split_manifest=manifest
    )
    if receipt.get("selection_contract") != recomputed_contract:
        raise LadderError("selection receipt contract differs from current files")
    if {doc.split for doc in documents} != {"train", "validation"}:
        raise LadderError(
            "selection bundle must contain exactly train and validation splits"
        )
    validation = read_gold(validation_silver_path)
    if any(doc.split != "validation" for doc in validation):
        raise LadderError("validation view contains a non-validation document")
    expected_validation = [
        doc.document_id for doc in documents if doc.split == "validation"
    ]
    if [doc.document_id for doc in validation] != expected_validation:
        raise LadderError("validation view is not the exact ordered selection subset")
    counts = receipt.get("counts")
    if not isinstance(counts, Mapping):
        raise LadderError("selection receipt document counts are malformed")
    excluded = counts.get("historically_named_test_documents_excluded")
    expected_source_documents = len(documents)
    if expected_active == ("BIB",):
        if not isinstance(excluded, int) or isinstance(excluded, bool) or excluded <= 0:
            raise LadderError("BIB selection receipt lacks its excluded-test count")
        expected_source_documents += excluded
    else:
        expected_excluded = config.get("historical_partition_usage", {}).get(
            "historical_test_document_count"
        )
        if excluded != expected_excluded:
            raise LadderError("joint selection receipt excluded-test count drift")
    if (
        counts.get("train_documents") != sum(doc.split == "train" for doc in documents)
        or counts.get("validation_documents") != len(validation)
        or counts.get("source_documents") != expected_source_documents
        or counts.get("source_materialized_documents") != expected_source_documents
    ):
        raise LadderError("selection receipt document counts differ from current files")
    return documents, validation, receipt


def _prediction_model_ids(path: str | Path) -> set[str]:
    model_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            model_id = row.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise LadderError(f"prediction row {row_number}: model_id is required")
            model_ids.add(model_id)
    return model_ids


def _mark_silver_safety_unavailable(
    metrics: dict[str, Any], active_classes: Sequence[str] = ("BIB",)
) -> None:
    mark_silver_safety_unavailable(metrics, active_classes)


def evaluate_validation(
    *,
    selection_silver_path: str | Path,
    selection_manifest_path: str | Path,
    validation_silver_path: str | Path,
    selection_receipt_path: str | Path,
    config_path: str | Path,
    baseline_path: str | Path,
    candidate_paths: Mapping[str, str | Path],
    arm_receipt_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    _require_new_outputs(output_path)
    config = _load_json(config_path)
    documents, validation, selection_receipt = verify_selection_bundle(
        selection_silver_path=selection_silver_path,
        selection_manifest_path=selection_manifest_path,
        validation_silver_path=validation_silver_path,
        selection_receipt_path=selection_receipt_path,
        config_path=config_path,
    )
    active_classes = active_classes_for_documents(documents, config)
    if tuple(sorted(candidate_paths)) != tuple(sorted(EXPECTED_CANDIDATES)):
        raise LadderError(
            f"validation comparison requires exactly {EXPECTED_CANDIDATES!r}"
        )
    expected_arm_ids = {"c0-rust-lr-hysteresis", *EXPECTED_CANDIDATES}
    if set(arm_receipt_paths) != expected_arm_ids:
        raise LadderError("validation comparison requires one receipt for every arm")
    c0_receipt = _load_json(arm_receipt_paths["c0-rust-lr-hysteresis"])
    if (
        c0_receipt.get("schema_version") != "academic-structure-c0-reference-v2"
        or c0_receipt.get("status") != "passed_descriptive_reference_prediction"
        or c0_receipt.get("comparison_role") != "historical_reference_only"
        or c0_receipt.get("production_eligible") is not False
        or c0_receipt.get("target") != target_name(active_classes)
        or c0_receipt.get("active_classes") != list(active_classes)
        or c0_receipt.get("outputs", {}).get("validation_predictions_sha256")
        != sha256_file(baseline_path)
    ):
        raise LadderError("C0 descriptive reference receipt is invalid")
    expected_c0_model_id = (
        "c0-rust-lr-hysteresis-python-bib-head"
        if active_classes == ("BIB",)
        else "c0-rust-lr-hysteresis-python-joint-heads"
    )
    baseline_ids = _prediction_model_ids(baseline_path)
    if baseline_ids != {expected_c0_model_id}:
        raise LadderError(
            f"unexpected C0 prediction model IDs: {sorted(baseline_ids)!r}"
        )
    baseline_predictions = read_predictions(baseline_path, validation)
    baseline_metrics, _ = evaluate(validation, baseline_predictions, split="validation")
    _mark_silver_safety_unavailable(baseline_metrics, active_classes)

    candidates: dict[str, Any] = {}
    prediction_receipts: dict[str, str] = {
        "c0-rust-lr-hysteresis": sha256_file(baseline_path)
    }
    for architecture_id in EXPECTED_CANDIDATES:
        path = candidate_paths[architecture_id]
        arm_receipt = _load_json(arm_receipt_paths[architecture_id])
        expected_schema = (
            "academic-structure-n1-training-v2"
            if architecture_id == "n1-bytecnn-tcn-masked-crf"
            else "academic-structure-feature-crf-training-v2"
        )
        if (
            arm_receipt.get("schema_version") != expected_schema
            or arm_receipt.get("architecture_id") != architecture_id
            or arm_receipt.get("target") != target_name(active_classes)
            or arm_receipt.get("active_classes") != list(active_classes)
            or arm_receipt.get("production_eligible") is not False
            or arm_receipt.get("outputs", {}).get("validation_predictions_sha256")
            != sha256_file(path)
        ):
            raise LadderError(f"{architecture_id}: arm receipt is invalid")
        if _prediction_model_ids(path) != {architecture_id}:
            raise LadderError(f"{architecture_id}: prediction model_id does not match")
        predictions = read_predictions(path, validation)
        metrics, _ = evaluate(validation, predictions, split="validation")
        _mark_silver_safety_unavailable(metrics, active_classes)
        candidates[architecture_id] = {
            "metrics": metrics,
            "calibration": arm_receipt["calibration"],
            "comparison_note": (
                "absolute LLM-silver replay metric; no fair held-out gain is claimed"
            ),
        }
        prediction_receipts[architecture_id] = sha256_file(path)

    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA,
        "status": "sealed_retrospective_comparison_complete",
        "evidence_tier": "LLM_silver",
        "target": target_name(active_classes),
        "active_classes": list(active_classes),
        "production_eligible": False,
        "selection_receipt_sha256": sha256_file(selection_receipt_path),
        "source_rehydration_receipt_sha256": selection_receipt["source"][
            "rehydration_receipt_sha256"
        ],
        "config_sha256": sha256_file(config_path),
        "validation_silver_sha256": sha256_file(validation_silver_path),
        "prediction_sha256": dict(sorted(prediction_receipts.items())),
        "arm_receipt_sha256": {
            architecture_id: sha256_file(path)
            for architecture_id, path in sorted(arm_receipt_paths.items())
        },
        "validation_document_count": len(validation),
        "historically_named_test_partition": {
            "documents_loaded_by_model_or_validation_processes": 0,
            "predictions_written": 0,
            "semantics": "sealed_retrospective_comparison_not_unbiased_never_seen_test",
        },
        "baseline": {
            "architecture_id": "c0-rust-lr-hysteresis",
            "comparison_role": "historical_reference_only",
            "overlap_caveat": c0_receipt["overlap_caveat"],
            "metrics": baseline_metrics,
        },
        "candidates": candidates,
        "selection_decision": {
            "automated": False,
            "reason": (
                "This is retrospective LLM-silver replay. It requires no human-gold dataset or "
                "annotation campaign, cannot establish production safety, and does not provide "
                "an unbiased final estimate. Future independent evaluation may use newly sampled "
                "LLM-silver documents."
            ),
        },
    }
    _atomic_json(output_path, report)
    return report


def _parse_artifacts(values: Sequence[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise LadderError(f"invalid --artifact value {value!r}; expected role=path")
        if role in artifacts:
            raise LadderError(f"duplicate artifact role {role!r}")
        artifacts[role] = Path(raw_path)
    if set(artifacts) != EXPECTED_ARTIFACT_ROLES:
        raise LadderError(
            "final artifact roles differ: "
            f"missing={sorted(EXPECTED_ARTIFACT_ROLES - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - EXPECTED_ARTIFACT_ROLES)}"
        )
    return artifacts


def finalize_run(
    *,
    run_root: str | Path,
    artifacts: Mapping[str, str | Path],
    config_path: str | Path,
    source_rehydration_receipt_path: str | Path,
    uenv: str,
    code_commit: str,
    job_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    _require_new_outputs(output_path)
    root = Path(run_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise LadderError("run root must be a real directory")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise LadderError("code commit must be an exact 40-character Git SHA")
    resolved: dict[str, Path] = {}
    for role, raw_path in artifacts.items():
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise LadderError(f"{role}: artifact must be a non-symlink regular file")
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise LadderError(f"{role}: artifact is outside the run root") from error
        resolved[role] = path
    if set(resolved) != EXPECTED_ARTIFACT_ROLES:
        raise LadderError(
            "finalization requires the exact sequence-ladder artifact role set"
        )
    discovered: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LadderError(f"run root contains a symlink: {path}")
        if path.is_file():
            discovered.add(path.resolve())
    if discovered != set(resolved.values()):
        raise LadderError(
            "run root contains unreceipted files: "
            f"{sorted(str(path.relative_to(root)) for path in discovered - set(resolved.values()))}"
        )

    config = _load_json(config_path)
    current_runtime = configure_runtime(
        config, uenv=uenv, effective_seed=int(config["execution"]["seed"])
    )
    documents, validation, selection_receipt = verify_selection_bundle(
        selection_silver_path=resolved["selection_silver"],
        selection_manifest_path=resolved["selection_manifest"],
        validation_silver_path=resolved["validation_silver"],
        selection_receipt_path=resolved["selection_receipt"],
        config_path=config_path,
    )
    active_classes = active_classes_for_documents(documents, config)
    if any(document.split == "test" for document in documents):
        raise LadderError("final selection verification loaded a historical test row")
    source_receipt_file = Path(source_rehydration_receipt_path)
    if source_receipt_file.is_symlink() or not source_receipt_file.is_file():
        raise LadderError(
            "source rehydration receipt must be a regular non-symlink file"
        )
    source_receipt = _load_json(source_receipt_file)
    source_silver_path, source_split_path = _source_artifact_paths(
        source_receipt_file, source_receipt
    )
    _validate_rehydration_receipt(
        source_receipt,
        silver_path=source_silver_path,
        split_manifest_path=source_split_path,
        config_path=config_path,
    )
    source_documents = read_gold(source_silver_path)
    source_manifest = _load_json(source_split_path)
    if source_manifest != build_split_manifest(source_documents, config["split"]):
        raise LadderError(
            "source split no longer recomputes from current identities/config"
        )
    source_rows = []
    with source_silver_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source_rows.append(json.loads(line))
    rows_by_id = {row["document_id"]: row for row in source_rows}
    source_selected = [
        document
        for document in source_documents
        if document.split in ("train", "validation")
    ]
    source_validation = [
        document for document in source_documents if document.split == "validation"
    ]
    expected_selection_payload = b"".join(
        (
            json.dumps(rows_by_id[doc.document_id], ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for doc in source_selected
    )
    expected_validation_payload = b"".join(
        (
            json.dumps(rows_by_id[doc.document_id], ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for doc in source_validation
    )
    if (
        resolved["selection_silver"].read_bytes() != expected_selection_payload
        or resolved["validation_silver"].read_bytes() != expected_validation_payload
        or _load_json(resolved["selection_manifest"])
        != _manifest_for_subset(source_selected, source_manifest)
        or resolved["selection_manifest"].read_bytes()
        != _json_bytes(_manifest_for_subset(source_selected, source_manifest))
    ):
        raise LadderError(
            "selection bundle does not exactly derive from the current source silver"
        )
    if (
        sha256_file(source_rehydration_receipt_path)
        != selection_receipt["source"]["rehydration_receipt_sha256"]
        or source_receipt.get("config_sha256") != sha256_file(config_path)
        or source_receipt.get("silver_sha256")
        != selection_receipt["source"]["silver_sha256"]
        or source_receipt.get("split_manifest_sha256")
        != selection_receipt["source"]["split_manifest_sha256"]
        or source_receipt.get("production_eligible") is not False
    ):
        raise LadderError(
            "current source rehydration receipt differs from selection provenance"
        )
    validation_report = _load_json(resolved["validation_report"])
    c0_receipt = _load_json(resolved["c0_receipt"])
    c1_receipt = _load_json(resolved["c1_receipt"])
    c2_receipt = _load_json(resolved["c2_receipt"])
    n1_receipt = _load_json(resolved["n1_train_receipt"])
    profile_receipt = _load_json(resolved["n1_profile_receipt"])
    expected_selection_sha = selection_receipt["outputs"]["selection_silver_sha256"]
    expected_manifest_sha = selection_receipt["outputs"]["selection_manifest_sha256"]
    expected_validation_sha = selection_receipt["outputs"]["validation_silver_sha256"]
    expected_selection_receipt_sha = sha256_file(resolved["selection_receipt"])
    config_sha = sha256_file(config_path)
    expected_common_inputs = {
        "selection_silver_sha256": expected_selection_sha,
        "selection_manifest_sha256": expected_manifest_sha,
        "validation_silver_sha256": expected_validation_sha,
        "selection_receipt_sha256": expected_selection_receipt_sha,
        "config_sha256": config_sha,
    }

    def validate_runtime(receipt: Mapping[str, Any], *, seed: int | None) -> None:
        runtime = receipt.get("execution", {})
        for key in (
            "device",
            "accelerator_used",
            "uenv",
            "python_version",
            "python_implementation",
            "numpy_version",
            "torch_version",
            "python_hash_seed",
            "effective_seed",
            "torch_intraop_threads",
            "torch_interop_threads",
            "deterministic_algorithms",
            "omp_num_threads",
            "mkl_num_threads",
            "slurm_cpus_per_task",
        ):
            expected = seed if key == "effective_seed" else current_runtime.get(key)
            if runtime.get(key) != expected:
                raise LadderError(
                    f"arm runtime {key} differs from finalization runtime"
                )
        if receipt.get("effective_seed") != seed:
            raise LadderError("arm effective seed differs from the config")
        measurements = (runtime.get("wall_seconds"), runtime.get("peak_rss_bytes"))
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in measurements
        ):
            raise LadderError("arm receipt lacks measured wall/RSS")

    expected_source_receipt_sha = sha256_file(source_rehydration_receipt_path)
    if (
        c0_receipt.get("schema_version") != "academic-structure-c0-reference-v2"
        or c0_receipt.get("status") != "passed_descriptive_reference_prediction"
        or c0_receipt.get("comparison_role") != "historical_reference_only"
        or c0_receipt.get("production_eligible") is not False
        or c0_receipt.get("target") != target_name(active_classes)
        or c0_receipt.get("active_classes") != list(active_classes)
        or c0_receipt.get("inputs")
        != {
            **expected_common_inputs,
            "source_rehydration_receipt_sha256": expected_source_receipt_sha,
        }
        or c0_receipt.get("outputs", {}).get("validation_predictions_sha256")
        != sha256_file(resolved["c0_validation_predictions"])
    ):
        raise LadderError("C0 reference receipt fails final validation")
    validate_runtime(c0_receipt, seed=None)

    from .baseline import EVAL_DIR, _load_json as load_baseline_json, predict_document

    bib_path = EVAL_DIR / "span_line_lr_struct_model.json"
    toc_path = EVAL_DIR / "toc_line_lr_model.json"
    decoder_path = EVAL_DIR / "struct_smooth_params.json"
    bib, toc, decoder = map(load_baseline_json, (bib_path, toc_path, decoder_path))
    c0_recomputed = {
        document.document_id: predict_document(
            document, bib, toc, decoder, active_classes=active_classes
        )[0]
        for document in validation
    }
    if (
        read_predictions(resolved["c0_validation_predictions"], validation)
        != c0_recomputed
    ):
        raise LadderError(
            "C0 predictions do not reproduce from tracked reference artifacts"
        )
    c0_metrics, _ = evaluate(validation, c0_recomputed, split="validation")
    c0_reference_precision = float(c0_metrics["token"]["action_precision"])

    def validate_calibration(receipt: Mapping[str, Any]) -> None:
        calibration = receipt.get("calibration", {})
        expected_rule = (
            "maximize_BIB_recall_on_PR_frontier_at_or_above_descriptive_C0_action_precision"
            if active_classes == ("BIB",)
            else "maximize_joint_action_recall_on_PR_frontier_at_or_above_descriptive_C0_action_precision"
        )
        if (
            calibration.get("schema_version")
            != "academic-structure-shared-calibration-v1"
            or calibration.get("rule") != expected_rule
            or calibration.get("active_classes") != list(active_classes)
            or calibration.get("reference", {}).get("action_precision")
            != c0_reference_precision
            or [row.get("deletion_bias") for row in calibration.get("grid", [])]
            != [float(value) for value in config["calibration"]["deletion_bias_grid"]]
            or calibration.get("selected") not in calibration.get("pareto_frontier", [])
        ):
            raise LadderError(
                "arm calibration grid/frontier differs from the shared rule"
            )

    from .feature_crf import (
        LinearChainCRF,
        calibrate_deletion_bias,
        make_examples,
        predict_documents,
    )
    from .features import FeatureEncoder

    for architecture_id, model_role, prediction_role, receipt in (
        (
            "c1-feature-bioes-crf",
            "c1_model",
            "c1_validation_predictions",
            c1_receipt,
        ),
        (
            "c2-char-ngram-feature-bioes-crf",
            "c2_model",
            "c2_validation_predictions",
            c2_receipt,
        ),
    ):
        if (
            receipt.get("schema_version")
            != "academic-structure-feature-crf-training-v2"
            or receipt.get("status")
            != "passed_cpu_fit_checkpoint_reload_and_validation_prediction"
            or receipt.get("architecture_id") != architecture_id
            or receipt.get("production_eligible") is not False
            or receipt.get("target") != target_name(active_classes)
            or receipt.get("active_classes") != list(active_classes)
            or receipt.get("effective_seed") != int(config["execution"]["seed"])
            or receipt.get("inputs")
            != {
                **expected_common_inputs,
                "reference_predictions_sha256": sha256_file(
                    resolved["c0_validation_predictions"]
                ),
            }
            or receipt.get("outputs", {}).get("model_sha256")
            != sha256_file(resolved[model_role])
            or receipt.get("outputs", {}).get("validation_predictions_sha256")
            != sha256_file(resolved[prediction_role])
        ):
            raise LadderError(
                f"{architecture_id}: per-arm receipt fails final validation"
            )
        validate_runtime(receipt, seed=int(config["execution"]["seed"]))
        validate_calibration(receipt)
        model, metadata = LinearChainCRF.load(resolved[model_role])
        architecture = next(
            row for row in config["architecture_ladder"] if row["id"] == architecture_id
        )
        encoder = FeatureEncoder(
            char_hash_dim=int(architecture.get("char_hash_dim", 0)),
            char_ngram_min=int(architecture.get("char_ngram_min", 2)),
            char_ngram_max=int(architecture.get("char_ngram_max", 5)),
        )
        if (
            metadata.get("architecture_id") != architecture_id
            or metadata.get("silver_sha256") != expected_selection_sha
            or metadata.get("split_manifest_sha256") != expected_manifest_sha
            or metadata.get("validation_silver_sha256") != expected_validation_sha
            or metadata.get("selection_receipt_sha256")
            != expected_selection_receipt_sha
            or metadata.get("config_sha256") != config_sha
            or metadata.get("reference_predictions_sha256")
            != sha256_file(resolved["c0_validation_predictions"])
            or metadata.get("active_classes") != list(active_classes)
            or metadata.get("test_used_for_training_or_calibration") is not False
            or metadata.get("production_eligible") is not False
            or model.n_features != encoder.n_features
            or metadata.get("calibration") != receipt.get("calibration")
            or metadata.get("deletion_bias")
            != receipt.get("calibration", {}).get("selected", {}).get("deletion_bias")
        ):
            raise LadderError(
                f"{architecture_id}: model metadata fails the receipt contract"
            )
        recomputed_calibration = calibrate_deletion_bias(
            make_examples(validation, encoder),
            model,
            config["calibration"]["deletion_bias_grid"],
            reference_action_precision=c0_reference_precision,
            active_classes=active_classes,
        )
        if recomputed_calibration != receipt.get("calibration"):
            raise LadderError(
                f"{architecture_id}: calibration does not reproduce from model"
            )
        recomputed = predict_documents(
            validation,
            encoder,
            model,
            deletion_bias=float(metadata["deletion_bias"]),
        )
        if read_predictions(resolved[prediction_role], validation) != recomputed:
            raise LadderError(
                f"{architecture_id}: predictions do not reproduce from model"
            )

    from .char_tcn_crf import count_neural_sequences, validate_n1_profile_receipt

    try:
        validate_n1_profile_receipt(
            profile_receipt,
            config,
            expected_inputs={
                **expected_common_inputs,
                "source_rehydration_receipt_sha256": expected_source_receipt_sha,
            },
            expected_uenv=uenv,
            expected_code_commit=code_commit,
            current_runtime=current_runtime,
            expected_counts={
                "train_documents": sum(
                    document.split == "train" for document in documents
                ),
                "validation_documents_contract_checked_not_scored_by_profile": len(
                    validation
                ),
                "train_sequences": count_neural_sequences(
                    [document for document in documents if document.split == "train"]
                ),
            },
        )
    except RuntimeError as error:
        raise LadderError("N1 profile receipt fails final validation") from error
    if (
        n1_receipt.get("schema_version") != "academic-structure-n1-training-v2"
        or n1_receipt.get("status")
        != "passed_cpu_fit_checkpoint_reload_and_validation_prediction"
        or n1_receipt.get("architecture_id") != "n1-bytecnn-tcn-masked-crf"
        or n1_receipt.get("inputs")
        != {
            **expected_common_inputs,
            "reference_predictions_sha256": sha256_file(
                resolved["c0_validation_predictions"]
            ),
            "profile_receipt_sha256": sha256_file(resolved["n1_profile_receipt"]),
        }
        or n1_receipt.get("outputs", {}).get("model_sha256")
        != sha256_file(resolved["n1_model"])
        or n1_receipt.get("outputs", {}).get("validation_predictions_sha256")
        != sha256_file(resolved["n1_validation_predictions"])
        or n1_receipt.get("historically_named_test_partition", {}).get(
            "documents_loaded"
        )
        != 0
        or n1_receipt.get("production_eligible") is not False
        or n1_receipt.get("target") != target_name(active_classes)
        or n1_receipt.get("active_classes") != list(active_classes)
        or n1_receipt.get("execution", {}).get("code_commit") != code_commit
    ):
        raise LadderError("N1 training metadata fails the receipt contract")
    validate_runtime(n1_receipt, seed=int(config["execution"]["seed"]))
    validate_calibration(n1_receipt)

    from .char_tcn_crf import (
        _cache_emissions,
        _predictions_from_emissions,
        calibrate_n1,
        load_n1_checkpoint,
        make_neural_examples,
    )

    n1_model, n1_checkpoint = load_n1_checkpoint(resolved["n1_model"], config)
    if n1_checkpoint["inputs"] != n1_receipt["inputs"]:
        raise LadderError("N1 checkpoint input bindings fail final validation")
    n1_encoder = FeatureEncoder(char_hash_dim=0)
    n1_architecture = n1_checkpoint["architecture"]
    n1_examples = make_neural_examples(
        validation,
        n1_encoder,
        max_bytes=int(n1_architecture["max_utf8_bytes_per_line"]),
    )
    n1_emissions = _cache_emissions(
        n1_model, n1_examples, batch_size=int(n1_architecture["batch_size"])
    )
    n1_recomputed_calibration = calibrate_n1(
        validation,
        n1_examples,
        n1_emissions,
        n1_model,
        config["calibration"]["deletion_bias_grid"],
        reference_action_precision=c0_reference_precision,
        active_classes=active_classes,
    )
    if (
        n1_recomputed_calibration != n1_receipt.get("calibration")
        or n1_checkpoint.get("deletion_bias")
        != n1_recomputed_calibration["selected"]["deletion_bias"]
        or n1_receipt.get("deletion_bias")
        != n1_recomputed_calibration["selected"]["deletion_bias"]
    ):
        raise LadderError(
            "N1 calibration does not reproduce from the strict checkpoint"
        )
    n1_recomputed = _predictions_from_emissions(
        validation,
        n1_examples,
        n1_emissions,
        n1_model,
        deletion_bias=float(n1_checkpoint["deletion_bias"]),
    )
    if (
        read_predictions(resolved["n1_validation_predictions"], validation)
        != n1_recomputed
    ):
        raise LadderError("N1 predictions do not reproduce from the strict checkpoint")

    expected_predictions = {
        "c0-rust-lr-hysteresis": sha256_file(resolved["c0_validation_predictions"]),
        "c1-feature-bioes-crf": sha256_file(resolved["c1_validation_predictions"]),
        "c2-char-ngram-feature-bioes-crf": sha256_file(
            resolved["c2_validation_predictions"]
        ),
        "n1-bytecnn-tcn-masked-crf": sha256_file(resolved["n1_validation_predictions"]),
    }
    if validation_report.get("prediction_sha256") != expected_predictions:
        raise LadderError(
            "validation report prediction bindings differ from run artifacts"
        )
    expected_arm_receipts = {
        "c0-rust-lr-hysteresis": sha256_file(resolved["c0_receipt"]),
        "c1-feature-bioes-crf": sha256_file(resolved["c1_receipt"]),
        "c2-char-ngram-feature-bioes-crf": sha256_file(resolved["c2_receipt"]),
        "n1-bytecnn-tcn-masked-crf": sha256_file(resolved["n1_train_receipt"]),
    }
    if (
        validation_report.get("schema_version") != VALIDATION_REPORT_SCHEMA
        or validation_report.get("status") != "sealed_retrospective_comparison_complete"
        or validation_report.get("production_eligible") is not False
        or validation_report.get("target") != target_name(active_classes)
        or validation_report.get("active_classes") != list(active_classes)
        or validation_report.get("config_sha256") != config_sha
        or validation_report.get("selection_receipt_sha256")
        != expected_selection_receipt_sha
        or validation_report.get("validation_silver_sha256") != expected_validation_sha
        or validation_report.get("source_rehydration_receipt_sha256")
        != sha256_file(source_rehydration_receipt_path)
        or validation_report.get("arm_receipt_sha256") != expected_arm_receipts
        or validation_report.get("historically_named_test_partition", {}).get(
            "documents_loaded_by_model_or_validation_processes"
        )
        != 0
    ):
        raise LadderError("validation report status/provenance fails final validation")
    with tempfile.TemporaryDirectory(prefix="bib-ladder-final-report-") as directory:
        recomputed_report = evaluate_validation(
            selection_silver_path=resolved["selection_silver"],
            selection_manifest_path=resolved["selection_manifest"],
            validation_silver_path=resolved["validation_silver"],
            selection_receipt_path=resolved["selection_receipt"],
            config_path=config_path,
            baseline_path=resolved["c0_validation_predictions"],
            candidate_paths={
                "c1-feature-bioes-crf": resolved["c1_validation_predictions"],
                "c2-char-ngram-feature-bioes-crf": resolved[
                    "c2_validation_predictions"
                ],
                "n1-bytecnn-tcn-masked-crf": resolved["n1_validation_predictions"],
            },
            arm_receipt_paths={
                "c0-rust-lr-hysteresis": resolved["c0_receipt"],
                "c1-feature-bioes-crf": resolved["c1_receipt"],
                "c2-char-ngram-feature-bioes-crf": resolved["c2_receipt"],
                "n1-bytecnn-tcn-masked-crf": resolved["n1_train_receipt"],
            },
            output_path=Path(directory) / "report.json",
        )
    if recomputed_report != validation_report:
        raise LadderError(
            "validation report content does not reproduce from bound predictions"
        )

    artifact_rows = []
    for role, path in sorted(resolved.items()):
        artifact_rows.append(
            {
                "role": role,
                "relative_path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    receipt = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "status": "passed_cpu_sealed_retrospective_comparison",
        "architecture_ids": [
            "c0-rust-lr-hysteresis",
            *EXPECTED_CANDIDATES,
        ],
        "target": target_name(active_classes),
        "active_classes": list(active_classes),
        "evidence_tier": "LLM_silver",
        "production_eligible": False,
        "execution": {
            **current_runtime,
            "accelerator_requested": False,
            "code_commit": code_commit,
            "slurm_job_id": str(job_id),
        },
        "source": selection_receipt["source"],
        "config_sha256": config_sha,
        "selection_receipt_sha256": sha256_file(resolved["selection_receipt"]),
        "validation_report_sha256": sha256_file(resolved["validation_report"]),
        "historically_named_test_partition": {
            "documents_loaded_by_model_or_validation_processes": 0,
            "predictions_written": 0,
            "semantics": "sealed_retrospective_comparison_not_unbiased_test",
        },
        "artifacts": artifact_rows,
        "artifact_inventory_sha256": canonical_json_sha256(artifact_rows),
        "decision": "LLM_silver_replay_only_no_automatic_selection_no_production_change",
        "human_annotation": {
            "required": False,
            "campaign_planned": False,
            "future_independent_option": "newly sampled LLM-silver documents",
        },
        "resource_gate_note": (
            "config deployment resource limits are promotion-only; per-arm wall/RSS are recorded"
        ),
    }
    _atomic_json(output_path, receipt)
    return receipt


def verify_published_run(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    receipt_path = root / "run.receipt.json"
    if root.is_symlink() or not root.is_dir() or receipt_path.is_symlink():
        raise LadderError("published run root/receipt must be real paths")
    receipt = _load_json(receipt_path)
    artifact_rows = receipt.get("artifacts")
    expected_architectures = ["c0-rust-lr-hysteresis", *EXPECTED_CANDIDATES]
    if (
        receipt.get("schema_version") != RUN_RECEIPT_SCHEMA
        or receipt.get("status") != "passed_cpu_sealed_retrospective_comparison"
        or receipt.get("production_eligible") is not False
        or receipt.get("architecture_ids") != expected_architectures
        or receipt.get("active_classes") not in (["BIB"], ["BIB", "TOC"])
        or receipt.get("target") != target_name(receipt.get("active_classes", []))
        or receipt.get("decision")
        != "LLM_silver_replay_only_no_automatic_selection_no_production_change"
        or receipt.get("human_annotation")
        != {
            "required": False,
            "campaign_planned": False,
            "future_independent_option": "newly sampled LLM-silver documents",
        }
        or not isinstance(artifact_rows, list)
    ):
        raise LadderError("published run receipt status is invalid")
    roles = [row.get("role") for row in artifact_rows if isinstance(row, Mapping)]
    relative_paths = [
        row.get("relative_path") for row in artifact_rows if isinstance(row, Mapping)
    ]
    if (
        len(artifact_rows) != len(EXPECTED_ARTIFACT_ROLES)
        or len(roles) != len(artifact_rows)
        or set(roles) != EXPECTED_ARTIFACT_ROLES
        or len(set(roles)) != len(roles)
        or len(relative_paths) != len(artifact_rows)
        or any(not isinstance(value, str) or not value for value in relative_paths)
        or len(set(relative_paths)) != len(relative_paths)
        or receipt.get("artifact_inventory_sha256")
        != canonical_json_sha256(artifact_rows)
    ):
        raise LadderError("published run artifact inventory is invalid")
    expected = {receipt_path.resolve()}
    for row in artifact_rows:
        relative_path = Path(row["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise LadderError("published artifact path escapes the run root")
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise LadderError("published artifact is absent or a symlink")
        if (
            not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or not isinstance(row.get("bytes"), int)
            or isinstance(row.get("bytes"), bool)
            or row["bytes"] < 0
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            raise LadderError("published artifact differs from the run receipt")
        expected.add(path.resolve())
    discovered = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LadderError("published run contains a symlink")
        if path.is_file():
            discovered.add(path.resolve())
    if discovered != expected:
        raise LadderError("published run contains an unreceipted or missing file")
    return {
        "schema_version": "academic-structure-bib-ladder-publication-check-v1",
        "status": "pass",
        "run_receipt_sha256": sha256_file(receipt_path),
        "artifact_count": len(expected) - 1,
        "production_eligible": False,
    }


def _candidate_map(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise LadderError(f"invalid candidate mapping {value!r}")
        result[name] = path
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-selection")
    prepare.add_argument("--silver", required=True)
    prepare.add_argument("--split-manifest", required=True)
    prepare.add_argument("--rehydration-receipt", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--selection-silver", required=True)
    prepare.add_argument("--selection-manifest", required=True)
    prepare.add_argument("--validation-silver", required=True)
    prepare.add_argument("--receipt", required=True)

    verify = sub.add_parser("verify-selection")
    verify.add_argument("--selection-silver", required=True)
    verify.add_argument("--selection-manifest", required=True)
    verify.add_argument("--validation-silver", required=True)
    verify.add_argument("--selection-receipt", required=True)
    verify.add_argument("--config", required=True)

    compare = sub.add_parser("evaluate-validation")
    compare.add_argument("--selection-silver", required=True)
    compare.add_argument("--selection-manifest", required=True)
    compare.add_argument("--validation-silver", required=True)
    compare.add_argument("--selection-receipt", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", action="append", default=[])
    compare.add_argument("--arm-receipt", action="append", default=[])
    compare.add_argument("--output", required=True)

    final = sub.add_parser("finalize-run")
    final.add_argument("--run-root", required=True)
    final.add_argument("--artifact", action="append", default=[])
    final.add_argument("--config", required=True)
    final.add_argument("--source-rehydration-receipt", required=True)
    final.add_argument("--uenv", required=True)
    final.add_argument("--code-commit", required=True)
    final.add_argument("--job-id", required=True)
    final.add_argument("--output", required=True)

    published = sub.add_parser("verify-published-run")
    published.add_argument("--run-root", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare-selection":
        receipt = prepare_selection(
            silver_path=args.silver,
            split_manifest_path=args.split_manifest,
            rehydration_receipt_path=args.rehydration_receipt,
            config_path=args.config,
            selection_silver_path=args.selection_silver,
            selection_manifest_path=args.selection_manifest,
            validation_silver_path=args.validation_silver,
            receipt_path=args.receipt,
        )
    elif args.command == "verify-selection":
        documents, validation, selection = verify_selection_bundle(
            selection_silver_path=args.selection_silver,
            selection_manifest_path=args.selection_manifest,
            validation_silver_path=args.validation_silver,
            selection_receipt_path=args.selection_receipt,
            config_path=args.config,
        )
        receipt = {
            "schema_version": "academic-structure-bib-selection-verification-v1",
            "status": "pass",
            "selection_receipt_sha256": sha256_file(args.selection_receipt),
            "train_documents": sum(doc.split == "train" for doc in documents),
            "validation_documents": len(validation),
            "historically_named_test_documents_loaded": 0,
            "source_rehydration_receipt_sha256": selection["source"][
                "rehydration_receipt_sha256"
            ],
        }
    elif args.command == "evaluate-validation":
        receipt = evaluate_validation(
            selection_silver_path=args.selection_silver,
            selection_manifest_path=args.selection_manifest,
            validation_silver_path=args.validation_silver,
            selection_receipt_path=args.selection_receipt,
            config_path=args.config,
            baseline_path=args.baseline,
            candidate_paths=_candidate_map(args.candidate),
            arm_receipt_paths=_candidate_map(args.arm_receipt),
            output_path=args.output,
        )
    elif args.command == "finalize-run":
        receipt = finalize_run(
            run_root=args.run_root,
            artifacts=_parse_artifacts(args.artifact),
            config_path=args.config,
            source_rehydration_receipt_path=args.source_rehydration_receipt,
            uenv=args.uenv,
            code_commit=args.code_commit,
            job_id=args.job_id,
            output_path=args.output,
        )
    else:
        receipt = verify_published_run(args.run_root)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
