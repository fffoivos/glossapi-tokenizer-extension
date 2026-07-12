#!/usr/bin/env python3
"""Receipt-oriented validation-only deterministic/hybrid ablation evaluation.

This runner consumes only paths named explicitly on the command line.  It does
not discover data, fit a model, mutate a corpus, or read a sealed/test split.
The resulting metrics measure agreement with imported LLM-silver labels; they
are not human-gold accuracy or authorization for production removal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import GoldDocument, read_gold, sha256_file, validate_silver
from .deterministic_adapter import (
    AblationMode,
    build_prediction_rows,
    read_base_predictions,
    write_prediction_rows,
)
from .deterministic_structure import DECODER_ID, RULES_ID
from .evaluate import (
    _mark_silver_safety_unavailable,
    evaluate,
    read_predictions,
)


REPORT_SCHEMA = "academic-structure-deterministic-ablation-evaluation-v1"
RUN_MODES = (
    AblationMode.RULES_ONLY,
    AblationMode.BASE_PLUS_RULES,
    AblationMode.BASE_RULES_VETO,
    AblationMode.BASE_PLUS_RULES_VETO,
)
ALLOWED_DEVELOPMENT_SPLITS = frozenset(("train", "validation"))
FORBIDDEN_SPLIT_NAMES = frozenset(
    (
        "test",
        "historical_test",
        "sealed_test",
        "test_sealed",
        "held_out_test",
        "holdout",
        "heldout",
        "final_eval",
        "final_evaluation",
        "retrospective_test",
    )
)


class AblationRunError(ValueError):
    """Raised before output when the comparison contract is not satisfied."""


class SealedDataError(AblationRunError):
    """Raised when any explicit input names a sealed/test-like partition."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationRunError(message)


def _normalised_split(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _is_forbidden_split(value: object) -> bool:
    name = _normalised_split(value)
    if not name:
        return False
    parts = set(name.split("_"))
    return (
        name in FORBIDDEN_SPLIT_NAMES
        or "test" in parts
        or "sealed" in parts
        or name.startswith("sealed")
        or name.endswith("sealed")
    )


def validate_allowed_split(value: str) -> str:
    """Validate the requested split before any input document is materialised."""

    name = _normalised_split(value)
    if _is_forbidden_split(name):
        raise SealedDataError(f"sealed/test split {value!r} is forbidden")
    _require(
        name in ALLOWED_DEVELOPMENT_SPLITS,
        "allowed split must be one of: train, validation",
    )
    return name


def _read_json_object(path: str | Path, description: str) -> Mapping[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), f"{description} must be a JSON object")
    return raw


def preflight_jsonl_splits(path: str | Path, description: str) -> int:
    """Reject sealed aliases while rows are still raw mappings, before GoldLine."""

    row_count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            _require(
                isinstance(raw, Mapping),
                f"{description} row {row_number}: expected object",
            )
            split = raw.get("split")
            if _is_forbidden_split(split):
                raise SealedDataError(
                    f"{description} row {row_number}: sealed/test split {split!r} is forbidden"
                )
            row_count += 1
    _require(row_count > 0, f"{description} is empty")
    return row_count


def prediction_model_ids(path: str | Path) -> list[str]:
    model_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            _require(
                isinstance(raw, Mapping),
                f"base C0 row {row_number}: expected object",
            )
            model_id = raw.get("model_id")
            _require(
                isinstance(model_id, str) and bool(model_id),
                f"base C0 row {row_number}: model_id is required",
            )
            model_ids.add(model_id)
    _require(bool(model_ids), "base C0 model inventory is empty")
    return sorted(model_ids)


def preflight_manifest_splits(manifest: Mapping[str, Any]) -> None:
    assignments = manifest.get("assignments")
    _require(isinstance(assignments, Mapping), "split manifest lacks assignments")
    for document_id, split in assignments.items():
        _require(
            isinstance(document_id, str) and bool(document_id),
            "split manifest has an invalid document identity",
        )
        if _is_forbidden_split(split):
            raise SealedDataError(
                f"split manifest document {document_id!r}: sealed/test split {split!r} is forbidden"
            )


def _input_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _git_revision(anchor: Path) -> dict[str, Any]:
    common = ["git", "-C", str(anchor)]
    try:
        root = subprocess.run(
            [*common, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            [*common, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [*common, "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_available": False,
            "clean": None,
            "commit": None,
            "reason": "git_metadata_unavailable",
        }
    clean = not bool(status.strip())
    return {
        "git_available": True,
        "repository_root": root,
        "clean": clean,
        "commit": head if clean else None,
        "reason": None if clean else "worktree_not_clean",
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable report {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _metric_options(config: Mapping[str, Any]) -> dict[str, int | float]:
    gates = config.get("deployment_gates")
    _require(isinstance(gates, Mapping), "config lacks deployment_gates")
    try:
        return {
            "maximum_false_fraction": float(
                gates["maximum_false_deletion_fraction_per_document"]
            ),
            "maximum_contiguous_false_tokens": int(
                gates["maximum_contiguous_false_deletion_tokens"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AblationRunError("config has invalid evaluation gates") from error


def _silver_metrics(
    documents: Sequence[GoldDocument],
    predictions: Mapping[str, Sequence[str]],
    *,
    split: str,
    metric_options: Mapping[str, int | float],
) -> dict[str, Any]:
    metrics, _per_document = evaluate(
        documents,
        predictions,
        split=split,
        maximum_false_fraction=float(metric_options["maximum_false_fraction"]),
        maximum_contiguous_false_tokens=int(
            metric_options["maximum_contiguous_false_tokens"]
        ),
    )
    _mark_silver_safety_unavailable(metrics)
    return metrics


def run_ablation_evaluation(
    *,
    silver_path: str | Path,
    split_manifest_path: str | Path,
    base_c0_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    allowed_split: str = "validation",
) -> dict[str, Any]:
    """Run all four deterministic ablations and create one immutable report."""

    selected_split = validate_allowed_split(allowed_split)
    paths = {
        "base_c0_predictions": Path(base_c0_path),
        "config": Path(config_path),
        "imported_llm_silver": Path(silver_path),
        "split_manifest": Path(split_manifest_path),
    }
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable output directory {destination}"
        )

    # These checks precede read_gold/read_base_predictions, so no GoldLine or
    # adapter prediction line is materialised when a forbidden alias is present.
    preflight_jsonl_splits(paths["imported_llm_silver"], "LLM-silver input")
    manifest = _read_json_object(paths["split_manifest"], "split manifest")
    preflight_manifest_splits(manifest)
    preflight_jsonl_splits(paths["base_c0_predictions"], "base C0 predictions")
    base_model_ids = prediction_model_ids(paths["base_c0_predictions"])
    initial_inputs = _input_hashes(paths)

    config = _read_json_object(paths["config"], "config")
    silver_contract = config.get("silver_contract")
    _require(isinstance(silver_contract, Mapping), "config lacks silver_contract")
    _require(
        silver_contract.get("comparison_only") is True
        and silver_contract.get("production_eligible") is False,
        "config must explicitly mark LLM-silver as comparison-only/non-production",
    )
    metric_options = _metric_options(config)

    documents = read_gold(paths["imported_llm_silver"])
    contract_receipt = validate_silver(
        documents, silver_contract, split_manifest=manifest
    )
    selected = [document for document in documents if document.split == selected_split]
    _require(bool(selected), f"no documents in allowed split {selected_split!r}")
    _require(
        all(document.split == selected_split for document in selected),
        "internal split selection failure",
    )

    base_predictions = read_base_predictions(paths["base_c0_predictions"], selected)
    base_metrics = _silver_metrics(
        selected,
        base_predictions,
        split=selected_split,
        metric_options=metric_options,
    )

    rows_by_mode: dict[AblationMode, list[dict[str, Any]]] = {}
    for mode in RUN_MODES:
        base = None if mode is AblationMode.RULES_ONLY else base_predictions
        rows_by_mode[mode] = build_prediction_rows(selected, mode, base)

    # Hash once after all reads as well as before them: an input changing during
    # evaluation invalidates the run rather than producing a misleading receipt.
    inputs = _input_hashes(paths)
    _require(initial_inputs == inputs, "an input changed during evaluation")
    revision = _git_revision(Path(__file__).resolve().parent)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    output_receipts: dict[str, dict[str, Any]] = {}
    mode_reports: dict[str, dict[str, Any]] = {}
    for mode in RUN_MODES:
        filename = f"{mode.value}.predictions.jsonl"
        path = destination / filename
        write_prediction_rows(path, rows_by_mode[mode])
        predictions = read_predictions(path, selected)
        metrics = _silver_metrics(
            selected,
            predictions,
            split=selected_split,
            metric_options=metric_options,
        )
        output_receipts[mode.value] = {
            "path": filename,
            "sha256": sha256_file(path),
            "document_count": len(rows_by_mode[mode]),
        }
        mode_reports[mode.value] = {
            "evidence_tier": "LLM_silver",
            "metrics": metrics,
            "output": output_receipts[mode.value],
        }

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "evidence_tier": "LLM_silver",
        "allowed_split": selected_split,
        "document_count": len(selected),
        "source_counts": {
            source: sum(document.source == source for document in selected)
            for source in sorted({document.source for document in selected})
        },
        "inputs": inputs,
        "outputs": output_receipts,
        "code_revision": revision,
        "silver_contract_receipt": contract_receipt,
        "base_c0": {
            "role": "explicit_input_baseline",
            "model_ids": base_model_ids,
            "evidence_tier": "LLM_silver",
            "metrics": base_metrics,
        },
        "modes": mode_reports,
        "component_ids": {
            "deterministic_rules": RULES_ID,
            "deterministic_decoder": DECODER_ID,
        },
        "execution_claims": {
            "model_fitting_performed": False,
            "data_discovery_performed": False,
            "corpus_mutation_performed": False,
            "sealed_or_test_data_accessed": False,
            "human_gold_used": False,
            "production_eligible": False,
            "production_action_authorized": False,
        },
        "caveats": [
            "All metrics are agreement with imported LLM-silver labels, not human-gold accuracy.",
            "LLM-silver ablations are comparison-only and cannot authorize production removal.",
            "Independent running-prose safety metrics are unavailable and are reported as null.",
            "No model was fit and no corpus document was changed by this runner.",
        ],
    }
    _write_immutable_json(destination / "ablation.report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--base-c0", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-split", default="validation")
    args = parser.parse_args(argv)
    report = run_ablation_evaluation(
        silver_path=args.silver,
        split_manifest_path=args.split_manifest,
        base_c0_path=args.base_c0,
        config_path=args.config,
        output_dir=args.output_dir,
        allowed_split=args.allowed_split,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "allowed_split": report["allowed_split"],
                "document_count": report["document_count"],
                "production_eligible": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
