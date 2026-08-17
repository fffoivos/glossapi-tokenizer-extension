#!/usr/bin/env python3
"""Run and receipt all benchmark-clean document-local validation panels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import file_binding, read_json, require, write_json_atomic


def resolve_export(run_root: Path, *, scale: str, iteration: int) -> tuple[Path, Path]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in run_root.resolve().rglob("result.json"):
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            value.get("schema_version")
            == "apertus_hard_h_to_g_checkpoint_export_evaluation_v1"
            and value.get("status") == "completed"
            and value.get("scale") == scale
            and int(value.get("iteration", -1)) == iteration
        ):
            matches.append((path, value))
    require(
        len(matches) == 1,
        f"expected one completed checkpoint export for {scale}@{iteration}; found {len(matches)}",
    )
    _result_path, result = matches[0]
    export_receipt = Path(str(result["checkpoint_export"]["path"])).resolve()
    require(
        file_binding(export_receipt) == result["checkpoint_export"],
        "checkpoint export binding drift",
    )
    receipt = read_json(export_receipt)
    hf_root = Path(str(receipt["hf_export"]["path"])).resolve()
    require(
        hf_root.is_dir() and receipt.get("ready_for_frozen_evaluators") is True,
        "HF export is not evaluation-ready",
    )
    return hf_root, export_receipt


def validate_panels(
    manifest_path: Path, output: Path, *, model: Path
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    panels = manifest.get("panels")
    require(
        isinstance(panels, list) and len(panels) == 13,
        "validation manifest must contain 13 panels",
    )
    names = [str(row["name"]) for row in panels]
    require(len(set(names)) == 13, "validation panel names are not unique")
    results = []
    for row in panels:
        name = str(row["name"])
        receipt_path = output / f"{name}.receipt.json"
        documents_path = output / f"{name}.documents.jsonl"
        receipt = read_json(receipt_path)
        require(
            receipt.get("schema_version") == "apertus_per_document_validation_v1"
            and receipt.get("status") == "completed",
            f"panel did not complete: {name}",
        )
        require(
            Path(str(receipt.get("model", ""))).resolve() == model,
            f"panel model drift: {name}",
        )
        expected_input = row["raw_jsonl"]
        require(
            receipt.get("input")
            == {key: expected_input[key] for key in ("path", "bytes", "sha256")},
            f"panel input binding drift: {name}",
        )
        require(
            receipt.get("output")
            == file_binding(documents_path) | {"rows": receipt["output"]["rows"]},
            f"panel output binding drift: {name}",
        )
        aggregate = receipt.get("aggregate")
        require(
            isinstance(aggregate, dict) and int(aggregate.get("documents", 0)) > 0,
            f"empty panel: {name}",
        )
        results.append(
            {
                "name": name,
                "receipt": file_binding(receipt_path),
                "documents": file_binding(documents_path),
                "aggregate": aggregate,
            }
        )
    require(
        len(list(output.glob("*.receipt.json"))) == 13, "unexpected panel receipt count"
    )
    require(
        len(list(output.glob("*.documents.jsonl"))) == 13,
        "unexpected panel document count",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")
    hf_root, export_receipt = resolve_export(
        args.run_root, scale=args.scale, iteration=args.iteration
    )
    final_output = args.output / "panels"
    wrapper = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/run_offline_panels_1node_debug.sbatch"
    )
    require(wrapper.is_file(), "frozen panel wrapper missing")
    env = os.environ.copy()
    env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_VALIDATION_MANIFEST": str(args.validation_manifest.resolve()),
            "H2G_HF_MODEL": str(hf_root),
            "H2G_HF_TOKENIZER": str(args.tokenizer_dir.resolve()),
            "H2G_DOCVAL_OUTPUT": str(final_output),
            "H2G_DOCVAL_OUTPUT_FINAL": str(final_output),
        }
    )
    try:
        subprocess.run(["bash", str(wrapper)], env=env, check=True)
        panels = validate_panels(args.validation_manifest, final_output, model=hf_root)
        write_json_atomic(
            result_path,
            {
                "schema_version": "apertus_hard_h_to_g_offline_panels_evaluation_v1",
                "status": "completed",
                "campaign_id": args.campaign_id,
                "evaluator_id": args.evaluator_id,
                "iteration": args.iteration,
                "attempt": args.attempt,
                "contract_digest": args.contract_digest,
                "scale": args.scale,
                "checkpoint_export": file_binding(export_receipt),
                "validation_manifest": file_binding(args.validation_manifest),
                "panels": panels,
            },
        )
    except BaseException:
        if final_output.exists():
            shutil.rmtree(final_output)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
