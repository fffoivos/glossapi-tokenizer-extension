#!/usr/bin/env python3
"""Inventory and semantically validate surviving assets for the matched study.

This tool is read-only except for its immutable JSON receipt. Directories are
not recursively hashed. Critical immutable files are hashed when the asset
spec explicitly asks for it, including the published overlap table.
"""

from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import os
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, read_json, require, sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-spec", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def count_files(root: Path) -> int:
    count = 0
    for _, _, files in os.walk(root):
        count += len(files)
    return count


def decimal_product_equal(left: object, first: object, second: object) -> bool:
    """Compare frozen decimal recipe values without binary-float artifacts."""
    return Decimal(str(left)) == Decimal(str(first)) * Decimal(str(second))


def inspect_asset(name: str, row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    kind = row["kind"]
    expected_missing = bool(row.get("expected_missing", False))
    exists = path.exists()
    if expected_missing:
        require(not exists, f"{name}: expected missing asset exists: {path}")
        return {
            "path": str(path),
            "kind": kind,
            "exists": False,
            "expected_missing": True,
            "status": "passed",
        }
    if bool(row.get("required", False)):
        require(exists, f"{name}: required asset missing: {path}")
    if not exists:
        return {"path": str(path), "kind": kind, "exists": False, "status": "optional_missing"}
    require(kind in {"file", "directory"}, f"{name}: invalid kind {kind}")
    require(path.is_file() if kind == "file" else path.is_dir(), f"{name}: kind mismatch: {path}")
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "kind": kind,
        "exists": True,
        "status": "passed",
    }
    stat = path.stat()
    result.update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if kind == "directory" and "expected_file_count" in row:
        file_count = count_files(path)
        result["file_count"] = file_count
        require(file_count == int(row["expected_file_count"]), f"{name}: file-count drift")
    if kind == "file" and bool(row.get("hash", False)):
        digest = sha256_file(path)
        result["sha256"] = digest
        if "expected_sha256" in row:
            require(digest == row["expected_sha256"], f"{name}: SHA-256 drift")
    return result


def semantic_checks(assets: dict[str, dict[str, Any]], experiment: dict[str, Any]) -> dict[str, Any]:
    run = read_json(Path(assets["historical_run_metadata"]["path"]))
    target = experiment["historical_target"]
    training = experiment["training"]
    require(run["megatron_commit"] == target["training_code_revision"], "historical Megatron revision drift")
    require(run["data_seed"] == target["data_seed"], "historical data seed drift")
    require(run["curriculum_order_mode"] == target["curriculum_order_mode"], "historical order drift")
    require(int(run["megatron_gpt_dataset_no_shuffle"]) == target["megatron_gpt_dataset_no_shuffle"], "historical shuffle drift")
    require(run["train_iters"] == 3218 and run["train_tokens"] == 13_500_000_000, "historical horizon drift")
    require(run["global_batch_tokens"] == training["global_batch_token_slots"], "historical global tokens drift")
    require(str(run["lr_peak"]) == training["peak_lr_8b"], "historical peak LR drift")
    require(decimal_product_equal(run["lr_final"], training["peak_lr_8b"], training["terminal_lr_ratio"]), "historical final LR drift")
    require(run["rotary_base"] == training["rope_theta"], "historical RoPE theta drift")
    require(run["rope_scaling_factor"] == training["rope_scaling_factor"], "historical RoPE scale drift")
    require(run["make_vocab_size_divisible_by"] == experiment["tokenizer"]["make_vocab_size_divisible_by"], "historical vocab divisor drift")

    token_counts = read_json(Path(assets["release_token_counts"]["path"]))
    anonymization = read_json(Path(assets["anonymization_manifest"]["path"]))
    publication = read_json(Path(assets["release_publication_receipt"]["path"]))
    public = read_json(Path(assets["release_public_access_receipt"]["path"]))
    expected_dataset = experiment["data"]["source_dataset"]
    require(token_counts.get("status") == "passed", "release token count receipt is not passed")
    require(anonymization.get("counts", {}).get("rows") == expected_dataset["expected_rows"], "release row count drift")
    require(publication.get("status") == "passed" and public.get("status") == "passed", "release publication is not passed")
    publication_text = Path(assets["release_publication_receipt"]["path"]).read_text(encoding="utf-8")
    require(expected_dataset["revision"] in publication_text, "release revision absent from publication receipt")

    clean = read_json(Path(assets["greekmmlu_clean_manifest"]["path"]))
    require(clean.get("clean_count") == experiment["data"]["greekmmlu"]["clean_count"], "GreekMMLU clean-count drift")
    gate = read_json(Path(assets["native_suite_execution_gate"]["path"]))
    require(str(gate.get("status", "")).lower() in {"passed", "accepted", "completed"}, "native-suite gate is not passed")
    replay_acquisition = read_json(Path(assets["replay_acquisition_receipt"]["path"]))
    replay_contract = experiment["data"]["replay_reconstruction"]
    require(replay_acquisition.get("schema_version") == "full_cpt_replay_acquisition_receipt_v1", "replay acquisition schema drift")
    require(replay_acquisition.get("status") == "completed", "replay acquisition is not completed")
    require(replay_acquisition.get("output_count") == len(replay_acquisition.get("outputs", [])) == replay_contract["selected_files"], "replay acquisition file-count drift")
    selection = replay_acquisition.get("selection_plan", {})
    require(selection.get("policy") == replay_contract["acquisition_policy"], "replay acquisition policy drift")
    capacity = selection.get("capacity_sampling", {})
    require(capacity.get("seed") == replay_contract["acquisition_seed"], "replay acquisition seed drift")
    require(selection.get("selected_remote_bytes") == replay_contract["selected_remote_bytes"], "replay acquisition byte-count drift")

    return {
        "historical_run": {
            "megatron_commit": run["megatron_commit"],
            "data_seed": run["data_seed"],
            "train_iters": run["train_iters"],
            "train_tokens": run["train_tokens"],
            "curriculum_order_mode": run["curriculum_order_mode"],
        },
        "release": {
            "rows": anonymization["counts"]["rows"],
            "shards": anonymization["counts"]["shards"],
            "training_tokens": token_counts["totals"]["training_tokens"],
            "revision": expected_dataset["revision"],
            "anonymized": True,
        },
        "greekmmlu_clean_count": clean["clean_count"],
        "replay_reconstruction": {
            "receipt_sha256": assets["replay_acquisition_receipt"]["sha256"],
            "selected_files": replay_acquisition["output_count"],
            "selected_remote_bytes": selection["selected_remote_bytes"],
            "historical_document_identity_claimed": False,
            "named_difference": replay_contract["named_reconstruction_difference"],
        },
        "historical_payload_reconstruction_required": True,
        "additional_global_deduplication_allowed": False,
    }


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    spec = read_json(args.asset_spec)
    experiment = read_json(args.experiment)
    require(spec.get("schema_version") == "apertus_hard_h_to_g_asset_spec_v1", "asset spec schema drift")
    inventory = {
        name: inspect_asset(name, row)
        for name, row in spec["assets"].items()
    }
    semantics = semantic_checks(inventory, experiment)
    payload = {
        "schema_version": "apertus_hard_h_to_g_asset_inventory_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "assets": inventory,
        "semantic_checks": semantics,
        "read_only_inventory": True,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
