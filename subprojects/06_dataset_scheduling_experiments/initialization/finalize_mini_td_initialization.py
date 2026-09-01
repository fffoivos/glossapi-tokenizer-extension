#!/usr/bin/env python3
"""Freeze the production Mini tied-TD initialization receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


SLICES = ("hplt", "non_hplt", "polytonic")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_tree(root: Path, rows: list[dict]) -> None:
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"tree drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-td-dir", type=Path, required=True)
    parser.add_argument("--pilot-selection", type=Path, required=True)
    parser.add_argument("--conversion-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.full_td_dir / "tied_td_manifest.json"
    verification_path = args.full_td_dir / "initialization_verification.json"
    inventory_path = args.full_td_dir / "full_added_token_ids_receipt.json"
    manifest = read_json(manifest_path)
    verification = read_json(verification_path)
    selection = read_json(args.pilot_selection)
    inventory = read_json(inventory_path)
    conversion = read_json(args.conversion_receipt)
    if (
        manifest.get("status") != "completed"
        or manifest.get("scope") != "full"
        or int(manifest.get("requested_token_count", -1)) != 17_920
        or float(manifest.get("trained_token_fraction", 0.0)) < 0.90
        or int(manifest.get("min_accepted_snippets_per_token", -1)) != 25
        or manifest.get("input_output_share_storage") is not True
        or manifest.get("norm_collapse_gate_passed") is not True
    ):
        raise ValueError("full TD manifest failed")
    if (
        selection.get("status") != "selected"
        or selection.get("fvt_macro_non_regression_gate_passed") is not True
        or float(selection.get("selected_delta_macro_bpb_vs_fvt", 1.0)) > 0.0
        or manifest.get("target_layer") != selection.get("selected_target_layer")
        or manifest.get("loss_profile") != selection.get("selected_loss_profile")
    ):
        raise ValueError("full TD recipe is not the selected pilot recipe")
    if verification.get("status") != "pass" or not all(verification.get("checks", {}).values()):
        raise ValueError("full TD structural/collapse verification failed")
    if (
        inventory.get("status") != "frozen"
        or int(inventory.get("count", -1)) != 17_920
        or inventory.get("ordered_and_contiguous") is not True
    ):
        raise ValueError("full added-token inventory failed")
    inventory_ids = Path(inventory["path"])
    if (
        not inventory_ids.is_file()
        or inventory_ids.stat().st_size != int(inventory["bytes"])
        or sha256_file(inventory_ids) != inventory["sha256"]
        or manifest.get("token_ids_file_sha256") != inventory["sha256"]
    ):
        raise ValueError("full added-token inventory payload drift")
    if conversion.get("status") != "passed" or not all(conversion.get("checks", {}).values()):
        raise ValueError("Megatron conversion/roundtrip failed")
    initial_root = Path(conversion["initial_checkpoint_root"])
    if conversion.get("schema_version") != "apertus_mini_td_megatron_conversion_v2":
        raise ValueError("Megatron conversion receipt is not release-checkpoint schema v2")
    latest = initial_root / "latest_checkpointed_iteration.txt"
    if not latest.is_file() or latest.read_text().strip() != "release":
        raise ValueError("Megatron initialization tracker is not release")
    verify_tree(initial_root / "release", conversion["release_tree"])
    metrics = {}
    for name in SLICES:
        path = args.full_td_dir / "metrics" / name / "tokenizer_fair_metrics.json"
        value = read_json(path).get("global", {}).get("bpb_bits_per_byte")
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise ValueError(f"invalid full TD {name} BPB")
        metrics[name] = {"bpb": float(value), "receipt": file_receipt(path)}
    payload = {
        "schema_version": "apertus_mini_tied_td_initialization_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": "canonical FVT pre-initialization followed by selected tied-embedding Token Distillation recipe",
        "selected_pilot_id": selection["selected_pilot_id"],
        "target_layer": selection["selected_target_layer"],
        "loss_profile": selection["selected_loss_profile"],
        "fvt_baseline_macro_bpb": float(selection["fvt_baseline"]["macro_bpb"]),
        "selected_pilot_macro_bpb": float(selection["selected_macro_bpb"]),
        "selected_delta_macro_bpb_vs_fvt": float(
            selection["selected_delta_macro_bpb_vs_fvt"]
        ),
        "fvt_macro_non_regression_gate_passed": True,
        "requested_rows": 17_920,
        "trained_rows": int(manifest["trained_token_count"]),
        "trained_fraction": float(manifest["trained_token_fraction"]),
        "low_coverage_rows_retain_exact_fvt": True,
        "input_output_embeddings_tied": True,
        "base_rows_bitwise_preserved": True,
        "initial_checkpoint_root": str(initial_root.resolve()),
        "heldout_bpb": metrics,
        "evidence": {
            "full_td_manifest": file_receipt(manifest_path),
            "full_td_verification": file_receipt(verification_path),
            "full_token_inventory": file_receipt(inventory_path),
            "pilot_selection": file_receipt(args.pilot_selection),
            "conversion": file_receipt(args.conversion_receipt),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "initial_checkpoint_root": str(initial_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
