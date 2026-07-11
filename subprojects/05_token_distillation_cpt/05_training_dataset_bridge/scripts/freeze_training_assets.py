#!/usr/bin/env python3
"""Freeze launch-time code, TD layer-11 init, and semantic evidence receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from bridge_common import (
    bound_code_sha,
    file_tree_receipt,
    read_json,
    sha256_file,
    utc_now,
    validate_file_tree_receipt,
    write_json_atomic,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _validate_existing(
    path: Path,
    *,
    bridge_sha: str,
    config_sha: str,
    implementation_sha: str,
) -> bool:
    if not path.is_file():
        return False
    value = read_json(path)
    if (
        value.get("schema_version") != "full_cpt_training_assets_receipt_v1"
        or value.get("status") != "completed"
        or value.get("bridge_manifest_sha256") != bridge_sha
        or value.get("config_sha256") != config_sha
        or value.get("implementation_sha256") != implementation_sha
    ):
        raise ValueError("existing training-assets receipt has different bindings")
    validate_file_tree_receipt(value["init_checkpoint"]["tree"])
    megatron_root = validate_file_tree_receipt(value["megatron"]["tree"])
    validate_file_tree_receipt(value["td_layer11_evidence"]["bundle_tree"])
    if _git(megatron_root, "rev-parse", "HEAD") != value["megatron"]["commit"]:
        raise ValueError("existing training-assets Megatron commit drift")
    if _git(megatron_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("existing training-assets Megatron tree is dirty")
    for label in ("trainer", "training_environment"):
        receipt = value[label]
        file_path = Path(str(receipt["path"]))
        if not file_path.is_file() or sha256_file(file_path) != receipt["sha256"]:
            raise ValueError(f"existing training asset drift: {label}")
    if not all(value["td_layer11_evidence"]["semantic_checks"].values()):
        raise ValueError("existing TD semantic evidence has failed checks")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--training-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge = read_json(args.bridge_manifest)
    config = read_json(args.config)
    if (
        bridge.get("schema_version") != "full_cpt_training_bridge_manifest_v1"
        or bridge.get("status") != "completed"
        or not all(bridge.get("invariants", {}).values())
    ):
        raise ValueError("training assets require a passed finalized bridge")
    if config.get("schema_version") != "full_cpt_training_bridge_config_v1":
        raise ValueError("unsupported training-bridge config")
    bridge_sha = sha256_file(args.bridge_manifest.resolve())
    config_sha = sha256_file(args.config.resolve())
    if bridge.get("config", {}).get("sha256") != config_sha:
        raise ValueError("bridge and launch config differ")
    input_receipt_path = Path(str(bridge["input_receipt"]["path"])).resolve()
    if sha256_file(input_receipt_path) != bridge["input_receipt"]["sha256"]:
        raise ValueError("bridge input receipt drift")
    input_receipt = read_json(input_receipt_path)
    implementation_sha = bound_code_sha(input_receipt, Path(__file__))
    bound_code_sha(input_receipt, Path(__file__).with_name("bridge_common.py"))
    if _validate_existing(
        args.output,
        bridge_sha=bridge_sha,
        config_sha=config_sha,
        implementation_sha=implementation_sha,
    ):
        print(json.dumps({"ok": True, "resumed": True, "output": str(args.output)}))
        return 0

    assets = config["training_assets"]
    init_checkpoint = Path(
        str(assets["init_checkpoint"]).format(
            scratch_root=str(args.scratch_root.resolve())
        )
    ).resolve()
    checkpoint_tree = file_tree_receipt(init_checkpoint)
    marker = init_checkpoint / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != str(
        assets["expected_checkpoint_marker"]
    ):
        raise ValueError("TD init checkpoint is not a validated release checkpoint")
    checkpoint_payload = init_checkpoint / str(assets["expected_checkpoint_marker"])
    if not checkpoint_payload.is_dir() or not any(checkpoint_payload.rglob("*")):
        raise ValueError("TD init checkpoint release payload is absent")

    evidence_bundle = (args.repo_root / str(assets["evidence_bundle"])).resolve()
    evidence_tree = file_tree_receipt(evidence_bundle)
    evidence_manifest_path = evidence_bundle / str(assets["evidence_manifest"])
    verification_path = evidence_bundle / str(assets["verification"])
    commit_marker_path = evidence_bundle / str(assets["megatron_commit_marker"])
    evidence_manifest = read_json(evidence_manifest_path)
    verification = read_json(verification_path)
    if (
        Path(str(evidence_manifest.get("patched_megatron_dir", ""))).resolve()
        != init_checkpoint
    ):
        raise ValueError("TD evidence names a different patched Megatron checkpoint")
    if f"layer{int(assets['distillation_layer'])}" not in str(
        evidence_manifest.get("hf_dir", "")
    ):
        raise ValueError("TD evidence does not identify distillation layer 11")
    if (
        Path(str(evidence_manifest.get("verification_json", ""))).name
        != verification_path.name
    ):
        raise ValueError("TD evidence manifest points to different verification output")
    if int(evidence_manifest.get("target_tensor_parallel_size", -1)) != int(
        assets["expected_tensor_parallel_size"]
    ) or int(evidence_manifest.get("target_pipeline_parallel_size", -1)) != int(
        assets["expected_pipeline_parallel_size"]
    ):
        raise ValueError("TD evidence has incompatible parallel checkpoint geometry")
    for field in assets["required_zero_metrics"]:
        if float(verification.get(field, float("nan"))) != 0.0:
            raise ValueError(f"TD roundtrip evidence is not exact: {field}")
    for field in assets["required_empty_metrics"]:
        if verification.get(field) != []:
            raise ValueError(f"TD roundtrip evidence reports differences: {field}")
    logits = verification.get("logits", {})
    if (
        float(logits.get("logit_max_abs_diff", float("nan"))) != 0.0
        or any(not row.get("top_id_match") for row in logits.get("per_prompt", []))
        or not logits.get("per_prompt")
    ):
        raise ValueError("TD roundtrip logit evidence is incomplete or nonzero")

    megatron_dir = args.megatron_dir.resolve()
    expected_commit = str(config["builder"]["expected_megatron_commit"])
    if _git(megatron_dir, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("effective Megatron commit drift")
    if _git(megatron_dir, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("effective Megatron tree is dirty or has untracked patches")
    if commit_marker_path.read_text(encoding="utf-8").strip() != expected_commit:
        raise ValueError("TD evidence was generated with a different Megatron commit")
    megatron_tree = file_tree_receipt(megatron_dir, exclude_top_level=(".git",))
    frozen_megatron_tree = input_receipt["megatron"]["tree"]
    if megatron_tree["tree_sha256"] != frozen_megatron_tree["tree_sha256"]:
        raise ValueError(
            "effective Megatron source tree differs from the bridge freeze"
        )

    for path, label in (
        (args.trainer.resolve(), "trainer"),
        (args.training_env.resolve(), "training environment"),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} is absent: {path}")

    payload: dict[str, Any] = {
        "schema_version": "full_cpt_training_assets_receipt_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "bridge_manifest": str(args.bridge_manifest.resolve()),
        "bridge_manifest_sha256": bridge_sha,
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": implementation_sha,
        "init_checkpoint": {
            "distillation_layer": int(assets["distillation_layer"]),
            "marker": str(assets["expected_checkpoint_marker"]),
            "tree": checkpoint_tree,
        },
        "td_layer11_evidence": {
            "bundle_tree": evidence_tree,
            "manifest": {
                "path": str(evidence_manifest_path),
                "sha256": sha256_file(evidence_manifest_path),
            },
            "verification": {
                "path": str(verification_path),
                "sha256": sha256_file(verification_path),
            },
            "megatron_commit_marker": {
                "path": str(commit_marker_path),
                "sha256": sha256_file(commit_marker_path),
                "commit": expected_commit,
            },
            "semantic_checks": {
                "distillation_layer_11": True,
                "patched_checkpoint_path_matches": True,
                "r17_and_standard_tensor_roundtrip_exact": True,
                "logit_roundtrip_exact": True,
                "parallel_geometry_matches": True,
            },
        },
        "megatron": {
            "root": str(megatron_dir),
            "commit": expected_commit,
            "clean": True,
            "tree": megatron_tree,
        },
        "trainer": {
            "path": str(args.trainer.resolve()),
            "sha256": sha256_file(args.trainer.resolve()),
        },
        "training_environment": {
            "path": str(args.training_env.resolve()),
            "sha256": sha256_file(args.training_env.resolve()),
        },
    }
    write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps({"ok": True, "output": str(args.output.resolve())}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
