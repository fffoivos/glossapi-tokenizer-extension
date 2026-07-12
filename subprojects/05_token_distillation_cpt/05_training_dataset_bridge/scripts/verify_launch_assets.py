#!/usr/bin/env python3
"""Fail-closed launch validation for bridge, binaries, code, init, and resume."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bridge_common import (
    bound_code_sha,
    read_json,
    sha256_file,
    validate_file_tree_receipt,
    validate_frozen_repository,
    validate_launch_dependency_receipts,
    validate_tokenizer_tree_receipt,
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


def validate_tokenizer_asset(
    input_receipt: dict[str, object],
    assets: dict[str, object],
    expected_tokenizer_dir: Path,
) -> Path:
    frozen_tokenizer = input_receipt.get("tokenizer", {})
    if not isinstance(frozen_tokenizer, dict):
        raise ValueError("input receipt has no tokenizer binding")
    input_tokenizer_tree = {
        "root": str(frozen_tokenizer.get("root", "")),
        "files": frozen_tokenizer.get("files"),
        "tree_sha256": frozen_tokenizer.get("tree_sha256"),
    }
    expected_tokenizer_asset = {
        "root": str(Path(str(frozen_tokenizer.get("root", ""))).resolve()),
        "tree": input_tokenizer_tree,
    }
    if assets.get("tokenizer") != expected_tokenizer_asset:
        raise ValueError("training assets are bound to a different tokenizer tree")
    tokenizer_root = validate_tokenizer_tree_receipt(input_tokenizer_tree)
    if tokenizer_root != expected_tokenizer_dir.resolve():
        raise ValueError("exported tokenizer root differs from the frozen input tree")
    return tokenizer_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--training-data-env", type=Path, required=True)
    parser.add_argument("--training-assets-receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--training-env", type=Path, required=True)
    parser.add_argument("--common-training-env", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--runtime-wrapper", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--expected-load-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-megatron-dir", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--start-iteration", type=int, required=True)
    parser.add_argument("--probe-plan", type=Path)
    parser.add_argument("--resume-checkpoint-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge = read_json(args.bridge_manifest)
    assets = read_json(args.training_assets_receipt)
    if (
        bridge.get("schema_version") != "full_cpt_training_bridge_manifest_v1"
        or bridge.get("status") != "completed"
        or bridge.get("recipe_id") != "full_corpus_td_25b_79_20_1_v1"
        or not all(bridge.get("invariants", {}).values())
    ):
        raise ValueError("bridge manifest is not launchable")
    bridge_sha = sha256_file(args.bridge_manifest.resolve())
    if (
        assets.get("schema_version") != "full_cpt_training_assets_receipt_v1"
        or assets.get("status") != "completed"
        or assets.get("bridge_manifest_sha256") != bridge_sha
    ):
        raise ValueError("training assets are not bound to this finalized bridge")
    input_path = Path(str(bridge["input_receipt"]["path"])).resolve()
    if sha256_file(input_path) != bridge["input_receipt"]["sha256"]:
        raise ValueError("bridge input receipt drift")
    input_receipt = read_json(input_path)
    repository = validate_frozen_repository(input_receipt, args.repo_root)
    if assets.get("repository") != repository:
        raise ValueError("training assets are bound to a different repository checkout")
    tokenizer_root = validate_tokenizer_asset(
        input_receipt, assets, args.expected_tokenizer_dir
    )
    bound_code_sha(input_receipt, Path(__file__))
    bound_code_sha(input_receipt, Path(__file__).with_name("bridge_common.py"))
    assets_impl = Path(str(assets.get("implementation", ""))).resolve()
    if not assets_impl.is_file() or sha256_file(assets_impl) != assets.get(
        "implementation_sha256"
    ):
        raise ValueError("training-assets implementation receipt drift")
    bound_code_sha(input_receipt, assets_impl)

    validate_launch_dependency_receipts(
        assets.get("launch_dependencies"),
        {
            "common_training_environment": args.common_training_env,
            "launcher": args.launcher,
            "runtime_wrapper": args.runtime_wrapper,
            "trainer": args.trainer,
            "training_environment": args.training_env,
        },
    )

    if (
        sha256_file(args.training_data_env.resolve())
        != bridge["training_env"]["sha256"]
    ):
        raise ValueError("generated training-data environment drift")
    if (
        Path(str(bridge["training_env"]["path"])).resolve()
        != args.training_data_env.resolve()
    ):
        raise ValueError("launcher selected a different training-data environment")
    if (
        sha256_file(args.training_env.resolve())
        != assets["training_environment"]["sha256"]
    ):
        raise ValueError("frozen 25B training environment drift")
    if (
        args.training_env.resolve()
        != Path(str(assets["training_environment"]["path"])).resolve()
    ):
        raise ValueError("launcher selected a different training environment")
    if sha256_file(args.trainer.resolve()) != assets["trainer"]["sha256"]:
        raise ValueError("frozen training script drift")
    if args.trainer.resolve() != Path(str(assets["trainer"]["path"])).resolve():
        raise ValueError("launcher selected a different training script")

    for row in bridge["outputs"]:
        for label in ("bin", "idx"):
            receipt = row[label]
            path = Path(str(receipt["path"]))
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(receipt["bytes"])
                or sha256_file(path) != receipt["sha256"]
            ):
                raise ValueError(f"Megatron {label} payload drift: {path}")
    if not all(
        row.get("passed") for row in bridge["capacity"].get("physical_prefixes", [])
    ) or not bridge["capacity"].get("physical_prefixes"):
        raise ValueError("physical-prefix sample-capacity proof is absent or failed")

    init_root = validate_file_tree_receipt(assets["init_checkpoint"]["tree"])
    validate_file_tree_receipt(assets["td_layer11_evidence"]["bundle_tree"])
    megatron_root = validate_file_tree_receipt(assets["megatron"]["tree"])
    if megatron_root != Path(str(assets["megatron"].get("root", ""))).resolve():
        raise ValueError("runtime-selected Megatron root differs from its frozen tree")
    if megatron_root != args.expected_megatron_dir.resolve():
        raise ValueError("exported Megatron root differs from the frozen training asset")
    if not all(assets["td_layer11_evidence"]["semantic_checks"].values()):
        raise ValueError("TD layer-11 semantic evidence has a failed check")
    if _git(megatron_root, "rev-parse", "HEAD") != assets["megatron"]["commit"]:
        raise ValueError("effective Megatron commit drift")
    if _git(megatron_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("effective Megatron tree is dirty or patched")

    result = {
        "bridge_manifest_sha256": bridge_sha,
        "training_assets_receipt_sha256": sha256_file(
            args.training_assets_receipt.resolve()
        ),
        "start_iteration": args.start_iteration,
        "initial_checkpoint": str(init_root),
        "megatron_root": str(megatron_root),
        "tokenizer_root": str(tokenizer_root),
    }
    if args.start_iteration == 0:
        if args.resume_checkpoint_receipt is not None:
            raise ValueError("initial segment must not provide a resume receipt")
        result["load_checkpoint"] = str(init_root)
        result["resume"] = False
    else:
        if args.probe_plan is None or args.resume_checkpoint_receipt is None:
            raise ValueError("segmented relaunch requires plan and checkpoint receipt")
        plan = read_json(args.probe_plan)
        assets_sha = result["training_assets_receipt_sha256"]
        if (
            plan.get("schema_version") != "full_cpt_25b_probe_plan_v2"
            or plan.get("bridge_manifest", {}).get("sha256") != bridge_sha
            or plan.get("training_assets_receipt", {}).get("sha256") != assets_sha
            or int(plan.get("iterations", -1)) != 5960
            or int(plan.get("effective_tokens", -1)) != 24_998_051_840
        ):
            raise ValueError("immutable probe plan differs from frozen launch inputs")
        resume = read_json(args.resume_checkpoint_receipt)
        resume_impl = Path(str(resume.get("implementation", ""))).resolve()
        if not resume_impl.is_file() or sha256_file(resume_impl) != resume.get(
            "implementation_sha256"
        ):
            raise ValueError("resume-checkpoint implementation receipt drift")
        bound_code_sha(input_receipt, resume_impl)
        if (
            resume.get("schema_version") != "full_cpt_segment_checkpoint_receipt_v1"
            or resume.get("status") != "completed"
            or int(resume.get("expected_iteration", -1)) != args.start_iteration
            or resume.get("probe_plan", {}).get("sha256")
            != sha256_file(args.probe_plan.resolve())
            or resume.get("training_assets_receipt", {}).get("sha256") != assets_sha
            or not all(resume.get("invariants", {}).values())
        ):
            raise ValueError("resume checkpoint receipt has incompatible bindings")
        checkpoint_root = validate_file_tree_receipt(resume["checkpoint_tree"])
        result["load_checkpoint"] = str(checkpoint_root)
        result["resume"] = True
        result["resume_checkpoint_receipt_sha256"] = sha256_file(
            args.resume_checkpoint_receipt.resolve()
        )
    if Path(str(result["load_checkpoint"])).resolve() != args.expected_load_checkpoint.resolve():
        raise ValueError("exported load checkpoint differs from the verified launch receipt")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
