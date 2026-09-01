#!/usr/bin/env python3
"""Validate one production segment at job start before any GPU process launches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from campaign_contract import (
    ARMS,
    SEGMENT_BOUNDARY,
    TOKENIZER_JSON_SHA256,
    TOTAL_ITERATIONS,
    atomic_write_json,
    read_json,
    sha256_file,
    verify_code_bundle_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--recovery-start", type=int)
    parser.add_argument("--runtime-scientific-bundle", type=Path)
    parser.add_argument("--runtime-scientific-bundle-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_file_receipt(row: dict) -> None:
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"asset missing or size drift: {path}")
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"asset hash drift: {path}")


def main() -> int:
    args = parse_args()
    campaign = read_json(args.campaign_manifest)
    if campaign.get("schema_version") != "apertus_mini_campaign_manifest_v1" or campaign.get("status") != "frozen":
        raise ValueError("campaign manifest is not frozen")
    runtime_override = args.runtime_scientific_bundle is not None
    if runtime_override != (args.runtime_scientific_bundle_receipt is not None):
        raise ValueError("runtime scientific bundle and receipt must be supplied together")
    if runtime_override:
        verify_code_bundle_receipt(
            args.runtime_scientific_bundle_receipt,
            args.runtime_scientific_bundle,
            "scientific",
        )
    expected = campaign["segments"][args.segment_id]
    if expected["segment_id"] != args.segment_id:
        raise ValueError("segment manifest ordering drift")
    for name in (
        "experiment_matrix",
        "schedule_manifest",
        "checkpoint_plan",
        "validation_manifest",
        "initialization_receipt",
        "lr_selection_receipt",
        "token_byte_lengths_receipt",
        "endpoint_benchmark_contract",
        "lm_eval_runtime_receipt",
        "megatron_runtime_receipt",
        "scientific_bundle_receipt",
        "efficiency_bundle_receipt",
    ):
        verify_file_receipt(campaign["assets"][name])
    verify_code_bundle_receipt(
        Path(campaign["assets"]["scientific_bundle_receipt"]["path"]),
        Path(campaign["assets"]["scientific_bundle"]),
        "scientific",
    )
    verify_code_bundle_receipt(
        Path(campaign["assets"]["efficiency_bundle_receipt"]["path"]),
        Path(campaign["assets"]["efficiency_bundle"]),
        "efficiency",
    )
    for row in campaign["assets"]["gate_receipts"]:
        verify_file_receipt(row)
    lr_selection = read_json(Path(campaign["assets"]["lr_selection_receipt"]["path"]))
    if (
        lr_selection.get("status") != "frozen"
        or float(lr_selection.get("selected_peak_lr", -1.0))
        != float(campaign["scientific_contract"]["peak_lr"])
        or float(lr_selection.get("selected_min_lr", -1.0))
        != float(campaign["scientific_contract"]["minimum_lr"])
        or lr_selection.get("same_lr_for_all_five_arms") is not True
    ):
        raise ValueError("frozen common LR selection drift at job start")
    tokenizer = Path(campaign["assets"]["tokenizer_dir"])
    if sha256_file(tokenizer / "tokenizer.json") != TOKENIZER_JSON_SHA256:
        raise ValueError("tokenizer drift at job start")
    byte_receipt = read_json(Path(campaign["assets"]["token_byte_lengths_receipt"]["path"]))
    byte_path = Path(campaign["assets"]["token_byte_lengths"])
    if sha256_file(byte_path) != byte_receipt["byte_lengths"]["sha256"]:
        raise ValueError("token byte-length table drift at job start")
    clean_subset = read_json(Path(campaign["assets"]["greekmmlu_clean_subset"]))
    if clean_subset.get("status") != "frozen" or int(clean_subset.get("clean_count", 0)) <= 0:
        raise ValueError("GreekMMLU clean subset drift at job start")
    lm_eval = read_json(Path(campaign["assets"]["lm_eval_runtime_receipt"]["path"]))
    if (
        lm_eval.get("status") != "frozen"
        or Path(lm_eval.get("root", "")).resolve() != Path(campaign["assets"]["lm_eval_root"]).resolve()
        or not (Path(campaign["assets"]["lm_eval_root"]) / "lm_eval" / "__main__.py").is_file()
        or lm_eval.get("distributions", {}).get("lm-eval") != "0.4.11"
        or lm_eval.get("distributions", {}).get("accelerate") != "1.13.0"
        or lm_eval.get("runtime_environment", {}).get("uenv_image")
        != "pytorch/v2.9.1:v2"
        or lm_eval.get("runtime_environment", {}).get("external_distributions")
        != {
            "huggingface-hub": "0.36.0",
            "psutil": "7.1.0",
            "safetensors": "0.6.2",
            "torch": "2.9.1",
            "transformers": "4.57.0",
        }
        or len(
            lm_eval.get("custom_task_aliases", {})
            .get("global_mmlu", {})
            .get("expands_to", [])
        )
        != 15
    ):
        raise ValueError("lm-eval runtime drift at job start")
    lm_lock = lm_eval.get("requirements_lock", {})
    verify_file_receipt(lm_lock)
    lm_eval_root = Path(campaign["assets"]["lm_eval_root"])
    for row in lm_eval.get("files", []):
        path = lm_eval_root / row["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"frozen lm-eval file drift at job start: {path}")
    megatron_root = Path(campaign["assets"]["megatron_dir"]).resolve()
    megatron_runtime = read_json(
        Path(campaign["assets"]["megatron_runtime_receipt"]["path"])
    )
    expected_runtime_files = {
        megatron_root / "megatron" / "training" / "arguments.py",
        megatron_root / "megatron" / "training" / "training.py",
        megatron_root / "pretrain_gpt.py",
    }
    observed_runtime_files = set()
    if (
        megatron_runtime.get("schema_version")
        != "apertus_mini_patched_megatron_runtime_v1"
        or megatron_runtime.get("status") != "frozen"
        or Path(megatron_runtime.get("output_root", "")).resolve() != megatron_root
        or megatron_runtime.get("upstream_commit")
        != "c92402e39ef3c8e69ea378a59e79059dc14541f4"
        or not megatron_runtime.get("checks")
        or not all(megatron_runtime["checks"].values())
    ):
        raise ValueError("patched Megatron runtime drift at job start")
    for row in megatron_runtime.get("patched_files", []):
        path = Path(row["path"]).resolve()
        observed_runtime_files.add(path)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"patched Megatron runtime file drift at job start: {path}")
    if observed_runtime_files != expected_runtime_files:
        raise ValueError("patched Megatron runtime file inventory drift at job start")
    schedule = read_json(Path(campaign["assets"]["schedule_manifest"]["path"]))
    if tuple(row["arm_id"] for row in schedule["arms"]) != ARMS:
        raise ValueError("five-arm schedule drift")
    validation = read_json(Path(campaign["assets"]["validation_manifest"]["path"]))
    for panel in validation["panels"]:
        prefix = Path(panel["megatron_prefix"])
        for suffix in (".bin", ".idx"):
            if not Path(str(prefix) + suffix).is_file():
                raise ValueError(f"validation payload missing: {prefix}{suffix}")
    nominal_start = int(expected["start_iteration"])
    start = nominal_start if args.recovery_start is None else int(args.recovery_start)
    end = int(expected["end_iteration"])
    if not nominal_start <= start < end:
        raise ValueError(f"invalid recovery start {start} for segment {args.segment_id}")
    if start == 0:
        if args.resume_receipt is not None:
            raise ValueError("iteration-0 start must not have a resume receipt")
        load_roots = {arm: campaign["assets"]["initial_checkpoint_root"] for arm in ARMS}
    else:
        if args.resume_receipt is None:
            raise ValueError("a nonzero start requires a common checkpoint receipt")
        receipt = read_json(args.resume_receipt)
        if (
            receipt.get("schema_version") != "apertus_mini_segment_checkpoint_receipt_v1"
            or receipt.get("status") != "passed"
            or int(receipt.get("iteration", -1)) != start
        ):
            raise ValueError("resume receipt iteration/status drift")
        if receipt.get("campaign_manifest_sha256") != sha256_file(args.campaign_manifest):
            raise ValueError("resume receipt is bound to another campaign")
        rows = {row["arm_id"]: row for row in receipt["arms"]}
        if tuple(sorted(rows)) != tuple(sorted(ARMS)):
            raise ValueError("resume receipt arm set drift")
        load_roots = {arm: rows[arm]["checkpoint_root"] for arm in ARMS}
        for arm, root_text in load_roots.items():
            root = Path(root_text)
            marker = root / "latest_checkpointed_iteration.txt"
            iteration_root = root / f"iter_{start:07d}"
            if not marker.is_file() or marker.read_text().strip() != str(start):
                raise ValueError(f"resume checkpoint marker drift for {arm}: {root}")
            if not (iteration_root / ".metadata").is_file():
                raise ValueError(f"incomplete resume checkpoint for {arm}: {root}")
            receipted_iteration = Path(rows[arm]["checkpoint"]["root"]).resolve()
            if iteration_root.resolve() != receipted_iteration:
                raise ValueError(f"resume checkpoint view target drift for {arm}: {root}")
    free = shutil.disk_usage(args.run_root.parent if args.run_root.exists() else args.run_root.parent).free
    hard_minimum = int(campaign["runtime"]["hard_checkpoint_headroom_bytes"])
    if free < hard_minimum:
        raise ValueError(f"checkpoint headroom below hard minimum: {free} < {hard_minimum}")
    output = {
        "schema_version": "apertus_mini_segment_preflight_v1",
        "status": "passed",
        "segment_id": args.segment_id,
        "start_iteration": start,
        "end_iteration": end,
        "campaign_manifest": str(args.campaign_manifest.resolve()),
        "campaign_manifest_sha256": sha256_file(args.campaign_manifest),
        "load_roots": load_roots,
        "free_bytes": free,
        "expected_final_iteration": TOTAL_ITERATIONS,
        "runtime_scientific_bundle": (
            str(args.runtime_scientific_bundle.resolve()) if runtime_override else None
        ),
        "runtime_scientific_bundle_receipt": (
            str(args.runtime_scientific_bundle_receipt.resolve())
            if runtime_override
            else None
        ),
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
