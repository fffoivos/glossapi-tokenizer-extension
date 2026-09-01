#!/usr/bin/env python3
"""Freeze the complete five-arm campaign only after every launch gate passes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from campaign_contract import (
    ACTIVE_TOKENS,
    ARMS,
    EXPECTED_GREEKMMLU_TOTAL,
    GLOBAL_BATCH_SEQUENCES,
    MODEL_REVISION,
    SCHEDULED_TOKEN_SLOTS,
    SCHEDULE_MANIFEST_SHA256,
    SEGMENT_BOUNDARY,
    SEQUENCE_LENGTH,
    TOKENIZER_JSON_SHA256,
    TOKENIZER_REVISION,
    TOTAL_ITERATIONS,
    atomic_write_json,
    checkpoint_iterations,
    file_receipt,
    read_json,
    require_status,
    sha256_file,
    verify_checkpoint_plan,
    verify_code_bundle_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-matrix", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-plan", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--initialization-receipt", type=Path, required=True)
    parser.add_argument("--lr-selection-receipt", type=Path, required=True)
    parser.add_argument("--token-byte-lengths-receipt", type=Path, required=True)
    parser.add_argument("--greekmmlu-clean-subset", type=Path, required=True)
    parser.add_argument("--endpoint-benchmark-contract", type=Path, required=True)
    parser.add_argument("--lm-eval-runtime-receipt", type=Path, required=True)
    parser.add_argument("--lm-eval-root", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, action="append", required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--megatron-runtime-receipt", type=Path, required=True)
    parser.add_argument("--native-greek-eval-root", type=Path, required=True)
    parser.add_argument("--python-compat-dir", type=Path, required=True)
    parser.add_argument("--scientific-bundle", type=Path, required=True)
    parser.add_argument("--efficiency-bundle", type=Path, required=True)
    parser.add_argument("--scientific-bundle-receipt", type=Path, required=True)
    parser.add_argument("--efficiency-bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(args.experiment_matrix)
    if matrix.get("launch_authorized") is not True or matrix.get("not_valid_for_launch") is True:
        raise ValueError("experiment matrix has not been explicitly launch-authorized")
    authorization = matrix.get("launch_authorization", {})
    expected_gate_ids = tuple(matrix.get("launch_gates", []))
    authorized_gate_receipts = authorization.get("gate_receipts", [])
    if (
        len(expected_gate_ids) != 16
        or authorization.get("gate_ids") != list(expected_gate_ids)
        or len(authorized_gate_receipts) != len(expected_gate_ids)
    ):
        raise ValueError("authorized matrix gate chain is incomplete")
    schedule = require_status(
        args.schedule_manifest,
        schemas={"apertus_mini_five_data_order_schedules_v1"},
    )
    if sha256_file(args.schedule_manifest) != SCHEDULE_MANIFEST_SHA256:
        raise ValueError("schedule manifest hash drift")
    observed_arms = tuple(row["arm_id"] for row in schedule["arms"])
    if observed_arms != ARMS:
        raise ValueError(f"schedule arm ordering drift: {observed_arms}")
    source_matrix_receipt = authorization.get("source_matrix", {})
    source_matrix_path = Path(source_matrix_receipt.get("path", ""))
    if (
        not source_matrix_path.is_file()
        or source_matrix_path.stat().st_size
        != int(source_matrix_receipt.get("bytes", -1))
        or sha256_file(source_matrix_path) != source_matrix_receipt.get("sha256")
    ):
        raise ValueError("authorized source experiment matrix receipt drift")
    plan = verify_checkpoint_plan(
        args.checkpoint_plan,
        args.schedule_manifest,
        source_matrix_path,
    )
    iterations = checkpoint_iterations(plan)
    if int(plan.get("native_greekmmlu_evaluations_total", -1)) != EXPECTED_GREEKMMLU_TOTAL:
        raise ValueError("GreekMMLU evaluation count drift")
    validation = require_status(args.validation_manifest)
    panels = validation.get("panels", [])
    if len(panels) != 13 or len({row["name"] for row in panels}) != 13:
        raise ValueError("validation manifest must freeze exactly 13 named panels")
    require_status(args.initialization_receipt)
    lr_selection = require_status(
        args.lr_selection_receipt,
        schemas={"apertus_mini_common_lr_selection_v1"},
    )
    peak_lr = float(lr_selection.get("selected_peak_lr", -1.0))
    min_lr = float(lr_selection.get("selected_min_lr", -1.0))
    if (
        peak_lr not in {3.0e-4, 1.5e-4}
        or min_lr != peak_lr * 0.1
        or lr_selection.get("same_lr_for_all_five_arms") is not True
    ):
        raise ValueError("common stability-smoke LR selection drift")
    byte_lengths = require_status(
        args.token_byte_lengths_receipt,
        schemas={"apertus_token_utf8_byte_lengths_v1"},
    )
    byte_payload = byte_lengths["byte_lengths"]
    byte_path = Path(byte_payload["path"])
    if (
        byte_lengths.get("vocab_size") != 148_992
        or not byte_path.is_file()
        or byte_path.stat().st_size != int(byte_payload["bytes"])
        or sha256_file(byte_path) != byte_payload["sha256"]
    ):
        raise ValueError("token UTF-8 byte-length receipt drift")
    clean_subset = require_status(
        args.greekmmlu_clean_subset,
        schemas={"apertus_mini_greekmmlu_clean_subset_v1"},
    )
    if (
        clean_subset.get("dataset_revision")
        != "6a03aa06b68beb932fb75edff3a34e50b3674649"
        or int(clean_subset.get("full_count", -1)) != 16_632
        or int(clean_subset.get("clean_count", -1)) <= 0
    ):
        raise ValueError("GreekMMLU clean-subset receipt drift")
    endpoint_contract = require_status(
        args.endpoint_benchmark_contract,
        schemas={"apertus_mini_endpoint_benchmark_contract_v1"},
    )
    if set(endpoint_contract.get("benchmarks", {})) != {"greek_belebele", "demosqa"}:
        raise ValueError("Greek endpoint benchmark contract drift")
    lm_eval = require_status(
        args.lm_eval_runtime_receipt,
        schemas={"apertus_mini_lm_eval_runtime_v1"},
    )
    if Path(lm_eval.get("root", "")).resolve() != args.lm_eval_root.resolve():
        raise ValueError("lm-eval receipt/root binding drift")
    if (
        lm_eval.get("distributions", {}).get("lm-eval") != "0.4.11"
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
    ):
        raise ValueError("lm-eval package/uenv version drift")
    global_alias = lm_eval.get("custom_task_aliases", {}).get("global_mmlu", {})
    if len(global_alias.get("expands_to", [])) != 15:
        raise ValueError("global_mmlu retention alias is not the frozen 15-language suite")
    lm_lock = lm_eval.get("requirements_lock", {})
    lock_path = Path(lm_lock.get("path", ""))
    if (
        not lock_path.is_file()
        or lock_path.stat().st_size != int(lm_lock.get("bytes", -1))
        or sha256_file(lock_path) != lm_lock.get("sha256")
    ):
        raise ValueError("lm-eval requirements-lock drift")
    if not (args.lm_eval_root / "lm_eval" / "__main__.py").is_file():
        raise ValueError("frozen lm-eval runtime is incomplete")
    for row in lm_eval.get("files", []):
        path = args.lm_eval_root / row["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"frozen lm-eval file drift: {path}")
    gates = []
    observed_gate_ids = []
    for path in args.gate_receipt:
        gate = require_status(path, schemas={"apertus_mini_launch_gate_receipt_v1"})
        gates.append(file_receipt(path))
        observed_gate_ids.append(gate.get("gate_id"))
    if observed_gate_ids != list(expected_gate_ids) or gates != authorized_gate_receipts:
        raise ValueError("campaign gate receipts differ from the matrix authorization chain")
    verify_code_bundle_receipt(
        args.scientific_bundle_receipt, args.scientific_bundle, "scientific"
    )
    verify_code_bundle_receipt(
        args.efficiency_bundle_receipt, args.efficiency_bundle, "efficiency"
    )
    megatron_runtime = require_status(
        args.megatron_runtime_receipt,
        schemas={"apertus_mini_patched_megatron_runtime_v1"},
    )
    megatron_root = args.megatron_dir.resolve()
    expected_runtime_files = {
        megatron_root / "megatron" / "training" / "arguments.py",
        megatron_root / "megatron" / "training" / "training.py",
        megatron_root / "pretrain_gpt.py",
    }
    observed_runtime_files = set()
    if (
        Path(megatron_runtime.get("output_root", "")).resolve() != megatron_root
        or megatron_runtime.get("upstream_commit")
        != "c92402e39ef3c8e69ea378a59e79059dc14541f4"
        or not megatron_runtime.get("checks")
        or not all(megatron_runtime["checks"].values())
    ):
        raise ValueError("patched Megatron runtime receipt drift")
    for row in megatron_runtime.get("patched_files", []):
        path = Path(row["path"]).resolve()
        observed_runtime_files.add(path)
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"patched Megatron runtime file drift: {path}")
    if observed_runtime_files != expected_runtime_files:
        raise ValueError("patched Megatron runtime file inventory drift")
    tokenizer_json = args.tokenizer_dir.resolve() / "tokenizer.json"
    if sha256_file(tokenizer_json) != TOKENIZER_JSON_SHA256:
        raise ValueError("tokenizer.json drift")
    latest = args.initial_checkpoint_root.resolve() / "latest_checkpointed_iteration.txt"
    initial_release = args.initial_checkpoint_root.resolve() / "release"
    if (
        not latest.is_file()
        or latest.read_text(encoding="utf-8").strip() != "release"
        or not (initial_release / ".metadata").is_file()
    ):
        raise ValueError("initial checkpoint root must be a complete torch_dist release checkpoint")
    payload = {
        "schema_version": "apertus_mini_campaign_manifest_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_contract": {
            "arms": list(ARMS),
            "only_factor": "modern_greek_temporal_order",
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "active_tokens_per_arm": ACTIVE_TOKENS,
            "scheduled_token_slots_per_arm": SCHEDULED_TOKEN_SLOTS,
            "optimizer_iterations": TOTAL_ITERATIONS,
            "global_batch_sequences": GLOBAL_BATCH_SEQUENCES,
            "sequence_length": SEQUENCE_LENGTH,
            "learning_rate_treatment": "L0_wsd10",
            "peak_lr": peak_lr,
            "minimum_lr": min_lr,
            "checkpoint_averaging": False,
        },
        "segments": [
            {"segment_id": 0, "start_iteration": 0, "end_iteration": SEGMENT_BOUNDARY},
            {
                "segment_id": 1,
                "start_iteration": SEGMENT_BOUNDARY,
                "end_iteration": TOTAL_ITERATIONS,
            },
        ],
        "parallelism": {
            "tensor": 1,
            "pipeline": 1,
            "context": 1,
            "data_per_arm": 16,
            "nodes_per_arm": 4,
            "training_nodes": 20,
            "gpus_per_node": 4,
            "micro_batch_sequences": 4,
            "gradient_accumulation_steps": 8,
        },
        "evaluation": {
            "checkpoint_iterations": list(iterations),
            "checkpoints_per_arm": len(iterations),
            "native_greekmmlu_bindings": EXPECTED_GREEKMMLU_TOTAL,
            "validation_panels": [row["name"] for row in panels],
            "validation_bindings": len(iterations) * len(ARMS) * len(panels),
        },
        "assets": {
            "experiment_matrix": file_receipt(args.experiment_matrix),
            "schedule_manifest": file_receipt(args.schedule_manifest),
            "checkpoint_plan": file_receipt(args.checkpoint_plan),
            "validation_manifest": file_receipt(args.validation_manifest),
            "initialization_receipt": file_receipt(args.initialization_receipt),
            "lr_selection_receipt": file_receipt(args.lr_selection_receipt),
            "token_byte_lengths_receipt": file_receipt(args.token_byte_lengths_receipt),
            "token_byte_lengths": str(byte_path.resolve()),
            "greekmmlu_clean_subset": str(args.greekmmlu_clean_subset.resolve()),
            "endpoint_benchmark_contract": file_receipt(args.endpoint_benchmark_contract),
            "lm_eval_runtime_receipt": file_receipt(args.lm_eval_runtime_receipt),
            "lm_eval_root": str(args.lm_eval_root.resolve()),
            "initial_checkpoint_root": str(args.initial_checkpoint_root.resolve()),
            "tokenizer_dir": str(args.tokenizer_dir.resolve()),
            "megatron_dir": str(args.megatron_dir.resolve()),
            "megatron_runtime_receipt": file_receipt(args.megatron_runtime_receipt),
            "native_greek_eval_root": str(args.native_greek_eval_root.resolve()),
            "python_compat_dir": str(args.python_compat_dir.resolve()),
            "scientific_bundle": str(args.scientific_bundle.resolve()),
            "efficiency_bundle": str(args.efficiency_bundle.resolve()),
            "scientific_bundle_receipt": file_receipt(args.scientific_bundle_receipt),
            "efficiency_bundle_receipt": file_receipt(args.efficiency_bundle_receipt),
            "gate_receipts": gates,
        },
        "runtime": {
            "account": "a0140",
            "partition": "normal",
            "uenv": "pytorch/v2.9.1:v2",
            "uenv_view": "default",
            "walltime_per_segment": "12:00:00",
            "minimum_starting_checkpoint_headroom_bytes": 4_500_000_000_000,
            "hard_checkpoint_headroom_bytes": 500_000_000_000,
            "maximum_infrastructure_retries": 2,
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
