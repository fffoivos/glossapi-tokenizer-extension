#!/usr/bin/env python3
"""Validate all launch semantics and emit the only authorizable gate set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from campaign_contract import verify_checkpoint_plan


SCHEDULE_SHA = "ffeaa69492b0a30768efb5c34a942e1b7d11ca5df0d962d001ae6387d6f20955"
TOKENIZER_SHA = "cc3f544817da0e8d1623e3f7484df7f67464aeb00867aece956880e9b407ef8f"
ARMS = ("D0_mixed", "D1_hard_h_to_g", "D2_hard_g_to_h", "D3_gradual_h_to_g", "D4_gradual_g_to_h")


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(path)
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict:
    if not path.is_file(): raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def require(value: bool, message: str) -> None:
    if not value: raise ValueError(message)


def write_gate(output_dir: Path, matrix: Path, gate_id: str, evidence: list[Path], assertions: list[str]) -> None:
    validator_path = Path(__file__).resolve()
    payload = {
        "schema_version": "apertus_mini_launch_gate_receipt_v1",
        "status": "passed",
        "launch_authorized": True,
        "gate_id": gate_id,
        "passed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_matrix": file_receipt(matrix),
        "assertions": assertions,
        "evidence": [file_receipt(path) for path in evidence],
        "semantic_validation": {
            "schema_version": "apertus_mini_launch_gate_semantics_v1",
            "validator": "production/finalize_launch_gate_set.py",
            "validator_sha256": sha(validator_path),
            "all_gate_specific_checks_passed": True,
        },
    }
    output = output_dir / f"{gate_id}.json"
    temporary = Path(str(output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-matrix", type=Path, required=True)
    p.add_argument("--overlay-manifest", type=Path, required=True)
    p.add_argument("--initialization-receipt", type=Path, required=True)
    p.add_argument("--pool-corpus-receipt", type=Path, required=True)
    p.add_argument("--packed-corpus-receipt", type=Path, required=True)
    p.add_argument("--schedule-manifest", type=Path, required=True)
    p.add_argument("--schedule-audit", type=Path, required=True)
    p.add_argument("--goldfish-uniformity", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--checkpoint-plan", type=Path, required=True)
    p.add_argument("--greekmmlu-runtime-smoke", type=Path, required=True)
    p.add_argument("--greekmmlu-wave-smoke", type=Path, required=True)
    p.add_argument("--b1-restart-receipt", type=Path, required=True)
    p.add_argument("--b2-contention-receipt", type=Path, required=True)
    p.add_argument("--lr-selection-receipt", type=Path, required=True)
    p.add_argument("--prelaunch-smoke-receipt", type=Path, required=True)
    p.add_argument("--megatron-dir", type=Path, required=True)
    p.add_argument("--megatron-runtime-receipt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    matrix = read(a.experiment_matrix)
    required_gates = matrix.get("launch_gates", [])
    require(matrix.get("launch_authorized") is False and len(required_gates) == 16, "source matrix/gate inventory drift")
    require(not a.output_dir.exists(), f"refusing to replace {a.output_dir}")

    overlay, initialization = read(a.overlay_manifest), read(a.initialization_receipt)
    pool, packed, schedule = read(a.pool_corpus_receipt), read(a.packed_corpus_receipt), read(a.schedule_manifest)
    audit, goldfish, validation = read(a.schedule_audit), read(a.goldfish_uniformity), read(a.validation_manifest)
    checkpoint, greek_smoke, wave = read(a.checkpoint_plan), read(a.greekmmlu_runtime_smoke), read(a.greekmmlu_wave_smoke)
    b1, b2, lr, prelaunch = read(a.b1_restart_receipt), read(a.b2_contention_receipt), read(a.lr_selection_receipt), read(a.prelaunch_smoke_receipt)
    megatron_runtime = read(a.megatron_runtime_receipt)

    require(overlay.get("status") == "completed" and overlay.get("target_vocab_size") == 148_992, "overlay incomplete")
    require(overlay.get("alignment") == {"divisor": 256, "quotient": 582, "remainder": 0, "padding_tokens": 0}, "overlay alignment/padding drift")
    require(overlay.get("output", {}).get("tokenizer_json_sha256") == TOKENIZER_SHA, "overlay tokenizer hash drift")
    require(initialization.get("status") == "passed" and initialization.get("input_output_embeddings_tied") is True and initialization.get("base_rows_bitwise_preserved") is True, "TD initialization incomplete")
    require(initialization.get("trained_fraction", 0) >= 0.90 and initialization.get("requested_rows") == 17_920, "TD coverage drift")
    require(
        initialization.get("fvt_macro_non_regression_gate_passed") is True
        and float(initialization.get("selected_delta_macro_bpb_vs_fvt", 1.0)) <= 0.0
        and float(initialization.get("fvt_baseline_macro_bpb", 0.0)) > 0.0
        and float(initialization.get("selected_pilot_macro_bpb", 0.0)) > 0.0,
        "TD selection did not pass the tied-FVT macro-BPB non-regression gate",
    )

    require(pool.get("status") == "completed" and pool.get("schema_version") == "apertus_mini_schedule_pool_corpus_v1", "pool receipt incomplete")
    identity = pool.get("global_identity_proof", {})
    require(identity.get("record_duplicates_or_collisions") == 0 and identity.get("modern_greek_exact_content_duplicates_or_collisions") == 0, "training identity/content dedup failed")
    require(identity.get("replay_content_policy") == "audit_only_preserve_original_training_replay_records_and_report_duplicates", "replay distribution policy drift")
    require(pool.get("modern_greek", {}).get("hplt_tokens") == 44_042_201_419 and pool.get("modern_greek", {}).get("glossapi_non_hplt_tokens") == 19_734_450_444, "post-exclusion Greek token counts drift")
    require(packed.get("status") == "completed" and packed.get("global") == {"sequence_count": 19_709_692, "active_tokens": 80_729_939_067, "duplicate_sequence_ids": 0}, "packed corpus totals drift")
    expected_pool_tokens = {"hplt_new_greek": 44_042_201_419, "non_hplt_new_greek": 19_734_450_444, "foreign_replay": 16_145_987_813, "old_greek_replay": 807_299_391}
    require({key: row["active_tokens"] for key, row in packed.get("pools", {}).items()} == expected_pool_tokens, "packed pool token totals drift")

    require(sha(a.schedule_manifest) == SCHEDULE_SHA and schedule.get("status") == "completed", "schedule hash/status drift")
    require(tuple(row["arm_id"] for row in schedule.get("arms", [])) == ARMS, "schedule arms drift")
    require(all(row.get("training_slots") == 19_709_952 and row.get("optimizer_updates") == 38_496 for row in schedule["arms"]), "schedule horizon drift")
    common = schedule.get("common_contract", {})
    require(common.get("same_exact_sequence_multiset") is True and common.get("same_replay_sequence_ids_at_same_global_positions") is True, "schedule identity/replay contract failed")
    require(audit.get("status") == "passed" and audit.get("same_replay_sequence_ids_at_same_positions") is True and audit.get("same_exact_sequence_and_active_token_inventory") is True and audit.get("five_distinct_modern_greek_order_trajectories") is True, "only-factor schedule audit failed")
    require(goldfish.get("status") == "passed" and goldfish.get("added_tokens", {}).get("count") == 17_920 and goldfish.get("added_tokens", {}).get("all_share_exact_complete_residue_drop_count") is True, "Goldfish token uniformity failed")

    require(validation.get("status") == "frozen" and validation.get("panel_count") == 13, "validation panel incomplete")
    neutral = {row["name"]: row for row in validation.get("panels", [])}.get("neutral_external_modern_greek")
    require(neutral is not None and 10_000_000 <= neutral.get("tokens", 0) <= 20_000_000, "neutral external Greek panel missing")
    checkpoint = verify_checkpoint_plan(
        a.checkpoint_plan,
        a.schedule_manifest,
        a.experiment_matrix,
    )
    require(greek_smoke.get("status") == "passed" and greek_smoke.get("native_greekmmlu", {}).get("examples") == 16_632, "GreekMMLU runtime smoke failed")
    require(wave.get("status") == "passed" and wave.get("pipelines", {}).get("exact_receipts") == 5, "GreekMMLU wave smoke failed")

    require(b1.get("status") == "passed" and b1.get("restart_parity", {}).get("first_post_checkpoint_exact") is True, "B1 restart parity failed")
    require(b2.get("status") == "passed" and b2.get("training_allocation", {}).get("data_parallel_per_arm") == 16 and b2.get("training_allocation", {}).get("nodes") == 20, "B2 geometry failed")
    require(b2.get("checkpoint_integrity", {}).get("skipped_iterations_total") == 0 and b2.get("checkpoint_integrity", {}).get("nan_iterations_total") == 0, "B2 numerical integrity failed")
    require(b2.get("forecast", {}).get("below_24_hour_preferred_training_budget") is True and b2.get("forecast", {}).get("below_36_hour_hard_round_target") is True, "runtime target forecast failed")
    require(lr.get("status") == "frozen" and lr.get("selected_peak_lr") in {3e-4, 1.5e-4} and lr.get("same_lr_for_all_five_arms") is True, "common LR selection failed")
    megatron_root = a.megatron_dir.resolve()
    expected_runtime_files = {
        megatron_root / "megatron" / "training" / "arguments.py",
        megatron_root / "megatron" / "training" / "training.py",
        megatron_root / "pretrain_gpt.py",
    }
    observed_runtime_files = set()
    require(
        megatron_runtime.get("schema_version") == "apertus_mini_patched_megatron_runtime_v1"
        and megatron_runtime.get("status") == "frozen"
        and Path(megatron_runtime.get("output_root", "")).resolve() == megatron_root
        and megatron_runtime.get("upstream_commit") == "c92402e39ef3c8e69ea378a59e79059dc14541f4"
        and megatron_runtime.get("checks")
        and all(megatron_runtime["checks"].values()),
        "patched Megatron runtime receipt drift",
    )
    for row in megatron_runtime.get("patched_files", []):
        path = Path(row["path"]).resolve()
        observed_runtime_files.add(path)
        require(
            path.is_file()
            and path.stat().st_size == int(row["bytes"])
            and sha(path) == row["sha256"],
            f"patched Megatron runtime file drift: {path}",
        )
    require(observed_runtime_files == expected_runtime_files, "patched Megatron runtime file inventory drift")
    prelaunch_results = prelaunch.get("results", [])
    require(
        prelaunch.get("status") == "passed"
        and prelaunch.get("concurrent_arms") == 5
        and prelaunch.get("data_parallel_size_per_arm") == 16
        and len(prelaunch_results) == 10
        and all(
            row.get("arm_id") in ARMS
            and row.get("phase") in {"initial_to_64", "resume_64_to_128"}
            and row.get("checks")
            and all(row["checks"].values())
            for row in prelaunch_results
        )
        and {
            (row["arm_id"], row["phase"])
            for row in prelaunch_results
        }
        == {
            (arm, phase)
            for arm in ARMS
            for phase in ("initial_to_64", "resume_64_to_128")
        },
        "five-arm real-data prelaunch/resume smoke failed",
    )

    a.output_dir.mkdir(parents=True)
    evidence = {
        "overlay": a.overlay_manifest, "init": a.initialization_receipt, "pool": a.pool_corpus_receipt,
        "packed": a.packed_corpus_receipt, "schedule": a.schedule_manifest, "audit": a.schedule_audit,
        "goldfish": a.goldfish_uniformity, "validation": a.validation_manifest, "checkpoint": a.checkpoint_plan,
        "greek": a.greekmmlu_runtime_smoke, "wave": a.greekmmlu_wave_smoke, "b1": a.b1_restart_receipt,
        "b2": a.b2_contention_receipt, "lr": a.lr_selection_receipt, "prelaunch": a.prelaunch_smoke_receipt,
        "runtime": a.megatron_runtime_receipt,
    }
    mapping = {
        required_gates[0]: ([evidence["overlay"], evidence["init"]], ["Mini IDs and merges roundtrip; vocab 148992 is divisible by 256 without padding"]),
        required_gates[1]: ([evidence["init"]], ["one four-cell tied-TD pilot recipe passed the tied-FVT macro-BPB non-regression gate and was applied to all requested rows"]),
        required_gates[2]: ([evidence["pool"]], ["post-exclusion eligible record identity set is unique and frozen"]),
        required_gates[3]: ([evidence["pool"], evidence["packed"]], ["fresh exact selected-tokenizer counts close to the frozen active-token totals"]),
        required_gates[4]: ([evidence["pool"], evidence["schedule"], evidence["audit"]], ["replay preserves original multiplicity and exact IDs/positions across arms"]),
        required_gates[5]: ([evidence["lr"]], ["candidate-first common stability smoke selected one LR for every arm"]),
        required_gates[6]: ([evidence["schedule"], evidence["audit"]], ["five schedules use one exact sequence/token inventory"]),
        required_gates[7]: ([evidence["audit"]], ["immutable sequence labels imply identical deterministic Goldfish masks"]),
        required_gates[8]: ([evidence["goldfish"]], ["all 17920 added IDs are neutral under the pinned Goldfish hash rule"]),
        required_gates[9]: ([evidence["audit"]], ["only HPLT/non-HPLT temporal order differs"]),
        required_gates[10]: ([evidence["pool"], evidence["validation"]], ["13 cluster-level globally deduplicated panels are frozen"]),
        required_gates[11]: ([evidence["checkpoint"], evidence["greek"], evidence["wave"]], ["83 checkpoints per arm and 415 native GreekMMLU bindings use the frozen evaluator"]),
        required_gates[12]: ([evidence["b1"], evidence["b2"]], ["real scheduled data passed DP16 single-arm and five-arm systems gates"]),
        required_gates[13]: ([evidence["b2"]], ["five concurrent DP16 arms passed on 20 nodes"]),
        required_gates[14]: ([evidence["b2"]], ["controlling-arm forecast is below 24 training hours and 36 end-to-end hours"]),
        required_gates[15]: ([evidence["init"], evidence["b1"], evidence["prelaunch"], evidence["runtime"]], ["production initialization load, exact restart, hash-checked patched runtime, all validation panels and five-arm resume passed"]),
    }
    require(set(mapping) == set(required_gates), "semantic gate map coverage drift")
    for gate_id in required_gates: write_gate(a.output_dir, a.experiment_matrix, gate_id, *mapping[gate_id])
    manifest = {"schema_version": "apertus_mini_launch_gate_set_v1", "status": "passed", "gate_count": len(required_gates), "gate_ids": required_gates, "receipts": [file_receipt(a.output_dir / f"{gate}.json") for gate in required_gates]}
    (a.output_dir / "gate_set_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "gates": len(required_gates), "output_dir": str(a.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
