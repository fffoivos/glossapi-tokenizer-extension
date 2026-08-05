#!/usr/bin/env python3
"""Shared fail-closed contract helpers for the production campaign."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any


ARMS = (
    "D0_mixed",
    "D1_hard_h_to_g",
    "D2_hard_g_to_h",
    "D3_gradual_h_to_g",
    "D4_gradual_g_to_h",
)
TOTAL_ITERATIONS = 38_496
SEGMENT_BOUNDARY = 19_456
GLOBAL_BATCH_SEQUENCES = 512
SEQUENCE_LENGTH = 4_096
SCHEDULED_TOKEN_SLOTS = 80_731_963_392
ACTIVE_TOKENS = 80_729_939_067
EXPECTED_CHECKPOINTS_PER_ARM = 83
EXPECTED_GREEKMMLU_TOTAL = EXPECTED_CHECKPOINTS_PER_ARM * len(ARMS)
SCHEDULE_MANIFEST_SHA256 = "ffeaa69492b0a30768efb5c34a942e1b7d11ca5df0d962d001ae6387d6f20955"
MODEL_REVISION = "1b7276176e564fc0cc7d7c3b991a8d653c8b8792"
TOKENIZER_REVISION = "fcd33ec09fb7d86bc072b3a4b3e890efa6473b66"
TOKENIZER_JSON_SHA256 = "cc3f544817da0e8d1623e3f7484df7f67464aeb00867aece956880e9b407ef8f"
PASS_STATUSES = {"pass", "passed", "completed", "frozen"}
AUTHORITATIVE_EVALUATION_DTYPE = "float32"
DEFAULT_EVALUATION_NAMESPACE = "fp32_v1"
EVALUATION_NAMESPACE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def evaluation_namespace() -> str:
    """Return the isolated namespace for authoritative checkpoint evaluation."""

    value = os.environ.get(
        "EVALUATION_NAMESPACE", DEFAULT_EVALUATION_NAMESPACE
    ).strip()
    if EVALUATION_NAMESPACE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid evaluation namespace: {value!r}")
    return value


def scoped_evaluation_root(run_root: Path, stem: str) -> Path:
    """Keep a recovered evaluator from reusing legacy/provisional artifacts."""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", stem):
        raise ValueError(f"invalid evaluation root stem: {stem!r}")
    return run_root.resolve() / f"{stem}_{evaluation_namespace()}"


def resolve_evaluation_runtime(
    campaign_manifest_path: Path,
    campaign: dict[str, Any],
) -> tuple[Path, Path]:
    """Resolve a receipt-bound evaluation-only runtime override.

    Training always remains bound to the scientific and Megatron roots in the
    frozen campaign manifest.  A live evaluation recovery may select different
    roots only through an append-only receipt that binds the manifest and every
    modified runtime file by SHA-256.
    """

    assets = campaign["assets"]
    originals = {
        "scientific_bundle": Path(assets["scientific_bundle"]).resolve(),
        "megatron_dir": Path(assets["megatron_dir"]).resolve(),
    }
    requested = {
        "scientific_bundle": Path(
            os.environ.get("SCIENTIFIC_BUNDLE", str(originals["scientific_bundle"]))
        ).resolve(),
        "megatron_dir": Path(
            os.environ.get("EVALUATION_MEGATRON_DIR", str(originals["megatron_dir"]))
        ).resolve(),
    }
    changed = {key for key in originals if requested[key] != originals[key]}
    if not changed:
        return requested["scientific_bundle"], requested["megatron_dir"]

    receipt_text = os.environ.get("OPERATIONAL_RECOVERY_RECEIPT", "")
    if not receipt_text:
        raise ValueError("evaluation runtime override lacks OPERATIONAL_RECOVERY_RECEIPT")
    receipt_path = Path(receipt_text).resolve()
    receipt = require_status(
        receipt_path,
        schemas={"apertus_mini_evaluation_runtime_recovery_v1"},
    )
    if (
        receipt.get("evaluation_namespace") != evaluation_namespace()
        or receipt.get("authoritative_evaluation_dtype")
        != AUTHORITATIVE_EVALUATION_DTYPE
    ):
        raise ValueError("evaluation recovery namespace/dtype binding drift")
    manifest_path = campaign_manifest_path.resolve()
    manifest = receipt.get("campaign_manifest", {})
    if (
        Path(manifest.get("path", "")).resolve() != manifest_path
        or manifest.get("sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("evaluation recovery campaign-manifest binding drift")
    overrides = receipt.get("overrides", {})
    for key in changed:
        row = overrides.get(key, {})
        if (
            Path(row.get("from_path", "")).resolve() != originals[key]
            or Path(row.get("to_path", "")).resolve() != requested[key]
        ):
            raise ValueError(f"evaluation recovery override drift: {key}")
    runtime_files = receipt.get("runtime_files", [])
    if not runtime_files:
        raise ValueError("evaluation recovery lacks runtime file bindings")
    for row in runtime_files:
        path = Path(row.get("path", "")).resolve()
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
        ):
            raise ValueError(f"evaluation recovery runtime-file drift: {path}")
    bundle_reference = receipt.get("scientific_bundle_receipt", {})
    bundle_receipt_path = Path(bundle_reference.get("path", "")).resolve()
    if (
        not bundle_receipt_path.is_file()
        or sha256_file(bundle_receipt_path) != bundle_reference.get("sha256")
    ):
        raise ValueError("evaluation recovery scientific-bundle receipt drift")
    verify_code_bundle_receipt(
        bundle_receipt_path,
        requested["scientific_bundle"],
        "scientific",
    )
    diagnostic_reference = receipt.get("semantic_parity_diagnostics", {})
    diagnostic_path = Path(diagnostic_reference.get("path", "")).resolve()
    if (
        not diagnostic_path.is_file()
        or sha256_file(diagnostic_path) != diagnostic_reference.get("sha256")
    ):
        raise ValueError("evaluation recovery semantic-parity diagnostic drift")
    require_status(
        diagnostic_path,
        schemas={"apertus_mini_conversion_semantic_parity_diagnostic_v1"},
    )
    return requested["scientific_bundle"], requested["megatron_dir"]


def file_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_status(path: Path, *, schemas: set[str] | None = None) -> dict[str, Any]:
    value = read_json(path)
    if schemas is not None and value.get("schema_version") not in schemas:
        raise ValueError(f"{path}: unexpected schema {value.get('schema_version')!r}")
    if str(value.get("status", "")).lower() not in PASS_STATUSES:
        raise ValueError(f"{path}: non-passing status {value.get('status')!r}")
    return value


def verify_code_bundle_receipt(path: Path, root: Path, kind: str) -> dict[str, Any]:
    receipt = require_status(
        path, schemas={"apertus_mini_immutable_code_bundle_v1"}
    )
    resolved_root = root.resolve()
    rows = receipt.get("files", [])
    if (
        receipt.get("kind") != kind
        or Path(receipt.get("root", "")).resolve() != resolved_root
        or int(receipt.get("file_count", -1)) != len(rows)
        or not rows
    ):
        raise ValueError(f"{kind} bundle receipt/root drift")
    seen = set()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        candidate = (resolved_root / relative).resolve()
        if (
            not relative
            or relative in seen
            or candidate.parent != resolved_root
            and resolved_root not in candidate.parents
            or not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(candidate) != row.get("sha256")
        ):
            raise ValueError(f"{kind} bundle file drift: {relative}")
        seen.add(relative)
    canonical = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != receipt.get("tree_sha256"):
        raise ValueError(f"{kind} bundle tree hash drift")
    return receipt


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_iterations(plan: dict[str, Any]) -> tuple[int, ...]:
    values = tuple(int(row["iteration"]) for row in plan["checkpoint_rows"])
    if len(values) != EXPECTED_CHECKPOINTS_PER_ARM or values != tuple(sorted(set(values))):
        raise ValueError("checkpoint plan does not contain 83 unique ordered iterations")
    if values[0] != 0 or values[-1] != TOTAL_ITERATIONS:
        raise ValueError("checkpoint plan endpoints drift")
    return values


def verify_checkpoint_plan(
    plan_path: Path,
    schedule_path: Path,
    source_matrix_path: Path,
) -> dict[str, Any]:
    """Verify the plan semantically and bind it to the exact launch inputs.

    The plan hash is deliberately not compiled into the orchestration code.  The
    experiment matrix can acquire stricter, cadence-neutral launch policies after
    an earlier plan was generated.  In that case the plan must be regenerated
    against the new source-matrix hash; accepting the earlier hard-coded plan hash
    would silently break the evidence chain.
    """

    plan = require_status(
        plan_path,
        schemas={"apertus_mini_checkpoint_evaluation_plan_v1"},
    )
    schedule = require_status(
        schedule_path,
        schemas={"apertus_mini_five_data_order_schedules_v1"},
    )
    schedule_receipt = plan.get("schedule_manifest", {})
    matrix_receipt = plan.get("experiment_matrix", {})
    resolved_schedule = schedule_path.resolve()
    resolved_matrix = source_matrix_path.resolve()
    if (
        Path(schedule_receipt.get("path", "")).resolve() != resolved_schedule
        or schedule_receipt.get("sha256") != sha256_file(resolved_schedule)
        or int(schedule_receipt.get("bytes", -1)) != resolved_schedule.stat().st_size
    ):
        raise ValueError("checkpoint plan does not bind the exact schedule manifest")
    if (
        Path(matrix_receipt.get("path", "")).resolve() != resolved_matrix
        or matrix_receipt.get("sha256") != sha256_file(resolved_matrix)
        or int(matrix_receipt.get("bytes", -1)) != resolved_matrix.stat().st_size
    ):
        raise ValueError("checkpoint plan does not bind the exact source experiment matrix")
    if (
        int(plan.get("optimizer_steps", -1)) != TOTAL_ITERATIONS
        or int(plan.get("checkpoint_count_per_arm", -1))
        != EXPECTED_CHECKPOINTS_PER_ARM
        or int(plan.get("native_greekmmlu_evaluations_total", -1))
        != EXPECTED_GREEKMMLU_TOTAL
        or plan.get("greekmmlu_origin") != "natively_authored_greek"
        or plan.get("greekmmlu_dataset")
        != {
            "repo_id": "dascim/GreekMMLU",
            "revision": "6a03aa06b68beb932fb75edff3a34e50b3674649",
            "config": "All",
            "split": "test",
        }
    ):
        raise ValueError("checkpoint/GreekMMLU plan contract drift")
    iterations = checkpoint_iterations(plan)
    rows = plan.get("checkpoint_rows", [])
    by_iteration = {int(row["iteration"]): row for row in rows}
    required_common = {
        0: "initial_checkpoint",
        800: "after_warmup",
        SEGMENT_BOUNDARY: "normal_partition_segment_boundary",
        int(0.8 * TOTAL_ITERATIONS): "cooldown_start",
        TOTAL_ITERATIONS: "raw_final_endpoint",
    }
    for iteration in range(512, TOTAL_ITERATIONS + 1, 512):
        required_common[iteration] = "regular_512_step_cadence"
    for iteration, reason in required_common.items():
        if iteration not in by_iteration or reason not in by_iteration[iteration].get(
            "reasons", []
        ):
            raise ValueError(f"checkpoint plan is missing {reason} at {iteration}")
    metrics = (
        "official_zero_shot_accuracy",
        "multiple_choice_cross_entropy_from_frozen_normalized_choice_scores",
        "correct_answer_continuation_bpb",
    )
    for row in rows:
        iteration = int(row["iteration"])
        if (
            row.get("all_arms") != list(ARMS)
            or row.get("full_state_checkpoint_required") is not True
            or row.get("fast_source_conditioned_panel_required") is not True
            or row.get("native_greekmmlu_required") is not True
            or tuple(row.get("native_greekmmlu_metrics", [])) != metrics
            or row.get("same_frozen_evaluator_contract") is not True
            or int(row.get("nominal_consumed_tokens", -1))
            != iteration * GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH
        ):
            raise ValueError(f"checkpoint evaluation row drift at iteration {iteration}")
    transitions = plan.get("hard_transitions", {})
    if set(transitions) != {"D1_hard_h_to_g", "D2_hard_g_to_h"}:
        raise ValueError("hard-transition checkpoint inventory drift")
    schedule_arms = {row["arm_id"]: row for row in schedule.get("arms", [])}
    if tuple(schedule_arms) != ARMS:
        raise ValueError("checkpoint plan schedule-arm inventory drift")

    def derive_transition(arm_id: str) -> dict[str, int]:
        target_pool = 1 if arm_id == "D1_hard_h_to_g" else 0
        receipt = schedule_arms[arm_id].get("sequence_ids", {})
        ids_path = Path(receipt.get("path", ""))
        if (
            not ids_path.is_file()
            or ids_path.is_symlink()
            or ids_path.stat().st_size != int(receipt.get("bytes", -1))
            or ids_path.stat().st_size % 8
            or sha256_file(ids_path) != receipt.get("sha256")
        ):
            raise ValueError(f"hard-arm sequence payload drift: {arm_id}")
        slot = 0
        with ids_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                for (sequence_id,) in struct.iter_unpack("<Q", chunk):
                    if sequence_id != 2**64 - 1 and sequence_id >> 62 == target_pool:
                        update = slot // GLOBAL_BATCH_SEQUENCES + 1
                        return {
                            "first_destination_sequence_slot": slot,
                            "optimizer_update_containing_first_destination_sequence": update,
                            "checkpoint_immediately_before": update - 1,
                            "checkpoint_after_first_complete_transition_update": update,
                        }
                    slot += 1
        raise ValueError(f"hard-arm destination pool is absent: {arm_id}")

    for arm_id, transition in transitions.items():
        if transition != derive_transition(arm_id):
            raise ValueError(f"hard-transition derivation drift: {arm_id}")
        before = int(transition.get("checkpoint_immediately_before", -1))
        after = int(transition.get("checkpoint_after_first_complete_transition_update", -1))
        if (
            after != before + 1
            or before not in by_iteration
            or after not in by_iteration
            or f"matched_{arm_id}_pre_transition"
            not in by_iteration[before].get("reasons", [])
            or f"matched_{arm_id}_post_transition"
            not in by_iteration[after].get("reasons", [])
        ):
            raise ValueError(f"hard-transition checkpoint binding drift: {arm_id}")
    if tuple(by_iteration) != iterations:
        raise ValueError("checkpoint row order drift")
    return plan


def padded_iteration(iteration: int) -> str:
    if not 0 <= int(iteration) <= TOTAL_ITERATIONS:
        raise ValueError(f"iteration outside campaign horizon: {iteration}")
    return f"iter_{int(iteration):07d}"
