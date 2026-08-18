#!/usr/bin/env python3
"""Build a current-controller continuation contract without changing S2 science.

Workaround for apertus-cscs-efficiency issue #115.  This adapter reads the
historical, completed S1 campaign and its signed postprocess-recovery permit,
then writes a new continuation campaign rooted at S2.  It preserves the
training argv, data/model/tokenizer/init revisions, every S2+ segment override,
and the exact S1 checkpoint reference.  The only scientific-contract addition
is the current runner's required bound entrypoint provenance.

It writes only a fresh experiment-owned output directory.  It never changes
the historical run root, any data artifact, or a canonical tooling bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "apertus_hard_h_to_g_current_s2_continuation_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a receipt byte-for-byte into a fresh immutable contract root."""

    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with source.open("rb") as origin, os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(origin, handle)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def entrypoint_from_argv(argv: list[Any]) -> dict[str, Any]:
    candidates = [Path(item) for item in argv if isinstance(item, str) and item.startswith("/") and item.endswith(".py")]
    if len(candidates) != 1:
        raise ValueError(f"expected one absolute Python entrypoint in train argv, found {candidates!r}")
    result = binding(candidates[0])
    result.update({"kind": "file", "verify_at_submit": True})
    return result


def resolve_source_binding(value: Any, *, source_directory: Path, label: str) -> dict[str, Any]:
    """Resolve a legacy relative receipt binding without changing its content."""

    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"source campaign lacks {label} binding")
    recorded_path = Path(value["path"])
    path = recorded_path if recorded_path.is_absolute() else source_directory / recorded_path
    resolved = binding(path)
    expected = {key: value.get(key) for key in ("bytes", "sha256")}
    if {key: resolved[key] for key in expected} != expected:
        raise ValueError(f"{label} binding drift: {path}")
    return resolved


def portable_relative_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{label} must be a portable non-empty relative path")
    return path


def prepared_gate_sources(training_manifest: Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    """Return the small receipt closure required to verify a data manifest."""

    manifest = read_json(training_manifest)
    root_value = str(manifest.get("root", "."))
    root = Path(root_value)
    if root_value != ".":
        root = portable_relative_path(root_value, label="training_data_manifest.root")
    else:
        root = Path(".")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("training_data_manifest lacks datasets")
    sources: list[tuple[Path, Path, dict[str, Any]]] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("training_data_manifest dataset is not an object")
        gate = dataset.get("prepared_gate")
        if not isinstance(gate, dict) or not isinstance(gate.get("path"), str):
            raise ValueError("training_data_manifest lacks prepared_gate binding")
        relative = portable_relative_path(gate["path"], label="prepared_gate.path")
        source_directory = training_manifest.parent / root
        source = resolve_source_binding(
            gate, source_directory=source_directory, label=f"prepared_gate:{dataset.get('id', '?')}"
        )
        sources.append((root / relative, Path(source["path"]), source))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--recovery-permit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"output root must not exist: {args.output_root}")
    campaign = read_json(args.source_campaign)
    runtime = read_json(args.source_runtime)
    evaluation = read_json(args.source_evaluation)
    permit = read_json(args.recovery_permit)
    if permit.get("status") != "passed" or permit.get("target_segment_id") != "s2":
        raise ValueError("recovery permit is not an accepted S2 permit")
    checkpoint = permit.get("checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("path"), str):
        raise ValueError("recovery permit lacks checkpoint binding")
    checkpoint_path = Path(checkpoint["path"])
    if binding(checkpoint_path) != {key: checkpoint[key] for key in ("path", "bytes", "sha256")}:
        raise ValueError("recovery checkpoint binding drift")

    science = campaign.get("science")
    segments = campaign.get("segments")
    if not isinstance(science, dict) or not isinstance(segments, list):
        raise ValueError("source campaign missing science or segments")
    s2_index = next((index for index, row in enumerate(segments) if row.get("id") == "s2"), None)
    if s2_index is None:
        raise ValueError("source campaign lacks s2")
    training_data_manifest = resolve_source_binding(
        science.get("training_data_manifest"),
        source_directory=args.source_campaign.parent,
        label="training_data_manifest",
    )
    gate_sources = prepared_gate_sources(Path(training_data_manifest["path"]))
    readiness_plan = resolve_source_binding(
        campaign.get("readiness_plan"),
        source_directory=args.source_campaign.parent,
        label="readiness_plan",
    )

    continuation = json.loads(json.dumps(campaign))
    # The approved readiness horizon binds campaign_id.  The new immutable run
    # root, not a renamed science contract, identifies this continuation.
    continuation["campaign_id"] = campaign["campaign_id"]
    continuation["readiness_plan"] = {
        "path": "readiness_plan.json",
        "bytes": readiness_plan["bytes"],
        "sha256": readiness_plan["sha256"],
    }
    continuation_science = continuation["science"]
    continuation_science["entrypoint"] = entrypoint_from_argv(continuation_science["train_argv"])
    continuation_science.pop("required_env", None)
    # Legacy contracts intentionally used a path relative to their own
    # directory.  A continuation has a different immutable directory, so copy
    # the checked manifest byte-for-byte and keep a portable relative binding.
    continuation_science["training_data_manifest"] = {
        "path": "training_data_manifest.json",
        "bytes": training_data_manifest["bytes"],
        "sha256": training_data_manifest["sha256"],
    }
    continuation_segments = continuation["segments"][s2_index:]
    continuation_segments[0]["load_checkpoint"] = str(checkpoint_path)
    continuation["segments"] = continuation_segments
    continuation["legacy_continuation"] = {
        "source_manifest": binding(args.source_campaign),
        "recovery_permit": binding(args.recovery_permit),
        "preserve_train_argv": True,
    }

    minimum = int(continuation_segments[0]["start_iteration"])
    maximum = int(continuation_segments[-1]["end_iteration"])
    continuation_evaluation = json.loads(json.dumps(evaluation))
    continuation_evaluation["evaluators"] = [
        {
            **row,
            "milestone_iterations": [
                value
                for value in row["milestone_iterations"]
                if minimum <= int(value) <= maximum
            ],
        }
        for row in continuation_evaluation["evaluators"]
        if any(minimum <= int(value) <= maximum for value in row["milestone_iterations"])
    ]

    runtime_candidate = json.loads(json.dumps(runtime))
    runtime_candidate["status"] = "candidate"
    runtime_candidate.pop("qualification_receipt", None)

    args.output_root.mkdir(parents=True, exist_ok=False)
    campaign_path = args.output_root / "campaign.json"
    runtime_path = args.output_root / "runtime-candidate.json"
    evaluation_path = args.output_root / "evaluation.json"
    continuation_manifest_path = args.output_root / "training_data_manifest.json"
    continuation_readiness_path = args.output_root / "readiness_plan.json"
    atomic_copy(Path(training_data_manifest["path"]), continuation_manifest_path)
    if binding(continuation_manifest_path)["sha256"] != training_data_manifest["sha256"]:
        raise ValueError("copied training_data_manifest hash mismatch")
    copied_prepared_gates: list[dict[str, Any]] = []
    for relative, source, source_binding in gate_sources:
        destination = args.output_root / relative
        atomic_copy(source, destination)
        copied = binding(destination)
        if copied["sha256"] != source_binding["sha256"]:
            raise ValueError(f"copied prepared gate hash mismatch: {relative}")
        copied_prepared_gates.append(copied)
    atomic_copy(Path(readiness_plan["path"]), continuation_readiness_path)
    if binding(continuation_readiness_path)["sha256"] != readiness_plan["sha256"]:
        raise ValueError("copied readiness_plan hash mismatch")
    atomic_json(campaign_path, continuation)
    atomic_json(runtime_path, runtime_candidate)
    atomic_json(evaluation_path, continuation_evaluation)
    receipt = {
        "schema_version": SCHEMA,
        "status": "passed",
        "issue": "https://github.com/fffoivos/apertus-cscs-efficiency/issues/115",
        "adapter": binding(Path(__file__).resolve()),
        "purpose": "current-controller continuation at S2; train science unchanged",
        "source": {
            "campaign": binding(args.source_campaign),
            "runtime": binding(args.source_runtime),
            "evaluation": binding(args.source_evaluation),
            "recovery_permit": binding(args.recovery_permit),
            "training_data_manifest": training_data_manifest,
            "readiness_plan": readiness_plan,
        },
        "s2_checkpoint": binding(checkpoint_path),
        "outputs": {
            "campaign": binding(campaign_path),
            "runtime_candidate": binding(runtime_path),
            "evaluation": binding(evaluation_path),
            "training_data_manifest": binding(continuation_manifest_path),
            "prepared_gates": copied_prepared_gates,
            "readiness_plan": binding(continuation_readiness_path),
        },
        "invariants": {
            "train_argv_unchanged": continuation_science["train_argv"] == science["train_argv"],
            "campaign_id_unchanged": continuation["campaign_id"] == campaign["campaign_id"],
            "immutable_inputs_unchanged": continuation_science["immutable_inputs"] == science["immutable_inputs"],
            "s2_and_later_segment_overrides_unchanged": continuation_segments[0]["argv_overrides"] == segments[s2_index]["argv_overrides"],
            "s2_load_checkpoint_rebound_to_recovery_permit": continuation_segments[0]["load_checkpoint"] == str(checkpoint_path),
            "entrypoint_added_from_existing_train_argv": True,
            "training_data_manifest_rebound_without_content_change": True,
            "prepared_gate_receipts_rebound_without_content_change": True,
            "readiness_plan_rebound_without_content_change": True,
            "evaluation_milestones_restricted_to_continuation_horizon": True,
            "no_dataset_transformation_performed": True,
        },
    }
    atomic_json(args.output_root / "receipt.json", receipt)
    print(json.dumps({"status": "passed", "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
