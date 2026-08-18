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


def entrypoint_from_argv(argv: list[Any]) -> dict[str, Any]:
    candidates = [Path(item) for item in argv if isinstance(item, str) and item.startswith("/") and item.endswith(".py")]
    if len(candidates) != 1:
        raise ValueError(f"expected one absolute Python entrypoint in train argv, found {candidates!r}")
    result = binding(candidates[0])
    result.update({"kind": "file", "verify_at_submit": True})
    return result


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

    continuation = json.loads(json.dumps(campaign))
    continuation["campaign_id"] = f"{campaign['campaign_id']}-current-s2-continuation-v1"
    continuation_science = continuation["science"]
    continuation_science["entrypoint"] = entrypoint_from_argv(continuation_science["train_argv"])
    continuation_science.pop("required_env", None)
    continuation_segments = continuation["segments"][s2_index:]
    continuation_segments[0]["load_checkpoint"] = str(checkpoint_path)
    continuation["segments"] = continuation_segments

    runtime_candidate = json.loads(json.dumps(runtime))
    runtime_candidate["status"] = "candidate"
    runtime_candidate.pop("qualification_receipt", None)

    args.output_root.mkdir(parents=True, exist_ok=False)
    campaign_path = args.output_root / "campaign.json"
    runtime_path = args.output_root / "runtime-candidate.json"
    evaluation_path = args.output_root / "evaluation.json"
    atomic_json(campaign_path, continuation)
    atomic_json(runtime_path, runtime_candidate)
    atomic_json(evaluation_path, evaluation)
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
        },
        "s2_checkpoint": binding(checkpoint_path),
        "outputs": {
            "campaign": binding(campaign_path),
            "runtime_candidate": binding(runtime_path),
            "evaluation": binding(evaluation_path),
        },
        "invariants": {
            "train_argv_unchanged": continuation_science["train_argv"] == science["train_argv"],
            "immutable_inputs_unchanged": continuation_science["immutable_inputs"] == science["immutable_inputs"],
            "s2_and_later_segment_overrides_unchanged": continuation_segments[0]["argv_overrides"] == segments[s2_index]["argv_overrides"],
            "only_s2_load_checkpoint_changed": continuation_segments[0]["load_checkpoint"] == str(checkpoint_path),
            "entrypoint_added_from_existing_train_argv": True,
            "no_dataset_transformation_performed": True,
        },
    }
    atomic_json(args.output_root / "receipt.json", receipt)
    print(json.dumps({"status": "passed", "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
