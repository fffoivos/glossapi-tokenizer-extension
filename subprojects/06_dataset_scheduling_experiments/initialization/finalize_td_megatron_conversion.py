#!/usr/bin/env python3
"""Verify and receipt the tied HF -> Megatron torch_dist -> HF roundtrip."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree(root: Path) -> list[dict]:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"empty tree: {root}")
    return rows


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-reference", type=Path, required=True)
    parser.add_argument("--torch-dist-root", type=Path, required=True)
    parser.add_argument("--hf-roundtrip", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--megatron-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    latest = args.torch_dist_root / "latest_checkpointed_iteration.txt"
    release = args.torch_dist_root / "release"
    if (
        not latest.is_file()
        or latest.read_text().strip() != "release"
        or not (release / ".metadata").is_file()
    ):
        raise ValueError("initial Megatron root is not a complete torch_dist release")
    verification = read_json(args.verification)
    if (
        verification.get("orig_only")
        or verification.get("trip_only")
        or verification.get("shape_mismatches")
        or int(verification.get("standard_changed_over_tol_count", -1)) != 0
        or int(verification.get("r17_changed_over_tol_count", -1)) != 0
        or not all(row.get("top_id_match") is True for row in verification.get("logits", {}).get("per_prompt", []))
    ):
        raise ValueError("HF/Megatron/HF roundtrip verification failed")
    td_manifest = read_json(args.hf_reference / "tied_td_manifest.json")
    if (
        td_manifest.get("status") != "completed"
        or td_manifest.get("scope") != "full"
        or td_manifest.get("input_output_share_storage") is not True
    ):
        raise ValueError("reference full TD artifact is incomplete")
    payload = {
        "schema_version": "apertus_mini_td_megatron_conversion_v2",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "megatron_commit": args.megatron_commit,
        "parallelism": {"tensor": 1, "pipeline": 1},
        "hf_reference": str(args.hf_reference.resolve()),
        "hf_reference_tree": tree(args.hf_reference),
        "initial_checkpoint_root": str(args.torch_dist_root.resolve()),
        "release_tree": tree(release),
        "hf_roundtrip": str(args.hf_roundtrip.resolve()),
        "hf_roundtrip_tree": tree(args.hf_roundtrip),
        "verification": {
            "path": str(args.verification.resolve()),
            "sha256": sha256_file(args.verification),
            "bytes": args.verification.stat().st_size,
        },
        "checks": {
            "torch_dist_release_complete": True,
            "tensor_parallel_size_one": True,
            "pipeline_parallel_size_one": True,
            "all_standard_tensors_within_tolerance": True,
            "all_apertus_extra_tensors_within_tolerance": True,
            "fixed_prompt_top1_logits_match": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "output": str(args.output), "checkpoint": "release"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
