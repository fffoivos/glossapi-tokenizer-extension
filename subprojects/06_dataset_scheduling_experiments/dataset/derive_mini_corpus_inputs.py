#!/usr/bin/env python3
"""Derive Mini-overlay corpus inputs from the frozen, vetted 8B corpus receipt.

This intentionally reuses document identities, heldout/decontamination policy,
replay acquisition and source files.  Only the tokenizer binding and output
stage change.  The existing two-way partitions remain useful physical shards;
the scheduling layer later recombines them into HPLT, non-HPLT, foreign replay
and Old-Greek pools.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_bridge_common(path: Path):
    spec = importlib.util.spec_from_file_location("schedule_bridge_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def link_or_copy(source: Path, destination: Path, expected_sha: str, expected_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.stat().st_size != expected_bytes
            or sha256_file(destination) != expected_sha
        ):
            raise ValueError(f"existing exclusion payload drift: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if destination.stat().st_size != expected_bytes or sha256_file(destination) != expected_sha:
        raise ValueError(f"copied exclusion payload drift: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input-receipt", type=Path, required=True)
    parser.add_argument("--source-heldout-manifest", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output-stage", type=Path, required=True)
    parser.add_argument("--bridge-common", type=Path, required=True)
    parser.add_argument("--derivation-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_input = args.output_stage / "input_receipt.json"
    output_heldout = args.output_stage / "heldouts" / "heldout_manifest.json"
    if output_input.exists() or output_heldout.exists():
        raise SystemExit("refusing to overwrite an existing derived input freeze")

    source_input = read_json(args.source_input_receipt)
    source_heldout = read_json(args.source_heldout_manifest)
    source_input_sha = sha256_file(args.source_input_receipt)
    source_heldout_sha = sha256_file(args.source_heldout_manifest)
    if (
        source_input.get("schema_version") != "full_cpt_training_bridge_input_receipt_v1"
        or source_input.get("status") != "frozen"
    ):
        raise ValueError("source input receipt is not frozen")
    if (
        source_heldout.get("schema_version") != "full_cpt_training_heldouts_v1"
        or source_heldout.get("status") != "completed"
        or source_heldout.get("input_receipt_sha256") != source_input_sha
    ):
        raise ValueError("source heldout manifest is not bound to the source input receipt")

    overlay_manifest_path = args.overlay_dir / "overlay_manifest.json"
    overlay = read_json(overlay_manifest_path)
    tokenizer_path = args.overlay_dir / "tokenizer.json"
    tokenizer_sha = sha256_file(tokenizer_path)
    if (
        overlay.get("schema_version") != "apertus_mini_greek_tokenizer_overlay_v1"
        or overlay.get("status") != "completed"
        or overlay.get("output", {}).get("tokenizer_json_sha256") != tokenizer_sha
        or int(overlay.get("target_vocab_size", -1)) != 148_992
        or int(overlay.get("alignment", {}).get("remainder", -1)) != 0
        or int(overlay.get("alignment", {}).get("padding_tokens", -1)) != 0
    ):
        raise ValueError("Mini overlay manifest failed the frozen tokenizer gates")
    bridge_common = load_bridge_common(args.bridge_common.resolve())
    tokenizer_tree = bridge_common.tokenizer_tree_receipt(args.overlay_dir.resolve())

    created_at = datetime.now(timezone.utc).isoformat()
    derived = copy.deepcopy(source_input)
    derived["created_at"] = created_at
    derived["recipe_id"] = args.derivation_id
    derived["tokenizer"] = {
        "root": str(args.overlay_dir.resolve()),
        "repo_id": "swiss-ai/Apertus-v1.1-0.5B+fffoivos/apertus-tokenizer-extension",
        "revision": (
            "1b7276176e564fc0cc7d7c3b991a8d653c8b8792+"
            "fcd33ec09fb7d86bc072b3a4b3e890efa6473b66"
        ),
        "subfolder": "mini-compatible-greek-modern-polytonic-overlay",
        "tokenizer_json_sha256": tokenizer_sha,
        "vocab_size": 148_992,
        "tree_sha256": tokenizer_tree["tree_sha256"],
        "tree": tokenizer_tree,
        "overlay_manifest": {
            "path": str(overlay_manifest_path.resolve()),
            "sha256": sha256_file(overlay_manifest_path),
        },
    }
    derived["derivation"] = {
        "schema_version": "apertus_mini_schedule_corpus_derivation_v1",
        "purpose": "retokenize frozen vetted corpus with Mini-compatible Greek overlay",
        "source_input_receipt": {
            "path": str(args.source_input_receipt.resolve()),
            "sha256": source_input_sha,
        },
        "source_heldout_manifest": {
            "path": str(args.source_heldout_manifest.resolve()),
            "sha256": source_heldout_sha,
        },
        "documents_reselected": False,
        "heldouts_reselected": False,
        "replay_reacquired": False,
        "only_material_change": "tokenizer binding and binary output stage",
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_atomic(output_input, derived)
    output_input_sha = sha256_file(output_input)

    heldout = copy.deepcopy(source_heldout)
    heldout["completed_at"] = created_at
    heldout["input_receipt"] = str(output_input.resolve())
    heldout["input_receipt_sha256"] = output_input_sha
    heldout["derivation"] = {
        "policy_and_identity_sets_byte_preserved": True,
        "source_manifest": {
            "path": str(args.source_heldout_manifest.resolve()),
            "sha256": source_heldout_sha,
        },
    }

    task_exclusions: dict[str, Path] = {}
    for task in derived["tasks"]:
        key = str(task.get("exclusion_key") or "")
        relative = str(task.get("exclusion_file") or "")
        if not key and not relative:
            continue
        if not key or not relative:
            raise ValueError("task has an incomplete exclusion binding")
        destination = (args.output_stage / relative).resolve()
        previous = task_exclusions.setdefault(key, destination)
        if previous != destination:
            raise ValueError(f"exclusion key maps to multiple paths: {key}")

    for key, receipt in heldout.get("exclusions", {}).items():
        if key not in task_exclusions:
            continue
        source = Path(str(receipt["path"]))
        if (
            not source.is_file()
            or source.stat().st_size != int(receipt["bytes"])
            or sha256_file(source) != receipt["sha256"]
        ):
            raise ValueError(f"source exclusion payload drift: {source}")
        destination = task_exclusions[key]
        link_or_copy(source, destination, receipt["sha256"], int(receipt["bytes"]))
        receipt["path"] = str(destination)

    missing = sorted(set(task_exclusions) - set(heldout.get("exclusions", {})))
    if missing:
        raise ValueError(f"missing required exclusion receipts: {missing}")
    write_json_atomic(output_heldout, heldout)
    print(
        json.dumps(
            {
                "ok": True,
                "input_receipt": str(output_input),
                "input_receipt_sha256": output_input_sha,
                "heldout_manifest": str(output_heldout),
                "training_tasks": len(derived["tasks"]),
                "heldout_sets": len(heldout["sets"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
