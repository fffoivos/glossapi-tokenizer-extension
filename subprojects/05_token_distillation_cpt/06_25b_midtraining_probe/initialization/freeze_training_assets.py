#!/usr/bin/env python3
"""Freeze the exact model, code, tokenizer, bridge, and launch dependencies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
REPO_ROOT_DEFAULT = PROBE_ROOT.parents[2]
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import (  # noqa: E402
    file_tree_receipt,
    read_json,
    sha256_file,
    tokenizer_tree_receipt,
    utc_now,
    write_json_atomic,
)


MEGATRON_COMMIT = "c92402e39ef3c8e69ea378a59e79059dc14541f4"
TOKENIZER_SHA = "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--init-evidence", type=Path, required=True)
    parser.add_argument("--roundtrip-verification", type=Path, required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--training-data-env", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("refusing to freeze a dirty training repository")
    repository_commit = _git(repo_root, "rev-parse", "HEAD")

    megatron = args.megatron_dir.resolve()
    if _git(megatron, "rev-parse", "HEAD") != MEGATRON_COMMIT:
        raise ValueError("Megatron commit drift")
    if _git(megatron, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("refusing to freeze a dirty Megatron tree")

    tokenizer = args.tokenizer_dir.resolve()
    if sha256_file(tokenizer / "tokenizer.json") != TOKENIZER_SHA:
        raise ValueError("production tokenizer drift")
    from tokenizers import Tokenizer

    if Tokenizer.from_file(str(tokenizer / "tokenizer.json")).get_vocab_size(with_added_tokens=True) != 148992:
        raise ValueError("production tokenizer vocabulary drift")

    evidence = read_json(args.init_evidence.resolve())
    required_evidence = {
        "schema_version": "production_polytonic_td_init_verification_v1",
        "status": "passed",
        "base_vocab_size": 148480,
        "final_vocab_size": 148992,
        "target_layer": 11,
        "existing_input_rows_exact": True,
        "existing_output_rows_exact": True,
        "non_embedding_tensors_exact": True,
        "new_rows_finite": True,
    }
    for name, value in required_evidence.items():
        if evidence.get(name) != value:
            raise ValueError(f"production init evidence drift ({name})")
    roundtrip = read_json(args.roundtrip_verification.resolve())
    for name in ("standard_max_abs_diff", "r17_max_abs_diff", "xielu_max_abs_diff", "qk_norm_max_abs_diff"):
        if float(roundtrip.get(name, -1)) != 0.0:
            raise ValueError(f"production Megatron roundtrip is not zero-drift ({name})")
    for name in ("orig_only", "trip_only", "standard_changed_over_tol", "r17_changed_over_tol", "shape_mismatches"):
        if roundtrip.get(name) != []:
            raise ValueError(f"production Megatron roundtrip inventory drift ({name})")

    checkpoint = args.init_checkpoint.resolve()
    marker = checkpoint / "latest_checkpointed_iteration.txt"
    if marker.read_text(encoding="utf-8").strip() != "release" or not (checkpoint / "release").is_dir():
        raise ValueError("production init is not a Megatron release checkpoint")
    bridge = read_json(args.bridge_manifest.resolve())
    if bridge.get("schema_version") != "greek_cpt_two_phase_bridge_manifest_v1" or bridge.get("status") != "completed":
        raise ValueError("two-phase bridge is not complete")
    if bridge.get("training_data_env", {}).get("sha256") != sha256_file(args.training_data_env.resolve()):
        raise ValueError("training data environment drift")

    dependencies = {
        "recipe": PROBE_ROOT / "configs" / "recipe_25b_midtraining.json",
        "phase_config": PROBE_ROOT / "train" / "phase_config.env",
        "phase_relative_wrapper": PROBE_ROOT / "train" / "runtime_patches" / "phase_relative_data_index.py",
        "segment_launcher": PROBE_ROOT / "train" / "submit_segment.sh",
        "segment_sbatch": PROBE_ROOT / "clariden" / "train_segment.sbatch",
        "segment_preflight": PROBE_ROOT / "clariden" / "preflight_segment.py",
        "freeze_resume_checkpoint": PROBE_ROOT / "train" / "freeze_resume_checkpoint.py",
        "smoke_phase_config": PROBE_ROOT / "train" / "smoke_phase_config.env",
        "smoke_launcher": PROBE_ROOT / "train" / "submit_smoke.sh",
        "smoke_segment_sbatch": PROBE_ROOT / "clariden" / "smoke_train_segment.sbatch",
        "smoke_verifier": PROBE_ROOT / "train" / "verify_smoke.py",
        "smoke_verify_sbatch": PROBE_ROOT / "clariden" / "verify_smoke.sbatch",
        "greekmmlu_checkpoint_watcher": PROBE_ROOT / "eval" / "watch_greekmmlu_checkpoints.sbatch",
        "trainer": repo_root / "subprojects" / "03_apertus_extension_and_embedding_adaptation" / "03_4_implementation_experiments" / "init_bakeoff" / "bakeoff_training" / "bakeoff_train.sbatch",
        "te_guard": repo_root / "subprojects" / "03_apertus_extension_and_embedding_adaptation" / "03_4_implementation_experiments" / "init_bakeoff" / "megatron_patches" / "runtime" / "pretrain_gpt_te_guard.py",
        "bridge_manifest": args.bridge_manifest.resolve(),
        "training_data_env": args.training_data_env.resolve(),
        "init_evidence": args.init_evidence.resolve(),
        "roundtrip_verification": args.roundtrip_verification.resolve(),
    }
    payload = {
        "schema_version": "greek_cpt_training_assets_receipt_v1",
        "status": "frozen",
        "created_at": utc_now(),
        "repository": {"root": str(repo_root), "commit": repository_commit},
        "init_checkpoint": {"root": str(checkpoint), "tree": file_tree_receipt(checkpoint)},
        "init_evidence": _file(args.init_evidence.resolve()),
        "roundtrip_verification": _file(args.roundtrip_verification.resolve()),
        "megatron": {"root": str(megatron), "commit": MEGATRON_COMMIT, "tree": file_tree_receipt(megatron, exclude_top_level=(".git",))},
        "tokenizer": {"root": str(tokenizer), "tokenizer_json_sha256": TOKENIZER_SHA, "vocab_size": 148992, "tree": tokenizer_tree_receipt(tokenizer)},
        "dependencies": {name: _file(path) for name, path in sorted(dependencies.items())},
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "repository_commit": repository_commit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
