#!/usr/bin/env python3
"""Freeze the derived dataset, code, tokenizer, TD init, and trainer assets."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {"root": str(root), "files": files, "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-data-env", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != "apertus8b_lr_floor_dataset_manifest_v1" or dataset.get("status") != "completed":
        raise ValueError("derived dataset manifest is incomplete")
    if dataset["recipe"]["sha256"] != sha256_file(args.recipe):
        raise ValueError("derived dataset is bound to another recipe")
    runtime = recipe["runtime"]
    source_repo = Path(runtime["source_repository"]).resolve()
    megatron = Path(runtime["megatron"]).resolve()
    if git(source_repo, "rev-parse", "HEAD") != runtime["source_repository_commit"] or git(source_repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source training repository is not the clean pinned commit")
    if git(megatron, "rev-parse", "HEAD") != runtime["megatron_commit"] or git(megatron, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Megatron is not the clean pinned commit")
    tokenizer = Path(dataset["source"]["root"]).resolve()
    del tokenizer
    tokenizer_root = Path(
        json.loads(Path(dataset["source"]["input_receipt"]["path"]).read_text(encoding="utf-8"))["tokenizer"]["root"]
    ).resolve()
    tokenizer_json = tokenizer_root / "tokenizer.json"
    if sha256_file(tokenizer_json) != recipe["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer drift")
    init = recipe["initialization"]
    init_evidence = Path(init["verification"]).resolve()
    roundtrip = Path(init["roundtrip_verification"]).resolve()
    if sha256_file(init_evidence) != init["verification_sha256"] or sha256_file(roundtrip) != init["roundtrip_verification_sha256"]:
        raise ValueError("TD initialization evidence drift")
    init_value = json.loads(init_evidence.read_text(encoding="utf-8"))
    required = {"status": "passed", "target_layer": 11, "base_vocab_size": 148480, "final_vocab_size": 148992, "existing_input_rows_exact": True, "existing_output_rows_exact": True, "non_embedding_tensors_exact": True, "new_rows_finite": True}
    for name, value in required.items():
        if init_value.get(name) != value:
            raise ValueError(f"TD initialization invariant failed: {name}")
    trip_value = json.loads(roundtrip.read_text(encoding="utf-8"))
    for name in ("standard_max_abs_diff", "r17_max_abs_diff", "xielu_max_abs_diff", "qk_norm_max_abs_diff"):
        if float(trip_value.get(name, -1)) != 0.0:
            raise ValueError(f"roundtrip drift: {name}")
    checkpoint = Path(init["checkpoint"]).resolve()
    if (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
        raise ValueError("TD initialization checkpoint is not a release checkpoint")
    code_root = args.code_root.resolve()
    dependencies = [
        code_root / "configs" / "recipe_13b_lr_floor.json",
        code_root / "train" / "lr_floor_config.env",
        code_root / "train" / "submit_three_lr_tails.sh",
        code_root / "clariden" / "preflight_segment.py",
        code_root / "clariden" / "train_segment.sbatch",
        code_root / "clariden" / "freeze_checkpoint.sbatch",
        code_root / "train" / "freeze_resume_checkpoint.py",
        code_root / "train" / "runtime_patches" / "lr_floor_resume.py",
        code_root / "train" / "runtime_patches" / "megatron_extra_valid_c92402e.patch",
        code_root / "dataset" / "download_replay_supplements.py",
        code_root / "dataset" / "prepare_replay_supplements.py",
        code_root / "dataset" / "freeze_derived_schedule.py",
        code_root / "clariden" / "download_replay_supplements.sbatch",
        code_root / "clariden" / "prepare_replay_supplements.sbatch",
        code_root / "clariden" / "build_replay_supplements.sbatch",
        code_root / "clariden" / "freeze_dataset.sbatch",
        source_repo / "subprojects" / "05_token_distillation_cpt" / "06_25b_midtraining_probe" / "train" / "runtime_patches" / "phase_relative_data_index.py",
        source_repo / "subprojects" / "03_apertus_extension_and_embedding_adaptation" / "03_4_implementation_experiments" / "init_bakeoff" / "bakeoff_training" / "bakeoff_train.sbatch",
        args.dataset_manifest.resolve(),
        args.training_data_env.resolve(),
    ]
    payload = {
        "schema_version": "apertus8b_lr_floor_training_assets_v1",
        "status": "frozen",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe": file_receipt(args.recipe),
        "dataset_manifest": file_receipt(args.dataset_manifest),
        "training_data_env": file_receipt(args.training_data_env),
        "code_root": str(code_root),
        "dependencies": {path.name + f"_{index:02d}": file_receipt(path) for index, path in enumerate(dependencies)},
        "source_repository": {"root": str(source_repo), "commit": runtime["source_repository_commit"]},
        "megatron": {"root": str(megatron), "commit": runtime["megatron_commit"]},
        "tokenizer": {"root": str(tokenizer_root), "tokenizer_json": file_receipt(tokenizer_json), "vocab_size": 148992},
        "initialization": {"checkpoint": {"root": str(checkpoint), "tree": tree_receipt(checkpoint)}, "verification": file_receipt(init_evidence), "roundtrip_verification": file_receipt(roundtrip)},
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
