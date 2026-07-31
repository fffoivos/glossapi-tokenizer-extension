#!/usr/bin/env python3
"""Freeze the exact two-phase corpus, tokenizer, replay, and code inputs.

This is intentionally an operational Clariden step: it hashes all 431 Greek
Parquets plus all replay files. It must not be run on the coordination Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
REPO_ROOT_DEFAULT = PROBE_ROOT.parents[2]
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import (  # noqa: E402
    HEX_COMMIT,
    canonical_sha256,
    read_json,
    safe_name,
    sha256_file,
    tokenizer_tree_receipt,
    utc_now,
    write_json_atomic,
)


def _load_shared_freezer():
    path = SHARED / "freeze_inputs.py"
    spec = importlib.util.spec_from_file_location("legacy_bridge_freezer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _absolute_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_decontamination(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"implementation", "queries", "benchmark_manifest", "policy"}
    if not required <= set(value):
        raise ValueError("incomplete GreekMMLU decontamination binding")
    for name in ("implementation", "queries", "benchmark_manifest"):
        receipt = value[name]
        path = Path(str(receipt.get("path", "")))
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            raise ValueError(f"GreekMMLU {name} receipt drift")
    return dict(value)


def _validate_replay_receipts(
    acquisition_path: Path,
    old_build_path: Path,
    foreign_tasks: list[Mapping[str, Any]],
    old_files: list[Mapping[str, Any]],
) -> dict[str, Any]:
    acquisition_path = acquisition_path.resolve()
    acquisition = read_json(acquisition_path)
    if (
        acquisition.get("schema_version")
        != "full_cpt_replay_acquisition_receipt_v1"
        or acquisition.get("status") != "completed"
    ):
        raise ValueError("replay acquisition is not completed")
    acquisition_impl = Path(str(acquisition.get("implementation", ""))).resolve()
    acquisition_config = Path(str(acquisition.get("config", ""))).resolve()
    if (
        not acquisition_impl.is_file()
        or sha256_file(acquisition_impl) != acquisition.get("implementation_sha256")
        or not acquisition_config.is_file()
        or sha256_file(acquisition_config) != acquisition.get("config_sha256")
    ):
        raise ValueError("replay acquisition implementation/config receipt drift")
    phase04_pin = acquisition.get("phase04_sources_config", {})
    phase04_pin_path = Path(str(phase04_pin.get("path", ""))).resolve()
    if (
        not phase04_pin_path.is_file()
        or sha256_file(phase04_pin_path) != phase04_pin.get("sha256")
    ):
        raise ValueError("replay acquisition source-pin receipt drift")
    acquired_rows = [
        row for row in acquisition.get("outputs", [])
        if row.get("role") == "foreign_replay"
    ]
    acquired = {Path(str(row["path"])).resolve(): row for row in acquired_rows}
    if len(acquired) != len(acquired_rows):
        raise ValueError("replay acquisition has duplicate output paths")
    replay_paths = {Path(str(task["input_path"])).resolve() for task in foreign_tasks}
    if set(acquired) != replay_paths:
        raise ValueError("foreign replay inventory differs from acquisition receipt")
    for path, row in acquired.items():
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"acquired replay payload drift: {path}")

    old_build_path = old_build_path.resolve()
    old_build = read_json(old_build_path)
    if (
        old_build.get("schema_version") != "full_cpt_old_greek_build_receipt_v1"
        or old_build.get("status") != "completed"
        or old_build.get("acquisition_receipt_sha256")
        != sha256_file(acquisition_path)
    ):
        raise ValueError("old-Greek replay build is not bound to this acquisition")
    old_impl = Path(str(old_build.get("implementation", ""))).resolve()
    if (
        not old_impl.is_file()
        or sha256_file(old_impl) != old_build.get("implementation_sha256")
        or old_build.get("config_sha256") != acquisition.get("config_sha256")
    ):
        raise ValueError("old-Greek implementation/config receipt drift")
    old_rows = old_build.get("outputs", [])
    old_outputs = {Path(str(row["path"])).resolve(): row for row in old_rows}
    if len(old_outputs) != len(old_rows):
        raise ValueError("old-Greek build has duplicate output paths")
    expected_old = {Path(str(row["path"])).resolve() for row in old_files}
    if set(old_outputs) != expected_old:
        raise ValueError("old-Greek replay inventory differs from build receipt")
    for path, row in old_outputs.items():
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"old-Greek replay payload drift: {path}")
    return {
        "acquisition_receipt": _absolute_receipt(acquisition_path),
        "old_greek_build_receipt": _absolute_receipt(old_build_path),
    }


def _validate_greek_release(
    recipe: Mapping[str, Any], root: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    dataset = recipe["dataset"]
    if sha256_file(manifest_path) != dataset["published_manifest_sha256"]:
        raise ValueError("published dataset manifest drift")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != "agent1_v5_deduplicated_release_manifest_v1"
        or manifest.get("status") != "passed"
        or manifest.get("repository_id") != dataset["repo_id"]
        or int(manifest.get("rows", -1)) != dataset["documents"]
    ):
        raise ValueError("published Greek release identity drift")
    expected = manifest.get("files", [])
    if len(expected) != dataset["parquet_files"]:
        raise ValueError("published Greek release file-count drift")
    expected_paths = {str(row["path"]) for row in expected}
    actual_paths = {
        path.relative_to(root).as_posix() for path in (root / "data").glob("*.parquet")
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"Greek Parquet inventory drift: missing={sorted(expected_paths-actual_paths)[:5]}, "
            f"extra={sorted(actual_paths-expected_paths)[:5]}"
        )
    required_columns = set(dataset["required_columns"])
    files: list[dict[str, Any]] = []
    for row in sorted(expected, key=lambda item: item["path"]):
        path = root / str(row["path"])
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Greek Parquet payload drift: {path}")
        metadata = pq.ParquetFile(path)
        columns = metadata.schema_arrow.names
        if not required_columns <= set(columns):
            raise ValueError(f"Greek Parquet schema drift: {path}")
        if metadata.metadata.num_rows != int(row["rows"]):
            raise ValueError(f"Greek Parquet row-count drift: {path}")
        files.append(
            {
                "path": str(path.resolve()),
                "input_relative": str(row["path"]),
                "sha256": row["sha256"],
                "bytes": int(row["bytes"]),
                "rows": int(row["rows"]),
                "columns": list(columns),
                "origin": row.get("origin"),
            }
        )
    if sum(row["rows"] for row in files) != dataset["documents"]:
        raise ValueError("Greek release rows do not reconcile")
    if sum(row["bytes"] for row in files) != dataset["parquet_bytes"]:
        raise ValueError("Greek release bytes do not reconcile")
    return files, manifest


def _task(
    *,
    task_index: int,
    pool: str,
    source_name: str,
    source_weight: str | None,
    file: Mapping[str, Any],
    text_column: str,
    identity_columns: list[str],
    identity_scope: str,
    decontaminate: bool,
    exclusion_key: str | None,
    output_prefix: str,
    phase_partition: Mapping[str, Any] | None = None,
    filter_field: str | None = None,
    filter_min: float | None = None,
) -> dict[str, Any]:
    return {
        "task_index": task_index,
        "task_id": f"train-{task_index:05d}-{safe_name(source_name)}",
        "kind": "training",
        "pool": pool,
        "source_name": source_name,
        "source_weight_within_pool": source_weight,
        "input_path": str(Path(str(file["path"])).resolve()),
        "input_relative": str(file["input_relative"]),
        "input_sha256": str(file["sha256"]),
        "input_bytes": int(file["bytes"]),
        "input_rows": int(file["rows"]),
        "text_column": text_column,
        "identity_columns": identity_columns,
        "identity_scope": identity_scope,
        "filter_field": filter_field,
        "filter_min": filter_min,
        "decontaminate_greekmmlu": decontaminate,
        "requires_heldout_exclusion": exclusion_key is not None,
        "exclusion_key": exclusion_key or "",
        "exclusion_file": (
            f"heldouts/exclusions/{safe_name(exclusion_key)}.jsonl"
            if exclusion_key
            else ""
        ),
        "phase_partition": dict(phase_partition) if phase_partition else None,
        "output_prefix": output_prefix,
    }


def _heldout_task(
    index: int,
    *,
    pool: str,
    source_name: str,
    file: Mapping[str, Any],
    text_column: str,
    identity_columns: list[str],
    identity_scope: str,
    filter_field: str | None = None,
    filter_min: float | None = None,
) -> dict[str, Any]:
    return {
        "task_index": index,
        "task_id": f"heldout-source-{index:05d}-{safe_name(source_name)}",
        "kind": "heldout_source",
        "pool": pool,
        "source_name": source_name,
        "input_path": str(Path(str(file["path"])).resolve()),
        "input_relative": str(file["input_relative"]),
        "input_sha256": str(file["sha256"]),
        "input_bytes": int(file["bytes"]),
        "input_rows": int(file["rows"]),
        "text_column": text_column,
        "identity_columns": identity_columns,
        "identity_scope": identity_scope,
        "filter_field": filter_field,
        "filter_min": filter_min,
        "decontaminate_greekmmlu": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=PROBE_ROOT / "configs" / "recipe_25b_midtraining.json")
    parser.add_argument("--legacy-bridge-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--replay-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--old-greek-build-receipt", type=Path, required=True)
    parser.add_argument("--decontamination-binding", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-repo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = read_json(args.recipe.resolve())
    legacy_config = read_json(args.legacy_bridge_config.resolve())
    greek_files, published_manifest = _validate_greek_release(
        recipe, args.dataset_root.resolve(), args.dataset_manifest.resolve()
    )
    decontamination = _validate_decontamination(
        read_json(args.decontamination_binding.resolve())
    )

    tokenizer = recipe["tokenizer"]
    tokenizer_root = args.tokenizer_root.resolve()
    if sha256_file(tokenizer_root / "tokenizer.json") != tokenizer["tokenizer_json_sha256"]:
        raise ValueError("production tokenizer JSON drift")
    from tokenizers import Tokenizer

    loaded = Tokenizer.from_file(str(tokenizer_root / "tokenizer.json"))
    if loaded.get_vocab_size(with_added_tokens=True) != tokenizer["vocab_size"]:
        raise ValueError("production tokenizer vocabulary drift")
    tokenizer_tree = tokenizer_tree_receipt(tokenizer_root)

    repo_root = args.repo_root.resolve()
    repo_commit = _git(repo_root, "rev-parse", "HEAD")
    if not HEX_COMMIT.fullmatch(repo_commit):
        raise ValueError("repository HEAD is not a full commit")
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty and not args.allow_dirty_repo:
        raise ValueError("refusing to freeze a dirty training repository")

    legacy = _load_shared_freezer()
    foreign_tasks, foreign_inventory, legacy_recipe = legacy.resolve_replay_sources(
        args.legacy_bridge_config.resolve(), legacy_config, args.scratch_root.resolve()
    )
    old = legacy_config["old_greek_replay"]
    old_files: list[dict[str, Any]] = []
    for path in legacy._source_paths(str(old["local_parquet"]), args.scratch_root.resolve()):
        receipt = legacy._parquet_receipt(path, relative=f"old_greek/{path.name}")
        receipt["path"] = str(path.resolve())
        receipt["input_relative"] = f"old_greek/{path.name}"
        old_files.append(receipt)
    replay_staging = _validate_replay_receipts(
        args.replay_acquisition_receipt,
        args.old_greek_build_receipt,
        foreign_tasks,
        old_files,
    )

    phase_path = HERE / "phase_partition.py"
    phase_implementation = {"path": str(phase_path.resolve()), "sha256": sha256_file(phase_path)}
    partition_base = {"implementation": phase_implementation, "seed": int(recipe["seed"])}
    tasks: list[dict[str, Any]] = []
    heldout_tasks: list[dict[str, Any]] = []

    for file in greek_files:
        heldout_tasks.append(
            _heldout_task(
                len(heldout_tasks),
                pool="new_greek",
                source_name="cleaned_greek_v2",
                file=file,
                text_column="text",
                identity_columns=["source_dataset", "source_doc_id"],
                identity_scope="global",
            )
        )
        for phase, logical_pool in ((1, "hplt_new_greek"), (2, "hplt_new_greek"), (2, "non_hplt_new_greek")):
            tasks.append(
                _task(
                    task_index=len(tasks),
                    pool="new_greek",
                    source_name="cleaned_greek_v2",
                    source_weight=None,
                    file=file,
                    text_column="text",
                    identity_columns=["source_dataset", "source_doc_id"],
                    identity_scope="global",
                    decontaminate=True,
                    exclusion_key="new_greek",
                    phase_partition={**partition_base, "corpus": "new_greek", "phase": phase, "logical_pool": logical_pool},
                    output_prefix=f"phase_{phase}/{logical_pool}/{Path(str(file['input_relative'])).stem}_text_document",
                )
            )

    heldout_sources = {str(row["source_name"]) for row in recipe["heldouts"]["foreign_replay"]}
    for source_task in foreign_tasks:
        file = {
            "path": source_task["input_path"],
            "input_relative": source_task["input_relative"],
            "sha256": source_task["input_sha256"],
            "bytes": source_task["input_bytes"],
            "rows": source_task["input_rows"],
        }
        if source_task["source_name"] in heldout_sources:
            heldout_tasks.append(
                _heldout_task(
                    len(heldout_tasks),
                    pool="foreign_replay",
                    source_name=source_task["source_name"],
                    file=file,
                    text_column=source_task["text_column"],
                    identity_columns=list(source_task["identity_columns"]),
                    identity_scope=source_task["identity_scope"],
                    filter_field=source_task.get("filter_field"),
                    filter_min=source_task.get("filter_min"),
                )
            )
        for phase in (1, 2):
            tasks.append(
                _task(
                    task_index=len(tasks),
                    pool="foreign_replay",
                    source_name=source_task["source_name"],
                    source_weight=source_task.get("source_weight_within_pool"),
                    file=file,
                    text_column=source_task["text_column"],
                    identity_columns=list(source_task["identity_columns"]),
                    identity_scope=source_task["identity_scope"],
                    decontaminate=True,
                    exclusion_key=source_task["source_name"] if source_task["source_name"] in heldout_sources else None,
                    phase_partition={**partition_base, "corpus": "replay", "phase": phase, "logical_pool": "foreign_replay"},
                    filter_field=source_task.get("filter_field"),
                    filter_min=source_task.get("filter_min"),
                    output_prefix=f"phase_{phase}/foreign_replay/{safe_name(source_task['source_name'])}/{Path(str(file['input_relative'])).stem}_text_document",
                )
            )

    old_name = str(old["name"])
    for file in old_files:
        heldout_tasks.append(
            _heldout_task(
                len(heldout_tasks), pool="old_greek_replay", source_name=old_name,
                file=file, text_column=str(old["text_column"]),
                identity_columns=[str(value) for value in old["identity_columns"]],
                identity_scope=str(old["identity_scope"]),
            )
        )
        for phase in (1, 2):
            tasks.append(
                _task(
                    task_index=len(tasks), pool="old_greek_replay", source_name=old_name,
                    source_weight="1", file=file, text_column=str(old["text_column"]),
                    identity_columns=[str(value) for value in old["identity_columns"]],
                    identity_scope=str(old["identity_scope"]), decontaminate=True,
                    exclusion_key=old_name,
                    phase_partition={**partition_base, "corpus": "replay", "phase": phase, "logical_pool": "old_greek_replay"},
                    output_prefix=f"phase_{phase}/old_greek_replay/{Path(str(file['input_relative'])).stem}_text_document",
                )
            )

    code_paths = [
        Path(__file__).resolve(),
        phase_path.resolve(),
        SHARED / "bridge_common.py",
        SHARED / "build_heldouts.py",
        SHARED / "build_binary_shard.py",
        HERE / "finalize_phase_bridge.py",
    ]
    code_files = [_absolute_receipt(path) for path in code_paths]
    payload = {
        "schema_version": "full_cpt_training_bridge_input_receipt_v1",
        "status": "frozen",
        "created_at": utc_now(),
        "recipe_id": recipe["recipe_id"],
        "config": {"path": str(args.recipe.resolve()), "sha256": sha256_file(args.recipe.resolve())},
        "repository": {
            "root": str(repo_root), "commit": repo_commit, "dirty_allowed": bool(args.allow_dirty_repo),
            "code_files": code_files,
        },
        "dataset": {
            "repo_id": recipe["dataset"]["repo_id"], "revision": recipe["dataset"]["revision"],
            "root": str(args.dataset_root.resolve()), "published_manifest": _absolute_receipt(args.dataset_manifest.resolve()),
            "published_manifest_schema": published_manifest["schema_version"], "files": greek_files,
        },
        "tokenizer": {
            "root": str(tokenizer_root), "repo_id": tokenizer["repo_id"], "revision": tokenizer["revision"],
            "subfolder": tokenizer["subfolder"], "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
            "vocab_size": tokenizer["vocab_size"], "tree_sha256": tokenizer_tree["tree_sha256"], "tree": tokenizer_tree,
        },
        "decontamination": decontamination,
        "replay_staging": replay_staging,
        "foreign_inventory": foreign_inventory,
        "legacy_foreign_recipe": legacy_recipe,
        "old_greek_inventory": old_files,
        "heldout_tasks": heldout_tasks,
        "tasks": tasks,
        "tasks_sha256": canonical_sha256(tasks),
        "heldout_tasks_sha256": canonical_sha256(heldout_tasks),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "training_tasks": len(tasks), "heldout_source_tasks": len(heldout_tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
