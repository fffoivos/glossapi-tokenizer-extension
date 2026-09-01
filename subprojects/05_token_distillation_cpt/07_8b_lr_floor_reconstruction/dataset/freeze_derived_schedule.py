#!/usr/bin/env python3
"""Freeze a 13.5B two-phase schedule over completed receipt-bound binaries."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Mapping


getcontext().prec = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_payload(receipt: Mapping[str, Any], label: str) -> Path:
    path = Path(str(receipt.get("path", ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} payload missing or unsafe: {path}")
    if path.stat().st_size != int(receipt.get("bytes", -1)):
        raise ValueError(f"{label} payload size drift: {path}")
    expected = str(receipt.get("sha256", ""))
    if len(expected) != 64:
        raise ValueError(f"{label} payload has no SHA-256 binding: {path}")
    return path


def validate_shard(
    task: Mapping[str, Any], source_root: Path, input_sha: str, heldout_sha: str
) -> dict[str, Any]:
    prefix = source_root / "megatron" / str(task["output_prefix"])
    manifest_path = Path(str(prefix) + ".manifest.json")
    manifest = read_json(manifest_path)
    expected = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "task_id": task["task_id"],
        "task_sha256": canonical_sha256(task),
        "input_receipt_sha256": input_sha,
        "heldout_manifest_sha256": heldout_sha,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"training shard manifest drift ({name}): {manifest_path}")
    for label in ("bin", "idx", "dropped_ledger", "retained_ledger"):
        validate_payload(manifest["outputs"][label], f"training/{task['task_id']}/{label}")
    tokens = int(manifest["counts"]["tokens"])
    if int(manifest["outputs"]["bin"]["bytes"]) != tokens * 4:
        raise ValueError(f"token/byte accounting drift: {manifest_path}")
    return {
        **manifest,
        "task": dict(task),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
    }


def validate_heldouts(source_root: Path, heldouts: Mapping[str, Any]) -> list[dict[str, Any]]:
    new_names = {"hplt", "non_hplt", "openarchives", "greek_phd", "historical_polytonic"}
    result = []
    for row in sorted(heldouts["sets"], key=lambda item: (item["pool"], item["name"])):
        name = str(row["name"])
        stem = f"val_{name}" if name in new_names else f"val_forget_{name}"
        prefix = source_root / "megatron" / "heldout" / f"{stem}_ext_text_document"
        manifest_path = Path(str(prefix) + ".manifest.json")
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != "full_cpt_megatron_shard_v1" or manifest.get("status") != "completed":
            raise ValueError(f"heldout shard is incomplete: {manifest_path}")
        for label in ("bin", "idx"):
            validate_payload(manifest["outputs"][label], f"heldout/{name}/{label}")
        result.append(
            {
                "name": name,
                "pool": row["pool"],
                "prefix": str(prefix.resolve()),
                "tokens": int(manifest["counts"]["tokens"]),
                "documents": int(manifest["counts"]["documents"]),
                "manifest": {
                    "path": str(manifest_path.resolve()),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                },
            }
        )
    if len(result) != 12 or len({row["name"] for row in result}) != 12:
        raise ValueError("expected exactly 12 uniquely named heldout sets")
    return result


def blend_row(shard: Mapping[str, Any], pool: str, weight: Decimal) -> dict[str, Any]:
    prefix = str(Path(shard["outputs"]["bin"]["path"]).with_suffix(""))
    return {
        "task_index": int(shard["task"]["task_index"]),
        "task_id": shard["task_id"],
        "logical_pool": pool,
        "source_name": shard["source_name"],
        "prefix": prefix,
        "tokens": int(shard["counts"]["tokens"]),
        "documents": int(shard["counts"]["documents"]),
        "weight_exact": format(weight, "f"),
        "weight_cli": format(weight, ".17g"),
        "manifest": shard["manifest"],
        "payloads": {
            name: dict(shard["outputs"][name]) for name in ("bin", "idx")
        },
    }


def make_blend(
    phase_recipe: Mapping[str, Any], shards: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    source_phase = int(phase_recipe["source_phase"])
    allowed = set(phase_recipe["pool_remap"])
    selected = [
        row
        for row in shards
        if int(row["task"]["phase_partition"]["phase"]) == source_phase
        and str(row["task"]["phase_partition"]["logical_pool"]) in allowed
        and int(row["counts"]["tokens"]) > 0
    ]
    by_pool: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        pool = str(row["task"]["phase_partition"]["logical_pool"])
        by_pool[pool].append(row)
        by_source[(pool, str(row["source_name"]))].append(row)
    mix = {name: Decimal(str(value)) for name, value in phase_recipe["mix_exact"].items()}
    if set(by_pool) != set(mix) or sum(mix.values()) != Decimal(1):
        raise ValueError("selected phase pools or logical weights differ from recipe")
    result: list[dict[str, Any]] = []
    for pool in sorted(mix):
        if pool == "foreign_replay":
            source_weights: dict[str, Decimal] = {}
            for (candidate_pool, source), rows in by_source.items():
                if candidate_pool != pool:
                    continue
                values = {str(row.get("source_weight_within_pool")) for row in rows}
                if len(values) != 1:
                    raise ValueError(f"foreign replay weight drift: {source}")
                source_weights[source] = Decimal(next(iter(values)))
            source_total = sum(source_weights.values())
            for source in sorted(source_weights):
                rows = by_source[(pool, source)]
                source_tokens = sum(int(row["counts"]["tokens"]) for row in rows)
                for row in rows:
                    weight = (
                        mix[pool]
                        * source_weights[source]
                        / source_total
                        * Decimal(int(row["counts"]["tokens"]))
                        / Decimal(source_tokens)
                    )
                    result.append(blend_row(row, pool, weight))
        else:
            pool_tokens = sum(int(row["counts"]["tokens"]) for row in by_pool[pool])
            for row in by_pool[pool]:
                weight = mix[pool] * Decimal(int(row["counts"]["tokens"])) / Decimal(pool_tokens)
                result.append(blend_row(row, pool, weight))
    if abs(sum(Decimal(row["weight_exact"]) for row in result) - Decimal(1)) > Decimal("1e-50"):
        raise ValueError("physical phase blend does not sum to one")
    return sorted(result, key=lambda row: int(row["task_index"]))


def capacity_report(
    phase_recipe: Mapping[str, Any], blend: list[Mapping[str, Any]], seq: int, gbs: int
) -> dict[str, Any]:
    iterations = int(phase_recipe["iteration_end"]) - int(phase_recipe["iteration_start"])
    phase_samples = iterations * gbs
    ratio = Decimal("1.005")
    failures = []
    prefixes = []
    for row in blend:
        planned = math.ceil(Decimal(phase_samples) * Decimal(row["weight_exact"]))
        required = math.ceil(Decimal(planned) * ratio) + 1
        available = max(0, (int(row["tokens"]) - 1) // seq)
        passed = available >= required
        if not passed:
            failures.append(f"{row['task_id']}: {available} < {required}")
        prefixes.append(
            {
                "task_id": row["task_id"],
                "planned_samples": planned,
                "required_samples": required,
                "available_indexed_samples": available,
                "passed": passed,
            }
        )
    by_pool: dict[str, dict[str, int]] = defaultdict(lambda: {"tokens": 0, "planned": 0})
    for row in blend:
        pool = str(row["logical_pool"])
        by_pool[pool]["tokens"] += int(row["tokens"])
        by_pool[pool]["planned"] += math.ceil(
            Decimal(phase_samples) * Decimal(row["weight_exact"])
        )
    pools = {}
    for pool, values in sorted(by_pool.items()):
        available = max(0, (values["tokens"] - 1) // seq)
        required = math.ceil(Decimal(values["planned"]) * ratio) + 1
        passed = available >= required
        if not passed:
            failures.append(f"pool/{pool}: {available} < {required}")
        pools[pool] = {**values, "available_indexed_samples": available, "required_samples": required, "passed": passed}
    return {
        "basis": "retained_indexed_tokens_with_atomic_shard_hash_receipts",
        "minimum_capacity_ratio": "1.005",
        "boundary_samples": 1,
        "training_samples": phase_samples,
        "prefixes": prefixes,
        "pools": pools,
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    recipe_path = args.recipe.resolve()
    recipe = read_json(recipe_path)
    source_root = Path(recipe["source_stage"]["root"]).resolve()
    input_path = source_root / "input_receipt.json"
    heldout_path = source_root / "heldouts" / "heldout_manifest.json"
    input_sha = sha256_file(input_path)
    heldout_sha = sha256_file(heldout_path)
    if input_sha != recipe["source_stage"]["input_receipt_sha256"]:
        raise ValueError("source input receipt drift")
    if heldout_sha != recipe["source_stage"]["heldout_manifest_sha256"]:
        raise ValueError("source heldout manifest drift")
    inputs = read_json(input_path)
    heldouts = read_json(heldout_path)
    if inputs.get("status") != "frozen" or len(inputs.get("tasks", [])) != int(recipe["source_stage"]["training_tasks"]):
        raise ValueError("source training input receipt is incomplete")
    if heldouts.get("status") != "completed" or len(heldouts.get("sets", [])) != int(recipe["source_stage"]["heldout_sets"]):
        raise ValueError("source heldout manifest is incomplete")
    greek_shards = [
        validate_shard(task, source_root, input_sha, heldout_sha)
        for task in inputs["tasks"]
        if task["pool"] == "new_greek"
    ]
    replay_root = Path(recipe["replay_repartition"]["root"]).resolve()
    replay_input_path = replay_root / "input_receipt.json"
    replay_heldout_path = replay_root / "heldouts" / "heldout_manifest.json"
    replay_input_sha = sha256_file(replay_input_path)
    replay_heldout_sha = sha256_file(replay_heldout_path)
    if replay_input_sha != recipe["replay_repartition"]["input_receipt_sha256"]:
        raise ValueError("base replay input receipt drift")
    if replay_heldout_sha != recipe["replay_repartition"]["heldout_manifest_sha256"]:
        raise ValueError("base replay heldout manifest drift")
    replay_inputs = read_json(replay_input_path)
    replay_heldouts = read_json(replay_heldout_path)
    if replay_inputs.get("parent_input_receipt", {}).get("sha256") != input_sha:
        raise ValueError("replay repartition is derived from another source input")
    if replay_heldouts.get("input_receipt_sha256") != replay_input_sha:
        raise ValueError("replay heldouts are bound to another replay input")
    if len(replay_inputs.get("tasks", [])) != int(recipe["replay_repartition"]["expected_tasks"]):
        raise ValueError("replay repartition task-count drift")
    replay_shards = [
        validate_shard(task, replay_root, replay_input_sha, replay_heldout_sha)
        for task in replay_inputs["tasks"]
    ]
    supplement_root = Path(recipe["replay_supplements"]["root"]).resolve()
    supplement_input_path = supplement_root / "input_receipt.json"
    supplement_heldout_path = supplement_root / "heldouts" / "heldout_manifest.json"
    supplement_download_path = supplement_root / "download_receipt.json"
    supplement_input_sha = sha256_file(supplement_input_path)
    supplement_heldout_sha = sha256_file(supplement_heldout_path)
    supplement_download_sha = sha256_file(supplement_download_path)
    supplement_inputs = read_json(supplement_input_path)
    supplement_heldouts = read_json(supplement_heldout_path)
    supplement_download = read_json(supplement_download_path)
    supplement_config = recipe["replay_supplements"]
    if supplement_download_sha != supplement_config["download_receipt_sha256"]:
        raise ValueError("supplement download receipt drift")
    if supplement_input_sha != supplement_config["input_receipt_sha256"]:
        raise ValueError("supplement input receipt drift")
    if supplement_heldout_sha != supplement_config["heldout_manifest_sha256"]:
        raise ValueError("supplement heldout manifest drift")
    if supplement_inputs.get("parent_input_receipt", {}).get("sha256") != replay_input_sha:
        raise ValueError("replay supplement is derived from another base replay input")
    if supplement_inputs.get("supplement_download_receipt", {}).get("sha256") != supplement_download_sha:
        raise ValueError("replay supplement is bound to another download receipt")
    if supplement_heldouts.get("input_receipt_sha256") != supplement_input_sha:
        raise ValueError("supplement heldouts are bound to another supplement input")
    if supplement_heldouts.get("parent_heldout_manifest", {}).get("sha256") != replay_heldout_sha:
        raise ValueError("supplement heldouts are derived from another base heldout manifest")
    if (
        supplement_download.get("repo_id") != supplement_config["repo_id"]
        or supplement_download.get("revision") != supplement_config["revision"]
    ):
        raise ValueError("supplement download source drift")
    if len(supplement_download.get("files", [])) != int(recipe["replay_supplements"]["expected_files"]):
        raise ValueError("supplement download file-count drift")
    if len(supplement_inputs.get("tasks", [])) != int(recipe["replay_supplements"]["expected_tasks"]):
        raise ValueError("supplement replay task-count drift")
    expected_files = {
        (row["source_name"], row["path"]): (int(row["bytes"]), row["sha256"])
        for row in supplement_config["files"]
    }
    actual_files = {
        (row["source_name"], row["path"]): (int(row["bytes"]), row["sha256"])
        for row in supplement_download["files"]
    }
    if actual_files != expected_files:
        raise ValueError("supplement download inventory drift")
    supplement_shards = [
        validate_shard(task, supplement_root, supplement_input_sha, supplement_heldout_sha)
        for task in supplement_inputs["tasks"]
    ]
    shards = greek_shards + replay_shards + supplement_shards
    heldout_rows = validate_heldouts(source_root, heldouts)
    phases = {
        number: make_blend(recipe["phases"][f"phase_{number}"], shards)
        for number in (1, 2)
    }
    seq = int(recipe["geometry"]["sequence_length"])
    gbs = int(recipe["geometry"]["global_batch_sequences"])
    capacity = {
        str(number): capacity_report(
            recipe["phases"][f"phase_{number}"], phases[number], seq, gbs
        )
        for number in (1, 2)
    }
    failures = [failure for phase in capacity.values() for failure in phase["failures"]]
    if failures:
        raise ValueError("indexed-capacity gate failed: " + "; ".join(failures[:10]))
    output_root = args.output_root.resolve()
    env_path = output_root / "training_data.env"
    names = [row["name"] for row in heldout_rows]
    phase_paths = {
        number: " ".join(f"{row['weight_cli']} {row['prefix']}" for row in phases[number])
        for number in (1, 2)
    }
    env = "\n".join(
        [
            "# Receipt-generated. Do not edit.",
            f'FULL_CPT_TOKENIZER_DIR="{inputs["tokenizer"]["root"]}"',
            f'PHASE1_CPT_DATA_PREFIX="{phase_paths[1]}"',
            f'PHASE2_CPT_DATA_PREFIX="{phase_paths[2]}"',
            f'VAL_DATA_DIR="{source_root / "megatron" / "heldout"}"',
            f'EXTRA_VALID_SETS="{" ".join(names)}"',
            'NEW_GREEK_VALID_SETS="hplt non_hplt openarchives greek_phd historical_polytonic"',
            f'LR13_DATASET_MANIFEST="{output_root / "dataset_manifest.json"}"',
            f'LR13_SOURCE_INPUT_RECEIPT="{input_path}"',
            f'LR13_SOURCE_HELDOUT_MANIFEST="{heldout_path}"',
            "",
        ]
    )
    write_atomic(env_path, env)
    payload = {
        "schema_version": "apertus8b_lr_floor_dataset_manifest_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe": {"path": str(recipe_path), "sha256": sha256_file(recipe_path)},
        "source": {
            "root": str(source_root),
            "input_receipt": {"path": str(input_path), "sha256": input_sha},
            "heldout_manifest": {"path": str(heldout_path), "sha256": heldout_sha},
        },
        "replay_repartition": {
            "root": str(replay_root),
            "input_receipt": {"path": str(replay_input_path), "sha256": replay_input_sha},
            "heldout_manifest": {"path": str(replay_heldout_path), "sha256": replay_heldout_sha},
            "tasks": len(replay_inputs["tasks"]),
        },
        "replay_supplements": {
            "root": str(supplement_root),
            "download_receipt": {
                "path": str(supplement_download_path),
                "sha256": supplement_download_sha,
            },
            "input_receipt": {
                "path": str(supplement_input_path),
                "sha256": supplement_input_sha,
            },
            "heldout_manifest": {
                "path": str(supplement_heldout_path),
                "sha256": supplement_heldout_sha,
            },
            "files": len(supplement_download["files"]),
            "tasks": len(supplement_inputs["tasks"]),
        },
        "phases": {
            str(number): {
                "blend": phases[number],
                "blend_sha256": canonical_sha256(phases[number]),
                "weight_sum": format(sum(Decimal(row["weight_exact"]) for row in phases[number]), "f"),
                "documents": sum(int(row["documents"]) for row in phases[number]),
                "indexed_tokens": sum(int(row["tokens"]) for row in phases[number]),
            }
            for number in (1, 2)
        },
        "heldouts": heldout_rows,
        "capacity": capacity,
        "training_data_env": {
            "path": str(env_path),
            "bytes": env_path.stat().st_size,
            "sha256": sha256_file(env_path),
        },
        "invariants": {
            "source_documents_globally_deduplicated_before_heldout_split": True,
            "heldout_identities_excluded_before_binary_build": True,
            "phase_assignment_is_disjoint_by_source_receipt": True,
            "no_binary_payload_copied": True,
            "same_phase_data_prefixes_for_all_lr_tails": True,
            "indexed_capacity_at_least_1_005_plus_boundary": True,
            "base_replay_payloads_reused_without_rebuild": True,
            "supplement_downloads_revision_and_sha256_pinned": True,
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    write_json(manifest_path, payload)
    print(json.dumps({"ok": True, "manifest": str(manifest_path), "phase_1_prefixes": len(phases[1]), "phase_2_prefixes": len(phases[2])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
