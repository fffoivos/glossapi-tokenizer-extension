#!/usr/bin/env python3
"""Build deterministic, source-disjoint LM-loss heldouts and exclusion lists."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from bridge_common import (
    bound_code_sha,
    canonical_sha256,
    document_key,
    read_json,
    safe_name,
    selected_by_threshold,
    sha256_file,
    utc_now,
    write_json_atomic,
)


def _load_decontaminator(receipt: Mapping[str, Any]):
    implementation = receipt["decontamination"]["implementation"]
    path = Path(implementation["path"])
    if sha256_file(path) != implementation["sha256"]:
        raise ValueError("decontamination implementation drift")
    spec = importlib.util.spec_from_file_location(
        "bridge_decontaminate_full_corpus", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    decontam = receipt["decontamination"]
    policy = decontam["policy"]
    index, _ = module.load_benchmark_index(
        Path(decontam["queries"]["path"]),
        Path(decontam["benchmark_manifest"]["path"]),
        k=int(policy["k"]),
        min_coverage=float(policy["min_coverage"]),
        minhash_threshold=float(policy["minhash_threshold"]),
        min_matched_grams=int(policy["min_matched_grams"]),
        max_gap_tokens=int(policy["max_gap_tokens"]),
    )
    return module, index


def _eligible(
    row: Mapping[str, Any], task: Mapping[str, Any], heldout: Mapping[str, Any]
) -> bool:
    field = str(task.get("filter_field") or "")
    if field:
        value = row.get(field)
        if value is None:
            return False
        minimum = task.get("filter_min")
        if minimum is not None:
            try:
                if float(value) < float(minimum):
                    return False
            except (TypeError, ValueError):
                return False
    selector_column = heldout.get("selector_column")
    if selector_column:
        value = row.get(str(selector_column))
        if heldout.get("selector_equals") is not None and str(value) != str(
            heldout["selector_equals"]
        ):
            return False
        if heldout.get("selector_regex") is not None and not re.search(
            str(heldout["selector_regex"]), str(value or "")
        ):
            return False
        if heldout.get("selector_not_regex") is not None and re.search(
            str(heldout["selector_not_regex"]), str(value or "")
        ):
            return False
    return True


def _columns(
    task: Mapping[str, Any], heldout: Mapping[str, Any], available: set[str]
) -> list[str]:
    requested = [str(task["text_column"])]
    for value in (
        *task.get("identity_columns", []),
        task.get("filter_field"),
        heldout.get("selector_column"),
        "source_dataset",
        "source_family_id",
    ):
        if value and str(value) in available and str(value) not in requested:
            requested.append(str(value))
    return requested


def _iter_docs(tasks: list[dict[str, Any]], heldout: Mapping[str, Any]):
    import pyarrow.parquet as pq

    for task in tasks:
        path = Path(task["input_path"])
        if not path.is_file() or path.stat().st_size != int(task["input_bytes"]):
            raise ValueError(f"input file-size drift while building heldout: {path}")
        parquet = pq.ParquetFile(path)
        columns = _columns(task, heldout, set(parquet.schema_arrow.names))
        row_index = 0
        for batch in parquet.iter_batches(
            columns=columns, batch_size=4096, use_threads=False
        ):
            data = batch.to_pydict()
            for offset in range(batch.num_rows):
                row = {column: data[column][offset] for column in columns}
                absolute_row = row_index + offset
                if not _eligible(row, task, heldout):
                    continue
                text = row.get(str(task["text_column"]))
                if not isinstance(text, str) or not text:
                    continue
                doc_id = document_key(
                    str(task["source_name"]),
                    str(task["input_relative"]),
                    absolute_row,
                    {
                        str(column): row.get(str(column))
                        for column in task.get("identity_columns", [])
                    },
                    identity_scope=str(task["identity_scope"]),
                )
                yield task, row, doc_id, text
            row_index += batch.num_rows


def _measure(
    tasks: list[dict[str, Any]], heldout: Mapping[str, Any]
) -> tuple[int, int]:
    documents = characters = 0
    for _, _, _, text in _iter_docs(tasks, heldout):
        documents += 1
        characters += len(text)
    return documents, characters


def _iter_group_docs(
    tasks: list[dict[str, Any]], heldout_specs: list[Mapping[str, Any]]
):
    """Scan a shared source inventory once and expose all selector columns."""

    import pyarrow.parquet as pq

    selector_columns = {
        str(spec["selector_column"])
        for spec in heldout_specs
        if spec.get("selector_column")
    }
    for task in tasks:
        path = Path(task["input_path"])
        if not path.is_file() or path.stat().st_size != int(task["input_bytes"]):
            raise ValueError(f"input file-size drift while building heldouts: {path}")
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        requested = [str(task["text_column"])]
        for value in (
            *task.get("identity_columns", []),
            task.get("filter_field"),
            "source_dataset",
            "source_family_id",
            *sorted(selector_columns),
        ):
            if value and str(value) in available and str(value) not in requested:
                requested.append(str(value))
        row_index = 0
        for batch in parquet.iter_batches(
            columns=requested, batch_size=4096, use_threads=False
        ):
            data = batch.to_pydict()
            for offset in range(batch.num_rows):
                row = {column: data[column][offset] for column in requested}
                absolute_row = row_index + offset
                if not _eligible(row, task, {}):
                    continue
                text = row.get(str(task["text_column"]))
                if not isinstance(text, str) or not text:
                    continue
                doc_id = document_key(
                    str(task["source_name"]),
                    str(task["input_relative"]),
                    absolute_row,
                    {
                        str(column): row.get(str(column))
                        for column in task.get("identity_columns", [])
                    },
                    identity_scope=str(task["identity_scope"]),
                )
                yield task, row, doc_id, text
            row_index += batch.num_rows


def _build_selector_group(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build new-Greek heldouts in two shared corpus passes."""

    if not payloads:
        return []
    tasks = payloads[0]["tasks"]
    if any(payload["tasks"] != tasks for payload in payloads[1:]):
        raise ValueError("grouped heldouts must share an exact task inventory")
    specs = [payload["heldout"] for payload in payloads]
    decontaminator = None
    if any(bool(task["decontaminate_greekmmlu"]) for task in tasks):
        decontaminator = _load_decontaminator(payloads[0]["input_receipt"])
    measurements = {
        str(spec["name"]): {"documents": 0, "characters": 0} for spec in specs
    }
    for task, row, _, text in _iter_group_docs(tasks, specs):
        for spec in specs:
            if _eligible(row, task, spec):
                bucket = measurements[str(spec["name"])]
                bucket["documents"] += 1
                bucket["characters"] += len(text)
    state: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        name = str(payload["heldout"]["name"])
        measured = measurements[name]
        if measured["documents"] <= 0 or measured["characters"] <= 0:
            raise ValueError(f"heldout {name!r} has no eligible source documents")
        target = min(
            int(payload["char_budget"]),
            max(1, math.floor(measured["characters"] * float(payload["max_fraction"]))),
        )
        output = Path(payload["output"])
        exclusion = Path(payload["exclusion"])
        output.parent.mkdir(parents=True, exist_ok=True)
        exclusion.parent.mkdir(parents=True, exist_ok=True)
        output_tmp = output.with_name(f".{output.name}.partial")
        exclusion_tmp = exclusion.with_name(f".{exclusion.name}.partial")
        output_tmp.unlink(missing_ok=True)
        exclusion_tmp.unlink(missing_ok=True)
        state[name] = {
            "payload": payload,
            "target": target,
            "output": output,
            "exclusion": exclusion,
            "output_tmp": output_tmp,
            "exclusion_tmp": exclusion_tmp,
            "out": output_tmp.open("w", encoding="utf-8"),
            "ids": exclusion_tmp.open("w", encoding="utf-8"),
            "seen": set(),
            "selected_documents": 0,
            "selected_characters": 0,
            "contaminated": 0,
        }
    try:
        for task, row, doc_id, text in _iter_group_docs(tasks, specs):
            selected_names: list[str] = []
            for spec in specs:
                name = str(spec["name"])
                item = state[name]
                if not _eligible(row, task, spec):
                    continue
                measured = measurements[name]
                if not selected_by_threshold(
                    seed=int(item["payload"]["seed"]),
                    set_name=name,
                    source_name=str(item["payload"]["selection_source_name"]),
                    doc_id=doc_id,
                    numerator=int(item["target"]),
                    denominator=int(measured["characters"]),
                ):
                    continue
                selected_names.append(name)
            if not selected_names:
                continue
            if decontaminator is not None:
                module, index = decontaminator
                action, _, _ = module.match_document(text, index)
                if action == "drop":
                    for name in selected_names:
                        state[name]["contaminated"] += 1
                    continue
            for name in selected_names:
                item = state[name]
                if doc_id in item["seen"]:
                    raise ValueError(f"duplicate heldout identity for {name}: {doc_id}")
                item["seen"].add(doc_id)
                item["out"].write(
                    json.dumps(
                        {
                            "text": text,
                            "doc_id": doc_id,
                            "source_dataset": str(
                                row.get("source_dataset") or task["source_name"]
                            ),
                            "training_source": task["source_name"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                item["ids"].write(
                    json.dumps({"doc_id": doc_id, "heldout": name}, sort_keys=True)
                    + "\n"
                )
                item["selected_documents"] += 1
                item["selected_characters"] += len(text)
    finally:
        for item in state.values():
            for key in ("out", "ids"):
                item[key].flush()
                os.fsync(item[key].fileno())
                item[key].close()
    results: list[dict[str, Any]] = []
    for name, item in state.items():
        if item["selected_documents"] <= 0:
            raise ValueError(
                f"deterministic heldout selection produced no documents: {name}"
            )
        os.replace(item["output_tmp"], item["output"])
        os.replace(item["exclusion_tmp"], item["exclusion"])
        measured = measurements[name]
        results.append(
            {
                "name": name,
                "pool": item["payload"]["pool"],
                "selection_source_name": item["payload"]["selection_source_name"],
                "source_documents": measured["documents"],
                "source_characters": measured["characters"],
                "target_characters": item["target"],
                "selected_documents": item["selected_documents"],
                "selected_characters": item["selected_characters"],
                "selected_fraction_by_characters": item["selected_characters"]
                / measured["characters"],
                "contaminated_selected_documents_dropped": item["contaminated"],
                "selection": {
                    "algorithm": "domain_separated_sha256_threshold_v1",
                    "seed": item["payload"]["seed"],
                    "threshold_numerator": item["target"],
                    "threshold_denominator": measured["characters"],
                },
                "inputs": [
                    {
                        "path": task["input_path"],
                        "sha256": task["input_sha256"],
                        "rows": task["input_rows"],
                    }
                    for task in tasks
                ],
                "output": {
                    "path": str(item["output"].resolve()),
                    "sha256": sha256_file(item["output"]),
                    "bytes": item["output"].stat().st_size,
                    "rows": item["selected_documents"],
                },
                "exclusion": {
                    "path": str(item["exclusion"].resolve()),
                    "sha256": sha256_file(item["exclusion"]),
                    "bytes": item["exclusion"].stat().st_size,
                    "rows": item["selected_documents"],
                },
            }
        )
    return results


def _build_one(payload: dict[str, Any]) -> dict[str, Any]:
    heldout = payload["heldout"]
    tasks = payload["tasks"]
    seed = int(payload["seed"])
    output = Path(payload["output"])
    exclusion = Path(payload["exclusion"])
    source_name = str(payload["selection_source_name"])
    documents, characters = _measure(tasks, heldout)
    if documents <= 0 or characters <= 0:
        raise ValueError(
            f"heldout {heldout['name']!r} has no eligible source documents"
        )
    target = min(
        int(payload["char_budget"]),
        max(1, math.floor(characters * float(payload["max_fraction"]))),
    )
    numerator, denominator = target, characters
    decontaminator = None
    if any(bool(task["decontaminate_greekmmlu"]) for task in tasks):
        decontaminator = _load_decontaminator(payload["input_receipt"])

    output.parent.mkdir(parents=True, exist_ok=True)
    exclusion.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(f".{output.name}.partial")
    exclusion_tmp = exclusion.with_name(f".{exclusion.name}.partial")
    output_tmp.unlink(missing_ok=True)
    exclusion_tmp.unlink(missing_ok=True)
    selected_docs = selected_chars = contaminated = 0
    seen_ids: set[str] = set()
    with (
        output_tmp.open("w", encoding="utf-8") as out,
        exclusion_tmp.open("w", encoding="utf-8") as ids,
    ):
        for task, row, doc_id, text in _iter_docs(tasks, heldout):
            if not selected_by_threshold(
                seed=seed,
                set_name=str(heldout["name"]),
                source_name=source_name,
                doc_id=doc_id,
                numerator=numerator,
                denominator=denominator,
            ):
                continue
            if doc_id in seen_ids:
                raise ValueError(
                    f"duplicate heldout identity for {heldout['name']}: {doc_id}"
                )
            seen_ids.add(doc_id)
            if decontaminator is not None:
                module, index = decontaminator
                action, _, _ = module.match_document(text, index)
                if action == "drop":
                    contaminated += 1
                    continue
            out.write(
                json.dumps(
                    {
                        "text": text,
                        "doc_id": doc_id,
                        "source_dataset": str(
                            row.get("source_dataset") or task["source_name"]
                        ),
                        "training_source": task["source_name"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            ids.write(
                json.dumps(
                    {"doc_id": doc_id, "heldout": heldout["name"]}, sort_keys=True
                )
                + "\n"
            )
            selected_docs += 1
            selected_chars += len(text)
        out.flush()
        os.fsync(out.fileno())
        ids.flush()
        os.fsync(ids.fileno())
    if selected_docs <= 0:
        output_tmp.unlink(missing_ok=True)
        exclusion_tmp.unlink(missing_ok=True)
        raise ValueError(
            f"deterministic heldout selection produced no documents: {heldout['name']}"
        )
    os.replace(output_tmp, output)
    os.replace(exclusion_tmp, exclusion)
    return {
        "name": heldout["name"],
        "pool": payload["pool"],
        "selection_source_name": source_name,
        "source_documents": documents,
        "source_characters": characters,
        "target_characters": target,
        "selected_documents": selected_docs,
        "selected_characters": selected_chars,
        "selected_fraction_by_characters": selected_chars / characters,
        "contaminated_selected_documents_dropped": contaminated,
        "selection": {
            "algorithm": "domain_separated_sha256_threshold_v1",
            "seed": seed,
            "threshold_numerator": numerator,
            "threshold_denominator": denominator,
        },
        "inputs": [
            {
                "path": task["input_path"],
                "sha256": task["input_sha256"],
                "rows": task["input_rows"],
            }
            for task in tasks
        ],
        "output": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": selected_docs,
        },
        "exclusion": {
            "path": str(exclusion.resolve()),
            "sha256": sha256_file(exclusion),
            "bytes": exclusion.stat().st_size,
            "rows": selected_docs,
        },
    }


def _validate_existing(
    manifest_path: Path,
    input_receipt_sha: str,
    config_sha: str,
    builder_sha: str,
    expected_exclusion_keys: set[str],
) -> bool:
    if not manifest_path.is_file():
        return False
    value = read_json(manifest_path)
    if (
        value.get("schema_version") != "full_cpt_training_heldouts_v1"
        or value.get("status") != "completed"
        or value.get("input_receipt_sha256") != input_receipt_sha
        or value.get("config_sha256") != config_sha
        or value.get("builder", {}).get("sha256") != builder_sha
    ):
        raise ValueError("existing heldout manifest is bound to different inputs")
    for row in value.get("sets", []):
        for key in ("output", "exclusion"):
            receipt = row[key]
            path = Path(receipt["path"])
            if (
                not path.is_file()
                or path.stat().st_size != receipt["bytes"]
                or sha256_file(path) != receipt["sha256"]
            ):
                raise ValueError(f"existing heldout output drift: {path}")
    if set(value.get("exclusions", {})) != expected_exclusion_keys:
        raise ValueError("existing heldout exclusion inventory drift")
    return True


def _specs(
    config: Mapping[str, Any], input_receipt: Mapping[str, Any], stage_root: Path
) -> list[dict[str, Any]]:
    heldouts = config["heldouts"]
    # Two-phase bridges may expose a single unpartitioned inventory for
    # heldout selection so a source file is not measured once per phase/pool.
    # Historical receipts have no separate inventory and retain old behavior.
    tasks = input_receipt.get("heldout_tasks", input_receipt["tasks"])
    result: list[dict[str, Any]] = []
    for pool in ("new_greek", "foreign_replay", "old_greek_replay"):
        for spec in heldouts[pool]:
            source = spec.get("source_name")
            selected_tasks = [
                task
                for task in tasks
                if task["pool"] == pool
                and (source is None or task["source_name"] == source)
            ]
            if not selected_tasks:
                raise ValueError(f"heldout {spec['name']}: no input tasks")
            output_name = (
                f"val_{spec['name']}.jsonl"
                if pool == "new_greek"
                else f"val_forget_{spec['name']}.jsonl"
            )
            exclusion_name = "new_greek" if pool == "new_greek" else str(source)
            result.append(
                {
                    "heldout": spec,
                    "pool": pool,
                    "tasks": selected_tasks,
                    "seed": config["seed"],
                    "char_budget": heldouts["char_budget_per_set"],
                    "max_fraction": heldouts["max_pool_fraction"],
                    "selection_source_name": str(source or "phase04_release"),
                    "output": str(stage_root / "heldouts" / "sets" / output_name),
                    "exclusion": str(
                        stage_root
                        / "heldouts"
                        / "exclusions"
                        / f"{safe_name(exclusion_name)}.{safe_name(spec['name'])}.jsonl"
                    ),
                    "input_receipt": input_receipt,
                }
            )
    return result


def _merge_exclusions(
    stage_root: Path, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Path]] = {}
    for row in rows:
        key = (
            "new_greek" if row["pool"] == "new_greek" else row["selection_source_name"]
        )
        grouped.setdefault(key, []).append(Path(row["exclusion"]["path"]))
    receipts: dict[str, dict[str, Any]] = {}
    for key, inputs in grouped.items():
        output = stage_root / "heldouts" / "exclusions" / f"{safe_name(key)}.jsonl"
        temporary = output.with_name(f".{output.name}.partial")
        seen: set[str] = set()
        overlapping: set[str] = set()
        duplicate_memberships = 0
        count = 0
        with temporary.open("w", encoding="utf-8") as out:
            for path in inputs:
                component_seen: set[str] = set()
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        value = json.loads(line)
                        doc_id = str(value["doc_id"])
                        if doc_id in component_seen:
                            raise ValueError(
                                f"document appears more than once within heldout component {path}: {doc_id}"
                            )
                        component_seen.add(doc_id)
                        if doc_id in seen:
                            # Evaluation slices intentionally overlap (for example,
                            # broad HPLT and historical/source-specific slices).  The
                            # training exclusion is their set union, so retain one
                            # identity while receipting every duplicate membership.
                            overlapping.add(doc_id)
                            duplicate_memberships += 1
                            continue
                        seen.add(doc_id)
                        out.write(line)
                        count += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, output)
        receipts[key] = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": count,
            "membership_rows": count + duplicate_memberships,
            "duplicate_memberships": duplicate_memberships,
            "overlapping_documents": len(overlapping),
            "merge_policy": "set_union_across_heldout_components_v1",
            "component_files": [str(path) for path in inputs],
        }
    return receipts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    input_receipt = read_json(args.input_receipt)
    config = read_json(args.config)
    if (
        input_receipt.get("schema_version")
        != "full_cpt_training_bridge_input_receipt_v1"
    ):
        raise ValueError("unsupported training-bridge input receipt")
    builder_sha = bound_code_sha(input_receipt, Path(__file__))
    bound_code_sha(input_receipt, Path(__file__).with_name("bridge_common.py"))
    if sha256_file(args.config.resolve()) != input_receipt.get("config", {}).get(
        "sha256"
    ):
        raise ValueError("heldout config differs from the frozen input receipt")
    input_sha = sha256_file(args.input_receipt.resolve())
    config_sha = sha256_file(args.config.resolve())
    expected_exclusion_keys = {
        str(task["exclusion_key"])
        for task in input_receipt["tasks"]
        if task.get("requires_heldout_exclusion")
    }
    manifest_path = args.stage_root / "heldouts" / "heldout_manifest.json"
    if _validate_existing(
        manifest_path,
        input_sha,
        config_sha,
        builder_sha,
        expected_exclusion_keys,
    ):
        print(
            json.dumps(
                {"ok": True, "resumed": True, "manifest": str(manifest_path)},
                sort_keys=True,
            )
        )
        return 0
    heldout_root = args.stage_root / "heldouts"
    for child in (heldout_root / "sets", heldout_root / "exclusions"):
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)
    specs = _specs(config, input_receipt, args.stage_root.resolve())
    new_specs = [spec for spec in specs if spec["pool"] == "new_greek"]
    other_specs = [spec for spec in specs if spec["pool"] != "new_greek"]
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(other_specs) + 1)
    ) as executor:
        grouped = executor.submit(_build_selector_group, new_specs)
        futures = [executor.submit(_build_one, spec) for spec in other_specs]
        rows = grouped.result()
        for future in futures:
            rows.append(future.result())
    exclusions = _merge_exclusions(args.stage_root.resolve(), rows)
    if set(exclusions) != expected_exclusion_keys:
        raise ValueError(
            "heldout exclusion inventory does not match frozen training tasks: "
            f"expected={sorted(expected_exclusion_keys)}, actual={sorted(exclusions)}"
        )
    payload = {
        "schema_version": "full_cpt_training_heldouts_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "input_receipt": str(args.input_receipt.resolve()),
        "input_receipt_sha256": input_sha,
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": builder_sha},
        "selection_policy": config["heldouts"]["selection"],
        "char_budget_per_set": config["heldouts"]["char_budget_per_set"],
        "max_pool_fraction": config["heldouts"]["max_pool_fraction"],
        "sets": sorted(rows, key=lambda row: (row["pool"], row["name"])),
        "exclusions": exclusions,
        "sets_sha256": canonical_sha256(rows),
    }
    write_json_atomic(manifest_path, payload)
    print(
        json.dumps(
            {"ok": True, "manifest": str(manifest_path), "sets": len(rows)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
