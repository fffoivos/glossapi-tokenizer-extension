#!/usr/bin/env python3
"""Freeze new-modern binary tasks plus immutable inherited replay sources."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, sha256_file, write_json_atomic


def parquet_receipts(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.rglob("*.parquet"))
    require(files, f"no Parquet files below {root}")
    return [
        {
            **file_binding(path),
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "input_relative": path.relative_to(root).as_posix(),
        }
        for path in files
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--academic-pool-receipt", type=Path, required=True)
    parser.add_argument("--hplt-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--hplt-exclusion-audit", type=Path, required=True)
    parser.add_argument("--poly-pool-receipt", type=Path, required=True)
    parser.add_argument("--parent-pool-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--bridge-builder", type=Path, required=True)
    parser.add_argument("--bridge-common", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), f"immutable binary input root exists: {args.output_dir}")
    require(len(args.repository_commit) == 40, "repository commit must be full length")
    academic = read_json(args.academic_pool_receipt)
    hplt = read_json(args.hplt_exclusion_manifest)
    hplt_audit = read_json(args.hplt_exclusion_audit)
    poly = read_json(args.poly_pool_receipt)
    parent = read_json(args.parent_pool_receipt)
    bundle = read_json(args.code_bundle_receipt)
    require(academic.get("status") == "passed", "academic pool is not passed")
    require(hplt.get("status") == "completed" and hplt_audit.get("status") == "passed", "HPLT eligibility is not audited")
    require(poly.get("status") == "passed", "polytonic pool is not passed")
    require(parent.get("status") == "completed", "parent pool corpus is not completed")
    require(bundle.get("status") == "frozen" and bundle.get("kind") == "scientific", "code bundle is not frozen scientific code")
    sys.path.insert(0, str(args.bridge_common.resolve().parent))
    from bridge_common import canonical_sha256, tokenizer_tree_receipt

    tokenizer_tree = tokenizer_tree_receipt(args.tokenizer_root)
    tokenizer_json = args.tokenizer_root / "tokenizer.json"
    require(sha256_file(tokenizer_json) == "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b", "tokenizer drift")
    inherited = []
    replay_indices: list[int] = []
    unused_parent_modern_indices: list[int] = []
    parent_tasks = sorted(parent.get("tasks", []), key=lambda row: int(row["task_index"]))
    require(parent_tasks, "parent pool has no binary tasks")
    require(
        [int(row["task_index"]) for row in parent_tasks] == list(range(len(parent_tasks))),
        "parent binary task indices are not positional and contiguous",
    )
    for row in parent_tasks:
        pool = str(row["pool"])
        task_index = int(row["task_index"])
        is_replay = pool in {"foreign_replay", "old_greek_replay"}
        (replay_indices if is_replay else unused_parent_modern_indices).append(task_index)
        inherited.append(
            {
                "task_index": task_index,
                "task_id": f"inherited-parent-{task_index:05d}",
                "kind": "training",
                "pool": pool,
                "source_name": "inherited_replay" if is_replay else "inherited_unused_parent_modern",
                "task_origin": "inherited_parent_binary",
                "selected_by_targeted_experiment": is_replay,
                "output_prefix": str(row["output_prefix"]),
                "inherited_source_manifest": dict(row["source_manifest"]),
                "documents": int(row["documents"]),
                "tokens": int(row["tokens"]),
            }
        )
    require(replay_indices, "parent pool has no replay tasks")
    next_index = len(parent_tasks)
    first_new_index = next_index
    tasks = list(inherited)
    components = (
        ("hplt_new_greek", "hplt_quarter_candidates", Path(hplt["output"]), ["source_dataset", "source_doc_id"]),
        ("non_hplt_new_greek", "academic_openarchives_phd", Path(academic["training_data"]), ["source_dataset", "source_doc_id"]),
        ("non_hplt_new_greek", "release_polytonic_sources", Path(poly["training_data"]), []),
    )
    component_files: dict[str, list[dict[str, Any]]] = {}
    for pool, source_name, root, identities in components:
        receipts = parquet_receipts(root)
        component_files[source_name] = receipts
        for file in receipts:
            relative_stem = Path(file["input_relative"]).with_suffix("").as_posix()
            tasks.append(
                {
                    "task_index": next_index,
                    "task_id": f"targeted-modern-{next_index:05d}",
                    "kind": "training",
                    "task_origin": "targeted_new_modern",
                    "selected_by_targeted_experiment": True,
                    "pool": pool,
                    "source_name": source_name,
                    "source_weight_within_pool": None,
                    "input_path": file["path"],
                    "input_relative": file["input_relative"],
                    "input_sha256": file["sha256"],
                    "input_bytes": file["bytes"],
                    "input_rows": file["rows"],
                    "text_column": "text",
                    "identity_columns": identities,
                    "identity_scope": "file",
                    "filter_field": None,
                    "filter_min": None,
                    "decontaminate_greekmmlu": False,
                    "requires_heldout_exclusion": False,
                    "exclusion_key": "",
                    "exclusion_file": "",
                    "phase_partition": None,
                    "output_prefix": f"source_binary/{source_name}/{relative_stem}_text_document",
                }
            )
            next_index += 1
    input_path = args.output_dir / "input_receipt.json"
    payload = {
        "schema_version": "full_cpt_training_bridge_input_receipt_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recipe_id": "targeted8b-academic-hplt-poly-v1",
        "repository": {
            "commit": args.repository_commit,
            "code_bundle": file_binding(args.code_bundle_receipt),
            "code_bundle_tree_sha256": bundle["tree_sha256"],
            "code_files": [file_binding(args.bridge_builder), file_binding(args.bridge_common)],
        },
        "tokenizer": {
            "root": str(args.tokenizer_root.resolve()),
            "repo_id": "fffoivos/apertus-tokenizer-extension",
            "revision": "fcd33ec09fb7d86bc072b3a4b3e890efa6473b66",
            "subfolder": "greek-modern-polytonic-tokenizer",
            "tokenizer_json_sha256": sha256_file(tokenizer_json),
            "vocab_size": 148992,
            "pad_token_id": 3,
            "tree_sha256": tokenizer_tree["tree_sha256"],
            "tree": tokenizer_tree,
        },
        "decontamination": {"applied": False, "reason": "completed_and_independently_audited_before_binary_encoding"},
        "components": {
            "academic_pool_receipt": file_binding(args.academic_pool_receipt),
            "hplt_exclusion_manifest": file_binding(args.hplt_exclusion_manifest),
            "hplt_exclusion_audit": file_binding(args.hplt_exclusion_audit),
            "poly_pool_receipt": file_binding(args.poly_pool_receipt),
            "parent_pool_receipt": file_binding(args.parent_pool_receipt),
            "files": component_files,
        },
        "task_sets": {
            "inherited_parent_all": {
                "count": len(inherited),
                "indices": list(range(len(inherited))),
                "rebuilt": False,
            },
            "inherited_replay": {
                "count": len(replay_indices),
                "indices": replay_indices,
                "rebuilt": False,
            },
            "inherited_unused_parent_modern": {
                "count": len(unused_parent_modern_indices),
                "indices": unused_parent_modern_indices,
                "rebuilt": False,
                "selected_by_targeted_experiment": False,
            },
            "new_modern": {
                "count": len(tasks) - len(inherited),
                "first_index": first_new_index,
                "last_index": next_index - 1,
                "indices_contiguous": True,
                "must_be_built": True,
            },
        },
        "tasks": sorted(tasks, key=lambda row: int(row["task_index"])),
    }
    require(
        [int(row["task_index"]) for row in payload["tasks"]] == list(range(len(payload["tasks"]))),
        "targeted binary receipt task indices are not positional and contiguous",
    )
    payload["tasks_sha256"] = canonical_sha256(payload["tasks"])
    created_root = False
    try:
        args.output_dir.mkdir(parents=True)
        created_root = True
        write_json_atomic(input_path, payload)
        heldout = {
            "schema_version": "full_cpt_training_heldouts_v1",
            "status": "completed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_receipt": str(input_path.resolve()),
            "input_receipt_sha256": sha256_file(input_path),
            "sets": [],
            "exclusions": {},
            "reason": "all GreekMMLU and frozen-validation exclusions were completed before binary encoding",
        }
        # The heldout manifest is the commit marker for this two-file binary
        # input contract; failed freezes leave no reusable partial root.
        write_json_atomic(args.output_dir / "heldout_manifest.json", heldout)
    except BaseException:
        if created_root:
            shutil.rmtree(args.output_dir, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "inherited_parent_tasks": len(inherited),
                "replay_tasks": len(replay_indices),
                "new_modern_tasks": len(tasks) - len(inherited),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
