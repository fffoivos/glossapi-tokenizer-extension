#!/usr/bin/env python3
"""Hash and freeze every local source selected by the historical replay recipe."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from decimal import Decimal
import glob
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


EXPECTED_SOURCE_NAMES = [
    "replay_t1_english_edu", "replay_t1_fra_Latn", "replay_t1_deu_Latn",
    "replay_t1_ita_Latn", "replay_t1_spa_Latn", "replay_t1_rus_Cyrl",
    "replay_t1_arb_Arab", "replay_t1_cmn_Hani", "replay_t2_tur_Latn",
    "replay_t2_bul_Cyrl", "replay_t2_srp_Cyrl", "replay_t2_ron_Latn",
    "replay_t2_heb_Hebr", "replay_t2_por_Latn", "replay_t2_pol_Latn",
    "replay_t2_nld_Latn", "replay_t2_pes_Arab", "replay_t2_ukr_Cyrl",
    "replay_t2_jpn_Jpan", "replay_t3_lat_Latn", "replay_t3_hye_Armn",
    "replay_t3_kat_Geor", "replay_t3_als_Latn", "replay_t3_mkd_Cyrl",
    "code_starcoderdata_subset", "math_finemath",
    "greek_replay_apertus_original",
]


def identity_spec(
    source_name: str,
    schema_names: list[str],
    identity_columns: list[str],
    file_sha256: str,
) -> dict[str, Any]:
    present = [name for name in identity_columns if name in schema_names]
    if present:
        return {
            "identity_mode": "source_columns",
            "present_identity_columns": present,
        }
    require(
        source_name == "math_finemath",
        f"{source_name}: no configured identity column is present",
    )
    return {
        "identity_mode": "immutable_file_sha256_plus_zero_based_row_index",
        "present_identity_columns": [],
        "synthetic_identity": {
            "file_sha256": file_sha256,
            "row_index_origin": 0,
        },
    }


def inspect_file(
    source_name: str,
    path: Path,
    text_column: str,
    identity_columns: list[str],
) -> dict[str, Any]:
    binding = file_binding(path)
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    require(text_column in names, f"{path}: text column missing: {text_column}")
    identity = identity_spec(source_name, names, identity_columns, binding["sha256"])
    return {
        **binding,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
        "text_column": text_column,
        **identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--published-recipe", type=Path)
    parser.add_argument("--bulk-recipe", type=Path, required=True)
    parser.add_argument("--replay-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable inventory exists: {args.output}")
    require(1 <= args.workers <= 32, "workers must be in [1, 32]")
    recipe = read_json(args.recipe)
    acquisition = read_json(args.replay_acquisition_receipt)
    require(acquisition.get("schema_version") == "full_cpt_replay_acquisition_receipt_v1", "replay acquisition schema drift")
    require(acquisition.get("status") == "completed", "replay acquisition did not complete")
    require(acquisition.get("output_count") == len(acquisition.get("outputs", [])) == 355, "replay acquisition output count drift")
    require(
        recipe.get("derivation", {}).get("replay_acquisition_receipt") == file_binding(args.replay_acquisition_receipt),
        "derived recipe/acquisition receipt binding drift",
    )
    acquisition_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for row in acquisition["outputs"]:
        name = str(row.get("source_name", ""))
        path = str(Path(str(row.get("path", ""))).resolve())
        require(name and path, "invalid replay acquisition output")
        require(path not in acquisition_by_source.setdefault(name, {}), f"duplicate acquisition path: {path}")
        acquisition_by_source[name][path] = row
    sources = recipe.get("sources")
    require(isinstance(sources, list), "replay recipe sources missing")
    names = [str(source.get("name", "")) for source in sources]
    require(names == EXPECTED_SOURCE_NAMES, f"replay source inventory/order drift: {names}")
    require(len(names) == len(set(names)), "duplicate replay source name")
    weight_sum = sum(Decimal(str(source["weight"])) for source in sources)
    require(abs(weight_sum - Decimal("1")) <= Decimal("0.000001"), f"replay weights do not close: {weight_sum}")

    tasks: list[tuple[str, Path, str, list[str]]] = []
    expanded_by_source: dict[str, list[str]] = {}
    for source in sources:
        name = str(source["name"])
        local = os.path.expandvars(str(source.get("local_parquet", "")))
        require(local and "$" not in local, f"{name}: replay source is not frozen locally")
        paths = [Path(value) for value in sorted(glob.glob(local))]
        require(paths, f"{name}: local parquet glob matched no files: {local}")
        if name != "greek_replay_apertus_original":
            require(name in acquisition_by_source, f"{name}: acquisition source missing")
            expected_paths = sorted(acquisition_by_source[name])
            observed_paths = sorted(str(path.resolve()) for path in paths)
            require(observed_paths == expected_paths, f"{name}: live file set differs from acquisition receipt")
        expanded_by_source[name] = [str(path.resolve()) for path in paths]
        identity_columns = list(dict.fromkeys([
            str(source.get("doc_key_field", "doc_key")), "doc_key", "doc_id", "id", "source_doc_id",
        ]))
        for path in paths:
            tasks.append((name, path, str(source.get("text_column", "text")), identity_columns))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = list(
            executor.map(
                lambda task: inspect_file(task[0], task[1], task[2], task[3]),
                tasks,
            )
        )
    files_by_source: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for task, result in zip(tasks, inspected, strict=True):
        files_by_source[task[0]].append(result)
    source_rows: dict[str, Any] = {}
    for source in sources:
        name = str(source["name"])
        files = files_by_source[name]
        if name != "greek_replay_apertus_original":
            expected = acquisition_by_source[name]
            for row in files:
                authority = expected[str(Path(row["path"]).resolve())]
                require(row["sha256"] == authority.get("sha256"), f"{name}: acquired payload SHA-256 drift")
                require(int(row["bytes"]) == int(authority.get("bytes", -1)), f"{name}: acquired payload size drift")
                require(int(row["rows"]) == int(authority.get("rows", -1)), f"{name}: acquired payload row-count drift")
        source_rows[name] = {
            "bucket": source["bucket"],
            "weight": str(source["weight"]),
            "text_column": source.get("text_column", "text"),
            "doc_key_field": source.get("doc_key_field"),
            "drop_doc_keys_parquet": source.get("drop_doc_keys_parquet"),
            "expanded_paths": expanded_by_source[name],
            "files": files,
            "file_count": len(files),
            "rows": sum(int(row["rows"]) for row in files),
            "bytes": sum(int(row["bytes"]) for row in files),
        }
    payload = {
        "schema_version": "apertus_hard_h_to_g_replay_source_inventory_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "recipe": {
            **file_binding(args.recipe),
            "path": str((args.published_recipe or args.recipe).resolve()),
        },
        "bulk_recipe": file_binding(args.bulk_recipe),
        "replay_acquisition_receipt": file_binding(args.replay_acquisition_receipt),
        "replay_acquisition_selection_plan": acquisition["selection_plan"],
        "sources": source_rows,
        "totals": {
            "sources": len(sources),
            "files": len(tasks),
            "rows_across_source_capacity": sum(int(row["rows"]) for row in inspected),
            "bytes": sum(int(row["bytes"]) for row in inspected),
        },
        "invariants": {
            "all_sources_are_local_parquet": True,
            "all_selected_files_sha256_bound": True,
            "all_rows_have_stable_identity_within_immutable_file_sets": True,
            "synthetic_file_row_identity_is_limited_to_math_finemath": True,
            "all_non_greek_replay_files_match_acquisition_receipt_exactly": True,
            "historical_replay_document_identity_claimed": False,
            "no_upstream_refetch_allowed": True,
            "recipe_source_order_frozen": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
