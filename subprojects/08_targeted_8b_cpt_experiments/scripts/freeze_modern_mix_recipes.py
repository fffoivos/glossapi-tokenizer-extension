#!/usr/bin/env python3
"""Freeze exact-file recipes for the benchmark-clean HPLT and OA selections."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import tempfile

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


POOL_SPECS = {
    "hplt": {"source_name": "greek_hplt_70", "target_tokens": 8_500_000_000},
    "openarchives": {"source_name": "greek_openarchives_30", "target_tokens": 3_700_000_000},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), f"immutable recipe directory exists: {args.output_dir}")
    require(not args.output_receipt.exists(), f"immutable recipe receipt exists: {args.output_receipt}")
    source_view = read_json(args.source_view_receipt)
    require(source_view.get("schema_version") == "apertus_hard_h_to_g_source_views_v1", "source-view schema drift")
    require(source_view.get("status") == "passed", "source-view build did not pass")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", suffix=".partial", dir=args.output_dir.parent))
    recipe_summaries: dict[str, dict] = {}
    renamed = False
    try:
        for pool, spec in POOL_SPECS.items():
            output_files = source_view["pools"][pool]["output_files"]
            paths = sorted(str(Path(row["output"]["path"]).resolve()) for row in output_files)
            require(paths and len(paths) == len(set(paths)), f"{pool} source-view path inventory invalid")
            expected_bindings = {str(Path(row["output"]["path"]).resolve()): row["output"] for row in output_files}
            for path_string in paths:
                path = Path(path_string)
                require(path.is_file(), f"{pool} source-view file missing: {path}")
                observed = file_binding(path)
                expected = expected_bindings[path_string]
                require(observed["bytes"] == int(expected["bytes"]), f"{pool} source-view bytes drift: {path}")
                require(observed["sha256"] == expected["sha256"], f"{pool} source-view SHA drift: {path}")
            recipe = {
                "name": f"{pool}_only_benchmark_clean_reconstruction",
                "version": "hard_h_to_g_r2_v1",
                "seed": 20260611,
                "buckets": {"greek": 1.0},
                "sources": [{
                    "name": spec["source_name"],
                    "bucket": "greek",
                    "weight": 1.0,
                    "local_parquet_files": paths,
                    "text_column": "text",
                    "doc_key_field": "source_doc_id",
                }],
            }
            write_json_atomic(temporary_dir / f"{pool}_only.json", recipe)
            recipe_summaries[pool] = {
                "source_files": len(paths),
                "source_rows": int(source_view["pools"][pool]["counts"]["kept_rows"]),
                "target_tokens": spec["target_tokens"],
            }
        os.rename(temporary_dir, args.output_dir)
        renamed = True
        recipes = {
            pool: {**summary, "recipe": file_binding(args.output_dir / f"{pool}_only.json")}
            for pool, summary in recipe_summaries.items()
        }
        payload = {
            "schema_version": "apertus_hard_h_to_g_modern_mix_recipes_v1",
            "status": "frozen",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "executing_code_bundle": executing_code_bundle(),
            "source_view_receipt": file_binding(args.source_view_receipt),
            "recipes": recipes,
            "command_contract": {
                "seed": 20260611,
                "source_shards": 16,
                "source_shard_rule": "eligible_source_row_index_modulo_16",
                "concatenation_order": "source_shard_index_ascending",
                "selection_input_order": "source_view_output_path_lexicographic_then_parquet_row_order",
            },
            "named_reconstruction_difference": "historical mix geometry rerun over pinned v2 benchmark-clean source views; historical selected documents are not claimed",
        }
        write_json_atomic(args.output_receipt, payload)
    except BaseException:
        if renamed and args.output_dir.exists() and not args.output_receipt.exists():
            failed = args.output_dir.parent / f"_failed_{args.output_dir.name}_{os.getpid()}"
            if not failed.exists():
                os.rename(args.output_dir, failed)
        else:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
