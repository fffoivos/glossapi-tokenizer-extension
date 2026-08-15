#!/usr/bin/env python3
"""Replace unavailable raw-id heldouts with the frozen exact-text exclusion gate."""

from __future__ import annotations

import argparse
import copy
import glob
import os
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


ACQUISITION_AUXILIARY_SOURCES = {
    "apertus_overlap_drop_overlay",
    "nanochat_for_old_greek",
}


def receipt_bound_glob(paths: list[Path]) -> str:
    """Return one glob whose live expansion is exactly the receipted path set."""
    require(paths, "replay acquisition source has no selected files")
    resolved = sorted(path.resolve() for path in paths)
    parents = {path.parent for path in resolved}
    if len(parents) == 1:
        pattern = str(next(iter(parents)) / "*.parquet")
    else:
        common = Path(os.path.commonpath([str(path) for path in resolved]))
        relative_depths = {len(path.relative_to(common).parts) for path in resolved}
        require(len(relative_depths) == 1, "replay acquisition paths have heterogeneous depths")
        depth = next(iter(relative_depths))
        require(depth >= 2, "multi-directory replay selection has no directory level")
        pattern = str(common.joinpath(*(["*"] * (depth - 1)), "*.parquet"))
    expanded = sorted(Path(value).resolve() for value in glob.glob(pattern))
    require(expanded == resolved, f"receipt-derived replay glob is not exact: {pattern}")
    return pattern


def published_file_binding(actual: Path, published: Path | None) -> dict[str, object]:
    binding = file_binding(actual)
    if published is not None:
        binding["path"] = str(published.resolve())
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-recipe", type=Path, required=True)
    parser.add_argument("--published-historical-recipe", type=Path)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--replay-acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable replay recipe exists: {args.output}")
    historical = read_json(args.historical_recipe)
    validation = read_json(args.validation_receipt)
    acquisition = read_json(args.replay_acquisition_receipt)
    require(validation.get("schema_version") == "apertus_hard_h_to_g_reused_validation_panels_v1", "validation receipt schema drift")
    require(validation.get("status") == "passed", "validation receipt did not pass")
    require(acquisition.get("schema_version") == "full_cpt_replay_acquisition_receipt_v1", "replay acquisition schema drift")
    require(acquisition.get("status") == "completed", "replay acquisition did not complete")
    require(acquisition.get("output_count") == len(acquisition.get("outputs", [])) == 355, "replay acquisition output count drift")
    acquisition_by_source: dict[str, list[Path]] = {}
    for row in acquisition["outputs"]:
        source_name = str(row.get("source_name", ""))
        path = Path(str(row.get("path", "")))
        require(source_name and path.suffix == ".parquet", "invalid replay acquisition output")
        acquisition_by_source.setdefault(source_name, []).append(path)
    derived = copy.deepcopy(historical)
    removed: dict[str, str] = {}
    replaced_selectors: dict[str, dict[str, object]] = {}
    for source in derived["sources"]:
        name = str(source["name"])
        value = source.pop("drop_doc_keys_parquet", None)
        if value is not None:
            removed[name] = str(value)
        if name == "greek_replay_apertus_original":
            continue
        require(name in acquisition_by_source, f"replay source absent from acquisition receipt: {name}")
        old_selector = str(source.get("local_parquet", ""))
        new_selector = receipt_bound_glob(acquisition_by_source[name])
        source["local_parquet"] = new_selector
        replaced_selectors[name] = {
            "historical_selector": old_selector,
            "receipt_bound_selector": new_selector,
            "selected_files": len(acquisition_by_source[name]),
        }
    require(removed, "historical replay recipe had no heldout paths to replace")
    recipe_names = {str(source["name"]) for source in derived["sources"]}
    acquisition_names = set(acquisition_by_source)
    require(
        acquisition_names - recipe_names == ACQUISITION_AUXILIARY_SOURCES,
        f"unexpected replay acquisition source set: {sorted(acquisition_names - recipe_names)}",
    )
    derived["version"] = "v2_curriculum_r2_reused_validation_exact_text"
    derived["derivation"] = {
        "status": "frozen",
        "historical_recipe": published_file_binding(args.historical_recipe, args.published_historical_recipe),
        "executing_code_bundle": executing_code_bundle(),
        "removed_unavailable_raw_id_holdout_paths": removed,
        "replacement_validation_receipt": file_binding(args.validation_receipt),
        "replacement_training_exclusions": validation["training_exclusions"],
        "replay_acquisition_receipt": file_binding(args.replay_acquisition_receipt),
        "replay_selector_replacements": replaced_selectors,
        "named_reconstruction_difference": (
            "the deleted historical 13.5B replay parquet selection is replaced by the exact "
            "receipt-bound 2026-07-31 deterministic acquisition at the same frozen source weights"
        ),
        "historical_replay_document_identity_claimed": False,
        "application_stage": "after_mix_before_benchmark_scan_and_anonymization",
        "reason": "reused corrected panels expose stable exact text but not every upstream raw source key",
        "additional_deduplication": False,
    }
    write_json_atomic(args.output, derived)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
