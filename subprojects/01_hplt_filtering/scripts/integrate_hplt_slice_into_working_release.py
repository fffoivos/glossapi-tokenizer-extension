#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

TOKENIZER_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TOKENIZER_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace the stale HPLT slice in a working release snapshot with a corrected HPLT slice.")
    parser.add_argument("--working-release-root", type=Path, required=True)
    parser.add_argument("--hplt-release-root", type=Path, required=True)
    parser.add_argument("--old-dataset-name", default="HPLT/ell_Grek_ge8_no_mt")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path to write the integration summary JSON")
    parser.add_argument("--metadata-mode", choices=["row_counts_only", "full"], default="row_counts_only")
    return parser.parse_args()


def dataset_name_from_parquet(path: Path) -> str:
    parquet = pq.ParquetFile(path)
    batch = next(parquet.iter_batches(batch_size=1, columns=["source_dataset"]))
    row = batch.to_pylist()[0]
    return str(row["source_dataset"])


def dataset_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def main() -> None:
    args = parse_args()
    working_root = args.working_release_root.resolve()
    hplt_root = args.hplt_release_root.resolve()
    working_data = working_root / "data"
    hplt_data = hplt_root / "data"

    if not working_data.exists():
        raise SystemExit(f"Working release data dir does not exist: {working_data}")
    if not hplt_data.exists():
        raise SystemExit(f"HPLT release data dir does not exist: {hplt_data}")

    incoming_files = sorted(hplt_data.glob("*.parquet"))
    if not incoming_files:
        raise SystemExit(f"No HPLT parquet files under {hplt_data}")

    incoming_dataset_names = sorted({dataset_name_from_parquet(path) for path in incoming_files})
    if len(incoming_dataset_names) != 1:
        raise SystemExit(f"Expected exactly one incoming HPLT dataset name, got: {incoming_dataset_names}")
    incoming_dataset_name = incoming_dataset_names[0]
    incoming_stem = incoming_dataset_name.replace("/", "__")

    removed: list[dict[str, object]] = []
    removed_invalid: list[dict[str, object]] = []
    for path in sorted(working_data.glob("*.parquet")):
        try:
            dataset_name = dataset_name_from_parquet(path)
        except Exception as exc:
            if path.name.startswith(incoming_stem):
                removed_invalid.append(
                    {
                        "path": str(path),
                        "error": str(exc),
                    }
                )
                path.unlink()
                continue
            raise
        if dataset_name in {args.old_dataset_name, incoming_dataset_name}:
            removed.append(
                {
                    "path": str(path),
                    "dataset_name": dataset_name,
                    "row_count": dataset_row_count(path),
                }
            )
            path.unlink()

    copied: list[dict[str, object]] = []
    for path in incoming_files:
        dest = working_data / path.name
        shutil.copy2(path, dest)
        copied.append(
            {
                "path": str(dest),
                "dataset_name": incoming_dataset_name,
                "row_count": dataset_row_count(dest),
            }
        )

    from assemble_hf_release import build_row_counts  # noqa: PLC0415

    row_counts = build_row_counts(working_root)
    validation_summary_path = working_root / "validation_summary.csv"
    prepare_manifest_path = working_root / "prepare_manifest.json"
    manifest = None
    if args.metadata_mode == "full":
        from assemble_hf_release import build_validation  # noqa: PLC0415
        from prepare_publish_release import build_prepare_manifest  # noqa: PLC0415

        validation = build_validation(working_root)
        validation.to_csv(validation_summary_path, index=False)
        manifest = build_prepare_manifest(working_root)

    summary = {
        "working_release_root": str(working_root),
        "hplt_release_root": str(hplt_root),
        "old_dataset_name": args.old_dataset_name,
        "new_dataset_name": incoming_dataset_name,
        "metadata_mode": args.metadata_mode,
        "removed_file_count": len(removed),
        "removed_row_count_total": int(sum(int(item["row_count"]) for item in removed)),
        "removed_invalid_file_count": len(removed_invalid),
        "removed_invalid_files": removed_invalid,
        "copied_file_count": len(copied),
        "copied_row_count_total": int(sum(int(item["row_count"]) for item in copied)),
        "removed_files": removed,
        "copied_files": copied,
        "row_counts_csv": str(working_root / "row_counts.csv"),
        "validation_summary_csv": str(validation_summary_path) if validation_summary_path.exists() else None,
        "prepare_manifest_json": str(prepare_manifest_path) if prepare_manifest_path.exists() else None,
        "total_rows_after": int(row_counts["row_count"].sum()),
        "source_dataset_count_after": int(len(row_counts)),
        "prepare_manifest_totals": manifest["totals"] if manifest is not None else None,
    }

    summary_json = args.summary_json or (working_root / "hplt_integration_summary.json")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
