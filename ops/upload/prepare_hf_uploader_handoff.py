#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a working release snapshot and write a handoff manifest for the cheap HF uploader instance."
    )
    parser.add_argument("--working-release-root", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true")
    visibility.add_argument("--public", action="store_true")
    parser.add_argument("--remote-host", default="88.99.60.187")
    parser.add_argument("--remote-user", default="foivos")
    parser.add_argument("--remote-release-root", default="/srv/glossapi/data/hf_release_publish")
    parser.add_argument("--remote-repo-root", default="/srv/glossapi/repo")
    parser.add_argument("--remote-python", default="/srv/glossapi/repo/.venv/bin/python")
    parser.add_argument("--remote-detach-bin", default="glossapi-detach")
    parser.add_argument("--remote-unit-prefix", default="hf-upload")
    parser.add_argument("--publish-script-path", default="/srv/glossapi/repo/publish_hf_release.py")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-report-every", type=int, default=60)
    parser.add_argument("--hplt-dataset-name", default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--use-hf-xet-high-performance", action="store_true")
    return parser.parse_args()


def dataset_name_from_parquet(path: Path) -> str:
    parquet = pq.ParquetFile(path)
    batch = next(parquet.iter_batches(batch_size=1, columns=["source_dataset"]))
    return str(batch.to_pylist()[0]["source_dataset"])


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def resolve_visibility(args: argparse.Namespace) -> str:
    if args.public:
        return "public"
    if args.private:
        return "private"
    return "private"


def discover_hplt_dataset_name(working_root: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    summary_path = working_root / "hplt_integration_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing HPLT integration summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_name = summary.get("new_dataset_name")
    if not dataset_name:
        raise SystemExit(f"Could not read new_dataset_name from {summary_path}")
    return str(dataset_name)


def collect_release_paths(working_root: Path) -> list[str]:
    release_paths = ["data", "dedup_metadata"]
    optional_files = [
        "row_counts.csv",
        "validation_summary.csv",
        "prepare_manifest.json",
        "hplt_integration_summary.json",
    ]
    for relative in optional_files:
        if (working_root / relative).exists():
            release_paths.append(relative)
    return release_paths


def main() -> None:
    args = parse_args()
    working_root = args.working_release_root.resolve()
    handoff_root = args.handoff_root.resolve()
    data_root = working_root / "data"
    dedup_latest_path = working_root / "dedup_metadata" / "latest.json"

    if not data_root.exists():
        raise SystemExit(f"Missing working release data dir: {data_root}")
    if not dedup_latest_path.exists():
        raise SystemExit(f"Missing dedup latest pointer: {dedup_latest_path}")

    hplt_dataset_name = discover_hplt_dataset_name(working_root, args.hplt_dataset_name)
    all_parquets = sorted(data_root.glob("*.parquet"))
    if not all_parquets:
        raise SystemExit(f"No parquet files found under {data_root}")

    hplt_files: list[Path] = []
    hplt_row_count_total = 0
    for path in all_parquets:
        if dataset_name_from_parquet(path) != hplt_dataset_name:
            continue
        hplt_files.append(path)
        hplt_row_count_total += parquet_row_count(path)

    if not hplt_files:
        raise SystemExit(f"No parquet files for HPLT dataset {hplt_dataset_name} under {data_root}")

    dedup_latest = json.loads(dedup_latest_path.read_text(encoding="utf-8"))
    builder_metadata_root = working_root / str(dedup_latest["builder_metadata_root"])
    if not builder_metadata_root.exists():
        raise SystemExit(f"Missing builder metadata root from latest.json: {builder_metadata_root}")

    visibility = resolve_visibility(args)
    unit_name = f"{args.remote_unit_prefix}-{dedup_latest['latest_run_id']}"
    publish_args = [
        args.remote_python,
        args.publish_script_path,
        "--release-root",
        args.remote_release_root,
        "--repo-id",
        args.repo_id,
        "--num-workers",
        str(args.num_workers),
        "--print-report-every",
        str(args.print_report_every),
        "--public" if visibility == "public" else "--private",
    ]
    env_prefix = "HF_XET_HIGH_PERFORMANCE=1 " if args.use_hf_xet_high_performance else ""
    remote_publish_command = env_prefix + shlex.join(publish_args)
    remote_detach_command = (
        f"mkdir -p {shlex.quote(args.remote_release_root)} && "
        f"{args.remote_detach_bin} {shlex.quote(unit_name)} {shlex.quote(remote_publish_command)}"
    )

    handoff_root.mkdir(parents=True, exist_ok=True)
    manifest_path = handoff_root / "uploader_handoff.json"
    launch_command_path = handoff_root / "remote_upload_command.txt"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_release_root": str(working_root),
        "sync_paths": collect_release_paths(working_root),
        "repo_id": args.repo_id,
        "visibility": visibility,
        "dataset_server": {
            "host": args.remote_host,
            "user": args.remote_user,
            "remote_release_root": args.remote_release_root,
            "remote_repo_root": args.remote_repo_root,
            "remote_python": args.remote_python,
            "remote_publish_script_path": args.publish_script_path,
            "remote_detach_bin": args.remote_detach_bin,
            "remote_unit_name": unit_name,
        },
        "upload": {
            "num_workers": args.num_workers,
            "print_report_every": args.print_report_every,
            "use_hf_xet_high_performance": bool(args.use_hf_xet_high_performance),
            "remote_publish_command": remote_publish_command,
            "remote_detach_command": remote_detach_command,
        },
        "contracts": {
            "hplt_dataset_name": hplt_dataset_name,
            "hplt_file_count": len(hplt_files),
            "hplt_row_count_total": hplt_row_count_total,
            "dedup_latest_run_id": dedup_latest["latest_run_id"],
            "dedup_builder_metadata_root": str(builder_metadata_root),
            "dedup_path": dedup_latest.get("path"),
        },
        "paths": {
            "manifest_path": str(manifest_path),
            "remote_upload_command_path": str(launch_command_path),
        },
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    launch_command_path.write_text(remote_detach_command + "\n", encoding="utf-8")

    summary = {
        "handoff_root": str(handoff_root),
        "manifest_path": str(manifest_path),
        "remote_upload_command_path": str(launch_command_path),
        "repo_id": args.repo_id,
        "visibility": visibility,
        "hplt_dataset_name": hplt_dataset_name,
        "hplt_file_count": len(hplt_files),
        "hplt_row_count_total": hplt_row_count_total,
        "dedup_latest_run_id": dedup_latest["latest_run_id"],
    }
    summary_path = args.summary_json.resolve() if args.summary_json else handoff_root / "handoff_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
