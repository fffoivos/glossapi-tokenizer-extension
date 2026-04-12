#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the latest dedup builder metadata bundle into a working release snapshot.")
    parser.add_argument("--working-release-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path("/home/foivos/data/glossapi_work"))
    parser.add_argument("--published-at", default=None, help="Optional YYYY-MM-DD string for dedup_metadata/latest.json")
    return parser.parse_args()


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> None:
    args = parse_args()
    working_root = args.working_release_root.resolve()
    state_root = args.state_root.resolve()
    code_root = args.code_root.resolve()

    latest_success_path = state_root / "latest_success.json"
    if not latest_success_path.exists():
        raise SystemExit(f"No latest_success.json under {state_root}")

    latest_success = json.loads(latest_success_path.read_text(encoding="utf-8"))
    run_id = str(latest_success["run_id"])
    run_root = Path(str(latest_success["run_root"])).resolve()
    builder_metadata_root = run_root / "builder_metadata"
    final_summary_path = run_root / "final" / "run_summary.json"

    if not builder_metadata_root.exists():
        raise SystemExit(f"Missing builder_metadata under {run_root}")
    if not final_summary_path.exists():
        raise SystemExit(f"Missing final run summary under {run_root}")

    release_dedup_root = working_root / "dedup_metadata"
    target_root = release_dedup_root / run_id
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    shutil.copytree(builder_metadata_root, target_root / "builder_metadata")
    copy_file(final_summary_path, target_root / "final" / "run_summary.json")

    code_files = [
        ("glossapi_corpus_cli/cli.py", "code/glossapi_corpus_cli/cli.py"),
        ("glossapi_corpus_cli/pipeline.py", "code/glossapi_corpus_cli/pipeline.py"),
        ("glossapi_corpus_cli/text_dedup.py", "code/glossapi_corpus_cli/text_dedup.py"),
        ("tests/test_pipeline.py", "code/tests/test_pipeline.py"),
        ("tests/test_text_dedup.py", "code/tests/test_text_dedup.py"),
    ]
    copied_code_files: list[str] = []
    for relative_src, relative_dest in code_files:
        src = code_root / relative_src
        if src.exists():
            copy_file(src, target_root / relative_dest)
            copied_code_files.append(relative_dest)

    readme_lines = [
        f"# Dedup Metadata {run_id}",
        "",
        "This overlay publishes builder-facing dedup metadata for the working release snapshot.",
        "",
        "Contents:",
        "",
        "- `builder_metadata/`",
        "- `final/run_summary.json`",
        "- `code/`",
        "",
        "The base corpus rows in `data/*.parquet` are unchanged. Downstream builders should load `data/*.parquet`, join against the builder metadata, and apply dedup policy at build time.",
        "",
    ]
    (target_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    latest_payload = {
        "latest_run_id": run_id,
        "path": f"dedup_metadata/{run_id}",
        "builder_metadata_root": f"dedup_metadata/{run_id}/builder_metadata",
        "code_root": f"dedup_metadata/{run_id}/code",
    }
    if args.published_at:
        latest_payload["published_at"] = args.published_at
    release_dedup_root.mkdir(parents=True, exist_ok=True)
    (release_dedup_root / "latest.json").write_text(
        json.dumps(latest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "working_release_root": str(working_root),
        "state_root": str(state_root),
        "run_id": run_id,
        "run_root": str(run_root),
        "builder_metadata_root": str(target_root / "builder_metadata"),
        "final_run_summary": str(target_root / "final" / "run_summary.json"),
        "copied_code_files": copied_code_files,
        "latest_json": str(release_dedup_root / "latest.json"),
    }
    (target_root / "publish_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
