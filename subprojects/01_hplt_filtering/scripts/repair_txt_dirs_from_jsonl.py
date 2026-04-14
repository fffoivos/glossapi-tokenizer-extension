#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild category *_txt folders from canonical *.jsonl ordering."
    )
    parser.add_argument(
        "--category-samples-root",
        type=Path,
        required=True,
        help="Path to the category_samples directory.",
    )
    return parser.parse_args()


def pick_source_file(txt_dir: Path, expected_name: str, doc_id: str) -> Path:
    exact = txt_dir / expected_name
    if exact.exists():
        return exact

    prefixed_exact = txt_dir / f"x_{expected_name}"
    if prefixed_exact.exists():
        return prefixed_exact

    suffix = f"_{doc_id}.txt"
    candidates = sorted(
        path
        for path in txt_dir.glob(f"*{suffix}")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No source txt found for doc id {doc_id} in {txt_dir}")

    unprefixed = [path for path in candidates if not path.name.startswith("x_")]
    if unprefixed:
        return unprefixed[0]
    return candidates[0]


def rebuild_txt_dir(jsonl_path: Path) -> dict[str, int | str]:
    bucket = jsonl_path.stem
    txt_dir = jsonl_path.with_name(f"{bucket}_txt")
    if not txt_dir.exists():
        raise FileNotFoundError(f"Missing txt dir for {jsonl_path}: {txt_dir}")

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tmp_dir = txt_dir.with_name(f"{txt_dir.name}.__repair_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        doc_id = row.get("id") or f"row_{index:03d}"
        expected_name = f"{index:03d}_{doc_id}.txt"
        source = pick_source_file(txt_dir, expected_name, doc_id)
        (tmp_dir / expected_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    backup_dir = txt_dir.with_name(f"{txt_dir.name}.__pre_repair_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    txt_dir.rename(backup_dir)
    tmp_dir.rename(txt_dir)
    shutil.rmtree(backup_dir)

    return {
        "jsonl": str(jsonl_path),
        "txt_dir": str(txt_dir),
        "selected": len(rows),
        "final_txt_files": sum(1 for _ in txt_dir.glob("*.txt")),
    }


def main() -> None:
    args = parse_args()
    root = args.category_samples_root
    if not root.exists():
        raise FileNotFoundError(root)

    reports: list[dict[str, int | str]] = []
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for jsonl_path in sorted(category_dir.glob("*.jsonl")):
            reports.append(rebuild_txt_dir(jsonl_path))

    for report in reports:
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
