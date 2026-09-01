#!/usr/bin/env python3
"""Prefetch or offline-verify the frozen retention task dataset cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import atomic_write_json


TASKS = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "winogrande",
    "piqa",
    "mmlu",
    "global_mmlu",
    "xnli",
    "xcopa",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_rows(task_dict: Mapping) -> list[dict]:
    rows: list[dict] = []

    def visit(value, names: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for name, child in sorted(value.items(), key=lambda item: str(item[0])):
                visit(child, names + (str(name),))
            return
        dataset = getattr(value, "dataset", None)
        if dataset is None:
            raise ValueError(f"retention task lacks a dataset: {'/'.join(names)}")
        for split, split_dataset in sorted(dataset.items()):
            fingerprint = getattr(split_dataset, "_fingerprint", None)
            if not fingerprint:
                raise ValueError(f"dataset fingerprint missing: {'/'.join(names)}/{split}")
            rows.append(
                {
                    "task": "/".join(names),
                    "split": str(split),
                    "rows": len(split_dataset),
                    "fingerprint": str(fingerprint),
                }
            )

    visit(task_dict, ())
    return sorted(rows, key=lambda row: (row["task"], row["split"]))


def payload(cache_root: Path) -> dict:
    from lm_eval.tasks import TaskManager, get_task_dict
    import lm_eval

    task_dict = get_task_dict(list(TASKS), task_manager=TaskManager())
    return {
        "schema_version": "apertus_mini_retention_dataset_cache_v1",
        "status": "completed",
        "cache_root": str(cache_root.resolve()),
        "requested_tasks": list(TASKS),
        "dataset_splits": dataset_rows(task_dict),
        "lm_eval": {
            "path": str(Path(lm_eval.__file__).resolve()),
            "sha256": sha256_file(Path(lm_eval.__file__).resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-receipt", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.expected_receipt is None):
        raise ValueError("choose exactly one of --output or --expected-receipt")
    observed = payload(args.cache_root)
    if args.expected_receipt is not None:
        expected = json.loads(args.expected_receipt.read_text())
        if observed != expected:
            raise ValueError("offline retention dataset-cache receipt drift")
        print(json.dumps({"ok": True, "mode": "offline_verify", "splits": len(observed["dataset_splits"])}))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_write_json(args.output, observed)
    print(json.dumps({"ok": True, "mode": "freeze", "splits": len(observed["dataset_splits"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
