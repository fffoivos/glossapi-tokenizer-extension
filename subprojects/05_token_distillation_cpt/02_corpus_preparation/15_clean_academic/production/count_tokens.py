#!/usr/bin/env python3
"""Plan, execute, and aggregate exact pinned-tokenizer release token counts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import atomic_write_json, load_json, sha256_file

TOKEN_PLAN_SCHEMA = "bibliography-cleaning-token-plan-v1"
TOKEN_RECEIPT_SCHEMA = "bibliography-cleaning-token-shard-receipt-v1"
TOKEN_SUMMARY_SCHEMA = "bibliography-cleaning-token-summary-v1"


def _manifest_files(manifest: dict[str, Any], root: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for row in manifest["files"]:
        rank = int(row["rank"])
        if rank in result:
            raise ValueError(f"duplicate manifest rank {rank}")
        result[rank] = {
            "rank": rank,
            "path": str(root / row["path"]),
            "sha256": row["sha256"],
            "bytes": int(row["bytes"]),
            "rows": int(row["rows"]),
        }
    return result


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    base_manifest_path = Path(args.base_manifest).resolve()
    cleaned_manifest_path = Path(args.cleaned_manifest).resolve()
    base = load_json(base_manifest_path)
    cleaned = load_json(cleaned_manifest_path)
    base_root = Path(base["root"]).resolve()
    cleaned_root = Path(cleaned["root"]).resolve()
    base_files = _manifest_files(base, base_root)
    cleaned_files = _manifest_files(cleaned, cleaned_root)
    if set(base_files) != set(cleaned_files) or base["rows"] != cleaned["rows"]:
        raise ValueError("base and cleaned release inventories are not row-compatible")
    transformed = {
        int(rank) for rank in cleaned["bibliography_cleaning"]["transformed_ranks"]
    }
    tasks = []
    for rank, row in sorted(base_files.items()):
        tasks.append({"task_id": f"base-{rank:06d}", "release": "base", **row})
    for rank in sorted(transformed):
        tasks.append(
            {
                "task_id": f"cleaned-{rank:06d}",
                "release": "cleaned",
                **cleaned_files[rank],
            }
        )
    tokenizer_path = Path(args.tokenizer_json).resolve()
    if sha256_file(tokenizer_path) != args.tokenizer_sha256:
        raise ValueError("tokenizer JSON does not match the pinned sha256")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != args.vocab_size:
        raise ValueError("tokenizer vocabulary differs from the pin")
    plan = {
        "schema_version": TOKEN_PLAN_SCHEMA,
        "status": "passed",
        "base_manifest": {
            "path": str(base_manifest_path),
            "sha256": sha256_file(base_manifest_path),
        },
        "cleaned_manifest": {
            "path": str(cleaned_manifest_path),
            "sha256": sha256_file(cleaned_manifest_path),
        },
        "tokenizer": {
            "repository_id": args.tokenizer_repo_id,
            "revision": args.tokenizer_revision,
            "tokenizer_json": str(tokenizer_path),
            "tokenizer_json_sha256": args.tokenizer_sha256,
            "vocab_size": args.vocab_size,
        },
        "transformed_ranks": sorted(transformed),
        "tasks": tasks,
    }
    atomic_write_json(args.output, plan)
    print(f"{args.output}: {len(tasks)} tasks")
    return plan


def _text_batches(path: Path, batch_size: int) -> Iterable[list[str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
        yield [value or "" for value in batch.column(0).to_pylist()]


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    if (
        plan.get("schema_version") != TOKEN_PLAN_SCHEMA
        or plan.get("status") != "passed"
    ):
        raise ValueError("token plan is not passed")
    try:
        task = plan["tasks"][args.task_index]
    except IndexError as error:
        raise ValueError(f"task index {args.task_index} is outside the plan") from error
    receipt_path = Path(args.receipt_dir) / f"{task['task_id']}.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema_version") != TOKEN_RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("plan_sha256") != sha256_file(plan_path)
            or receipt.get("task_id") != task["task_id"]
        ):
            raise ValueError(f"invalid existing token receipt: {receipt_path}")
        print(
            json.dumps({"status": "passed", "reused": True, "task_id": task["task_id"]})
        )
        return receipt

    tokenizer_record = plan["tokenizer"]
    tokenizer_path = Path(tokenizer_record["tokenizer_json"])
    if sha256_file(tokenizer_path) != tokenizer_record["tokenizer_json_sha256"]:
        raise ValueError("tokenizer bytes drifted after planning")
    shard = Path(task["path"])
    if (
        shard.is_symlink()
        or shard.stat().st_size != int(task["bytes"])
        or sha256_file(shard) != task["sha256"]
        or pq.ParquetFile(shard).metadata.num_rows != int(task["rows"])
    ):
        raise ValueError(f"shard identity failed: {shard}")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != int(
        tokenizer_record["vocab_size"]
    ):
        raise ValueError("tokenizer vocabulary drifted after planning")
    documents = 0
    text_tokens = 0
    for texts in _text_batches(shard, args.batch_size):
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        documents += len(encodings)
        text_tokens += sum(len(encoding.ids) for encoding in encodings)
    if documents != int(task["rows"]):
        raise ValueError(f"read {documents} documents, expected {task['rows']}")
    receipt = {
        "schema_version": TOKEN_RECEIPT_SCHEMA,
        "status": "passed",
        "plan_sha256": sha256_file(plan_path),
        "task_id": task["task_id"],
        "release": task["release"],
        "rank": int(task["rank"]),
        "input": {
            "path": str(shard),
            "bytes": int(task["bytes"]),
            "sha256": task["sha256"],
            "rows": documents,
        },
        "tokenizer": tokenizer_record,
        "documents": documents,
        "text_tokens": text_tokens,
        "eos_tokens": documents,
        "training_tokens": text_tokens + documents,
    }
    atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": "passed",
                "task_id": task["task_id"],
                "documents": documents,
                "training_tokens": text_tokens + documents,
            },
            sort_keys=True,
        )
    )
    return receipt


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    plan_sha = sha256_file(plan_path)
    receipt_dir = Path(args.receipt_dir)
    expected = {task["task_id"] for task in plan["tasks"]}
    actual = {path.stem for path in receipt_dir.glob("*.json")}
    if actual != expected:
        raise ValueError(
            f"token receipt set is not exact: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    by_task = {}
    for task_id in sorted(expected):
        path = receipt_dir / f"{task_id}.json"
        receipt = load_json(path)
        if (
            receipt.get("schema_version") != TOKEN_RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("plan_sha256") != plan_sha
            or receipt.get("task_id") != task_id
        ):
            raise ValueError(f"token receipt binding failed: {path}")
        by_task[task_id] = receipt

    base = {
        int(task_id.split("-")[1]): receipt
        for task_id, receipt in by_task.items()
        if task_id.startswith("base-")
    }
    cleaned_changes = {
        int(task_id.split("-")[1]): receipt
        for task_id, receipt in by_task.items()
        if task_id.startswith("cleaned-")
    }

    def totals(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        values = list(rows)
        return {
            key: sum(int(row[key]) for row in values)
            for key in ("documents", "text_tokens", "eos_tokens", "training_tokens")
        }

    base_totals = totals(base.values())
    cleaned_by_rank = {
        rank: cleaned_changes.get(rank, receipt) for rank, receipt in base.items()
    }
    for rank, receipt in cleaned_changes.items():
        if receipt["documents"] != base[rank]["documents"]:
            raise ValueError(f"rank {rank}: cleaning changed document count")
    cleaned_totals = totals(cleaned_by_rank.values())
    if base_totals["documents"] != cleaned_totals["documents"]:
        raise ValueError("base/cleaned document totals differ")
    summary = {
        "schema_version": TOKEN_SUMMARY_SCHEMA,
        "status": "passed",
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "tokenizer": plan["tokenizer"],
        "base": base_totals,
        "cleaned": cleaned_totals,
        "delta": {
            key: cleaned_totals[key] - base_totals[key]
            for key in ("text_tokens", "eos_tokens", "training_tokens")
        },
        "transformed_ranks": plan["transformed_ranks"],
        "receipts": {
            task_id: {"sha256": sha256_file(receipt_dir / f"{task_id}.json")}
            for task_id in sorted(expected)
        },
    }
    atomic_write_json(args.output, summary)
    print(
        f"{args.output}: base={base_totals['training_tokens']}, "
        f"cleaned={cleaned_totals['training_tokens']}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--base-manifest", required=True)
    plan.add_argument("--cleaned-manifest", required=True)
    plan.add_argument("--tokenizer-json", required=True)
    plan.add_argument("--tokenizer-sha256", required=True)
    plan.add_argument("--tokenizer-repo-id", required=True)
    plan.add_argument("--tokenizer-revision", required=True)
    plan.add_argument("--vocab-size", type=int, required=True)
    plan.add_argument("--output", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--plan", required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--receipt-dir", required=True)
    task.add_argument("--batch-size", type=int, default=256)
    summary = subparsers.add_parser("aggregate")
    summary.add_argument("--plan", required=True)
    summary.add_argument("--receipt-dir", required=True)
    summary.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        build_plan(args)
    elif args.command == "task":
        run_task(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
