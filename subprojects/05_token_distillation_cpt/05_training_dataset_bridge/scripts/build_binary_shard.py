#!/usr/bin/env python3
"""Build one receipt-bound Megatron indexed-dataset shard.

Training tasks read a single frozen Parquet file, exclude deterministic
heldouts, apply the Phase-04 GreekMMLU policy to replay pools, and encode every
remaining document once.  Heldout tasks read the already-frozen JSONL sets.
All payload files are written atomically and the manifest is published last,
which makes Slurm-array retries safe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from bridge_common import (
    bound_code_sha,
    canonical_sha256,
    document_key,
    iter_index_lengths,
    iter_jsonl,
    load_exclusion_ids,
    read_json,
    safe_name,
    sha256_file,
    task_output_prefix,
    tokenizer_tree_receipt,
    utc_now,
    write_index,
    write_json_atomic,
)


_TOKENIZER: Any = None
_EOS_ID: int | None = None
_VOCAB_SIZE = 0
_DECONTAM_MODULE: Any = None
_DECONTAM_INDEX: Any = None
_PHASE_MODULE: Any = None


def _install_decontaminator(receipt: Mapping[str, Any]) -> None:
    global _DECONTAM_MODULE, _DECONTAM_INDEX
    implementation = receipt["decontamination"]["implementation"]
    path = Path(implementation["path"])
    if sha256_file(path) != implementation["sha256"]:
        raise ValueError("decontamination implementation drift")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("bridge_binary_decontaminate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    decontam = receipt["decontamination"]
    policy = decontam["policy"]
    index, _ = module.load_benchmark_index(
        Path(decontam["queries"]["path"]),
        Path(decontam["benchmark_manifest"]["path"]),
        k=int(policy["k"]),
        min_coverage=float(policy["min_coverage"]),
        minhash_threshold=float(policy["minhash_threshold"]),
        min_matched_grams=int(policy["min_matched_grams"]),
        max_gap_tokens=int(policy["max_gap_tokens"]),
    )
    _DECONTAM_MODULE = module
    _DECONTAM_INDEX = index


def _worker_init(
    tokenizer_dir: str, expected_vocab: int, expected_tokenizer_sha: str
) -> None:
    global _TOKENIZER, _EOS_ID, _VOCAB_SIZE
    from tokenizers import Tokenizer

    tokenizer_json = Path(tokenizer_dir) / "tokenizer.json"
    if sha256_file(tokenizer_json) != expected_tokenizer_sha:
        raise ValueError("tokenizer bytes drift in worker")
    _TOKENIZER = Tokenizer.from_file(str(tokenizer_json))
    if _TOKENIZER.get_vocab_size(with_added_tokens=True) != expected_vocab:
        raise ValueError("tokenizer vocabulary drift in worker")
    tokenizer_config = read_json(Path(tokenizer_dir) / "tokenizer_config.json")
    eos_value = tokenizer_config.get("eos_token")
    eos_token = eos_value.get("content") if isinstance(eos_value, dict) else eos_value
    _EOS_ID = _TOKENIZER.token_to_id(eos_token) if isinstance(eos_token, str) else None
    _VOCAB_SIZE = expected_vocab
    if _EOS_ID is None:
        raise ValueError("tokenizer has no EOD/EOS id")


def _encode(record: tuple[str, str]) -> dict[str, Any]:
    doc_id, text = record
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if _DECONTAM_MODULE is not None:
        action, reason, evidence = _DECONTAM_MODULE.match_document(
            text, _DECONTAM_INDEX
        )
        if action == "drop":
            return {
                "doc_id": doc_id,
                "drop": True,
                "reason": reason,
                "text_sha256": text_sha,
                "evidence": evidence,
            }
    token_ids = _TOKENIZER.encode(text, add_special_tokens=False).ids
    token_ids.append(_EOS_ID)
    if not token_ids or min(token_ids) < 0 or max(token_ids) >= _VOCAB_SIZE:
        raise ValueError(f"token id outside frozen vocabulary for {doc_id}")
    return {
        "doc_id": doc_id,
        "drop": False,
        "token_ids": token_ids,
        "text_sha256": text_sha,
    }


def _eligible(row: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    field = str(task.get("filter_field") or "")
    if not field:
        return True
    value = row.get(field)
    if value is None:
        return False
    minimum = task.get("filter_min")
    if minimum is not None:
        try:
            return float(value) >= float(minimum)
        except (TypeError, ValueError):
            return False
    return True


def _row_policy_exclusion(
    row: Mapping[str, Any], task: Mapping[str, Any]
) -> str | None:
    """Return the frozen rule id when a row is excluded by eligibility policy."""

    rules = task.get("row_exclusion_rules", [])
    if not isinstance(rules, list):
        raise ValueError("row_exclusion_rules must be a list")
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ValueError("row exclusion rule must be an object")
        rule_id = str(rule.get("rule_id", ""))
        when_field = str(rule.get("when_field", ""))
        reject_field = str(rule.get("reject_if_true_field", ""))
        values = rule.get("when_values", [])
        if not rule_id or not when_field or not reject_field or not isinstance(values, list):
            raise ValueError("malformed row exclusion rule")
        if row.get(when_field) in values and row.get(reject_field) is True:
            return rule_id
    return None


def _load_phase_module(task: Mapping[str, Any]):
    """Load the receipt-bound optional two-phase assignment implementation."""

    global _PHASE_MODULE
    spec_value = task.get("phase_partition")
    if not isinstance(spec_value, Mapping):
        return None
    implementation = spec_value.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("phase-partition task has no implementation receipt")
    path = Path(str(implementation.get("path", "")))
    if not path.is_file() or sha256_file(path) != implementation.get("sha256"):
        raise ValueError("phase-partition implementation drift")
    if _PHASE_MODULE is None:
        module_spec = importlib.util.spec_from_file_location(
            "bridge_phase_partition", path
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        _PHASE_MODULE = module
    return _PHASE_MODULE


def _in_selected_phase(
    row: Mapping[str, Any], doc_id: str, task: Mapping[str, Any], module: Any
) -> bool:
    value = task.get("phase_partition")
    if not isinstance(value, Mapping):
        return True
    phase = int(value["phase"])
    seed = int(value["seed"])
    corpus = str(value["corpus"])
    if corpus == "new_greek":
        assignment = module.assign_new_greek(
            seed=seed,
            source_dataset=row.get("source_dataset"),
            source_doc_id=row.get("source_doc_id"),
        )
        expected_pool = value.get("logical_pool")
        if expected_pool and assignment.logical_pool != expected_pool:
            return False
    elif corpus == "replay":
        assignment = module.assign_replay(
            seed=seed,
            logical_pool=str(value["logical_pool"]),
            document_id=doc_id,
        )
    else:
        raise ValueError(f"unsupported phase-partition corpus: {corpus}")
    return int(assignment.phase) == phase


def _iter_parquet_records(
    task: Mapping[str, Any], exclusions: set[str], counters: dict[str, int]
):
    import pyarrow.parquet as pq

    path = Path(task["input_path"])
    parquet = pq.ParquetFile(path)
    columns = [str(task["text_column"])]
    partition_module = _load_phase_module(task)
    phase_columns = (
        ("source_dataset", "source_doc_id")
        if partition_module is not None
        and task.get("phase_partition", {}).get("corpus") == "new_greek"
        else ()
    )
    for value in (
        *task.get("identity_columns", []),
        task.get("filter_field"),
        *phase_columns,
    ):
        if value and str(value) not in columns:
            columns.append(str(value))
    for rule in task.get("row_exclusion_rules", []):
        for key in ("when_field", "reject_if_true_field"):
            value = str(rule[key])
            if value not in columns:
                columns.append(value)
    row_index = 0
    for batch in parquet.iter_batches(
        columns=columns, batch_size=4096, use_threads=False
    ):
        data = batch.to_pydict()
        for offset in range(batch.num_rows):
            counters["input_rows"] += 1
            row = {column: data[column][offset] for column in columns}
            absolute_row = row_index + offset
            if not _eligible(row, task):
                counters["filtered_rows"] += 1
                continue
            text = row.get(str(task["text_column"]))
            if not isinstance(text, str) or not text:
                counters["empty_rows"] += 1
                continue
            doc_id = document_key(
                str(task["source_name"]),
                str(task["input_relative"]),
                absolute_row,
                {
                    str(column): row.get(str(column))
                    for column in task.get("identity_columns", [])
                },
                identity_scope=str(task["identity_scope"]),
            )
            if partition_module is not None and not _in_selected_phase(
                row, doc_id, task, partition_module
            ):
                counters["phase_excluded_rows"] += 1
                continue
            policy_rule = _row_policy_exclusion(row, task)
            if policy_rule is not None:
                counters["policy_excluded_rows"] = (
                    counters.get("policy_excluded_rows", 0) + 1
                )
                continue
            if doc_id in exclusions:
                counters["heldout_rows"] += 1
                continue
            counters["candidate_rows"] += 1
            yield doc_id, text
        row_index += batch.num_rows


def _iter_heldout_records(task: Mapping[str, Any], counters: dict[str, int]):
    for row in iter_jsonl(Path(task["input_path"])):
        counters["input_rows"] += 1
        text = row.get("text")
        doc_id = row.get("doc_id")
        if not isinstance(text, str) or not text or doc_id is None:
            raise ValueError(f"malformed heldout row in {task['input_path']}")
        counters["candidate_rows"] += 1
        yield str(doc_id), text


def _heldout_tasks(heldouts: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(
        sorted(heldouts["sets"], key=lambda item: (item["pool"], item["name"]))
    ):
        stem = (
            f"val_{row['name']}"
            if row["pool"] == "new_greek"
            else f"val_forget_{row['name']}"
        )
        result.append(
            {
                "task_index": index,
                "task_id": f"heldout-{index:02d}-{safe_name(str(row['name']))}",
                "kind": "heldout",
                "pool": row["pool"],
                "source_name": row["selection_source_name"],
                "heldout_name": row["name"],
                "input_path": row["output"]["path"],
                "input_relative": Path(row["output"]["path"]).name,
                "input_sha256": row["output"]["sha256"],
                "input_bytes": row["output"]["bytes"],
                "input_rows": row["output"]["rows"],
                "decontaminate_greekmmlu": False,
                "identity_scope": "global",
                "identity_columns": ["doc_id"],
                "requires_heldout_exclusion": False,
                "exclusion_key": "",
                "exclusion_file": "",
                "output_prefix": f"heldout/{stem}_ext_text_document",
            }
        )
    return result


def _manifest_path(prefix: Path) -> Path:
    return Path(str(prefix) + ".manifest.json")


def _validate_resume(
    manifest_path: Path,
    *,
    task: Mapping[str, Any],
    input_receipt_sha: str,
    heldout_manifest_sha: str,
    tokenizer_tree_sha: str,
    stage_root: Path,
) -> bool:
    if not manifest_path.is_file():
        return False
    value = read_json(manifest_path)
    expected = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "task_id": task["task_id"],
        "input_receipt_sha256": input_receipt_sha,
        "heldout_manifest_sha256": heldout_manifest_sha,
        "tokenizer_tree_sha256": tokenizer_tree_sha,
        "task_sha256": canonical_sha256(task),
        "task_index": int(task["task_index"]),
        "kind": str(task["kind"]),
        "pool": str(task["pool"]),
        "source_name": str(task["source_name"]),
        "heldout_name": task.get("heldout_name"),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"existing binary manifest binding drift ({key}): {manifest_path}"
            )
    if value.get("input") != {
        "path": str(Path(task["input_path"]).resolve()),
        "sha256": task["input_sha256"],
        "bytes": task["input_bytes"],
        "rows": task["input_rows"],
    }:
        raise ValueError(f"existing binary input binding drift: {manifest_path}")
    expected_prefix = task_output_prefix(stage_root, task)
    if Path(str(value.get("output_prefix", ""))).resolve() != expected_prefix.resolve():
        raise ValueError(f"existing binary output-prefix drift: {manifest_path}")
    for key in ("bin", "idx", "dropped_ledger", "retained_ledger"):
        receipt = value["outputs"][key]
        path = Path(receipt["path"])
        if (
            not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"existing binary payload drift: {path}")
    sequences, documents, tokens = iter_index_lengths(
        Path(value["outputs"]["idx"]["path"])
    )
    if (sequences, documents, tokens) != (
        value["counts"]["documents"],
        value["counts"]["document_index_entries"],
        value["counts"]["tokens"],
    ):
        raise ValueError(f"existing binary index accounting drift: {manifest_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-index", type=int)
    group.add_argument("--heldout-index", type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.chunksize < 1:
        raise ValueError("--workers and --chunksize must be positive")
    input_receipt = read_json(args.input_receipt)
    heldouts = read_json(args.heldout_manifest)
    if (
        input_receipt.get("schema_version")
        != "full_cpt_training_bridge_input_receipt_v1"
    ):
        raise ValueError("unsupported input receipt")
    builder_sha = bound_code_sha(input_receipt, Path(__file__))
    bound_code_sha(input_receipt, Path(__file__).with_name("bridge_common.py"))
    if (
        heldouts.get("schema_version") != "full_cpt_training_heldouts_v1"
        or heldouts.get("status") != "completed"
    ):
        raise ValueError("heldouts are not completed")
    input_receipt_sha = sha256_file(args.input_receipt)
    heldout_manifest_sha = sha256_file(args.heldout_manifest)
    if heldouts.get("input_receipt_sha256") != input_receipt_sha:
        raise ValueError("heldouts are bound to a different input receipt")
    kind = "training" if args.task_index is not None else "heldout"
    tasks = input_receipt["tasks"] if kind == "training" else _heldout_tasks(heldouts)
    index = args.task_index if args.task_index is not None else args.heldout_index
    if index is None or index < 0 or index >= len(tasks):
        raise ValueError(f"{kind} task index is outside [0,{len(tasks)})")
    task = tasks[index]
    input_path = Path(task["input_path"])
    if not input_path.is_file() or input_path.stat().st_size != int(
        task["input_bytes"]
    ):
        raise ValueError(f"input size drift: {input_path}")
    if sha256_file(input_path) != task["input_sha256"]:
        raise ValueError(f"input checksum drift: {input_path}")

    tokenizer = input_receipt["tokenizer"]
    tokenizer_dir = Path(tokenizer["root"])
    if (
        sha256_file(tokenizer_dir / "tokenizer.json")
        != tokenizer["tokenizer_json_sha256"]
    ):
        raise ValueError("tokenizer JSON drift")
    if tokenizer_tree_receipt(tokenizer_dir)["tree_sha256"] != tokenizer["tree_sha256"]:
        raise ValueError("tokenizer directory tree drift")
    prefix = task_output_prefix(args.stage_root.resolve(), task)
    manifest_path = _manifest_path(prefix)
    if _validate_resume(
        manifest_path,
        task=task,
        input_receipt_sha=input_receipt_sha,
        heldout_manifest_sha=heldout_manifest_sha,
        tokenizer_tree_sha=tokenizer["tree_sha256"],
        stage_root=args.stage_root.resolve(),
    ):
        print(
            json.dumps(
                {"ok": True, "resumed": True, "task": task["task_id"]}, sort_keys=True
            )
        )
        return 0

    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = Path(str(prefix) + ".bin")
    idx_path = Path(str(prefix) + ".idx")
    dropped_path = Path(str(prefix) + ".dropped.jsonl")
    retained_path = Path(str(prefix) + ".retained.jsonl")
    for path in (bin_path, idx_path, dropped_path, retained_path):
        if path.is_symlink():
            raise ValueError(f"refusing to replace generated symlink: {path}")
        path.unlink(missing_ok=True)
    bin_tmp = Path(str(bin_path) + ".partial")
    idx_tmp = Path(str(idx_path) + ".partial")
    dropped_tmp = Path(str(dropped_path) + ".partial")
    retained_tmp = Path(str(retained_path) + ".partial")
    for path in (bin_tmp, idx_tmp, dropped_tmp, retained_tmp):
        path.unlink(missing_ok=True)

    exclusions: set[str] = set()
    exclusion_binding: dict[str, Any] = {"required": False}
    if kind == "training" and task.get("requires_heldout_exclusion"):
        if not task.get("exclusion_file") or not task.get("exclusion_key"):
            raise ValueError("task requires a heldout exclusion but binds no file/key")
        exclusion_path = args.stage_root / str(task["exclusion_file"])
        key = str(task["exclusion_key"])
        exclusion_receipt = heldouts.get("exclusions", {}).get(key)
        if not isinstance(exclusion_receipt, dict):
            raise ValueError(f"required heldout exclusion is absent: {key}")
        if Path(exclusion_receipt["path"]).resolve() != exclusion_path.resolve():
            raise ValueError("heldout exclusion path drift")
        if not exclusion_path.is_file() or exclusion_path.is_symlink():
            raise FileNotFoundError(exclusion_path)
        if exclusion_path.stat().st_size != int(exclusion_receipt["bytes"]):
            raise ValueError("heldout exclusion size drift")
        if sha256_file(exclusion_path) != exclusion_receipt["sha256"]:
            raise ValueError("heldout exclusion checksum drift")
        exclusions = load_exclusion_ids(exclusion_path)
        if len(exclusions) != int(exclusion_receipt["rows"]):
            raise ValueError("heldout exclusion row/identity accounting drift")
        exclusion_binding = {
            "required": True,
            "key": key,
            "path": str(exclusion_path.resolve()),
            "sha256": exclusion_receipt["sha256"],
            "bytes": exclusion_receipt["bytes"],
            "rows": exclusion_receipt["rows"],
        }
    elif kind == "training" and (
        task.get("exclusion_file") or task.get("exclusion_key")
    ):
        raise ValueError("optional exclusions are forbidden; task binding is ambiguous")

    if bool(task["decontaminate_greekmmlu"]):
        _install_decontaminator(input_receipt)
    counters = {
        "input_rows": 0,
        "filtered_rows": 0,
        "empty_rows": 0,
        "heldout_rows": 0,
        "phase_excluded_rows": 0,
        "candidate_rows": 0,
        "contaminated_rows": 0,
        "documents": 0,
        "tokens": 0,
    }
    records: Iterable[tuple[str, str]]
    if kind == "training":
        records = _iter_parquet_records(task, exclusions, counters)
    else:
        records = _iter_heldout_records(task, counters)
    try:
        context = mp.get_context("fork")
    except ValueError as exc:  # pragma: no cover - Clariden is Linux/fork.
        raise RuntimeError(
            "the receipt-bound builder requires a fork-capable CPU node"
        ) from exc
    lengths: list[int] = []
    with (
        context.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(
                str(tokenizer_dir),
                int(tokenizer["vocab_size"]),
                tokenizer["tokenizer_json_sha256"],
            ),
        ) as pool,
        bin_tmp.open("wb") as binary,
        dropped_tmp.open("w", encoding="utf-8") as dropped,
        retained_tmp.open("w", encoding="utf-8") as retained,
    ):
        for result in pool.imap(_encode, records, chunksize=args.chunksize):
            if result["drop"]:
                counters["contaminated_rows"] += 1
                dropped.write(
                    json.dumps(
                        {
                            "doc_id": result["doc_id"],
                            "reason": result["reason"],
                            "text_sha256": result["text_sha256"],
                            "evidence": result["evidence"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue
            values = np.asarray(result["token_ids"], dtype=np.int32)
            binary.write(values.tobytes(order="C"))
            retained.write(
                json.dumps(
                    {
                        "doc_id": result["doc_id"],
                        "text_sha256": result["text_sha256"],
                        "tokens": int(values.size),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            lengths.append(int(values.size))
            counters["documents"] += 1
            counters["tokens"] += int(values.size)
        binary.flush()
        os.fsync(binary.fileno())
        dropped.flush()
        os.fsync(dropped.fileno())
        retained.flush()
        os.fsync(retained.fileno())
    sequences, document_entries, exact_tokens = write_index(idx_tmp, lengths)
    if sequences != counters["documents"] or exact_tokens != counters["tokens"]:
        raise RuntimeError("Megatron index accounting does not reconcile")
    if bin_tmp.stat().st_size != exact_tokens * 4:
        raise RuntimeError("Megatron binary byte count does not reconcile")
    if (
        counters["candidate_rows"]
        != counters["documents"] + counters["contaminated_rows"]
    ):
        raise RuntimeError("candidate document accounting does not reconcile")
    os.replace(bin_tmp, bin_path)
    os.replace(idx_tmp, idx_path)
    os.replace(dropped_tmp, dropped_path)
    os.replace(retained_tmp, retained_path)
    builder_path = Path(__file__).resolve()
    payload = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "task_id": task["task_id"],
        "task_sha256": canonical_sha256(task),
        "task_index": index,
        "kind": kind,
        "pool": task["pool"],
        "source_name": task["source_name"],
        "heldout_name": task.get("heldout_name"),
        "source_weight_within_pool": task.get("source_weight_within_pool"),
        "output_prefix": str(prefix.resolve()),
        "input": {
            "path": str(input_path.resolve()),
            "sha256": task["input_sha256"],
            "bytes": task["input_bytes"],
            "rows": task["input_rows"],
        },
        "input_receipt": str(args.input_receipt.resolve()),
        "input_receipt_sha256": input_receipt_sha,
        "heldout_manifest": str(args.heldout_manifest.resolve()),
        "heldout_manifest_sha256": heldout_manifest_sha,
        "tokenizer": {
            "root": str(tokenizer_dir.resolve()),
            "revision": tokenizer["revision"],
            "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
            "tree_sha256": tokenizer["tree_sha256"],
            "vocab_size": tokenizer["vocab_size"],
        },
        "tokenizer_tree_sha256": tokenizer["tree_sha256"],
        "builder": {
            "path": str(builder_path),
            "sha256": builder_sha,
            "repository_commit": input_receipt["repository"]["commit"],
            "format": "Megatron indexed dataset v1",
            "dtype": "int32",
            "append_eod": True,
            "tokenization": "encode(add_special_tokens=false)+eod",
        },
        "decontamination": (
            input_receipt["decontamination"]
            if task["decontaminate_greekmmlu"]
            else {"applied": False}
        ),
        "heldout_exclusion": exclusion_binding,
        "counts": {**counters, "document_index_entries": document_entries},
        "outputs": {
            "bin": {
                "path": str(bin_path),
                "sha256": sha256_file(bin_path),
                "bytes": bin_path.stat().st_size,
            },
            "idx": {
                "path": str(idx_path),
                "sha256": sha256_file(idx_path),
                "bytes": idx_path.stat().st_size,
            },
            "dropped_ledger": {
                "path": str(dropped_path),
                "sha256": sha256_file(dropped_path),
                "bytes": dropped_path.stat().st_size,
                "rows": counters["contaminated_rows"],
            },
            "retained_ledger": {
                "path": str(retained_path),
                "sha256": sha256_file(retained_path),
                "bytes": retained_path.stat().st_size,
                "rows": counters["documents"],
            },
        },
    }
    write_json_atomic(manifest_path, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "task": task["task_id"],
                "documents": counters["documents"],
                "tokens": counters["tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
