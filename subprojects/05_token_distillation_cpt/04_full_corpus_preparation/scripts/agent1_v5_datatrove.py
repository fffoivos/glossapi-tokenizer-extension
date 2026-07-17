#!/usr/bin/env python3
"""DataTrove-backed, actual-Jaccard-verified deduplication for Agent-1 v5.

The stock DataTrove MinHash implementation is used to tokenize Greek text and
produce LSH candidates.  Candidate edges are accepted only after exact
5-shingle Jaccard verification.  Strict and conservative normalized-exact
edges are composed with the verified near-duplicate edges in a disk-backed
union-find.  Nanochat-base rows have immutable representative priority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import sys
import unicodedata
import zlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v5_pipeline import (  # noqa: E402
    COMBINED_MANIFEST_SCHEMA,
    canonical_json,
    canonical_schema,
    contract,
    file_receipt,
    load_config,
    sha256_file,
    validate_file_receipt,
    write_json_atomic,
)


RUNTIME_SCHEMA = "agent1_v5_datatrove_runtime_v1"
EXACT_RECEIPT_SCHEMA = "agent1_v5_exact_index_task_receipt_v1"
EXACT_MANIFEST_SCHEMA = "agent1_v5_exact_index_manifest_v1"
SIGNATURE_RECEIPT_SCHEMA = "agent1_v5_minhash_signature_task_receipt_v1"
SIGNATURE_MANIFEST_SCHEMA = "agent1_v5_minhash_signature_manifest_v1"
BUCKET_RECEIPT_SCHEMA = "agent1_v5_minhash_bucket_task_receipt_v1"
PAIR_MANIFEST_SCHEMA = "agent1_v5_lsh_pair_manifest_v1"
SHINGLE_RECEIPT_SCHEMA = "agent1_v5_shingle_task_receipt_v1"
SHINGLE_MANIFEST_SCHEMA = "agent1_v5_shingle_manifest_v1"
VERIFY_RECEIPT_SCHEMA = "agent1_v5_jaccard_verify_task_receipt_v1"
VERIFY_MANIFEST_SCHEMA = "agent1_v5_jaccard_verify_manifest_v1"
CLUSTER_MANIFEST_SCHEMA = "agent1_v5_dedup_cluster_manifest_v1"
FILTER_RECEIPT_SCHEMA = "agent1_v5_dedup_filter_task_receipt_v1"
DEDUP_MANIFEST_SCHEMA = "agent1_v5_deduplicated_release_manifest_v1"

WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
PAIR_STRUCT = struct.Struct("<4I")
REF_SHIFT = 32
MAX_UINT32 = (1 << REF_SHIFT) - 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def git_output(root: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def encode_ref(rank: int, document: int) -> int:
    if not 0 <= rank <= MAX_UINT32 or not 0 <= document <= MAX_UINT32:
        raise ValueError(f"dedup reference exceeds uint32: rank={rank}, doc={document}")
    return (rank << REF_SHIFT) | document


def decode_ref(reference: int) -> tuple[int, int]:
    return reference >> REF_SHIFT, reference & MAX_UINT32


def _load_release(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    value = read_object(manifest_path)
    if value.get("schema_version") != COMBINED_MANIFEST_SCHEMA or value.get("status") != "passed":
        raise ValueError("combined release manifest is not passed")
    root = Path(str(value["root"])).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = value.get("files")
    if not isinstance(files, list) or [int(row["rank"]) for row in files] != list(range(len(files))):
        raise ValueError("combined release inventory ranks must be contiguous from zero")
    for row in files:
        validate_file_receipt(row, root=root)
    return value, root


def _dedup_paths(run_root: Path, stage: str, task: int, suffix: str) -> tuple[Path, Path]:
    output = run_root / "60-dedup" / stage / f"{task:06d}{suffix}"
    receipt = run_root / "60-dedup" / stage / "receipts" / f"{task:06d}.json"
    return output, receipt


def _verify_runtime(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = read_object(path)
    if receipt.get("schema_version") != RUNTIME_SCHEMA or receipt.get("status") != "passed":
        raise ValueError("DataTrove runtime receipt is not passed")
    expected = config["pins"]
    if receipt.get("version") != expected["datatrove_version"]:
        raise ValueError("DataTrove runtime version drift")
    if receipt.get("commit") != expected["datatrove_commit"]:
        raise ValueError("DataTrove runtime commit drift")
    return receipt


def build_runtime_receipt(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = args.datatrove_root.resolve()
    expected_commit = str(config["pins"]["datatrove_commit"])
    expected_version = str(config["pins"]["datatrove_version"])
    if git_output(root, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("DataTrove checkout is not at the pinned commit")
    if git_output(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("DataTrove checkout must be clean")
    installed = importlib.metadata.version("datatrove")
    if installed != expected_version:
        raise ValueError(f"installed DataTrove is {installed}, expected {expected_version}")
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.text import TextNormConfig
    from datatrove.utils.word_tokenizers import load_word_tokenizer

    dedup = config["dedup"]
    norm = TextNormConfig(norm_unicode_diacritics=not bool(dedup["preserve_greek_diacritics"]))
    minhash = MinhashConfig(
        n_grams=int(dedup["shingle_size"]),
        num_buckets=int(dedup["num_buckets"]),
        hashes_per_bucket=int(dedup["hashes_per_bucket"]),
        seed=int(dedup["seed"]),
        norm_config=norm,
        hash_config=HashConfig(precision=int(dedup["hash_precision"])),
    )
    tokenizer = load_word_tokenizer(str(dedup["language"]))
    result: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "root": str(root),
        "commit": expected_commit,
        "version": installed,
        "config": str(minhash),
        "language": str(dedup["language"]),
        "tokenizer_class": type(tokenizer).__name__,
        "norm_config": vars(norm),
        "actual_jaccard_threshold": float(dedup["verified_jaccard_threshold"]),
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "version": installed, "commit": expected_commit}))
    return 0


def _quality(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 1.0e30
    return result if math.isfinite(result) else 1.0e30


def relaxed_exact_text(text: str) -> str:
    """Conservative exact normalization; keeps Greek accents and numbers."""

    text = unicodedata.normalize("NFC", text).casefold()
    text = PUNCT_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def exact_index_task(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = contract(args.contract)
    manifest, root = _load_release(args.combined_manifest)
    rank = int(args.task_index)
    inventory = manifest["files"][rank]
    path = validate_file_receipt(inventory, root=root)
    run_root = Path(str(run["run_root"]))
    output, receipt_path = _dedup_paths(run_root, "exact-index", rank, ".parquet")
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != EXACT_RECEIPT_SCHEMA:
            raise ValueError(f"invalid exact-index receipt: {receipt_path}")
        validate_file_receipt(receipt["output"], root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": rank}))
        return 0
    schema = pa.schema(
        [
            ("reference", pa.uint64()),
            ("rank", pa.uint32()),
            ("doc_index", pa.uint32()),
            ("origin_rank", pa.uint8()),
            ("source_dataset", pa.large_string()),
            ("source_doc_id", pa.large_string()),
            ("strict_hash", pa.string()),
            ("relaxed_hash", pa.string()),
            ("quality_rank", pa.float64()),
            ("cleaner_loss", pa.uint64()),
            ("text_chars", pa.uint64()),
        ]
    )
    parquet = pq.ParquetFile(path)
    counters: Counter[str] = Counter()

    def batches() -> Iterable[list[dict[str, Any]]]:
        document = 0
        for batch in parquet.iter_batches(
            columns=[
                "source_dataset",
                "source_doc_id",
                "text",
                "greek_badness_score",
                "mojibake_badness_score",
                "cleaner_chars_before",
                "cleaner_chars_after",
            ],
            batch_size=2048,
        ):
            rows = []
            for row in batch.to_pylist():
                text = str(row.get("text") or "")
                relaxed = relaxed_exact_text(text)
                before = int(row.get("cleaner_chars_before") or len(text))
                after = int(row.get("cleaner_chars_after") or len(text))
                rows.append(
                    {
                        "reference": encode_ref(rank, document),
                        "rank": rank,
                        "doc_index": document,
                        "origin_rank": 0 if inventory["origin"] == "nanochat_base" else 1,
                        "source_dataset": str(row.get("source_dataset") or ""),
                        "source_doc_id": str(row.get("source_doc_id") or ""),
                        "strict_hash": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
                        "relaxed_hash": hashlib.sha256(relaxed.encode("utf-8")).hexdigest() if relaxed else None,
                        "quality_rank": _quality(row.get("greek_badness_score"))
                        + _quality(row.get("mojibake_badness_score")),
                        "cleaner_loss": max(0, before - after),
                        "text_chars": len(text),
                    }
                )
                document += 1
            counters["rows"] += len(rows)
            yield rows

    from agent1_v5_pipeline import write_parquet_atomic

    rows = write_parquet_atomic(output, schema, batches())
    if rows != int(inventory["rows"]):
        raise ValueError("exact index row closure failed")
    receipt: dict[str, Any] = {
        "schema_version": EXACT_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": rank,
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "input": inventory,
        "output": file_receipt(output, root=run_root, rows=rows),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": rank, "rows": rows}))
    return 0


def merge_exact_index(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    run = contract(args.contract)
    combined, _ = _load_release(args.combined_manifest)
    run_root = Path(str(run["run_root"]))
    shards = []
    rows = 0
    for inventory in combined["files"]:
        rank = int(inventory["rank"])
        _, receipt_path = _dedup_paths(run_root, "exact-index", rank, ".parquet")
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != EXACT_RECEIPT_SCHEMA or receipt.get("input", {}).get("sha256") != inventory["sha256"]:
            raise ValueError(f"exact-index receipt drift for rank {rank}")
        path = validate_file_receipt(receipt["output"], root=run_root)
        rows += pq.ParquetFile(path).metadata.num_rows
        shards.append(receipt["output"])
    if rows != int(combined["rows"]):
        raise ValueError("exact-index merged row closure failed")
    result: dict[str, Any] = {
        "schema_version": EXACT_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "task_count": len(shards),
        "rows": rows,
        "shards": shards,
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "tasks": len(shards), "rows": rows}))
    return 0


def _minhash_config(config: Mapping[str, Any]) -> Any:
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.text import TextNormConfig

    dedup = config["dedup"]
    return MinhashConfig(
        n_grams=int(dedup["shingle_size"]),
        num_buckets=int(dedup["num_buckets"]),
        hashes_per_bucket=int(dedup["hashes_per_bucket"]),
        seed=int(dedup["seed"]),
        norm_config=TextNormConfig(
            norm_unicode_diacritics=not bool(dedup["preserve_greek_diacritics"])
        ),
        hash_config=HashConfig(precision=int(dedup["hash_precision"])),
    )


def signature_task(args: argparse.Namespace) -> int:
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.readers import ParquetReader

    config = load_config(args.config)
    _verify_runtime(args.runtime_receipt, config)
    run = contract(args.contract)
    combined, root = _load_release(args.combined_manifest)
    rank = int(args.task_index)
    world_size = len(combined["files"])
    run_root = Path(str(run["run_root"]))
    output_root = run_root / "60-dedup" / "minhash-signatures"
    _, receipt_path = _dedup_paths(run_root, "minhash-signatures", rank, ".unused")
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != SIGNATURE_RECEIPT_SCHEMA:
            raise ValueError(f"invalid signature receipt: {receipt_path}")
        for binding in receipt["outputs"]:
            validate_file_receipt(binding, root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": rank}))
        return 0
    reader = ParquetReader(
        str(root / "data"),
        text_key="text",
        id_key="source_doc_id",
        read_metadata=False,
        recursive=False,
        glob_pattern="*.parquet",
        file_progress=False,
        doc_progress=False,
    )
    step = MinhashDedupSignature(
        output_folder=str(output_root),
        config=_minhash_config(config),
        language=str(config["dedup"]["language"]),
    )
    step.run(reader.run(rank=rank, world_size=world_size), rank=rank, world_size=world_size)
    outputs = []
    for bucket in range(int(config["dedup"]["num_buckets"])):
        path = output_root / f"bucket_{bucket:03d}" / f"{rank:05d}.minhash.sig"
        outputs.append(file_receipt(path, root=run_root))
    receipt: dict[str, Any] = {
        "schema_version": SIGNATURE_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": rank,
        "runtime_receipt_sha256": sha256_file(args.runtime_receipt),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "input": combined["files"][rank],
        "outputs": outputs,
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": rank, "files": len(outputs)}))
    return 0


def _signature_dtype(step: Any) -> Any:
    """Return DataTrove's on-disk MinHash record dtype."""

    import numpy as np

    return np.dtype(
        [
            (f"field{i + 1}", f"<{step.config.hash_config.struct_format}")
            for i in range(step.config.hashes_per_bucket)
        ]
        + [(f"field{step.config.hashes_per_bucket + 1}", "<I")]
    )


def _signature_partial_paths(run_root: Path, rank: int, row_group: int) -> tuple[Path, Path]:
    root = run_root / "60-dedup" / "minhash-signatures"
    partial = root / "partials" / f"{rank:05d}" / f"group-{row_group:03d}"
    receipt = root / "partial-receipts" / f"{rank:05d}" / f"group-{row_group:03d}.json"
    return partial, receipt


def signature_row_group_task(args: argparse.Namespace) -> int:
    """Create one durable, sorted MinHash-signature row-group partial.

    A few legacy NanoChat parquet files contain very large text rows.  The
    stock DataTrove step treats the full file as one non-checkpointed unit,
    which cannot fit in Clariden's debug walltime.  This uses the identical
    tokenizer, signature calculation, binary layout, and sort order, but
    checkpoints each parquet row group before the final rank-level merge.
    """

    import pyarrow.parquet as pq
    from datatrove.pipeline.dedup import MinhashDedupSignature

    config = load_config(args.config)
    _verify_runtime(args.runtime_receipt, config)
    run = contract(args.contract)
    combined, root = _load_release(args.combined_manifest)
    rank = int(args.task_index)
    row_group = int(args.row_group)
    run_root = Path(str(run["run_root"]))
    source = root / str(combined["files"][rank]["path"])
    parquet = pq.ParquetFile(source)
    if not 0 <= row_group < parquet.metadata.num_row_groups:
        raise ValueError(f"row group out of range: rank={rank} row_group={row_group}")
    partial_root, receipt_path = _signature_partial_paths(run_root, rank, row_group)
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("status") != "passed":
            raise ValueError(f"invalid signature partial receipt: {receipt_path}")
        for binding in receipt["outputs"]:
            validate_file_receipt(binding, root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": rank, "row_group": row_group}))
        return 0

    step = MinhashDedupSignature(
        output_folder=str(partial_root),
        config=_minhash_config(config),
        language=str(config["dedup"]["language"]),
    )
    dtype = _signature_dtype(step)
    row_offset = sum(parquet.metadata.row_group(i).num_rows for i in range(row_group))
    texts = parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
    records: list[list[tuple[Any, ...]]] = [[] for _ in range(int(config["dedup"]["num_buckets"]))]
    for document_index, text in enumerate(texts, start=row_offset):
        shingles = step.get_shingles(str(text))
        if shingles.size == 0:
            continue
        for bucket, signature in enumerate(step.get_signature(shingles)):
            records[bucket].append((*signature, document_index))

    outputs = []
    for bucket, values in enumerate(records):
        path = partial_root / f"bucket_{bucket:03d}" / f"{rank:05d}.minhash.sig"
        path.parent.mkdir(parents=True, exist_ok=True)
        array = __import__("numpy").array(values, dtype=dtype)
        array.sort(order=dtype.names)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(array.tobytes())
        os.replace(temporary, path)
        outputs.append(file_receipt(path, root=run_root))
    write_json_atomic(
        receipt_path,
        {
            "schema_version": "agent1_v5_minhash_signature_partial_v1",
            "status": "passed",
            "created_at": utc_now(),
            "task_index": rank,
            "row_group": row_group,
            "rows": len(texts),
            "outputs": outputs,
        },
    )
    print(canonical_json({"ok": True, "task": rank, "row_group": row_group, "files": len(outputs)}))
    return 0


def merge_signature_row_groups(args: argparse.Namespace) -> int:
    """Merge durable signature row-group partials into the normal rank output."""

    import numpy as np
    import pyarrow.parquet as pq
    from datatrove.pipeline.dedup import MinhashDedupSignature

    config = load_config(args.config)
    _verify_runtime(args.runtime_receipt, config)
    run = contract(args.contract)
    combined, root = _load_release(args.combined_manifest)
    rank = int(args.task_index)
    run_root = Path(str(run["run_root"]))
    output_root = run_root / "60-dedup" / "minhash-signatures"
    _, receipt_path = _dedup_paths(run_root, "minhash-signatures", rank, ".unused")
    if receipt_path.exists():
        print(canonical_json({"ok": True, "reused": True, "task": rank}))
        return 0
    parquet = pq.ParquetFile(root / str(combined["files"][rank]["path"]))
    step = MinhashDedupSignature(output_folder=str(output_root), config=_minhash_config(config), language=str(config["dedup"]["language"]))
    dtype = _signature_dtype(step)
    partials = []
    for row_group in range(parquet.metadata.num_row_groups):
        _, partial_receipt = _signature_partial_paths(run_root, rank, row_group)
        value = read_object(partial_receipt)
        if value.get("status") != "passed" or value.get("row_group") != row_group:
            raise ValueError(f"missing or invalid signature partial: {partial_receipt}")
        partials.append(value)
    outputs = []
    for bucket in range(int(config["dedup"]["num_buckets"])):
        chunks = []
        for partial in partials:
            binding = partial["outputs"][bucket]
            validate_file_receipt(binding, root=run_root)
            chunks.append(np.frombuffer((run_root / str(binding["path"])).read_bytes(), dtype=dtype))
        merged = np.concatenate(chunks) if chunks else np.array([], dtype=dtype)
        merged.sort(order=dtype.names)
        path = output_root / f"bucket_{bucket:03d}" / f"{rank:05d}.minhash.sig"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(merged.tobytes())
        os.replace(temporary, path)
        outputs.append(file_receipt(path, root=run_root))
    write_json_atomic(receipt_path, {
        "schema_version": SIGNATURE_RECEIPT_SCHEMA, "status": "passed", "created_at": utc_now(),
        "task_index": rank, "runtime_receipt_sha256": sha256_file(args.runtime_receipt),
        "combined_manifest_sha256": sha256_file(args.combined_manifest), "input": combined["files"][rank], "outputs": outputs,
    })
    print(canonical_json({"ok": True, "task": rank, "files": len(outputs)}))
    return 0


def merge_signatures(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run = contract(args.contract)
    combined, _ = _load_release(args.combined_manifest)
    run_root = Path(str(run["run_root"]))
    receipts = []
    bytes_total = 0
    for inventory in combined["files"]:
        rank = int(inventory["rank"])
        _, receipt_path = _dedup_paths(run_root, "minhash-signatures", rank, ".unused")
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != SIGNATURE_RECEIPT_SCHEMA or receipt.get("input", {}).get("sha256") != inventory["sha256"]:
            raise ValueError(f"signature receipt drift for rank {rank}")
        if len(receipt["outputs"]) != int(config["dedup"]["num_buckets"]):
            raise ValueError(f"signature bucket closure failed for rank {rank}")
        for binding in receipt["outputs"]:
            validate_file_receipt(binding, root=run_root)
            bytes_total += int(binding["bytes"])
        receipts.append(receipt)
    result: dict[str, Any] = {
        "schema_version": SIGNATURE_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "task_count": len(receipts),
        "bucket_count": int(config["dedup"]["num_buckets"]),
        "bytes": bytes_total,
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "tasks": len(receipts), "bytes": bytes_total}))
    return 0


def bucket_task(args: argparse.Namespace) -> int:
    from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets

    config = load_config(args.config)
    _verify_runtime(args.runtime_receipt, config)
    signatures = read_object(args.signature_manifest)
    if signatures.get("schema_version") != SIGNATURE_MANIFEST_SCHEMA or signatures.get("status") != "passed":
        raise ValueError("signature manifest is not passed")
    run = contract(args.contract)
    run_root = Path(str(run["run_root"]))
    rank = int(args.task_index)
    world_size = int(config["dedup"]["num_buckets"])
    output_root = run_root / "60-dedup" / "minhash-buckets"
    output = output_root / f"{rank:05d}_00.dups"
    _, receipt_path = _dedup_paths(run_root, "minhash-buckets", rank, ".unused")
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != BUCKET_RECEIPT_SCHEMA:
            raise ValueError(f"invalid bucket receipt: {receipt_path}")
        validate_file_receipt(receipt["output"], root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": rank}))
        return 0
    step = MinhashDedupBuckets(
        input_folder=str(run_root / "60-dedup" / "minhash-signatures"),
        output_folder=str(output_root),
        config=_minhash_config(config),
        only_dedup_in_index=False,
    )
    step.run(rank=rank, world_size=world_size)
    if output.stat().st_size % PAIR_STRUCT.size:
        raise ValueError(f"DataTrove produced truncated pair output: {output}")
    pairs = output.stat().st_size // PAIR_STRUCT.size
    receipt: dict[str, Any] = {
        "schema_version": BUCKET_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": rank,
        "runtime_receipt_sha256": sha256_file(args.runtime_receipt),
        "signature_manifest_sha256": sha256_file(args.signature_manifest),
        "pairs": pairs,
        "output": file_receipt(output, root=run_root),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": rank, "pairs": pairs}))
    return 0


def _iter_pairs(path: Path) -> Iterable[tuple[int, int, int, int]]:
    with path.open("rb") as handle:
        while data := handle.read(PAIR_STRUCT.size):
            if len(data) != PAIR_STRUCT.size:
                raise ValueError(f"truncated DataTrove pair file: {path}")
            yield PAIR_STRUCT.unpack(data)


def merge_lsh_pairs(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run = contract(args.contract)
    combined, _ = _load_release(args.combined_manifest)
    run_root = Path(str(run["run_root"]))
    database = args.output.resolve()
    if database.exists():
        raise FileExistsError(database)
    database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE pairs(pair_id INTEGER PRIMARY KEY, left_ref INTEGER NOT NULL, right_ref INTEGER NOT NULL, UNIQUE(left_ref, right_ref))"
    )
    max_documents = int(config["dedup"]["max_bucket_documents"])
    counters: Counter[str] = Counter()
    oversized: list[dict[str, int]] = []

    def commit_chain(bucket: int, chain: list[tuple[int, int]]) -> None:
        if not chain:
            return
        nodes = {node for edge in chain for node in edge}
        if len(nodes) > max_documents:
            counters["oversized_bucket_groups"] += 1
            counters["oversized_bucket_documents"] += len(nodes)
            counters["oversized_bucket_edges_excluded"] += len(chain)
            oversized.append({"bucket": bucket, "documents": len(nodes), "edges": len(chain)})
            return
        for left, right in chain:
            if left == right:
                counters["self_pairs"] += 1
                continue
            left, right = sorted((left, right))
            before = connection.total_changes
            connection.execute("INSERT OR IGNORE INTO pairs(left_ref, right_ref) VALUES (?, ?)", (left, right))
            counters["unique_pairs"] += connection.total_changes - before

    for bucket in range(int(config["dedup"]["num_buckets"])):
        _, receipt_path = _dedup_paths(run_root, "minhash-buckets", bucket, ".unused")
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != BUCKET_RECEIPT_SCHEMA:
            raise ValueError(f"missing/invalid bucket receipt {bucket}")
        path = validate_file_receipt(receipt["output"], root=run_root)
        chain: list[tuple[int, int]] = []
        last_right: int | None = None
        for file_a, doc_a, file_b, doc_b in _iter_pairs(path):
            if file_a >= len(combined["files"]) or file_b >= len(combined["files"]):
                raise ValueError("DataTrove pair references an unknown input rank")
            left = encode_ref(file_a, doc_a)
            right = encode_ref(file_b, doc_b)
            counters["raw_pairs"] += 1
            if chain and left != last_right:
                commit_chain(bucket, chain)
                chain = []
            chain.append((left, right))
            last_right = right
        commit_chain(bucket, chain)
        connection.commit()
    pair_count = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    connection.execute("CREATE INDEX pair_left ON pairs(left_ref)")
    connection.execute("CREATE INDEX pair_right ON pairs(right_ref)")
    connection.commit()
    connection.close()
    if pair_count != counters["unique_pairs"]:
        raise ValueError("LSH pair database closure failed")
    oversized_path = database.with_suffix(".oversized.json")
    write_json_atomic(
        oversized_path,
        {
            "schema_version": "agent1_v5_oversized_lsh_groups_v1",
            "status": "passed",
            "created_at": utc_now(),
            "max_documents": max_documents,
            "groups": oversized,
        },
    )
    manifest_path = database.with_suffix(".manifest.json")
    result: dict[str, Any] = {
        "schema_version": PAIR_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "database": file_receipt(database),
        "oversized_groups": file_receipt(oversized_path),
        "pairs": pair_count,
        "counters": dict(counters),
    }
    write_json_atomic(manifest_path, result)
    print(canonical_json({"ok": True, "pairs": pair_count, "oversized": len(oversized)}))
    return 0


def _pair_manifest(database: Path) -> dict[str, Any]:
    manifest = read_object(database.with_suffix(".manifest.json"))
    if manifest.get("schema_version") != PAIR_MANIFEST_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("LSH pair manifest is not passed")
    validate_file_receipt(manifest["database"])
    validate_file_receipt(manifest["oversized_groups"])
    return manifest


def shingle_task(args: argparse.Namespace) -> int:
    import numpy as np
    import pyarrow.parquet as pq
    from datatrove.pipeline.dedup import MinhashDedupSignature

    config = load_config(args.config)
    _verify_runtime(args.runtime_receipt, config)
    run = contract(args.contract)
    combined, root = _load_release(args.combined_manifest)
    pairs = _pair_manifest(args.pair_database)
    rank = int(args.task_index)
    run_root = Path(str(run["run_root"]))
    output, receipt_path = _dedup_paths(run_root, "shingles", rank, ".sqlite")
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != SHINGLE_RECEIPT_SCHEMA:
            raise ValueError(f"invalid shingle receipt: {receipt_path}")
        validate_file_receipt(receipt["output"], root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": rank}))
        return 0
    pair_connection = sqlite3.connect(f"file:{args.pair_database.resolve()}?mode=ro", uri=True)
    lower = encode_ref(rank, 0)
    upper = encode_ref(rank + 1, 0) if rank < MAX_UINT32 else 1 << 63
    needed = {
        int(reference & MAX_UINT32)
        for row in pair_connection.execute(
            "SELECT left_ref FROM pairs WHERE left_ref >= ? AND left_ref < ? "
            "UNION SELECT right_ref FROM pairs WHERE right_ref >= ? AND right_ref < ?",
            (lower, upper, lower, upper),
        )
        for reference in row
    }
    pair_connection.close()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.exists():
        raise FileExistsError(output)
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE shingles(doc_index INTEGER PRIMARY KEY, count INTEGER NOT NULL, hashes BLOB NOT NULL)")
    source = validate_file_receipt(combined["files"][rank], root=root)
    parquet = pq.ParquetFile(source)
    hasher = MinhashDedupSignature(
        output_folder=str(run_root / "60-dedup" / ".unused-shingle-output"),
        config=_minhash_config(config),
        language=str(config["dedup"]["language"]),
    )
    seen = 0
    document = 0
    for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
        for text in batch.column(0).to_pylist():
            if document in needed:
                values = np.unique(hasher.get_shingles(str(text or "")).reshape(-1)).astype("<u8", copy=False)
                payload = zlib.compress(values.tobytes(order="C"), level=3)
                connection.execute(
                    "INSERT INTO shingles VALUES (?, ?, ?)",
                    (document, int(values.size), sqlite3.Binary(payload)),
                )
                seen += 1
            document += 1
        if seen and seen % 10000 == 0:
            connection.commit()
    connection.commit()
    actual = int(connection.execute("SELECT COUNT(*) FROM shingles").fetchone()[0])
    connection.close()
    if actual != len(needed):
        raise ValueError(f"rank {rank}: missing shingle rows, needed={len(needed)}, wrote={actual}")
    receipt: dict[str, Any] = {
        "schema_version": SHINGLE_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": rank,
        "runtime_receipt_sha256": sha256_file(args.runtime_receipt),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "pair_manifest_sha256": sha256_file(args.pair_database.with_suffix(".manifest.json")),
        "input": combined["files"][rank],
        "needed_documents": len(needed),
        "output": file_receipt(output, root=run_root, rows=actual),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": rank, "documents": actual, "pairs": pairs["pairs"]}))
    return 0


def merge_shingles(args: argparse.Namespace) -> int:
    run = contract(args.contract)
    combined, _ = _load_release(args.combined_manifest)
    pair_manifest = _pair_manifest(args.pair_database)
    run_root = Path(str(run["run_root"]))
    receipts = []
    documents = 0
    for inventory in combined["files"]:
        rank = int(inventory["rank"])
        _, receipt_path = _dedup_paths(run_root, "shingles", rank, ".sqlite")
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != SHINGLE_RECEIPT_SCHEMA or receipt.get("input", {}).get("sha256") != inventory["sha256"]:
            raise ValueError(f"shingle receipt drift for rank {rank}")
        validate_file_receipt(receipt["output"], root=run_root)
        documents += int(receipt["needed_documents"])
        receipts.append(receipt)
    result: dict[str, Any] = {
        "schema_version": SHINGLE_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "pair_manifest_sha256": sha256_file(args.pair_database.with_suffix(".manifest.json")),
        "task_count": len(receipts),
        "pairs": int(pair_manifest["pairs"]),
        "documents": documents,
        "shards": [receipt["output"] for receipt in receipts],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, "tasks": len(receipts), "documents": documents}))
    return 0


class ShingleStore:
    def __init__(self, run_root: Path, max_open: int = 32):
        self.run_root = run_root
        self.max_open = max_open
        self.connections: OrderedDict[int, sqlite3.Connection] = OrderedDict()

    def _connection(self, rank: int) -> sqlite3.Connection:
        if rank in self.connections:
            connection = self.connections.pop(rank)
            self.connections[rank] = connection
            return connection
        path, _ = _dedup_paths(self.run_root, "shingles", rank, ".sqlite")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.connections[rank] = connection
        if len(self.connections) > self.max_open:
            _, old = self.connections.popitem(last=False)
            old.close()
        return connection

    def get(self, reference: int) -> Any:
        import numpy as np

        rank, document = decode_ref(reference)
        row = self._connection(rank).execute(
            "SELECT count, hashes FROM shingles WHERE doc_index = ?", (document,)
        ).fetchone()
        if row is None:
            raise ValueError(f"missing shingle payload for rank={rank}, doc={document}")
        values = np.frombuffer(zlib.decompress(row[1]), dtype="<u8")
        if values.size != int(row[0]):
            raise ValueError("shingle payload length drift")
        return values

    def close(self) -> None:
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()


def _jaccard_sorted(left: Any, right: Any) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    i = 0
    j = 0
    intersection = 0
    while i < left.size and j < right.size:
        a = int(left[i])
        b = int(right[j])
        if a == b:
            intersection += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    union = int(left.size) + int(right.size) - intersection
    return intersection / union if union else 0.0


def verify_task(args: argparse.Namespace) -> int:
    import pyarrow as pa

    config = load_config(args.config)
    _verify_runtime(args.runtime_receipt, config)
    run = contract(args.contract)
    _pair_manifest(args.pair_database)
    shingle_manifest = read_object(args.shingle_manifest)
    if shingle_manifest.get("schema_version") != SHINGLE_MANIFEST_SCHEMA or shingle_manifest.get("status") != "passed":
        raise ValueError("shingle manifest is not passed")
    task = int(args.task_index)
    world_size = int(args.tasks)
    if not 0 <= task < world_size:
        raise ValueError("invalid Jaccard verifier task")
    run_root = Path(str(run["run_root"]))
    output, receipt_path = _dedup_paths(run_root, "verified-edges", task, ".parquet")
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != VERIFY_RECEIPT_SCHEMA:
            raise ValueError(f"invalid verifier receipt: {receipt_path}")
        validate_file_receipt(receipt["output"], root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": task}))
        return 0
    schema = pa.schema(
        [
            ("pair_id", pa.uint64()),
            ("left_ref", pa.uint64()),
            ("right_ref", pa.uint64()),
            ("jaccard", pa.float64()),
        ]
    )
    threshold = float(config["dedup"]["verified_jaccard_threshold"])
    connection = sqlite3.connect(f"file:{args.pair_database.resolve()}?mode=ro", uri=True)
    cursor = connection.execute(
        "SELECT pair_id, left_ref, right_ref FROM pairs WHERE (pair_id - 1) % ? = ? ORDER BY pair_id",
        (world_size, task),
    )
    store = ShingleStore(run_root, max_open=int(args.max_open_shards))
    counters: Counter[str] = Counter()

    def batches() -> Iterable[list[dict[str, Any]]]:
        rows = []
        for pair_id, left_ref, right_ref in cursor:
            similarity = _jaccard_sorted(store.get(int(left_ref)), store.get(int(right_ref)))
            counters["candidates"] += 1
            if similarity >= threshold:
                rows.append(
                    {
                        "pair_id": int(pair_id),
                        "left_ref": int(left_ref),
                        "right_ref": int(right_ref),
                        "jaccard": similarity,
                    }
                )
                counters["verified"] += 1
            else:
                counters["rejected"] += 1
            if len(rows) >= 2048:
                yield rows
                rows = []
        if rows:
            yield rows

    from agent1_v5_pipeline import write_parquet_atomic

    try:
        rows = write_parquet_atomic(output, schema, batches())
    finally:
        store.close()
        connection.close()
    if rows != counters["verified"]:
        raise ValueError("Jaccard verifier output closure failed")
    receipt: dict[str, Any] = {
        "schema_version": VERIFY_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": task,
        "world_size": world_size,
        "runtime_receipt_sha256": sha256_file(args.runtime_receipt),
        "pair_manifest_sha256": sha256_file(args.pair_database.with_suffix(".manifest.json")),
        "shingle_manifest_sha256": sha256_file(args.shingle_manifest),
        "threshold": threshold,
        "counters": dict(counters),
        "output": file_receipt(output, root=run_root, rows=rows),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": task, **dict(counters)}))
    return 0


def merge_verified(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    run = contract(args.contract)
    pair_manifest = _pair_manifest(args.pair_database)
    run_root = Path(str(run["run_root"]))
    receipts = []
    counters: Counter[str] = Counter()
    for task in range(int(args.tasks)):
        _, receipt_path = _dedup_paths(run_root, "verified-edges", task, ".parquet")
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != VERIFY_RECEIPT_SCHEMA or int(receipt.get("world_size", -1)) != int(args.tasks):
            raise ValueError(f"verifier receipt drift for task {task}")
        path = validate_file_receipt(receipt["output"], root=run_root)
        if pq.ParquetFile(path).metadata.num_rows != int(receipt["counters"].get("verified", 0)):
            raise ValueError(f"verified edge count drift for task {task}")
        counters.update({key: int(value) for key, value in receipt["counters"].items()})
        receipts.append(receipt)
    if counters["candidates"] != int(pair_manifest["pairs"]):
        raise ValueError("Jaccard verification candidate closure failed")
    if counters["verified"] + counters["rejected"] != counters["candidates"]:
        raise ValueError("Jaccard verification decision closure failed")
    result: dict[str, Any] = {
        "schema_version": VERIFY_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "pair_manifest_sha256": sha256_file(args.pair_database.with_suffix(".manifest.json")),
        "task_count": len(receipts),
        "counters": dict(counters),
        "shards": [receipt["output"] for receipt in receipts],
    }
    write_json_atomic(args.output, result)
    print(canonical_json({"ok": True, **dict(counters)}))
    return 0


class SqliteUnionFind:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def ensure(self, reference: int) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO nodes(reference, parent, size) VALUES (?, ?, 1)",
            (reference, reference),
        )

    def find(self, reference: int) -> int:
        self.ensure(reference)
        trail = []
        current = reference
        while True:
            row = self.connection.execute(
                "SELECT parent FROM nodes WHERE reference = ?", (current,)
            ).fetchone()
            if row is None:
                raise AssertionError(current)
            parent = int(row[0])
            if parent == current:
                break
            trail.append(current)
            current = parent
        if trail:
            self.connection.executemany(
                "UPDATE nodes SET parent = ? WHERE reference = ?",
                [(current, node) for node in trail],
            )
        return current

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        size_left = int(
            self.connection.execute(
                "SELECT size FROM nodes WHERE reference = ?", (root_left,)
            ).fetchone()[0]
        )
        size_right = int(
            self.connection.execute(
                "SELECT size FROM nodes WHERE reference = ?", (root_right,)
            ).fetchone()[0]
        )
        if (size_left, -root_left) < (size_right, -root_right):
            root_left, root_right = root_right, root_left
            size_left, size_right = size_right, size_left
        self.connection.execute(
            "UPDATE nodes SET parent = ? WHERE reference = ?", (root_left, root_right)
        )
        self.connection.execute(
            "UPDATE nodes SET size = ? WHERE reference = ?",
            (size_left + size_right, root_left),
        )
        return True


def _union_hash_groups(
    connection: sqlite3.Connection,
    union_find: SqliteUnionFind,
    column: str,
) -> tuple[int, int]:
    if column not in {"strict_hash", "relaxed_hash"}:
        raise ValueError(column)
    cursor = connection.execute(
        f"SELECT {column}, reference FROM docs WHERE {column} IS NOT NULL ORDER BY {column}, reference"
    )
    previous_hash: str | None = None
    representative: int | None = None
    groups = 0
    edges = 0
    group_size = 0
    for value, reference in cursor:
        reference = int(reference)
        if value != previous_hash:
            if group_size > 1:
                groups += 1
            previous_hash = str(value)
            representative = reference
            group_size = 1
            continue
        group_size += 1
        if representative is None:
            raise AssertionError("hash group lacks representative")
        if union_find.union(representative, reference):
            edges += 1
    if group_size > 1:
        groups += 1
    return groups, edges


def cluster_duplicates(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = contract(args.contract)
    combined, _ = _load_release(args.combined_manifest)
    exact_manifest = read_object(args.exact_manifest)
    verified_manifest = read_object(args.verified_manifest)
    if exact_manifest.get("schema_version") != EXACT_MANIFEST_SCHEMA or exact_manifest.get("status") != "passed":
        raise ValueError("exact-index manifest is not passed")
    if verified_manifest.get("schema_version") != VERIFY_MANIFEST_SCHEMA or verified_manifest.get("status") != "passed":
        raise ValueError("verified-edge manifest is not passed")
    run_root = Path(str(run["run_root"]))
    database = args.output.resolve()
    if database.exists():
        raise FileExistsError(database)
    database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE docs(
          reference INTEGER PRIMARY KEY,
          rank INTEGER NOT NULL,
          doc_index INTEGER NOT NULL,
          origin_rank INTEGER NOT NULL,
          source_dataset TEXT NOT NULL,
          source_doc_id TEXT NOT NULL,
          strict_hash TEXT,
          relaxed_hash TEXT,
          quality_rank REAL NOT NULL,
          cleaner_loss INTEGER NOT NULL,
          text_chars INTEGER NOT NULL
        );
        CREATE TABLE nodes(reference INTEGER PRIMARY KEY, parent INTEGER NOT NULL, size INTEGER NOT NULL);
        CREATE TABLE removals(
          rank INTEGER NOT NULL,
          doc_index INTEGER NOT NULL,
          reference INTEGER PRIMARY KEY,
          representative_rank INTEGER NOT NULL,
          representative_doc_index INTEGER NOT NULL,
          representative_reference INTEGER NOT NULL,
          cluster_id TEXT NOT NULL,
          method TEXT NOT NULL,
          UNIQUE(rank, doc_index)
        );
        """
    )
    rows = 0
    for binding in exact_manifest["shards"]:
        path = validate_file_receipt(binding, root=run_root)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096):
            values = [
                (
                    int(row["reference"]),
                    int(row["rank"]),
                    int(row["doc_index"]),
                    int(row["origin_rank"]),
                    str(row["source_dataset"]),
                    str(row["source_doc_id"]),
                    row.get("strict_hash"),
                    row.get("relaxed_hash"),
                    float(row["quality_rank"]),
                    int(row["cleaner_loss"]),
                    int(row["text_chars"]),
                )
                for row in batch.to_pylist()
            ]
            connection.executemany("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            rows += len(values)
        connection.commit()
    if rows != int(combined["rows"]) or rows != int(exact_manifest["rows"]):
        raise ValueError("cluster document index row closure failed")
    connection.execute("CREATE INDEX docs_strict ON docs(strict_hash)")
    connection.execute("CREATE INDEX docs_relaxed ON docs(relaxed_hash)")
    connection.commit()
    union_find = SqliteUnionFind(connection)
    strict_groups, strict_edges = _union_hash_groups(connection, union_find, "strict_hash")
    connection.commit()
    relaxed_groups, relaxed_edges = _union_hash_groups(connection, union_find, "relaxed_hash")
    connection.commit()
    verified_edges = 0
    verified_union_edges = 0
    for binding in verified_manifest["shards"]:
        path = validate_file_receipt(binding, root=run_root)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["left_ref", "right_ref"], batch_size=4096):
            for row in batch.to_pylist():
                verified_edges += 1
                verified_union_edges += int(union_find.union(int(row["left_ref"]), int(row["right_ref"])))
        connection.commit()
    if verified_edges != int(verified_manifest["counters"]["verified"]):
        raise ValueError("verified edge closure failed during clustering")
    connection.execute("CREATE TEMP TABLE components(reference INTEGER PRIMARY KEY, root INTEGER NOT NULL)")
    node_rows = [int(row[0]) for row in connection.execute("SELECT reference FROM nodes ORDER BY reference")]
    connection.executemany(
        "INSERT INTO components VALUES (?, ?)",
        [(reference, union_find.find(reference)) for reference in node_rows],
    )
    connection.execute("CREATE INDEX components_root ON components(root)")
    connection.commit()

    def priority(row: Sequence[Any]) -> tuple[Any, ...]:
        return (
            int(row[3]),
            float(row[6]),
            int(row[7]),
            -int(row[8]),
            str(row[4]),
            str(row[5]),
            int(row[0]),
        )

    cluster_count = 0
    removed = 0
    current_root: int | None = None
    members: list[Sequence[Any]] = []

    def commit_component(root: int | None, component: list[Sequence[Any]]) -> None:
        nonlocal cluster_count, removed
        if root is None or len(component) < 2:
            return
        representative = min(component, key=priority)
        representative_ref = int(representative[0])
        representative_rank = int(representative[1])
        representative_doc = int(representative[2])
        cluster_id = hashlib.sha256(f"agent1-v5:{root}".encode()).hexdigest()[:24]
        cluster_count += 1
        for row in component:
            reference = int(row[0])
            if reference == representative_ref:
                continue
            rank = int(row[1])
            document = int(row[2])
            connection.execute(
                "INSERT INTO removals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rank,
                    document,
                    reference,
                    representative_rank,
                    representative_doc,
                    representative_ref,
                    cluster_id,
                    "strict_or_normalized_exact_or_verified_minhash",
                ),
            )
            removed += 1

    cursor = connection.execute(
        "SELECT c.root, d.reference, d.rank, d.doc_index, d.origin_rank, "
        "d.source_dataset, d.source_doc_id, d.quality_rank, d.cleaner_loss, d.text_chars "
        "FROM components c JOIN docs d ON d.reference = c.reference ORDER BY c.root, d.reference"
    )
    # Reorder the SQL row into the shape expected by priority()/ledger logic:
    # ref, rank, doc, origin, source, id, quality, loss, chars.
    for raw in cursor:
        root = int(raw[0])
        row = raw[1:]
        if current_root is not None and root != current_root:
            commit_component(current_root, members)
            members = []
        current_root = root
        members.append(row)
    commit_component(current_root, members)
    connection.commit()
    removal_count = int(connection.execute("SELECT COUNT(*) FROM removals").fetchone()[0])
    if removal_count != removed:
        raise ValueError("dedup removal database closure failed")
    connection.execute("CREATE INDEX removals_rank_doc ON removals(rank, doc_index)")
    connection.commit()
    ledger_schema = pa.schema(
        [
            ("cluster_id", pa.string()),
            ("rank", pa.uint32()),
            ("doc_index", pa.uint32()),
            ("source_dataset", pa.large_string()),
            ("source_doc_id", pa.large_string()),
            ("representative_rank", pa.uint32()),
            ("representative_doc_index", pa.uint32()),
            ("representative_source_dataset", pa.large_string()),
            ("representative_source_doc_id", pa.large_string()),
            ("method", pa.string()),
        ]
    )
    from agent1_v5_pipeline import write_parquet_atomic

    ledger = database.with_suffix(".ledger.parquet")
    ledger_cursor = connection.execute(
        "SELECT x.cluster_id, x.rank, x.doc_index, d.source_dataset, d.source_doc_id, "
        "x.representative_rank, x.representative_doc_index, r.source_dataset, "
        "r.source_doc_id, x.method FROM removals x "
        "JOIN docs d ON d.reference = x.reference "
        "JOIN docs r ON r.reference = x.representative_reference "
        "ORDER BY x.cluster_id, x.rank, x.doc_index"
    )

    def ledger_batches() -> Iterable[list[dict[str, Any]]]:
        while values := ledger_cursor.fetchmany(8192):
            yield [
                {
                    "cluster_id": row[0],
                    "rank": row[1],
                    "doc_index": row[2],
                    "source_dataset": row[3],
                    "source_doc_id": row[4],
                    "representative_rank": row[5],
                    "representative_doc_index": row[6],
                    "representative_source_dataset": row[7],
                    "representative_source_doc_id": row[8],
                    "method": row[9],
                }
                for row in values
            ]

    ledger_count = write_parquet_atomic(ledger, ledger_schema, ledger_batches())
    connection.close()
    if ledger_count != removed:
        raise ValueError("dedup ledger closure failed")
    manifest_path = database.with_suffix(".manifest.json")
    result: dict[str, Any] = {
        "schema_version": CLUSTER_MANIFEST_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "exact_manifest_sha256": sha256_file(args.exact_manifest),
        "verified_manifest_sha256": sha256_file(args.verified_manifest),
        "input_rows": rows,
        "clusters": cluster_count,
        "removed_rows": removed,
        "retained_rows": rows - removed,
        "strict_exact_groups": strict_groups,
        "strict_exact_union_edges": strict_edges,
        "normalized_exact_groups": relaxed_groups,
        "normalized_exact_union_edges": relaxed_edges,
        "verified_minhash_edges": verified_edges,
        "verified_minhash_union_edges": verified_union_edges,
        "protected_origin": "nanochat_base",
        "removal_database": file_receipt(database),
        "decision_ledger": file_receipt(ledger, rows=ledger_count),
    }
    write_json_atomic(manifest_path, result)
    print(canonical_json({"ok": True, "clusters": cluster_count, "removed": removed}))
    return 0


def _cluster_manifest(database: Path) -> dict[str, Any]:
    manifest = read_object(database.with_suffix(".manifest.json"))
    if manifest.get("schema_version") != CLUSTER_MANIFEST_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("dedup cluster manifest is not passed")
    validate_file_receipt(manifest["removal_database"])
    validate_file_receipt(manifest["decision_ledger"])
    return manifest


def filter_task(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    run = contract(args.contract)
    combined, root = _load_release(args.combined_manifest)
    _cluster_manifest(args.removal_database)
    rank = int(args.task_index)
    inventory = combined["files"][rank]
    source = validate_file_receipt(inventory, root=root)
    run_root = Path(str(run["run_root"]))
    output, receipt_path = _dedup_paths(run_root, "filtered", rank, ".parquet")
    if receipt_path.exists():
        receipt = read_object(receipt_path)
        if receipt.get("schema_version") != FILTER_RECEIPT_SCHEMA:
            raise ValueError(f"invalid filter receipt: {receipt_path}")
        validate_file_receipt(receipt["output"], root=run_root)
        print(canonical_json({"ok": True, "reused": True, "task": rank}))
        return 0
    removal_connection = sqlite3.connect(f"file:{args.removal_database.resolve()}?mode=ro", uri=True)
    removals = {
        int(row[0])
        for row in removal_connection.execute(
            "SELECT doc_index FROM removals WHERE rank = ?", (rank,)
        )
    }
    removal_connection.close()
    parquet = pq.ParquetFile(source)
    document = 0
    retained = 0

    def batches() -> Iterable[list[dict[str, Any]]]:
        nonlocal document, retained
        for batch in parquet.iter_batches(batch_size=1024):
            rows = []
            for row in batch.to_pylist():
                if document not in removals:
                    rows.append(row)
                    retained += 1
                document += 1
            yield rows

    from agent1_v5_pipeline import write_parquet_atomic

    rows = write_parquet_atomic(output, canonical_schema(), batches())
    if document != int(inventory["rows"]):
        raise ValueError("dedup filter input row closure failed")
    if rows != retained or rows + len(removals) != document:
        raise ValueError("dedup filter decision closure failed")
    receipt: dict[str, Any] = {
        "schema_version": FILTER_RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "task_index": rank,
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "cluster_manifest_sha256": sha256_file(args.removal_database.with_suffix(".manifest.json")),
        "input": inventory,
        "removed_rows": len(removals),
        "retained_rows": rows,
        "output": file_receipt(output, root=run_root, rows=rows),
    }
    write_json_atomic(receipt_path, receipt)
    print(canonical_json({"ok": True, "task": rank, "removed": len(removals), "retained": rows}))
    return 0


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise RuntimeError("dedup release requires same-filesystem hardlinks") from exc


def merge_filtered_release(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = contract(args.contract)
    combined, _ = _load_release(args.combined_manifest)
    cluster = _cluster_manifest(args.removal_database)
    run_root = Path(str(run["run_root"]))
    release_root = args.output.resolve()
    release_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    files = []
    input_rows = 0
    output_rows = 0
    removed_rows = 0
    try:
        for inventory in combined["files"]:
            rank = int(inventory["rank"])
            output, receipt_path = _dedup_paths(run_root, "filtered", rank, ".parquet")
            receipt = read_object(receipt_path)
            if receipt.get("schema_version") != FILTER_RECEIPT_SCHEMA or receipt.get("input", {}).get("sha256") != inventory["sha256"]:
                raise ValueError(f"dedup filter receipt drift for rank {rank}")
            source = validate_file_receipt(receipt["output"], root=run_root)
            relative = Path("data") / f"{rank:06d}.parquet"
            destination = release_root / relative
            _hardlink(source, destination)
            if not pq.ParquetFile(destination).schema_arrow.equals(canonical_schema(), check_metadata=False):
                raise ValueError(f"dedup canonical schema drift: {destination}")
            binding = file_receipt(destination, root=release_root, rows=int(receipt["retained_rows"]))
            files.append({"rank": rank, "origin": inventory["origin"], **binding})
            input_rows += int(inventory["rows"])
            output_rows += int(receipt["retained_rows"])
            removed_rows += int(receipt["removed_rows"])
        if input_rows != int(combined["rows"]):
            raise ValueError("dedup release input closure failed")
        if removed_rows != int(cluster["removed_rows"]) or output_rows != int(cluster["retained_rows"]):
            raise ValueError("dedup release cluster closure failed")
        if input_rows != output_rows + removed_rows:
            raise ValueError("dedup release row waterfall failed")
        inventory_schema = pa.schema(
            [
                ("rank", pa.int64()),
                ("origin", pa.string()),
                ("path", pa.string()),
                ("rows", pa.int64()),
                ("bytes", pa.int64()),
                ("sha256", pa.string()),
            ]
        )
        from agent1_v5_pipeline import write_parquet_atomic

        inventory_path = release_root / "manifests" / "deduplicated_inventory.parquet"
        write_parquet_atomic(inventory_path, inventory_schema, [files])
        shutil.copy2(
            run_root / "license_override_receipt.json",
            release_root / "manifests" / "license_override_receipt.json",
        )
        shutil.copy2(
            args.removal_database.with_suffix(".ledger.parquet"),
            release_root / "manifests" / "dedup_decision_ledger.parquet",
        )
        card = (
            "---\n"
            "configs:\n"
            "- config_name: default\n"
            "  data_files: data/*.parquet\n"
            "---\n\n"
            f"# Greek Nanochat plus new sources — deduplicated `{run['run_id']}`\n\n"
            "Private, receipt-bound release. Deduplication uses pinned DataTrove MinHash candidates, "
            "actual 5-token-shingle Jaccard verification at 0.85, strict/normalized exact matching, "
            "and protected preference for the original Nanochat base.\n"
        )
        (release_root / "README.md").write_text(card, encoding="utf-8")
        result: dict[str, Any] = {
            "schema_version": DEDUP_MANIFEST_SCHEMA,
            "status": "passed",
            "created_at": utc_now(),
            "run_id": run["run_id"],
            "repository_id": run["private_repositories"]["deduplicated"],
            "private_only": True,
            "root": str(release_root),
            "combined_manifest_sha256": sha256_file(args.combined_manifest),
            "cluster_manifest_sha256": sha256_file(args.removal_database.with_suffix(".manifest.json")),
            "input_rows": input_rows,
            "removed_rows": removed_rows,
            "rows": output_rows,
            "files": files,
            "inventory": file_receipt(inventory_path, root=release_root, rows=len(files)),
            "decision_ledger": file_receipt(
                release_root / "manifests" / "dedup_decision_ledger.parquet",
                root=release_root,
                rows=removed_rows,
            ),
        }
        write_json_atomic(release_root / "manifests" / "deduplicated_manifest.json", result)
    except BaseException:
        shutil.rmtree(release_root, ignore_errors=True)
        raise
    print(canonical_json({"ok": True, "input": input_rows, "removed": removed_rows, "rows": output_rows}))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("build-runtime-receipt")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--datatrove-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=build_runtime_receipt)

    command = subparsers.add_parser("exact-index-task")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=exact_index_task)

    command = subparsers.add_parser("merge-exact-index")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_exact_index)

    command = subparsers.add_parser("signature-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=signature_task)

    command = subparsers.add_parser("signature-row-group-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.add_argument("--row-group", type=int, required=True)
    command.set_defaults(func=signature_row_group_task)

    command = subparsers.add_parser("merge-signature-row-groups")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=merge_signature_row_groups)

    command = subparsers.add_parser("merge-signatures")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_signatures)

    command = subparsers.add_parser("bucket-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--signature-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=bucket_task)

    command = subparsers.add_parser("merge-lsh-pairs")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_lsh_pairs)

    command = subparsers.add_parser("shingle-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--pair-database", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=shingle_task)

    command = subparsers.add_parser("merge-shingles")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--pair-database", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_shingles)

    command = subparsers.add_parser("verify-task")
    command.add_argument("--config", type=Path, default=root / "configs" / "agent1_v5_eiger_pipeline.json")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--pair-database", type=Path, required=True)
    command.add_argument("--shingle-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.add_argument("--tasks", type=int, required=True)
    command.add_argument("--max-open-shards", type=int, default=32)
    command.set_defaults(func=verify_task)

    command = subparsers.add_parser("merge-verified")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--pair-database", type=Path, required=True)
    command.add_argument("--tasks", type=int, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_verified)

    command = subparsers.add_parser("cluster")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--exact-manifest", type=Path, required=True)
    command.add_argument("--verified-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=cluster_duplicates)

    command = subparsers.add_parser("filter-task")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--removal-database", type=Path, required=True)
    command.add_argument("--task-index", type=int, required=True)
    command.set_defaults(func=filter_task)

    command = subparsers.add_parser("merge-filtered-release")
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--removal-database", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=merge_filtered_release)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
