#!/usr/bin/env python3
"""Count exact per-token firing frequencies for a pinned cleaned release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

PLAN_SCHEMA = "bibliography-cleaning-token-plan-v1"
RECEIPT_SCHEMA = "bibliography-cleaning-token-frequency-receipt-v1"
SUMMARY_SCHEMA = "bibliography-cleaning-token-frequency-summary-v1"
RESERVED_TOKEN_COUNT = 1_000


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_npy(path: str | Path, values: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_parquet(path: str | Path, table: pa.Table) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_plan(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    plan_path = Path(path).resolve()
    plan = load_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "passed":
        raise ValueError(f"{plan_path}: token plan is not passed")
    task_ids = [task["task_id"] for task in plan["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{plan_path}: duplicate task IDs")
    return plan_path, plan, sha256_file(plan_path)


def text_batches(path: Path, batch_size: int) -> Iterable[list[str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
        yield [value or "" for value in batch.column(0).to_pylist()]


def validate_frequency_array(
    path: Path, expected_sha256: str, vocab_size: int, text_tokens: int
) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path}: frequency-array SHA-256 mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.uint64 or values.shape != (vocab_size,):
        raise ValueError(
            f"{path}: expected uint64[{vocab_size}], got {values.dtype}{values.shape}"
        )
    if int(values.sum(dtype=np.uint64)) != text_tokens:
        raise ValueError(f"{path}: frequency sum differs from text-token total")
    return values


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    plan_path, plan, plan_sha256 = load_plan(args.plan)
    try:
        task = plan["tasks"][args.task_index]
    except IndexError as error:
        raise ValueError(f"task index {args.task_index} is outside the plan") from error

    receipt_path = Path(args.receipt_dir) / f"{task['task_id']}.json"
    counts_path = Path(args.counts_dir) / f"{task['task_id']}.counts.npy"
    vocab_size = int(plan["tokenizer"]["vocab_size"])
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("plan_sha256") != plan_sha256
            or receipt.get("task_id") != task["task_id"]
            or receipt.get("frequency_counts", {}).get("path") != str(counts_path.resolve())
        ):
            raise ValueError(f"invalid existing frequency receipt: {receipt_path}")
        validate_frequency_array(
            counts_path,
            receipt["frequency_counts"]["sha256"],
            vocab_size,
            int(receipt["text_tokens"]),
        )
        print(json.dumps({"reused": True, "status": "passed", "task_id": task["task_id"]}))
        return receipt

    tokenizer_record = plan["tokenizer"]
    tokenizer_path = Path(tokenizer_record["tokenizer_json"])
    if sha256_file(tokenizer_path) != tokenizer_record["tokenizer_json_sha256"]:
        raise ValueError("tokenizer bytes drifted after planning")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise ValueError("tokenizer vocabulary differs from the plan")

    shard = Path(task["path"])
    parquet = pq.ParquetFile(shard)
    if (
        shard.is_symlink()
        or shard.stat().st_size != int(task["bytes"])
        or sha256_file(shard) != task["sha256"]
        or parquet.metadata.num_rows != int(task["rows"])
    ):
        raise ValueError(f"shard identity failed: {shard}")

    frequencies = np.zeros(vocab_size, dtype=np.uint64)
    documents = 0
    text_tokens = 0
    for texts in text_batches(shard, args.batch_size):
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        documents += len(encodings)
        batch_tokens = sum(len(encoding.ids) for encoding in encodings)
        text_tokens += batch_tokens
        if batch_tokens:
            token_ids = np.fromiter(
                (token_id for encoding in encodings for token_id in encoding.ids),
                dtype=np.int32,
                count=batch_tokens,
            )
            frequencies += np.bincount(token_ids, minlength=vocab_size).astype(
                np.uint64, copy=False
            )

    if documents != int(task["rows"]):
        raise ValueError(f"read {documents} documents, expected {task['rows']}")
    if int(frequencies.sum(dtype=np.uint64)) != text_tokens:
        raise AssertionError("frequency sum differs from observed text-token total")

    atomic_write_npy(counts_path, frequencies)
    counts_sha256 = sha256_file(counts_path)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "plan_sha256": plan_sha256,
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
        "frequency_counts": {
            "path": str(counts_path.resolve()),
            "sha256": counts_sha256,
            "dtype": "uint64",
            "shape": [vocab_size],
            "sum": text_tokens,
        },
    }
    atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "documents": documents,
                "status": "passed",
                "task_id": task["task_id"],
                "text_tokens": text_tokens,
            },
            sort_keys=True,
        )
    )
    return receipt


def bytes_to_unicode_decoder() -> dict[str, int]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values.copy()
    extra = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, codepoints)}


def token_metadata(tokenizer: Tokenizer, vocab_size: int) -> dict[str, list[Any]]:
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    id_to_token: list[str | None] = [None] * vocab_size
    for token, token_id in vocabulary.items():
        if not 0 <= token_id < vocab_size or id_to_token[token_id] is not None:
            raise ValueError(f"invalid or duplicate vocabulary ID {token_id}")
        id_to_token[token_id] = token
    if any(token is None for token in id_to_token):
        raise ValueError("tokenizer vocabulary IDs are not contiguous")

    decoder = bytes_to_unicode_decoder()
    model_tokens: list[str] = []
    payload_hex: list[str | None] = []
    decoded_texts: list[str | None] = []
    utf8_valid: list[bool] = []
    codepoint_counts: list[int | None] = []
    is_subcharacter_fragment: list[bool] = []
    is_single_codepoint: list[bool] = []
    is_char_sized_or_smaller: list[bool] = []
    contains_greek: list[bool] = []
    is_reserved: list[bool] = []

    for token_id, optional_token in enumerate(id_to_token):
        token = str(optional_token)
        reserved = token_id < RESERVED_TOKEN_COUNT
        try:
            payload = bytes(decoder[character] for character in token)
        except KeyError:
            payload = None
        if payload is None:
            decoded = None
            valid = False
        else:
            try:
                decoded = payload.decode("utf-8", errors="strict")
                valid = True
            except UnicodeDecodeError:
                decoded = None
                valid = False
        subcharacter = not reserved and payload is not None and len(payload) > 0 and not valid
        single_codepoint = not reserved and valid and decoded is not None and len(decoded) == 1
        greek = (
            not reserved
            and decoded is not None
            and any("GREEK" in unicodedata.name(character, "") for character in decoded)
        )

        model_tokens.append(token)
        payload_hex.append(None if payload is None else payload.hex())
        decoded_texts.append(decoded)
        utf8_valid.append(valid)
        codepoint_counts.append(None if decoded is None else len(decoded))
        is_subcharacter_fragment.append(subcharacter)
        is_single_codepoint.append(single_codepoint)
        is_char_sized_or_smaller.append(subcharacter or single_codepoint)
        contains_greek.append(greek)
        is_reserved.append(reserved)

    return {
        "model_token": model_tokens,
        "byte_payload_hex": payload_hex,
        "decoded_text": decoded_texts,
        "utf8_valid": utf8_valid,
        "unicode_codepoints": codepoint_counts,
        "is_subcharacter_utf8_fragment": is_subcharacter_fragment,
        "is_single_unicode_codepoint": is_single_codepoint,
        "is_char_sized_or_smaller": is_char_sized_or_smaller,
        "contains_greek": contains_greek,
        "is_reserved_or_special": is_reserved,
    }


def quantiles(values: np.ndarray) -> dict[str, int]:
    probabilities = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1)
    labels = ("min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p999", "max")
    if not values.size:
        return {label: 0 for label in labels}
    results = np.quantile(values, probabilities, method="nearest")
    return {label: int(value) for label, value in zip(labels, results)}


def firing_histogram(values: np.ndarray) -> list[dict[str, int | str]]:
    buckets: dict[str, dict[str, int | str]] = {
        "0": {"range": "0", "token_types": 0, "occurrences": 0}
    }
    for value in values:
        count = int(value)
        if count == 0:
            record = buckets["0"]
        else:
            exponent = count.bit_length() - 1
            lower = 1 << exponent
            upper = (1 << (exponent + 1)) - 1
            key = f"{lower}-{upper}"
            record = buckets.setdefault(
                key, {"range": key, "token_types": 0, "occurrences": 0}
            )
        record["token_types"] = int(record["token_types"]) + 1
        record["occurrences"] = int(record["occurrences"]) + count
    return list(buckets.values())


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    plan_path, plan, plan_sha256 = load_plan(args.plan)
    receipt_dir = Path(args.receipt_dir)
    counts_dir = Path(args.counts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {task["task_id"] for task in plan["tasks"]}
    actual_receipts = {path.stem for path in receipt_dir.glob("*.json")}
    if actual_receipts != expected:
        raise ValueError(
            f"frequency receipt set is not exact: missing={sorted(expected - actual_receipts)}, "
            f"extra={sorted(actual_receipts - expected)}"
        )
    expected_count_files = {f"{task_id}.counts" for task_id in expected}
    actual_count_files = {path.stem for path in counts_dir.glob("*.counts.npy")}
    if actual_count_files != expected_count_files:
        raise ValueError(
            f"frequency array set is not exact: missing={sorted(expected_count_files - actual_count_files)}, "
            f"extra={sorted(actual_count_files - expected_count_files)}"
        )

    vocab_size = int(plan["tokenizer"]["vocab_size"])
    base_counts = np.zeros(vocab_size, dtype=np.uint64)
    transformed_base: dict[int, np.ndarray] = {}
    transformed_cleaned: dict[int, np.ndarray] = {}
    base_text_tokens = 0
    cleaned_replacement_tokens: dict[int, int] = {}
    base_rank_tokens: dict[int, int] = {}
    receipt_records: dict[str, dict[str, Any]] = {}

    for task in plan["tasks"]:
        task_id = task["task_id"]
        receipt_path = receipt_dir / f"{task_id}.json"
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("plan_sha256") != plan_sha256
            or receipt.get("task_id") != task_id
        ):
            raise ValueError(f"frequency receipt binding failed: {receipt_path}")
        counts_path = counts_dir / f"{task_id}.counts.npy"
        values = validate_frequency_array(
            counts_path,
            receipt["frequency_counts"]["sha256"],
            vocab_size,
            int(receipt["text_tokens"]),
        )
        rank = int(task["rank"])
        if task["release"] == "base":
            base_counts += values
            base_text_tokens += int(receipt["text_tokens"])
            base_rank_tokens[rank] = int(receipt["text_tokens"])
            if rank in plan["transformed_ranks"]:
                transformed_base[rank] = np.array(values, copy=True)
        else:
            transformed_cleaned[rank] = np.array(values, copy=True)
            cleaned_replacement_tokens[rank] = int(receipt["text_tokens"])
        receipt_records[task_id] = {
            "receipt_sha256": sha256_file(receipt_path),
            "frequency_counts_sha256": receipt["frequency_counts"]["sha256"],
        }

    transformed_ranks = set(int(rank) for rank in plan["transformed_ranks"])
    if set(transformed_base) != transformed_ranks or set(transformed_cleaned) != transformed_ranks:
        raise ValueError("transformed-rank frequency arrays are incomplete")
    cleaned_counts = base_counts.copy()
    for rank in sorted(transformed_ranks):
        cleaned_counts -= transformed_base[rank]
        cleaned_counts += transformed_cleaned[rank]
    cleaned_text_tokens = base_text_tokens - sum(
        base_rank_tokens[rank] for rank in transformed_ranks
    ) + sum(cleaned_replacement_tokens.values())

    if int(base_counts.sum(dtype=np.uint64)) != base_text_tokens:
        raise AssertionError("base frequency sum differs from receipt totals")
    if int(cleaned_counts.sum(dtype=np.uint64)) != cleaned_text_tokens:
        raise AssertionError("cleaned frequency sum differs from receipt totals")

    previous_summary = load_json(args.previous_summary)
    if (
        previous_summary.get("status") != "passed"
        or previous_summary["plan"]["sha256"] != plan_sha256
        or int(previous_summary["base"]["text_tokens"]) != base_text_tokens
        or int(previous_summary["cleaned"]["text_tokens"]) != cleaned_text_tokens
    ):
        raise ValueError("frequency totals do not reproduce the previous exact token count")

    tokenizer_path = Path(plan["tokenizer"]["tokenizer_json"])
    if sha256_file(tokenizer_path) != plan["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer bytes drifted before frequency aggregation")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    metadata = token_metadata(tokenizer, vocab_size)
    base_path = output_dir / "base_token_frequencies.npy"
    cleaned_path = output_dir / "cleaned_token_frequencies.npy"
    table_path = output_dir / "token_frequencies.parquet"
    atomic_write_npy(base_path, base_counts)
    atomic_write_npy(cleaned_path, cleaned_counts)

    delta = cleaned_counts.astype(np.int64) - base_counts.astype(np.int64)
    table = pa.table(
        {
            "token_id": np.arange(vocab_size, dtype=np.int32),
            **metadata,
            "base_count": base_counts,
            "cleaned_count": cleaned_counts,
            "cleaned_minus_base": delta,
        }
    )
    atomic_write_parquet(table_path, table)

    content_mask = np.arange(vocab_size) >= RESERVED_TOKEN_COUNT
    active_mask = cleaned_counts > 0
    char_mask = np.asarray(metadata["is_char_sized_or_smaller"], dtype=bool)
    greek_mask = np.asarray(metadata["contains_greek"], dtype=bool)
    single_mask = np.asarray(metadata["is_single_unicode_codepoint"], dtype=bool)
    fragment_mask = np.asarray(metadata["is_subcharacter_utf8_fragment"], dtype=bool)

    def mask_stats(mask: np.ndarray) -> dict[str, int | float]:
        occurrences = int(cleaned_counts[mask].sum(dtype=np.uint64))
        return {
            "token_types": int(mask.sum()),
            "active_token_types": int((mask & active_mask).sum()),
            "never_fired_token_types": int((mask & ~active_mask).sum()),
            "occurrences": occurrences,
            "share_of_text_tokens": occurrences / cleaned_text_tokens,
        }

    greek_rows = []
    for token_id in np.flatnonzero(greek_mask):
        greek_rows.append(
            {
                "token_id": int(token_id),
                "model_token": metadata["model_token"][token_id],
                "decoded_text": metadata["decoded_text"][token_id],
                "count": int(cleaned_counts[token_id]),
            }
        )
    never_fired_greek = sorted(
        (row for row in greek_rows if row["count"] == 0), key=lambda row: row["token_id"]
    )
    lowest_positive_greek = sorted(
        (row for row in greek_rows if row["count"] > 0),
        key=lambda row: (row["count"], row["token_id"]),
    )[: args.lowest_greek_limit]
    greek_path = output_dir / "lowest_firing_greek_tokens.json"
    atomic_write_json(
        greek_path,
        {
            "definition": "Non-reserved tokens whose standalone strict UTF-8 payload contains at least one Unicode character with GREEK in its Unicode name.",
            "never_fired": never_fired_greek,
            "lowest_positive": lowest_positive_greek,
        },
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "passed",
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "previous_token_summary": {
            "path": str(Path(args.previous_summary).resolve()),
            "sha256": sha256_file(args.previous_summary),
        },
        "tokenizer": plan["tokenizer"],
        "vocab_size": vocab_size,
        "reserved_or_special_token_count": RESERVED_TOKEN_COUNT,
        "base_text_tokens": base_text_tokens,
        "cleaned_text_tokens": cleaned_text_tokens,
        "frequency_sum_matches_previous_count": True,
        "definitions": {
            "text_firing": "Occurrences emitted by encode_batch(add_special_tokens=False); EOS is excluded.",
            "single_unicode_codepoint": "A non-reserved token whose ByteLevel payload is valid UTF-8 and decodes to exactly one Unicode scalar.",
            "subcharacter_utf8_fragment": "A non-reserved token with a non-empty ByteLevel payload that is not valid standalone UTF-8.",
            "char_sized_or_smaller": "Union of single_unicode_codepoint and subcharacter_utf8_fragment.",
            "greek_token": "A non-reserved token whose standalone strict UTF-8 payload contains a character with GREEK in its Unicode name.",
        },
        "all_token_types": mask_stats(np.ones(vocab_size, dtype=bool)),
        "content_token_types": mask_stats(content_mask),
        "single_unicode_codepoint": mask_stats(single_mask),
        "subcharacter_utf8_fragment": mask_stats(fragment_mask),
        "char_sized_or_smaller": mask_stats(char_mask),
        "greek_tokens": mask_stats(greek_mask),
        "firing_distribution": {
            "all_content_types_quantiles": quantiles(cleaned_counts[content_mask]),
            "positive_content_types_quantiles": quantiles(
                cleaned_counts[content_mask & active_mask]
            ),
            "log2_histogram": firing_histogram(cleaned_counts[content_mask]),
        },
        "artifacts": {
            "base_token_frequencies": {
                "path": str(base_path.resolve()),
                "sha256": sha256_file(base_path),
            },
            "cleaned_token_frequencies": {
                "path": str(cleaned_path.resolve()),
                "sha256": sha256_file(cleaned_path),
            },
            "token_frequency_table": {
                "path": str(table_path.resolve()),
                "sha256": sha256_file(table_path),
                "rows": vocab_size,
            },
            "lowest_firing_greek_tokens": {
                "path": str(greek_path.resolve()),
                "sha256": sha256_file(greek_path),
            },
        },
        "receipts": receipt_records,
    }
    summary_path = output_dir / "frequency_summary.json"
    atomic_write_json(summary_path, summary)
    print(
        f"{summary_path}: cleaned_text_tokens={cleaned_text_tokens}, "
        f"active_types={int(active_mask.sum())}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--plan", required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--receipt-dir", required=True)
    task.add_argument("--counts-dir", required=True)
    task.add_argument("--batch-size", type=int, default=256)
    summary = subparsers.add_parser("aggregate")
    summary.add_argument("--plan", required=True)
    summary.add_argument("--receipt-dir", required=True)
    summary.add_argument("--counts-dir", required=True)
    summary.add_argument("--previous-summary", required=True)
    summary.add_argument("--output-dir", required=True)
    summary.add_argument("--lowest-greek-limit", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "task":
        run_task(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
