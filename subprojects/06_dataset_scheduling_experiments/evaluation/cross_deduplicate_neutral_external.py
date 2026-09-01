#!/usr/bin/env python3
"""Cross-deduplicate a clustered neutral-Greek reserve against CPT training.

The pipeline deliberately reuses the exact-hash and sorted MinHash indexes that
created the frozen training release.  It has three receipt-bound subcommands:

* ``candidate-signatures`` builds fragment- and sitting-level candidate docs;
* ``match-bucket`` compares one of 32 candidate bands with all training ranks;
* ``finalize`` verifies real shingle Jaccard, removes whole leaking clusters,
  and selects a deterministic 10--20M-token panel.

No LSH collision is treated as a proven near duplicate until the underlying
5-token shingle Jaccard is at least 0.85.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping


BUCKETS = 32
ROWS_PER_BAND = 4
PERMUTATIONS = 128
SHINGLE_SIZE = 5
THRESHOLD = 0.85
CANDIDATE_RANK = 90_000
TARGET_TOKENS = 15_000_000
MIN_TOKENS = 10_000_000
MAX_TOKENS = 20_000_000
MAX_GROUP_DOCUMENTS = 5_000
AGGREGATE_DOC_TYPE = "complete_sitting"
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path, **extra: object) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def verify_file(row: Mapping[str, Any], *, root: Path | None = None) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute() and root is not None:
        path = root / path
    path = path.resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise ValueError(f"file receipt drift: {path}")
    return path


def relaxed_exact_text(text: str) -> str:
    """Match the conservative exact normalization used by Agent-1 v5."""

    value = unicodedata.normalize("NFC", text).casefold()
    value = PUNCT_RE.sub(" ", value)
    return WHITESPACE_RE.sub(" ", value).strip()


def validate_training_config(path: Path) -> dict[str, Any]:
    value = read_json(path)
    dedup = value.get("dedup", {})
    expected = {
        "hash_precision": 64,
        "hashes_per_bucket": ROWS_PER_BAND,
        "language": "ell_Grek",
        "num_buckets": BUCKETS,
        "num_permutations": PERMUTATIONS,
        "preserve_greek_diacritics": True,
        "seed": 1,
        "shingle_size": SHINGLE_SIZE,
        "verified_jaccard_threshold": THRESHOLD,
    }
    if any(dedup.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("training MinHash geometry drift")
    return value


def minhash_config(training_config: Mapping[str, Any]) -> Any:
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.text import TextNormConfig

    dedup = training_config["dedup"]
    return MinhashConfig(
        n_grams=SHINGLE_SIZE,
        num_buckets=BUCKETS,
        hashes_per_bucket=ROWS_PER_BAND,
        seed=1,
        norm_config=TextNormConfig(norm_unicode_diacritics=not bool(dedup["preserve_greek_diacritics"])),
        hash_config=HashConfig(precision=64),
    )


def source_inputs(source_receipt_path: Path) -> tuple[dict[str, Any], Path, Path]:
    receipt = read_json(source_receipt_path)
    if receipt.get("schema_version") != "apertus_mini_neutral_source_preparation_receipt_v1" or receipt.get("status") != "passed":
        raise ValueError("neutral source preparation is not passed")
    fragments = verify_file(receipt["candidate_fragments"])
    snapshot = verify_file(receipt["source_snapshot"])
    verify_file(receipt["cluster_manifest"])
    return receipt, fragments, snapshot


def candidate_signatures(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datatrove.data import Document
    from datatrove.pipeline.dedup import MinhashDedupSignature

    source_receipt, fragments_path, _ = source_inputs(args.source_preparation_receipt)
    training_config = validate_training_config(args.training_config)
    signature_manifest = read_json(args.training_signature_manifest)
    if (
        signature_manifest.get("schema_version") != "agent1_v5_minhash_signature_manifest_v1"
        or signature_manifest.get("status") != "passed"
        or int(signature_manifest.get("bucket_count", -1)) != BUCKETS
        or int(signature_manifest.get("task_count", -1)) != 431
        or signature_manifest.get("combined_manifest_sha256")
        != "4ff9f598f0e592324ae08c139e7f241344bdd180497b39f011d810a990ffdacf"
    ):
        raise ValueError("training signature manifest is not a complete 32x431 index")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    cluster_fragments: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with fragments_path.open(encoding="utf-8") as handle:
        for source_position, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            cluster = str(row["cluster_id"])
            text = str(row["text"])
            order = int(row["fragment_order"])
            rows.append(
                {
                    "doc_index": len(rows),
                    "cluster_id": cluster,
                    "source_id": str(row["source_id"]),
                    "source_doc_id": str(row["source_doc_id"]),
                    "doc_type": "speech_fragment",
                    "source_position": source_position,
                    "fragment_order": order,
                    "text": text,
                    "strict_hash": hashlib.sha256(text.encode()).hexdigest(),
                    "relaxed_hash": hashlib.sha256(relaxed_exact_text(text).encode()).hexdigest(),
                }
            )
            cluster_fragments[cluster].append((order, text))
    fragment_count = len(rows)
    for cluster in sorted(cluster_fragments):
        ordered = sorted(cluster_fragments[cluster])
        text = "\n\n".join(value for _, value in ordered)
        rows.append(
            {
                "doc_index": len(rows),
                "cluster_id": cluster,
                "source_id": "zenodo_greek_parliament_proceedings_2587904",
                "source_doc_id": f"{cluster}:complete-sitting",
                "doc_type": "complete_sitting",
                "source_position": -1,
                "fragment_order": -1,
                "text": text,
                "strict_hash": hashlib.sha256(text.encode()).hexdigest(),
                "relaxed_hash": hashlib.sha256(relaxed_exact_text(text).encode()).hexdigest(),
            }
        )
    if not rows or len(cluster_fragments) != int(read_json(Path(source_receipt["cluster_manifest"]["path"]))["reserve_counts"]["document_clusters"]):
        raise ValueError("candidate document/cluster closure failed")

    docs_path = output_root / "candidate_documents.parquet"
    schema = pa.schema(
        [
            ("doc_index", pa.uint32()),
            ("cluster_id", pa.string()),
            ("source_id", pa.string()),
            ("source_doc_id", pa.string()),
            ("doc_type", pa.string()),
            ("source_position", pa.int64()),
            ("fragment_order", pa.int32()),
            ("text", pa.large_string()),
            ("strict_hash", pa.string()),
            ("relaxed_hash", pa.string()),
        ]
    )
    partial = Path(str(docs_path) + ".partial")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), partial, compression="zstd", compression_level=5)
    partial.replace(docs_path)

    signatures = output_root / "candidate_minhash_signatures"
    step = MinhashDedupSignature(
        output_folder=str(signatures),
        config=minhash_config(training_config),
        language="ell_Grek",
    )
    step.run((Document(text=row["text"], id=str(row["doc_index"])) for row in rows), rank=CANDIDATE_RANK, world_size=1)
    outputs = []
    for bucket in range(BUCKETS):
        path = signatures / f"bucket_{bucket:03d}" / f"{CANDIDATE_RANK:05d}.minhash.sig"
        outputs.append({"bucket": bucket, **file_receipt(path)})
    payload = {
        "schema_version": "apertus_mini_neutral_candidate_signatures_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_preparation_receipt": file_receipt(args.source_preparation_receipt),
        "training_config": file_receipt(args.training_config),
        "training_signature_manifest": file_receipt(args.training_signature_manifest),
        "candidate_rank": CANDIDATE_RANK,
        "candidate_documents": file_receipt(
            docs_path,
            rows=len(rows),
            speech_fragments=fragment_count,
            complete_sittings=len(cluster_fragments),
        ),
        "geometry": {
            "token_shingle_size": SHINGLE_SIZE,
            "permutations": PERMUTATIONS,
            "bands": BUCKETS,
            "rows_per_band": ROWS_PER_BAND,
            "threshold": THRESHOLD,
            "language": "ell_Grek",
            "preserve_greek_diacritics": True,
            "seed": 1,
            "hash_precision": 64,
            "hash_function": "xxhash",
        },
        "outputs": outputs,
    }
    receipt_path = output_root / "candidate_signatures_receipt.json"
    write_json_atomic(receipt_path, payload)
    print(json.dumps({"ok": True, "documents": len(rows), "clusters": len(cluster_fragments), "receipt": str(receipt_path)}, sort_keys=True))
    return 0


def _signature_dtypes() -> tuple[Any, Any]:
    import numpy as np

    record = np.dtype([(f"s{index}", "<u8") for index in range(ROWS_PER_BAND)] + [("doc", "<u4")])
    signature = np.dtype([(f"s{index}", "<u8") for index in range(ROWS_PER_BAND)])
    return record, signature


def _signature_only(records: Any, signature_dtype: Any) -> Any:
    import numpy as np

    result = np.empty(len(records), dtype=signature_dtype)
    for index in range(ROWS_PER_BAND):
        result[f"s{index}"] = records[f"s{index}"]
    return result


def _candidate_groups(records: Any, signatures: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    if not len(records):
        return signatures, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    changed = np.ones(len(records), dtype=np.bool_)
    changed[1:] = signatures[1:] != signatures[:-1]
    starts = np.flatnonzero(changed)
    ends = np.r_[starts[1:], len(records)]
    return signatures[starts], starts, ends


def _require_sorted_signatures(signatures: Any, *, label: str) -> None:
    import numpy as np

    if len(signatures) < 2:
        return
    order = np.argsort(signatures, order=list(signatures.dtype.names), kind="stable")
    if not np.array_equal(order, np.arange(len(signatures))):
        raise ValueError(f"MinHash signatures are not sorted: {label}")


def match_bucket(args: argparse.Namespace) -> int:
    import numpy as np

    bucket = int(args.bucket)
    if not 0 <= bucket < BUCKETS:
        raise ValueError(bucket)
    candidate = read_json(args.candidate_signatures_receipt)
    if candidate.get("schema_version") != "apertus_mini_neutral_candidate_signatures_v1" or candidate.get("status") != "passed":
        raise ValueError("candidate signatures are not passed")
    if verify_file(candidate["training_signature_manifest"]) != args.training_signature_manifest.resolve():
        raise ValueError("candidate/training signature-manifest binding drift")
    training_manifest = read_json(args.training_signature_manifest)
    if int(training_manifest.get("task_count", -1)) != 431 or int(training_manifest.get("bucket_count", -1)) != BUCKETS:
        raise ValueError("training signature inventory drift")
    candidate_binding = next((row for row in candidate["outputs"] if int(row["bucket"]) == bucket), None)
    if candidate_binding is None:
        raise ValueError("candidate bucket missing")
    candidate_path = verify_file(candidate_binding)
    record_dtype, signature_dtype = _signature_dtypes()
    candidate_records = np.fromfile(candidate_path, dtype=record_dtype)
    candidate_signatures = _signature_only(candidate_records, signature_dtype)
    _require_sorted_signatures(candidate_signatures, label=str(candidate_path))
    unique_signatures, starts, ends = _candidate_groups(candidate_records, candidate_signatures)

    output = args.output.resolve()
    if output.exists() or Path(str(output) + ".receipt.json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    training_root = args.training_signature_root.resolve()
    run_root = training_root.parents[1]
    input_bindings: list[dict[str, Any]] = []
    training_match_groups = 0
    raw_cross_pairs = 0
    internal_groups = 0
    oversized_groups = 0
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        for start, end in zip(starts, ends):
            if int(end - start) > 1:
                docs = candidate_records["doc"][start:end].astype(int).tolist()
                if len(docs) > MAX_GROUP_DOCUMENTS:
                    oversized_groups += 1
                handle.write(json.dumps({"kind": "candidate_internal", "candidate_doc_ids": docs}, sort_keys=True) + "\n")
                internal_groups += 1
        for rank in range(int(training_manifest["task_count"])):
            receipt_path = training_root / "receipts" / f"{rank:06d}.json"
            receipt = read_json(receipt_path)
            if receipt.get("status") != "passed" or int(receipt.get("task_index", -1)) != rank:
                raise ValueError(f"training signature receipt drift: {receipt_path}")
            suffix = f"bucket_{bucket:03d}/{rank:05d}.minhash.sig"
            binding = next((row for row in receipt.get("outputs", []) if str(row.get("path", "")).endswith(suffix)), None)
            if binding is None:
                raise ValueError(f"training bucket binding missing: rank={rank} bucket={bucket}")
            path = Path(str(binding["path"]))
            if not path.is_absolute():
                path = run_root / path
            path = path.resolve()
            if path.is_symlink() or not path.is_file() or path.stat().st_size != int(binding["bytes"]):
                raise ValueError(f"training signature stat drift: {path}")
            records = np.fromfile(path, dtype=record_dtype)
            observed_sha = hashlib.sha256(memoryview(records).cast("B")).hexdigest()
            if observed_sha != binding["sha256"]:
                raise ValueError(f"training signature content drift: {path}")
            signatures = _signature_only(records, signature_dtype)
            _require_sorted_signatures(signatures, label=str(path))
            left = np.searchsorted(signatures, unique_signatures, side="left")
            right = np.searchsorted(signatures, unique_signatures, side="right")
            matched = np.flatnonzero(left < right)
            for group_index in matched:
                candidate_docs = candidate_records["doc"][starts[group_index] : ends[group_index]].astype(int).tolist()
                training_docs = records["doc"][left[group_index] : right[group_index]].astype(int).tolist()
                group_size = len(candidate_docs) + len(training_docs)
                if group_size > MAX_GROUP_DOCUMENTS:
                    oversized_groups += 1
                raw_cross_pairs += len(candidate_docs) * len(training_docs)
                handle.write(
                    json.dumps(
                        {
                            "kind": "candidate_training",
                            "candidate_doc_ids": candidate_docs,
                            "training_rank": rank,
                            "training_doc_ids": training_docs,
                            "oversized": group_size > MAX_GROUP_DOCUMENTS,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                training_match_groups += 1
            input_bindings.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "bytes": int(binding["bytes"]),
                    "sha256": observed_sha,
                    "receipt_sha256": sha256_file(receipt_path),
                }
            )
    partial.replace(output)
    receipt_payload = {
        "schema_version": "apertus_mini_neutral_minhash_bucket_matches_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bucket": bucket,
        "candidate_signatures_receipt": file_receipt(args.candidate_signatures_receipt),
        "training_signature_manifest": file_receipt(args.training_signature_manifest),
        "candidate_signature": file_receipt(candidate_path),
        "training_signature_files": len(input_bindings),
        "training_signature_bytes": sum(row["bytes"] for row in input_bindings),
        "training_signature_binding_set_sha256": hashlib.sha256(canonical_json(input_bindings).encode()).hexdigest(),
        "counts": {
            "candidate_internal_signature_groups": internal_groups,
            "candidate_training_signature_groups": training_match_groups,
            "raw_candidate_training_pairs": raw_cross_pairs,
            "oversized_signature_groups": oversized_groups,
        },
        "matches": file_receipt(output),
    }
    receipt_path = Path(str(output) + ".receipt.json")
    write_json_atomic(receipt_path, receipt_payload)
    print(json.dumps({"ok": True, "bucket": bucket, **receipt_payload["counts"], "receipt": str(receipt_path)}, sort_keys=True))
    return 0


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            keep, drop = sorted((a, b))
            self.parent[drop] = keep


def _verify_exact_rank(args: tuple[int, Path, Path, Path]) -> dict[str, Any]:
    rank, exact_root, receipt_root, run_root = args
    receipt_path = receipt_root / f"{rank:06d}.json"
    receipt = read_json(receipt_path)
    if receipt.get("status") != "passed" or int(receipt.get("task_index", -1)) != rank:
        raise ValueError(f"exact-index receipt drift: {receipt_path}")
    binding = receipt["output"]
    path = Path(str(binding["path"]))
    if not path.is_absolute():
        path = run_root / path
    path = path.resolve()
    if path != (exact_root / f"{rank:06d}.parquet").resolve():
        raise ValueError("exact-index path drift")
    verify_file(binding, root=run_root)
    return {"rank": rank, "path": str(path), "bytes": int(binding["bytes"]), "sha256": binding["sha256"], "receipt_sha256": sha256_file(receipt_path)}


def _jaccard(left: Any, right: Any) -> float:
    import numpy as np

    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    union = int(left.size) + int(right.size) - intersection
    # Two texts shorter than the configured shingle size both produce empty
    # shingle sets.  That is absence of comparison evidence, not a perfect
    # near-duplicate match.  Treating empty/empty as 1.0 can bridge thousands
    # of otherwise unrelated document clusters through short boilerplate.
    return intersection / union if union else 0.0


def _aggregate_doc_ids(doc_ids: Iterable[int], by_doc: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """Return cluster-level documents used for candidate-internal dedup.

    Speech turns are valid probes for leakage into the training corpus, but a
    repeated turn such as a procedural phrase does not make two complete
    parliamentary sittings duplicate documents.  Candidate-internal
    near-dedup therefore compares only the complete cluster aggregates.
    """

    return [doc for doc in doc_ids if str(by_doc[doc]["doc_type"]) == AGGREGATE_DOC_TYPE]


def _scan_exact_matches(
    rows: list[dict[str, Any]], exact_bindings: list[dict[str, Any]]
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]], set[str], list[list[int]]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    strict_lookup: dict[str, list[int]] = defaultdict(list)
    relaxed_lookup: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        doc = int(row["doc_index"])
        strict_lookup[str(row["strict_hash"])].append(doc)
        relaxed_lookup[str(row["relaxed_hash"])].append(doc)
    strict_values = pa.array(sorted(strict_lookup), type=pa.string())
    relaxed_values = pa.array(sorted(relaxed_lookup), type=pa.string())
    strict_matches: set[tuple[int, int, int]] = set()
    relaxed_matches: set[tuple[int, int, int]] = set()
    source_inventory: set[str] = set()
    for binding in exact_bindings:
        table = pq.read_table(
            binding["path"],
            columns=["rank", "doc_index", "source_dataset", "strict_hash", "relaxed_hash"],
        )
        source_inventory.update(str(value) for value in pc.unique(table["source_dataset"]).to_pylist())
        strict_rows = table.filter(pc.is_in(table["strict_hash"], value_set=strict_values))
        for digest, rank, training_doc in zip(
            strict_rows["strict_hash"].to_pylist(),
            strict_rows["rank"].to_pylist(),
            strict_rows["doc_index"].to_pylist(),
        ):
            strict_matches.update(
                (candidate_doc, int(rank), int(training_doc))
                for candidate_doc in strict_lookup[str(digest)]
            )
        relaxed_rows = table.filter(pc.is_in(table["relaxed_hash"], value_set=relaxed_values))
        for digest, rank, training_doc in zip(
            relaxed_rows["relaxed_hash"].to_pylist(),
            relaxed_rows["rank"].to_pylist(),
            relaxed_rows["doc_index"].to_pylist(),
        ):
            relaxed_matches.update(
                (candidate_doc, int(rank), int(training_doc))
                for candidate_doc in relaxed_lookup[str(digest)]
            )
    internal_exact = [docs for docs in strict_lookup.values() if len(docs) > 1]
    return strict_matches, relaxed_matches, source_inventory, internal_exact


def finalize(args: argparse.Namespace) -> int:
    import numpy as np
    import pyarrow.parquet as pq
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from tokenizers import Tokenizer

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    candidate = read_json(args.candidate_signatures_receipt)
    if candidate.get("schema_version") != "apertus_mini_neutral_candidate_signatures_v1" or candidate.get("status") != "passed":
        raise ValueError("candidate signatures are not passed")
    if verify_file(candidate["training_config"]) != args.training_config.resolve():
        raise ValueError("candidate signature training-config binding drift")
    if verify_file(candidate["training_signature_manifest"]) != args.training_signature_manifest.resolve():
        raise ValueError("candidate signature training-manifest binding drift")
    docs_path = verify_file(candidate["candidate_documents"])
    verify_file(candidate["source_preparation_receipt"])
    _, _, source_snapshot_path = source_inputs(Path(candidate["source_preparation_receipt"]["path"]))
    training_config = validate_training_config(args.training_config)
    pool = read_json(args.pool_corpus_receipt)
    if pool.get("schema_version") != "apertus_mini_schedule_pool_corpus_v1" or pool.get("status") != "completed":
        raise ValueError("training pool corpus is not frozen")
    combined = read_json(args.training_combined_manifest)
    if combined.get("status") != "passed" or int(combined.get("rows", -1)) != 53_046_533 or len(combined.get("files", [])) != 431:
        raise ValueError("training pre-dedup manifest drift")
    if sha256_file(args.training_combined_manifest) != "4ff9f598f0e592324ae08c139e7f241344bdd180497b39f011d810a990ffdacf":
        raise ValueError("training combined-manifest hash drift")
    training_run_root = args.training_exact_root.resolve().parents[1]
    exact_jobs = [
        (rank, args.training_exact_root.resolve(), args.training_exact_receipts.resolve(), training_run_root)
        for rank in range(431)
    ]
    with ThreadPoolExecutor(max_workers=args.hash_workers) as executor:
        exact_bindings = list(executor.map(_verify_exact_rank, exact_jobs))

    table = pq.read_table(docs_path)
    rows = table.to_pylist()
    by_doc = {int(row["doc_index"]): row for row in rows}
    if len(by_doc) != len(rows):
        raise ValueError("candidate doc-index collision")
    clusters = {str(row["cluster_id"]) for row in rows}
    aggregates = {str(row["cluster_id"]): row for row in rows if row["doc_type"] == "complete_sitting"}
    if set(aggregates) != clusters:
        raise ValueError("candidate aggregate closure failed")

    strict_matches, relaxed_matches, source_inventory_set, internal_exact = _scan_exact_matches(rows, exact_bindings)
    source_inventory = sorted(source_inventory_set)

    external_pairs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    internal_pairs: set[tuple[int, int]] = set()
    conservative_excluded_docs: set[int] = set()
    bucket_receipts = []
    for receipt_path in args.bucket_receipt:
        receipt = read_json(receipt_path)
        if receipt.get("schema_version") != "apertus_mini_neutral_minhash_bucket_matches_v1" or receipt.get("status") != "passed":
            raise ValueError(f"bucket receipt is not passed: {receipt_path}")
        if verify_file(receipt["candidate_signatures_receipt"]) != args.candidate_signatures_receipt.resolve():
            raise ValueError("bucket candidate-signature binding drift")
        if verify_file(receipt["training_signature_manifest"]) != args.training_signature_manifest.resolve():
            raise ValueError("bucket training-signature binding drift")
        match_path = verify_file(receipt["matches"])
        with gzip.open(match_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidate_docs = [int(value) for value in row["candidate_doc_ids"]]
                if row["kind"] == "candidate_internal":
                    candidate_docs = _aggregate_doc_ids(candidate_docs, by_doc)
                    if len(candidate_docs) > MAX_GROUP_DOCUMENTS:
                        raise ValueError("oversized aggregate-only internal MinHash group")
                    for position, left in enumerate(candidate_docs):
                        for right in candidate_docs[position + 1 :]:
                            internal_pairs.add(tuple(sorted((left, right))))
                elif row["kind"] == "candidate_training":
                    if len(candidate_docs) > MAX_GROUP_DOCUMENTS or row.get("oversized") is True:
                        conservative_excluded_docs.update(candidate_docs)
                        continue
                    rank = int(row["training_rank"])
                    training_docs = [int(value) for value in row["training_doc_ids"]]
                    if len(training_docs) + len(candidate_docs) > MAX_GROUP_DOCUMENTS:
                        conservative_excluded_docs.update(candidate_docs)
                        continue
                    for candidate_doc in candidate_docs:
                        external_pairs[candidate_doc].update((rank, training_doc) for training_doc in training_docs)
                else:
                    raise ValueError("unknown bucket match row")
        bucket_receipts.append(file_receipt(receipt_path, bucket=int(receipt["bucket"])))
    if sorted(row["bucket"] for row in bucket_receipts) != list(range(BUCKETS)):
        raise ValueError("exactly one passed receipt is required for each MinHash bucket")

    for docs in internal_exact:
        values = _aggregate_doc_ids((int(value) for value in docs), by_doc)
        for position, left in enumerate(values):
            for right in values[position + 1 :]:
                internal_pairs.add(tuple(sorted((left, right))))
    exact_external_docs = {int(row[0]) for row in strict_matches | relaxed_matches}
    exactly_matched_clusters = {str(by_doc[doc]["cluster_id"]) for doc in exact_external_docs}
    conservatively_excluded_clusters = {str(by_doc[doc]["cluster_id"]) for doc in conservative_excluded_docs}
    externally_matched_clusters = exactly_matched_clusters | conservatively_excluded_clusters
    exact_overlap_by_doc_type = {
        doc_type: sum(1 for doc in exact_external_docs if str(by_doc[doc]["doc_type"]) == doc_type)
        for doc_type in ("speech_fragment", "complete_sitting")
    }
    exact_fragment_cluster_counts_by_min_chars = {
        str(threshold): len(
            {
                str(by_doc[doc]["cluster_id"])
                for doc in exact_external_docs
                if str(by_doc[doc]["doc_type"]) == "speech_fragment"
                and len(str(by_doc[doc]["text"])) >= threshold
            }
        )
        for threshold in (50, 100, 200, 500, 1000, 2000)
    }

    output_root.mkdir(parents=True)
    hasher = MinhashDedupSignature(
        output_folder=str(output_root / ".unused-signatures"),
        config=minhash_config(training_config),
        language="ell_Grek",
    )
    candidate_shingles: dict[int, Any] = {}

    def cshingles(doc: int) -> Any:
        if doc not in candidate_shingles:
            candidate_shingles[doc] = np.unique(hasher.get_shingles(str(by_doc[doc]["text"])).reshape(-1))
        return candidate_shingles[doc]

    needed_training: dict[int, set[int]] = defaultdict(set)
    for candidate_doc, refs in external_pairs.items():
        if str(by_doc[candidate_doc]["cluster_id"]) in externally_matched_clusters:
            continue
        if len(refs) > args.max_training_refs_per_candidate:
            conservative_excluded_docs.add(candidate_doc)
            externally_matched_clusters.add(str(by_doc[candidate_doc]["cluster_id"]))
            continue
        for rank, doc in refs:
            needed_training[rank].add(doc)
    training_text: dict[tuple[int, int], str] = {}
    combined_root = Path(str(combined["root"]))
    for rank, doc_ids in needed_training.items():
        binding = combined["files"][rank]
        source = verify_file(binding, root=combined_root)
        texts = pq.read_table(source, columns=["text"]).column(0)
        for doc in sorted(doc_ids):
            if not 0 <= doc < len(texts):
                raise ValueError("training MinHash reference out of bounds")
            training_text[(rank, doc)] = str(texts[doc].as_py() or "")
    training_shingles: dict[tuple[int, int], Any] = {}

    def tshingles(ref: tuple[int, int]) -> Any:
        if ref not in training_shingles:
            training_shingles[ref] = np.unique(hasher.get_shingles(training_text[ref]).reshape(-1))
        return training_shingles[ref]

    actual_external_pairs = 0
    actual_external_docs: set[int] = set()
    for candidate_doc in sorted(external_pairs):
        cluster = str(by_doc[candidate_doc]["cluster_id"])
        if cluster in externally_matched_clusters:
            continue
        for ref in sorted(external_pairs[candidate_doc]):
            if ref not in training_text:
                continue
            if _jaccard(cshingles(candidate_doc), tshingles(ref)) >= THRESHOLD:
                externally_matched_clusters.add(cluster)
                actual_external_pairs += 1
                actual_external_docs.add(candidate_doc)
                break
    minhash_only_matched_clusters = (
        externally_matched_clusters - exactly_matched_clusters - conservatively_excluded_clusters
    )

    union = UnionFind()
    actual_internal_pairs = 0
    actual_internal_pair_docs: set[int] = set()
    actual_internal_pairs_by_min_chars = {threshold: 0 for threshold in (50, 100, 200, 500, 1000, 2000)}
    for left, right in sorted(internal_pairs):
        left_cluster = str(by_doc[left]["cluster_id"])
        right_cluster = str(by_doc[right]["cluster_id"])
        if left_cluster == right_cluster:
            continue
        if _jaccard(cshingles(left), cshingles(right)) >= THRESHOLD:
            union.union(left_cluster, right_cluster)
            actual_internal_pairs += 1
            actual_internal_pair_docs.update((left, right))
            shorter_chars = min(len(str(by_doc[left]["text"])), len(str(by_doc[right]["text"])))
            for threshold in actual_internal_pairs_by_min_chars:
                if shorter_chars >= threshold:
                    actual_internal_pairs_by_min_chars[threshold] += 1
    components: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        components[union.find(cluster)].append(cluster)
    internally_removed_clusters: set[str] = set()
    internally_linked_components = 0
    largest_internal_component = 1
    for members in components.values():
        largest_internal_component = max(largest_internal_component, len(members))
        if len(members) > 1:
            internally_linked_components += 1
            keep = min(members, key=lambda value: hashlib.sha256(f"neutral-internal-keep-v1:{value}".encode()).digest())
            internally_removed_clusters.update(value for value in members if value != keep)
    eligible = sorted(clusters - externally_matched_clusters - internally_removed_clusters)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir.resolve() / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != 148_992:
        raise ValueError("neutral selection tokenizer vocabulary drift")
    config = read_json(args.tokenizer_dir.resolve() / "tokenizer_config.json")
    eos_value = config.get("eos_token")
    eos_token = eos_value.get("content") if isinstance(eos_value, dict) else eos_value
    eos_id = tokenizer.token_to_id(eos_token) if isinstance(eos_token, str) else None
    if eos_id is None:
        raise ValueError("neutral selection tokenizer has no EOS")
    selected: list[tuple[str, int]] = []
    total_tokens = 0
    ordered_eligible = sorted(eligible, key=lambda value: hashlib.sha256(f"neutral-final-v1:{value}".encode()).digest())
    for cluster in ordered_eligible:
        tokens = len(tokenizer.encode(str(aggregates[cluster]["text"]), add_special_tokens=False).ids) + 1
        if total_tokens + tokens > MAX_TOKENS:
            continue
        selected.append((cluster, tokens))
        total_tokens += tokens
        if total_tokens >= args.target_tokens:
            break
    if not MIN_TOKENS <= total_tokens <= MAX_TOKENS:
        failure = {
            "schema_version": "apertus_mini_neutral_external_insufficient_reserve_v1",
            "status": "blocked_insufficient_after_cross_dedup",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "counts": {
                "reserve_clusters": len(clusters),
                "exactly_matched_clusters": len(exactly_matched_clusters),
                "minhash_only_matched_clusters": len(minhash_only_matched_clusters),
                "conservatively_excluded_candidate_docs": len(conservative_excluded_docs),
                "conservatively_excluded_clusters": len(conservatively_excluded_clusters),
                "internally_removed_clusters": len(internally_removed_clusters),
                "eligible_clusters": len(eligible),
                "selected_tokens_including_eos": total_tokens,
            },
            "exact_overlap": {
                "strict_match_triplets": len(strict_matches),
                "relaxed_match_triplets": len(relaxed_matches),
                "candidate_documents": len(exact_external_docs),
                "candidate_documents_by_type": exact_overlap_by_doc_type,
                "speech_fragment_clusters_by_minimum_character_length": exact_fragment_cluster_counts_by_min_chars,
            },
            "verified_minhash": {
                "candidate_internal_pairs_at_or_above_threshold": actual_internal_pairs,
                "candidate_to_training_pairs_at_or_above_threshold": actual_external_pairs,
                "candidate_internal_documents_at_or_above_threshold": len(actual_internal_pair_docs),
                "candidate_to_training_documents_at_or_above_threshold": len(actual_external_docs),
                "candidate_internal_documents_by_type": {
                    doc_type: sum(
                        1
                        for doc in actual_internal_pair_docs
                        if str(by_doc[doc]["doc_type"]) == doc_type
                    )
                    for doc_type in ("speech_fragment", "complete_sitting")
                },
                "candidate_to_training_documents_by_type": {
                    doc_type: sum(
                        1
                        for doc in actual_external_docs
                        if str(by_doc[doc]["doc_type"]) == doc_type
                    )
                    for doc_type in ("speech_fragment", "complete_sitting")
                },
                "candidate_internal_pairs_by_minimum_document_character_length": {
                    str(threshold): count
                    for threshold, count in actual_internal_pairs_by_min_chars.items()
                },
                "candidate_to_training_clusters_by_minimum_document_character_length": {
                    str(threshold): len(
                        {
                            str(by_doc[doc]["cluster_id"])
                            for doc in actual_external_docs
                            if len(str(by_doc[doc]["text"])) >= threshold
                        }
                    )
                    for threshold in (50, 100, 200, 500, 1000, 2000)
                },
                "internally_linked_components": internally_linked_components,
                "largest_internal_component_clusters": largest_internal_component,
                "threshold": THRESHOLD,
            },
            "bindings": {
                "candidate_signatures_receipt": file_receipt(args.candidate_signatures_receipt),
                "training_combined_manifest": file_receipt(args.training_combined_manifest),
                "training_signature_manifest": file_receipt(args.training_signature_manifest),
                "pool_corpus_receipt": file_receipt(args.pool_corpus_receipt),
            },
        }
        write_json_atomic(output_root / "insufficient_reserve_receipt.json", failure)
        raise ValueError(
            f"deduplicated neutral reserve is insufficient: eligible_clusters={len(eligible)} selected_tokens={total_tokens}"
        )

    candidate_output = output_root / "neutral_external_candidates.jsonl"
    with candidate_output.open("w", encoding="utf-8") as handle:
        for cluster, tokens in selected:
            row = aggregates[cluster]
            handle.write(
                json.dumps(
                    {
                        "cluster_id": cluster,
                        "source_id": str(row["source_id"]),
                        "source_doc_id": str(row["source_doc_id"]),
                        "text": str(row["text"]),
                        "selection_tokens_including_eos": tokens,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    snapshot = read_json(source_snapshot_path)
    source_snapshot_receipts = [
        file_receipt(source_snapshot_path),
        snapshot["metadata"],
        snapshot["archive"],
    ]
    for binding in source_snapshot_receipts:
        verify_file(binding)
    receipt = {
        "schema_version": "apertus_mini_neutral_external_dedup_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_reference": {"pool_corpus_receipt": file_receipt(args.pool_corpus_receipt)},
        "dataset_separation": {
            "document_cluster_split": True,
            "publishers_or_domains_absent_from_training": "zenodo_greek_parliament_proceedings_2587904" not in source_inventory,
            "source_time_window_absent_from_training": False,
            "source_separation_rule": "publisher_or_domain_or_time_window",
            "candidate_documents_never_used_for_training": True,
            "evaluation_use_authorized": snapshot.get("license") == "cc-by-4.0",
        },
        "exact_dedup": {
            "algorithm": "sha256_utf8_text",
            "candidate_internal_duplicate_rows": 0,
            "candidate_to_training_match_rows": 0,
            "prefilter_candidate_to_training_strict_matches": len(strict_matches),
            "prefilter_candidate_to_training_relaxed_matches": len(relaxed_matches),
            "training_exact_index_files": len(exact_bindings),
            "training_exact_index_binding_set_sha256": hashlib.sha256(canonical_json(exact_bindings).encode()).hexdigest(),
        },
        "minhash_dedup": {
            "token_shingle_size": SHINGLE_SIZE,
            "permutations": PERMUTATIONS,
            "bands": BUCKETS,
            "rows_per_band": ROWS_PER_BAND,
            "threshold": THRESHOLD,
            "candidate_internal_pairs_at_or_above_threshold": 0,
            "candidate_to_training_pairs_at_or_above_threshold": 0,
            "prefilter_verified_internal_pairs_at_or_above_threshold": actual_internal_pairs,
            "prefilter_verified_candidate_to_training_pairs_at_or_above_threshold": actual_external_pairs,
            "conservative_oversized_or_high_fanout_candidate_docs": len(conservative_excluded_docs),
            "training_signature_manifest": file_receipt(args.training_signature_manifest),
            "bucket_match_receipts": bucket_receipts,
        },
        "source_snapshot_receipts": source_snapshot_receipts,
        "candidate_output": file_receipt(candidate_output, rows=len(selected), tokens_including_eos=total_tokens),
        "selection": {
            "target_tokens_including_eos": args.target_tokens,
            "actual_tokens_including_eos": total_tokens,
            "reserve_clusters": len(clusters),
            "eligible_clusters_after_cross_dedup": len(eligible),
            "selected_clusters": len(selected),
            "externally_matched_clusters_removed": len(externally_matched_clusters),
            "internal_duplicate_clusters_removed": len(internally_removed_clusters),
        },
        "bindings": {
            "candidate_signatures_receipt": file_receipt(args.candidate_signatures_receipt),
            "training_config": file_receipt(args.training_config),
            "training_combined_manifest": file_receipt(args.training_combined_manifest),
            "training_source_inventory_sha256": hashlib.sha256(canonical_json(source_inventory).encode()).hexdigest(),
            "tokenizer_json": file_receipt(args.tokenizer_dir.resolve() / "tokenizer.json"),
        },
    }
    if not (
        receipt["dataset_separation"]["publishers_or_domains_absent_from_training"]
        or receipt["dataset_separation"]["source_time_window_absent_from_training"]
    ):
        raise ValueError("neutral source is not externally separated from training")
    receipt_path = output_root / "neutral_external_dedup_receipt.json"
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({"ok": True, "documents": len(selected), "tokens": total_tokens, "receipt": str(receipt_path)}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    signatures = commands.add_parser("candidate-signatures")
    signatures.add_argument("--source-preparation-receipt", type=Path, required=True)
    signatures.add_argument("--training-config", type=Path, required=True)
    signatures.add_argument("--training-signature-manifest", type=Path, required=True)
    signatures.add_argument("--output-root", type=Path, required=True)
    signatures.set_defaults(func=candidate_signatures)

    match = commands.add_parser("match-bucket")
    match.add_argument("--candidate-signatures-receipt", type=Path, required=True)
    match.add_argument("--training-signature-manifest", type=Path, required=True)
    match.add_argument("--training-signature-root", type=Path, required=True)
    match.add_argument("--bucket", type=int, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.set_defaults(func=match_bucket)

    finish = commands.add_parser("finalize")
    finish.add_argument("--candidate-signatures-receipt", type=Path, required=True)
    finish.add_argument("--bucket-receipt", type=Path, action="append", required=True)
    finish.add_argument("--training-config", type=Path, required=True)
    finish.add_argument("--training-signature-manifest", type=Path, required=True)
    finish.add_argument("--training-combined-manifest", type=Path, required=True)
    finish.add_argument("--training-exact-root", type=Path, required=True)
    finish.add_argument("--training-exact-receipts", type=Path, required=True)
    finish.add_argument("--pool-corpus-receipt", type=Path, required=True)
    finish.add_argument("--tokenizer-dir", type=Path, required=True)
    finish.add_argument("--output-root", type=Path, required=True)
    finish.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    finish.add_argument("--hash-workers", type=int, default=16)
    finish.add_argument("--max-training-refs-per-candidate", type=int, default=10_000)
    finish.set_defaults(func=finalize)
    return root


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "target_tokens", TARGET_TOKENS) not in range(MIN_TOKENS, MAX_TOKENS + 1):
        raise ValueError("neutral target tokens must be in [10M,20M]")
    if getattr(args, "hash_workers", 1) < 1:
        raise ValueError("hash workers must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
