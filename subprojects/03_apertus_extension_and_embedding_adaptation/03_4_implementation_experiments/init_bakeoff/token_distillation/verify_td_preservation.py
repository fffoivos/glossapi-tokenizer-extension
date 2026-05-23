#!/usr/bin/env python3
"""Verify Token Distillation changed only the intended embedding rows.

This is the post-TD preservation gate for the Apertus adapter. It compares a
TD checkpoint against its ReTok reference checkpoint and enforces:

* state-dict keys and shapes are identical;
* every non-embedding tensor is unchanged within tolerance;
* input/output embedding rows outside the trained-token set are unchanged;
* trained new input/output rows are finite and moved.

The upstream TD loop already checks original rows before saving. This script is
the persisted artifact-level check we can run independently after the job.

The implementation intentionally parses Safetensors files directly instead of
importing torch. That keeps the gate runnable on Clariden `xfer`, where CPU-only
jobs do not have the PyTorch uenv.
"""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple


EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")
XIELU_PATTERN = re.compile(r"act_fn\.(alpha_p|alpha_n|beta|eps)")
QK_PATTERN = re.compile(r"self_attn\.(q_norm|k_norm)\.weight")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
CHUNK_BYTES = 64 * 1024 * 1024


class TensorMeta(NamedTuple):
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]


class SafeTensorStore:
    def __init__(self, hf_dir: Path, weight_map: Dict[str, str]):
        self.hf_dir = hf_dir
        self.weight_map = weight_map
        self._headers: Dict[str, Tuple[int, Dict[str, TensorMeta]]] = {}

    def _load_header(self, shard_name: str) -> Tuple[int, Dict[str, TensorMeta]]:
        if shard_name in self._headers:
            return self._headers[shard_name]
        path = self.hf_dir / shard_name
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        tensors: Dict[str, TensorMeta] = {}
        for key, value in header.items():
            if key == "__metadata__":
                continue
            tensors[key] = TensorMeta(
                dtype=value["dtype"],
                shape=tuple(int(x) for x in value["shape"]),
                data_offsets=(int(value["data_offsets"][0]), int(value["data_offsets"][1])),
            )
        data_start = 8 + header_len
        self._headers[shard_name] = (data_start, tensors)
        return self._headers[shard_name]

    def meta(self, key: str) -> TensorMeta:
        shard_name = self.weight_map[key]
        _, tensors = self._load_header(shard_name)
        return tensors[key]

    def read_bytes(self, key: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        shard_name = self.weight_map[key]
        data_start, tensors = self._load_header(shard_name)
        meta = tensors[key]
        tensor_len = meta.data_offsets[1] - meta.data_offsets[0]
        if length is None:
            length = tensor_len - offset
        if offset < 0 or length < 0 or offset + length > tensor_len:
            raise ValueError("bad tensor read for %s: offset=%d length=%d tensor_len=%d" % (key, offset, length, tensor_len))
        with (self.hf_dir / shard_name).open("rb") as fh:
            fh.seek(data_start + meta.data_offsets[0] + offset)
            return fh.read(length)

    def iter_bytes(self, key: str, offset: int = 0, length: Optional[int] = None, chunk_bytes: int = CHUNK_BYTES):
        shard_name = self.weight_map[key]
        data_start, tensors = self._load_header(shard_name)
        meta = tensors[key]
        tensor_len = meta.data_offsets[1] - meta.data_offsets[0]
        if length is None:
            length = tensor_len - offset
        if offset < 0 or length < 0 or offset + length > tensor_len:
            raise ValueError("bad tensor read for %s: offset=%d length=%d tensor_len=%d" % (key, offset, length, tensor_len))
        remaining = length
        with (self.hf_dir / shard_name).open("rb") as fh:
            fh.seek(data_start + meta.data_offsets[0] + offset)
            while remaining:
                chunk = fh.read(min(chunk_bytes, remaining))
                if not chunk:
                    raise EOFError("unexpected EOF while reading %s" % key)
                remaining -= len(chunk)
                yield chunk


def read_weight_map(hf_dir: Path) -> Dict[str, str]:
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = hf_dir / "model.safetensors"
    if single.exists():
        store = SafeTensorStore(hf_dir, {})
        _, tensors = store._load_header(single.name)
        return {key: single.name for key in tensors}
    raise FileNotFoundError("found neither model.safetensors.index.json nor model.safetensors in %s" % hf_dir)


def tensor_nbytes(meta: TensorMeta) -> int:
    itemsize = DTYPE_BYTES.get(meta.dtype)
    if itemsize is None:
        raise ValueError("unsupported dtype: %s" % meta.dtype)
    count = 1
    for dim in meta.shape:
        count *= dim
    return count * itemsize


def sha256_tensor(store: SafeTensorStore, key: str) -> str:
    h = hashlib.sha256()
    for chunk in store.iter_bytes(key):
        h.update(chunk)
    return h.hexdigest()


def byte_ranges_equal(ref: SafeTensorStore, td: SafeTensorStore, key: str, offset: int, length: int) -> bool:
    ref_iter = ref.iter_bytes(key, offset=offset, length=length)
    td_iter = td.iter_bytes(key, offset=offset, length=length)
    for ref_chunk, td_chunk in zip(ref_iter, td_iter):
        if ref_chunk != td_chunk:
            return False
    return True


def load_trained_ids(manifest: Path) -> Set[int]:
    data = json.loads(manifest.read_text())
    ids = data.get("trained_token_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("manifest has no non-empty trained_token_ids: %s" % manifest)
    out = set()
    for value in ids:
        if not isinstance(value, int):
            raise ValueError("non-integer trained token id in %s: %r" % (manifest, value))
        out.add(value)
    return out


def sample_ids(ids: Iterable[int], limit: int) -> List[int]:
    out = sorted(set(ids))
    if limit <= 0 or len(out) <= limit:
        return out
    head = out[: limit // 2]
    tail = out[-(limit - len(head)) :]
    return head + tail


def complement_ranges(vocab: int, excluded_ids: Set[int]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for token_id in sorted(excluded_ids):
        if token_id < 0 or token_id >= vocab:
            raise ValueError("token ID outside vocab: %d >= %d" % (token_id, vocab))
        if start < token_id:
            ranges.append((start, token_id))
        start = token_id + 1
    if start < vocab:
        ranges.append((start, vocab))
    return ranges


def row_has_nonfinite(raw: bytes, dtype: str) -> bool:
    if dtype == "BF16":
        for (value,) in struct.iter_unpack("<H", raw):
            if value & 0x7F80 == 0x7F80:
                return True
        return False
    if dtype == "F16":
        for (value,) in struct.iter_unpack("<H", raw):
            if value & 0x7C00 == 0x7C00:
                return True
        return False
    if dtype == "F32":
        for (value,) in struct.iter_unpack("<I", raw):
            if value & 0x7F800000 == 0x7F800000:
                return True
        return False
    return False


def verify_embedding_rows(
    ref: SafeTensorStore,
    td: SafeTensorStore,
    key: str,
    trained_ids: Set[int],
) -> Dict[str, object]:
    meta = ref.meta(key)
    if len(meta.shape) != 2:
        raise AssertionError("expected 2D embedding tensor for %s, got %s" % (key, meta.shape))
    itemsize = DTYPE_BYTES[meta.dtype]
    vocab, width = meta.shape
    row_bytes = width * itemsize
    preserved_ranges = complement_ranges(vocab, trained_ids)
    preserved_changed_ranges = []
    preserved_bytes_compared = 0

    for start, end in preserved_ranges:
        offset = start * row_bytes
        length = (end - start) * row_bytes
        preserved_bytes_compared += length
        if not byte_ranges_equal(ref, td, key, offset, length):
            preserved_changed_ranges.append([start, end])

    changed_rows = 0
    unchanged_trained_ids: List[int] = []
    nonfinite_trained_ids: List[int] = []
    zero_trained_ids: List[int] = []
    for token_id in sorted(trained_ids):
        offset = token_id * row_bytes
        ref_row = ref.read_bytes(key, offset=offset, length=row_bytes)
        td_row = td.read_bytes(key, offset=offset, length=row_bytes)
        if ref_row != td_row:
            changed_rows += 1
        else:
            unchanged_trained_ids.append(token_id)
        if row_has_nonfinite(td_row, meta.dtype):
            nonfinite_trained_ids.append(token_id)
        if not any(td_row):
            zero_trained_ids.append(token_id)

    return {
        "dtype": meta.dtype,
        "shape": list(meta.shape),
        "row_bytes": row_bytes,
        "preserved_range_count": len(preserved_ranges),
        "preserved_bytes_compared": preserved_bytes_compared,
        "preserved_changed_ranges": preserved_changed_ranges[:20],
        "preserved_changed_range_count": len(preserved_changed_ranges),
        "trained_rows_changed": changed_rows,
        "trained_rows_unchanged": len(unchanged_trained_ids),
        "trained_rows_nonfinite": len(nonfinite_trained_ids),
        "trained_rows_zero": len(zero_trained_ids),
        "sample_unchanged_trained_ids": unchanged_trained_ids[:20],
        "sample_nonfinite_trained_ids": nonfinite_trained_ids[:20],
        "sample_zero_trained_ids": zero_trained_ids[:20],
        "sample_trained_ids": sample_ids(trained_ids, 20),
    }


def verify(args: argparse.Namespace) -> Dict[str, object]:
    ref_map = read_weight_map(args.reference_hf_dir)
    td_map = read_weight_map(args.td_hf_dir)
    ref_store = SafeTensorStore(args.reference_hf_dir, ref_map)
    td_store = SafeTensorStore(args.td_hf_dir, td_map)
    ref_keys = set(ref_map)
    td_keys = set(td_map)
    orig_only = sorted(ref_keys - td_keys)
    td_only = sorted(td_keys - ref_keys)
    if orig_only or td_only:
        raise AssertionError("state_dict keys differ: orig_only=%s td_only=%s" % (orig_only[:5], td_only[:5]))

    trained_ids = load_trained_ids(args.manifest)

    summary = {
        "reference_hf_dir": str(args.reference_hf_dir),
        "td_hf_dir": str(args.td_hf_dir),
        "manifest": str(args.manifest),
        "trained_token_count": len(trained_ids),
        "mode": "safetensors_raw_exact",
        "non_embedding_changed": [],
        "non_embedding_changed_count": 0,
        "xielu_changed": [],
        "qk_norm_changed": [],
        "embedding_rows": {},
        "shape_mismatches": [],
        "dtype_mismatches": [],
    }

    changed_non_embedding = []
    shape_mismatches = []
    dtype_mismatches = []

    for key in sorted(ref_keys):
        ref_meta = ref_store.meta(key)
        td_meta = td_store.meta(key)
        if ref_meta.shape != td_meta.shape:
            shape_mismatches.append("%s: %s vs %s" % (key, ref_meta.shape, td_meta.shape))
            continue
        if ref_meta.dtype != td_meta.dtype:
            dtype_mismatches.append("%s: %s vs %s" % (key, ref_meta.dtype, td_meta.dtype))
            continue
        if tensor_nbytes(ref_meta) != ref_meta.data_offsets[1] - ref_meta.data_offsets[0]:
            raise AssertionError("bad byte length in reference tensor %s" % key)
        if tensor_nbytes(td_meta) != td_meta.data_offsets[1] - td_meta.data_offsets[0]:
            raise AssertionError("bad byte length in TD tensor %s" % key)

        if key not in EMBED_KEYS:
            ref_hash = sha256_tensor(ref_store, key)
            td_hash = sha256_tensor(td_store, key)
            if ref_hash != td_hash:
                changed_non_embedding.append([key, ref_hash, td_hash])
            continue

        row_summary = verify_embedding_rows(ref_store, td_store, key, trained_ids)
        summary["embedding_rows"][key] = row_summary
        if row_summary["preserved_changed_range_count"]:
            changed_non_embedding.append([key + "[preserved_rows]", row_summary["preserved_changed_ranges"]])
        if row_summary["trained_rows_nonfinite"]:
            raise AssertionError("non-finite trained rows in %s: %s" % (key, row_summary["sample_nonfinite_trained_ids"]))
        if row_summary["trained_rows_zero"]:
            raise AssertionError("zero trained rows in %s: %s" % (key, row_summary["sample_zero_trained_ids"]))
        if row_summary["trained_rows_unchanged"]:
            raise AssertionError("unchanged trained rows in %s: %s" % (key, row_summary["sample_unchanged_trained_ids"]))

    summary["non_embedding_changed"] = changed_non_embedding[:50]
    summary["non_embedding_changed_count"] = len(changed_non_embedding)
    summary["xielu_changed"] = [row for row in changed_non_embedding if XIELU_PATTERN.search(row[0])]
    summary["qk_norm_changed"] = [row for row in changed_non_embedding if QK_PATTERN.search(row[0])]
    summary["shape_mismatches"] = shape_mismatches
    summary["dtype_mismatches"] = dtype_mismatches

    if shape_mismatches:
        raise AssertionError("shape mismatches: %s" % shape_mismatches[:5])
    if dtype_mismatches:
        raise AssertionError("dtype mismatches: %s" % dtype_mismatches[:5])
    if changed_non_embedding:
        raise AssertionError("preservation violations: %s" % changed_non_embedding[:5])
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-hf-dir", type=Path, required=True)
    ap.add_argument("--td-hf-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary = verify(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote: %s" % args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
