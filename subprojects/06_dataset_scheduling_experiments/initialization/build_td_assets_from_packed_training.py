#!/usr/bin/env python3
"""Build exact tied-TD coverage and contexts from frozen packed training rows.

The old Modern-Greek Token Distillation snippet index is no longer present on
CSCS.  Re-tokenizing a loosely reconstructed JSONL would weaken the experiment
contract, because it could include heldouts or documents that are not in the
five-arm schedule.  This builder instead reads a deterministic sample from the
already frozen, post-exclusion packed sequence inventory.

The first N rows of each per-pool sequence catalogue are already a frozen
SplitMix64 permutation.  They therefore define an immutable sample without a
new shuffle.  The complete Old-Greek packed pool is scanned by default so the
second-stage polytonic rows receive the strongest available coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


BASE_VOCAB_SIZE = 131_072
POLYTONIC_START = 148_480
TARGET_VOCAB_SIZE = 148_992
PACKED_WIDTH = 4_097
SEQUENCE_DTYPE = np.dtype(
    [
        ("sequence_id", "<u8"),
        ("packing_task_index", "<u4"),
        ("row_index", "<u4"),
        ("active_tokens", "<u2"),
    ],
    align=False,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def status_for(accepted: int) -> tuple[str, str]:
    if accepted >= 100:
        return "enough_100", "td_100"
    if accepted >= 25:
        return "enough_25", "td_25"
    if accepted >= 20:
        return "low_20_24", "keep_retok"
    if accepted > 0:
        return "low_lt20", "keep_retok"
    return "zero", "inspect"


def contains_subsequence(sequence: list[int], needle: list[int]) -> bool:
    if not needle or len(needle) > len(sequence):
        return False
    width = len(needle)
    target = tuple(needle)
    return any(
        tuple(sequence[index : index + width]) == target
        for index in range(len(sequence) - width + 1)
    )


def receipt_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def require_file_receipt(record: Mapping[str, Any], label: str, *, hash_file: bool) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} size or path drift: {path}")
    if hash_file and sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 drift: {path}")
    return path


def requested_sample_count(available: int, requested: int) -> int:
    if requested == -1:
        return available
    if requested <= 0:
        raise ValueError("sample counts must be positive or -1 for the complete pool")
    if requested > available:
        raise ValueError(f"requested {requested} sequences from a pool with {available}")
    return requested


def load_frozen_samples(
    packed_receipt: Mapping[str, Any], requested: Mapping[str, int]
) -> tuple[dict[int, list[tuple[str, np.void]]], dict[str, dict[str, int]]]:
    by_task: dict[int, list[tuple[str, np.void]]] = defaultdict(list)
    sample_summary: dict[str, dict[str, int]] = {}
    for pool, requested_count in requested.items():
        pool_record = packed_receipt["pools"][pool]
        catalogue_record = pool_record["sequence_catalog"]
        catalogue_path = require_file_receipt(
            catalogue_record, f"{pool} sequence catalogue", hash_file=True
        )
        available = int(catalogue_record["rows"])
        count = requested_sample_count(available, requested_count)
        catalogue = np.memmap(catalogue_path, mode="r", dtype=SEQUENCE_DTYPE, shape=(available,))
        selected = np.array(catalogue[:count], copy=True)
        active_tokens = int(selected["active_tokens"].astype(np.uint64).sum())
        for record in selected:
            by_task[int(record["packing_task_index"])].append((pool, record))
        sample_summary[pool] = {
            "available_sequences": available,
            "selected_sequences": count,
            "selected_active_tokens": active_tokens,
        }
    return by_task, sample_summary


def load_task_manifests(packed_receipt: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifests: dict[int, dict[str, Any]] = {}
    for record in packed_receipt["packing_task_manifests"]:
        task_index = int(record["task_index"])
        manifest_path = Path(record["manifest_path"])
        if sha256_file(manifest_path) != record["manifest_sha256"]:
            raise ValueError(f"packing task manifest SHA-256 drift: {manifest_path}")
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") != "apertus_mini_fixed_sequence_bucket_v1"
            or manifest.get("status") != "completed"
            or int(manifest.get("task_index", -1)) != task_index
        ):
            raise ValueError(f"invalid packed task manifest: {manifest_path}")
        manifests[task_index] = manifest
    return manifests


def task_arrays(manifest: Mapping[str, Any]) -> tuple[np.memmap, np.memmap]:
    bin_path = require_file_receipt(manifest["outputs"]["bin"], "packed binary", hash_file=False)
    active_path = require_file_receipt(
        manifest["outputs"]["active_counts"], "packed active counts", hash_file=True
    )
    sequence_count = int(manifest["sequence_count"])
    if bin_path.stat().st_size != sequence_count * PACKED_WIDTH * 4:
        raise ValueError(f"packed binary geometry drift: {bin_path}")
    binary = np.memmap(bin_path, mode="r", dtype=np.int32).reshape(sequence_count, PACKED_WIDTH)
    active = np.memmap(active_path, mode="r", dtype=np.uint16, shape=(sequence_count,))
    return binary, active


def iter_round_robin_batches(
    samples_by_task: Mapping[int, list[tuple[str, np.void]]],
    *,
    batch_size: int,
    seed: int,
) -> Iterable[tuple[int, list[tuple[str, np.void]]]]:
    task_ids = sorted(samples_by_task)
    random.Random(seed).shuffle(task_ids)
    offsets = {task_id: 0 for task_id in task_ids}
    while True:
        progressed = False
        for task_id in task_ids:
            start = offsets[task_id]
            rows = samples_by_task[task_id]
            if start >= len(rows):
                continue
            end = min(len(rows), start + batch_size)
            offsets[task_id] = end
            progressed = True
            yield task_id, rows[start:end]
        if not progressed:
            return


def collect_firings_and_candidates(
    samples_by_task: Mapping[int, list[tuple[str, np.void]]],
    manifests: Mapping[int, Mapping[str, Any]],
    *,
    candidate_cap: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    token_count = TARGET_VOCAB_SIZE - BASE_VOCAB_SIZE
    firings = np.zeros(token_count, dtype=np.uint64)
    candidate_count = np.zeros(token_count, dtype=np.uint16)
    needed = np.ones(token_count, dtype=np.bool_)
    candidates = {
        "task_index": np.full((token_count, candidate_cap), -1, dtype=np.int32),
        "row_index": np.zeros((token_count, candidate_cap), dtype=np.uint32),
        "token_index": np.zeros((token_count, candidate_cap), dtype=np.uint16),
        "sequence_id": np.zeros((token_count, candidate_cap), dtype=np.uint64),
        "pool_code": np.zeros((token_count, candidate_cap), dtype=np.uint8),
        "count": candidate_count,
    }
    pool_codes = {"hplt_new_greek": 0, "non_hplt_new_greek": 1, "old_greek_replay": 3}
    completed_batches = 0
    for task_id, records in iter_round_robin_batches(
        samples_by_task, batch_size=batch_size, seed=seed
    ):
        manifest = manifests.get(task_id)
        if manifest is None:
            raise ValueError(f"sample references unknown packing task {task_id}")
        binary, active = task_arrays(manifest)
        row_indexes = np.fromiter(
            (int(record[1]["row_index"]) for record in records), dtype=np.int64
        )
        expected_active = np.fromiter(
            (int(record[1]["active_tokens"]) for record in records), dtype=np.uint16
        )
        if np.any(active[row_indexes] != expected_active):
            raise ValueError(f"sequence catalogue/active-count drift in task {task_id}")
        rows = np.asarray(binary[row_indexes])
        payload = rows[:, 1:]
        target_values = payload[
            (payload >= BASE_VOCAB_SIZE) & (payload < TARGET_VOCAB_SIZE)
        ]
        if target_values.size:
            firings += np.bincount(
                target_values.astype(np.int64) - BASE_VOCAB_SIZE,
                minlength=token_count,
            ).astype(np.uint64, copy=False)

        if bool(needed.any()):
            for record_index, (_pool, record) in enumerate(records):
                width = int(record["active_tokens"])
                ids = payload[record_index, :width]
                target_positions = np.flatnonzero(
                    (ids >= BASE_VOCAB_SIZE) & (ids < TARGET_VOCAB_SIZE)
                )
                if not target_positions.size:
                    continue
                relative_ids = ids[target_positions].astype(np.int64) - BASE_VOCAB_SIZE
                still_needed = needed[relative_ids]
                if not bool(still_needed.any()):
                    continue
                positions = target_positions[still_needed]
                relative_ids = relative_ids[still_needed]
                unique_ids, first = np.unique(relative_ids, return_index=True)
                positions = positions[first]
                for relative_id, token_position in zip(unique_ids, positions, strict=True):
                    relative_id = int(relative_id)
                    slot = int(candidate_count[relative_id])
                    if slot >= candidate_cap:
                        continue
                    candidates["task_index"][relative_id, slot] = task_id
                    candidates["row_index"][relative_id, slot] = int(record["row_index"])
                    candidates["token_index"][relative_id, slot] = int(token_position)
                    candidates["sequence_id"][relative_id, slot] = int(record["sequence_id"])
                    candidates["pool_code"][relative_id, slot] = pool_codes[_pool]
                    candidate_count[relative_id] += 1
                    if int(candidate_count[relative_id]) == candidate_cap:
                        needed[relative_id] = False
        completed_batches += 1
        if completed_batches % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "packed_td_scan_progress",
                        "completed_batches": completed_batches,
                        "tokens_with_candidate_cap": int(np.count_nonzero(~needed)),
                        "tokens_with_any_firing": int(np.count_nonzero(firings)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return firings, candidates


def decode_and_write_snippets(
    candidates: Mapping[str, np.ndarray],
    manifests: Mapping[int, Mapping[str, Any]],
    target_tokenizer: Any,
    base_tokenizer: Any,
    exact_base_ids: Mapping[int, list[int]],
    output: Path,
    *,
    accepted_cap: int,
    radius: int,
    decode_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[int, list[str]]]:
    token_count = TARGET_VOCAB_SIZE - BASE_VOCAB_SIZE
    accepted = np.zeros(token_count, dtype=np.uint16)
    distinct_sequences = np.zeros(token_count, dtype=np.uint16)
    example_refs: dict[int, list[str]] = defaultdict(list)
    rejection_counts = {
        "empty_decoded_text": 0,
        "non_nfc_decoded_text": 0,
        "exact_base_phrase_absent": 0,
        "candidate_skipped_after_token_cap": 0,
    }
    by_task: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for relative_id in range(token_count):
        for slot in range(int(candidates["count"][relative_id])):
            by_task[int(candidates["task_index"][relative_id, slot])].append(
                (relative_id, slot)
            )

    pool_names = {0: "hplt_new_greek", 1: "non_hplt_new_greek", 3: "old_greek_replay"}
    temporary = Path(str(output) + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for task_id in sorted(by_task):
            binary, active = task_arrays(manifests[task_id])
            entries = by_task[task_id]
            for start in range(0, len(entries), decode_batch_size):
                batch = entries[start : start + decode_batch_size]
                windows: list[list[int]] = []
                metadata: list[tuple[int, int, int, int, int]] = []
                for relative_id, slot in batch:
                    if int(accepted[relative_id]) >= accepted_cap:
                        rejection_counts["candidate_skipped_after_token_cap"] += 1
                        continue
                    row_index = int(candidates["row_index"][relative_id, slot])
                    token_index = int(candidates["token_index"][relative_id, slot])
                    width = int(active[row_index])
                    left = max(0, token_index - radius)
                    right = min(width, token_index + radius + 1)
                    ids = binary[row_index, 1 + left : 1 + right].astype(np.int64).tolist()
                    windows.append(ids)
                    metadata.append((relative_id, slot, row_index, token_index, width))
                if not windows:
                    continue
                texts = target_tokenizer.decode_batch(windows, skip_special_tokens=False)
                base_encodings = base_tokenizer.encode_batch(texts, add_special_tokens=False)
                for text, encoding, (relative_id, slot, row_index, token_index, _width) in zip(
                    texts, base_encodings, metadata, strict=True
                ):
                    if int(accepted[relative_id]) >= accepted_cap:
                        rejection_counts["candidate_skipped_after_token_cap"] += 1
                        continue
                    if not text:
                        rejection_counts["empty_decoded_text"] += 1
                        continue
                    if text != unicodedata.normalize("NFC", text):
                        rejection_counts["non_nfc_decoded_text"] += 1
                        continue
                    token_id = BASE_VOCAB_SIZE + relative_id
                    phrase = exact_base_ids[token_id]
                    if not contains_subsequence(encoding.ids, phrase):
                        rejection_counts["exact_base_phrase_absent"] += 1
                        continue
                    snippet_index = int(accepted[relative_id])
                    snippet_id = f"{token_id}:{snippet_index:04d}"
                    pool = pool_names[int(candidates["pool_code"][relative_id, slot])]
                    row = {
                        "snippet_id": snippet_id,
                        "new_token_id": token_id,
                        "doc_ref": (
                            f"packed:{pool}:task={task_id}:row={row_index}:"
                            f"sequence={int(candidates['sequence_id'][relative_id, slot])}"
                        ),
                        "doc_id": f"packed-sequence:{int(candidates['sequence_id'][relative_id, slot])}",
                        "source": pool,
                        "lang": "grc" if pool == "old_greek_replay" else "el",
                        "token_index": token_index,
                        "surface": target_tokenizer.decode([token_id], skip_special_tokens=False),
                        "span_base_subtoken_ids": phrase,
                        "snippet_text": text,
                        "provenance_policy": "exact_frozen_post_exclusion_packed_training_sequence",
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    accepted[relative_id] += 1
                    distinct_sequences[relative_id] += 1
                    if len(example_refs[token_id]) < 5:
                        example_refs[token_id].append(snippet_id)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return accepted, distinct_sequences, rejection_counts, example_refs


def coverage_fraction(accepted: np.ndarray, start: int, end: int) -> tuple[int, float]:
    relative_start = start - BASE_VOCAB_SIZE
    relative_end = end - BASE_VOCAB_SIZE
    values = accepted[relative_start:relative_end]
    enough = int(np.count_nonzero(values >= 25))
    return enough, enough / int(values.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packing-plan", type=Path, required=True)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--hplt-eval-jsonl", type=Path, required=True)
    parser.add_argument("--non-hplt-eval-jsonl", type=Path, required=True)
    parser.add_argument("--polytonic-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hplt-sequences", type=int, default=350_000)
    parser.add_argument("--non-hplt-sequences", type=int, default=150_000)
    parser.add_argument("--old-greek-sequences", type=int, default=-1)
    parser.add_argument("--candidate-snippets-per-token", type=int, default=64)
    parser.add_argument("--accepted-snippets-per-token", type=int, default=25)
    parser.add_argument("--snippet-token-radius", type=int, default=50)
    parser.add_argument("--scan-batch-size", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=2_026_080_2)
    parser.add_argument("--min-overall-enough25-fraction", type=float, default=0.90)
    parser.add_argument("--min-modern-enough25-fraction", type=float, default=0.90)
    parser.add_argument("--min-polytonic-enough25-count", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    from tokenizers import Tokenizer

    from tokenizer_geometry import derive_added_token_base_ids

    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.accepted_snippets_per_token != 25:
        raise ValueError("this campaign contract requires exactly 25 accepted snippets per token")
    if args.candidate_snippets_per_token < args.accepted_snippets_per_token:
        raise ValueError("candidate snippet cap is below accepted snippet cap")

    packing_plan = read_json(args.packing_plan)
    packed_receipt = read_json(args.packed_receipt)
    input_receipt = read_json(args.input_receipt)
    if packing_plan.get("schema_version") != "apertus_mini_fixed_sequence_packing_plan_v1":
        raise ValueError("unsupported packing plan")
    if (
        packed_receipt.get("schema_version") != "apertus_mini_packed_sequence_corpus_v1"
        or packed_receipt.get("status") != "completed"
    ):
        raise ValueError("packed corpus receipt is not completed")
    plan_sha = sha256_file(args.packing_plan)
    if packed_receipt["packing_plan"] != {
        "path": str(args.packing_plan.resolve()),
        "sha256": plan_sha,
    }:
        raise ValueError("packed receipt does not bind the requested packing plan")
    target_tokenizer_json = args.target_tokenizer / "tokenizer.json"
    base_tokenizer_json = args.base_tokenizer / "tokenizer.json"
    if sha256_file(target_tokenizer_json) != input_receipt["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("target tokenizer differs from the packed-corpus tokenizer")

    requested = {
        "hplt_new_greek": args.hplt_sequences,
        "non_hplt_new_greek": args.non_hplt_sequences,
        "old_greek_replay": args.old_greek_sequences,
    }
    samples_by_task, sample_summary = load_frozen_samples(packed_receipt, requested)
    manifests = load_task_manifests(packed_receipt)
    firings, candidates = collect_firings_and_candidates(
        samples_by_task,
        manifests,
        candidate_cap=args.candidate_snippets_per_token,
        batch_size=args.scan_batch_size,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True)
    snippet_dir = args.output_dir / "td_snippet_index"
    snippet_dir.mkdir()
    snippets_path = snippet_dir / "snippets.jsonl"
    target_tokenizer = Tokenizer.from_file(str(target_tokenizer_json))
    base_tokenizer = Tokenizer.from_file(str(base_tokenizer_json))
    if target_tokenizer.get_vocab_size(with_added_tokens=True) != TARGET_VOCAB_SIZE:
        raise ValueError("target tokenizer vocabulary size drift")
    exact_base_ids = derive_added_token_base_ids(
        target_tokenizer_json,
        base_vocab_size=BASE_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
    )
    accepted, distinct_sequences, rejection_counts, example_refs = decode_and_write_snippets(
        candidates,
        manifests,
        target_tokenizer,
        base_tokenizer,
        exact_base_ids,
        snippets_path,
        accepted_cap=args.accepted_snippets_per_token,
        radius=args.snippet_token_radius,
        decode_batch_size=args.decode_batch_size,
    )

    coverage_path = args.output_dir / "td_coverage_prepass.jsonl"
    coverage_temporary = Path(str(coverage_path) + ".partial")
    status_counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    with coverage_temporary.open("w", encoding="utf-8") as handle:
        for token_id in range(BASE_VOCAB_SIZE, TARGET_VOCAB_SIZE):
            relative_id = token_id - BASE_VOCAB_SIZE
            usable = int(accepted[relative_id])
            status, action = status_for(usable)
            status_counts[status] += 1
            action_counts[action] += 1
            row = {
                "new_token_id": token_id,
                "raw_token": target_tokenizer.id_to_token(token_id),
                "token_string": target_tokenizer.decode([token_id], skip_special_tokens=False),
                "base_subtoken_ids": exact_base_ids[token_id],
                "base_subtoken_len": len(exact_base_ids[token_id]),
                "base_decomposition_policy": "exact_dependency_ordered_appended_merge_dag_leaves",
                "extended_firings": int(firings[relative_id]),
                "raw_surface_occurrences": None,
                "usable_snippets_25": min(usable, 25),
                "usable_snippets_100": min(usable, 100),
                "docs_with_firing": int(distinct_sequences[relative_id]),
                "docs_with_firing_policy": "distinct_packed_training_sequences_with_accepted_context_lower_bound",
                "example_snippet_refs": example_refs.get(token_id, []),
                "status": status,
                "recommended_action": action,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(coverage_temporary, coverage_path)

    modern_enough, modern_fraction = coverage_fraction(
        accepted, BASE_VOCAB_SIZE, POLYTONIC_START
    )
    poly_enough, poly_fraction = coverage_fraction(
        accepted, POLYTONIC_START, TARGET_VOCAB_SIZE
    )
    overall_enough = int(np.count_nonzero(accepted >= 25))
    overall_fraction = overall_enough / int(accepted.size)
    gates = {
        "overall_enough25_fraction": {
            "observed": overall_fraction,
            "required_minimum": args.min_overall_enough25_fraction,
            "passed": overall_fraction >= args.min_overall_enough25_fraction,
        },
        "modern_enough25_fraction": {
            "observed": modern_fraction,
            "required_minimum": args.min_modern_enough25_fraction,
            "passed": modern_fraction >= args.min_modern_enough25_fraction,
        },
        "polytonic_enough25_count": {
            "observed": poly_enough,
            "required_minimum": args.min_polytonic_enough25_count,
            "passed": poly_enough >= args.min_polytonic_enough25_count,
        },
    }
    passed = all(bool(record["passed"]) for record in gates.values())
    evaluation_inputs = {
        "hplt": receipt_file(args.hplt_eval_jsonl),
        "non_hplt": receipt_file(args.non_hplt_eval_jsonl),
        "polytonic": receipt_file(args.polytonic_eval_jsonl),
    }
    summary = {
        "schema_version": "apertus_mini_packed_training_td_assets_v1",
        "status": "passed" if passed else "failed_coverage_gate",
        "provenance_policy": "contexts_are_drawn_only_from_frozen_post_exclusion_packed_training_sequences",
        "heldouts_used_for_training_contexts": False,
        "packing_plan": receipt_file(args.packing_plan),
        "packed_receipt": receipt_file(args.packed_receipt),
        "input_receipt": receipt_file(args.input_receipt),
        "base_tokenizer_json": receipt_file(base_tokenizer_json),
        "target_tokenizer_json": receipt_file(target_tokenizer_json),
        "frozen_sequence_sample": sample_summary,
        "sample_total_sequences": sum(v["selected_sequences"] for v in sample_summary.values()),
        "sample_total_active_tokens": sum(v["selected_active_tokens"] for v in sample_summary.values()),
        "candidate_snippets_per_token": args.candidate_snippets_per_token,
        "accepted_snippets_per_token": args.accepted_snippets_per_token,
        "snippet_token_radius": args.snippet_token_radius,
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "coverage": {
            "overall": {
                "token_types": int(accepted.size),
                "enough_25": overall_enough,
                "enough_25_fraction": overall_fraction,
            },
            "modern": {
                "id_range": [BASE_VOCAB_SIZE, POLYTONIC_START],
                "token_types": POLYTONIC_START - BASE_VOCAB_SIZE,
                "enough_25": modern_enough,
                "enough_25_fraction": modern_fraction,
            },
            "polytonic": {
                "id_range": [POLYTONIC_START, TARGET_VOCAB_SIZE],
                "token_types": TARGET_VOCAB_SIZE - POLYTONIC_START,
                "enough_25": poly_enough,
                "enough_25_fraction": poly_fraction,
            },
        },
        "gates": gates,
        "rejection_counts": rejection_counts,
        "evaluation_inputs": evaluation_inputs,
        "artifacts": {
            "coverage_jsonl": receipt_file(coverage_path),
            "snippet_index_jsonl": receipt_file(snippets_path),
        },
        "counting_rule": "A firing means the exact packed training row contains the emitted added-token ID.",
        "candidate_selection": (
            "first N rows of each frozen SplitMix64-ordered pool catalogue, then "
            "deterministic task-round-robin context collection"
        ),
    }
    summary_path = args.output_dir / "td_coverage_summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
