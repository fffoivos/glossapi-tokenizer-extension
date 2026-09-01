#!/usr/bin/env python3
"""Build a resume-safe hybrid schedule from the parent prefix and unseen suffix."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from contract_utils import file_binding, read_json, require, sha256_file, write_json_atomic


FILLER_ID = np.uint64(2**64 - 1)
POOL_SHIFT = 62
POOL_CODES = {"H": 0, "G": 1, "F": 2, "O": 3}


def smooth_categories(counts: dict[str, int], priority: tuple[str, ...]) -> np.ndarray:
    total = sum(counts.values())
    result = np.empty(total, dtype="S1")
    remaining = np.arange(total, dtype=np.int64)
    ordered = sorted(counts, key=lambda key: (counts[key], -priority.index(key)))
    for key in ordered[:-1]:
        count = counts[key]
        if count == 0:
            continue
        local = np.floor((np.arange(count, dtype=np.float64) + 0.5) * remaining.size / count).astype(np.int64)
        result[remaining[local]] = key.encode("ascii")
        keep = np.ones(remaining.size, dtype=bool)
        keep[local] = False
        remaining = remaining[keep]
    result[remaining] = ordered[-1].encode("ascii")
    return result


def nearest_prefix(active: np.ndarray, target: int) -> int:
    if target < 0:
        raise ValueError("negative replay target")
    if target == 0:
        return 0
    cumulative = np.cumsum(active, dtype=np.uint64)
    if cumulative.size == 0 or int(cumulative[-1]) < target:
        raise ValueError(f"remaining replay capacity is below target {target}")
    index = int(np.searchsorted(cumulative, target, side="left"))
    candidates = [index + 1]
    if index > 0:
        candidates.append(index)
    return min(candidates, key=lambda count: (abs((int(cumulative[count - 1]) if count else 0) - target), count))


def windows(total: int, count: int) -> list[tuple[int, int]]:
    base, extra = divmod(total, count)
    sizes = [base + int(i < extra) for i in range(count)]
    offsets = np.cumsum([0] + sizes)
    return [(int(offsets[i]), int(offsets[i + 1])) for i in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-schedule-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--parent-arm", default="D0_mixed")
    parser.add_argument("--checkpoint-iteration", type=int, default=9536)
    parser.add_argument("--global-batch-sequences", type=int, default=1024)
    parser.add_argument("--expected-modern-active-tokens", type=int, default=9123187023)
    parser.add_argument("--window-count", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite schedule directory: {args.output_dir}")
    parent = read_json(args.parent_schedule_manifest)
    require(parent.get("schema_version") == "apertus_data_order_schedules_v1" and parent.get("status") == "completed", "parent schedule receipt drift")
    require(parent.get("common_contract", {}).get("same_exact_sequence_multiset") is True, "parent multiset is not exact")
    checkpoint_receipt = read_json(args.parent_checkpoint_receipt)
    require(
        checkpoint_receipt.get("schema_version") == "megatron_exact_checkpoint_view_v1"
        and int(checkpoint_receipt.get("iteration", -1)) == args.checkpoint_iteration,
        "parent checkpoint receipt drift",
    )
    checkpoint_dir = args.parent_checkpoint_dir.resolve()
    expected_checkpoint_dir = (
        Path(checkpoint_receipt.get("source_checkpoint_root", "")).resolve()
        / f"iter_{args.checkpoint_iteration:07d}"
    )
    require(checkpoint_dir == expected_checkpoint_dir, "parent checkpoint directory/receipt drift")
    checkpoint_rows = checkpoint_receipt.get("source_files", [])
    require(checkpoint_rows and checkpoint_rows[0].get("relative_path") == ".metadata", "parent checkpoint inventory drift")
    metadata = checkpoint_dir / ".metadata"
    metadata_row = checkpoint_rows[0]
    require(
        metadata.is_file()
        and metadata.stat().st_size == int(metadata_row.get("bytes", -1))
        and sha256_file(metadata) == metadata_row.get("sha256"),
        "parent checkpoint metadata drift",
    )
    arms = {row["arm_id"]: row for row in parent["arms"]}
    require(args.parent_arm in arms, "parent arm absent")
    arm = arms[args.parent_arm]
    ids_info, active_info = arm["sequence_ids"], arm["active_tokens"]
    ids_path, active_path = Path(ids_info["path"]), Path(active_info["path"])
    require(file_binding(ids_path)["sha256"] == ids_info["sha256"], "parent sequence IDs hash drift")
    require(file_binding(active_path)["sha256"] == active_info["sha256"], "parent active-token hash drift")
    count = int(arm["training_slots"])
    ids = np.memmap(ids_path, mode="r", dtype="<u8", shape=(count,))
    active = np.memmap(active_path, mode="r", dtype="<u2", shape=(count,))
    prefix = args.checkpoint_iteration * args.global_batch_sequences
    require(0 < prefix < count and prefix % args.global_batch_sequences == 0, "invalid parent prefix boundary")
    packed_info = parent["packed_corpus_receipt"]
    packed_path = Path(packed_info["path"])
    require(sha256_file(packed_path) == packed_info["sha256"], "parent packed receipt hash drift")
    packed = read_json(packed_path)
    require(packed.get("global", {}).get("duplicate_sequence_ids") == 0, "parent packed inventory has duplicate IDs")

    suffix_ids = ids[prefix:]
    suffix_active = active[prefix:]
    suffix_pool = suffix_ids >> np.uint64(POOL_SHIFT)
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, code in {key: POOL_CODES[key] for key in ("G", "F", "O")}.items():
        mask = (suffix_pool == code) & (suffix_ids != FILLER_ID) & (suffix_active > 0)
        streams[key] = (np.asarray(suffix_ids[mask]), np.asarray(suffix_active[mask]))
    modern = int(streams["G"][1].astype(np.uint64).sum())
    require(modern == args.expected_modern_active_tokens, f"unseen non-HPLT drift: {modern}")
    foreign_target, old_target = round(modern * 20 / 79), round(modern / 79)
    selected: dict[str, tuple[np.ndarray, np.ndarray]] = {"G": streams["G"]}
    targets = {"G": modern, "F": foreign_target, "O": old_target}
    for key in ("F", "O"):
        take = nearest_prefix(streams[key][1], targets[key])
        selected[key] = (streams[key][0][:take], streams[key][1][:take])
    realized = {key: int(value[1].astype(np.uint64).sum()) for key, value in selected.items()}
    counts = {key: int(value[0].size) for key, value in selected.items()}
    categories = smooth_categories(counts, ("G", "F", "O"))
    continuation_ids = np.empty(categories.size, dtype="<u8")
    continuation_active = np.empty(categories.size, dtype="<u2")
    for key in ("G", "F", "O"):
        positions = np.flatnonzero(categories == key.encode("ascii"))
        continuation_ids[positions] = selected[key][0]
        continuation_active[positions] = selected[key][1]
    continuation_real_sequences = int(continuation_ids.size)
    pad = (-continuation_real_sequences) % args.global_batch_sequences
    if pad:
        continuation_ids = np.concatenate((continuation_ids, np.full(pad, FILLER_ID, dtype="<u8")))
        continuation_active = np.concatenate((continuation_active, np.zeros(pad, dtype="<u2")))
    require(
        np.unique(continuation_ids[:continuation_real_sequences]).size == continuation_real_sequences,
        "selected continuation IDs are not unique",
    )
    parent_prefix_real = np.asarray(ids[:prefix][ids[:prefix] != FILLER_ID])
    overlap = int(np.intersect1d(parent_prefix_real, continuation_ids[:continuation_real_sequences], assume_unique=True).size)
    require(overlap == 0, f"continuation reuses {overlap} parent-prefix sequences")
    schedule_ids = np.concatenate((np.asarray(ids[:prefix]), continuation_ids))
    schedule_active = np.concatenate((np.asarray(active[:prefix]), continuation_active))
    require(np.array_equal(schedule_ids[:prefix], ids[:prefix]), "parent sequence-ID prefix changed")
    require(np.array_equal(schedule_active[:prefix], active[:prefix]), "parent active-token prefix changed")
    window_rows = []
    for index, (start, end) in enumerate(windows(continuation_real_sequences, args.window_count)):
        cats = categories[start:end]
        window_rows.append({
            "window": index,
            "continuation_slot_start": start,
            "continuation_slot_end": end,
            "absolute_slot_start": prefix + start,
            "absolute_slot_end": prefix + end,
            "sequence_counts": {key: int(np.count_nonzero(cats == key.encode("ascii"))) for key in ("G", "F", "O")},
            "active_tokens": {key: int(continuation_active[start:end][cats == key.encode("ascii")].astype(np.uint64).sum()) for key in ("G", "F", "O")},
        })
    continuation_active_total = sum(realized.values())
    full_pool_active: dict[str, int] = {}
    full_pool_counts: dict[str, int] = {}
    real_mask = schedule_ids != FILLER_ID
    full_pool_codes = schedule_ids >> np.uint64(POOL_SHIFT)
    for key, code in POOL_CODES.items():
        mask = real_mask & (full_pool_codes == code)
        full_pool_active[key] = int(schedule_active[mask].astype(np.uint64).sum())
        full_pool_counts[key] = int(np.count_nonzero(mask))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", suffix=".partial", dir=args.output_dir.parent)
    )
    ids_output = temporary_dir / "D0_mixed.sequence_ids.u64"
    active_output = temporary_dir / "D0_mixed.active_tokens.u16"
    final_ids_output = args.output_dir / ids_output.name
    final_active_output = args.output_dir / active_output.name
    try:
        schedule_ids.tofile(ids_output)
        schedule_active.tofile(active_output)
        ids_binding = file_binding(ids_output)
        active_binding = file_binding(active_output)
        ids_binding["path"] = str(final_ids_output.resolve())
        active_binding["path"] = str(final_active_output.resolve())
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    manifest = {
        "schema_version": "apertus_data_order_schedules_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "packed_corpus_receipt": {"path": str(packed_path.resolve()), "sha256": sha256_file(packed_path)},
        "common_contract": {
            "arms": ["D0_mixed"],
            "window_count": args.window_count,
            "global_batch_sequences": args.global_batch_sequences,
            "same_exact_sequence_multiset": True,
            "same_replay_sequence_ids_at_same_global_positions": True,
            "same_per_sequence_active_token_count": True,
            "single_arm_vacuous_cross_arm_properties": True,
        },
        "continuation_contract": {
            "parent_schedule_manifest": file_binding(args.parent_schedule_manifest),
            "parent_checkpoint_receipt": file_binding(args.parent_checkpoint_receipt),
            "parent_checkpoint_directory": str(checkpoint_dir),
            "parent_checkpoint_metadata": file_binding(metadata),
            "parent_arm": args.parent_arm,
            "parent_checkpoint_iteration": args.checkpoint_iteration,
            "parent_prefix_slots": prefix,
            "selection_domain": "strict_parent_schedule_suffix",
            "parent_prefix_overlap_selected_sequences": overlap,
            "parent_packed_duplicate_sequence_ids": 0,
            "parent_prefix_is_byte_exact": True,
            "relative_order_within_each_pool_preserved": True,
            "target_pool_active_tokens": targets,
            "realized_pool_active_tokens": realized,
            "active_token_residuals": {key: realized[key] - targets[key] for key in ("G", "F", "O")},
            "realized_fractions": {key: str(realized[key] / continuation_active_total) for key in ("G", "F", "O")},
            "continuation_real_sequences": continuation_real_sequences,
            "continuation_pad_only_filler_slots": pad,
            "continuation_optimizer_updates": int(continuation_ids.size // args.global_batch_sequences),
            "absolute_final_optimizer_update": int(schedule_ids.size // args.global_batch_sequences),
        },
        "arms": [{
            "arm_id": "D0_mixed",
            "sequence_ids": ids_binding,
            "active_tokens": active_binding,
            "training_slots": int(schedule_ids.size),
            "real_sequences": int(np.count_nonzero(real_mask)),
            "pad_only_filler_slots": pad,
            "optimizer_updates": int(schedule_ids.size // args.global_batch_sequences),
            "pool_sequence_counts": full_pool_counts,
            "pool_active_tokens": full_pool_active,
            "continuation_pool_sequence_counts": counts,
            "continuation_pool_active_tokens": realized,
            "windows": window_rows,
        }],
    }
    try:
        write_json_atomic(temporary_dir / "schedule_manifest.json", manifest)
        require(not args.output_dir.exists(), f"immutable schedule output appeared concurrently: {args.output_dir}")
        os.rename(temporary_dir, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    print(args.output_dir / "schedule_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
