#!/usr/bin/env python3
"""Materialize D0-D4 as lightweight, exact-once packed-sequence schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


WINDOWS = 128
FILLER_ID = np.uint64(2**64 - 1)
SEQUENCE_DTYPE = np.dtype(
    [
        ("sequence_id", "<u8"),
        ("packing_task_index", "<u4"),
        ("row_index", "<u4"),
        ("active_tokens", "<u2"),
    ],
    align=False,
)
POOL_KEYS = {
    "H": "hplt_new_greek",
    "G": "non_hplt_new_greek",
    "F": "foreign_replay",
    "O": "old_greek_replay",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def smooth_categories(counts: Mapping[str, int], priority: tuple[str, ...]) -> np.ndarray:
    total = sum(counts.values())
    if total == 0:
        return np.empty(0, dtype="S1")
    result = np.empty(total, dtype="S1")
    remaining_positions = np.arange(total, dtype=np.int64)
    ordered = sorted(
        counts,
        key=lambda key: (counts[key], -priority.index(key)),
    )
    used: dict[str, int] = {}
    for key in ordered[:-1]:
        count = int(counts[key])
        if count == 0:
            used[key] = 0
            continue
        local = np.floor(
            (np.arange(count, dtype=np.float64) + 0.5)
            * remaining_positions.size
            / count
        ).astype(np.int64)
        selected = remaining_positions[local]
        result[selected] = key.encode("ascii")
        keep = np.ones(remaining_positions.size, dtype=bool)
        keep[local] = False
        remaining_positions = remaining_positions[keep]
        used[key] = count
    final_key = ordered[-1]
    result[remaining_positions] = final_key.encode("ascii")
    used[final_key] = int(remaining_positions.size)
    if used != {key: int(value) for key, value in counts.items()}:
        raise RuntimeError("balanced category construction did not exhaust exact counts")
    return result


def window_sizes(total: int) -> list[int]:
    base, extra = divmod(total, WINDOWS)
    return [base + int(index < extra) for index in range(WINDOWS)]


def apportion(raw: list[float], capacities: list[int], target: int) -> list[int]:
    total_raw = sum(raw)
    scaled = [value * target / total_raw if total_raw else 0.0 for value in raw]
    result = [min(capacity, int(math.floor(value))) for value, capacity in zip(scaled, capacities, strict=True)]
    remaining = target - sum(result)
    ranking = sorted(
        range(len(raw)),
        key=lambda index: (scaled[index] - math.floor(scaled[index]), -index),
        reverse=True,
    )
    while remaining:
        changed = False
        for index in ranking:
            if result[index] < capacities[index]:
                result[index] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("quota apportionment exceeded window capacities")
    return result


def modern_window_quotas(arm: str, capacities: list[int], g_total: int, q_g: float) -> list[int] | None:
    if arm in {"D1_hard_h_to_g", "D2_hard_g_to_h"}:
        return None
    if arm == "D0_mixed":
        raw = [capacity * q_g for capacity in capacities]
    else:
        exponent = 1.0 / q_g - 1.0
        raw = []
        for index, capacity in enumerate(capacities):
            left = index / WINDOWS
            right = (index + 1) / WINDOWS
            mean = WINDOWS * (right ** (exponent + 1) - left ** (exponent + 1)) / (
                exponent + 1
            )
            if arm == "D4_gradual_g_to_h":
                mean = WINDOWS * (
                    (1 - left) ** (exponent + 1) - (1 - right) ** (exponent + 1)
                ) / (exponent + 1)
            raw.append(capacity * mean)
    return apportion(raw, capacities, g_total)


def hard_glossapi_token_quotas(
    capacities: list[int], g_total: int, *, glossapi_first: bool
) -> list[int]:
    result = [0] * len(capacities)
    remaining = int(g_total)
    order = range(len(capacities)) if glossapi_first else range(len(capacities) - 1, -1, -1)
    for index in order:
        value = min(int(capacities[index]), remaining)
        result[index] = value
        remaining -= value
        if remaining == 0:
            break
    if remaining:
        raise ValueError("hard schedule token quotas did not exhaust GlossAPI total")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-batch-sequences", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite schedule directory: {args.output_dir}")
    packed = read_json(args.packed_receipt)
    packed_schema = packed.get("schema_version")
    if packed_schema not in {
        "apertus_mini_packed_sequence_corpus_v1",
        "apertus_packed_sequence_corpus_v1",
    }:
        raise ValueError("unsupported packed corpus receipt")
    generic_schema = packed_schema == "apertus_packed_sequence_corpus_v1"
    if args.global_batch_sequences <= 0:
        raise ValueError("global batch size must be positive")
    streams: dict[str, np.ndarray] = {}
    for short, pool in POOL_KEYS.items():
        receipt = packed["pools"][pool]["sequence_catalog"]
        path = Path(receipt["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"sequence catalog drift: {path}")
        streams[short] = np.memmap(path, mode="r", dtype=SEQUENCE_DTYPE)

    counts = {key: int(value.size) for key, value in streams.items()}
    pool_active_totals = {
        key: int(stream["active_tokens"].astype(np.uint64).sum())
        for key, stream in streams.items()
    }
    total = sum(counts.values())
    skeleton = smooth_categories(
        {"M": counts["H"] + counts["G"], "F": counts["F"], "O": counts["O"]},
        ("M", "F", "O"),
    )
    replay_positions = np.flatnonzero(skeleton != b"M").astype(np.uint64)
    window_lengths = window_sizes(total)
    offsets = np.cumsum([0] + window_lengths)
    modern_capacities = [int(np.count_nonzero(skeleton[offsets[j] : offsets[j + 1]] == b"M")) for j in range(WINDOWS)]
    q_g = int(packed["pools"][POOL_KEYS["G"]]["active_tokens"]) / (
        int(packed["pools"][POOL_KEYS["H"]]["active_tokens"])
        + int(packed["pools"][POOL_KEYS["G"]]["active_tokens"])
    )
    arms = (
        "D0_mixed",
        "D1_hard_h_to_g",
        "D2_hard_g_to_h",
        "D3_gradual_h_to_g",
        "D4_gradual_g_to_h",
    )
    args.output_dir.mkdir(parents=True)
    common_replay_path = args.output_dir / "common_replay_positions.u64"
    replay_positions.tofile(common_replay_path)
    arm_receipts = []
    reference_replay_ids: np.ndarray | None = None
    reference_replay_active: np.ndarray | None = None

    for arm in arms:
        quotas = modern_window_quotas(arm, modern_capacities, counts["G"], q_g)
        if quotas is None:
            if arm == "D1_hard_h_to_g":
                modern_categories = np.concatenate(
                    (np.full(counts["H"], b"H", dtype="S1"), np.full(counts["G"], b"G", dtype="S1"))
                )
            else:
                modern_categories = np.concatenate(
                    (np.full(counts["G"], b"G", dtype="S1"), np.full(counts["H"], b"H", dtype="S1"))
                )
        else:
            parts = []
            for capacity, g_count in zip(modern_capacities, quotas, strict=True):
                parts.append(smooth_categories({"H": capacity - g_count, "G": g_count}, ("H", "G")))
            modern_categories = np.concatenate(parts)

        category_schedule = skeleton.copy()
        category_schedule[category_schedule == b"M"] = modern_categories
        cursors = {key: 0 for key in streams}
        ids = np.empty(total, dtype=np.uint64)
        active = np.empty(total, dtype=np.uint16)
        for key in ("H", "G", "F", "O"):
            positions = np.flatnonzero(category_schedule == key.encode("ascii"))
            stream = streams[key]
            if positions.size != stream.size:
                raise RuntimeError(f"schedule category count drift: {arm}/{key}")
            ids[positions] = stream["sequence_id"]
            active[positions] = stream["active_tokens"]
            cursors[key] = int(stream.size)
        pad_slots = (-total) % args.global_batch_sequences
        if pad_slots:
            ids = np.concatenate((ids, np.full(pad_slots, FILLER_ID, dtype=np.uint64)))
            active = np.concatenate((active, np.zeros(pad_slots, dtype=np.uint16)))
        replay_ids = ids[replay_positions]
        replay_active = active[replay_positions]
        if reference_replay_ids is None:
            reference_replay_ids = replay_ids.copy()
            reference_replay_active = replay_active.copy()
        elif not np.array_equal(replay_ids, reference_replay_ids) or not np.array_equal(
            replay_active, reference_replay_active
        ):
            raise RuntimeError(
                f"replay sequence IDs or active-token counts moved across arms: {arm}"
            )
        if int(np.count_nonzero(ids[:total] == FILLER_ID)) != 0:
            raise RuntimeError(f"filler leaked into real schedule slots: {arm}")
        if int(np.count_nonzero(ids[total:] != FILLER_ID)) != 0 or int(
            active[total:].astype(np.uint64).sum()
        ) != 0:
            raise RuntimeError(f"filler tail is not loss-inactive: {arm}")
        ids_path = args.output_dir / f"{arm}.sequence_ids.u64"
        active_path = args.output_dir / f"{arm}.active_tokens.u16"
        ids.tofile(ids_path)
        active.tofile(active_path)

        windows = []
        for index in range(WINDOWS):
            start, end = int(offsets[index]), int(offsets[index + 1])
            categories = category_schedule[start:end]
            windows.append(
                {
                    "window": index,
                    "slot_start": start,
                    "slot_end": end,
                    "sequence_counts": {
                        key: int(np.count_nonzero(categories == key.encode("ascii")))
                        for key in ("H", "G", "F", "O")
                    },
                    "active_tokens": {
                        key: int(active[start:end][categories == key.encode("ascii")].astype(np.uint64).sum())
                        for key in ("H", "G", "F", "O")
                    },
                }
            )
        window_totals = [sum(row["active_tokens"].values()) for row in windows]
        ideal_old = apportion(
            [total * 0.01 for total in window_totals],
            window_totals,
            pool_active_totals["O"],
        )
        foreign_capacities = [
            total - old for total, old in zip(window_totals, ideal_old, strict=True)
        ]
        ideal_foreign = apportion(
            [total * 0.20 for total in window_totals],
            foreign_capacities,
            pool_active_totals["F"],
        )
        ideal_modern = [
            total - old - foreign
            for total, old, foreign in zip(
                window_totals, ideal_old, ideal_foreign, strict=True
            )
        ]
        if arm == "D1_hard_h_to_g":
            ideal_glossapi = hard_glossapi_token_quotas(
                ideal_modern, pool_active_totals["G"], glossapi_first=False
            )
        elif arm == "D2_hard_g_to_h":
            ideal_glossapi = hard_glossapi_token_quotas(
                ideal_modern, pool_active_totals["G"], glossapi_first=True
            )
        else:
            ideal_glossapi = modern_window_quotas(
                arm, ideal_modern, pool_active_totals["G"], q_g
            )
            if ideal_glossapi is None:
                raise RuntimeError(f"missing ideal token quotas for {arm}")
        ideal_hplt = [
            modern - glossapi
            for modern, glossapi in zip(ideal_modern, ideal_glossapi, strict=True)
        ]
        ideal_by_pool = {
            "H": ideal_hplt,
            "G": ideal_glossapi,
            "F": ideal_foreign,
            "O": ideal_old,
        }
        cumulative = {key: 0 for key in ("H", "G", "F", "O")}
        max_abs_residual = {key: 0 for key in cumulative}
        for index, row in enumerate(windows):
            row["ideal_active_token_quota"] = {
                key: int(ideal_by_pool[key][index]) for key in cumulative
            }
            residuals = {
                key: int(row["active_tokens"][key] - ideal_by_pool[key][index])
                for key in cumulative
            }
            row["realized_minus_ideal_active_tokens"] = residuals
            for key, value in residuals.items():
                cumulative[key] += value
                max_abs_residual[key] = max(max_abs_residual[key], abs(value))
            row["cumulative_active_token_residual"] = dict(cumulative)
        if cumulative != {key: 0 for key in cumulative}:
            raise RuntimeError(f"terminal token quota residual did not close: {arm}/{cumulative}")
        arm_receipts.append(
            {
                "arm_id": arm,
                "sequence_ids": {"path": str(ids_path), "sha256": sha256_file(ids_path), "bytes": ids_path.stat().st_size},
                "active_tokens": {"path": str(active_path), "sha256": sha256_file(active_path), "bytes": active_path.stat().st_size},
                "training_slots": int(ids.size),
                "real_sequences": total,
                "pad_only_filler_slots": pad_slots,
                "optimizer_updates": int(ids.size // args.global_batch_sequences),
                "pool_sequence_counts": counts,
                "pool_active_tokens": pool_active_totals,
                "max_abs_window_active_token_residual": max_abs_residual,
                "terminal_cumulative_active_token_residual": cumulative,
                "windows": windows,
            }
        )

    if reference_replay_ids is None or reference_replay_active is None:
        raise RuntimeError("no replay reference was constructed")
    canonical_sequence_inventory = hashlib.sha256()
    for key in ("H", "G", "F", "O"):
        canonical_sequence_inventory.update(streams[key]["sequence_id"].tobytes())
        canonical_sequence_inventory.update(streams[key]["active_tokens"].tobytes())
    replay_assignment = hashlib.sha256()
    replay_assignment.update(replay_positions.tobytes())
    replay_assignment.update(reference_replay_ids.tobytes())
    replay_assignment.update(reference_replay_active.tobytes())
    receipt = {
        "schema_version": (
            "apertus_data_order_schedules_v1"
            if generic_schema
            else "apertus_mini_five_data_order_schedules_v1"
        ),
        "status": "completed",
        "packed_corpus_receipt": {
            "path": str(args.packed_receipt.resolve()),
            "sha256": sha256_file(args.packed_receipt),
        },
        "common_contract": {
            "arms": list(arms),
            "window_count": WINDOWS,
            "global_batch_sequences": args.global_batch_sequences,
            "same_exact_sequence_multiset": True,
            "same_replay_sequence_ids_at_same_global_positions": True,
            "same_per_sequence_active_token_count": True,
            "canonical_sequence_inventory_sha256": canonical_sequence_inventory.hexdigest(),
            "common_replay_assignment_sha256": replay_assignment.hexdigest(),
            "replay_positions": {
                "path": str(common_replay_path),
                "sha256": sha256_file(common_replay_path),
                "bytes": common_replay_path.stat().st_size,
                "rows": int(replay_positions.size),
            },
            "modern_greek_token_fraction_glossapi": str(q_g),
            "gradual_curve_exponent": str(1.0 / q_g - 1.0),
            "sequence_granularity_residuals_reported_per_window": True,
        },
        "arms": arm_receipts,
    }
    write_json_atomic(args.output_dir / "schedule_manifest.json", receipt)
    print(json.dumps({"ok": True, "arms": len(arms), "real_sequences": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
