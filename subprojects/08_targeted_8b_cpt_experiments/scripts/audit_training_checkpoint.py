#!/usr/bin/env python3
"""Audit a completed targeted-CPT DCP checkpoint before issuing a resume permit."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from build_checkpoint_permit import REQUIRED_CHECKS
from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from freeze_phase_blend_cache import PHASE_START
from freeze_phase_blend_cache import validate_receipt as validate_phase_cache

TRAINING_ROW = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?"
    r"consumed samples:\s*(?P<samples>\d+).*?"
    r"lm loss:\s*(?P<loss>[+\-0-9.Ee]+).*?"
    r"grad norm:\s*(?P<grad>[+\-0-9.Ee]+).*?"
    r"params norm:\s*(?P<params>[+\-0-9.Ee]+).*?"
    r"number of skipped iterations:\s*(?P<skipped>\d+).*?"
    r"number of nan iterations:\s*(?P<nan>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--source-phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-phase-cache-receipt", type=Path, required=True)
    parser.add_argument("--segment-preflight", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-graceful-stop", action="store_true")
    return parser.parse_args()


def validate_claim_window(
    preflight: dict[str, Any], *, observed_update: int, allow_graceful_stop: bool
) -> tuple[int, int]:
    claimed_start = int(preflight.get("start_update", -1))
    claimed_exit = int(preflight.get("exit_update", -1))
    if allow_graceful_stop:
        require(
            claimed_start < observed_update < claimed_exit,
            "graceful checkpoint is outside the preflight claim",
        )
    else:
        require(claimed_exit == observed_update, "segment preflight exit drift")
    return claimed_start, claimed_exit


def storage_inventory(root: Path, metadata: Any) -> list[dict[str, Any]]:
    maximum_ends: dict[str, int] = {}
    chunks: dict[str, int] = {}
    for value in metadata.storage_data.values():
        relative = str(value.relative_path)
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe DCP storage path")
        end = int(value.offset) + int(value.length)
        require(int(value.offset) >= 0 and int(value.length) > 0, "invalid DCP storage range")
        maximum_ends[relative] = max(maximum_ends.get(relative, 0), end)
        chunks[relative] = chunks.get(relative, 0) + 1
    require(bool(maximum_ends), "DCP storage inventory is empty")
    rows = []
    for relative in sorted(maximum_ends):
        path = root / relative
        require(path.is_file(), f"DCP storage file missing: {path}")
        size = path.stat().st_size
        require(size >= maximum_ends[relative], f"DCP storage file is truncated: {path}")
        rows.append({
            "relative_path": relative,
            "bytes": size,
            "referenced_chunks": chunks[relative],
            "maximum_referenced_end": maximum_ends[relative],
        })
    return rows


def parse_training_log(path: Path, update: int) -> dict[str, Any]:
    require(path.is_file(), f"training log missing: {path}")
    rows = []
    saved = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRAINING_ROW.search(line)
        if match:
            row = {key: int(value) if key in {"iteration", "samples", "skipped", "nan"} else float(value) for key, value in match.groupdict().items()}
            require(all(math.isfinite(float(row[name])) for name in ("loss", "grad", "params")), "non-finite training metric")
            require(row["skipped"] == 0 and row["nan"] == 0, "skipped or NaN training update")
            require(row["samples"] == row["iteration"] * 1024, "training-log global sample cursor drift")
            rows.append(row)
        if re.search(rf"successfully saved checkpoint from iteration\s+{update}\b", line):
            saved = True
    require(bool(rows), "training log contains no parseable optimizer updates")
    require(rows[-1]["iteration"] == update, "training log does not end at requested checkpoint update")
    require(saved, "training log lacks successful checkpoint-save confirmation")
    return {
        "optimizer_update_rows": len(rows),
        "first_update": rows[0]["iteration"],
        "last_update": rows[-1]["iteration"],
        "last_metrics": rows[-1],
        "successful_save_confirmation": True,
    }


def main() -> int:
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    args = parse_args()
    require(not args.output.exists(), f"immutable checkpoint audit exists: {args.output}")
    root = args.checkpoint_root.resolve()
    require(root.is_dir(), "checkpoint root missing")
    common_path = root / "common.pt"
    require(common_path.is_file(), "checkpoint common.pt missing")

    source_cache = read_json(args.source_phase_cache_receipt)
    validate_phase_cache(
        source_cache,
        phase=args.source_phase,
        data_path_spec=Path(str(source_cache.get("data_path_spec", {}).get("path", ""))),
        cache_root=Path(str(source_cache.get("cache_root", ""))),
    )
    preflight = read_json(args.segment_preflight)
    require(preflight.get("schema_version") == "apertus_hard_h_to_g_train_segment_preflight_v1", "segment preflight schema drift")
    require(preflight.get("status") == "passed" and preflight.get("scale") == args.scale, "segment preflight status/scale drift")
    require(preflight.get("phase") == args.source_phase, "segment preflight phase drift")
    _, claimed_exit = validate_claim_window(
        preflight,
        observed_update=args.update,
        allow_graceful_stop=args.allow_graceful_stop,
    )
    require(preflight.get("phase_cache_receipt") == file_binding(args.source_phase_cache_receipt), "segment preflight phase-cache binding drift")

    common = torch.load(common_path, map_location="cpu", weights_only=False)
    require(isinstance(common, dict), "checkpoint common.pt is not a mapping")
    require(int(common.get("iteration", -1)) == args.update, "checkpoint common iteration drift")
    scheduler = common.get("opt_param_scheduler")
    require(isinstance(scheduler, dict) and int(scheduler.get("num_steps", -1)) == args.update * 1024, "scheduler/global sample cursor drift")
    expected_phase_local_samples = (args.update - PHASE_START[args.source_phase]) * 1024
    require(expected_phase_local_samples > 0, "checkpoint does not advance its source phase")

    metadata = FileSystemReader(root).read_metadata()
    mcore_data = getattr(metadata, "mcore_data", None)
    state_keys = tuple(str(key) for key in metadata.state_dict_metadata)
    model_keys = tuple(key for key in state_keys if not key.startswith(("optimizer.", "rng_state/", "rerun_state_machine_state/")))
    optimizer_keys = tuple(key for key in state_keys if key.startswith("optimizer."))
    rng_keys = tuple(key for key in state_keys if key.startswith("rng_state/"))
    require(bool(model_keys), "DCP model-state metadata missing")
    require(bool(optimizer_keys), "DCP optimizer-state metadata missing")
    require(bool(rng_keys), "DCP RNG-state metadata missing")
    require(
        isinstance(mcore_data, dict) and bool(mcore_data),
        "DCP Megatron model-space optimizer metadata missing",
    )
    storage = storage_inventory(root, metadata)
    log = parse_training_log(args.training_log, args.update)
    checkpoint_files = sorted(path for path in root.rglob("*") if path.is_file())
    require(bool(checkpoint_files), "checkpoint file inventory is empty")
    for path in checkpoint_files:
        path.chmod(path.stat().st_mode & ~0o222)
        require(
            path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
            f"checkpoint file remains writable: {path}",
        )

    checks = {
        "model_state_metadata_complete": bool(model_keys),
        "optimizer_state_metadata_complete": bool(optimizer_keys),
        "optimizer_model_space_metadata_complete": bool(mcore_data),
        "rng_state_complete": bool(rng_keys),
        "scheduler_state_complete": int(scheduler.get("num_steps", -1)) == args.update * 1024,
        "data_cursor_verified": expected_phase_local_samples > 0,
        "training_log_no_nonfinite_updates": (
            log["last_update"] == args.update
            and log["last_metrics"]["skipped"] == 0
            and log["last_metrics"]["nan"] == 0
        ),
        "checkpoint_storage_inventory_verified": bool(storage),
        "checkpoint_files_read_only": all(
            path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
            for path in checkpoint_files
        ),
    }
    require(
        set(checks) == set(REQUIRED_CHECKS) and all(checks.values()),
        "checkpoint audit checks are incomplete",
    )
    payload: dict[str, object] = {
        "schema_version": "apertus_hard_h_to_g_checkpoint_state_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scale": args.scale,
        "source_phase": args.source_phase,
        "update": args.update,
        "claimed_exit_update": claimed_exit,
        "gracefully_stopped": args.allow_graceful_stop,
        "checkpoint_root": str(root),
        "checks": checks,
        "state_metadata": {
            "total_keys": len(state_keys),
            "model_keys": len(model_keys),
            "optimizer_keys": len(optimizer_keys),
            "mcore_data_entries": len(mcore_data),
            "rng_keys": len(rng_keys),
        },
        "scheduler": {
            "num_steps": int(scheduler["num_steps"]),
            "max_lr": scheduler.get("max_lr"),
            "min_lr": scheduler.get("min_lr"),
            "lr_decay_style": scheduler.get("lr_decay_style"),
        },
        "data_cursor": {
            "global_consumed_samples": args.update * 1024,
            "phase_local_consumed_samples": expected_phase_local_samples,
            "phase_start_update": PHASE_START[args.source_phase],
        },
        "storage_inventory": storage,
        "checkpoint_file_count": len(checkpoint_files),
        "training_log_summary": log,
        "source_phase_cache_receipt": file_binding(args.source_phase_cache_receipt),
        "segment_preflight": file_binding(args.segment_preflight),
        "training_log": file_binding(args.training_log),
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "update": args.update, "storage_files": len(storage)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
