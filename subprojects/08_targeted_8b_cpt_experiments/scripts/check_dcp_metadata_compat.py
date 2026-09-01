#!/usr/bin/env python3
"""Prove the frozen runtime preserves Megatron dynamic DCP metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, Metadata


def file_binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-compat-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable compatibility receipt exists: {args.output}")
    shim = args.runtime_compat_dir.resolve() / "sitecustomize.py"
    if not shim.is_file():
        raise FileNotFoundError(shim)

    shim_loaded = bool(
        getattr(FileSystemWriter.finish, "_apertus_preserves_dynamic_metadata", False)
    )
    if not shim_loaded:
        raise RuntimeError("frozen DCP metadata compatibility shim was not loaded")
    sentinel = {"sentinel": {"nd_reformulated_orig_global_shape": (2, 3)}}
    metadata = Metadata(state_dict_metadata={})
    metadata.mcore_data = sentinel
    with tempfile.TemporaryDirectory(prefix="h2g-dcp-metadata-") as temporary:
        root = Path(temporary)
        FileSystemWriter(root).finish(metadata, [])
        observed = getattr(FileSystemReader(root).read_metadata(), "mcore_data", None)
    if observed != sentinel:
        raise RuntimeError("Megatron mcore_data did not survive DCP metadata round trip")

    payload = {
        "schema_version": "apertus_hard_h_to_g_dcp_metadata_compat_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
        "shim_loaded": shim_loaded,
        "mcore_data_round_trip": True,
        "runtime_compat_dir": str(args.runtime_compat_dir.resolve()),
        "sitecustomize": file_binding(shim),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    print(json.dumps({"ok": True, "torch_version": torch.__version__}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
