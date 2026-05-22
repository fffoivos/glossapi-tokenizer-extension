#!/usr/bin/env python3
"""Run Megatron's checkpoint converter with a single-rank process group.

Megatron's `loader core` can load `torch_dist` checkpoints whose validation path
calls `torch.distributed.get_world_size()`. The upstream converter does not
initialize a process group, which is fine for release checkpoints but fails for
our TP=2 training checkpoints.
"""
from __future__ import annotations

import os
import runpy
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import api as sharded_tensor_api


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_megatron_convert_with_pg.py <convert.py> [convert args...]")
    convert_py = sys.argv[1]
    convert_dir = str(Path(convert_py).resolve().parent)
    if convert_dir not in sys.path:
        sys.path.insert(0, convert_dir)
    sys.argv = [convert_py] + sys.argv[2:]

    if not dist.is_initialized():
        backend = os.environ.get("CONVERT_DIST_BACKEND", "gloo")
        port = int(os.environ.get("CONVERT_DIST_PORT") or free_port())
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )

    from megatron.core import parallel_state as mpu

    # Leave TP/PP sizing to loader_core, which reads the checkpoint args and
    # then loads TP ranks sequentially. The torch_dist path still asks for a DP
    # replica id and DP world-size through Megatron sharded-state helpers; these
    # overrides avoid requiring a full data-parallel group in the single-process
    # converter.
    mpu.set_data_parallel_rank(0)
    mpu._MPU_DATA_PARALLEL_WORLD_SIZE = 1

    from megatron.core.dist_checkpointing import validation as dist_ckpt_validation
    from megatron.core.dist_checkpointing.strategies import torch as dist_ckpt_torch

    def _skip_single_process_sharding_integrity(*_args, **_kwargs) -> None:
        return None

    # A TP=2 checkpoint is loaded rank-by-rank by loader_core in one process.
    # The default validator expects all TP shards to be represented as current
    # distributed ranks and rejects that access pattern before loading. Missing
    # or malformed tensors still fail during the actual load.
    dist_ckpt_validation.validate_sharding_integrity = _skip_single_process_sharding_integrity

    real_dist_world_size = dist.get_world_size()
    orig_sharded_tensor_to_torch_sharded_tensor = (
        dist_ckpt_torch.sharded_tensor_to_torch_sharded_tensor
    )

    def _single_process_sequential_tp_sharded_tensor(*args, **kwargs):
        """Preserve TP shard placements while the converter runs in one process.

        Megatron's converter iterates TP ranks sequentially, but PyTorch's
        ShardedTensor constructor uses torch.distributed.get_world_size() to
        decide which shard placements are local. With a real one-rank process
        group, remote TP shards wrap back to rank 0 and PyTorch rejects the
        metadata as having more local shards than tensors. For this narrow
        construction path, report the checkpoint TP size so non-current TP
        shards remain remote in metadata. The actual checkpoint read still uses
        the real one-rank process group.
        """

        fake_world_size = int(os.environ.get("CONVERT_FAKE_SHARDING_WORLD_SIZE", "2"))
        if fake_world_size <= 1:
            return orig_sharded_tensor_to_torch_sharded_tensor(*args, **kwargs)

        orig_get_world_size = torch.distributed.get_world_size

        def _fake_get_world_size(group=None):
            if group is None:
                return fake_world_size
            return orig_get_world_size(group)

        torch.distributed.get_world_size = _fake_get_world_size
        try:
            return orig_sharded_tensor_to_torch_sharded_tensor(*args, **kwargs)
        finally:
            torch.distributed.get_world_size = orig_get_world_size

    dist_ckpt_torch.sharded_tensor_to_torch_sharded_tensor = (
        _single_process_sequential_tp_sharded_tensor
    )

    orig_parse_remote_device = sharded_tensor_api._parse_and_validate_remote_device

    def _parse_remote_device_allow_sequential_tp(pg, remote_device):
        fake_world_size = int(os.environ.get("CONVERT_FAKE_SHARDING_WORLD_SIZE", "2"))
        if fake_world_size <= real_dist_world_size:
            return orig_parse_remote_device(pg, remote_device)
        if remote_device is None:
            raise ValueError("remote device is None")
        worker_name = remote_device.worker_name()
        if worker_name is not None:
            return orig_parse_remote_device(pg, remote_device)
        return remote_device.rank(), remote_device.device()

    # The synthetic remote TP placements above are metadata-only. PyTorch's
    # ShardedTensor API normally checks that every placement rank exists in the
    # current process group; for the sequential converter they deliberately do
    # not. Keep the bypass local to rank-only placements used by this shim.
    sharded_tensor_api._parse_and_validate_remote_device = _parse_remote_device_allow_sequential_tp

    runpy.run_path(convert_py, run_name="__main__")


if __name__ == "__main__":
    main()
