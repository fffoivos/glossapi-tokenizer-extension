#!/usr/bin/env python3
"""Minimal torch.distributed NCCL smoke for Clariden/uenv diagnostics."""

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = socket.gethostname()

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    value = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2

    if rank == 0:
        print(
            "allreduce_smoke "
            f"host={host} world_size={world_size} "
            f"value={value.item()} expected={expected} "
            f"torch={torch.__version__} cuda={torch.version.cuda}",
            flush=True,
        )

    if value.item() != expected:
        raise SystemExit(f"bad all_reduce result: got {value.item()} expected {expected}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
