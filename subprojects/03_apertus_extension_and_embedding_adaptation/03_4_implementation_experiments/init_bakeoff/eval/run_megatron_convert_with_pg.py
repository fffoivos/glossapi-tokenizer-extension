#!/usr/bin/env python3
"""Run Megatron's checkpoint converter with a single-rank process group.

Megatron's `loader core` can load `torch_dist` checkpoints whose validation path
calls `torch.distributed.get_world_size()`. The upstream converter does not
initialize a process group, which is fine for release checkpoints but fails for
our TP=2 training checkpoints. A single-rank gloo group is enough because the
loader reads TP ranks sequentially inside one process.
"""
from __future__ import annotations

import os
import runpy
import socket
import sys

import torch.distributed as dist


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_megatron_convert_with_pg.py <convert.py> [convert args...]")
    convert_py = sys.argv[1]
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

    runpy.run_path(convert_py, run_name="__main__")


if __name__ == "__main__":
    main()
