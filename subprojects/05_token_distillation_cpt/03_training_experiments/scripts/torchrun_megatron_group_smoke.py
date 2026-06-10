#!/usr/bin/env python3
"""Pure PyTorch NCCL smoke for Megatron-style TP/DP process groups."""

import os
import socket

import torch
import torch.distributed as dist


TP_SIZE = int(os.environ.get("SMOKE_TP_SIZE", "2"))
N_ITERS = int(os.environ.get("SMOKE_ITERS", "8"))
PAYLOAD_ELEMENTS = int(os.environ.get("SMOKE_PAYLOAD_ELEMENTS", str(1024 * 1024)))
DTYPE_NAME = os.environ.get("SMOKE_DTYPE", "float32")
COLLECTIVES = tuple(
    item.strip() for item in os.environ.get("SMOKE_COLLECTIVES", "all_reduce").split(",") if item.strip()
)
VALID_COLLECTIVES = {"all_reduce", "reduce_scatter", "all_gather"}
unknown_collectives = sorted(set(COLLECTIVES) - VALID_COLLECTIVES)
if unknown_collectives:
    raise SystemExit(f"unknown SMOKE_COLLECTIVES values: {unknown_collectives}")
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
if DTYPE_NAME not in DTYPES:
    raise SystemExit(f"SMOKE_DTYPE={DTYPE_NAME} not in {sorted(DTYPES)}")


def build_groups(world_size: int):
    if world_size % TP_SIZE != 0:
        raise SystemExit(f"world_size={world_size} is not divisible by TP_SIZE={TP_SIZE}")

    tp_groups = [list(range(start, start + TP_SIZE)) for start in range(0, world_size, TP_SIZE)]
    dp_groups = [[tp_rank + dp_rank * TP_SIZE for dp_rank in range(world_size // TP_SIZE)] for tp_rank in range(TP_SIZE)]
    return [("tp", ranks) for ranks in tp_groups] + [("dp_with_cp", ranks) for ranks in dp_groups]


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = socket.gethostname()

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    group_specs = build_groups(world_size)
    groups = []
    for name, ranks in group_specs:
        groups.append((name, ranks, dist.new_group(ranks=ranks, backend="nccl")))

    dist.barrier()

    for iteration in range(N_ITERS):
        for name, ranks, group in groups:
            if rank not in ranks:
                continue

            local_group_rank = ranks.index(rank)
            tiny = torch.tensor([float(local_group_rank)], device="cuda")
            dist.all_reduce(tiny, op=dist.ReduceOp.SUM, group=group)
            expected_tiny = len(ranks) * (len(ranks) - 1) / 2
            if tiny.item() != expected_tiny:
                raise SystemExit(
                    f"{name} ranks={ranks} iter={iteration} bad tiny all_reduce: "
                    f"got {tiny.item()} expected {expected_tiny}"
                )

            expected_payload = len(ranks) * (len(ranks) + 1) / 2

            if "all_reduce" in COLLECTIVES:
                payload = torch.full(
                    (PAYLOAD_ELEMENTS,),
                    float(local_group_rank + 1),
                    device="cuda",
                    dtype=DTYPES[DTYPE_NAME],
                )
                dist.all_reduce(payload, op=dist.ReduceOp.SUM, group=group)
                if payload[0].item() != expected_payload:
                    raise SystemExit(
                        f"{name} ranks={ranks} iter={iteration} bad payload all_reduce: "
                        f"got {payload[0].item()} expected {expected_payload}"
                    )

            if "reduce_scatter" in COLLECTIVES:
                scatter_input = torch.full(
                    (len(ranks) * PAYLOAD_ELEMENTS,),
                    float(local_group_rank + 1),
                    device="cuda",
                    dtype=DTYPES[DTYPE_NAME],
                )
                scatter_output = torch.empty(
                    (PAYLOAD_ELEMENTS,),
                    device="cuda",
                    dtype=DTYPES[DTYPE_NAME],
                )
                dist.reduce_scatter_tensor(scatter_output, scatter_input, op=dist.ReduceOp.SUM, group=group)
                if scatter_output[0].item() != expected_payload:
                    raise SystemExit(
                        f"{name} ranks={ranks} iter={iteration} bad reduce_scatter: "
                        f"got {scatter_output[0].item()} expected {expected_payload}"
                    )

            if "all_gather" in COLLECTIVES:
                gather_input = torch.full(
                    (PAYLOAD_ELEMENTS,),
                    float(local_group_rank + 1),
                    device="cuda",
                    dtype=DTYPES[DTYPE_NAME],
                )
                gather_output = torch.empty(
                    (len(ranks) * PAYLOAD_ELEMENTS,),
                    device="cuda",
                    dtype=DTYPES[DTYPE_NAME],
                )
                dist.all_gather_into_tensor(gather_output, gather_input, group=group)
                for chunk_rank in range(len(ranks)):
                    got = gather_output[chunk_rank * PAYLOAD_ELEMENTS].item()
                    expected = float(chunk_rank + 1)
                    if got != expected:
                        raise SystemExit(
                            f"{name} ranks={ranks} iter={iteration} bad all_gather chunk={chunk_rank}: "
                            f"got {got} expected {expected}"
                        )

        dist.barrier()

    if rank == 0:
        print(
            "megatron_group_smoke "
            f"host={host} world_size={world_size} tp_size={TP_SIZE} "
            f"groups={[(name, ranks) for name, ranks, _ in groups]} "
            f"payload_elements={PAYLOAD_ELEMENTS} dtype={DTYPE_NAME} "
            f"collectives={COLLECTIVES} "
            f"iters={N_ITERS} torch={torch.__version__} cuda={torch.version.cuda}",
            flush=True,
        )

    for _, _, group in groups:
        dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
