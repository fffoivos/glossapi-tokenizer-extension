#!/usr/bin/env python3
"""Request exact Megatron checkpoints at predeclared optimizer iterations."""

from __future__ import annotations

import functools
import os
from pathlib import Path


def parse_iterations(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not values or values[0] <= 0:
        raise ValueError("exact checkpoint iterations must be positive integers")
    return values


def install_from_environment() -> tuple[int, ...]:
    iterations = parse_iterations(os.environ.get("MINI_SCHEDULE_SAVE_ITERATIONS", ""))
    no_save_exit_raw = os.environ.get("MINI_SCHEDULE_NO_SAVE_EXIT_ITERATION", "").strip()
    no_save_exit = int(no_save_exit_raw) if no_save_exit_raw else None
    if no_save_exit is not None and no_save_exit <= 0:
        raise ValueError("no-save exit iteration must be a positive integer")
    if no_save_exit is not None and no_save_exit in iterations:
        raise ValueError("one iteration cannot be both an exact save and a no-save exit")
    if not iterations and no_save_exit is None:
        return iterations

    from megatron.training import get_args
    from megatron.training import training as training_runtime

    original = training_runtime.checkpoint_and_decide_exit
    if getattr(original, "_mini_exact_checkpoint_hook", False):
        raise RuntimeError("exact checkpoint hook already installed")

    @functools.wraps(original)
    def wrapped(
        model,
        optimizer,
        opt_param_scheduler,
        iteration,
        num_floating_point_operations_so_far,
        checkpointing_context,
        train_data_iterator,
    ):
        # A one-update, same-allocation causal-control gate needs to stop after
        # the first post-checkpoint update without spending several minutes or
        # hundreds of GB on a throwaway checkpoint.  This is deliberately
        # environment-gated and mutually exclusive with an exact save above.
        if no_save_exit is not None and int(iteration) == no_save_exit:
            return True
        if int(iteration) in iterations:
            args = get_args()
            if not args.save:
                raise RuntimeError("exact checkpoint iteration reached without --save")
            Path(args.save_trigger).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_trigger).touch(exist_ok=True)
        return original(
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
        )

    wrapped._mini_exact_checkpoint_hook = True
    training_runtime.checkpoint_and_decide_exit = wrapped
    return iterations
