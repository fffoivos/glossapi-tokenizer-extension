#!/usr/bin/env python3
"""Use phase-relative data progress while preserving global training state.

Megatron checkpoints store the global ``consumed_train_samples`` value.  Once
the run switches to the phase-2 blend, each resumed segment must pass
``global_consumed - phase_start_iteration * global_batch_size`` to the TRAIN
data loader.  Optimizer, scheduler, RNG, and the checkpoint iteration are not
changed.

Usage:
  python phase_relative_data_index.py <te-guard.py> <pretrain_gpt.py> <args...>

Required environment:
  CPT_PHASE_START_ITERATION
  CPT_GLOBAL_BATCH_SIZE
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import runpy
import sys


def phase_relative_consumed_samples(
    global_consumed_samples: int,
    phase_start_iteration: int,
    global_batch_size: int,
) -> int:
    values = (global_consumed_samples, phase_start_iteration, global_batch_size)
    if any(not isinstance(value, int) for value in values):
        raise TypeError("phase-relative accounting inputs must be integers")
    if global_consumed_samples < 0 or phase_start_iteration < 0:
        raise ValueError("sample and iteration counts must be non-negative")
    if global_batch_size <= 0:
        raise ValueError("global batch size must be positive")
    phase_offset = phase_start_iteration * global_batch_size
    if global_consumed_samples < phase_offset:
        raise ValueError(
            "checkpoint consumed-sample count predates the selected phase: "
            f"{global_consumed_samples} < {phase_offset}"
        )
    return global_consumed_samples - phase_offset


def _required_uint(name: str, *, positive: bool = False) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isdigit():
        raise RuntimeError(f"{name} must be set to an unsigned integer")
    value = int(raw)
    if positive and value == 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def install_phase_relative_guard() -> None:
    phase_start = _required_uint("CPT_PHASE_START_ITERATION")
    global_batch = _required_uint("CPT_GLOBAL_BATCH_SIZE", positive=True)
    target = "megatron.legacy.data.data_samplers"

    def patch_module(module) -> None:
        original = getattr(module, "build_pretraining_data_loader", None)
        if original is None or getattr(original, "_phase_relative_guard", False):
            return
        state = {"train_seen": False}

        def patched(dataset, consumed_samples):
            if not state["train_seen"]:
                state["train_seen"] = True
                relative = phase_relative_consumed_samples(
                    int(consumed_samples), phase_start, global_batch
                )
                rank = os.environ.get("RANK", "?")
                print(
                    "[phase_relative_data_index] "
                    f"rank={rank} train consumed_samples {consumed_samples} -> "
                    f"{relative} (phase_start_iteration={phase_start}; global state kept)",
                    file=sys.stderr,
                    flush=True,
                )
                return original(dataset, relative)
            return original(dataset, consumed_samples)

        patched._phase_relative_guard = True
        module.build_pretraining_data_loader = patched

    if target in sys.modules:
        patch_module(sys.modules[target])
        return

    class PatchLoader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            if hasattr(self.wrapped, "create_module"):
                return self.wrapped.create_module(spec)
            return None

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            patch_module(module)

    class PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname != target:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            spec.loader = PatchLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, PatchFinder())
    print(
        "[phase_relative_data_index] lazy import hook installed",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    install_phase_relative_guard()
    if len(sys.argv) < 3:
        raise SystemExit("usage: phase_relative_data_index.py TE_GUARD PRETRAIN_GPT ...")
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
