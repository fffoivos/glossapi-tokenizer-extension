#!/usr/bin/env python3
"""Run Megatron with a fail-closed phase-local training-data cursor.

Megatron restores global ``consumed_train_samples`` from a checkpoint. A new
phase-specific dataset must instead start at the samples consumed *inside that
phase*, without changing optimizer, scheduler, RNG, or the global update. This
wrapper patches only the first (training) data-loader construction.

Required environment when ``PHASE_LOCAL_DATA_INDEX=1``:

* ``PHASE_START_UPDATE``: checkpoint update immediately before phase sample 0;
* ``EXPECTED_GLOBAL_UPDATE``: update restored by the checkpoint;
* ``GLOBAL_BATCH_SIZE``: frozen global sequence batch (1024 here);
* ``EXPECTED_PHASE_LOCAL_CONSUMED_SAMPLES``: exact phase cursor; and
* ``EXPECTED_DATA_CACHE_SHA256`` / ``ACTUAL_DATA_CACHE_SHA256``: immutable
  phase-cache binding checked before Megatron starts.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import runpy
import sys


def required_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required")
    value = int(raw)
    if value < 0:
        raise RuntimeError(f"{name} must be non-negative")
    return value


def validate_contract() -> tuple[int, int]:
    phase_start = required_int("PHASE_START_UPDATE")
    global_update = required_int("EXPECTED_GLOBAL_UPDATE")
    global_batch = required_int("GLOBAL_BATCH_SIZE")
    local_samples = required_int("EXPECTED_PHASE_LOCAL_CONSUMED_SAMPLES")
    if global_batch <= 0:
        raise RuntimeError("GLOBAL_BATCH_SIZE must be positive")
    if global_update < phase_start:
        raise RuntimeError("global update precedes phase start")
    derived = (global_update - phase_start) * global_batch
    if local_samples != derived:
        raise RuntimeError(
            f"phase-local cursor drift: configured={local_samples} derived={derived}"
        )
    expected_cache = os.environ.get("EXPECTED_DATA_CACHE_SHA256", "")
    actual_cache = os.environ.get("ACTUAL_DATA_CACHE_SHA256", "")
    if len(expected_cache) != 64 or actual_cache != expected_cache:
        raise RuntimeError("phase data-cache SHA-256 missing or mismatched")
    return global_update * global_batch, local_samples


def override_train_dataset_samples(
    sizes: list[int], expected_global: int, expected_local: int
) -> tuple[list[int], bool]:
    if expected_local <= 0 or expected_local >= expected_global:
        raise RuntimeError("phase-local dataset-size override is not a strict positive reduction")
    if not sizes or int(sizes[0]) != expected_global:
        return sizes, False
    adjusted = list(sizes)
    adjusted[0] = expected_local
    return adjusted, True


def install_guard() -> None:
    if os.environ.get("PHASE_LOCAL_DATA_INDEX", "0") != "1":
        return
    expected_global_samples, local_samples = validate_contract()
    sampler_target = "megatron.legacy.data.data_samplers"
    builder_target = "megatron.core.datasets.blended_megatron_dataset_builder"

    def patch_sampler_module(module) -> None:
        original = getattr(module, "build_pretraining_data_loader", None)
        if original is None or getattr(original, "_phase_local_data_index_guard", False):
            return
        state = {"train_seen": False}

        def patched(dataset, consumed_samples):
            if not state["train_seen"]:
                state["train_seen"] = True
                if int(consumed_samples) != expected_global_samples:
                    raise RuntimeError(
                        "checkpoint consumed-sample drift: "
                        f"loader={consumed_samples} expected={expected_global_samples}"
                    )
                print(
                    "[phase_local_data_index_guard] "
                    f"rank={os.environ.get('RANK', '?')} global_samples={consumed_samples} "
                    f"phase_local_samples={local_samples}",
                    file=sys.stderr,
                    flush=True,
                )
                return original(dataset, local_samples)
            return original(dataset, consumed_samples)

        patched._phase_local_data_index_guard = True
        module.build_pretraining_data_loader = patched

    def patch_builder_module(module) -> None:
        if os.environ.get("PHASE_LOCAL_DATASET_SIZE_OVERRIDE", "0") != "1":
            return
        expected_global = required_int("EXPECTED_GLOBAL_TRAIN_SAMPLES")
        expected_local = required_int("EXPECTED_PHASE_DATASET_SAMPLES")
        builder = getattr(module, "BlendedMegatronDatasetBuilder", None)
        if builder is None:
            raise RuntimeError("BlendedMegatronDatasetBuilder is missing")
        original = builder.__init__
        if getattr(original, "_phase_local_dataset_size_guard", False):
            return
        state = {"replaced": False}

        def patched(self, cls, sizes, is_built_on_rank, config):
            adjusted, replaced = override_train_dataset_samples(
                sizes, expected_global, expected_local
            ) if not state["replaced"] else (sizes, False)
            if replaced:
                state["replaced"] = True
                print(
                    "[phase_local_data_index_guard] "
                    f"rank={os.environ.get('RANK', '?')} dataset samples "
                    f"{expected_global} -> {expected_local}",
                    file=sys.stderr,
                    flush=True,
                )
            return original(self, cls, adjusted, is_built_on_rank, config)

        patched._phase_local_dataset_size_guard = True
        builder.__init__ = patched

    patchers = {
        sampler_target: patch_sampler_module,
        builder_target: patch_builder_module,
    }
    for target, patcher in patchers.items():
        if target in sys.modules:
            patcher(sys.modules[target])

    class PatchLoader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            create = getattr(self.wrapped, "create_module", None)
            return create(spec) if create is not None else None

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            patchers[module.__name__](module)

    class PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname not in patchers:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            spec.loader = PatchLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, PatchFinder())


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: phase_local_data_index_guard.py TARGET [ARGS ...]")
    install_guard()
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
