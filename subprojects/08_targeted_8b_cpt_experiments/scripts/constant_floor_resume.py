#!/usr/bin/env python3
"""Hold the exact LR floor after restore, then apply phase-local data indexing."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import os
from pathlib import Path
import runpy
import sys


def enforce_constant_floor(scheduler, param_group: dict, expected_lr: float) -> bool:
    """Overwrite every scheduler/param-group quantity that can re-anchor LR."""
    before = (
        scheduler.init_lr, scheduler.max_lr, scheduler.min_lr,
        scheduler.lr_warmup_steps, scheduler.lr_decay_style,
        param_group.get("max_lr"), param_group.get("min_lr"), param_group.get("lr"),
    )
    scheduler.init_lr = expected_lr
    scheduler.max_lr = expected_lr
    scheduler.min_lr = expected_lr
    scheduler.lr_warmup_steps = 0
    scheduler.lr_decay_style = "constant"
    param_group["max_lr"] = expected_lr
    param_group["min_lr"] = expected_lr
    param_group["lr"] = expected_lr * float(param_group.get("lr_mult", 1.0))
    after = (
        scheduler.init_lr, scheduler.max_lr, scheduler.min_lr,
        scheduler.lr_warmup_steps, scheduler.lr_decay_style,
        param_group.get("max_lr"), param_group.get("min_lr"), param_group.get("lr"),
    )
    return before != after


def required_floor() -> float:
    raw = os.environ.get("H2G_CONSTANT_FLOOR_LR")
    if raw is None:
        raise RuntimeError("H2G_CONSTANT_FLOOR_LR must be set")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("H2G_CONSTANT_FLOOR_LR must be finite and positive")
    return value


def install() -> None:
    expected = required_floor()
    target = "megatron.core.optimizer_param_scheduler"

    def patch_module(module) -> None:
        scheduler_class = getattr(module, "OptimizerParamScheduler", None)
        if scheduler_class is None:
            raise RuntimeError("OptimizerParamScheduler is missing")
        original = scheduler_class.get_lr
        if getattr(original, "_h2g_constant_floor_guard", False):
            return
        reported = {"value": False}

        def patched(self, param_group):
            changed = enforce_constant_floor(self, param_group, expected)
            result = original(self, param_group)
            if result != expected:
                raise RuntimeError(f"constant-floor scheduler returned {result!r}, expected {expected!r}")
            if changed and not reported["value"] and os.environ.get("RANK", "0") == "0":
                print(f"[h2g_constant_floor] restored scheduler forced to {expected:.17g}", file=sys.stderr, flush=True)
                reported["value"] = True
            return result

        patched._h2g_constant_floor_guard = True
        scheduler_class.get_lr = patched

    if target in sys.modules:
        patch_module(sys.modules[target])
        return

    class Loader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            return self.wrapped.create_module(spec) if hasattr(self.wrapped, "create_module") else None

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            patch_module(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname != target:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
    print(f"[h2g_constant_floor] lazy guard installed for {expected:.17g}", file=sys.stderr, flush=True)


def main() -> None:
    install()
    if len(sys.argv) < 3:
        raise SystemExit("usage: constant_floor_resume.py TE_GUARD PRETRAIN_GPT ...")
    code_root = os.environ.get("H2G_CODE_ROOT")
    if not code_root:
        raise RuntimeError("H2G_CODE_ROOT must be set")
    phase_wrapper = (
        Path(code_root)
        / "subprojects"
        / "08_targeted_8b_cpt_experiments"
        / "scripts"
        / "phase_local_data_index_guard.py"
    )
    if not phase_wrapper.is_file():
        raise RuntimeError(f"phase-relative wrapper missing: {phase_wrapper}")
    sys.argv = [str(phase_wrapper), *sys.argv[1:]]
    runpy.run_path(str(phase_wrapper), run_name="__main__")


if __name__ == "__main__":
    main()
