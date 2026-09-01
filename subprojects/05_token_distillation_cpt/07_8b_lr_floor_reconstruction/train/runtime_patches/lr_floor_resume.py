#!/usr/bin/env python3
"""Apply the selected LR floor after optimizer-state restore.

Megatron stores ``min_lr`` both in the optimizer-param scheduler and in each
optimizer parameter group.  ``--override-opt_param-scheduler`` updates the
scheduler when a branch resumes, but loading the optimizer state restores the
old parameter-group value.  ``OptimizerParamScheduler.get_lr`` gives that
parameter-group value precedence, so every branch would otherwise continue
with the floor from the shared checkpoint.

This wrapper installs a narrow import hook that reapplies the selected floor to
both locations immediately before every LR calculation.  It then delegates to
the existing phase-relative data-index wrapper.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import os
from pathlib import Path
import runpy
import sys


def enforce_lr_floor(scheduler, param_group: dict, expected_min_lr: float) -> bool:
    """Set the branch floor in both scheduler and restored optimizer state."""
    changed = scheduler.min_lr != expected_min_lr or param_group.get("min_lr") != expected_min_lr
    scheduler.min_lr = expected_min_lr
    param_group["min_lr"] = expected_min_lr
    return changed


def _required_lr_floor() -> float:
    raw = os.environ.get("CPT_MIN_LR_OVERRIDE")
    if raw is None:
        raise RuntimeError("CPT_MIN_LR_OVERRIDE must be set")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("CPT_MIN_LR_OVERRIDE must be a finite positive float")
    return value


def install_lr_floor_override() -> None:
    expected_min_lr = _required_lr_floor()
    target = "megatron.core.optimizer_param_scheduler"

    def patch_module(module) -> None:
        scheduler_class = getattr(module, "OptimizerParamScheduler", None)
        if scheduler_class is None:
            raise RuntimeError(f"{target} has no OptimizerParamScheduler")
        original = scheduler_class.get_lr
        if getattr(original, "_lr_floor_resume_guard", False):
            return
        state = {"reported": False}

        def patched(self, param_group):
            changed = enforce_lr_floor(self, param_group, expected_min_lr)
            if changed and not state["reported"] and os.environ.get("RANK", "0") == "0":
                print(
                    "[lr_floor_resume] restored optimizer param-group min_lr "
                    f"overridden to {expected_min_lr:.17g}",
                    file=sys.stderr,
                    flush=True,
                )
                state["reported"] = True
            return original(self, param_group)

        patched._lr_floor_resume_guard = True
        scheduler_class.get_lr = patched

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
        f"[lr_floor_resume] lazy import hook installed for min_lr={expected_min_lr:.17g}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    install_lr_floor_override()
    if len(sys.argv) < 3:
        raise SystemExit("usage: lr_floor_resume.py TE_GUARD PRETRAIN_GPT ...")
    source_repo = os.environ.get("LR13_SOURCE_REPO")
    if not source_repo:
        raise RuntimeError("LR13_SOURCE_REPO must be set")
    phase_wrapper = (
        Path(source_repo)
        / "subprojects/05_token_distillation_cpt/06_25b_midtraining_probe"
        / "train/runtime_patches/phase_relative_data_index.py"
    )
    if not phase_wrapper.is_file():
        raise RuntimeError(f"phase-relative wrapper is missing: {phase_wrapper}")
    sys.argv = [str(phase_wrapper), *sys.argv[1:]]
    runpy.run_path(str(phase_wrapper), run_name="__main__")


if __name__ == "__main__":
    main()
