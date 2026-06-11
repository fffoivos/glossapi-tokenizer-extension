#!/usr/bin/env python3
"""Two-phase curriculum gap-filler: reset the TRAIN dataloader's data index at the phase boundary
WITHOUT touching the optimizer/scheduler state.

Mechanism (verified on Megatron-LM-Swiss-AI @ c92402e):
  MegatronPretrainingSampler.__iter__ iterates `range(self.consumed_samples, total_samples)`, i.e.
  it SKIPS consumed_samples indices into the dataset. On a normal resume args.consumed_train_samples
  is restored from the checkpoint and forwarded to build_pretraining_data_loader. For a SWITCHED
  phase-2 binary that is WRONG — it would skip ~N1 samples into the (smaller) GlossAPI binary.
  --finetune would zero it but ALSO drops optim/rng/scheduler load (AdEMAMix + WSD num_steps), which
  we MUST keep. So when RESET_DATA_INDEX=1 we monkeypatch build_pretraining_data_loader so the FIRST
  (=train) loader is built with consumed_samples=0 against binary B, while consumed_train_samples
  (scheduler num_steps + AdEMAMix counters) stays at the restored phase-1 value -> continuous WSD,
  carried optimizer, fresh data index. Extra-valid loaders already pass consumed_samples=0
  (pretrain_gpt.build_extra_valid_iterators), so they're untouched.

Usage (wraps the existing TE guard so both apply):
  python reset_data_index_guard.py <pretrain_gpt_te_guard.py> <pretrain_gpt.py> <megatron args...>
Set RESET_DATA_INDEX=1 ONLY on the first phase-2 segment; default (unset/0) is a transparent passthrough.
"""
import importlib.abc
import importlib.machinery
import os
import sys
import runpy


def install_data_index_reset_guard():
    if os.environ.get("RESET_DATA_INDEX", "0") != "1":
        return
    target = "megatron.legacy.data.data_samplers"

    def patch_module(module):
        orig = getattr(module, "build_pretraining_data_loader", None)
        if orig is None or getattr(orig, "_reset_data_index_guard", False):
            return
        state = {"train_seen": False}

        def patched(dataset, consumed_samples):
            # The first build call in build_train_valid_test_data_loaders is the TRAIN loader.
            if not state["train_seen"]:
                state["train_seen"] = True
                rank = os.environ.get("RANK", "?")
                print(f"[reset_data_index_guard] rank={rank} train dataloader consumed_samples "
                      f"{consumed_samples} -> 0 (phase-2 binary restart; scheduler num_steps kept)",
                      file=sys.stderr, flush=True)
                return orig(dataset, 0)
            return orig(dataset, consumed_samples)

        patched._reset_data_index_guard = True
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
    print("[reset_data_index_guard] lazy import hook installed for data_samplers",
          file=sys.stderr, flush=True)


def main():
    install_data_index_reset_guard()
    target = sys.argv[1]          # the TE guard, which itself runpy's pretrain_gpt.py
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
