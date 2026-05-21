#!/usr/bin/env python3
"""Run Megatron pretrain with a narrow Transformer Engine load guard.

The HF -> Megatron converted init checkpoints contain empty Transformer Engine
`_extra_state` tensors. Transformer Engine tries to unpickle those bytes during
`load_state_dict()` and raises `EOFError`. For this bf16 bakeoff those empty
extra states are not carrying FP8 runtime state, so ignore only the truly empty
case and still raise on non-empty corrupt state.
"""

import atexit
import os
import runpy
import sys


def install_te_empty_extra_state_guard() -> None:
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    original = TransformerEngineBaseModule.set_extra_state
    skipped_empty_extra_state = {"count": 0}

    def report_skipped_empty_extra_state() -> None:
        count = skipped_empty_extra_state["count"]
        if count:
            rank = os.environ.get("RANK", "?")
            print(
                "[pretrain_gpt_te_guard] "
                f"rank={rank} skipped {count} empty Transformer Engine _extra_state tensors",
                file=sys.stderr,
                flush=True,
            )

    def set_extra_state_allow_empty(self, state):  # type: ignore[no-untyped-def]
        try:
            return original(self, state)
        except EOFError:
            if getattr(state, "numel", lambda: None)() == 0:
                skipped_empty_extra_state["count"] += 1
                return None
            raise

    atexit.register(report_skipped_empty_extra_state)
    TransformerEngineBaseModule.set_extra_state = set_extra_state_allow_empty


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: pretrain_gpt_te_guard.py /path/to/pretrain_gpt.py [args...]")
    target = sys.argv[1]
    install_te_empty_extra_state_guard()
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
