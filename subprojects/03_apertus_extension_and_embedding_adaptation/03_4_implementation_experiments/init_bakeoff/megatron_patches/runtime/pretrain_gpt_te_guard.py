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


def install_numpy_product_alias() -> None:
    import numpy as np

    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]


def set_cuda_device_from_local_rank() -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
    except Exception as exc:
        rank = os.environ.get("RANK", "?")
        print(
            "[pretrain_gpt_te_guard] "
            f"rank={rank} could not set cuda device from LOCAL_RANK={local_rank}: {exc!r}",
            file=sys.stderr,
            flush=True,
        )


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


def _optimizer_param_refs(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj is None or obj_id in seen:
        return set(), set(), set()
    seen.add(obj_id)

    direct_ids = set()
    model_ids = set()
    main_ids = set()

    param_groups = getattr(obj, "param_groups", None)
    if param_groups is not None:
        for group in param_groups:
            for param in group.get("params", []):
                direct_ids.add(id(param))
                main_param = getattr(param, "main_param", None)
                if main_param is not None:
                    main_ids.add(id(main_param))

    # Megatron's distributed optimizer rewrites optimizer.param_groups to hold
    # local fp32 shards/main params. The original model Parameters are tracked
    # separately here; direct object-id membership in param_groups is therefore
    # not enough to prove trainability.
    model_param_group_index_map = getattr(obj, "model_param_group_index_map", None)
    if model_param_group_index_map is not None:
        for param in model_param_group_index_map:
            model_ids.add(id(param))
            main_param = getattr(param, "main_param", None)
            if main_param is not None:
                main_ids.add(id(main_param))

    for attr in (
        "optimizer",
        "optimizers",
        "_optimizer",
        "_optimizers",
        "chained_optimizers",
        "base_optimizer",
    ):
        child = getattr(obj, attr, None)
        if isinstance(child, (list, tuple)):
            for item in child:
                child_direct, child_model, child_main = _optimizer_param_refs(item, seen)
                direct_ids.update(child_direct)
                model_ids.update(child_model)
                main_ids.update(child_main)
        elif child is not None:
            child_direct, child_model, child_main = _optimizer_param_refs(child, seen)
            direct_ids.update(child_direct)
            model_ids.update(child_model)
            main_ids.update(child_main)
    return direct_ids, model_ids, main_ids


def install_xielu_optimizer_audit() -> None:
    from megatron.training import training
    import torch

    original = training.setup_model_and_optimizer

    def setup_model_and_optimizer_with_xielu_audit(*args, **kwargs):  # type: ignore[no-untyped-def]
        model, optimizer, opt_param_scheduler = original(*args, **kwargs)

        named_params = []
        for chunk in model if isinstance(model, list) else [model]:
            for name, param in chunk.named_parameters():
                if "activation_func.alpha_" in name:
                    named_params.append((name, param))

        if named_params:
            frozen = [name for name, param in named_params if not param.requires_grad]
            if frozen:
                preview = ", ".join(frozen[:8])
                raise RuntimeError(f"xIELU alpha params are frozen: {preview}")

            direct_ids, model_ids, main_ids = _optimizer_param_refs(optimizer)

            def optimizer_covers(param):  # type: ignore[no-untyped-def]
                main_param = getattr(param, "main_param", None)
                return (
                    id(param) in direct_ids
                    or id(param) in model_ids
                    or (main_param is not None and id(main_param) in direct_ids)
                    or (main_param is not None and id(main_param) in main_ids)
                )

            local_covered = [optimizer_covers(param) for _, param in named_params]
            dp_covered = local_covered
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    from megatron.core import parallel_state

                    try:
                        dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
                    except AssertionError:
                        dp_group = parallel_state.get_data_parallel_group()
                    device = torch.device("cuda", torch.cuda.current_device())
                    coverage = torch.tensor(local_covered, dtype=torch.int32, device=device)
                    torch.distributed.all_reduce(
                        coverage, op=torch.distributed.ReduceOp.SUM, group=dp_group
                    )
                    dp_covered = [count > 0 for count in coverage.cpu().tolist()]
                except Exception as exc:
                    rank = os.environ.get("RANK", "?")
                    print(
                        "[pretrain_gpt_te_guard] "
                        f"rank={rank} xIELU DP ownership reduce failed: {exc!r}; "
                        "falling back to local ownership only",
                        file=sys.stderr,
                        flush=True,
                    )

            missing = [
                name for (name, _), is_covered in zip(named_params, dp_covered) if not is_covered
            ]
            rank = os.environ.get("RANK", "?")
            print(
                "[pretrain_gpt_te_guard] "
                f"rank={rank} xIELU optimizer audit: model_alpha_params={len(named_params)} "
                f"local_optimizer_alpha_params={sum(local_covered)} "
                f"dp_optimizer_alpha_params={len(named_params) - len(missing)} "
                f"missing={len(missing)} "
                f"direct_refs={len(direct_ids)} model_refs={len(model_ids)} main_refs={len(main_ids)}",
                file=sys.stderr,
                flush=True,
            )
            if missing:
                preview = ", ".join(missing[:8])
                raise RuntimeError(f"xIELU alpha params missing from optimizer: {preview}")

        return model, optimizer, opt_param_scheduler

    training.setup_model_and_optimizer = setup_model_and_optimizer_with_xielu_audit


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: pretrain_gpt_te_guard.py /path/to/pretrain_gpt.py [args...]")
    target = sys.argv[1]
    set_cuda_device_from_local_rank()
    install_numpy_product_alias()
    install_te_empty_extra_state_guard()
    install_xielu_optimizer_audit()
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
