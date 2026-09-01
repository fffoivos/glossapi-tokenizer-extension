# 07 · runtime_compat — one compatibility shim

> **In one line:** a `sitecustomize.py` that lets the pinned Megatron checkpoint code run under PyTorch 2.9.1 / NumPy 2 without touching any tensor or optimizer math.
> **Period:** 2026-08-05 (commit `300d6244`, "Restore pinned Megatron checkpoint compatibility"). **Status:** frozen; used by every 8B training and evaluation job.

## Why this existed

The scientific runtime is pinned to Swiss-AI Megatron `c92402e3…` under the `pytorch/v2.9.1:v2` uenv. NumPy 2 removed `numpy.product`, which Megatron's checkpoint-validation path still calls, and the newer stack also loses some of the dynamic distributed-checkpoint metadata Megatron expects. Rather than patch the pinned scientific tree — which would break its receipt — the fix is a dependency-closed `sitecustomize.py` that Python imports automatically.

## What it does, and what it must not do

[`sitecustomize.py`](sitecustomize.py) restores the historical `numpy.product` alias and preserves Megatron's dynamic distributed-checkpoint metadata. The recipe's own description is the constraint: this affects **checkpoint format compatibility only, not tensors or optimizer math**. Anything beyond that would be a silent change to the scientific runtime and belongs in the hash-pinned patch set instead ([`../configs/recipe_8b_full_mixed.json`](../configs/recipe_8b_full_mixed.json) → `software.megatron_runtime_patchset`, and [`../../06_dataset_scheduling_experiments/training/runtime_patches/`](../../06_dataset_scheduling_experiments/training/runtime_patches/)).

## Outcome

One file, no follow-up commits — it worked and was never revisited. The runtime it protects is revalidated before every job by [`../scripts/validate_megatron_runtime.py`](../scripts/validate_megatron_runtime.py) against the frozen receipt `20260803T093500Z-megatron-production-c92402e-v1.receipt.json` (SHA-256 `99b9ecbd…`).
