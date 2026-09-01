# runtime_compat — pinned-Megatron shim for PyTorch 2.9.1

> **In one line:** a `sitecustomize.py` that keeps the pinned SwissAI Megatron checkpoint format working under the newer PyTorch in the CSCS uenv.
> **Period:** committed 2026-08-16 (`de6d9b79`). **Status:** complete; loaded by the training and checkpoint-audit paths for the whole study.

## Why this existed

The scientific contract pins the Megatron fork (`c92402e`) and therefore the distributed-checkpoint (DCP) semantics that the 8B and 1.5B checkpoints must keep. The available uenv ships PyTorch 2.9.1, which drops `numpy.product` and — more importantly — discards the *dynamic* (non-dataclass) fields Megatron attaches to DCP metadata when `FileSystemWriter.finish()` writes it. Losing those fields would silently change checkpoint contents rather than fail loudly.

## What it does

[`sitecustomize.py`](sitecustomize.py) restores `numpy.product` and wraps `FileSystemWriter.finish` so that any attribute not declared on the metadata dataclass is preserved across the write. It is idempotent (it marks the wrapped function) and installs itself only if the import succeeds, so it is inert in environments that do not use DCP.

## Outcome

Adopted as the standard runtime shim for every job that reads or writes a checkpoint in this subproject; `scripts/check_dcp_metadata_compat.py` is the corresponding preflight check.
