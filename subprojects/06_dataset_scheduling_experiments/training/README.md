# 06 · training — the Megatron entrypoint, the schedule reader, and the LR selection

> **In one line:** the code that actually ran the five arms — a receipt-bound schedule reader over immutable packed rows, an exact-iteration checkpoint hook, and the 1,024-step smoke that reversed the planned peak learning rate.
> **Period:** 2026-08-02 → 2026-08-03; the reader and the hook were extended again 2026-08-13 for subprojects 08 and 10. **Status:** completed, then reused as a library.
> **Came from / led to:** [`../dataset/`](../dataset/) (the frozen schedules) → this → [`../production/`](../production/) (segment orchestration) and [`../evaluation/`](../evaluation/) (every checkpoint it emits).

## Why this existed

SwissAI Megatron had to consume a *schedule*, not a sampler. The arms share one set of packed payloads and differ only in the order their stable sequence IDs are read, so the dataloader had to be an adapter over a receipt-bound ID list — and it had to be impossible for an arm to repack a payload or alter a Goldfish mask. Separately, the campaign needed checkpoints at exact non-periodic iterations (warmup, hard transitions, matched controls, cooldown start, segment boundary), which Megatron's `--save-interval` alone cannot express.

## History

- **The reader.** [`scheduled_sequence_reader.py`](scheduled_sequence_reader.py) reads immutable packed rows through a receipt-bound sequence-ID schedule; [`scheduled_packed_dataset.py`](scheduled_packed_dataset.py) wraps it as a Megatron `Dataset`; [`smoke_scheduled_packed_dataset.py`](smoke_scheduled_packed_dataset.py) smokes two arms sharing one payload. [`pretrain_scheduled_gpt.py`](pretrain_scheduled_gpt.py) is the entrypoint that runs one frozen arm.
- **Exact checkpoints.** [`exact_checkpoint_hook.py`](exact_checkpoint_hook.py) installs a narrow trigger that calls Megatron's own asynchronous `torch_dist` save at the declared optimizer iteration from the frozen `MINI_SCHEDULE_SAVE_ITERATIONS` list. It does not implement a second checkpoint format, and it must not approximate a transition checkpoint by polling logs and saving one update late.
- **Goldfish gate.** [`audit_goldfish_added_token_uniformity.py`](audit_goldfish_added_token_uniformity.py) proves every added token ID has identical hash-table eligibility; it passed for all 17,920 added IDs at a 0.0199 drop rate (`../evidence/static_prelaunch_evidence_20260803.json`).
- **The LR reversal (2026-08-03).** The candidate peak `1.5e-4`… was not the first choice. Transferring the tested dimensionless fraction from the 8B recipe gave `6e-4 × 0.5 = 3e-4`. The 1,024-step smoke (2,147,483,648 tokens, real 800-step warmup leaving 224 steps at peak) passed finite-loss, gradient, checkpoint, added-token and catastrophic-regression checks at `3e-4` but **failed the predeclared retention-panel non-inferiority gate**. The identical smoke at the predeclared `1.5e-4` fallback passed everything, so `1.5e-4` peak / `1.5e-5` floor was frozen for every arm. [`evaluate_common_stability_smoke.py`](evaluate_common_stability_smoke.py) is the fail-closed evaluator and [`finalize_stability_lr_selection.py`](finalize_stability_lr_selection.py) freezes the candidate-first, fallback-only decision (selection receipt SHA-256 `39c74abd…`).
- **Prelaunch smoke.** [`evaluate_five_arm_prelaunch_smoke.py`](evaluate_five_arm_prelaunch_smoke.py) verifies the real-data five-arm initial-load and resume outputs.
- **Runtime patches.** [`runtime_patches/`](runtime_patches/) pins two Megatron patches against upstream `c92402e3…` (extra validation sets, exact evaluation iterations) plus their applier. Subproject 07 re-pins these same two patch hashes in its 8B recipe.
- **Later reuse (2026-08-13 →).** `8cee8894` and `77227e6a` (branch `agent/replay-reader-v1`) generalized the reader to immutable *multi-corpus* replay schedules; `7a2d7993` (branch `agent/early-cooldown-causal`) extended the checkpoint hook for the causal early-cooldown experiment in [`../../10_early_cooldown_causal_experiment/`](../../10_early_cooldown_causal_experiment/).

## Outcome

- The frozen common trajectory: start at `0.1 × peak`, 800-step warmup (preserving the 8B recipe's ≈1.678B warmup-token mass at Mini's 2,097,152-token batch), stable to 80%, `1-sqrt` cooldown to `0.1 × peak`; AdEMAMix `(0.9, 0.999, 0.999, α=4)`, weight decay 0.1, clip 0.1, Goldfish `k=h=50`, bf16 with fp32 main gradients.
- The `3e-4 → 1.5e-4` reversal is the single most consequential result produced by this directory, and it was produced by a *disposable* run: all five arms restart from the same frozen TD initialization, so the smoke chose nothing but the LR.
- The reader and the hook outlived the experiment; they are now shared infrastructure for subprojects 07, 08 and 10.

## Working documents

Nine scripts plus `runtime_patches/`. Contract: [`../INITIALIZATION_AND_TRAINING_DECISIONS.md`](../INITIALIZATION_AND_TRAINING_DECISIONS.md) §§3, 5. Tests: [`../tests/test_scheduled_sequence_reader.py`](../tests/test_scheduled_sequence_reader.py), [`../tests/test_exact_checkpoint_hook.py`](../tests/test_exact_checkpoint_hook.py), [`../tests/test_stability_smoke.py`](../tests/test_stability_smoke.py).
