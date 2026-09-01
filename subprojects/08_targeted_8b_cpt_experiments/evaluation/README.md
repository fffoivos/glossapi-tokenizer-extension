# evaluation — scorers, checkpoint export, and parity evidence

> **In one line:** the evaluation half of the study — three generations of GreekMMLU scorer (frozen subset → full public panel → pinned legacy BF16), the panel evaluator, and the checkpoint export path whose parity receipts bound how the results may be read.
> **Period:** 2026-08-16 → 2026-08-22 (`de6d9b79`, then `7f476bb9`, `134a5ffa`, `3817baa6`, `825e60be`, `3ace4a2d`). **Status:** complete — all 34 trajectory checkpoints, all 17 full-panel 8B checkpoints and the three legacy-BF16 checkpoints were scored.

## Why this existed

Two constraints shaped everything here. Checkpoints are Megatron distributed-checkpoint (DCP) trees, so every score requires an HF export first — and an export that is not provably weight-exact would silently move the result. And the benchmark question set changed twice during the study, so the scorers had to be able to run a frozen subset, the full public panel, and the historical evaluator *without* any of them being quietly substituted for another.

## History

| Date | What happened | Evidence |
| --- | --- | --- |
| 2026-08-16 | Initial scorers, sentinel builders/validators, export contract and the exact-weight-mapping verifier land with the squashed drop | `de6d9b79` |
| 2026-08-22 | `aggregate_cross_scale_greekmmlu_trajectory.py` added; the 34-checkpoint matched trajectory is aggregated with SHA-256 bindings to every result receipt | `7f476bb9` |
| 2026-08-22 | Five exports miss the stricter cross-runtime logit/prediction threshold. Rather than relabel them, a distinct receipt schema scopes them to the matched HF trajectory evaluator only; a second commit accounts for three incomplete optional diagnostics | `134a5ffa`, `3817baa6` |
| 2026-08-22 | Full-public panel freezing (`freeze_full_public_greekmmlu_examples.py`) and the legacy evaluator path (`run_legacy_public_greekmmlu_evaluator.py`, `finalize_legacy_bf16_matrix.py`) added for the corrected primary metric and the replication comparison | `825e60be`, `3ace4a2d` |

## What is here

- **GreekMMLU scorers:** [`run_greekmmlu_evaluator.py`](run_greekmmlu_evaluator.py) and [`score_frozen_greekmmlu_shard.py`](score_frozen_greekmmlu_shard.py) (sharded, restart-safe); [`run_legacy_public_greekmmlu_evaluator.py`](run_legacy_public_greekmmlu_evaluator.py) for the pinned `cfdd0e7b` BF16 path; calibration, fallback and plateau variants for the sentinel design that was ultimately not needed.
- **Panels:** [`freeze_greekmmlu_examples.py`](freeze_greekmmlu_examples.py) / [`freeze_full_public_greekmmlu_examples.py`](freeze_full_public_greekmmlu_examples.py) (16,159-question subset and 16,632-question public panel), [`build_greekmmlu_sentinels.py`](build_greekmmlu_sentinels.py) + [`validate_greekmmlu_sentinels.py`](validate_greekmmlu_sentinels.py), [`run_offline_panels_evaluator.py`](run_offline_panels_evaluator.py) for the 13 source-conditioned validation panels, and [`run_native_suite_evaluator.py`](run_native_suite_evaluator.py).
- **Checkpoint export and parity:** [`run_checkpoint_export_evaluator.py`](run_checkpoint_export_evaluator.py), [`checkpoint_export_contract.py`](checkpoint_export_contract.py), [`checkpoint_export_receipt.py`](checkpoint_export_receipt.py), [`verify_exact_checkpoint_weight_mapping.py`](verify_exact_checkpoint_weight_mapping.py). (The converter overlay and the float32 probability-parity runtime patch live only on the unmerged `agent/h2g-safe-open-verifier-20260817` branch, commit `4d019212`; the exports themselves were produced on Clariden with an approved overlay bound by `806c87a1`/`58eb4e9a`.)
- **Aggregation and finalization:** [`aggregate_cross_scale_greekmmlu_trajectory.py`](aggregate_cross_scale_greekmmlu_trajectory.py), [`aggregate_frozen_greekmmlu.py`](aggregate_frozen_greekmmlu.py), [`finalize_legacy_bf16_matrix.py`](finalize_legacy_bf16_matrix.py), [`finalize_legacy_public_greekmmlu.py`](finalize_legacy_public_greekmmlu.py), [`finalize_matched_study_evidence.py`](finalize_matched_study_evidence.py), [`finalize_trajectory_checkpoint_export.py`](finalize_trajectory_checkpoint_export.py).

## Outcome

- All 34 trajectory checkpoints passed exact tensor mapping (every source parameter covered, every mapped tensor bit-exact, every HF tensor accounted for). 29 of 34 also passed the stricter canonical export gate; the remaining five — 1.5B updates 238, 952, 1,428, 2,856 and 8B update 714 — are receipted as scoped to the common HF trajectory evaluator only, which is why very small checkpoint differences must not be over-read ([`../HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md`](../HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md) §5).
- Conversion-probe prediction agreement spanned 98.63–100.00%.
- The sentinel machinery (4,096/8,192-question nested panels) was built and gated but never became the decision path: the corrected primary metric is the complete 16,632-question public panel.
