# 06 · evidence — dated prelaunch and launch receipts

> **In one line:** the fifteen frozen receipts that carried the five-arm 0.5B screen from "pinned model" to "campaign submitted", each one a gate that had to pass before the next.
> **Period:** 2026-08-01 → 2026-08-03. **Status:** frozen; superseded only within the directory (see `checkpoint_plan_rebind_*`).
> **Came from / led to:** the gates declared in [`../FACTORIAL_EXPERIMENT_DESIGN.md`](../FACTORIAL_EXPERIMENT_DESIGN.md) §10 → these receipts → the campaign manifest and live launch.

## Why this existed

The campaign was fail-closed by construction: no arm could launch until a named gate had a hash-bound receipt on disk. This directory is the local copy of those receipts, so that a claim in [`../README.md`](../README.md) can be checked without CSCS access. Every file records what was *actually observed*, including the two receipts whose purpose is to say that a smoke result is **not** a scientific result.

## History

| Date | Receipt | Result / decision |
|---|---|---|
| 08-01 | `apertus_v1_1_0_5b_snapshot.json` | Pinned `swiss-ai/Apertus-v1.1-0.5B` at revision `1b727617…`: 20 layers, hidden 1024, tied embeddings, RoPE base 500,000, max position 4096, vocab 131,072, `pad_token_id=3` pointing at `[INST]`. |
| 08-01 | `tokenizer_compatibility_audit_20260801.json` | Status `requires_mini_base_overlay`: base merge prefix exact and the 17,920 appended merges dependency-safe, but 14 base IDs differ and the front end differs. Resolution: overlay, do not copy. |
| 08-01 | `token_distillation_upstream_snapshot_20260801.json` | Canonical TD pinned at `35702b58…`; three vendored files byte-checked; tied path confirmed to call `tie_weights()`. |
| 08-02 | [`clariden_capacity_probe_20260802.md`](clariden_capacity_probe_20260802.md) | 362 idle `normal` nodes; non-submitting `--test-only` predicted near-term starts for 20/40/80/160 nodes. Explicitly "an estimate, not a reservation". |
| 08-02 | `native_greekmmlu_runtime_smoke_20260802.json` | Job `2981227`: all three GreekMMLU metrics executed over 16,632 questions on an FVT model (accuracy 0.3703). Marked `runtime_and_metric_path_smoke_only`. |
| 08-02 | `exact_checkpoint_native_greekmmlu_smoke_20260802.json` | Iteration-48 mock-data checkpoint converted through the canonical SwissAI chain: **99.99% prediction agreement, 99.88% close logits**, full 16,632-question evaluation bound to source and HF tree hashes. `scientific_result: false`. |
| 08-02 | `native_greekmmlu_four_lane_wave_smoke_20260802.json` | Job `2983668`: four isolated one-GPU evaluation lanes on one node plus a fifth in the next wave, five exact receipts in 516 s. Proved the evaluation service can keep up with a five-arm checkpoint wave. |
| 08-02 | `neutral_external_panel_20260802.json` | Greek Parliament Proceedings reserve; a first pass mis-clustered 512 sittings into one component; the corrected pass removed 106 verified training-overlapping clusters and froze **345 clusters / 15,057,527 tokens**, zero surviving matches. |
| 08-02 | `dataset_schedule_and_native_greekmmlu_plan_20260802.json` | The consolidated data/runtime receipt: 512-task packing in 489 s; schedule v2 `ffeaa694…` with 19,709,692 + 260 sequences → 38,496 updates and 80,729,939,067 active tokens/arm; B1 DP16 job `2983591` at 1666.8 ms median; B2 five-arm job `2983767` → 19.9979 h forecast. `scientific_launch_authorized: false`. |
| 08-02 | `checkpoint_plan_rebind_required_20260802.json` | **Reversal:** fail-closed launch-policy edits changed the matrix hash after the checkpoint plan was frozen, so the plan stopped being launch-authorizing even though the scientific cadence had not changed. |
| 08-02 | `checkpoint_plan_rebind_resolved_20260802.json` | Plan v7 regenerated and re-bound; cadence unchanged at **83 checkpoints/arm, 415 GreekMMLU evaluations**; `scientific_cadence_changed: false`. |
| 08-03 | `static_prelaunch_evidence_20260803.json` | Job `2989033`: schedule-only-factor audit passed; Goldfish uniformity passed (all 17,920 added IDs share a 0.0199 drop rate); GreekMMLU clean subset frozen at 16,632 − 473 = **16,159**. |
| 08-03 | `retention_runtime_reconstruction_20260803.json` | Job `2989039`: the old lm-eval target had lost its sources, so `lm-eval==0.4.11` was rebuilt with an explicit `global_mmlu` alias over 15 language groups. Comparisons to previously reported retention numbers are **evaluator-reconstructed, not byte-identical**. |
| 08-03 | [`final_prelaunch_closure_20260803.json`](final_prelaunch_closure_20260803.json) | All **16 gates passed**; campaign manifest `85311d99…` frozen; status `passed_dry_run_only`, zero GPU jobs, run root not created. |
| 08-03 | [`live_campaign_submission_20260803.json`](live_campaign_submission_20260803.json) | Campaign `mini_cpt5_20260803T074854Z` submitted `07:48:55Z`: jobs `2989297`/`2989298`/`2989299`/`2989300`. Initial validation completed in 19m20s, 13 panels, shared by all five arms. |

## Outcome

- Every numeric claim in the parent README about the pre-launch phase traces to one of these files.
- Three of the fifteen exist specifically to prevent a false claim: the two `*_smoke_*` receipts (`scientific_result: false`) and the `rebind_required` receipt (a plan that was correct but no longer authorizing).
- What these receipts do **not** cover: the campaign's own restarts, the float32 evaluation re-run, and the endpoint analysis. Those live in the run root and in [`../presentations/data/dataset_order_20260805/`](../presentations/data/dataset_order_20260805/).

## Working documents

All fifteen files are historical receipts and are listed above. `checkpoint_plan_rebind_required_20260802.json` is superseded by `..._resolved_20260802.json`; nothing else in this directory supersedes anything else.
