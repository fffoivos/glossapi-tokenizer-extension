# 06 — Dataset scheduling experiments (0.5B five-arm data-order screen)

> **In one line:** five 0.5B CPT trajectories that differed only in the temporal order of HPLT vs. GlossAPI/non-HPLT Greek; stationary mixing (`D0`) was the observed leader and was carried into the full 8B run, but no formal winner was ever declared.
> **Period:** 2026-08-01 → 2026-08-05 (science); shared tooling was touched again 2026-08-13 → 2026-08-18 by later subprojects. **Status:** completed.
> **Came from / led to:** [`05_token_distillation_cpt`](../05_token_distillation_cpt/) (corpus, tokenizer, replay and resource map) → this → [`07_full_8b_cpt`](../07_full_8b_cpt/) (D0 at 8B).

## Why this existed

The Greek CPT corpus is dominated by HPLT web text (~69% of Modern Greek after exclusions), with the curated GlossAPI/non-HPLT material as the smaller, higher-quality remainder. Before spending an 8B allocation, the program wanted to know whether the *order* in which those two pools are consumed changes what the model ends up knowing and forgetting — a question raised by the curriculum/LR-decay literature ([arXiv:2511.18903](https://arxiv.org/html/2511.18903)). The design answer was a strict single-factor screen on `swiss-ai/Apertus-v1.1-0.5B`: identical documents, identical loss-active token multiset, identical replay positions, identical WSD-10 learning-rate trajectory, five different orderings. Full scientific authority is [`FACTORIAL_EXPERIMENT_DESIGN.md`](FACTORIAL_EXPERIMENT_DESIGN.md); the settings rationale is [`INITIALIZATION_AND_TRAINING_DECISIONS.md`](INITIALIZATION_AND_TRAINING_DECISIONS.md).

## History

The whole subproject was committed as one already-formed tree in `ba80bb0c` (2026-08-05); the internal chronology below therefore comes from the dated receipts under [`evidence/`](evidence/) and from the run receipts they cite, not from commit granularity.

### 2026-08-01 — pin the model, find the tokenizer that does not fit

`evidence/apertus_v1_1_0_5b_snapshot.json` pinned the base model at revision `1b7276176e...` (20 layers, hidden 1024, tied embeddings, RoPE base 500,000, max position 4096). `evidence/tokenizer_compatibility_audit_20260801.json` then blocked the obvious plan: Mini and the production Greek extension share the full 269,443-entry base merge prefix and the 17,920 appended merges are dependency-safe, but **14 base token IDs mean different things**, and Mini's config declares `pad_token_id=3` while ID 3 is `[INST]` and `<pad>` already sits at ID 10. Status was recorded as `requires_mini_base_overlay`. The resolution — keep Mini IDs `0..131071`, append the exact production IDs `131072..148991`, declare the existing ID 10 as pad, change no merge — is implemented in [`initialization/build_mini_tokenizer_overlay.py`](initialization/build_mini_tokenizer_overlay.py). `evidence/token_distillation_upstream_snapshot_20260801.json` pinned the canonical Token Distillation implementation at commit `35702b58...` and byte-checked the vendored copy.

### 2026-08-02 — build the data, prove the machinery

- **Packing and schedules.** Source-local packing ran as a 512-task array in 489 s (jobs `2982978`–`2982982`); the frozen schedule manifest `ffeaa694...` fixes 19,709,692 real sequences plus 260 loss-inactive filler sequences = 19,709,952 scheduled sequences = **38,496 optimizer updates and 80,729,939,067 active tokens per arm**, with zero terminal quota residual in every pool and arm. A first schedule (`schedules_source_local`) was superseded by v2 because it claimed per-window residual reporting without emitting the fields; the v2 sequence-ID arrays were byte-identical to v1 (`evidence/dataset_schedule_and_native_greekmmlu_plan_20260802.json`).
- **Checkpoint plan, twice.** `evidence/checkpoint_plan_rebind_required_20260802.json` recorded that fail-closed launch-policy edits had changed the experiment-matrix hash after the plan was frozen, so the plan was no longer launch-authorizing; `..._rebind_resolved_20260802.json` records the regenerated plan v7 with the cadence unchanged: **83 checkpoints per arm, 415 native-GreekMMLU evaluations**, segment boundary at update 19,456.
- **Evaluator path proven end to end.** A runtime smoke on an FVT model (job `2981227`) executed all three GreekMMLU metrics over 16,632 questions; the exact-checkpoint smoke then converted iteration 48 of a mock-data checkpoint through the canonical SwissAI chain with **99.99% prediction agreement and 99.88% close logits** and bound the result to source and HF tree hashes. Both receipts state explicitly that the metrics are not model results.
- **Neutral external panel.** A Greek Parliament Proceedings reserve (Zenodo 2587904, CC-BY-4.0) was clustered by sitting; a first cross-dedup run mis-clustered 512 sittings into one erroneous component. The corrected pass removed 106 minhash-verified training-overlapping clusters and froze 345 clusters / 15,057,527 tokens with zero surviving exact or verified matches (`evidence/neutral_external_panel_20260802.json`).
- **Runtime selected.** A non-submitting scheduler probe found 362 idle `normal` nodes and near-term starts for 20/40/80/160 nodes ([`evidence/clariden_capacity_probe_20260802.md`](evidence/clariden_capacity_probe_20260802.md)). B1 (single arm, job `2983591`) measured DP=16 at 1666.8 ms median / 17.82 projected compute hours; B2 (job `2983767`, all five arms on 20 nodes / 80 GPUs for 288 updates each, with a separate four-lane GreekMMLU service `2983768`) finished with zero skipped and zero NaN iterations, controlling arm D3 at 19.7912 compute hours → **19.9979 h end-to-end forecast**, under both the 24 h and 36 h targets. The ladder was stopped there; DP=32/64/128 were never run ([`RUNTIME_SCALING_36H_PLAN.md`](RUNTIME_SCALING_36H_PLAN.md)).

### 2026-08-03 — LR reversal, static gates, launch

The common 1,024-step stability smoke was run at the derived candidate peak `3e-4` (Mini pretraining `6e-4` × the tested CPT fraction 0.5). It passed finite-loss, gradient, checkpoint, added-token and catastrophic-regression checks but **failed the predeclared retention-panel non-inferiority gate**; the identical smoke at the predeclared `1.5e-4` fallback passed every gate, so `1.5e-4` peak / `1.5e-5` floor was frozen for all five arms (`INITIALIZATION_AND_TRAINING_DECISIONS.md` §3; selection receipt `39c74abd...`).

Static prelaunch evidence (job `2989033`) passed the schedule-only-factor audit, the Goldfish added-token uniformity gate (all 17,920 added IDs share the same 0.0199 drop rate) and froze the decontaminated GreekMMLU subset: 16,632 public items − 473 contaminated = **16,159 clean**. The retention runtime had to be reconstructed rather than reused, because the old lm-eval target directory had lost its sources: official `lm-eval==0.4.11` plus an explicit `global_mmlu` alias over the 15 language groups. That is recorded as *evaluator-reconstructed, not byte-identical* to previously reported retention values (`evidence/retention_runtime_reconstruction_20260803.json`).

`evidence/final_prelaunch_closure_20260803.json` then froze the campaign manifest (`85311d99...`), reported all **16 declared gates passed**, and closed with status `passed_dry_run_only` and zero GPU jobs. The owner authorized the live launch the same day: campaign `mini_cpt5_20260803T074854Z` was submitted at `2026-08-03T07:48:55Z` (jobs `2989297` initial validation, `2989298` five-arm training, `2989299` evaluation watcher, `2989300` supervisor). Initial validation completed in 19m20s with all 13 panels shared by the five arms (`evidence/live_campaign_submission_20260803.json`).

### 2026-08-03 → 08-05 — the run, its restarts, and an evaluation-dtype correction

All five arms reached update 38,496. The run was not a single clean allocation: the stitched validation trajectory records three authoritative attempts (attempt 0 through iteration 36,352, attempt 2 through 38,400, attempt 4 through 38,496 — `presentations/data/dataset_order_20260805/validation_trajectory.json`), consistent with the recovery machinery in [`production/`](production/) (`freeze_common_recovery_checkpoint.py`, `supervise_production_segment.py`, and the segment-1 recovery tests). Checkpoint evaluations were re-run under an authoritative **float32** namespace `fp32_v1` after a bfloat16-vs-float32 conversion-parity investigation (`clariden/diagnose_native_greekmmlu_dtype*.sbatch`, the five `evaluation/runtime_patches/*.patch` files, and `production/finalize_fp32_evaluation_recovery.py` / `finalize_exact_mapping_evaluation_recovery.py`). The final evidence set is complete: **415/415** GreekMMLU checkpoint evaluations and **5,395/5,395** validation rows (13 panels × 83 checkpoints × 5 arms).

### 2026-08-05 — selection, and the decision not to declare a winner

[`evaluation/analyze_dataset_order_selection.py`](evaluation/analyze_dataset_order_selection.py) applied the predeclared hierarchy plus a 5% source-retention safety screen. D2 and D4 failed the screen (historical/polytonic +13.1% and +7.2% vs D0). Among the three passing arms the lexicographic order was **D0 → D3 → D1**. The receipt nevertheless sets `winner_selected: false`, blocked on two honest gaps: the frozen validation outputs contain aggregate panel losses but no document-cluster rows for bootstrap intervals, and no numeric general-benchmark non-inferiority margin was frozen before endpoints were inspected. The result was published as [`presentations/DATA_ORDER_MIX_RESULTS_20260805.html`](presentations/DATA_ORDER_MIX_RESULTS_20260805.html).

### 2026-08-13 → 08-18 — later reuse of this subproject's code

The directory stopped being an experiment and became a library. `agent/replay-reader-v1` generalized [`training/scheduled_sequence_reader.py`](training/scheduled_sequence_reader.py) for immutable multi-corpus replay schedules (`8cee8894`, `77227e6a`); `agent/early-cooldown-causal` reused [`training/exact_checkpoint_hook.py`](training/exact_checkpoint_hook.py) for subproject 10 (`7a2d7993`); and the `agent/h2g-*` branches hardened the code-bundle tooling in [`production/`](production/) for subproject 08's H→G work (`c13dacf1`, `772cf450`, `e79f09ca`).

## Outcome

- **Order matters, and it trades off predictably.** D1 (hard HPLT→GlossAPI) finishes best on every curated GlossAPI family but pays +0.0473 BPB on HPLT and +0.0184 on neutral Greek versus D0; D2 (hard GlossAPI→HPLT) finishes best on HPLT but gives back +0.0823 BPB on aggregate non-HPLT and +0.0855 on historical polytonic (`DATA_ORDER_MIX_RESULTS_20260805.html`).
- **D0 is the observed all-round leader.** Neutral-external Greek BPB 0.43346, balanced HPLT/GlossAPI 0.45631, GlossAPI macro 0.48758; lowest observed replay-forgetting macro (0.09475 BPB, D3 adjacent at 0.09487); best final clean GreekMMLU choice NLL (1.2869) with paired 95% intervals excluding zero against every alternative.
- **But no arm won on accuracy.** Clean GreekMMLU accuracy: D3 42.37%, D0 42.13%, D4 42.00%, D1 41.96%, D2 41.47% (n = 16,159). D3's +0.24 pp lead has a paired interval of −0.22 to +0.71 pp. `winner_selected: false` stands.
- **Carried into 07:** D0 stationary mixing, the 148,992-token tokenizer, WSD-10, no checkpoint averaging, the 13-panel validation design, the GreekMMLU clean subset and the whole receipt-gated campaign pattern. The owner accepted D0 on **explicit point-estimate acceptance** rather than on resolved statistics ([`../07_full_8b_cpt/configs/owner_decisions_20260805.json`](../07_full_8b_cpt/configs/owner_decisions_20260805.json)).
- **Left open:** document-cluster BPB intervals (would require re-running endpoint validation with per-document numerators — the estimate and scorer for that live in [`../07_full_8b_cpt/configs/per_document_validation_estimate.json`](../07_full_8b_cpt/configs/per_document_validation_estimate.json)); the deferred 10/20/30% LR-floor study; the two-extra-seed confirmation of D0 vs D3 named in the analysis receipt.
- **Not established:** that 0.5B schedule rankings transfer to 8B. Subproject 09 later concluded they do not ([`../09_full_8b_cpt_results_analysis/RESULTS.md`](../09_full_8b_cpt_results_analysis/RESULTS.md) §3).

## Sub-subprojects

| Dir | Role | Period | Status | Result (one line) |
|---|---|---|---|---|
| [`initialization/`](initialization/) | Mini tokenizer overlay + tied Token-Distillation init | 08-01 → 08-02 | completed | One frozen tied-TD artifact used unchanged by all five arms. |
| [`dataset/`](dataset/) | Pool freeze, immutable packing, the five schedules | 08-02 | completed | 19,709,952 scheduled sequences / 38,496 updates, exact-once in every arm. |
| [`training/`](training/) | Megatron entrypoint, schedule reader, LR smoke | 08-02 → 08-03 | completed | `1.5e-4` selected after `3e-4` failed retention; reader later reused by 08/10. |
| [`evaluation/`](evaluation/) | 13 panels, GreekMMLU, retention runtime, selection | 08-02 → 08-05 | completed | 415 GreekMMLU + 5,395 validation bindings; D0 leader, no formal winner. |
| [`production/`](production/) | Launch gates, campaign manifest, supervisors, bundles | 08-02 → 08-18 | completed, then reused | 16/16 gates; recovery machinery carried the multi-attempt run. |
| [`clariden/`](clariden/) | 73 Slurm launchers for every stage | 08-02 → 08-05 | completed | The executable form of the pipeline above. |
| [`evidence/`](evidence/) | 15 dated prelaunch/gate receipts | 08-01 → 08-03 | frozen | The audit trail every claim in this README rests on. |
| [`presentations/`](presentations/) | Result reports (data-order + two LR reports) | 08-01 → 08-05 | completed | `DATA_ORDER_MIX_RESULTS_20260805.html` is the canonical 06 result. |
| [`configs/`](configs/) | `experiment_matrix.json` — the machine contract | 08-02 | frozen | Single source for arms, gates and planning arithmetic. |
| [`scripts/`](scripts/) | Matrix validator, capacity probe, runtime forecast | 08-02 | completed | Dependency-free local checks. |
| [`tests/`](tests/) | 27 unit tests over the contracts above | 08-02 → 08-16 | completed | Reject schedule leakage, coverage drift and planning-math drift. |

## Where things are

| What | Path |
|---|---|
| Scientific authority | [`FACTORIAL_EXPERIMENT_DESIGN.md`](FACTORIAL_EXPERIMENT_DESIGN.md) |
| Settings + gates rationale | [`INITIALIZATION_AND_TRAINING_DECISIONS.md`](INITIALIZATION_AND_TRAINING_DECISIONS.md) |
| Machine contract | [`configs/experiment_matrix.json`](configs/experiment_matrix.json) |
| Result | [`presentations/DATA_ORDER_MIX_RESULTS_20260805.html`](presentations/DATA_ORDER_MIX_RESULTS_20260805.html) + [`presentations/data/dataset_order_20260805/`](presentations/data/dataset_order_20260805/) |
| Selection receipt | `presentations/data/dataset_order_20260805/dataset_order_selection_analysis.json` |
| Launch closure | [`evidence/final_prelaunch_closure_20260803.json`](evidence/final_prelaunch_closure_20260803.json) |
| Live submission | [`evidence/live_campaign_submission_20260803.json`](evidence/live_campaign_submission_20260803.json) |
| Run root (CSCS) | `/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z` |
| Local checks | `python3 scripts/validate_experiment_matrix.py` and `python3 -m unittest discover -s tests` |

## Working documents

Everything below is historical; nothing was deleted.

- **Plans / designs (still accurate as design records):** `FACTORIAL_EXPERIMENT_DESIGN.md` (header still says "designed; GPU launch not authorized" — that state was superseded on 2026-08-03), `RUNTIME_SCALING_36H_PLAN.md`, `INITIALIZATION_AND_TRAINING_DECISIONS.md`.
- **Frozen receipts:** `evidence/*_20260801.json`, `evidence/*_20260802.json`, `evidence/*_20260803.json` — dated gate evidence, superseded only by later files in the same series (notably `checkpoint_plan_rebind_required` → `..._resolved`).
- **Reports carried in from adjacent work:** `presentations/LR_SCHEDULES_AS_RUN_AND_NEXT_20260801.html`, `presentations/LR_SCHEDULE_TAIL_EXPERIMENTS_20260801.pptx`(+`.inspect.ndjson`) and `presentations/LR_FLOOR_EXPERIMENT_RESULTS_20260802.html` describe the **8B** LR sweep and T10/T20/T30 floor study, not this 0.5B screen; they are kept here because they motivated the fixed WSD-10 choice.
- **Runtime patches:** `training/runtime_patches/*.patch` and `evaluation/runtime_patches/*.patch` are pinned against Megatron `c92402e3...`; the evaluation set records the bf16/fp32 conversion-parity work.

> The owner's main clone also holds copies of several files in this directory that differ from the committed versions. They were **not** imported; the committed versions here are authoritative for this history.
