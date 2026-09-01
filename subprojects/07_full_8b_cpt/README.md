# 07 — Full Apertus-8B mixed CPT

> **In one line:** the production Greek continued-pretraining run of Apertus-8B — one stationary-mix D0/WSD-10 trajectory that was launched once, stopped after 30B tokens for a missing PII pass, rebuilt on a sanitized corpus, and completed on 2026-08-11 at 76.685B active tokens.
> **Period:** 2026-08-05 → 2026-08-11 (run); analyses to 2026-08-12; one retrospective plan added 2026-09-01. **Status:** completed — the run finished, its evidence set is closed, and post-hoc conclusions moved to subproject 09.
> **Came from / led to:** [`06_dataset_scheduling_experiments`](../06_dataset_scheduling_experiments/) (D0 selected) → this → [`09_full_8b_cpt_results_analysis`](../09_full_8b_cpt_results_analysis/) (results authority), [`08_targeted_8b_cpt_experiments`](../08_targeted_8b_cpt_experiments/) (follow-ups) and [`10_early_cooldown_causal_experiment`](../10_early_cooldown_causal_experiment/).

## Why this existed

Subproject 06 had screened five data orders at 0.5B and produced an observed — not statistically resolved — leader. This subproject spent the 8B allocation on exactly one of them. It owns the recipe, the data sanitization, the Clariden orchestration, the checkpoints and the raw evaluation receipts; it deliberately does **not** own the post-hoc conclusions, which live in subproject 09. Everything here is receipt-gated: the machine authority is [`configs/recipe_8b_full_mixed.json`](configs/recipe_8b_full_mixed.json) (`recipe_id: full8b-mixed-79-20-1-wsd10-v1`) and the owner's own decisions are recorded as data in [`configs/owner_decisions_20260805.json`](configs/owner_decisions_20260805.json).

## History

### 2026-08-05 — the recipe, the owner's decisions, and a day of hardening

The recipe froze the 79/20/1 stream (Modern Greek / foreign source-family replay / Greek "old_greek" replay), the 148,992-token tokenizer, the verified **untied layer-11** Token-Distillation initialization, peak LR `5.5e-5` with a 400-update warmup and a 20% `1-sqrt` cooldown to the WSD-10 floor, AdEMAMix `(0.9, 0.999, 0.999, α=4)`, Goldfish `k=h=50`, TP=2, global batch 1,024 × 4,096, six 3,208-update segments on 16 nodes / 64 GH200 GPUs, 13 validation panels every 25 updates, 20 GreekMMLU checkpoints and per-document validation at three milestones. Planned horizon: 80,729,939,067 active tokens over 19,248 updates.

`configs/owner_decisions_20260805.json` (recorded `11:17:54Z`) is unusually explicit about what was *not* resolved: D0 was accepted on `explicit_point_estimate_acceptance` — before schedule-selection confidence intervals existed; `libduth` was included on `explicit_recorded_risk_acceptance` with `legal_conclusion_claimed: false`, i.e. the conflicting licence evidence was accepted as a risk, not reconciled; checkpoint averaging was excluded by explicit owner exclusion.

Nineteen more commits the same day hardened the launcher rather than the science: a validated Clariden memory ceiling (`e25d7bdc`), benchmark restart and Python-runtime hardening (`a313f7a9`), Megatron caches kept in mutable run state (`460d5bfc`), a dependency-closed `sitecustomize.py` restoring the historical `numpy.product` checkpoint alias under NumPy 2 (`300d6244`), a 75-minute GreekMMLU allocation replacing the inherited 30-minute Mini value after a measured 54m11s baseline (`6a54d25a`), and an 8B-specific export verifier (`a0ca3e88`) because the converter's simultaneous Megatron/HF `--test-logits` check exceeds one 96 GB GH200 — the gate instead requires all **323 learned source tensors** to map bit-exactly into the four HF shards. `8e9980b3` closed the prelaunch evidence gaps and made two provenance corrections in the README: the inherited `old_greek` panel was **rejected** because 5,784 of its 5,833 documents are exact training-content matches, and the 1% pool was relabelled *Greek replay retention* because its source is dominated by Modern-Greek HPLT and FineWiki rows and measures no Ancient-Greek capability.

### 2026-08-05/06 — the first trajectory, and why it was stopped

Production started on the pre-sanitization corpus and ran through segment 0 (attempt 0), segment 1 (attempt 2) and into segment 2, reaching **update 7,152 = 29,997,662,208 token slots**, 37.2% of the planned horizon. The training text was GreekMMLU-decontaminated but had never received the required Apertus PII anonymization pass, so the run was stopped for that reason. It was written up as an explicitly exploratory control, not a result: [`presentations/FULL8_EXPLORATORY_PREFIX_20260806.html`](presentations/FULL8_EXPLORATORY_PREFIX_20260806.html). Its findings — broad Greek learning including on the neutral panel, small but visible foreign-panel departures from their running minima, and a GreekMMLU curve that improves sharply then oscillates (35.28% at init → 56.58% at update 5,960 → 55.24% at 7,152, full 16,632-question set) — are what set the expectations for the rerun.

### 2026-08-06 — nested Slurm, disjoint panels, and an uncomfortable disclosure

Four commits made nested `sbatch` work under the pinned uenv (`381f4e3c`, `98d3bd27`, `f2f905f8`, `17eaa9dd`). `8deb1976` rebuilt every validation panel to be training-disjoint. `c3c84fdb` restricted segment recovery to verified clean checkpoints. And `14a61ca0` added [`evidence/DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md`](evidence/DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md), which states that the restart gradient-norm tolerances (`0.001` absolute / `0.02` relative) were added in commit `71d1bba2` **after** the DP32 restart result existed, withdraws the earlier "collective reduction order" explanation as unsupported, and downgrades the recovery claim from bitwise-exact to *numerically continuous*.

### 2026-08-07 — the sanitized restart

[`SANITIZED_RESTART_RUNBOOK_20260807.md`](SANITIZED_RESTART_RUNBOOK_20260807.md) and commit `5b6dd260` (plus ~18 follow-ups) rebuilt the data contract: PII masking with the Apertus-parity email/IP masker and a validated country-length IBAN masker; exclusion of exactly 6,648 `openarchives.gr` rows flagged `needs_ocr`; global exact deduplication of the masked text; and recomputation of the whole D0 schedule from the retained mass. The follow-ups read as a list of bugs found and closed:

- dedup keyed by `(doc_id, masked SHA-256)` with row multiplicity preserved, after v7 found 27 repeated IDs spanning 32 extra records in one shard — a set of document IDs would have dropped the wrong text (`317636c2`, `e80ad793`);
- Old-Greek replay capacity preserved: the original task-index-only ordering rule left only 11,529,074 Old-Greek tokens against a 2,666,110,500-token source and failed the 1% capacity gate (`2bc4cf40`);
- bridge accounting separated documents from index sentinels — summing index entries inflated the v1 bridge by exactly 1,457, one per task (`20fb7294`, `303ccf67`);
- replay heldouts reconstructed *through* the same masker before comparing hashes, since comparing raw source text to masked training text fails on any document containing an email, IP or IBAN (`5f4383f3`);
- retention alerts pre-registered on the seven fixed foreign/code/math/Greek-replay panels: +0.05 nats twice = warning, +0.08 on any panel or +0.05 on the macro = critical, explicitly as reporting constraints and not automatic stop rules (`ca729f0b`).

The same day produced two hard operational reversals. The v35 sanitized benchmark **invalidated the inherited async-save evidence**: two independent DP32 restarts matched each other exactly but both differed from the uninterrupted update-161 gradient norm (3.202 vs 2.210), so that benchmark was quarantined and asynchronous save was forbidden at resumable boundaries pending a synchronous-parity proof (`3d1c015b`). The v36 parity attempt was spread over six leaf switches and showed 8.7–72.0 second updates against the proven ≈8.74 s single-switch trajectory, so multi-node training was pinned to `--switches=1` plus an explicit leaf-group exclusion list (`85f2b755`, `c7652b1e`). Finally `701741f7` added a second, independently fail-closed path to select the already-proven DP32 profile without re-running DP64 at all.

### 2026-08-08 — the run that finished

The sanitized recipe derived a new horizon: **76,685,490,476 active tokens → 18,284 optimizer updates**; the runbook records that the earlier 19,248-update / 80.7B figure belongs to a superseded corpus geometry. The v45 scientific bundle was frozen (tree SHA-256 `fe6993bc…`, scientific digest `41998a04…`). DP64 was **rejected on trajectory drift despite being faster**; `dp32_16node` was selected with a 44.13-hour compute-only estimate at a median 8.6892 s/update. Five segments were planned: 0–4,000, 4,000–8,000, 8,000–12,000, 12,000–14,627 and 14,627–18,284. Run root: `/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12`. The launch and operations contract is [`FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md`](FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md).

### 2026-08-09 — running it against the queue

Segment 0 (job `3034915`) completed updates 0–4,000 in `10:02:48`. The rest of the day was allocation engineering, all of it receipt-bound and recorded in the handoff's operational addendum: `debug-qos` allows one running and two submitted jobs per user, so the evaluation chain had to be serial and no debug controller could wait on debug children; at most one audited, delayed, permit-gated successor holder may be prequeued, with the hold derived from the *target* segment's runtime and the trigger from the *source* segment's conservative budget. Supervisor bundles v29 → v30 → v31 each replaced a still-pending predecessor only after auditing the replacement and reconstructing one missing legacy receipt. Segment 1 (holder `3037145`) got its allocation at 06:20 and ran the 4,000–8,000 range at a stabilized median 8.614 s/update and 7,608 tokens/s/GPU with zero skipped or NaN updates. The same day the owner **deferred** review of the post-mask deduplication ([`DEFERRED_POSTMASK_DEDUP_REVIEW_20260809.md`](DEFERRED_POSTMASK_DEDUP_REVIEW_20260809.md)): the second global dedup dropped 2,378,595 exact duplicates plus 8,081 validation collisions and was outside the requested anonymization scope, but nothing about the running job was allowed to change because of it. A mid-run status report was published as [`presentations/FULL8_SANITIZED_RERUN_PROGRESS_20260809.html`](presentations/FULL8_SANITIZED_RERUN_PROGRESS_20260809.html), whose headline was that *learning replicates better than the benchmark*.

### 2026-08-11 — completion, then the drift analyses

The completion receipt was written at 10:49 Athens time and binds **19 GreekMMLU receipts, 39 document-local receipts, 5 training-attempt audits**, the launch gate, the selected DP32 profile and the terminal HF export. [`presentations/FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811.html`](presentations/FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811.html) is the canonical trajectory report and [`presentations/FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260811.html`](presentations/FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260811.html) the cross-scale comparison. Because GreekMMLU peaked mid-run and then fell back, a further investigation ran the same evening and into 08-12: [`analysis/GREEKMMLU_RESPONSE_DISPLACEMENT_BY_CATEGORY_20260811.md`](analysis/GREEKMMLU_RESPONSE_DISPLACEMENT_BY_CATEGORY_20260811.md) plus a family of drift simulations and stress tests in [`presentations/`](presentations/). Its finding: the dominant full-run effect is *early* response reorganization (strongest boundary at updates 3,576→4,768, ≈15–20B tokens, selected by 27 of 31 subject labels), while the post-peak window's strongest boundary is 14,627→15,496 — aligned with the declared cooldown start, but explicitly not proof that LR decay caused it.

### 2026-09-01 — one plan, not executed

[`RETROSPECTIVE_TOKEN_ACCURACY_PLAN_20260901.md`](RETROSPECTIVE_TOKEN_ACCURACY_PLAN_20260901.md) proposes reconstructing document-local top-1 next-token accuracy across all 19 checkpoints (never logged during training), costed at 27.25 node-hours on one debug node at a time. Status: ready to implement, **not submitted**.

## Outcome

- **The run completed:** 76,685,490,476 active tokens over 18,284 updates — 60.582B Modern Greek, 15.337B foreign replay, 0.767B Greek replay — with zero skipped and zero non-finite updates ([`../09_full_8b_cpt_results_analysis/RESULTS.md`](../09_full_8b_cpt_results_analysis/RESULTS.md), [`DATA_AND_LIMITATIONS.md`](../09_full_8b_cpt_results_analysis/DATA_AND_LIMITATIONS.md)).
- **Adaptation is broad:** all six Greek learning panels and all 13 exact per-document panels improved from initialization to the terminal checkpoint, and every per-document panel improved during cooldown.
- **Retention is imperfect:** English, code, math, German, Russian and Chinese all ended above their own earlier best BPB while remaining far better than at initialization.
- **GreekMMLU separates from loss:** 35.782% at initialization → **56.810% at update 9,536 (39.997B active tokens)** → 54.855% terminal, on the fixed 16,159-question decontaminated subset, while source-conditioned Greek BPB kept improving. The update-9,536 checkpoint was preserved as the observed leader.
- **Execution facts worth reusing:** DP64 is faster but drifts; async save is unproven at resumable boundaries; single-leaf-switch placement is worth up to an 8× per-update difference; `debug-qos` limits force serial evaluation chains.
- **Left open:** the deferred post-mask dedup review; the retrospective token-accuracy campaign; the fact that this is one trajectory and cannot isolate sanitation, dedup, replay share, scale, tokenizer, TD or LR shape as causes.
- **Carried forward:** subproject 09 owns the conclusions; 08 owns the follow-up designs; 10 owns the early-cooldown causal test motivated by the cooldown-aligned displacement boundary.

## Sub-subprojects

| Dir | Role | Period | Status | Result (one line) |
|---|---|---|---|---|
| [`configs/`](configs/) | Recipe, execution profiles, owner decisions, ETA | 08-05 → 08-07 | frozen | The machine authority for everything the run did. |
| [`dataset/`](dataset/) | Inventory freeze, packing reuse, disjoint validation | 08-05 → 08-07 | completed | Reused the 8B binaries without re-tokenizing; rebuilt every leaking panel. |
| [`dataset/anonymization/`](dataset/anonymization/) | PII masking, OCR exclusion, post-mask dedup, bridge | 08-07 | completed | The sanitized corpus the completed run consumed. |
| [`train/`](train/) | Checkpoint freezing and training-attempt auditing | 08-05 → 08-07 | completed | Five audited training attempts, all clean. |
| [`evaluation/`](evaluation/) | 13 panels, GreekMMLU, per-document scoring, exports | 08-05 → 08-09 | completed | 19 GreekMMLU + 39 document-local receipts, bit-exact HF exports. |
| [`scripts/`](scripts/) | Validators, launch gate, supervisors, campaign status | 08-05 → 08-09 | completed | The fail-closed control plane, including the DP32 selection. |
| [`clariden/`](clariden/) | 76 Slurm entry points | 08-05 → 08-09 | completed | Ran the data graph, the benchmarks, the five segments and the evidence chain. |
| [`runtime_compat/`](runtime_compat/) | One `sitecustomize.py` shim | 08-05 | frozen | Restores Megatron's `numpy.product` alias under NumPy 2. |
| [`evidence/`](evidence/) | One disclosure document | 08-06 | frozen | The post-hoc restart tolerance, disclosed rather than buried. |
| [`presentations/`](presentations/) | 11 reports + builders + payloads | 08-06 → 08-12 | completed | Exploratory prefix, mid-run progress, final results, cross-scale, drift. |
| [`analysis/`](analysis/) | GreekMMLU drift and source-exposure analysis | 08-11 → 08-12 | completed | AHWD: early reorganization dominates; post-peak boundary sits at cooldown. |
| [`tests/`](tests/) | 6 test modules over the orchestration and masking | 08-05 → 08-09 | completed | Every gate above has an executable test. |

## Where things are

| What | Path |
|---|---|
| Machine authority | [`configs/recipe_8b_full_mixed.json`](configs/recipe_8b_full_mixed.json) |
| Owner decisions | [`configs/owner_decisions_20260805.json`](configs/owner_decisions_20260805.json) |
| Sanitized data contract | [`SANITIZED_RESTART_RUNBOOK_20260807.md`](SANITIZED_RESTART_RUNBOOK_20260807.md) |
| Launch + operations contract | [`FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md`](FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md) |
| Final result | [`presentations/FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811.html`](presentations/FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811.html) |
| Conclusions | [`../09_full_8b_cpt_results_analysis/RESULTS.md`](../09_full_8b_cpt_results_analysis/RESULTS.md) |
| Resource/provenance map | [`../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md`](../05_token_distillation_cpt/CPT_LAUNCH_RESOURCE_SPEC_20260801.md) |
| Public training contract | [`eellak/greek-apertus` PR #1](https://github.com/eellak/greek-apertus/pull/1), merge commit `c1cb8510…` |
| Completed run root (CSCS) | `/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12` |
| Scientific bundle (CSCS) | `/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt/20260808T023300Z-sanitized-v45`, tree `fe6993bc…` |

## Working documents

- **Contracts, still accurate:** `SANITIZED_RESTART_RUNBOOK_20260807.md`, `FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md` (its own header notes that the 2026-08-09 addendum supersedes the historical "training had not started" paragraph).
- **Open decisions, deliberately parked:** `DEFERRED_POSTMASK_DEDUP_REVIEW_20260809.md` (deferred by the owner 2026-08-09), `RETROSPECTIVE_TOKEN_ACCURACY_PLAN_20260901.md` (never submitted).
- **Disclosure:** `evidence/DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md`.
- **Superseded by the sanitized rerun but kept as evidence:** `presentations/FULL8_EXPLORATORY_PREFIX_20260806.*` (the stopped pre-anonymization trajectory) and `presentations/FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260810.*` (superseded next day by the `20260811` build).
- **Exploratory analyses that changed no decision:** the GreekMMLU drift simulations, stress tests, history gallery and wrong-cell displacement bake-off under `presentations/`. Subproject 09 deliberately did not copy these forward.
- The historical README stated the run as "owner-authorized and receipt-gated" with a "Not yet done" list; that state ended on 2026-08-11. Some of its numbers (19,248 updates, 80.7B tokens, six segments) describe the **pre-sanitization** geometry and are superseded by 18,284 / 76.685B / five segments.

> The owner's main clone also holds copies of several files in this directory — including `configs/recipe_8b_full_mixed.json`, scripts and sbatch wrappers — that differ from the committed versions. They were **not** imported; the committed versions here are authoritative for this history.
