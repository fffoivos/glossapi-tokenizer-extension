# 04 — CPT training regime on Vanilla Apertus-8B (Task 1 regime diagnostic)

> **In one line:** a 5 B-token continued-pretraining run of *unmodified* Apertus-8B on Greek, built to test whether the four-arm bakeoff's CPT **regime** (not the tokenizer extension) caused the native-Greek MCQ degradation — it did, and the corrected Apertus-faithful regime plus a late Path-A geometry probe became the settings Task 2 inherited.
> **Period:** 2026-05-28 → 2026-06-11 (experimental work 2026-05-28 → 2026-06-01; commits `37888147` 2026-06-01, `a19c136f` 2026-06-11). **Status:** completed and closed; no further runs from this directory.
> **Came from / led to:** [`../03_apertus_extension_and_embedding_adaptation`](../03_apertus_extension_and_embedding_adaptation) (the four-arm bakeoff whose Vanilla arm degraded) → this → [`../05_token_distillation_cpt`](../05_token_distillation_cpt) (Task 2).

## Why this existed

The bakeoff had left an ambiguity: every arm — including the Vanilla arm with no tokenizer change at all — lost native-Greek MCQ accuracy against Apertus-Base. Either Greek CPT itself is harmful, or the bakeoff's *training regime* was wrong. Task 1 removed every confound except the regime: base 131,072-token tokenizer, no extension, HPLT-only Greek, and the Apertus paper's own continuation settings (Goldfish loss `k=h=50`, LR `1.1e-5` with a 1.2 B-token warmup then constant, AdEMAMix `β3=0.99`) instead of the bakeoff's (NTP, `1.5e-5`, trapezoid, `β3=0.9999`). Everything else — including the bakeoff's Path-B positional geometry (`rope_theta=500K`, `max_position=4096`) — was held fixed so that this run and the bakeoff arms stay directly comparable. Scope and stop conditions are in [`goal/goal.md`](goal/goal.md); the full plan is [`_archive/superseded_drafts/task1_20260601/cpt-plan.md`](_archive/superseded_drafts/task1_20260601/cpt-plan.md) §2.

## History

### 2026-05-28 — smoke, dataset, and two false starts on the training chain

The day opened with an end-to-end smoke: dataset build on the CPU-only `xfer` partition (jobs `2415392` and `2415400` both died before doing work because `uenv` is not available on `xfer`; `2415472` passed in 2m42s under a `uv` Python env), a 2-iteration Goldfish training smoke on `debug` (`2415525`, 6m32s), Megatron→HF conversion (`2415569`), and native-MCQ + BPB eval smokes ([`RUN_LOG_20260528.md`](RUN_LOG_20260528.md) §Jobs → §Smoke Result). The real dataset job `2415688` then produced 5,000,000,250 tokens / 3,739,911 rows / 22.94 GB in 178.8 min at the intended 70/24/4/2 Greek/replay/code/math mix, `validation ok:true`.

The training chain took three submissions to stand up. Segment 1 (`2417278`, target iter 119) failed instantly on Megatron's `OptimizerParamScheduler` assertion `lr_warmup_steps < lr_decay_steps` — the segment was shorter than the 287-step warmup. Re-chained to iter 357 (`2417297`), then cancelled again once real timing showed ~131 s/iter would blow the 12 h `normal` cap. The surviving chain is `2417446` → `2417453` with segment 1 targeting iter 300 and `SAVE_INTERVAL=119` so the iter-119/238 report checkpoints still land inside it. iter 119 appeared at 19:17 UTC; its first sidecar conversion (`2419080`) failed in 9 s because the Clariden mirror was missing `_train_config_common.env`, which was synced and the chain resubmitted as `2419108`–`2419114`.

The first adversarial review (`Vanilla-0.5B`, 20:38–20:49 UTC) was the only one actually run by `codex exec` / `gpt-5.5`, and it produced the finding that shaped the rest of the run: the checkpoint's positional geometry (`rope_theta=500K`, `max_position=4096`) does not match the released Apertus-Base (`rope_theta=12M`, `max_position=65536`, llama3 scaling). That was written up the same evening as [`reports/config_geometry_audit_iter_0000119.md`](reports/config_geometry_audit_iter_0000119.md).

### 2026-05-29 — the `--export` comma bug, the matched-config dead end, and first statistical support

Codex went offline overnight; from `2026-05-29T00:24Z` every remaining review ran as a Claude Code subagent with the same prompt template, recorded as `BACKEND="claude-code-subagent"` in each `review_metadata.env` (RUN_LOG §"Codex Handoff to Claude Code"). The first subagent review (`Vanilla-1B`) immediately found a **critical silent data bug**: `submit_checkpoint_sidecars.sh:156` passed `BENCHMARKS="$NATIVE_BENCHMARKS"` inside `--export=ALL,…`, Slurm split on the commas, and the iter-238 and iter-477 native-MCQ jobs evaluated GreekMMLU only while still reporting a "3-task headline policy". iter 119 escaped only because it had been resubmitted by hand. The fix hoisted `BENCHMARKS=` onto the sbatch shell line (`7eb4667e…` → `e865c65a…`), and iters 238/477 were re-evaluated as jobs `2422769` / `2422770` (RUN_LOG §Decision A/B; [`reports/script_audit_20260529.md`](reports/script_audit_20260529.md) C1).

Decisions A–G were logged the same morning and then cross-checked by three parallel investigators plus a synthesis agent into the 24-row [`reports/decisions_matrix_20260529.md`](reports/decisions_matrix_20260529.md) (rows A–X) and the 3-critical / 7-major / 9-minor [`reports/script_audit_20260529.md`](reports/script_audit_20260529.md).

The attempt to remove the geometry confound failed and was recorded as such. A "matched-config" Apertus-Base — the released Path-A weights re-served with `rope_theta=500K`, `max_position=4096` (jobs `2422890`/`2422891`/`2422892`) — scored 0.4272 on the 3-task headline and **1.2216 Greek BPB** against ~0.43 for the CPT checkpoints. The Vanilla-2B critique read this correctly: the override *perturbs* the base rather than re-anchoring it, so it is a diagnostic bookend, never a baseline (RUN_LOG §"Vanilla-2B Critique Returned"; [`reports/5B_REPORT.md`](reports/5B_REPORT.md) §10.4). The docs were corrected the same day — `cpt-plan.md` §2.1 and `goal/hyperparameters.json` had been carrying a mixed Path-A/Path-B description matching neither the base nor the training command — and the Path-A-for-Task-2 recommendation was written into `cpt-plan.md` §3.4 Q3.4.10.

V4 bootstrap CIs were re-emitted as **v2** and gave the first statistically clean result: iter 477 (post-warmup, 2 B) beat bakeoff-Vanilla-2B by **+4.65 pp, CI [+0.0299, +0.0629]**. The same entry extrapolated the post-warmup slope to a 5 B estimate — an extrapolation later marked as an error ([`_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md`](_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md) §2.4).

Late that night iter 834 landed **flat**: 0.4790 against iter 477's 0.4792. The Vanilla-3.5B critique confirmed the aggregate plateau (paired Δ = −0.0002, CI [−0.0123, +0.0114]) but showed it was task-level cancellation, not saturation — GreekMMLU +2.09 pp and ASEP −1.92 pp, both outside zero.

### 2026-05-30 — endpoint, and the report that had to be corrected twice

The chain finished at `2026-05-30T15:31Z`. iter 1192 came in at headline **0.4973**, Plutus **0.4356** (−5.33 pp) and Greek BPB **0.4132**. Four agents ran in parallel: the Vanilla-5B review, the V4 **v3** re-emit (10 models, 83 delta rows), the Plutus investigation, and the [`reports/5B_REPORT.md`](reports/5B_REPORT.md) synthesis. The plateau turned out transient — iter 1192 vs iter 834 = +1.84 pp with CI clear of zero — so the 3.5 B reviewer's plateau reading was superseded (TASK2_HANDOFF §2.5).

Two self-corrections landed before the day closed. The report's §7 retention table had used iter 119 as the "starting point"; iter 119 is 0.5 B tokens *into* training and has already paid the rope re-adaptation cost, so the table was rebuilt against the matched-config Path-B init — the true iter 0 — which changed the sign or size of several deltas (TASK2_HANDOFF §2.6/§2.17). And the folder was reorganized: the live draft went to `_archive/superseded_drafts/5B_REPORT_DRAFT.md`, five in-flight bootstrap workspaces to `_archive/v4_workspaces/`, and `TASK2_HANDOFF.md` was written as the reply to the planning agent.

### 2026-05-31 — the Path-A geometry probe (the run's most efficient result)

Rather than leave the geometry confound as a caveat, a 0.5 B probe was run under Path A ([`_archive/superseded_drafts/task1_20260601/PATH_A_GEOMETRY_PROBE_PLAN.md`](_archive/superseded_drafts/task1_20260601/PATH_A_GEOMETRY_PROBE_PLAN.md)). Three submissions failed first — `2437889` (missing `ARM=vanilla`), `2437893` (`--ntasks-per-node=1` → `WORLD_SIZE=1`), `2437896` (the *same* warmup-decay assertion Task 1 had already hit, fixed here more cleanly by setting a notional `TRAIN_TOKENS=1.5e9` with `EXIT_INTERVAL=119`). `2437909` then ran clean in 4h21m. Verdict **CONFIRMED**: Path A at 0.5 B scored 0.4942 [0.4747, 0.5133] — between Task 1's Path-B 3.5 B and 5 B marks at one tenth the compute — with every Task-1 rope-re-adaptation retention regression simply absent ([`reports/path_a_probe_results_20260531.md`](reports/path_a_probe_results_20260531.md)). `TASK2_HANDOFF.md` §3.1 flipped from "RECOMMENDED working position" to "CONFIRMED, LOCKED".

### 2026-06-01 → 2026-06-11 — handoff, one retraction, and closure

[`reports/PLANNING_AGENT_REPLY_20260601.md`](reports/PLANNING_AGENT_REPLY_20260601.md) supplemented the handoff with the probe verdict and the three new operational errors. A same-day audit claimed the bakeoff's TD `target_layer=11` had been picked heuristically; a retrospective dig **retracted that** — both candidates (`-1` and `11`) were trained at two scales in the same job and layer 11 won on heldout BPB (0.008 pilot / 0.040 full) plus new-token recall ([`reports/TD_LAYER_11_SELECTION_PROVENANCE_20260601.md`](reports/TD_LAYER_11_SELECTION_PROVENANCE_20260601.md)). Everything to this point was committed as `37888147`. On 2026-06-11, commit `a19c136f` closed the directory: `cpt-plan.md`, `TASK2_HANDOFF.md`, `PATH_A_GEOMETRY_PROBE_PLAN.md` and a snapshot of `goal/` moved under `_archive/superseded_drafts/task1_20260601/`, the README was repointed at the archive, and `scripts/watch_and_submit_checkpoint_sidecars.sbatch` gained a `CHECKPOINTS_FILE` override "so other runs (e.g. the 13.5B 2-arm experiment) feed their own cadence" — Task 2 reusing Task 1's tooling.

## Outcome

- **Regime hypothesis supported.** iter 1192 (5.0 B) headline 3-task native Greek MCQ = **0.4973, 95 % CI [0.4779, 0.5156]** — verified directly against [`reports/v4_bootstrap_cis_native_mcq.json`](reports/v4_bootstrap_cis_native_mcq.json) (`models["iter-1192-Vanilla-5B"].headline_3task` = 0.49733 [0.47789, 0.51560]), confirming the old README's figure. Paired deltas, all outside zero: **+6.69 pp** [+0.0513, +0.0830] vs bakeoff-Vanilla-5B, **+7.01 pp** [+0.0537, +0.0857] vs the matched-config Path-B init, **+1.56 pp** [+0.0016, +0.0284] vs Apertus-Base Path A (barely), **+1.82 pp** vs iter 477, **+1.84 pp** vs iter 834.
- **Trajectory shape, not just level.** warmup → +3.05 pp burst at iter 477 → flat 1.5 B segment (iter 834 − iter 477 = −0.0002, CI straddles zero) → +1.84 pp endpoint lift. That is not the bakeoff's "peak early then drift" ([`reports/5B_REPORT.md`](reports/5B_REPORT.md) §4).
- **Greek BPB monotone post-warmup** — 0.4313 → 0.4197 → 0.4132 at iter 477/834/1192, and monotone in each of the four per-source registers, so the gain is not one register carrying the rest; code and math BPB also fall monotonically (5B report §8).
- **Path A confirmed and locked for Task 2**: +5.51 pp [+0.0379, +0.0725] over Path B at matched 0.5 B tokens, +1.25 pp over the released base, for 23.6 GPU-h ([`reports/path_a_probe_results_20260531.md`](reports/path_a_probe_results_20260531.md)).
- **Compute:** 217.2 GPU-h for the 5 B run (TASK2_HANDOFF §1) plus 23.6 GPU-h for the probe; ~45 h wall from training start to iter 1192; ~8 050 tokens/sec/GPU on 1 GH200 node (4 GPUs, TP=2). Note that [`reports/gpu_hours_breakdown_20260530.md`](reports/gpu_hours_breakdown_20260530.md) is a **mid-run snapshot** (145.85 billed GPU-h, ≈204.9 projected), not the 217.2 final — `reports/README.md`'s old "final total 217.2" attribution to that file was wrong and is corrected in the rewritten index.
- **Left open at the end** (5B report §13, all deferred, none gating): decontamination MinHash of the training pool against the four Greek MCQ prompt sets; `non_truncated_subset_bpb` over the 354 untruncated heldout docs; per-subject GreekMMLU/ASEP/Plutus breakdowns; a KL-to-base probe for the Plutus drop; the `xfer`→`normal` re-route of the watcher/checksum sidecars (Decisions Matrix row I); the 10 B stretch, deliberately left as an unforced choice.
- **Carried into [`../05_token_distillation_cpt`](../05_token_distillation_cpt):** the Path-A geometry (`scripts/train_config_td_path_a.env` there sets `ROTARY_BASE=12000000`, `USE_ROPE_SCALING=1`), the 1.2 B-token warmup constant, the sidecar-eval + checksum-manifest + adversarial-review pattern (`scripts/run_td_checkpoint_adversarial_review.sh` there still references this directory), and the generalized checkpoint watcher. Task 2 later re-derived its own LR and replay mix by sweep (see that subproject's `LOG.md`, 2026-07-11), so the regime here is its starting point, not its final setting.

## Sub-subprojects

| Dir | Role | Period | Status | Result (one line) |
|---|---|---|---|---|
| [`goal/`](goal) | Locked spec: scope, authoritative hyperparameters, canonical eval-task lockdown | 2026-05-28 → 2026-05-30 | frozen | The settings every other artifact cites; `canonical_eval_tasks.json` is the fix for the 201-vs-12 retention-task sprawl. |
| [`scripts/`](scripts) | Dataset build, training chain, sidecar fan-out, verifiers, status renderer, Path-A probe | 2026-05-28 → 2026-06-11 | complete, reused by 05 | 25 files; carries the `--export` comma fix and the env-var-driven Path-A config. |
| [`reports/`](reports) | Every result: 5 B report, decisions matrix, bootstrap CIs, audits, plots | 2026-05-28 → 2026-06-01 | complete | `5B_REPORT.md` + `v4_bootstrap_cis_native_mcq.json` are load-bearing for every number quoted above. |
| [`adversarial_reviews/`](adversarial_reviews) | One critique per checkpoint (0.5B → 5B) plus the Path-A probe | 2026-05-28 → 2026-05-31 | complete | Found the comma bug, the geometry confound and the matched-config perturbation; backend switched codex → Claude subagent mid-run. |
| [`monitor_logs/`](monitor_logs) | Watcher-side handoff timestamps | 2026-05-28 → 2026-05-30 | historical | Five one-line files recording when each checkpoint's sidecars went `handoff_ready`. |
| [`_archive/`](_archive) | Superseded plan/handoff/draft docs and bootstrap scratch workspaces | archived 2026-05-30 and 2026-06-11 | historical | Holds `cpt-plan.md` and `TASK2_HANDOFF.md`, which are still the best narrative sources. |

## Where things are

| Path | What it is |
|---|---|
| [`reports/5B_REPORT.md`](reports/5B_REPORT.md) | The endpoint report — 14 sections, results + caveats + Task-2 implications. |
| [`reports/v4_bootstrap_cis_native_mcq.json`](reports/v4_bootstrap_cis_native_mcq.json) | V4 v3: 10 models, 83 paired/marginal delta rows, 1000 resamples, `rng_seed=20260529`. Authoritative for every cross-arm claim. |
| [`reports/path_a_probe_results_20260531.md`](reports/path_a_probe_results_20260531.md) | The Path-A verdict that set Task 2's geometry. |
| [`reports/decisions_matrix_20260529.md`](reports/decisions_matrix_20260529.md) | Rows A–X: issue, severity, plan ref, recommendation, action, status. |
| [`goal/hyperparameters.json`](goal/hyperparameters.json) | Machine-readable settings; Path A in `base_model.architecture`, Path B in `training_geometry`, Path A again in `task2_geometry_recommendation`. |
| [`RUN_LOG_20260528.md`](RUN_LOG_20260528.md) | 2 350-line append-only narrative of the whole run — every job ID, failure and decision in order. |
| [`_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md`](_archive/superseded_drafts/task1_20260601/TASK2_HANDOFF.md) | What Task 1 established, 17 numbered errors and recoveries, 11 recommendations, 6 open questions. |
| Clariden (external) | Run dir `…/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z`, eval root `…/eval_04_vanilla_goldfish_5b_20260528T112539Z`, Megatron prefix `…/megatron/hplt_b1_base_text_document`, init checkpoint `…/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched` (5B report §14). |

## Working documents

Historical, kept for provenance — nothing here is current guidance.

- **Run narrative:** [`RUN_LOG_20260528.md`](RUN_LOG_20260528.md) — the single append-only log; every other doc is a summary of some slice of it.
- **Superseded plans and handoffs:** everything under [`_archive/superseded_drafts/task1_20260601/`](_archive/superseded_drafts/task1_20260601) (`cpt-plan.md`, `TASK2_HANDOFF.md`, `PATH_A_GEOMETRY_PROBE_PLAN.md`, a frozen copy of `goal/`) — moved out of the root by `a19c136f`; the archived `goal/goal.md` differs from the live one only in link targets.
- **Superseded drafts:** [`_archive/superseded_drafts/5B_REPORT_DRAFT.md`](_archive/superseded_drafts/5B_REPORT_DRAFT.md) — the live draft that `reports/5B_REPORT.md` replaced.
- **Status snapshots:** [`reports/latest_5b_report_status.md`](reports/latest_5b_report_status.md) and `reports/latest_5b_report_state.json` — the renderer's last output, frozen at iter 300 / 2026-05-30T16:03Z, i.e. *not* the final state.
- **Machine state files:** `reports/iter_*_checkpoint_sidecar_*.json`, `monitor_logs/*/iter_*.handoff_done`, `adversarial_reviews/*_watch_state/*` — watcher and verifier audit trail.
- **Scratch:** [`_archive/v4_workspaces/`](_archive/v4_workspaces) — the five bootstrap-computation workspaces behind the V4 v2/v3, iter-834, iter-1192 and Plutus numbers.
