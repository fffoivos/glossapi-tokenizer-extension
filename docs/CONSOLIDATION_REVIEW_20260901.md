# Consolidation review — 2026-09-01

This note accompanied branch `deslop/history-consolidation-20260901`, which is now
`main`. It records what
was found, what the branch does, what the owner still has to decide, and the two
lists the README writers produced but did not act on: documents that could be
archived, and contradictions between documents. Nothing in this note has been
applied; the READMEs describe, they do not move or delete.

## What was found

- GitHub `origin/main` stopped at 2026-06-11 (`d06b1ac4`, subprojects 02–05). The local
  `main` continued to 2026-07-22 (`84f3067f`, 546 commits further) and was never pushed.
- Subprojects 06–10 existed only on `agent/*` branches, in two families that never met:
  `agent/full8b-production-launch` → `replay-reader-v1` / `early-cooldown-causal` /
  `full8-results-analysis` (06, 07, 09, 10; branched from the June `main`), and the
  `agent/h2g-*` family (06, 08; branched from the July `main`). Subproject 06 diverged
  between them.
- Seven worktrees held uncommitted work: the 2026-07-29 polytonic tokenizer decision and
  the 148,992-token ship bundle, `05/07_8b_lr_floor_reconstruction`, `09_1`, `09_3`,
  `08/publication`, all of `11_greek_posttraining` (139 files), the 2026-08-06 RoPE
  correction to `CURRENT_HYPERPARAMETERS.md`, and 06/07 result presentations.
- READMEs were agent status snapshots ("new agents should start at…", absolute
  `/home/foivos/...` links; 39 broken links), not histories. `05_token_distillation_cpt`
  — 646 commits — had no README at all.

## What the branch does

1. Merges every leaf branch into local `main` (ten branches). Four add/add conflicts:
   `05/05_training_dataset_bridge/scripts/build_binary_shard.py` three-way merged with
   both sides' additions kept; three `06/production|tests` code-bundle files resolved to
   the newer `h2g` versions. Recorded in `600148e6`.
2. Recovers the uncommitted files (`2aec4a66`, 422 files, provenance per source worktree
   in the commit message). Not imported: the three 18 MB duplicate candidate tokenizers,
   `*.env`/`*.err`/`*.out`, 119 copies of 06/07 files that differ from committed versions,
   the root-level `frozen_*` deployment bundles, and two diffs that did not apply
   (`run_native_greek_mcq_eval.py`, `SUBPROJECTS_OVERVIEW.md`).
3. Rewrites every subproject and sub-subproject README as a dated, evidence-cited history
   (`f5b19d0e`: 73 created, 56 rewritten, 7,811 lines), plus a new root README, a dated
   `docs/PROJECT_INDEX.md`, and historical banners on the old status docs.

Left unmerged on purpose: `agent/h2g-safe-open-verifier-20260817` (13 commits of
2026-08-17 that conflict with the later `h2g` line; described in subproject 08's README).

## Decisions — executed 2026-09-01/03 at the owner's request

The owner answered "push, default to your preferences for the rest, clear the
branches". What happened:

1. **Published.** The branch was pushed, `main` fast-forwarded onto it and pushed
   (`origin/main` = the consolidation). The `deslop/...` branch ref was then removed —
   `main` *is* the consolidation now.
2. **Renamed** `10_greek_posttraining` → `11_greek_posttraining` (`c99c144c`).
3. **Working-tree variants archived, worktrees cleared.** Every uncommitted working-tree
   state was committed and preserved as an annotated tag, pushed to origin:
   `archive/worktree-{main-clone,full8-results,full8b-launch,h2g-deploy,h2g-extension-gate,h2g-nopilot,h2g-resize}-20260901`.
   The 119 differing 06/07 copies live inside those snapshots. Excluded on purpose:
   `.codex_tmp/` and the two 18 MB duplicate candidate tokenizers (still untracked in
   the main clone).
4. **`agent/h2g-safe-open-verifier-20260817`** preserved as the pushed tag
   `archive/branch-h2g-safe-open-verifier-20260817` (its tip content is byte-identical
   in `main`), then deleted as a branch.
5. **Branches cleared.** All other local and remote branches were deleted — each was
   either an ancestor of `main` or had every file of its unique commits byte-identical
   in `main` (verified per file before deletion). End state: one branch (`main`),
   8 `archive/*` tags, one worktree (the main clone). Two former worktree directories
   remain on disk unregistered (~400 MB each of mostly gitignored artifacts):
   `~/Projects/.codex-worktrees/train-apertus-h2g-{deploy-49d04335,nopilot-20260816}` —
   left for the owner to delete after a glance.
6. **Archive moves: deferred** (as recommended). The candidate lists below stand.

The original decision framing is kept below for the record.

### The decisions as originally posed

1. **Publish or not.** Nothing has been pushed. Options: fast-forward `main` to this
   branch and push; or push the branch and open a PR; or keep it local.
2. ~~Rename `10_greek_posttraining`~~ — **done 2026-09-01**: renamed to `11_greek_posttraining`
   (it collided with `10_early_cooldown_causal_experiment`); its own documents still say "10".
3. **The 119 differing 06/07 working-tree copies** in `~/Projects/train-apertus-with-glossapi`
   (scripts, sbatch, `recipe_8b_full_mixed.json`, READMEs). The committed versions were
   treated as authoritative. Discard the copies, or diff and decide file by file.
4. **`agent/h2g-safe-open-verifier-20260817`**: merge by hand (2–3 content conflicts in
   `08/clariden/export_checkpoint_for_evaluation_debug.sbatch` and
   `08/tests/test_canonical_train_adapter.py`) or leave as a side branch.
5. **Archive moves.** The lists below are candidates only. Moving them changes links in
   older documents; doing it per subproject with a link check is the safe path.

## Verification performed

Twelve Opus writers produced the READMEs from the git chronology and the documents
themselves, citing a path or commit for every dated claim. Spot checks against the
cited artifacts: 04's headline 0.49733 [0.47789, 0.51560] (bootstrap JSON); 02.2's
113,184 / 86.35 % coverage (PMI summary); 03's 0.4204/0.4076, 0.4602/0.4872, 0.8994
(BAKEOFF_FINAL) and 0.4305/0.4109/0.4817 (native suite); 09's 56.81 / 35.78 / 54.85 /
76.685 B (RESULTS.md and data JSON); 08's 57.9004 / 59.9627 / 2.0623 / 0.6698 (execution
notes and handoff). Link check: 0 broken relative links across all 184 README files and
the top-level `docs/`. Not verified: anything that lives only on CSCS Clariden or Hugging
Face (receipts, checkpoints, run roots) — the READMEs say so where it applies.
`docs/_archive/*` still contains pre-existing broken links between archived documents;
untouched.

## Archive candidates (by subproject, as reported by the writers; nothing moved)

- **01 Corpus phase (_archive) / papers** — _archive/01_0/{CURRENT_STATUS.md, HANDOFF_2026-04-28.md, PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md, WAVE2_PIPELINE_RUN_2026-04-26.md, WAVE3_PRODUCTION_PROGRESS_2026-04-28.md, NEXT_SAMPLE_WAVE_PLAN.md, PER_LINE_CLEANER_BRANCH_PLAN_2026-05-04.md, CORPUS_CLEAN_WAVE2_PLAN.md, scripts/README.md}; all four 01_*/TODO.md.
- **02 Apertus Tokenizer Spec / 02.1 Tokenizer Experiments** — 02_1/TODO.md; 02_1_1/CONTINUOUS_BPE_EXTENSION_{PLAN,TODO}.md (four-arm framing closed by fiat); 02_1_6/TODO.md, REVIEW_INTEGRATION_20260517.md, 12_gini_optimization.md (SUPERSEDED, prediction wrong), 01_explicit_goals.md, 02_implicit_constraints.md, _deprecated_20260518/** (13 files); 02_1_7/PLAN.md, FIRING_COUNT_PLAN.md; polytonic/ANCIENT_GREEK_AFTER_C3_PLAN.md (overturned 07-29); 02_spec/TODO.md.
- **02.2 Tokenizer implementation** — TODO.md (stale "87% complete"); CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md (snapshot; two tables superseded); 02_2_1/TODO.md, v3_2_INTEGRATION_REPORT_20260515.md, FEEDBACK_FROM_PMI_PROMOTION_CONSUMER_20260515.md; 02_2_2/analysis/main_token_sets/README.md (self-declared superseded by main_token_sets_pmi/); german_review/REVIEW_ISSUES_20260514.md; 02_2_3/PLAN.md, 02_2_4/PLAN.md + METHODOLOGY.md (never built).
- **03 Apertus extension and embedding adaptation** — TODO.md (frozen 05-23, absolute paths); eval/trajectory_analysis_20260524/{BAKEOFF_TRAJECTORY_ANALYSIS_20260524, CONTINUATION_3P5B_RESULTS_20260525, CONTINUATION_5B_RESULTS_20260526}.md; eval/v4_baseline_20260521/V4_RESULTS.md (partial); eval/live_summaries/* (≈20); td_pilot/td_full25 intrinsics dirs; bakeoff_training/RUN_LOG_5B_TD_VS_VANILLA_20260525.md; token_distillation/RUN_LOG_20260523.md (90 KB); corpus_build/HPLT_TOKEN_COUNT_RUN_20260526.md; 03_2 partial run report + REVIEW_INTEGRATION rounds 1–4 + READY_TO_SPIN_UP.md; 03_3 REVIEW_PACKET.md, ANALYSIS.md, CURRICULUM_AND_INIT_CORPUS.md, REPLAY_LANGUAGE_SELECTION.md (v0.7-superseded banners); 03_4 AUTH_AND_NODE_FINDING/STORAGE_AND_EXISTING_WORK/ENVIRONMENT_AND_BENCHMARKS.md (05-20 recon); _archive/** (21 docs).
- **04 CPT training regime on Vanilla Apertus-8B** — reports/latest_5b_report_status.md + .json (mid-run renderer snapshot); reports/iter_*_checkpoint_sidecar_{handoff_pass,verify_latest}.json (10 machine snapshots); reports/iter_0000119_checkpoint_sidecar_precheck.json; adversarial_reviews/*_watch_state/*; monitor_logs/*_sidecar_verify_state/*.handoff_done; reports/config_geometry_audit_iter_0000119.{md,json}; reports/PLANNING_AGENT_REPLY_20260601.md (§4 retracted same day).
- **05 Token-Distillation CPT (Task 2)** — EXECUTION_LOG_CURRICULUM_SWEEPS_V2.md (2,408-line poll log); LOG.md (per-poll Slurm narrative); ARCHIVE.md; RUNBOOK.md (run can't be relaunched, data deleted); ROADMAP_20260611.md (superseded on 3 axes); EPISTEMIC_PLAN.md; 03/curriculum_sweeps_v2/{RUNBOOK.md, BETA2_SWEEP_PLAN_20260616.md, BETA3_SWEEP_PLAN_20260613.md, train/UPSTREAM_EDITS.md}.
- **05/02 Corpus preparation** — 10_clean_hplt/reports/ superseded scoreboards (40 policy_gate_audit + 24 policy_recommendation + 16 pending_boundary + 24 boundary materialization + 20 queue/pilot snapshots + 4,573 heldout .txt); 30_decontaminate/_archive_k13_overlap_method/ (self-archived) + 3 same-day audit iterations; 15_clean_academic/eval/BIBLIOGRAPHY_NEXTGEN_*_2026072*.md (SUPERSEDED banners), WORST_DOCS_REVIEW_20260722.md, ANNOTATION_RUNBOOK.md (archived), CODEX_LIMITS.md, CONFUSION_MATRIX.md + EVAL_B_FINDINGS.md (pilot), ITERATION_ROUND{1,2,4}.md; sequence_models/BIB_LINE_TO_BLOCK_CLASSIFIER_PLAN.md, HUMAN_GOLD_RUNBOOK.md (never executed), results/** (9 run dirs), evolution/legacy_launch_20260718/; BIB_CLEANING_HANDOVER_20260727.md (superseded but the only record of the contaminated first attempt).
- **05/04 Full CPT corpus preparation (full-corpus v2) / 05/05 Training-dataset bridge** — 04/RUNBOOK.md (v2 DAG never ran); 04/FINALIZATION_PIPELINE.md; 04/STRUCTURAL_SPAN_PRODUCTION.md; 04/docs/agent1_v4_gfm_normalization.md; 04/requirements-gfm-prototype.txt. Suggestion: root docs/AGENT1_*_2026-07-18/21.md belong next to 05/04 (they are its primary evidence).
- **06 Dataset scheduling experiments / 07 Full Apertus-8B mixed CPT** — 06/presentations/LR_SCHEDULES_AS_RUN_AND_NEXT_20260801.html, LR_SCHEDULE_TAIL_EXPERIMENTS_20260801.pptx+.inspect.ndjson, LR_FLOOR_EXPERIMENT_RESULTS_20260802.html (8B LR material, not 06's experiment); 06/evidence/checkpoint_plan_rebind_required_20260802.json, dataset_schedule_and_native_greekmmlu_plan_20260802.json, three *_smoke_20260802.json; 07/presentations/FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260810.* (superseded next day), GREEKMMLU_DRIFT_*/HISTORICAL_*/WRONG_CELL_*/DRIFT_HISTORY_GALLERY_20260811.* (exploratory statistic selection 09 dropped); 07/clariden/train_segment.sbatch.orig; 07/configs/eta_16node_to_20260809.json.
- **08 Targeted 8B CPT experiments (hard HPLT→OpenArchives cross-scale study)** — REVIEW_ULTRACODE_HARD_H_TO_G_PLAN_20260814.md (157 KB R1, superseded by R2); REVIEW_ULTRACODE_R2_...md (164 KB); ULTRACODE_R2_REMEDIATION_20260814.md; SCALE_PREDICTIVITY_STUDY_20260812.md (design-only); CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md; CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md; configs/experiment_{a,b}_recipe.json, allocation_plan.json, owner_authorization.json, hard_h_to_g_allocation_v1.json; evidence/restart_smoke_*_20260812.json; presentations/greekmmlu_endpoint_20260822/ and greekmmlu_trajectory_20260822/ (superseded same day by full-panel); patches/train_segment_targeted_benchmark_offset.patch.
- **09 Full 8B CPT results analysis / 10 Early-cooldown causal experiment** — 09/presentations/NATIVE_GREEK_3CP_BENCHMARKS.{html,data.json}; 09/NATIVE_GREEK_3CP_RESULTS_20260812.md; 09/NATIVE_GREEK_BENCHMARKS.md (Tier 2 never executed); 10/PROCESS_REVIEW_20260813.md.
- **10 Greek post-training** — MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md (827-line recommendation, "nothing built"); DATA_SURVEY_GREEK_SFT_20260823.md (snapshot); ROOT_CAUSES_v1.md (proposal, not applied); pilot_no_robots_100/FEEDBACK.md (frozen at 9/100 rows); pilot_v2/prompts/* (byte-identical dupes); pilot_v2/v2/batches/* (32 regenerable payloads).

## Contradictions and unverifiable claims (by subproject)

- **01 Corpus phase (_archive) / papers** — 01_2 "Already Decided" says 70/30 but C3 shipped 50/50; "wave-2 broad cleaner" label maps to two different branches; wave-4 exporter fix introduced the row-level key that caused the split leak; DEDUP repair plan's golden equivalence test never delivered; CITATION_AUDIT corrections not propagated to CURRENT_HYPERPARAMETERS References list; wave-3 "0 rows missing badness" vs wave-4 291,107; "200-document human review" pool has no artifact.
- **02 Apertus Tokenizer Spec / 02.1 Tokenizer Experiments** — three competing "ship" vocab declarations in May (153,600 in 03_3 05-20; 148,480 public release 05-25/27; 148,992 production 07-29) — all cited; 02_1_1 "4 phases" vs six listed; "curated delta essentially zero" vs −0.013 (pruned) / −0.0005 (backfilled); 02_1_4 and 02_1_5 both claim to be terminal (02_1_7 was); splitter row-vs-doc leak never fixed, only worked around; polytonic sweep numbers run 05-18 but committed 06-11.
- **02.2 Tokenizer implementation** — unmapped keys 7 (checkpoint) vs 5 (committed manifest; als/lat later mapped); bit counts drift across docs (YAMLs authoritative); German sample 1,005,777,069 vs 1,005,786,784; firings 113.4B/1,933 vs 114.37B/1,934 (English re-run row); 02_1 README's claim of "compatibility checks in 02_2" is false (done in 03_3); artifacts/ outputs/ parquet/npz gitignored → primary artifacts unverifiable in-repo.
- **03 Apertus extension and embedding adaptation** — TD Greek lead "+1.40→+1.28 pp" called "widened" (it narrowed) and "+0.69 pp" elsewhere; BPB crossover ~6.5 B vs ~6.8 B; vocab scope 153,600 declared "active CPT base" then 148,480 pinned same day (bakeoff used 148,480); replay languages 34 recommended vs 24 shipped; Greek-agg baseline 0.525 unreconciled; PEER_GREEKMMLU_BASELINE "ready to submit" but never executed; 03_1 artifacts/ gitignored (numbers second-hand); "142.6 M extra params" mislabelled per-matrix; an unreported third dedup run 20260518T231526Z; production_cpt README's "Vanilla selected" vs CPT_MASTER "not rule-bound".
- **04 CPT training regime on Vanilla Apertus-8B** — 217.2 GPU-h only in TASK2_HANDOFF (gpu_hours_breakdown is a mid-run snapshot 145.85/≈204.9); Plutus CIs differ between 5B_REPORT §5.4 and v4 JSON (JSON used); wall-clock 45h vs 43h; reports/train_logs_cache_5b/ never committed (plot script can't re-run); adversarial_reviews/Vanilla-5B lacks prompt.md/metadata; Decisions-Matrix row I (move sidecar off xfer) never applied.
- **05 Token-Distillation CPT (Task 2)** — PRODUCTION_MIX_DECISION cites reports/cpt_curriculum_forgetting_learning.html (absent); its "peak" GreekMMLU = iter-1190 readings, decision taken before sidecars drained; ROADMAP GreekMMLU 48.8→55.6→59.3 vs reports 48.3→55.3→58.7 (different metric; ROADMAP numbers unreproduced); 01_decontam estimate_inputs.json references absent audit JSON; RUNBOOK says β₂=0.999 but run used 0.995; two failed-status vanilla retention jobs never resolved; 07_8b_lr_floor_reconstruction: three tails completed but NO floor decision committed and 06/07 don't cite it though both use WSD-10; 04 recommended Path-A geometry but the pilot launched 4096/500k-with-scaling geometry.
- **05/02 Corpus preparation** — BIB_DETECTOR_BAKEOFF_20260725 says "do not port heading_lexgate" — port began next day and it is what production runs; heading_lexgate both passes (0.98503 opened cohort) and fails (0.95921 sealed) the 0.98 gate; β-precision quoted 0.85–0.90 / ≈0.70 / 0.762; two different "final" β-gates (F1 0.886 vs 0.898); CODEX_LIMITS self-contradicts; canonical-prompt ownership conflict; "no human campaign planned" vs HUMAN_GOLD_RUNBOOK 2,720-action plan; cohort-2 agreement 97.7583% vs 98.04% (different denominators); seven-role vs eight-label; test count 9/9 vs 7; ERROR_CATALOG.md and stage1_validate_deduped.codex_review.md referenced but absent. NO apply-run receipts in-tree for 15_clean_academic/production (authorized 07-28) or 40_anonymize/hf_v2_release (08-11).
- **05/04 Full CPT corpus preparation (full-corpus v2) / 05/05 Training-dataset bridge** — old 04 README "not yet a completed corpus build" never updated (corpus built by v5 lane 3 days later); agent1_v5_eiger_pipeline.json records max_bucket_documents 5000 while the 07-21 diagnosis moved to 50,000 cluster-side; release.private_only true vs eventual public dataset (publisher change e8fbec2c 07-28); no in-repo receipt for the final v5 publication (51,839,746 corroborated from 3 later docs); several v2 HF revisions cited at different lifecycle stages; exact v5 completion/publication date not recorded (bounded 07-20..07-28).
- **06 Dataset scheduling experiments / 07 Full Apertus-8B mixed CPT** — old 07 README + recipe_8b_full_mixed.json say 19,248 updates/80.73B/6 segments (pre-sanitization) vs runbook/09 18,284/76.685B/5; FACTORIAL_EXPERIMENT_DESIGN header still says "not authorized" though launched 08-03; DP32 restart tolerances added after the result they accept (disclosed); 0.5B TD pilot per-cell numbers only in CSCS receipts; 5% retention margin reused from LR smoke; exploratory-prefix GreekMMLU n=16,632 vs final n=16,159 not comparable.
- **08 Targeted 8B CPT experiments (hard HPLT→OpenArchives cross-scale study)** — 1p5b_td_acceptance_policy_v2.json still "proposal_pending_owner_approval" though 1.5B trained (approval receipt on Clariden only); replication_v1.json status string predates execution; tokenizer sha 358ae3f2 vs 37c110e7 (serialization difference, resolved in publication doc); two "8B historical" curves conflated (59.9627% β₂ arm vs ~58.75% TD study); TD-scan 08-15 measurement has no receipt beyond README; production allocations 0→3,694 not itemised; A/B compute overran plan (183.98 vs ~112 GPU-h; 55.64 vs ~35).
- **09 Full 8B CPT results analysis / 10 Early-cooldown causal experiment** — 10's own README/PROCESS_REVIEW leave v6 "queued" while the out-of-repo watcher state records failure; 09_1/09_2 recovered READMEs were stale (averaging/ensembles done 08-19/20; metadata unblocked 08-19); CHECKPOINT_AVERAGE_RESULTS has a self-corrected data error; canonical 19-cp presentation has no averaged models so "best model" disagrees with the averaging doc; post-08-19 results explicitly unreviewed; SUBPROJECTS_OVERVIEW predates 09_1/09_2/09_3.
- **10 Greek post-training** — no in-repo evidence of the natural-greek-sft repo; DATA_SURVEY says "takes 10 to avoid collision" but 10 was already taken; ~9% content-filter loss claimed then retracted same day; Skywork RM chosen then rejected same day; pilot-1 model claim (Opus 5) has no run log/receipt; pilot-1 artifact URL unverifiable; pilot 2 never reviewed.
