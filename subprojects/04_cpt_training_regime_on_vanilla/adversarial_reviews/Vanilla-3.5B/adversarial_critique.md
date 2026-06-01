**1. Verdict on iter 834 and on the plateau hypothesis**

Mechanically clean and statistically interesting. `iter_0000834` is fully landed (`latest_checkpointed_iteration.txt=834`), HF conversion `iter_0000834_hf` carries 4 safetensor shards + tokenizer, the sidecar manifest enumerates all 8 expected kinds (convert, native_mcq, greek_nlp, heldout_greek_bpb, retention, code_bpb, math_bpb, checksum), and training-log tail through iter 834 is healthy: `lm loss ≈ 1.39–1.42`, `learning rate = 1.100000E-05` (peak), `skipped=0`, `nan=0`, `params norm` drifting 7091.54 → 7091.42 (slow, monotone, no instability), throughput ~8050 tok/s/GPU. Training officially terminated normally at iter 834 with `=== bakeoff arm vanilla done ===`.

The native-MCQ artifact is operationally clean: `run_metadata.json` lists all 4 benchmarks, `Vanilla-3.5B_native_mcq_aggregate.json` reports `headline.macro_accuracy=0.4789658 / n_tasks=3` with explicit `headline_policy.headline_benchmarks=[greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep]` and `diagnostic_benchmarks=[plutus_qa]`. The `--export` comma-fix and the Plutus-out-of-headline structure both persist (Vanilla-1B/Vanilla-2B Finding 2/5: fixed and verified at iter 834 — second checkpoint after the fix).

**On the plateau hypothesis at the aggregate level**: the paired bootstrap iter-477-vs-iter-834 headline_3task delta is **Δ = −0.000186, 95% CI [−0.0123, +0.0114] — STRADDLES ZERO**. The aggregate trajectory between 2 B and 3.5 B is statistically indistinguishable from flat (Hypothesis 3 is *not* ruled out by the aggregate alone).

But the aggregate hides genuine, statistically significant motion at the per-task level (new finding — see §8):

- **greekmmlu** (n=16,632): +2.09 pp, CI [+0.0164, +0.0254], OUTSIDE zero — **continues to improve**.
- **ilsp_mcqa_asep** (n=1,200): −1.92 pp, CI [−0.0367, −0.0033], OUTSIDE zero — **statistically significant regression**.
- ilsp_medical_mcqa (n=432): −0.23 pp, CI [−0.0348, +0.0278], inside zero (underpowered).
- plutus_qa (n=225): exactly 0 pp, CI [−0.0444, +0.0444], inside zero (underpowered).

Task-level cancellation produces the −0.02 pp aggregate. The cleanest reading is **"under the hood, two of three headline tasks are moving in opposite directions in a statistically significant way; the aggregate plateau is partly a cancellation artifact, not a single saturated process."**

On the cross-arm regime claim at the 3.5 B mark: my paired bootstrap of iter-834 vs bakeoff-Vanilla-3.5B gives **Δ = +0.0420, CI [+0.0262, +0.0581], OUTSIDE zero**, with **every per-task delta positive** and 2 of 3 individually significant. The regime hypothesis (cpt-plan §2.2) is **reconfirmed at 3.5 B** — at the same +4.20 pp magnitude as iter 477 vs bakeoff-2B (+4.65 pp). Lower bound of iter-834's marginal CI (0.4606) exceeds bakeoff-3.5B's upper bound (0.4545), confirming non-overlap.

Verdict: **trustworthy as the iter 834 post-warmup probe.** Regime gain holds at 3.5 B; the aggregate iter-477→iter-834 plateau is real at the aggregate level but masks per-task structure. The "regime is materially better than the bakeoff regime" half of the hypothesis is now confirmed at *two* matched token marks (2 B and 3.5 B). The "matches Apertus-Base" half remains held by the Path-A/Path-B confound (carried forward from Vanilla-2B C1/C2).

---

**2. Critical Findings**

**C1. iter-834 vs iter-477 aggregate plateau hides a significant per-task regression on ilsp_mcqa_asep.** Paired bootstrap (1000 resamples, rng_seed=20260529, per-task item-level, same methodology as `reports/v4_bootstrap_cis_native_mcq.json` v2): ilsp_mcqa_asep delta = −1.92 pp with 95% CI [−0.0367, −0.0033]; the upper bound sits just below zero — small but statistically significant. By contrast greekmmlu *gained* +2.09 pp [+0.0164, +0.0254]. The +3.05 pp aggregate jump from iter 238 to iter 477 was monotone and broad-based; the iter 477 → iter 834 segment is *not* — it is a task-level reallocation, not a uniform plateau. **Implication:** the 5 B report cannot interpret the −0.02 pp aggregate as "training stopped helping the headline tasks"; it must report the per-task deltas as separate items. This also weakens the "1.2 B more tokens of HPLT clean60 produces a uniform gain on Greek MCQ" narrative — at this regime, an MCQ-test-format-sensitive task (ASEP is a Greek public-sector exam corpus) is losing accuracy while a broad-knowledge MCQ task (GreekMMLU) keeps gaining. The 5 B report should investigate ilsp_mcqa_asep specifically (subject breakdown? choice-pattern shift?) before drawing a Section-8 trajectory verdict at iter 1192.

**C2. Path-A vs Path-B confound for the "recovers to Apertus-Base" claim still live (carried from Vanilla-0.5B C1, Vanilla-1B C2, Vanilla-2B M1).** `iter_0000834_hf/config.json` confirms `max_position_embeddings=4096, rope_theta=500000, rope_scaling=null`; the live training command is `--max-position-embeddings 4096 --rotary-base 500000`. The released base ships Path A (12M / 65536 / llama3). `cpt-plan.md` §2.1 "Training-time positional geometry override (Path B)" now explicitly documents the override and its empirical cost, and §3.4 Q3.4.10 records the Path-A recommendation for Task 2. That documentation closes the *plan-coherence* gap from prior critiques. But the comparison gap is not closed: there is still no Path-A baseline run at matched compute, and the matched-config Apertus-Base eval (BPB 1.22, MCQ 0.4272) is documented as diagnostic-only (perturbed, not a clean baseline). The iter-834 point estimate 0.4790 sits inside the Path-A Apertus-Base CI [0.4629, 0.4997] **but the geometry mismatch is unresolved**. Implication: the 5 B report can defend "iter 834 sits at Apertus-Base point-estimate level on native MCQ" only with the explicit "Path-A baseline, Path-B run; matched-config Path-B is perturbed, not a clean baseline; ~1 B tokens of rope-adaptation cost is paid in the first segment and not in the iter 834 number" caveat. No new "recovers to Apertus-Base" claim is licensed by iter 834 over iter 477.

**C3. Decontamination of Greek MCQ benchmark prompts vs `hplt_b1_5b.jsonl` still absent (Vanilla-0.5B / 1B / 2B C3).** No `*decont*` / `*contam*` run artifact exists in either `/capstor/scratch/cscs/fffoivos` or `/iopsstor/scratch/cscs/fffoivos`. The 20260519 overlay (`audit_id=20260519T010924Z`) addresses Apertus-pretraining overlap, not Greek MCQ benchmark prompts. Per goal `production_blockers_status.V1.status="not_required_for_diagnostic"` this is plan-coherent, but the unmeasured asymmetric-contamination risk applies to *all* "regime is materially better than bakeoff" claims (now landed at 2 of 2 matched token marks). At 3.5 B the per-task pattern (GreekMMLU gains +6.13 pp vs bakeoff-3.5B while ASEP gains only +1.83 pp) is *the* shape one would expect if HPLT clean60 differentially contains GreekMMLU-adjacent prose vs ASEP exam material. Cannot be cleanly separated from "the regime trains better Greek knowledge" without an n-gram or MinHash scan against the four benchmark prompts. This is now a load-bearing caveat for both regime-fix claims.

---

**3. Major Findings**

**M1. Greek BPB heldout still 29.2 % prefix-truncated at iter 834 (Vanilla-0.5B M1 / Vanilla-1B M1 / Vanilla-2B M2 persist).** `iter_0000834/heldout_greek_bpb.json` truncation block is identical: `n_docs_truncated=146, n_tokens_dropped=6,906,358, fraction_truncated=0.292`. **NB: the prompt's claim that `iter_0000834 BPB JSON shows trunc=null` is incorrect — the schema field name is `truncation`, not `trunc`, and the iter-834 value is the same 0.292 used at iter 119/238/477. The schema has not changed; the heldout file has not been swapped. Trajectory within-run: 0.6049 → 0.4684 → 0.4313 → 0.4197 (iter 119 / 238 / 477 / 834). Like-for-like monotone decrease, with the iter-477 → iter-834 BPB delta of −0.0116 markedly smaller than the iter-238 → iter-477 delta of −0.0371. The per-source breakdown shows the same pattern (greek_academic 0.3490 → 0.3366, greek_hplt_clean60 0.4052 → 0.3916, greek_dialogue_textbooks 0.6313 → 0.6217, greek_legal_civic 0.2942 → 0.2867 — all small monotone decreases). So BPB *is* still improving (slowly), even though the aggregate MCQ is flat — these two signals diverge in iter 477 → iter 834 in a way they did not diverge in iter 238 → iter 477. The cpt-plan §2.3 mapping for "BPC improves but native MCQ stays flat below Apertus-Base" reads as "Investigate forgetting via KL-to-base on fixed probes; consider higher replay share" — a new diagnostic suggestion that wasn't relevant at iter 477.

**M2. Retention at iter 834: xnli_en finally recovers, but mmlu / global_mmlu_en/fr regress vs iter 477.** Comparing iter 477 vs iter 834 retention numbers:

| Task | iter 477 | iter 834 | Δ |
|---|---:|---:|---:|
| mmlu | 0.5962 | 0.5891 | −0.71 pp |
| global_mmlu_en | 0.6375 | 0.6150 | −2.25 pp |
| global_mmlu_fr | 0.5800 | 0.5600 | −2.00 pp |
| global_mmlu_de | 0.5875 | 0.6025 | +1.50 pp |
| arc_challenge | 0.5247 | 0.5188 | −0.59 pp |
| arc_easy | 0.8035 | 0.8114 | +0.79 pp |
| piqa | 0.7905 | 0.7824 | −0.81 pp |
| hellaswag | 0.5916 | 0.5905 | −0.11 pp |
| winogrande | 0.7143 | 0.7096 | −0.47 pp |
| xnli_en | 0.4904 | 0.5462 | **+5.58 pp** |
| xnli_fr | 0.4679 | 0.5044 | +3.66 pp |
| xnli_de | 0.4847 | 0.5016 | +1.69 pp |
| xnli_ru | 0.4880 | 0.4912 | +0.32 pp |
| xnli_el | 0.4161 | 0.4325 | +1.64 pp |

The **xnli_en regression flagged in Vanilla-2B M2** has fully *inverted* at iter 834 (−2.85 pp net vs iter 119 at iter 477 → +2.73 pp net vs iter 119 at iter 834). XNLI gains broadly across all 5 measured languages. **But MMLU and global_mmlu_en/fr regress vs iter 477** for the first time in the run. These are not (yet) below iter-119 levels (iter 119 mmlu = 0.5674, iter 834 mmlu = 0.5891 — net +2.17 pp; iter 119 global_mmlu_en = 0.605, iter 834 = 0.615 — net +1.0 pp), so the iter 477 → iter 834 regression is recovering from a particularly high iter-477 readout rather than dropping below the warmup baseline. Still, the cpt-plan §2.3 diagnostic table row "Multilingual retention degrades → Replay too low or LR too high" applies in miniature on MMLU-family tasks; the report should explicitly call this out as a *new* iter-477 → iter-834 retention movement that was not present at iter 238 → iter 477. **Note**: the same `xstorycloze_*` `None` coverage and `global_mmlu_ru` `None` from iter 477 persist at iter 834 (Vanilla-1B M2 / Vanilla-2B M3 carry forward).

**M3. Per-task signal-to-noise constrains the iter-477-vs-iter-834 verdict.** Only the headline_3task aggregate plus greekmmlu (n=16,632) have tight CIs at this n. ilsp_mcqa_asep (n=1,200) is borderline (CI = [−0.0367, −0.0033], upper bound near zero), ilsp_medical_mcqa (n=432) is underpowered (CI width 0.062 vs |Δ|=0.0023), plutus_qa (n=225) is too small to detect any plausible single-checkpoint movement (CI width 0.089 vs |Δ|=0). The 5 B report's per-task interpretation must be sized for these CIs; in particular **any "the regime maintains progress on Medical MCQA and Plutus" claim is unsupportable** at iter 477 → iter 834 step granularity. This is a power problem with the headline_3task suite, not a regime problem — the suite is dominated by greekmmlu, and aggregate motion is principally greekmmlu motion modulated by smaller-n noise.

**M4. iter_0000833 saved alongside iter_0000834 — duplicate save pattern persists (Vanilla-2B M8).** `ls checkpoints/` shows both `iter_0000833` and `iter_0000834`. Per the training log, iter 833 is the segment's penultimate `SAVE_INTERVAL=119`-aligned save (715 + 118 = 833) and iter 834 is the end-of-segment save (matches the iter 476/477 duplicate at the prior segment boundary). Both are mechanically clean and the HF/sidecar pipeline targets iter 834 only. But the duplicate-save pattern now occurs at *every* segment boundary that aligns close to a `SAVE_INTERVAL=119` cycle (476/477, 833/834). The 5 B report should either (a) document the rule, (b) delete the iter 833 dir to free `iopsstor` quota, or (c) skip iter 1192 / iter 1191 dual-save by tuning `SAVE_INTERVAL` for the final segment.

**M5. `run_metadata.json[lr_decay_style] = "1-sqrt"` while training argv uses `--lr-decay-style constant` (Vanilla-0.5B Minor / Vanilla-1B M5 / Vanilla-2B M4 persist).** Top-level metadata at `04_vanilla_goldfish_5b_20260528T112539Z/run_metadata.json` still carries `"lr_decay_style": "1-sqrt"` plus a parallel correct `"lr_schedule_style": "constant"`. The live argv is authoritative; metadata-only readers will draw the wrong post-warmup conclusion. Same fix as before: delete or set the stale field. This has now survived 5 segments and 5 critiques.

**M6. iter 834 sidecar `native_mcq` still consumes a 4-GPU GH200 node for a 1-GPU eval (Vanilla-1B M6 / Vanilla-2B M5 persist).** Per the sidecar TSV the convert/native_mcq/greek_nlp/bpb/retention chain ran on `normal` with full-node allocation. Documented as Decisions Matrix row Q (acceptable for trajectory sidecars). At ~7 GPU-h per checkpoint × 5 checkpoints = ~35 GPU-h overhead vs ~118 GPU-h training (per `reports/gpu_hours_breakdown_20260530.md`). The 5 B report's compute-justification block must surface this; if iter 1192 sidecars run the same shape, the projected ~205 billed GPU-h leaves room before a 250-300 GPU-h hard budget.

**M7. greek_nlp_s100 at iter 834 — same 0.0-F1 prompt-echo failure mode (Vanilla-2B M6 persists).** Not re-inspected file-by-file; pattern is the same family as iter 477 (intent_classification.accuracy=0, legal_classification.accuracy=0, ner.entity_f1 ≈ 0.01). Out of headline scope but worth labeling explicitly so the report's "Greek capability" claim doesn't pick up greek_nlp_s100 numbers downstream.

---

**4. Minor findings and hygiene notes**

- **`native_mcq` runner's `max_input_tokens=3072`** persists at iter 834 (same as 0.5B/1B/2B). Still unverified whether the bakeoff Vanilla numbers cited in cpt-plan.md §1.1 (0.4327 / 0.4370 / 0.4305) used the same `max_input_tokens`. If the bakeoff used 4096, +4.20 pp at iter 834 vs bakeoff-3.5B is partly an eval-setup delta. (Vanilla-2B Missing-Evidence-8 carries forward.) Recommend a one-time sanity rerun of bakeoff-Vanilla-3.5B's predictions at `max_input_tokens=3072` to bound the delta.
- **xnli_el** (Greek XNLI) is in the retention bundle and lifts from 0.4161 to 0.4325 (+1.64 pp) between iter 477 and iter 834. cpt-plan.md treats XNLI-EL as a "MT-derived Greek task, demoted to diagnostic", but it's the *one* Greek-relevant retention task in the lm-eval bundle; the 5 B report should call out this within-Greek-task gain as evidence-of-Greek-improvement in a way that is independent of the prefix-truncated BPB and the +4.2 pp MCQ headline.
- **Predictions JSONL example-id alignment verified.** `iter_0000477_hf` and `iter_0000834_hf` predictions JSONLs share identical example_ids in identical row order for all four benchmarks (verified pre-bootstrap); paired-bootstrap deltas are *paired*, not pseudo-paired.
- **Plutus marginal CI at iter 834 (0.4267, 0.5556) overlaps bakeoff-Vanilla-3.5B Plutus (0.3378 [0.2756, 0.4044]).** Implies the Plutus pseudo-headline at 0.4889 is *not* statistically distinguishable from bakeoff-3.5B's Plutus at 0.3378 — the CI width (0.13) is wider than the delta (+0.067) at n=225. The 5 B report should not promote Plutus as evidence of regime fix (size of CI > size of effect at the standard 95 % level).
- **Sidecar manifest matches expected kinds.** `sidecar_jobs.tsv` enumerates convert/native_mcq/greek_nlp/heldout_greek_bpb/retention/code_bpb/math_bpb/checksum — all 8 expected for the post-Decisions-Matrix-row-P contract. (`checksum` is not in `expected_kinds` per design; documented hygiene-only.)
- **Heldout BPB truncation field name** in the iter-834 JSON is `"truncation"` (not `"trunc"`). The reviewer prompt's note "iter 834 BPB JSON shows `trunc=null`" is incorrect — there's no `trunc` field in this schema; the long-name `truncation` field is populated and matches prior iters exactly. **No schema drift.**
- **xfer-bound watcher** for the iter-834 sidecar fan-out (Decisions Matrix row I / script_audit §C1) appears to have worked — iter 834 sidecars all ran and `checksum` is the only `xfer`-routed sidecar in the manifest (job 2428543). The xfer reservation lifts 2026-06-11, but the iter-834 checksum slot didn't pend. For iter 1192 the same watcher still risks pending behind the reservation if it falls during the maintenance window. Re-routing (Decisions Matrix row I) remains unapplied in the live deployment.
- **Code BPB iter 834: 0.2697** (vs iter 477: 0.2807) — small monotone decrease. **Math BPB iter 834: 0.5491** (vs iter 477: 0.5589) — same. Consistent with the language-modeling-improving-slowly read.

---

**5. Missing evidence**

To raise iter 834 to "load-bearing post-2-B probe":

1. **Sanity rerun of bakeoff-Vanilla-3.5B at `max_input_tokens=3072`** (the iter-834 runner setting) to bound the +4.20 pp regime delta against an eval-setup confound. Cheap (1 GPU-h).
2. **Per-subject GreekMMLU breakdown for iter 477 vs iter 834** — the +2.09 pp aggregate greekmmlu gain (CI excludes 0) should be allocated to specific subjects to test the "the regime is teaching the model more general Greek knowledge" hypothesis vs "the regime is benefiting from contamination on subject X." `greekmmlu:idx` example_ids in the predictions JSONL already carry per-subject metadata; a 5-minute pandas pass on the JSONL is sufficient.
3. **Per-subject ilsp_mcqa_asep breakdown** — to localize the −1.92 pp regression. Is it broad-based (suggesting a true regime-induced format-sensitivity drop) or concentrated in one subject group (suggesting a noisy realisation at small n)?
4. **Decontamination report.** Now applies to two checkpoints' worth of cross-arm gain claims. A single MinHash pass against the 4 benchmark prompts is the cheapest insurance policy in the entire run.
5. **English KL-to-base on a fixed probe set** at iter 477 and iter 834. The MMLU / global_mmlu_en/fr regression in §M2 is the first sign of forgetting on English; KL-to-base would distinguish "the model is drifting in policy space" from "test-time noise."
6. **Verification that the heldout JSONL `cpt_greek_heldout_500_20260522.jsonl` was not regenerated** between iter 477 and iter 834 (an inode/mtime check). The BPB trajectory is only meaningful as a within-run delta if the file is byte-identical.
7. **A paired-bootstrap delta CI for `iter-834 vs bakeoff-Vanilla-3.5B` and `iter-834 vs iter-477`** baked into `reports/v4_bootstrap_cis_native_mcq.json` (currently only computed in this critique's workspace). The 5 B report quote-source for the regime CIs should be the V4 artifact.
8. **A KL-to-base check on Russian** (`global_mmlu_ru` is `None` in the lm-eval bundle and has been since iter 119). The 5 B report can't say "EN/FR/DE/RU retention" without a Russian signal; should either source the eval differently or say "EN/FR/DE measured, RU absent" outright.
9. **A v4 artifact re-emit** including the iter-834 entry with paired CIs against bakeoff-Vanilla-3.5B and against iter-477. The artifact at `reports/v4_bootstrap_cis_native_mcq.json` is still "v2" (iter 238/477 + matched-config); iter-834 not yet added.

---

**6. Recommended next actions before reading iter 1192 or drawing 5 B-report conclusions**

1. **Build the iter-834 entry into `reports/v4_bootstrap_cis_native_mcq.json`** (delta vs iter-477, vs iter-238, vs bakeoff-Vanilla-3.5B, vs bakeoff-Vanilla-2B/5B, vs Apertus-Base, vs matched-config Path B). Reuse `run_bootstrap_v2.py` pattern; add a v3 revision tag. The numbers in §7 below give the paired headline_3task delta CI for iter-834 vs iter-477 to seed this update.
2. **Run per-subject breakdowns** of greekmmlu and ilsp_mcqa_asep for iter 477 and iter 834 from the existing predictions JSONLs. If ASEP's regression is broad-based, the 5 B report needs a separate paragraph; if it's concentrated, it's still mentionable but downgrades to "subject-localized noise."
3. **Address C3 (decontamination)** now, not "before final 5 B report." A single MinHash pass is plan-coherent under V1 documented-caveat policy, but with two checkpoints' worth of cross-arm gain claims the caveat has compounded.
4. **Do NOT extrapolate iter 1192 from the iter 238 → iter 477 slope.** The 2026-05-29 RUN_LOG entry "Implication for the 5B / 10B trajectory" projected iter 1192 to land near 0.515 from the iter 238 → iter 477 +3.05 pp / 1 B slope, halved. The iter 477 → iter 834 paired delta CI [−0.0123, +0.0114] is inconsistent with even half-rate continuation — the upper bound +0.0114 implies at most +1.14 pp over 1.5 B tokens, ~+0.76 pp / 1 B, well below the halved-rate projection (+1.5 pp / 1 B). The 5 B report should adopt a Bayesian-credible interval bracket on iter 1192 anchored on the iter-477-iter-834 CI, not on the steep early slope.
5. **Apply Decisions Matrix row I** (re-route watcher + checksum from xfer to normal) before iter 1192 lands, so the iter 1192 sidecar fan-out doesn't pend behind the 2026-06-11 maintenance window.
6. **Document the duplicate-save pattern** (iter 833/834 and prior 476/477) before iter 1191/1192 saves the same way.
7. **For the iter-1192 review**, recompute the paired iter-477-vs-iter-1192 *and* iter-834-vs-iter-1192 CIs jointly so the 5 B trajectory verdict has three breakpoints, not two.

---

**7. Persistence of prior critical findings + iter-477-vs-iter-834 paired CI**

Persistence (iter 119 → 238 → 477 → 834):

- **Path-B vs Path-A geometry mismatch** (Vanilla-0.5B C1 / 1B C2 / 2B M1): **still live**. cpt-plan §2.1 now properly documents Path B as a training-time override + §3.4 Q3.4.10 records the Path-A recommendation for Task 2. *Doc gap closed; comparison gap remains.* The matched-config Apertus-Base baseline is now formally diagnostic-only (perturbed). No new Apertus-Base anchor exists at iter 834. Verdict: still live, properly documented, no fresh resolution.
- **Plutus-in-headline + Slurm `--export` comma bug** (Vanilla-0.5B C2 / 1B C1 / 2B Finding 2 & 5): **fixed and verified at iter 834.** `run_metadata.json` lists 4 benchmarks; aggregate JSON has `headline.n_tasks=3`, `diagnostics.n_tasks=1`, explicit `headline_policy` block; fix-class survives the second sidecar-watcher-driven checkpoint after the patch. Verdict: closed.
- **Decontamination gap for Greek MCQ prompts** (Vanilla-0.5B C3 / 1B C3 / 2B C3): **still live.** No artifact exists; the +4.20 pp / 3.5B regime claim now joins the +4.65 pp / 2B claim as load-bearing-without-this-evidence. Verdict: still live, documented caveat, *increased load-bearingness*.
- **Greek BPB heldout 29.2 % truncation** (Vanilla-0.5B M1 / 1B M1 / 2B M2): **still live**, schema/file unchanged at iter 834. The prompt's note about "trunc=null" was a false alarm — the field name is `truncation`, not `trunc`, and it's populated as before. Verdict: still live, sensitivity check still owed for the 5 B report.
- **iter-238 retention regression** (Vanilla-1B Major-2): inverted at iter 477; *partial new movement at iter 834* — global_mmlu_en/fr regress vs iter 477 (still above iter 119); xnli_en finally recovers to net positive vs iter 119. Verdict: closed at iter 477 ("warmup transient"), reopened in a different shape at iter 834 (see §M2).
- **Matched-config Apertus-Base eval = diagnostic, not baseline** (Vanilla-2B C2): **resolved as design choice**. cpt-plan §2.1 + hyperparameters.json `training_geometry.matched_config_diagnostic.status_note` now both flag the matched-config dir explicitly as "DIAGNOSTIC of rope-perturbation, NOT a clean baseline." Verdict: closed as a design clarification; no fresh action needed.

**Paired bootstrap iter-834 vs iter-477 (3-task headline, 1000 resamples, rng_seed=20260529, 95 % percentile, per-task item-level paired bootstrap — same methodology as `reports/v4_bootstrap_cis_native_mcq.json` v2):**

- iter-477 point: 0.479152
- iter-834 point: 0.478966
- **Δ (iter-834 − iter-477) = −0.000186, 95 % CI = [−0.0123, +0.0114], OUTSIDE-ZERO = false (CI straddles zero)**
- Per-task paired:
  - greekmmlu: Δ = +0.0209, CI [+0.0164, +0.0254], **OUTSIDE zero**
  - ilsp_medical_mcqa: Δ = −0.0023, CI [−0.0348, +0.0278], inside zero
  - ilsp_mcqa_asep: Δ = −0.0192, CI [−0.0367, −0.0033], **OUTSIDE zero**
  - plutus_qa: Δ = 0.000, CI [−0.0444, +0.0444], inside zero
- headline_4task_with_plutus paired: Δ = −0.000139, CI [−0.0142, +0.0146], inside zero.

**Paired bootstrap iter-834 vs bakeoff-Vanilla-3.5B (same methodology):**

- iter-834 point: 0.478966
- bakeoff-Vanilla-3.5B point: 0.436980
- **Δ = +0.0420, CI [+0.0262, +0.0581], OUTSIDE zero.**
- Per-task: greekmmlu +6.13 pp (OUTSIDE), medical +4.63 pp (OUTSIDE, barely), asep +1.83 pp (inside, CI [−0.0033, +0.0392]), plutus +6.67 pp (inside, CI [0.0000, +0.1333]).

Both new CIs are written to `reports/v4_workspace_iter834/paired_iter477_vs_iter834.json` and `…/paired_iter834_vs_bakeoff_3p5B.json` for inclusion in the next V4 re-emit. The downloaded prediction JSONLs were removed after computation; the scripts and JSON results remain.

---

**8. Trajectory verdict**

The corrected regime produced a real one-shot improvement on the 3-task headline that landed by iter 477 (+3.05 pp paired, CI [+0.0156, +0.0458] outside zero — already in V4 v2) and has been **statistically held but not extended** between iter 477 and iter 834. The aggregate is consistent with three readings of the data, distinguishable per-task:

- **Plateau (Hypothesis 1).** Wrong at the per-task level — greekmmlu is still significantly improving (CI excludes 0). At most: *plateau on the suite-average*, driven by task-level cancellation. Cannot be promoted to "CPT has done what it can on this benchmark."
- **One-shot warmup-finished jump, now stabilized (Hypothesis 2).** Partially correct: the +3.05 pp iter-238 → iter-477 lift was unique and has not repeated. But "stabilized" hides the −1.92 pp ASEP regression and +2.09 pp greekmmlu gain — the model is *moving*, the aggregate is not. The cleanest framing is "the bulk regime-fix gain landed by 2 B tokens; subsequent improvement is task-specific and slow."
- **Noise / need iter 1192 (Hypothesis 3).** The paired CI [−0.0123, +0.0114] cannot rule out a real continued slope of up to +0.0076 pp / 1 B tokens (upper bound, halved over 1.5 B), which extrapolates to ~+0.011 over the final 1.5 B and an iter-1192 headline of ~0.490. iter 1192 *is* needed to discriminate from genuine flat — but the magnitude of any remaining slope is now bounded above by the CI.

**My favored hypothesis: a hybrid of (2) and (3) — the regime-fix delivered a one-shot post-warmup gain that landed by 2 B tokens; further headline gains are bounded above by ~+1 pp / 1.5 B by the iter-477-iter-834 CI; iter 1192 is needed to discriminate from genuine flat, with greekmmlu trajectory the load-bearing per-task signal.** The 5 B report should frame the regime-fix as a real one-shot improvement (iter 477's +4.65 pp delta over bakeoff-2B, now compounded by iter 834's +4.20 pp delta over bakeoff-3.5B at matched token marks, both CIs excluding zero) plus a slow per-task drift (greekmmlu still up, ASEP slowly down, language-modeling BPB still slowly down) rather than as a sustained Greek-side improvement on the headline aggregate.

Critically: **the paired bootstrap CI for iter-477 vs iter-834 does *not* exclude zero**, so the cleanest single-sentence answer is "we cannot statistically distinguish iter 834 from iter 477 on the headline_3task aggregate; per-task structure tells a more interesting story." This is the load-bearing CI for §8 and is bracketed throughout above.

**Genuinely new finding at Vanilla-3.5B not present at 0.5 B / 1 B / 2 B:** the iter-477 → iter-834 *task-level cancellation*. At 0.5 B / 1 B / 2 B every paired delta was either non-significant (warmup) or broadly positive across all 3 headline tasks (post-warmup). iter 834 is the first checkpoint where one headline task gains significantly AND another regresses significantly within the same paired-bootstrap window. ASEP is the first of the 3 headline tasks to show a paired CI excluding zero in the *negative* direction within the regime-fixed run. This is what the prior critiques could not have observed because they had no iter-477 → iter-N pair available. Combined with the M2 retention movement (xnli regions gain, MMLU regions lose), the read is "the model is moving in task-specific directions in the second half of the post-warmup window, even when the aggregate looks flat." Suggests cpt-plan §2.3 row 3 ("BPC improves; native MCQ stays flat below Apertus-Base") deserves to be invoked at iter 834 with the *specific* recommendation "Investigate forgetting via KL-to-base on fixed probes; consider higher replay share" — *that's* the new diagnostic prescription this trajectory invites.

---

**9. Implication for the 5 B endpoint (iter 1192)**

If the iter-477-iter-834 plateau holds, iter 1192 lands at ~0.479 — at the lower edge of Apertus-Base Path-A CI [0.4629, 0.4997] and well above bakeoff-Vanilla-5B [0.4134, 0.4485]. **What's the bracket?** Anchoring on the paired iter-477-iter-834 CI [−0.0123, +0.0114] for the 1.5 B step (iter 477 → iter 834), and assuming the iter 834 → iter 1192 step (1.5 B more tokens) draws from the *same* distribution under the post-warmup regime:

- **Pessimistic bracket (iter 1192 ≈ iter 834 + lower-CI-bound)**: 0.479 − 0.012 = **0.467** — slightly above Apertus-Base Path-A CI lower bound (0.4629) and still well above bakeoff-Vanilla-5B upper (0.4485).
- **Central bracket (iter 1192 ≈ iter 834)**: **0.479** — the plateau prediction.
- **Optimistic bracket (iter 1192 ≈ iter 834 + upper-CI-bound)**: 0.479 + 0.012 = **0.491** — at Apertus-Base Path-A point estimate (0.4817) and inside its CI.

If we instead anchor on the **upper-bound slope from iter 238 → iter 834** (+0.0301 over 2.5 B = +0.0120 / 1 B, halved for "typical flattening" → +0.0060 / 1 B → +0.009 over 1.5 B), iter 1192 would land at ~0.488 — still below Apertus-Base point estimate, still inside its CI. This is the most charitable continued-slope projection given iter-477-iter-834 data and is *much* lower than the 2026-05-29 RUN_LOG projection of ~0.515.

**The single load-bearing iter-1192 question for the regime hypothesis**: does iter 1192 land *meaningfully above 0.479* (e.g., outside the iter-477-iter-834 paired CI upper bound at +0.0114)? If yes, the slope is not zero and the headline trajectory has more to give; if no (lands within [0.467, 0.491]), the regime is genuinely on a plateau on the aggregate and the 5 B → 10 B continuation decision should be driven by *per-task* signal (greekmmlu still improving suggests "yes," ASEP still regressing suggests "no") and BPB / retention trajectory rather than headline_3task.

**Bracket summary for the 5 B report**:

| Scenario | iter 1192 prediction | vs Apertus-Base Path-A CI [0.4629, 0.4997] |
|---|---:|---|
| Pessimistic (CI lower) | 0.467 | inside CI, lower edge |
| Central (plateau) | 0.479 | inside CI |
| Charitable continued slope (half iter 238 → 834 rate) | 0.488 | inside CI, near point estimate |
| Optimistic (CI upper) | 0.491 | inside CI, above point estimate |
| **Old RUN_LOG-2026-05-29 projection (half iter 238 → 477 rate)** | **~0.515** | **above CI upper bound — but inconsistent with iter-477 → iter-834 paired CI; should be retired** |

The 5 B report should **retire** the +0.515 iter-1192 projection from the 2026-05-29 RUN_LOG entry (it assumed continuation at half the iter 238 → 477 rate; iter 834 evidence rules this out within the 95 % CI). The plausible bracket per the iter-477-iter-834 paired CI is [0.467, 0.491]. Iter 1192 above 0.491 would surprise the data; iter 1192 below 0.467 would also surprise (the aggregate plateau is symmetric around the iter-834 point at this CI).

For the 5 B → 10 B continuation decision (cpt-plan §2.4 commitments), the iter-1192 readout should be paired-bootstrapped against iter-834 immediately on landing — that CI plus the per-task breakdown decides whether 5 B → 10 B has expected information value.
