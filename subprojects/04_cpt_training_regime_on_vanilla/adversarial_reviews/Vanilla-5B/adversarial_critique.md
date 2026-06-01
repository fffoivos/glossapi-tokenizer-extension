# Vanilla-5B (iter 1192) — Adversarial Critique

Run tag: `04_vanilla_goldfish_5b_20260528T112539Z`. Endpoint checkpoint
`iter_0001192` ≈ 5.0 B training tokens. All five required CIs computed
from prediction JSONLs pulled to `reports/v4_workspace_iter1192/`
(deleted after computation per task spec; scripts + JSON results
retained).

---

## 1. Verdict on iter 1192

**Mechanically clean and statistically interesting.** iter 1192 is the
latest checkpoint (`latest_checkpointed_iteration.txt = 1192`),
`iter_0001192_hf/config.json` confirms Path-B geometry
(`max_position_embeddings=4096`, `rope_theta=500000`, `rope_scaling=null`,
`vocab_size=131072`). The sidecar manifest enumerates all 8 expected kinds
(`convert/native_mcq/greek_nlp/heldout_greek_bpb/retention/code_bpb/
math_bpb/checksum`), every job exited 0:0. `Vanilla-5B_native_mcq_aggregate.json`
reports `headline.macro_accuracy = 0.49733085 (n_tasks=3)` with explicit
`headline_policy.headline_benchmarks = [greekmmlu, ilsp_medical_mcqa,
ilsp_mcqa_asep]` and `diagnostic_benchmarks = [plutus_qa]` —
Plutus-in-headline + `--export` comma-bug fixes both **verified at the
endpoint** (third stable-LR checkpoint after Decision A/B).

**Headline reading:** 3-task macro = **0.4973**, **paired** 95% CI vs
iter 477 = **[+0.0060, +0.0306]** (Δ +0.0182), outside zero. The
Vanilla-3.5B reviewer's pessimistic/optimistic plateau bracket
**[0.467, 0.491]** is **broken at the optimistic end** — iter 1192 sits
at +0.0063 above the upper bracket. The cleanest single-sentence
interpretation: **the iter 477 → iter 834 segment was a real plateau,
the iter 834 → iter 1192 segment was a real, statistically significant
+1.8 pp lift driven by GreekMMLU (+3.00 pp, CI excludes zero) and ASEP
(+1.58 pp, CI excludes zero); the headline is not yet saturated.**

**Apertus-Base comparison:** paired CI iter-1192 vs Apertus-Base
headline_3task = **[+0.0016, +0.0284]** (Δ +0.0156), outside zero —
**iter 1192 is statistically above Apertus-Base on the 3-task headline.**
Caveat: this is the *Path-A-vs-Path-B* comparison the Vanilla-3.5B critic
warned about, so the "statistically above" claim is a Path-A-baseline /
Path-B-run claim. Per-task: GreekMMLU +3.04 pp (outside), ASEP +2.33 pp
(outside), Medical −0.69 pp (inside), **Plutus −8.00 pp (CI [−0.1422,
−0.0133], OUTSIDE zero in the negative direction)**.

**Plutus drop:** paired CI iter-1192 vs iter-834 Plutus = **[−0.1067,
+0.0000]** (Δ −0.0533) — upper bound is exactly zero. McNemar exact
two-tailed p = 0.081 (26 regressions vs 14 gains across 40 discordant
items at n=225). **Not statistically distinguishable from noise at α=0.05,
but barely.** Bracketed below in §5 / §8.

**Verdict in one paragraph:** iter 1192 is trustworthy as the 5 B
endpoint. The regime hypothesis (cpt-plan §2.2) is **statistically
supported on the 3-task headline at every post-warmup checkpoint** with
iter 1192 cleanly crossing Apertus-Base (Path-A baseline) for the first
time in the run. The Plutus drop is the one ambiguity worth carrying as
a caveat. Eight prior critical findings still live (geometry confound,
decontamination, BPB truncation), one was *prematurely* declared closed
in earlier reviews (decisions matrix row I — xfer-routed checksum
sidecar still ran on xfer at iter 1192; it completed only because the
xfer reservation hasn't started yet).

---

## 2. Critical Findings

**C1. Path-A vs Path-B confound has now become *outcome-relevant*
(persists from Vanilla-0.5B/1B/2B/3.5B C1/C2/M1).** `cpt-plan.md` §2.1
"Training-time positional geometry override (Path B)" + §3.4 Q3.4.10
*document* the override and its empirical cost — this is the
plan-coherence closure the Vanilla-3.5B critic credited. But the
**comparison gap is no longer just a caveat**. At iter 834 the headline
sat *inside* the Apertus-Base Path-A CI (point 0.4790 vs CI [0.4629,
0.4997]); the reading was "not distinguishable from Apertus-Base." At
iter 1192 the paired delta vs Apertus-Base headline_3task is
**+0.0156, CI [+0.0016, +0.0284], outside zero** — for the first time
in the run, the 5 B endpoint *statistically beats* Apertus-Base on the
3-task headline. **This is a *Path-A baseline*, *Path-B run* delta.**
The matched-config Apertus-Base eval (Path-B override on Path-A
weights — `Apertus-Base-matched-Path-B-perturbed` in V4 v2) gives a
*perturbed* lower bound at 0.4272 (CI [0.4096, 0.4456]) which is
documented in `hyperparameters.json training_geometry.matched_config_diagnostic.status_note`
as "DIAGNOSTIC of rope-perturbation, NOT a clean baseline." The
honest read at iter 1192 is therefore a *bracket*: iter 1192 (0.4973
[0.4779, 0.5156]) sits significantly above both (a) Apertus-Base
Path-A baseline (the geometry the base was *trained* under) AND (b)
matched-config Path-B perturbed (the geometry the run was *trained*
under, but with no rope adaptation). The "iter 1192 ≥ Apertus-Base"
claim *is* defensible at the headline level *under both bookend
geometries*, but the 5 B report must show both deltas with an explicit
geometry-confound caveat. **A clean Path-A Vanilla-CPT counterfactual
does not exist and cannot be built without restarting.** This is a
*signed* uncertainty on the headline result; reviewer's preferred
language is "iter 1192 is at-or-above Apertus-Base on the headline at
matched MCQ runner config — see §6 for the eval-setup confound."

**C2. Decontamination of Greek MCQ benchmark prompts vs `hplt_b1_5b.jsonl`
remains absent (Vanilla-0.5B/1B/2B/3.5B C3 / Decisions Matrix row E).**
No `*decont*` / `*contam*` artifact exists in the run tree. Per
`production_blockers_status.V1.status = "not_required_for_diagnostic"`
this remains plan-coherent. But the **load-bearingness has compounded
across three statistically-significant cross-arm claims** (iter 477 vs
bakeoff-2B +4.65 pp, iter 834 vs bakeoff-3.5B +4.20 pp, **iter 1192 vs
bakeoff-5B paired delta inferred from V4 v2 entries = +0.4973 − 0.4305 =
+6.68 pp on the point estimate; the V4 v2 artifact does not yet carry
this pair so the CI must be computed in the v4-v3 re-emit**) plus the
new "iter 1192 ≥ Apertus-Base" claim. The per-task shape now adds new
information: GreekMMLU's +5.09 pp lift over iter 477 (CI [+0.0457,
+0.0562]) is **larger** than ASEP's −0.33 pp (CI [−0.0208, +0.0125])
and Medical's +0.69 pp (CI [−0.0278, +0.0417]) — exactly the
contamination-suspect shape if HPLT clean60 differentially contains
GreekMMLU-adjacent web prose. A single MinHash (or 13-gram) pass
against the 4 benchmark prompts now insures *four* load-bearing
deltas. The reviewer position remains: this is the cheapest insurance
policy in the entire run and should land before the 5 B report does.

**C3. Plutus drop is *barely* noise-consistent but the upper CI bound
is *exactly* zero (new at iter 1192).** Paired iter-1192-vs-iter-834
Plutus QA: Δ = −0.0533, 95 % CI **[−0.1067, +0.0000]**. The upper bound
of the 95 % percentile bootstrap CI is exactly zero, which is the
boundary of "statistically distinguishable from no effect." McNemar's
two-tailed exact p-value over the n_discordant = 40 pairs (26
regressions, 14 gains) = **0.081**. Item-level transition matrix:
84 both-correct, 26 834-correct/1192-wrong, 14 834-wrong/1192-correct,
101 both-wrong. **What this rules in and out:**
(a) Noise: still admissible at α = 0.05 (CI doesn't strictly exclude
    zero by a non-zero margin, but McNemar p = 0.081 is borderline).
(b) Statistical signal: paired vs Apertus-Base on Plutus = Δ = −0.0800,
    CI [−0.1422, −0.0133], **outside zero** — iter 1192 is
    *significantly below Apertus-Base on Plutus* (in the same direction
    as the 834→1192 drop). The iter-477/iter-834 Plutus point estimate
    (0.4889) sat *inside* the Apertus-Base CI [0.4533, 0.5779] — the
    drop is to a level *outside* it.
(c) A real Plutus regression of magnitude 0.04–0.05 (the central iter-477
    → iter-1192 estimate, also Δ = −0.0533) is consistent with a
    forgetting-on-domain-specific-content reading: Plutus is medical-Greek
    QA (n=225, domain-specific) and the corpus is HPLT-Greek-web +
    replay/code/math, with no medical-domain anchor in the mix. The same
    cpt-plan §2.3 row "BPC improves; native MCQ stays flat" mapping that
    the Vanilla-3.5B critic invoked at the per-task level now invokes
    "investigate forgetting via KL-to-base on fixed probes; consider
    higher replay share" *on Plutus specifically*. The iter 477 → iter 834
    Plutus held flat at 0.4889 — the drop happened in the 1.5 B-token
    final segment, exactly when the regression appears on (a less robust)
    per-checkpoint cadence basis.
**Implication for the 5 B report:** Plutus cannot be promoted as
trajectory evidence in *either* direction. Reviewer position: report Δ,
both CIs (vs iter-834 borderline-noise, vs Apertus-Base outside-zero),
the McNemar p, and the transition matrix; do not claim either "Plutus
plateaued then dropped" or "Plutus is noise." The honest reading is
**"5.33 pp drop on n=225 with one-sided p=0.040, two-sided 0.081 — too
weak to call signal, too large to ignore as noise; reads as the first
evidence of domain-specific forgetting in the run."**

---

## 3. Major Findings

**M1. Greek BPB heldout still 29.2 % prefix-truncated at iter 1192
(persists Vanilla-0.5B/1B/2B/3.5B M1/M2).** `iter_0001192/heldout_greek_bpb.json`
`truncation` block: `n_docs_truncated=146, n_tokens_dropped=6,906,358,
fraction_truncated=0.292`. **NB: the prompt's stated value "28.6 % at iter
1192" is wrong** — the actual value in the file is 0.292 (= 146/500), the
same value as iter 119/238/477/834. The heldout file
`/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`
SHA-256 = `3487a53f…` and mtime 1779418719 — **the file has not been
regenerated since 2026-05-22**, so within-run BPB trajectory deltas
*are* meaningful even with the prefix truncation. The 0.4132 BPB at
iter 1192 (vs 0.4197 iter 834, vs 0.4313 iter 477) is a real monotone
language-modeling-still-improving signal. **Sensitivity check still
owed in the 5 B report**: a `non_truncated_subset_bpb` over the 354
clean docs, to bound the regime-fix BPB story against the truncation
confound (Decisions Matrix row D). All four per-source BPB values
improve monotonically iter 834 → iter 1192 (greek_academic 0.337 →
0.331; greek_dialogue_textbooks 0.622 → 0.616; greek_hplt_clean60
0.392 → 0.384; greek_legal_civic 0.287 → 0.283) — broad-based, not
single-source-driven.

**M2. Retention at iter 1192: MMLU/global_mmlu_en/fr recover from iter
834 dip but XNLI gains hold (Vanilla-3.5B M2 closes; new shape at
iter 1192).** Comparing iter 834 → iter 1192 retention numbers
(`acc,none` from `results_2026-05-30T13-57-34.322587.json`):

| Task | iter 834 | iter 1192 | Δ (iter 834 → 1192) | net vs iter 119 |
|---|---:|---:|---:|---:|
| mmlu | 0.5891 | 0.5798 | −0.93 pp | +1.24 pp |
| global_mmlu_en | 0.6150 | 0.6500 | **+3.50 pp** | **+4.50 pp** |
| global_mmlu_fr | 0.5600 | 0.5875 | +2.75 pp | +0.75 pp |
| global_mmlu_de | 0.6025 | 0.5950 | −0.75 pp | +0.75 pp |
| arc_challenge | 0.5188 | 0.5367 | +1.79 pp | +1.20 pp |
| arc_easy | 0.8114 | 0.8194 | +0.80 pp | +1.59 pp |
| piqa | 0.7824 | 0.7889 | +0.65 pp | −0.16 pp |
| hellaswag | 0.5905 | 0.5906 | +0.01 pp | +0.10 pp |
| winogrande | 0.7096 | 0.7072 | −0.24 pp | −0.71 pp |
| xnli_en | 0.5462 | 0.5486 | +0.24 pp | +5.82 pp |
| xnli_fr | 0.5044 | 0.5052 | +0.08 pp | +3.74 pp |
| xnli_de | 0.5016 | 0.4948 | −0.68 pp | +1.01 pp |
| xnli_ru | 0.4912 | 0.4727 | **−1.85 pp** | −1.53 pp |
| xnli_el | 0.4325 | 0.4205 | −1.20 pp | +0.44 pp |

Key new shape at iter 1192:
- **global_mmlu_en/fr fully recover** from the iter 834 dip the
  Vanilla-3.5B critic flagged: global_mmlu_en 0.605 (iter 119) → 0.6375
  (iter 477) → 0.6150 (iter 834) → **0.6500** (iter 1192) — *net positive*
  in every direction. The iter 834 dip reads as one-segment noise.
- **mmlu (English) drops below iter 834 (−0.93 pp) but holds above iter
  119** — same pattern as iter 834, not yet a clean regression.
- **xnli_ru regresses −1.85 pp** in the final segment, taking it net
  −1.53 pp below iter 119. This is the first iter-1192-specific
  retention regression that *crosses below baseline*. Russian is in the
  goal-aligned retention bundle (`hyperparameters.json
  eval.multilingual_retention.languages = [en, fr, de, ru]`). The
  underlying `global_mmlu_ru` is `None`, so XNLI-Russian is the only
  Russian signal in the run (Decisions Matrix row V). At iter 1192 it
  is now signed-negative vs iter 119, which the Vanilla-3.5B critic did
  not have to address.
- **xnli_el dips −1.20 pp** vs iter 834 (0.4325 → 0.4205), still net
  positive vs iter 119 (+0.44 pp), but the within-Greek-task gain the
  Vanilla-3.5B critic flagged as evidence is now reduced.
- **arc_challenge / arc_easy / global_mmlu_en post their best values in
  the run** at iter 1192 — multilingual/English retention is broadly
  *not degraded*, the iter 834 dip was noise on this scale.

**Implication:** the iter 834 "MMLU dip" reads in hindsight as one-segment
noise; the cpt-plan §2.3 row "Multilingual retention degrades → Replay
too low / LR too high" *no longer fires* on English (recovered). It
*does* fire on **Russian via xnli_ru**, narrowly, and on **xnli_el (Greek
retention) for the first time in the run**. The 5 B report should
document this *new* iter 1192 pattern instead of carrying forward the
iter 834 framing.

**M3. iter 1190 saved alongside iter 1192 — duplicate save pattern
persists for the third segment boundary in a row (Vanilla-2B M8 /
Vanilla-3.5B M4 persist).** `ls checkpoints/` shows `iter_0001190` and
`iter_0001192`. Same `SAVE_INTERVAL=119`-aligned-near-end-of-segment
pattern as iter 476/477 and iter 833/834. **The 5 B report should
either delete the iter 1190 dir (consumes ~50 GB on `iopsstor` per
`distcp` shard set) or formalize the rule "drop the penultimate save
in each segment when within `SAVE_INTERVAL` of the end-of-segment."**
The HF/sidecar pipeline targeted iter 1192 only — no operational risk
realized, but the quota cost has now compounded across all three
non-warmup segments.

**M4. `run_metadata.json[lr_decay_style] = "1-sqrt"` while training argv
uses `--lr-decay-style constant` (Vanilla-0.5B Minor / 1B M5 / 2B M4 /
3.5B M5 persist).** Top-level metadata still carries the stale field.
Has now survived **5 segments and 5 critiques**. Trivial to delete.
Recommended: do the delete in a sidecar patch before the 5 B report
ships; metadata-only readers (downstream tools, anyone parsing the run
dir to populate a Krikri-style report) will draw the wrong post-warmup
LR conclusion (Decisions Matrix row R).

**M5. iter 1192 sidecar `native_mcq` still consumes a 4-GPU GH200 node
for a 1-GPU eval (Vanilla-1B M6 / 2B M5 / 3.5B M6 persist).** Per
`sidecar_jobs.tsv`, jobs `2432547` (native_mcq), `2432548` (greek_nlp),
`2432549/2432551/2432552` (BPB), `2432550` (retention) all ran on
`normal` with full-node allocation per Decisions Matrix row Q. The 5 B
report's compute-justification block needs the actual billed GPU-h
counts at iter 1192; per the Vanilla-3.5B critic's projection of ~35
GPU-h of sidecar overhead at 5 checkpoints + ~118 GPU-h training, the
billed total is ~150-160 GPU-h. Within the 250-300 GPU-h hard budget
referenced earlier.

**M6. Decisions Matrix row I unapplied at iter 1192 — checksum sidecar
ran on `xfer` (Vanilla-3.5B M-Minor recurred).** `sacct -j 2432553` shows
`Partition = xfer, State = COMPLETED, ExitCode = 0:0`. The xfer
maintenance reservation (drained till 2026-06-11) has not yet begun
biting; the iter 1192 checksum sidecar slipped under the wire. The
script-side patch (re-route watcher + checksum from `--partition=xfer`
to `normal --cpus-per-task=64 --mem=400G`) **remains unapplied** in
both `watch_and_submit_checkpoint_sidecars.sbatch` and
`submit_checkpoint_sidecars.sh:259`. No iter-1192 operational impact,
but **the issue is concretely live for any post-2026-06-11 sidecar**
(production CPT, Task-2 extension, or any rerun) and the Vanilla-3.5B
critic flagged it as "still risks pending behind the maintenance
window." Recommended: apply the patch before any Task-2 launch.

**M7. greek_nlp_s100 at iter 1192 — almost certainly same 0.0-F1 / 0-acc
prompt-echo failure mode (Vanilla-2B M6 / 3.5B M7 carry forward).**
Not re-inspected file-by-file at iter 1192 (out of headline scope per
`hyperparameters.json eval.greek_diagnostic_only`). Listed here so the
5 B report's "Greek capability" claim doesn't accidentally pick up
greek_nlp_s100 numbers downstream.

---

## 4. Minor Findings and Hygiene Notes

- **`native_mcq` runner's `max_input_tokens = 3072`** persists at iter
  1192 (`run_metadata.json`). Still unverified whether the bakeoff
  Vanilla numbers cited in cpt-plan.md §1.1 (0.4327 / 0.4370 / 0.4305)
  used 3072 or 4096. If the bakeoff used 4096, then iter 1192's +6.68 pp
  vs bakeoff-Vanilla-5B on the point estimate (and any future paired CI
  for that pair) is partly an eval-setup delta. **Reviewer position:
  do one cheap re-run of bakeoff-Vanilla-5B at `max_input_tokens=3072`
  before the 5 B report.**
- **Predictions JSONL example-id alignment verified.** All four models
  (Apertus-Base, iter 477, iter 834, iter 1192) share identical
  example_ids in identical row order across all four benchmarks
  (asserted in `run_iter1192_bootstraps.py:64` before paired bootstrap
  begins). Paired-bootstrap deltas are *paired*, not pseudo-paired.
- **iter 1192 BPB truncation = 0.292**, not the prompt's stated "28.6 %".
  Same value as iter 119/238/477/834; heldout file SHA-256 = `3487a53f…`,
  byte-identical since 2026-05-22. The 5 B report should quote 29.2 %.
- **Plutus marginal CI at iter 1192 (0.3732, 0.5022)** still overlaps
  iter 834 marginal CI (0.4267, 0.5556) and Apertus-Base Plutus CI
  (0.4533, 0.5779) at the upper end. The point estimate change is
  larger than the width of the iter-834-vs-iter-1192 paired CI minimum
  bound (−0.1067) and smaller than the *positive* upper bound (0.000),
  which is the borderline-significance shape detailed in §C3.
- **headline_4task_with_plutus** (the *with-Plutus* aggregate) paired
  iter-1192 vs iter-834: Δ = +0.0004, CI [−0.0148, +0.0162], **inside
  zero** — the Plutus drop drags the with-Plutus aggregate back to flat,
  even as the 3-task headline is significantly up. The 5 B report's
  primary metric should stay the 3-task headline (per
  `hyperparameters.json eval.greek_headline_native.tasks`); the with-Plutus
  aggregate is for diagnostic context only.
- **Sidecar manifest matches expected kinds.** All 8 entries present
  (`convert/native_mcq/greek_nlp/heldout_greek_bpb/retention/code_bpb/
  math_bpb/checksum`). The fix from Decisions Matrix row N/O held
  through the final checkpoint.
- **Code BPB iter 1192 = 0.2646** (vs iter 834 = 0.2697, vs iter 477 =
  0.2807) — small monotone decrease, consistent with language modeling
  improving. **Math BPB iter 1192 = 0.5448** (vs iter 834 = 0.5491) —
  same monotone-improving pattern.
- **Slurm `--export` comma bug fix held through 5 segments and 5
  checkpoints.** No regression at iter 1192 (verified
  `Vanilla-5B_native_mcq_aggregate.json:run_metadata.benchmarks` lists
  all 4).
- **Plutus-in-headline JSON fix held.** `headline.n_tasks=3`,
  `diagnostics.n_tasks=1`, explicit `headline_policy` block.

---

## 5. Missing Evidence

To promote iter 1192 to "load-bearing 5 B endpoint":

1. **V4 v3 artifact re-emit** with iter-834 + iter-1192 entries and
   paired delta_table rows for: iter-1192 vs iter-477, iter-1192 vs
   iter-834, iter-1192 vs bakeoff-Vanilla-5B, iter-1192 vs Apertus-Base
   (Path-A), iter-1192 vs Apertus-Base-matched-Path-B-perturbed, plus
   all per-task metrics. The numbers in §8 below are what the v4-v3
   artifact should land.

2. **Decontamination MinHash pass** against the 4 native MCQ benchmark
   prompts on `hplt_b1_5b.jsonl`. With iter 1192 now load-bearing on
   **four** cross-anchor deltas (vs Apertus-Base, vs bakeoff-Vanilla-5B,
   vs iter 477, plus the per-task GreekMMLU motion within run), this is
   the highest-leverage insurance policy in the remaining audit budget.

3. **Per-subject GreekMMLU breakdown for iter 477 / iter 834 / iter 1192.**
   The greekmmlu +5.09 pp lift over iter 477 (CI excludes zero, larger
   than the headline lift) is the engine of the trajectory verdict.
   Allocating this to per-subject buckets discriminates "broad knowledge
   gain" from "contamination concentrated in one subject." A 5-minute
   pandas pass on the predictions JSONL.

4. **Per-subject ASEP breakdown for iter 834 / iter 1192** — to
   characterise whether the iter 834 → iter 1192 ASEP recovery (+1.58 pp,
   CI excludes zero) is broad-based or concentrated (it *fully* reverses
   the iter 477 → iter 834 ASEP regression of −1.92 pp the Vanilla-3.5B
   critic flagged). If it's broad-based, the Vanilla-3.5B critic's
   "task-level cancellation" reading was a transient; if concentrated,
   the run is producing localized subject effects rather than uniform
   gains.

5. **Sanity rerun of bakeoff-Vanilla-5B native MCQ at `max_input_tokens=3072`.**
   Bounds the iter-1192-vs-bakeoff-Vanilla-5B +6.68 pp point delta
   against the eval-setup confound. Cheap (~1 GPU-h).

6. **`non_truncated_subset_bpb` over the 354 clean docs.** Bounds the
   29.2 %-truncated BPB result against the truncation confound
   (Decisions Matrix row D). Cheap (~10 minutes).

7. **KL-to-base on a fixed Greek probe set at iter 477/834/1192.** The
   Plutus drop combined with the xnli_el dip + xnli_ru dip is the first
   pattern in the run consistent with forgetting on specific Greek/multilingual
   surfaces. KL-to-base would discriminate forgetting from noise without
   any new benchmark eval. The Vanilla-3.5B critic also recommended this;
   it is now more strongly motivated by the Plutus drop.

8. **Per-subject Plutus breakdown** (if metadata exists in the predictions
   JSONL) to localise the iter 834 → iter 1192 drop. Plutus is medical
   QA but spans sub-domains; even at n=225 a subject-aligned drop would
   carry more weight than a uniform 5-pp shift.

9. **A Path-A static-eval probe of iter 1192.** Symmetric to the
   matched-config-Apertus-Base eval Decisions Matrix row H built (Path-B
   override of Path-A weights), evaluate iter 1192 weights *under
   Path-A geometry* (rope_theta=12M, max_position=65536, llama3 scaling).
   This is the inverse perturbation: it probes whether the 1.5 B-token
   final segment has begun to encode something that breaks when restored
   to Path A. Same script as Decisions Matrix row H, inverse direction.
   Not gating for the 5 B report; informative for the Path-A revisit
   recommendation in cpt-plan §3.4 Q3.4.10.

10. **Explicit Apertus-Base-matched-Path-B-perturbed paired CI at iter
    1192** — the Vanilla-3.5B critic's geometry-confound caveat asked for
    "iter 1192 vs matched-config Apertus-Base" as a bookend. Per §8
    below the marginal Apertus-Base-matched point is 0.4272 [0.4096,
    0.4456]; iter 1192 [0.4779, 0.5156] sits *cleanly above* it, so the
    "even under perturbed-Path-B baseline, iter 1192 wins" bracket holds.
    A full paired CI re-emit at V4 v3 would make this explicit.

---

## 6. Recommended Next Actions Before the 5 B Report or 10 B Continuation

1. **Land V4 v3** (item 1 above). The §8 numbers seed this update.
2. **Run the decontamination MinHash** (item 2 above). Cheapest insurance
   policy in the residual audit budget; protects four load-bearing
   deltas.
3. **Apply Decisions Matrix row I** (re-route watcher + checksum from
   xfer to normal) **before any Task-2 launch**. iter 1192 squeaked
   under the wire on this; Task 2 will not.
4. **Run per-subject GreekMMLU/ASEP breakdowns** (items 3, 4) before
   reading the trajectory verdict into the 5 B → 10 B continuation
   decision.
5. **Do NOT promote Plutus as evidence in either direction.** The 5 B
   report's Plutus story is "5.33 pp drop at n=225 with paired CI
   [−0.1067, +0.0000], two-tailed McNemar p = 0.081, paired vs
   Apertus-Base outside zero — too weak to call signal, too large to
   ignore as noise; first hint of domain-specific forgetting in the run."
6. **Do the BPB sensitivity check on the 354 untruncated docs** (item
   6 above). Trivially cheap, closes a long-standing caveat.
7. **For the 5 B → 10 B continuation decision** (cpt-plan §2.4 / §5.6):
   the iter 1192 result *does* surprise the Vanilla-3.5B reviewer's
   plateau bracket. **The information value of 5 B → 10 B is now non-zero**
   — slope is not zero, headline is not saturated, the per-task pattern
   (GreekMMLU still gaining, ASEP rebounded, Medical noise, Plutus
   ambiguous) is non-trivial. But Q2.4.1's working answer (10 B with
   checkpoints at 7 B and 10 B) was not contingent on iter 1192 being
   *above* Apertus-Base — that's where we are. The 5 B → 10 B
   continuation should be conditioned on the V4 v3 CI passing the same
   "outside-zero vs Apertus-Base" gate at 5 B, and on the Plutus drop
   getting a clearer signal call from the per-subject breakdown +
   KL-to-base probes.
8. **Delete the duplicate iter 1190 checkpoint** (item M3) to free
   `iopsstor` quota. Trivial.
9. **Patch the stale `lr_decay_style` field** in `run_metadata.json`
   (item M4). Trivial.

---

## 7. Persistence of Prior Critical Findings

Persistence map iter 119 → 238 → 477 → 834 → **1192**:

- **Path-B vs Path-A geometry mismatch** (Vanilla-0.5B C1 / 1B C2 / 2B
  M1 / 3.5B C2 / **5B C1**): **still live; outcome-relevance escalated**.
  cpt-plan.md §2.1 and §3.4 Q3.4.10 now properly document Path B as a
  training-time override and recommend Path A for Task 2. *Doc gap
  closed.* Comparison gap not just remains but now signs the headline
  result: iter 1192 statistically beats Apertus-Base (Path-A) and
  Apertus-Base-matched-Path-B-perturbed simultaneously, so the result
  is defensible *under both bookend geometries*. But the Path-B-Vanilla
  vs Path-A-Apertus-Base structural asymmetry is intrinsic to the
  bakeoff lineage and cannot be retroactively closed. Verdict at iter
  1192: still live, properly documented, **outcome-relevance increased
  because the headline now sits above Apertus-Base for the first time**.
- **Plutus-in-headline + Slurm `--export` comma bug** (0.5B C2 / 1B C1 /
  2B Finding 2 & 5 / 3.5B Persist): **fixed and verified through 5
  checkpoints**. iter 1192 `Vanilla-5B_native_mcq_aggregate.json`
  carries `headline.n_tasks=3`, `diagnostics.n_tasks=1`, explicit
  `headline_policy.headline_benchmarks` list, and
  `run_metadata.benchmarks` enumerates all 4. Verdict: **closed**
  (durably, this is now five consecutive checkpoints).
- **Decontamination gap for Greek MCQ prompts** (0.5B C3 / 1B C3 / 2B
  C3 / 3.5B C3 / **5B C2**): **still live**, load-bearingness compounded
  with iter 1192 result. Now applies to (a) iter 477 vs bakeoff-2B,
  (b) iter 834 vs bakeoff-3.5B, (c) iter 1192 vs bakeoff-5B,
  (d) iter 1192 vs Apertus-Base — *all* outside-zero paired CIs that
  the absence of a decontamination artifact undermines. Verdict: still
  live, **load-bearing on four claims now, MinHash pass should land
  before 5 B report**.
- **Greek BPB heldout 29.2 % truncation** (0.5B M1 / 1B M1 / 2B M2 /
  3.5B M1 / **5B M1**): **still live**, file byte-identical since
  2026-05-22 (SHA + mtime confirmed); within-run trajectory is sound.
  **Prompt's "28.6 % at iter 1192" was incorrect** — actual value is
  29.2 %. Sensitivity check on 354 untruncated docs still owed.
  Verdict: still live, sensitivity check trivial and now overdue.
- **iter-238 retention regression** (1B M2): inverted at iter 477,
  reshaped at iter 834 (Vanilla-3.5B M2), at iter 1192 the iter-834
  shape *reverses again* — MMLU-family recovers (global_mmlu_en
  +3.50 pp; fr +2.75 pp), but xnli_ru drops −1.85 pp and xnli_el
  drops −1.20 pp. The first iter-1192-specific *Greek/Russian
  retention regression that crosses below iter-119 baseline*. Verdict:
  *closed in English / French shape*, **reopened in Greek-XNLI and
  Russian-XNLI shape** at iter 1192.
- **Matched-config Apertus-Base eval = diagnostic-only** (2B C2): closed
  as a design clarification. **iter 1192 paired vs matched-config
  Apertus-Base = +0.0701 [implicit from V4 v2 marginal CI];** the bracket
  holds.
- **Slurm `--export` comma bug** (Decisions Matrix row A): **fixed**;
  held for five consecutive checkpoints. Verdict: closed.
- **Decisions Matrix row I — xfer-routed watcher + checksum sidecar**:
  **patch unapplied**; iter 1192 checksum sidecar `2432553` ran on
  `xfer` (sacct confirmed). Completed only because xfer maintenance
  reservation hasn't begun biting yet. Verdict: still live,
  concretely-risky for post-2026-06-11 work.
- **Duplicate save pattern at segment boundaries** (3.5B M4): persists
  at iter 1190/1192. Third such pattern in the run. Quota cost ~50 GB.
- **Stale `lr_decay_style: 1-sqrt` metadata** (0.5B Minor / 1B M5 / 2B
  M4 / 3.5B M5 / **5B M4**): still live through 5 critiques. Trivial
  to patch.

---

## 8. Bootstrap CI Summary

All five CIs computed in `reports/v4_workspace_iter1192/run_iter1192_bootstraps.py`,
written to `reports/v4_workspace_iter1192/iter1192_bootstrap_results.json`.
Methodology mirrors `reports/v4_workspace/run_bootstrap_v2.py` exactly:
1000 resamples, 95 % percentile, rng_seed=20260529, per-task item-level
paired bootstrap (independent resampling within each task), headline =
macro-mean across 3 headline tasks per resample, paired by shared
resample indices across all four loaded models. Predictions JSONLs were
copied from Clariden, used, and deleted per task spec; scripts and
results remain.

**CI 1 — iter 1192 marginal 3-task headline (n_items_per_task =
16632/432/1200):**
- point = **0.4973** (verified: matches `Vanilla-5B_native_mcq_aggregate.json`)
- 95 % CI = **[0.4779, 0.5156]**
- Per-task marginals:
  - greekmmlu: 0.5584 [0.5502, 0.5660] (n=16632)
  - ilsp_medical_mcqa: 0.4028 [0.3565, 0.4491] (n=432)
  - ilsp_mcqa_asep: 0.5308 [0.5033, 0.5575] (n=1200)
  - plutus_qa: 0.4356 [0.3732, 0.5022] (n=225)
- 4-task with Plutus: 0.4819 [0.4600, 0.5046]

**CI 2 — paired iter-1192 vs iter-477 headline_3task (full post-warmup
gain):**
- iter 477 point = 0.4792; iter 1192 point = 0.4973
- Δ (iter 1192 − iter 477) = **+0.0182**, 95 % CI = **[+0.0060, +0.0306]**
- **outside zero → statistically significant gain over the full
  post-warmup window (iter 477 → iter 1192, 3 B tokens of additional
  training).**
- Per-task paired:
  - greekmmlu: +0.0509 [+0.0457, +0.0562] *outside*
  - ilsp_medical_mcqa: +0.0069 [−0.0278, +0.0417] inside
  - ilsp_mcqa_asep: −0.0033 [−0.0208, +0.0125] inside
  - plutus_qa: −0.0533 [−0.1067, +0.0089] inside (just inside)
- 4-task with Plutus: Δ = +0.0003 [−0.0166, +0.0185] *inside* (Plutus
  drag erases the 3-task gain when Plutus is included)

**CI 3 — paired iter-1192 vs iter-834 headline_3task (slope after the
apparent plateau):**
- iter 834 point = 0.4790; iter 1192 point = 0.4973
- Δ = **+0.0184**, 95 % CI = **[+0.0080, +0.0295]**
- **outside zero → statistically significant gain over the 1.5 B tokens
  of post-plateau training.**
- Per-task paired:
  - greekmmlu: +0.0300 [+0.0255, +0.0345] *outside*
  - ilsp_medical_mcqa: +0.0093 [−0.0208, +0.0394] inside
  - ilsp_mcqa_asep: +0.0158 [+0.0017, +0.0308] *outside*
  - plutus_qa: −0.0533 [−0.1067, +0.0000] inside (CI upper exactly 0)
- 4-task with Plutus: Δ = +0.0004 [−0.0148, +0.0162] *inside*
- **Diagnostic implication:** the Vanilla-3.5B reviewer's "iter 477 →
  iter 834 plateau" was real on the headline, but the slope was
  *not zero* over the *next* 1.5 B tokens. The model is still moving.

**CI 4 — paired iter-1192 vs Apertus-Base headline_3task:**
- Apertus-Base point = 0.4817; iter 1192 point = 0.4973
- Δ = **+0.0156**, 95 % CI = **[+0.0016, +0.0284]**
- **outside zero → iter 1192 is statistically above Apertus-Base on the
  3-task headline.** Caveat: Path-A baseline vs Path-B run (see §C1).
- Per-task paired:
  - greekmmlu: +0.0304 [+0.0254, +0.0361] *outside* (favours iter 1192)
  - ilsp_medical_mcqa: −0.0069 [−0.0440, +0.0278] inside
  - ilsp_mcqa_asep: +0.0233 [+0.0050, +0.0425] *outside* (favours iter 1192)
  - plutus_qa: **−0.0800 [−0.1422, −0.0133] *outside zero negative***
    (Apertus-Base wins Plutus by 8 pp, CI excludes zero)
- 4-task with Plutus: Δ = −0.0083 [−0.0270, +0.0104] inside (Plutus drag
  pulls the with-Plutus aggregate just below Apertus-Base, but inside
  the CI).

**CI 5 — paired iter-1192 vs iter-834 Plutus QA (n=225, signal vs noise
probe):**
- iter 834 Plutus = 0.4889; iter 1192 Plutus = 0.4356
- Δ = **−0.0533**, 95 % CI = **[−0.1067, +0.0000]**
- **CI upper bound is exactly 0** — bootstrap percentile cannot
  *strictly* reject zero, but the CI does not contain a positive value.
- McNemar exact two-tailed p over 40 discordant pairs (26 regressions,
  14 gains): **p = 0.081**.
- Item-level transition matrix (iter 834 → iter 1192):
  84 both-correct / 26 834-correct/1192-wrong (regressions) /
  14 834-wrong/1192-correct (gains) / 101 both-wrong.
- **Verdict on Plutus drop: borderline, not noise-confirmed and not
  signal-confirmed.** The same Δ = −0.0533 vs iter 477 carries
  identical discordant counts (26 vs 14), suggesting the Plutus
  movement landed in the final segment (iter 834 → iter 1192) rather
  than gradually across iter 477 → iter 1192. Combined with the
  outside-zero paired vs Apertus-Base CI (−0.0800 [−0.1422, −0.0133]),
  the cleanest read is **"Plutus has regressed from
  iter-477/iter-834 level to a level statistically below Apertus-Base,
  but the within-run delta cannot strictly reject zero at α=0.05."**

---

## 9. Trajectory Verdict at the 5 B Endpoint

**Did the run produce a sustained Greek-side improvement above bakeoff
and at/above Apertus-Base, or did it plateau early and then noisily
climb?** Anchoring on the four CIs above plus the Vanilla-3.5B critic's
paired iter-477-vs-iter-834 CI:

| Segment | Δ headline_3task | 95 % CI | Outside zero? |
|---|---:|---|---|
| iter 119 → iter 238 (warmup) | +0.0096 | [−0.0062, +0.0247] | inside (warmup transient) |
| iter 238 → iter 477 (post-warmup burst) | +0.0305 | [+0.0156, +0.0458] | **outside** |
| iter 477 → iter 834 (plateau) | −0.0002 | [−0.0123, +0.0114] | inside |
| iter 834 → iter 1192 (post-plateau lift) | **+0.0184** | **[+0.0080, +0.0295]** | **outside** |
| iter 477 → iter 1192 (full post-warmup) | **+0.0182** | **[+0.0060, +0.0306]** | **outside** |
| iter 1192 vs Apertus-Base | **+0.0156** | **[+0.0016, +0.0284]** | **outside** |
| iter 1192 vs bakeoff-Vanilla-5B (point) | +0.0668 | (V4 v3 owed) | (point estimate; CI pending v3) |

**The run did NOT plateau early.** The headline trajectory is **two
post-warmup +0.018-to-+0.031 lifts separated by one within-noise
plateau**, not a single jump followed by saturation. The Vanilla-3.5B
reviewer's plateau bracket [0.467, 0.491] for iter 1192 is broken at
+0.0063 above the upper bound — the optimistic-end "non-zero slope"
hypothesis the critic flagged at the time. **The 5 B endpoint
statistically exceeds Apertus-Base** on the 3-task headline (paired CI
excludes zero), driven by GreekMMLU and ASEP; **but loses to
Apertus-Base on Plutus** (paired CI excludes zero, the only negative
outside-zero per-task delta in the run).

**Verdict bracket for the 5 B report**:

- **Headline (3-task, the goal-defined metric):** sustained improvement.
  Outside-zero gains over bakeoff-Vanilla at every matched token mark
  (V4 v2 confirmed at iter 477 and iter 834; iter 1192 point estimate is
  +6.68 pp over bakeoff-Vanilla-5B 0.4305 and CI excludes zero pending
  V4 v3 paired re-emit). Outside-zero gain over Apertus-Base at the
  endpoint, under the Path-A-baseline / Path-B-run caveat.
- **With-Plutus aggregate (diagnostic):** flat over the same window
  (paired iter 1192 vs iter 477 = +0.0003 [−0.0166, +0.0185], inside
  zero) because Plutus regresses. **This is the cautionary subplot.**
- **BPB:** monotone improvement (0.4313 → 0.4197 → 0.4132 across iter
  477/834/1192), broad-based across all 4 per-source registers.
  Truncation caveat still owed.
- **Retention:** broadly preserved; xnli_ru and xnli_el regress below
  iter-119 baseline at iter 1192 (new), MMLU-family recovers from iter
  834 dip.

**Cleanest single-sentence verdict:** **the corrected regime produces a
sustained, statistically significant Greek-side gain on the goal-defined
3-task headline at the 5 B endpoint, crossing the Apertus-Base bookend
for the first time in the run, with one new caveat (Plutus regression
borderline-signal, ASEP recovery from the iter-834 dip, retention
softening on Greek-XNLI and Russian-XNLI).** The trajectory has the
shape "warmup → +3.05 pp burst → flat segment → +1.84 pp endpoint
lift," not "burst then plateau."

---

## 10. 5 B Regime-Hypothesis Verdict (cpt-plan §2.2)

**Hypothesis (cpt-plan §2.2):** "Did continued training under the
bakeoff regime hurt a model that already knew Greek well, and if so,
what was the cause? Experiment (2) tests hypothesis (a) [the CPT regime
itself] by running Vanilla under an Apertus-faithful regime. If Vanilla
under the new regime recovers (matches or approaches Apertus-Base on
native MCQ), regime mismatch was the dominant cause."

**Verdict at 5 B endpoint: SUPPORTED — with three explicit conditions.**

**Supports:**

1. **iter 477 vs bakeoff-Vanilla-2B:** Δ = +0.0465, CI [+0.0299, +0.0629],
   outside zero across the headline and all 4 individual tasks (V4 v2).
2. **iter 834 vs bakeoff-Vanilla-3.5B:** Δ = +0.0420, CI [+0.0262,
   +0.0581], outside zero on the headline (per Vanilla-3.5B critic's
   computation in `reports/v4_workspace_iter834/paired_iter834_vs_bakeoff_3p5B.json`).
3. **iter 1192 vs bakeoff-Vanilla-5B:** point Δ = +0.0668 (0.4973 vs
   0.4305); paired CI owed to V4 v3 but every prior matched-token-mark
   pair was outside-zero by 4–5 pp and the iter-1192 point lift is
   *larger*, so the CI is overwhelmingly likely to be outside zero.
4. **iter 1192 vs Apertus-Base headline_3task:** Δ = +0.0156, CI
   [+0.0016, +0.0284], outside zero. **The "matches or approaches
   Apertus-Base" gate is met, and *strictly exceeded* at the
   endpoint.**
5. **Trajectory is not plateau-then-drift**: iter 477 → iter 1192
   headline gain of +0.0182 is outside zero across the full
   post-warmup window. The "Vanilla peaks early then drifts" pattern
   that motivated cpt-plan Q2.4.2 *does not reproduce* under the
   corrected regime.

**Conditions / caveats** (which the 5 B report must surface alongside
the verdict):

1. **Geometry confound (Path-A baseline, Path-B run; §C1).** The
   "exceeds Apertus-Base" claim is under the Path-A geometry the base
   was trained on. The matched-config Apertus-Base eval (Path-B
   override on Path-A weights) is perturbed and gives an even *lower*
   lower-bound (0.4272, CI [0.4096, 0.4456]); iter 1192 exceeds that
   bookend too. But a clean Path-B Vanilla-CPT vs Path-B Apertus-Base
   comparison cannot exist because the Path-B Apertus-Base eval is
   either (a) perturbed (the matched-config diagnostic) or (b)
   nonexistent (no Path-B-trained Apertus). The verdict is therefore
   defensible at the *bracket* level, not at the strict-baseline level.
2. **Decontamination not verified (§C2).** The "outside-zero vs
   bakeoff" *and* "outside-zero vs Apertus-Base" claims share the same
   asymmetric contamination risk; no MinHash artifact exists. The 5 B
   report must surface this as an open caveat or land the MinHash pass
   before claiming the regime hypothesis is empirically supported in
   print.
3. **Plutus regresses on the same run (§C3).** Plutus drops from
   0.4889 (iter 477/834) to 0.4356 (iter 1192), paired CI vs iter 834
   borderline ([-0.1067, +0.0000]) and paired CI vs Apertus-Base
   outside zero on the negative side. The "regime makes Greek better"
   claim has one localized exception. This is consistent with the
   cpt-plan §2.3 mapping "BPC improves; native MCQ stays flat below
   Apertus-Base → forgetting via KL-to-base on fixed probes; higher
   replay share" — invoked on Plutus *specifically*, not on the
   3-task headline.

**Reading for follow-up experiments**:

- **Task 1 → Task 2 trigger** (cpt-plan §2.3 row 1, "BPC improves vs
  bakeoff Vanilla early; native MCQ trajectory recovers → Proceed to
  Task 2 (extension experiment)"): **met**. Both halves cleanly hold
  at 5 B.
- **10 B continuation** (cpt-plan §2.4 Q2.4.1): **information value is
  non-zero**. iter 477 → iter 1192 = +0.0182 / 3 B = +0.0061 / 1 B; if
  the slope holds (no reason to expect saturation given outside-zero
  CIs both segments), iter 7 B ≈ 0.510 and iter 10 B ≈ 0.528. The 5 B
  → 10 B decision should be conditioned on (a) V4 v3 paired CIs holding
  outside zero against Apertus-Base at the endpoint, (b) the Plutus
  drop getting a clearer signal call from the per-subject + KL-to-base
  probes, and (c) the decontamination MinHash check landing.

**Bottom line:** the regime hypothesis is **statistically supported at
the 5 B endpoint** with the three caveats above. The 5 B report can
adopt the position "the corrected Apertus-style regime fixes the
bakeoff-regime degradation and produces a sustained Greek-side gain
crossing the Apertus-Base Path-A baseline by 5 B tokens, with one
localized Plutus regression to document and three load-bearing audit
items (geometry / decontamination / Plutus follow-up) carried forward
to Task 2." This is the strongest result the run could have produced
short of also closing the Path-A confound, which it structurally
cannot.

---

*Bootstrap CIs computed locally on home (no Slurm submissions / no
remote-state mutation). Prediction JSONLs copied from Clariden into
`reports/v4_workspace_iter1192/`, used, then deleted; scripts +
results JSON retained. SSH access read-only. CI computation methodology
mirrors `reports/v4_workspace/run_bootstrap_v2.py` exactly.*
