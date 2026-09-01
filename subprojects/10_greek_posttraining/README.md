# 10 — Greek post-training: SFT data survey and adaptation pilots

> **In one line:** a two-day survey of what Greek instruction-tuning data actually exists, a
> proposed minimal SFT mix and build pipeline for it, and two 100-row pilots of a
> "translate the prompt, write the answer natively" adaptation of `HuggingFaceH4/no_robots` into
> Greek — which produced a measured adaptation spec and a list of open owner decisions, and stopped
> there in this repo.
> **Period:** 2026-08-23 → 2026-08-24 (from file mtimes and the dates the documents carry; there is
> no commit history — see note 2). **Status:** ended at the pilot-2 stage. No SFT training was run,
> no corpus was built, and the licence/register/glossary decisions were still open.
> **Came from / led to:** [`../09_full_8b_cpt_results_analysis/`](../09_full_8b_cpt_results_analysis/README.md)
> (which supplied the checkpoint candidates this phase would have SFT'd) → this → out of this repo.

## Three notes before reading

1. **The directory number collides.** [`../10_early_cooldown_causal_experiment/`](../10_early_cooldown_causal_experiment/README.md)
   also carries the number 10 and is an unrelated subproject (a causal LR-schedule branch off the
   40 B checkpoint). Nothing here has been renamed. The collision is explained by
   [`DATA_SURVEY_GREEK_SFT_20260823.md`](DATA_SURVEY_GREEK_SFT_20260823.md) line 8, which says this
   subproject "takes 10 to avoid collision" with `09_*` — the author was avoiding a different
   collision and did not know about the other 10. Only `10_early_cooldown_causal_experiment/` is
   listed in [`../SUBPROJECTS_OVERVIEW.md`](../SUBPROJECTS_OVERVIEW.md); this directory is not.
2. **There is no development history for this directory.** All 139 files arrive in a single commit,
   `2aec4a66` (2026-09-01, "Recover uncommitted working-tree files from local worktrees") — they were
   recovered from the owner's uncommitted working tree, so no commit records what was written when.
   Every date below therefore comes from a date written inside
   a document, from a filename (`*_20260823.md`), or from a file mtime — and the mtimes are coarse:
   all eleven pilot-1 output files share one minute (2026-08-23 20:29) and all 122 pilot-2 files
   share another (2026-08-24 03:23), which is what a bulk copy looks like. Treat within-evening
   ordering as approximate.
3. **Continuation.** Per the owner's notes this line of work continued in a separate private
   repository, "natural-greek-sft", renamed 2026-08-24 *(owner note, not verified in-repo — no
   document, script or comment in this directory mentions such a repository; the only GitHub URLs
   here are `github.com/swiss-ai/posttraining` and `github.com/swiss-ai/apertus-format`)*.

## Why this existed

The 8 B Greek CPT run (subproject 07) and its evaluation (subproject 09) produced a base model, not
a usable assistant. This phase asked the next question: what would it take to post-train that
checkpoint into a Greek instruct model? The first finding governs everything after it — **Greek SFT
is greenfield**. ILSP released none of the SFT or preference data behind Meltemi and Llama-Krikri,
and the total reusable quality Greek material anywhere is roughly 3.5 k rows
([`DATA_SURVEY_GREEK_SFT_20260823.md`](DATA_SURVEY_GREEK_SFT_20260823.md) §0, §2.2). So the corpus
had to be built, and the work turned into a data-engineering problem: which English sources to adapt,
who translates and who generates, what a gate can and cannot check in Greek, and how much a single
human reviewer can actually get through.

## History

### 2026-08-23 — the survey

[`DATA_SURVEY_GREEK_SFT_20260823.md`](DATA_SURVEY_GREEK_SFT_20260823.md) (mtime 15:54, self-dated
2026-08-23, status "SURVEY / RECOMMENDATION — no decisions locked, no builds started") was assembled
from three parallel research agents, each verifying dataset ids, row counts and licences against the
HF hub that day. Its findings: the Krikri paper (arXiv 2505.13772) is the proven blueprint —
translate prompts, **regenerate responses natively**, Magpie natively in Greek, ground synthetic QA
in Greek corpora; `swiss-ai/apertus-sft-mixture` (3,942,208 rows, ODC-BY) is reusable as the
non-Greek backbone but carries only ~3,215 Greek rows (0.082%); no public Greek preference data
exists at all (largest find: 30 DPO pairs). It ranked eight translation candidates with licences
verified, proposed a four-stream corpus shape, and closed with five open owner decisions (licence
posture, register policy, translator/generator choice, stream-B scale, checkpoint A/B). Its §5
argued for initialising SFT from the benchmark-best *averaged* checkpoint rather than the lowest-loss
one, citing a peak single checkpoint of 56.81% GreekMMLU at iter 9,536 — a figure that
[`../09_full_8b_cpt_results_analysis/README.md`](../09_full_8b_cpt_results_analysis/README.md)
independently confirms.

[`MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md`](MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md) (mtime 19:10,
status "RECOMMENDATION — nothing locked, nothing built") answered the *how*. It sized a floor mix
(~62 k rows, ~51% Greek) and a comfortable mix (~161 k, ~50% Greek) bucket by bucket; picked
Gemma-3-27B-IT as generator, Krikri-8B-Instruct-v1.5 as translator and `rubricreward/mR3-Qwen3-8B`
as the bulk judge, each with the licence cost named; and established that **compute is not the cost**
($22–$70 of GPU time versus a human-review budget 10–100× that). Its sharpest technical results are
negative ones: the stock `google-research/instruction_following_eval` checker is **broken on Greek in
four distinct ways** and actively rewards orthographically wrong Greek (§11a); `rouge_score` scores
two identical Greek instructions at 0.0 because it tokenises on `[^a-z0-9]+`, so the canonical
Self-Instruct diversity filter silently accepts everything (§11b); `distilabel` is
author-abandoned and Argilla is too (§10b, §12); and by the doc's own estimate one reviewer clears
~38 items/hour, so 10 k reviewed rows ≈ 12 weeks and 40 k ≈ a year (§12). §13a records the one
**owner decision** taken that day: route rows by their source dataset's own metadata into **eight
translation classes**, assigned in stage 1 and carried into stage 2.

### 2026-08-23 evening — pilot 1, the stress sample

[`pilot_no_robots_100/`](pilot_no_robots_100/README.md) executed §15's step 2: 100 `no_robots` rows
through classify → translate → generate, both stages on Claude Opus 5 at medium reasoning effort,
nothing hand-edited between them. The sample was deliberately **stratified, not proportional** —
20 rows each from Generation, Open QA, Chat, Rewrite and Closed QA, chosen to stress different
failure modes. Results: 99/100 passed the hard gates, 88/100 with no advisory either
(recomputed from [`pilot_no_robots_100/out/gated.json`](pilot_no_robots_100/out/gated.json)); zero
failures on NFC, homoglyphs, accented all-caps, polytonic bleed and identity leakage. The gate suite
itself was wrong three times, and **every sigma hard-flag in all three versions was a false
positive**. The bake-off of §10 (Gemma vs DeepSeek vs Qwen3 vs Krikri) was never run — pilot 1 used
Opus for both stages instead.

### 2026-08-24 — the owner review, and the reversal it forced

The owner reviewed **9 of the 100 rows** — all of them Chat, which is what made the sampling problem
visible — and the exchange was logged in
[`pilot_no_robots_100/FEEDBACK.md`](pilot_no_robots_100/FEEDBACK.md) (mtime 00:24) rather than fixed
row by row. Seven owner findings (F1–F7) and seven observations (O1–O7) came out of nine rows. Two
changed the plan rather than the prompts:

- **D1 (owner decision, 2026-08-24):** the target is not a translation of `no_robots` but
  **Greek-native instruction data seeded by it** — invented frames relocated into Greek reality,
  completely or not at all, governed by one test: *would swapping this entity change whether the
  answer is true? Yes → content, freeze it; no → frame, transpose it.* Measured scope over the 100:
  20 transposable, 52 locked by a real entity, 12 locked by a source text, 16 neutral — and 19 of the
  20 transposable rows are Chat or Generation.
- **F7:** the pilot's own frequencies do not describe the corpus. Closed QA was over-represented
  7.75×, Chat 2.39×, Generation *under*-represented at 0.44× despite being 45.8% of the data, and
  five categories (24.3%) were absent entirely — including Coding, the natural home of
  `VERBATIM-FREEZE`. Re-weighted to corpus level, **`CONSTRAINT-PRESERVING` is the largest class at
  ~37%, not `LITERAL`** — the pilot had inverted this, which re-prioritised the (broken) Greek
  constraint checker from side issue to highest-leverage piece.

The measurements that came out of the review are the durable part: 4.94% of tokens are Latin-script
but only 97 distinct lowercase Latin tokens exist, so the vocabulary is small enough to legislate
term by term (F2); `chatbot` was chosen 30 times with zero alternatives ever considered (F4);
person names were **localised in Generation/Rewrite and transliterated in Chat** — 6 vs 28, with
nothing asking for that split (F6); 38 rows made a transliteration call against no standard (O4);
7 rows had to invent a grammatical gender English left open (O3); and the pipeline **silently
corrected errors in the human-written English reference in 8 rows** (O2).

Two independent attempts at cheap automated detection failed for the same reason: the register gate
cannot separate formality from grammatical number (the Three Little Pigs row, where the wolf
addresses one pig then two), and a regex scan for marked word order returned 1 true positive in 11
hits while missing two of the three known cases (F5, O7). The review's conclusion —
**the human budget should be spent on naturalness, not correctness**, because the deterministic gates
already handle correctness at 99/100 and are structurally incapable of seeing naturalness.

### 2026-08-24, small hours — the spec, the causes, and pilot 2

[`TRANSLATION_SPEC_v1.md`](TRANSLATION_SPEC_v1.md) (mtime 01:38) turned the findings into rules,
adding a structural analysis of all 9,499 rows of the train split. Its organising move: **two
families before ten categories** — generative (Generation, Open QA, Chat, Brainstorm, Coding) vs
source-bearing (Rewrite, Closed QA, Summarize, Classify, Extract), separated cleanly by
answer-content-token recall in the prompt (0.00–0.10 vs 0.53–0.88). Five of the ten categories were
profiled without ever having been piloted, and the spec says so on each.

[`ROOT_CAUSES_v1.md`](ROOT_CAUSES_v1.md) (mtime 02:21, status "for discussion — nothing here has
been applied") reduced the fifteen findings to eleven causes. RC1 is the root of most of them: *the
prompt has no memory from one row to the next*, so "consistent" and "arbitrary" are
indistinguishable from inside the process — fixed by a style guide injected into every call plus
`guide_applied` / `guide_gaps` fields that make drift a worklist instead of an invisible process.

[`pilot_v2/`](pilot_v2/README.md) (all files mtime 03:23) then ran RC11's prescription: **sample A**,
the 9 reviewed rows re-run for a clean before/after, and **sample B**, a proportional 100 covering
all ten categories including the four never tested. It also became a two-model bake-off, running
every batch through **Claude Opus 5** and **gpt-5.6-sol** (via `codex exec`) with identical inputs.
All four arms completed: 9 + 100 rows × 2 models × 2 stages, with no missing or unparseable batches.
No review of pilot 2 and no results document exist in this directory — it is where the work stops.

## Outcome

- **Nothing was trained, and no corpus was built.** Both survey documents state this in their own
  status lines ("no builds started", "nothing locked, nothing built"), and the mix in
  `MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md` §5 was never assembled.
- **The pipeline design is validated at the 100-row scale and only there.** Classify → translate →
  generate natively runs end to end and passes deterministic Greek gates at 99/100
  ([`pilot_no_robots_100/out/gated.json`](pilot_no_robots_100/out/gated.json)), while the same
  evidence shows the gates cannot see the problems that matter (F5, O7).
- **One decision was taken by the owner and is binding on everything downstream:** D1, hard
  transposition of the frame into Greek reality, freeze the content
  ([`pilot_no_robots_100/FEEDBACK.md`](pilot_no_robots_100/FEEDBACK.md) §D1). Its consequence —
  compounded by O2's 8 silently corrected references — is that the artefact could not honestly be
  described as "no_robots translated into Greek".
- **The eight-class routing scheme** (§13a of the pipeline doc) survived two pilots and is the
  pipeline's spine; the two-family split (`TRANSLATION_SPEC_v1.md` Part II) is the correction pilot 1
  forced on it.
- **Left open at the end:** the whole glossary (`chatbot`, `email`, `bot`, jargon, transliteration
  standard — all "pending"); the register policy (fixed vs mirroring vs labelled); the gender policy;
  the licence posture (`no_robots` is CC-BY-NC, `ilsp/ifeval_greek` CC-BY-NC-SA, Gemma-as-teacher
  makes the student a Gemma Model Derivative); the checkpoint A/B for SFT init; a Greeklish /
  English-typed / code-switched input stream, of which the corpus has **zero rows** (F3); and the
  human review of pilot 1 itself, 9 rows of 100 done.
- **Carried forward:** per the owner's note, into the separate "natural-greek-sft" repository
  *(not verified in-repo)*.

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`pilot_no_robots_100/`](pilot_no_robots_100/README.md) | Pilot 1 — 100-row **stress** sample, 5 of 10 categories, Opus both stages | 2026-08-23 → 08-24 | reviewed 9/100 rows, then superseded as a sampling design | 99/100 hard gates; produced F1–F7, O1–O7 and decision D1 |
| [`pilot_v2/`](pilot_v2/README.md) | Pilot 2 — sample A (9 re-runs) + sample B (proportional 100, all 10 categories), Opus vs gpt-5.6-sol | 2026-08-24 | ran to completion; never reviewed | 218 rows per model arm, 4/4 arms complete, no results document |

## Where things are

| What | Path |
|---|---|
| What Greek SFT data exists, with licences | [`DATA_SURVEY_GREEK_SFT_20260823.md`](DATA_SURVEY_GREEK_SFT_20260823.md) |
| The mix, the model picks, the gates, the traps, the review-capacity arithmetic | [`MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md`](MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md) |
| The rules the pipeline must follow, plus per-category profiles over all 9,499 rows | [`TRANSLATION_SPEC_v1.md`](TRANSLATION_SPEC_v1.md) |
| Eleven root causes and their proposed fixes, ranked | [`ROOT_CAUSES_v1.md`](ROOT_CAUSES_v1.md) |
| Owner feedback verbatim, with the D1 decision | [`pilot_no_robots_100/FEEDBACK.md`](pilot_no_robots_100/FEEDBACK.md) |
| Machine-readable per-category spec, imported by the pilot-2 runner | [`pilot_v2/spec_data.py`](pilot_v2/spec_data.py) |
| Raw pilot output (the only measurement substrate that survives) | [`pilot_no_robots_100/out/`](pilot_no_robots_100/out/), [`pilot_v2/v2/`](pilot_v2/v2/) |

## Working documents

All four top-level documents are historical snapshots of a two-day investigation; none was updated
after 2026-08-24.

- **Surveys / plans (never executed as written):** `DATA_SURVEY_GREEK_SFT_20260823.md`,
  `MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md`. Both are explicitly labelled recommendations with
  nothing locked. Their §15 / §6 "first things to do" lists are only partly done — the 100-item
  pilot ran, the generator/translator bake-off and the ~100-pair native-Greek gate calibration did
  not.
- **Specs still marked v1 / for discussion:** `TRANSLATION_SPEC_v1.md` ("v1, for discussion"),
  `ROOT_CAUSES_v1.md` ("nothing here has been applied"). Read them as proposals, not as rules that
  governed any run — the only run after them, pilot 2, used the prompts in
  [`pilot_v2/prompts/`](pilot_v2/prompts/), which encode a subset.
- **Review log:** `pilot_no_robots_100/FEEDBACK.md` — a living document frozen mid-review at 9 of
  100 rows. Its own warning banner (read every rate with F7 in mind) still applies.
- **Data not recovered:** `raw_train.jsonl`, `sample_100.jsonl`, `sample_A.jsonl`, `sample_B.jsonl`
  and the built `artifact_v2.html` are referenced by the scripts but are not present in this
  directory; they are regenerable from `fetch.py` / `sample.py` / `build_samples.py`.
