# Pilot 2 — proportional sample, rewritten prompts, two model arms

> **In one line:** the re-run that pilot 1's review demanded — the nine reviewed rows again for a
> before/after, plus a *proportional* 100 covering all ten `no_robots` categories — run twice, once
> through Claude Opus 5 and once through gpt-5.6-sol, under prompts rebuilt around the spec.
> **Period:** 2026-08-24 (every file carries mtime 2026-08-24 03:23; the directory has no development
> history — see [`../README.md`](../README.md) note 2). **Status:** all four arms ran to completion; **nothing here was ever reviewed**
> and no results document was written. This is where the work stops in this repo.
> **Came from / led to:** [`../pilot_no_robots_100/`](../pilot_no_robots_100/README.md) →
> [`../ROOT_CAUSES_v1.md`](../ROOT_CAUSES_v1.md) RC11 → this → nothing further in this repo.

## Why this existed

Pilot 1 was stratified for stress, not proportion, so its rates described nothing about the corpus
([`../pilot_no_robots_100/FEEDBACK.md`](../pilot_no_robots_100/FEEDBACK.md) F7). RC11 prescribed two
samples: **A**, the nine rows the owner had actually reviewed, re-run to check the fixes exactly
where the problems were found; and **B**, a proportional 100 that covers all ten categories and
finally tests Coding — the untested home of `VERBATIM-FREEZE`. The run also became an unplanned
two-model comparison, which is the closest thing this subproject has to the generator bake-off that
`MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md` §10 asked for and never got.

## What ran

[`build_samples.py`](build_samples.py) builds both samples deterministically: A is the nine reviewed
`prompt_id`s, hard-coded — all nine are **Chat**, which is why the owner reported seeing "nothing but
named chatbots"; B is `random.Random(20260824)` drawn to true category shares (Generation 45,
Open QA 12, Brainstorm 11, Chat 8, Rewrite 7, Summarize 4, Coding 4, Classify 4, Closed QA 3,
Extract 2), with Generation stratified across three prompt-length bands because it is 46% of the
corpus and the most heterogeneous.

[`runner.py`](runner.py) batches at 10 rows per call and splices two blocks into each prompt from
[`spec_data.py`](spec_data.py) — the row's **family** and its **category profile**, both lifted from
[`../TRANSLATION_SPEC_v1.md`](../TRANSLATION_SPEC_v1.md) Part IV (its per-category counts sum to
9,499, matching the fetched split). [`drive_sol.py`](drive_sol.py) runs the sol arm via
`codex exec -m gpt-5.6-sol -c model_reasoning_effort=medium`, four workers, resuming from whatever
output already parses; the Opus arm was driven separately by a Workflow script over the same batch
files (`runner.py` docstring — that script is not in this directory).

The prompts were rewritten, not patched. [`prompts/v2_stage1.md`](prompts/v2_stage1.md) opens with
"You are **not** translating… a Greek reader should never be able to tell the row began in English",
and adds: the D1 frame/content test as Step 1; **two labels for source-bearing rows**
(`source_handling` × `output_handling`) instead of one; a devices step that names what a name *does*
so the effect rather than the spelling is reproduced (F6); required `derived_elements` for
propagation, where an empty list is a claim (F1); and `guide_gaps` as the mechanism that turns
ad-hoc lexical calls into the future glossary (RC1) — the prompt states plainly that "there is not
yet a settled house glossary". [`prompts/v2_stage2.md`](prompts/v2_stage2.md) makes the English
answer an **anchor, not a source text**, forbids silent repair while requiring
`reference_corrections`, orders stage 2 to honour stage 1's class and record any
`class_disagreement` (O1), and splits the defect note by family (RC8).
[`prompts/stage1_classify_translate.md`](prompts/stage1_classify_translate.md) and
[`prompts/stage2_generate.md`](prompts/stage2_generate.md) are byte-identical copies of the pilot-1
prompts, kept for comparison.

## Outcome

**Completion is the one uncontested result:** 4/4 arms produced every batch, with no missing or
unparseable file — A 9 rows and B 100 rows, for each of two stages, for each of two models
(counted from [`v2/`](v2/) against [`v2/batches/*_manifest.json`](v2/batches/)).

**The F1 fix is verified in the output.** On row `6d2fe44a` — the pig-latin row whose persona name
pilot 1 left as *Το Γουρουνάκι*, pointing at a game that no longer existed — **both arms re-derived
the name to Κοράκι** from κορακίστικα and recorded it in `derived_elements` with the dependency
named (`v2/A/{opus,sol}/stage1_A_Chat_0.json`). Both also kept the row as `REGENERATE-NATIVE`.

**The two arms diverge sharply**, which is why this needed a review it never got. Sample B, counted
from the stage-1 and stage-2 JSON:

| | Opus | sol |
|---|---|---|
| generative classes (80 rows) | LITERAL 37 · CONSTRAINT-PRESERVING 24 · LOCALIZE 10 · REGISTER-CRITICAL 7 · REGENERATE-NATIVE 2 | CONSTRAINT-PRESERVING 33 · LITERAL 28 · REGISTER-CRITICAL 14 · LOCALIZE 4 · REGENERATE-NATIVE 1 |
| source-bearing (20 rows) | LITERAL 14 · PRESERVE-DEFECT 3 · LOCALIZE 3 | LITERAL 13 · PRESERVE-DEFECT 5 · VERBATIM-FREEZE 1 · LOCALIZE 1 |
| target register | ενικός 73 · neutral 17 · πληθυντικός 10 | ενικός 51 · neutral 42 · πληθυντικός 7 |
| stage 2 used the register stage 1 targeted | **86/100** | **100/100** |
| transpositions recorded | 101 | 36 |
| `derived_elements` | 110 | 47 |
| `guide_gaps` (both stages) | 521 | 203 |
| `reference_corrections` | 332 | 162 |
| `self_flags` | 170 | 17 |
| `defect_note` written | 19 | 6 |
| `class_disagreement` | 0 | 0 |

Read carefully, none of these is a quality score — they are volumes of self-reported annotation, and
the arm that annotates more is not thereby better. Two observations do hold. First, **sol's
classification is closer to F7's corpus-level prediction** that `CONSTRAINT-PRESERVING`, not
`LITERAL`, is the largest class (~37%): sol 33, Opus 24. Second, **`guide_gaps` fired hundreds of
times in both arms**, which is exactly what RC1 predicted for a run with no settled glossary — the
instrument worked, and the worklist it produced was never triaged.

**Left open:** no human review, no accept/edit/reject rate, no A-vs-B naturalness judgement, and no
decision between the two model arms. The comparison artifact that [`build_v2.py`](build_v2.py)
generates (`artifact_v2.html`, three columns: original | opus | sol) is not present in this
directory, and no artifact URL is recorded anywhere for pilot 2 — unlike pilot 1.

## Where things are

| What | Path |
|---|---|
| Sample construction (deterministic, seed 20260824) | [`build_samples.py`](build_samples.py) |
| Prompt assembly + the sol driver | [`runner.py`](runner.py), [`drive_sol.py`](drive_sol.py) |
| Per-category spec injected into every call | [`spec_data.py`](spec_data.py) — n, share, family, `piloted` flag, essentials, what is safe to adapt, what is checkable, review priority |
| The v2 prompts as run | [`prompts/v2_stage1.md`](prompts/v2_stage1.md), [`prompts/v2_stage2.md`](prompts/v2_stage2.md) |
| Raw output, 4 arms | [`v2/A/opus/`](v2/A/opus/), [`v2/A/sol/`](v2/A/sol/), [`v2/B/opus/`](v2/B/opus/), [`v2/B/sol/`](v2/B/sol/) |
| Batch prompt files + manifests (the run's receipts) | [`v2/batches/`](v2/batches/) |
| Comparison-artifact builder | [`build_v2.py`](build_v2.py) + [`template_v2.html`](template_v2.html) |

## Working documents

This directory has no prose documents of its own — it is code, prompts and raw output. The
reasoning behind it lives one level up in [`../ROOT_CAUSES_v1.md`](../ROOT_CAUSES_v1.md) (RC11 for
the sampling, RC1/RC3/RC6/RC8 for the prompt rewrite) and
[`../TRANSLATION_SPEC_v1.md`](../TRANSLATION_SPEC_v1.md) (the rules and category profiles that
`spec_data.py` encodes). `sample_A.jsonl`, `sample_B.jsonl`, `raw_train.jsonl` and
`artifact_v2.html` are referenced by the scripts but absent; the samples are regenerable from
`build_samples.py` given a re-fetched `raw_train.jsonl`.
