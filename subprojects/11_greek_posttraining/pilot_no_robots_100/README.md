# Pilot 1 — no_robots → Greek, 100 rows (the stress sample)

> **In one line:** the first end-to-end run of the classify → translate → generate-natively pipeline;
> it passed its own gates at 99/100 and then, under nine rows of owner review, produced the two
> findings that changed the programme — decision D1 and the admission that the sample's frequencies
> describe nothing.
> **Period:** 2026-08-23 (run) → 2026-08-24 (review). **Status:** completed as a bug-finder,
> superseded as a sampling design by [`../pilot_v2/`](../pilot_v2/README.md); the human review was
> stopped at 9 of 100 rows.
> **Came from / led to:** [`../MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md`](../MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md)
> §15.2 → this → [`FEEDBACK.md`](FEEDBACK.md) → [`../TRANSLATION_SPEC_v1.md`](../TRANSLATION_SPEC_v1.md)
> + [`../ROOT_CAUSES_v1.md`](../ROOT_CAUSES_v1.md) → [`../pilot_v2/`](../pilot_v2/README.md).

## Why this existed

The pipeline document called for a 100-item end-to-end pilot before anything was built, on the
grounds that the accept/edit/reject rate is the only honest input to a schedule dominated by human
review. This is that pilot. Both model stages ran on **Claude Opus 5 at medium reasoning effort**, in
5 batches of 20 (one per category), stage 2 reading stage 1's output, with nothing hand-edited
between them.

## History

**2026-08-23 — build and run.** The full `train` split of `HuggingFaceH4/no_robots` was pulled
page-by-page from the datasets-server ([`fetch.py`](fetch.py) + [`repair.py`](repair.py)), yielding
**9,499 unique `prompt_id` of 9,500** — one page failed persistently after five retries.
[`sample.py`](sample.py) then drew `random.Random(20260823).sample()`, 20 rows from each of five
categories picked to stress *different* failure modes: Generation (wordplay that cannot survive
translation), Open QA (over-adaptation of real entities), Chat (register; all 20 sampled rows are
multi-turn and carry a system prompt), Rewrite (operations on English text), Closed QA (the answer
must stay true to the *translated* passage). Outputs are the ten
[`out/stage{1,2}_b*.json`](out/) files, 100 rows each stage.

**2026-08-23 — the gate suite was wrong three times.** [`gates.py`](gates.py) went through three
versions, and **every version's final-sigma hard flag was a false positive**:

| ver | "clean" | what it was really measuring |
|---|---|---|
| v1 | 74/100 | 7 sigma flags = all elisions (`σ' αγαπώ`, `Δώσ' μου`) with the apostrophe stripped by the tokenizer; 11 "low Greek ratio" = rows full of correctly-preserved proper names |
| v2 | 97/100 | the 2 remaining sigma flags = `κ.σ.` tokenised to a bare `σ`, and a concatenation faithfully mirroring the source's own `Singersongwriteractress` |
| v3 | 99/100 | abbreviations and concatenation artefacts demoted to advisories |

**2026-08-24 — nine rows of owner review, logged not patched.** The artifact was frozen while the
owner read; findings were accumulated in [`FEEDBACK.md`](FEEDBACK.md) rather than fixed one at a
time. Nine rows — all of them Chat, which is precisely what exposed F7 — produced seven owner
findings and seven observations. Decision **D1** (hard transposition of the invented frame into Greek
reality, freeze anything whose swap would change whether the answer is true) and finding **F7** (the
sample is stratified, not proportional, so its rates are not evidence about the corpus) between them
ended this pilot's usefulness as a measurement and started `pilot_v2`.

## Outcome

Numbers recomputed from [`out/gated.json`](out/gated.json) unless stated:

- **Gates: 99/100 with no hard flag, 88/100 with no advisory either.** The single hard flag is one
  `LOW_GREEK_RATIO`; the advisories are 5 register self-report disagreements, 4 length outliers, 1
  register review and 1 concatenated token. Zero failures on NFC, mixed-script homoglyphs, accented
  all-caps, polytonic bleed and identity leakage — the five modes most expected going in.
- **Classes assigned:** LITERAL 50 · REGISTER-CRITICAL 23 · CONSTRAINT-PRESERVING 18 · LOCALIZE 5 ·
  PRESERVE-DEFECT 2 · VERBATIM-FREEZE 1 · REGENERATE-NATIVE 1 · RE-EXECUTE 0. **Category ≠ class**:
  Chat is 16/20 register-critical, Closed QA 18/20 literal, Generation scattered across four classes.
- **Register:** 63 neutral / 31 ενικός / 6 πληθυντικός as targeted; 60 / 32 / 8 as actually used.
  Two-thirds of rows can be written without committing to an address form. This is the one pilot
  figure that survives F7's re-weighting to corpus level essentially unchanged.
- **The two gate lessons that stuck.** The register gate counts second-person markers, so it called
  the Three Little Pigs row mixed-register — the wolf says `το σπίτι σου` to one pig and `το σπίτι
  σας` to two. That is grammatical **number**, not formality, and no marker-counting heuristic
  separates them. Second, a regex scan for marked word order returned **1 true positive in 11 hits
  and missed two of the three known cases** ([`FEEDBACK.md`](FEEDBACK.md) F5). Together these are the
  pilot's strongest result: the deterministic gates handle correctness well and are structurally
  incapable of seeing naturalness, so the human budget belongs on naturalness.
- **The pipeline silently improved the human data in 8 rows** (O2) — misspellings *Alberdeen* /
  *Nort-East*, a factual contradiction, a wrong name — which, with D1, means the output is not a
  translation of `no_robots` at all.
- **Left undone:** the human review of the other 91 rows, hence no accept/edit/reject rate — the
  number the pilot was run to obtain. Also open at the end: ratifying the κορακίστικα encoding rule
  with a native speaker, and whether English piercing jargon (*septum*, *bridge*) is acceptable Greek.

Rows worth keeping as worked examples, from the old README: `6d2fe44a` (pig-latin →
**κορακίστικα**, the only REGENERATE-NATIVE row, and the source of finding F1 because the persona
name was not re-derived); `b4bbd3c8` (`$480.00` → `480,00 ευρώ` relabelled, but `650 sq ft` →
`~60 τ.μ.` converted, because a physical measurement must stay true); `591f26b0` (the English asks
"when" twice; the defect is reproduced, not repaired).

## Where things are

| What | Path |
|---|---|
| The two prompts, verbatim as run | [`prompts/stage1_classify_translate.md`](prompts/stage1_classify_translate.md), [`prompts/stage2_generate.md`](prompts/stage2_generate.md) — copied unchanged into [`../pilot_v2/prompts/`](../pilot_v2/prompts/) |
| Raw model output, 100 rows per stage | [`out/stage1_b0..b4.json`](out/), [`out/stage2_b0..b4.json`](out/) |
| Per-row gate record (the numbers above) | [`out/gated.json`](out/gated.json) |
| The gate stage, v3 | [`gates.py`](gates.py) — stdlib only; 15 checks incl. NFC, Greek ratio with proper-noun discount, mixed script, final sigma, accented caps, polytonic, identity leak, register markers, length ratio |
| Corpus fetch / repair / sampling | [`fetch.py`](fetch.py), [`repair.py`](repair.py), [`sample.py`](sample.py) |
| Review artifact builder | [`build_artifact.py`](build_artifact.py) + [`template.html`](template.html) |

## Working documents

- [`FEEDBACK.md`](FEEDBACK.md) — the review log, historical. A living document frozen at 9 of 100
  rows: Part 0 the D1 decision, Part 1 owner findings F1–F7, Part 2 observations O1–O7, Part 3 four
  cross-cutting themes, Part 4 the questions left open. Its banner warning — read every rate with F7
  in mind — is still the right way to read it. Superseded as a plan by
  [`../ROOT_CAUSES_v1.md`](../ROOT_CAUSES_v1.md), which reduces its fifteen findings to eleven causes.
- The old README recorded a review artifact at `https://claude.ai/code/artifact/927a2ebe-8ad9-4519-aafd-64759554cc30`
  (claimed there and in `FEEDBACK.md`; not verified here).
- `raw_train.jsonl` and `sample_100.jsonl` are written by `fetch.py` and `sample.py` but are not
  present in this directory; both are regenerable, the sample deterministically (seed 20260823).
