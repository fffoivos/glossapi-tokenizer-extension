# Pilot 1 — no_robots → Greek, 100 prompts

Run 2026-08-23. Artifact: https://claude.ai/code/artifact/927a2ebe-8ad9-4519-aafd-64759554cc30

## What it is

First end-to-end test of the two-stage design from `../MINIMAL_SFT_MIX_AND_PIPELINE_20260823.md`
(§13a): **classify the English prompt → translate it → generate the Greek answer natively.**
Both model stages ran on **Claude Opus 5 at medium reasoning effort**, 5 batches of 20 (one per
category), stage 2 reading stage 1's output. Nothing hand-edited between stages.

## Sample

Full `train` split of `HuggingFaceH4/no_robots` pulled page-by-page from the datasets-server
(`fetch.py` + `repair.py`) → **9,499 unique `prompt_id` of 9,500** (one page failed persistently
after 5 retries). Then `sample.py`: `random.Random(20260823).sample()`, 20 rows from each of
5 categories chosen to stress *different* failure modes:

| Category | Stresses |
|---|---|
| Generation | mixed bucket; rhyme/acrostic/wordplay that cannot survive translation |
| Open QA | over-adaptation risk — real entities must NOT be localized |
| Chat | register; all 20 sampled rows are multi-turn **and** carry a system prompt |
| Rewrite | operations *on English text* needing different Greek machinery |
| Closed QA | answer must stay true to the *translated* passage (self-checking) |

## Results

- **Classes assigned**: LITERAL 50 · REGISTER-CRITICAL 23 · CONSTRAINT-PRESERVING 18 · LOCALIZE 5 ·
  PRESERVE-DEFECT 2 · VERBATIM-FREEZE 1 · REGENERATE-NATIVE 1 · RE-EXECUTE 0.
- **Category ≠ class**, confirmed: Chat is 16/20 register-critical, Closed QA 18/20 literal,
  Generation is scattered across 4 classes. The `category` field alone is not actionable.
- **Register**: 63 neutral / 31 ενικός / 6 πληθυντικός. Two-thirds of rows can be written without
  committing to an address form — the εσύ/εσείς decision is forced on ~a third, concentrated in Chat
  and personal-message Generation. Provisional policy under test = **mirror the prompt's register**.
- **Annotation volume**: 127 localization calls, 25 constraint notes, 182 recorded deviations from
  the English reference, 119 self-flags for human review.
- **Gates**: 99/100 pass hard gates, 88/100 with no advisory either.

## The gate lesson (worth keeping)

`gates.py` was wrong three times, and every version's sigma flag was a false positive:

| ver | "clean" | what it was really measuring |
|---|---|---|
| v1 | 74/100 | 7 final-sigma flags = **all elisions** (`σ' αγαπώ`, `Δώσ' μου`) with the apostrophe stripped by the tokenizer; 11 "low Greek ratio" = rows full of correctly-preserved proper names |
| v2 | 97/100 | remaining 2 sigma flags = `κ.σ.` (tablespoon) tokenized to a bare `σ`, and `τραγουδίστριατραγουδοποιόςηθοποιός` faithfully mirroring the source's own `Singersongwriteractress` |
| v3 | 99/100 | abbreviations + concatenation artifacts demoted to advisories |

**The register gate cannot do its job.** It counts 2nd-person markers, so it called the Three Little
Pigs row mixed-register — the wolf says `το σπίτι σου` to one pig and `το σπίτι σας` to two. That is
grammatical **number**, not formality, and no marker-counting heuristic separates them. This is the
direct evidence that register-critical rows must reach a human.

Zero failures on NFC, mixed-script homoglyphs, accented all-caps, polytonic bleed, and identity
leakage — the five modes most expected going in.

## Showcase rows

- `6d2fe44a` — "Piggy is a chatbot that answers in pig-latin" → **κορακίστικα**, the Greek
  syllable-insertion children's language, with all assistant turns regenerated under that rule.
  The only REGENERATE-NATIVE row.
- `dca7c1d3` — Aberdeen/Granite City: transliterated not localized, English epithet kept in brackets;
  the Greek also fixes the human reference's "Alberdeen"/"Nort-East" typos.
- `591f26b0` — English asks "when" twice; the defect is reproduced, not repaired, and the answer
  names the ambiguity.
- `b4bbd3c8` — Craigslist ad: `$480.00` → `480,00 ευρώ` (relabel, no rescale) but `650 sq ft` →
  `~60 τ.μ.` (a physical measurement must stay true).
- `425a5425` — email to daughter "Jenna" → Ελένη, Italy stays Italy, register forced to ενικός.

## Files

`prompts/stage1_classify_translate.md`, `prompts/stage2_generate.md` — the two prompts, verbatim.
`out/stage1_b*.json`, `out/stage2_b*.json` — raw model output. `out/gated.json` — per-row gate
record. `gates.py` — the gate stage. `build_artifact.py` + `template.html` — regenerate the artifact.

## Open next steps

1. Human review of the 100 (this is the pilot's real purpose — it yields the accept/edit/reject
   rate that sizes the whole programme). Prioritize the 23 register-critical rows.
2. Ratify the κορακίστικα encoding rule with a native speaker (the model flagged regional variation).
3. Decide the register policy from the measured evidence rather than in the abstract.
4. Decide whether English jargon (the piercing row: *septum*, *bridge*) is acceptable Greek usage.
