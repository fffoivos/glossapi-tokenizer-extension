# Stub — Multilingual benchmark coverage as selection criterion

Status: **OPEN**. Not yet investigated.

## Hypothesis

A language gets prioritized in pretraining if it has a usable
multilingual evaluation benchmark — because that's how the model's
multilingual claim is *demonstrable*. Languages without benchmarks
are harder to publish numbers for, so they're deprioritized even if
they're large in absolute terms.

Multilingual benchmark candidates Mistral / Apertus would have known
of at training time:

- **FLORES+ / FLORES-200** (NLLB Team, 2022 / 2024) — translation, 200
  language varieties.
- **XNLI** (Conneau et al., 2018) — 15 languages.
- **XCOPA** (Ponti et al., 2020) — 11 languages.
- **XQuAD** (Artetxe et al., 2020) — 11 languages.
- **TyDi-QA** (Clark et al., 2020) — 11 languages.
- **MLQA** (Lewis et al., 2020) — 7 languages.
- **MMLU-Multilingual** / **Belebele** (Bandarkar et al., 2023) —
  122 languages.
- **MT-Bench (multilingual variants)**.
- **AGIEval-MultiL** — various languages.
- **MGSM** — 11 languages for math reasoning.

If Mistral-11 / HQ-20 tracks benchmark availability, the languages
on the lists should be the **intersection of multiple major
benchmarks**.

## Sources to check

For each benchmark, harvest its official language list. Then:

1. Compute Mistral-11 ∩ each-benchmark and find which benchmarks
   Mistral-11 nearly-matches.
2. Compute HQ-20 ∩ each-benchmark.
3. Specifically check what's in **all of {XNLI, XCOPA, XQuAD,
   TyDi-QA, FLORES-55}** — that intersection might be Mistral-11.

## What to test

### Q1 — Is Mistral-11 = "languages in all major multilingual benchmarks"?

XNLI's 15 languages: en, fr, es, de, el, bg, ru, tr, ar, vi, th, zh,
hi, sw, ur. Mistral-11 ∩ XNLI = {en, fr, es, de, ar, zh, hi} = 7.
Languages in Mistral-11 NOT in XNLI: {it, pt, ja, ko}. Languages in
XNLI NOT in Mistral-11: {el, bg, ru, tr, vi, th, sw, ur}.

The intersection isn't tight. Maybe Mistral-11 is "union of XNLI +
XCOPA + popular-translation-benchmarks" rather than intersection.

Test all the intersections systematically.

### Q2 — Is HQ-20 = "languages in any standard multilingual benchmark"?

HQ-20 + FLORES-55 = 55 ∪ 20 = 56 (every HQ-20 is also in FLORES-55).
Verify which benchmarks every HQ-20 language is in.

### Q3 — Is Greek in any benchmark?

Greek is in XNLI (15 langs), FLORES+ (55), Belebele (122). Greek is
NOT in XCOPA, XQuAD, TyDi-QA, MLQA, MGSM. So Greek has medium
benchmark coverage — better than Bengali (XQuAD includes Bengali but
XNLI / XCOPA don't), worse than Spanish/French.

## Why this matters for Greek

If benchmark availability is the operative criterion, Greek's HQ-20
inclusion is justified by FLORES+ and XNLI coverage. Its Mistral-11
exclusion would track its absence from XCOPA / XQuAD / etc. — the
"less benchmark-able" tier.

A "fair share by benchmark coverage" frame would put Greek alongside
the other XNLI-but-not-XCOPA-XQuAD languages: roughly Turkish-tier
or Bulgarian-tier (both XNLI but limited elsewhere).

## Output format

A markdown doc, ~1500-2000 words:

1. Per-benchmark language tables.
2. Set arithmetic: Mistral-11 / HQ-20 against each benchmark.
3. Greek's specific position in the benchmark landscape.
4. Verdict.

## Priority

**LOWER-MEDIUM.** Strongly overlaps with Hypothesis 8 (inherited from
priors) — benchmarks track model coverage which tracks prior
multilingual standards. May not yield a fundamentally distinct frame.
But quick to do.

## Estimated effort

~45 min — benchmark language lists are publicly published in their
canonical papers.
