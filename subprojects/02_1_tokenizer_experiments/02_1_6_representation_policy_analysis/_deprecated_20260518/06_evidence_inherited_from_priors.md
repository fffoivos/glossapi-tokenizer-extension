# Stub — Inherited from prior multilingual model lineage

Status: **OPEN**. Not yet investigated.

## Hypothesis

Mistral-11 / HQ-20 may be effectively a subset of (or inherited from)
the language coverage of earlier multilingual models that established
field standards. Specifically:

- **mBERT** (Devlin et al., 2018) — 104 languages
- **XLM-RoBERTa** (Conneau et al., 2020) — **100 languages**
- **mT5** (Xue et al., 2021) — 101 languages
- **BLOOM** (BigScience, 2022) — 46 languages
- **EuroLLM** (UTTER project, 2024) — 24 EU + 11 additional = 35
- **NLLB-200** (Meta, 2022) — 200 languages

Apertus's own quality-filtering classifier IS XLM-RoBERTa-100
(verified — Messmer et al. §3.3). So at least one direct inheritance
chain exists: HQ-20 ⊂ XLM-R-100.

## Sources to check

### Each model's official language list

- **mBERT**: paper Appendix B language list; Devlin et al. 2018.
- **XLM-RoBERTa**: 100-language list in Conneau et al. 2020 Table 6.
- **mT5**: 101-language list in Xue et al. 2021.
- **BLOOM**: 46-language list in BLOOM paper §3.
- **EuroLLM-9B**: 35-language list (24 EU + 11 extras) — confirmed
  in our `01_explicit_goals.md` (Apertus paper / EuroBlocks
  references).
- **NLLB-200**: 200 language list.

For each, harvest the verbatim language list and cross-reference
against Mistral-11 + HQ-20.

## What to test

Three nested questions:

### Q1 — Is Mistral-11 ⊂ XLM-R-100?

XLM-R-100 covers English, French, Spanish, German, Russian, Chinese,
Japanese, etc. Does it cover ALL 11 Mistral languages? Almost
certainly yes; verify.

### Q2 — Is HQ-20 ⊂ XLM-R-100?

Apertus's FW2-HQ uses XLM-RoBERTa-100 as classifier base. HQ-20 must
be ⊂ XLM-R-100 by construction. Verify which 80 XLM-R languages
**aren't** in HQ-20 and why (presumably because they have less FW2
data).

### Q3 — Does Mistral-11 better fit some prior model's top-tier than HPLT/speakers?

Is Mistral-11 the **intersection of "languages well-represented in
XLM-R" AND "languages with significant CommonCrawl share"**? This
hypothesis predicts:

- Languages in XLM-R-100 with strong CC presence: included in
  Mistral-11.
- Languages in XLM-R-100 but small CC presence: not included
  (Bengali / Marathi / Tamil — but these ARE in XLM-R-100).
- Languages with strong CC presence but not in XLM-R-100 or earlier
  multilingual standards: not included (some long-tail CC languages).

### Q4 — Specifically, what does BLOOM cover?

BLOOM's 46-language list was a curated multilingual research target
in 2022. Many BLOOM languages were later refined down to ~20 in HQ.
The pattern from BLOOM → HQ-20 may show the policy lineage.

## Why this matters for Greek

- Greek IS in XLM-R-100 (verify).
- Greek IS in mBERT-104 (verify).
- Greek IS NOT in BLOOM-46 (verify).
- Greek IS in EuroLLM-35 (24 EU + extras).

If Mistral-11 ⊂ BLOOM-46, Greek's omission from Mistral-11 might be
explained by BLOOM's exclusion. If HQ-20 ⊃ Greek because XLM-R-100
covers it AND it's in CC, then Greek's HQ-20 inclusion is a
"both-condition" satisfaction.

The "size we should give Greek" depends on which prior-model lineage
Apertus's tokenizer is actually inheriting from.

## Output format

A markdown doc, ~1500-2500 words:

1. Language-coverage tables for each prior multilingual model.
2. Set-intersection / set-difference analysis:
   - Mistral-11 ∩ XLM-R-100 = ?
   - Mistral-11 ⊂ XLM-R-100 ? (predicted yes)
   - HQ-20 ∩ BLOOM-46 = ?
   - Mistral-11 ⊂ NLLB-200 ?
3. Greek's status in each prior model.
4. Verdict: does Mistral-11 fit "subset of XLM-R-100" or "subset of
   BLOOM-46" or "subset of EuroLLM-35" better than any other tested
   hypothesis?

## Priority

**HIGH.** Most direct structural answer available — Apertus literally
uses XLM-R as a tool, so XLM-R-100 IS the upper-bound coverage list
for HQ-20. The question is whether Mistral-11 is also a recognizable
subset of a prior lineage.

## Estimated effort

~45 min — language lists are published in canonical papers; the work
is set arithmetic + Greek cross-check.
