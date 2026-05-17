# Stub — Team / institutional nationality bias

Status: **OPEN**. Not yet investigated.

## Hypothesis

The Mistral-11 priority list might reflect the **nationality / language
fluency of the Mistral team** rather than any data-driven criterion.
Similarly, Apertus's affordances for Romansh / Swiss German / German /
French / Italian reflect the Swiss institutional environment.

Specifically:
- Mistral founders: Arthur Mensch, Guillaume Lample, Timothée Lacroix
  — all French nationals. Headquartered in Paris.
- Apertus team: Swiss Federal Institute of Technology Lausanne (EPFL,
  French-speaking Switzerland) + ETH Zurich (German-speaking
  Switzerland) + Swiss National Supercomputing Centre (CSCS, Italian-
  speaking Switzerland / Lugano). Multi-cantonal Swiss collaboration.

If team nationality is the driver, Mistral-11 should over-include
French and adjacent Romance/Western European languages relative to
data-driven baselines; Apertus's commitments should over-include the
Swiss national languages.

## Sources to check

### Mistral team

- Mistral founders' Wikipedia / LinkedIn profiles for nationality and
  prior education.
- Mistral team blog posts about company culture / language priorities.
- Locations of Mistral offices (Paris, ?).
- Funding sources — does French government investment shape priorities?

### Apertus team

- Apertus paper author affiliations — already cited in repo: EPFL +
  ETH + CSCS.
- Swiss AI Initiative public documents about scope.
- ETH press release framing on "Swiss values."
- Swiss AI Charter (referenced but not located as standalone doc).

### EPFL / ETH / CSCS prior multilingual work

- EPFL's NLP lab publications — any pre-existing language focus?
- ETH's language tech work.
- CSCS's institutional language commitments (CSCS is in Italian-
  speaking Lugano; do their materials reflect that?).

## What to test

### Q1 — Does Mistral-11's set composition match a French Western-European bias?

Specifically:
- Mistral-11 includes 6 Western European languages (en, fr, de, es,
  it, pt, NL not in list) and 5 non-European (zh, ja, ko, ar, hi).
- The Western European 6 are roughly the "Romance + Germanic core"
  of EU.
- Is this fit tighter than chance? Compare to a random selection of
  11 from the top-22 by HPLT 3.0.

### Q2 — Is Apertus's named priority list (Swiss German, Romansh + commercial DE/FR/IT/EN) a Swiss-cantonal mirror?

Apertus apps page: "Multilingual competence (German, French, Italian,
English) + native support for all Swiss national languages."

Swiss national languages: German (~63 %), French (~23 %), Italian
(~8 %), Romansh (~0.5 %). The Apertus list maps 1:1 onto Switzerland's
language demographics.

Verify: does the Apertus priority list track Swiss national language
proportions, or does it reflect something else?

### Q3 — Does the Apertus team's institutional split (EPFL/ETH/CSCS) explain Romansh's specific affordance?

Romansh is Swiss-only (~60,000 speakers globally, all in Switzerland).
A Swiss institutional team explains the Romansh investment in a way
no data-driven hypothesis does. Confirm via:
- Romansh-specific funding mentions in Swiss AI Initiative materials.
- Romansh community organizations (Lia Rumantscha) credited in
  Apertus paper §J.1.

## Why this matters for Greek

If Mistral-11 reflects French-nationality bias, Greek's exclusion is
**not a comment on Greek's relative merit** — it's a comment on
Greek not being French. Same for the small EU-but-Western-tilt HQ-20
additions (Greek included but Russian still nominally absent until
Apertus's EPFL re-add).

This puts the principled-budget question in a different frame: if
Mistral-11 is partially arbitrary national bias, then "fair share by
Mistral-11 reference class" doesn't carry normative weight — only
the data-driven layer does. Apertus's EPFL-led HQ-20 corrects some
of Mistral's biases (by adding Russian, Polish, Vietnamese, etc.)
but introduces its own (small EU languages including Greek).

## Output format

A markdown doc, ~1000-1500 words:

1. Team nationality / location facts.
2. Mistral-11 composition against "French Western-European bias"
   hypothesis.
3. Apertus priority list against "Swiss multi-cantonal" hypothesis.
4. Verdict: FIT / PARTIAL FIT / INVALIDATED for each subhypothesis.

## Priority

**MEDIUM.** Quick to investigate; offers a partial explanation for
patterns no data-driven hypothesis fits. Won't fully explain
Mistral-11 (the East Asian + Hindi + Arabic inclusions don't fit
French bias) but should clarify the Apertus / Romansh / Swiss
affordance story.

## Estimated effort

~30 min — team affiliations are public; the analysis is
straightforward.
