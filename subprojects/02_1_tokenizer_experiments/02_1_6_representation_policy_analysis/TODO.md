# TODO — 02.1.6 representation policy analysis

Method and source inventory are in
[`README.md`](README.md). Execute in phase order. Output of each
phase is its own markdown file; phases can be split across sessions
but must not be merged into a single doc.

## Status (2026-05-18 — archive mode)

[`INVESTIGATIONS_TRACKER.md`](INVESTIGATIONS_TRACKER.md) is the
authoritative status doc. Final summary:

- **Phase 1** (explicit goals → `01_explicit_goals.md`) — **DONE**.
- **Phase 2** (implicit constraints → `02_implicit_constraints.md`) — **DONE**.
- **Phase 3** (effective policy synthesis) — **NOT EXECUTED**. Earlier draft moved to `_deprecated_20260518/synthesis_with_known_errors/03_effective_policy.md` with known math errors.
- **Phase 4** (rational policy + Greek budget) — **NOT EXECUTED**. Earlier draft moved to `_deprecated_20260518/synthesis_with_known_errors/` (recommendation `+5,120` was wrong by 3×; actual cutoff `+17,408`).
- **Open hypotheses (5-10)** — **DEPRIORITIZED**. Six stub investigations moved to `_deprecated_20260518/`. Cutoff decided empirically in `02_1_7` without resolving them.

The four-phase plan below is preserved for historical reference but
**no active work is pending in this subproject**. The experimental
thread moved to
[`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/),
which decided the cutoff (17,408 added, curated + backfilled) on
2026-05-18.

Remaining items that could still be done — none of them blocking the
cutoff decision:

- **Tail-token audit** of the 486 dropped Mistral BPE entries
  (verification of `11_tokenizer_provenance.md`). ~30 min effort.
  Useful as provenance verification; not decision-relevant.
- **Hypotheses 7-9 stubs** in `_deprecated_20260518/` could still be
  executed if the cutoff is ever defended against "why didn't you
  match Mistral's per-language policy" questions. Not currently
  prioritised.

## Phase 1 — Explicit stated goals → `01_explicit_goals.md`

For every claim about language coverage / priority / intent, record
`(source, section, exact quote, ≤2-line paraphrase)` in a single
table. No synthesis yet.

- [ ] Apertus paper §1 (abstract + intro): multilingual framing, "1,800+
      languages", Swiss / EU / global positioning, any
      "primary languages" list.
- [ ] Apertus paper §2.2 + Appendix I: tokenizer-selection
      criteria. Exact wording of why Mistral-Nemo was picked. What
      notion of fairness is operative (Gini on FLORES+ 55 langs).
- [ ] Apertus paper §3 intro: stated goals of the data mixture.
- [ ] Apertus paper §3.1 (filters): full list of which filters apply
      to which languages. Special attention to §3.1.3 (toxicity
      classifier covers 9 langs — Greek not among them).
- [ ] Apertus paper §3.2: every dataset's stated coverage scope.
      Note where the paper explicitly says a dataset is single-lang
      vs multilingual.
- [ ] Apertus paper §3.3 (curriculum): any per-language curriculum
      decisions (sampler rates, quality cutoffs).
- [ ] Apertus paper §3.4 (long-context): per-language treatment of
      FineWeb-Long and Institutional Books.
- [ ] Apertus paper §4 (post-training): which languages get SFT,
      Romansh appendix J.1, multilingual instruction data.
- [ ] Apertus paper Appendix G: Table G.6 wording on "20 high-
      resource languages" — does the paper justify why these 20?
- [ ] Apertus paper Appendix H.2: stated non-use of `apertus-pretrain-
      swiss` and `apertus-pretrain-romansh` in v1.
- [ ] `swiss-ai/Apertus-8B-2509` model card: every claim about
      language coverage ("1811 natively supported languages",
      "Compliant", etc.).
- [ ] ETH Zurich press release (September 2025): public framing on
      Swiss-vs-multilingual-vs-global.
- [ ] Swiss AI Charter: scope, value commitments, language
      commitments if any.
- [ ] Mistral-Nemo-Base-2407 model card / Mistral release blog:
      tekken v3 design goals, claimed multilingual coverage.
- [ ] FineWeb-2 paper / dataset card (Penedo et al., 2025): stated
      goals — 1,811 langs, long-tail focus.
- [ ] FineWeb-2-HQ paper / dataset card (Messmer et al., 2025): HQ-20
      selection — what criterion produced these 20 specifically?
- [ ] FineWeb-HQ / FineWeb-Edu / DCLM-Edu dataset cards: any
      language-coverage statements (expected English-only).
- [ ] Clean-Wikipedia, EuroParl, ParaDocs, Institutional Books,
      EuroBlocks-SFT dataset cards: coverage scope statements.
- [ ] FineMath / MegaMath / StarCoder / CommonPile cards:
      language-coverage statements (expected English-mostly).

## Phase 2 — Implicit constraints → `02_implicit_constraints.md`

For every observed per-language behavior, identify the structural
constraint that produces it. Distinguish "policy chose this" from
"data / web / tokenizer-math forced this."

- [ ] Web language distribution baseline: cite W3Techs / Internet
      World Stats / similar primary source. Quantify what share of
      the web each major language gets, as the constant input any
      web-derived corpus inherits.
- [ ] English-only-by-construction datasets: tabulate which Apertus
      datasets are English-only (FineWeb-HQ / FineWeb-Edu / DCLM-Edu
      / FineMath / MegaMath / StarCoder / CommonPile / most
      Gutenberg). Compute their share of Apertus's total token
      budget. Conclude: English dominance is forced before any
      policy choice.
- [ ] Filter coverage gaps:
  - [ ] Toxicity classifier — 9 languages cover-set, Greek excluded.
        Effect: no toxicity-filter haircut applied to Greek text.
  - [ ] PII redaction — language-agnostic (regex), confirm.
  - [ ] robots.txt opt-out — language-agnostic, ~8 % English /
        ~4 % multilingual loss. Confirm symmetry.
- [ ] HQ-20 selection criterion: empirically what gates entry?
  - [ ] Cross-check: Korean is rank 14 by FW2 docs, not in HQ-20.
        Vietnamese is rank 23, in HQ-20. Greek rank 21, in HQ-20.
        Why?
  - [ ] Hypothesis to test: HQ-20 = "languages with a usable
        quality classifier," not "languages with most docs."
- [ ] FW2-HQ filter mechanics: `p × 0.95` haircut for the secondary
      ring of 12 HQ-20 languages, `p × 1.0` for primary ring of 8.
      Document which languages are in each ring and why.
- [ ] BPE merge-order biases:
  - [ ] Latin-script merges share across languages → big-Latin
        languages compound advantage.
  - [ ] Script-isolated languages get no sharing benefit.
        Greek, Korean, Japanese, Chinese, Arabic, Hebrew, Hindi,
        Bengali, Thai etc. all script-isolated under Apertus's vocab.
  - [ ] Quantify from `summary.tsv`: ratio of `masked_count` to
        sum-of-pairwise-overlaps per language as a sharing-benefit
        index.
- [ ] Acknowledged biases in Apertus's own materials:
  - [ ] Search paper + model card + blog for any explicit
        admission of language imbalance, English-dominance,
        Eurocentric framing, missing Asian / African coverage, etc.
  - [ ] Specifically check whether the authors flag low-resource
        underperformance or treat ~1,800-language coverage as a
        win irrespective of per-language quality.

## Phase 3 — Effective policy synthesis → `03_effective_policy.md`

- [ ] Cross-walk every per-language pattern visible in `summary.tsv`
      and `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md` against
      Phase 1 (stated) and Phase 2 (constraint).
- [ ] Write the policy as a numbered rule list. Example shape:
      `Rule 1: English data is unconditionally primary because all
      English-only datasets have no multilingual counterpart and
      they sum to ~60 % of consumed tokens (Phase 2 constraint).`
- [ ] For each rule, mark its source: `[stated]`, `[constraint]`,
      or `[unaccounted]`. Rules tagged `[unaccounted]` are the
      candidates for accidental policy.
- [ ] Sanity check: does the rule set predict the observed
      `summary.tsv` numbers within ~10 %? If not, the rule set is
      incomplete.

## Phase 4 — Rational policy + Greek budget → `04_rational_policy.md`, `FAIRNESS_DEFINITION.md`, `GREEK_BUDGET.md`

- [ ] For each rule in Phase 3, classify *necessary* vs *accidental*:
  - **necessary**: forced by hardware, available data, tokenizer
    math, or a principle the authors stated and would defend.
  - **accidental**: reflects a missing classifier, a filter-
    coverage gap, an unintended downstream consequence, or a
    historical priority without a stated principle.
- [ ] State the **rational core** = union of necessary rules.
- [ ] Apply the rational core to Greek:
  - [ ] What share of the vocab does it imply for Greek?
  - [ ] Cross-check against `summary.tsv` Greek baseline of 1,479
        PMI tokens (1.13 % of 131,072).
  - [ ] What `added` budget on top of the 1,479 base is implied?
        Align to the existing cutoff grid
        `{10240, 15360, 20480, 25600}` and the `+11,264` /
        `+11,520` variants in
        `02_1_4_cutoff_analysis/REPORT.md` §6.
- [ ] Write `FAIRNESS_DEFINITION.md` as a 1–2 page standalone
      statement so it can be cited from
      `02_1_4_cutoff_analysis/REPORT.md` §1 without needing the
      full analysis trail.
- [ ] Write `GREEK_BUDGET.md` with: the recommended number, the
      derivation chain, and an honest section on which Phase-2
      constraints would have to change for the recommendation to
      change.

## Open questions to surface during the work (don't pre-decide)

- Whether the rational-core policy can produce a *unique* Greek
  budget, or only a range. A range is acceptable as output.
- Whether Mistral-Nemo's stated tokenizer goals materially differ
  from what Apertus inherits (since Apertus did not retrain the
  tokenizer). If yes, the tokenizer-level policy and the
  pretraining-data-level policy may need separate fairness
  definitions.
- Whether the toxicity-filter coverage gap (Greek excluded) counts
  as a *positive* accident for Greek (i.e., Greek text was not
  haircut by toxicity filter) and what that implies for the
  rational-policy adjustment.

## Research rule

Match the citation discipline of `02_1_4_cutoff_analysis/REPORT.md`:
every numeric claim cites a primary source. Where the source is the
authors' own words, use an exact quote, not a paraphrase. Where
evidence is absent, say so explicitly and do not extrapolate.
