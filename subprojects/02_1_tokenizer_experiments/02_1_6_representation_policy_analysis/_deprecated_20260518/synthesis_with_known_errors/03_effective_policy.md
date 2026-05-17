# Phase 3 — Effective policy synthesis

The actual rule set producing Apertus's observed per-language vocab +
pretraining allocation. Each rule is tagged:

- `[stated]` — author explicitly stated this as intent.
- `[constraint]` — structural consequence of data, web, or
  tokenizer-math; cited in Phase 2.
- `[tooling gap]` — produced by a missing classifier / OCR / probe /
  pair set; not a stated choice but a contingent infrastructure
  outcome.
- `[unaccounted]` — no primary-source explanation; observable in the
  artifact but justified nowhere.

Sources: `01_explicit_goals.md` (E.x), `02_implicit_constraints.md`
(C.x), measurements in `docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`
and `02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/`.

## Effective-policy rules

### Rule 1 — English is the unconditional primary language

- **Observed**: ~60 % of consumed 8B pretraining tokens are English;
  English PMI count is 19,009 (14.5 % of vocab), 3× the next-largest
  language.
- **Sources**: `[constraint]` — English-only datasets sum to ~11 TB
  pool vs ~3.5 TB for multilingual per stage (Phase 2 §2). `[stated]` —
  no Apertus source frames Apertus as English-primary, but no source
  denies it either; the apps page advertises "40 % non-English" as
  the explicit framing.
- **Necessary or accidental**: Necessary given that Apertus chose to
  use FineWeb-Edu / FW-HQ / DCLM-Edu (which are English-only by
  construction). English dominance is a chosen consequence of dataset
  selection, not an explicit policy. The choice TO use those datasets
  is a chosen policy; the per-language consequence then follows.

### Rule 2 — Multilingual content allocated by natural web frequency

- **Observed**: FineWeb-2 per-language doc counts (Apertus paper
  Table G.6) roughly track web language share. Greek 0.97 % of FW2
  vs 0.5 % of W3Techs web; Russian 13.26 % of FW2 vs 3.5 % of web (over-
  representation comes from FW2's crawler bias toward Russian sites).
- **Sources**: `[stated]` — Penedo et al. and Apertus §3.2.2: "preserve
  all languages present in the dataset in their natural frequency."
- **Necessary or accidental**: A chosen policy. Apertus could have
  rebalanced (e.g. equal-share, or proportional to a stated priority
  list) but chose not to. Stated rationale is pipeline simplicity.

### Rule 3 — Quality filtering applies only to languages with a classifier

- **Observed**: 20 languages get FW2-HQ; the other 1,791 in FW2 get
  random sampling.
- **Sources**: `[stated, with critical gap]` — Messmer et al. §2 says
  "we limit our scope to 20 languages as the number of documents
  drops quickly." The specific 20 are listed but **the criterion is
  not stated** (Phase 1 §D.5).
- **Necessary or accidental**: The *practice* of language-specific
  quality filtering is `[stated]`; the *specific 20-language list* is
  `[unaccounted]`. Korean / Romanian / Ukrainian unexplained
  exclusions, Vietnamese unexplained inclusion. Plausibly the
  binding constraint is per-language quality-classifier training-data
  availability (Aya / MMLU coverage) but this is not stated.

### Rule 4 — Within HQ-20, retention is 10 % or 33 % (curriculum-driven)

- **Observed**: Apertus uses 33 % retention in Stages 1-3, 10 % in
  Stages 4-5.
- **Sources**: `[stated]` — Apertus paper Table 6, Appendix G.
- **Necessary or accidental**: Chosen policy. Stated rationale is
  curriculum design (more breadth early, more quality late). Not
  forced by data or compute.

### Rule 5 — HQ-20 secondary ring gets a 0.95 sampler haircut

- **Observed**: 12 of 20 HQ languages including Greek have
  `sampler.rate = 0.95`; the 8 primary-ring HQ languages have rate
  1.0. (Reference: `apertus_greek_extension.yaml` / pretrain-data
  pipeline configs.)
- **Sources**: `[unaccounted]` — neither the Messmer paper nor the
  Apertus paper documents this ring split. The terminology
  "primary/secondary ring" is project-local. The 0.95 value appears
  in code (`swiss-ai/pretrain-data/pipelines/fineweb-2/main.py`) but
  not in any paper or card.
- **Necessary or accidental**: `[unaccounted]` — by default, accidental.
  A 5 % haircut applied to 12 of 20 HQ languages, but not to the other
  8, has no stated rationale. Greek's allocation is one of the 12
  haircut.

### Rule 6 — Toxicity haircut for top-9-by-data-availability

- **Observed**: Top-5 % toxicity haircut on en, zh, fr, de, it, nl, pl,
  es, pt. None on Greek or the other 11 HQ-20 languages.
- **Sources**: `[stated]` — Apertus §3.1.3 explicit list. `[tooling
  gap]` — Greek's exclusion is justified by classifier-data
  availability not by Apertus-policy.
- **Necessary or accidental**: Tooling gap → accidental. Greek-favorable
  direction (no haircut).

### Rule 7 — Long-context OCR post-processing for top-5 EU languages

- **Observed**: Institutional Books 1.0 OCR-post-processes eng / deu /
  fra / ita / spa. 249 other languages including Greek are present
  but raw OCR.
- **Sources**: `[stated]` — Institutional Books card. `[tooling gap]`
  for the 5-language scope.
- **Necessary or accidental**: Tooling gap → accidental. Direction:
  slightly negative for Greek's long-context contribution.

### Rule 8 — Parallel translation pairs follow ParaDocs's 6-pair scope

- **Observed**: ParaDocs covers en-de / en-fr / en-es / en-it / en-pl /
  en-pt; no Greek pair.
- **Sources**: `[stated]` — JHU ACL paper. `[tooling gap]` for Greek.
- **Necessary or accidental**: Tooling gap → accidental. EuroParl
  (separate dataset) does include Greek; ParaDocs's gap is partially
  filled by EuroParl bitexts.

### Rule 9 — Mistral's "particularly strong 11" defines downstream tooling priority

- **Observed**: The 11 languages Mistral flagged as model-strong (en,
  fr, de, es, it, pt, zh, ja, ko, ar, hi) substantially overlap with
  what downstream LLM ecosystems treat as "supported." Greek is not
  on this list and so doesn't inherit Mistral-aligned third-party
  tooling.
- **Sources**: `[stated]` — Mistral blog.
- **Necessary or accidental**: Inherited stated policy from Mistral.
  Apertus did not re-derive this list; it inherited tekken's
  downstream consequences.

### Rule 10 — Tokenizer chosen for Gini fairness across FLORES+ 55 languages

- **Observed**: Apertus picked Mistral-Nemo tekken over Llama-3.1 /
  Qwen-2.5 / Gemma-2 because tekken had the lowest Gini coefficient
  on FLORES+. The 55 FLORES+ languages **include Greek**.
- **Sources**: `[stated]` — Apertus §2.2, Appendix I.
- **Necessary or accidental**: Necessary given Apertus's stated
  fairness goal. The Gini-on-55-langs metric *does* count Greek; it
  just averages Greek's fairness contribution with 54 other
  languages, so Greek's specific footprint can be poor while the
  aggregate is fair.

### Rule 11 — Script-isolated languages need dedicated vocab slots

- **Observed**: Greek ∩ Bulgarian PMI overlap = 0; Greek ∩
  any-Latin-language overlap = 0. Greek tokens are 100 % dedicated.
- **Sources**: `[constraint]` — BPE math. UTF-8 codepoints are
  disjoint between scripts.
- **Necessary or accidental**: Mathematical, non-negotiable.

### Rule 12 — BPE merge order is frequency-driven on Mistral's training data

- **Observed**: Mistral's undisclosed data mix sets the per-language
  merge frequency in tekken. Apertus inherits.
- **Sources**: `[constraint]` — BPE algorithm.
- **Necessary or accidental**: Mistral's policy was stated (compression
  efficiency vs Llama-3, 11 strong languages); Apertus inherits the
  consequence.

### Rule 13 — Public framing: Swiss-multilingual-global triad

- **Observed**: Three positioning frames coexist (Swiss-first,
  largest-effort-worldwide, European-sovereignty) without
  reconciliation.
- **Sources**: `[stated]` — ETH press, EPFL article, swiss-ai.org,
  apertus.ai.
- **Necessary or accidental**: Public-communication choice. The
  multilingual claim ("1,811 / 1,000+ / 1,800+ languages") creates
  an implicit commitment to non-trivial coverage for every claimed
  language.

### Rule 14 — Named priority languages get explicit affordances

- **Observed**: Swiss German + Romansh receive a dedicated SFT split
  (Romansh: 46,923 SFT examples covering 6 written varieties); German /
  French / Italian get commercial-page "Multilingual competence"
  badge; English is unconditional primary. **Greek receives no named
  affordance.**
- **Sources**: `[stated]` — Apertus §4.1.3, Appendix J.1, apertus.ai
  apps page.
- **Necessary or accidental**: Chosen policy. Apertus chose to invest
  in Romansh-specific tooling explicitly; the absence of analogous
  Greek-specific tooling is the residual.

### Rule 15 — Aggregate language count is the primary marketing claim

- **Observed**: Headline figure is "1,811" / "1,000+" / "1,800+"
  rather than per-language quality. Performance evaluations cover
  "around 100 languages" — an order of magnitude smaller than the
  headline.
- **Sources**: `[stated]` — model card, press release, apps page.
- **Necessary or accidental**: Public-communication choice.

## Predictive check

If these rules are the effective policy, do they predict the observed
per-language footprint within Apertus's vocab and pretraining?

**For Greek**, the rules predict:
- Tokenizer footprint = Mistral's allocation (1,479 PMI tokens) [Rule
  12, inherited].
- Pretraining-token share = (FW2 share 0.97 %) × (HQ-20 quality
  retention 10–33 %) × (secondary ring 0.95) × (stage weighting) =
  ~0.02 % [Rules 2-5].
- Toxicity haircut = 0 [Rule 6, tooling gap].
- No long-context OCR boost [Rule 7].
- No parallel-pair boost beyond EuroParl [Rule 8].
- No Mistral-strong-11 inheritance [Rule 9].

Measured: 1,479 PMI tokens (1.13 % of vocab), 0.023 % of consumed
pretraining tokens. **Both match the predictions.**

**For Korean** (named-excluded HQ-20):
- Tokenizer footprint = Mistral's allocation (4,438 PMI tokens). Korean
  is in Mistral-strong-11 (Rule 9) and is one of the 2 languages where
  tekken has 2× the compression efficiency of Llama-3.
- Pretraining-token share = FW2 random-33% only (no HQ filter, Rule 3).
- Toxicity haircut = 0 (not in 9-cover-set, Rule 6).

Korean's PMI footprint (4,438) is **3× Greek's** despite being absent
from HQ-20. The explanation: Mistral allocated heavily to Korean
(Rule 12), and that's inherited. Korean's pretraining-data share is
small (random-33 % from FW2) but the tokenizer was already Korean-
favorable before Apertus's data choices.

**For English** (Rule 1 primary):
- Tokenizer: 19,009 PMI tokens (14.5 % of vocab).
- Pretraining: ~60 % of consumed tokens.
- Matches observed.

The rules predict the observed footprints with reasonable accuracy.
They form a coherent (if not principled) effective policy.

## Where the rules don't fit cleanly

- **Catalan**: 4,267 PMI tokens (3.6 % of vocab) despite only 15.5 M FW2
  docs (rank 32). Predicted by Rule 11 (Latin-shared) — Catalan's mass
  is mostly shared Latin tokens, not Catalan-unique.
- **Estonian / Albanian / Latvian**: each have 1,000-1,500 PMI tokens
  with 8-12 M docs. Same explanation — Latin-shared merges lift the
  count regardless of doc volume.
- **Hindi**: 1,388 PMI tokens (1.1 %) despite only 20.6 M FW2 docs and
  being in Mistral-strong-11. The Mistral-strong-11 inheritance is
  weaker for Hindi than for Korean / Arabic — why is unclear
  ([unaccounted]).

These outliers don't change the effective-policy picture for Greek;
they show that the policy is more like "default rules + per-
language tooling investments" than "uniform rule set."

## Summary

The effective policy producing the observed allocation can be stated
as 15 rules. Of these:

- **5 are stated by primary sources**: Rules 1 (data dominance), 2
  (natural frequency), 4 (10/33 retention), 10 (Gini fairness), 13
  (public framing), 15 (aggregate count marketing).
- **3 are mathematical/structural constraints**: Rules 1 partial, 11
  (script isolation), 12 (BPE frequency).
- **5 are tooling gaps with unstated rationale**: Rules 3 (HQ-20 list),
  5 (0.95 ring), 6 (toxicity-9), 7 (OCR-5), 8 (ParaDocs-6), 9
  (Mistral-11).
- **1 is a chosen public-investment policy**: Rule 14 (Romansh
  affordance).

The next phase (`04_rational_policy.md`) splits these into necessary
and accidental components and derives what the rational core implies
for Greek.
