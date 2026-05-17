# Investigations tracker — Apertus per-language policy analysis

Single index for this sub-subproject. **Final update 2026-05-18 (archive mode)**.

## Final state

The C3 cutoff was decided empirically on 2026-05-18 in
[`../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md):
**17,408 added units (curated + backfilled), total vocab 148,480**.

This subproject's policy-archaeology approach **did not produce the
budget number**. What it did contribute, which informed the
downstream measurement:

1. **Provenance finding** ([`11_tokenizer_provenance.md`](11_tokenizer_provenance.md))
   reframed the question from "what is Apertus's vocab policy" to
   "what is Mistral's, since Apertus inherited it verbatim." This
   pulled the multi-criteria Apertus-pretraining hypotheses out of
   scope for the cutoff question.
2. **Suite discovery** seeded `02_1_7`: the
   `swiss-ai/tokenizer-intrinsic-evals` repo (TokEval fork) is what
   `02_1_7` uses to measure cutoff variants on Apertus's own metric
   surface.
3. **Speaker-count hypothesis invalidation** closed a dead-end
   framing.
4. **Reference evidence** (HPLT 3.0, FLORES+ 55-list, Latin-vs-
   Ancient-Greek, FW2-HQ HQ-20 documentation) remains useful as
   background.

Six open hypotheses (5-10) were stubbed but never executed; the
cutoff was decided without resolving them. Phases 3-4 (synthesis +
fairness definition + Greek budget) were attempted, found buggy, and
also superseded by the empirical measurement. Both groups archived in
[`_deprecated_20260518/`](_deprecated_20260518/).

## Goal

Derive **the policy that produced the per-language token allocation
in Apertus's tokenizer** (1,479 Greek tokens, 2,768 Latin tokens, etc.)
so the user can reason about a principled C3 cutoff for Greek.

**Scope-down (2026-05-17)**: per
[`11_tokenizer_provenance.md`](11_tokenizer_provenance.md), the
per-language BPE allocation is **inherited 100 % from Mistral's tekken
v3**. Apertus changed only the special-token block (ids 0-999),
adding 58 new code/math/PII/chat-template tokens and truncating 486
trailing rare-frequency BPE entries. Apertus made **zero per-language
BPE modifications**.

Therefore the relevant policy is **Mistral's tokenizer-training
policy**, not Apertus's pretraining-data policy. Hypotheses about
Apertus's pretraining (HQ-20, EuroParl inclusion, toxicity-9, etc.)
explain the **pretraining data**, not the **vocab allocation we're
trying to extend**.

The goal is **not** to recommend a Greek budget directly. The goal is
to map the policy space so that the user can reason about which
budget is principled given that Mistral's allocation is the inherited
base and the C3 extension is the first per-language vocab decision
*inside Apertus's stack*.

## Why this matters

The current C3 cutoff REPORT
([`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md))
surveys six rhetorical anchors ("match Korean / German / French /
English-unique / English-total / fertility-elbow") none of which is
derived from a principle. To replace these with a principled anchor,
we need to know what policy Apertus actually follows. Each hypothesis
in this tracker is a candidate explanation for that policy.

## Outputs (deliverables when all hypotheses tested)

- This tracker, kept up to date.
- One numbered evidence doc per investigation.
- (Out of scope for now) A standalone fairness definition + Greek
  budget recommendation, picking from the surviving hypotheses. Set
  aside in `_my_synthesis_set_aside/` from the previous pass; can be
  revived once hypotheses are settled.

## Status table

| # | Hypothesis | File | Verdict | Notes |
|---|---|---|---|---|
| 0 | Tokenizer provenance + Apertus's modifications | [`11_tokenizer_provenance.md`](11_tokenizer_provenance.md) | **DONE** | Apertus inherits Mistral's BPE table 100 %. Only changed special-token block (ids 0-999, expanded from 514, +58 new named tokens, 486 trailing BPE entries truncated). Frames everything below. |
| 1 | Explicit author-stated goals | [`01_explicit_goals.md`](01_explicit_goals.md) | DONE | ~70 quotes across 15+ sources. No source cites Greek individually. |
| 2 | Implicit data + filter + math constraints | [`02_implicit_constraints.md`](02_implicit_constraints.md) | DONE | 8 constraints identified. Greek is excluded from every named tooling subset except FW2-HQ + EuroParl + Clean-Wiki + EuroBlocks. **Apertus-side**; doesn't affect inherited tokenizer. |
| 3 | Web footprint (HPLT 3.0) as selection criterion | [`03_evidence_HPLT3_FLORES_classical.md`](03_evidence_HPLT3_FLORES_classical.md) §1 | **PARTIAL FIT** | Apertus-HQ-20 ≈ HPLT 3.0 top-22 (good fit). Mistral-11 ≠ HPLT top-11. **HPLT is post-2024; tests Apertus's pretraining policy, not Mistral's tokenizer-training policy.** |
| 4 | FLORES+ Gini fairness as Apertus's explicit tokenizer criterion | [`03_evidence_HPLT3_FLORES_classical.md`](03_evidence_HPLT3_FLORES_classical.md) §2 | DOCUMENTED | Gini is **aggregate** over 55 langs, not per-language. Doesn't commit Apertus to Greek-specific fertility parity. Note: Apertus's Gini metric *selected the tokenizer*, didn't *train it* — Mistral's training mix is what produced the per-language allocation. |
| 5 | Latin / Ancient-Greek asymmetry | [`03_evidence_HPLT3_FLORES_classical.md`](03_evidence_HPLT3_FLORES_classical.md) §3 | DOCUMENTED | Latin gets ~2× Modern Greek's vocab tokens despite no policy mention. **Confirmed to be a Mistral-side artifact, not Apertus policy.** |
| 6 | Speaker count (L1 or total) as selection criterion | [`04_evidence_speakers.md`](04_evidence_speakers.md) | **INVALIDATED** | Top-by-L1 / top-by-total both fail. Bengali (rank 6 L1) absent from both lists; Greek (rank ~70 L1) in HQ-20. No source cites speakers. |
| 7 | Pre-2024 dataset landscape (what Mistral's BPE training corpus might have been) | [`_deprecated_20260518/05_evidence_pre2024_datasets.md`](_deprecated_20260518/05_evidence_pre2024_datasets.md) | **DEPRIORITIZED** | Cutoff decided in 02_1_7 without resolving this. Stub remains a viable starting point if ever needed. |
| 8 | Mistral inherits language list from prior multilingual models | [`_deprecated_20260518/06_evidence_inherited_from_priors.md`](_deprecated_20260518/06_evidence_inherited_from_priors.md) | **DEPRIORITIZED** | Same — superseded by direct empirical measurement. |
| 9 | Team / institutional nationality bias | [`_deprecated_20260518/07_evidence_team_institutional.md`](_deprecated_20260518/07_evidence_team_institutional.md) | **DEPRIORITIZED** | Same. |
| 10 | Commercial market footprint | [`_deprecated_20260518/08_evidence_commercial_markets.md`](_deprecated_20260518/08_evidence_commercial_markets.md) | **DEPRIORITIZED** | Was already scoped-out for tokenizer-vocab question; now archived. |
| 11 | Available multilingual benchmarks (XNLI / XCOPA / Belebele / FLORES) | [`_deprecated_20260518/09_evidence_benchmark_coverage.md`](_deprecated_20260518/09_evidence_benchmark_coverage.md) | **DEPRIORITIZED** | Same. |
| 12 | Reddit per-language footprint (pre-2024 API closure) | [`_deprecated_20260518/10_evidence_reddit_proxy.md`](_deprecated_20260518/10_evidence_reddit_proxy.md) | **DEPRIORITIZED** | Same. |

## Summary of what's been tested so far

Three hypothesis-classes have verdicts:

- **Web footprint** explains HQ-20 well, Mistral-11 only partially. The
  gap in Mistral-11 (Russian omitted) is the Western-European-tilt
  finding.
- **Speaker count** explains neither list. No source cites speakers.
- **Apertus's own Gini-on-FLORES+** is real but aggregate — it doesn't
  obligate Greek-specific fairness.

This leaves the **Mistral-11 selection criterion still unexplained**
under all tested hypotheses. Hypotheses 7-12 are the candidate next
explanations.

## Open questions Mistral-11 still poses

If neither web rank, speaker count, nor any stated criterion explains
Mistral's choice of:

- include Italian (HPLT 8, L1 rank 25) and Portuguese (HPLT 9, L1 5)
  — but the smaller Romance languages are HPLT-top-10 organically.
- include Korean (HPLT 22, L1 17) — a notable web-rank outlier among
  East Asian languages.
- include Arabic-MSA (HPLT 28) — a rank-28 web language with 0 L1
  speakers.
- include Hindi (HPLT 35, L1 4) — speaker-count Yes / web-rank No.
- **exclude Russian** (HPLT 2) — the second-largest web language.
- **exclude Bengali** (HPLT >50, L1 rank 6) — Asia's 6th-largest L1
  community.
- **exclude Vietnamese / Indonesian / Polish / Dutch / Turkish /
  Persian / Czech** (all HPLT 10-17) — these are FW2-HQ inclusions
  that Mistral didn't make.

… then there must be another rule. Hypotheses 7-12 are the candidates.

## Order of priority for the open investigations (scope-down applied)

After the provenance finding, only investigations that bear on
**Mistral's tokenizer-training policy** matter. Three remain, plus
two cross-cutting verifications:

1. **Hypothesis 8 — Mistral inherits from prior multilingual models.**
   Strongest structural answer. Mistral's tekken v3 added merges to
   tiktoken's base. The tiktoken base + Mistral's "11 particularly
   strong" language list closely tracks the XLM-R-100 / mT5-101 /
   BLOOM-46 lineage. Verify language-list intersections.
2. **Hypothesis 7 — Pre-2024 dataset landscape (narrowed).** What
   data was available to feed Mistral's BPE training in early 2024
   when tekken v3 was finalized? OSCAR-23.01, mC4, CulturaX,
   MADLAD-400, Wikipedia mid-2024 — match per-language sizes against
   Mistral-11.
3. **Hypothesis 9 (narrowed) — Mistral French team data bias.** The
   Mistral founders are French nationals in Paris. Did the
   tokenizer-training data over-weight French content? Per-language
   token-density patterns (do French tokens fragment less than
   Spanish per byte of training data?) can test this.

**Cross-cutting verifications (added 2026-05-17 review)**:

- **Gini-on-FLORES+ sweep** ([`12_gini_optimization.md`](12_gini_optimization.md)).
  Currently an experiment plan; not yet run. Confirms or invalidates
  the +3-5k optimum hypothesis.
- **Tail-token audit of the 486 dropped Mistral BPE entries**
  ([`11_tokenizer_provenance.md`](11_tokenizer_provenance.md) §
  "Difference 1"). Currently flagged as unverified. Decode Mistral's
  last 486 BPE entries, classify by script, check overlap with our
  Greek / Latin / Arabic PMI-promoted sets. Confirms or invalidates
  the "no impact on major-language allocations" hypothesis.

**Scoped out**: hypotheses 10, 11, 12 (commercial markets, benchmark
coverage, Reddit) — these affect Apertus's pretraining choices, not
the inherited Mistral BPE table that's the actual subject of the C3
extension.

**Review log**: see [`REVIEW_INTEGRATION_20260517.md`](REVIEW_INTEGRATION_20260517.md)
for the most recent review pass and how each finding was addressed.

## How to use this tracker

- Each open hypothesis is its own stub doc (05-10) with: scope,
  sources to check, output format, priority.
- When an investigation completes, update this tracker's status table
  to FIT / PARTIAL FIT / INVALIDATED with a one-line note.
- If a hypothesis is invalidated, the corresponding evidence doc
  stays — invalidation is itself a finding.
- The tracker is the single index. Other docs are evidence; this is
  navigation.

## Cross-references

- [`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
  — the cutoff decision this analysis ultimately serves.
- [`../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`](../../../docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
  — measured Greek share of Apertus pretraining.
- [`_my_synthesis_set_aside/`](_my_synthesis_set_aside/) — previous
  synthesis attempt; set aside pending completion of hypothesis testing.
