# 02.1.6 — Representation Policy Analysis

> **In one line:** an attempt to derive Greek's vocab budget from Apertus's implicit language policy instead of picking an anchor from a menu; it never produced a number, was wrong by ~3× where it did predict one, and its most valuable output was finding the evaluation suite that settled the question empirically.
> **Period:** 2026-05-17 (evidence + review pass) → 2026-05-18 (archive mode). **Status:** abandoned. All work committed 2026-05-18 (`7deea009`), already carrying its own archive banners.
> **Came from / led to:** [`../02_1_4_cutoff_analysis/`](../02_1_4_cutoff_analysis/README.md) (whose "match language X" anchors it tried to replace) → this → [`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/README.md) (which it seeded and which superseded it)

## Why this existed

[`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md) surveyed six anchors for Greek's budget — match Korean, German, French, English-unique (~13k), English-total (~19k), or the fertility elbow (~17k). None was derived from a principle; each was another language's empirical footprint, so choosing between them was choosing a rhetorical framing. This subproject asked the question one level up: *what policy produced those footprints?* Four planned phases — harvest stated goals, identify implicit structural constraints, synthesize the effective policy, split it into necessary core vs. accident and read off Greek's implied share.

## History

### 2026-05-17 — the provenance finding kills the premise

[`11_tokenizer_provenance.md`](11_tokenizer_provenance.md) diffed Apertus-8B-2509's tokenizer against `mistralai/Mistral-Nemo-Base-2407`. Result: **Apertus's per-language BPE allocation is 100 % inherited from Mistral.** Apertus changed only the front block — reserved ids 514 → 1000, 58 slots repurposed for code/math/PII/chat markers, and 486 trailing BPE entries truncated to hold vocab at 131,072. Zero per-language modifications. The relevant policy was therefore Mistral's, not Apertus's, and every hypothesis about Apertus's *pretraining* mix (HQ-20 coverage, EuroParl, the 9-language toxicity filter) explained the pretraining data rather than the vocab allocation being extended. The investigation scope collapsed on the spot.

### 2026-05-17 — a reviewer pass, and a hypothesis dies

[`04_evidence_speakers.md`](04_evidence_speakers.md) tested whether Mistral-11 / Apertus-HQ-20 membership tracks speaker counts. It does not — Bengali (232 M L1), W. Punjabi, Yue and Wu Chinese are in neither list. Hypothesis **invalidated**, dead-end framing closed.

[`REVIEW_INTEGRATION_20260517.md`](REVIEW_INTEGRATION_20260517.md) records a review-agent pass and how each finding was resolved: three internal docs disagreed about whether the fairness definition and Greek budget were deliverables (resolved: the tracker is the single source of truth); [`12_gini_optimization.md`](12_gini_optimization.md) overstated Gini as Apertus's "sole stated criterion" when the paper lists four metrics plus a smaller-vocab preference, and presented an unrun prediction as a result (resolved: relabelled "experiment plan", status banner added).

### 2026-05-18 — archive mode

The cutoff was decided empirically downstream at **17,408**. Two things were archived under [`_deprecated_20260518/`](_deprecated_20260518/README.md):

- the Phase 3–4 synthesis (effective policy, rational core, fairness definition, Greek budget), which recommended **+5,120** and contains two documented arithmetic errors — "both internally buggy AND wrong about the budget";
- six hypothesis stubs (05–10: pre-2024 datasets, inherited-from-priors, team/institutional, commercial markets, benchmark coverage, Reddit proxy) that were never executed.

[`12_gini_optimization.md`](12_gini_optimization.md) was marked SUPERSEDED and kept deliberately: it predicted N\* in the +3–5k range under Gini-only logic against a measured +17,408, and that gap is itself the finding — single-metric aggregate-fairness reasoning overweighted FLORES+ fairness against in-domain Greek fertility.

## Outcome

- **No budget produced.** The method did not reach a number; the number came from measurement in `02_1_7`.
- **Net contributions**: the Mistral-inheritance provenance finding, which reframed the question and pulled the Apertus-pretraining hypotheses out of scope; the speaker-count invalidation; and — the one with real downstream leverage — the discovery of `swiss-ai/tokenizer-intrinsic-evals` (TokEval), the suite Apertus itself used, which became the whole evidence base of `02_1_7`.
- **Reference evidence retained**: HPLT 3.0 baseline, the FLORES+ 55-language list, the Latin-vs-Ancient-Greek asymmetry, and FineWeb-2-HQ's 20-language coverage, in [`03_evidence_HPLT3_FLORES_classical.md`](03_evidence_HPLT3_FLORES_classical.md).
- **Standing lesson**, stated in the deprecation notes: policy reasoning and single-metric optimization both under-predicted the useful vocab budget by roughly 3×.

## Where things are

| What | Where |
|---|---|
| Final hypothesis-status table | [`INVESTIGATIONS_TRACKER.md`](INVESTIGATIONS_TRACKER.md) |
| The provenance finding | [`11_tokenizer_provenance.md`](11_tokenizer_provenance.md) |
| Invalidated speaker-count hypothesis | [`04_evidence_speakers.md`](04_evidence_speakers.md) |
| Superseded Gini plan (kept for the contrast) | [`12_gini_optimization.md`](12_gini_optimization.md) |
| Archived synthesis + unrun stubs | [`_deprecated_20260518/`](_deprecated_20260518/README.md) |

## Working documents

- Phase 1–2 evidence harvest: [`01_explicit_goals.md`](01_explicit_goals.md), [`02_implicit_constraints.md`](02_implicit_constraints.md) — thorough, did not inform the pick, kept as reference.
- Review record: [`REVIEW_INTEGRATION_20260517.md`](REVIEW_INTEGRATION_20260517.md). Status/scope banner: [`TODO.md`](TODO.md).
- `_deprecated_20260518/synthesis_with_known_errors/` — do not revive without redoing from scratch; the known errors are listed in its own README.
