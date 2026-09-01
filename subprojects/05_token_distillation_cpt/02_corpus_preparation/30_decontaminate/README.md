# 30 — Decontaminate

> **In one line:** a k-gram overlap scanner was built, LLM-judged, found unsound, and replaced by a DCLM-style question+correct-answer rule with a proximity gate (92.4% judged precision) — which became the production GreekMMLU filter; two months later a second, *reusable* audit was built for the whole native-Greek benchmark suite as a post-hoc evaluation correction.
> **Period:** 2026-06-02 → 2026-08-12 (content timestamps; git entry 2026-06-11 `a19c136f` for the June work, 2026-08-12 for the audit). **Status:** completed. The `correct_only` GreekMMLU filter is the launch decision; the 2026-08-12 audit is a scoring correction, not a corpus change.
> **Came from / led to:** [`../20_dedup`](../20_dedup/README.md) → this → [`../40_anonymize`](../40_anonymize/README.md); the reusable audit feeds [`../../../09_full_8b_cpt_results_analysis`](../../../09_full_8b_cpt_results_analysis)

## Why this existed

The CPT run is scored on GreekMMLU and a wider native-Greek suite. Greek web text contains
driving-test question banks, exam rehearsal sites and study-guide hosts that reproduce
benchmark items verbatim, so any headline gain could be leakage. The stage had to answer two
different questions: *which corpus documents must be removed before training* (June), and
*which already-trained scores must be discounted after the fact* (August).

## History

### 2026-06-02 — the k-gram overlap method (built, then archived)

`scan_5b_decontamination.py` streamed the 5 B Task-1 JSONL and scored **exact normalized word
n-gram overlap** at k ∈ {8, 13} against four benchmarks, with overlap-fraction thresholds
{0, 0.1, 0.2, 0.4, 0.6, 0.8}. Run `05_decontam_5b_20260602T011447Z` read 22,935,074,496 bytes
over 3,739,911 rows and indexed 1,709,528 unique query grams from 147,912 surface keys
(20,964 of which produced **zero** shingles). Matching docs: greekmmlu 5,459, ilsp_mcqa_asep
2,594, ilsp_medical_mcqa 36, plutus_qa 37
([`_archive_k13_overlap_method/reports/decontamination_audit_5b_20260602T011447Z.json`](_archive_k13_overlap_method/reports/decontamination_audit_5b_20260602T011447Z.json)).
1,202 matches were rated
([`_archive_k13_overlap_method/reports/match_review_ratings_20260602.csv`](_archive_k13_overlap_method/reports/match_review_ratings_20260602.csv)):
942 `EVAL_REHEARSAL_SITE` and 120 `STUDY_GUIDE_HOST` at severity 5, but also 43
`DOMAIN_BLOG` and 35 `REGULATION_SOURCE` at severity 2–3. The whole method was moved to
`_archive_k13_overlap_method/` — the directory name *is* the verdict.

### 2026-06-02 — why it failed: the answer space collapses

A discovery-pass clustering of `dascim/GreekMMLU` (`All`/`test`, 16,632 items) explains the
problem: the answer-count histogram is `{2: 3152, 3: 3478, 4: 10002}` and
`answer_token_collapse_at_k13_fraction = 0.907347` — **90.7% of items cannot form a 13-token
answer k-gram at all**, because Greek MCQ options are frequently one token (Σωστό/Λάθος).
Evidence: [`classification/reports/greekmmlu_clusters_20260602T140045Z.json`](classification/reports/greekmmlu_clusters_20260602T140045Z.json),
produced by [`classification/scripts/cluster_greekmmlu_items.py`](classification/scripts/cluster_greekmmlu_items.py)
(whose docstring is explicit: *"This script does NOT label categories. It surfaces the
empirical clusters."*).

### 2026-06-02 — v2: the DCLM rule plus a proximity gate, LLM-validated

`decontaminate.py` re-implements the DCLM MMLU check (Li et al. 2024, arXiv 2406.11794)
for Greek and adds one thing the paper does not have: the option side must occur **near and
after** the question-stem match, within `--max-gap-tokens` (50 long, 5 short). Short options
fall back to contiguous-token-subsequence presence. Three option-side rules are emitted in
parallel — `correct_only`, `any` (the paper rule), `all` — plus a four-way categorization
(`q_only`, `q_plus_correct_only`, `q_plus_wrong_only`, `q_plus_correct_and_wrong`).
A 250-pair gpt-5.5 codex judge audit
([`experiments/v2_validation_250pair_codex_20260602/`](experiments/v2_validation_250pair_codex_20260602))
graded each match `NO_OVERLAP` / `TOPIC_OVERLAP` / `SOURCE_OVERLAP` / `STRONG_CHEATING` /
`DEFINITE_CHEATING`.

### 2026-06-03 — the production rule

Full 5 B scan under `correct_only`
([`experiments/v2_full_5B_correctonly_20260603/greekmmlu_dclm_audit_20260603T074147Z.json`](experiments/v2_full_5B_correctonly_20260603/greekmmlu_dclm_audit_20260603T074147Z.json),
schema `greekmmlu-dclm-decontam-v3-categorized`, k=8): **270 contaminated items (1.6234%),
379 item-doc pairs, 96 contaminated document ids**. The headline block records the rationale
verbatim: *"PRODUCTION RULE… 92.4% codex-judge precision on a 118-doc sample (gpt-5.5,
250-pair audit, 2026-06-02). Residual false positives are source-overlap on Greek regulation
portals (opengov.gr, eur-lex, elinyae) — these docs contain the regulation text that becomes
the eval question, which is appropriate to remove from the training corpus anyway."*
A second judge pass over all correct-only matches is in
[`experiments/v2_full_5B_correctonly_20260603/codex_review/`](experiments/v2_full_5B_correctonly_20260603/codex_review).

This is the launch decision, restated in [`../../ARCHIVE.md`](../../ARCHIVE.md):
*"Decontamination scope for this launch: GreekMMLU only, rule `correct_only`, with the
DCLM-style scanner parameters encoded in `03_training_experiments/dataset_build/stageA_clean_decontam.sbatch`."*

### 2026-07-12 — folded into the full-corpus pipeline

Commit `01cba0ee` ("Complete receipt-bound full-corpus CPT pipeline") wired the queries and
scanner into the Phase-04 build as a GreekMMLU freeze + decontamination stage; see
[`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md).

### 2026-08-12 — the reusable benchmark-contamination audit

Eight commits in one day (`efd9eb8a` → `159b5bb5`) built a *different* instrument, documented
in [`BENCHMARK_CONTAMINATION_AUDIT.md`](BENCHMARK_CONTAMINATION_AUDIT.md). Its opening
decision is that `decontaminate.py` **stays** the calibrated GreekMMLU corpus filter but is
**not** a safe reusable audit for the wider suite, for six stated reasons: stems shorter than
K were invisible; OYXOY prompts contain evaluator-authored scaffolding absent from the source;
OYXOY NLI scores one source pair as three binary decisions; WIC/WSD/metaphor labels are
carried by source pairs, not by the generated Ναι/Όχι; the scanner kept document ids but not
shard/row/line evidence; and *"its 92.4% precision estimate is specific to the reviewed
GreekMMLU rule and must not be asserted for a new benchmark family without review."*

The replacement uses 8-token k-grams for questions of ≥8 tokens and the **complete normalized
stem** for 3–7-token questions, and publishes every question-anchor hit while recommending
exclusion only when a second **label-bearing** surface appears in the proximity window (the
per-family table in the doc: correct answer for the MCQ families; hypothesis for NLI;
definition for WSD; second usage for WIC). Question-only matches stay `candidate` and are
never auto-discounted. Output is a receipt-bound Hugging Face payload under
`benchmark_contamination/native_greek_suite_v1/`. Execution: one CPU-only `debug`
allocation, four lanes over the 431 local Parquet shards, finalizing only after all 431 shard
receipts validate. Text-only Protipa is excluded because its HF access gate was not approved.
The doc is emphatic that this is *"a post-hoc evaluation correction for a completed run.
It does not claim to undo training exposure and it does not mutate the published dataset."*

## Outcome

- **Production filter (June):** DCLM-adapted, proximity-gated, rule `correct_only`, k=8,
  gaps 50/5 → 270 GreekMMLU items (1.62%) and 96 documents removed from the 5 B corpus,
  at 92.4% judged precision.
- **Method reversal:** the k-gram overlap approach was archived, not tuned. Its failure mode
  is quantified (90.7% of GreekMMLU items collapse at k=13) rather than asserted.
- **Known residual:** the surviving false positives are Greek regulation portals, accepted on
  the argument that removing them is right anyway.
- **August correction:** a second, benchmark-family-aware audit that publishes evidence and a
  strict `recommended_excluded_example_ids.jsonl`, consumed by
  `09_full_8b_cpt_results_analysis/evaluation/rescore_contamination_filtered.py` to report
  both full and strict-filtered scores. No corpus row changes.

## Where things are

| Path | What |
|---|---|
| [`BENCHMARK_CONTAMINATION_AUDIT.md`](BENCHMARK_CONTAMINATION_AUDIT.md) | The 2026-08-12 decision, match/discount policy and published-table contract. |
| [`scripts/decontaminate.py`](scripts/decontaminate.py) | The production DCLM-adapted scanner (its docstring is the method spec). |
| [`scripts/build_decontamination_queries.py`](scripts/build_decontamination_queries.py) | Binds the audit to the exact frozen examples the evaluator scored. |
| [`scripts/audit_benchmark_contamination_parquet.py`](scripts/audit_benchmark_contamination_parquet.py) · [`run_benchmark_contamination_audit.sbatch`](scripts/run_benchmark_contamination_audit.sbatch) | The resumable 431-shard scan. |
| [`scripts/finalize_benchmark_contamination_audit.py`](scripts/finalize_benchmark_contamination_audit.py) · [`publish_benchmark_contamination_audit.py`](scripts/publish_benchmark_contamination_audit.py) | Receipt-bound tables and the separate publish step. |
| [`scripts/sample_benchmark_contamination_evidence.py`](scripts/sample_benchmark_contamination_evidence.py) · [`build_benchmark_contamination_adjudication_packet.py`](scripts/build_benchmark_contamination_adjudication_packet.py) | Deterministic review sample and the compact adjudication packet. |
| [`scripts/codex_judge_contamination.py`](scripts/codex_judge_contamination.py) · [`codex_review_matches.py`](scripts/codex_review_matches.py) | The LLM judges used for the June precision estimate. |
| [`tests/test_benchmark_contamination_audit.py`](tests/test_benchmark_contamination_audit.py) | Audit unit tests. |

## Working documents

Historical run evidence, nothing deleted:

- [`experiments/v2_validation_250pair_codex_20260602/`](experiments/v2_validation_250pair_codex_20260602) — the 250-pair judge validation that produced the 92.4% figure (three audit JSONs at 19:41 / 20:45 / 20:54 Z, two summary CSVs, a doc-id→URL map).
- [`experiments/v2_full_5B_correctonly_20260603/`](experiments/v2_full_5B_correctonly_20260603) — the production scan: audit JSON, the 96 contaminated doc ids, and the full-correct-only judge CSV.
- [`_archive_k13_overlap_method/`](_archive_k13_overlap_method) — **superseded** k=8/13 overlap scanner, its 5 B audit JSON and the 1,202-row rating CSV. Kept as the record of why the method was abandoned.
- [`classification/`](classification) — the GreekMMLU cluster-discovery pass (script + one report) that quantified the answer-token collapse.
