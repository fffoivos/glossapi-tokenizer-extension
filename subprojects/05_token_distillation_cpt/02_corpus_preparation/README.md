# 02 — Corpus preparation

> **In one line:** the five-stage pipeline that turned raw Greek web and academic text into the CPT training corpus — and, more often than not, measured a proposed cleaning rule carefully enough to decide *not* to ship it.
> **Period:** 2026-06-02 (earliest report timestamps) → 2026-08-12 (the reusable contamination audit), with a late recovery of uncommitted work on 2026-09-01. **Status:** completed for the CPT launch. Two stages ended with a deliberate near-no-op; one ended at a dry run it never applied.
> **Came from / led to:** the mix decision in [`../PRODUCTION_MIX_DECISION_20260612.md`](../PRODUCTION_MIX_DECISION_20260612.md) → this → the production build in [`../04_full_corpus_preparation`](../04_full_corpus_preparation/README.md) and the training runs in [`../03_training_experiments`](../03_training_experiments/README.md)

## Why this existed

The ~60 B-new-token CPT mix is ~79% new Greek, drawn from HPLT web text plus eight academic
sources. None of it arrives clean: HPLT rows carry extraction residue and boilerplate that
document-level filtering does not touch; academic PDFs converted to Markdown carry bibliographies,
tables of contents and footnote apparatus; Greek web text contains verbatim benchmark items; and
the corpus had to be anonymized to Apertus parity before it could be used or published. The
directory is the reusable component library for all of that. Production orchestration — the
receipted, resumable build that actually consumes these components — lives in
[`../04_full_corpus_preparation`](../04_full_corpus_preparation/README.md).

The stage order is fixed and recorded in [`../ARCHIVE.md`](../ARCHIVE.md):
**clean → dedup-validate → decontaminate → anonymize → shard**, CPU-only throughout (no GPU is
requested for cleaning, decontamination, anonymization or tokenizer preprocessing).

## History

Two of the five stages were built before this directory entered git: `10_clean_hplt`, `20_dedup`,
`30_decontaminate` and `40_anonymize` all arrived in one bulk commit, `a19c136f`
(2026-06-11, "Checkpoint pending project updates"), so their chronology comes from report
timestamps inside the files rather than from commit dates.

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-06-02 | k-gram overlap decontamination scanner built and LLM-reviewed on the 5 B corpus | Archived, not tuned — 90.7% of GreekMMLU items cannot form a 13-token answer k-gram | [`30_decontaminate/_archive_k13_overlap_method/`](30_decontaminate/README.md) |
| 2026-06-02/03 | DCLM-adapted rule with a proximity gate, judged by gpt-5.5 on 250 pairs | **Production filter:** rule `correct_only`, 270 items (1.62%) / 96 documents removed at 92.4% judged precision | [`30_decontaminate/experiments/v2_full_5B_correctonly_20260603/`](30_decontaminate/README.md) |
| 2026-06-05/06 | Two-day HPLT audit: 21-class error taxonomy, 947 rows hand-reviewed, five gates | **No broad rule shipped.** Exact-action precision `trim_span` 0.083, `quarantine` 0.364, `drop_doc` 0.568 — none near the 95% bar | [`10_clean_hplt/`](10_clean_hplt/README.md) |
| before 2026-06-11 | Apertus-parity PII masker with a Greek 27-char IBAN fix; a Greek structured-ID recall scan | Masker shipped; the ΑΦΜ/ΑΜΚΑ/plate/name detectors deliberately **not** shipped | [`40_anonymize/`](40_anonymize/README.md) |
| before 2026-06-11 | Dedup validator for the existing SELECTED artifact, hardened by adversarial `codex exec` review | Dedup for this launch is *validation*, not re-derivation | [`20_dedup/`](20_dedup/README.md) |
| 2026-06-13 | 25-agent study of academic reference handling; Rust `reference_detector` built and run at corpus scale | REMOVE-by-segmentation adopted; STRUCTURE and MASK rejected; the **footnote stream, not the end list**, is the dominant sink | [`15_clean_academic/`](15_clean_academic/README.md) |
| 2026-06-16 → 07-23 | Four generations of bibliography model, three annotation regimes, two sealed cohorts | `heading_lexgate` @0.98 supersedes the incumbent — 86.0% of bibliography characters removed for 0.258% body damage, against 53.9% / 0.505% | [`15_clean_academic/eval/`](15_clean_academic/eval/README.md) |
| 2026-07-12 | Sequence supervision retro-labelled | The 2,000-item `STRUCT_2K_gold.jsonl` is **LLM silver, never human gold** | `e7236f48` |
| 2026-07-26 | Rust port of the shipped line model | Decision-equivalent: **210,704/210,704** line mask, 142 s per cohort against ~15 days of single-stream Python | [`15_clean_academic/bib_line_model/`](15_clean_academic/bib_line_model/README.md) |
| 2026-07-27/28 | Receipt-bound dry run over all 202,792 academic documents, then a 209-item QA gate | 2.93 B characters (6.06%) would be removed; QA passed; apply authorized but **not run here** | [`15_clean_academic/production/`](15_clean_academic/production/README.md) |
| 2026-08-11 | Row-preserving anonymized v2 release pipeline written (recovered 2026-09-01) | Written and unit-tested; **no run receipt exists in this tree** | [`40_anonymize/hf_v2_release/`](40_anonymize/README.md) |
| 2026-08-12 | Reusable benchmark-contamination audit for the whole native-Greek suite | A **post-hoc scoring correction**, not a corpus change; the June rule stays the corpus filter | [`30_decontaminate/BENCHMARK_CONTAMINATION_AUDIT.md`](30_decontaminate/README.md) |

## Outcome

What actually shipped into the CPT launch is much narrower than what was built — and that is the
result, not a shortfall. Per [`../ARCHIVE.md`](../ARCHIVE.md):

- **HPLT cleaning:** only the E001 replacement-character/control-residue fix is in the launch path
  (32 matching rows in 46.5 M, 166 characters). *"Broader HPLT cleaning categories were explored
  but not approved as destructive production overlays."* Two standing rules came out of it: only
  exact, observable, high-confidence artifacts are eligible for automatic transformation, and
  **source text is immutable — cleaning happens through derived/shadow outputs.**
- **Academic cleaning:** for this CPT run the tracked structural policy stayed `audit_only`, both
  materialization flags false, and Stage58 a deterministic no-op. The bibliography cleaner targets
  the *published v2 dataset*, not the training corpus that had already been built.
- **Decontamination:** GreekMMLU only, rule `correct_only`, parameters encoded in
  `../03_training_experiments/dataset_build/stageA_clean_decontam.sbatch`.
- **Anonymization:** email, IP and IBAN masked to the reserved Apertus tokens, after
  decontamination.
- **Dedup:** validation and characterization of the existing selected artifact.

Carried forward: the receipt-bound production pattern (immutable contract, stable unit IDs,
fail-closed gates, atomic ledgers) became the template used in
[`../04_full_corpus_preparation`](../04_full_corpus_preparation/README.md); the contamination
audit tables feed `../../09_full_8b_cpt_results_analysis`; and the cleaned-v2 apply run, its token
count and its publication were all left open.

## Stages

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`10_clean_hplt/`](10_clean_hplt/README.md) | Audit the 46.5 M-row Greek HPLT slice for in-row artifacts | 2026-06-05 → 06-06 | completed; broad cleaner **not** promoted | 947 rows reviewed; no destructive action reached 95% Wilson-lower precision |
| [`15_clean_academic/`](15_clean_academic/README.md) | Bibliography / ToC / reference removal for the academic sources | 2026-06-13 → 07-28 | completed to dry-run + QA | `heading_lexgate` @0.98; 6.06% of characters removed in a 202,792-document dry run |
| [`20_dedup/`](20_dedup/README.md) | Prove the existing deduplicated artifact is the right one | before 2026-06-11 | completed | Anchored validation + provenance manifest; superseded by the Phase-04 content-bound dedup |
| [`30_decontaminate/`](30_decontaminate/README.md) | Remove benchmark leakage; later, audit it post hoc | 2026-06-02 → 08-12 | completed | 270 GreekMMLU items / 96 documents removed at 92.4% judged precision |
| [`40_anonymize/`](40_anonymize/README.md) | Apertus-parity PII masking, and the anonymized v2 release | before 2026-06-11; 2026-08-11 | masker completed; release pipeline written, unrun here | Greek 27-char IBAN fix over Apertus's regex; fixed-point masking added 2026-09-01 |

## Where things are

| Path | What |
|---|---|
| [`15_clean_academic/production/policy.json`](15_clean_academic/production/policy.json) | The frozen bibliography-cleaning policy: 13 ranks, 202,792 analysed / 175,242 apply rows, model, QA gate, licence override. |
| [`15_clean_academic/eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](15_clean_academic/eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json) | The machine-readable model decision with receipts and known limitations. |
| [`10_clean_hplt/reports/policy_recommendation_20260606T122100Z.json`](10_clean_hplt/reports/policy_recommendation_20260606T122100Z.json) | The per-action HPLT verdicts — the reason no broad rule shipped. |
| [`30_decontaminate/scripts/decontaminate.py`](30_decontaminate/scripts/decontaminate.py) | The production decontamination scanner; its docstring is the method specification. |
| [`40_anonymize/scripts/pii_masker.py`](40_anonymize/scripts/pii_masker.py) | The masker, with the one deliberate deviation from Apertus parity documented in-file. |
| [`40_anonymize/hf_v2_release/configs/release.json`](40_anonymize/hf_v2_release/configs/release.json) | The pinned anonymized-release contract (input revision, tokenizer, dedup receipt, 37-source taxonomy). |
| [`../ARCHIVE.md`](../ARCHIVE.md) | "Corpus-Prep Method Summary" — the standing decisions, after the corpus-prep markdown tree was collapsed. |

## Notes on this directory's history in git

Much of the July 2026 work happened on parallel `codex/*` branches and reaches the consolidation
branch as merges, so the log reads as a merged chronology rather than the order the work was done:

- `codex/toc-bib-sealed-annotation`, `codex/toc-bib-evolution`, `codex/toc-bib-sealed-inference`
  and `codex/toc-bib-header-deploy` were merged on 2026-07-22 into
  `codex/worktree-consolidation-v2-20260722` (`3b8fe685`, `e2378060`, `de10e413`, `8f07b616`).
- `codex/bib-nextgen-lexicon-gate` carried the 07-22/07-23 lexicon-gate and cohort-2 work; it is
  named as the branch of record in `15_clean_academic/eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json`.
- `codex/bib-cleaning-production-hardening` (corpus workflow) and `codex/bibliography-hardening`
  (glossAPI) carried the 2026-07-27/28 production work, and are named in
  [`15_clean_academic/BIB_CLEANING_IMPLEMENTATION_20260727.md`](15_clean_academic/BIB_CLEANING_IMPLEMENTATION_20260727.md).
- Commit `2aec4a66` (2026-09-01) recovered files that had existed only as untracked or modified
  files in local worktrees: `15_clean_academic/BIB_CLEANING_HANDOVER_20260727.md`, the whole
  `40_anonymize/hf_v2_release/` unit, and a fixed-point rewrite of `40_anonymize/scripts/pii_masker.py`.
