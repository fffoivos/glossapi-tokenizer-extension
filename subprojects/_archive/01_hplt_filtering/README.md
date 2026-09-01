# 01 — HPLT filtering (the "clean60" Greek slice)

> **In one line:** built the filtered Greek slice of HPLT 2.0 — `HPLT/ell_Grek_ge8_no_mt_clean60` — that supplied half of C3's tokenizer-training mix; finished and archived.
> **Period:** 2026-04-10 (`f21eed85`) → 2026-05-14 (`002bddc5`, archive move). **Status:** completed; the slice was published as [`fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`](https://huggingface.co/datasets/fffoivos/hplt-greek-ge8-no-mt-clean60-wave4) and consumed by C3.
> **Came from / led to:** project restructure `f21eed85` → this → [`../01_1_corpus_dedup`](../01_1_corpus_dedup/README.md) (dedup over the integrated corpus) and [`../01_2_training_dataset_mix`](../01_2_training_dataset_mix/README.md) (the mix that used it).

## Why this existed

The GlossAPI corpus alone was ~52 B characters after quality filtering — too small and too heavily academic/PDF-derived to train a Greek tokenizer extension on. HPLT 2.0's `ell_Grek` slice was the only web-scale Greek pool available, but raw HPLT carries machine-translated pages, low-quality bins and its own schema. This subproject turned it into a source that the rest of the pipeline could treat like any other GlossAPI dataset: same 21-column canonical parquet schema, same quality columns, provenance folded into `source_metadata_json`.

## History

The subproject's own commit trail is thin (the directory was moved into `_archive/` in a single commit), so the sequence below is reconstructed from the README's own revisions plus the run reports that reference it.

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-04-10 | Subproject created in the first clean-subprojects restructure. Deliverable was a *manifest* (shard, doc id, URL, quality bin, register labels, char count). | Decided: no contiguous-prefix sampling from sorted shards; stratify on HPLT metadata; keep quality bins 8–10; use web-register labels for diversity; "do not trust released HPLT dedup as sufficient by itself". | `f21eed85` |
| 2026-04-10 | Canonical web-register tuple analysis added. | 19 register tuples frozen as the fast-review target set (news report, opinion blog, research article, … and `Machine translated or generated`). | `ee234083`; [`fast_review_target_categories.txt`](fast_review_target_categories.txt); `scripts/analyze_hplt_register_tuples.py`, `scripts/sample_hplt_full_register_tuples.py` |
| 2026-04-11 | Deliverable rewritten from "manifest" to "upload-ready parquet in the canonical GlossAPI schema"; `build_hplt_hf_slice.py` added. | Added the exclusion of `Machine translated or generated`. The "don't trust HPLT dedup" line was dropped — dedup ownership moved to `01_1`. | `2c803082` (README diff) |
| 2026-04-12 | Track explicitly decoupled from the tokenizer critical path. | Decided: the tokenizer path may proceed from the local canonical source-parquet tree without waiting for the HF upload to finish. | `0a8a50be` (README diff) |
| 2026-04-12 | The **"clean60" rule** landed. | The slice must run real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring and drop `greek_badness_score > 60` — this is what the `clean60` suffix names. | `ed598b30` (README diff) |
| 2026-04-13 → 04-14 | Worker orchestration + repo canonicalization. | `wait_for_hplt_and_integrate.sh` / `integrate_hplt_slice_into_working_release.py` became the repo-owned entry points; this repo became the canonical home of the pipeline code. | `ca6f6a36`, `a062d0aa` |
| 2026-04-15 | Recovery plan records the build as finished. | "HPLT `clean60` build is complete. HPLT integration into the working source release is complete." | [`../../../docs/_archive/PIPELINE_RECOVERY_AND_SCALE_PLAN.md`](../../../docs/_archive/PIPELINE_RECOVERY_AND_SCALE_PLAN.md) § Current Diagnosis |
| 2026-04-28 | **Dead end / abort.** The wave-3 production run was aborted at `09:20Z` because HPLT rows carried no upstream `greek_badness_score` and the split exporter still admitted null-badness rows — making the HPLT-only and 70/30 tokenizer inputs unjudgeable. | Fix: the reclean driver fills missing Greek badness for HPLT without overwriting mojibake scores, and strict filters fail closed on missing scores. After the fix: 48,728,774 HPLT rows scored, **0 missing**. | [`../01_0_cleaning_iteration_and_thresholds/PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md`](../01_0_cleaning_iteration_and_thresholds/PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md); [`../01_0_cleaning_iteration_and_thresholds/WAVE3_PRODUCTION_PROGRESS_2026-04-28.md`](../01_0_cleaning_iteration_and_thresholds/WAVE3_PRODUCTION_PROGRESS_2026-04-28.md) |
| 2026-04-28 | Independent tokenizer review of the wave-3 arms. | The HPLT-only fresh tokenizer (F2) showed **0** glyph/PDF-residue tokens, **0** PostScript literals, **1** mojibake marker out of 50,000 — "HPLT looks clean by the artifact classes we were targeting". | [`../01_0_cleaning_iteration_and_thresholds/WAVE3_TOKENIZER_REVIEW_2026-04-28.md`](../01_0_cleaning_iteration_and_thresholds/WAVE3_TOKENIZER_REVIEW_2026-04-28.md) |
| 2026-05-11 | Slice frozen as the wave-4 release and used by C3. | 250 parquets, ≈103 GB, **48,728,774 rows**, 44.196 B tokens under the ModernGreek-148k tokenizer. | [`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md) § Source pool 2 |
| 2026-05-14 | Archived with a "DONE for the C3 shipping path (as of 2026-05-11)" status block. | No further work planned. | `002bddc5` |

## Outcome

- **Shipped artifact:** `HPLT/ell_Grek_ge8_no_mt_clean60`, published as `fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`. Six filters define it: `ell_Grek` language tag; HPLT quality bin ≥ 8; drop `Machine translated or generated`; run upstream `Corpus.clean` scoring; drop `greek_badness_score > 60`; normalize to the 21-column canonical schema with `source_doc_id = "hplt::<shard>::<id>"` ([`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md)).
- **Consumed by C3:** of 48,577,489 rows available at mix-build time, **13,906,493 rows / 52.47 B chars** were sampled into the 50/50 mix — 28.6 % of the slice. The unsampled ~34.7 M docs later became the pool for the verified-virgin held-out eval that replaced the leaky val/test splits ([`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md) § Held-out integrity).
- **Carried forward:** the integrated slice went into `01_1`'s dedup run and `01_2`'s mix builder; nothing HPLT-specific survived past `01_2`.
- **Left open** (from [`TODO.md`](TODO.md), never closed): freezing the filtering policy in machine-readable form; deciding whether `filter == keep` is required; a real-document smoke fixture for HPLT filtering/integration rather than synthetic contract tests only; rebuilding the human-review sample under the final quality policy instead of the stale exploratory one.

## Where things are

| Artifact | Role |
|---|---|
| `scripts/build_hplt_hf_slice.py` | Builds the upload-ready canonical-schema slice — the entry point for the whole subproject. |
| `scripts/integrate_hplt_slice_into_working_release.py`, `scripts/wait_for_hplt_and_integrate.sh` | Integration into the working source release; the repo-owned wrappers the E2E verification exercised. |
| `scripts/inspect_hplt_greek_metadata.py`, `scripts/build_hplt_greek_manifest.py`, `scripts/analyze_hplt_quality_bins.py` | The exploratory metadata / quality-bin probes that set the ≥8 bin policy. |
| `scripts/hplt_web_register.py`, `scripts/export_hplt_web_register_mapping.py`, `scripts/analyze_hplt_register_tuples.py` | Web-register mapping and diversity analysis. |
| [`fast_review_target_categories.txt`](fast_review_target_categories.txt) | The 19 register tuples targeted for fast human review. |
| [`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md) | The authoritative record of what the slice contained and how C3 sampled it. |

## Working documents

Historical, kept for traceability:

- [`TODO.md`](TODO.md) — the open action list as of the archive move; several items were never closed (see Outcome).
- `scripts/mark_full_sample_by_greek_badness.py`, `scripts/repair_txt_dirs_from_jsonl.py`, `scripts/download_public_dataset_snapshot.sh` — one-off review/repair helpers from the exploratory phase.
