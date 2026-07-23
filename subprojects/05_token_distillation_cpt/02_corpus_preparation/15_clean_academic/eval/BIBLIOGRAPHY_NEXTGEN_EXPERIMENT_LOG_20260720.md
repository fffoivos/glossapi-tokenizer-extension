# Bibliography next-generation experiment log (2026-07-20)

> **SUPERSEDED (2026-07-23).** The deployment conclusion below is out of date.
> On the fresh 150-document `bibliography_150_20260723_v2` cohort the incumbent is
> *strictly dominated*: it removes 53.9% of bibliography characters while destroying
> 0.505% of body characters, against 86.0% / 0.258% for `heading_lexgate`.
> See `BIBLIOGRAPHY_NEXTGEN_COHORT2_BAKEOFF_20260723.md` and
> `RECOMMENDED_BIBLIOGRAPHY_MODEL.json`. Metrics in this document are measured on
> the 20260718 cohort, which was subsequently shown to be optimistic.

## Objective and protocol

Improve the original bibliography pipeline without changing the sealed 143-document consensus-silver test set. Model and decoder choices are made from grouped document-level out-of-fold (OOF) development predictions over the 1,118-document silver corpus. The test labels remain unopened until all candidates, thresholds, and topology settings are frozen.

The deployment gate is:

- line precision at least 0.98;
- character precision at least 0.98;
- zero emitted non-BIB Markdown headings;
- at most 0.02 spurious blocks per zero-BIB document;
- among passing candidates, maximize the lower of line and character recall.

## Development data and shared feature table

- Source: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/struct2k_sources/struct2k_joint_20260712b/struct2k.LLM_silver.jsonl`
- Base table: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/table`
- Next-generation table: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_20260720/full_table_0a18c2a_r1`
- Materialization job: `2800604` (`COMPLETED`)
- Inventory: 1,118 documents, 939,014 lines, 139,243 BIB lines, 124 base features.

All line-model figures below are grouped OOF figures: each document is predicted only by a model that did not train on that document.

## Generations tried

| Generation | Main change | Line P | Line R | Char P | Char R | Spurious blocks / zero-BIB doc | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Incumbent entry decoder | Frozen entry probability plus constrained topology | 0.9927 | 0.5354 | 0.9901 | 0.6093 | 0.0150 | Passes gate, low recall |
| Full-document HistGB | All 939k lines; ordinary negatives restored; document weighting | 0.9815 | 0.8817 | 0.9802 | 0.9336 | 0.0902 | High recall, scope failures |
| Full-document TCN | Unweighted grouped OOF TCN | 0.9804 | 0.8749 | 0.9816 | 0.9270 | 0.0902 | Dominated by HistGB |
| Position-aware HistGB | Adds normalized document and physical-segment position | 0.9806 | 0.8851 | 0.9808 | 0.9350 | 0.0677 | Better recall and scope, still misses gate |
| Position-aware HistGB + linear component scope | OOF component-level keep/veto classifier at frozen threshold 0.90 | **0.9974** | **0.7891** | **0.9972** | **0.8426** | **0.0000** | **Passes gate; new development winner** |

The component scope model saw 2,425 OOF-predicted components: 2,273 touched a gold BIB region and 152 did not. It consumes only aggregate block evidence: relative document position, component size, line-model probability distribution, deterministic/role-signal density, neighboring line probabilities, and nearby Markdown heading-role probabilities. Its folds are grouped by document, like the line model.

## False-positive analysis

Exact OOF error receipts are under:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_20260720/audit_*_2c62547_r1.json`

The added false positives were overwhelmingly complete spurious components rather than boundary overruns. Manual inspection of the highest-contributing documents found citation-shaped footnote runs, reading/source lists, and URL-heavy appendices that the silver labels mark as outside bibliography sections. This explains why adding line features alone raises recall but cannot preserve scope. The component classifier addresses that error at the component level instead of weakening bibliographic feature detection or adding document-specific regexes.

## Frozen one-shot test candidates

Freeze receipt:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_20260720/frozen_candidates_099c6b1_r1.json`

Frozen before test-label access:

1. `incumbent_entry`: deployment-gated reference.
2. `position_hist_unscoped`: predeclared line-model ablation; does not pass the development spurious-block gate.
3. `position_hist_component_scope`: deployment-gated new candidate.

The test set is consensus silver, not human gold. No thresholds, topology settings, or candidate ranking may be changed after the one-shot report is produced.

## Code and verification

- Branch: `codex/toc-bib-sealed-annotation`
- Current test/inference code commit at freeze: `099c6b1`
- Immutable Clariden bundle: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/code_bundles/bib_nextgen_099c6b1_fullruntime`
- Focused Clariden verification: 20 tests passed.

Key modules:

- `bibliography_nextgen_table.py`: full-development feature materialization.
- `bibliography_nextgen_models.py`: grouped linear/HistGB line models and position/context features.
- `bibliography_nextgen_tcn.py`: grouped full-document TCN.
- `bibliography_nextgen_decode.py`: topology-constrained line-to-block decoder.
- `bibliography_nextgen_error_audit.py`: exact FP/FN component audit.
- `bibliography_nextgen_scope.py`: grouped component scope classifier.
- `bibliography_nextgen_freeze.py`: immutable candidate freeze before test-label access.
- `bibliography_nextgen_unseen_features.py`, `bibliography_nextgen_unseen_predict.py`, and `bibliography_nextgen_unseen_evaluate.py`: label-blind test features/predictions followed by one-shot evaluation.

## One-shot test result

The frozen one-shot evaluation is complete. The test contains 143 documents and 173,609 lines; 172,905 lines (99.5945%) have trusted bibliography-membership labels. These are dual-annotator consensus-silver labels, not human gold.

| Frozen candidate | Line P | Line R | Char P | Char R | FP lines | FN lines | Gate interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `incumbent_entry` | **0.9998** | 0.6409 | **0.9998** | 0.6916 | **3** | 7,384 | Safe precision reference |
| `position_hist_unscoped` | 0.9090 | **0.9415** | 0.9500 | **0.9645** | 1,938 | **1,203** | Recall ablation; unsafe precision |
| `position_hist_component_scope` | 0.9680 | 0.9171 | 0.9763 | 0.9350 | 623 | 1,704 | Large recall gain, but below the frozen 0.98 precision gate |

The component scope layer removes 147 of 479 unscoped predicted components and reduces false-positive lines from 1,938 to 623 while retaining most of the recall gain. It generalizes as a useful scope mechanism, but not strongly enough for precision-first deployment. Its weakest source is Kallipos (line precision 0.9578 and character precision 0.9292); common remaining errors include examples that demonstrate citation formatting, footnote/citation runs, URL/list appendices, tables, and other bibliography-shaped material outside the annotated bibliography section.

The deployment conclusion is therefore conservative:

- keep `incumbent_entry` as the production-safe model;
- retain `position_hist_component_scope` as a research candidate, not a signed-off replacement;
- do not tune thresholds or choose another candidate using this test set.

Immutable test artifacts:

- Label-blind features: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_20260720/unseen_features_099c6b1_r1`
- Label-blind predictions: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_20260720/unseen_predictions_099c6b1_r1`
- One-shot report: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_nextgen_20260720/one_shot_test_099c6b1_r2.json`

Evaluation job `2801038` stopped before metrics because the sealed line-key order differed from the document-derived feature order. Commit `60db955` changed only the identity join to use stable `(document_id, line_id, abs_idx)` keys. Job `2801045` then evaluated the exact same frozen candidates and prediction arrays successfully. No model, threshold, decoder, or prediction changed between the failed and successful evaluation attempts.
