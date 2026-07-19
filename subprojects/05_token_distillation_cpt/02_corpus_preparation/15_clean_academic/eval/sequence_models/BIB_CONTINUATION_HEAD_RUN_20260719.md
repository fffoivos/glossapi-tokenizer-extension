# Dedicated continuation-head experiment — 2026-07-19

## Outcome

The compact dedicated `CONTINUATION` head is a better line-ranking model than
the frozen shared connector cascade, but none of the tested integrations
improves the structured bibliography-block pipeline. The line head is retained
as research evidence; all block integrations are rejected. The frozen
`block_model_23b4034_r1` remains the current best block model.

No retrospective validation or consensus-silver evaluation labels were opened.
All selection and comparison used the existing grouped train-only role table.

## Line-head comparison

The target was `CONTINUATION` versus trusted non-`ENTRY` candidates. `ENTRY`
remained owned by the frozen P0D model and `UNKNOWN` stayed masked. The table
contained 2,688 eligible lines, including 535 continuations from 597 documents.

Clariden job `2798800` compared the frozen cascade with four newly fitted OOF
arms under the existing document-grouped folds.

| arm | features | pooled OOF PR-AUC | document-macro PR-AUC |
|---|---:|---:|---:|
| frozen connector cascade | 177-feature cascade | 0.782005 | 0.891311 |
| new all-feature continuation head | 177 | 0.809345 | 0.956822 |
| compact core | 87 | 0.810560 | 0.939597 |
| compact + directional joins | 95 | 0.829802 | 0.940261 |
| compact + joins + table interactions | 110 | **0.838079** | 0.939614 |

The selected compact/table arm beat the frozen cascade in four of five grouped
folds. Complete-source holdout PR-AUC was 0.847408 on Greek PhD, 0.890605 on
Kallipos, and 0.745454 on OpenArchives. These results support the feature
hypothesis: directional joined-line evidence and explicit table interactions
help recognize continuation fragments. They do not by themselves establish a
better block decoder.

## Paired block tests

The first paired test replaced the old continuation probability and used the
maximum of the old connector and new continuation scores. Table job `2798804`
and block job `2798904` completed successfully. Recall regressed, primarily on
Kallipos.

A second conservative sweep blended 25%, 50%, 75%, or 100% of the new
continuation score into the old continuation channel while preserving the old
connector probability. Table jobs were `2798981`–`2798984`; paired structured
model jobs were `2798986`–`2798989`.

| block arm | line precision | line recall | char precision | char recall | false-positive lines |
|---|---:|---:|---:|---:|---:|
| frozen current best | 0.999431 | **0.943087** | 0.999698 | **0.943466** | 2 |
| replace + connector max | 0.999428 | 0.938523 | 0.999695 | 0.934786 | 2 |
| 25% blend, connector preserved | 0.998011 | 0.942819 | 0.999579 | 0.943448 | 7 |
| 50% blend, connector preserved | 0.999428 | 0.938255 | 0.999695 | 0.934776 | 2 |
| 75% blend, connector preserved | **1.000000** | 0.939597 | **1.000000** | 0.939438 | 0 |
| 100% replacement, connector preserved | 0.999429 | 0.939329 | 0.999696 | 0.939241 | 2 |

The 25% blend is almost recall-neutral but buys nothing: it adds five false
positives and slightly lowers both recalls. Stronger blends lose 12–19 Kallipos
recall points relative to its source-specific scale (about 0.009–0.015 absolute
line recall), while OpenArchives is almost unchanged. None passes the existing
0.95 line/character recall gates.

## Interpretation and next evolutionary step

Higher continuation PR-AUC did not repair the block result because the current
failure is no longer chiefly continuation ranking. Block proposals still
depend on two P0D entry seeds, and their reachability ceiling is unchanged by
this experiment. The new head also has a different calibration and
source-specific geometry; treating it as the old connector probability removes
useful Kallipos behaviour.

Do not tune another continuation blend. The next candidate should attack
proposal reachability directly while leaving P0D and the frozen decoder as
controls:

1. an outward edge proposal for a strong one-seed/heading-supported region;
2. a conservative weak/unseeded proposal path with an explicit whole-component
   precision gate; or
3. an auxiliary continuation score supplied as an additional structured
   feature rather than replacing an existing probability channel, only if its
   expected value can be isolated from the proposal change.

## Artifacts

Clariden root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_continuation_head_20260719`

Durable reports and receipts are archived locally under:

`sequence_models/results/bibliography_role_pipeline/20260719/`

Relevant code commits:

- `397ff43` — compact continuation-head OOF ablation;
- `606557a` — direct continuation probability port into the block table; and
- `1d04de7` — conservative blend and connector-preservation sweep.
