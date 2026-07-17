# Gap-only connector pilot — 2026-07-17

This is the durable summary of the first gap-only connector experiment.  It
used train-OOF upstream probabilities and LLM-silver region labels, never
placed the two anchor lines in the model tensor, and did not open validation.
It is a research baseline and is not approved for corpus deletion.

## Immutable artifacts

- code commit: `80010cb52430eff11e44b399489455a53ccd689b`
- table job: `2782661`
- model job: `2782674`
- Clariden table: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_gap_connect_20260717/table_80010cb_r2`
- Clariden models: `/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_gap_connect_20260717/models_80010cb_r1`

## Dataset

- 1,857 gap examples / 35,261 interior lines;
- 1,590 connect and 267 break targets;
- break targets came from only 93 works;
- positive gap median: 3 lines; negative gap median: 40 lines;
- negatives by source: 60 Greek PhD, 159 Kallipos, 48 OpenArchives.

The large class-conditional gap-length difference is a shortcut.  These data
cannot answer whether ordered gap structure generalizes.

## Five-fold grouped OOF result

| arm | break PR-AUC | connect precision | connect recall | false connects |
|---|---:|---:|---:|---:|
| pooled histogram boosting | 0.932651 | 0.995995 | 0.469182 | 3/267 |
| ordered residual TCN | 0.848774 | 0.989407 | 0.293711 | 5/267 |
| shuffled-order TCN | 0.892957 | 0.990632 | 0.266038 | 4/267 |

The ordered-minus-shuffled break PR-AUC was `-0.044184`.  The pooled model was
the clear pilot winner, but its observed 1.12% false-connect rate fails the
subsequently chosen safety target.  OpenArchives was the weakest pooled slice
(`0.592983` break PR-AUC).  The next experiment therefore changes candidate
materialization and negative coverage before revisiting model architecture.
