# Gap-connection dataset and model study — 2026-07-17

This train-only study asks whether a model should connect two bibliography-entry
seed components across the lines between them. The two endpoint seeds are never
placed in the model tensor. Upstream entry, heading, and continuation/filler
probabilities are grouped out-of-fold predictions. Validation remained sealed,
and no result here authorizes corpus deletion.

## Immutable executions

- multi-regime table: job `2784098`, commit `a08b98b854645b2b8d4fd62c10804f2552eb0e9b`;
- pooled regime/size screen: job `2784158`, commit
  `d7381afe177437e9a1b4e89c9527026b1361cf81`;
- pooled end-to-end decoder: job `2784447`, commit
  `a0a8286c9d68a14fbd4b86d9083e317719c8e355`;
- oracle reachability audit: job `2784457`, commit
  `9f91b845fd5bfabb0fb214ec804dafaa5421a5b6`;
- ordered/shuffled sequence controls: job `2784446`, commit
  `d7381afe177437e9a1b4e89c9527026b1361cf81`, completed in `02:17:01`.

The full artifacts are under this Clariden root:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_gap_connect_v2_20260717
```

## Candidate data

The materialized table contains 42,047 examples and 475,465 interior model
lines with 190 features per line. Its fixed genuine deployment evaluation set
contains 14,121 connect gaps and 777 break gaps. The break examples span 184
works, versus 267 breaks from 93 works in the first pilot.

The cumulative preparation regimes were:

1. real gaps at the deployment entry threshold `0.25`;
2. natural fragmentation at thresholds `0.10`, `0.15`, `0.40`, `0.60`, and
   `0.80`;
3. down-weighted typed-heading miss simulations;
4. one positive-length-matched hard non-bibliography span per work; and
5. one matched easy non-bibliography span per work.

Negative-boundary learning curves used 250, 500, 1,000, 2,000, and all
available groups where possible. Positives were capped at two per negative and
four per gold bibliography block. Every selected subset contains both classes
in every fold. Synthetic examples are training-only and at most half of fit
weight; all metrics use the same 14,898 genuine candidates.

## Pooled screen

Selection retained configurations within one work-bootstrap standard error of
the best break PR-AUC, preferred arms no worse than the real/all histogram
baseline on each source's false-connect count, then minimized false connects,
training size, and false splits in that order.

The selected pooled arm is shallow histogram boosting trained with the
threshold-ladder regime at the 1,000-negative-boundary rung:

- break PR-AUC `0.835896`;
- connect precision `0.999529`;
- connect recall `0.300262`;
- 2 false connects out of 777 (`0.2574%`).

The paired real-only/500 control had break PR-AUC `0.837135`, 3 false
connects, and `0.261596` recall. Header-ablation/1,000 produced the highest raw
break PR-AUC (`0.845719`) but 6 false connects. Hard and easy random
non-bibliography spans reduced ranking quality; they should not be added to the
chosen training regime. Using every threshold-fragmented example also hurt
safety (20 false connects).

## End-to-end effect and ceiling

On complete train-OOF masks, the selected connector changed:

| policy | line P | line R | token P | token R |
|---|---:|---:|---:|---:|
| seed-only | 0.993557 | 0.475069 | 0.994395 | 0.524334 |
| selected connector | 0.994427 | 0.558757 | 0.995049 | 0.608251 |
| oracle over available gap candidates | 0.996314 | 0.832724 | 0.996558 | 0.855608 |

The learned connector therefore adds about 8.4 percentage points of line and
token recall without weakening precision, but fails the 0.95 recall gates. It
also inherits `0.0526` spurious blocks per zero-bibliography document from the
seed policy. There were zero trusted hard-stop crossings.

The oracle audit partitions its 23,292 missed gold lines as:

- 6,025 lines in blocks with no eligible seed;
- 3,096 lines in blocks with only one eligible seed;
- 5,496 leading-edge lines before the first seed;
- 6,007 trailing-edge lines after the last seed; and
- 2,668 unreachable internal lines between seeds.

Thus about 49% of misses are outer edges, about 39% are weak/unseeded blocks,
and only about 11% are internal gaps. More internal-gap augmentation cannot
meet the end-to-end recall gate by itself. The next architecture needs a
separate outward edge-expander and a conservative proposal path for weakly
seeded blocks; the gap connector can remain an interior connection expert.

## Ordered sequence control

The residual TCN did not earn representation credit. Its execution status
records that both selected controls ran successfully; the per-configuration
`representation_evidence_passed` fields are both false.

| training data | arm | break PR-AUC | false connects | connect recall |
|---|---|---:|---:|---:|
| threshold ladder / 1,000 | pooled histogram | 0.835896 | 2 | 0.300262 |
| threshold ladder / 1,000 | ordered TCN | 0.820921 | 7 | 0.097939 |
| threshold ladder / 1,000 | shuffled TCN | 0.818487 | 5 | 0.110049 |
| real gaps / 500 | pooled histogram | 0.837135 | 3 | 0.261596 |
| real gaps / 500 | ordered TCN | 0.756901 | 6 | 0.041144 |
| real gaps / 500 | shuffled TCN | 0.750419 | 2 | 0.068692 |

For the selected threshold-ladder configuration, ordered minus shuffled break
PR-AUC was `0.002368` with work-bootstrap 95% interval
`[-0.026807, 0.031423]`; ordered minus the same-data pooled arm was
`-0.014442 [-0.049310, 0.017588]`. For the real-gap control, the corresponding
intervals were `0.006487 [-0.022924, 0.041070]` and
`-0.078205 [-0.118044, -0.044898]`. Order therefore has no supported benefit,
and no sequence end-to-end decoder was run.

## Decision

Retain the pooled histogram model trained on the 1,000-boundary threshold
ladder as the current *interior gap expert*. Reject random-span augmentation
and the ordered TCN for this role. This is not a corpus-removal or deployment
approval: the complete decoder still fails recall and zero-document spurious
block gates. The next model-development stage should target outward block
edges and zero/one-seed block proposals, while preserving the pooled connector
as one input rather than expanding the internal-gap dataset further.

## Training-dataset presentation

Clariden job `2789232` restored source text for 180 examples selected from the
exact 3,000-row winning training subset: 30 `CONNECT` and 30 `BREAK` examples
per source, spread across work folds and gap-length buckets. The presentation
marks the two seed endpoints as excluded and highlights only the strictly
interior span that supplied the model input. It also shows the underlying
silver line labels, which is important because a `BREAK` span may contain some
silver `BIB` lines while still being an invalid connection as a whole.

The source packet and receipt are `training-presentation.packet.json` and
`training-presentation.receipt.json` in this directory. The rendered local
reader is:

```text
http://127.0.0.1:8775/
/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-gap-training-dataset-20260718/index.html
```
