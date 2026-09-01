# 09.1 - Downstream-task instability

This sub-subproject adapts the analysis in Nishida, Isonuma, and Oda,
*Instability in Downstream Task Performance During LLM Pretraining*
([arXiv:2510.04848v1](https://arxiv.org/abs/2510.04848)), to the saved Apertus
checkpoint evaluations.

The first implementation stage covers only expressions (1) and (2). It does
not yet implement checkpoint averaging or checkpoint ensembles.

## Expression (1): mean total variation

For an ordered sequence of `m` checkpoints and a reference-based score
`L(theta_t)` for one example,

```text
             1       m-1
MTV(Theta) = --- *   sum |L(theta_(t+1)) - L(theta_t)|.
            m-1      t=1
```

For deterministic multiple-choice evaluation, this subproject maps
`L(theta_t)` to the stored `correct` value (`0` or `1`) for each example. It
computes MTV separately for every example, then reports the unweighted mean
over examples. This preserves the paper's example-level interpretation. The
adapter also reports MTV of the aggregate accuracy trajectory as a separate
diagnostic; it must not be substituted for mean example-level MTV because
simultaneous gains and losses can cancel in aggregate accuracy.

## Expression (2): instability score

For the model outputs `x_(theta_t)` for one example,

```text
            1       m-1
IS(Theta) = --- *   sum [1 - sim(x_(theta_t), x_(theta_(t+1)))].
           m-1      t=1
```

The current prediction artifacts are forced-choice evaluations. Therefore the
model output is the stored `pred_index`, and `sim` is exact match. Under this
mapping, IS is the fraction of adjacent checkpoint transitions at which the
selected answer changes. The metric core accepts a custom similarity function
for later generation tasks such as character-F1.

## Files

- `checkpoint_instability.py` - validated implementations of expressions (1)
  and (2), plus example-level aggregation.
- `analyze_predictions.py` - adapter for the repository's
  `predictions.jsonl` artifacts.
- `test_checkpoint_instability.py` - synthetic formula, validation, alignment,
  and adapter regression tests.

## Run

The order of repeated `--checkpoint` arguments is the scientific checkpoint
order. All files must contain the same example IDs.

```bash
python3 analyze_predictions.py \
  --checkpoint iter_0000000=/path/to/iter0/predictions.jsonl \
  --checkpoint iter_0009536=/path/to/iter9536/predictions.jsonl \
  --checkpoint iter_0018284=/path/to/iter18284/predictions.jsonl \
  --output /path/to/checkpoint_instability.json
```

Use `--ids /path/to/clean_ids.txt` to restrict every checkpoint to the same
decontaminated example set. The adapter fails closed on duplicate IDs,
checkpoint-label duplication, missing examples, non-finite scores, invalid
similarities, or checkpoint-count drift below two.

The paper reports final-stage results on the last 20% of its checkpoints. That
window selection is intentionally not hidden inside these two formulas. Supply
only the desired ordered checkpoint window to the adapter, and record that
selection in the calling analysis.

## Test

```bash
python3 -m unittest -v test_checkpoint_instability.py
```
