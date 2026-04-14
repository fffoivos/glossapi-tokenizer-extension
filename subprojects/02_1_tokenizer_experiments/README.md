# 02.1 Tokenizer Experiments

## Scope

Research the tokenizer discovery experiments before implementation is frozen.

## Already Decided

- discovery tokenizers should use a working vocab fixed at `50k` for the first runs
- compare four arms, not just two corpus views:
  - fresh discovery `BPE` on `GlossAPI-only`
  - fresh discovery `BPE` on `GlossAPI + HPLT`
  - continuous `BPE` from Apertus on `GlossAPI-only`
  - continuous `BPE` from Apertus on `GlossAPI + HPLT`
- the mixed view uses `70/30` GlossAPI/HPLT by training-token mass
- the continuous-BPE arms should start from the Apertus tokenizer and merge table and be allowed to grow by roughly the same estimated extension budget we are already evaluating, i.e. up to about `25k` new units
- evaluate candidate extension sizes analytically around:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- use `modern_greek_eval` as the primary decision set
- prefer the simpler `GlossAPI-only` variant unless `GlossAPI + HPLT` gives a clearly better result

## Evaluation Phase

Evaluation is staged. We do not jump directly from tokenizer training to a shipping choice.

### Phase 1: Intrinsic Tokenizer Comparison

Run all four arms on the same held-out bundle and compare:
- `bytes_per_token`
- `tokens_per_byte`
- fertility on the held-out texts
- added-token utilization rate
- vocabulary utilization rate
- unreachable added tokens
- byte-fallback rate

Use the same evaluation slices for every arm:
- `GlossAPI` held-out
- `HPLT` held-out
- mixed `GlossAPI + HPLT` held-out
- `modern_greek_eval`

### Phase 2: Apertus-Compatible Mergeback

After the raw four-arm comparison:
- diff learned units against Apertus
- build Apertus-compatible merged variants
- evaluate the cutoff grid:
  - `10240`
  - `15360`
  - `20480`
  - `25600`

These are the variants that matter for shipping decisions, not the raw standalone discovery tokenizers.

### Phase 3: Fertility Tests

Fertility tests are part of the core decision bundle, not a side check.

Run fertility tests on the Apertus-compatible merged variants and compare:
- overall fertility
- fertility on `modern_greek_eval`
- fertility on `GlossAPI` held-out
- fertility on `HPLT` held-out

We want lower token counts for Greek text without creating obvious regressions in the broader mixed view.

### Phase 4: Downstream Confirmation

Only after the intrinsic and fertility screens:
- promote the best one or two arms
- run downstream training/evaluation
- compare throughput, memory profile, and task results under the same training budget

The evaluation phase therefore has three roles:
- choose between `GlossAPI-only` and `GlossAPI + HPLT`
- choose between fresh discovery and continuous `BPE`
- choose the cutoff for the Apertus-compatible merged extension

## Research Rule

This subproject is research-heavy. Any new experimental decision that is not already documented should be made explicitly with the user rather than assumed silently.
