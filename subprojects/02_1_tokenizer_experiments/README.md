# 02.1 Tokenizer Experiments

## Scope

Research the tokenizer discovery experiments before implementation is frozen.

## Already Decided

- discovery tokenizers should use a working vocab fixed at `50k` for the first runs
- compare `GlossAPI-only` vs `GlossAPI + HPLT`
- the mixed view uses `70/30` GlossAPI/HPLT by training-token mass
- evaluate candidate extension sizes analytically around:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- use `modern_greek_eval` as the primary decision set
- prefer the simpler `GlossAPI-only` variant unless `GlossAPI + HPLT` gives a clearly better result

## Research Rule

This subproject is research-heavy. Any new experimental decision that is not already documented should be made explicitly with the user rather than assumed silently.
