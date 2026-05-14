# 02.1 Tokenizer Experiments

## Active Scope

**Active arm: C3** (`C3_wave2_broad_glossapi_plus_hplt_50_50`). See
[../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md).

The four-arm exploration (`F1`, `F2`, `C1`, `C2`) is closed. The
remaining research scope in this subproject is the cutoff-decision
sweep on C3's added units. Candidate cutoffs are frozen at
`{10240, 15360, 20480, 25600}` (added merges atop the Apertus base of
`131072`).

## Current Execution Plan

- [CONTINUOUS_BPE_EXTENSION_PLAN.md](./CONTINUOUS_BPE_EXTENSION_PLAN.md)
- [CONTINUOUS_BPE_EXTENSION_TODO.md](./CONTINUOUS_BPE_EXTENSION_TODO.md)

Both docs were originally written under the four-arm framing. The
header in each marks the multi-arm sections as archived; the active
sections (cutoff grid + evaluation bundle + mergeback) still drive the
C3 cutoff sweep.

## Already Decided

- **arm = C3** — continuous BPE from Apertus on `GlossAPI + HPLT` at
  `50 / 50` by training-token mass, trained on the wave-2 broad cleaner
  output, total vocab `156672`
- evaluate candidate extension sizes analytically at the cutoff grid:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- `modern_greek_eval` is the primary decision set
- preserve Apertus front-end behavior during analysis: same
  normalization, same regex split, same byte-level regime
- shipping path remains: build Apertus-compatible merged variants of
  C3, evaluate, pick the cutoff, then hand off to `02_2`

## Evaluation Phase

Evaluation is now a single sweep across C3's four cutoffs, not a
four-arm comparison.

### Phase 1: Apertus-Compatible Mergeback

Build merged Apertus-compatible variants of C3 at each cutoff:
- `10240`
- `15360`
- `20480`
- `25600`

These are the variants that matter for the shipping decision, not the
raw 156672-token tokenizer.

### Phase 2: Intrinsic Metric Bundle

Run on every merged variant:
- `bytes_per_token`
- `tokens_per_byte`
- fertility
- added-token utilization rate
- vocabulary utilization rate
- unreachable added tokens
- byte-fallback rate

Evaluation slices:
- `GlossAPI` held-out
- `HPLT` held-out
- mixed `GlossAPI + HPLT` held-out
- `modern_greek_eval` (primary)

### Phase 3: Cutoff Selection

- find the elbow on the intrinsic + fertility curves
- diff C3 against Apertus `model.vocab` / `model.merges` to confirm
  extension quality at the chosen cutoff
- freeze the shipped cutoff (already `128`-aligned for all four
  candidates)

### Phase 4: Downstream Confirmation

After the cutoff is picked, the shipped variant goes through the
downstream confirmation in `02_2_tokenizer_implementation` (compatibility
checks) and then `03_apertus_extension_and_embedding_adaptation`
(embedding adaptation + CPT).

## Research Rule

Any new experimental decision that is not already documented should be
made explicitly with the user rather than assumed silently.

## Archived: Four-Arm Comparison

The original phase-1/phase-2/phase-3/phase-4 framing compared four arms
(`F1`, `F2`, `C1`, `C2`) and was decided in favor of C3. The
archived raw artifacts live under:
- `runs/production_strict_v2/tokenizers/` — F1, F2, C1 (wave-3 strict)
- `runs/wave4_production_strict_v1_20260429/tokenizers/` — wave-4 strict
- `../../tokenizer_analysis/hf_snapshots/apertus-tokenizer-extension/` —
  earlier C1/C2 156672 snapshots used as analyzed baselines

Do not run new evaluations against those arms unless explicitly
revisiting the convergence decision.
