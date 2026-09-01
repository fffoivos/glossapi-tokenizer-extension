# 02.1.2 — Cutoff Variant Builder

> **In one line:** a single script that slices a fully-trained continuous-BPE arm into Apertus-compatible tokenizers at any cutoff — the mechanism that made the whole cutoff question cheap enough to answer empirically.
> **Period:** used for the C3 sweep from 2026-05-11; committed to this directory 2026-05-18 (`7deea009`). **Status:** completed; still the reusable stage-2 tool.
> **Came from / led to:** [`../02_1_1_tokenizer_training/`](../02_1_1_tokenizer_training/README.md) → this → [`../02_1_3_fertility_evaluation/`](../02_1_3_fertility_evaluation/README.md)

## Why this existed

Continuous BPE preserves Apertus ids `0..131,071` and appends new ids in merge order. That makes *every prefix* of the added block a valid Apertus-compatible tokenizer — so a cutoff decision does not need a retrain per candidate, only a truncation. This directory is the realisation of that observation: given the full arm and a list of cutoffs `N`, emit one loadable HF tokenizer directory per `N` with total vocab `131,072 + N`.

## History

| Date | What happened | Result | Evidence |
|---|---|---|---|
| 2026-04-29 | The mergeback idea was written into the plan as a phase, before any builder existed | Cutoff grid frozen at four points | [`../02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md`](../02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md) §1.4 |
| 2026-05-11 | Used to build the C3 sweep at **25 cutoffs** (`seq 1024 1024 25600`), superseding the four-point grid | Whole 1k-resolution sweep became affordable — seconds per cutoff | [`../../../docs/C3_CUTOFF_REPORT.md`](../../../docs/C3_CUTOFF_REPORT.md) |
| 2026-05-17/18 | The same builder fed the extended TokEval sweep in `02_1_7` (the parent script was re-implemented inline there as `01_build_variants_inline.py`) | 33 tokenizers evaluated | [`../02_1_7_intrinsic_eval_sweep/REPORT.md`](../02_1_7_intrinsic_eval_sweep/REPORT.md) |
| 2026-05-18 | Committed here with its README in the pipeline reorg | — | `7deea009` |

## Outcome

- `scripts/build_cutoff_variants.py` wraps `build_continuous_cutoff(...)` from the shared `tokenizer_analysis/run_wave4_fertility_eval.py` and generalises its hardcoded `c1_added_*` output prefix into an `--arm-prefix` argument.
- Each output directory `<arm_prefix>_added_<N>/` carries `tokenizer.json` plus `tokenizer_config.json` and `special_tokens_map.json` copied **byte-identical from the full arm**. That copy is what transfers the Apertus front-end contract: first 1,000 ids preserved, special tokens preserved, regex split + ByteLevel untouched, `normalizer: null`. The contract is verified once on the full arm in `02_1_1`; variants inherit it rather than re-proving it.
- Cutoffs must be 128-aligned, preferably 256-aligned — the same rule that later made `148,480 = 256 × 580` and `148,992 = 256 × 582` acceptable ship sizes.
- Wave-3 C1 variants from the earlier strict run remain on the gcloud instance as historical evidence of the builder working on a different arm; they are not in git.

## Where things are

| What | Where |
|---|---|
| The builder | `scripts/build_cutoff_variants.py` |
| C3 invocation (25 cutoffs) | example command in this directory's history; outputs went to `runs/c3_cutoff_eval_20260511/cutoff_tokenizers/` on the gcloud worker |
| Inline re-implementation used by the final sweep | [`../02_1_7_intrinsic_eval_sweep/scripts/01_build_variants_inline.py`](../02_1_7_intrinsic_eval_sweep/scripts/01_build_variants_inline.py) |

## Working documents

None. This directory is one script and this README.
