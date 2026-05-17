# TODO

## C3 cutoff handoff

- [x] choose the active C3 arm over the earlier four-arm exploration
- [x] build Apertus-compatible cutoff variants from the C3 full tokenizer
- [x] run intrinsic, fertility, MorphScore, and in-domain Greek evaluations
- [x] freeze the canonical cutoff at **17,408 added units**
- [x] curate the added-token noise list and build the
  curated+backfilled 17,408 tokenizer
- [x] document the final decision in
  [`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)
- [ ] publish the canonical tokenizer + minimal evidence bundle to
  `fffoivos/apertus-tokenizer-extension` on HF
- [ ] hand
  `variants/c3_added_17408_curated_padded/tokenizer.json` to
  `02_2_tokenizer_implementation`
- [ ] run downstream compatibility and embedding-adaptation confirmation
  in `02_2` / `03`

## Historical tokenizer-training plan

- follow the numbered execution plan in:
  - [`02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md`](02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_PLAN.md)
  - [`02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_TODO.md`](02_1_1_tokenizer_training/CONTINUOUS_BPE_EXTENSION_TODO.md)
  - (both moved into the `02_1_1_tokenizer_training/` sub-subproject in the 2026-05-12 reorg)

- [x] run the fixed intrinsic metric bundle:
  - `bytes_per_token`
  - `tokens_per_byte`
  - fertility
  - added-token utilization rate
  - vocabulary utilization rate
  - unreachable added tokens
  - byte-fallback rate
- [x] run that bundle on the common held-out evaluation slices:
  - `GlossAPI` held-out
  - `HPLT` held-out
  - mixed `GlossAPI + HPLT` held-out
  - `modern_greek_eval`
- [x] supersede the original four-candidate grid with the 1k-spaced
  `02_1_7` sweep. Original grid:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- [x] run fertility tests on merged Apertus-compatible variants, not just
  raw standalone tokenizer arms
- [x] define the continuous-`BPE` training procedure starting from the
  Apertus tokenizer and merge table
- [ ] review whether additional modern-Greek-only control slices are
  needed after downstream adaptation results arrive

## Parallel polytonic-Greek arm

- [ ] freeze the source-selection policy in
  `02_1_polytonic_greek_extension/`
- [x] rerun strict Wikisource/Scholarios filtering with plain tonos/oxia
  excluded from the polytonic signal
- [x] deduplicate the selected corpus with Greek diacritics preserved
- [ ] review kept/dropped decisions and representative choices before
  constructing tokenizer-training shards
- [ ] define held-out polytonic and modern-Greek control slices before
  tokenizer extension training
