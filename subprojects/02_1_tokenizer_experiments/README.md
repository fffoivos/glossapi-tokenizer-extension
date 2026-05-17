# 02.1 Tokenizer Experiments

## Active Scope

**Active arm: C3** (`C3_wave2_broad_glossapi_plus_hplt_50_50`). See
[../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md).

The four-arm exploration (`F1`, `F2`, `C1`, `C2`) is closed. The
cutoff-decision sweep on C3 is now the center of the tokenizer-extension
story:

**Canonical cutoff: 17,408 added C3 units, curated + backfilled.**

- ship vocab: `148,480` = `131,072` Apertus base ids + `17,408` added ids
- alignment: `148,480 = 128 * 1160 = 256 * 580`
- contract: Apertus base ids `0..131,071` preserved verbatim
- curation: 69 noise tokens structurally skipped and backfilled with the
  next valid C3 merges
- decision doc:
  [`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)

The cutoff can still be revisited if downstream adaptation teaches us
something new, but the current handoff target is the 17,408
curated+backfilled tokenizer.

## Sub-subprojects — pipeline ordering

The subproject is split into pipeline-oriented sub-subprojects (same
convention used by `02_2_tokenizer_implementation`):

```
[02_1_1 tokenizer training]
        → produces a full tokenizer.json at the target max vocab
[02_1_2 cutoff variant builder]
        → derives Apertus-compatible merged variants at each cutoff
[02_1_3 fertility evaluation]
        → measures intrinsic + fertility metrics per (variant, slice)
[02_1_4 cutoff analysis]
        → combines comparable-language sizes + fertility + glossary +
          char-lang masks into a cutoff recommendation
[02_1_5 added-token curation]
        → per-token keep/remove decision; emits implementation
          manifest for 02_2 (does not modify tokenizer files)
[02_1_6 representation policy analysis]
        → investigates Apertus's implicit language-budget policy
[02_1_7 intrinsic eval sweep]
        → measures the C3 cutoff grid and freezes the 17,408 decision
[02_1_polytonic_greek_extension]
        → parallel ancient/polytonic source-selection arm
```

- [`02_1_1_tokenizer_training/`](02_1_1_tokenizer_training/README.md) —
  continuous-BPE trainer + the original `CONTINUOUS_BPE_EXTENSION_PLAN.md`
  and `_TODO.md` (still apply to the cutoff grid, mergeback, evaluation
  shape; multi-arm-specific sections are archived in-place).
- [`02_1_2_cutoff_variant_builder/`](02_1_2_cutoff_variant_builder/README.md) —
  derives `<arm>_added_<N>/` variants per cutoff.
- [`02_1_3_fertility_evaluation/`](02_1_3_fertility_evaluation/README.md) —
  runs the intrinsic + fertility suite on clean held-out slices. Also
  hosts the held-out-cleaner scripts (`clean_holdouts.py`,
  `build_virgin_hplt_eval.py`) needed because the splitter has a known
  row-vs-doc bug.
- [`02_1_4_cutoff_analysis/`](02_1_4_cutoff_analysis/README.md) — the
  cutoff-recommendation stage. First-draft report at
  [`02_1_4_cutoff_analysis/REPORT.md`](02_1_4_cutoff_analysis/REPORT.md).
- [`02_1_5_added_token_curation/`](02_1_5_added_token_curation/README.md) —
  per-class keep/remove policy for added tokens (Latin-1 mojibake,
  font-substitution mojibake, cleaner extraction tags). Emits a
  machine-readable removal manifest at
  [`02_1_5_added_token_curation/manifests/removal_list.jsonl`](02_1_5_added_token_curation/manifests/removal_list.jsonl)
  for `02_2_tokenizer_implementation` to consume. Reasoning report at
  [`02_1_5_added_token_curation/CURATION_REPORT.md`](02_1_5_added_token_curation/CURATION_REPORT.md).
- [`02_1_6_representation_policy_analysis/`](02_1_6_representation_policy_analysis/README.md) —
  policy-archaeology investigation into Apertus's implicit
  per-language representation policy. Net outputs:
  the provenance finding (Apertus inherited Mistral's BPE table
  verbatim — [`11_tokenizer_provenance.md`](02_1_6_representation_policy_analysis/11_tokenizer_provenance.md)),
  speaker-count hypothesis invalidation, HPLT 3.0 / FLORES+ /
  classical-language reference evidence, and the discovery of the
  swiss-ai TokEval suite that seeded `02_1_7`. The Greek-budget
  question was ultimately settled empirically in `02_1_7`, not by
  policy reasoning here. Deprecated drafts (synthesis with known
  math errors + 6 unrun hypothesis stubs) archived in
  `02_1_6_.../\_deprecated_20260518/`.
- [`02_1_7_intrinsic_eval_sweep/`](02_1_7_intrinsic_eval_sweep/README.md) —
  TokEval intrinsic-evaluation suite (Meister 2025, swiss-ai fork)
  applied to a 1k-spaced grid of C3 cutoff variants (0 → 25,600
  added tokens). **Cutoff decided 2026-05-18 at 17,408 added units
  (curated + backfilled, vocab 148,480)** per
  [`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md).
  Full sweep report at
  [`02_1_7_intrinsic_eval_sweep/REPORT.md`](02_1_7_intrinsic_eval_sweep/REPORT.md).
- [`02_1_polytonic_greek_extension/`](02_1_polytonic_greek_extension/README.md) —
  parallel Ancient/Polytonic Greek arm. It selects eligible ancient /
  liturgical sources, filters mixed Wikisource/Scholarios rows by
  distinctive polytonic orthography, deduplicates the selected corpus,
  and then hands a frozen source mix to the tokenizer-training stage.

## Already Decided

- **arm = C3** — continuous BPE from Apertus on `GlossAPI + HPLT` at
  `50 / 50` by training-token mass, trained on the wave-2 broad cleaner
  output, total vocab `156672`
- **cutoff = 17,408 added units** — selected from the extended 0 → 25,600
  sweep because it is the first point where the next 1k added units buys
  <1% additional Greek-fertility improvement while preserving Apertus
  alignment
- **ship artifact = curated + backfilled variant** — the raw 17,408
  variant was useful analytically, but the canonical tokenizer skips the
  69 curated noise tokens and backfills to keep exactly 17,408 useful
  added units
- `modern_greek_eval` is the primary decision set
- preserve Apertus front-end behavior during analysis: same
  normalization, same regex split, same byte-level regime
- shipping path: publish the canonical tokenizer artifact to the
  Apertus-extension HF repo, then hand it to `02_2_tokenizer_implementation`
  and `03_apertus_extension_and_embedding_adaptation`

## Cutoff Decision Phase

The cutoff evaluation is complete. The historical four-arm comparison is
closed, and the old four-candidate grid (`10,240`, `15,360`, `20,480`,
`25,600`) was superseded by the 1k-spaced intrinsic-eval sweep in
`02_1_7`.

### Evidence Bundle

- in-domain Greek fertility from the `02_1_3` held-out harness
- TokEval lines/words metrics over Apertus-55 FLORES+
- MorphScore Greek UD morphology probe
- added-token utilization and knee analysis
- curated-vs-raw comparison for the final cutoff

See [`02_1_7_intrinsic_eval_sweep/REPORT.md`](02_1_7_intrinsic_eval_sweep/REPORT.md)
for the evidence and
[`02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)
for the pinned ship contract.

### Artifact Policy

Git tracks the process: scripts, reports, small decision manifests, and
the report plots. Bulky generated tokenizers, raw metric tables, parquet
outputs, and rerunnable vendor/eval caches stay out of Git. The canonical
tokenizer and its minimal evidence bundle belong on HF.

### Downstream Confirmation

The shipped variant goes through downstream confirmation in
`02_2_tokenizer_implementation` (compatibility checks) and then
`03_apertus_extension_and_embedding_adaptation` (embedding adaptation +
CPT).

## Research Rule

Any new experimental decision that is not already documented should be
made explicitly with the user rather than assumed silently.

## Parallel Arm: Polytonic Greek

The polytonic arm is not a continuation of the C3 cutoff sweep. It is a
parallel source-selection and tokenizer-training path motivated by the
finding that polytonic Greek should be treated as its own orthographic
lane. See
[`02_1_polytonic_greek_extension/`](02_1_polytonic_greek_extension/README.md).

## Archived: Four-Arm Comparison

The original phase-1/phase-2/phase-3/phase-4 framing compared four arms
(`F1`, `F2`, `C1`, `C2`) and was decided in favor of C3. The
archived raw artifacts live under:
- `runs/_archive/production_strict_v2/tokenizers/` — F1, F2, C1 (wave-3 strict)
- `../../tokenizer_analysis/hf_snapshots/apertus-tokenizer-extension/` —
  earlier C1/C2 156672 snapshots used as analyzed baselines

Do not run new evaluations against those arms unless explicitly
revisiting the convergence decision.
