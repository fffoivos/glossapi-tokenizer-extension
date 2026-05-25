# Global Decisions

These are the active high-level decisions already made.

## Goal

Extend `swiss-ai/Apertus-8B-2509` for Greek in a way that generalizes well.

This means:
- discovering reusable Greek subword units through true `BPE` training
- not using whole-word `add_tokens(...)` as the shipping method
- extending Apertus through `model.vocab` and `model.merges`

## Tokenizer Arm

- **converged: C3** (`C3_wave2_broad_glossapi_plus_hplt_50_50`) — see
  [C3_CONVERGENCE.md](C3_CONVERGENCE.md)
  - continuous BPE from Apertus
  - base `131072` + added `25600` = total `156672`
  - corpus mix: `glossapi + hplt` at `50 / 50` by training-token mass
  - trained on the wave-2 broad cleaner output
- `F1`, `F2`, `C1`, `C2` are retained as analyzed baselines only and do
  not drive shipping work
- the only remaining tokenizer-side decision is the cutoff on C3 from
  the frozen grid `{10240, 15360, 20480, 25600}` added units; the
  shipped cutoff must remain `128`-aligned (all four candidates already
  are)

## Hard Constraints

- match Apertus tokenization behavior as exactly as possible
- preserve the fixed first `1000` ids
- preserve special-token behavior
- preserve the regex split plus `ByteLevel` regime
- final vocab size must remain divisible by `128`
- `tie_word_embeddings = false`, so embeddings and `lm_head` both matter

## Dataset Integration Constraints

- HPLT must be integrated into `fffoivos/glossapi-greek-nanochat-pretraining-dataset`
- the integration target is the upstream broad corpus stage under `data/*.parquet`
- HPLT must be uploaded in the existing canonical `21`-column source-parquet schema
- the published source parquets remain undeduplicated; deduplication is applied downstream through refreshed `dedup_metadata`
- HPLT-specific provenance should live in `source_metadata_json`
- `title` and `author` should only be promoted to top-level if they are credibly available
- the downstream builder should stay lightweight after HF download
- local tokenizer progress does not need to wait for HF upload once the filtered local source-parquet slice exists

## Corpus Constraints

- the training corpus must be diverse
- the training corpus must be deduplicated before any proper tokenizer training
- that cleaning/dedup work is owned by the upstream dataset pipeline, not by a separate tokenizer-project builder stage
- HPLT should be sampled with metadata awareness, not raw prefix sampling
- HPLT should currently be treated with a provisional `>=8` quality-bin filter
- after that metadata filter, HPLT must also run through real `corpus.clean`-compatible quality scoring before it is accepted as the tokenizer/CPT slice
- rows with `greek_badness_score > 60` must be excluded from the HPLT tokenizer/CPT slice
- HPLT documents labeled `Machine translated or generated` must be excluded from the final training dataset
- same-source overlap between GlossAPI and HPLT should be reduced before final freeze
- `openarchives.gr` rows with `needs_ocr == true` must remain excluded from the CPT-ready dataset used for tokenizer work

## Dedup Repair Constraint

- the dedup implementation may be changed for efficiency, storage layout, resumability, and parallelism
- dedup functionality must remain the same
- exact and near dedup decisions must remain semantically equivalent after the repair
- the repaired path must pass golden equivalence, resume equivalence, and downstream contract tests before it becomes the live default

## Operational Constraint

Use the existing dataset-build scripts as the operational path. Do not invent a second independent release builder when the current work can be expressed through the existing release pipeline and overlays.

## Upload Constraint

- dataset publication must run on a separate cheap uploader instance, not on the tokenizer worker
- temporary exception:
  - if the cheap uploader instance is unreachable, a low-priority `source_only` upload may run from the active GCP worker
  - this is a fallback only, not the intended steady-state topology
- that uploader track must stay independent of the tokenizer critical path
- the uploader payload must include:
  - the complete filtered HPLT source parquet slice
  - the refreshed `dedup_metadata` bundle so downstream builder-time dedup works
- the upload path should use the official Hugging Face large-folder upload mechanism, not an ad hoc custom uploader
- `publish_hf_release.py` is the intended upload entrypoint and should use the official large-folder upload strategy
- the upload instance should be configured for Xet-backed uploads when available

## Experimental Structure

The four-arm comparison phase is over. C3 is the converged arm; see
[C3_CONVERGENCE.md](C3_CONVERGENCE.md). What remains in this section
applies to C3 only.

- the active arm is C3 — continuous BPE from Apertus on `GlossAPI + HPLT`
  at `50 / 50` by training-token mass, trained on the wave-2 broad
  cleaner output, base `131072` + added `25600` = total `156672`
- preserve the Apertus front-end behavior on C3: same normalization,
  same regex split, same byte-level regime
- diff C3 against Apertus `model.vocab` / `model.merges` and drop units
  that should not be merged back as new tokenizer entries
- the cutoff grid on C3's added units is frozen at:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- fertility tests must be run on Apertus-compatible merged variants of
  C3 at each cutoff, not on the raw 156672-token tokenizer
- the shared tokenizer evaluation bundle must include:
  - `bytes_per_token`
  - `tokens_per_byte`
  - fertility
  - added-token utilization rate
  - vocabulary utilization rate
  - unreachable added tokens
  - byte-fallback rate
- the shared tokenizer evaluation slices must include:
  - `GlossAPI` held-out
  - `HPLT` held-out
  - mixed `GlossAPI + HPLT` held-out
  - `modern_greek_eval` — primary decision set
- the evaluation phase on C3 is:
  - Apertus-compatible mergeback at each of the four cutoffs
  - intrinsic + fertility metrics on each merged variant
  - cutoff elbow identification → shipped cutoff
- the divisibility-by-`128` rule applies to the whole final tokenizer,
  not just the newly added units (all four candidates already satisfy
  this)
- tokenizer experiments read from the same CPT-ready dataset used for
  continued pretraining

## Model-Adaptation Measurement Constraint

For Apertus adaptation experiments that compare different tokenizer
vocabularies, raw Megatron `lm loss` is not a cross-arm selection metric. It is
per-target-token cross entropy, so it changes with both softmax size and
tokenizer compression. Use it for health checks and within-arm trends only.

Cross-tokenizer loss evidence must use heldout tokenizer-fair BPC/BPB and
downstream evals. When the training loop emits dense `bpb`, `bpt`,
`base_loss`, `new_loss`, and `n_new`, those fields are measurement-only and
must be computed on the same loss-mask positions as optimizer `lm loss`.
Heldout checkpoint BPC/BPB remains the selection anchor.

## Execution Structure

There are now two parallel tracks:

Execution boundary:
- `home` is coordination-only for this project
- do not run tokenizer filtering, export, or training workloads on `home`
- operational tokenizer work should run on GCP workers and be stopped when done

1. Tokenizer critical path
- freeze eval manifests
- lock the literal Apertus tokenizer-replication checklist
- assemble Apertus-compatible merged variants of C3 at the four cutoffs
- run intrinsic + fertility tests at each cutoff on the four eval slices
- pick the cutoff at the elbow
- implement the final merge-rule extension

Prior steps (HPLT filter, dedup, mix, four-arm training, four-arm
comparison) are settled; the corpus is frozen, the arms have been
trained and analyzed, and C3 has been chosen — see
[C3_CONVERGENCE.md](C3_CONVERGENCE.md).

2. Dataset operational sidetrack
- finish uploading the filtered HPLT slice into the upstream HF dataset
- rerun the full prepared-source dataset view locally with HPLT included, using the existing release scripts
- keep the upload path and the local tokenizer path decoupled
- refresh published `dedup_metadata` later as a separate step once the intended dataset state is settled

## Open Decisions

- **C3 cutoff** from the frozen grid `{10240, 15360, 20480, 25600}` —
  the only open tokenizer-side decision
- exact HPLT-to-canonical-schema field mapping inside `source_metadata_json`
- exact literal tokenizer replication checklist beyond the already confirmed settings
- new-row initialization method
- multilingual replay ratio during full continued pretraining
