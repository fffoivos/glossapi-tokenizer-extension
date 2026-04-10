# Apertus Greek Tokenizer Extension Plan

Date: `2026-04-09`

## Goal

Extend `swiss-ai/Apertus-8B-2509` for Greek by adding genuinely useful Greek `BPE` units and merge rules, while preserving the existing tokenizer and model as much as possible.

The target method is:
- train compatible Greek `BPE` tokenizers on curated Greek corpora
- identify new Greek subword units absent from base Apertus
- extend Apertus through `model.vocab` and `model.merges`
- adapt the model mainly for the new token rows

The target method is not:
- `add_tokens(...)`
- exact whole-word memorization
- replacing the tokenizer outright as a first step

## Reset From The Previous Attempt

The completed `2026-04-08` sweep is now a baseline only.

What it did:
- mined frequent Greek words
- appended them as Hugging Face added tokens
- measured fertility improvement

Why it is not the real method:
- added tokens match ahead of the base `BPE` model
- the method does not learn compositional Greek subwords
- it generalizes poorly for a morphologically rich language
- it does not extend Apertus through merge rules

Keep that run as:
- evidence that Greek adaptation is worth doing
- a comparison point for the real method

## Confirmed Starting Point

- Apertus tokenizer type: `BPE`
- Apertus base vocab size: `131072`
- Apertus uses regex split plus `ByteLevel` pretokenization
- final shipped vocab must remain divisible by `128`
- Krikri remains the current Greek fertility reference on the curated benchmark

## Locked Compatibility Target

The extension work must preserve the actual published Apertus tokenizer mechanics.

Confirmed directly from the public Apertus files and local tokenizer inspection:

- `vocab_size = 131072`
- `bos_token_id = 1`
- `eos_token_id = 2`
- `pad_token_id = 3`
- `add_bos_token = true`
- `add_eos_token = false`
- `add_prefix_space = false`
- core special tokens `<unk>`, `<s>`, `</s>`, `<pad>` all have:
  - `lstrip = false`
  - `rstrip = false`
  - `normalized = false`
  - `single_word = false`
- `normalizer = null`
- pre-tokenizer is:
  - `Split` by the Tekken-style regex
  - then `ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false)`
- decoder is:
  - `ByteLevel(add_prefix_space=true, trim_offsets=true, use_regex=true)`
- post-processor inserts `<s>` for single and pair sequences
- `model.ignore_merges = true`
- `model.dropout = null`
- `model.unk_token = null`
- `model.continuing_subword_prefix = null`
- `model.end_of_word_suffix = null`
- `model.fuse_unk = false`
- `model.byte_fallback = false`
- `tie_word_embeddings = false` on the model side, so output weights must be handled explicitly if we resize vocab

Also confirmed from the published `tokenizer.json`:

- the first `1000` ids are reserved in the Tekken-style front block
- among those, `996` are marked special
- ids `73-999` are mostly placeholder special-token slots, which leaves about `927` front-block slots that should be treated as fixed reserved space rather than extension room
- the only non-special entries in that front block are:
  - `\\begin{`
  - `\\end{`
  - `\\text{`
  - `\\boxed{`
- the published `Apertus-8B-2509` and `Apertus-70B-2509` `tokenizer.json` files are byte-identical
  - shared `sha256`: `bb201fb226cde11f66c3cf51c5344fb37b1611f00c21e75c324546d854eff2e1`

Implication for the extension builder:

- do not change the first `1000` ids
- do not change special-token behavior
- do not introduce a normalizer or prefix-space behavior that diverges from Apertus
- keep compatibility with the exact regex split plus byte-level regime

## Confirmed Consumer Stack

The actual Swiss-AI training and conversion stack confirms that the extension must behave as a Hugging Face tokenizer artifact, not merely as an internal Megatron tokenizer patch.

Confirmed from the public Swiss-AI code:

- Apertus pretraining scripts use:
  - `--tokenizer-type HuggingFaceTokenizer`
  - `--tokenizer-model swiss-ai/Apertus-8B-2509`
  - `--tokenizer-model swiss-ai/Apertus-70B-2509`
- Megatron's `HuggingFaceTokenizer` path reloads the tokenizer with:
  - `transformers.AutoTokenizer.from_pretrained(...)`
- the HF-to-Megatron conversion loader also reloads the tokenizer through:
  - `AutoTokenizer.from_pretrained(margs.tokenizer_model)`
  - and computes `true_vocab_size` from the fast-tokenizer backend with added tokens included
- the Swiss-AI HF saver copies tokenizer artifacts with:
  - `tokenizer.save_pretrained(save_dir)`

Practical implication:

- the extension output must be a valid Hugging Face tokenizer directory
- at minimum we should expect:
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
- and those files must agree with the resized model config

## Corpus Strategy

We should not train directly on full HPLT first.

Instead, build a controlled Greek training corpus with two components:

1. `nanochat_train`
- primary downstream-aligned Greek corpus
- use the existing GlossAPI Greek nanochat train split

2. `hplt_matched_sample`
- a representative HPLT sample
- roughly matched to nanochat scale at first
- designed using HPLT metadata rather than raw random sampling

We will decide later whether the final mixture should be:
- `70/30`
- `50/50`
- or another ratio

That choice should be made after inspecting the sampled corpus and the tokenizer results.

We will explicitly compare two tokenizer-training variants:

1. `GlossAPI-only`
- trained only on `nanochat_train`

2. `GlossAPI + HPLT`
- trained on `nanochat_train` plus the frozen `hplt_matched_sample`

The HPLT mixing ratio remains intentionally undecided until the sample has been inspected.

## Decided But Not Yet Executed

- final shipped vocab must remain divisible by `128`
- discovery-tokenizer vocab size and shipped extension size are different decisions
- the first discovery tokenizers should use a working vocab size around `40k-50k`
- we will compare `GlossAPI-only` against `GlossAPI + HPLT`
- HPLT mixing ratio remains open pending content inspection
- model adaptation should be two-phase:
  - frozen-base warmup focused on the new rows
  - then full continued pretraining
- new-row initialization is still an experiment choice between:
  - mean-of-subtokens
  - `FOCUS`
  - token distillation

## HPLT Sampling Plan

Build the first HPLT sample with source awareness, not just volume matching.

### HPLT sampling objectives

- match nanochat approximately in total text volume for the first training run
- preserve diversity across source families
- avoid over-indexing on one source or one crawl artifact
- retain metadata needed for later audits

### Metadata to use

Use whatever HPLT metadata is available and stable, especially:
- source URL or host/domain
- content type or document type
- shard identity
- document length

### Sampling rules

1. Bucket HPLT by source family.
2. Bucket again by content type when available.
3. Sample within buckets proportionally, with caps so one domain cannot dominate.
4. Keep a manifest with:
   - document id
   - shard
   - URL or host
   - content type
   - character count
   - selected split
5. Do not sample from contiguous row prefixes inside the sorted shards.
   The first probe on `10_1.jsonl.zst` showed extreme domain clumping, with `docplayer.gr` covering `931/1000` consecutive rows.
   Use randomized or reservoir-style sampling across shard positions instead.

### Source-overlap controls

Before finalizing the HPLT training sample:
- detect obvious overlap with nanochat source domains or mirrored collections
- exclude same-source duplicates where feasible
- then run text-level or doc-level dedup on the combined training candidate pool

The order should be:
1. source-family overlap reduction
2. exact or near-exact dedup
3. final sample freeze

## Training Sample Review

Before tokenizer training, create a manual review slice of about `200` documents from the frozen training sample.

The review set should be stratified across:
- nanochat vs HPLT
- major source families
- short, medium, and long documents
- modern monotonic Greek vs historical or polytonic material

Manual review goals:
- verify the sample is really Greek
- verify the sample is not dominated by OCR garbage or boilerplate
- verify the domain mix looks plausible
- estimate how much polytonic or historical material remains

The review output should be a small CSV or JSONL with reviewer notes.

## Held-Out Test Sets For Fertility

Tokenizer training and fertility evaluation must use different documents.

Create at least three held-out evaluation slices:

1. `nanochat_eval`
- from nanochat `val` and `test`
- never used for tokenizer training

2. `hplt_eval`
- sampled from HPLT after excluding all training-sample documents
- source-balanced where possible

3. `modern_greek_eval`
- a stricter evaluation subset emphasizing modern monotonic Greek
- used as the main decision set for extension quality

Optional extra slice:
- `polytonic_stress_eval`

This is useful for analysis, but it should not dominate the main decision rule.

## Actual Tokenizer Method

The intended tokenizer workflow is:

1. Train a compatible Greek `BPE` tokenizer on:
   - `nanochat_train`
   - and then on one or more frozen mixtures with `hplt_matched_sample`
   - using a discovery vocab size around `40k-50k`
2. Keep the tokenizer training setup as compatible with Apertus as possible:
   - same or closely matched pretokenization behavior
   - byte-level setup if Apertus compatibility requires it
3. Extract candidate new tokens and merges from the trained Greek tokenizer.
4. Diff them against base Apertus:
   - existing vocab entries
   - existing merge rules
5. Build extension variants from the genuinely new Greek units.
6. Patch Apertus through merge rules and vocab entries, not `added_tokens`.

## Candidate Sizes

The analytic cutoff sweep should probe the elbow around:

- `5k`
- `10k`
- `15k`
- `20k`

These round numbers are for analysis and model selection.

Because Apertus must stay `128`-aligned, the actual shipped or large-scale built variants should use the nearest valid counts:

- `5120`
- `10240`
- `15360`
- `20480`

Hard constraint:
- base vocab is `131072`
- final vocab must remain divisible by `128`
- therefore the shipped added count must also be divisible by `128`

Decision rule:
- prefer the smallest size that captures most of the Greek gain
- treat larger sizes as justified only if they improve materially on held-out modern Greek
- if the elbow lands between aligned values, use the nearest valid `128`-aligned size for the final build
- if the elbow looks ambiguous, run a small local probe around that region before freezing the shipped size

Comparison rule for `GlossAPI-only` vs `GlossAPI + HPLT`:
- prefer `GlossAPI-only` by default because it is simpler and closer to the downstream data distribution
- only prefer `GlossAPI + HPLT` if it produces a clearly better held-out result
- the default threshold should be:
  - at least `5%` relative improvement on `modern_greek_eval`
  - or a clear robustness gain across both `nanochat_eval` and `hplt_eval` without a non-Greek regression
- if the gain is smaller or ambiguous, keep the simpler `GlossAPI-only` variant

## Selection Logic For New Units

Candidate ranking should combine:
- frequency in the frozen training corpus
- utility across both nanochat and HPLT
- reduction in fragmentation under base Apertus
- subword usefulness, not just surface-form frequency
- stability across accent and inflection variants
- support across more than one source document or domain when possible

Reject or downweight:
- URLs
- numbers
- punctuation artifacts
- OCR garbage
- boilerplate fragments
- source-specific junk
- units that appear in only one source document unless they show unusually strong fragmentation benefit
- units that are effectively confined to one narrow domain and do not generalize across the held-out sets

## Integration With Apertus

The extension target is:
- preserve all existing Apertus token ids
- append only new ids
- preserve the old tokenizer behavior for already-covered units as much as possible

Implementation requirements:
- add new vocab items at the end
- append the corresponding merge rules in a deterministic order
- emit a manifest of:
  - new token string
  - new token id
  - source counts
  - rank or score
  - originating corpus mix
  - whether the token appeared in nanochat, HPLT, or both

## Model Adaptation

The expected model-side change is limited but real:

- resize token embeddings to the new vocab size
- resize `lm_head` if it is not tied
- initialize only the new rows explicitly
- continue training so the new rows become meaningful

The old rows should remain intact at initialization time.

Current adaptation plan:

1. initialize the new rows with one of:
   - mean-of-subtokens
   - `FOCUS`
   - token distillation
2. run a short warmup where the old base rows are frozen and the new rows learn to stabilize
3. run full continued pretraining with the whole model unfrozen

Open mixing decision for phase `2` continued pretraining:
- the Greek-focused corpus should not be the only data stream in the full-CPT phase
- the English or multilingual replay ratio is still open
- this ratio should be chosen explicitly to limit catastrophic forgetting while still giving the new Greek rows enough signal

## Evaluation Criteria

Primary metrics:
- tokens per `100` chars
- average tokens per Greek word
- single-token Greek word share

Secondary metrics:
- p90 tokens per Greek word
- fragmentation of common Greek suffix/stem patterns
- behavior on accent variants
- non-Greek regression smoke test

Acceptance target:
- clear gain over baseline Apertus on held-out modern Greek
- positive result on both `nanochat_eval` and `hplt_eval`
- no material regression on a small non-Greek smoke set

## Compute Plan

This is still a single-node CPU problem.

### GCP default

Use one large CPU-only instance per serious run.

Initial recommendation:
- `n2-custom-32-131072` for data prep and smaller tokenizer experiments
- `n2-custom-64-262144` for full tokenizer training and extension assembly

If the representative HPLT sample is kept close to nanochat size, one machine should be enough for the first real sweep.

### Parallelism

Use parallelism where it helps:
- parallel shard reads
- parallel sample extraction
- parallel dedup
- parallel tokenizer sweeps across `N`

Do not distribute one tokenizer-training job across many instances unless profiling proves it is necessary.

## Immediate Execution Order

1. Freeze the corrected project state in docs.
2. Build a source-aware HPLT sampling manifest.
3. Build combined training-candidate manifests with source-overlap controls.
4. Create the `200`-document manual review sample.
5. Create held-out fertility test sets.
6. Train the first true Greek-compatible `BPE` tokenizer at about `40k-50k` vocab size.
7. Diff it against Apertus and build merge-rule extension candidates.
8. Run the analytic cutoff sweep around `5k` / `10k` / `15k` / `20k` new tokens without forcing the decision logic itself onto `128`-aligned values.
9. Identify the elbow from the analytic sweep, then snap the shipped build to the nearest valid `128`-aligned size or probe locally around that elbow if needed.
10. Move to embedding initialization and two-phase adaptation.
