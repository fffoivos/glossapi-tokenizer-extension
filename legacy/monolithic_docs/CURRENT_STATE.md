# Current State

Date: `2026-04-09`

## Executive Status

The project is not at the tokenizer-extension stage yet.

We have:
- a valid Greek tokenizer-fertility baseline
- a working Greek data-preparation path for nanochat
- proven GCP execution plumbing
- a completed `add_tokens(...)` experiment that should now be treated only as a diagnostic baseline
- a working project config that now captures the locked tokenizer constraints and the intended corpus pipeline:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/config/apertus_greek_extension.yaml`

We do not yet have:
- a representative HPLT training sample
- a deduplicated source-aware Greek training mixture
- held-out fertility test sets designed for the new method
- a true Greek `BPE` training run
- a merge-rule-based Apertus tokenizer extension

## What Is Confirmed And Reusable

- Reusable fertility evaluator:
  - `/home/foivos/data/glossapi_work/nanochat_glossapi_en_vs_el/scripts/evaluate_greek_tokenizer_fertility.py`
- Reusable nanochat Greek data-preparation script:
  - `/home/foivos/data/glossapi_work/nanochat_glossapi_en_vs_el/scripts/prepare_glossapi_greek_experiment_data.py`
- Curated baseline benchmark artifacts:
  - `/home/foivos/data/glossapi_work/analysis/tokenizer_fertility_20260407_curated`
- Confirmed Apertus tokenizer facts:
  - model type `BPE`
  - base vocab size `131072`
  - regex split plus `ByteLevel` pretokenization
  - `tie_word_embeddings = false`
  - final shipped vocab must stay divisible by `128`
  - the first `1000` ids are a fixed front block
  - about `927` of those front-block ids are placeholder special-token slots spanning `73-999`
- Proven execution backend facts:
  - GCP CPU-only instances are suitable for tokenizer work
  - CSCS Clariden is accessible, but less attractive for this CPU-heavy workflow

## New Findings From The First Execution Pass

- The actual Apertus pretraining scripts do not use the internal Megatron `TikTokenizer` path.
  They use:
  - `--tokenizer-type HuggingFaceTokenizer`
  - `--tokenizer-model swiss-ai/Apertus-8B-2509`
  - `--tokenizer-model swiss-ai/Apertus-70B-2509`
- The Megatron Hugging Face tokenizer path reloads the tokenizer through:
  - `transformers.AutoTokenizer.from_pretrained(...)`
- The HF-to-Megatron conversion path also reloads the tokenizer through:
  - `AutoTokenizer.from_pretrained(...)`
  - and computes `true_vocab_size` with added tokens included
- The Swiss-AI HF saver copies tokenizer artifacts via:
  - `tokenizer.save_pretrained(save_dir)`
- Practical implication:
  - any extension we build must remain a valid Hugging Face tokenizer directory, not just an internal merge table patch

- HPLT Greek metadata is rich enough for source-aware sampling.
  The first saved probe is:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/artifacts/20260409_hplt_metadata_probe_10_1_1000`
- Confirmed fields available in sampled `ell_Grek` rows:
  - `u`
  - `c`
  - `ts`
  - `crawl_id`
  - `lang`
  - `prob`
  - `cluster_size`
  - `seg_langs`
  - `filter`
  - `doc_scores`
  - `web-register`
  - plus `id`, `text`, `html_lang`, and source-position fields
- HPLT's `web-register` schema is mixed rather than perfectly clean.
  We now have a full-word mapping export at:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/artifacts/20260409_hplt_web_register_mapping/hplt_web_register_mapping.md`
  - `/home/foivos/Projects/glossapi-tokenizer-extension/artifacts/20260409_hplt_web_register_mapping/hplt_web_register_mapping.json`
  The important distinction is:
  - the Turku docs expose a simplified two-level hierarchy
  - HPLT's schema also carries older CORE fine labels
  - so broad labels like `nb` and `dtp` coexist with older more specific labels like `pb`, `tb`, `dp`, and `dt`
- Important sampling warning:
  - the sorted HPLT shards can be extremely domain-clumped
  - in the first `1000` rows of `10_1.jsonl.zst`, `docplayer.gr` accounts for `931` rows
  - therefore we must not build the matched sample from contiguous row prefixes; sampling has to be randomized or reservoir-based across shards
- The first exact HPLT representative-sample pass is now running as a durable background unit:
  - `hplt-review-sample-20260409-fast.service`
  - output directory:
    - `/home/foivos/Projects/glossapi-tokenizer-extension/artifacts/20260409_hplt_review_sample_200`
  It streams all `9` Greek shards in parallel and will write the final `200`-document review set when complete.

## What Was Finished But Is Now A Baseline Only

The following work is complete, but it is not the intended final extension method:

- `add_tokens(...)` builder:
  - `/home/foivos/data/glossapi_work/nanochat_glossapi_en_vs_el/scripts/build_apertus_greek_tokenizer_extensions.py`
- GCP sweep outputs:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/artifacts/20260408T160000Z`
- Postmortem analysis for the `add_tokens(...)` artifact:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/scripts/analyze_added_tokenizer_variant.py`

Reason:
- the run appended frequent whole-word Greek strings as Hugging Face added tokens
- it did not train Greek `BPE` merges
- it did not extend Apertus through `model.vocab` and `model.merges`
- it would generalize poorly across Greek morphology and accent variation

These artifacts are still useful as:
- a fertility upper-bound sanity check for exact-word memorization
- evidence that Greek adaptation is worthwhile
- a negative control against the real merge-extension path

## Current Working Conclusion

The plan has reset from:
- "mine frequent Greek words and add them"

to:
- "train compatible Greek `BPE` tokenizers on curated Greek corpora, identify genuinely new Greek subword units and merges, then extend Apertus through merge rules rather than `add_tokens(...)`"

## Decided But Not Yet Executed

- the final extended Apertus vocab must remain divisible by `128`
- embeddings and `lm_head` must both be handled because `tie_word_embeddings = false`
- embedding initialization is still to be chosen between:
  - mean-of-subtokens
  - `FOCUS`
  - token distillation
- model adaptation should be two-phase:
  - frozen-base warmup focused on the new rows
  - then full continued pretraining
- we will compare two tokenizer-training views:
  - `GlossAPI-only`
  - `GlossAPI + HPLT`
- the HPLT mixing ratio is intentionally still open pending content inspection

## Immediate Next Planning Work

The next stage is corpus design, not tokenizer execution:

1. Build a representative HPLT sample matched roughly to nanochat scale.
2. Use HPLT metadata such as source URL and content type to stratify that sample.
3. Remove same-source overlap between nanochat and HPLT where possible, then deduplicate.
4. Define a manual review subset of about `200` documents from the training sample.
5. Define held-out fertility test sets that are not used for `BPE` training.
6. Train true Greek-compatible `BPE` tokenizers at a working vocab size around `40k-50k`, so candidate discovery is not prematurely constrained and multiple extension cutoffs remain possible.
7. Evaluate fertility at multiple cutoff points around `5k`, `10k`, `15k`, and `20k` new tokens, then pick the elbow and align the shipped size to a valid multiple of `128`.
