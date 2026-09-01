# publication — private HF release of the two TD initialization checkpoints

> **In one line:** the 8B and 1.5B 148,480-vocabulary Token-Distillation initializations were published as private Hugging Face repos so the matched experiment had a durable, hash-verified starting point — after the R1 review found that the *historical* as-consumed init existed nowhere but an HF payload.
> **Period:** 2026-08-21 (from the `agent/h2g-teardown-workaround-128` working tree; recovered and committed 2026-09-01 in `2aec4a66`). **Status:** complete; both repos published private.

## Why this existed

The first ultracode review's live `ssh clariden` pass found that the as-consumed Megatron TD initialization of the historical run had been reduced to a 12K directory skeleton on CSCS, and that its only full-payload survivor was the HF publish `fffoivos/apertus-tokenizer-extension @ fcd33ec`. Repeating that failure with the new experiment's own initializations was avoidable, so both were released with a full verification transaction rather than left on scratch.

## What was released

| Contract | Repo (private) | Parent | Geometry |
| --- | --- | --- | --- |
| [`TD_INIT_8B_148480_RELEASE.json`](TD_INIT_8B_148480_RELEASE.json) | `fffoivos/apertus-8b-token-distillation-init-148480` | `swiss-ai/Apertus-8B-2509@3162c996…` | vocab 148,480, hidden 4,096, 32 layers, 32 heads / 8 KV, RoPE θ 500,000, 4,096 positions |
| [`TD_INIT_1P5B_148480_RELEASE.json`](TD_INIT_1P5B_148480_RELEASE.json) | `fffoivos/apertus-1p5b-token-distillation-init-148480` | `swiss-ai/Apertus-v1.1-1.5B@dbe8919b…` | vocab 148,480, hidden 2,048, 16 layers, 32 heads / 8 KV, same RoPE |

Each release transaction verifies the checkpoint inventory, the tokenizer SHA-256, tensor shapes, embedding geometry and an exact HF→Megatron→HF tensor *and logits* round trip before publication, and records stage / freeze / upload / inspection receipts. The model cards state plainly that these are initializations, not completed CPT models, and must not be presented as such.

## The tokenizer-serialization question

[`TOKENIZER_SERIALIZATION_RELEASE_BINDING_20260821.md`](TOKENIZER_SERIALIZATION_RELEASE_BINDING_20260821.md) resolves a discrepancy that would otherwise look like a data-integrity failure: the emitted `hf_roundtrip/tokenizer.json` hashes to `37c110e7…`, while the older 1.5B training-geometry artifact records a *compact* JSON serialization hashing to `358ae3f2…` (the value quoted in the cross-scale handoff). The two documents were independently parsed and found exactly equal — no vocabulary, merge, normalizer or special-token value differs. The release contract binds the emitted artifact's exact bytes; the 1.5B round-trip receipt remains the evidence for the initialized tensors and logits.

## Workarounds kept with it

Three shell helpers record how the release was actually pushed from CSCS: `workaround_canonical_xfer_release_with_ephemeral_token.sh` (upload from the `xfer` partition with an ephemeral token), `workaround_vendor_python_exec.sh` (execute the vendored Python), and `workaround_archive_capstor_bootstrap_inodes.sh` (archive bootstrap inodes on capstor).
