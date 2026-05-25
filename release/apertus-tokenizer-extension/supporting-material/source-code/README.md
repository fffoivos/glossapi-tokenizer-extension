# Source Code

The runnable implementation lives in GitHub:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

This Hugging Face release keeps artifact payloads, manifests, compact evidence,
and hydration instructions. It does not mirror the full scripts tree.

Important eval policy:

```text
../provenance/evals/LOSS_MEASUREMENT_POLICY.md
```

That file explains why raw Megatron `lm loss` is health-only across different
tokenizers, and why heldout BPC/BPB plus downstream evals are the cross-arm loss
evidence.

See:

```text
manifest.json
```
