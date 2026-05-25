# TokenDistil-Init

Human name: `TokenDistil-Init`.

Weights status: location only in this HF release.

Clariden checkpoint:

```text
/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release
```

Tokenizer: `ModernGreek-148k`.

Technical notes:

- Token Distillation target layer: `11`;
- layer selection was based on the layer-pilot comparison;
- format: Megatron `torch_dist`, TP=2;
- includes the R17/xIELU/QK-Norm preservation patch;
- roundtrip verification is recorded in
  `../../supporting/provenance/conversion-roundtrip/td_layer11_r17_roundtrip_verification.json`.
