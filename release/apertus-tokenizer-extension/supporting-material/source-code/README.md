# Source Code

The runnable implementation lives in GitHub:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

This Hugging Face release keeps artifact payloads, manifests, compact evidence,
and hydration instructions. It does not mirror the full scripts tree.

## Where to start reading

Canonical entry point on the GitHub side:

```text
subprojects/03_apertus_extension_and_embedding_adaptation/CPT_MASTER_20260526.md
```

That doc is the master synthesis: mission, plans (v0.12 + v0.7), Apertus
fidelity, bakeoff results, discrepancy log, operational reference, and what's
needed for production. A copy is also mirrored to this release at
`../provenance/decisions/CPT_MASTER_20260526.md`.

## Important policies

Loss measurement:

```text
../provenance/evals/LOSS_MEASUREMENT_POLICY.md
```

Raw Megatron `lm loss` is health-only across different tokenizers. Cross-arm
loss conclusions use heldout BPB plus downstream evals. Older files may call
BPB `BPC`; that is a legacy bits-per-byte label.

Greek aggregate rule: explicit MT diagnostics (`arc_challenge_mt_el`,
`global_piqa_completions_ell_grek`) are excluded from the Greek aggregate per
v0.12 §10 Q6. Per-task scores remain visible.

## Manifests

Top-level: `../../manifest.json`. Per-checkpoint: each
`experiment-checkpoints/<Arm>-<scale>/manifest.json` carries source Clariden
paths and human-readable metadata.
