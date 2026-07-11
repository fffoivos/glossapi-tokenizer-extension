# Frozen CPT hyperparameters — 2026-07-11

**Status:** signed off for the full-corpus probe. No additional 13.5 B sweep is
required. Raw Clariden run metadata and logs take precedence over earlier prose.

## Frozen recipe

| Setting | Value | Evidence |
|---|---:|---|
| tokenizer/init | TD layer 11, ModernGreek-148480 | completed tokenizer/TD bakeoff |
| replay mix | 79% new Greek / 20% foreign / 1% old Greek | replay sweep; `PRODUCTION_MIX_DECISION_20260612.md` |
| peak/final LR | `5.5e-5` / `5.5e-6` | four-arm LR sweep; `PRODUCTION_LR_DECISION_20260613.md` |
| LR warmup | **400 iterations, fixed** | the beta2 sweep held this constant; do not recouple it to beta2 |
| WSD cooldown | final 20%, `1-sqrt` | frozen prior; no new cooldown sweep is authorized |
| AdEMAMix beta1/beta2/beta3 | `0.9 / 0.999 / 0.999` | beta2 and beta3 sweeps below |
| AdEMAMix alpha | `4` | alpha sweep decision table |
| alpha/beta3 ramp | whole run | exact schedule used by the selected arms |
| weight decay / grad clip | `0.1 / 0.1` | unchanged common recipe |
| sequence / RoPE | 4096 / 500,000 with Llama-3 scaling | unchanged common recipe |
| global batch | 1,024 sequences (4,194,304 tokens) | unchanged common recipe |
| loss | Goldfish `k=h=50` | unchanged common recipe |

The selected beta2 value is valid only with the **fixed 400-iteration LR
warmup**. Applying the old `2/(1-beta2)` rule would yield 2,000 iterations and
would be a different, untested recipe.

## Sweep readout

- Replay: `R=0.25` tied `R=0.35` on adaptation/retention and beat `R=0.15`;
  split replay was then materialized as 79/20/1.
- Peak LR: `5.5e-5` was the loss-first knee. Higher arms gained little
  GreekMMLU while degrading foreign held-out loss.
- Alpha: `4` produced 59.48% final GreekMMLU, versus 56.63% at alpha 0 and
  57.82% at alpha 8, with a better adaptation/retention balance.
- Beta3: `0.999` produced 59.48% final GreekMMLU, versus 57.91% at `0.995`
  and 57.20% at `0.99`. Its foreign-loss difference was small, and it had the
  best old-Greek absolute final loss.
- Beta2: `0.999` produced 59.94% final GreekMMLU and the best new-Greek mean
  held-out loss (1.6941). Foreign mean loss was only 0.0021 above the `0.995`
  control, while all foreign deltas remained negative.

The complete beta3 and beta2 tables are under
`curriculum_sweeps_v2/results/`. Existing replay, LR, and alpha decision
artifacts remain authoritative for their metrics.

## Mechanical comparability audit

All 14 informative TD runs reached checkpoint 3218. For each sweep, raw
`run_metadata.json` objects were normalized for JSON numeric equivalence,
run-local fields (`output_dir`, `init_ckpt`, Slurm job, start time) were removed,
and the declared sweep field was removed. Every run within each sweep then had
the same SHA-256 fingerprint.

For beta2 specifically, the `0.99`, `0.995`, and `0.999` arms shared:

- Megatron commit `c92402e39ef3c8e69ea378a59e79059dc14541f4`;
- the same data prefixes, seed, tokenizer, geometry and 3,218-step horizon;
- LR `5.5e-5`, beta3 `0.999`, alpha `4`, and 79/20/1 replay;
- `lr_warmup_samples=409600`, exactly 400 global-batch iterations.

After removing beta2 and run-local paths, their normalized metadata fingerprint
is `72992288d2117774d5e8d7a68e170acf1a36ea9a09a4e2470d6c083bcf40f9ca`.
Thus the deterministic policy selects beta2 `0.999`; the `0.995` fallback is not
needed. One beta2 `0.999` segment allocation failed before useful training and
was cleanly retried; the resumed chain completed checkpoint 3218.

The machine-readable evidence, raw metadata hashes, and live verifier are:

- `curriculum_sweeps_v2/results/sweep_config_audit_20260711.json`
- `curriculum_sweeps_v2/analysis/audit_sweep_configs.py`

## Remaining evidence limitation

Clariden retains all run logs, checkpoints, metadata, and evaluation sidecars,
but the old `curriculum_v2/megatron` training binaries have been removed (zero
files; 424 KiB stage directory at audit time). Payload hashes therefore cannot
be recomputed. The raw as-run metadata still proves identical data paths,
tokenizer, seed, geometry, and non-swept settings, but reconstructing the old
dataset would be required for a byte-level rerun. This does not block the new
full-corpus probe, which will materialize and hash a new dataset.
