# Polytonic tokenizer cutoff probe

This directory freezes the decision procedure for appending the first 512 or
1,024 polytonic BPE merges to the production ModernGreek-148480 tokenizer.
It deliberately stops before tokenizing the new cleaned corpus.

## Decision contract

- Preserve all IDs and merges below 148,480 exactly.
- Keep structural ByteLevel merge fragments. They are not standalone UTF-8,
  but the audit proves that their downstream compositions decode to valid
  polytonic Greek. They are merge-chain initialized and excluded only from
  independent token-distillation targets.
- Run the same fixed NFC FineWeb-2 ancient and modern Greek evaluation streams
  for the ModernGreek-148480 baseline, +512, and +1,024 arms.
- Reject any evaluation with tokenizer-specific document truncation.
- Reject a candidate if modern-Greek BPB regresses by more than 0.5% against
  the baseline.
- Select +1,024 only when it passes that guard and improves ancient-Greek BPB
  by at least 1% relative to +512. Otherwise select +512 if it passes.
- If +512 fails the modern guard, select nothing.

The model-side probe uses the public TokenDistil-3.5B ModernGreek-148480
checkpoint. New rows are first initialized through their exact BPE merge chain;
non-structural rows with sufficient real ancient-Greek coverage then receive
one bounded layer-11 token-distillation epoch. Existing vocabulary rows are
preserved exactly by both stages.

The first pass exposed a positive-only output-row calibration bug. The
production decision therefore includes a second, disjoint balanced
ancient/modern pass which freezes the model and updates only appended LM-head
rows. Both the failed uncalibrated result and corrected result are retained.

## Result

**+512 selected** on 2026-07-29. The frozen tokenizer is:

```text
../../03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/
sha256: bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b
```

See [`PRODUCTION_DECISION_20260729.md`](PRODUCTION_DECISION_20260729.md).

## Run

The code and tokenizer candidates are staged under a unique Clariden
`RUN_ROOT`, then:

```bash
RUN_ROOT=/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024 \
  bash submit_probe.sh
```

The final machine-readable decision is
`production_cutoff_candidates/model_probe/selection_calibrated.json`.
