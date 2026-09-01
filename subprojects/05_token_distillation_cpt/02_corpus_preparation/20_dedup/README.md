# 20 — Dedup validation

> **In one line:** two files that *prove* the already-deduplicated corpus artifact is the right one, rather than re-deriving it.
> **Period:** work predates 2026-06-11; both files entered git in the bulk checkpoint commit `a19c136f` (2026-06-11). **Status:** completed — superseded for the full-corpus build by the content-bound dedup in [`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md).
> **Came from / led to:** [`../15_clean_academic`](../15_clean_academic/README.md) → this → [`../30_decontaminate`](../30_decontaminate/README.md)

## Why this existed

The deduplicated corpus artifact — `selected_after_apertus_and_internal_dedup.parquet` — was
produced *outside* this subproject by `glossapi_corpus_cli mix-prepare-selected-input`. The
pipeline order recorded in [`../../ARCHIVE.md`](../../ARCHIVE.md) ("clean, dedup-validate,
decontaminate, anonymize, shard") therefore makes this stage a **validation** step, not a
derivation: *"Dedup work for this launch is validation/characterization of the existing
selected corpus artifact, not a fresh full dedup derivation."*

## History

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| before 2026-06-11 | `stage1_validate_deduped.py` written to prove the SELECTED artifact: drop-list integrity, key-space type soundness, zero Apertus-overlap keys remaining, a data-loss guard against expected per-source nanochat totals, and an atomic provenance manifest | Every check anchored to an EXPECTED value so a vacuous PASS is impossible | [`stage1_validate_deduped.py`](stage1_validate_deduped.py) docstring |
| before 2026-06-11 | The script was hardened after an adversarial `codex exec` review; `codex_review.sh` was generalized into a reusable pre-run reviewer (gpt-5.5, high effort, agent prompt and user config stripped) | Membership moved to vectorized Arrow `is_in`; manifest write made atomic; tri-state reporting for the overlap check when the key-space is unsound | script docstring; [`codex_review.sh`](codex_review.sh) |
| 2026-06-11 | Both files committed in the bulk `Checkpoint pending project updates` commit | — | `a19c136f` |

The referenced review output (`stage1_validate_deduped.codex_review.md`) is **not** in the
tree; only the script's summary of its findings survives.

## Outcome

- The dedup contract for the CPT launch is *validation of an existing artifact*, with a
  provenance manifest as the deliverable — recorded in [`../../ARCHIVE.md`](../../ARCHIVE.md).
- `codex_review.sh` became the reusable "adversarially review this script before it runs"
  instrument for corpus-prep scripts.
- The later full-corpus build did not reuse this path: [`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md)
  implements content-bound exact and near deduplication with its own receipts. The
  51,839,746-row / 431-shard deduplicated release those receipts bind is the input that
  [`../15_clean_academic/production`](../15_clean_academic/production/README.md) later cleaned.

## Where things are

| Path | What |
|---|---|
| [`stage1_validate_deduped.py`](stage1_validate_deduped.py) | The validator (pyarrow; run on a compute node). Expects `--selected`, `--drop-list`, key columns, `--output-manifest`. |
| [`codex_review.sh`](codex_review.sh) | Pre-run adversarial script review via `codex exec`; writes `<script>.codex_review.md`. |
