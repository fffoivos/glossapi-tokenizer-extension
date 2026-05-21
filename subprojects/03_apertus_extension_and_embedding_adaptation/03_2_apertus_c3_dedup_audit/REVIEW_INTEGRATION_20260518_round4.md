# Review integration — 2026-05-18 round 4

Round 4 catches **implementation drift** from the measurement contract.
Six findings, all integrated. One conceptual reframe accepted.

## Conceptual reframe (Finding 1 — RETRACTED + REFRAMED)

**Reviewer retraction**: the earlier "nanochat differs from C3 by ~36%"
framing was based on stale GlossAPI-only release thinking. C3 was
trained on a 1:1 GlossAPI-nanochat + HPLT mix; the current HF material
includes HPLT-side artifacts too.

**Reviewer's reframed point**: the audit should explicitly distinguish
two scopes:
- `c3_exact_mix_overlap` — exact sampled 1:1 C3 mix/train against Apertus.
  Requires the C3 mix manifest (on the TERMINATED `apertus-greek-tokenizer`
  instance).
- `hf_source_pool_overlap` — broader HF source pool against Apertus.
  Useful for future CPT curation; not the load-bearing CPT-replay
  number.

**Status**: **INTEGRATED** as scope-mode split.

**Changes**:
- `01_bytes_balanced_partition.py` now sets `partition.scope =
  "hf_source_pool"` explicitly and uses `corpus_id = "hf_source_pool"`
  (was `c3`) for all GlossAPI-nanochat + HPLT-clean60 source tags.
- `09_build_summary_report.py` REPORT.md preamble now reads: **"This
  audit measures HF source-pool overlap with Apertus pretraining, NOT
  the exact 1:1 sampled C3 training mix."** The c3_exact_mix scope is
  explicitly named as a separate future run.
- All output filenames and columns renamed: `c3_doc_key` →
  `hf_pool_doc_key`, `per_c3_doc_overlap.parquet` →
  `per_hf_pool_doc_overlap.parquet`, etc. Downstream readers cannot
  confuse the universes.

The c3_exact_mix scope is **not** produced by this run — flagged as
the proper follow-on audit when the C3 mix manifest is accessible.

## Findings 2-6 (CRITICAL/HIGH/MEDIUM — STAND)

### Finding 2 (CRITICAL) — Apertus source loaders too brittle

**Issues**:
- EuroParl `pattern="*el*"` was a substring match, not glob. Would match `aeleal-*` and other false positives.
- EuroBlocks `pattern=""` + no `language == 'Greek'` filter → would slurp non-Greek SFT rows.
- Worker only accepted `text`/`content`/`raw_content` columns → would miss EuroParl which has a `translation` dict.

**Status**: **INTEGRATED.**

**Changes**:
- `01_bytes_balanced_partition.py` now enumerates EuroParl via an **explicit list of 20 `el-*` and `*-el` bitexts** (`bg-el`, `cs-el`, …, `el-sv`). Each file-filter is a closure that checks `fn.startswith(b + "/")` for those exact bitexts. No more substring matching.
- `01_bytes_balanced_partition.py` uses **explicit file_filter closures** per source instead of a single `pattern` string; loaders are correct-by-construction.
- `hash_pass.py` gained a `extract_greek_text(row, source_id)` function that:
  - For `europarl_greek`: extracts `row['translation']['el']` from the bitext dict (handles the `{"el": ..., "X": ...}` shape).
  - For `euroblocks_greek`: filters rows to `language == "Greek"` and tries `text`/`content`/`answer`/`output` columns.
  - For FW2-HQ / Clean-Wikipedia / others: standard `text` / `content` / `raw_content` fields.
- Non-Greek rows are skipped (not hashed); skip counts are recorded in `hash_log.jsonl` as `rows_in - rows_kept`.

### Finding 3 (HIGH) — Worker doesn't actually drive text_dedup.py canonical functions

**Issues**:
- `hash_pass.py` used truncated blake3 (`hexdigest(length=16)`) while canonical `text_dedup.hash_bytes` returns full 64-char hex. **Would have produced incompatible hashes with the published bundle's `strict_exact_group_hash`** — silent data corruption.
- MinHash was reimplemented locally instead of calling canonical `text_dedup.minhash_signature`.
- Short docs (<20 tokens) were returning an all-zero signature; their LSH bands would all collide → all short docs falsely "near-dup" each other.

**Status**: **INTEGRATED.**

**Changes**:
- `hash_pass.py` replaced local hashing with `td.hash_bytes(strict_norm.encode("utf-8"))` for both strict + relaxed — full 64-char blake3 hex per the published-bundle spec.
- `hash_pass.py` replaced local MinHash impl with `td.minhash_signature(shingle_hashes, num_perm=NUM_PERM)` — the canonical 128-perm signature.
- Short-doc behaviour: `text_dedup.shingle_hashes_from_text` already short-circuits to `[]` for docs with <20 tokens. Worker now emits `minhash_sig = None` and `lsh_band_hashes = None` for those rows, so short docs cannot collide via LSH band lookup. They still contribute strict_exact + relaxed_exact hashes.

### Finding 4 (HIGH) — Held-out contamination optional + must not be claimed if skipped

**Issue**: The original `run_all` script invoked `08_holdout_contamination_check.py || true`, so the step could silently skip while the REPORT still claimed eval-contamination coverage.

**Status**: **INTEGRATED.**

**Changes**:
- `run_all_with_teardown_trap.sh` keeps step 08 as non-fatal (`|| true`) since the holdout doc-id list isn't always available.
- `09_build_summary_report.py` now **checks whether `holdout_contamination.parquet` exists** and renders the REPORT.md scope block accordingly:
  - If present: "Held-out contamination check: INCLUDED — see holdout_contamination.parquet (N rows)"
  - If missing: "Held-out contamination check: **SKIPPED** — no `holdout_doc_ids.parquet` was provided. This report does NOT verify C3 val/test integrity."
- The "What this audit does NOT cover" section at the end of REPORT.md lists held-out check as a known gap when applicable.

### Finding 5 (MEDIUM) — Teardown trap + HF_TOKEN plaintext

**Two safety issues**:
- (a) The READY doc listed 9 manual coordinator steps; if any failed or the user SIGINT'd, workers would keep burning until manual teardown.
- (b) `03_dispatch_bootstrap.sh` baked `HF_TOKEN` into worker scripts via `sed` substitution → secrets sitting in plaintext on workers.

**Status**: **INTEGRATED.**

**Changes (a) teardown trap**:
- New driver `scripts/coordinator/run_all_with_teardown_trap.sh` wraps steps 03 → 09 in `trap teardown EXIT INT TERM` where teardown calls `99_teardown.sh`. Works on success, error, SIGINT, SIGTERM, anything.
- Also accepts `--include-spinup` flag to make the cost-event step explicit (default is to assume spin-up already happened).

**Changes (b) secret handling**:
- `03_dispatch_bootstrap.sh` no longer `sed`-substitutes `HF_TOKEN` into worker scripts.
- Instead, it sets `hf-token-secret` as **instance metadata** via `gcloud compute instances add-metadata`. The worker `run_all_on_worker.sh` reads the token at boot time from `http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token-secret` (only reachable from inside the VM).
- Same pattern for `worker-idx`, `bucket`, `run-id` — metadata-driven, no plaintext in scripts.

### Finding 6 (MEDIUM) — Local concat may not fit memory

**Issue**: `05_concat_per_source.py` did `pl.concat([pl.read_parquet(p) for p in files])` — eagerly materialised all worker outputs in memory. With 8 workers × multi-GB outputs each, this could OOM on home.

**Status**: **INTEGRATED.**

**Changes**:
- `05_concat_per_source.py` now streams. For each (corpus_id, source_id) group, it opens output via `pyarrow.parquet.ParquetWriter` once, then iterates each contributing shard via `ParquetFile.iter_batches(batch_size=20_000)` and writes batch-by-batch. Never materialises a single union frame.
- Source assignment determined by reading a single batch per shard for the `corpus_id` / `source_id` tag (~ms per shard), then routing.

## Verification

- All 17 scripts re-linted: `bash -n` + `py_compile` clean.
- Pre-flight re-run: 9/9 checks PASS.

## Net effect on the run

The audit now:
- Produces correctly-labelled outputs (`hf_source_pool_*`, not `c3_*`).
- Uses canonical hashes that match the published bundle's `exact_strict_norm_v1`.
- Skips short docs from near-dup pool (no false collisions).
- Filters non-Greek EuroBlocks rows at source.
- Handles EuroParl's dict-typed `translation` field.
- Tears down workers automatically on ANY exit path.
- Never bakes secrets into worker scripts.
- Streams concat without OOM risk.
- Honestly reports skipped held-out check when no holdout list is provided.

The conceptual scope split (Finding 1 reframed) is captured in the REPORT.md preamble — readers downstream cannot confuse this with the exact-C3-mix audit.

The audit is now **safe to spin up** when the user approves. ~$30 / ~55-70 min.
