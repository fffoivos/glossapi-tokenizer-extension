# Vocab-Attribution Run Report — 2026-05-13

**Status: COMPLETE.** Run finished 2026-05-13 16:37:59 UTC. All 8 workers reported done; 1933/1933 canonical-language histograms produced; aggregator ran on `home`; instances stopped (boot disks preserved); HF tokens stripped from worker profiles before stop.

Final artifacts at `02_2_2_vocab_lang_attribution/outputs/`:
- `histogram_matrix.npz` — 58.4 MB compressed; `(1933, 131072)` int64 raw counts; total 113.4 B tokens across all langs.
- `lang_metadata.json` — 1933 entries: ISO 639-3, script tag, sources contributed, sample size, vocab-entries-fired.
- `token_metadata.parquet` — 131,072 rows: decoded_string + ~20 script-flag booleans + structural flags + totals.
- `zero_sum_keys.json` — 4 canonical keys with zero-sum histograms (`chy_Latn`, `gan_Hani`, `abs_Latn`, `kon_Latn`) flagged for re-run targeting.

This doc reviews everything we've built for the per-token language attribution problem: methodology, scripts, infrastructure, artifacts in flight, and operational issues encountered. It exists so a reviewer (or a future agent) can audit and reproduce.

---

## 1. Goal

**Problem**: Apertus-8B-2509 has 131,072 vocab entries. We need to know, for each one, *which natural language(s) it primarily serves*. Existing in-project tooling uses script-codepoint heuristics (e.g. "any token containing U+0370–03FF is Greek"), which:

- Works cleanly for Greek / Cyrillic / CJK (script-mapped languages).
- **Fails for Latin-script tokens** — the heuristic just calls everything "English-baseline" because German / French / Spanish / Italian / Dutch / Polish / etc. tokens that happen to be ASCII-only with ≥3 letters all collapse into one bucket.

This bottleneck blocks per-language embedding-structure experiments (E and U matrix per-language norm distributions, clustering analyses, cross-language overlap studies, candidate-init lookup for C3 extension merges, behavioral per-token-NLL diagnostics that need a clean per-language slice).

**End deliverable**: a per-token table joinable with E / U / `embedding-norm.npy` arrays by `token_id`, containing:
- `primary_lang` (canonical key, e.g. `ell_Grek`)
- `primary_lang_prob` = P(primary | token)
- top-5 language signature with probabilities
- script flags
- structural flags
- confidence tier (high / medium / low / floor)
- raw per-language counts retained in a separate matrix for re-cutoff downstream

---

## 2. Methodology

For each canonical language L:
1. Trust the source dataset's per-language partition as the language label (FW-2 used GlotLID v3; same label Apertus's pipeline trusted).
2. Stream documents from that language's slice, deterministic shard order with seed=42 shuffle.
3. Tokenize each document with the Apertus tokenizer (`swiss-ai/Apertus-8B-2509`, `add_special_tokens=False`).
4. Increment `histogram[L][token_id]` for each emitted token via `np.bincount`.
5. Stop when cumulative tokens for L hit the cap (currently **1 B tokens per language**).

Then post-hoc:
- `P(L | t) = histogram[L][t] / Σ_L' histogram[L'][t]`
- `primary[t] = argmax_L histogram[L][t]`
- entropy, top-K signature, confidence flag per token.

Why not token-level LangID: BPE tokens are too short for reliable LangID classification (single-char or 2-char tokens). Using document-level labels and co-occurrence is the only sound way.

---

## 3. Inputs

### 3.1 Source datasets (verified open + accessible with `HF_TOKEN`)

| Source | Repo | Per-language layout | Used for |
|---|---|---|---|
| FineWeb-2 (raw) | `HuggingFaceFW/fineweb-2` | `data/<iso639-3>_<script>/` | 1,811 langs, base multilingual web |
| FineWeb-2-HQ | `epfml/FineWeb2-HQ` | `<iso639-3>_<script>/` | 20 high-resource langs Apertus consumed via HQ |
| Clean-Wikipedia | `HuggingFaceFW/clean-wikipedia` | `<wikicode>/` (319 langs) | per-language Wikipedia |
| EuroParl | `Helsinki-NLP/europarl` | 210 bitexts (`xx-yy/`) | 21 EU langs, parallel side |
| ParaDocs | `jhu-clsp/paradocs` | 18 `en-XX/` pairs | non-English side per pair |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `data/` (English) | English |
| FineWeb-HQ | `epfml/FineWeb-HQ` | `data/` (English) | English |
| DCLM-Edu | `HuggingFaceTB/dclm-edu` | `data/` (English) | English |

All 8 are Apertus pretraining data sources (excluding code+math which are language-irrelevant). Code/math datasets were deliberately excluded — see the earlier-shipped data-share doc for the rationale.

### 3.2 Canonical language keys

We unify each source's local language code into the FW-2 convention `<ISO 639-3>_<ISO 15924>` (e.g., `ell_Grek`, `deu_Latn`, `cmn_Hani`).

Hand-coded mapping at `subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/scripts/lang_code_map.json` covers:

- 1,811 FW-2 keys (already canonical) → seed set.
- 319 Wikipedia codes → canonical, with explicit rationale per row for quirky cases:
  - `zh` → `cmn_Hani`, `zh-classical` → `lzh_Hani`, `zh-min-nan` → `nan_Latn`, `zh-yue` → `yue_Hani`
  - `nb`/`no` → `nob_Latn`, `nn` → `nno_Latn`
  - `be`/`be-x-old` → `bel_Cyrl` (orthographic variants lumped)
  - `bat-smg` → `sgs_Latn`, `nds-nl` → `nds_Latn`, `roa-tara` → `ita_Latn`
- 21 EuroParl 2-letter codes (ISO 639-1) → canonical via the same table.
- 18 ParaDocs 2-letter codes → canonical.
- English: synthesized as `eng_Latn` (not in FW-2 itself; sourced from FW-Edu/FW-HQ/DCLM-Edu).

Final canonical keys: **1,933**.

### 3.3 Worker partition

8 workers; each canonical key assigned to one worker by `int(sha1(canonical_key).hexdigest(), 16) % 8`. Balanced per-worker counts:
```
w0:242  w1:249  w2:263  w3:224  w4:241  w5:247  w6:208  w7:259
```

Stored at `subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/scripts/partition.json`.

### 3.4 Pre-computed file listings

To avoid HF API rate-limiting (1,000 calls / 5 min per user — which we hit hard on the first attempt), we pre-fetched all 8 source repos' full file listings once from `home` and shipped a 1.25 MB JSON to each worker:

```
file_listings.json: 31,750 parquet/gz file paths across 8 repos
```

Each worker reads this once, then filters per language via local string ops — zero HF API calls during tokenization (only `hf_hub_download` for the actual shards).

---

## 4. Scripts

All under `subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/scripts/`:

| File | Purpose |
|---|---|
| `build_lang_code_map.py` | One-shot script that emits `lang_code_map.json` + `partition.json` |
| `entrypoint.sh` | Worker bootstrap: apt deps, venv with `tokenizers / huggingface_hub / pyarrow / numpy`, env-file with `HF_TOKEN` + `RAYON_NUM_THREADS=192` |
| `tokenize_lang_for_worker.py` | Per-worker driver. Reads partition.json + lang_code_map.json + file_listings.json, processes each assigned canonical key against all applicable sources, writes per-language `arrays/<canonical_key>.npy` + `<canonical_key>.summary.json` |
| `aggregate.py` | Final aggregator (runs on `home`). Loads all per-language `.npy` from `02_2_2_vocab_lang_attribution/staging/<idx>/`, builds `histogram_matrix.npz` + `lang_metadata.json` + `token_metadata.parquet` + `zero_sum_keys.json` |
| `file_listings.json` | Pre-cached HF API responses (31k file paths). Avoids per-worker API rate-limit |
| `lang_code_map.json` | 1,933 canonical keys with source-code mappings |
| `partition.json` | Worker-idx → list of canonical keys |
| `RUN_REPORT.md` | This file |

All scripts have been pushed to all 8 workers under `~/run/`.

---

## 5. Infrastructure

### 5.1 Worker fleet

8 × `c4-highcpu-192` (192 vCPU, 384 GB RAM each) in `europe-west4-b`:

| Worker | Instance name | Provisioning |
|---|---|---|
| w0 | apertus-vocab-attr-w0-20260513t120959z | SPOT |
| w1 | apertus-vocab-attr-w1-20260513t120316z | STANDARD (on-demand) |
| w2 | apertus-vocab-attr-w2-20260513t120959z | SPOT |
| w3 | apertus-vocab-attr-w3-20260513t120959z | SPOT |
| w4 | apertus-vocab-attr-w4-20260513t120959z | SPOT |
| w5 | apertus-vocab-attr-w5-20260513t120959z | SPOT |
| w6 | apertus-vocab-attr-w6-20260513t120316z | STANDARD (on-demand) |
| w7 | apertus-vocab-attr-w7-20260513t120959z | SPOT |

Each: 500 GB hyperdisk-balanced boot disk, default compute service account, `enable-oslogin=TRUE`.

Per-worker peak resource: ~17 GB RAM resident, ~100 % CPU saturation across all 192 cores via Rust `tokenizers` rayon pool.

Total compute: 8 × 192 = 1,536 vCPUs working in parallel.

### 5.2 GCS bucket

`gs://eellak-glossapi-20251008-vocab-attribution` (europe-west4, uniform-bucket-level-access). Created for cross-worker handoff. **Currently unused** — see §7 below.

### 5.3 Run identifier

`RUN_ID=vocab_attr_20260513t120258z`

Persistent env on `home` at `/tmp/vocab_attr_env.sh`, worker name map at `/tmp/vocab_attr_worker_names.txt`.

### 5.4 Cost

Approximate run cost (~2.5 hr total wall):
- 2 standard c4-highcpu-192: ~$8/hr × 2.5 hr × 2 = $40
- 6 spot c4-highcpu-192: ~$4/hr × 2.5 hr × 6 = $60
- Hyperdisk-balanced (500 GB × 8 × 2.5 hr): ~$3
- HF egress + GCS: negligible
- **Total ~$100-110**

---

## 6. Operational issues hit + how we resolved them

These are the lessons captured for future agents:

1. **gcloud auth expired mid-session** — Required user to run `gcloud auth login` interactively. Subsequent gcloud calls worked.

2. **C4 family per-region quota cap ~384 vCPU** — On-demand `c4-highcpu-192` only allowed 2 instances simultaneously in europe-west4 (we tried -a, -b, -c — all gave same error). The general `CPUS` quota is 5,000, but C4 has its own undocumented family cap. Solution: use spot for the other 6 workers, which draws from the much larger `PREEMPTIBLE_CPUS=10,000` pool. (Multi-zone didn't help — same C4 family cap region-wide.)

3. **HF API rate limit (1,000 req / 5 min / user)** — Initial driver called `api.list_repo_files(repo)` once per language per source → ~12,000 API calls in seconds → mass 429 errors → workers silently failed most of their assigned languages (w4 reported "DONE" with only 5 valid files out of 241). Solution: pre-compute all 8 repo's file listings once from `home`, ship as JSON to workers. Zero HF API listing calls during tokenization.

4. **GCS upload from workers failed (403)** — Default compute service account has `devstorage.read_only` OAuth scope baked into the instance metadata, regardless of bucket IAM. Without recreating the instances with `--scopes=cloud-platform`, GCS writes are blocked. Solution: skip GCS uploads from workers, use periodic `tar`-stream over SSH from `home` to pull per-worker `arrays/*.npy` files. One SSH per worker per poll cycle, ~4-min cadence.

5. **Monitor tool 1-hr timeout** — Re-armed when first instance hit its limit. State (done_seen set) is rebuilt on each new monitor run by probing for `_done.flag` files.

---

## 7. Current artifact state

### 7.1 On `home` (durable, what matters)

Sub-subproject root: `subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/`

- **`02_2_2_vocab_lang_attribution/staging/<idx>/*.npy`** — per-language histograms pulled so far, distributed across 8 worker-idx subdirectories. Each is a 131,072-element int64 array (~1 MB).
- **`02_2_2_vocab_lang_attribution/staging/<idx>/*.summary.json`** — per-language metadata: sample tokens, vocab entries fired, sources used, wall time per language.
- **`02_2_2_vocab_lang_attribution/scripts/`** — all scripts, mappings, listings:
  - `build_lang_code_map.py`, `lang_code_map.json` (1,933 canonical keys)
  - `partition.json` (8-worker assignment by hash)
  - `file_listings.json` (pre-cached HF API responses, 31k file paths)
  - `entrypoint.sh`, `tokenize_lang_for_worker.py`, `aggregate.py`
- **`02_2_2_vocab_lang_attribution/outputs/`** — destination for final aggregated artifacts (populated by `scripts/aggregate.py` after the run completes).
- **`02_2_2_vocab_lang_attribution/RUN_REPORT.md`** — this file.

Convenience symlink `/home/foivos/vocab_attr_staging` → `02_2_2_vocab_lang_attribution/staging/`, kept while the in-flight monitor has the old path baked into its loop.

### 7.2 On each worker (transient — will be lost on instance delete)

- **`/mnt/data/outputs/arrays/<canonical_key>.npy`** — same files as on home (single source of truth).
- **`/mnt/data/outputs/_done.flag`** — written when worker finishes its full partition. 5 workers (w1, w2, w5, w6, w7) have this. w0, w3, w4 still running.
- **`/mnt/data/logs/worker.log`** — full per-language tokenization log on each worker. Kept across `stop` (boot disk preserved).

### 7.3 GCS bucket

Empty (unused after we pivoted to SCP-based pull, see §6 issue 4). Bucket will be cleaned up at end of run.

---

## 8. What's next

1. **Wait for w0, w3, w4 to finish** — currently on big languages (San_Latn/etc, kor_Hang at 768M/1B tokens, hin_Latn). ETA ~45-60 min more for slowest worker.
2. **Aggregate on `home`** (post-review, simplified per user direction):
   `python3 scripts/aggregate.py --staging staging --lang-map scripts/lang_code_map.json --out outputs`
   produces:
   - `outputs/histogram_matrix.npz` — `(N_langs, 131072)` int64 raw counts. Lossless.
   - `outputs/lang_metadata.json` — per canonical key: row index, ISO 639-3, ISO 15924 script, name, family, sources_contributed (with per-source token counts from the worker's summary.json), sample_tokens_total, vocab_entries_fired, vocab_entries_fired_geq_10/100.
   - `outputs/token_metadata.parquet` — 131,072 rows: decoded_string, byte_length, script-flag booleans (has_greek_mono, has_greek_poly, has_cyrillic, has_latin_basic, has_latin_extended, has_han, has_hiragana, has_katakana, has_hangul, has_arabic, has_hebrew, has_devanagari, has_thai, etc.), structural flags (is_special, is_byte_fragment, is_pure_ascii, is_pure_digits, is_pure_whitespace, is_structural_only), total_count_all_langs, total_langs_with_any_count.
   - `outputs/zero_sum_keys.json` — list of canonical keys whose histogram is all-zeros (for re-run targeting).
   - **Deliberately NOT** produced: `primary_lang`, `primary_lang_prob`, `signature_entropy`, `confidence_flag`, `top5_*`. Those are downstream investigation decisions per user direction; the raw histogram matrix carries the full information.
3. **Strip HF_TOKEN from `/mnt/data/profile.sh` on each worker before stop** (security: stop-preserved boot disks would otherwise retain plaintext token).
4. **Stop (not delete) all 8 instances** — preserves boot disks for follow-up runs without re-bootstrapping. Disk cost while stopped: $0.04/GB-month × 500 GB × 8 ≈ $5/month.
5. **Re-run any zero-sum canonical keys** (the driver's "skip-if-exists" logic fossilized 4 zero-output histograms; the aggregator surfaces them in `zero_sum_keys.json` so the next pass can target only those).
6. **Index update**: add pointers from `docs/PROJECT_INDEX.md` and from the embedding-norm analysis docs.
7. **Downstream analyses** (next session):
   - Script-family overlap from raw counts + `script_iso15924` per language.
   - "Out of place" detection: tokens with `has_greek_*` codepoints but non-zero counts in non-Greek-script languages.
   - Join `histogram_matrix.npz` with E/U norm arrays from `runs/apertus_greek_diagnostic_20260511/` for per-language norm distributions.
   - Per-source per-token breakdown (currently we only have summed-per-language histograms — if debug-grade breakdown is needed, a separate re-tokenize pass with per-source histograms tracking).

---

## 9. How to reproduce / resume

If this run dies and we need to resume:

1. Stop the broken monitor: `TaskStop bi2tdlcy8` (or whichever).
2. The driver's per-language `.npy` resume logic kicks in automatically when re-launched — it skips languages with existing `.npy` files.
3. Re-launch by SSHing to each worker:
   ```bash
   tmux new -d -s vocab_resume "bash -lc 'source /mnt/data/profile.sh && python3 ~/run/tokenize_lang_for_worker.py --partition ~/run/partition.json --lang-map ~/run/lang_code_map.json --file-listings ~/run/file_listings.json --worker-idx <IDX> --token-cap 1000000000'"
   ```
4. Re-arm the SCP-pull monitor (same shape as in this session).

If a spot worker is preempted, the boot disk is gone (we used `--instance-termination-action=DELETE`). To resume:
1. Re-create with same worker-idx label and the same RUN_ID.
2. Re-run entrypoint + driver.
3. Driver re-tokenizes from scratch (no resume across instances). Per-language sync to `home` was happening every ~4 min so loss is bounded to <4 min of work.

---

## 10. Key open questions for the user when this completes

1. **Cutoff for "this token is language L"** — multiple flavors available from the raw matrix: absolute count, P(L|t), normalized by per-language sample size. Want me to pre-compute a few default cutoffs as separate `per_language_token_lists_at_*.json` artifacts, or leave that purely as an analysis-time decision?
2. **Cross-language overlap presentation** — script-aware bitwise overlap matrices for: Slavic-Cyrillic, Slavic-Latin, Western-European Latin, Romance Latin, Germanic Latin, Han-using, Arabic-script-using. Want me to ship these as derived artifacts or just keep the raw matrix?
3. **Instance teardown timing** — stop immediately after aggregation, or leave running for 24-48 hrs in case we want to drill into any specific worker's logs?
