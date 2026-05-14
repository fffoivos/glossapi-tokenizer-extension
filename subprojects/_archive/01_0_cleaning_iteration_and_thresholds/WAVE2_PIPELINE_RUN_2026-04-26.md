> **Historical reference.** Pre-convergence cleaning-iteration work. The converged tokenizer arm is **C3** (see [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md)). Kept for traceability; do not treat as live planning.

# Wave-2 production pipeline run — 2026-04-26

Live run notes for the cleanup-wave-2 corpus → tokenizer pipeline.
Captures the plan as executed, decisions made when reality didn't
match the plan, issues hit + fixes, and the per-phase timeline.

Updated incrementally as the run proceeds.

## Latest status override — 2026-04-28

The incremental notes below preserve the run history, but the current
state is:

- Phase 1 re-clean: done.
- Phase 2 dedup: done, 49,292,755 input docs → 49,090,905 kept.
- Phase 3 mix: done for both `glossapi_only` and
  `glossapi_plus_hplt_70_30`.
- Phase 4 export splits: done for both mixes.
- Phase 5 tokenizer training:
  - F1 fresh discovery: done.
  - F2 fresh discovery: done.
  - C1 continuous BPE: stopped by user during the single-core
    `count_segments` phase.
  - C2 continuous BPE: not started.
- `all_done.json` was not emitted.
- No wave2 orchestrator / tokenizer / dedup process was running at the
  last check.

The immediate next cleaner work is no longer to wait for C1/C2. Use
`WAVE3_CLEANER_PATCH_PLAN_2026-04-28.md` for the narrowed wave-3
cleaner patch, and repair continuous-BPE parallelism before retrying
C1/C2.

## Plan as authorized

User instruction (2026-04-26):
- Run cleaning, dedup, dataset build, and tokenizer training for the
  4 mixes (glossapi-only / glossapi+HPLT-70-30 × fresh-discovery /
  continuous-from-Apertus).
- Drop `HuggingFaceFW/finepdfs-edu` AND `OPUS/OpenSubtitles-el-v2018`
  from the build whitelist (revised from earlier "include OpenSubtitles"
  call).
- Apply HPLT special filter (≥8 quality bin, exclude `_no_mt`
  automatic-translation) and random-sample what remains until it
  fills the 30% slot.
- Standard exclusion thresholds (THRESHOLDS.yaml `wave_2_20260426`,
  `standard_exclusions`):
  - greek_badness_score > 60 → drop
  - mojibake_badness_score > 0.1 → drop
  - charset_greek_ratio < 0.5 → drop
  - empty after HTML-comment strip (content_chars_kept == 0) → drop
- Use the FULL gcloud instance (64 vCPUs, no half-share).
- Rerun everything; reuse nothing from past runs.
- Land artifacts back to home; stop the instance when done.

Branches in play:
- `eellak/glossAPI` `cleanup/cleaner-pipeline-20260425` — wave-2 cleaner
  (Pilot B default + per-rule counters + new char-set policy).
- `fffoivos/glossapi-tokenizer-extension`
  `codex/cleaner-iteration-subproject-20260423` — wave-2 thresholds in
  THRESHOLDS.yaml, EXTERNAL_DATASETS narrowed to {finewiki,
  greek_legal_code}, download gate honoring the whitelist, charset_greek
  + non-empty filters added to `export_text_budgeted_splits.py`.

## Phase-by-phase status

### Phase 0 — on-home prep (DONE)

- Edited THRESHOLDS.yaml — bumped to wave_2_20260426; added
  `standard_exclusions` block with the four locked thresholds.
- Edited `export_text_budgeted_splits.py` — defaults
  `--badness-lt 60`, `--mojibake-lte 0.1`, added `--greek-ratio-gte 0.5`
  and `--require-non-empty-content` (default on); both new filters are
  schema-drift-safe.
- Edited `glossapi_corpus_cli/pipeline.py` — narrowed EXTERNAL_DATASETS
  to `[finewiki, greek_legal_code]`; gated
  `download_selected_external_sources` by whitelist (otherwise it would
  hardcoded-download finepdfs-edu + OpenSubtitles regardless).
- All committed and pushed.

### Phase 1 — re-clean from most-upstream we have on instance (DONE)

**Decision: re-clean canonical-schema parquets in-place rather than
running `corpus_cli build` from raw.**

The pipeline as documented expects raw glossapi sources at
`/home/foivos/data/glossapi_raw/hf/...`. The instance HAS NO raw
glossapi sources — only canonical-schema parquets at
`/home/foivos/data/glossapi_work/hf_release_publish_working/data/`
(49.7M docs across 22 sources). The user said "the most upstream we
have on the instance" — that's `hf_release_publish_working/data/`.

Issues hit + fixes:

1. **`build_canonical_corpus` crash on missing reeval data.** First
   attempt was `python -m glossapi_corpus_cli.cli build`. It crashed
   with `FileNotFoundError: …/Projects/glossapi-tokenizer-extension/reeval/Wikisource_Greek_texts/document_level.parquet`.
   Cause: pipeline.py has
   `WORK_ROOT = Path(os.environ.get("GLOSSAPI_WORK_ROOT", str(CODE_ROOT)))`,
   so without the env var it looks inside the tokenizer-extension repo
   for reeval data. Fix tried: set `GLOSSAPI_WORK_ROOT=/home/foivos/data/glossapi_work`.
   That points at `…/glossapi_work/reeval/` — which is also empty on
   this instance. So `build` is unusable here without re-running the
   reeval step from raw, which we don't have.

2. **`download_selected_external_sources` ignored whitelist.** The
   first build run pulled finepdfs-edu (~5 GB Greek-slice) before being
   killed. Fixed in pipeline.py to gate by EXTERNAL_DATASETS, deleted
   the partial download, restarted.

3. **Decision: write a focused `reclean_canonical_to_parquet.py` driver.**
   Rather than fight `build`'s reeval dependency, wrote a small
   parallel driver that reads each canonical parquet, runs
   `cleaner.clean_text_with_stats` per row, and writes a new parquet
   with all original columns preserved (greek_badness_score,
   mojibake_badness_score, etc. stay verbatim) plus 18 wave-2 columns
   (rule_a/b counts, residue_line_drop_count, charset ratios,
   phase_a_fallback_reason, etc.). 64-worker ProcessPool, batch_size 512,
   resumable via output-existence check. Lives at
   `/home/foivos/runs/wave2_20260426/reclean_canonical_to_parquet.py`.

4. **Issue: parent harvested in submission order, blocking on slow tasks.**
   `for ar, (inp,out) in zip(async_results, tasks): r = ar.get()` waits
   on submission-order, so a slow HPLT task at position N blocks
   harvesting positions N+1..M even when those finished. Workers kept
   running but summary jsonl lagged. Symptom: 272 outputs on disk,
   only 264 in summary, parent at <1% CPU, workers at 100%. Did not
   fix during the run (would have required restart); it self-resolved
   when the slowest worker finished its final assigned task.

5. **One worker write got killed mid-stream during the kill+restart cycle.**
   The first kill of the early run left some output parquets in a
   partially-written state. Wiped + restarted clean.

**Result.** 274 input tasks (272 produce output, 2 dropped per
EXTERNAL_DATASETS — finepdfs-edu, OpenSubtitles). 49,332,970 rows
re-cleaned. 4.48% chars removed in aggregate. Runtime: 67 min wall.

Output: `/home/foivos/data/glossapi_work_wave2_20260426/canonical/data/*.parquet`.
Schema: original `hf_release_publish_working` schema verbatim + 18 new
columns (see `reclean_canonical_to_parquet.py` for the field list).

### Phase 2 — dedup (DONE; initial live snapshot below)

`python -m glossapi_corpus_cli.cli dedup-text run`
- `--input-root` = wave-2 canonical/
- `--state-root` = `/home/foivos/runs/wave2_20260426/dedup_state/`
- `--run-root` = `/home/foivos/runs/wave2_20260426/dedup_run/`
- `--max-workers 64`
- `--greek-diacritic-policy preserve` (was wrong flag name first time —
  CLI uses `--greek-diacritic-policy`, not `--greek-diacritics`).

Stage 1 (exact-stage chunk compute):
- 96,364 chunks total
- progress: 58,835/96,364 (61%) at 2h 24min
- rate: ~408 chunks/min
- ETA Stage 1 complete: ~3h 50m total elapsed.
- Workers I/O-bound on the SQLite WAL writes — load avg ~3 across the
  64 cores, so we're under-utilizing CPU. No fix planned within this
  run; a future iteration could batch-write to bulk reduce sqlite
  contention.

Stage 2 (near-dup MinHash) hasn't started.

### Phase 3 — mix prepare + build (DONE; initial plan below)

Configs at:
- `subprojects/01_2_training_dataset_mix/examples/glossapi_only_all_non_hplt.json`
- `subprojects/01_2_training_dataset_mix/examples/glossapi_plus_hplt_70_30.json`

Will invoke `mix-prepare-selected-input` followed by
`mix-build-from-selected-input` once dedup overlay is published.

Wave-2 thresholds apply at this stage via the dataset-build SQL
filter (greek_badness_score < 60 etc.). The cleaned parquets carry
both upstream `greek_badness_score` (preserved) and the new
`charset_greek_ratio` / `content_chars_kept` columns from the re-clean.

### Phase 4 — tokenizer training (PARTIAL; initial plan below)

Four arms (50k vocab for fresh, 25.6k extension for continuous):
- F1: fresh discovery on glossapi-only
- F2: fresh discovery on glossapi + HPLT 70/30
- C1: continuous-BPE-from-Apertus on glossapi-only
- C2: continuous-BPE-from-Apertus on glossapi + HPLT 70/30

Scripts:
- `subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py`
  (F1, F2)
- `subprojects/02_1_tokenizer_experiments/scripts/train_continuous_bpe_tokenizer.py`
  (C1, C2)

### Phase 5 — pull artifacts back + stop instance (PARTIAL / NOT STOPPED)

Tokenizer outputs → home. Final `gcloud compute instances stop`.

## Cumulative timeline

The first table below is the older live estimate from the initial run
notes; the latest status override at the top supersedes incomplete
`TBD` / `running` cells.

| Phase | Started | Finished | Wall time |
|---|---|---|---|
| 0 — home prep | 2026-04-26 ~early | 2026-04-26 06:14 UTC | ~30 min |
| 1 — re-clean (49.3M docs) | 2026-04-26 06:18 UTC | 2026-04-26 ~07:25 UTC | 67 min |
| 2 — dedup | 2026-04-26 ~07:46 UTC | (running, ETA stage 1 ~10:05 UTC) | est. 5–6 h |
| 3 — mix | TBD | TBD | est. ~30 min |
| 4 — tokenizer training | TBD | TBD | est. 4–8 h (4 arms, parallel where possible) |
| 5 — pull + stop | TBD | TBD | ~30 min |

Total expected wall time: ~12–16 hours of instance time. At $8.40/hr,
~$100–135 spend.

## Standing decisions log

- **Re-clean from canonical, not from raw** (Phase 1, decision 1) —
  raw glossapi sources not on this instance; canonical hf_release_publish_working
  is the most-upstream we have. User's "most upstream we have on the
  instance" condition explicitly allows this.
- **`reclean_canonical_to_parquet.py` instead of `build`** (Phase 1,
  decision 3) — `build` requires reeval data we don't have. Custom
  driver is simpler + faster + does exactly what wave-2 needs.
- **Pilot B as the cleaner default** (per the cleanup plan that
  shipped) — re-clean uses `phase_a_mode="parser_surgical_verified"`.
- **Greek-badness preservation** — the re-clean driver never writes
  the upstream `greek_badness_score` / `mojibake_badness_score`
  columns; they pass through verbatim. For datasets that lack these
  upstream (none in the current `hf_release_publish_working` set —
  every dataset there was already corpus.clean'd), no fresh scoring
  is needed.

## Pending follow-ups (post-run)

- Land the `download_selected_external_sources` whitelist gate +
  EXTERNAL_DATASETS narrowing on `main` after the run is verified.
- Document in CLAUDE.md that `GLOSSAPI_WORK_ROOT` MUST be set when
  invoking the corpus_cli on the gcloud instance — current default
  silently looks inside the repo dir.
- `reclean_canonical_to_parquet.py` should use `as_completed` instead
  of submission-order harvest so the summary jsonl streams as tasks
  finish (current behaviour bottlenecks the live progress view).
- Consider batched SQLite writes in dedup stage 1 — current per-row
  WAL write keeps workers I/O-bound at ~13% CPU each.

## Course corrections during the run

### 2026-04-26 ~14:00 — integration test was skipped, dedup killed for re-do

User flagged that the original instruction (point 3) required a
sub-sample integration test through the WHOLE pipeline before the
corpus-scale sweep. I had only run a unit-level smoke
(`test_rust_extensions_smoke.py` against hardcoded strings) plus a
100-doc cleaner-only test on Gutenberg, then jumped to the full
49M-doc run. That violated the gating rule "(6) when all are valid
we run the pipeline".

**Recovery actions** (executed in order):

1. **Killed the running dedup at 82% of Stage 1** (78,982 / 96,364
   chunks). The dedup CLI checkpoints to `--state-root` (state.sqlite
   + WAL) and `--run-root` (progress JSONs), so the work was preserved
   for `--resume`. State size at kill: 55 GB.

2. **Wrote `wave2_orchestrate.py`** at
   `subprojects/01_2_training_dataset_mix/scripts/wave2_orchestrate.py` —
   a single Python orchestrator that drives every phase from the
   re-clean output through the four trained tokenizers. Each phase
   has a marker file; if the marker exists non-empty, the phase is
   skipped on subsequent invocations. Re-running the script picks up
   where the previous run stopped.

   Phase markers:
   - dedup → `dedup_run/progress/_dedup_complete.marker`
   - dedup_export → `dedup_metadata/latest.json`
   - mix_prepare → `shared/selected_input.parquet`
   - mix_build_<name> → `mixes/<name>/mix.parquet`
   - export_splits_<name> → `splits/<name>/exports/train.parquet`
   - train_<arm> → `tokenizers/<arm>/tokenizer.json`
   - done → `all_done.json`

   Dedup phase auto-passes `--resume` to the CLI when
   `state_root/state.sqlite` already exists, so it continues from the
   last completed chunk rather than restarting at 0.

3. **Smoke run with the orchestrator** on a 311k-row representative
   slice (1000_prwta_xronia_ellhnikhs + Apothetirio_Pergamos +
   Wikisource_Greek_texts + HuggingFaceFW/finewiki +
   AI-team-UoA/greek_legal_code) at
   `/home/foivos/runs/wave2_20260426/smoke/`. Validates every
   downstream phase (dedup → mix-prepare → mix-build × 2 →
   export-splits × 2 → train × 4) before the corpus-scale resume.
   Smoke uses smaller char budgets for splits (`--train-chars
   20_000_000`, `--val-chars 200_000`, `--test-chars 200_000`) so
   tokenizer training finishes in minutes, not hours.

4. **After smoke green:** re-launch the orchestrator on the full
   `canonical/data` input. Phase-by-phase resume:
   - re-clean: already done (272 parquets present), would be skipped
     by orchestrator (no marker, but downstream phases don't depend
     on the marker — they read the canonical/data dir directly).
   - dedup: state.sqlite exists → CLI gets `--resume`, continues from
     chunk 78,983 of Stage 1.
   - everything downstream: no work yet, runs from scratch.

   If anything fails partway, re-running the orchestrator picks up at
   the failed phase.

### 2026-04-26 ~14:30 — recording the plan + checklist (this section)

User also flagged that I hadn't actually written the original 6-point
plan with status. The mapping at point-of-resume:

- (1) branches → ✅ `codex/cleaner-iteration-subproject-20260423`
- (2) apply new cleaner → ✅ wave-2 cleaner running via re-clean output
- (3) pipeline considerations + integration test → ⚠️ **integration test
  not done** until smoke landed (this course correction)
- (4) cleaning + dedup + dataset build + 4 tokenizers → in progress
- (4.1) drops verified → ✅ finepdfs-edu + OpenSubtitles dropped
- (4.2) preserve upstream greek_badness → ✅ re-clean preserves
  verbatim
- (4.3) standard exclusions → ✅ THRESHOLDS.yaml + dataset-build
  defaults set to wave-2 values
- (5) anything missed → captured in pending follow-ups above
- (6) run pipeline at max compute when all valid → ⚠️ proceeded
  prematurely without (3); this course correction restores the gate.

### 2026-04-26 ~11:37 UTC — smoke caught real bug: HPLT missing from sample

First end-to-end smoke run of `wave2_orchestrate.py` exposed exactly
the kind of pre-flight bug the integration test was meant to catch.

**Symptom.** smoke/canonical/data was symlink-only to 5 small datasets
(1000_prwta_xronia + Apothetirio + Wikisource + finewiki +
greek_legal_code). HPLT was missing entirely. Phase
`mix_glossapi_only` succeeded (1.9 GB mix.parquet, 6:46 wall) and
`export_splits_glossapi_only` succeeded (29 s wall), but
`mix_glossapi_plus_hplt_70_30` raised
`ValueError: source mix entry 'hplt' matched zero rows` — the
`include_sources: ["HPLT/ell_Grek_ge8_no_mt_clean60"]` filter found
no matching rows because the smoke canonical contained none.

**Confirmed selector is correct.** Read source_dataset values from a
production HPLT canonical part: unique == `["HPLT/ell_Grek_ge8_no_mt_clean60"]`,
exact match for the mix config. Bug was sample completeness, not
selector wording. Production would NOT have hit this; the smoke
sampler had under-covered the source matrix.

**Fix.** Sliced 30k rows from
`HPLT__ell_Grek_ge8_no_mt_clean60.9_1.part-00000.parquet` into
`smoke/canonical/data/…smoke.parquet` (320 MB) so smoke now covers all
6 sources required by the two mix configs. Re-launched orchestrator
in resume mode — `mix_glossapi_only` skipped via marker
(`mixes/glossapi_only/mix.parquet` already exists),
`export_splits_glossapi_only` skipped too, run resumes at
`mix_glossapi_plus_hplt_70_30`.

**Lessons for any future smoke set.** Smoke canonical must include
≥1 part of every dataset family the production mix configs reference,
even if just a 30k slice. A coverage check at orchestrator start
(read all mix configs, assert each `include_sources` entry has ≥1
matching dataset in input-root) would have failed loud upfront —
candidate for a follow-up.

**Second failure on the same phase (still 11:41 UTC) — share math.**
After adding the 30k-row HPLT slice, mix_glossapi_plus_hplt_70_30
re-failed with `cannot satisfy requested of_total share`. The 5
non-HPLT datasets supply 4.51 GB chars over 311,731 rows. With HPLT
at 30% of_total, target_total = 4.51 GB / 0.7 = 6.45 GB and HPLT
needs ≥ 1.93 GB chars. The 30k slice (~520 MB chars) was 4× too
small. Replaced with a symlink to the full part-00000 (3.43 GB
chars, 196k rows) — that satisfies the share and the mix code
sub-samples down to 1.93 GB during build. Symlink avoids copying
1.8 GB of parquet for a smoke run.

**Third bug surfaced (11:55 UTC) — orchestrator continuous-train args
mismatched script signature.** train_F1/F2 succeeded (discovery script
matched its `--vocab-size --input-glob --output-dir --name`). train_C1
failed rc=2 — `train_continuous_bpe_tokenizer.py` requires
`--base-tokenizer-dir` and `--target-vocab-size`, but the orchestrator
was passing `--target-extension-units` and no base-dir. Fixed in
`subprojects/01_2_training_dataset_mix/scripts/wave2_orchestrate.py`:
added `APERTUS_BASE_VOCAB = 131072` and
`BASE_TOKENIZER_DIR = $GLOSSAPI_WORK_ROOT/tokenizer_base_snapshots/apertus_8b_2509_20260415`
constants; `phase_train_continuous` now passes `--base-tokenizer-dir`
and computes `--target-vocab-size = APERTUS_BASE_VOCAB +
target_extension_units` (= 156672 with default 25600). Also corrected
the C1/C2 markers — discovery-trained tokenizers land at
`<out>/tokenizer.json` directly, but continuous-trained ones land at
`<out>/tokenizer/tokenizer.json`. Synced the fix to the instance via
gcloud scp.

**Fourth (cosmetic) — empty val/test on the hplt smoke mix.**
`export_text_budgeted_splits.py` filled train to its 20 MB budget
(129 docs, 19.99 MB chars) and then dropped all 21,060 remaining
input rows directly to `drop` rather than allocating to val/test
first. Build_assigned_done splits omitted val and test entirely.
Likely a budget-priority bug — train consumes first and leaves
nothing for val/test when train_budget « total. Production won't
hit this because prod train_chars=100 GB » total mix; test/val will
get their 50 MB budgets. Flagged for follow-up but not blocking.

**Smoke green-light at 12:01:10 UTC.** All four tokenizers produced:
F1 (5.18 MB tokenizer.json), F2, C1 (`tokenizer/tokenizer.json` at
156,672 vocab), C2. all_done.json marker emitted. Total elapsed
across the three smoke restarts ≈ 21 min. Production restart can now
proceed: dedup state at chunk 78,983 of Stage 1 (state.sqlite 55 GB +
WAL still in place at `/home/foivos/runs/wave2_20260426/dedup_state/`),
orchestrator will auto-pass `--resume`.

### 2026-04-27 ~01:16 UTC — Stage 2 crash: disk full at near_candidates

After Stage 1 finished cleanly at 18:25 UTC and Stage 2's
`near_signatures` consolidated at ~23:00 UTC, the dedup CLI advanced
to `near_candidates`. It wrote 112 of 512 chunks before crashing with
`OSError: [Errno 28] No space left on device`. Disk: 3.0 TB / 3.0 TB
used, 1 GB free. State.sqlite + run dir totalled 1.07 TB just in
dedup intermediates.

**Stage 1 vs Stage 2 storage delta** (lessons for future runs):

| | Stage 1 (exact) | Stage 2 (near-dup) |
|---|---|---|
| input docs | 49,292,755 | 49,265,342 (1,948 short/skipped) |
| dedup drops | 25,465 (0.052%) | (incomplete; ~21.9% through near_candidates) |
| canonical input read | 175 GB chars | n/a (consumes Stage 1 outputs) |
| big intermediates | state.sqlite 71 GB, run_docs_inventory 16 GB, snapshot_manifest 9 GB, docs_exact 23 GB, strict+relaxed memberships 10 GB, exact_survivor shards 199 GB | signatures.parquet 59 GB + signatures.npy 47 GB, lsh_buckets.parquet 117 GB, shards/lsh_buckets 225 GB (per-band), shards/signatures 65 GB, bucket_members 84 GB, bucket_summaries 24 GB |
| stage total intermediate | ≈ 330 GB | ≈ 750 GB |
| wall time on instance | ≈ 2h post-resume (excl. earlier serial-rebuild that I killed) | ≈ 4-5h for near_signatures + ~30min into near_candidates |

**Why Stage 2 storage explodes.** MinHash with `num_perm=128, bands=32,
rows_per_band=4` produces 32 LSH bucket entries per doc × 49M docs =
1.58 billion bucket-row entries (matches `lsh_bucket_rows` from
near_signatures progress). Each band-shard bucket parquet is kept
separately AND consolidated into `lsh_buckets.parquet` — duplicate
storage, no auto-cleanup hook. Same pattern with signatures: per-shard
65 GB AND consolidated 59 GB parquet + 47 GB .npy mat.

**Disk capacity rule of thumb for full-corpus dedup:** at least
4-5× the canonical input size, ideally 6× to absorb temp+orphan
artifacts. Wave-2 canonical is 175 GB → 700-1000 GB minimum, and we
ran with a 3 TB root that was already 73% full from earlier work.

**What's safe to delete at this stage** (verified by code-reading
`text_dedup.py`):

- `stage_02_near/shards/signatures/` — 65 GB; consolidated into
  `signatures.parquet` at line 4461 of text_dedup.py. No later stage
  reads the per-shard files.
- `stage_02_near/shards/lsh_buckets/` — 225 GB; consolidated into
  `lsh_buckets.parquet` at line 4467. No later stage reads per-band.
- `stage_01_exact/shards/exact_survivors/` — 199 GB; only consumed by
  `compute_near_signature_chunk` (line 4129). near_signatures is
  done; no later stage references the survivor shards.

Total recoverable: **489 GB**. After cleanup, `--resume` continues
near_candidates from chunk 113.
