# C3 Convergence

Date converged: **2026-05-11**.

## Decision

**C3 (`C3_wave2_broad_glossapi_plus_hplt_50_50`) is the tokenizer arm
under active consideration for shipping.**

C3 is a continuous BPE extension of `swiss-ai/Apertus-8B-2509`:
- base vocab: `131072` (Apertus, preserved exactly)
- added units: `25600`
- total vocab: `156672`
- corpus mix: `glossapi + hplt` at `50 / 50` by training-token mass
- cleaner: wave-2 broad cleaner (latest at training time)
- training record: `c3_driver.sh` →
  `/home/foivos/tmp_c3_patch/c3_driver.sh`

The other tokenizer arms (`F1`, `F2`, `C1`, `C2`) are retained as
**analyzed baselines only**. They do not drive any further work in this
project. Anything that still frames the tokenizer track as a four-arm
comparison is stale framing — treat it as archived background, not as
the live plan.

## What Is Still Open

The cutoff — i.e. how many of C3's 25,600 added units to keep — is the
only remaining tokenizer-side decision. Candidate cutoffs are frozen at
the grid:

- `10240`
- `15360`
- `20480`
- `25600`

Each cutoff corresponds to a merged-variant total vocab size of
`131072 + N`. The grid was frozen in
`subprojects/02_1_tokenizer_experiments/CONTINUOUS_BPE_EXTENSION_TODO.md`
§1.4 and stays valid under the C3-only framing.

The shipped cutoff is also required to be `128`-aligned per
`GLOBAL_DECISIONS.md`. All four candidates already satisfy this.

## Current C3 Artifacts On `home`

- Per-added-token analysis + corrected glossary
  (25,600 rows, contiguous ids `131072..156671`):
  - `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/`
  - `data/glossary/tokens_glossary.jsonl` — corrected glossary (the
    authoritative source for downstream analysis)
  - `data/glossary/distribution_summary.json` — full-extension
    category/structure/lexical breakdown
- C3 vs C1 Greek category diff:
  - `~/Projects/glossapi-tokenizer-extension/tokenizer_analysis/inspection/c1_vs_c3_greek_category_diff_20260507/`
- Cutoff-grid distribution (added 2026-05-11):
  - `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/cutoff_grid/`
  - `distribution_at_<N>.json` for every multiple of 1024 from 1024 to
    25600 (25 cutoffs)
  - `cutoff_grid_summary.md` — category × cutoff comparison table
  - `scripts/apply_cutoff_grid.py` — slicer (rerun on glossary
    re-correction)
- Full cutoff sweep evaluation (added 2026-05-11):
  - [C3_CUTOFF_REPORT.md](C3_CUTOFF_REPORT.md) — fertility +
    composition tables + plots over the 25 1k-step cutoffs, evaluated
    on three verified-clean held-out slices
- Training datasets inventory + source links:
  - [C3_TRAINING_DATASETS.md](C3_TRAINING_DATASETS.md) — every source
    that fed C3's BPE training, with HF / upstream links

The C3 `tokenizer.json` itself lives on the gcloud worker, under
`/home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/tokenizers/`,
not on `home`. Local-only analysis can run from the glossary.

## What Comes After Cutoff Is Picked

Downstream stages are unchanged from the project plan; they just narrow
from "evaluate the arms then pick" to "evaluate C3 cutoffs then pick":

1. Cutoff sweep on C3:
   - build merged Apertus-compatible variants at each grid cutoff
   - run intrinsic metrics (`bytes_per_token`, `tokens_per_byte`,
     fertility, added-token utilization, vocabulary utilization,
     unreachable added tokens, byte-fallback rate)
   - evaluation slices: `GlossAPI` held-out, `HPLT` held-out, mixed
     held-out, `modern_greek_eval` (primary)
2. Pick the elbow → shipped cutoff (already `128`-aligned).
3. `subprojects/02_2_tokenizer_implementation` — patch Apertus through
   `model.vocab` + `model.merges` (no `add_tokens(...)`), append-only,
   preserve first 1000 ids and special-token behavior.
4. `subprojects/03_apertus_extension_and_embedding_adaptation` —
   embedding + `lm_head` init for new rows only, frozen-base warmup,
   then full CPT.

## Held-out integrity

The C3 train/val/test splits live at
`/home/foivos/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/splits/glossapi_plus_hplt_50_50/exports/`
on the gcloud instance and were produced by
`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`.

Verification on 2026-05-11 found the splits are **not perfectly
disjoint** from train at the text level:

- train: 14,401,554 rows, 14,372,027 unique text md5 — **29,527
  duplicate texts inside train**.
- val: 7,654 rows / 7,654 unique md5 (clean internally).
- test: 7,282 rows / 7,282 unique md5 (clean internally).
- train ∩ val = **30** docs.
- train ∩ test = **36** docs.
- val ∩ test = 0 (clean).

Cause: the splitter partitions by `source_split_row_id` (row index),
not by text or `source_doc_id`. When duplicate texts existed in the
input mix, the row-level partition independently sent the copies to
different splits. See the comment at the top of
`subprojects/_archive/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`.

Practical impact on the C3 cutoff fertility sweep: the contamination
is ~0.4–0.5% of val/test, and the fertility suite samples 300 docs per
slice → expected leak in each sampled subset is ~1 doc. Below the
metric noise floor, but the slices are not verifiable held-outs.

**Clean held-out path used instead**: virgin HPLT evaluation slice at
`/home/foivos/runs/c3_cutoff_eval_20260511/virgin_hplt_eval/hplt_virgin_eval_20260511.parquet`,
built by anti-joining the `fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`
HPLT release against C3 train text-md5. Every row in that file is
guaranteed not-seen by C3's BPE training. See
`/home/foivos/runs/c3_cutoff_eval_20260511/build_virgin_hplt.py` for
the builder.

Fix path for future tokenizer arms: either dedup the input mix on
`text` before split assignment, or change `stable_key` in the splitter
to be a hash of the text content (so duplicate texts collapse to a
single key and land in the same split). Not back-ported to C3's
exports.

## What Is Now Archived Framing

The following docs and sections still describe the pre-convergence
multi-arm plan. They are retained for traceability but should not drive
new execution. Each will carry an archive banner pointing back here:

- `docs/CURRENT_STATUS.md` — pre-convergence active phase
- `docs/ACTIVE_BACKLOG.md` — tokenizer-critical-path items #11–#17 that
  describe the four-arm matrix and shipping-size selection across arms
- `subprojects/02_1_tokenizer_experiments/README.md`
- `subprojects/02_1_tokenizer_experiments/CONTINUOUS_BPE_EXTENSION_PLAN.md`
- `subprojects/02_1_tokenizer_experiments/CONTINUOUS_BPE_EXTENSION_TODO.md`
- `tokenizer_analysis/hf_snapshots/apertus-tokenizer-extension/README.md`
  (bundles the four raw arms; treat as a frozen comparison snapshot)

If you are an agent reading this project for the first time: read this
file, `GLOBAL_DECISIONS.md`, and `ACTIVE_BACKLOG.md`'s "Active C3
cutoff decision" block. Skip the multi-arm sections unless you are
explicitly doing historical reconstruction.
