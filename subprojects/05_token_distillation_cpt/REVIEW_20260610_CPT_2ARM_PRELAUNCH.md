# Pre-Launch Review — Greek Apertus 13.5B Two-Arm CPT

Reviewer pass: 2026-06-10. Reviews the execution agent's work logged in
`RUN_LOG_20260609_CPT_2ARM.md`, against the live Clariden state.

Method: 6-dimension adversarial audit (17 agents). Each dimension deep-read the
local **and** deployed Clariden files end-to-end; every BLOCKER/MAJOR finding
was independently verified twice (one agent re-reading source to refute it, one
checking live runtime/data on Clariden). Findings below are post-verification.

> **UPDATE 2026-06-10 — run launched; see `STATUS_CONFORMANCE_20260610.md`.**
> **B1 (BLOCKER) RESOLVED before launch** — `pretrain_gpt_te_guard.py` is now
> present in the full-repo tree; both arms loaded the checkpoints and are
> training (proof it's fixed). **M1 (MAJOR) PARTIALLY addressed** — per-set
> held-out loss **is** emitting to the `.out` for all 3 sets, and `EVAL_INTERVAL`
> was raised to 25 (the efficiency suggestion below); the TensorBoard /
> `collect_metrics.py` separability gap still stands if CSV/TB curves are wanted.
> The MINORs remain open but are non-blocking.

## Verdict

**Do NOT launch yet — 1 BLOCKER.** Everything that would *silently corrupt the
science* (holdout leakage, the 70/30 split, decontam/anon coverage, geometry,
the WSD/AdEMAMix/Goldfish schedule, init-checkpoint fidelity) was verified
**correct** — by code-trace and empirically against live data. The one
launch-blocker is a packaging gap (a missing trainer wrapper file), plus one
MAJOR observability defect on the per-set held-out loss the user explicitly
asked for. Both are recoverable and neither has fired yet (nothing has trained;
mix is ~55% built, Stage A/B dependency-pending).

Live state matches the run log exactly: mix `2509367_0-3` running, `4-7` queued;
Stage A `2509368` + Stage B `2509486` dependency-pending; all 6
`val_*_{base,ext}_text_document.{bin,idx}` non-empty; both init `verification.json`
present and clean.

---

## BLOCKER — must fix before launch

### B1. Trainer runtime wrapper `pretrain_gpt_te_guard.py` is missing from the tree the launcher points at → segment 1 crashes immediately

- **Chain:** `launch_all.sh` → `submit_two_arm_full_run.sh` defaults
  `TRAIN_DIR=$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training`
  (`REPO_ROOT=$SC/repo/glossapi-tokenizer-extension`) and exports it as
  `SCRIPT_DIR_OVERRIDE`. `bakeoff_train.sbatch` then runs
  `python3 "$SCRIPT_DIR/../megatron_patches/runtime/pretrain_gpt_te_guard.py" …`.
- **Defect:** that `megatron_patches/` subtree does **not exist** in the rsync'd
  full-repo tree (`init_bakeoff/` contains only `{bakeoff_training, eval,
  slurm_cpu_only_guard.sh}`; the deploy is an rsync, not a git checkout, and the
  subtree was never synced). The file exists **only** in the separate legacy
  standalone tree:
  `$SC/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/runtime/pretrain_gpt_te_guard.py`
  (11,549 B). Reproduced live: `te_guard MISSING -> python3 would fail: No such file`.
- **Impact:** on a real launch (`DRY_RUN=0 CONFIRM_LAUNCH=1`), every rank of
  segment 1 of **both** arms dies at `python3` process start. The wrapper is
  load-bearing — it installs the TransformerEngine empty-`_extra_state` EOFError
  guard required to **load** the HF→Megatron init checkpoints — so it cannot
  simply be dropped. Both verifiers: REAL, BLOCKER, high confidence.
- **Fix:** rsync the missing subtree into the full-repo tree:
  ```bash
  rsync -a \
    "$SC/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/" \
    "$SC/repo/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/"
  # then re-confirm:
  ls "$SC/repo/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/runtime/pretrain_gpt_te_guard.py"
  ```
- **Do NOT** "fix" this by repointing `TRAIN_DIR` at the legacy
  `bakeoff_training/` — its `bakeoff_train.sbatch` is the **older** version
  (lacks `EXTRA_VALID_ARGS`, the `LR_WSD_DECAY_SAMPLES` knob, the
  `--make-vocab-size-divisible-by` override, and the `td`/`extension` ARM cases),
  so it would silently drop the 3 held-out val sets **and** the WSD cooldown.

---

## MAJOR — should fix before launch (user's explicit per-set-loss requirement)

### M1. Per-set held-out loss is not separable in TensorBoard and not parsed by `collect_metrics.py`

- **TensorBoard:** Megatron `evaluate_and_print_results` (training.py ~L1868)
  writes the scalar keyed only on the loss name —
  `writer.add_scalar('{} validation'.format(key), …)` — so all three extra-valid
  sets write the **same** scalar `lm loss validation` each eval and overwrite each
  other (greek_phd wins, last in dict). The per-set `[name]` prefix only enters
  the printed `.out` string. Both eval code paths (periodic + final) have this.
- **collect_metrics.py:** gate (L44) requires the substring `lm loss:`, but the
  validation line uses `lm loss value:` → it parses **zero** per-set losses (it
  only ever parses the training-iteration line).
- **`EXTRA_VALID_README.md`** claims `[hplt] lm loss validation` scalars appear
  and are parseable by `collect_metrics.py` — **false** for the deployed code.
- **Impact:** the 3 per-set numbers *are* printed to the `.out` (recoverable;
  run not corrupted), but the stated observability mechanism the user asked for
  is broken. Both verifiers: REAL, MAJOR, high confidence.
- **Fix (either/both):** (a) inject the set name into the TB scalar key in the
  patch's eval call, e.g. `add_scalar(f'[{name}] {key} validation', …)`; and/or
  (b) add to `collect_metrics.py` a parser for
  `r'\[(\w+)\].*lm loss value:\s*([\d.E+-]+)'`. Correct the README claim.
- **Note:** the README's own §Verification step (a 50-iter smoke with
  `EVAL_INTERVAL=5` to confirm 3 distinct scalars) would have caught this and has
  not been run. Worth running as the post-fix smoke.

---

## MINOR / process / observability

- **MN1 (hyperparams, real):** the submit export-hygiene guard
  (`submit_two_arm_full_run.sh:85-90`) covers 11 sweep vars but omits several
  overridable schedule vars (`LR_WSD_DECAY_SAMPLES`, `ADEMA_BETA3/ALPHA_WARMUP_STEPS`,
  `LR_WARMUP_INIT/TOKENS`, `EVAL_*`); `--export=ALL` would forward a stray shell
  export of these. Mitigated by `gate_cpt2arm_artifacts.sh` (asserts
  `LR_WSD_DECAY_SAMPLES=659179` etc.) and the post-launch `run_metadata.json`
  review. Fix: extend the guard list, or launch from a sanitized shell + run the
  gate first.
- **MN2 (launch, real):** the richer `gate_cpt2arm_artifacts.sh` is **not wired
  into the documented launch sequence** (`LAUNCH_RUNBOOK.md`/HANDOFF go straight
  to `launch_all.sh`); only the weaker in-script `.idx`-existence preflight runs
  automatically. Fix: add `bash scripts/gate_cpt2arm_artifacts.sh` (exit-on-fail)
  as an explicit step before launch when `DRY_RUN=0`.
- **MN3 (launch, real, inert today):** `LAUNCH_RUNBOOK.md:83` xfer-down fallback
  `WATCHER_PARTITION=normal` would be killed by `slurm_cpu_only_guard.sh`
  (`exit 88`, allowed partition = xfer). Moot now — live `sinfo` shows xfer **up**
  with an idle node (contradicts the drained-till-2026-06-11 memo). Fix if ever
  needed: also export `CPU_ONLY_PARTITION=normal`, or use the login-node poll loop.
- **MN4 (holdout, real):** `val_openarchives.jsonl` has 2,150 byte-identical
  duplicate docs (~19% internal duplication; same content-hash id). **No leakage**
  (drop is set-membership, catches all copies); only marginally biases the
  openarchives held-out loss. Optional: dedup on `source_doc_id` before
  tokenizing that val set.
- **MN5 (dataset, real, cosmetic):** `share_within_bucket` fields in
  `bulk_13b.json` sum to 1.3455, but the field is **dead metadata** — mix_builder
  derives within-bucket proportions from the numeric `weight` field. No effect on
  the mix.

## Downgraded / false-positive (recorded for honesty about the review)

- **greek_replay "incidental zero-leakage" → OBSERVATION (downgraded):** the
  finding's mechanism (an HPLT shard split 10_1 vs 8_2/9_2) is **factually wrong**.
  The real guarantee is **structural**: `greek_replay` = docs that ARE in
  `apertus_overlap_drop`; train/holdout = docs in the post-drop `SELECTED` pool —
  disjoint by the dedup keep/drop partition itself (a doc can't be both dropped
  and kept). So zero-leakage is precluded, not coincidental. A belt-and-suspenders
  `drop_doc_keys_parquet` on the greek_replay source would still be cheap if you
  want it enforced rather than structural.
- **"extra-valid patch lacks a provenance copy" → FALSE POSITIVE (refuted):** the
  finding's directory listing was wrong and the named provenance dir doesn't exist
  in subproject 05. At most a stale README sentence.

---

## Verified CORRECT (high-value coverage — these were the real risks)

**Dataset / holdout (code-trace + empirical):**
- 70/30 unseen split is **exact**: hplt 0.5185187 = 0.70×0.740741, oa 0.2222223 =
  0.30×0.740741; ratio 0.700000/0.300000; sums to the greek bucket 0.740741. All
  buckets ≈ 1.0; 8 shards × 1.6875e9 = 13.5e9.
- **Holdout drop join is correct + empirically zero-leak:** parquet col `doc_id`
  holds source_doc_ids; mix_builder drops on `source_doc_id` (same namespace).
  630,024 sampled unseen-Greek docs → **0 leaks**; all 549,839 val ids present in
  the 547,689-unique drop set; 548,258 matching rows already in SELECTED will be
  dropped.
- greek_phd genuinely out-of-training (not a bucket; 0 of its ids in greek_replay).
- Decontam covers the **Greek replay** (it's a real in-mix bucket) and anonymize
  covers the **full stream incl. replay** — both user requirements met. Decontam
  **drops** (not just flags); queries file exists (52 MB / 18,489); `zcat` used for
  gzipped datatrove output; `split -n l/64` → 64 shards; order enforced by Slurm deps.

**Hyperparameters / schedule (config + trainer source):**
- LR warmup = 2/(1−β₂) = **400 iters** (409,600 samples). WSD cooldown = **exactly
  20%** (659,179 samples), 1-sqrt, anchored to the **global** sample count and
  **survives segment restarts** (num_steps restored from checkpoint; the explicit
  fix over the old 5B chain). Peak 5.5e-5 / min 5.5e-6.
- AdEMAMix β1/β2/β3/α = 0.9/0.995/0.999/4; β3/α warmups in **iterations** (3218),
  β3 from β1, α from 0. Goldfish k=h=50 (2%) in-dataloader.
- Geometry reverted: rope 500000, seq/max_pos 4096, llama3 factor 8; and
  `--use-checkpoint-args` is **NOT** set, so the checkpoint's 12M/65536 cannot
  resurface. Vocab ÷256. Token arithmetic closes (3218 × 4.194M = 13.497B).

**Init checkpoints (empirical weight inspection):**
- Zero-diff verification is **meaningful** (two distinct HF dirs, per-group numeric
  compare, R17 enforced; pre-patch max-diff 173.78 → 0.0 proves non-vacuity).
- TD arm relearned **both** E (embed_tokens) **and** U (lm_head) on the new rows
  [131072:148480) — MSE-on-hiddens at layer 11 for E, CE for U — with originals
  byte-identical (frozen). Vanilla = clean geometry-reverted convert, vocab 131072,
  **no** embedding surgery. R17 (xIELU + QK-Norm restore) applied identically to
  both. TD [148480,4096] (÷256=580), vanilla [131072,4096] (÷256=512). TP2 rank
  files + `latest=release` present.

**Launch orchestration:**
- Each arm → correct init checkpoint + data prefix + tokenizer (vanilla→base,
  td→ext); init sizes differ as expected → no INIT_CKPT bleed. Export-hygiene
  aborts on all 11 guarded vars (`ALLOW_OVERRIDES=1` bypasses). Data preflight
  blocks launch while bulk binaries absent. Run stops **exactly** at iter 3218 with
  a final checkpoint. 14 benchmark checkpoints/arm on saved iters. The full gate,
  if run, FAILS today (bulk binaries missing) and the INIT_CKPT-bleed bug is fixed
  (clean child-shell `env -u …` resolution).
- All 7 orchestration scripts + 3 configs + 5 dataset_build scripts are
  byte-identical local↔Clariden. Megatron fork at pinned commit `c92402e`.

---

## Launch readiness checklist (gate before `DRY_RUN=0`)

1. [ ] **B1** — rsync the `megatron_patches/` subtree into the full-repo tree;
   confirm `pretrain_gpt_te_guard.py` resolves.
2. [ ] **M1** — fix per-set TB scalar key and/or `collect_metrics.py` parser;
   correct `EXTRA_VALID_README.md`; run the 50-iter `EVAL_INTERVAL=5` smoke to
   confirm 3 distinct per-set curves.
3. [ ] (recommended) MN1 — extend the export guard or launch from a sanitized
   shell; MN2 — run `gate_cpt2arm_artifacts.sh` as an explicit pre-launch step.
4. [ ] Stage A/B complete → `bulk_mix_{base,ext}_text_document.{bin,idx}` non-empty;
   the artifact gate passes (it correctly fails today).
