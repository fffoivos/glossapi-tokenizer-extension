# CPT Log

This is the one living log for the subproject. It records the decisions that
matter after cleanup; old reports are summarized in `ARCHIVE.md`.

## 2026-06-10

- Launched the full two-arm CPT run with `STAMP=20260610T200344Z` after the
  artifact gate passed on Clariden. Vanilla chain: `2516051-2516054`, watcher
  `2516055`; TD chain: `2516056-2516059`, watcher `2516060`. The first
  training segments requested 16 nodes each and initially entered `PENDING
  (Priority)` while both xfer watcher jobs started.
- Applied the schedule review without relaunching: updated the live Slurm jobs
  in place to 6h for segments 1-3 and 3h for final segments, then patched the
  submitter defaults. Vanilla segment 1 reached iteration 25, printed separate
  held-out losses for `hplt`, `openarchives`, and `greek_phd`, and ran at about
  8.6s/iter with no skipped or NaN iterations.
- Fixed the prelaunch blocker: the tree used by the launcher now has the
  `pretrain_gpt_te_guard.py` runtime wrapper needed for TE empty-extra-state
  checkpoint loading.
- Fixed per-set validation observability: Megatron now prints and logs separate
  held-out validation losses for `hplt`, `openarchives`, and `greek_phd`.
- Confirmed the dataset and init checkpoint gates: held-out ids are excluded,
  replay positions are preserved, new-Greek slots are ordered HPLT then
  OpenArchives, and both tokenizers/checkpoints pass the 256-alignment and R17
  checks.
- Diagnosed the multi-node `NET/OFI ... NO_SPACE` failure. The durable fix was
  to stop forcing `NCCL_NET_FORCE_FLUSH=1`; the trainer now defaults to
  `NCCL_NET_FORCE_FLUSH=0`.
- Validated launch-scale CXI at 16 nodes / 64 GPUs per arm with no-flush:
  mock-data smoke `2515665`, real-data timing `2515841`, held-out validation
  smoke `2515891`, and checkpoint-save smoke `2515966`.
- Updated the production shape: 16 nodes per arm, `torchrun`, 4 segments of
  952 iterations, 119-iteration checkpoint cadence, 25-iteration held-out
  validation cadence, expected 8.3-8.5h allocated runtime per arm.
- Cleaned the docs so the runbook is the operational source of truth, with this
  log plus `ARCHIVE.md` for provenance.

## 2026-06-09

- Built and validated the two-arm 13.5B dataset path: 10B new Greek, 70% HPLT
  then 30% OpenArchives/GlossAPI, plus 35% replay relative to new Greek.
- Preserved replay/code/math/Greek-replay positions while reordering only the
  new-Greek slots.
- Set Stage-A to HPLT confident-only E001 cleaning plus GreekMMLU
  `correct_only` decontamination.
- Set Stage-B anonymization after decontamination.
- Kept the full-run hyperparameter policy centered in
  `../CURRENT_HYPERPARAMETERS.md` and encoded in `configs/common_cpt.env`.

## Earlier Context

- The earlier 5B runs were diagnostics, not the full training recipe.
- The corrected hyperparameter regime is the central finding from those runs:
  AdEMAMix, Goldfish, 4096/500k geometry with RoPE scaling, 256-aligned extended
  tokenizer, full-run WSD schedule, and explicit replay.
- TD layer 11 remains the current TD-init choice for this two-arm run. Further
  TD-layer sweeps are separate experiments and should not change this launch
  unless deliberately re-scoped.
