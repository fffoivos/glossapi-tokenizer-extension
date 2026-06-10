# CPT Log

This is the one living log for the subproject. It records the decisions that
matter after cleanup; old reports are summarized in `ARCHIVE.md`.

## 2026-06-11

- Vanilla segment 1 completed cleanly as job `2516051` after `02:26:32`,
  saved `iter_0000952`, and exited at iteration 952. Segment 2 job `2516052`
  started from the dependency chain, loaded the `iter_0000952` checkpoint, and
  resumed training through iteration 981 with no skipped or NaN iterations.
  The resumed held-out validation line at iteration 975 still prints separate
  losses for `hplt`, `openarchives`, and `greek_phd`.
- TD segment 1 job `2516056` continued cleanly through iteration 842 and saved
  `iter_0000833`. It is still running at roughly 8.7s/iter with no skipped or
  NaN iterations; TD segment 2 remains pending on the dependency chain.
- The restarted TD watcher `2516378` submitted clean sidecars for iterations
  238 and 476 after the tokenizer-path fix. The TD converters `2516379` and
  `2516388` completed, and the TD retention jobs at both checkpoints completed
  after the `pytablewriter` repair.
- The `pytablewriter` repair is also validated on the vanilla side by
  retention job `2516432` for `iter_0000714`, which completed successfully.
  Earlier vanilla retention jobs for iterations 238 and 476 still have failed
  Slurm status because they hit the pre-repair table-rendering crash after
  writing timestamped results; decide later whether to rerun them for clean
  accounting or treat the result artifacts as usable.

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
- Committed and pushed the durable walltime defaults as `4924bb0`. A Clariden
  dry-run confirmed both arms now submit as four segments with 6h, 6h, 6h, and
  3h walltimes.
- Vanilla segment 1 saved `iter_0000119`, continued past iteration 188, and
  kept printing clean held-out validation losses. The sidecar manifests target
  benchmark checkpoints at iterations 238, 476, ..., 3218, so no sidecar launch
  is expected from the 119-iteration resume checkpoint.
- TD segment 1 started on 16 nodes, reached iteration 25 with separate held-out
  validation losses, and continued past iteration 37 with no skipped or NaN
  iterations.
- Vanilla reached benchmark checkpoint `iter_0000238`; the watcher submitted
  sidecars `2516200-2516207`. Conversion, checksum, and held-out Greek BPB
  completed; retention started; the native MCQ and Greek-NLP jobs failed fast
  on eval-launcher plumbing and were resubmitted as `2516230` and `2516231`
  after patching the shared eval sbatches.
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
- Reviewed the launch-scale compute audit against live logs. The training math
  is now validated in production shape: 16 nodes / 64 GPUs per arm, TP2/PP1,
  `mbs=2`, `NCCL_NET_FORCE_FLUSH=0`, and about 8.6-8.7s/iter with roughly
  390-395 TFLOP/s/GPU. No further comm/microbatch tuning is planned for this
  launch; the useful lever was the already-applied 6h/6h/6h/3h segment
  walltime schedule.
- Repaired TD checkpoint sidecar plumbing after the live watcher inherited the
  raw extended tokenizer directory. That directory is correct for
  training/corpus preprocessing but lacks `config.json`, so it is invalid for
  Megatron-to-HF conversion. `launch_all.sh` now passes the HF roundtrip
  skeleton at
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip`,
  and the TD sidecar submitter now fails early unless the path contains both
  `config.json` and `tokenizer.json`.
- Canceled the poisoned TD watcher `2516060` and dead TD sidecar dependency
  trees for iterations 238 and 476, cleared those two submitted markers, and
  restarted the TD watcher as `2516378`. It resubmitted clean converter jobs
  `2516379` and `2516388` with the HF roundtrip skeleton; both reached
  checkpoint loading instead of the previous path guard failure.
- Found a second eval-sidecar issue in retention jobs: vanilla retention jobs
  `2516204` and `2516284` completed the long lm-eval pass and wrote timestamped
  `results_*.json` plus sample logs, but exited nonzero during final
  pretty-table rendering because `pytablewriter` was present only as
  sourceless bytecode and the repair script had not exposed it. Patched
  `repair_lm_eval_cli_install.py` to expose the pytablewriter dependency stack
  and verified in the Clariden uenv that
  `from pytablewriter import LatexTableWriter` now succeeds.

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
