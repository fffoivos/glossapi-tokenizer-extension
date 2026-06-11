# Historical Archive

This archive replaces the many overlapping launch notes that existed before the
2026-06-10 cleanup. Use `RUNBOOK.md` for operations and `LOG.md` for current
chronology.

## Superseded Documents

The following docs were removed after their actionable content was folded into
the runbook/log/archive:

- `HANDOFF.md`
- `PLAN.md`
- `README.md`
- `RUN_LOG_20260609_CPT_2ARM.md`
- `REVIEW_20260610_CPT_2ARM_PRELAUNCH.md`
- `STATUS_CONFORMANCE_20260610.md`
- `CXI_NOSPACE_DEEP_DIVE_20260610.md`
- `apertus_pretrain_checkpoints_notes.md`
- `reports/CLARIDEN_MEGATRON_NCCL_NO_SPACE_20260610.md`
- `reports/CPT_16NODE_CXI_TIMING_20260610.md`
- `reports/README.md`
- `03_training_experiments/BUILD_PLAN.md`
- `03_training_experiments/HANDOFF.md`
- `03_training_experiments/LAUNCH_RUNBOOK.md`
- `03_training_experiments/README.md`
- `03_training_experiments/TOOLING_DECISIONS.md`
- `03_training_experiments/dataset_build/HANDOFF.md`
- `03_training_experiments/dataset_build/EXTRA_VALID_README.md`
- `03_training_experiments/docs/SCHEDULER_MATH.md`
- `03_training_experiments/docs/TOKEN_DISTILLATION_E_AND_U.md`

The content remains recoverable from git history. The current operational
summary is below.

## Prelaunch Review Summary

The review agent found one blocker and one major issue before launch:

- Blocker: the trainer path lacked `megatron_patches/runtime/pretrain_gpt_te_guard.py`.
  This would have crashed both arms before training and was fixed.
- Major: per-set held-out validation losses were not separable in TensorBoard
  and were not parsed by metrics collection. This was fixed by prefixing the
  validation scalar/print path and updating parsing.

Positive checks from the review:

- No sampled holdout leakage.
- 70/30 HPLT/OpenArchives split.
- Important post-run caveat: the dataset artifact was physically ordered, but
  Megatron randomized training sample consumption. The completed 13.5B run
  should not be described as an executed HPLT-to-OpenArchives curriculum.
- Greek replay covered by decontamination and the full stream covered by
  anonymization.
- Warmup, WSD cooldown, AdEMAMix, Goldfish, geometry, and vocab divisibility
  matched the written policy.
- Vanilla and TD init checkpoints were meaningful zero-diff/R17-checked
  conversions, not accidental stale checkpoints.

## Apertus Checkpoint Notes

The selected base is `swiss-ai/Apertus-8B-2509` main, not the older pre-decay
branch. `main` carries long-context geometry, so conversion reverts
`rope_theta` to 500000 and context length to 4096 while preserving Apertus
main-pretraining RoPE scaling. The declined alternative was starting from
`step2400000-tokens13112B`, the last full-peak-LR pre-cooldown checkpoint.

The practical rule for this launch is simple: load `main`, revert only the
long-context geometry fields, preserve `rope_scaling`, convert to Megatron
TP=2/PP=1, patch R17, and verify logits/roundtrip.

## CXI Failure Summary

Initial multi-node Megatron runs failed before iteration 1 with
`NET/OFI ... NO_SPACE` in the first inter-node data-parallel collective.
Single-node training worked, and pure PyTorch/NCCL inter-node controls worked.

Several mitigations were explored: request-buffer sizing, hybrid/software CXI
matching, Socket fallback, and smaller node counts. The decisive finding was
that the training wrapper forced `NCCL_NET_FORCE_FLUSH=1`. With force-flush
disabled (`NCCL_NET_FORCE_FLUSH=0`), 2-node, 4-node, and 16-node CXI Megatron
smokes passed. Socket remains fallback only.

## 16-Node Timing Evidence

- `2515665`: 16-node / 64-GPU mock-data CXI no-flush smoke passed in 1m48s.
- `2515841`: 16-node real-data timing passed in 2m56s. Iteration 1 was about
  15.45s; iterations 2-10 averaged about 8.63s.
- `2515891`: 16-node per-set held-out validation smoke passed in 2m57s. The
  three held-out validation sets printed separately, with about 11s overhead.
- `2515966`: 16-node checkpoint-save smoke passed in 2m05s. Save overhead was
  about 22s and the smoke checkpoint directory was about 136G.

These measurements produce the 8.3-8.5h per-arm allocated-runtime estimate in
`RUNBOOK.md`.

## Dataset Build Summary

The production dataset is a full 13.5B-token run:

- 10B new Greek.
- New Greek = 70% HPLT and 30% OpenArchives/GlossAPI.
- Replay = 24% multilingual, 4% code, 2% math, 5% Greek replay, all measured
  relative to new Greek.
- Three 0.5B held-out validation sets are excluded from training.
- Stage-C preserves replay positions and orders only new-Greek slots in the
  physical artifact. This does not imply sequential curriculum consumption by
  Megatron.
- Stage-A applies HPLT E001 cleaning and GreekMMLU `correct_only`
  decontamination.
- Stage-B anonymizes after decontamination.

## Corpus-Prep Method Summary

The corpus-prep markdown tree was also collapsed during cleanup. The retained
method decisions are:

- Ordered pipeline: clean, dedup-validate, decontaminate, anonymize, shard.
- CPU-only for corpus work. Do not request GPU/GRES for cleaning,
  decontamination, anonymization, or tokenizer preprocessing.
- HPLT cleaning posture: only exact observable, high-confidence artifacts are
  eligible for automatic transformation. Avoid semantic rules. Source text is
  immutable; any cleaning must happen through derived/shadow outputs.
- HPLT cleaning active production posture for this run: only the confident E001
  replacement-character/control residue cleanup is in the launch path. Broader
  HPLT cleaning categories were explored but not approved as destructive
  production overlays.
- Decontamination scope for this launch: GreekMMLU only, rule `correct_only`,
  with the DCLM-style scanner parameters encoded in
  `03_training_experiments/dataset_build/stageA_clean_decontam.sbatch`.
- Anonymization runs after decontamination. Mask email, IP, and IBAN to reserved
  single tokens; keep the Apertus parity email/IP logic and the Greek-IBAN fix.
- Dedup work for this launch is validation/characterization of the existing
  selected corpus artifact, not a fresh full dedup derivation.
- Old generated review packs and policy snapshots were useful exploration
  artifacts, but they are not launch instructions. Their operational conclusions
  are represented here and in `RUNBOOK.md`.

## TD Initialization Summary

The TD arm uses the 148,480-vocab tokenizer, adding 17,408 modern-Greek rows.
Layer 11 is the current initialization choice. Input embeddings are trained
with hidden-state MSE distillation and output/lm-head rows with CE; original
rows are frozen during initialization. The full CPT run then trains both arms
with the same data stream and same hyperparameters.
