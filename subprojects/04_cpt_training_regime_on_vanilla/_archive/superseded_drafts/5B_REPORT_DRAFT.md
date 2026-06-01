# 5B Report Draft - Vanilla Apertus CPT Regime

Status: live draft. Do not use this as the final conclusion until every
completion gate in this file is satisfied from current artifacts.

Run tag: `04_vanilla_goldfish_5b_20260528T112539Z`

Goal and hyperparameter source docs:
`goal/goal.md` and `goal/hyperparameters.json`.

Training run dir:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z`

Eval root:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z`

Structured state snapshots:

| Artifact | Purpose |
| --- | --- |
| `goal/goal.md` | Human-readable scope, locked settings, required artifacts, and stop conditions. |
| `goal/hyperparameters.json` | Authoritative machine-readable settings for sbatch generation and config audits. |
| `scripts/collect_5b_report_state.py` | Home-side collector that reads Clariden through SSH and emits report-ready JSON. |
| `reports/latest_5b_report_state.json` | Latest collected snapshot of training health, queue state, checkpoint/eval presence, sidecar watcher state, and local adversarial-review artifacts. |
| `scripts/render_5b_report_status.py` | Converts the JSON snapshot into a compact Markdown status block. |
| `reports/latest_5b_report_status.md` | Latest rendered training/queue/checkpoint status for quick report updates. |
| `scripts/update_5b_report_status.sh` | One-command refresh wrapper for the JSON snapshot and Markdown status block. |
| `scripts/submit_checkpoint_sidecars.sh` | Per-checkpoint sidecar launcher; now also submits a CPU-only checksum manifest job after HF conversion. |
| `scripts/watch_and_submit_checkpoint_sidecars.sbatch` | Clariden `xfer` watcher that calls the sidecar launcher as checkpoints appear. |
| `scripts/verify_checkpoint_sidecars.py` | Per-iteration verifier for checkpoint metadata, sidecar manifest, expected sidecar job kinds, Slurm status, and local review artifacts. |
| `scripts/watch_checkpoint_sidecar_verification.sh` | Home-side watcher that reruns the verifier for planned checkpoints and writes pass snapshots only after outputs, Slurm completion, and checksum are ready. |
| `scripts/watch_and_run_adversarial_reviews.sh` | Home-side watcher that launches local adversarial reviews only after the hardened verifier reports `handoff_ready=true`. |
| `scripts/write_checkpoint_checksum_manifest.py` | Streams SHA256 manifests for Megatron checkpoint shards and converted HF checkpoint files. |
| `reports/config_geometry_audit_iter_0000119.md` | Current positional/RoPE geometry audit for iter 119 and final-report comparison caveat. |
| `reports/iter_0000119_checkpoint_sidecar_precheck.json` | Pre-check snapshot for the first 0.5B checkpoint before checkpoint arrival. |
| `reports/iter_*_checkpoint_sidecar_verify_latest.json` | Latest handoff-verifier snapshots for each planned checkpoint; expected to remain incomplete until the corresponding checkpoint lands. |

## Question

Does the corrected Apertus-style CPT regime improve Vanilla Apertus-8B against
the previous bakeoff Vanilla trajectory, without introducing the tokenizer
extension confound?

Primary comparison targets:

| Model / checkpoint | Native MCQ aggregate | Native MCQ incl. Plutus | Source |
| --- | ---: | ---: | --- |
| Apertus-Base | 0.4817 | 0.4902 | `cpt-plan.md` section 1.1 |
| Prior bakeoff Vanilla-2B | 0.4327 | 0.4256 | `cpt-plan.md` section 1.1 |
| Prior bakeoff Vanilla-3.5B | 0.4370 | 0.4333 | `cpt-plan.md` section 1.1 |
| Prior bakeoff Vanilla-5B | 0.4305 | 0.4329 | `cpt-plan.md` section 1.1 |
| This run, 0.5B | 0.4391 | 0.4293 | `iter_0000119/native_mcq/*_headline.json`; Plutus diagnostic separated in `*_diagnostics.json` |
| This run, 1B | TBD | TBD | sidecar evals |
| This run, 2B | TBD | TBD | sidecar evals |
| This run, 3.5B | TBD | TBD | sidecar evals |
| This run, 5B | TBD | TBD | sidecar evals |

Conclusion placeholder: TBD after the 5B checkpoint, sidecar evals, and
adversarial reviews are complete.

## Fixed Scope Verification

| Requirement | Evidence |
| --- | --- |
| Base model/tokenizer only | `goal/goal.md`; tokenizer in `run_metadata.json` is `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509`; dataset validation token scan has `max_id=131071`. |
| HPLT-only Greek data | dataset validation `greek_sources=["greek_hplt_clean60"]`. |
| Mix 70/24/4/2 | validation JSON actual weights: Greek `0.7000002012`, replay `0.2399995782`, code `0.0400000390`, math `0.0200001816`. |
| No tokenizer extension | `goal/hyperparameters.json`; base vocab 131072; no extension arm. |
| Goldfish loss | run metadata `loss_objective=goldfish`; goal lock `k=50`, `h=50`, hash seed `2971215073`. |
| AdEMAMix settings | run metadata: beta1 `0.9`, beta2 `0.999`, beta3 `0.99`, alpha `8.0`; warmups `287`. |
| CPU-only dataset work | dataset build ran under `xfer`; see `RUN_LOG_20260528.md`. |
| Sidecar eval pattern | watcher job `2417454` runs on `xfer`; sidecars wait for checkpoints; local tmux `cpt_sidecar_verify` records handoff verification. |

## Dataset Artifacts

Dataset run dir:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_hplt_b1_dataset_5b_20260528T112539Z`

Megatron data prefix:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_hplt_b1_dataset_5b_20260528T112539Z/megatron/hplt_b1_base_text_document`

Key artifacts:

| Artifact | Path / value |
| --- | --- |
| Dataset paths env | `.../dataset_paths.env` |
| JSONL | `.../jsonl/hplt_b1_5b.jsonl` |
| JSONL manifest | `.../jsonl/hplt_b1_5b.manifest.json` |
| Recipe | `.../recipe/hplt_b1_vanilla_regime.json` |
| Megatron bin/idx | `.../megatron/hplt_b1_base_text_document.{bin,idx}` |
| Megatron manifest | `.../megatron/hplt_b1_base_text_document.manifest.json` |
| Validation JSON | `.../validation/dataset_validation.json` |
| JSONL tokens / rows | `5,000,000,250` tokens / `3,739,911` rows |
| Megatron tokens / documents | `5,007,480,072` tokens / `3,739,912` documents |

Bucket totals from the JSONL manifest:

| Bucket | Tokens | Rows |
| --- | ---: | ---: |
| Greek | 3,500,001,181 | 2,422,129 |
| Replay | 1,199,997,951 | 1,184,199 |
| Code | 200,000,205 | 76,040 |
| Math | 100,000,913 | 57,543 |

Heldouts:

| Heldout | Path |
| --- | --- |
| Greek | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl` |
| Code | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_code_heldout_200_20260528.jsonl` |
| Math | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_math_heldout_200_20260528.jsonl` |

## Training Submission

State dir:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z_submit_state`

Exact command log:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z_submit_state/training_sbatch_commands.sh`

Training chain:
`/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z_submit_state/training_chain.tsv`

Initial checkpoint:
`/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched`

Megatron commit from `run_metadata.json`:
`c92402e39ef3c8e69ea378a59e79059dc14541f4`

Segment chain:

| Segment | Target iter | Target tokens | Job ID | Dependency | Status |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 300 | 1,258,291,200 | 2417446 | none | running as of 2026-05-28T21:17Z |
| 2 | 477 | 2,000,683,008 | 2417447 | afterok:2417446 | pending |
| 3 | 596 | 2,499,805,184 | 2417448 | afterok:2417447 | pending |
| 4 | 715 | 2,998,927,360 | 2417449 | afterok:2417448 | pending |
| 5 | 834 | 3,498,049,536 | 2417450 | afterok:2417449 | pending |
| 6 | 953 | 3,997,171,712 | 2417451 | afterok:2417450 | pending |
| 7 | 1072 | 4,496,293,888 | 2417452 | afterok:2417451 | pending |
| 8 | 1192 | 4,999,610,368 | 2417453 | afterok:2417452 | pending |

Sidecar watcher:

| Job ID | Partition | Role |
| ---: | --- | --- |
| 2417454 | `xfer` | checkpoint watcher and eval sidecar submitter; running as of 2026-05-28T21:14Z |

Preserved failed/cancelled attempts:

| Job ID | Outcome | Reason | Preserved path |
| ---: | --- | --- | --- |
| 2417278 | failed | warmup assert, target iter below 287 warmup steps | `failed_attempt_2417278_warmup_assert_20260528T144523Z` |
| 2417297 | cancelled | segment target 357 was too long for 12h wall time | `canceled_attempt_2417297_segment357_too_long_20260528T145517Z` |

## Training Health

Latest verified live line before this draft:

| UTC poll | Iter | Consumed tokens | Loss | Skipped | NaN | Tokens/sec/GPU | Next-checkpoint ETA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-28T21:17:21Z | 174/300 | 0.730B | 1.898097 | 0 | 0 | 8001.7 | 2026-05-28T23:37:08Z for iter 238 |

Stop-condition checks so far:

| Stop condition | Current evidence |
| --- | --- |
| Dataset validation failure | validation JSON `ok: true`. |
| Model load failure | current `2417446` entered training and has reached iter 174. |
| NaNs | none observed in latest parsed log scan. |
| Repeated skipped iterations | none observed in latest parsed log scan. |
| OOM / severe errors | none observed in latest parsed log scan. |
| Checkpoint save failure | iter 119 checkpoint exists with `.metadata`; later checkpoints pending. |
| Corrupted conversion | iter 119 HF conversion job `2419108` completed `0:0`; later checkpoints pending. |
| Broken eval loading | iter 119 sidecars completed `0:0`; later checkpoints pending. |
| Accidental GPU use for CPU dataset work | dataset/heldout work used `xfer`; no GPU dataset job recorded in the current run log. |

## Checkpoints And Sidecars

Planned checkpoints:

| Label | Iter | Token mark | Megatron checkpoint | HF checkpoint | Sidecar manifest | Checksum | Review |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| Vanilla-0.5B | 119 | 499,122,176 | present | present | complete | present and sampled-verified | present |
| Vanilla-1B | 238 | 998,244,352 | pending | pending | pending | pending | pending |
| Vanilla-2B | 477 | 2,000,683,008 | pending | pending | pending | pending | pending |
| Vanilla-3.5B | 834 | 3,498,049,536 | pending | pending | pending | pending | pending |
| Vanilla-5B | 1192 | 4,999,610,368 | pending | pending | pending | pending | pending |

Expected sidecar jobs per checkpoint:

| Eval family | Required output |
| --- | --- |
| HF conversion | converted HF checkpoint directory |
| Native Greek MCQ | GreekMMLU, ILSP Medical MCQA, ILSP ASEP |
| Greek diagnostics | Plutus QA, greek-nlp/benchmark sample, heldout Greek BPB |
| Retention | English, French, German, Russian |
| Code/math | heldout BPB or loss outputs |
| Adversarial review | `prompt.md`, `codex_events.jsonl`, `review_metadata.env`, `adversarial_critique.md` |

### Iter 119 Completed Evidence

Sidecar jobs for `Vanilla-0.5B`:

| Kind | Job ID | Status | Output |
| --- | ---: | --- | --- |
| HF conversion | 2419108 | completed `0:0` | `iter_0000119_hf` |
| Native Greek MCQ | 2419109 | completed `0:0` | `iter_0000119/native_mcq` |
| Greek NLP diagnostic | 2419110 | completed `0:0` | `iter_0000119/greek_nlp_s100` |
| Heldout Greek BPB | 2419111 | completed `0:0` | `iter_0000119/heldout_greek_bpb.json` |
| Retention | 2419112 | completed `0:0` | `iter_0000119/retention` |
| Code BPB | 2419113 | completed `0:0` | `iter_0000119/heldout_code_bpb.json` |
| Math BPB | 2419114 | completed `0:0` | `iter_0000119/heldout_math_bpb.json` |
| Checksum manifest | 2419684 | completed `0:0` | `iter_0000119/checksums/Vanilla-0.5B_iter_0000119_checksum_manifest.json` |

Native Greek MCQ:

| Benchmark | n | Accuracy |
| --- | ---: | ---: |
| GreekMMLU | 16632 | 0.4984968735 |
| ILSP Medical MCQA | 432 | 0.3379629630 |
| ILSP ASEP | 1200 | 0.4808333333 |
| Plutus QA, diagnostic only | 225 | 0.4000000000 |

Headline macro is `0.4390977233` over GreekMMLU, ILSP Medical MCQA, and ILSP
ASEP. Headline plus diagnostic Plutus macro is `0.4293232924`; do not use that
as the headline.

Heldout BPB:

| Heldout | BPB | Docs | Truncated docs | Truncation fraction | STRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greek | 0.6049364592 | 500 | 146 | 0.292 | 0.2702694597 |
| Code | 0.4177850318 | 200 | 23 | 0.115 | 0.6963093083 |
| Math | 0.7270752513 | 200 | 15 | 0.075 | 0.7285759714 |

Retention highlights from `results_2026-05-28T21-49-44.233218.json`:

| Task | Accuracy |
| --- | ---: |
| Global MMLU EN | 0.605 |
| Global MMLU FR | 0.535 |
| Global MMLU DE | 0.525 |
| XNLI EN | 0.5188755020 |
| XNLI FR | 0.4546184739 |
| XNLI DE | 0.4855421687 |
| XNLI RU | 0.4819277108 |

Integrity notes:

- Checksum manifest schema `04-vanilla-checkpoint-checksum-manifest-v1` was
  generated at `2026-05-28T21:09:20Z` and lists 11 Megatron files plus 11 HF
  files, totaling `161089256625` bytes across the two checkpoint forms.
- Independent verification found no missing files, no bad sizes, and matching
  first/last sampled SHA256 hashes for both Megatron and HF sections.
- Local adversarial review exists at
  `adversarial_reviews/Vanilla-0.5B/adversarial_critique.md`. Its critical
  remaining issue is the positional/RoPE geometry confound; the MCQ headline
  split, sidecar attempt history, and checksum manifest gaps have since been
  addressed.

## Interpretation Plan

Use the native Greek MCQ aggregate excluding MT-derived Greek tasks as the
headline. Plutus remains diagnostic. The final 5B report must compare this run
to Apertus-Base and prior bakeoff Vanilla at matched token marks where possible.
Because `reports/config_geometry_audit_iter_0000119.md` shows this run uses
4096/RoPE-500K geometry while the local official HF config is 65536/RoPE-12M
with scaling, final comparisons must use a matched 4096/RoPE-500K baseline or
explicitly label geometry as a confound.

Decision skeleton:

| Observation | Interpretation |
| --- | --- |
| New Vanilla improves over prior bakeoff Vanilla and approaches Apertus-Base | corrected regime likely addressed the bakeoff CPT-regime failure. |
| BPB improves early but native MCQ stays below Apertus-Base | language modeling improves, but benchmark-relevant behavior or calibration may be degraded. |
| New Vanilla tracks prior bakeoff Vanilla or degrades | regime change did not fix the Vanilla drift; investigate data mix or intrinsic CPT interference. |
| Adversarial review finds critical artifact/eval flaws | do not draw conclusions until fixed or explicitly bounded. |

## Completion Gates

Do not call this report final until all gates are complete and cited with paths:

- All planned training segments completed or a documented stop condition halted
  the run.
- Checkpoints exist and have metadata at iters 119, 238, 477, 834, and 1192.
- HF conversions exist for each required checkpoint.
- Native Greek MCQ, Plutus, greek-nlp sample, heldout Greek BPB, retention, and
  code/math outputs exist for each checkpoint.
- Adversarial review artifacts exist for each checkpoint.
- Any critical adversarial findings are addressed or explicitly bounded.
- `sacct` status and final job IDs are recorded.
- Throughput/loss health and restart history are summarized from logs.
- Final comparison tables are filled from actual result files, not copied from
  transient console output.
- A clear 5B decision is written: stop, continue to 7B/10B, or rerun/fix.
