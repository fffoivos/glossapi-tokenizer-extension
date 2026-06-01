**1. Verdict**

Not trustworthy enough to use, even as a comparative checkpoint, until two new problems unique to this checkpoint are fixed. The iter-238 (`Vanilla-1B`) checkpoint itself is mechanically sound — Megatron `iter_0000238` is intact, `iter_0000238_hf` converted cleanly, the checksum manifest (`Vanilla-1B_iter_0000238_checksum_manifest.json`) covers 11 Megatron + 11 HF files, training is healthy (iter 237/238 `lm loss=1.594/1.608`, `skipped=0`, `nan=0`, `grad_norm=0.5`, `params_norm=7092.6`), and all seven mandatory sidecars (`2420146`–`2420152`) plus checksum (`2420153`) completed with exit `0:0`. But the headline native-Greek MCQ output is *quantitatively wrong by construction* (only GreekMMLU ran; ILSP Medical, ILSP ASEP, and Plutus QA were silently dropped), so the only headline number you can compare against bakeoff/Apertus-Base is a single-task slice of the intended 3-task aggregate. Combined with the persisting RoPE/seqlen geometry mismatch, persisting Greek-MCQ decontamination gap, and persisting ~29% Greek-BPB prefix truncation, no MCQ-based delta or "Vanilla-CPT improves over bakeoff Vanilla" claim should be drawn from this checkpoint yet. Also, the prompt's framing of `Vanilla-1B = 998,244,352 tokens at iter 238` matches `consumed tokens: 0.998B` at iter 238 in `04van5b_i300-2417446.out:1230`, but at that point LR is still `9.336e-6` (target `1.1e-5`, warmup ends at iter 287) — this is a still-in-warmup probe, not a stable-LR measurement.

**2. Critical Findings**

1. **Native Greek MCQ sidecar dropped 3 of 4 benchmarks at iter 238 — silently.**

   Submitter log `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z_sidecar_watch/iter_238_submit.log` shows `NATIVE_BENCHMARKS: greekmmlu,ilsp_medical_mcqa,ilsp_mcqa_asep,plutus_qa`. But `iter_0000238/native_mcq/run_metadata.json` records `benchmarks: ["greekmmlu"]`, and the job stdout (`04native_i238-2420147.out`) starts with `BENCHMARKS: greekmmlu`.

   Root cause: in `scripts/submit_checkpoint_sidecars.sh` L156 the submitter passes a comma-separated list inside Slurm's `--export=ALL,KEY=VAL,KEY=VAL,...` form: `--export=ALL,MODEL_SPEC=...,OUTPUT_DIR=...,BENCHMARKS="$NATIVE_BENCHMARKS",SAMPLE_SIZE=...`. Slurm splits `--export` on every top-level comma, so `BENCHMARKS=greekmmlu` is captured, and `ilsp_medical_mcqa`, `ilsp_mcqa_asep`, `plutus_qa,SAMPLE_SIZE=...` are interpreted as further export entries — silently truncating the benchmark list to GreekMMLU only.

   Why iter 119 looked OK: the iter-119 native MCQ job was *manually resubmitted* with `BENCHMARKS=all` after the original `2419080`-`2419087` chain was killed by the missing-config bug; the watcher's broken `--export` path was never exercised on iter 119. So the iter-119 MCQ artifact (`benchmarks: greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep, plutus_qa`) is correct and the iter-238 artifact is the first watcher-driven run — and the bug shipped to iter 238.

   Consequence: the only iter-238 "headline" value is GreekMMLU `0.5026` (n=16632). The plan's defined headline is `mean(GreekMMLU, ILSP Medical, ILSP ASEP)`, which cannot be computed from this run, and the diagnostic (Plutus) is missing. `Vanilla-1B_native_mcq_aggregate.json` declares the headline_policy (`headline_benchmarks=[greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep]`, `diagnostic_benchmarks=[plutus_qa]`) but then computes `headline.macro_accuracy=0.5026` over `n_tasks=1`. This is a misleading artifact: it reads like a real headline but is a single-task value silently masquerading as a 3-task mean. Any cross-checkpoint comparison using these JSONs will compare iter-119 3-task (`0.43910`) against iter-238 1-task (`0.5026`) and falsely register an enormous "gain". This must be flagged before any iter-238 number enters the 5B report.

2. **RoPE/seqlen mismatch persists at iter 238; the converted HF checkpoint is *not* drop-in for Apertus-Base evaluation.**

   `iter_0000238_hf/config.json`:
   ```
   max_position_embeddings: 4096
   rope_theta: 500000
   rope_scaling: null
   vocab_size: 131072
   ```
   versus the official base model under `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json` (`max_position_embeddings=65536`, `rope_theta=12000000`, llama3 RoPE scaling). The Vanilla-0.5B critique surfaced this as Critical-1; nothing has changed at iter 238. The live training command (in `04_vanilla_goldfish_5b_20260528T112539Z/training_command.sh`) is still `--seq-length 4096 --max-position-embeddings 4096 --rotary-base 500000`. Any comparison of `Vanilla-1B` against Apertus-Base or against bakeoff Vanilla at matched tokens is a comparison of two different positional geometries unless the base/bakeoff are re-evaluated under the 4096/500K geometry. The `reports/config_geometry_audit_iter_0000119.{json,md}` documents the issue but the run continues to train under it. This is the canonical Vanilla-0.5B Critical-1 still live.

3. **No decontamination evidence for any public Greek MCQ benchmark.**

   Searches under `/capstor/scratch/cscs/fffoivos` and `/iopsstor/scratch/cscs/fffoivos/cpt_corpus` for filenames matching `*decont*` or `*contam*` returned nothing. The HPLT clean60 source (2.42M docs, 3.5B Greek tokens) has had Apertus-pretrain dedup overlay (`audit_id=20260519T010924Z`, 2,099,756 dropped docs) — but that addresses Apertus *pretraining* overlap, not overlap with GreekMMLU / ILSP Medical / ILSP ASEP / Plutus prompts. Goal `production_blockers_status.V1_decontamination_against_native_bench_prompts.status="not_required_for_diagnostic"` documents the choice but the choice still invalidates any sharp MCQ-improvement claim. Persistence of Vanilla-0.5B Critical-3.

**3. Major Findings**

1. **Greek BPB heldout is heavily prefix-truncated at exactly the same rate as iter 119.**

   `iter_0000238/heldout_greek_bpb.json`: `n_docs=500`, `n_docs_truncated=146`, `fraction_truncated=0.292`, `n_tokens_dropped=6,906,358`. Iter 119 was `0.292 / 146 / 6,906,358` — identical (same heldout set, same `max_context=4096`). The same metric on the same set means iter 119 → 238 deltas are comparable, but the *absolute* BPB and any cross-arm comparison against a model evaluated at a longer context window remain biased by ~29% prefix-only scoring. The header note in the JSON warns "Look at this if it's > ~10 % of docs" — at 29.2% it cannot be ignored. Persistence of Vanilla-0.5B Major-1.

2. **Multilingual retention shows a small but non-zero regression on MMLU and global_mmlu_en between iter 119 and iter 238.**

   Iter 119 retention vs iter 238 retention:

   | Task | iter 119 | iter 238 | Δ |
   |---|---:|---:|---:|
   | mmlu | 0.5674 | 0.5511 | −1.63 pp |
   | global_mmlu_en | 0.605 | 0.600 | −0.5 pp |
   | global_mmlu_de | 0.525 | 0.540 | +1.5 pp |
   | global_mmlu_fr | 0.535 | 0.525 | −1.0 pp |
   | arc_challenge | 0.5085 | 0.5290 | +2.1 pp |
   | arc_easy | 0.811 | 0.823 | +1.2 pp |
   | piqa | 0.786 | 0.795 | +1.0 pp |
   | hellaswag | 0.5711 | 0.5783 | +0.7 pp |
   | xnli_en | 0.5189 | 0.5120 | −0.7 pp |
   | xnli_fr | 0.4546 | 0.4944 | +4.0 pp |

   Net: ARC/Hellaswag/PIQA improving (training fluency); MMLU clearly down (−1.6 pp). global_mmlu_ru is `None` in both runs — the retention bundle does not actually score Russian global_mmlu, despite the plan listing `ru` as one of the four retention languages. xstorycloze_en/fr both `None` as well. So "EN/FR/DE/RU retention" is only partially measured; the run did *not* match the goal's per-language coverage.

3. **The retention sidecar runs 201 tasks (`global_mmlu_*` across many non-target languages, including ar/bn/hi/id/sw/te/yo/zh, plus xnli for multiple langs)** and writes 180 sample JSONLs — far more than the EN/FR/DE/RU + code/math focus the goal calls for. This is OK as raw data, but no per-checkpoint summary restricts to the four languages, so comparisons by reader will use whichever subset they reach for first.

4. **LR is still in warmup; this checkpoint is not the stable-regime evidence the 1B label might imply.**

   Iter 238 LR = `9.335537e-06`; target = `1.1e-5`; warmup ends at iter 287 (`lr_warmup_samples=292968`, `global_batch_samples=1024` ⇒ 286.1 → ceil 287). So `Vanilla-1B` is roughly 83% of warmup complete, ~84.9% of the way to peak LR. Drawing "the regime is working" or "Vanilla CPT improves" conclusions from this checkpoint mixes warmup dynamics with the steady regime. The 5B report should explicitly mark `Vanilla-1B` as a *late-warmup* point, not a stable measurement.

5. **`run_metadata.json` reports `lr_decay_style: 1-sqrt` while the actual command uses `--lr-decay-style constant`.**

   Top-level `run_metadata.json` line `"lr_decay_style": "1-sqrt"` contradicts the `training_command.sh` (`--lr-decay-style constant`). This is the same inconsistency the Vanilla-0.5B critique flagged; the metadata still carries a stale field that does not match the actual training. Anyone reading metadata-only will draw the wrong conclusion about post-warmup behaviour.

6. **Iter-238 native-MCQ sidecar `04native_i238` allocated `gres/gpu=4` for a single-GPU eval.**

   `sacct` for `2420147`: `AllocTRES=billing=288,cpu=288,energy=1032296,gres/gpu=4,mem=220G,node=1`, elapsed 12m42s. The sbatch script requests `--gpus-per-node=1` and `--cpus-per-task=18`, but Slurm at Clariden gave a whole GH200 node anyway because the partition is `normal`. Every iter-238 eval (`convert`, `native_mcq`, `greek_nlp`, `bpb`, `retention`, `code_bpb`, `math_bpb`) sits on 4 GPUs for whatever wallclock it takes (the 1h01m `04gnlp_i238` job consumed `energy=3.75M` units on 4 GPUs). Total ~2h00m of 4-GPU node billing for one checkpoint's sidecars. Hygiene problem the 0.5B critique already flagged; unchanged.

7. **The dataset token scan covers id `0` but the smoke validation said `[1, 131070]`.**

   `validation/dataset_validation.json` reports `token_scan.min_id=0, max_id=131071`. The smoke run log claimed `[1, 131070]`. Apertus tokenizer's `pad_token_id=3, bos_token_id=1, eos_token_id=2`; id `0` is reserved (likely `<unk>`). It is probably the EOD/preprocess marker (`append_eod=true`) injected by Megatron's `preprocess_data.py`. Worth confirming so we can rule out `<unk>`-spill corrupting Greek text. Not flagged as a critical risk yet, but the discrepancy with the smoke validation deserves an explicit note in the 5B report.

**4. Minor Findings and Hygiene Notes**

- `Vanilla-1B_native_mcq_diagnostics.json` is `[]` (empty array). It should at minimum carry Plutus QA — but Plutus was silently dropped by Finding C1. The "no diagnostic" emptiness is itself a symptom of the bug.
- The sidecar manifest's `expected_kinds` does **not** include `checksum`. The verifier extends the row to `kinds=[..., checksum]` but the verifier check (`manifest_has_all_expected_kinds`) does not gate on it; checksum acts as an out-of-band column. This is fine today, but it means a missing checksum job would still produce `handoff_ready=true`. Confirmed by reading `expected_kinds` in `iter_0000238_checkpoint_sidecar_verify_latest.json`.
- The training command still has `--main-grads-dtype fp32` (correct per plan), `--cross-entropy-loss-fusion`, `--ckpt-format torch_dist`, `--ckpt-fully-parallel-load`, `--dist-ckpt-strictness assume_ok_unexpected`, `--no-load-optim --no-load-rng`. All consistent with the plan; nothing surprising.
- The 5B chain segment that produced this checkpoint (`2417446 04van5b_i300`) is still running on `normal` with `gres/gpu=4`. Next checkpoint (iter 477 / `Vanilla-2B`) appeared shortly after iter 238; the same broken `NATIVE_BENCHMARKS` parsing will recur in iter 477 unless the submitter is fixed before then.
- Greek-NLP sidecar `04gnlp_i238` took 1h01m (4-GPU node), and its outputs (24 child files) are likely still in the same base-LM "prompt echoing / 0.0 F1" failure mode flagged at iter 119; this critique did not re-inspect them but the Vanilla-0.5B critique's "greek-nlp not interpretable as capability evidence" probably persists. Confirm before any Greek-NLP number enters the 5B report.
- The native MCQ runner's argparse default is `--benchmarks all` (`run_native_greek_mcq_eval.py:36`); a hardened iter-119 path would have been to keep `BENCHMARKS=all` from the watcher rather than re-listing them comma-separated through `--export`. As a defensive simple fix, the watcher should either pass `BENCHMARKS=all` (then enforce headline policy inside the runner, which the runner already does via `HEADLINE_MCQ_BENCHMARKS`) or quote the export properly (`--export=ALL,BENCHMARKS="$NATIVE_BENCHMARKS"` after `set` substitution does not help because Slurm still parses commas).

**5. Missing Evidence**

To trust this checkpoint, the following are needed and currently absent:

- **A re-run of iter 238 native MCQ with all four benchmarks.** Reuses the existing HF dir under `iter_0000238_hf`, no retraining. Without this, no headline MCQ value exists for iter 238.
- **A matched-config baseline.** Either Apertus-Base re-evaluated with `max_position_embeddings=4096`, `rope_theta=500000`, identical tokenizer + sampling, or the run retrained under the official `65536 / 12000000` long-context geometry. Until then the bakeoff Vanilla numbers (themselves trained under the bakeoff-recipe `4096/RoPE-500K` regime per `_train_config_common.env` — so they *do* match this run's geometry) are the only fair anchor, and Apertus-Base in the 5B report must be re-evaluated under the same geometry.
- **Decontamination report for GreekMMLU / ILSP Medical / ILSP ASEP / Plutus against the final HPLT+B1 JSONL** (`jsonl/hplt_b1_5b.jsonl`, 3.74M rows). Even a simple n-gram or paraphrased-prompt scan is missing.
- **A `Vanilla-1B vs Vanilla-0.5B` per-task retention diff and a `Vanilla-1B vs Apertus-Base (same geometry)` table**, with EN/FR/DE/RU each represented. Currently `global_mmlu_ru` and `xstorycloze_*` are missing from results.
- **An explicit annotation that iter 238 is mid-warmup (`lr ≈ 0.849 × peak`)** in any report that uses this checkpoint as evidence.
- **A note on the `min_id=0` token in the dataset scan** and what byte/text it corresponds to in the Apertus tokenizer.
- **Bootstrap CIs on GreekMMLU at this n** so that a 0.5026 vs 0.4984 (iter 119) "improvement" (~+0.4 pp) is read against the binomial noise band for n=16,632 (≈±0.39 pp at 1σ).
- **A statement of whether the bakeoff Vanilla numbers in the cpt-plan's table** (`Vanilla-2B=0.4327` MCQ general) were produced at the same `max_input_tokens=3072` MCQ context the iter-238 run used.

**6. Recommended Next Actions**

Before reading or acting on this checkpoint:

1. **Stop relying on the iter-238 MCQ aggregate JSON.** Manually re-submit the native MCQ sidecar for `iter_0000238_hf` with `BENCHMARKS=all` (the runner's default) — this reuses the existing HF dir and a single GPU-node, ~12 minutes more than what was already spent. Also patch `scripts/submit_checkpoint_sidecars.sh` L156 (and analogous lines for other sidecars if they share the pattern) so `BENCHMARKS` is quoted-and-isolated or passed via environment variable export *outside* the comma-separated `--export` list (e.g. `sbatch --export=ALL ... -v BENCHMARKS="$NATIVE_BENCHMARKS"` form, or pre-`export BENCHMARKS=...; sbatch --export=ALL ...`). Then back-fill iter 477 / iter 834 / iter 1192 as they appear, since they will inherit the same bug.

2. **Mark all `headline.macro_accuracy` values produced by the broken submitter as invalid** in `reports/5B_REPORT_DRAFT.md` and in the live status renderer. The current `latest_5b_report_status.md` row pulls `0.5026` as iter-238 headline MCQ; this is a 1-task slice, not the 3-task headline.

3. **Re-evaluate Apertus-Base under the live 4096/RoPE-500K geometry** as the anchor, OR re-run training under the official `65536/12000000`. Without one of these, no Vanilla-vs-Apertus-Base claim is defensible.

4. **Run a decontamination pass** (exact-match and n-gram MinHash) of GreekMMLU/ILSP/Plutus prompts against `hplt_b1_5b.jsonl`. Store the report under `reports/` and cite it before any MCQ number lands.

5. **Treat iter 238 as a "late-warmup probe."** Do not start the 5B-vs-10B continuation decision from this checkpoint; wait at least until iter 287+ (warmup complete) and preferably iter 477 (`Vanilla-2B`) for the first stable-LR snapshot.

6. **Re-issue retention with the goal-defined EN/FR/DE/RU coverage** (it currently misses `global_mmlu_ru` and `xstorycloze_*`).

7. **Acknowledge the prior-attempt history for iter 119** in any release evidence (sidecar attempt `2419080-2419087` failed and was archived) — `iter_0000238` has no archived prior attempts (clean), so this is iter-119-specific.

8. **Add a `BENCHMARKS` smoke check** to the watcher: after submitting `04native_i<iter>`, the watcher should grep the resulting sbatch stdout for `BENCHMARKS:` and assert the value matches `NATIVE_BENCHMARKS`. This would have caught the silent truncation.

**7. Persistence of Vanilla-0.5B Critical Findings**

- **(1) RoPE/seqlen mismatch (training 4096/RoPE-500K vs Apertus base 65536/RoPE-12M):** **STILL PRESENT.** Evidence: `iter_0000238_hf/config.json` reports `max_position_embeddings=4096`, `rope_theta=500000`, `rope_scaling=null`; live `training_command.sh` uses `--max-position-embeddings 4096 --rotary-base 500000 --seq-length 4096`; official base at `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json` still `65536 / 12000000`. The 0.5B `reports/config_geometry_audit_iter_0000119.{json,md}` documents the gap and the training continues.

- **(2) Headline native-MCQ JSON included Plutus alongside the 3 headline tasks at iter 119:** **MITIGATED IN STRUCTURE BUT NOW SUPERSEDED BY A NEW BUG.** Evidence: the runner was patched (`*_native_mcq_headline.json` now excludes Plutus, `*_native_mcq_aggregate.json` carries an explicit `headline_policy.headline_benchmarks=[greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep]` and `diagnostic_benchmarks=[plutus_qa]`, iter-119 was backfilled to headline=0.43910 / headline+diag=0.42932). At iter 238 the schema is now correct, but the submitter delivered only `greekmmlu`, so the headline aggregate is computed over 1 task, not 3 — fixing one ambiguity created another. Net: structurally fixed, *operationally re-broken at iter 238*.

- **(3) No decontamination evidence for public Greek MCQ benchmarks:** **STILL PRESENT.** Evidence: filesystem scans under `/capstor/scratch/cscs/fffoivos` and `/iopsstor/scratch/cscs/fffoivos/cpt_corpus` for `*decont*` / `*contam*` returned no artifacts. The `production_blockers_status.V1` flag in `goal/hyperparameters.json` documents the caveat (`not_required_for_diagnostic`) but no run-time evidence exists.

- **(4) Greek BPB heldout heavily prefix-truncated (~29% on Greek):** **STILL PRESENT.** Evidence: `iter_0000238/heldout_greek_bpb.json` `fraction_truncated=0.292`, `n_docs_truncated=146 / 500`, `n_tokens_dropped=6,906,358`. Identical to iter 119 (same heldout set, same `max_context=4096`). The metric is therefore prefix-BPB, not full-document BPB; the iter 119 → 238 delta (Greek BPB `0.6049` → `0.4684` global, −22.6%) is on a like-for-like prefix metric, but the absolute number is biased by truncation and not directly comparable to a longer-context evaluation.
