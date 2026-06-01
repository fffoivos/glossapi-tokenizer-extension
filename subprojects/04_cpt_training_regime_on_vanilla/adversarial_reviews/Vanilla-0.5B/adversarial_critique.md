**1. Verdict**

Provisionally usable as an early checkpoint artifact, not trustworthy enough yet for scientific interpretation or checkpoint selection.

The iter `0000119` checkpoint appears real: training job `2417446` saved it, conversion job `2419108` completed, eval sidecars completed, and the training log shows finite loss with `skipped iterations 0` and `nan iterations 0` at `499122176` consumed tokens. But I would not act on the reported gains/losses yet. The comparison is materially confounded by a base-model positional-config mismatch, ambiguous native-MCQ aggregation, no visible MCQ decontamination, heavy BPB truncation, and brittle sidecar/reporting automation.

**2. Critical Findings**

1. **The “Vanilla” checkpoint does not preserve the official Apertus HF positional config.**

   Official local HF base config at `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json` reports:

   - `max_position_embeddings = 65536`
   - `rope_theta = 12000000`

   The init conversion log also showed the imported base using `max_position_embeddings 65536`, `rotary_base 12000000`, and RoPE scaling. But the actual training command for job `2417446` uses:

   - `--seq-length 4096`
   - `--max-position-embeddings 4096`
   - `--rotary-base 500000`

   The converted HF checkpoint at:

   `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0000119_hf/config.json`

   also has `max_position_embeddings: 4096` and `rope_theta: 500000`.

   This is not a small metadata wart. It means comparison to `Apertus-Base` may be apples-to-reconfigured-applesauce unless the base was evaluated under the same 4096/500K geometry. The local training recipe says the architecture should be inherited via checkpoint args, while `bakeoff_train.sbatch` re-declares the positional settings. Until this is resolved, differences could be due to geometry/config changes, not CPT.

2. **Native MCQ headline aggregation is ambiguous and easy to misreport.**

   The plan says the headline should use GreekMMLU + ILSP Medical + ILSP ASEP, with Plutus as diagnostic. But:

   `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z/native_mcq/Vanilla-0.5B_native_mcq_headline.json`

   includes Plutus alongside the headline tasks.

   Observed accuracies:

   - GreekMMLU: `0.4984968735`
   - ILSP ASEP: `0.4808333333`
   - ILSP Medical: `0.3379629630`
   - Plutus QA: `0.4000000000`

   The intended 3-task mean is about `0.43910`; including Plutus gives about `0.42932`. That is a material difference relative to the previously reported nearby Vanilla numbers around `0.43-0.44`.

3. **No visible decontamination evidence for public Greek MCQ benchmarks.**

   The Greek BPB heldout doc IDs appear clean against the final training JSONL; I found `overlap_count: 0` for the 500 heldout IDs. But that only covers BPB heldout doc IDs. GreekMMLU, ILSP Medical, ILSP ASEP, and Plutus are public-style benchmark artifacts, and the training source is HPLT web text. I found no exact/n-gram/minhash contamination artifact proving benchmark questions/answers are absent from the CPT data. This can invalidate headline MCQ claims.

**3. Major Findings**

1. **BPB eval is heavily prefix-truncated.**

   Greek BPB:

   - `bpb = 0.6049364592`
   - `n_docs = 500`
   - `truncated_docs = 146`
   - truncation rate `29.2%`
   - `tokens_dropped = 6906358`

   Code BPB:

   - `bpb = 0.4177850318`
   - `truncated_docs = 23 / 200`
   - truncation rate `11.5%`

   The metric is therefore mostly “prefix BPB under 4096-token context,” not full-document BPB. That may be acceptable, but it must be reported that way.

2. **Greek NLP diagnostic is not interpretable as capability evidence.**

   The Greek NLP sidecar completed, but outputs look like base-LM prompt echoing or malformed generation rather than task completion. Examples include `intent_classification` and `legal_classification` at `0.0` accuracy/F1, NER entity F1 around `0.0096`, and summarization degeneracy. Treat this as a failure-mode diagnostic only, not a reliable benchmark.

3. **Sidecar automation had a real broken-script failure before repair.**

   Failed conversion job:

   - `2419080`
   - failed because `_train_config_common.env` was missing from the expected eval/bakeoff path

   Repaired sidecars then ran as `2419108-2419114`. The current manifest points to the repaired jobs, while the failed attempt is only preserved separately. A verifier that says “all good” while omitting the earlier failed submission is too forgiving for release evidence.

4. **Compute hygiene problem: sidecars requested/used whole GPU nodes.**

   `sacct` shows the eval/conversion sidecars allocated `gres/gpu=4` and `billing=288`, despite the sidecar script requesting one GPU. Even the failed 9-second conversion attempt allocated a full GPU node. This is not an eval-validity problem, but it is bad cost/resource hygiene.

5. **Converted checkpoint integrity is insufficiently proven.**

   The Megatron checkpoint has eight large `.distcp` shards and the HF conversion has four safetensor shards, but I did not see a checksum manifest tying source checkpoint shards to converted HF shards. The conversion log also warns: “Original vocab size not specified, leaving embedding table as-is.”

6. **The checkpoint is still inside warmup.**

   At iter `119`, LR was `5.217768E-06`; target LR is `1.1e-5`, warmup is `287` iterations. This checkpoint is an early-training shock probe, not evidence for the stable post-warmup regime.

**4. Minor Findings And Hygiene Notes**

- `run_metadata.json` says `lr_schedule_style: constant` but also `lr_decay_style: 1-sqrt`; the actual command uses `--lr-decay-style constant`.
- Local docs disagree about Apertus positional settings: `hyperparameters.json`, `cpt-plan.md`, `TRAINING_RECIPE.md`, and the live scripts do not tell one clean story.
- Retention eval includes more than the requested EN/FR/DE/RU focus, including broad multilingual tasks. Fine, but comparisons must use matched task sets.
- Local report state artifact had `training_latest`, `training_health`, `evals`, and `adversarial_reviews` as null despite live logs/evals existing, so report-state automation is not enough.
- Automated adversarial review sidecar appears still in progress locally; no final `adversarial_critique.md` was present when checked.
- Mandatory cloud cost checks could not verify active AWS/GCP instances because AWS auth failed and GCloud required reauth in a non-interactive session.

**5. Missing Evidence**

Required before trusting this checkpoint:

- A matched-config baseline eval: Apertus-Base evaluated with the same `max_position_embeddings=4096`, `rope_theta=500000`, tokenizer, and eval harness, or a rerun preserving official `65536` / `12000000`.
- Explicit native MCQ aggregate artifact separating:
  - 3-task headline: GreekMMLU, ILSP Medical, ILSP ASEP
  - diagnostic-only: Plutus
  - confidence intervals and baseline deltas
- Decontamination report for GreekMMLU / ILSP / Plutus against final HPLT training JSONL.
- Checksum manifest for:
  - Megatron `iter_0000119` `.distcp` shards
  - converted HF safetensor shards
- Full sidecar history in verifier output, including failed/canceled attempts `2419080-2419087`.
- A clear canonical statement of intended architecture: official Apertus long-context geometry or bakeoff-local 4096/500K geometry.
- BPB rerun or annotation that reports truncation prominently, preferably with sliding-window or shorter-doc heldouts.

**6. Recommended Next Actions**

Do not select or interpret `Vanilla-0.5B` yet except as an early smoke checkpoint.

First resolve the positional-config issue. Either rerun/evaluate everything under official Apertus `65536` / `12000000`, or explicitly define this bakeoff as a 4096/500K regime and evaluate the base model under the same regime.

Then regenerate native MCQ reporting with the intended 3-task headline, Plutus separated, CIs included, and contamination checks attached. Treat BPB as prefix-only until truncation is fixed or fully documented. Exclude Greek NLP diagnostics from capability claims until the base-LM evaluation adapter is repaired.

Only after those are done should this checkpoint be compared to Apertus-Base, prior Vanilla checkpoints, or TD checkpoints.

