# Apertus Tokenizer Extension Release Reorganization Plan

Date: 2026-05-25.

Status: superseded by the human-readable naming pass in
[`APERTUS_EXTENSION_RELEASE_RENAMING_PLAN_20260525.md`](APERTUS_EXTENSION_RELEASE_RENAMING_PLAN_20260525.md).
This document records the first cleanup plan; the implemented public layout now
uses names like `TokenDistil-3.5B`, `ModernGreek-148k`, and `CPT-7B-mix`.

Purpose: reorganize the public-facing artifact layout around the real product:
the tokenizer extension, the selected checkpoint(s), the CPT dataset recipe, and
the evidence that explains how they were produced.

Companion map:

- [`APERTUS_EXTENSION_ARTIFACT_MAP_20260525.md`](APERTUS_EXTENSION_ARTIFACT_MAP_20260525.md)

## Goal

Make `fffoivos/apertus-tokenizer-extension` read as:

1. here is the Apertus Greek tokenizer extension;
2. here is the selected modern Greek extension size and why;
3. here is the CPT dataset graph used to test it;
4. here are the trained checkpoint artifacts;
5. here is the benchmark evidence;
6. here are the scripts and provenance if you want to audit the work.

It should not read as an unfiltered mirror of every script, intermediate run,
and historical branch.

## Repository Ownership Split

The public release should use two repositories with distinct jobs:

- Hugging Face model repo:
  `https://huggingface.co/fffoivos/apertus-tokenizer-extension`
  should be the artifact-facing release page. It should center tokenizers,
  checkpoint payloads or hydration pointers, dataset provenance, benchmark
  summaries, and compact audit evidence.
- GitHub source repo:
  `https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation`
  should be the canonical home for runnable scripts, Slurm launchers,
  conversion utilities, corpus-build code, eval code, and implementation docs
  for the Apertus extension work.

Do not make the Hugging Face repo a broad mirror of the scripts directory. The
HF repo should link to the GitHub subproject for code, and include only small
script excerpts or command snippets where they are needed to explain hydration
or reproduction.

## Release Principles

1. Center the product artifacts first.
   The tokenizer, checkpoint, dataset recipe, and final results should be
   visible before logs or implementation internals.

2. Separate payloads from provenance.
   A 16 GB checkpoint and a 41 GB JSONL dataset are payloads. A manifest,
   hydration command, hash, and benchmark table are provenance. They should not
   be mixed in one flat tree.

3. Promote one final checkpoint by default.
   The likely primary checkpoint is TD layer 11 at 3.5B, iter 834. Vanilla and
   ReTok should remain comparable baselines, but they should not crowd the front
   page unless their payloads are intentionally published too.

4. Preserve the audit trail without letting it dominate.
   Reviewer packets, dry runs, failed attempts, and watcher logs are useful, but
   they belong under archive or provenance indexes.

5. Make every large object intentional.
   If a checkpoint payload is uploaded, it should have a manifest, source path,
   hash or shard list, conversion status, eval status, and clear relation to the
   tokenizer. Do not upload entire run directories.

6. Make loss measurement explicit.
   Benchmark summaries must say that raw Megatron `lm loss` is health-only
   across different tokenizer vocabularies. Cross-arm loss evidence is heldout
   BPC/BPB plus downstream evals, with dense `bpb` logs used only when the
   training loop actually emits them.

7. Prefer additive reorganization first.
   Build the clean structure and top-level routing without deleting remote files
   immediately. Only delete or move legacy remote paths after a review pass.

8. Keep executable code in GitHub.
   Scripts and launchers should be promoted to the GitHub subproject path above,
   not duplicated into the HF release tree except as minimal hydration snippets.

## Superseded Initial Public Layout

The initial layout below was intentionally replaced because names like
`td-layer11-cpt-3p5b-iter834` mixed the human-facing artifact with technical
metadata. The implemented layout is:

```text
README.md
ARTIFACTS.md
manifest.json
checksums.sha256

tokenizers/
  ModernGreek-148k/
  ModernGreek-Polytonic-154k/

checkpoints/
  README.md

locations/
  TokenDistil-3.5B.md
  TokenDistil-2B.md
  TokenDistil-Init.md
  Vanilla-3.5B.md
  ReTok-3.5B.md
  CPT-7B-mix.md

datasets/
  CPT-7B-mix/

results/
  3.5B-comparison/

source-code/
  README.md
  manifest.json

provenance/
  tokenizer-selection/
  dataset-build/
  token-distillation/
  conversion-roundtrip/
  evals/

archive/
  legacy-layout.md
```

`checkpoints/` now contains only a README until actual model weights are
uploaded. Pointer-only checkpoint entries live under `locations/`.

## Original Proposed Public Layout

```text
README.md
MANIFEST.json
ARTIFACT_GRAPH.md
SHA256SUMS

tokenizer/
  modern-greek-17408/
    README.md
    manifest.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
  polytonic-plus-5120/
    README.md
    manifest.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json

checkpoints/
  td-layer11-cpt-3p5b-iter834/
    README.md
    manifest.json
    hydration.md
    # optional payload if we decide this repo should host weights
  td-layer11-init-r17-tp2/
    README.md
    manifest.json
    hydration.md
  baselines/
    vanilla-3p5b-iter834/
      README.md
      manifest.json
      hydration.md
    retok-3p5b-iter834/
      README.md
      manifest.json
      hydration.md

training-data/
  cpt-7b-mix/
    README.md
    manifest.json
    source_graph.json
    hydration.md

results/
  SUMMARY.md
  continuation_3p5b_summary.json
  benchmark_table.csv
  plots/

provenance/
  tokenizer-selection/
  dataset-build/
  token-distillation/
  conversion-roundtrip/
  evals/

code-links/
  README.md
  github_subproject_manifest.json

archive/
  legacy-hf-layout-index.md
  dry-runs/
  failed-attempts/
  old-review-packets/
  slurm-log-indexes/
```

## What To Promote

### Tokenizers

Promote:

- `apertus_greek_modern_only_148480`;
- `apertus_greek_extended_153600`.

Source paths:

- `subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480/`;
- `subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600/`.

Validation before upload:

- `AutoTokenizer.from_pretrained()` loads both bundles;
- vocab sizes are exactly `148480` and `153600`;
- base ids `0..131071` match Apertus;
- SHA-256 values match manifests.

### Checkpoints

Promote as first-class entries:

- TD layer 11 3.5B checkpoint, iter 834;
- TD layer 11 init checkpoint;
- Vanilla and ReTok 3.5B checkpoint pointers for comparison.

Do not promote:

- all intermediate per-iter checkpoints;
- failed attempts;
- raw run directories;
- optimizer states unless they are needed for continued training.

Recommended payload decision:

- publish the selected TD layer 11 3.5B checkpoint as a real HF model payload
  if the goal is for others to load and use it directly;
- keep Vanilla/ReTok as hydration pointers unless we want to support exact
  independent comparison without Clariden access;
- do not upload Centroid unless a reviewer specifically needs it.

Minimum checkpoint manifest fields:

```json
{
  "artifact_id": "checkpoint.td-layer11-cpt-3p5b-iter834",
  "tokenizer": "tokenizer.modern-greek-17408",
  "source_format": "Megatron torch_dist TP=2",
  "public_format": "HF safetensors or hydration pointer",
  "source_path": "/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_td_layer11/checkpoints/iter_0000834",
  "eval_hf_path": "/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000834_hf",
  "training_tokens": "3.5B target continuation",
  "base_model": "swiss-ai/Apertus-8B-2509",
  "status": "candidate-main"
}
```

### Training Data

Promote:

- source graph;
- dedup overlay reference;
- final mix recipe;
- hydration commands;
- exact token counts with the tokenizer distinction:
  - about `7.0B` by extended-tokenizer/mix-builder budget;
  - `9,831,704,774` base-tokenized Megatron tokens.

Do not promote:

- raw parquets;
- full JSONL payload;
- Megatron `.bin/.idx` payload, unless a separate dataset repo is explicitly
  created for it.

Recommended public form:

- `training-data/cpt-7b-mix/source_graph.json`;
- `training-data/cpt-7b-mix/manifest.json`;
- `training-data/cpt-7b-mix/hydration.md`.

### Results

Promote:

- 3.5B summary table;
- final benchmark table CSV;
- plot images;
- compact JSON summaries.

Source path:

- `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/`.

Keep per-task raw samples out of the front page.

### Scripts And Implementation Code

Promote scripts to GitHub, not Hugging Face.

Canonical GitHub destination:

- `https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation`

Script families that belong there:

- tokenizer shipping and validation helpers;
- corpus-build scripts and Slurm launchers;
- init-arm builders and conversion scripts;
- R17/xIELU/QK-Norm patch and verification scripts;
- Token Distillation coverage, layer-pilot, and conversion scripts;
- bakeoff and continuation launchers;
- eval conversion, packed eval, BPC, and new-token diagnostic scripts;
- production CPT launchers and hydration checks.

The HF repo should contain:

- links to the relevant GitHub paths;
- compact command snippets for loading artifacts;
- hydration instructions that call the GitHub scripts;
- manifests recording which GitHub commit produced each artifact.

The HF repo should not contain:

- a full `subprojects/` mirror;
- `__pycache__` files;
- dry-run command dumps as code;
- old failed-launch scripts unless they are explicitly archived as provenance.

## What To Demote Or Archive

Demote to provenance:

- cutoff sweep internals;
- firing-count bundles;
- TD coverage prepass;
- R17 patch verification;
- conversion reports;
- eval JSON snapshots.

Archive:

- stale C1/C2/fresh tokenizer bundles;
- broad `subprojects/` mirror;
- dry-run sbatch command dumps;
- watcher logs;
- old review packets;
- failed launch logs.

The archive should be indexed, not hidden. The goal is not to erase history; it
is to stop history from being the first thing a reviewer sees.

## Planning Docs To Preserve

The planning docs should be presented as an indexed source trail, not as a flat
dump. Some of them predate the 3.5B continuation and therefore need status
labels in the release.

### Current Or Still Load-Bearing

| Doc | Role in the story |
|---|---|
| `cpt_plan.md` | Design-space reference for the Apertus Greek CPT objective, replay/curriculum, init experiments, eval strategy, and production scale. |
| `TRAINING_RECIPE.md` | Apertus-faithful training recipe: Megatron-LM-Swiss-AI, AdEMAMix, LR, batch shape, Goldfish/NTP split, document masks, architecture constraints. |
| `apertus_fidelity_checklist.md` | Must-preserve Apertus constraints: AdEMAMix, 0.1 grad clip, xIELU, QK-Norm, Goldfish, special tokens, NFC, untied E/U. |
| `TOKEN_DISTILLATION_PLAN.md` | Sensitive TD plan: fixed-ID tokenizer handling, layer choice, firing prepass, untied output embeddings, preservation checks. |
| `03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md` | Original 2B bakeoff design: arms, pre-Clariden checklist, V-gates, Slurm shape, eval cadence. |
| `03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md` | CPT data recipe: dedup order, 70/24/4/2 mix, NFC build, CPU-only xfer rule. |
| `03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md` | Eval tasks, V4 comparator logic, checkpoint cadence, BPC/diagnostics, and hard-gate rubric. |
| `ARTIFACTS_AND_HYDRATION.md` | Repo ownership and hydration policy: what belongs in git/HF vs Clariden, production path checks. |
| `CLARIDEN_INVENTORY_20260524.md` | Remote path inventory for models, tokenizers, datasets, checkpoints, eval outputs, code, and envs. |
| `03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/CONTINUATION_3P5B_RESULTS_20260525.md` | Latest 3.5B continuation result; this is the current result anchor for the release story. |

### Review And Flight-Readiness Context

| Doc | Role in the story |
|---|---|
| `REVIEW_HANDOFF_20260524.md` | Broad reviewer map of docs, scripts, artifacts, and checks as of 2026-05-24. Useful, but pre-3.5B. |
| `PRODUCTION_DECISION_STATE.md` | 2B + TD-challenger production overlay. Important historical decision state, but it predates the 3.5B continuation where TD layer 11 became the leading release candidate. |
| `COMPLETENESS_CHECK.md` | Gap audit against `cpt_plan.md` v0.7; records which missing scripts/checks were later implemented. |
| `RISKS.md` | Silent-failure register; especially useful for explaining why roundtrip, xIELU/QK-Norm, heldout, xfer CPU, and harness checks exist. |
| `TAKEOVER_LOG_20260521.md` | Chronological operations log: what ran, what failed, what was restarted, what was fixed. |
| `CSCS_OVERNIGHT_STATE.md` | Operational handoff during the first CSCS run; useful for failure/restart provenance, not a current release entrypoint. |
| `03_2_apertus_c3_dedup_audit/READY_TO_SPIN_UP.md` | Earlier GCP dedup-audit flight-readiness example: preflight checks, teardown trap, cost boundary, and output gates. Archive as dedup provenance, not as the current training launcher. |

### Production Launcher Context

| Doc or artifact | Role in the story |
|---|---|
| `03_4_implementation_experiments/init_bakeoff/production_cpt/README.md` | Dry-run-validated production launcher runbook for the pre-3.5B Vanilla/base default. Keep as launcher provenance, but do not make it the final release decision if the selected checkpoint is TD layer 11. |
| `03_4_implementation_experiments/init_bakeoff/production_cpt/submit_vanilla_base_15b_chain.sh` | Dry-run-by-default production submitter; useful as a template and evidence of launch hygiene. |
| `03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/submission_plan.json` | Flight-readiness evidence: chain plan, token budget, dependency mode, loss objective, save cadence, no live submission. |
| `03_4_implementation_experiments/init_bakeoff/production_cpt/dryrun_default_vanilla_base_15b_nfc_20260524T121007/submission_chain.tsv` | Expanded Slurm chain from the dry run. |

## Flight-Ready Checks To Surface

These are the checks that made an experiment or launch "flight ready." The
clean release should expose them as a checklist, with links to the evidence
files and current status.

### Decision And Scope Checks

- Confirm the selected artifact is the current one, not a stale decision from an
  earlier checkpoint window.
- Label `PRODUCTION_DECISION_STATE.md` and `production_cpt/README.md` as
  pre-3.5B context if the release promotes TD layer 11.
- Confirm the target tokenizer family: base tokenizer, modern-only 148,480, or
  polytonic 153,600.
- Confirm training budget language is unambiguous: per-arm training tokens,
  total experiment tokens, available corpus tokens, and base-tokenized Megatron
  tokens are different numbers.

### Tokenizer Checks

- `AutoTokenizer.from_pretrained()` loads the shipped tokenizer bundle.
- Vocab size matches the intended target: `148480` modern-only or `153600`
  polytonic-stacked.
- Base ids `0..131071` are byte-identical to Apertus.
- First 1000 special/reserved ids are preserved.
- `unk`, `bos`, `eos`, and `pad` ids match Apertus.
- Tokenizer SHA-256 matches the manifest.
- New-token curation/backfill manifest is present for the 17,408 modern
  extension.

### Dataset And Corpus Checks

- Build order is correct: Apertus-overlap hard drop first, then nanochat
  internal dedup replay, then mix build.
- The final source pool is the selected post-Apertus-drop and post-internal
  dedup parquet.
- Mix recipe is the intended `70/24/4/2` Greek/replay/code/math recipe.
- NFC is enforced on source parquets and on the final JSONL stream.
- The production Megatron prefix points to the NFC-safe base-tokenized data.
- The manifest records rows, sequences, documents, bytes, tokenizer, and token
  count.
- CPU-only dataset/preprocess/coverage jobs use `xfer`, not GPU partitions.
- `check_cpu_only_slurm.sh` passes before submitting dataset or preprocessing
  work.

### Init And Conversion Checks

- Init checkpoints were built from the intended tokenizer and base model.
- Untied input embedding `E` and output head `U` are both handled.
- HF -> Megatron -> HF roundtrip passes for standard tensors and logits.
- R17/xIELU/QK-Norm patching passes with zero or accepted max diff.
- Raw unpatched conversion is not used for production or release checkpoints.
- Megatron TP=2 format is used for Apertus-faithful execution.
- TE empty-extra-state guard remains narrow and logged, not a blanket exception
  swallow.

### Training Config Checks

- Megatron-LM-Swiss-AI commit is pinned or recorded.
- Apertus-specific recipe is preserved: AdEMAMix, xIELU, QK-Norm, RMSNorm,
  bf16, fp32 main grads, 0.1 grad clip, sequence length 4096.
- Global batch tokens are documented even if microbatch differs.
- Loss objective is intentional: NTP for bakeoff; Goldfish only when production
  fidelity is intended and configured.
- Cross-document attention reset and EoD loss mask are enabled.
- Checkpoint save cadence matches the eval plan.
- Resume path preserves optimizer/RNG state when continuing a run.

### Smoke And Slurm Checks

- Run at least one single-arm smoke before firing expensive parallel chains when
  a new failure mode has appeared.
- The first forward/backward pass completes without OOM, NaN, or skipped
  iterations.
- Two-node training remains disabled unless the NCCL/OFI failure mode has been
  fixed and re-smoked.
- The production or continuation submitter runs in dry-run mode first.
- Dry run writes `submission_plan.json` and `submission_chain.tsv`.
- Slurm dependencies are `afterok` for real chains so failed jobs do not blindly
  launch successors.
- Check `squeue -u fffoivos` before launch to avoid overlapping accidental
  jobs.

### Eval Checks

- V4-HF and post-conversion comparator context is recorded.
- The eval task list is the pretraining-stage suite, not SFT/post-training
  tasks by accident.
- Greek task availability is verified against the actual harness clone.
- Harness commit is recorded for every eval run.
- Heldout JSONL path is staged and documented.
- BPC/NLL and new-token diagnostics run alongside downstream benchmarks where
  relevant.
- Full evals are packed when possible so training does not stop and GPU nodes
  are not wasted by serial single-GPU evals.
- Final decision uses late-window checkpoints, not only an early operational
  canary.

### Hydration And Release Checks

- Required remote paths exist before launch or release:
  - base model;
  - tokenizer bundle;
  - dataset `.bin/.idx` prefix;
  - init or final checkpoint;
  - Megatron code;
  - eval runtime.
- Large payloads are either intentionally uploaded or represented by hydration
  pointers with exact paths.
- `MANIFEST.json` records code refs, hashes, source paths, remote paths, and
  dependency edges.
- The GitHub subproject contains the runnable scripts used to produce the
  artifacts.
- HF README links to GitHub for source code and does not present a broad
  `subprojects/` mirror as the primary interface.

### Explicit Not-Ready Items

- Anneal data is not ready unless a separate xfer build has produced it.
- Polytonic TD/CPT is not ready from the modern-only corpus alone.
- ILSP Greek YAML tasks remain separate work unless they are ported and
  verified in the eval harness.
- A launcher that was dry-run validated for Vanilla/base should not be reused
  unchanged for TD layer 11 without a new dry run and manifest update.

## Concrete Implementation Phases

### Phase 1: Create A Clean Local Release Staging Tree

Create:

```text
release/apertus-tokenizer-extension/
```

Populate it with the target layout above. This is additive and does not disturb
the working repo or current HF remote.

Outputs:

- `release/apertus-tokenizer-extension/MANIFEST.json`;
- `release/apertus-tokenizer-extension/ARTIFACT_GRAPH.md`;
- `release/apertus-tokenizer-extension/SHA256SUMS`;
- per-artifact `README.md` and `manifest.json` files.

### Phase 2: Verify And Promote Scripts In GitHub

Before publishing the cleaned HF layout, verify that the executable scripts
needed to recreate, hydrate, convert, train, and evaluate the artifacts are
present in:

```text
subprojects/03_apertus_extension_and_embedding_adaptation/
```

and are committed to:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

Create a small `code-links/github_subproject_manifest.json` in the HF staging
tree that records:

- GitHub repo URL;
- commit SHA;
- script family;
- relative path in the GitHub repo;
- artifact(s) that script family can reproduce or validate.

This phase keeps the HF repo lightweight while still making every artifact
auditable from source code.

### Phase 3: Generate The Artifact Manifest

Create a machine-readable manifest with one entry per first-class artifact:

- `tokenizer.modern-greek-17408`;
- `tokenizer.polytonic-plus-5120`;
- `dataset.cpt-7b-mix`;
- `checkpoint.td-layer11-init-r17-tp2`;
- `checkpoint.td-layer11-cpt-3p5b-iter834`;
- `checkpoint.baseline-vanilla-3p5b-iter834`;
- `checkpoint.baseline-retok-3p5b-iter834`;
- `results.continuation-3p5b`.

Each entry should have:

- `artifact_id`;
- `type`;
- `status`: `payload-present`, `hydration-pointer`, `compact-evidence`, or
  `archived`;
- `local_source`;
- `remote_source`;
- `public_path`;
- `hashes` where applicable;
- `depends_on`;
- `review_docs`.
- `github_code_refs` for the scripts that produced or validate the artifact.

### Phase 4: Copy Only Curated Small Payloads

Copy into the staged tree:

- tokenizer bundles;
- compact result summaries;
- benchmark CSV/JSON/plots;
- source graph and hydration docs;
- compact provenance files.

Do not copy:

- `.safetensors`, `.distcp`, `.bin`, `.idx`, `.pt`, `.pth`, `.ckpt`, `.gguf`;
- raw corpora;
- full Slurm logs;
- entire run directories.

Exception: if the user explicitly decides that the selected TD checkpoint should
be hosted in this repo, upload only that converted HF checkpoint payload with
Xet/LFS and a manifest. Do not upload every checkpoint by inertia.

### Phase 5: Rewrite The Public README

The top-level README should have this order:

1. What this repo is.
2. Main artifact table.
3. Quick load examples for the tokenizer and selected checkpoint.
4. Result summary.
5. Dataset provenance summary.
6. Reproduction and hydration links that point to the GitHub subproject.
7. Script/source-code section with the canonical GitHub URL.
8. Archive/provenance links.

The README should not start with run history.

### Phase 6: Upload Additively To Hugging Face

Upload the staged tree additively first.

Safety rules:

- do not delete existing remote paths in the first pass;
- do not move legacy paths until the staged layout is reviewed;
- after review, either:
  - leave old paths in place but route users through the new README; or
  - move old paths under `archive/legacy-hf-layout-20260525/`.

### Phase 7: Validate The Published Repo

Validation checklist:

- `AutoTokenizer.from_pretrained()` works for both tokenizer bundles;
- `MANIFEST.json` parses;
- every `public_path` in `MANIFEST.json` exists;
- SHA-256 checks pass for tokenizer files and compact evidence;
- README links resolve;
- no accidental large binary files appear outside intentional checkpoint
  payload directories;
- selected checkpoint manifest names the exact tokenizer and training data;
- results table matches
  `CONTINUATION_3P5B_RESULTS_20260525.md`.
- every artifact manifest has a `github_code_refs` entry pointing to the
  GitHub subproject path;
- the HF README links to the GitHub source subproject instead of mirroring all
  implementation scripts.

## Proposed Main README Artifact Table

| Artifact | Status | Where |
|---|---|---|
| Modern Greek tokenizer, 148,480 vocab | payload | `tokenizer/modern-greek-17408/` |
| Polytonic tokenizer, 153,600 vocab | payload | `tokenizer/polytonic-plus-5120/` |
| TD layer 11 3.5B checkpoint | payload or pointer | `checkpoints/td-layer11-cpt-3p5b-iter834/` |
| TD layer 11 init checkpoint | pointer | `checkpoints/td-layer11-init-r17-tp2/` |
| CPT 7B mix recipe | pointer and manifest | `training-data/cpt-7b-mix/` |
| 3.5B benchmark results | compact payload | `results/` |
| Source scripts | GitHub canonical | `code-links/` points to GitHub subproject |
| Full provenance | compact payload | `provenance/` |
| Legacy upload contents | indexed archive | `archive/` |

## Open Decisions Before Uploading Weights

1. Should the selected TD 3.5B checkpoint be uploaded as a payload to
   `fffoivos/apertus-tokenizer-extension`, or should it get a separate HF model
   repo linked from this one?
2. Should Vanilla and ReTok 3.5B baselines be payloads or hydration pointers?
3. Should optimizer states be published, or only inference-ready HF weights?
4. Should the full Megatron dataset binary be published as a dataset repo, or
   should the public repo keep only recipe and hydration instructions?

Recommended answers:

- upload inference-ready HF weights for the selected TD checkpoint;
- keep baseline checkpoints as pointers unless a reviewer asks for payloads;
- do not publish optimizer states by default;
- keep the dataset as recipe/hydration unless there is a separate dataset-release
  pass.

## Review Checklist For The Reorg

Before changing the existing HF remote layout destructively, review:

- the staged release tree;
- the generated `MANIFEST.json`;
- the proposed README;
- which checkpoint payloads are included;
- which old remote paths are merely archived versus deleted;
- whether the final TD checkpoint is loadable outside Clariden.

## Final Intended Shape

After reorganization, a reviewer should be able to answer these questions in
under five minutes:

1. What tokenizer should I use?
2. What model checkpoint should I use?
3. What dataset was used to train it?
4. What won the bakeoff?
5. Where is the audit trail if I want to verify every step?
