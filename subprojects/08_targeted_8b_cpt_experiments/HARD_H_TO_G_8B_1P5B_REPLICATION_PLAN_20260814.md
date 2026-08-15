# Matched benchmark-clean 8B and 1.5B HPLT-to-GlossAPI replication and extension plan

Date: 2026-08-14
Status: R2-review-adjusted design; implementation incomplete; **not
launch-authorized**
Scope: two matched training trajectories, each with a frozen 13.497B-token
replication endpoint and a 2B-token OpenArchives continuation

Review basis:
`REVIEW_ULTRACODE_HARD_H_TO_G_PLAN_20260814.md` and
`REVIEW_ULTRACODE_R2_HARD_H_TO_G_PLAN_20260814.md`. Both R2 blockers and all
verified major contract findings are incorporated below; none of the newly named
builders, scorers or receipts may be treated as existing until their launch
gates actually pass.

## Decision summary

The experiment has three predeclared goals:

1. **A — replication:** reconstruct the successful 13.497B-token hard
   HPLT-to-GlossAPI curriculum and test whether the 8B result is recovered;
2. **B — scale mirroring:** run the same data and schedule at 1.5B and test
   whether its learning, forgetting and benchmark trajectories mirror 8B; and
3. **C — OpenArchives continuation:** after freezing the 13.497B-token endpoint,
   continue the OpenArchives phase for two more approximately 1B-token
   intervals at both scales and test whether adaptation and retention extend
   coherently.

The two main trajectories are:

1. an 8B benchmark-clean near-replication of the historical
   `curr_td_b20p999_b3p999_13b_20260616T093527Z` trajectory; and
2. a token-, data- and schedule-matched 1.5B trajectory.

Before either training stream is tokenized, freeze every example from every
benchmark that may be reported, then decontaminate **all** HPLT,
GlossAPI/OpenArchives, foreign-replay and Old-Greek-replay pools against that
complete union. The same post-removal documents, Megatron weighted blends and
randomized GPTDataset index caches are used for both model sizes.

Both models will consume the same tokenized documents, global batches and
replay identities in the same order. The architecture and a separately
calibrated peak learning rate are the only intended model-scale differences.
The 1.5B learning-rate calibration will not inspect GreekMMLU.

Update 3,218 remains the immutable historical-horizon endpoint used for goal
A. Only after saving its complete model, optimizer, RNG and data-cursor state
does each trajectory continue to updates 3,456 and 3,694. The continuation
does not retroactively redefine the replication endpoint.

**Do not run the complete GreekMMLU set at every checkpoint unless the sentinel
calibration fails.** Freeze nested, deterministic 4,096- and 8,192-question
sentinel panels for the dense trajectory. Validate them independently against
the complete panel in both an early window (updates 0/238/476/714) and the
decision-critical late window (2,618/2,856/3,094/3,218). Fall back to the
complete clean panel at all decision-bearing checkpoints unless both tests
resolve the relevant changes. Run the complete 16,159-question clean panel at
all eight calibration checkpoints, update 3,694, and the earliest additional
member of a mechanically detected plateau when required.

The 8B update-3,218 replication endpoint additionally receives the exact
historical public GreekMMLU evaluator needed to compare with the old 59.94%
headline. That score is a historical reference, not a strict replication
pass/fail criterion, because the new corpus removes contamination for more
benchmarks than the old GreekMMLU-only Stage A did.

No checkpoint averaging, 0.5B arm, stationary-mix control or internal
GlossAPI curriculum is part of this experiment.

## 1. Questions answered

### A — replication question

Does a benchmark-clean reconstruction of the 8B hard HPLT-to-GlossAPI recipe
recover the high GreekMMLU result and the associated
Greek-adaptation/foreign-retention trajectory of the historical selected
beta2 arm?

### B — scale-mirroring question

Does the 1.5B model reproduce the *shape* of the matched 8B trajectory closely
enough to act as a checkpoint-timing proxy?

This experiment does **not** establish that 1.5B can select between competing
data schedules. That would require a second matched treatment, such as a
stationary Mixed control, at both scales.

### C — OpenArchives-continuation question

After the historical endpoint, do two additional approximately 1B-token
intervals of the same OpenArchives-plus-replay phase continue improving
OpenArchives and Greek capability, and do the 1.5B and 8B models show the same
direction and trade-off against HPLT, foreign and Old-Greek retention?

“Extend appropriately” does not mean every benchmark must improve
monotonically. It means the continuation is numerically stable, OpenArchives
loss responds in the expected direction, benchmark changes are measured with
uncertainty, and any extra adaptation is reported together with its marginal
retention cost.

## 2. Historical target

The target trajectory is the selected beta2 arm:

| Field | Historical value |
| --- | ---: |
| Run tag | `curr_td_b20p999_b3p999_13b_20260616T093527Z` |
| Base model | `swiss-ai/Apertus-8B-2509` |
| Final update | 3,218 |
| Tokens per update | 4,194,304 |
| Total token slots | 13,497,270,272 |
| HPLT-to-GlossAPI switch | update 2,261 / 9,483,321,344 tokens |
| Mix in both phases | 79% active Modern Greek / 20% foreign / 1% Old Greek |
| Peak/final LR | `5.5e-5` / `5.5e-6` |
| Warmup | fixed 400 updates / 1,677,721,600 token slots; init LR `5.5e-6` |
| Scheduler anchor | nominal `TRAIN_TOKENS=13,500,000,000`; `--train-samples=3,295,898` |
| Cooldown | `659,179` samples; one-minus-square-root, nominally ending at 0.1 peak |
| Optimizer | AdEMAMix, beta1/beta2/beta3 `0.9/0.999/0.999`, alpha `4` |
| Alpha and beta3 ramp | full 3,218-update run |
| Weight decay / gradient clip | `0.1 / 0.1` |
| Sequence length | 4,096 |
| RoPE | base 500,000 with the historical Llama-3 scaling configuration |
| Loss | Goldfish `k=h=50` |
| Data seed | `20260609` |
| Sample order | randomized Megatron GPTDataset mapping; `CURRICULUM_ORDER_MODE=randomized`, `MEGATRON_GPT_DATASET_NO_SHUFFLE=0` |
| Global batch | 1,024 sequences |
| Historical 8B microbatch | 2 sequences; may differ at 1.5B only after parity-gated profile selection |
| Training precision | BF16 parameters/compute with FP32 main gradients |
| Training code | Megatron-LM-Swiss-AI commit `c92402e39ef3c8e69ea378a59e79059dc14541f4` |
| Historical final GreekMMLU | `0.5993867244` |
| Historical best observed GreekMMLU | `0.5996272246` |

Primary authorities:

- `../05_token_distillation_cpt/PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/results/beta2_decision_table_20260711.csv`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/results/sweep_config_audit_20260711.json`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/train/curriculum_common.env`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/train/phase1_hplt.env`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/train/phase2_glossapi.env`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/train/runtime_patches/reset_data_index_guard.py`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/eval/cadence_curriculum_3218.tsv`

Authority is field-specific. The selected run's `run_metadata.json` controls
the run tag, revision, seed, randomized data mode and realized recipe. The
frozen common/phase env files and surviving training command control sample
horizons, WSD, precision, batch geometry, optimizer, Goldfish and RoPE flags.
The cadence TSV controls checkpoint positions. The beta2 decision table and
`sweep_config_audit_20260711.json` justify arm selection only; the latter is
not cited as evidence for scheduler or precision fields it does not contain.

The authoritative sample-order receipt is the selected run's CSCS metadata:

`/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/curr_td_b20p999_b3p999_13b_20260616T093527Z/run_metadata.json`

It overrides the conflicting local `physical_order` default, which predates
the selected run by five days but was not the value the run actually recorded.
The execution bundle must pin the receipted historical randomized mode
explicitly rather than inherit either value from the submitter environment.

The complete curriculum-v2 staging tree is known to be an empty 424 KiB
skeleton: packed binaries, intermediate JSONL, split outputs and historical
heldout-id Parquets are gone. The unversioned primary Greek SELECTED Parquet is
also gone. The new run additionally removes documents contaminated against the
complete native-Greek benchmark suite, while historical Stage A screened only
GreekMMLU. Therefore this run cannot be called either byte-for-byte or a strict
data replication. The correct name is **benchmark-clean near-replication**.
The historical configuration and result are the target; every unavoidable
data reconstruction difference remains an explicit report field.

## 3. Experimental matrix

| Contract row | Model | Update range | Data schedule | Purpose |
| --- | --- | ---: | --- | --- |
| `R-HG-8B` | pinned `swiss-ai/Apertus-8B-2509` revision | 0–3,218 | HPLT then OpenArchives | benchmark-clean near-replication of the historical 8B result |
| `R-HG-1p5B` | pinned `swiss-ai/Apertus-v1.1-1.5B` revision | 0–3,218 | identical HPLT then OpenArchives | test 1.5B trajectory similarity |
| `E-OA-8B` | exact `R-HG-8B` endpoint | 3,218–3,694 | continued OpenArchives plus replay | measure two 1B-scale extension intervals |
| `E-OA-1p5B` | exact `R-HG-1p5B` endpoint | 3,218–3,694 | identical continuation | test whether the smaller-model tail mirrors 8B |

The `E-*` rows are checkpoint-continuation suffixes, not new initializations or
independent optimization runs. There remain two model trajectories in total.

The 1.5B model is the better architectural proxy than Apertus 0.5B for this
question because it has untied input/output embeddings and the same 32
attention-head / eight-KV-head counts as the 8B model. It is not an exact
miniature: it has 16 rather than 32 layers, hidden width 2,048 rather than
4,096 and head dimension 64 rather than 128.

## 4. Tokenizer and Token Distillation contract

### Default replication stack

Use the historical Modern-Greek 148,480-token tokenizer for both cells. The
tokenizer survives with byte-level receipts, but the as-consumed Megatron TD
checkpoint does not. Its only full-payload recovery authority is:

- repository `fffoivos/apertus-tokenizer-extension`;
- revision `fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`;
- path `experiment-checkpoints/TokenDistil-Init`; and
- expected payload: four safetensors totalling 16,391,939,480 bytes with
  `target_layer=11` in its manifest.

The replication stack is accepted only when all of the following pass:

- tokenizer and merge SHA-256 receipts;
- `materialize_base_init.py` verifies the pinned HF payload;
- a fresh HF-to-Megatron R17 conversion and Megatron-to-HF round-trip receipt;
- the historical in-loop untied-output procedure; and
- the exact historical tokenizer reconstruction path.

Record `as-consumed Megatron initialization payload was purged and could not
be re-hashed` as a named reconstruction difference. Do not silently treat the
new conversion as the historical checkpoint merely because its source is the
documented HF round trip.

This is the only stack that removes the tokenizer change as a confound when
comparing with the old 59.94% result. The additional benchmark-wide training
exclusions still mean that comparison is a historical reference rather than a
strict replication.

If any required historical artifact cannot be reconstructed, both cells may
instead use the current verified 148,992-token production tokenizer. That
fallback changes the experiment's name to **production-stack
near-replication** and prohibits claiming direct reproduction of the absolute
historical score.

### 8B initialization

Reuse the verified historical untied layer-11 Token-Distillation procedure:

- one epoch;
- 25 snippets per added token;
- batch size 8;
- LR `1e-4`;
- BF16 with seed `20260523`;
- target layer 11;
- input rows trained by hidden-state MSE;
- output rows learned inside the same TD loop with
  `learn_output_with_ce=True`, with original-row output gradients zeroed; and
- base 131,072 vocabulary rows byte-identical to the parent model.

### 1.5B initialization

Port the same untied procedure to the 1.5B architecture. Predeclare target
layer 6. The proportional mapping is an exact tie (`16 * 11 / 32 = 5.5`);
choose 6 as the frozen upward/even tie-break. In executable terms this means
`target_layer=6` in the vendored trainer, i.e. `hidden_states[6]` where
`hidden_states[0]` is the embedding output—not decoder-module index 6.

The pinned `swiss-ai/Apertus-v1.1-1.5B` tokenizer package is not byte-identical
to the historical 8B-derived tokenizer in reserved special-token slots. Its
own tokenizer metadata leaves padding undeclared and places the `<pad>`
surface at id 10, while the model config declares `pad_token_id=3`; the target
148,480 tokenizer declares `<pad>` at id 3. Across the base 131,072 rows,
131,058 token strings remain at the same id and the complete 269,443-rule base
merge list is an exact prefix. The exact 14 content differences and 18
added-token-record differences are frozen in
`configs/1p5b_tokenizer_compatibility_v1.json`.

For the matched experiment, preserve every original input and output row by
**id**, without permutation, and apply the shared target tokenizer. This keeps
all ordinary-token weights and all base rows byte-identical, aligns padding
with the model config, and makes the reserved-slot overlay explicit rather
than silently pretending the two HF tokenizer packages are identical.

The historical token-id selection survives inside `retok_td_manifest.json`,
but the snippet text corpus is gone. Before building the 1.5B init:

1. verify and hash the original coverage input if it survives;
2. otherwise regenerate snippets deterministically from the frozen
   benchmark-clean Modern-Greek stream using the historical coverage prepass:
   2,000,000,000 target extended tokens, 100 candidate snippets per token,
   radius 50 and seed `20260523`;
3. retain the historical selected-token order from the manifest;
4. freeze the new `snippets.jsonl` and token-id ledger before training; and
5. report `regenerated TD snippet text` as a named initialization difference.

Do not claim identical Token Distillation inputs when the historical snippet
text cannot be recovered.

Do not choose the 1.5B distillation layer from GreekMMLU. It must pass the
intrinsic initialization gates:

- every original vocabulary row byte-identical;
- all appended input and output rows finite;
- no zero or missing appended rows;
- row-norm distribution within a pre-init acceptance interval derived from the
  verified 8B TD artifact: for input and output matrices separately, normalize
  every added-row norm by the median base-row norm, freeze the 0.5th/99.5th
  percentile interval with 20% outward padding, where
  `w = p99.5 - p0.5`, `lower = p0.5 - 0.2w`, and
  `upper = p99.5 + 0.2w`; require at least 99% of 1.5B
  added rows inside it, and require its median ratio inside
  `[0.8 * m_8B, 1.2 * m_8B]`, where `m_8B` is the frozen median added-row ratio
  of the verified 8B artifact;
- at least 25 accepted snippets for every trained token, trained-token fraction
  at least `0.99`, and an explicit skipped-token ledger whose rows retain the
  merge-chain fallback (the historical run trained 17,377/17,392 and skipped
  15; 100% coverage is not required);
- exact HF-to-Megatron-to-HF round-trip for all tensors; and
- two-update plus checkpoint/restart parity before production training.

The norm-band receipt must be created from the 8B artifact **before** the 1.5B
TD job is run. The 1.5B bridge, conversion verifier and restart-parity harness
are new deliverables; the existing scripts are 8B-specific and cannot be
relabelled as 1.5B evidence.

If layer 6 fails an intrinsic gate, stop and diagnose it. Do not silently try
layers until a benchmark improves.

## 5. Dataset reconstruction and provenance

Build one immutable Megatron weighted-blend dataset contract and reuse it at
both model scales.

### Historical pipeline baseline

The historical log receipts only the post-Stage-A clean document counts:

| Stream | Receipted clean documents | Input/dropped counts |
| --- | ---: | --- |
| HPLT | 9,535,742 | not receipted; re-derive, do not use as a gate |
| GlossAPI/OpenArchives | 77,136 | not receipted; re-derive, do not use as a gate |
| Replay before split | 5,031,733 | not receipted; re-derive, do not use as a gate |

Authorities:

- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/dataset/stageA_clean_decontam_binary.sbatch`
- `../05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/dataset/stageB_anon_preprocess_binary.sbatch`
- `../05_token_distillation_cpt/EXECUTION_LOG_CURRICULUM_SWEEPS_V2.md`

Stage A first applies the HPLT E001 confident-only character residue strip to
both Greek streams; it preserves row count but changes text. It then uses
`decontaminate.py --benchmark greekmmlu --primary-rule correct_only` on all
three streams. Stage B consumes `_decontam.jsonl`, applies the historical
email/IP/IBAN anonymization and tokenizes. The clean counts are historical
provenance only; they are **not** expected counts for the new benchmark-clean
corpus.

The selected 79/20/1 runs then split `replay_only_final.jsonl` with
`split_replay_final_for_lr.py` and tokenized two distinct binaries. The
historical receipt recorded 3,808,235 foreign rows and 1,223,498 Old-Greek
rows, consumed through active-stream-relative weights:

- `FOREIGN_REPLAY_R=0.253164557` (`20/79`); and
- `OLD_GREEK_REPLAY_R=0.012658228` (`1/79`).

The reconstruction must reproduce and receipt this split. A single
`replay_only` binary is not a valid substitute.

### Direct application of the published benchmark overlaps

The new Stage A must cover the union of all examples that can be reported in
this experiment, primarily by applying the already-published overlaps:

- GreekMMLU: the union of the clean 16,159-question evaluation and the exact
  historical/public evaluator; the 4,096-question sentinel is nested in this
  union and introduces no extra examples;
- DemosQA;
- Medical MCQA;
- ASEP MCQA;
- GPCR;
- OYXOY NLI;
- OYXOY WSD-definition;
- OYXOY WiC; and
- OYXOY metaphor.

Freeze benchmark revisions, splits, example ids, prompts and raw fields before
building the training manifest. The primary evaluation contract is:

`/Users/foivoskarounos-zamparloukos/Projects/.codex-worktrees/train-apertus-full8-results/subprojects/09_full_8b_cpt_results_analysis/evaluation/native_greek_3cp_contract.json`

The native-suite overlaps have already been computed over the exact Hugging
Face dataset and are the primary training-exclusion input, not merely a source
of matching logic:

- decision record:
  `/Users/foivoskarounos-zamparloukos/Projects/.codex-worktrees/train-apertus-full8-results/subprojects/09_full_8b_cpt_results_analysis/evaluation/CONTAMINATION_DROP_DECISION_20260812.md`;
- immutable CSCS audit:
  `/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1`;
- [Hugging Face evidence
  folder](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/tree/b28806da9b9bc5a569acbf923a378c967b1c1752/benchmark_contamination/native_greek_suite_v1):
  revision `b28806da9b9bc5a569acbf923a378c967b1c1752`, under
  `benchmark_contamination/native_greek_suite_v1`;
- audited data revision:
  `987b8955fcd395c6219e39df9e64715457f69065` (431 Parquet shards,
  51,839,746 rows);
- frozen query SHA-256:
  `ef9d601b8f91c6845818b9584c6634a13337c77b07e3f101f755a4884634c0eb`;
- match-table SHA-256:
  `1b23a9dc14a6175c18e0530210cc47795e24e3841bb1b3229c666877ac4b4b19`;
- exclusion SHA-256:
  `7a8559461b15a308f599faf0ff25cd16c07be0a597078864f779af7f2f1fdd32`.

`qa_document_line_matches.parquet` is directly applicable to every selected
row inherited from the audited data revision. It records the immutable dataset
shard, zero-based row, `source_dataset`, `source_doc_id`, 1-based source-line
range and match class. Build the exclusion manifest by joining these published
document matches to the selected HPLT/GlossAPI/other-v2 rows. Verify all four
coordinates—revision, shard/row, source identifiers and the published
`document_text_sha256` over raw UTF-8 text—before excluding a row. Perform this
join against pristine v2 rows before E001 changes the text; any hash mismatch
fails the build. Do **not** rerun the native-suite content matcher over those
51,839,746 rows.

The full match table, rather than
`recommended_excluded_example_ids.jsonl`, is the training-data authority. The
JSONL file is a post-hoc benchmark-score filter; the Parquet table is what
identifies the training documents to remove before packing. Use the copy
already present at the immutable CSCS audit root and verify its published
SHA-256 instead of transferring its 6.344 GB payload through the MacBook.

The published strict evidence rule is question plus correct-answer evidence
for MCQ; premise plus hypothesis for NLI; both usages for WiC; usage plus the
correct definition for WSD; and usage plus source definition for metaphor.
Apply the published strong/strict document matches exactly. Preserve
question-only hits as a separate candidate ledger; do not convert all 18.17M
trace rows into blanket training exclusions. Any additional candidate
exclusion requires a recorded adjudication and is reported separately from the
published strict policy.

Remove exactly the matched published document rows selected for training. Do
not perform another global deduplication or infer broader cross-document
clusters that are absent from the published evidence. If multiple published
rows represent the same matched document, the table already exposes each row
and all of them are removed through the same join.

### Coverage boundaries

The published native-suite payload covers DemosQA, Medical MCQA, ASEP MCQA,
GPCR and all four OYXOY tasks. It explicitly does **not** contain GreekMMLU or
Protipa.

- GreekMMLU cannot use the deleted historical Stage-A query artifact as current
  authority. Regenerate the exact query set from pinned revision
  `6a03aa06...` and freshly scan **every rebuilt selected stream** with frozen
  `k=8`, `correct_only`, after-direction and 50/5-token gap rules. The surviving
  2026-07-31 query/ledger artifacts are provenance and a positive
  reconciliation check, not exclusions for v2.
- Protipa is unavailable and frozen as excluded from both training-union
  claims and evaluation. It cannot be added post hoc.
- Any foreign- or Old-Greek-replay row that does not originate from audited HF
  revision `987b895...` is outside the published table's scope. Apply the
  frozen benchmark queries to every selected external row through the named
  heterogeneous-schema replay adapter. Source-level disjointness is not an
  escape from content scanning.

### Decontamination receipts and fail-closed gate

For every benchmark-by-training-pool pair, record:

- frozen benchmark query count and revision;
- candidate and strict match counts;
- published shard/row and source-document exclusion keys;
- documents and token mass removed from each pool;
- the source artifact and policy used (published HF join, fresh GreekMMLU
  scan, or heterogeneous external-replay scan); and
- residual exclusion-key counts.

After anonymization and tokenization, verify row lineage and anti-join the
actual selected document manifest against the published and GreekMMLU
exclusion keys. Require **zero selected excluded documents**. This is a cheap
identity verification, not another content scan of the HF corpus. The packer
must consume only this verified manifest. Both 8B and 1.5B recipes must bind to
the same manifest and decontamination receipt hashes.

The published audit's 10,048 strict evaluation units and 10,076 scored
exclusions are benchmark-unit counts, not training-document counts. Derive the
actual removed-document and removed-token counts by joining the published
match table to this experiment's selected rows; do not rerun the overlap scan
or mistake scored-example counts for corpus-removal counts.

Protipa is permanently out of scope for this frozen experiment and may not be
added as a post-training test.

### Reconstruction rules

The historical intermediate stage is empty and the old primary Greek SELECTED
Parquet is deleted. There is no live reuse branch. The full rebuild is:

1. inventory and hash the surviving `replay/` assets, `greek_replay.parquet`,
   StarCoder staging/manifest and deleted/surviving GreekMMLU evidence on CSCS;
   bind the non-Greek replay payload to the completed 2026-07-31 acquisition
   receipt (`9ea630cb...`, 355 files, 250,673,537,368 selected remote bytes,
   seed `20260609`) and record that its deterministic file selection replaces
   the deleted historical 13.5B replay files while preserving the frozen
   source weights; never claim historical replay document identity;
2. treat every unversioned or missing upstream asset as unavailable—not as
   permission to refetch a drifting `main` revision;
3. use the pinned HF v2 revision as the named Greek-source reconstruction
   difference; inventory observed labels; select HPLT exactly as
   `HPLT/ell_Grek_ge8_no_mt_clean60` and OpenArchives by
   `^openarchives\\.gr`; freeze counts; and anti-join both active views against
   Greek replay natural keys;
4. freeze and reuse the corrected 13-panel full-8B validation manifest at
   SHA-256 `a4b1d696...`; derive exact stored-UTF-8 text hashes for every panel
   document and exclude those hashes from every rebuilt training stream before
   tokenization. This improves cross-scale and current-full-8B comparability,
   but it is a named difference from the historical H→G heldouts, so panel-loss
   values are not claimed as an exact historical-loss replication;
5. verify and apply the published native-suite exclusions to pristine v2 rows
   using raw-text hashes, failing on any coordinate or hash mismatch;
6. reproduce the historical modern-stream mix-builder geometry over those
   benchmark-clean source views before E001: 8,500,000,000 aggregate HPLT
   target tokens and 3,700,000,000 aggregate OpenArchives target tokens, each
   with 16 eligible-row-modulo shards, respective per-shard targets of
   531,250,000 and 231,250,000 tokens, seed `20260611`, and shard-index-order
   concatenation. Preserve release coordinates and source metadata in the
   selected JSONL through a lineage-only output-schema extension; a parity
   test must prove that it changes neither eligible rows nor the historical
   scheduler, token counts, RNG choices or selection order. Because the old
   selected Parquets were deleted, this reproduces the algorithm and geometry
   over the pinned v2 benchmark-clean views but does not claim historical
   document identity;
7. run E001 on the selected HPLT and OpenArchives rows;
8. rebuild the historical replay mix with `mix_builder` and
   `make_phase_recipes.py`, recording all source revisions, counts and hashes
   and applying the frozen panel-text exclusions to the selected replay
   stream. Reproduce the historical replay builder geometry exactly:
   5,000,000,000 aggregate target tokens,
   16 eligible-row-modulo source shards, 312,500,000 target tokens per shard,
   seed `20260611`, and concatenation in ascending shard-index order. This
   reproduces the selection algorithm but, because the historical Parquets
   were deleted, does not claim the same replay documents;
9. regenerate GreekMMLU queries and freshly scan the selected HPLT and
   OpenArchives streams. Normalize the exact selected replay aggregate through
   the frozen heterogeneous adapter, scan it against the complete native-suite
   union, remove those hits, then scan the surviving replay against the frozen
   GreekMMLU query union and remove those hits. The replay split happens later,
   so these aggregate scans cover every eventual foreign and Old-Greek row;
10. verify historical Stage-B anonymization is a byte-preserving no-op on the
   already-anonymized v2 HPLT and OpenArchives text. Apply Stage B only to the
   native-suite-clean and GreekMMLU-clean replay JSONL. Then audit the exact
   Stage-B replay bytes in `audit-only` mode against GreekMMLU and materialize
   those same bytes for the native-suite post-filter scan; both post-Stage-B
   scans must report zero before splitting or tokenization;
11. split the final post-Stage-B replay JSONL into foreign and Old-Greek streams with the
   historical splitter and freeze its manifest. Then tokenize the HPLT,
   OpenArchives, foreign-replay and Old-Greek Stage-B streams separately with
   the pinned 148,480 tokenizer, historical `preprocess_data.py`, `--append-eod`,
   `--json-keys text` and 64 workers. Each `.bin/.idx` pair must reconcile to
   exactly one indexed document per input JSONL row and carry a bundle-,
   tokenizer-, Megatron- and upstream-stream-bound receipt; and
12. freeze Phase-1/2 `.bin/.idx`, blend specs, seed, randomized index caches
    and heldout receipts; enumerate the Phase-2 realized document ledger; then
    build Phase 3 separately from only unseen documents.

Do **not** add a second global deduplication, new quality filter or new PII
policy. Any such change is a separate data experiment.

The faithful trainer stack is the historical Megatron weighted blend, not the
later explicit schedule reader. Both model scales bind to identical `.bin`/
`.idx` hashes, phase-specific blend strings, randomized seed and exported
GPTDataset index-cache hashes. Masks and batch assignment remain runtime
Megatron behavior; the plan no longer claims that training consumes an
external global sequence manifest. A read-only realized-sample ledger may be
exported for cross-scale verification, but it is evidence—not the dataloader.

The post-boundary replay-identity ledger is new-run-only evidence. It is
compared between 8B and 1.5B, not against history, because no historical
sample-identity receipt survives.

No training job may launch if the selected HF rows cannot be reconciled with
the audited revision and published match coordinates, if an evaluated
benchmark lacks a document-exclusion or disjointness receipt, if a strict
residual match survives, if required scratch replay assets are unpinned or
missing, or if a reconstruction difference remains unnamed.

## 6. Exact data schedule

The schedule is one continuous optimizer/LR trajectory with a data-index reset
at the phase boundary. Both phases use the historical randomized Megatron
GPTDataset mapping with seed `20260609`; fail if the no-shuffle patch or an
inherited environment variable changes the recorded mode.

### Phase 1: HPLT

- initial checkpoint at update 0, then optimizer updates 1 through 2,261;
- 79% HPLT;
- 20% foreign replay;
- 1% Old-Greek replay.

The executable weighted blend is active stream weight `1.0`, foreign replay
weight `0.253164557` and Old-Greek replay weight `0.012658228`.
Freeze this as an ordered, role-labelled three-prefix `--data-path` spec. Hash
each component `.bin/.idx` once on `debug`, make every component read-only,
and bind an explicit writable cache root. The patched trainer must pass that
root with `--data-cache-path`; implicit caches beside a data prefix are
forbidden.

### Phase 2: GlossAPI

- boundary checkpoint at update 2,261, then optimizer updates 2,262 through
  3,218;
- 79% GlossAPI/OpenArchives;
- 20% the same foreign-replay policy;
- 1% the same Old-Greek-replay policy.

Use the same relative replay weights as phase 1.

The first phase-2 invocation must reproduce the historical
`RESET_DATA_INDEX=1` behavior while loading the optimizer, AdEMAMix slow state,
LR scheduler, RNG state and global update from the boundary checkpoint. The
historical guard set `consumed_samples=0` for the entire first phase-2 training
loader, so the new GlossAPI-plus-replay weighted blend restarted its data
sampling together. Reproduce that behavior rather than continuing a phase-1
replay cursor. Replay composition remains 20% foreign and 1% Old Greek, and
the realized post-boundary replay identities must be frozen explicitly.

### Phase 3: OpenArchives continuation

After the complete update-3,218 endpoint is frozen:

- run optimizer updates 3,219 through 3,456, adding 998,244,352 total token
  slots;
- continue to update 3,694, adding another 998,244,352 total token slots;
- keep the 79/20/1 OpenArchives/foreign/Old-Greek proportions but construct a
  separate immutable blend from documents not realized earlier in the main
  trajectory; and
- preserve optimizer, AdEMAMix slow state and RNG without a global reset.

Phase 3 is **not** an appended index-cache tail. Historical Phase 2 used a
randomized GPTDataset construction spanning roughly three epochs; its realized
sample stream contains document repetition, estimated near 26%, and cannot be
described as an unseen-document prefix. First export the exact realized
Phase-1/2 document-id ledger from both frozen caches. OpenArchives candidates
are thereby compared with Phase 2, while foreign and Old-Greek replay
candidates are compared with both phases. Then anti-join that ledger from every
Phase-3 pool, prove adequate token capacity, build a separate randomized
weighted blend/cache, and consume it from cursor zero. No document may repeat
within Phase 3.

The optimizer/scheduler and dataset horizons are deliberately decoupled only
for Phase 3. The scheduler retains the terminal global sample horizon
`3,782,656`, but the guarded GPTDataset builder receives exactly `487,424`
Phase-3 samples. This prevents the independent extension cache from building
multiple epochs merely because the restored scheduler carries the global
horizon. Its train `document_index` arrays must contain no repeated document
indices. Phase 1 and Phase 2 retain the historical full-horizon GPTDataset
request of `3,295,898` samples; in particular, the historical Phase-2 repeats
are reproduced rather than corrected.

The c92402e blend builder applies a fixed `1.005` per-component construction
margin. Therefore the one-epoch capacity gate is stronger than the bare
79/20/1 token quota: OpenArchives, foreign replay and Old-Greek replay must
cover respectively `386,991`, `97,973` and `4,899` component samples, or at
least `1,585,115,137`, `401,297,409` and `20,066,305` EOD-inclusive tokens.
Checking only the un-margined target plus one 4,096-token sequence would allow
the cache builder to tile a second epoch and is forbidden.

The generalized phase-local guard applies to every phase-entry **and every
restart inside Phase 2 or Phase 3**. It must restore the phase-local cursor,
verify the phase/cache hash and next sample identities, while preserving the
global optimizer update, optimizer/AdEMAMix state and RNG. For Phase 3:

- update 3,218: `0` consumed Phase-3 sequences;
- update 3,456: `(3456 - 3218) * 1024 = 243,712` consumed sequences; and
- update 3,694: `(3694 - 3218) * 1024 = 487,424` consumed sequences.

The update-3,218-to-3,219 and update-3,456-to-3,457 smokes must prove exact next
sample-index-cache identity against an uninterrupted control.

These are checkpoints separated by approximately 1B **total token slots**.
Each interval therefore contains approximately 0.789B OpenArchives/GlossAPI
tokens, not 1B active OpenArchives tokens. The complete extension is
1,996,488,704 token slots and the terminal horizon is 15,493,758,976 token
slots.

Do not wrap Phase 2 or graft new indices into its cache. A pre-extension
receipt must prove at least approximately 1.577B active OpenArchives token
slots plus the required unseen replay capacity after the realized-document
anti-join. Phase-3 construction cannot alter any sequence identity through
update 3,218 because it is an independent cache activated only after the
signed endpoint permits exist.

Document-boundary behavior is frozen to the realized historical launcher:
`--reset-attention-mask`, `--reset-position-ids`, and `--eod-mask-loss` are all
enabled. Thus attention does not cross an EOD boundary, positions restart at
the boundary, and the EOD target is loss-inactive. These flags remain identical
between the two cells. Sequence length, global tokens per update, main and extension
horizons, checkpoint token positions and phase-switch token position are fixed.

## 7. Learning-rate policy

### 8B

Use the historical peak LR `5.5e-5` exactly. There is no new 8B LR search.

### 1.5B

Keep the same warmup-token mass, WSD shape, cooldown start, cooldown duration
and final/peak ratio. Calibrate only the peak magnitude.

A scale interpolation between the previously stable 0.5B peak `1.5e-4` and
the selected 8B peak `5.5e-5` gives a 1.5B prior near `1.0e-4`. The proposed
three-point pilot is:

`7.5e-5`, `1.0e-4`, `1.25e-4`.

This intentionally differs from the sibling scale-study proposal to center on
the model's original pretraining LR. The present grid instead interpolates two
measured Greek-CPT optima; record that methodological choice explicitly rather
than implying the two plans use the same prior.

Before any pilot output exists, freeze:

- a single shared HPLT prefix of approximately 1B token slots;
- the exact three candidates;
- the loss/gradient stability limits; and
- the source-conditioned adaptation and replay-retention selector.

GreekMMLU and the downstream benchmark suite are forbidden during LR
selection. Reset to the exact Token-Distillation initialization after choosing
the LR, then run the full 13.497B-token trajectory.

The pilot cannot run before a 1.5B execution profile exists. First benchmark
scientifically identical one-, two- and four-node `normal` candidates (or the
smallest feasible subset) and promote one through fixed-batch loss/gradient
and restart parity. Then freeze a pilot allocation receipt containing nodes,
partition, wall time, microbatch, accumulation and projected p90 completion.
Run each LR candidate from the same initialization on the same 238-update
prefix in a separate receipt-bound `normal` allocation; no candidate may reuse
another candidate's optimizer state. The production profile may be promoted
only after the three-arm pilot and a longer throughput confirmation.

The profile selector is frozen before measurements: every candidate runs 256
updates, timing discards updates 1–32, and scientific parity is mandatory.
Among at least two passing candidates, maximize end-to-end tokens/GPU-hour.
Candidates within 2% of the best efficiency are tied; choose the lowest p90
step time, then fewer nodes. The one-node 1.5B trajectory is the fixed-batch
reference. Its own update-1 checkpoint is also the sole source of the resumed
update-2 restart comparison, so restart parity is not confounded by an
independent 0-to-1 run. The 8B profile is not re-selected: its bounded
TP2/DP32 16-node geometry is revalidated against itself for throughput and
against its own update-1 checkpoint for restart parity.

The LR selector is also frozen before measurements. Each candidate must finish
238 finite updates with zero skipped/non-finite steps. Use only update-0 and
update-238 `lm_loss` for HPLT, the macro-average of English/German/Russian/
Chinese/code replay, and Old Greek. A candidate is eligible only when every
foreign/Old-Greek final-minus-initial loss is at most `0.01`. Rank eligible
candidates by the mean of the three relative loss improvements; differences
within `1e-6` tie and choose the lower peak LR. If no candidate is eligible,
production remains blocked for owner review. GreekMMLU and every downstream
benchmark are forbidden inputs to this selector.

### Extension LR for both scales

The nominal sample-domain WSD floor is reached at approximately update
3,218.65, so update 3,218 is the frozen endpoint checkpoint rather than a claim
of exact sample-boundary equality. The extension deliberately starts from that
checkpoint at the nominal 0.1-times-peak floor and holds it constant:

- 8B: `5.5e-6`;
- 1.5B: 0.1 times its independently selected peak LR.

Do not restart warmup, rewind the cooldown, raise LR or stretch the original
WSD schedule. Any of those choices would either alter the replication prefix
or turn goal C into a second LR experiment. The continuation is a constant
floor tail beginning from the exact historical-horizon optimizer state.
Likewise, the AdEMAMix alpha and beta3 ramps must finish at update 3,218 exactly
as in the main recipe, then remain fixed at alpha `4` and beta3 `0.999` through
update 3,694. Do not recompute either ramp against the longer terminal horizon.

The executable extension recipe sets the total train-sample horizon to
`3,782,656`, uses a constant scheduler with `lr=min_lr`, sets LR warmup samples
to zero, and freezes both alpha and beta3 ramp denominators at 3,218 updates.
Loading the shorter checkpoint therefore requires the explicitly expected
optimizer-parameter-scheduler override; a guard must reject any override that
reanchors WSD, warmup, alpha or beta3 to the longer horizon.

The scheduler continuation implementation must be unit-tested on `debug` and
then proven across updates 3,218–3,219 in a profile-matched checkpoint/resume
smoke. The 8B distributed smoke uses its bounded 16-node `normal` profile; the
1.5B smoke uses its promoted profile. Receipts must show the exact floor LR,
fixed alpha/beta3, continuous optimizer state and expected next sequence
identity.

## 8. Checkpoint and evaluation policy

### Saved checkpoints

Preserve the historical 1B-token trajectory positions and the phase boundary:

| Update | Token slots | Role |
| ---: | ---: | --- |
| 0 | 0 | Token-Distillation initialization |
| 238 | 998,244,352 | approximately 1B |
| 476 | 1,996,488,704 | approximately 2B |
| 714 | 2,994,733,056 | approximately 3B |
| 952 | 3,992,977,408 | approximately 4B |
| 1,190 | 4,991,221,760 | approximately 5B |
| 1,428 | 5,989,466,112 | approximately 6B |
| 1,666 | 6,987,710,464 | approximately 7B |
| 1,904 | 7,985,954,816 | approximately 8B |
| 2,142 | 8,984,199,168 | approximately 9B |
| 2,261 | 9,483,321,344 | HPLT/GlossAPI boundary |
| 2,380 | 9,982,443,520 | approximately 10B |
| 2,618 | 10,980,687,872 | near cooldown start |
| 2,856 | 11,978,932,224 | approximately 12B |
| 3,094 | 12,977,176,576 | approximately 13B |
| 3,218 | 13,497,270,272 | frozen replication endpoint |
| 3,456 | 14,495,514,624 | OpenArchives extension +1B checkpoint |
| 3,694 | 15,493,758,976 | OpenArchives extension +2B terminal checkpoint |

Use the historical update-119 save interval for restart safety, and additionally
force exact save-and-exit walls at updates 3,456 and 3,694; neither is on the
119-update grid. Convert only evaluation checkpoints. Do not average
checkpoints. After all receipts
are final, retain the initial, boundary, cooldown-near, frozen replication
endpoint, both extension checkpoints and any distinct sentinel-best
checkpoint; intermediate optimizer checkpoints may be pruned only under a
separate storage receipt.

### Source-conditioned loss

Keep the historical online panel and the new selection panel separate.

The **historical online panel** runs at the exact 25-update cadence and contains
the nine historical extra-valid sets only:

`hplt`, `openarchives`, `greek_phd`, `english`, `de`, `ru`, `zh`, `code`, and
`old_greek`.

It reports the historical Megatron `lm_loss` and perplexity fields. Do not
claim that it historically supplied BPB, source-family loss, neutral-Greek
loss or base-versus-added-token strata. Reconstructed versions of these panels
remain useful trajectory evidence, but any panel with exact or cluster-level
training overlap is visibly labelled contaminated and receives
`selection_authorized=false`.

The **new offline selection panel** is a pre-launch build deliverable. It
contains document-cluster-clean HPLT; OpenArchives aggregate and source-family
panels; foreign replay by language family; Old Greek; a neutral external
Modern-Greek panel; and base-token versus added-token target strata. Freeze
document ids, cluster ids, UTF-8 byte counts, tokenizer revision, prompt or
prefix rules, code-bundle hash and a zero-overlap receipt for each panel. Score
these panels at all 18 saved checkpoints on a receipt-bound `debug` queue.

For source panel `s`, report token negative log-likelihood and

`BPB_s = -sum_t log2 p(x_t | x_<t>) / UTF8_bytes_s`.

Define `BPB_OA,macro` as the unweighted mean across the predeclared
OpenArchives source families and define the primary balanced-Greek diagnostic
as `0.5 * BPB_HPLT + 0.5 * BPB_OA,macro`. Define replay macro BPB analogously
over the frozen foreign-language families; always publish the complete family
vector beside the macro.

Never collapse the panels into a single micro-average. The finalizer must
refuse checkpoint selection from any receipt whose
`selection_authorized` field is not exactly `true`; prose warnings alone are
not sufficient.

### GreekMMLU sentinel

Freeze nested 4,096- and 8,192-question subsets of the clean 16,159-question
panel before any new-model predictions exist. The builder must use the
following exact rule:

1. use `subject x educational_level` as the stratum only if the frozen dataset
   manifest proves that `educational_level` is populated and stable; otherwise
   use `subject` and record the fallback;
2. select a stable lowest-hash floor question for every subject, then
   incrementally apportion only each larger panel's additional slots over the
   remaining stratum capacities by Hamilton largest remainder;
3. within every stratum, sort by
   `SHA256(UTF8("greekmmlu-sentinel-v1") || byte(0) || UTF8(canonical_question_id))`;
4. break Hamilton remainder ties by a salted SHA-256 of the stratum id; retain
   every selected id forever so a later target cannot remove an earlier id; and
5. assert that every subject is represented and that 4,096 is a strict subset
   of 8,192.

Use the same ids, prompt, answer order, tokenizer and FP32 scorer at both model
scales. Use candidate batch size **1** for the clean-panel scorer. Existing
FP32 batch-1-versus-4 evidence preserved predictions but failed its raw-score
tolerance, so it does not authorize batch 4 or 16. Batch 16 remains only in the
separate historical BF16 public compatibility evaluator.

For question `i` and candidate `j`, let `s_ij` be the candidate continuation
log-likelihood divided by its number of scored continuation tokens. Define

`p_ij = exp(s_ij) / sum_k exp(s_ik)`

and define **choice NLL** as

`L_choice = -(1/n) sum_i log p_i,y_i`.

Also report correct-answer BPB and accuracy at every checkpoint. Accuracy
remains the historical endpoint headline even though choice NLL is the denser
selection signal; any disagreement in checkpoint ranking must be explicit.

The historical target run cannot validate this subset: its per-question files
were deleted. The surviving 8B production-run predictions and the separate
0.5B D0 predictions use different tokenizers, datasets or schedules and may be
used only as separately labelled, non-binding sampler dry runs after their
exact paths and hashes are frozen. They cannot satisfy the sentinel gate.

Instead, validate the sentinel **on the new run and scoring stack** against
full clean-panel predictions twice: early at updates 0/238/476/714 and late at
2,618/2,856/3,094/3,218. Use `B=10,000` paired-question bootstrap replicates,
seed `20260814`, and percentile 95% confidence intervals. Set one resolution
target per window mechanically to

`tau = 0.5 * median(|Delta_full_choice_NLL(t,t-1)|)`

over that window's three adjacent pairs. The formula is frozen before results;
only the two data-dependent values are computed later. A sentinel size is
authorized only if it passes **both** windows.

The 4,096 panel passes only when, for all three adjacent pairs:

- its delta has the same sign as the complete-panel delta whenever the full
  paired 95% interval excludes zero; and
- its paired delta standard error is at most `tau`.

If it fails, apply the identical test once to the nested 8,192 panel. If that
also fails, the sentinel remains descriptive only and the complete 16,159-
question panel is required at every checkpoint that can affect selection or a
trajectory-mirroring decision. This is a valid terminal outcome, not a reason
to tune the sample, seed or threshold.

Evaluate the authorized sentinel at all 18 checkpoints. Define the minimum-NLL
**plateau set** as all checkpoints `c` for which the paired 95% interval of
`L_choice(c) - L_choice(c_min)` includes zero. Peak comparisons use overlap of
plateau sets, not a noise-dominated argmin. If the plateau set is not exactly
the singleton `{3,218}`, mechanically run the full panel at its earliest
member other than 3,218. Update 0 has no historical
curriculum GreekMMLU counterpart and must not be shown as one.

### Complete GreekMMLU

Run the complete clean panel at updates 0, 238, 476, 714, 2,618, 2,856, 3,094,
3,218 and 3,694 of both models, plus the earliest non-3,218 plateau member
whenever the plateau is not exactly `{3,218}`. Do not transfer
the 8B subset-resolution receipt to 1.5B or vice versa. If the sentinel gate
fails, use the complete panel at every decision-bearing checkpoint.

At the 8B update-3,218 endpoint only, also reproduce the exact historical
public evaluation:

- evaluator:
  `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py`;
- historical code commit:
  `cfdd0e7b00761a736be660867bf3d09733e24a92`;
- registry dataset: `dascim/GreekMMLU`;
- complete public panel: 16,632 questions;
- dtype: BF16; and
- exact historical prompt, choice normalization and sidecar arguments copied
  from the historical launcher into a machine-readable contract.

The historical final is `9969/16632 = 0.5993867244`; the historical best is
`9973/16632 = 0.5996272246`. Never compare either number to the 16,159-question
clean FP32 panel. The clean result is the scientific primary; the BF16 public
result is the like-for-like legacy replication reference. The historical code
did not forward a dataset revision, so execution must use a pre-materialized
snapshot at revision `6a03aa06...` and a loader-only compatibility wrapper
that passes parity against the unmodified loader on that pinned revision. If
the 148,992 fallback tokenizer is used, skip this score or label it explicitly
not comparable; it cannot decide Goal A.

Historical full-set jobs required roughly 9--13 minutes after conversion. The
live `debug` queue may still serialize many such jobs, so evidence-complete ETA
must include the measured conversion/evaluation backlog rather than treating
evaluation as free.

### Complementary native-Greek suite

Reuse the frozen scorer and examples from the canonical results worktree:

`/Users/foivoskarounos-zamparloukos/Projects/.codex-worktrees/train-apertus-full8-results/subprojects/09_full_8b_cpt_results_analysis/evaluation/native_greek_3cp_contract.json`

The immutable CSCS evaluation authority is
`/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812`, with
frozen code revision `036e1e2e53e13d3cd58cc8cb22b3c38a52881580`. Freeze a
copy of both the contract **and that exact evaluator code tree** into the new
execution bundle, record its tree hash, and verify it at every job start rather
than depending on a Mac-only path at runtime. The suite contains:

- DemosQA;
- Medical MCQA;
- ASEP MCQA;
- GPCR;
- OYXOY NLI;
- OYXOY WSD-definition;
- OYXOY WiC; and
- OYXOY metaphor.

Score the suite at six decision checkpoints per model:

- initialization;
- update 2,261, immediately before the data switch;
- update 2,618, near cooldown start and after GlossAPI exposure;
- update 3,218, the frozen replication endpoint;
- update 3,456, the first OpenArchives extension checkpoint; and
- update 3,694, the second OpenArchives extension checkpoint.

If the GreekMMLU sentinel selects another checkpoint, append that checkpoint
to the suite. Because every materialized benchmark was excluded from the new
training corpus before tokenization, the complete frozen benchmark sets are
the primary results. A strict-filtered view may also be reported for direct
compatibility with the earlier completed-model analysis, but it is no longer
the mechanism that makes this run benchmark-clean.

Protipa is permanently unscored for this experiment because owner-side access
was unavailable at contract freeze. Do not report it as evaluated or add it
after training.

## 9. Analysis and success criteria

### A — 8B near-replication report

The operational legacy-equivalence target is
`p0 = 9969 / 16632 = 0.5993867244`. Predeclare an absolute accuracy margin of
`M = 0.015`. The margin is a predeclared operational tolerance: roughly one
percentage point for evaluation sampling variability plus a separate half-
percentage-point allowance for unavoidable source reconstruction and broader
decontamination. It is not derived by adding two exact confidence limits and
does not claim that the old contaminated and new benchmark-clean sets are
identical.

Compute a 90% Wilson interval for the new 16,632-question BF16 public-evaluator
accuracy:

- **PASS:** the entire interval lies inside `[p0 - M, p0 + M]`;
- **FAIL:** the entire interval lies outside that equivalence band; and
- **INCONCLUSIVE:** the interval intersects a band boundary.

This decision is available only under the historical 148,480 tokenizer and
the pinned legacy evaluator. If the study falls back to the 148,992 production
stack, goal A's absolute legacy-equivalence test is `NOT TESTABLE`; the run is
reported only as a benchmark-clean trajectory comparison.

Regardless of the verdict, report clean GreekMMLU accuracy, choice NLL and
correct-answer BPB; source-conditioned curves; foreign and Old-Greek
forgetting; the update-2,261-to-2,380 switch response; the NLL plateau set; any
NLL-versus-accuracy rank disagreement; and every reconstruction difference.
An equivalence failure is a real result but does not by itself identify whether
data reconstruction or training numerics caused the difference.

### B — 1.5B trajectory-mirroring gate

Normalize time by consumed token progress. Use `B=10,000` paired-question or
document-cluster bootstrap replicates, seed `20260814`; report percentile 95%
intervals for losses and slopes and 90% intervals for the proxy-decision
correlations. These intervals quantify evaluation-panel sampling conditional
on one realized training run per scale; they do **not** estimate seed-to-seed
training variance. Freeze these windows:

- pre-switch OLS: updates `0, 238, ..., 2,142, 2,261`;
- post-switch OLS: updates `2,261, 2,380, 2,618, 2,856, 3,094, 3,218`; and
- extension OLS: updates `3,218, 3,456, 3,694`.

The boundary belongs to phase 1 for data accounting but is reused as the
post-switch response baseline. Fit each metric against normalized token
progress with ordinary least squares. Also report slope magnitudes after
normalizing each metric by its update-0 value; direction alone is the gate.

For every slope or immediate-switch comparison:

- **PASS:** both model intervals exclude zero with the same sign;
- **FAIL:** both exclude zero with opposite signs; and
- **INCONCLUSIVE:** either interval includes zero.

The immediate switch response is the paired change from update 2,261 to 2,380
and is a primary criterion, not merely a plot annotation.

Do not correlate cumulative improvement from initialization. For GreekMMLU
choice NLL and balanced-Greek BPB, compute Spearman correlation across matched
**adjacent-checkpoint first differences**. In each bootstrap replicate,
resample the paired questions or document clusters, rebuild both complete
delta series and recompute Spearman. Score each correlation as:

- **PASS:** point estimate at least `0.45` and the 90% lower bound is above
  zero;
- **FAIL:** the 90% upper bound is below zero; and
- **INCONCLUSIVE:** otherwise.

For panel `s`, define historical-horizon forgetting as

`F_s = L_s(3218) - min_{t <= 3218} L_s(t)`.

Before final-checkpoint results are visible, define each panel's material-
forgetting margin exactly as
`delta_s = 2 * median(document-bootstrap SE_s(t))` over updates
0/238/476/714. Recompute the historical minimum inside every bootstrap
replicate. Classify `s` as **material forgetting** when the 95% lower bound
of `F_s` exceeds `delta_s`, **no material forgetting** when its upper bound is
at most `delta_s`, and **inconclusive** otherwise. The two scales agree only
when HPLT, foreign replay and Old Greek receive the same non-inconclusive
class; otherwise this criterion is inconclusive or failed when classes are
opposite.

Goal B has four primary families: every pre/post slope-direction test, every
immediate-switch test, plateau-set overlap, and the retention-class vector.
Its two secondary tests are first-difference Spearman for GreekMMLU choice NLL
and balanced-Greek BPB. Goal B **passes** only if all primary families pass,
at least one secondary passes, and neither secondary fails. It **fails** if any
primary fails or both secondary tests fail; otherwise it is inconclusive.
Extension intervals use Goal C rather than a three-point Spearman. Never
convert uncertainty into a pass from a point estimate alone.

### C — OpenArchives continuation analysis

For each model, treat updates 3,218, 3,456 and 3,694 as a three-point extension
trajectory and report:

- OpenArchives/GlossAPI BPB change per interval and cumulatively;
- HPLT, foreign-replay and Old-Greek BPB change over the same intervals;
- neutral-Greek BPB and all native-Greek benchmark deltas;
- GreekMMLU sentinel choice-NLL, correct-answer BPB and accuracy;
- marginal OpenArchives BPB gain per 1B total token slots; and
- the complete adaptation-versus-retention vector, without hiding it in one
  micro-average.

Before looking at extension results, set each panel's margin to
`2 * median(document-bootstrap SE_s(t))` over updates
2,618/2,856/3,094/3,218. Positive OpenArchives improvement is
`BPB_start - BPB_end`; positive retention regression is
`BPB_end - BPB_start`. Use the same frozen per-panel margins for each interval
and cumulatively. For each interval and cumulatively:

- **PASS/useful:** the OpenArchives BPB improvement interval excludes its
  minimum-improvement margin and every retention upper bound stays inside its
  non-inferiority margin;
- **FAIL:** OpenArchives does not improve beyond the margin or any retention
  lower bound exceeds its allowed regression; and
- **INCONCLUSIVE:** all other cases.

If OpenArchives improves but retention or neutral Greek materially worsens,
report a measured trade-off, not success. Declare saturation only when the
first interval passes, the second interval's OpenArchives-improvement test
fails, and retention remains non-inferior. An inconclusive second interval is
not saturation.

## 10. Launch gates and execution order

1. **Historical asset inventory:** hash surviving replay assets, Greek replay,
   StarCoder manifests, the deleted/surviving GreekMMLU evidence, tokenizer and
   the HF-only 8B init;
   record every deleted or unpinned historical asset as a reconstruction
   difference.
2. **Replication-stack decision:** freeze historical 148,480 or explicitly
   rename the study to the 148,992 production-stack near-replication.
3. **Published-overlap freeze:** verify the audited HF parent revision, native-
   suite auxiliary revision and 6.344 GB match-table SHA-256; freeze Protipa as
   permanently excluded.
4. **Complete data rebuild:** freeze the corrected 13-panel full-8B validation
   set and its exact-text exclusion manifest; inventory exact labels;
   verify/apply published raw-text-hash exclusions before E001; regenerate and
   freshly scan GreekMMLU over the selected Greek streams; normalize the exact
   selected replay through the heterogeneous adapter, filter it against the
   native-suite union and then GreekMMLU, apply Stage B, and audit those exact
   Stage-B bytes against both query unions. Only after both post-Stage-B audits
   are zero, split replay into foreign and Old-Greek streams, tokenize all four
   streams, and build Phase-1/2 payloads. Require zero selected excluded
   documents and complete count/hash reconciliation.
5. **Weighted-blend freeze:** freeze the phase-specific blend strings,
   `DATA_SEED=20260609`, randomized GPTDataset mode, exported index-cache hashes,
   replay-split receipt and generalized phase-local restart contract. Do not
   substitute the later explicit-schedule reader. Phase-3 capacity is a
   separate pre-extension gate after realized Phase-2 ids exist.
6. **Initialization materialization:** convert the pinned HF 8B TD init back to
   Megatron, round-trip it, regenerate the 1.5B layer-6 snippet corpus and TD
   init, and pass preservation, coverage and frozen row-norm gates using the
   new 1.5B bridge/harness.
7. **Evaluation build:** implement and freeze the clean offline-panel scorer,
   nested sentinel builder, bootstrap validator, historical public evaluator,
   finalizer selection guard and native-suite code bundle. Building subsets is
   pre-launch; same-stack sentinel validation happens after the first four
   checkpoints exist.
8. **Static contract tests and runtime freeze:** pin training code
   `c92402e39ef3c8e69ea378a59e79059dc14541f4`; build a new read-only training
   clone on one `debug` node; apply only the historically proven named
   extra-validation patch with SHA-256
   `2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6`;
   freeze its exact three-file git diff and runtime receipt; freeze the
   scale-aware bakeoff trainer and Transformer-Engine guard inside the code
   bundle; do not apply the unrelated exact-evaluation-iteration patch;
   verify that model recipes differ only in declared geometry, init hash,
   promoted parallel decomposition, microbatch/accumulation and selected peak
   LR while the global batch and scientific recipe remain fixed.
9. **Allocation-free launch preflight:** on the login/control path, resolve and
   hash every load path, read the actual model/checkpoint/config/tokenizer/cache
   metadata, import the frozen runtime modules, verify tensor and parallel
   geometry, simulate scheduler and phase-local cursor transitions, and render
   the exact segment environment and command. Freeze that contract before
   `sbatch`; the allocated trainer must byte-match it. A missing path, import or
   configuration mismatch is our preflight failure, never a reason to request
   a GPU allocation.
10. **Reuse proven 8B execution evidence:** retain the already-promoted
    16-node TP2/DP32 profile and historical restart evidence when its producer
    compatibility check passes. Do not rerun a proof-only 8B allocation.
11. **1.5B LR pilot and profile selection:** submit three independent
    receipt-bound `normal`
    allocations on the promoted profile, one per frozen LR candidate and the
    same 238-update HPLT prefix; do not evaluate GreekMMLU. These are actual
    scientific experiments, and their measured memory/throughput evidence also
    selects the conservative 1.5B profile. Do not add a separate profile or
    restart-proof allocation.
12. **Production timing:** derive 8B timing from the compatible completed runs
    and 1.5B timing from the actual LR pilots, including startup, checkpoint and
    control overhead; freeze conservative allocation budgets for both scales.
13. **Launch main segmented trajectories:** run the historical segmented
    harness for `R-HG-8B` and `R-HG-1p5B`, concurrently only when disjoint
    audited allocations are available.
14. **Same-stack sentinel calibration:** score the complete clean panel and both
    nested subsets in the early and late four-checkpoint windows; authorize
    4,096, expand to 8,192 or activate full-panel fallback only after both
    window tests resolve exactly as Section 8 specifies.
15. **Replication-endpoint freeze:** at update 3,218, verify complete
    model/optimizer/RNG/phase-local data-cursor receipts before any extension
    update becomes authoritative.
16. **Extension:** enumerate the Phase-2 realized-document ledger, build the
    separate unseen Phase-3 blend/cache, consume it from cursor zero at constant
    floor LR, and force exact saves at 3,456 and 3,694; prove both phase-local
    transitions before the corresponding segment is authorized.
17. **Evaluation and finalization:** complete all source, GreekMMLU and native-
    Greek receipts; enforce selection authorization; publish raw per-question
    predictions; and generate separate A, B and C verdicts.

No launch gate may be stamped true by construction. Every gate requires the
path and SHA-256 of its backing receipt and the immutable executable bundle.
The gates are chronological: `pre_main`, `pre_extension`,
`pre_second_extension`, and `pre_finalization`. Post-update artifacts are never
required by an earlier gate.

## 11. CSCS resource policy

- Use one-node `debug` jobs for source inspection, receipts, sentinel building,
  conversion smokes, evaluation control and metadata work that fit the live
  debug wall limit.
- The exact 1.5B Token-Distillation command measured 8.4 batches/s over a
  54,203-batch epoch, implying about 110 minutes including startup. This does
  not fit the 90-minute `debug` ceiling. Run that unchanged scientific command
  through `build_1p5b_td_init_normal.sbatch` on one `normal` node with a
  2:30 allocation; both resource wrappers enter the same immutable
  `run_1p5b_td_init_common.sh` body.
- On 2026-08-14 the observed `debug` limits were 1:30:00 wall,
  `MaxJobsPU=1` and `MaxSubmitPU=2`; refresh them live before submission and
  size the serialized evaluation backlog explicitly.
- Use `normal` for production training, the 1.5B LR pilot and bounded
  distributed parity/profile tests that cannot fit on `debug`.
- Keep the proven 8B TP2/PP1/DP32 16-node profile unless a different profile
  passes the existing scientific parity gate.
- Benchmark a separate 1.5B profile; do not assume that assigning more GPUs is
  efficient. Global batch and sample order remain fixed regardless of the
  selected decomposition.
- Use the already-proven segmented historical harness by default. No existing
  receipt proves a single allocation that performs the phase switch and
  extension correctly, so do not put new single-allocation machinery on the
  launch critical path.
- Treat historical 8B timing as a geometry transfer from the receipted sibling
  curriculum arms and the identical-geometry audit; the execution log predates
  the selected beta2 run and is not a direct wall-time receipt for it.
- Keep **at most one pending successor total**. The pending successor is the
  next checkpoint-gated segment; never prequeue another job beyond it.
- Submit the successor as a direct delayed `normal` holder with
  `after:<source-job>+<derived-minutes>`. Derive its maximum hold from target
  p90 runtime plus reserve, and its trigger from the source segment's frozen
  conservative wall-time budget—not from compute-only seconds/update.
- A continuation holder may start early only within the audited unused-time
  bound. It must verify the signed update-3,218 checkpoint permit, the exact
  blend/cache suffix receipt and sufficient live allocation time for target p90
  runtime plus reserve before training.
- Run evaluations on debug nodes without occupying or delaying a production
  training allocation. Submit only the next dependency-ready evaluation under
  the observed per-user limits; do not flood a 70-job queue.

Any completion estimate remains `planning_only` until production-equivalent
1.5B and 8B measurements exist. Report compute-complete, training-complete and
evidence-complete times separately.

## 12. Required deliverables

- immutable experiment contract for both cells;
- historical-asset inventory distinguishing surviving, deleted and unpinned
  inputs;
- frozen complete benchmark union and per-task revision/example manifests;
- verified HF dataset/audit revisions and published match-table hash;
- exact observed source-label inventory and HPLT/OpenArchives population-
  reconstruction-difference receipt;
- per-benchmark, per-pool published-document exclusion receipts;
- regenerated GreekMMLU query receipt plus fresh per-stream scan receipts;
- heterogeneous replay-schema adapter, tests and complete replay scan receipt;
- receipt-bound 8.5B HPLT and 3.7B OpenArchives 16-shard selection manifests,
  including the lineage-only mix-builder parity result, plus the regenerated
  heldout-id, 5B replay mix-builder, phase-recipe and replay-split receipts;
- four immutable tokenized-stream receipts binding the exact HPLT,
  OpenArchives, foreign-replay and Old-Greek Stage-B JSONLs to their historical
  148,480-tokenizer Megatron `.bin/.idx` pairs, document counts and EOD-inclusive
  token counts;
- phase-specific weighted-blend, randomized GPTDataset and index-cache
  contracts shared by both scales;
- read-only pinned Megatron training clone with the hash-pinned named
  extra-validation patch only, exact changed-file/diff receipt, and a
  scale-aware trainer plus runtime guard frozen in the code bundle;
- pinned 148,480 tokenizer, HF-init recovery, HF-to-Megatron conversion and
  round-trip receipts;
- regenerated TD snippets, token-id selection difference ledger and frozen
  pre-init row-norm acceptance contract;
- geometry-aware 1.5B TD bridge, round-trip verifier and restart-parity harness;
- reconstructed source/document/token manifests, Phase-1/2 realized-document
  ledger, and a separate Phase-3 unseen blend/cache consumed from cursor zero;
- unseen OpenArchives/replay capacity including the c92402e `1.005` component
  construction margin, and no-within-Phase-3-repeat receipt;
- generalized phase-local cursor guard and Phase-2/Phase-3 restart receipts;
- constant-floor extension scheduler environment and no-reanchor guard;
- post-anonymization/tokenization lineage anti-join proving zero selected
  excluded documents;
- exact training and heldout content-overlap audit;
- exact nine-panel historical online-validation binary inventory with full
  hashes; it is comparability-only and never selection-authorized;
- frozen historical nine-panel online contract and new offline selection-panel
  manifests with explicit `selection_authorized` fields;
- nested sentinel builder, deterministic sampling receipt, same-stack
  calibration receipt and fallback state;
- exact choice-NLL/bootstrap/statistical-decision contract;
- exact 16,632-question legacy evaluator contract and BF16 receipt;
- frozen native-suite evaluator code tree, contract and tree hash;
- 1.5B LR-pilot decision receipt;
- 1.5B pilot allocation contract and promoted execution-profile receipt;
- per-scale execution-profile and restart-parity receipts;
- per-scale training run permits binding the promoted nodes/TP/microbatch and
  exact selected peak/floor LR to the immutable command bundle;
- ordered phase data-path specs, one-time component hashes, explicit
  `--data-cache-path` roots and cache-build receipts;
- update-3,218 full-state checkpoint and signed continuation permit per scale;
- structural checkpoint audits covering DCP model/optimizer/RNG metadata,
  scheduler/global cursor, source-phase cache, finite/skipped-update logs,
  storage-range completeness and read-only checkpoint files;
- source-conditioned loss trajectory through update 3,694;
- authorized 4,096/8,192 sentinel or full-panel predictions for every
  decision-bearing checkpoint;
- complete clean GreekMMLU endpoint/candidate predictions;
- complete benchmark-clean native-Greek suite predictions and compatibility
  strict-filtered views;
- separate A replication, B scale-mirroring and C continuation analyses in
  Markdown and single-page HTML;
- raw job ids, code bundle hashes, checkpoint hashes and CSCS evidence roots.

## 13. Staged authorization checklists

Every stage is fail-closed against its own receipt-backed artifact set. An
artifact that only exists after training cannot deadlock an earlier gate.

### Pre-main launch

- [ ] historical/fallback tokenizer, both base revisions and both TD inits are
      pinned, preserved and round-trip verified;
- [ ] historical inventory records every surviving, deleted or reconstructed
      asset, including deleted Stage-A GreekMMLU queries;
- [ ] audited HF revision and native-suite payload hashes are verified against
      pristine raw UTF-8 rows before E001;
- [ ] exact source labels/counts and the v2 population reconstruction
      difference are receipted;
- [ ] regenerated GreekMMLU queries freshly scan every rebuilt stream;
- [ ] all selected external replay passes the heterogeneous benchmark adapter;
- [ ] Protipa is frozen excluded, no second global dedup or new sanitation is
      introduced, and Stage-B is verified a no-op on anonymized v2;
- [ ] post-tokenization anti-joins prove zero selected benchmark and heldout
      documents;
- [ ] Phase-1/2 weighted blends, randomized mode, seed and GPTDataset caches are
      identical across scales; Phase-2 reset/restart guards pass;
- [ ] each training command carries a run permit that exactly matches its
      promoted profile and selected LR; arbitrary candidate-grid values are
      not accepted by segment preflight;
- [ ] the pinned training Megatron clone is read-only, its live diff matches
      the receipt-bound named-validation patch, and the scale-aware trainer and
      runtime guard match the immutable code-bundle hash;
- [ ] offline panels, nested sentinels, historical public evaluator,
      statistical contract and immutable evaluation code are frozen;
- [ ] batch-1 FP32 clean scorer and batch-16 BF16 legacy scorer remain separate;
- [ ] 8B restart parity, 1.5B profile/restart parity and benchmark-free LR pilot
      pass on their proper `normal` profiles;
- [ ] production timing, one-successor allocation schedule, dry-run dependency
      graph, immutable code bundle and explicit owner production authorization
      all pass.

### Pre-extension (after both update-3,218 permits)

- [ ] both complete update-3,218 model/optimizer/RNG permits exist;
- [ ] 8B and 1.5B realized sample ledgers match through update 3,218;
- [ ] early and late same-stack sentinel tests resolve to `4096_pass`,
      `8192_pass` or `full_panel_required`;
- [ ] the Phase-1/2 realized-document ledger is frozen and anti-joined from the
      separate Phase-3 OpenArchives/foreign/Old-Greek pools;
- [ ] Phase-3 capacity and no-within-phase-repeat receipts pass;
- [ ] cursor-zero Phase-3 entry and constant-floor scheduler/no-reanchor smokes
      pass for both scales; and
- [ ] explicit owner extension authorization exists.

### Pre-second extension segment

- [ ] both exact update-3,456 permits exist; and
- [ ] both 3,456-to-3,457 resumes restore Phase-3 cursor 243,712 and the exact
      next samples without optimizer/RNG drift.

### Pre-finalization

- [ ] every required source-panel, clean/public GreekMMLU and native-suite
      receipt is complete and hash-bound;
- [ ] mechanical plateau confirmations and full-panel fallback obligations are
      satisfied; and
- [ ] the finalizer's `selection_authorized` receipt passes before any winner
      or A/B/C verdict is published.
