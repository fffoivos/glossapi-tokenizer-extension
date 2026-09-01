# Targeted Apertus 8B CPT experiment and resource plan

Date: 2026-08-11  
Status: Experiment A implementation in progress; training is not launch-authorized

Scope update, 2026-08-12: Experiment A remains active. The former update-9,536
continuation Experiment B was dropped by the owner and is retained below only
as an archived design and reproducibility record. Pending jobs `3061757` and
`3061758` were cancelled before allocation or training (`00:00:00`). The
replacement scale-predictivity proposal is in
`SCALE_PREDICTIVITY_STUDY_20260812.md` and is not launch-authorized.

## Scientific constants shared by both experiments

The tokenizer, model geometry, Token-Distillation initialization policy,
AdEMAMix optimizer, Goldfish loss, masking, batch geometry and DP decomposition
remain exactly those proven in subproject 07. The authoritative inherited
recipe is `subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json`; the
production implementation must bind the corrected immutable v45-or-later code
bundle, not the older local Slurm defaults.

| Field | Frozen value |
|---|---:|
| Tokenizer | `fffoivos/apertus-tokenizer-extension@fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer` |
| `tokenizer.json` SHA-256 | `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b` |
| Vocabulary | 148,992; 0 padding tokens; divisible by 256 |
| Model | Apertus 8B, 32 layers, hidden 4,096, FFN 21,504, 32 heads, 8 query groups |
| RoPE | base 500,000; scaling factor 8; max position 4,096 |
| Sequence length | 4,096 |
| Global batch | 1,024 sequences = 4,194,304 token slots/update |
| Optimizer | AdEMAMix; beta1 0.9, beta2 0.999, beta3 0.999, alpha 4.0 |
| Peak LR / floor | 5.5e-5 / 5.5e-6 (WSD-10) |
| Loss | Goldfish k=50, h=50 |
| Packing masks | reset attention, reset position IDs, mask EOD loss |
| Geometry | TP=2, PP=1, CP=1, DP=32 on 16 nodes / 64 GH200 GPUs |
| Checkpoint averaging | disabled |

DP64 is prohibited: its benchmark was about 1.95 times faster per update but
failed the predeclared trajectory RMSE and signed-mean parity bounds. No speed
change may be promoted without the same trajectory/restart gate.

## Immutable data authorities

The modern source authority is the public anonymized release:

- Hugging Face: `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`
- immutable revision: `987b8955fcd395c6219e39df9e64715457f69065`
- CSCS release root:
  `/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run`
- publication receipt:
  `/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run/publication/receipt.json`
- public-access receipt:
  `/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run/publication/public_access_receipt.json`
- anonymization receipt:
  `/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run/release/manifests/anonymization_manifest.json`
- tokenizer-bound counts:
  `/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run/release/manifests/token_counts.json`

The release has 51,839,746 rows and 63,780,757,593 training tokens including
one EOD per document. Its anonymization receipt records that row order,
multiplicity and content other than the Apertus-standard PII substitutions were
preserved. **This subproject must not run a second global exact or near
deduplication.** Only explicit GreekMMLU contamination removals may change the
selected training set, and every removal must appear in the decontamination
ledger.

The replay and validation authorities remain the frozen, content-clean assets
from the completed sanitized run at:

`/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3`

## Experiment A — balanced academic plus polytonic run

### Dataset

Consume each of the following selected modern documents once:

| Component | Rows | Training tokens before new decontamination |
|---|---:|---:|
| `openarchives.gr` | 126,597 | 6,132,237,924 |
| `greek_phd` | 31,692 | 3,881,474,423 |
| Academic subtotal | 158,289 | 10,013,712,347 |
| HPLT | deterministic SHA-256 quarter candidate pool, then a seeded packed prefix with active-token mass exactly equal to the post-decontamination and heldout-excluded academic subtotal | 12,156,522 candidate rows; final packed prefix frozen later | frozen later |
| Release-internal direct polytonic sources | 7,088 | 111,327,131 manifest-derived training tokens; exact count after required exclusions pending |

Experiment A uses only the pinned anonymized Hugging Face release. Its
polytonic component is every row from the four direct polytonic source datasets
that are present in that release: `1000_prwta_xronia_ellhnikhs`,
`Ekklisiastika_Keimena`, `Wikisource_Greek_texts`, and
`klasikh_arx_ell_grammateia`. The earlier standalone `poly_train` manifest is
provenance only, not a required input. Its Scholarios source is absent from the
published release and will not be fetched, recreated, or substituted. A
release-internal Unicode audit must establish polytonic evidence before this
source set is frozen. The MacBook is not a data worker.

The planning-only release-polytonic estimate is 111,327,131 training tokens.
Before
new GreekMMLU removals, that gives:

- modern Greek = 20,138,751,825 tokens;
- foreign replay = 5,098,418,184 tokens;
- Old-Greek replay = 254,920,909 tokens;
- total active = 25,492,090,918 tokens;
- 6,078 updates and 888,794 loss-inactive tail slots.

These are estimates, not launch values. `freeze_experiment_contract.py` derives
the final geometry from exact post-decontamination and exact polytonic receipts.

The schedule is a stationary, window-balanced random mixture. Modern is 79%,
foreign replay 20%, and Old-Greek replay 1%. Within modern, HPLT and the
OpenArchives-plus-PhD academic pool have equal active-token mass; the
polytonic pass is added once. Source-specific permutations are deterministic
and frozen. Packing preserves the cross-document boundaries and metadata
ledgers.

### Decontamination and validation

The selected OpenArchives, PhD, HPLT and polytonic documents are rescanned
against the exact pinned `dascim/GreekMMLU` revision
`6a03aa06b68beb932fb75edff3a34e50b3674649`. Use the existing conservative
rules in
`subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/decontaminate_full_corpus.py`:
exact long prompt, exact question plus answer, or aligned >=85% shingle and
MinHash match with the correct answer nearby. Answer-only matches remain audit
evidence and never cause removal.

The decontamination gate requires:

1. immutable query JSONL and query manifest hashes;
2. input = kept + dropped row reconciliation;
3. a per-document decision ledger bound to input text hashes;
4. zero GreekMMLU high-confidence matches in the kept output on an independent
   post-scan;
5. zero unrecorded content removals and explicit proof that no dedup stage ran.

Before packing, compare every selected document against the exact UTF-8 text
hashes of all 13 frozen validation panels and exclude exact matches with a
separate per-document ledger. This preserves the old panels as heldouts under
the new selection. It is a train/validation split operation, not
deduplication: duplicates that are not validation content retain their original
multiplicity. A post-scan must report zero selected-training/validation exact
content overlap.

Use the same 13 content-clean source-conditioned validation panels as the
sanitized run. The lightweight aggregate source-conditioned panel runs every
25 optimizer updates, exactly as the training launcher executes it. Save the
more expensive checkpoint/per-document evidence at update 0, post-warmup,
approximately every 2B active tokens, cooldown start, segment boundaries and
final. GreekMMLU runs asynchronously on debug/control resources at those
checkpoint milestones. Report accuracy, choice NLL and correct-answer BPB;
selection is not based on accuracy alone.

### LR and optimizer horizon

Warmup stays 400 updates. AdEMAMix alpha and beta3 ramps are scaled to the exact
new run horizon because A starts from the fixed Token-Distillation model and is
a fresh CPT trajectory. WSD-10 begins at `floor(0.8 * updates)` and occupies
the rest of the run with the existing `1-sqrt` shape.

At the planning 6,092-update horizon: cooldown starts at update 4,873 and has
1,219 updates. The two production segments are [0, 3,046] and [3,046, 6,092].

## Retired Experiment B — archived update-9,536 continuation design

**Retired by owner on 2026-08-12. Do not launch.** The following details are
kept so that the already-built schedule and reusable continuation builder are
not lost or silently reinterpreted.

In simple terms: start from the already best-performing checkpoint at update
9,536, take every non-HPLT packed sequence that had not yet been consumed at
that point, mix it with still-unseen replay sequences at 79/20/1, and decay the
learning rate from peak to 10% over that entire continuation. Do not restart
the optimizer warmup or reinitialize any model state.

Authority:

- parent run:
  `/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12`
- parent schedule:
  `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3/schedules/schedule_manifest.json`
- checkpoint: exact iteration 9,536 with its checkpoint receipt and completed
  GreekMMLU receipt;
- exposure evidence:
  `subprojects/07_full_8b_cpt/presentations/data/full8_checkpoint_drift_20260811/checkpoint_source_exposure.json`.

“Unseen” means the exact packed-sequence token spans after schedule slot
`9,536 * 1,024`; it does not mean that a long source document was never touched
by an earlier packed sequence. The remaining non-HPLT active-token total is
exactly 9,123,187,023. The planning 79/20/1 geometry is 2,309,667,601 foreign,
115,483,380 Old Greek, 11,548,338,004 total active tokens, 2,754 continuation
updates, and at most a sequence-granularity replay quota residual. The final
absolute model update is 12,290.

`build_continuation_b_schedule.py` proves that no selected sequence appeared in
the parent prefix, selects only post-9,536 replay sequences, preserves each
pool's original relative order, writes a compatible receipt-bound schedule,
and reports the exact realized 79/20/1 residuals. The final production recipe
must use those realized values rather than the planning numbers.

For B, the optimizer, RNG and sample cursor load from update 9,536. Alpha and
beta3 ramp state and the sanitized parent's 18,284-update ramp horizon are
preserved exactly; there is no new warmup and no ramp rescaling. WSD-10 starts
immediately at continuation update 0 and decays for
the full continuation. GreekMMLU is evaluated at the parent anchor and about
every 1B additional active tokens, plus the final checkpoint. Source validation
continues every 25 updates.

## CSCS resource and launch plan

### Preparation and gates

All schema inspection, receipt creation, source selection, GreekMMLU scan,
packing control, schedule construction, conversion control and evaluation
orchestration use one-node `debug` jobs with `--partition=debug`. Array workers
may be used only when each task fits the debug limits and the live
`debug-qos` submission ceiling is respected. No dataset preparation runs on a
normal 16-node GPU allocation.

Required launch order for each experiment:

1. debug: schema and immutable-source receipt inspection;
2. debug: selection and GreekMMLU query freeze;
3. debug/array: decontamination, post-scan and exact token counts;
4. debug/array: deterministic packing, schedule finalization and validation
   panel bindings;
5. debug: checkpoint conversion, initial source validation and GreekMMLU
   anchor;
6. normal, one leaf, 16 nodes, at most one hour: run exactly two uninterrupted
   updates, save synchronously after the first, then load that exact control
   checkpoint and repeat only the first post-checkpoint update under the frozen
   DP32 profile. The control and resumed paths share one allocation, for three
   optimizer updates total. This is the only preparation test that cannot fit
   a debug node;
7. debug: bind the restart receipt and every other receipt into the launch and
   operational gates; only then submit production-horizon training.

The restart thresholds are frozen before submission: logged loss and parameter
norm must be exactly equal; the logged gradient norm uses the already disclosed
DP32 restart bound (`atol=0.001`, `rtol=0.02`). The receipt also requires exact
checkpoint/load-view provenance, the absolute schedule cursor, zero skipped or
non-finite updates, 16 nodes and a single leaf. These thresholds must not be
edited after the result exists.

The bounded parity smoke must use a hard-pinned, audited leaf exclusion.
`--switches=1` remains mandatory but is insufficient by itself: Slurm may relax
that switch preference after its wait threshold and place the job across
several leaves. The completed training receipts must independently prove the
single-leaf allocation. Production uses the same receipt-bound placement
contract.

### Production allocations

Every production job requests `normal`, account `a0140`, 16 nodes, one leaf
switch, four ranks/GPUs per node, TP2/DP32, exclusive nodes and 450G host
memory. The established launcher uses 64 ranks as four tasks per node with one
GPU and 72 CPUs per task. The wall limit is 12 hours with a B:USR1 signal ten
minutes before timeout and synchronous resumable checkpoints.

Measured sanitized-run wall time was approximately 10.49–11.05 seconds/update;
ETA uses full observed wall time, not the faster step-time metric.

| Experiment | Updates | Normal allocations | Conservative active compute | Allocation policy |
|---|---:|---:|---:|---|
| A | about 6,092; exact after receipts | 2 x 12h | about 18.2–18.8h | [0,3046], [3046,6092] planning split |
| B (retired) | 2,754 in archived receipt | 0 | none | no launch authorized |

Each experiment additionally needs one bounded 16-node restart-parity
allocation of at most one hour after debug preparation. It runs three optimizer
updates total and is accounted separately from the production allocations
above. No other metadata, receipt, conversion, evaluation-control or packing
work may consume a `normal` allocation.

For A, budget 10h for each segment including startup, checkpoint and validation
overhead, with a 20-minute reserve. Thus:

- maximum harmless successor hold = 12h - 10h - 20m = 1h40m;
- delayed successor trigger = 10h - 1h40m = 8h20m after segment 0 starts.

Only one audited successor may be pending. It is submitted directly to
`normal` with `after:<source-job>+500minutes`; a scarce debug slot is not used as
a timer. Before it may train it must verify the immutable scientific and
operational bundle receipts, its exact manifest row, a signed checkpoint permit
and at least 10h20m of live allocation time remaining. `sbatch --test-only`
must pass without mutating the manifest before real submission.

The retired B plan requests no allocation. A new live capacity/leaf-switch
snapshot is required immediately before every real A normal submission. A
pending allocation is not evidence of capacity, and no job may silently relax
the one-leaf placement gate.

## Implementation status (2026-08-12 Europe/Athens)

- **Experiment B retirement:** the owner dropped the update-9,536 continuation
  in favor of a replication and cross-scale predictivity study. Exact pending
  production job `3061757` and dependent supervisor `3061758` were cancelled
  at `00:00:00`; no node allocation or optimizer update was consumed. Builder,
  schedule and receipt artifacts remain immutable for possible future use.

- **Scope correction (2026-08-12):** Experiment A needs no external dataset or
  historical standalone `poly_train` artifact. Its polytonic data are selected
  directly from the pinned public anonymized release, using the four named
  source datasets above. Earlier archive/path searches are superseded evidence,
  not an operational blocker. Debug job `3062017`, which began a broader
  extensionless-file search under the old interpretation, was cancelled as soon
  as this was clarified.
- Experiment A now awaits the release-internal polytonic source audit,
  extraction, post-anonymization GreekMMLU decontamination, frozen-validation
  exclusion and exact production-tokenizer count. Only then can its geometry
  and recipe be frozen.
- Experiment A academic extraction, post-anonymization decontamination and
  frozen-validation exclusion are complete and independently audited: 158,289
  inputs, 9 GreekMMLU removals, 188 validation exclusions, 158,092 retained
  documents and exactly 10,000,949,141 retained training tokens. Both kept-set
  post-scans are zero and no multiplicity-changing deduplication ran.
- Immutable scientific bundle
  `/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260812T084500Z-targeted8b-v39`
  is frozen and independently hash-verified under tree SHA-256
  `4734ab501674c6ea6b5af7ba8ce53653b50930b654bee89709b04779b4c4a131`.
  It corrects receipt evidence so the source-conditioned cadence is the exact
  executable 25-update cadence (104,857,600 active-token slots), while A and B
  record their predeclared GreekMMLU cadences as 2B/477 updates and 1B/238
  updates respectively. It also makes the exact-poly authority policy
  executable: only an existing repository, Hugging Face, or CSCS artifact is
  accepted, and reconstruction or substitution is forbidden. CSCS debug
  validator job `3061532` passed all 31 tests plus static contracts and the
  pinned Hugging Face release inspection. The bundle-bound nested submission
  proof also passed in parent/child jobs `3061533`/`3061538`.
- Experiment A HPLT extraction completed as debug job `3058831`: 12,156,522
  deterministically selected candidate rows across the expected 250 source
  shards, with row multiplicity preserved. Decontamination job `3059161`
  retained 12,156,307 and removed 215 benchmark matches. Independent audit job
  `3059186` passed with exact input/ledger reconciliation, zero residual
  high-confidence GreekMMLU matches and explicit proof that no deduplication
  ran. Frozen-validation exclusion job `3059240` completed, and independent
  audit job `3060122` proved 12,152,435 kept rows, 3,872 exclusions, zero
  frozen-validation overlap and no deduplication. Exact polytonic artifact
  resolution and counting remain before A can freeze its schedule.
- Experiment B's current exact parent-suffix schedule completed as debug job
  `3060140`:
  all 9,123,187,023 unseen non-HPLT tokens, zero prefix overlap, 2,754
  continuation updates and final absolute update 12,290. Whole-sequence replay
  rounding is -1,151 foreign and +1,692 Old-Greek active tokens. Consolidated
  asset job `3060148` froze the receipt-bound DP32 recipe and contract with
  source validation every 25 updates and the 238-update / approximately 1B
  continuation-token GreekMMLU cadence. Exact update-9,536 evaluation job
  `3059235` completed all 13 per-document panels, and corrected anchor job
  `3060178` completed source validation, GreekMMLU and conversion smoke under
  the training RoPE geometry.
  The evaluator's inherited Llama-3 scaling warning is already resolved by the
  parent's frozen `evaluator_rope_parity_v1.json`: all 64 inverse-frequency
  elements match the training runtime bit-exactly (`max_abs_difference=0.0`,
  attention factor 1.0). The warning is preserved rather than editing the
  checkpoint config. A complete v39 diagnostic launch-gate pass ran on debug
  as job `3061545`; its parity fixture is explicitly marked synthetic and
  forbidden for production. B's genuine bundle-bound, group29-pinned 16-node
  DP32 restart-parity job `3061556` is pending for `normal` capacity. Debug
  controller `3061572` is held by `afterok:3061556`; it will rebuild the launch
  and operational gates from the genuine parity receipt and may submit the
  production continuation only after those gates pass. Diagnostic operational
  gate job `3061603` and end-to-end production dry-run job `3061611` also
  passed; the latter verified the exact 9,536-to-12,290 submission with
  `sbatch --test-only` and created neither a run root nor a production job.
  A read-only all-leaf test-only comparison confirmed that retaining the
  accrued group29 request is faster than replacing it with a fresh request;
  no queue state was changed.

These blockers prevent accidental GPU launch; they do not justify changing the
dataset definition.
