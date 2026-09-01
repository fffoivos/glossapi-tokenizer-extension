# Retrospective token-accuracy plan for the main Apertus 8B run

**Prepared:** 2026-09-01  
**Run:** `07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12`  
**Status:** Ready to implement and qualify; **not yet ready for production submission**  
**Default execution target:** one Clariden `debug` node at a time, account `a0140`

## 1. Decision summary

The scientific inputs required to reconstruct top-1 next-token accuracy are present:

- all 19 authoritative Hugging Face checkpoint exports, including initialization and the final checkpoint;
- the frozen 13-panel validation corpus and its manifest;
- the exact tokenizer used by the run;
- a working document-local NLL/BPB scorer whose existing model forwards can also produce top-1 correctness counts;
- the current Clariden PyTorch uenv; and
- measured wall times from prior full-checkpoint evaluations on the same data and hardware geometry.

The missing pieces are operational, not scientific data gaps:

1. add exact top-1 correctness counters and a versioned receipt schema to the experiment-owned scorer;
2. freeze a checkpoint-to-export manifest and an immutable code bundle;
3. qualify the changed scorer first on a smoke panel and then on one full terminal checkpoint;
4. obtain exact canonical preflight and `sbatch --test-only` receipts; and
5. only after those gates pass, submit the remaining checkpoint evaluations.

The proposed production geometry is **one node with all four GH200 GPUs**, with four validation panels scored concurrently and the four panel groups processed sequentially. The peak requirement is therefore **1 CSCS node / 4 GPUs**, not 19 simultaneous nodes.

The full default campaign requests **27.25 node-hours (109 GPU-hours)**, including the smoke test. A timing-based central estimate is **21.45 node-hours (85.82 GPU-hours)**. These are allocation-hours, not a claim about queue wait time.

## 2. Metric being reconstructed

The metric is **retrospective document-local top-1 token accuracy**:

\[
\text{accuracy} = \frac{\sum_t \mathbf{1}[\arg\max_v z_{t,v} = y_t]}{\sum_t 1}
\]

where `z` is the checkpoint's next-token logit vector and `y` is the observed next token. Scoring preserves the existing contract:

- teacher-forced causal language modelling;
- BOS/EOS document context;
- no context carried across documents;
- text target positions only; and
- the frozen tokenizer and validation panels.

This is not the original training-stream accuracy, because that metric was not logged during training. It is a comparable checkpoint-by-checkpoint evaluation series on a fixed held-out corpus.

The headline values should be micro-averaged within each panel and should include:

- overall correct targets / overall targets;
- base-vocabulary correct targets / base-vocabulary targets; and
- added-token correct targets / added-token targets.

Panel curves are the primary result. A global micro-average may be included as a secondary summary. A macro-average across panels must be labelled explicitly and must not replace the panel-level evidence.

## 3. Audited inventory

### 3.1 Live Clariden facts

Live probe observed at `2026-09-01T10:56:05Z`:

| Item | Verified value |
|---|---|
| SSH identity | `fffoivos` |
| Slurm account | `a0140` |
| `debug` partition | up, `01:30:00` maximum |
| `debug-qos` user limits | one running job, two submitted jobs |
| GPUs per Clariden node | 4 |
| Current working uenv | `pytorch/v2.9.1:v2`, image ID `c05f143d5fbf0927` |
| Live queue at probe time | empty |

The live receipt is:

`/Users/foivoskarounos-zamparloukos/Projects/apertus-cscs-efficiency/evidence/skill_guidance/probes/live_clariden.latest.json`

The queue observation is time-sensitive and must be refreshed immediately before submission. The partition/QoS limits, rather than the empty queue observation, determine the default serial schedule.

### 3.2 Frozen scientific inputs

| Input | Evidence and result |
|---|---|
| Run root | `/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12` |
| Validation manifest | `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260808T064500Z-d0-v4-v45bridge/validation/validation_manifest.json` |
| Validation panels | 13 panels, 59,749 documents |
| Manifest token count | 116,644,405 |
| Historical scorer target count | 116,584,656 |
| Token-count difference | 59,749, exactly one per document |
| Raw validation size | 656,536,049 bytes |
| Overlap audit | zero overlap reported for every panel |
| Tokenizer | `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992` |
| `tokenizer.json` SHA-256 | `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b` |
| Checkpoint exports | all 19 authoritative HF export directories present |
| Model size | about 16,419,542,007 bytes per checkpoint; four safetensor shards |
| Existing checkpoint evaluations | about 581 GB |
| Existing checkpoint storage | about 3.1 TB |

The one-token-per-document difference is consistent with the historical scorer excluding the first document-context token from prediction targets. Resource calculations use the measured **116,584,656 scored targets**, while the implementation gate requires reproducing this count exactly. Do not silently substitute the manifest's broader token count.

All 13 JSONL files passed live existence and size checks. Their full content hashes were not recomputed on the login node. The first debug allocation must verify every manifest hash before scoring.

### 3.3 Checkpoint matrix

Training tokens are computed as:

\[
\text{tokens at checkpoint} = \text{iteration} \times 4{,}194{,}304
\]

| Point | Iteration | Computed training tokens | Billions of tokens |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.000000 |
| 2 | 400 | 1,677,721,600 | 1.677722 |
| 3 | 1,192 | 4,999,610,368 | 4.999610 |
| 4 | 2,384 | 9,999,220,736 | 9.999221 |
| 5 | 3,576 | 14,998,831,104 | 14.998831 |
| 6 | 4,768 | 19,998,441,472 | 19.998441 |
| 7 | 5,960 | 24,998,051,840 | 24.998052 |
| 8 | 7,152 | 29,997,662,208 | 29.997662 |
| 9 | 8,344 | 34,997,272,576 | 34.997273 |
| 10 | 9,536 | 39,996,882,944 | 39.996883 |
| 11 | 10,728 | 44,996,493,312 | 44.996493 |
| 12 | 11,920 | 49,996,103,680 | 49.996104 |
| 13 | 13,112 | 54,995,714,048 | 54.995714 |
| 14 | 14,304 | 59,995,324,416 | 59.995324 |
| 15 | 14,627 | 61,350,084,608 | 61.350085 |
| 16 | 15,496 | 64,994,934,784 | 64.994935 |
| 17 | 16,688 | 69,994,545,152 | 69.994545 |
| 18 | 17,880 | 74,994,155,520 | 74.994156 |
| 19 | 18,284 | 76,688,654,336 | 76.688654 |

Initialization is stored at:

`/capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/20260731T124000Z-cpt25b-v1/hf_roundtrip`

The other 18 exports are under the run's `checkpoint_evaluations/iter_XXXXXXX/attempt_N/export/hf` directories. Iterations 14,627 and 15,496 use `attempt_1`; the remaining exported checkpoints use `attempt_0`. The checkpoint manifest must bind exact paths and export-receipt hashes rather than infer an attempt number at runtime.

## 4. Required implementation

The current experiment-owned scorer is:

`subprojects/07_full_8b_cpt/evaluation/score_documents_hf.py`

It already performs the required model forward pass and computes document-local NLL/BPB. Add, without changing the loss path:

```python
predicted = logits.argmax(dim=-1)
correct = predicted.eq(labels)
```

Then accumulate integer correct/target counters overall and for the existing base/added vocabulary split (`added_token_start = 131072`). Emit counters in both the per-document JSONL and a versioned aggregate receipt. Accuracy is derived from integers; do not store only display-rounded percentages.

Required invariants:

- `correct_total == correct_base + correct_added`;
- `target_total == target_base + target_added`;
- `0 <= correct_* <= target_*`;
- summed per-document counters exactly equal panel receipt counters;
- the 13-panel target total is exactly 116,584,656;
- existing target/base/added token counts are unchanged;
- existing NLL/BPB results remain semantically unchanged; and
- reruns create a new immutable attempt and never overwrite an accepted result.

The `argmax` reduction traverses the vocabulary dimension and can add measurable time even though it does not add another forward pass. Its overhead is therefore a qualification measurement, not an assumption to conceal.

The canonical campaign runner should own preflight, scheduling, `sbatch --test-only`, submission, supervision, and receipts. Experiment-local code should contain only the scientific evaluator, tests, and frozen checkpoint/metric manifests. Do not add another ad hoc submission system.

## 5. Execution DAG and promotion gates

Production submission is not authorized by this document. Each phase promotes only after its receipt passes.

### Phase A — local implementation and static validation (0 CSCS nodes)

1. Add top-1 counters and version the metric receipt.
2. Add tiny deterministic fixtures with hand-computed targets and predictions.
3. Prove integer aggregation and base/added partition invariants.
4. Prove the loss path is unchanged using existing local tests and stored historical output.
5. Create an immutable 19-row checkpoint manifest binding iteration, training-token count, exact HF path, export receipt, and model/config hashes.
6. Create an experiment evaluation manifest binding the validation manifest, tokenizer hash, metric-contract version, code revision, output root, and resource profile.
7. Compile the canonical evaluation plan and run its metadata-only preflight.

**Gate A:** tests pass; all 19 paths and hashes resolve; the plan contains exactly 19 unique milestones; no checkpoint or historical result is mutated.

### Phase B — single-panel smoke test (1 debug node, 4 GPUs allocated, 20 minutes requested)

Use the final checkpoint and the `code` panel. It historically contains about 8.46 million scored targets and exercises both base and added-token counters. The job may use only one GPU, but the allocation is costed as the full one-node Clariden unit.

Before scoring, verify all frozen validation file hashes and tokenizer identity. Run the panel, recompute its aggregate from per-document output, and compare its NLL/BPB and target counts against the historical receipt.

**Gate B:** exact count invariants pass; the historical target counts match; no OOM; hashes and identities are recorded; loss parity is established at machine precision. If floating reduction is not bit-identical, determine and freeze an evidence-based tolerance from the unrounded values rather than accepting display-rounded equality.

### Phase C — full terminal-checkpoint qualification (1 debug node, 4 GPUs, 85 minutes requested)

Run all 13 panels for iteration 18,284 using the existing resource-aware pattern: four panel workers concurrently, one per GPU, with four panel groups sequentially. This run becomes the accepted production result for the terminal checkpoint if it passes; it is not repeated later.

**Gate C:** all Phase B invariants pass over 116,584,656 targets; elapsed time is at most 75 minutes, leaving 10 minutes of the 85-minute request for receipt finalization and failure handling; outputs and logs are immutable and hashed; the canonical runner records the exact `sbatch --test-only` and job receipts.

If elapsed time is greater than 75 minutes, do not launch the default suffix. Use the split-group fallback in Section 7.4.

### Phase D — remaining 18 checkpoints (18 serial debug jobs)

After Gate C, submit the other 18 one-node jobs through the canonical runner. Respect the live `debug-qos` limit by keeping at most one running and one pending. Each job independently verifies the frozen identities before scoring and writes an attempt-scoped receipt.

**Gate D:** exactly 19 accepted checkpoint receipts exist; every receipt binds the same metric, validation and tokenizer identities; target totals and aggregation invariants pass; no duplicate checkpoint is counted.

### Phase E — analysis and presentation update (0 CSCS nodes)

Aggregate only accepted receipts. Plot raw checkpoint points versus computed training tokens. Do not imply that interpolated lines are measurements. Show overall and panel-level trajectories, and label the result “retrospective document-local top-1 token accuracy.” Keep NLL and accuracy as separate metrics.

## 6. Measured timing basis

The existing evaluation executes four panel groups sequentially, with panels inside a group concurrent on four GPUs. Therefore the checkpoint wall-time predictor is:

\[
T_{checkpoint} = \sum_{g=1}^{4} \max_{p \in g}(T_p)
\]

Measured historical values using the same 13 panels and model geometry:

| Checkpoint | Evidence type | Full-checkpoint wall equivalent |
|---|---|---:|
| Initialization | maximum panel time summed over four groups | 4,025.54 s = 67.09 min |
| Iteration 14,627 | completed Slurm job `3046003` | 3,625 s = 60.42 min |
| Iteration 18,284 | completed Slurm job `3053602` | 3,651 s = 60.85 min |

Mean of the two actual completed continuation jobs:

\[
\frac{3625 + 3651}{2} = 3638\text{ s} = 60.6333\text{ min}
\]

The initialization value is retained as the slower measured reference. The top-1 implementation has not yet been benchmarked, so the overhead percentages below are explicit provisional planning assumptions to be replaced by the Phase B/C receipts.

## 7. Resource calculation

### 7.1 Geometry and peak demand

Per full checkpoint:

- nodes: 1;
- GPUs: 4;
- CPUs requested by the established job shape: 288;
- memory requested by the established job shape: 450 GB;
- partition: `debug`;
- requested wall time: 85 minutes.

Because `debug-qos` permits only one running job for this user, the planned peak is:

\[
\boxed{1\text{ node} \times 4\text{ GPUs}}
\]

There are 19 full-checkpoint jobs in total, including the Phase C qualifying terminal result, plus one 20-minute smoke job.

### 7.2 Central timing estimate

Assume a provisional 10% `argmax` overhead on the mean measured Slurm wall time:

\[
60.6333 \times 1.10 = 66.6967\text{ min/checkpoint}
\]

Then:

\[
19 \times \frac{66.6967}{60} + \frac{20}{60}
= 21.4539\text{ node-hours}
\]

With four GPUs per node:

\[
21.4539 \times 4 = 85.8158\text{ GPU-hours}
\]

Rounded central estimate: **21.45 node-hours / 85.82 GPU-hours**.

### 7.3 Conservative estimate and requested budget

Use the slower 67.0923-minute historical reference, add 15% provisional accuracy overhead, and add five fixed minutes for startup, validation, and receipts:

\[
67.0923 \times 1.15 + 5 = 82.1562\text{ min/checkpoint}
\]

Campaign estimate:

\[
19 \times \frac{82.1562}{60} + \frac{20}{60}
= 26.3495\text{ node-hours}
\]

\[
26.3495 \times 4 = 105.398\text{ GPU-hours}
\]

Each full job will request 85 minutes, so the auditable requested allocation is:

\[
19 \times \frac{85}{60} + \frac{20}{60}
= \boxed{27.25\text{ node-hours}}
\]

\[
27.25 \times 4 = \boxed{109\text{ GPU-hours}}
\]

Thus the default request is **20 one-node allocations**: 19 full evaluations and one smoke allocation. Only one is intended to run at any time.

The estimated serial compute time is about 21.5 hours centrally and 26.4 hours conservatively. Queue delay and scheduler gaps are reported separately; they are not node-hours and are not included in those totals.

### 7.4 Split-group fallback

If Phase C exceeds 75 minutes or leaves inadequate receipt reserve, split each checkpoint into four independently receipted panel-group jobs while retaining one-node peak usage:

| Group | Requested time |
|---|---:|
| 0 | 25 min |
| 1 | 35 min |
| 2 | 30 min |
| 3 | 30 min |
| Total/checkpoint | 120 min = 2 node-hours |

Worst-case requested fallback budget:

\[
19 \times 2 + \frac{20}{60} = 38.3333\text{ node-hours}
\]

\[
38.3333 \times 4 = 153.333\text{ GPU-hours}
\]

This creates 76 group jobs plus the smoke job and increases scheduler overhead, so it is a safety fallback rather than the default.

Using `normal` nodes merely to parallelize this short evaluation campaign is not the default: each checkpoint is expected to fit the 90-minute `debug` limit, and the live QoS deliberately serializes the campaign. A deadline-driven `normal`-partition alternative would require a separate explicit resource and queue review.

## 8. Storage and I/O

No new model conversion is required; the 19 HF exports already exist. Historical per-checkpoint evaluation output is about 25.04 MiB, so expected new metric output is:

\[
19 \times 25.04\text{ MiB} \approx 475.76\text{ MiB} < 0.5\text{ GiB}
\]

Allow **1 GiB** for immutable per-document results, receipts, manifests, and logs. Verify free space and inode availability in preflight. Do not copy the 19 model exports into a new campaign directory.

## 9. Exact pre-apply checklist

Before any production submission:

- [ ] evaluator tests and metric invariants pass locally;
- [ ] code bundle, metric contract, tokenizer, validation manifest, and 19 checkpoint exports have immutable identities;
- [ ] every validation JSONL hash is verified inside the smoke allocation;
- [ ] canonical plan compiles to exactly 19 unique milestones;
- [ ] login-node metadata preflight passes without loading model weights;
- [ ] Phase B smoke receipt passes;
- [ ] Phase C full terminal receipt passes and records measured accuracy overhead;
- [ ] the remaining budget is recalculated from the measured Phase C wall time;
- [ ] live partition, QoS, account, queue and uenv facts are refreshed;
- [ ] an exact `sbatch --test-only` receipt exists for the immutable evaluator bundle;
- [ ] the test-only operation is proven not to mutate manifests or accepted results;
- [ ] output roots are attempt-scoped and collision-safe;
- [ ] production application is separately and explicitly authorized.

## 10. Completion receipts

The campaign is complete only when the final aggregate can be rebuilt from immutable inputs and receipts. Preserve:

- checkpoint manifest and its digest;
- validation manifest and all verified file hashes;
- tokenizer path and hash;
- code revision/bundle digest and uenv image ID;
- exact `sbatch --test-only` and submitted-job receipts;
- Slurm job IDs, resource shapes, elapsed times and terminal states;
- per-document result hashes and per-panel aggregate receipts;
- exact integer correct/target counters for overall/base/added tokens;
- NLL/BPB parity evidence; and
- a final 19-row accepted-result manifest used by the plot.

The result can then support a defensible statement to the Apertus community: accuracy was not recorded during training, but it was reconstructed consistently at all preserved checkpoints on the same frozen validation corpus.
