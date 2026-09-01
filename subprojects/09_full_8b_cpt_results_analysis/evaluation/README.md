# Three-checkpoint native-Greek evaluation

This directory freezes the requested first/40B/final screen for DemosQA,
Medical MCQA, ASEP MCQA, GPCR and OYXOY. The first point is the iteration-zero
model; the other points are update 9,536 (39.997B token slots) and the terminal
update 18,284 (76.689B token slots).

The execution shape is four chained 22-minute Clariden debug jobs. Each job
uses four nodes and up to sixteen independent FP32 evaluator processes, respecting
the live 90 node-minute-per-job QoS cap. Per checkpoint, DemosQA, the other
three MCQ suites and OYXOY NLI plus metaphor are separate shards; OYXOY WSD is
split ten ways and WIC eight ways. Every process is pinned to one GH200. A
fail-closed aggregator verifies that all sixty-three shards contain every
frozen example exactly once before it writes checkpoint-level metrics.

The authoritative scorer is the legacy full-logit implementation in FP32.
BF16 was tested and rejected by a frozen parity gate because it changed answer
rankings. An FP32 batch-one versus batch-four test preserved all sampled
predictions but missed the predeclared raw-score tolerances, so production uses
candidate batch one. Speed comes only from independent-example and
independent-checkpoint parallelism; arithmetic within each scored example is
unchanged. The matrix reads its dtype and batch sizes from an immutable
execution-profile receipt rather than from free-form overrides.

OYXOY is reported as four zero-shot base-model tasks: multilabel NLI through
three independent binary decisions, definition-based WSD, words-in-context and
metaphor detection. This is a causal-LM checkpoint comparison, not a
reproduction of OYXOY's supervised encoder training.

Text-only Protipa remains in the frozen contract but is not in the materialized
examples while the current Hugging Face token lacks approval for its manual
gate. It must be appended with the same source revision and scorer after access
is granted; the other five benchmarks do not wait on it.

The completed matrix is summarized in
[`../NATIVE_GREEK_3CP_RESULTS_20260812.md`](../NATIVE_GREEK_3CP_RESULTS_20260812.md).
The post-hoc contamination filter, exact immutable ids and decision boundary
are specified in
[`CONTAMINATION_DROP_DECISION_20260812.md`](CONTAMINATION_DROP_DECISION_20260812.md).

## Token-aligned D0 0.5B replication

`run_d0_0p5b_three_checkpoint_matrix.sbatch` reuses the same frozen examples,
FP32 legacy scorer, prompt serialization, metrics and contamination exclusions
for the stationary-mix D0 0.5B trajectory. Its predeclared checkpoints are
initialization, update 18,944 (39.728B tokens, the closest saved D0 checkpoint
to the 8B 39.997B point) and final update 38,496 (80.732B tokens). The model
contract is rebound explicitly because the 0.5B model has tied embeddings and
different architecture geometry; benchmark and scoring fields remain
byte-equivalent JSON values to the 8B source contract. The job uses four debug
nodes for at most 22.5 minutes and does not share nodes or dependencies with a
training allocation.

When another campaign occupies the account's two submitted debug-job slots,
`coordinate_d0_0p5b_matrix.sh` waits on the Mac and submits only after one
slot is released. It additionally requires the active campaign's long-running
training guard to be pending or to retain at least one hour. If the D0 job
does not start within five minutes, the coordinator cancels only that D0 job
so it cannot block the active campaign's next evaluation submission.
