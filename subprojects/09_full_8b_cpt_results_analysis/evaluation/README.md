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
