# Three-checkpoint native-Greek evaluation

This directory freezes the requested first/40B/final screen for DemosQA,
Medical MCQA, ASEP MCQA, GPCR and OYXOY. The first point is the iteration-zero
model; the other points are update 9,536 (39.997B token slots) and the terminal
update 18,284 (76.689B token slots).

The execution shape is two Clariden debug nodes and six independent evaluator
processes. Each checkpoint has a core shard and an OYXOY WIC shard, so all
three checkpoints and both task groups run concurrently. Every process is
pinned to one GH200; this uses six of the eight allocated GPUs and keeps the
two uneven shards from serializing the critical path.

The authoritative scorer is the legacy full-logit implementation in FP32.
BF16 was tested and rejected by a frozen parity gate because it changed answer
rankings. The production candidate batch is four only if an FP32 batch-one
versus batch-four gate preserves every sampled prediction and stays inside the
predeclared score tolerances. The matrix reads its dtype and batch sizes from
that immutable execution-profile receipt rather than from free-form overrides.

OYXOY is reported as four zero-shot base-model tasks: multilabel NLI through
three independent binary decisions, definition-based WSD, words-in-context and
metaphor detection. This is a causal-LM checkpoint comparison, not a
reproduction of OYXOY's supervised encoder training.

Text-only Protipa remains in the frozen contract but is not in the materialized
examples while the current Hugging Face token lacks approval for its manual
gate. It must be appended with the same source revision and scorer after access
is granted; the other five benchmarks do not wait on it.
