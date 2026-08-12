# Three-checkpoint native-Greek evaluation

This directory freezes the requested first/40B/final screen for DemosQA,
Medical MCQA, ASEP MCQA, GPCR and OYXOY. The first point is the iteration-zero
model; the other points are update 9,536 (39.997B token slots) and the terminal
update 18,284 (76.689B token slots).

The execution shape is one Clariden debug node and three independent evaluator
processes, each pinned to a different GH200. Each process loads its checkpoint
once and runs every frozen benchmark. This removes repeated model loads and
avoids waiting for three separate allocations. A fourth GPU remains unused;
splitting one checkpoint over it would require a duplicate model load and would
not reduce the critical path evenly.

OYXOY is reported as four zero-shot base-model tasks: multilabel NLI through
three independent binary decisions, definition-based WSD, words-in-context and
metaphor detection. This is a causal-LM checkpoint comparison, not a
reproduction of OYXOY's supervised encoder training.

Text-only Protipa remains in the frozen contract but is not in the materialized
examples while the current Hugging Face token lacks approval for its manual
gate. It must be appended with the same source revision and scorer after access
is granted; the other five benchmarks do not wait on it.
