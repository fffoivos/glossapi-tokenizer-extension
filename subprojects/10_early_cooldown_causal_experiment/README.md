# Causal early-cooldown branch from the 40B Apertus-8B checkpoint

This subproject implements one controlled intervention: resume the exact parent
D0 checkpoint at update 9,536 and consume the exact parent D0 samples at updates
9,537 through 13,193, but begin the parent's 3,657-update `1-sqrt` WSD-10
cooldown immediately.

Everything except the LR trajectory remains fixed, including full optimizer
and RNG state, the 18,284-update AdEMAMix alpha/beta3 ramps, tokenizer, RoPE,
Goldfish masking, batch geometry, packing, document masks and sample order. The
branch consumes 15.338569728B token slots and ends at 55.335452672B total token
slots.

## Lean execution path

The run performs no corpus preparation. It reads the immutable parent D0
schedule in prefix mode and reuses the parent checkpoint, 13-panel validation
manifest, tokenizer, Megatron runtime, GreekMMLU subset and native-Greek frozen
examples in place. It does not copy, repack, reshuffle, rededuplicate, anonymize,
decontaminate or retokenize data. The complete update-9,536 GreekMMLU and
per-document baselines are reused by hash.

The parity check runs inside the production allocation. It reloads update 9,536
under the original parent LR schedule, executes update 9,537 without saving a
throwaway checkpoint, and compares logged losses, token counts, parameter norm,
skipped/NaN counts and gradient norm with the original parent update. The branch
begins only after this passes. This costs one update, not another 16-node job.

Short preparation, orchestration, conversion and evaluation work runs on
Clariden `debug`. The 16-node `normal` allocation is reserved for the control
update and branch. It requests one leaf switch, DP32/TP2, 12 hours and
`B:USR1@600`; a graceful-stop receipt can launch one bounded recovery allocation
if needed.

## Measurements

Checkpoints are saved at updates 10,728, 11,920, 13,112 and 13,193. Each receives
clean GreekMMLU and all 13 per-document learning/forgetting panels. The endpoint
also receives DemosQA, Medical MCQA, ASEP, GPCR and OYXOY through the previously
parity-gated FP32 evaluator. Protipa remains access-blocked and non-blocking.
Checkpoint averaging is disabled.

The normative numerical and artifact bindings are in
[`configs/experiment_contract.json`](configs/experiment_contract.json).
