# Causal early-cooldown branch from the 40B Apertus-8B checkpoint

This subproject implements one controlled intervention: reconstruct the exact
parent state at update 9,536 from the parent's proven segment-start checkpoint
at update 8,000, then consume the exact parent D0 samples at updates 9,537
through 13,193 while beginning the parent's 3,657-update `1-sqrt` WSD-10
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

The first attempted direct reload of update 9,536 was correctly rejected: its
target counts and bytes were exact, but the gradient norm was 2.016 rather than
the parent's 0.669. The repaired path never relaxes that gate. Allocation one
reloads the exact update-8,000 segment boundary, gates update 8,001, replays the
parent peak-LR trajectory through 9,537, saves synchronously at 9,536 and checks
the parent rows at 8,001, 9,536 and 9,537. Allocation two first reloads that new
9,536 checkpoint and gates its pre-update loss and gradient against the replay;
only then does the cooldown branch begin.

Short preparation, orchestration, conversion and evaluation work runs on
Clariden `debug`. The two 16-node `normal` allocations are reserved for verified
replay and the branch. The branch holder is requested 200 minutes after the
replay allocation starts. Its maximum early hold is 4,200 seconds, leaving a
frozen 37,800-second training budget plus a 1,200-second reserve. Both request
one leaf switch, DP32/TP2 and `B:USR1@600`; a graceful-stop receipt can launch
one bounded recovery allocation if needed.

## Measurements

Checkpoints are saved at updates 10,728, 11,920, 13,112 and 13,193. Each receives
clean GreekMMLU and all 13 per-document learning/forgetting panels. The endpoint
also receives DemosQA, Medical MCQA, ASEP, GPCR and OYXOY through the previously
parity-gated FP32 evaluator. Protipa remains access-blocked and non-blocking.
Checkpoint averaging is disabled.

The normative numerical and artifact bindings are in
[`configs/experiment_contract.json`](configs/experiment_contract.json).
