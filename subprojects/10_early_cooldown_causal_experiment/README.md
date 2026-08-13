# Causal early-cooldown branch from the 40B Apertus-8B checkpoint

This subproject implements one controlled intervention: load the parent's exact
update-9,536 checkpoint and consume the exact parent D0 samples at updates 9,537
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

The parent checkpoint receipt hashes all 131 checkpoint files, not only
`.metadata`. Its distributed-checkpoint inventory contains 230 logical state
entries, including 52 optimizer, two RNG and two rerun-state entries.

Two rejected attempts established that a historical gradient norm is not a
cross-allocation restart invariant. A direct update-9,536 reload matched the
target counts, bytes, losses and parameter norm but not the historical gradient
norm. Replaying from update 8,000 was exact for its first update, then accumulated
BF16 cross-allocation trajectory drift from the second update. The project did
not loosen the numerical tolerance after observing either result.

Instead, the production gate performs a paired causal check on the same 16-node
allocation. It loads the fully hashed update-9,536 checkpoint once under the
parent peak-LR schedule and once under the early-cooldown schedule. At update
9,537 every pre-step field—including loss components, token counts/bytes,
gradient norm and parameter norm—must match exactly; only the logged LR may
differ. The cooldown probe saves update 9,537 and that exact optimizer/RNG/data
state becomes the branch, so the gate itself does not add a throwaway scientific
update.

This removes the unnecessary 1,536-update replay allocation. Short preparation,
orchestration, conversion and evaluation work runs on Clariden `debug`; only one
16-node `normal` allocation is requested for the branch. It requests one leaf
switch, DP32/TP2 and `B:USR1@600`. A graceful-stop receipt can launch one bounded
recovery allocation if the 12-hour segment ends before update 13,193.

## Measurements

Checkpoints are saved at updates 10,728, 11,920, 13,112 and 13,193. Each receives
clean GreekMMLU and all 13 per-document learning/forgetting panels. The endpoint
also receives DemosQA, Medical MCQA, ASEP, GPCR and OYXOY through the previously
parity-gated FP32 evaluator. Protipa remains access-blocked and non-blocking.
Checkpoint averaging is disabled.

The normative numerical and artifact bindings are in
[`configs/experiment_contract.json`](configs/experiment_contract.json).
