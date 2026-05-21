"""Scaffold — post-conversion patcher for xIELU + QK-Norm tensors.

NOT IMPLEMENTED. This is a placeholder + design doc for the patcher that
closes RISKS.md R17 (xIELU αp / αn / β / ε and QK-Norm q_norm / k_norm
land at XIELU.__init__ / RMSNorm.__init__ defaults after HF → Megatron
conversion via convert.py + saver_core, because saver_core.py:L357-443
has no protocol slot for these Apertus-specific keys).

Use case: AFTER `convert.py --loader apertus_hf --saver core` produces
a Megatron `torch_dist` directory, run this script to overwrite the
extras-tensors with their HF-source values. For the bakeoff this is
optional (all three arms lose the same defaults → comparison still
valid). For production CPT it's required (fidelity loss vs Apertus base).

Design sketch:

  1. Load HF Apertus: tok + ApertusForCausalLM (bf16).
  2. Open the Megatron save_dir. With --ckpt-format torch_dist, the format is
     one `iter_*/__*.distcp` per (pp, tp, dp, ep) shard.
  3. For each layer i:
     a) Locate the shards holding the i-th transformer layer's tensors.
     b) Read the relevant per-layer state_dict shards.
     c) Overwrite, per layer:
        - encoder.layers[i].mlp.activation_func.alpha_p ← HF
            model.layers.{i}.mlp.act_fn.alpha_p
        - encoder.layers[i].mlp.activation_func.alpha_n ← HF
            model.layers.{i}.mlp.act_fn.alpha_n
        - encoder.layers[i].mlp.activation_func.beta ← HF (if present)
        - encoder.layers[i].mlp.activation_func.eps ← HF (if present)
        - encoder.layers[i].self_attention.q_layernorm.weight ← HF
            model.layers.{i}.self_attn.q_norm.weight
        - encoder.layers[i].self_attention.k_layernorm.weight ← HF
            model.layers.{i}.self_attn.k_norm.weight
     d) Write the shards back.
  4. Verify by reloading the Megatron model + sampling logits on Greek
     and English; should now match unmodified Apertus (modulo bf16 noise).

Implementation requires understanding Megatron's distributed-checkpoint
shard layout — easiest path is to use Megatron's own
`megatron.core.dist_checkpointing.serialization.load/save` helpers from
inside an MPU-initialized process (single-rank works for read+modify+write).

This scaffold is committed so the task is tracked + the design is
recorded; actual implementation needs Clariden + actual Apertus weights
+ Megatron environment and so is deferred to production-CPT prep.

Status: SCAFFOLD ONLY — see RISKS.md R17.
"""
from __future__ import annotations
import sys


def main() -> int:
    print(__doc__, file=sys.stderr)
    print("\nERROR: patch_apertus_extras.py is a scaffold, not yet implemented.", file=sys.stderr)
    print("       For bakeoff: skip this step (R17 is acceptable across the 3 arms).", file=sys.stderr)
    print("       For production CPT: implement before submission.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
