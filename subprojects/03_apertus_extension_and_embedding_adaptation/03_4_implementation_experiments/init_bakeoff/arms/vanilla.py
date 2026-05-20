"""Vanilla arm: no vocab extension, no init changes.

Per v0.7 § 5 and `old_experiments_plan.md` § 5, the Vanilla arm uses
the original Apertus tokenizer (vocab 131,072) and trains the
unmodified Apertus-8B-2509 checkpoint on the Greek + replay mix.
This is the load-bearing baseline: if Vanilla matches the extension
arms on retention + Greek quality and beats them on throughput, the
extension's parameter overhead is not justified (Yuan et al. 2024,
*LLaMA Beyond English*).

This module exists for symmetry with `retok.py` and `centroid.py`
and to centralize the "Vanilla arm config" so the bakeoff harness
can spin up all three arms the same way.

Practically: there is nothing to compute. The training job for the
Vanilla arm should:
1. Load `swiss-ai/Apertus-8B-2509` unmodified.
2. Use the original tokenizer (vocab 131,072).
3. Train under the same Megatron-LM-Swiss-AI config + same dataloader
   seed + same NTP loss as the other two arms for 2 B tokens.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VanillaArmSpec:
    """Verification config emitted by the Vanilla arm. No init artifacts."""
    arm_name: str = "vanilla"
    base_model: str = "swiss-ai/Apertus-8B-2509"
    vocab_size: int = 131_072
    tokenizer_path: str = "swiss-ai/Apertus-8B-2509"  # use base, not our extended ship bundle
    init_method: str = "none"
    new_token_count: int = 0


def get_arm_spec() -> VanillaArmSpec:
    return VanillaArmSpec()


def main() -> None:
    spec = get_arm_spec()
    print(f"Vanilla arm: no init needed.")
    print(f"  arm: {spec.arm_name}")
    print(f"  base model: {spec.base_model}")
    print(f"  tokenizer: {spec.tokenizer_path}")
    print(f"  vocab: {spec.vocab_size:,}")
    print(f"  new tokens: {spec.new_token_count}")
    print(f"  → training job loads the base Apertus checkpoint unmodified.")


if __name__ == "__main__":
    main()
