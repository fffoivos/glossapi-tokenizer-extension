#!/usr/bin/env python3
"""Convert a Megatron torch checkpoint to torch_dist without optimization.

This is the exact inverse of SwissAI's pinned `torchdist_2_torch.py`: model
construction and checkpoint I/O stay in Megatron, while the source and target
formats are reversed.  It is deliberately a format conversion, not a training
step.
"""

from megatron.core.enums import ModelType
from megatron.training.global_vars import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from pretrain_gpt import model_provider


def main() -> None:
    args_defaults = {
        "transformer_impl": "transformer_engine",
        "use_checkpoint_args": True,
        "no_load_rng": True,
        "no_load_optim": True,
        "no_save_optim": True,
        "no_save_rng": True,
        "exit_on_missing_checkpoint": True,
        "micro_batch_size": 1,
        "train_iters": 1,
        "lr": 0.0,
    }
    initialize_megatron(args_defaults=args_defaults)
    args = get_args()
    if args.load is None or args.ckpt_convert_save is None:
        raise ValueError("--load and --ckpt-convert-save are required")
    # Megatron's parser defaults the source format to torch_dist, so the
    # inverse conversion must make both formats explicit on the command line.
    if args.ckpt_format != "torch" or args.ckpt_convert_format != "torch_dist":
        raise ValueError(
            "format contract drift: pass --ckpt-format torch "
            "--ckpt-convert-format torch_dist"
        )
    setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)


if __name__ == "__main__":
    main()
