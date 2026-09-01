#!/usr/bin/env python3
"""Run SwissAI Megatron GPT pretraining over one frozen data-order arm.

The training dataset is receipt-bound while validation uses a fixed one-global-
batch sample from each named heldout prefix.  Validation deliberately disables
Goldfish masking: Goldfish is the training objective, not the heldout metric.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import torch

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)

from pretrain_gpt import (
    get_batch,
    is_dataset_built_on_rank,
    loss_func as upstream_loss_func,
    model_provider,
    stimer,
)
from exact_checkpoint_hook import install_from_environment
from scheduled_packed_dataset import ScheduledPackedGPTDataset


BASE_VOCAB_SIZE = 131_072
_TOKEN_BYTE_LENGTHS: torch.Tensor | None = None


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    if args.skip_train:
        print_rank_0(
            "> skip-train evaluation: omitting the exhausted scheduled training dataset"
        )
        return None, None, None
    tokenizer = get_tokenizer()
    schedule_manifest = Path(os.environ["MINI_SCHEDULE_MANIFEST"]).resolve()
    arm_id = os.environ["MINI_SCHEDULE_ARM"]
    required = int(train_val_test_num_samples[0])
    prefix_mode = os.environ.get("MINI_SCHEDULE_ALLOW_PREFIX", "0") == "1"
    train = ScheduledPackedGPTDataset(
        schedule_manifest,
        arm_id,
        eod_token_id=tokenizer.eod,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        goldfish_loss=args.goldfish_loss,
        goldfish_k=args.goldfish_k,
        goldfish_h=args.goldfish_h,
        max_samples=required if prefix_mode else None,
    )
    if len(train) != required:
        raise ValueError(
            f"schedule/training horizon mismatch for {arm_id}: {len(train)} != {required}"
        )
    print_rank_0(
        f"> receipt-bound scheduled dataset: arm={arm_id} samples={len(train)} "
        f"manifest={schedule_manifest}"
    )
    return train, None, None


def extra_valid_datasets_provider():
    """Build independently named, fixed validation datasets.

    The pinned Megatron runtime adds ``--extra-valid-data-path`` as alternating
    NAME/PREFIX pairs.  Each dataset is exactly ``eval_iters * global_batch``
    samples, so a cyclic iterator replays the identical examples at every
    checkpoint rather than changing the validation distribution over time.
    """

    args = get_args()
    values = args.extra_valid_data_path or []
    if len(values) % 2:
        raise ValueError("--extra-valid-data-path requires alternating NAME PREFIX pairs")
    names = values[0::2]
    if len(names) != len(set(names)):
        raise ValueError("--extra-valid-data-path names must be unique")
    num_samples = [args.eval_iters * args.global_batch_size, 0, 0]
    result = []
    for name, prefix in zip(names, values[1::2]):
        if not name or not prefix:
            raise ValueError("validation names and prefixes must be non-empty")
        config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=([prefix], [1.0]),
            blend_per_split=None,
            split="100,0,0",
            num_dataset_builder_threads=args.num_dataset_builder_threads,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=get_tokenizer(),
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
            s3_cache_path=args.s3_cache_path,
            goldfish_loss=False,
            goldfish_k=args.goldfish_k,
            goldfish_h=args.goldfish_h,
        )
        print_rank_0(f"> building fixed source-conditioned validation set {name}: {prefix}")
        dataset, _, _ = BlendedMegatronDatasetBuilder(
            GPTDataset,
            num_samples,
            is_dataset_built_on_rank,
            config,
        ).build()
        result.append((name, dataset))
    return result


def stratified_loss_func(
    loss_mask: torch.Tensor,
    labels: torch.Tensor,
    output_tensor: torch.Tensor,
):
    """Preserve the upstream objective and add base/added-target diagnostics."""

    loss, local_num_tokens, reporting = upstream_loss_func(loss_mask, output_tensor)
    # Megatron's legacy (non ``--calculate-per-token-loss``) pipeline divides
    # every microbatch loss by the returned token count before backward.  The
    # immutable five-arm schedule contains receipt-declared pad-only filler
    # slots, so a microbatch can legitimately contain zero loss-active tokens.
    # Dividing its zero loss by zero creates NaN gradients even though the
    # global batch still contains real tokens.  A denominator of one is exact
    # for an empty microbatch (0 / 1 == 0) and leaves every non-empty
    # microbatch bit-for-bit on the existing normalization path.  Reporting
    # retains the true zero denominators from ``upstream_loss_func`` below.
    backward_num_tokens = torch.clamp_min(local_num_tokens, 1)
    losses = output_tensor.float().view(-1)
    flat_mask = loss_mask.view(-1).float()
    flat_labels = labels.view(-1)
    base_mask = flat_mask * (flat_labels < BASE_VOCAB_SIZE).float()
    added_mask = flat_mask * (flat_labels >= BASE_VOCAB_SIZE).float()
    global _TOKEN_BYTE_LENGTHS
    if _TOKEN_BYTE_LENGTHS is None:
        import numpy as np

        table_path = Path(os.environ["MINI_TOKEN_BYTE_LENGTHS"])
        table = np.load(table_path, allow_pickle=False)
        if table.shape != (148_992,) or table.dtype != np.uint16:
            raise ValueError("token UTF-8 byte-length table geometry drift")
        _TOKEN_BYTE_LENGTHS = torch.from_numpy(table.astype(np.float32)).to(labels.device)
    label_bytes = _TOKEN_BYTE_LENGTHS[flat_labels]
    values = torch.stack(
        (
            torch.sum(losses * base_mask),
            base_mask.sum(),
            torch.sum(losses * added_mask),
            added_mask.sum(),
            torch.sum(label_bytes * base_mask),
            torch.sum(label_bytes * added_mask),
        )
    )
    if get_args().context_parallel_size > 1:
        torch.distributed.all_reduce(values, group=mpu.get_context_parallel_group())
    reduced = values.detach().clone()
    torch.distributed.all_reduce(reduced, group=mpu.get_data_parallel_group())
    # Megatron aggregates a list of per-microbatch dictionaries by taking the
    # keys from the first dictionary and indexing every later dictionary with
    # those keys.  A target class can be absent from one data-parallel-reduced
    # microbatch even though it is present elsewhere in the global batch, so
    # conditionally omitting a key makes the aggregation order-dependent and
    # can raise KeyError.  Zero denominators are intentionally retained here:
    # summing the (numerator, denominator) pairs across all microbatches gives
    # the exact token-weighted diagnostic for the global batch.
    reporting["base-token target loss"] = (reduced[0], reduced[1])
    reporting["base-token target count"] = (
        reduced[1],
        torch.ones_like(reduced[1]),
    )
    reporting["base-token target bytes"] = (
        reduced[4],
        torch.ones_like(reduced[4]),
    )
    reporting["added-token target loss"] = (reduced[2], reduced[3])
    reporting["added-token target count"] = (
        reduced[3],
        torch.ones_like(reduced[3]),
    )
    reporting["added-token target bytes"] = (
        reduced[5],
        torch.ones_like(reduced[5]),
    )
    return loss, backward_num_tokens, reporting


def forward_step(data_iterator, model):
    """Upstream-equivalent forward step with target-ID-stratified reporting."""

    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()
    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(stratified_loss_func, loss_mask, labels)


if __name__ == "__main__":
    install_from_environment()
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_valid_datasets_provider=extra_valid_datasets_provider,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
