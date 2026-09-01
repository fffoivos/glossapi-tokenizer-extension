#!/usr/bin/env python3
"""Patch the pinned trainer for scale geometry and an explicit data-cache root."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD_NETWORK = """NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 21504
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
"""
NEW_NETWORK = """NETWORK_SIZE_ARGS=(
    --num-layers ${MODEL_NUM_LAYERS:-32}
    --hidden-size ${MODEL_HIDDEN_SIZE:-4096}
    --ffn-hidden-size ${MODEL_FFN_HIDDEN_SIZE:-21504}
    --num-attention-heads ${MODEL_NUM_ATTENTION_HEADS:-32}
    --group-query-attention
    --num-query-groups ${MODEL_NUM_QUERY_GROUPS:-8}
"""
OLD_METADATA = '  "megatron_commit": "$MEGATRON_LM_SWISSAI_COMMIT",\n'
NEW_METADATA = OLD_METADATA + """  "model_num_layers": ${MODEL_NUM_LAYERS:-32},
  "model_hidden_size": ${MODEL_HIDDEN_SIZE:-4096},
  "model_ffn_hidden_size": ${MODEL_FFN_HIDDEN_SIZE:-21504},
  "model_num_attention_heads": ${MODEL_NUM_ATTENTION_HEADS:-32},
  "model_num_query_groups": ${MODEL_NUM_QUERY_GROUPS:-8},
"""
OLD_DATA_THREADS = """    --num-workers $DATALOADER_WORKERS
    --num-dataset-builder-threads 4
"""
NEW_DATA_THREADS = OLD_DATA_THREADS + """    --data-cache-path "${H2G_PHASE_CACHE_ROOT:?set frozen phase cache root}"
"""


def patch_trainer(trainer: Path) -> bool:
    """Install the exact-anchor patch; return True only on the first install."""
    text = trainer.read_text(encoding="utf-8")
    if NEW_NETWORK in text and NEW_METADATA in text and NEW_DATA_THREADS in text:
        if (
            OLD_NETWORK in text
            or text.count(NEW_NETWORK) != 1
            or text.count(NEW_METADATA) != 1
            or text.count(NEW_DATA_THREADS) != 1
        ):
            raise ValueError("partially or multiply patched trainer")
        return False
    if text.count(OLD_NETWORK) != 1 or text.count(OLD_METADATA) != 1 or text.count(OLD_DATA_THREADS) != 1:
        raise ValueError("pinned trainer anchors drifted")
    text = (
        text.replace(OLD_NETWORK, NEW_NETWORK)
        .replace(OLD_METADATA, NEW_METADATA)
        .replace(OLD_DATA_THREADS, NEW_DATA_THREADS)
    )
    trainer.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()
    installed = patch_trainer(args.trainer)
    print("installed scale-geometry trainer patch" if installed else "scale-geometry patch already installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
