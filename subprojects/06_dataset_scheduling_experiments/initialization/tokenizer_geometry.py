"""Exact BPE geometry helpers for the Mini-compatible Greek extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def split_merge(value: Any) -> tuple[str, str]:
    if isinstance(value, list) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, str):
        parts = value.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"unsupported merge representation: {value!r}")


def derive_added_token_base_ids(
    tokenizer_json: Path,
    *,
    base_vocab_size: int,
    target_vocab_size: int,
) -> dict[int, list[int]]:
    """Expand every appended merge to leaves in the original base vocabulary.

    This avoids decoding ByteLevel token strings, which can be lossy for a token
    containing only part of a multi-byte character.  The extension contract has
    one dependency-ordered appended merge per appended vocabulary item.
    """

    root = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = root.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ValueError("tokenizer.json must contain BPE vocab and merges")

    ids = {int(token_id): str(token) for token, token_id in vocab.items()}
    if sorted(ids) != list(range(target_vocab_size)):
        raise ValueError("tokenizer vocabulary is not the expected contiguous range")

    added_count = target_vocab_size - base_vocab_size
    if len(merges) < added_count:
        raise ValueError("tokenizer has fewer merges than appended vocabulary rows")
    appended_merges = merges[len(merges) - added_count :]
    decomposition_by_token = {
        ids[token_id]: [token_id] for token_id in range(base_vocab_size)
    }
    result: dict[int, list[int]] = {}
    for offset, merge in enumerate(appended_merges):
        token_id = base_vocab_size + offset
        left, right = split_merge(merge)
        if left not in decomposition_by_token or right not in decomposition_by_token:
            raise ValueError(f"token {token_id}: appended merge operand is not live")
        token = left + right
        if ids[token_id] != token:
            raise ValueError(f"token {token_id}: merge result/vocabulary ID drift")
        leaves = decomposition_by_token[left] + decomposition_by_token[right]
        if not leaves or any(value < 0 or value >= base_vocab_size for value in leaves):
            raise ValueError(f"token {token_id}: invalid base-vocabulary leaves")
        decomposition_by_token[token] = leaves
        result[token_id] = leaves

    if sorted(result) != list(range(base_vocab_size, target_vocab_size)):
        raise AssertionError("did not derive every appended token decomposition")
    return result

