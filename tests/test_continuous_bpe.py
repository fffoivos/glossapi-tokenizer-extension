from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from glossapi_corpus_cli.continuous_bpe import (
    build_extended_tokenizer_dir,
    load_base_tokenizer_artifacts,
    run_continuation_training,
    verify_front_end_contract,
    verify_tokenizer_identity,
)


def _build_tiny_base_tokenizer(output_dir: Path) -> Path:
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "a": 4,
        "b": 5,
        "c": 6,
        " ": 7,
    }
    model = BPE(vocab=vocab, merges=[], unk_token="<unk>", ignore_merges=True)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder(add_prefix_space=True, trim_offsets=True, use_regex=True)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A",
        pair="<s> $A <s>:1 $B:1",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(output_dir)
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": {"content": "<s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "eos_token": {"content": "</s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "pad_token": {"content": "<pad>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
                "unk_token": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_json = json.loads((output_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_json["normalizer"] = None
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir


def test_run_continuation_training_appends_new_merge(tmp_path: Path) -> None:
    base_dir = _build_tiny_base_tokenizer(tmp_path / "base")
    base_artifacts = load_base_tokenizer_artifacts(str(base_dir))
    sequence_counter = {
        (4, 5, 6): 10,
        (4, 5): 5,
    }
    result = run_continuation_training(
        base_artifacts=base_artifacts,
        sequence_counter=sequence_counter,  # type: ignore[arg-type]
        target_vocab_size=base_artifacts.base_vocab_size + 1,
        checkpoint_path=tmp_path / "state.pkl",
        checkpoint_every=1,
        progress_callback=lambda _patch: None,
        min_pair_frequency=1,
    )
    assert result["added_tokens"] == ["ab"]
    assert result["added_merges"] == [["a", "b"]]
    assert len(result["token_strings"]) == base_artifacts.base_vocab_size + 1


def test_extended_tokenizer_preserves_front_end_contract(tmp_path: Path) -> None:
    base_dir = _build_tiny_base_tokenizer(tmp_path / "base")
    base_artifacts = load_base_tokenizer_artifacts(str(base_dir))
    token_strings = [None] * base_artifacts.base_vocab_size
    for token, token_id in base_artifacts.vocab.items():
        token_strings[token_id] = token
    token_strings.append("ab")
    output_dir = tmp_path / "extended"
    build_extended_tokenizer_dir(
        base_artifacts=base_artifacts,
        output_dir=output_dir,
        token_strings=token_strings,
        added_merges=[["a", "b"]],
    )
    contract = verify_front_end_contract(str(base_dir), str(output_dir))
    assert contract["candidate_vocab_size"] == len(token_strings)
    identity = verify_tokenizer_identity(str(base_dir), str(base_dir), ["abc", "ab c"])
    assert identity["vocab_size"] == base_artifacts.base_vocab_size
