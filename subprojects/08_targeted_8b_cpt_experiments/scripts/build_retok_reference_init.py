#!/usr/bin/env python3
"""Build the architecture-aware ReTok reference before Token Distillation."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from contract_utils import (
    executing_code_bundle,
    file_binding,
    require,
    require_file_binding,
    require_receipt,
    require_relative_inventory,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-materialization-receipt", type=Path, required=True)
    parser.add_argument("--extended-tokenizer", type=Path, required=True)
    parser.add_argument("--extended-tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-compatibility", type=Path, required=True)
    parser.add_argument("--retok-tool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--expected-hidden-size", type=int, required=True)
    parser.add_argument("--expected-hidden-layers", type=int, required=True)
    return parser.parse_args()


def ordered_vocab(tokenizer_json: dict, *, minimum_size: int) -> list[str]:
    vocab = tokenizer_json.get("model", {}).get("vocab")
    require(isinstance(vocab, dict) and len(vocab) >= minimum_size, "tokenizer vocabulary inventory missing")
    ids = sorted(int(value) for value in vocab.values())
    require(ids == list(range(len(vocab))), "tokenizer vocabulary ids are not contiguous")
    ordered = [""] * len(vocab)
    for token, token_id in vocab.items():
        ordered[int(token_id)] = str(token)
    require(len(set(ordered)) == len(ordered), "tokenizer vocabulary tokens are not unique")
    return ordered


def verify_tokenizer_compatibility(
    *, base_model: Path, extended_tokenizer: Path, contract_path: Path
) -> dict:
    contract = require_receipt(
        contract_path,
        schemas={"apertus_1p5b_tokenizer_compatibility_v1"},
        statuses={"frozen"},
    )
    base_path = base_model / "tokenizer.json"
    target_path = extended_tokenizer / "tokenizer.json"
    require(sha256_file(base_path) == contract["source"]["tokenizer_sha256"], "source tokenizer hash drift")
    require(sha256_file(target_path) == contract["target"]["tokenizer_sha256"], "target tokenizer hash drift")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    base_size = int(contract.get("base_vocab_size", -1))
    require(base_size == 131_072, "tokenizer compatibility base size drift")
    base_vocab = ordered_vocab(base, minimum_size=base_size)
    target_vocab = ordered_vocab(target, minimum_size=base_size)
    require(len(base_vocab) == base_size and len(target_vocab) == 148_480, "tokenizer compatibility size drift")
    actual_content_differences = [
        {"id": token_id, "source": base_vocab[token_id], "target": target_vocab[token_id]}
        for token_id in range(base_size)
        if base_vocab[token_id] != target_vocab[token_id]
    ]
    require(
        actual_content_differences == contract.get("expected_content_differences"),
        "base-vocabulary content-difference contract drift",
    )
    for component in contract.get("required_identical_tokenizer_components", []):
        require(base.get(component) == target.get(component), f"tokenizer component drift: {component}")
    base_model_json = base.get("model", {})
    target_model_json = target.get("model", {})
    base_merges = base_model_json.get("merges")
    target_merges = target_model_json.get("merges")
    require(
        isinstance(base_merges, list)
        and isinstance(target_merges, list)
        and target_merges[: len(base_merges)] == base_merges,
        "extended tokenizer does not preserve the complete base merge prefix",
    )
    for key in set(base_model_json) | set(target_model_json):
        if key not in {"vocab", "merges"}:
            require(base_model_json.get(key) == target_model_json.get(key), f"tokenizer model-field drift: {key}")
    base_added = {int(row["id"]): row for row in base.get("added_tokens", []) if int(row["id"]) < base_size}
    target_added = {int(row["id"]): row for row in target.get("added_tokens", []) if int(row["id"]) < base_size}
    require(set(base_added) == set(target_added), "base added-token id inventory drift")
    actual_added_differences = sorted(
        token_id for token_id in base_added if base_added[token_id] != target_added[token_id]
    )
    require(
        actual_added_differences == contract.get("expected_added_token_record_difference_ids"),
        "base added-token metadata difference drift",
    )
    row_policy = contract.get("row_policy", {})
    require(
        row_policy.get("preserve_every_base_input_and_output_row_by_id") is True
        and row_policy.get("permute_base_rows") is False,
        "unsupported base-row tokenizer migration policy",
    )
    return {
        "contract": file_binding(contract_path),
        "base_vocab_rows": base_size,
        "ordinary_same_id_rows": base_size - len(actual_content_differences),
        "content_differences": actual_content_differences,
        "added_token_record_difference_ids": actual_added_differences,
        "base_merge_count": len(base_merges),
        "target_merge_count": len(target_merges),
        "base_merge_prefix_exact": True,
        "row_policy": row_policy,
    }


def load_retok(path: Path):
    spec = importlib.util.spec_from_file_location("_frozen_apertus_retok", path)
    require(spec is not None and spec.loader is not None, "could not load frozen ReTok tool")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The canonical ReTok implementation imports its sibling `_common.py` as
    # a top-level module.  Load it from the immutable frozen tool directory,
    # then restore the caller's module/path state so no ambient package can
    # satisfy or retain that dependency accidentally.
    tool_dir = str(path.resolve().parent)
    previous_common = sys.modules.pop("_common", None)
    sys.path.insert(0, tool_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        path_order_ok = bool(sys.path) and sys.path[0] == tool_dir
        if path_order_ok:
            sys.path.pop(0)
        sys.modules.pop("_common", None)
        if previous_common is not None:
            sys.modules["_common"] = previous_common
        require(path_order_ok, "ReTok import path ordering drift")
    return module.compute_retok_init


def main() -> int:
    args = parse_args()
    require(not args.output_root.exists() and not args.output_receipt.exists(), "immutable ReTok output exists")
    materialization = require_receipt(
        args.base_materialization_receipt,
        schemas={"apertus_pinned_hf_model_materialization_v1"},
    )
    require(Path(str(materialization.get("output_root", ""))).resolve() == args.base_model.resolve(), "base materialization root drift")
    require_relative_inventory(root=args.base_model, rows=materialization.get("files"))
    tokenizer_receipt = require_receipt(
        args.extended_tokenizer_receipt,
        schemas={"apertus_historical_tokenizer_148480_v1"},
    )
    require(Path(str(tokenizer_receipt.get("output_root", ""))).resolve() == args.extended_tokenizer.resolve(), "extended tokenizer root drift")
    tokenizer_files = tokenizer_receipt.get("files")
    require(isinstance(tokenizer_files, dict) and bool(tokenizer_files), "extended tokenizer inventory missing")
    for relative, binding in tokenizer_files.items():
        require_file_binding(binding, expected_path=args.extended_tokenizer / relative)
    tokenizer_compatibility = verify_tokenizer_compatibility(
        base_model=args.base_model,
        extended_tokenizer=args.extended_tokenizer,
        contract_path=args.tokenizer_compatibility,
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    extended = AutoTokenizer.from_pretrained(args.extended_tokenizer, trust_remote_code=True)
    require(len(base_tokenizer) == 131_072 and len(extended) == 148_480, "tokenizer vocabulary drift")
    row_policy = tokenizer_compatibility["row_policy"]
    require(base_tokenizer.bos_token_id == extended.bos_token_id == 1, "BOS token id drift")
    require(base_tokenizer.eos_token_id == extended.eos_token_id == 2, "EOS token id drift")
    require(base_tokenizer.unk_token_id == extended.unk_token_id == 0, "UNK token id drift")
    require(base_tokenizer.pad_token_id == row_policy["source_declared_pad_token_id"], "source declared pad id drift")
    require(base_tokenizer.convert_tokens_to_ids("<pad>") == row_policy["source_pad_surface_vocab_id"], "source pad surface id drift")
    require(extended.pad_token_id == row_policy["target_declared_pad_token_id"], "target declared pad id drift")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    config = model.config
    require(config.hidden_size == args.expected_hidden_size and config.num_hidden_layers == args.expected_hidden_layers, "model geometry drift")
    require(config.tie_word_embeddings is False, "TD requires untied embeddings")
    require(config.pad_token_id == row_policy["model_config_pad_token_id"], "model config pad id drift")
    state_before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    require(tuple(input_weight.shape) == (131_072, args.expected_hidden_size), "input embedding geometry drift")
    require(tuple(output_weight.shape) == (131_072, args.expected_hidden_size), "output embedding geometry drift")
    compute_retok_init = load_retok(args.retok_tool)
    new_input, new_output = compute_retok_init(
        base_E=input_weight.detach().cpu().float().numpy(),
        base_U=output_weight.detach().cpu().float().numpy(),
        base_tokenizer=base_tokenizer,
        extended_tokenizer=extended,
        new_id_range=(131_072, 148_480),
        verbose=True,
    )
    try:
        model.resize_token_embeddings(148_480, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(148_480)
    with torch.no_grad():
        model.get_input_embeddings().weight[131_072:] = torch.from_numpy(new_input).to(model.get_input_embeddings().weight.dtype)
        model.get_output_embeddings().weight[131_072:] = torch.from_numpy(new_output).to(model.get_output_embeddings().weight.dtype)
    after = dict(model.named_parameters())
    input_name = next(name for name, parameter in after.items() if parameter is model.get_input_embeddings().weight)
    output_name = next(name for name, parameter in after.items() if parameter is model.get_output_embeddings().weight)
    require(set(after) == set(state_before), "parameter inventory drift after resize")
    changed_non_embedding = []
    for name, previous in state_before.items():
        current = after[name].detach().cpu()
        if name in {input_name, output_name}:
            require(torch.equal(current[:131_072], previous), f"base embedding rows changed: {name}")
        elif not torch.equal(current, previous):
            changed_non_embedding.append(name)
    require(not changed_non_embedding, f"non-embedding tensors changed: {changed_non_embedding[:10]}")
    require(torch.isfinite(model.get_input_embeddings().weight).all().item(), "non-finite input rows")
    require(torch.isfinite(model.get_output_embeddings().weight).all().item(), "non-finite output rows")

    args.output_root.mkdir(parents=True)
    model.save_pretrained(args.output_root, safe_serialization=True)
    extended.save_pretrained(args.output_root)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        source = args.extended_tokenizer / name
        if source.is_file():
            shutil.copyfile(source, args.output_root / name)
    weight_metadata = args.output_root / "model.safetensors.index.json"
    if not weight_metadata.is_file():
        weight_metadata = args.output_root / "model.safetensors"
    require(weight_metadata.is_file(), "saved ReTok weight payload missing")
    output_files = [
        {"path": str(path.relative_to(args.output_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(value for value in args.output_root.rglob("*") if value.is_file())
    ]
    require(bool(output_files), "saved ReTok output inventory is empty")
    receipt = {
        "schema_version": "apertus_retok_reference_init_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_vocab_size": 131_072,
        "extended_vocab_size": 148_480,
        "hidden_size": args.expected_hidden_size,
        "hidden_layers": args.expected_hidden_layers,
        "tie_word_embeddings": False,
        "base_input_rows_exact": True,
        "base_output_rows_exact": True,
        "base_rows_preserved_by_id_without_permutation": True,
        "tokenizer_compatibility": tokenizer_compatibility,
        "non_embedding_tensors_exact": True,
        "retok_tool": file_binding(args.retok_tool),
        "base_materialization_receipt": file_binding(args.base_materialization_receipt),
        "extended_tokenizer_receipt": file_binding(args.extended_tokenizer_receipt),
        "base_config": file_binding(args.base_model / "config.json"),
        "extended_tokenizer": file_binding(args.extended_tokenizer / "tokenizer.json"),
        "output_config": file_binding(args.output_root / "config.json"),
        "output_weight_metadata": file_binding(weight_metadata),
        "output_root": str(args.output_root.resolve()),
        "output_files": output_files,
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output_receipt, receipt)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
