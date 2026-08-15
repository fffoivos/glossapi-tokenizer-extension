#!/usr/bin/env python3
"""Prove an Apertus Megatron-to-HF export is tensor-exact at either study scale."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from checkpoint_export_contract import expected_source_keys, validate_geometry
from contract_utils import file_binding, read_json, require, write_json_atomic


class SafeTensorTree:
    """Read a single-file or sharded safetensors model without loading it all."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        index_path = self.root / "model.safetensors.index.json"
        single_path = self.root / "model.safetensors"
        if index_path.is_file():
            index = read_json(index_path)
            weight_map = index.get("weight_map")
            require(
                isinstance(weight_map, dict) and weight_map,
                "HF safetensors index is empty",
            )
            self.weight_map = {
                str(key): str(value) for key, value in weight_map.items()
            }
            shards = set(self.weight_map.values())
            for shard in shards:
                path = self.root / shard
                require(
                    path.is_file() and not path.is_symlink(),
                    f"HF shard missing: {path}",
                )
        else:
            require(
                single_path.is_file() and not single_path.is_symlink(),
                "HF safetensors weights missing",
            )
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self.weight_map = {key: single_path.name for key in handle}

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        require(key in self.weight_map, f"HF tensor missing: {key}")
        path = self.root / self.weight_map[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def file_bindings(self) -> list[dict[str, Any]]:
        paths = sorted({self.root / name for name in self.weight_map.values()})
        return [file_binding(path) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-checkpoint", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--true-vocab-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    intermediate = args.intermediate_checkpoint.resolve()
    hf_root = args.hf_root.resolve()
    config_path = hf_root / "config.json"
    require(
        intermediate.is_file() and not intermediate.is_symlink(),
        "intermediate checkpoint missing",
    )
    require(config_path.is_file() and not config_path.is_symlink(), "HF config missing")
    config = read_json(config_path)
    contract = read_json(args.model_contract)
    geometry = validate_geometry(
        config, contract, scale=args.scale, true_vocab_size=args.true_vocab_size
    )
    layers = geometry["num_hidden_layers"]
    heads = geometry["num_attention_heads"]
    kv_heads = geometry["num_key_value_heads"]
    hidden = geometry["hidden_size"]
    tied = geometry["tie_word_embeddings"]
    require(hidden % heads == 0 and heads % kv_heads == 0, "invalid GQA geometry")
    head_size = hidden // heads

    checkpoint = torch.load(intermediate, map_location="cpu", weights_only=False)
    source = checkpoint.get("model")
    require(isinstance(source, dict), "intermediate checkpoint lacks model state")
    expected_source = expected_source_keys(layers, tied=tied)
    meaningful_source = {
        key
        for key, value in source.items()
        if isinstance(value, torch.Tensor) and not key.endswith("._extra_state")
    }
    require(
        meaningful_source == expected_source,
        f"source parameter coverage drift: missing={sorted(expected_source - meaningful_source)}, extra={sorted(meaningful_source - expected_source)}",
    )

    tensors = SafeTensorTree(hf_root)
    expected_hf = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    checked = 0

    def require_equal(
        source_key: str, hf_key: str, hf_value: torch.Tensor | None = None
    ) -> None:
        nonlocal checked
        source_value = source[source_key].detach().cpu()
        target = (
            tensors.get_tensor(hf_key) if hf_value is None else hf_value.detach().cpu()
        )
        checked += 1
        require(
            source_value.shape == target.shape,
            f"parameter shape drift: {source_key} -> {hf_key}",
        )
        if not torch.equal(source_value, target):
            maximum = (source_value.float() - target.float()).abs().max().item()
            raise ValueError(
                f"non-exact parameter mapping: {source_key} -> {hf_key}; max_abs={maximum}"
            )

    require_equal("embedding.word_embeddings.weight", "model.embed_tokens.weight")
    for layer in range(layers):
        source_prefix = f"decoder.layers.{layer}"
        target_prefix = f"model.layers.{layer}"
        direct = (
            (
                "self_attention.linear_qkv.layer_norm_weight",
                "attention_layernorm.weight",
            ),
            ("mlp.linear_fc1.layer_norm_weight", "feedforward_layernorm.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.linear_fc1.weight", "mlp.up_proj.weight"),
            ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
            ("mlp.activation_func.alpha_p", "mlp.act_fn.alpha_p"),
            ("mlp.activation_func.alpha_n", "mlp.act_fn.alpha_n"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
        )
        for source_suffix, target_suffix in direct:
            source_key = f"{source_prefix}.{source_suffix}"
            hf_key = f"{target_prefix}.{target_suffix}"
            expected_hf.add(hf_key)
            require_equal(source_key, hf_key)

        total_heads = heads + 2 * kv_heads
        heads_per_group = heads // kv_heads
        qkv = source[f"{source_prefix}.self_attention.linear_qkv.weight"]
        qkv = qkv.reshape(total_heads, head_size, hidden)
        reconstructed = torch.empty_like(qkv)
        q_slice = torch.cat(
            [
                torch.arange(
                    (heads_per_group + 2) * group,
                    (heads_per_group + 2) * group + heads_per_group,
                )
                for group in range(kv_heads)
            ]
        )
        k_slice = torch.arange(heads_per_group, total_heads, heads_per_group + 2)
        v_slice = torch.arange(heads_per_group + 1, total_heads, heads_per_group + 2)
        q_key = f"{target_prefix}.self_attn.q_proj.weight"
        k_key = f"{target_prefix}.self_attn.k_proj.weight"
        v_key = f"{target_prefix}.self_attn.v_proj.weight"
        expected_hf.update((q_key, k_key, v_key))
        reconstructed[q_slice] = tensors.get_tensor(q_key).reshape(
            heads, head_size, hidden
        )
        reconstructed[k_slice] = tensors.get_tensor(k_key).reshape(
            kv_heads, head_size, hidden
        )
        reconstructed[v_slice] = tensors.get_tensor(v_key).reshape(
            kv_heads, head_size, hidden
        )
        require_equal(
            f"{source_prefix}.self_attention.linear_qkv.weight",
            f"{target_prefix}.self_attn.[qkv]_proj.weight",
            reconstructed.reshape_as(
                source[f"{source_prefix}.self_attention.linear_qkv.weight"]
            ),
        )

        beta_key = f"{target_prefix}.mlp.act_fn.beta"
        eps_key = f"{target_prefix}.mlp.act_fn.eps"
        expected_hf.update((beta_key, eps_key))
        beta = float(tensors.get_tensor(beta_key).float())
        eps = float(tensors.get_tensor(eps_key).float())
        require(
            beta == 0.5 and abs(eps - (-1.0e-6)) <= 1.0e-8,
            f"xIELU constant drift in layer {layer}",
        )

    require_equal("decoder.final_layernorm.weight", "model.norm.weight")
    require_equal(
        "embedding.word_embeddings.weight" if tied else "output_layer.weight",
        "lm_head.weight",
    )
    require(
        tensors.keys() == expected_hf,
        f"HF parameter coverage drift: missing={sorted(expected_hf - tensors.keys())}, extra={sorted(tensors.keys() - expected_hf)}",
    )

    payload = {
        "schema_version": "apertus_exact_checkpoint_weight_mapping_v2",
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "model_contract": file_binding(args.model_contract),
        "intermediate_checkpoint": file_binding(intermediate),
        "hf_config": file_binding(config_path),
        "hf_weight_files": tensors.file_bindings(),
        "geometry": geometry | {"head_size": head_size},
        "source_parameter_tensors_expected": len(expected_source),
        "source_parameter_tensors_checked_exact": checked,
        "xielu_constant_tensors_checked": 2 * layers,
        "all_source_parameters_covered": True,
        "all_hf_tensors_accounted_for": True,
        "all_mapped_parameter_tensors_bit_exact": True,
    }
    write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "ok": True,
                "scale": args.scale,
                "checked": checked,
                "output": str(args.output.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
