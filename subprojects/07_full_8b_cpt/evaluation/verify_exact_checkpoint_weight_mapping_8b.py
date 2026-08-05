#!/usr/bin/env python3
"""Prove the full Apertus-8B Megatron-to-HF parameter mapping bit-exact."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ShardedSafetensors:
    def __init__(self, root: Path, stack: contextlib.ExitStack) -> None:
        self.root = root.resolve()
        self.index_path = self.root / "model.safetensors.index.json"
        if not self.index_path.is_file() or self.index_path.is_symlink():
            raise FileNotFoundError(self.index_path)
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.weight_map = index.get("weight_map", {})
        if not self.weight_map:
            raise ValueError("empty sharded safetensors weight map")
        self.shard_paths = {
            name: (self.root / name).resolve() for name in sorted(set(self.weight_map.values()))
        }
        if any(not path.is_file() or path.is_symlink() for path in self.shard_paths.values()):
            raise FileNotFoundError("missing or symlinked HF shard")
        self.handles = {
            name: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name, path in self.shard_paths.items()
        }
        actual = {key for handle in self.handles.values() for key in handle.keys()}
        if actual != set(self.weight_map):
            raise ValueError("HF shard index/tensor-key drift")

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(key)
        return self.handles[shard].get_tensor(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-checkpoint", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    intermediate = args.intermediate_checkpoint.resolve()
    hf_root = args.hf_root.resolve()
    hf_config = args.hf_config.resolve()
    for path in (intermediate, hf_config):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)

    config = json.loads(hf_config.read_text(encoding="utf-8"))
    expected_geometry = {
        "vocab_size": 148992,
        "hidden_size": 4096,
        "intermediate_size": 21504,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "rope_theta": 500000,
    }
    for key, expected in expected_geometry.items():
        if config.get(key) != expected:
            raise ValueError(f"converted Apertus-8B geometry drift ({key}): {config.get(key)!r}")
    rope = config.get("rope_scaling") or {}
    if rope.get("factor") != 8.0 or rope.get("rope_type") != "llama3":
        raise ValueError(f"converted Apertus-8B RoPE scaling drift: {rope!r}")

    layers = expected_geometry["num_hidden_layers"]
    heads = expected_geometry["num_attention_heads"]
    kv_heads = expected_geometry["num_key_value_heads"]
    hidden = expected_geometry["hidden_size"]
    head_size = hidden // heads
    checkpoint = torch.load(intermediate, map_location="cpu", weights_only=False, mmap=True)
    source = checkpoint.get("model")
    if not isinstance(source, dict):
        raise ValueError("intermediate checkpoint lacks model state")

    expected_source = {
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "output_layer.weight",
    }
    for layer in range(layers):
        prefix = f"decoder.layers.{layer}"
        expected_source.update(
            {
                f"{prefix}.self_attention.linear_proj.weight",
                f"{prefix}.self_attention.linear_qkv.layer_norm_weight",
                f"{prefix}.self_attention.linear_qkv.weight",
                f"{prefix}.self_attention.q_layernorm.weight",
                f"{prefix}.self_attention.k_layernorm.weight",
                f"{prefix}.mlp.linear_fc1.layer_norm_weight",
                f"{prefix}.mlp.linear_fc1.weight",
                f"{prefix}.mlp.activation_func.alpha_p",
                f"{prefix}.mlp.activation_func.alpha_n",
                f"{prefix}.mlp.linear_fc2.weight",
            }
        )
    meaningful_source = {
        key for key, value in source.items()
        if isinstance(value, torch.Tensor) and not key.endswith("._extra_state")
    }
    if meaningful_source != expected_source:
        raise ValueError(
            "source parameter coverage drift: "
            f"missing={sorted(expected_source - meaningful_source)}, "
            f"extra={sorted(meaningful_source - expected_source)}"
        )

    checked = 0
    expected_hf = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    with contextlib.ExitStack() as stack:
        tensors = ShardedSafetensors(hf_root, stack)

        def require_equal(source_key: str, hf_key: str, hf_value: torch.Tensor | None = None) -> None:
            nonlocal checked
            source_value = source[source_key].detach().cpu()
            target_value = tensors.get_tensor(hf_key) if hf_value is None else hf_value
            target_value = target_value.detach().cpu()
            checked += 1
            if source_value.shape != target_value.shape or not torch.equal(source_value, target_value):
                maximum = (
                    float("inf") if source_value.shape != target_value.shape
                    else (source_value.float() - target_value.float()).abs().max().item()
                )
                raise ValueError(
                    f"non-exact parameter mapping: {source_key} -> {hf_key}; max_abs={maximum}"
                )

        require_equal("embedding.word_embeddings.weight", "model.embed_tokens.weight")
        require_equal("output_layer.weight", "lm_head.weight")
        direct = (
            ("self_attention.linear_qkv.layer_norm_weight", "attention_layernorm.weight"),
            ("mlp.linear_fc1.layer_norm_weight", "feedforward_layernorm.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.linear_fc1.weight", "mlp.up_proj.weight"),
            ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
            ("mlp.activation_func.alpha_p", "mlp.act_fn.alpha_p"),
            ("mlp.activation_func.alpha_n", "mlp.act_fn.alpha_n"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
        )
        for layer in range(layers):
            source_prefix = f"decoder.layers.{layer}"
            target_prefix = f"model.layers.{layer}"
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
            reconstructed[q_slice] = tensors.get_tensor(q_key).reshape(heads, head_size, hidden)
            reconstructed[k_slice] = tensors.get_tensor(k_key).reshape(kv_heads, head_size, hidden)
            reconstructed[v_slice] = tensors.get_tensor(v_key).reshape(kv_heads, head_size, hidden)
            require_equal(
                f"{source_prefix}.self_attention.linear_qkv.weight",
                f"{target_prefix}.self_attn.[qkv]_proj.weight",
                reconstructed.reshape_as(source[f"{source_prefix}.self_attention.linear_qkv.weight"]),
            )

            beta_key = f"{target_prefix}.mlp.act_fn.beta"
            eps_key = f"{target_prefix}.mlp.act_fn.eps"
            expected_hf.update((beta_key, eps_key))
            beta = float(tensors.get_tensor(beta_key).float())
            eps = float(tensors.get_tensor(eps_key).float())
            if beta != 0.5 or abs(eps - (-1.0e-6)) > 1.0e-8:
                raise ValueError(f"xIELU constant drift in layer {layer}: beta={beta}, eps={eps}")

        require_equal("decoder.final_layernorm.weight", "model.norm.weight")
        if tensors.keys() != expected_hf:
            raise ValueError(
                "HF parameter coverage drift: "
                f"missing={sorted(expected_hf - tensors.keys())}, "
                f"extra={sorted(tensors.keys() - expected_hf)}"
            )
        shards = [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in tensors.shard_paths.values()
        ]
        index_path = tensors.index_path

    payload = {
        "schema_version": "apertus_full_8b_exact_checkpoint_weight_mapping_v1",
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "intermediate_checkpoint": {
            "path": str(intermediate), "bytes": intermediate.stat().st_size,
            "sha256": sha256_file(intermediate),
        },
        "hf_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "hf_shards": shards,
        "hf_config": {"path": str(hf_config), "sha256": sha256_file(hf_config)},
        "geometry": expected_geometry | {"head_size": head_size, "rope_scaling": rope},
        "source_parameter_tensors_expected": len(expected_source),
        "source_parameter_tensors_checked_exact": checked,
        "xielu_constant_tensors_checked": 2 * layers,
        "all_source_parameters_covered": True,
        "all_hf_tensors_accounted_for": True,
        "all_mapped_parameter_tensors_bit_exact": True,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "checked": checked, "shards": len(shards)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
