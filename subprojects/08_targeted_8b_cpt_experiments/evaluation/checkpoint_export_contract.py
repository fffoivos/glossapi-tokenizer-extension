"""Pure scale-aware contracts shared by checkpoint export and verification."""

from __future__ import annotations

from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_source_keys(layers: int, *, tied: bool) -> set[str]:
    keys = {"embedding.word_embeddings.weight", "decoder.final_layernorm.weight"}
    if not tied:
        keys.add("output_layer.weight")
    for layer in range(layers):
        prefix = f"decoder.layers.{layer}"
        keys.update(
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
    return keys


def validate_geometry(
    config: dict[str, Any],
    contract: dict[str, Any],
    *,
    scale: str,
    true_vocab_size: int,
) -> dict[str, Any]:
    models = contract.get("models")
    require(
        isinstance(models, dict) and isinstance(models.get(scale), dict),
        "model contract scale missing",
    )
    model = models[scale]
    expected = {
        "hidden_size": int(model["hidden_size"]),
        "num_hidden_layers": int(model["num_hidden_layers"]),
        "num_attention_heads": int(model["num_attention_heads"]),
        "num_key_value_heads": int(model["num_key_value_heads"]),
        "tie_word_embeddings": bool(model["tie_word_embeddings"]),
        "vocab_size": true_vocab_size,
    }
    for key, value in expected.items():
        require(
            config.get(key) == value,
            f"converted HF geometry drift ({key}): {config.get(key)!r} != {value!r}",
        )
    training = contract.get("training")
    require(isinstance(training, dict), "training geometry missing from contract")
    require(
        config.get("rope_theta") == float(training["rope_theta"]),
        "converted HF RoPE theta drift",
    )
    require(
        int(config.get("max_position_embeddings", -1))
        == int(training["max_position_embeddings"]),
        "converted HF context geometry drift",
    )
    scaling = config.get("rope_scaling")
    require(isinstance(scaling, dict), "converted HF RoPE scaling missing")
    expected_scaling = {
        "factor": float(training["rope_scaling_factor"]),
        "original_max_position_embeddings": 8192,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "rope_type": "llama3",
    }
    for key, value in expected_scaling.items():
        require(
            scaling.get(key) == value,
            f"converted HF RoPE scaling drift ({key}): {scaling.get(key)!r} != {value!r}",
        )
    return expected | {
        "rope_theta": float(training["rope_theta"]),
        "max_position_embeddings": int(training["max_position_embeddings"]),
        "rope_scaling_factor": float(training["rope_scaling_factor"]),
    }
