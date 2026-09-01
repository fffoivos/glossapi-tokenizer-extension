#!/usr/bin/env python3
"""Stable, fail-closed semantic identity for a receipt-only stage rebind.

The original full-8B ``scientific_digest`` deliberately includes the complete
``data`` mapping.  That was appropriate before the post-review replacement
stage existed, but it also makes a *receipt location/hash* change look like a
different training experiment.  This module makes that distinction explicit:
the current receipt bindings remain validated independently, while this
identity hashes every training-relevant field and replaces only the two
duplicated receipt-binding subtrees with fixed sentinels.

It is intentionally a narrow compatibility layer for an already-proven,
byte-identical successor stage.  It must not be used to compare arbitrary
recipes or to hide changes to data, model, optimizer, schedule, tokenizer, or
evaluation settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "apertus_full_8b_successor_semantic_identity_v1"
_RECEIPT_SENTINEL = "<validated_receipt_binding_outside_semantic_identity_v1>"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _science_payload(recipe: dict[str, Any]) -> dict[str, Any]:
    """Match the v45 digest's coverage, after the two receipt normalizations."""

    data = copy.deepcopy(recipe["data"])
    # Both are evidence references.  The successor gate validates the live
    # objects (including hashes, policy, counts and post-mask dedup) directly.
    # Their physical path/hash must therefore not define training semantics.
    if "sanitized_source_receipt" not in data:
        raise ValueError("sanitized_source_receipt absent from recipe data")
    data["sanitized_source_receipt"] = _RECEIPT_SENTINEL
    eligibility = data.get("eligibility_policy")
    if not isinstance(eligibility, dict) or "proof" not in eligibility:
        raise ValueError("eligibility policy/proof absent from recipe data")
    eligibility["proof"] = _RECEIPT_SENTINEL

    batch = recipe["batch_and_parallelism"]
    return {
        "recipe_id": recipe["recipe_id"],
        "data": data,
        "tokenizer": recipe["tokenizer"],
        "initialization": recipe["initialization"],
        "model": recipe["model"],
        "optimization": recipe["optimization"],
        "batch": {
            key: batch[key]
            for key in (
                "global_batch_sequences",
                "global_batch_tokens",
                "micro_batch_sequences",
                "training_updates",
                "training_samples",
                "tensor_parallel",
                "pipeline_parallel",
                "context_parallel",
            )
        },
        "evaluation": recipe["evaluation"],
        "software": recipe["software"],
    }


def semantic_identity(recipe: dict[str, Any]) -> str:
    return sha256_json(_science_payload(recipe))


def normalized_recipe_payload(recipe: dict[str, Any]) -> dict[str, Any]:
    """Expose the exact covered payload for equality checks and diagnostics."""

    return _science_payload(recipe)


def normalized_profiles(profiles: dict[str, Any]) -> dict[str, Any]:
    """Profiles have no science-changing derivation fields after validation."""

    value = copy.deepcopy(profiles)
    if "derivation" not in value:
        raise ValueError("profiles derivation absent")
    value["derivation"] = _RECEIPT_SENTINEL
    return value


def changed_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [path]
        rows: list[str] = []
        for key in sorted(left):
            rows.extend(changed_paths(left[key], right[key], f"{path}.{key}"))
        return rows
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        rows: list[str] = []
        for index, (first, second) in enumerate(zip(left, right)):
            rows.extend(changed_paths(first, second, f"{path}[{index}]"))
        return rows
    return [] if left == right else [path]
