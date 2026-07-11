"""Single source of truth for the approved full-corpus dedup recipe."""

from __future__ import annotations

from typing import Any


PRODUCTION_RECIPE_ID = "greek_cpt_text_dedup_v1"
APPROVED_PRODUCTION_RECIPE: dict[str, Any] = {
    "greek_diacritic_policy": "preserve",
    "minhash_threshold": 0.85,
    "num_perm": 128,
    "bands": 32,
    "rows_per_band": 4,
    "shingle_mode": "token",
    "shingle_size": 5,
    "max_bucket_size": 5000,
}


def validate_recipe_parameters(
    parameters: dict[str, Any],
    *,
    experimental: bool,
) -> str:
    if set(parameters) != set(APPROVED_PRODUCTION_RECIPE):
        raise ValueError("dedup recipe parameters do not match the approved parameter names")
    changed = {
        key: {"approved": approved, "requested": parameters[key]}
        for key, approved in APPROVED_PRODUCTION_RECIPE.items()
        if parameters[key] != approved
    }
    if changed and not experimental:
        raise ValueError(
            "production dedup recipe is immutable; pass --experimental-parameters "
            f"for a non-production experiment: {changed}"
        )
    return "experimental" if experimental else "production"
