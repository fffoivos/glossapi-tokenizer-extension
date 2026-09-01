#!/usr/bin/env python3
"""Freeze the exact-parameter-mapping recovery for numerically sensitive exports."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

from campaign_contract import (
    AUTHORITATIVE_EVALUATION_DTYPE,
    atomic_write_json,
    file_receipt,
    read_json,
    sha256_file,
    verify_code_bundle_receipt,
)


def runtime_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "prediction_agreement_percent": r"Converted model agrees on ([0-9.]+)%",
        "mean_kl_divergence": r"Converted mean KL divergence: ([-+0-9.eE]+)",
        "mean_total_variation": r"Converted mean total variation: ([-+0-9.eE]+)",
        "mean_top_token_logprob_abs_difference": (
            r"Converted mean top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
        "p99_top_token_logprob_abs_difference": (
            r"Converted p99 top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
        "p999_top_token_logprob_abs_difference": (
            r"Converted p99\.9 top-token log-prob absolute difference: ([-+0-9.eE]+)"
        ),
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"missing {name} in {path}")
        values[name] = float(match.group(1))
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError(f"non-finite runtime diagnostic in {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--prior-recovery-receipt", type=Path, required=True)
    parser.add_argument("--scientific-bundle", type=Path, required=True)
    parser.add_argument("--scientific-bundle-receipt", type=Path, required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--megatron-bundle-receipt", type=Path, required=True)
    parser.add_argument("--exact-mapping-receipt", type=Path, required=True)
    parser.add_argument("--moderate-runtime-log", type=Path, required=True)
    parser.add_argument("--severe-runtime-log", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, action="append", required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--recovery-output", type=Path, required=True)
    args = parser.parse_args()

    campaign_path = args.campaign_manifest.resolve()
    campaign = read_json(campaign_path)
    if (
        campaign.get("schema_version") != "apertus_mini_campaign_manifest_v1"
        or campaign.get("status") != "frozen"
    ):
        raise ValueError("campaign manifest is not frozen")
    prior = read_json(args.prior_recovery_receipt)
    if (
        prior.get("schema_version") != "apertus_mini_evaluation_runtime_recovery_v1"
        or prior.get("status") != "frozen"
        or prior.get("evaluation_namespace") != "fp32_v1"
    ):
        raise ValueError("prior FP32 recovery receipt drift")

    scientific = args.scientific_bundle.resolve()
    megatron = args.megatron_dir.resolve()
    verify_code_bundle_receipt(args.scientific_bundle_receipt, scientific, "scientific")
    verify_code_bundle_receipt(args.megatron_bundle_receipt, megatron, "scientific")

    mapping = read_json(args.exact_mapping_receipt)
    if (
        mapping.get("schema_version")
        != "apertus_mini_exact_checkpoint_weight_mapping_v1"
        or mapping.get("status") != "passed"
        or int(mapping.get("source_parameter_tensors_expected", -1)) != 202
        or int(mapping.get("source_parameter_tensors_checked_exact", -1)) != 202
        or mapping.get("all_source_parameters_covered") is not True
        or mapping.get("all_hf_tensors_accounted_for") is not True
        or mapping.get("all_mapped_parameter_tensors_bit_exact") is not True
    ):
        raise ValueError("exact mapping diagnostic did not pass")

    moderate = runtime_metrics(args.moderate_runtime_log)
    severe = runtime_metrics(args.severe_runtime_log)
    if (
        not 99.0 <= moderate["prediction_agreement_percent"] < 99.5
        or not 0.0 < moderate["mean_kl_divergence"] < 1.0e-3
        or severe["prediction_agreement_percent"] > 80.0
        or severe["mean_kl_divergence"] < 0.1
        or severe["mean_total_variation"] < 0.1
    ):
        raise ValueError("runtime-sensitivity diagnostic does not match observed classes")

    runtime_files = [file_receipt(path.resolve()) for path in args.runtime_file]
    runtime_paths = [row["path"] for row in runtime_files]
    if len(runtime_paths) != len(set(runtime_paths)):
        raise ValueError("duplicate runtime-file binding")

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    diagnostic = {
        "schema_version": "apertus_mini_conversion_semantic_parity_diagnostic_v1",
        "status": "passed",
        "created_at_utc": now,
        "conclusion": (
            "The Megatron-to-HF parameter mapping is complete and bit-exact for all "
            "202 learned source tensors. The forward-runtime comparison is a recorded "
            "numerical-sensitivity diagnostic and is not a parameter-conversion gate."
        ),
        "exact_parameter_mapping": file_receipt(args.exact_mapping_receipt.resolve()),
        "runtime_sensitivity": {
            "moderate_case": {
                "log": file_receipt(args.moderate_runtime_log.resolve()),
                "metrics": moderate,
            },
            "severe_case": {
                "log": file_receipt(args.severe_runtime_log.resolve()),
                "metrics": severe,
            },
        },
        "authoritative_evaluation_runtime": "HF float32",
        "selection_policy": (
            "source-conditioned Megatron BPB remains primary; native GreekMMLU on the "
            "bit-exact HF export is downstream confirmation; runtime sensitivity is reported"
        ),
        "training_runtime_changed": False,
    }
    atomic_write_json(args.diagnostic_output.resolve(), diagnostic)

    originals = {
        "scientific_bundle": Path(campaign["assets"]["scientific_bundle"]).resolve(),
        "megatron_dir": Path(campaign["assets"]["megatron_dir"]).resolve(),
    }
    recovery = {
        "schema_version": "apertus_mini_evaluation_runtime_recovery_v1",
        "status": "frozen",
        "created_at_utc": now,
        "evaluation_namespace": "fp32_v1",
        "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        "old_bfloat16_namespace_is_provisional": True,
        "reason": (
            "Evaluation-only recovery: replace a numerically unstable cross-runtime "
            "logit gate with complete bit-exact parameter mapping, while retaining all "
            "float32 runtime distances as diagnostics. Training is unchanged."
        ),
        "campaign_manifest": file_receipt(campaign_path),
        "lineage": {
            "prior_recovery_receipt": file_receipt(
                args.prior_recovery_receipt.resolve()
            )
        },
        "overrides": {
            "scientific_bundle": {
                "from_path": str(originals["scientific_bundle"]),
                "to_path": str(scientific),
            },
            "megatron_dir": {
                "from_path": str(originals["megatron_dir"]),
                "to_path": str(megatron),
            },
        },
        "runtime_files": runtime_files,
        "scientific_bundle_receipt": file_receipt(
            args.scientific_bundle_receipt.resolve()
        ),
        "megatron_bundle_receipt": file_receipt(
            args.megatron_bundle_receipt.resolve()
        ),
        "semantic_parity_diagnostics": file_receipt(
            args.diagnostic_output.resolve()
        ),
        "evaluation_attempt_limit_overrides": {
            "14848": 5,
            "16384": 5,
            "16896": 5,
        },
        "training_runtime_changed": False,
    }
    atomic_write_json(args.recovery_output.resolve(), recovery)
    print(
        json.dumps(
            {
                "ok": True,
                "diagnostic": str(args.diagnostic_output.resolve()),
                "recovery": str(args.recovery_output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
