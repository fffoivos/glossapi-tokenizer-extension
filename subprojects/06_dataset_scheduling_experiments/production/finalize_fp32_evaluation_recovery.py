#!/usr/bin/env python3
"""Freeze the evidence and runtime binding for an FP32 evaluation recovery."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path

from campaign_contract import (
    AUTHORITATIVE_EVALUATION_DTYPE,
    DEFAULT_EVALUATION_NAMESPACE,
    atomic_write_json,
    read_json,
    sha256_file,
    verify_code_bundle_receipt,
)


def file_receipt(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe recovery evidence: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def headline(path: Path, dtype: str) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"invalid GreekMMLU headline: {path}")
    row = rows[0]
    values = {
        "accuracy": row.get("accuracy"),
        "choice_nll": row.get("choice_nll"),
        "correct_answer_bpb": row.get("correct_answer_bpb"),
    }
    if (
        row.get("benchmark") != "greekmmlu"
        or row.get("subject") != "__all__"
        or int(row.get("n", -1)) != 16_632
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values.values()
        )
    ):
        raise ValueError(f"incomplete GreekMMLU headline: {path}")
    return {"dtype": dtype, "n": 16_632, **values, "artifact": file_receipt(path)}


def prediction_flips(fp32_path: Path, bf16_path: Path) -> int:
    fp32_rows = [
        json.loads(line) for line in fp32_path.open() if line.strip()
    ]
    bf16_rows = [
        json.loads(line) for line in bf16_path.open() if line.strip()
    ]
    if len(fp32_rows) != 16_632 or len(bf16_rows) != 16_632:
        raise ValueError("paired dtype diagnostic does not cover full GreekMMLU")
    flips = 0
    for fp32, bf16 in zip(fp32_rows, bf16_rows, strict=True):
        if fp32.get("example_id") != bf16.get("example_id"):
            raise ValueError("paired dtype diagnostic example-order drift")
        flips += int(fp32.get("pred_index") != bf16.get("pred_index"))
    return flips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--scientific-bundle", type=Path, required=True)
    parser.add_argument("--scientific-bundle-receipt", type=Path, required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--authoritative-export-receipt", type=Path, required=True)
    parser.add_argument("--bf16-parity-log", type=Path, required=True)
    parser.add_argument("--fp32-d0-headline", type=Path, required=True)
    parser.add_argument("--bf16-d0-headline", type=Path, required=True)
    parser.add_argument("--fp32-d4-headline", type=Path, required=True)
    parser.add_argument("--bf16-d4-headline", type=Path, required=True)
    parser.add_argument("--fp32-d4-predictions", type=Path, required=True)
    parser.add_argument("--bf16-d4-predictions", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, action="append", required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--recovery-output", type=Path, required=True)
    args = parser.parse_args()

    campaign = read_json(args.campaign_manifest)
    if (
        campaign.get("schema_version") != "apertus_mini_campaign_manifest_v1"
        or campaign.get("status") != "frozen"
    ):
        raise ValueError("campaign manifest is not frozen")
    verify_code_bundle_receipt(
        args.scientific_bundle_receipt,
        args.scientific_bundle,
        "scientific",
    )
    export = read_json(args.authoritative_export_receipt)
    conversion = export.get("conversion", {})
    metrics = conversion.get("semantic_probability_metrics", {})
    if (
        export.get("status") != "completed"
        or export.get("ready_for_frozen_native_greekmmlu") is not True
        or conversion.get("semantic_parity_dtype")
        != AUTHORITATIVE_EVALUATION_DTYPE
        or conversion.get("parity_gate_policy")
        != "float32_probability_tail_v2"
        or conversion.get("semantic_probability_thresholds_passed") is not True
        or float(conversion.get("prediction_agreement_percent", 0.0)) < 99.5
        or float(metrics.get("p99_top_token_logprob_abs_difference", math.inf))
        > 0.075
        or float(metrics.get("p999_top_token_logprob_abs_difference", math.inf))
        > 0.2
    ):
        raise ValueError("authoritative FP32 conversion did not pass the recovery gate")
    bf16_log = args.bf16_parity_log.read_text(encoding="utf-8", errors="replace")
    required_bf16_fragments = (
        "Converted semantic parity dtype: bfloat16",
        "Converted model agrees on 89.19% of predictions",
        "Converted mean KL divergence: 1.812",
    )
    if not all(fragment in bf16_log for fragment in required_bf16_fragments):
        raise ValueError("BF16 parity-failure evidence drift")

    paired = {
        "D0_mixed": {
            "float32": headline(args.fp32_d0_headline, "float32"),
            "bfloat16": headline(args.bf16_d0_headline, "bfloat16"),
        },
        "D4_gradual_g_to_h": {
            "float32": headline(args.fp32_d4_headline, "float32"),
            "bfloat16": headline(args.bf16_d4_headline, "bfloat16"),
        },
    }
    fp32_order = (
        paired["D4_gradual_g_to_h"]["float32"]["choice_nll"]
        < paired["D0_mixed"]["float32"]["choice_nll"]
    )
    bf16_order = (
        paired["D4_gradual_g_to_h"]["bfloat16"]["choice_nll"]
        < paired["D0_mixed"]["bfloat16"]["choice_nll"]
    )
    if fp32_order == bf16_order:
        raise ValueError("paired diagnostic no longer proves dtype-dependent arm ordering")
    flips = prediction_flips(
        args.fp32_d4_predictions, args.bf16_d4_predictions
    )
    if flips <= 0:
        raise ValueError("paired diagnostic found no FP32/BF16 answer flips")

    diagnostic = {
        "schema_version": "apertus_mini_conversion_semantic_parity_diagnostic_v1",
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": "replace_provisional_bfloat16_greekmmlu_with_authoritative_float32",
        "training_runtime_changed": False,
        "evaluation_namespace": DEFAULT_EVALUATION_NAMESPACE,
        "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        "bf16_semantic_parity_failed": True,
        "paired_d4_prediction_choice_flips": flips,
        "paired_d0_d4_choice_nll": paired,
        "dtype_reversed_d0_d4_choice_nll_order": True,
        "authoritative_fp32_export": file_receipt(
            args.authoritative_export_receipt
        ),
        "bf16_parity_log": file_receipt(args.bf16_parity_log),
        "paired_prediction_artifacts": {
            "float32": file_receipt(args.fp32_d4_predictions),
            "bfloat16": file_receipt(args.bf16_d4_predictions),
        },
        "conversion_gate": conversion,
    }
    atomic_write_json(args.diagnostic_output, diagnostic)

    originals = campaign["assets"]
    runtime_files = [file_receipt(path) for path in args.runtime_file]
    if len({row["path"] for row in runtime_files}) != len(runtime_files):
        raise ValueError("duplicate recovery runtime-file binding")
    recovery = {
        "schema_version": "apertus_mini_evaluation_runtime_recovery_v1",
        "status": "frozen",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": (
            "Evaluation-only recovery: BF16 inference changed the D0/D4 ordering. "
            "Replay all ready checkpoints with FP32, an isolated namespace, and "
            "a robust float32 probability-tail conversion gate. Training is unchanged."
        ),
        "evaluation_namespace": DEFAULT_EVALUATION_NAMESPACE,
        "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        "campaign_manifest": file_receipt(args.campaign_manifest),
        "overrides": {
            "scientific_bundle": {
                "from_path": str(Path(originals["scientific_bundle"]).resolve()),
                "to_path": str(args.scientific_bundle.resolve()),
            },
            "megatron_dir": {
                "from_path": str(Path(originals["megatron_dir"]).resolve()),
                "to_path": str(args.megatron_dir.resolve()),
            },
        },
        "runtime_files": runtime_files,
        "scientific_bundle_receipt": file_receipt(
            args.scientific_bundle_receipt
        ),
        "semantic_parity_diagnostics": file_receipt(args.diagnostic_output),
        "evaluation_attempt_limit_overrides": {},
        "old_bfloat16_namespace_is_provisional": True,
        "training_runtime_changed": False,
    }
    atomic_write_json(args.recovery_output, recovery)
    print(
        json.dumps(
            {
                "ok": True,
                "prediction_flips": flips,
                "diagnostic": str(args.diagnostic_output.resolve()),
                "recovery": str(args.recovery_output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
