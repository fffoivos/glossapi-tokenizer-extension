#!/usr/bin/env python3
"""Apply the frozen dataset-order screen and paired endpoint uncertainty checks.

The source-conditioned validation jobs saved only panel aggregates, so this
script does not manufacture document-cluster intervals from checkpoint-to-
checkpoint variation.  It applies the only numeric retention margin frozen in
the project before the campaign (the 5% common-stability safety margin), then
uses the saved per-question predictions for paired benchmark uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

import numpy as np


ARMS = [
    "D0_mixed",
    "D1_hard_h_to_g",
    "D2_hard_g_to_h",
    "D3_gradual_h_to_g",
    "D4_gradual_g_to_h",
]
RETENTION_PANELS = [
    "code",
    "de",
    "english",
    "historical_polytonic",
    "math",
    "old_greek",
    "ru",
    "zh",
]
GLOSSAPI_PANELS = ["non_hplt", "greek_phd", "openarchives", "historical_polytonic"]
GENERAL_TASKS = [
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "global_mmlu",
    "xnli",
    "xcopa",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def question_metrics(row: dict) -> tuple[float, float, float, float]:
    answer = int(row["answer_index"])
    scores = [float(item["avg_logprob"]) for item in row["choice_scores"]]
    nll = logsumexp(scores) - scores[answer]
    bits = -float(row["choice_scores"][answer]["sum_logprob"]) / math.log(2.0)
    return float(bool(row["correct"])), nll, bits, float(row["correct_answer_utf8_bytes"])


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def paired_bootstrap(
    control: list[dict],
    candidate: list[dict],
    *,
    samples: int,
    seed: int,
    batch_size: int = 256,
) -> dict:
    if len(control) != len(candidate) or not control:
        raise ValueError("paired rows must be non-empty and equal-length")
    c = np.asarray([question_metrics(row) for row in control], dtype=np.float64)
    a = np.asarray([question_metrics(row) for row in candidate], dtype=np.float64)
    if not np.array_equal(c[:, 3], a[:, 3]):
        raise ValueError("paired correct-answer byte counts differ")
    n = len(control)
    rng = np.random.default_rng(seed)
    accuracy = np.empty(samples, dtype=np.float64)
    choice_nll = np.empty(samples, dtype=np.float64)
    answer_bpb = np.empty(samples, dtype=np.float64)
    cursor = 0
    while cursor < samples:
        width = min(batch_size, samples - cursor)
        indices = rng.integers(0, n, size=(width, n), endpoint=False)
        accuracy[cursor : cursor + width] = (a[indices, 0] - c[indices, 0]).mean(axis=1)
        choice_nll[cursor : cursor + width] = (a[indices, 1] - c[indices, 1]).mean(axis=1)
        answer_bpb[cursor : cursor + width] = (
            a[indices, 2].sum(axis=1) / a[indices, 3].sum(axis=1)
            - c[indices, 2].sum(axis=1) / c[indices, 3].sum(axis=1)
        )
        cursor += width
    control_correct = c[:, 0].astype(bool)
    candidate_correct = a[:, 0].astype(bool)
    control_only = int(np.sum(control_correct & ~candidate_correct))
    candidate_only = int(np.sum(~control_correct & candidate_correct))
    discordant = control_only + candidate_only
    tail = sum(math.comb(discordant, k) for k in range(min(control_only, candidate_only) + 1))
    mcnemar_p = min(1.0, 2.0 * float(tail / (2**discordant))) if discordant else 1.0
    return {
        "n": n,
        "bootstrap_samples": samples,
        "seed": seed,
        "delta_candidate_minus_d0": {
            "accuracy": {
                "point": float((a[:, 0] - c[:, 0]).mean()),
                "percentile_95_ci": percentile_interval(accuracy),
                "higher_is_better": True,
            },
            "choice_nll": {
                "point": float((a[:, 1] - c[:, 1]).mean()),
                "percentile_95_ci": percentile_interval(choice_nll),
                "lower_is_better": True,
            },
            "correct_answer_bpb": {
                "point": float(a[:, 2].sum() / a[:, 3].sum() - c[:, 2].sum() / c[:, 3].sum()),
                "percentile_95_ci": percentile_interval(answer_bpb),
                "lower_is_better": True,
            },
        },
        "mcnemar_accuracy": {
            "d0_correct_candidate_wrong": control_only,
            "d0_wrong_candidate_correct": candidate_only,
            "discordant": discordant,
            "exact_two_sided_p": mcnemar_p,
        },
    }


def metrics_from_rows(rows: list[dict]) -> dict:
    values = np.asarray([question_metrics(row) for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "accuracy": float(values[:, 0].mean()),
        "choice_nll": float(values[:, 1].mean()),
        "correct_answer_bpb": float(values[:, 2].sum() / values[:, 3].sum()),
    }


def find_one(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern!r} under {directory}, found {len(paths)}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_805)
    parser.add_argument("--retention-relative-margin", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    run = args.run_root.resolve()
    full_path = run / "full_endpoint_validation_recovery_v1" / "full_endpoint_validation_receipt.json"
    full = load_json(full_path)
    if full.get("status") != "completed" or full.get("row_count") != 65:
        raise ValueError("full endpoint validation receipt is not complete")
    full_by = {arm: {} for arm in ARMS}
    for row in full["rows"]:
        full_by[row["arm_id"]][row["panel"]] = float(row["bpb"])

    source_gate = {}
    d0 = full_by["D0_mixed"]
    for arm in ARMS:
        changes = {panel: (full_by[arm][panel] - d0[panel]) / d0[panel] for panel in RETENTION_PANELS}
        worst_panel = max(changes, key=changes.get)
        source_gate[arm] = {
            "status": "pass" if max(changes.values()) <= args.retention_relative_margin else "fail",
            "worst_panel": worst_panel,
            "worst_relative_regression": changes[worst_panel],
            "relative_change_vs_d0": changes,
        }

    q_h = 0.6905693562215841
    q_g = 1.0 - q_h
    endpoint_selection = {}
    for arm in ARMS:
        gloss = mean(full_by[arm][panel] for panel in GLOSSAPI_PANELS)
        endpoint_selection[arm] = {
            "source_safety_gate": source_gate[arm]["status"],
            "neutral_external_modern_greek_bpb": full_by[arm]["neutral_external_modern_greek"],
            "balanced_hplt_glossapi_bpb": 0.5 * (full_by[arm]["hplt"] + gloss),
            "natural_hplt_glossapi_bpb": q_h * full_by[arm]["hplt"] + q_g * gloss,
            "glossapi_macro_bpb": gloss,
        }
    passers = [arm for arm in ARMS if source_gate[arm]["status"] == "pass"]
    point_ranking = sorted(
        passers,
        key=lambda arm: (
            endpoint_selection[arm]["neutral_external_modern_greek_bpb"],
            endpoint_selection[arm]["balanced_hplt_glossapi_bpb"],
            endpoint_selection[arm]["glossapi_macro_bpb"],
        ),
    )

    gm_receipts = {}
    gm_rows = {}
    clean_ids = None
    evidence = {"full_endpoint_validation": file_receipt(full_path), "greekmmlu": {}, "greek_endpoints": {}, "general_retention": {}}
    for arm in ARMS:
        task_dir = run / "evaluations_fp32_v1" / "iteration_0038496" / "attempt_3" / "tasks" / arm
        receipt_path = task_dir / "exact_checkpoint_native_greekmmlu_receipt.json"
        receipt = load_json(receipt_path)
        predictions_path = Path(receipt["artifacts"]["predictions"]["path"])
        if sha256_file(predictions_path) != receipt["artifacts"]["predictions"]["sha256"]:
            raise ValueError(f"GreekMMLU prediction hash mismatch for {arm}")
        manifest_path = Path(receipt["clean_subset_manifest"]["path"])
        manifest = load_json(manifest_path)
        ids_path = Path(manifest["clean_example_ids"]["path"])
        ids = set(json.loads(line) if line.lstrip().startswith('"') else line.strip() for line in ids_path.open() if line.strip())
        if clean_ids is None:
            clean_ids = ids
        elif ids != clean_ids:
            raise ValueError("clean GreekMMLU ID sets differ across arms")
        rows = {str(row["example_id"]): row for row in load_jsonl(predictions_path)}
        filtered = [rows[key] for key in sorted(clean_ids)]
        measured = metrics_from_rows(filtered)
        expected = receipt["metrics"]["decontaminated"]
        for metric in ("accuracy", "choice_nll", "correct_answer_bpb"):
            if not math.isclose(measured[metric], float(expected[metric]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{arm} clean {metric} does not reproduce receipt")
        gm_rows[arm] = filtered
        gm_receipts[arm] = expected
        evidence["greekmmlu"][arm] = {"receipt": file_receipt(receipt_path), "predictions": file_receipt(predictions_path)}

    greekmmlu_paired = {}
    for index, arm in enumerate(ARMS[1:], start=1):
        greekmmlu_paired[arm] = paired_bootstrap(
            gm_rows["D0_mixed"],
            gm_rows[arm],
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )

    greek_endpoint_rows = {}
    greek_endpoint_metrics = {}
    for arm in ARMS:
        path = run / "greek_endpoint_benchmarks_recovery_v2" / "tasks" / arm / "predictions.jsonl"
        rows = load_jsonl(path)
        by_benchmark = {}
        for benchmark in sorted({str(row["benchmark"]) for row in rows}):
            by_benchmark[benchmark] = sorted(
                [row for row in rows if row["benchmark"] == benchmark], key=lambda row: str(row["example_id"])
            )
        greek_endpoint_rows[arm] = by_benchmark
        greek_endpoint_metrics[arm] = {benchmark: metrics_from_rows(group) for benchmark, group in by_benchmark.items()}
        evidence["greek_endpoints"][arm] = file_receipt(path)
    greek_endpoint_paired = {}
    for arm_index, arm in enumerate(ARMS[1:], start=1):
        greek_endpoint_paired[arm] = {}
        for benchmark_index, benchmark in enumerate(sorted(greek_endpoint_rows[arm])):
            control = greek_endpoint_rows["D0_mixed"][benchmark]
            candidate = greek_endpoint_rows[arm][benchmark]
            if [row["example_id"] for row in control] != [row["example_id"] for row in candidate]:
                raise ValueError(f"{benchmark} IDs do not align for {arm}")
            greek_endpoint_paired[arm][benchmark] = paired_bootstrap(
                control,
                candidate,
                samples=args.bootstrap_samples,
                seed=args.seed + 100 * arm_index + benchmark_index,
            )

    general_retention = {}
    for arm in ARMS:
        directory = run / "retention_endpoint_benchmarks_sharded_recovery_v2" / "tasks" / arm
        path = find_one(directory, "results_*.json")
        results = load_json(path)["results"]
        task_scores = {task: float(results[task]["acc,none"]) for task in GENERAL_TASKS}
        general_retention[arm] = {"task_accuracy": task_scores, "unweighted_task_macro_accuracy": mean(task_scores.values())}
        evidence["general_retention"][arm] = file_receipt(path)

    payload = {
        "schema_version": "apertus_dataset_order_selection_analysis_v1",
        "status": "completed",
        "run_root": str(run),
        "methods": {
            "source_retention_gate": {
                "threshold": args.retention_relative_margin,
                "comparison": "each arm relative to D0_mixed at the full endpoint",
                "panels": RETENTION_PANELS,
                "provenance": "the only numeric retention margin frozen in project code before the campaign: training/evaluate_common_stability_smoke.py; reused here as a safety-margin sensitivity, not misrepresented as a separately frozen final-winner margin",
            },
            "benchmark_uncertainty": {
                "paired_question_bootstrap_samples": args.bootstrap_samples,
                "percentile_interval": 0.95,
                "base_seed": args.seed,
                "accuracy_test": "exact two-sided McNemar",
            },
            "source_bpb_uncertainty": {
                "status": "unavailable_from_frozen_outputs",
                "reason": "full endpoint receipts contain aggregate panel losses but no per-document or document-cluster loss rows",
                "required_for_resolution": "rerun endpoint validation with per-document-cluster numerator and UTF-8-byte denominator outputs",
            },
            "general_benchmark_noninferiority": {
                "status": "not_formally_evaluable",
                "reason": "the design froze the requirement but no numeric final-selection margin was frozen before endpoint inspection",
            },
        },
        "source_retention_safety_gate": source_gate,
        "endpoint_selection_metrics": endpoint_selection,
        "point_estimate_selection": {
            "passing_arms": passers,
            "lexicographic_order": point_ranking,
            "observed_leader": point_ranking[0],
            "rule": ["neutral external Greek BPB", "balanced HPLT/GlossAPI BPB", "GlossAPI macro BPB"],
        },
        "greekmmlu_clean_endpoints": gm_receipts,
        "greekmmlu_paired_vs_d0": greekmmlu_paired,
        "greek_endpoint_benchmarks": greek_endpoint_metrics,
        "greek_endpoint_paired_vs_d0": greek_endpoint_paired,
        "general_retention_endpoints": general_retention,
        "decision": {
            "winner_selected": False,
            "provisional_observed_leader": point_ranking[0],
            "resolved_claim": "D0_mixed is the observed all-round leader under the predeclared hierarchy and the 5% source-retention safety screen; this is not yet a statistically resolved winner",
            "blocking_evidence": [
                "document-cluster uncertainty for primary source-conditioned BPB is absent from the frozen aggregate-only validation outputs",
                "no numeric general-benchmark noninferiority margin was frozen before endpoint inspection",
            ],
            "next_confirmatory_comparison": ["D0_mixed", "D3_gradual_h_to_g"],
        },
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "observed_leader": point_ranking[0], "source_gate": {k: v["status"] for k, v in source_gate.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
