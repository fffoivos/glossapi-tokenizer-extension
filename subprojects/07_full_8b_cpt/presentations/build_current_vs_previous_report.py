#!/usr/bin/env python3
"""Build a receipt-bound, single-page comparison of the live sanitized 8B CPT rerun."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any


TOKENS_PER_UPDATE = 4_194_304
OLD_UPDATES = 19_248
NEW_UPDATES = 18_284
WARMUP_END = 400
OLD_COOLDOWN_START = 15_398
NEW_COOLDOWN_START = 14_627
PEAK_LR = 5.5e-5
MIN_LR = 5.5e-6
GREEKMMLU_UPDATES = (
    0,
    400,
    1_192,
    2_384,
    3_576,
    4_768,
    5_960,
    7_152,
    8_344,
    9_536,
    10_728,
    11_920,
    13_112,
    14_304,
    14_627,
    15_496,
    16_688,
    17_880,
    18_284,
)

LEARNING = (
    "hplt",
    "non_hplt",
    "openarchives",
    "greek_phd",
    "historical_polytonic",
    "neutral_external_modern_greek",
)
RETENTION = ("english", "code", "math", "de", "ru", "zh", "old_greek")
LABELS = {
    "hplt": "HPLT broad Greek",
    "non_hplt": "GlossAPI / non-HPLT",
    "openarchives": "OpenArchives",
    "greek_phd": "Greek PhD",
    "historical_polytonic": "Historical polytonic",
    "neutral_external_modern_greek": "Neutral external Greek",
    "english": "English",
    "code": "Code",
    "math": "Math",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "old_greek": "Old Greek",
}

TRAIN_LINE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?consumed tokens:\s*([0-9.]+)B.*?"
    r"elapsed time per iteration \(ms\):\s*([0-9.]+).*?lm loss:\s*([0-9.Ee+-]+).*?"
    r"number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_parser(repo_root: Path):
    path = repo_root / "subprojects/06_dataset_scheduling_experiments/evaluation/collect_validation_trajectory.py"
    spec = importlib.util.spec_from_file_location("full8_validation_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_log, path


def collect_validation(paths: list[Path], parse_log) -> dict[str, list[dict[str, float]]]:
    rows: dict[tuple[int, str], dict[str, float]] = {}
    for path in paths:
        for row in parse_log(path):
            rows[(int(row["iteration"]), str(row["panel"]))] = row
    return {
        panel: [
            {
                "iteration": iteration,
                "bpb": float(row["bpb"]),
                "lm_loss": float(row["lm_loss"]),
                "base_target_count": float(row["base_target_count"]),
                "added_target_count": float(row["added_target_count"]),
                "base_target_bytes": float(row["base_target_bytes"]),
                "added_target_bytes": float(row["added_target_bytes"]),
            }
            for (iteration, name), row in sorted(rows.items())
            if name == panel
        ]
        for panel in sorted(set(LEARNING) | set(RETENTION))
    }


def filter_incomplete_validation_records(
    validation: dict[str, list[dict[str, float]]],
) -> tuple[dict[str, list[dict[str, float]]], list[dict[str, Any]]]:
    """Drop records truncated before their normally present added-token fields.

    Target counts legitimately vary between validation invocations, so a modal
    geometry signature is not a safe completeness test.  The observed logging
    failure instead leaves both added-token fields at the parser's zero default
    in a panel whose complete records almost always contain added-token targets.
    """
    filtered: dict[str, list[dict[str, float]]] = {}
    excluded: list[dict[str, Any]] = []
    for panel, rows in validation.items():
        positive_added = sum(
            float(row["added_target_count"]) > 0 and float(row["added_target_bytes"]) > 0
            for row in rows
        )
        added_fields_expected = len(rows) >= 5 and positive_added / len(rows) > 0.5
        filtered[panel] = []
        for row in rows:
            missing_added_fields = (
                float(row["added_target_count"]) == 0
                and float(row["added_target_bytes"]) == 0
            )
            total_targets = float(row["base_target_count"]) + float(row["added_target_count"])
            total_bytes = float(row["base_target_bytes"]) + float(row["added_target_bytes"])
            expected_bpb = float(row["lm_loss"]) * total_targets / total_bytes / math.log(2)
            inconsistent_bpb = abs(float(row["bpb"]) - expected_bpb) > 0.01
            if (added_fields_expected and missing_added_fields) or inconsistent_bpb:
                excluded.append(
                    {
                        "panel": panel,
                        "iteration": int(row["iteration"]),
                        "reason": (
                            "truncated log record produced BPB inconsistent with aggregate loss/count/byte identity"
                            if inconsistent_bpb
                            else "truncated log record omitted normally present added-token fields"
                        ),
                    }
                )
            else:
                filtered[panel].append(row)
    return filtered, excluded


def add_old_initial(validation: dict[str, list[dict[str, float]]], path: Path) -> None:
    receipt = read_json(path)
    for row in receipt["panels"]:
        panel = row["panel"]
        if panel not in validation:
            continue
        validation[panel].insert(
            0,
            {
                "iteration": 0,
                "bpb": float(row["bpb"]),
                "lm_loss": float(row["lm_loss"]),
                "base_target_count": float(row["base_target_count"]),
                "added_target_count": float(row.get("added_target_count", 0.0)),
                "base_target_bytes": float(row["base_target_bytes"]),
                "added_target_bytes": float(row.get("added_target_bytes", 0.0)),
            },
        )


def panel_signature(row: dict[str, float]) -> tuple[float, float, float, float]:
    return tuple(
        round(float(row[key]), 6)
        for key in ("base_target_count", "added_target_count", "base_target_bytes", "added_target_bytes")
    )


def collect_training(paths: list[Path]) -> list[dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = TRAIN_LINE.search(line)
            if match is None:
                continue
            iteration, planned, consumed_b, step_ms, loss, skipped, nan = match.groups()
            rows[int(iteration)] = {
                "iteration": int(iteration),
                "planned": int(planned),
                "consumed_tokens_b": float(consumed_b),
                "step_ms": float(step_ms),
                "loss": float(loss),
                "skipped": int(skipped),
                "nan": int(nan),
            }
    return [rows[key] for key in sorted(rows)]


def collect_greekmmlu(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("iter_*/exact_checkpoint_native_greekmmlu_receipt.json")):
        value = read_json(path)
        if value.get("status") != "completed":
            continue
        iteration = int(path.parent.name.split("_")[1])
        metrics = value["metrics"]
        rows.append(
            {
                "iteration": iteration,
                "accuracy": float(metrics["accuracy"]),
                "choice_nll": float(metrics["choice_nll"]),
                "correct_answer_bpb": float(metrics["correct_answer_bpb"]),
                "n": int(metrics["n"]),
                "clean_accuracy": float(metrics["decontaminated"]["accuracy"]),
                "clean_choice_nll": float(metrics["decontaminated"]["choice_nll"]),
                "clean_correct_answer_bpb": float(metrics["decontaminated"]["correct_answer_bpb"]),
                "clean_n": int(metrics["decontaminated"]["n"]),
            }
        )
    return sorted(rows, key=lambda row: int(row["iteration"]))


def snapshot_path(snapshot_root: Path, remote_path: str) -> Path:
    path = snapshot_root / remote_path.lstrip("/")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def collect_completion_evidence(snapshot_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    completion_paths = list(snapshot_root.rglob("campaign_evidence_completion_receipt.json"))
    if len(completion_paths) != 1:
        raise ValueError(f"expected one campaign completion receipt, found {len(completion_paths)}")
    completion = read_json(completion_paths[0])
    if completion.get("status") != "completed":
        raise ValueError("campaign completion receipt is not completed")

    def verify_bound(item: dict[str, Any], label: str) -> Path:
        path = snapshot_path(snapshot_root, item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"{label} hash mismatch: {path}")
        return path

    for key in ("launch_gate", "selected_profile", "training_completion"):
        verify_bound(completion[key], key)
    verify_bound(completion["terminal_model_export"]["receipt"], "terminal model export")
    for item in completion["training_attempt_audits"]:
        verify_bound(item, "training attempt audit")
    counts = completion["counts"]
    if int(counts["source_validated_segments"]) != len(completion["training_attempt_audits"]):
        raise ValueError("training-attempt audit count mismatch")

    greekmmlu: list[dict[str, Any]] = []
    for item in completion["greekmmlu_receipts"]:
        path = verify_bound(item, "GreekMMLU receipt")
        value = read_json(path)
        metrics = value["metrics"]
        full_metrics = metrics.get("full", metrics)
        match = re.search(r"/iter_(\d+)/", item["path"])
        iteration = int(match.group(1)) if match else 0
        greekmmlu.append(
            {
                "iteration": iteration,
                "accuracy": float(full_metrics["accuracy"]),
                "choice_nll": float(full_metrics["choice_nll"]),
                "correct_answer_bpb": float(full_metrics["correct_answer_bpb"]),
                "n": int(full_metrics["n"]),
                "clean_accuracy": float(metrics["decontaminated"]["accuracy"]),
                "clean_choice_nll": float(metrics["decontaminated"]["choice_nll"]),
                "clean_correct_answer_bpb": float(metrics["decontaminated"]["correct_answer_bpb"]),
                "clean_n": int(metrics["decontaminated"]["n"]),
                "receipt_path": item["path"],
                "receipt_sha256": item["sha256"],
            }
        )

    per_document: dict[str, list[dict[str, Any]]] = {
        panel: [] for panel in sorted(set(LEARNING) | set(RETENTION))
    }
    for item in completion["per_document_panel_receipts"]:
        path = verify_bound(item, "per-document receipt")
        value = read_json(path)
        panel = path.name.removesuffix(".receipt.json")
        if panel not in per_document:
            raise ValueError(f"unexpected validation panel: {panel}")
        match = re.search(r"/iter_(\d+)/", item["path"])
        iteration = int(match.group(1)) if match else 0
        aggregate = value["aggregate"]
        per_document[panel].append(
            {
                "iteration": iteration,
                "bpb": float(aggregate["bpb"]),
                "mean_nll": float(aggregate["mean_nll"]),
                "documents": int(aggregate["documents"]),
                "target_tokens": int(aggregate["target_tokens"]),
                "receipt_path": item["path"],
                "receipt_sha256": item["sha256"],
            }
        )
    for panel, rows in per_document.items():
        rows.sort(key=lambda row: int(row["iteration"]))
        if [int(row["iteration"]) for row in rows] != [0, NEW_COOLDOWN_START, NEW_UPDATES]:
            raise ValueError(f"incomplete per-document trajectory for {panel}: {rows}")

    greekmmlu.sort(key=lambda row: int(row["iteration"]))
    if [int(row["iteration"]) for row in greekmmlu] != list(GREEKMMLU_UPDATES):
        raise ValueError("GreekMMLU milestone set does not match the frozen schedule")
    if int(counts["greekmmlu"]) != len(greekmmlu):
        raise ValueError("GreekMMLU receipt count mismatch")
    if int(counts["per_document_panels"]) != sum(len(rows) for rows in per_document.values()):
        raise ValueError("per-document receipt count mismatch")
    return completion, greekmmlu, per_document


def running_gap(rows: list[dict[str, float]]) -> list[list[float]]:
    best = math.inf
    result = []
    for row in rows:
        best = min(best, row["bpb"])
        result.append([row["iteration"], row["bpb"] - best])
    return result


def nearest(rows: list[dict[str, float]], iteration: int) -> dict[str, float]:
    return min(rows, key=lambda row: abs(row["iteration"] - iteration))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def build_payload(
    repo_root: Path,
    evidence: Path,
    *,
    running_greekmmlu: int | None = None,
    final_evidence_root: Path | None = None,
) -> dict[str, Any]:
    parse_log, parser_path = load_validation_parser(repo_root)
    old_log_root = evidence / "old/logs"
    if not old_log_root.exists():
        old_log_root = repo_root / ".codex_tmp/full8_stop_report/logs"
    old_logs = sorted(old_log_root.glob("segments_*_training.log"))
    new_logs = sorted((evidence / "new/logs").glob("*.log"))
    if len(old_logs) != 3 or len(new_logs) < 3:
        raise ValueError(
            f"expected three old logs and at least three rerun logs, "
            f"got old={len(old_logs)} new={len(new_logs)}"
        )
    old_validation, old_validation_exclusions = filter_incomplete_validation_records(
        collect_validation(old_logs, parse_log)
    )
    add_old_initial(old_validation, repo_root / ".codex_tmp/full8_stop_report/initial_validation_receipt.json")
    new_validation, new_validation_exclusions = filter_incomplete_validation_records(
        collect_validation(new_logs, parse_log)
    )
    old_training = collect_training(old_logs)
    new_training = collect_training(new_logs)
    old_gm = collect_greekmmlu(evidence / "old/greekmmlu")
    completion = None
    per_document_validation = None
    if final_evidence_root is not None:
        completion, new_gm, per_document_validation = collect_completion_evidence(final_evidence_root)
    else:
        new_gm = collect_greekmmlu(evidence / "new/greekmmlu")

    comparable = {}
    for panel in sorted(set(LEARNING) | set(RETENTION)):
        old_nonzero = next(row for row in old_validation[panel] if row["iteration"] > 0)
        new_first = new_validation[panel][0]
        comparable[panel] = panel_signature(old_nonzero) == panel_signature(new_first)

    common_validation_iteration = min(
        max(row["iteration"] for values in old_validation.values() for row in values),
        max(row["iteration"] for values in new_validation.values() for row in values),
    )
    # Both runs have an exact validation pass at 7,152; use it when available.
    comparison_iteration = 7_152 if all(
        any(row["iteration"] == 7_152 for row in values)
        for values in (*old_validation.values(), *new_validation.values())
    ) else common_validation_iteration
    validation_comparison = []
    for panel in sorted(set(LEARNING) | set(RETENTION)):
        old_row = nearest(old_validation[panel], comparison_iteration)
        new_row = nearest(new_validation[panel], comparison_iteration)
        validation_comparison.append(
            {
                "panel": panel,
                "comparable": comparable[panel],
                "old_iteration": int(old_row["iteration"]),
                "new_iteration": int(new_row["iteration"]),
                "old_bpb": float(old_row["bpb"]),
                "new_bpb": float(new_row["bpb"]),
                "delta": float(new_row["bpb"] - old_row["bpb"]),
            }
        )

    old_gm_by_iteration = {row["iteration"]: row for row in old_gm}
    benchmark_comparison = []
    for row in new_gm:
        old_row = old_gm_by_iteration.get(row["iteration"])
        if old_row is None:
            continue
        benchmark_comparison.append(
            {
                "iteration": row["iteration"],
                "old_accuracy": old_row["clean_accuracy"],
                "new_accuracy": row["clean_accuracy"],
                "delta_accuracy": row["clean_accuracy"] - old_row["clean_accuracy"],
                "old_nll": old_row["clean_choice_nll"],
                "new_nll": row["clean_choice_nll"],
                "delta_nll": row["clean_choice_nll"] - old_row["clean_choice_nll"],
                "old_bpb": old_row["clean_correct_answer_bpb"],
                "new_bpb": row["clean_correct_answer_bpb"],
                "delta_bpb": row["clean_correct_answer_bpb"] - old_row["clean_correct_answer_bpb"],
            }
        )

    old_pool = read_json(evidence / "evidence/old_pool_corpus_receipt.json")
    new_pool = read_json(evidence / "evidence/new_pool_corpus_receipt.json")
    old_schedule = read_json(evidence / "evidence/old_schedule_manifest.json")
    new_schedule = read_json(evidence / "evidence/new_schedule_manifest.json")
    bridge = read_json(evidence / "evidence/sanitized_bridge_receipt.json")
    dedup = read_json(evidence / "evidence/postmask_dedup_receipt.json")

    latest = new_training[-1]
    recent_start = max(1, int(latest["iteration"]) - 750)
    recent_steps = [
        row["step_ms"] / 1000.0
        for row in new_training
        if row["iteration"] >= recent_start
    ]
    masked_documents = sum(int(value["masked_documents"]) for value in bridge["pool_counts"].values())
    old_active = int(old_pool["integer_79_20_1_geometry"]["active_tokens"])
    new_active = int(new_pool["integer_79_20_1_geometry"]["active_tokens"])

    bindings = [
        {"role": "validation_parser", "path": str(parser_path.resolve()), "sha256": sha256_file(parser_path)},
    ]
    for role, paths in (("old_training_log", old_logs), ("new_training_log", new_logs)):
        for path in paths:
            bindings.append({"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)})
    for path in sorted((evidence / "old/greekmmlu").glob("iter_*/*.json")) + sorted((evidence / "new/greekmmlu").glob("iter_*/*.json")):
        bindings.append({"role": "greekmmlu_receipt", "path": str(path.resolve()), "sha256": sha256_file(path)})
    for path in sorted((evidence / "evidence").glob("*.json")):
        bindings.append({"role": "data_contract", "path": str(path.resolve()), "sha256": sha256_file(path)})
    if final_evidence_root is not None:
        for path in sorted(final_evidence_root.rglob("*.json")):
            bindings.append({"role": "final_campaign_evidence", "path": str(path.resolve()), "sha256": sha256_file(path)})

    latest_greekmmlu = max(row["iteration"] for row in new_gm)
    next_greekmmlu = next(
        (value for value in GREEKMMLU_UPDATES if value > latest_greekmmlu),
        None,
    )
    if next_greekmmlu is None:
        next_greekmmlu_status = "all frozen milestones complete"
    elif running_greekmmlu == next_greekmmlu:
        next_greekmmlu_status = "running at snapshot"
    else:
        next_greekmmlu_status = "not yet complete at snapshot"

    return {
        "meta": {
            "title": "Apertus 8B CPT — final sanitized-run results",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "snapshot_iteration": int(latest["iteration"]),
            "snapshot_tokens_b": float(latest["consumed_tokens_b"]),
            "snapshot_fraction": int(latest["iteration"]) / NEW_UPDATES,
            "last_validation_iteration": max(row["iteration"] for values in new_validation.values() for row in values),
            "planned_updates": NEW_UPDATES,
            "active_tokens": new_active,
            "old_active_tokens": old_active,
            "active_token_change": new_active - old_active,
            "active_token_change_fraction": new_active / old_active - 1.0,
            "median_step_seconds": median(recent_steps),
            "p90_step_seconds": percentile(recent_steps, 0.9),
            "latest_training_loss": float(latest["loss"]),
            "skipped_updates": max(row["skipped"] for row in new_training),
            "nan_updates": max(row["nan"] for row in new_training),
            "comparison_iteration": comparison_iteration,
            "greekmmlu_latest_complete": latest_greekmmlu,
            "greekmmlu_next": next_greekmmlu,
            "greekmmlu_next_status": next_greekmmlu_status,
            "greekmmlu_required_milestones": len(GREEKMMLU_UPDATES),
            "greekmmlu_complete_milestones": len(new_gm) if any(row["iteration"] == 0 for row in new_gm) else len(new_gm) + 1,
            "raw_points_no_smoothing": True,
        },
        "constants": {
            "tokens_per_update": TOKENS_PER_UPDATE,
            "peak_lr": PEAK_LR,
            "minimum_lr": MIN_LR,
            "warmup_end": WARMUP_END,
            "old_updates": OLD_UPDATES,
            "new_updates": NEW_UPDATES,
            "old_cooldown_start": OLD_COOLDOWN_START,
            "new_cooldown_start": NEW_COOLDOWN_START,
        },
        "labels": LABELS,
        "learning_panels": LEARNING,
        "retention_panels": RETENTION,
        "comparable_panels": comparable,
        "validation_exclusions": {
            "old": old_validation_exclusions,
            "new": new_validation_exclusions,
        },
        "old_validation": old_validation,
        "new_validation": new_validation,
        "old_training": old_training,
        "new_training": new_training,
        "old_greekmmlu": old_gm,
        "new_greekmmlu": new_gm,
        "per_document_validation": per_document_validation,
        "completion": completion,
        "validation_comparison": validation_comparison,
        "benchmark_comparison": benchmark_comparison,
        "data": {
            "old_geometry": old_pool["integer_79_20_1_geometry"],
            "new_geometry": new_pool["integer_79_20_1_geometry"],
            "old_modern": old_pool["modern_greek"],
            "new_modern": new_pool["modern_greek"],
            "old_schedule_hash": old_schedule["common_contract"]["canonical_sequence_inventory_sha256"],
            "new_schedule_hash": new_schedule["common_contract"]["canonical_sequence_inventory_sha256"],
            "masked_documents": masked_documents,
            "dedup_counts": dedup["counts"],
            "anonymization": bridge["anonymization"],
        },
        "bindings": bindings,
    }


STYLE = r"""
:root{--paper:#f4f0e7;--paper2:#fbf8f1;--ink:#26272a;--muted:#6b6a67;--line:#cfc6b8;--old:#6f7f92;--new:#147b75;--red:#a5534f;--gold:#a78134;--blue:#486b91;--purple:#735f88;--green:#667a58}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.48}
main{max-width:1520px;margin:auto;padding:44px 52px 96px}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:500;margin:0}h1{font-size:clamp(46px,6.2vw,86px);line-height:.94;letter-spacing:-.045em;max-width:1220px}h2{font-size:37px;margin-bottom:9px}h3{font-size:20px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.18em;color:var(--red);font-weight:800}.lede{font-family:Georgia,serif;font-size:24px;max-width:1120px;color:#484743}.hero{padding:48px 0 44px;border-bottom:1px solid var(--ink)}.hero-meta{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-top:32px}.metric{background:rgba(255,255,255,.42);border-top:3px solid var(--new);padding:17px}.metric b{display:block;font-family:Georgia,serif;font-size:30px;font-weight:500}.metric span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.finding{margin-top:23px;padding:19px 22px;background:#efe3d6;border-left:4px solid var(--red);max-width:1200px}.finding.good{border-left-color:var(--new);background:#e3ece7}.section{padding:56px 0 22px;border-bottom:1px solid var(--line)}.section-intro{max-width:1120px;color:var(--muted);margin:0 0 26px}.chart-shell{background:var(--paper2);border:1px solid var(--line);padding:20px;margin:19px 0}.chart-title{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:7px}.chart-title small{color:var(--muted)}svg.chart{display:block;width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.mini{background:var(--paper2);border-top:3px solid var(--new);padding:16px}.mini.changed{border-top-color:var(--gold)}.mini h3{display:flex;justify-content:space-between;gap:8px}.mini h3 span{font-family:Inter,sans-serif;font-size:11px;color:var(--muted);font-weight:500}.benchmark-stack{display:grid;grid-template-columns:1fr;gap:22px}.benchmark-chart{background:var(--paper2);border-top:3px solid var(--new);padding:19px}.benchmark-chart h3{display:flex;justify-content:space-between;gap:10px}.benchmark-chart h3 span{font-family:Inter,sans-serif;font-size:12px;color:var(--muted);font-weight:500}.zoom{border-top-color:var(--purple)}.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:9px 0}.swatch{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}.swatch.old{background:var(--old)}.swatch.new{background:var(--new)}.notes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.note{background:var(--paper2);padding:21px;border-left:3px solid var(--blue)}.note.caution{border-left-color:var(--gold)}.note.good{border-left-color:var(--new)}.table-wrap{overflow:auto;background:var(--paper2);border:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #ddd5c8;font-size:13px}th:first-child,td:first-child{text-align:left}th{background:#e9e3d8;text-transform:uppercase;letter-spacing:.06em;font-size:11px}.better{color:var(--new);font-weight:700}.worse{color:var(--red);font-weight:700}.changed-text{color:var(--gold)}.mono{font-variant-numeric:tabular-nums}code{font-size:.9em;overflow-wrap:anywhere}.footer{padding-top:38px;color:var(--muted);font-size:12px}.axis{stroke:#a9a195;stroke-width:1}.gridline{stroke:#ddd6ca;stroke-width:1}.tick{font:11px Inter,sans-serif;fill:#716d66}.series{fill:none;stroke-width:2.15}.series.old{stroke:var(--old)}.series.new{stroke:var(--new)}.series.raw{stroke-width:.65;opacity:.23}.series.ideal{stroke-dasharray:6 5}.point-old{fill:var(--paper2);stroke:var(--old);stroke-width:1.4}.point-new{fill:var(--paper2);stroke:var(--new);stroke-width:1.4}.vline{stroke:var(--red);stroke-width:1;stroke-dasharray:4 4}.annotation{font:11px Inter,sans-serif;fill:var(--red)}.endpoint{font:600 11px Inter,sans-serif}.leader{stroke-width:1;opacity:.65}.bar-label{font:12px Inter,sans-serif;fill:var(--ink)}.bar-value{font:600 11px Inter,sans-serif;fill:var(--ink)}
@media(max-width:1050px){main{padding:30px 24px 78px}.hero-meta{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:720px){main{padding:24px 14px 64px}.hero-meta,.grid,.notes{grid-template-columns:1fr}.metric b{font-size:26px}.chart-shell,.mini,.benchmark-chart{padding:12px}h2{font-size:31px}.lede{font-size:21px}}@media print{body{background:#fff}main{max-width:none;padding:20px}.section{break-inside:auto}.mini,.chart-shell,.note,.benchmark-chart{break-inside:avoid}}
"""


SCRIPT = r"""
const D=JSON.parse(document.getElementById('report-data').textContent),NS='http://www.w3.org/2000/svg';
function s(tag,a={}){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e}
function moving(rows,n=101){const out=[];let sum=0,q=[];for(const p of rows){q.push(p);sum+=p[1];if(q.length>n)sum-=q.shift()[1];out.push([p[0],sum/q.length])}return out}
function fmt(v,opt){return opt.percent?`${(100*v).toFixed(opt.yDigits??2)}%`:v.toFixed(opt.yDigits??3)}
function plot(el,series,opt={}){const W=opt.width??1000,H=opt.height??360,m={l:72,r:opt.endLabels?138:28,t:24,b:44},svg=s('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':opt.label||'Training chart'});el.appendChild(svg);const all=series.flatMap(x=>x.values).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1])&&(opt.xmin===undefined||p[0]>=opt.xmin)&&(opt.xmax===undefined||p[0]<=opt.xmax));if(!all.length)return;let xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);let xmin=opt.xmin??Math.min(...xs),xmax=opt.xmax??Math.max(...xs),ymin=opt.ymin??Math.min(...ys),ymax=opt.ymax??Math.max(...ys);if(xmax===xmin)xmax=xmin+1;if(ymax===ymin){ymax+=1;ymin-=1}const pad=(ymax-ymin)*.08;if(opt.ymin===undefined)ymin-=pad;if(opt.ymax===undefined)ymax+=pad;const X=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),Y=y=>m.t+(ymax-y)/(ymax-ymin)*(H-m.t-m.b);
for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4,py=Y(y);svg.appendChild(s('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'gridline'}));const t=s('text',{x:m.l-9,y:py+4,'text-anchor':'end',class:'tick'});t.textContent=opt.percent?`${(100*y).toFixed(1)}%`:y.toFixed(opt.yDigits??3);svg.appendChild(t)}
for(let i=0;i<=4;i++){const x=xmin+(xmax-xmin)*i/4,px=X(x),t=s('text',{x:px,y:H-14,'text-anchor':'middle',class:'tick'});t.textContent=opt.tokens?`${(x*D.constants.tokens_per_update/1e9).toFixed(1)}B`:Math.round(x).toLocaleString();svg.appendChild(t)}svg.appendChild(s('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));svg.appendChild(s('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));
for(const marker of opt.markers||[]){if(marker.x<xmin||marker.x>xmax)continue;const x=X(marker.x);svg.appendChild(s('line',{x1:x,y1:m.t,x2:x,y2:H-m.b,class:'vline'}));const t=s('text',{x:x+5,y:m.t+12,class:'annotation'});t.textContent=marker.label;svg.appendChild(t)}
const ends=[];for(const item of series){const vals=item.values.filter(p=>p[0]>=xmin&&p[0]<=xmax&&Number.isFinite(p[1])),d=vals.map((p,i)=>`${i?'L':'M'}${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join(' ');svg.appendChild(s('path',{d,class:`series ${item.className||''}`}));if(item.points)for(const p of vals)svg.appendChild(s('circle',{cx:X(p[0]),cy:Y(p[1]),r:opt.pointRadius??3,class:item.className?.includes('old')?'point-old':'point-new'}));if(opt.endLabels&&vals.length)ends.push({item,p:vals[vals.length-1],actualY:Y(vals[vals.length-1][1])})}
if(opt.endLabels&&ends.length){ends.sort((a,b)=>a.actualY-b.actualY);let previous=m.t-18;for(const e of ends){e.labelY=Math.max(e.actualY,previous+17);previous=e.labelY}const overflow=ends.length?Math.max(0,ends[ends.length-1].labelY-(H-m.b)):0;for(const e of ends)e.labelY-=overflow;for(const e of ends){const color=e.item.className?.includes('old')?'var(--old)':'var(--new)',x0=X(e.p[0]),x1=W-m.r+8;svg.appendChild(s('line',{x1:x0,y1:e.actualY,x2:x1-3,y2:e.labelY,class:'leader',stroke:color}));const t=s('text',{x:x1,y:e.labelY+4,class:'endpoint',fill:color});t.textContent=`${e.item.label??''} ${fmt(e.p[1],opt)}`.trim();svg.appendChild(t)}}}
function val(which,p){return D[`${which}_validation`][p].map(r=>[r.iteration,r.bpb])}function gap(rows){let best=Infinity;return rows.map(([x,y])=>{best=Math.min(best,y);return[x,y-best]})}
const maxTrain=Math.max(D.meta.snapshot_iteration,...D.old_training.map(r=>r.iteration));plot(document.getElementById('train-loss'),[{values:D.old_training.map(r=>[r.iteration,r.loss]),className:'old raw'},{values:moving(D.old_training.map(r=>[r.iteration,r.loss])),className:'old',label:'previous'},{values:D.new_training.map(r=>[r.iteration,r.loss]),className:'new raw'},{values:moving(D.new_training.map(r=>[r.iteration,r.loss])),className:'new',label:'sanitized'}],{xmin:0,xmax:maxTrain,tokens:true,endLabels:true,label:'Raw and 101-update moving-average training loss'});
function lr(total,cool){const v=[];for(let u=0;u<=total;u+=40){let y;if(u<=D.constants.warmup_end)y=D.constants.minimum_lr+(D.constants.peak_lr-D.constants.minimum_lr)*u/D.constants.warmup_end;else if(u<=cool)y=D.constants.peak_lr;else{const q=(u-cool)/(total-cool);y=D.constants.minimum_lr+(D.constants.peak_lr-D.constants.minimum_lr)*(1-Math.sqrt(Math.min(1,q)))}v.push([u,y])}v.push([total,D.constants.minimum_lr]);return v}plot(document.getElementById('lr'),[{values:lr(D.constants.old_updates,D.constants.old_cooldown_start),className:'old ideal'},{values:lr(D.constants.new_updates,D.constants.new_cooldown_start),className:'new'}],{xmin:0,xmax:D.constants.old_updates,tokens:true,yDigits:6,markers:[{x:D.constants.warmup_end,label:'warmup ends'},{x:D.constants.new_cooldown_start,label:'new cooldown'},{x:D.constants.old_cooldown_start,label:'old cooldown'},{x:D.constants.new_updates,label:'new floor'}],label:'Idealized old and new WSD learning-rate schedules'});
for(const el of document.querySelectorAll('[data-val]')){const p=el.dataset.val;plot(el,[{values:val('old',p),className:'old',points:true},{values:val('new',p),className:'new',points:true}],{xmin:0,xmax:maxTrain,tokens:true,label:`${D.labels[p]} BPB comparison`})}
for(const el of document.querySelectorAll('[data-gap]')){const p=el.dataset.gap;plot(el,[{values:gap(val('old',p)),className:'old'},{values:gap(val('new',p)),className:'new'}],{xmin:0,xmax:maxTrain,tokens:true,ymin:0,yDigits:4,label:`${D.labels[p]} forgetting from best observed BPB`})}
for(const [id,key,percent] of [['gm-acc','clean_accuracy',true],['gm-nll','clean_choice_nll',false],['gm-bpb','clean_correct_answer_bpb',false]])plot(document.getElementById(id),[{values:D.old_greekmmlu.map(r=>[r.iteration,r[key]]),className:'old',points:true,label:'previous'},{values:D.new_greekmmlu.map(r=>[r.iteration,r[key]]),className:'new',points:true,label:'sanitized'}],{width:1200,height:440,xmin:0,xmax:D.constants.new_updates,tokens:true,percent,yDigits:percent?2:4,endLabels:true,markers:[{x:9536,label:'best rerun'},{x:D.constants.new_cooldown_start,label:'cooldown'}],label:`GreekMMLU ${key}`});
for(const [id,key,percent] of [['gm-acc-zoom','clean_accuracy',true],['gm-nll-zoom','clean_choice_nll',false]])plot(document.getElementById(id),[{values:D.new_greekmmlu.map(r=>[r.iteration,r[key]]),className:'new',points:true,label:'sanitized'}],{width:1200,height:390,xmin:5960,xmax:D.constants.new_updates,tokens:true,percent,yDigits:percent?2:4,endLabels:true,markers:[{x:9536,label:'best rerun'},{x:D.constants.new_cooldown_start,label:'cooldown'}],label:`GreekMMLU plateau zoom ${key}`});
function deltaBars(el){const rows=[];for(const p of [...D.learning_panels,...D.retention_panels]){const v=D.per_document_validation[p];rows.push({p,value:v[v.length-1].bpb-v[v.length-2].bpb})}const W=1100,H=530,m={l:210,r:90,t:28,b:36},svg=s('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'Document-local BPB change during cooldown'});el.appendChild(svg);const min=Math.min(...rows.map(r=>r.value))*1.12,max=0,X=x=>m.l+(x-min)/(max-min)*(W-m.l-m.r),step=(H-m.t-m.b)/rows.length;svg.appendChild(s('line',{x1:X(0),y1:m.t,x2:X(0),y2:H-m.b,class:'axis'}));rows.forEach((r,i)=>{const y=m.t+i*step+4,h=Math.max(10,step-9),x=X(r.value);svg.appendChild(s('rect',{x,y,width:X(0)-x,height:h,fill:'var(--new)',opacity:.78}));const lab=s('text',{x:m.l-12,y:y+h*.72,'text-anchor':'end',class:'bar-label'});lab.textContent=D.labels[r.p];svg.appendChild(lab);const val=s('text',{x:x-6,y:y+h*.72,'text-anchor':'end',class:'bar-value'});val.textContent=r.value.toFixed(4);svg.appendChild(val)});const zero=s('text',{x:X(0),y:H-10,'text-anchor':'middle',class:'tick'});zero.textContent='0 BPB';svg.appendChild(zero)}
if(D.per_document_validation)deltaBars(document.getElementById('cooldown-delta'));
"""


def validation_grid(payload: dict[str, Any], panels: tuple[str, ...], attribute: str) -> str:
    blocks = []
    for panel in panels:
        same = payload["comparable_panels"][panel]
        blocks.append(
            f"<article class='mini{' changed' if not same else ''}'><h3>{LABELS[panel]}"
            f"<span>{'same panel' if same else 'panel changed'}</span></h3><div {attribute}='{panel}'></div></article>"
        )
    return "".join(blocks)


def data_table(payload: dict[str, Any]) -> str:
    old = payload["data"]["old_geometry"]
    new = payload["data"]["new_geometry"]
    labels = (("modern_greek", "Modern Greek"), ("foreign_replay", "Foreign replay"), ("old_greek_replay", "Old-Greek replay"), ("active_tokens", "Total active"))
    return "".join(
        f"<tr><td>{label}</td><td>{old[key]/1e9:.3f}B</td><td>{new[key]/1e9:.3f}B</td><td>{(new[key]-old[key])/1e9:+.3f}B</td></tr>"
        for key, label in labels
    )


def benchmark_table(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload["benchmark_comparison"]:
        rows.append(
            f"<tr><td>{row['iteration']:,}</td><td>{row['old_accuracy']:.4f}</td><td>{row['new_accuracy']:.4f}</td>"
            f"<td class='{'better' if row['delta_accuracy'] > 0 else 'worse'}'>{row['delta_accuracy']:+.4f}</td>"
            f"<td>{row['old_nll']:.4f}</td><td>{row['new_nll']:.4f}</td>"
            f"<td class='{'better' if row['delta_nll'] < 0 else 'worse'}'>{row['delta_nll']:+.4f}</td></tr>"
        )
    return "".join(rows)


def validation_table(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload["validation_comparison"]:
        comparable = row["comparable"]
        cls = "better" if comparable and row["delta"] < 0 else "worse" if comparable else "changed-text"
        rows.append(
            f"<tr><td>{LABELS[row['panel']]}</td><td>{'yes' if comparable else 'no'}</td>"
            f"<td>{row['old_bpb']:.4f}</td><td>{row['new_bpb']:.4f}</td><td class='{cls}'>{row['delta']:+.4f}</td></tr>"
        )
    return "".join(rows)


def current_endpoint_table(payload: dict[str, Any]) -> str:
    rows = []
    for panel in (*LEARNING, *RETENTION):
        values = payload["new_validation"][panel]
        endpoint = values[-1]
        best = min(float(row["bpb"]) for row in values)
        forgetting = float(endpoint["bpb"]) - best
        rows.append(
            f"<tr><td>{LABELS[panel]}</td><td>{int(endpoint['iteration']):,}</td>"
            f"<td>{float(endpoint['bpb']):.4f}</td><td>{best:.4f}</td>"
            f"<td class='{'worse' if forgetting > 0 else 'better'}'>{forgetting:+.4f}</td></tr>"
        )
    return "".join(rows)


def per_document_table(payload: dict[str, Any]) -> str:
    rows = []
    assert payload["per_document_validation"] is not None
    for panel in (*LEARNING, *RETENTION):
        initial, cooldown, final = payload["per_document_validation"][panel]
        total_delta = float(final["bpb"]) - float(initial["bpb"])
        cooldown_delta = float(final["bpb"]) - float(cooldown["bpb"])
        total_fraction = total_delta / float(initial["bpb"])
        rows.append(
            f"<tr><td>{LABELS[panel]}</td><td>{int(final['documents']):,}</td>"
            f"<td>{float(initial['bpb']):.4f}</td><td>{float(cooldown['bpb']):.4f}</td>"
            f"<td>{float(final['bpb']):.4f}</td>"
            f"<td class='{'better' if cooldown_delta < 0 else 'worse'}'>{cooldown_delta:+.4f}</td>"
            f"<td class='{'better' if total_delta < 0 else 'worse'}'>{100*total_fraction:+.1f}%</td></tr>"
        )
    return "".join(rows)


def greekmmlu_trajectory_table(payload: dict[str, Any]) -> str:
    rows = []
    best_accuracy = max(float(row["clean_accuracy"]) for row in payload["new_greekmmlu"])
    best_nll = min(float(row["clean_choice_nll"]) for row in payload["new_greekmmlu"])
    for row in payload["new_greekmmlu"]:
        accuracy = float(row["clean_accuracy"])
        nll = float(row["clean_choice_nll"])
        rows.append(
            f"<tr><td>{int(row['iteration']):,}</td>"
            f"<td>{int(row['iteration'])*TOKENS_PER_UPDATE/1e9:.2f}B</td>"
            f"<td class='{'better' if accuracy == best_accuracy else ''}'>{100*accuracy:.2f}%</td>"
            f"<td class='{'better' if nll == best_nll else ''}'>{nll:.4f}</td>"
            f"<td>{float(row['clean_correct_answer_bpb']):.4f}</td></tr>"
        )
    return "".join(rows)


def build_html(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    progress = 100 * meta["snapshot_fraction"]
    if meta["snapshot_iteration"] >= payload["constants"]["new_updates"]:
        campaign_status = "has completed the planned optimizer trajectory"
        allocation_status = "The final 16-node segment has completed."
    elif meta["snapshot_iteration"] > payload["constants"]["new_cooldown_start"]:
        campaign_status = "is training through the cooldown in its final allocation"
        allocation_status = (
            "Segment 4, updates 14,628–18,284, is running on 16 normal nodes."
        )
    else:
        campaign_status = "has reached the cooldown boundary and is waiting for its final allocation"
        allocation_status = (
            "Segment 4, updates 14,628–18,284, is queued on 16 normal nodes."
        )
    dropped = payload["data"]["dedup_counts"]["dropped_documents"]
    latest_benchmark_comparison = payload["benchmark_comparison"][-1]
    latest_benchmark_sentence = (
        f"By update {latest_benchmark_comparison['iteration']:,}, the gap narrows to "
        f"{100 * abs(latest_benchmark_comparison['delta_accuracy']):.2f} accuracy points, "
        f"while the rerun NLL remains {latest_benchmark_comparison['delta_nll']:+.4f} higher."
    )
    next_greekmmlu = meta["greekmmlu_next"]
    if next_greekmmlu is None:
        benchmark_progress_sentence = "All frozen GreekMMLU milestones are complete."
    else:
        benchmark_progress_sentence = (
            f"Update {next_greekmmlu:,} is {meta['greekmmlu_next_status']}."
        )
    latest_greekmmlu = payload["new_greekmmlu"][-1]
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{meta['title']}</title><style>{STYLE}</style></head><body><main>
<header class='hero'><div class='eyebrow'>Live evidence review · 10 August 2026</div><h1>Sanitized 8B CPT:<br>learning replicates better than the benchmark</h1><p class='lede'>At {meta['snapshot_tokens_b']:.3f}B consumed token slots, source-conditioned validation is close to the stopped run on unchanged panels. GreekMMLU remains non-monotonic, while the rerun {campaign_status}.</p>
<div class='hero-meta'><div class='metric'><b>{meta['snapshot_iteration']:,}</b><span>updates complete · {progress:.1f}%</span></div><div class='metric'><b>{meta['snapshot_tokens_b']:.3f}B</b><span>consumed token slots</span></div><div class='metric'><b>{meta['median_step_seconds']:.2f}s</b><span>recent median / update</span></div><div class='metric'><b>0 / 0</b><span>skipped / non-finite</span></div><div class='metric'><b>{meta['greekmmlu_latest_complete']:,}</b><span>latest complete GreekMMLU</span></div></div>
<div class='finding'><strong>Interim finding.</strong> The rerun is stable and its Greek validation losses are not degraded in a broad way. However, at updates 2,384–4,768 it is 1.2–3.1 accuracy points below the earlier run on the decontaminated GreekMMLU subset, with worse continuous NLL as well. {latest_benchmark_sentence} The benchmark difference cannot yet be assigned uniquely to anonymization: the sanitized rebuild also changed the sampled sequence order and shortened horizon-scaled AdEMAMix schedules.</div></header>

<section class='section'><div class='eyebrow'>01 · Experimental geometry</div><h2>Same model; different corpus realization</h2><p class='section-intro'>Both runs load the same untied Token-Distilled checkpoint and 148,992-token Modern+Polytonic tokenizer, with 32 layers, width 4,096, 32 attention heads / 8 query groups, sequence length 4,096, RoPE base 500,000 with factor 8, BF16, TP=2, DP=32, global batch 1,024, seed 20260609, and peak LR 5.5×10⁻⁵. The changed fields are data-bound or operational.</p>
<div class='chart-shell'><div class='chart-title'><h3>Complete WSD schedules</h3><small>idealized from logged settings; old gray, sanitized teal</small></div><div class='legend'><span><i class='swatch old'></i>previous 19,248-update run</span><span><i class='swatch new'></i>sanitized 18,284-update run</span></div><div id='lr'></div></div>
<div class='notes'><div class='note'><h3>Architecture held fixed</h3><p>Tokenizer, initialization path, RoPE geometry, model dimensions, precision, batch geometry, optimizer family, peak/floor LR, warmup, clipping, and random seed match in the trainer argument dumps.</p></div><div class='note caution'><h3>Horizon schedules rescaled</h3><p>AdEMAMix α and β₃ warmups are 19,248 updates in the old run and 18,284 in the rerun. WSD cooldown starts at 15,398 versus 14,627. Before the cooldown the LR is identical, but AdEMAMix’s horizon-scaled state is not exactly identical at matched updates.</p></div></div>
<div class='table-wrap'><table><thead><tr><th>pool</th><th>previous</th><th>sanitized</th><th>change</th></tr></thead><tbody>{data_table(payload)}</tbody></table></div>
<p class='section-intro'>PII masking changed {payload['data']['masked_documents']:,} documents. Global post-mask processing dropped {dropped:,}: {payload['data']['dedup_counts']['duplicate_documents_dropped']:,} exact duplicates and {payload['data']['dedup_counts']['validation_collision_documents_dropped']:,} validation collisions. The active stream is {abs(meta['active_token_change'])/1e9:.3f}B tokens smaller ({100*meta['active_token_change_fraction']:.2f}%). Non-HPLT rises slightly within Modern Greek, from {100*float(payload['data']['old_modern']['glossapi_non_hplt_fraction']):.3f}% to {100*float(payload['data']['new_modern']['glossapi_non_hplt_fraction']):.3f}%.</p></section>

<section class='section'><div class='eyebrow'>02 · Optimization</div><h2>Training is numerically stable</h2><p class='section-intro'>Thin lines are every logged optimizer update; heavier lines are 101-update moving means shown only to make the trend legible. Raw points remain embedded and rendered. Training loss is batch-dependent and therefore diagnostic, not a fair winner metric across the rebuilt sequence schedule.</p><div class='chart-shell'><div class='chart-title'><h3>Training loss</h3><small>complete observed horizon</small></div><div class='legend'><span><i class='swatch old'></i>previous</span><span><i class='swatch new'></i>sanitized rerun</span></div><div id='train-loss'></div></div></section>

<section class='section'><div class='eyebrow'>03 · Greek learning</div><h2>Absolute BPB keeps improving</h2><p class='section-intro'>Lower is better. Nine panels have identical target-count/byte signatures and support direct cross-run comparison. HPLT and non-HPLT were rebuilt to remove exact training overlap, so their absolute values remain valid within each run but their vertical separation is not a clean treatment effect.</p><div class='legend'><span><i class='swatch old'></i>previous unsanitized training</span><span><i class='swatch new'></i>sanitized rerun</span><span class='changed-text'>gold rule = validation panel changed</span></div><div class='grid'>{validation_grid(payload, LEARNING, 'data-val')}</div></section>

<section class='section'><div class='eyebrow'>04 · Retention and forgetting</div><h2>Foreign BPB shows a small upward drift</h2><p class='section-intro'>The first grid shows absolute BPB. The second shows each run’s rise from its own best observed BPB. English and Old Greek changed heldout content and must not be compared vertically across runs; Code, Math, German, Russian, and Chinese are directly comparable.</p><div class='grid'>{validation_grid(payload, RETENTION, 'data-val')}</div><h3 style='margin:28px 0 10px'>Forgetting from best observed BPB</h3><div class='grid'>{validation_grid(payload, RETENTION, 'data-gap')}</div><div class='table-wrap' style='margin-top:18px'><table><thead><tr><th>rerun panel</th><th>latest update</th><th>latest BPB</th><th>best BPB</th><th>rise from best</th></tr></thead><tbody>{current_endpoint_table(payload)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>05 · Matched validation checkpoint</div><h2>At update {meta['comparison_iteration']:,}, differences are small on unchanged panels</h2><p class='section-intro'>Negative Δ favors the sanitized rerun. Changed-panel rows are printed for within-run context but colored gold and excluded from causal comparison.</p><div class='table-wrap'><table><thead><tr><th>panel</th><th>same heldout?</th><th>previous BPB</th><th>sanitized BPB</th><th>Δ</th></tr></thead><tbody>{validation_table(payload)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>06 · Native Greek benchmark</div><h2>GreekMMLU is not following validation loss monotonically</h2><p class='section-intro'>All three plots use the decontaminated 16,159-question subset. Accuracy is higher-is-better; choice NLL and correct-answer BPB are lower-is-better. The rerun is complete through update {meta['greekmmlu_latest_complete']:,}. {benchmark_progress_sentence}</p><div class='finding'><strong>Latest authoritative score · update {meta['greekmmlu_latest_complete']:,}.</strong> Decontaminated accuracy {100*latest_greekmmlu['clean_accuracy']:.2f}%, choice NLL {latest_greekmmlu['clean_choice_nll']:.4f}, and correct-answer BPB {latest_greekmmlu['clean_correct_answer_bpb']:.4f} on {latest_greekmmlu['clean_n']:,} questions.</div><div class='grid three'><article class='mini'><h3>Accuracy <span>higher is better</span></h3><div id='gm-acc'></div></article><article class='mini'><h3>Choice NLL <span>lower is better</span></h3><div id='gm-nll'></div></article><article class='mini'><h3>Correct-answer BPB <span>lower is better</span></h3><div id='gm-bpb'></div></article></div>
<div class='table-wrap' style='margin-top:18px'><table><thead><tr><th>update</th><th>old acc.</th><th>new acc.</th><th>Δ acc.</th><th>old NLL</th><th>new NLL</th><th>Δ NLL</th></tr></thead><tbody>{benchmark_table(payload)}</tbody></table></div></section>

<section class='section'><div class='eyebrow'>07 · Interpretation</div><h2>What can—and cannot—explain the difference</h2><div class='notes'><div class='note'><h3>Strongest explanation: changed sample trajectory</h3><p>Masking and post-mask deduplication change token sequences, document identities, the canonical sequence inventory hash, and therefore the randomized batch order. One seed cannot separate content effects from order effects. This is the largest confirmed difference.</p></div><div class='note'><h3>Secondary mechanism: horizon rescaling</h3><p>The shorter corpus advances AdEMAMix α/β₃ schedules about 5.3% faster at a matched update. The LR itself is identical over the currently observed stable phase. This could contribute, but the present evidence does not identify its effect size.</p></div><div class='note'><h3>Not an architecture explanation</h3><p>Trainer dumps agree on the model, tokenizer, TD initialization path, untied embeddings, RoPE 500k/4096/factor-8 geometry, precision, parallelism, and global batch. There is no observed architecture drift to explain GreekMMLU.</p></div><div class='note caution'><h3>Benchmark variance remains</h3><p>GreekMMLU accuracy is non-monotonic in both runs, and there is only one source-order seed. Continuous NLL confirms the current gap, so it is not merely an argmax artifact, but a final scientific conclusion should wait for later matched checkpoints and the completed cooldown.</p></div></div></section>

<section class='section'><div class='eyebrow'>08 · Evidence boundary</div><h2>Live report, not an endpoint decision</h2><p class='section-intro'>The rerun is {progress:.1f}% through optimizer updates; source validation is complete through update {meta['last_validation_iteration']:,}; GreekMMLU is complete through {meta['greekmmlu_latest_complete']:,} ({meta['greekmmlu_complete_milestones']}/{meta['greekmmlu_required_milestones']} frozen milestones including update 0). No checkpoint averaging is used. {allocation_status}</p><div class='note'><h3>Exact evidence</h3><p>Current run: <code>/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12</code><br>Previous run: <code>/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260805T154100Z-d0-wsd10-v3</code><br>Sanitized stage: <code>/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260808T064500Z-d0-v4-v45bridge</code></p></div></section>
<footer class='footer'>Self-contained report. Charts are generated from embedded, receipt-bound raw points; no network dependencies or smoothing-only views. Full bindings and SHA-256 values are present in the companion JSON.</footer>
<script id='report-data' type='application/json'>{json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')}</script><script>{SCRIPT}</script></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    parser.add_argument("--greekmmlu-running", type=int)
    parser.add_argument("--final-evidence-root", type=Path)
    args = parser.parse_args()
    payload = build_payload(
        args.repo_root.resolve(),
        args.evidence_root.resolve(),
        running_greekmmlu=args.greekmmlu_running,
        final_evidence_root=args.final_evidence_root.resolve() if args.final_evidence_root else None,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(payload), encoding="utf-8")
    data_output = (args.data_output or output.with_suffix(".data.json")).resolve()
    data_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(output), "data": str(data_output), "snapshot_iteration": payload["meta"]["snapshot_iteration"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
