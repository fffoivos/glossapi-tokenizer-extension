#!/usr/bin/env python3
"""Build the auditable analysis payload for the full-panel/stable-LR report."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
SNAPSHOTS = EVIDENCE / "remote_snapshots"
PRIOR_REPORT = ROOT.parent / "greekmmlu_trajectory_20260822" / "evidence"
UPDATES = [238, 476, 714, 952, 1190, 1428, 1666, 1904, 2142, 2261,
           2380, 2618, 2856, 3094, 3218, 3456, 3694]
PAIRED = [2618, 2856, 3094, 3218]
PANELS = ["hplt", "openarchives", "greek_phd", "english", "de", "ru", "zh", "code", "old_greek"]
VALIDATION = re.compile(
    r"validation loss at iteration\s+(\d+)\s+\[([^\]]+)\].*?lm loss value:\s*([-+0-9.Ee]+)"
)
OPTIMIZER = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?learning rate:\s*([-+0-9.Ee]+).*?"
    r"number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_summary_grid(arm: str, required: list[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in sorted((SNAPSHOTS / arm).glob("iter_*/full_public/aggregate/summary.json")):
        summary = load(path)
        update = int(summary["iteration"])
        metrics = summary["views"]["full_public"]["metrics"]
        overall = metrics["overall"]
        rows[update] = {
            "update": update,
            "accuracy": float(overall["accuracy"]),
            "correct": int(overall["correct"]),
            "n": int(overall["n"]),
            "choice_nll": float(overall["choice_nll"]),
            "correct_answer_bpb": float(overall["correct_answer_bpb"]),
            "by_subject": metrics["by_subject"],
            "by_educational_level": metrics["by_educational_level"],
            "summary": bind(path),
            "receipt": bind(path.with_name("receipt.json")),
        }
    if sorted(rows) != sorted(required):
        raise ValueError(f"{arm} checkpoint grid drift: got={sorted(rows)} expected={required}")
    return rows


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    validations: dict[str, dict[int, float]] = {panel: {} for panel in PANELS}
    for update, panel, value in VALIDATION.findall(text):
        if panel in validations:
            validations[panel][int(update)] = float(value)
    optimizer = {}
    skipped = nonfinite = 0
    for update, lr, skipped_raw, nan_raw in OPTIMIZER.findall(text):
        optimizer[int(update)] = float(lr)
        skipped = max(skipped, int(skipped_raw))
        nonfinite = max(nonfinite, int(nan_raw))
    if not optimizer:
        raise ValueError(f"no optimizer rows parsed: {path}")
    if any(not values for values in validations.values()):
        raise ValueError(f"validation panel coverage missing: {path}")
    return {
        "optimizer": optimizer,
        "validation": validations,
        "skipped": skipped,
        "nonfinite": nonfinite,
        "source": bind(path),
    }


def finite_tree(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value in analysis")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    if isinstance(value, list):
        for child in value:
            finite_tree(child)


def main() -> None:
    decayed = load_summary_grid("decayed", UPDATES)
    stable = load_summary_grid("stable", PAIRED)
    legacy_path = EVIDENCE / "legacy_bf16_matrix_receipt.json"
    legacy = load(legacy_path)
    if legacy.get("status") != "completed":
        raise ValueError("legacy matrix is incomplete")

    decayed_log = parse_log(EVIDENCE / "decayed_branch_train.log")
    stable_log = parse_log(EVIDENCE / "stable_branch_train.log")
    if stable_log["skipped"] or stable_log["nonfinite"]:
        raise ValueError("stable branch contains skipped or non-finite updates")
    stable_lrs = [stable_log["optimizer"].get(update) for update in range(2500, 3219)]
    if any(value is None or abs(value - 5.5e-5) > 1e-12 for value in stable_lrs):
        raise ValueError("stable branch LR drift")

    paired_rows = []
    for update in PAIRED:
        s, d = stable[update], decayed[update]
        paired_rows.append({
            "update": update,
            "stable": {key: s[key] for key in ("accuracy", "correct", "n", "choice_nll", "correct_answer_bpb")},
            "decayed": {key: d[key] for key in ("accuracy", "correct", "n", "choice_nll", "correct_answer_bpb")},
            "stable_minus_decayed_accuracy_pp": 100 * (s["accuracy"] - d["accuracy"]),
        })

    prior = load(PRIOR_REPORT / "training_trajectories.json")
    full_validation = prior["scales"]["8b"]["validation_loss"]
    full_lr = prior["scales"]["8b"]["learning_rate"]
    branch_validation = {}
    for panel in PANELS:
        stable_series = sorted((u, v) for u, v in stable_log["validation"][panel].items() if 2500 <= u <= 3218)
        decayed_series = sorted((u, v) for u, v in decayed_log["validation"][panel].items() if 2500 <= u <= 3218)
        common = sorted(set(dict(stable_series)) & set(dict(decayed_series)))
        if len(common) < 20:
            raise ValueError(f"insufficient paired validation coverage for {panel}: {len(common)}")
        branch_validation[panel] = {
            "stable": [[u, dict(stable_series)[u]] for u in common],
            "decayed": [[u, dict(decayed_series)[u]] for u in common],
            "endpoint_stable_minus_decayed": dict(stable_series)[common[-1]] - dict(decayed_series)[common[-1]],
        }

    stable_accuracy = [stable[u]["accuracy"] for u in PAIRED]
    interval_pp = [100 * (b - a) for a, b in zip(stable_accuracy, stable_accuracy[1:])]
    best_update = max(PAIRED, key=lambda update: stable[update]["accuracy"])
    output = {
        "schema_version": "apertus_h2g_full_panel_stable_lr_analysis_v1",
        "status": "completed",
        "panel": {"name": "full_public", "n": 16632, "dtype": "float32"},
        "tokens_per_update": 4_194_304,
        "phase_boundaries": {"hplt_end": 2261, "openarchives_end": 3218, "branch": 2499},
        "decayed_full_panel": [decayed[u] for u in UPDATES],
        "stable_full_panel": [stable[u] for u in PAIRED],
        "paired": paired_rows,
        "stable_trajectory": {
            "best_update": best_update,
            "best_accuracy": stable[best_update]["accuracy"],
            "interval_accuracy_change_pp": interval_pp,
            "endpoint_minus_first_pp": 100 * (stable_accuracy[-1] - stable_accuracy[0]),
            "noise_reference_pp": 0.4,
        },
        "legacy_replication": legacy,
        "learning_rate": {
            "decayed_full": full_lr,
            "stable_branch": [[2499, 5.5e-5], [3218, 5.5e-5]],
        },
        "validation": {"decayed_full": full_validation, "paired_branch": branch_validation},
        "optimizer_integrity": {
            "stable_skipped": stable_log["skipped"],
            "stable_nonfinite": stable_log["nonfinite"],
            "stable_first_update": min(stable_log["optimizer"]),
            "stable_last_update": max(stable_log["optimizer"]),
        },
        "sources": {
            "legacy_bf16_matrix": bind(legacy_path),
            "decayed_log": decayed_log["source"],
            "stable_log": stable_log["source"],
            "prior_training_trajectories": {
                "path": str((PRIOR_REPORT / "training_trajectories.json").relative_to(ROOT.parent.parent.parent.parent)),
                "bytes": (PRIOR_REPORT / "training_trajectories.json").stat().st_size,
                "sha256": sha256(PRIOR_REPORT / "training_trajectories.json"),
            },
        },
    }
    finite_tree(output)
    path = EVIDENCE / "analysis.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
