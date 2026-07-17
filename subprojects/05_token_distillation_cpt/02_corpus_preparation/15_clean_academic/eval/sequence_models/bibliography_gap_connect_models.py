#!/usr/bin/env python3
"""Grouped OOF gap-only connect baselines and an ordered residual TCN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import PINNED_SKLEARN_VERSION
from .bibliography_gap_connect_table import SCHEMA_VERSION as TABLE_SCHEMA_VERSION
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-connect-oof-v1"
ARMS = ("pooled_hist", "ordered_tcn", "shuffled_tcn")


def _sklearn() -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    return {
        "hist": HistGradientBoostingClassifier,
        "average_precision": average_precision_score,
        "roc_auc": roc_auc_score,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


class GapTable:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != TABLE_SCHEMA_VERSION:
            raise ValueError("unsupported gap table schema")
        if self.manifest.get("validation_opened") is not False:
            raise ValueError("gap experiment requires validation-isolated inputs")
        self.features = np.load(self.root / "features.npy", mmap_mode="r", allow_pickle=False)
        self.line_targets = np.load(self.root / "line_targets.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.root / "gap_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(self.root / "targets.npy", mmap_mode="r", allow_pickle=False)
        self.folds = np.load(self.root / "folds.npy", mmap_mode="r", allow_pickle=False)
        self.lengths = np.load(self.root / "gap_lengths.npy", mmap_mode="r", allow_pickle=False)
        self.metadata = tuple(
            json.loads(line) for line in (self.root / "gaps.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        n = len(self.targets)
        if not (
            len(self.folds) == len(self.lengths) == len(self.metadata) == n
            and len(self.offsets) == n + 1
            and int(self.offsets[-1]) == len(self.features) == len(self.line_targets)
            and np.all(np.diff(self.offsets.astype(np.int64)) == self.lengths)
        ):
            raise ValueError("gap table arrays are not aligned")

    def sequence(self, index: int, *, shuffled: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        features = np.asarray(self.features[start:end], dtype=np.float32)
        targets = np.asarray(self.line_targets[start:end], dtype=np.float32)
        if shuffled and len(features) > 1:
            identity = f"{seed}:{self.metadata[index]['document_id']}:{self.metadata[index]['left_local_index']}"
            local_seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little")
            order = np.random.default_rng(local_seed).permutation(len(features))
            features, targets = features[order], targets[order]
        return features, targets


def pooled_features(table: GapTable) -> np.ndarray:
    """Order-free comparator over exactly the same gap-only line vectors."""

    rows = []
    for index in range(len(table.targets)):
        values, _ = table.sequence(index, shuffled=False, seed=0)
        rows.append(np.concatenate((
            values.mean(axis=0), values.std(axis=0),
            values.min(axis=0), values.max(axis=0),
            np.asarray((math.log1p(len(values)),), dtype=np.float32),
        )))
    result = np.stack(rows).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("pooled gap features are non-finite")
    return result


def best_safety_threshold(
    targets: np.ndarray, probability: np.ndarray, *, max_false_connect_rate: float,
) -> dict[str, float]:
    if len(targets) != len(probability) or {int(value) for value in np.unique(targets)} != {0, 1}:
        raise ValueError("safety threshold selection requires both aligned classes")
    candidates = np.unique(np.concatenate((
        np.asarray((0.0, 1.0), dtype=np.float64),
        probability.astype(np.float64),
        np.nextafter(probability.astype(np.float64), np.inf),
    )))
    best: dict[str, float] | None = None
    negatives = targets == 0
    for threshold in candidates:
        prediction = probability >= threshold
        false_connect = int(np.count_nonzero(prediction & negatives))
        false_connect_rate = false_connect / max(1, int(np.count_nonzero(negatives)))
        true_connect = int(np.count_nonzero(prediction & (targets == 1)))
        connect_recall = true_connect / max(1, int(np.count_nonzero(targets == 1)))
        row = {
            "threshold": float(threshold),
            "false_connect_count": false_connect,
            "false_connect_rate": false_connect_rate,
            "connect_recall": connect_recall,
        }
        if false_connect_rate <= max_false_connect_rate and (
            best is None
            or (row["connect_recall"], -row["threshold"]) > (best["connect_recall"], -best["threshold"])
        ):
            best = row
    if best is None:
        raise RuntimeError("no safety threshold candidate satisfies the requested rate")
    return best


def binary_metrics(
    targets: np.ndarray, probability: np.ndarray, thresholds: np.ndarray,
) -> dict[str, Any]:
    tools = _sklearn()
    prediction = probability >= thresholds
    positive, negative = targets == 1, targets == 0
    tp = int(np.count_nonzero(prediction & positive))
    fp = int(np.count_nonzero(prediction & negative))
    fn = int(np.count_nonzero(~prediction & positive))
    tn = int(np.count_nonzero(~prediction & negative))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "connect_precision": precision,
        "connect_recall": recall,
        "connect_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_connect_count": fp,
        "false_connect_rate": fp / max(1, fp + tn),
        "false_split_count": fn,
        "true_connect_count": tp,
        "true_break_count": tn,
        "connect_pr_auc": float(tools["average_precision"](targets, probability)),
        "break_pr_auc": float(tools["average_precision"](negative, 1.0 - probability)),
        "roc_auc": float(tools["roc_auc"](targets, probability)),
        "brier": float(np.mean((probability - targets.astype(float)) ** 2)),
    }


def _balanced_weights(targets: np.ndarray) -> np.ndarray:
    positive = max(1, int(np.count_nonzero(targets == 1)))
    negative = max(1, int(np.count_nonzero(targets == 0)))
    return np.where(
        targets == 1, len(targets) / (2 * positive), len(targets) / (2 * negative)
    ).astype(np.float64)


def _fit_hist(features: np.ndarray, targets: np.ndarray, indices: np.ndarray, *, seed: int) -> Any:
    return _sklearn()["hist"](
        learning_rate=0.05, max_iter=180, max_depth=3,
        min_samples_leaf=10, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    ).fit(features[indices], targets[indices], sample_weight=_balanced_weights(targets[indices]))


def _torch() -> Any:
    import torch

    return torch


def _make_tcn(input_dim: int, *, hidden: int, dropout: float) -> Any:
    torch = _torch()
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
            self.norm1 = nn.GroupNorm(1, hidden)
            self.norm2 = nn.GroupNorm(1, hidden)
            self.dropout = nn.Dropout(dropout)

        def forward(self, values: Any, mask: Any) -> Any:
            residual = values
            values = self.dropout(torch.nn.functional.gelu(self.norm1(self.conv1(values))))
            values = self.dropout(torch.nn.functional.gelu(self.norm2(self.conv2(values))))
            return (residual + values) * mask[:, None, :]

    class GapTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(input_dim, hidden)
            self.blocks = nn.ModuleList(ResidualBlock(dilation) for dilation in (1, 2, 4, 8, 16, 32))
            self.attention = nn.Conv1d(hidden, 1, 1)
            self.line_head = nn.Conv1d(hidden, 1, 1)
            self.gap_head = nn.Sequential(
                nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
            )

        def forward(self, values: Any, mask: Any) -> tuple[Any, Any]:
            encoded = self.projection(values).transpose(1, 2) * mask[:, None, :]
            for block in self.blocks:
                encoded = block(encoded, mask)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (encoded * mask[:, None, :]).sum(dim=2) / count
            maximum = encoded.masked_fill(~mask[:, None, :].bool(), -1.0e4).max(dim=2).values
            score = self.attention(encoded).squeeze(1).masked_fill(~mask.bool(), -1.0e4)
            attention = torch.softmax(score, dim=1)
            attended = (encoded * attention[:, None, :]).sum(dim=2)
            gap = self.gap_head(torch.cat((mean, maximum, attended), dim=1)).squeeze(1)
            line = self.line_head(encoded).squeeze(1)
            return gap, line

    return GapTCN()


def _line_scaler(table: GapTable, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts = [
        np.asarray(table.features[int(table.offsets[index]) : int(table.offsets[index + 1])], dtype=np.float64)
        for index in indices
    ]
    values = np.concatenate(parts)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _batch(
    table: GapTable, indices: Sequence[int], *, mean: np.ndarray, scale: np.ndarray,
    shuffled: bool, seed: int,
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    sequences, line_targets = zip(*(
        table.sequence(int(index), shuffled=shuffled, seed=seed) for index in indices
    ))
    maximum = max(len(values) for values in sequences)
    values = np.zeros((len(indices), maximum, table.features.shape[1]), dtype=np.float32)
    mask = np.zeros((len(indices), maximum), dtype=np.float32)
    lines = np.zeros((len(indices), maximum), dtype=np.float32)
    for row, (sequence, target) in enumerate(zip(sequences, line_targets, strict=True)):
        length = len(sequence)
        values[row, :length] = np.clip((sequence - mean) / scale, -10.0, 10.0)
        mask[row, :length] = 1.0
        lines[row, :length] = target
    return (
        torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(lines),
        torch.from_numpy(table.targets[np.asarray(indices)].astype(np.float32)),
    )


def _predict_tcn(
    model: Any, table: GapTable, indices: np.ndarray, *, mean: np.ndarray,
    scale: np.ndarray, shuffled: bool, seed: int, batch_size: int,
) -> np.ndarray:
    torch = _torch()
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            local = indices[start : start + batch_size]
            values, mask, _, _ = _batch(
                table, local, mean=mean, scale=scale, shuffled=shuffled, seed=seed,
            )
            gap, _ = model(values, mask)
            result.extend(torch.sigmoid(gap).cpu().numpy().tolist())
    return np.asarray(result, dtype=np.float32)


def _train_tcn(
    table: GapTable, fit_indices: np.ndarray, *, epochs: int, hidden: int,
    dropout: float, learning_rate: float, auxiliary_weight: float,
    shuffled: bool, seed: int, batch_size: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    mean, scale = _line_scaler(table, fit_indices)
    model = _make_tcn(table.features.shape[1], hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    local_targets = table.targets[fit_indices]
    negative_weight = max(1.0, np.count_nonzero(local_targets == 1) / max(1, np.count_nonzero(local_targets == 0)))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = fit_indices[rng.permutation(len(fit_indices))]
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            values, mask, line_target, gap_target = _batch(
                table, indices, mean=mean, scale=scale, shuffled=shuffled, seed=seed,
            )
            gap_logit, line_logit = model(values, mask)
            gap_weight = torch.where(gap_target > 0.5, 1.0, float(negative_weight))
            gap_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gap_logit, gap_target, weight=gap_weight,
            )
            valid_line_target = line_target[mask.bool()]
            valid_line_logit = line_logit[mask.bool()]
            line_negative_weight = max(
                1.0,
                float((valid_line_target > 0.5).sum()) / max(1.0, float((valid_line_target <= 0.5).sum())),
            )
            line_weight = torch.where(valid_line_target > 0.5, 1.0, line_negative_weight)
            line_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                valid_line_logit, valid_line_target, weight=line_weight,
            )
            loss = gap_loss + auxiliary_weight * line_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model, mean, scale


def _select_tcn_epoch(
    table: GapTable, fit: np.ndarray, holdout: np.ndarray, *, max_epochs: int,
    hidden: int, dropout: float, learning_rate: float, auxiliary_weight: float,
    shuffled: bool, seed: int, batch_size: int,
) -> tuple[int, list[dict[str, Any]]]:
    # Checkpoints at a small predeclared grid keep the compute/tuning budget equal
    # for the ordered model and its shuffled-order control.
    candidates = sorted({max(1, max_epochs // 3), max(2, 2 * max_epochs // 3), max_epochs})
    reports = []
    best_epoch = candidates[0]
    best_score = (-1.0, -1.0)
    for epoch in candidates:
        model, mean, scale = _train_tcn(
            table, fit, epochs=epoch, hidden=hidden, dropout=dropout,
            learning_rate=learning_rate, auxiliary_weight=auxiliary_weight,
            shuffled=shuffled, seed=seed + epoch, batch_size=batch_size,
        )
        probability = _predict_tcn(
            model, table, holdout, mean=mean, scale=scale, shuffled=shuffled,
            seed=seed, batch_size=batch_size,
        )
        break_ap = float(_sklearn()["average_precision"](
            table.targets[holdout] == 0, 1.0 - probability,
        ))
        connect_ap = float(_sklearn()["average_precision"](
            table.targets[holdout], probability,
        ))
        reports.append({"epochs": epoch, "break_pr_auc": break_ap, "connect_pr_auc": connect_ap})
        if (break_ap, connect_ap) > best_score:
            best_epoch, best_score = epoch, (break_ap, connect_ap)
    return best_epoch, reports


def _slice_metrics(
    table: GapTable, probability: np.ndarray, thresholds: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"by_source": {}, "by_length_bucket": {}}
    sources = np.asarray([row["source"] for row in table.metadata])
    buckets = np.asarray([row["length_bucket"] for row in table.metadata])
    for values, key in ((sources, "by_source"), (buckets, "by_length_bucket")):
        for value in sorted(set(values.tolist())):
            rows = np.flatnonzero(values == value)
            if len(np.unique(table.targets[rows])) < 2:
                continue
            result[key][str(value)] = binary_metrics(
                table.targets[rows], probability[rows], thresholds[rows]
            )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = GapTable(Path(args.table_dir))
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    n = len(table.targets)
    pooled = pooled_features(table)
    n_folds = int(table.folds.max()) + 1
    arm_probability = {arm: np.full(n, np.nan, dtype=np.float32) for arm in ARMS}
    arm_threshold = {arm: np.full(n, np.nan, dtype=np.float32) for arm in ARMS}
    fold_reports: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}

    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        outer_holdout = np.flatnonzero(table.folds == outer)
        outer_fit = np.flatnonzero(table.folds != outer)
        inner_fit = np.flatnonzero((table.folds != outer) & (table.folds != inner))
        inner_holdout = np.flatnonzero(table.folds == inner)
        for name, rows in (
            ("outer holdout", outer_holdout), ("outer fit", outer_fit),
            ("inner fit", inner_fit), ("inner holdout", inner_holdout),
        ):
            if {int(value) for value in np.unique(table.targets[rows])} != {0, 1}:
                raise ValueError(f"fold {outer} {name} lacks a gap class")

        inner_model = _fit_hist(pooled, table.targets, inner_fit, seed=args.seed + outer)
        inner_probability = inner_model.predict_proba(pooled[inner_holdout])[:, 1]
        selection = best_safety_threshold(
            table.targets[inner_holdout], inner_probability,
            max_false_connect_rate=args.max_false_connect_rate,
        )
        model = _fit_hist(pooled, table.targets, outer_fit, seed=args.seed + 100 + outer)
        probability = model.predict_proba(pooled[outer_holdout])[:, 1].astype(np.float32)
        arm_probability["pooled_hist"][outer_holdout] = probability
        arm_threshold["pooled_hist"][outer_holdout] = selection["threshold"]
        with (output / f"pooled_hist.fold{outer}.pkl").open("xb") as handle:
            pickle.dump(model, handle)
        fold_reports["pooled_hist"].append({
            "outer_fold": outer, "inner_fold": inner,
            "fit_count": len(outer_fit), "holdout_count": len(outer_holdout),
            "threshold_selection": selection,
        })

        for arm, shuffled in (("ordered_tcn", False), ("shuffled_tcn", True)):
            selected_epoch, candidates = _select_tcn_epoch(
                table, inner_fit, inner_holdout, max_epochs=args.epochs,
                hidden=args.hidden, dropout=args.dropout,
                learning_rate=args.learning_rate, auxiliary_weight=args.auxiliary_weight,
                shuffled=shuffled, seed=args.seed + outer * 1000 + int(shuffled) * 100,
                batch_size=args.batch_size,
            )
            inner_model_tcn, inner_mean, inner_scale = _train_tcn(
                table, inner_fit, epochs=selected_epoch, hidden=args.hidden,
                dropout=args.dropout, learning_rate=args.learning_rate,
                auxiliary_weight=args.auxiliary_weight, shuffled=shuffled,
                seed=args.seed + outer * 1000 + int(shuffled) * 100 + 50,
                batch_size=args.batch_size,
            )
            inner_probability = _predict_tcn(
                inner_model_tcn, table, inner_holdout, mean=inner_mean, scale=inner_scale,
                shuffled=shuffled, seed=args.seed, batch_size=args.batch_size,
            )
            selection = best_safety_threshold(
                table.targets[inner_holdout], inner_probability,
                max_false_connect_rate=args.max_false_connect_rate,
            )
            final_model, mean, scale = _train_tcn(
                table, outer_fit, epochs=selected_epoch, hidden=args.hidden,
                dropout=args.dropout, learning_rate=args.learning_rate,
                auxiliary_weight=args.auxiliary_weight, shuffled=shuffled,
                seed=args.seed + outer * 1000 + int(shuffled) * 100 + 75,
                batch_size=args.batch_size,
            )
            probability = _predict_tcn(
                final_model, table, outer_holdout, mean=mean, scale=scale,
                shuffled=shuffled, seed=args.seed, batch_size=args.batch_size,
            )
            arm_probability[arm][outer_holdout] = probability
            arm_threshold[arm][outer_holdout] = selection["threshold"]
            torch = _torch()
            torch.save({
                "state_dict": final_model.state_dict(), "mean": mean, "scale": scale,
                "input_dim": int(table.features.shape[1]), "hidden": args.hidden,
                "dropout": args.dropout, "epochs": selected_epoch, "shuffled": shuffled,
            }, output / f"{arm}.fold{outer}.pt")
            fold_reports[arm].append({
                "outer_fold": outer, "inner_fold": inner,
                "fit_count": len(outer_fit), "holdout_count": len(outer_holdout),
                "selected_epochs": selected_epoch, "epoch_candidates": candidates,
                "threshold_selection": selection,
            })

    report_arms = {}
    for arm in ARMS:
        if not np.isfinite(arm_probability[arm]).all() or not np.isfinite(arm_threshold[arm]).all():
            raise RuntimeError(f"{arm} OOF predictions are incomplete")
        _save(output / f"{arm}.oof_probability.npy", arm_probability[arm])
        _save(output / f"{arm}.oof_threshold.npy", arm_threshold[arm])
        report_arms[arm] = {
            "oof_metrics": binary_metrics(
                table.targets, arm_probability[arm], arm_threshold[arm]
            ),
            **_slice_metrics(table, arm_probability[arm], arm_threshold[arm]),
            "folds": fold_reports[arm],
        }

    ordered_break = report_arms["ordered_tcn"]["oof_metrics"]["break_pr_auc"]
    shuffled_break = report_arms["shuffled_tcn"]["oof_metrics"]["break_pr_auc"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_gap_connect_research_oof",
        "validation_opened": False,
        "label_tier": table.manifest["label_tier"],
        "deployment_approved": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "gap_count": n,
        "positive_count": int(np.count_nonzero(table.targets)),
        "negative_count": int(np.count_nonzero(table.targets == 0)),
        "feature_count": int(table.features.shape[1]),
        "arms": report_arms,
        "ordered_minus_shuffled_break_pr_auc": ordered_break - shuffled_break,
        "interpretation_guard": (
            "Silver OOF compares representations only. The reviewed table has no negative gap; "
            "a fresh reviewed edge audit is required before model or threshold selection."
        ),
        "inputs": {
            "table_manifest_sha256": sha256_file(table.root / "manifest.json"),
            "table_receipt_sha256": sha256_file(table.root / "receipt.json"),
        },
        "training": {
            "epochs": args.epochs, "hidden": args.hidden, "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "auxiliary_weight": args.auxiliary_weight,
            "batch_size": args.batch_size,
            "max_false_connect_rate_inner": args.max_false_connect_rate,
        },
    }
    _write_json_new(output / "report.json", report)
    lines = [
        "# Gap-only connect OOF experiment\n\n",
        f"- weak-silver gaps: {n:,} ({report['positive_count']:,} connect; "
        f"{report['negative_count']:,} do not connect)\n",
        "- validation opened: no\n",
        "- deployment approved: no\n\n",
        "| arm | connect precision | connect recall | false connects | break PR-AUC |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for arm in ARMS:
        metrics = report_arms[arm]["oof_metrics"]
        lines.append(
            f"| {arm} | {metrics['connect_precision']:.6f} | {metrics['connect_recall']:.6f} | "
            f"{metrics['false_connect_count']} | {metrics['break_pr_auc']:.6f} |\n"
        )
    lines.append(
        "\nThe ordered model must beat both the pooled baseline and shuffled-order control "
        "before sequence order is credited. These weak-silver results cannot approve deployment.\n"
    )
    descriptor = os.open(output / "README.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("".join(lines))
    _write_json_new(output / "receipt.json", {**report, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-false-connect-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
