#!/usr/bin/env python3
"""Ordered-versus-shuffled TCN comparison on selected gap data regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_gap_candidate_screen import (
    CandidateTable,
    _source_metrics,
    _write_json_new,
)
from .bibliography_gap_connect_models import (
    _make_tcn,
    best_safety_threshold,
    binary_metrics,
)
from .bibliography_gap_sampling import fit_weights, select_training_rows
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-candidate-sequence-oof-v1"
ARMS = ("ordered_tcn", "shuffled_tcn")


def _torch() -> Any:
    import torch

    return torch


class SequenceTable(CandidateTable):
    def sequence_pair(
        self, index: int, *, shuffled: bool, seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        features = np.asarray(self.features[start:end], dtype=np.float32)
        targets = np.asarray(self.line_targets[start:end], dtype=np.float32)
        if shuffled and len(features) > 1:
            identity = f"{seed}:{self.metadata[index]['variant_id']}"
            local_seed = int.from_bytes(
                hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little"
            )
            order = np.random.default_rng(local_seed).permutation(len(features))
            features, targets = features[order], targets[order]
        return features, targets


def _line_scaler(table: SequenceTable, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([
        np.asarray(
            table.features[int(table.offsets[index]) : int(table.offsets[index + 1])],
            dtype=np.float64,
        )
        for index in rows
    ])
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _batch(
    table: SequenceTable,
    indices: Sequence[int],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    shuffled: bool,
    seed: int,
    weight_lookup: Mapping[int, float] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    torch = _torch()
    sequences, line_targets = zip(*(
        table.sequence_pair(int(index), shuffled=shuffled, seed=seed) for index in indices
    ))
    maximum = max(len(values) for values in sequences)
    values = np.zeros((len(indices), maximum, table.features.shape[1]), dtype=np.float32)
    mask = np.zeros((len(indices), maximum), dtype=np.float32)
    lines = np.zeros((len(indices), maximum), dtype=np.float32)
    weights = np.ones(len(indices), dtype=np.float32)
    for row, (index, sequence, target) in enumerate(
        zip(indices, sequences, line_targets, strict=True)
    ):
        length = len(sequence)
        values[row, :length] = np.clip((sequence - mean) / scale, -10.0, 10.0)
        mask[row, :length] = 1.0
        lines[row, :length] = target
        if weight_lookup is not None:
            weights[row] = float(weight_lookup[int(index)])
    weights /= max(1.0e-12, float(weights.mean()))
    return (
        torch.from_numpy(values),
        torch.from_numpy(mask),
        torch.from_numpy(lines),
        torch.from_numpy(table.targets[np.asarray(indices)].astype(np.float32)),
        torch.from_numpy(weights),
    )


def _train(
    table: SequenceTable,
    rows: np.ndarray,
    *,
    epochs: int,
    hidden: int,
    dropout: float,
    learning_rate: float,
    auxiliary_weight: float,
    shuffled: bool,
    seed: int,
    batch_size: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    mean, scale = _line_scaler(table, rows)
    model = _make_tcn(table.features.shape[1], hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    weights = fit_weights(table.metadata, table.targets, rows)
    weight_lookup = {int(index): float(weight) for index, weight in zip(rows, weights, strict=True)}
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rows[rng.permutation(len(rows))]
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            values, mask, line_target, gap_target, example_weight = _batch(
                table,
                indices,
                mean=mean,
                scale=scale,
                shuffled=shuffled,
                seed=seed,
                weight_lookup=weight_lookup,
            )
            gap_logit, line_logit = model(values, mask)
            gap_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    gap_logit, gap_target, reduction="none"
                )
                * example_weight
            ).mean()
            valid = mask.bool()
            raw_line_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                line_logit[valid], line_target[valid], reduction="none"
            )
            line_example_weight = example_weight[:, None].expand_as(mask)[valid]
            line_example_weight /= line_example_weight.mean().clamp_min(1.0e-12)
            line_loss = (raw_line_loss * line_example_weight).mean()
            loss = gap_loss + auxiliary_weight * line_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model, mean, scale


def _predict(
    model: Any,
    table: SequenceTable,
    rows: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    shuffled: bool,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    torch = _torch()
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            indices = rows[start : start + batch_size]
            values, mask, _, _, _ = _batch(
                table,
                indices,
                mean=mean,
                scale=scale,
                shuffled=shuffled,
                seed=seed,
            )
            logits, _ = model(values, mask)
            result.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(result, dtype=np.float32)


def _select_epochs(
    table: SequenceTable,
    fit_rows: np.ndarray,
    holdout_rows: np.ndarray,
    *,
    max_epochs: int,
    hidden: int,
    dropout: float,
    learning_rate: float,
    auxiliary_weight: float,
    shuffled: bool,
    seed: int,
    batch_size: int,
) -> tuple[int, list[dict[str, Any]]]:
    candidates = sorted({max(1, max_epochs // 3), max(2, 2 * max_epochs // 3), max_epochs})
    reports = []
    best_epoch, best = candidates[0], (-1.0, -1.0)
    from sklearn.metrics import average_precision_score

    for epochs in candidates:
        model, mean, scale = _train(
            table,
            fit_rows,
            epochs=epochs,
            hidden=hidden,
            dropout=dropout,
            learning_rate=learning_rate,
            auxiliary_weight=auxiliary_weight,
            shuffled=shuffled,
            seed=seed + epochs,
            batch_size=batch_size,
        )
        probability = _predict(
            model,
            table,
            holdout_rows,
            mean=mean,
            scale=scale,
            shuffled=shuffled,
            seed=seed,
            batch_size=batch_size,
        )
        break_ap = float(average_precision_score(
            table.targets[holdout_rows] == 0, 1.0 - probability
        ))
        connect_ap = float(average_precision_score(table.targets[holdout_rows], probability))
        reports.append({"epochs": epochs, "break_pr_auc": break_ap, "connect_pr_auc": connect_ap})
        if (break_ap, connect_ap) > best:
            best_epoch, best = epochs, (break_ap, connect_ap)
    return best_epoch, reports


def _bootstrap_delta(
    targets: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    works: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score

    unique = np.unique(works)
    by_work = {work: np.flatnonzero(works == work) for work in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_work[work] for work in sampled])
        if len(np.unique(targets[rows])) < 2:
            continue
        values.append(
            float(average_precision_score(targets[rows] == 0, 1.0 - first[rows]))
            - float(average_precision_score(targets[rows] == 0, 1.0 - second[rows]))
        )
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _run_configuration(
    *,
    table: SequenceTable,
    genuine_rows: np.ndarray,
    train_rows: np.ndarray,
    regime: str,
    size_label: str,
    args: argparse.Namespace,
    output: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_folds = int(table.folds.max()) + 1
    genuine_fold = table.folds[genuine_rows]
    arm_probability = {arm: np.full(len(genuine_rows), np.nan, dtype=np.float32) for arm in ARMS}
    arm_threshold = {arm: np.full(len(genuine_rows), np.nan, dtype=np.float32) for arm in ARMS}
    arm_folds = {arm: [] for arm in ARMS}
    config_key = f"{regime}__{size_label}"

    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        inner_fit = train_rows[
            (table.folds[train_rows] != outer) & (table.folds[train_rows] != inner)
        ]
        outer_fit = train_rows[table.folds[train_rows] != outer]
        inner_eval = genuine_rows[genuine_fold == inner]
        outer_eval_local = np.flatnonzero(genuine_fold == outer)
        outer_eval = genuine_rows[outer_eval_local]
        for arm, shuffled in (("ordered_tcn", False), ("shuffled_tcn", True)):
            selected_epochs, epoch_reports = _select_epochs(
                table,
                inner_fit,
                inner_eval,
                max_epochs=args.epochs,
                hidden=args.hidden,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                auxiliary_weight=args.auxiliary_weight,
                shuffled=shuffled,
                seed=args.seed + outer * 1000 + int(shuffled) * 100,
                batch_size=args.batch_size,
            )
            inner_model, inner_mean, inner_scale = _train(
                table,
                inner_fit,
                epochs=selected_epochs,
                hidden=args.hidden,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                auxiliary_weight=args.auxiliary_weight,
                shuffled=shuffled,
                seed=args.seed + outer * 1000 + int(shuffled) * 100 + 50,
                batch_size=args.batch_size,
            )
            inner_probability = _predict(
                inner_model,
                table,
                inner_eval,
                mean=inner_mean,
                scale=inner_scale,
                shuffled=shuffled,
                seed=args.seed,
                batch_size=args.batch_size,
            )
            selection = best_safety_threshold(
                table.targets[inner_eval],
                inner_probability,
                max_false_connect_rate=args.maximum_false_connect_rate,
            )
            final_model, mean, scale = _train(
                table,
                outer_fit,
                epochs=selected_epochs,
                hidden=args.hidden,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                auxiliary_weight=args.auxiliary_weight,
                shuffled=shuffled,
                seed=args.seed + outer * 1000 + int(shuffled) * 100 + 75,
                batch_size=args.batch_size,
            )
            arm_probability[arm][outer_eval_local] = _predict(
                final_model,
                table,
                outer_eval,
                mean=mean,
                scale=scale,
                shuffled=shuffled,
                seed=args.seed,
                batch_size=args.batch_size,
            )
            arm_threshold[arm][outer_eval_local] = selection["threshold"]
            _torch().save({
                "state_dict": final_model.state_dict(),
                "mean": mean,
                "scale": scale,
                "input_dim": int(table.features.shape[1]),
                "hidden": args.hidden,
                "dropout": args.dropout,
                "epochs": selected_epochs,
                "shuffled": shuffled,
                "regime": regime,
                "size": size_label,
            }, output / f"{config_key}__{arm}.fold{outer}.pt")
            arm_folds[arm].append({
                "outer_fold": outer,
                "inner_fold": inner,
                "selected_epochs": selected_epochs,
                "epoch_candidates": epoch_reports,
                "threshold_selection": selection,
            })
            print(
                "GAP_SEQUENCE_FOLD "
                f"configuration={config_key} arm={arm} outer_fold={outer} "
                f"epochs={selected_epochs}",
                flush=True,
            )

    targets = table.targets[genuine_rows]
    sources = np.asarray([table.metadata[index]["source"] for index in genuine_rows])
    report_arms = {}
    for arm in ARMS:
        if not np.isfinite(arm_probability[arm]).all():
            raise RuntimeError(f"{config_key}/{arm} predictions are incomplete")
        report_arms[arm] = {
            "oof_metrics": binary_metrics(
                targets, arm_probability[arm], arm_threshold[arm]
            ),
            "by_source": _source_metrics(
                targets, arm_probability[arm], arm_threshold[arm], sources
            ),
            "folds": arm_folds[arm],
        }
    return {
        "regime": regime,
        "size": size_label,
        "train_row_count": len(train_rows),
        "train_negative_boundary_group_count": len({
            table.metadata[index]["boundary_group_id"]
            for index in train_rows if table.targets[index] == 0
        }),
        "arms": report_arms,
    }, arm_probability, arm_threshold


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    table = SequenceTable(Path(args.table_dir))
    screen_root = Path(args.screen_dir).resolve()
    selected = json.loads((screen_root / "selected.json").read_text(encoding="utf-8"))
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    genuine_rows = np.asarray([
        index for index, row in enumerate(table.metadata)
        if bool(row["genuine_deployment_candidate"])
    ], dtype=np.int64)
    configs = []
    seen = set()
    pooled_arm_by_config = {}
    for row in selected["selected_for_sequence"]:
        key = (str(row["regime"]), str(row["size"]))
        pooled_arm_by_config[key] = str(row["arm"])
        if key not in seen:
            seen.add(key)
            configs.append(key)
    reports = []
    all_probability = {}
    all_threshold = {}
    for regime, size_label in configs:
        train_rows = select_training_rows(
            table.metadata,
            table.targets,
            regime=regime,
            negative_group_limit=None if size_label == "all" else int(size_label),
            seed=args.seed,
        )
        report, probability, threshold = _run_configuration(
            table=table,
            genuine_rows=genuine_rows,
            train_rows=train_rows,
            regime=regime,
            size_label=size_label,
            args=args,
            output=output,
        )
        reports.append(report)
        for arm in ARMS:
            all_probability[f"{regime}__{size_label}__{arm}"] = probability[arm]
            all_threshold[f"{regime}__{size_label}__{arm}"] = threshold[arm]

    pooled_selected = selected["selected_pooled"]
    targets = table.targets[genuine_rows]
    works = np.asarray([table.metadata[index]["work_id"] for index in genuine_rows])
    comparisons = []
    with np.load(screen_root / "oof_probabilities.npz") as archive:
        for report in reports:
            config = (str(report["regime"]), str(report["size"]))
            key = f"{config[0]}__{config[1]}"
            ordered = all_probability[f"{key}__ordered_tcn"]
            shuffled = all_probability[f"{key}__shuffled_tcn"]
            pooled_arm = pooled_arm_by_config[config]
            pooled_probability = np.asarray(
                archive[f"{key}__{pooled_arm}"], dtype=np.float32
            )
            ordered_minus_shuffled = _bootstrap_delta(
                targets, ordered, shuffled, works,
                seed=args.seed + 1,
                replicates=args.bootstrap_replicates,
            )
            ordered_minus_pooled = _bootstrap_delta(
                targets, ordered, pooled_probability, works,
                seed=args.seed + 2,
                replicates=args.bootstrap_replicates,
            )
            comparisons.append({
                "regime": report["regime"],
                "size": report["size"],
                "pooled_arm": pooled_arm,
                "ordered_minus_shuffled_break_pr_auc": ordered_minus_shuffled,
                "ordered_minus_same_data_pooled_break_pr_auc": ordered_minus_pooled,
                "representation_evidence_passed": (
                    ordered_minus_shuffled["lower_95"] > 0
                    and ordered_minus_pooled["lower_95"] > 0
                ),
                "order_credited": False,
                "order_credit_guard": "end-to-end block gates have not yet been rerun",
            })

    np.savez_compressed(output / "oof_probabilities.npz", **all_probability)
    np.savez_compressed(output / "oof_thresholds.npz", **all_threshold)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_selected_order_controls",
        "validation_opened": False,
        "deployment_approved": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "selected_pooled": pooled_selected,
        "configurations": reports,
        "comparisons": comparisons,
        "training": {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "auxiliary_weight": args.auxiliary_weight,
            "batch_size": args.batch_size,
        },
        "inputs": {
            "table_manifest_sha256": sha256_file(table.root / "manifest.json"),
            "table_receipt_sha256": sha256_file(table.root / "receipt.json"),
            "screen_report_sha256": sha256_file(screen_root / "report.json"),
            "screen_selection_sha256": sha256_file(screen_root / "selected.json"),
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(output / "receipt.json", {**report, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--maximum-false-connect-rate", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
