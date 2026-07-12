#!/usr/bin/env python3
"""NumPy BIOES linear-chain CRF for the C1/C2 CPU baselines.

This is a real conditional log-likelihood CRF (forward/backward training and
masked Viterbi), not a post-hoc smoother relabelled as a CRF.  It is deliberately
small and dependency-light so the feature baselines can run on a Clariden CPU
node.  The production Rust detector is not modified by this research module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sequence_models.bib_ladder import (
        active_classes_for_documents,
        configure_runtime,
        mark_silver_safety_unavailable,
        peak_rss_bytes,
        select_shared_calibration,
        target_name,
        verify_selection_bundle,
    )
    from sequence_models.contract import GoldDocument, sha256_file
    from sequence_models.evaluate import evaluate, read_predictions
    from sequence_models.features import (
        TAGS,
        FeatureEncoder,
        allowed_transition_mask,
        bioes_to_classes,
        document_tag_ids,
    )
else:
    from .bib_ladder import (
        active_classes_for_documents,
        configure_runtime,
        mark_silver_safety_unavailable,
        peak_rss_bytes,
        select_shared_calibration,
        target_name,
        verify_selection_bundle,
    )
    from .contract import GoldDocument, sha256_file
    from .evaluate import evaluate, read_predictions
    from .features import (
        TAGS,
        FeatureEncoder,
        allowed_transition_mask,
        bioes_to_classes,
        document_tag_ids,
    )

NEG_INF = -np.inf


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    safe_maximum = np.where(finite, maximum, 0.0)
    total = np.sum(np.exp(values - safe_maximum), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        result = np.where(finite, maximum + np.log(total), NEG_INF)
    if axis is None:
        return result.reshape(())
    return np.squeeze(result, axis=axis)


@dataclass
class SequenceExample:
    document: GoldDocument
    features: list[dict[int, float]]
    tags: np.ndarray
    line_indices: tuple[int, ...]


class LinearChainCRF:
    """Sparse-emission, masked BIOES linear-chain CRF."""

    def __init__(
        self,
        n_features: int,
        *,
        seed: int = 0,
        active_classes: Sequence[str] = ("BIB", "TOC"),
    ):
        self.n_features = int(n_features)
        self.n_tags = len(TAGS)
        rng = np.random.default_rng(seed)
        self.emission = rng.normal(0.0, 0.002, (self.n_features, self.n_tags)).astype(
            np.float64
        )
        self.emission_bias = np.zeros(self.n_tags, dtype=np.float64)
        self.transition = np.zeros((self.n_tags, self.n_tags), dtype=np.float64)
        self.start = np.zeros(self.n_tags, dtype=np.float64)
        self.end = np.zeros(self.n_tags, dtype=np.float64)
        unknown = set(active_classes) - {"BIB", "TOC"}
        if unknown:
            raise ValueError(f"unknown active classes: {sorted(unknown)!r}")
        self.active_classes = tuple(sorted(set(active_classes)))
        active = np.asarray(
            [
                tag == "O"
                or any(tag.endswith(f"-{target}") for target in self.active_classes)
                for tag in TAGS
            ]
        )
        transition, start, end = allowed_transition_mask()
        self.active_tag_mask = active
        self.transition_mask = transition & active[:, None] & active[None, :]
        self.start_mask = start & active
        self.end_mask = end & active

    def emission_scores(self, rows: Sequence[Mapping[int, float]]) -> np.ndarray:
        scores = np.repeat(self.emission_bias[None, :], len(rows), axis=0)
        for t, row in enumerate(rows):
            for index, value in row.items():
                scores[t] += self.emission[int(index)] * float(value)
        scores[:, ~self.active_tag_mask] = NEG_INF
        return scores

    def _masked_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transition = np.where(self.transition_mask, self.transition, NEG_INF)
        start = np.where(self.start_mask, self.start, NEG_INF)
        end = np.where(self.end_mask, self.end, NEG_INF)
        return transition, start, end

    def _forward_backward(
        self, emissions: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        length = emissions.shape[0]
        if length == 0:
            raise ValueError("CRF sequences cannot be empty")
        active = np.flatnonzero(self.active_tag_mask)
        local_emissions = emissions[:, active]
        local_transition_mask = self.transition_mask[np.ix_(active, active)]
        local_start_mask = self.start_mask[active]
        local_end_mask = self.end_mask[active]
        if (
            not local_start_mask.any()
            or not local_end_mask.any()
            or not local_transition_mask.any(axis=0).all()
            or not local_transition_mask.any(axis=1).all()
        ):
            raise ValueError("active CRF graph is not a complete BIOES state machine")
        transition = np.where(
            local_transition_mask,
            self.transition[np.ix_(active, active)],
            NEG_INF,
        )
        start = np.where(local_start_mask, self.start[active], NEG_INF)
        end = np.where(local_end_mask, self.end[active], NEG_INF)
        alpha_local = np.empty((length, len(active)), dtype=np.float64)
        alpha_local[0] = start + local_emissions[0]
        for t in range(1, length):
            alpha_local[t] = local_emissions[t] + _logsumexp(
                alpha_local[t - 1][:, None] + transition, axis=0
            )
        log_z = float(_logsumexp(alpha_local[-1, local_end_mask] + end[local_end_mask]))

        beta_local = np.empty_like(alpha_local)
        beta_local[-1] = end
        for t in range(length - 2, -1, -1):
            beta_local[t] = _logsumexp(
                transition
                + local_emissions[t + 1][None, :]
                + beta_local[t + 1][None, :],
                axis=1,
            )
        node_local = np.exp(alpha_local + beta_local - log_z)
        edge_local = np.empty(
            (max(0, length - 1), len(active), len(active)), dtype=np.float64
        )
        for t in range(1, length):
            edge_local[t - 1] = np.exp(
                alpha_local[t - 1][:, None]
                + transition
                + local_emissions[t][None, :]
                + beta_local[t][None, :]
                - log_z
            )
        start_marginal = np.exp(start + local_emissions[0] + beta_local[0] - log_z)
        end_marginal = np.exp(alpha_local[-1] + end - log_z)

        alpha = np.full((length, self.n_tags), NEG_INF, dtype=np.float64)
        beta = np.full_like(alpha, NEG_INF)
        node = np.zeros_like(alpha)
        edge = np.zeros(
            (max(0, length - 1), self.n_tags, self.n_tags), dtype=np.float64
        )
        boundary = np.zeros((2, self.n_tags), dtype=np.float64)
        alpha[:, active] = alpha_local
        beta[:, active] = beta_local
        node[:, active] = node_local
        edge[:, active[:, None], active[None, :]] = edge_local
        boundary[0, active] = start_marginal
        boundary[1, active] = end_marginal
        return log_z, alpha, beta, node, edge, boundary

    def nll_and_grad(
        self, rows: Sequence[Mapping[int, float]], gold: np.ndarray
    ) -> tuple[
        float, dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        if len(rows) != len(gold) or not len(gold):
            raise ValueError(
                "gold tag sequence must match a non-empty feature sequence"
            )
        if (
            not self.start_mask[gold[0]]
            or not self.end_mask[gold[-1]]
            or not self.active_tag_mask[gold].all()
            or any(
                not self.transition_mask[left, right]
                for left, right in zip(gold, gold[1:])
            )
        ):
            raise ValueError("gold tag sequence contains an illegal BIOES path")
        emissions = self.emission_scores(rows)
        log_z, _alpha, _beta, node, edge, boundary = self._forward_backward(emissions)
        transition, start, end = self._masked_parameters()
        path_score = start[gold[0]] + emissions[0, gold[0]] + end[gold[-1]]
        if len(gold) > 1:
            path_score += sum(
                transition[left, right] for left, right in zip(gold[:-1], gold[1:])
            )
            path_score += float(sum(emissions[t, gold[t]] for t in range(1, len(gold))))
        nll = float(log_z - path_score)

        emission_grad: dict[int, np.ndarray] = {}
        bias_grad = node.sum(axis=0)
        for t, (row, gold_tag) in enumerate(zip(rows, gold)):
            bias_grad[gold_tag] -= 1.0
            delta = node[t].copy()
            delta[gold_tag] -= 1.0
            for index, value in row.items():
                grad = emission_grad.setdefault(
                    int(index), np.zeros(self.n_tags, dtype=np.float64)
                )
                grad += float(value) * delta

        transition_grad = (
            edge.sum(axis=0) if len(edge) else np.zeros_like(self.transition)
        )
        for left, right in zip(gold[:-1], gold[1:]):
            transition_grad[left, right] -= 1.0
        transition_grad[~self.transition_mask] = 0.0
        start_grad = boundary[0].copy()
        start_grad[gold[0]] -= 1.0
        start_grad[~self.start_mask] = 0.0
        end_grad = boundary[1].copy()
        end_grad[gold[-1]] -= 1.0
        end_grad[~self.end_mask] = 0.0
        return nll, emission_grad, bias_grad, transition_grad, start_grad, end_grad

    def viterbi(
        self, rows: Sequence[Mapping[int, float]], *, deletion_bias: float = 0.0
    ) -> np.ndarray:
        if not rows:
            raise ValueError("CRF sequences cannot be empty")
        emissions = self.emission_scores(rows).copy()
        # A single conservative deletion penalty is calibrated on validation.
        # It shifts both structure classes without invalidating BIOES paths.
        for index, tag in enumerate(TAGS):
            if tag != "O":
                emissions[:, index] -= float(deletion_bias)
        active = np.flatnonzero(self.active_tag_mask)
        emissions = emissions[:, active]
        transition_mask = self.transition_mask[np.ix_(active, active)]
        start_mask = self.start_mask[active]
        end_mask = self.end_mask[active]
        transition = np.where(
            transition_mask,
            self.transition[np.ix_(active, active)],
            NEG_INF,
        )
        start = np.where(start_mask, self.start[active], NEG_INF)
        end = np.where(end_mask, self.end[active], NEG_INF)
        length = len(rows)
        score = np.empty((length, len(active)), dtype=np.float64)
        back = np.zeros((length, len(active)), dtype=np.int16)
        score[0] = start + emissions[0]
        for t in range(1, length):
            candidates = score[t - 1][:, None] + transition
            back[t] = np.argmax(candidates, axis=0)
            score[t] = emissions[t] + np.max(candidates, axis=0)
        tags = np.empty(length, dtype=np.int64)
        allowed_end = np.flatnonzero(end_mask)
        tags[-1] = int(
            allowed_end[np.argmax(score[-1, allowed_end] + end[allowed_end])]
        )
        for t in range(length - 1, 0, -1):
            tags[t - 1] = back[t, tags[t]]
        return active[tags]

    def save(self, path: str | Path, metadata: Mapping[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite immutable model {path}")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    emission=self.emission,
                    emission_bias=self.emission_bias,
                    transition=self.transition,
                    start=self.start,
                    end=self.end,
                    active_tag_mask=self.active_tag_mask,
                    metadata=np.asarray(
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                    ),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
            os.unlink(temporary)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> tuple["LinearChainCRF", dict[str, Any]]:
        required = {
            "emission",
            "emission_bias",
            "transition",
            "start",
            "end",
            "active_tag_mask",
            "metadata",
        }
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != required:
                raise ValueError("feature checkpoint has an unexpected array inventory")
            metadata = json.loads(str(values["metadata"]))
            if not isinstance(metadata, dict):
                raise ValueError("feature checkpoint metadata is not an object")
            emission = values["emission"].astype(np.float64)
            bias = values["emission_bias"].astype(np.float64)
            transition = values["transition"].astype(np.float64)
            start = values["start"].astype(np.float64)
            end = values["end"].astype(np.float64)
            active_mask = values["active_tag_mask"].astype(bool)
        n_tags = len(TAGS)
        if (
            emission.ndim != 2
            or emission.shape[1] != n_tags
            or bias.shape != (n_tags,)
            or transition.shape != (n_tags, n_tags)
            or start.shape != (n_tags,)
            or end.shape != (n_tags,)
            or active_mask.shape != (n_tags,)
        ):
            raise ValueError("feature checkpoint array shapes are invalid")
        for name, value in (
            ("emission", emission),
            ("emission_bias", bias),
            ("transition", transition),
            ("start", start),
            ("end", end),
        ):
            if not np.isfinite(value).all():
                raise ValueError(
                    f"feature checkpoint {name} contains non-finite values"
                )
        active_classes = metadata.get("active_classes")
        if not isinstance(active_classes, list) or not active_classes:
            raise ValueError("feature checkpoint active_classes are missing")
        model = cls(emission.shape[0], active_classes=active_classes)
        if not np.array_equal(active_mask, model.active_tag_mask):
            raise ValueError("feature checkpoint active-tag mask differs from metadata")
        model.emission = emission
        model.emission_bias = bias
        model.transition = transition
        model.start = start
        model.end = end
        return model, metadata


def make_examples(
    documents: Sequence[GoldDocument], encoder: FeatureEncoder
) -> list[SequenceExample]:
    """Build known-label segments; UNKNOWN is never coerced to a negative label."""
    examples: list[SequenceExample] = []
    for document in documents:
        encoded = encoder.encode_document(document)
        start = 0
        while start < len(document.lines):
            while (
                start < len(document.lines) and document.lines[start].label == "UNKNOWN"
            ):
                start += 1
            end = start
            while end < len(document.lines) and document.lines[end].label != "UNKNOWN":
                end += 1
            if end > start:
                tags = document_tag_ids(
                    type("Segment", (), {"lines": document.lines[start:end]})()
                )
                examples.append(
                    SequenceExample(
                        document=document,
                        features=encoded[start:end],
                        tags=tags,
                        line_indices=tuple(range(start, end)),
                    )
                )
            start = end + 1
    return examples


def _gradient_norm(parts: Iterable[np.ndarray]) -> float:
    return math.sqrt(sum(float(np.sum(part * part)) for part in parts))


def train_model(
    model: LinearChainCRF,
    examples: Sequence[SequenceExample],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    gradient_clip: float,
    seed: int = 0,
) -> list[float]:
    """Train with deterministic per-sequence AdaGrad updates."""
    rng = random.Random(seed)
    acc_emission = np.full_like(model.emission, 1.0e-8)
    acc_bias = np.full_like(model.emission_bias, 1.0e-8)
    acc_transition = np.full_like(model.transition, 1.0e-8)
    acc_start = np.full_like(model.start, 1.0e-8)
    acc_end = np.full_like(model.end, 1.0e-8)
    history: list[float] = []
    order = list(range(len(examples)))
    for _epoch in range(int(epochs)):
        rng.shuffle(order)
        epoch_loss = 0.0
        for example_index in order:
            example = examples[example_index]
            loss, emission_grad, bias_grad, transition_grad, start_grad, end_grad = (
                model.nll_and_grad(example.features, example.tags)
            )
            epoch_loss += loss
            # Decoupled weight decay is deterministic and avoids densifying the
            # sparse emission gradient for every document.
            if l2:
                model.emission *= max(0.0, 1.0 - learning_rate * l2)
                model.transition *= max(0.0, 1.0 - learning_rate * l2)
            parts = list(emission_grad.values()) + [
                bias_grad,
                transition_grad,
                start_grad,
                end_grad,
            ]
            norm = _gradient_norm(parts)
            scale = min(1.0, float(gradient_clip) / max(norm, 1.0e-12))
            for index, grad in emission_grad.items():
                grad = grad * scale
                acc_emission[index] += grad * grad
                model.emission[index] -= (
                    learning_rate * grad / np.sqrt(acc_emission[index])
                )
            for parameter, accumulator, grad in (
                (model.emission_bias, acc_bias, bias_grad),
                (model.transition, acc_transition, transition_grad),
                (model.start, acc_start, start_grad),
                (model.end, acc_end, end_grad),
            ):
                grad = grad * scale
                accumulator += grad * grad
                parameter -= learning_rate * grad / np.sqrt(accumulator)
        history.append(epoch_loss / max(len(examples), 1))
    return history


def _token_metrics(
    examples: Sequence[SequenceExample], model: LinearChainCRF, bias: float
) -> dict[str, float]:
    predicted = correct = gold_action = action_tp = prose = false_prose = 0
    gold_by_class = {"BIB": 0, "TOC": 0}
    tp_by_class = {"BIB": 0, "TOC": 0}
    for example in examples:
        pred = bioes_to_classes(
            [TAGS[x] for x in model.viterbi(example.features, deletion_bias=bias)]
        )
        for line_index, guess in zip(example.line_indices, pred):
            line = example.document.lines[line_index]
            weight = line.token_count
            if guess != "O":
                predicted += weight
                correct += weight * int(line.label != "O")
            if line.label != "O":
                gold_action += weight
                action_tp += weight * int(guess != "O")
            if line.is_running_prose:
                prose += weight
                false_prose += weight * int(guess != "O")
            if line.label != "O":
                gold_by_class[line.label] += weight
                tp_by_class[line.label] += weight * int(guess == line.label)
    return {
        "action_precision": correct / predicted if predicted else 1.0,
        "action_recall": action_tp / gold_action if gold_action else 0.0,
        "predicted_action_tokens": predicted,
        "prose_contamination": false_prose / prose if prose else 0.0,
        "bib_recall": tp_by_class["BIB"] / gold_by_class["BIB"]
        if gold_by_class["BIB"]
        else 0.0,
        "toc_recall": tp_by_class["TOC"] / gold_by_class["TOC"]
        if gold_by_class["TOC"]
        else 0.0,
    }


def calibrate_deletion_bias(
    examples: Sequence[SequenceExample],
    model: LinearChainCRF,
    candidates: Sequence[float],
    *,
    reference_action_precision: float,
    active_classes: Sequence[str] = ("BIB",),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = _token_metrics(examples, model, float(candidate))
        rows.append(
            {
                "deletion_bias": float(candidate),
                "action_precision": metrics["action_precision"],
                "action_recall": metrics["action_recall"],
                "bib_recall": metrics["bib_recall"],
                "toc_recall": metrics["toc_recall"],
                "predicted_action_tokens": int(metrics["predicted_action_tokens"]),
            }
        )
    return select_shared_calibration(
        rows,
        reference_action_precision=reference_action_precision,
        active_classes=active_classes,
    )


def predict_documents(
    documents: Sequence[GoldDocument],
    encoder: FeatureEncoder,
    model: LinearChainCRF,
    *,
    deletion_bias: float,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for document in documents:
        features = encoder.encode_document(document)
        classes = ["O"] * len(document.lines)
        start = 0
        while start < len(document.lines):
            while (
                start < len(document.lines) and document.lines[start].label == "UNKNOWN"
            ):
                start += 1
            end = start
            while end < len(document.lines) and document.lines[end].label != "UNKNOWN":
                end += 1
            if end > start:
                tags = model.viterbi(features[start:end], deletion_bias=deletion_bias)
                classes[start:end] = bioes_to_classes([TAGS[x] for x in tags])
            start = end + 1
        result[document.document_id] = classes
    return result


def write_predictions(
    path: str | Path,
    documents: Sequence[GoldDocument],
    encoder: FeatureEncoder,
    model: LinearChainCRF,
    *,
    model_id: str,
    deletion_bias: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable predictions {output}")
    predictions = predict_documents(
        documents, encoder, model, deletion_bias=deletion_bias
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for document in documents:
                classes = predictions[document.document_id]
                row = {
                    "schema_version": "academic-structure-predictions-v1",
                    "model_id": model_id,
                    "document_id": document.document_id,
                    "work_id": document.work_id,
                    "source": document.source,
                    "split": document.split,
                    "lines": [
                        {
                            "line_id": line.line_id,
                            "abs_idx": line.abs_idx,
                            "prediction": guess,
                        }
                        for line, guess in zip(document.lines, classes)
                    ],
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _architecture(config: Mapping[str, Any], architecture_id: str) -> Mapping[str, Any]:
    for architecture in config["architecture_ladder"]:
        if architecture["id"] == architecture_id:
            if architecture["kind"] != "feature_crf":
                raise ValueError(f"{architecture_id!r} is not a feature CRF")
            return architecture
    raise ValueError(f"unknown architecture {architecture_id!r}")


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable receipt {output}")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _scrub_silver_safety(
    metrics: dict[str, Any], active_classes: Sequence[str] = ("BIB",)
) -> None:
    mark_silver_safety_unavailable(metrics, active_classes)


def _require_clariden_cpu(confirmed: bool, config: Mapping[str, Any]) -> None:
    execution = config.get("execution", {})
    if (
        execution.get("fit_location") != "Clariden CPU node only"
        or execution.get("accelerator") != "none"
        or execution.get("local_training_forbidden") is not True
    ):
        raise RuntimeError(
            "feature CRF config must forbid local or accelerated fitting"
        )
    if not confirmed:
        raise RuntimeError("feature CRF fitting requires --confirm-clariden-cpu-only")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "feature CRF fitting is forbidden outside a Clariden Slurm allocation"
        )
    if os.environ.get("SLURM_JOB_PARTITION") not in {"normal", "debug"}:
        raise RuntimeError("feature CRF fitting requires a Clariden compute partition")
    if platform.machine() != "aarch64":
        raise RuntimeError(
            "feature CRF fitting requires the Clariden aarch64 CPU runtime"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError(
            "feature CRF fitting requires CUDA_VISIBLE_DEVICES to disable accelerators"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-silver", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--validation-silver", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("c1-feature-bioes-crf", "c2-char-ngram-feature-bioes-crf"),
    )
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--uenv", required=True)
    parser.add_argument("--confirm-clariden-cpu-only", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    _require_clariden_cpu(args.confirm_clariden_cpu_only, config)
    architecture = _architecture(config, args.architecture)
    seed = int(config["execution"]["seed"])
    runtime = configure_runtime(config, uenv=args.uenv, effective_seed=seed)
    documents, validation_docs, selection_receipt = verify_selection_bundle(
        selection_silver_path=args.selection_silver,
        selection_manifest_path=args.selection_manifest,
        validation_silver_path=args.validation_silver,
        selection_receipt_path=args.selection_receipt,
        config_path=args.config,
    )
    active_classes = active_classes_for_documents(documents, config)
    train_docs = [doc for doc in documents if doc.split == "train"]
    if not train_docs or not validation_docs:
        raise RuntimeError("feature CRF requires non-empty train and validation splits")
    reference_predictions = read_predictions(
        args.reference_predictions, validation_docs
    )
    reference_metrics, _ = evaluate(
        validation_docs, reference_predictions, split="validation"
    )
    reference_precision = float(reference_metrics["token"]["action_precision"])
    started = time.perf_counter()
    encoder = FeatureEncoder(
        char_hash_dim=int(architecture.get("char_hash_dim", 0)),
        char_ngram_min=int(architecture.get("char_ngram_min", 2)),
        char_ngram_max=int(architecture.get("char_ngram_max", 5)),
    )
    train_examples = make_examples(train_docs, encoder)
    validation_examples = make_examples(validation_docs, encoder)
    model = LinearChainCRF(encoder.n_features, seed=seed, active_classes=active_classes)
    history = train_model(
        model,
        train_examples,
        epochs=int(architecture["epochs"]),
        learning_rate=float(architecture["learning_rate"]),
        l2=float(architecture["l2"]),
        gradient_clip=float(architecture["gradient_clip"]),
        seed=seed,
    )
    calibration = config["calibration"]
    calibration_receipt = calibrate_deletion_bias(
        validation_examples,
        model,
        calibration["deletion_bias_grid"],
        reference_action_precision=reference_precision,
        active_classes=active_classes,
    )
    deletion_bias = float(calibration_receipt["selected"]["deletion_bias"])
    metadata = {
        "schema_version": "academic-structure-feature-crf-v1",
        "architecture_id": args.architecture,
        "config_sha256": sha256_file(args.config),
        "silver_sha256": sha256_file(args.selection_silver),
        "split_manifest_sha256": sha256_file(args.selection_manifest),
        "validation_silver_sha256": sha256_file(args.validation_silver),
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "reference_predictions_sha256": sha256_file(args.reference_predictions),
        "contract_inventory_sha256": selection_receipt["selection_contract"][
            "inventory_sha256"
        ],
        "feature_encoder": encoder.metadata(),
        "tags": list(TAGS),
        "active_classes": list(active_classes),
        "evidence_tier": "LLM_silver",
        "production_eligible": False,
        "safety_metrics_available": False,
        "metric_semantics": "agreement_with_LLM_silver; not production safety",
        "calibration": calibration_receipt,
        "deletion_bias": deletion_bias,
        "training_loss": history,
        "seed": seed,
        "runtime": runtime,
        "test_used_for_training_or_calibration": False,
    }
    model.save(args.model_out, metadata)
    loaded_model, loaded_metadata = LinearChainCRF.load(args.model_out)
    if loaded_metadata != metadata:
        raise RuntimeError("reloaded feature checkpoint metadata differs")
    write_predictions(
        args.validation_predictions,
        validation_docs,
        encoder,
        loaded_model,
        model_id=args.architecture,
        deletion_bias=deletion_bias,
    )
    selected_predictions = read_predictions(
        args.validation_predictions, validation_docs
    )
    validation_metrics, _ = evaluate(
        validation_docs, selected_predictions, split="validation"
    )
    _scrub_silver_safety(validation_metrics, active_classes)
    runtime["wall_seconds"] = time.perf_counter() - started
    runtime["peak_rss_bytes"] = peak_rss_bytes()
    receipt = {
        "schema_version": "academic-structure-feature-crf-training-v2",
        "status": "passed_cpu_fit_checkpoint_reload_and_validation_prediction",
        "architecture_id": args.architecture,
        "target": target_name(active_classes),
        "active_classes": list(active_classes),
        "production_eligible": False,
        "inputs": {
            "selection_silver_sha256": sha256_file(args.selection_silver),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "validation_silver_sha256": sha256_file(args.validation_silver),
            "selection_receipt_sha256": sha256_file(args.selection_receipt),
            "config_sha256": sha256_file(args.config),
            "reference_predictions_sha256": sha256_file(args.reference_predictions),
        },
        "execution": runtime,
        "effective_seed": seed,
        "calibration": calibration_receipt,
        "training_loss": history,
        "validation_metrics": validation_metrics,
        "outputs": {
            "model_sha256": sha256_file(args.model_out),
            "validation_predictions_sha256": sha256_file(args.validation_predictions),
        },
        "historically_named_test_partition": {
            "documents_loaded": 0,
            "predictions_written": 0,
            "semantics": "sealed_retrospective_comparison_not_unbiased_test",
        },
    }
    _atomic_json(args.receipt_out, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
