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
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sequence_models.contract import GoldDocument, read_gold, sha256_file, validate_gold
    from sequence_models.features import (
        TAGS,
        FeatureEncoder,
        allowed_transition_mask,
        bioes_to_classes,
        document_tag_ids,
    )
else:
    from .contract import GoldDocument, read_gold, sha256_file, validate_gold
    from .features import (
        TAGS,
        FeatureEncoder,
        allowed_transition_mask,
        bioes_to_classes,
        document_tag_ids,
    )

NEG_INF = -1.0e30


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
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

    def __init__(self, n_features: int, *, seed: int = 0):
        self.n_features = int(n_features)
        self.n_tags = len(TAGS)
        rng = np.random.default_rng(seed)
        self.emission = rng.normal(0.0, 0.002, (self.n_features, self.n_tags)).astype(np.float64)
        self.emission_bias = np.zeros(self.n_tags, dtype=np.float64)
        self.transition = np.zeros((self.n_tags, self.n_tags), dtype=np.float64)
        self.start = np.zeros(self.n_tags, dtype=np.float64)
        self.end = np.zeros(self.n_tags, dtype=np.float64)
        self.transition_mask, self.start_mask, self.end_mask = allowed_transition_mask()

    def emission_scores(self, rows: Sequence[Mapping[int, float]]) -> np.ndarray:
        scores = np.repeat(self.emission_bias[None, :], len(rows), axis=0)
        for t, row in enumerate(rows):
            for index, value in row.items():
                scores[t] += self.emission[int(index)] * float(value)
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
        transition, start, end = self._masked_parameters()
        alpha = np.empty((length, self.n_tags), dtype=np.float64)
        alpha[0] = start + emissions[0]
        for t in range(1, length):
            alpha[t] = emissions[t] + _logsumexp(alpha[t - 1][:, None] + transition, axis=0)
        log_z = float(_logsumexp(alpha[-1] + end))

        beta = np.empty_like(alpha)
        beta[-1] = end
        for t in range(length - 2, -1, -1):
            beta[t] = _logsumexp(
                transition + emissions[t + 1][None, :] + beta[t + 1][None, :], axis=1
            )
        node = np.exp(alpha + beta - log_z)
        edge = np.empty((max(0, length - 1), self.n_tags, self.n_tags), dtype=np.float64)
        for t in range(1, length):
            edge[t - 1] = np.exp(
                alpha[t - 1][:, None]
                + transition
                + emissions[t][None, :]
                + beta[t][None, :]
                - log_z
            )
        start_marginal = np.exp(start + emissions[0] + beta[0] - log_z)
        end_marginal = np.exp(alpha[-1] + end - log_z)
        return log_z, alpha, beta, node, edge, np.stack((start_marginal, end_marginal))

    def nll_and_grad(
        self, rows: Sequence[Mapping[int, float]], gold: np.ndarray
    ) -> tuple[float, dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        emissions = self.emission_scores(rows)
        log_z, _alpha, _beta, node, edge, boundary = self._forward_backward(emissions)
        transition, start, end = self._masked_parameters()
        path_score = start[gold[0]] + emissions[0, gold[0]] + end[gold[-1]]
        if len(gold) > 1:
            path_score += sum(transition[left, right] for left, right in zip(gold[:-1], gold[1:]))
            path_score += float(sum(emissions[t, gold[t]] for t in range(1, len(gold))))
        nll = float(log_z - path_score)

        emission_grad: dict[int, np.ndarray] = {}
        bias_grad = node.sum(axis=0)
        for t, (row, gold_tag) in enumerate(zip(rows, gold)):
            bias_grad[gold_tag] -= 1.0
            delta = node[t].copy()
            delta[gold_tag] -= 1.0
            for index, value in row.items():
                grad = emission_grad.setdefault(int(index), np.zeros(self.n_tags, dtype=np.float64))
                grad += float(value) * delta

        transition_grad = edge.sum(axis=0) if len(edge) else np.zeros_like(self.transition)
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

    def viterbi(self, rows: Sequence[Mapping[int, float]], *, deletion_bias: float = 0.0) -> np.ndarray:
        emissions = self.emission_scores(rows).copy()
        # A single conservative deletion penalty is calibrated on validation.
        # It shifts both structure classes without invalidating BIOES paths.
        for index, tag in enumerate(TAGS):
            if tag != "O":
                emissions[:, index] -= float(deletion_bias)
        transition, start, end = self._masked_parameters()
        length = len(rows)
        score = np.empty((length, self.n_tags), dtype=np.float64)
        back = np.zeros((length, self.n_tags), dtype=np.int16)
        score[0] = start + emissions[0]
        for t in range(1, length):
            candidates = score[t - 1][:, None] + transition
            back[t] = np.argmax(candidates, axis=0)
            score[t] = emissions[t] + np.max(candidates, axis=0)
        tags = np.empty(length, dtype=np.int64)
        tags[-1] = int(np.argmax(score[-1] + end))
        for t in range(length - 1, 0, -1):
            tags[t - 1] = back[t, tags[t]]
        return tags

    def save(self, path: str | Path, metadata: Mapping[str, Any]) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            emission=self.emission,
            emission_bias=self.emission_bias,
            transition=self.transition,
            start=self.start,
            end=self.end,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["LinearChainCRF", dict[str, Any]]:
        values = np.load(path, allow_pickle=False)
        model = cls(values["emission"].shape[0])
        model.emission = values["emission"].astype(np.float64)
        model.emission_bias = values["emission_bias"].astype(np.float64)
        model.transition = values["transition"].astype(np.float64)
        model.start = values["start"].astype(np.float64)
        model.end = values["end"].astype(np.float64)
        metadata = json.loads(str(values["metadata"]))
        return model, metadata


def make_examples(documents: Sequence[GoldDocument], encoder: FeatureEncoder) -> list[SequenceExample]:
    """Build known-label segments; UNKNOWN is never coerced to a negative label."""
    examples: list[SequenceExample] = []
    for document in documents:
        encoded = encoder.encode_document(document)
        start = 0
        while start < len(document.lines):
            while start < len(document.lines) and document.lines[start].label == "UNKNOWN":
                start += 1
            end = start
            while end < len(document.lines) and document.lines[end].label != "UNKNOWN":
                end += 1
            if end > start:
                labels = [line.label for line in document.lines[start:end]]
                tags = document_tag_ids(type("Segment", (), {"lines": document.lines[start:end]})())
                examples.append(SequenceExample(
                    document=document,
                    features=encoded[start:end],
                    tags=tags,
                    line_indices=tuple(range(start, end)),
                ))
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
            loss, emission_grad, bias_grad, transition_grad, start_grad, end_grad = model.nll_and_grad(
                example.features, example.tags
            )
            epoch_loss += loss
            # Decoupled weight decay is deterministic and avoids densifying the
            # sparse emission gradient for every document.
            if l2:
                model.emission *= max(0.0, 1.0 - learning_rate * l2)
                model.transition *= max(0.0, 1.0 - learning_rate * l2)
            parts = list(emission_grad.values()) + [bias_grad, transition_grad, start_grad, end_grad]
            norm = _gradient_norm(parts)
            scale = min(1.0, float(gradient_clip) / max(norm, 1.0e-12))
            for index, grad in emission_grad.items():
                grad = grad * scale
                acc_emission[index] += grad * grad
                model.emission[index] -= learning_rate * grad / np.sqrt(acc_emission[index])
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


def _token_metrics(examples: Sequence[SequenceExample], model: LinearChainCRF, bias: float) -> dict[str, float]:
    predicted = correct = prose = false_prose = 0
    gold_by_class = {"BIB": 0, "TOC": 0}
    tp_by_class = {"BIB": 0, "TOC": 0}
    for example in examples:
        pred = bioes_to_classes([TAGS[x] for x in model.viterbi(example.features, deletion_bias=bias)])
        for line_index, guess in zip(example.line_indices, pred):
            line = example.document.lines[line_index]
            weight = line.token_count
            if guess != "O":
                predicted += weight
                correct += weight * int(line.label != "O")
            if line.is_running_prose:
                prose += weight
                false_prose += weight * int(guess != "O")
            if line.label != "O":
                gold_by_class[line.label] += weight
                tp_by_class[line.label] += weight * int(guess == line.label)
    return {
        "deletion_precision": correct / predicted if predicted else 1.0,
        "prose_contamination": false_prose / prose if prose else 0.0,
        "bib_recall": tp_by_class["BIB"] / gold_by_class["BIB"] if gold_by_class["BIB"] else 0.0,
        "toc_recall": tp_by_class["TOC"] / gold_by_class["TOC"] if gold_by_class["TOC"] else 0.0,
    }


def calibrate_deletion_bias(
    examples: Sequence[SequenceExample],
    model: LinearChainCRF,
    candidates: Sequence[float],
    *,
    minimum_precision: float,
    maximum_contamination: float,
) -> tuple[float, dict[str, float]]:
    rows = []
    for candidate in candidates:
        metrics = _token_metrics(examples, model, float(candidate))
        rows.append((float(candidate), metrics))
    eligible = [
        row for row in rows
        if row[1]["deletion_precision"] >= minimum_precision
        and row[1]["prose_contamination"] <= maximum_contamination
    ]
    if not eligible:
        raise RuntimeError("no validation deletion bias satisfies the safety floor")
    return max(eligible, key=lambda row: (row[1]["bib_recall"] + row[1]["toc_recall"], -row[0]))


def write_predictions(
    path: str | Path,
    documents: Sequence[GoldDocument],
    encoder: FeatureEncoder,
    model: LinearChainCRF,
    *,
    model_id: str,
    deletion_bias: float,
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for document in documents:
            features = encoder.encode_document(document)
            classes = ["O"] * len(document.lines)
            start = 0
            while start < len(document.lines):
                while start < len(document.lines) and document.lines[start].label == "UNKNOWN":
                    start += 1
                end = start
                while end < len(document.lines) and document.lines[end].label != "UNKNOWN":
                    end += 1
                if end > start:
                    tags = model.viterbi(features[start:end], deletion_bias=deletion_bias)
                    classes[start:end] = bioes_to_classes([TAGS[x] for x in tags])
                start = end + 1
            row = {
                "schema_version": "academic-structure-predictions-v1",
                "model_id": model_id,
                "document_id": document.document_id,
                "work_id": document.work_id,
                "source": document.source,
                "split": document.split,
                "lines": [
                    {"line_id": line.line_id, "abs_idx": line.abs_idx, "prediction": guess}
                    for line, guess in zip(document.lines, classes)
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _architecture(config: Mapping[str, Any], architecture_id: str) -> Mapping[str, Any]:
    for architecture in config["architecture_ladder"]:
        if architecture["id"] == architecture_id:
            if architecture["kind"] != "feature_crf":
                raise ValueError(f"{architecture_id!r} is not a feature CRF")
            return architecture
    raise ValueError(f"unknown architecture {architecture_id!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--architecture", required=True,
                        choices=("c1-feature-bioes-crf", "c2-char-ngram-feature-bioes-crf"))
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--test-predictions", help="write only after architecture/calibration are frozen")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    architecture = _architecture(config, args.architecture)
    seed = int(config["execution"]["seed"] if args.seed is None else args.seed)
    documents = read_gold(args.gold)
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    contract_receipt = validate_gold(
        documents,
        config["gold_contract"],
        split_manifest=split_manifest,
        for_promotion=True,
    )
    train_docs = [doc for doc in documents if doc.split == "train"]
    validation_docs = [doc for doc in documents if doc.split == "validation"]
    if not train_docs or not validation_docs:
        raise RuntimeError("feature CRF requires non-empty train and validation splits")
    if any(
        line.label != "UNKNOWN" and line.is_running_prose is None
        for document in validation_docs for line in document.lines
    ):
        raise RuntimeError("validation calibration requires running-prose adjudication on every known line")
    encoder = FeatureEncoder(
        char_hash_dim=int(architecture.get("char_hash_dim", 0)),
        char_ngram_min=int(architecture.get("char_ngram_min", 2)),
        char_ngram_max=int(architecture.get("char_ngram_max", 5)),
    )
    train_examples = make_examples(train_docs, encoder)
    validation_examples = make_examples(validation_docs, encoder)
    model = LinearChainCRF(encoder.n_features, seed=seed)
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
    deletion_bias, validation_metrics = calibrate_deletion_bias(
        validation_examples,
        model,
        calibration["deletion_bias_grid"],
        minimum_precision=float(calibration["minimum_deletion_token_precision"]),
        maximum_contamination=float(calibration["maximum_prose_token_contamination"]),
    )
    metadata = {
        "schema_version": "academic-structure-feature-crf-v1",
        "architecture_id": args.architecture,
        "config_sha256": sha256_file(args.config),
        "gold_sha256": sha256_file(args.gold),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "contract_inventory_sha256": contract_receipt["inventory_sha256"],
        "feature_encoder": encoder.metadata(),
        "tags": list(TAGS),
        "deletion_bias": deletion_bias,
        "validation_metrics": validation_metrics,
        "training_loss": history,
        "seed": seed,
        "test_used_for_training_or_calibration": False,
    }
    model.save(args.model_out, metadata)
    write_predictions(
        args.validation_predictions,
        validation_docs,
        encoder,
        model,
        model_id=args.architecture,
        deletion_bias=deletion_bias,
    )
    if args.test_predictions:
        write_predictions(
            args.test_predictions,
            [doc for doc in documents if doc.split == "test"],
            encoder,
            model,
            model_id=args.architecture,
            deletion_bias=deletion_bias,
        )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
