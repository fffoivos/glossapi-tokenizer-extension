#!/usr/bin/env python3
"""Deterministic features and BIOES conversion for CPU sequence baselines."""
from __future__ import annotations

import collections
import hashlib
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# Reuse the exact hand-feature implementation already mirrored by Rust.  This
# package is nested below eval/, whose historical scripts are not a Python
# package, so add that directory explicitly without mutating their imports.
EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
import line_lr as existing_features  # noqa: E402
import span_signals  # noqa: E402

CLASSES = ("O", "BIB", "TOC")
TAGS = (
    "O",
    "B-BIB", "I-BIB", "E-BIB", "S-BIB",
    "B-TOC", "I-TOC", "E-TOC", "S-TOC",
)
TAG_TO_ID = {tag: index for index, tag in enumerate(TAGS)}
ID_TO_TAG = {index: tag for tag, index in TAG_TO_ID.items()}
HAND_FEATURES = tuple(existing_features.FEATS) + tuple(span_signals.TOC_KEYS) + (
    "bias",
    "pos2",
    "front_decay",
    "back_decay",
)


def classes_to_bioes(labels: Sequence[str]) -> list[str]:
    """Convert O/BIB/TOC line labels into valid class-specific BIOES tags."""
    out = ["O"] * len(labels)
    i = 0
    while i < len(labels):
        label = labels[i]
        if label == "O":
            i += 1
            continue
        if label == "UNKNOWN":
            raise ValueError("UNKNOWN labels must be masked or split before BIOES conversion")
        if label not in ("BIB", "TOC"):
            raise ValueError(f"unsupported class label {label!r}")
        j = i
        while j + 1 < len(labels) and labels[j + 1] == label:
            j += 1
        if i == j:
            out[i] = f"S-{label}"
        else:
            out[i] = f"B-{label}"
            for k in range(i + 1, j):
                out[k] = f"I-{label}"
            out[j] = f"E-{label}"
        i = j + 1
    return out


def bioes_to_classes(tags: Sequence[str]) -> list[str]:
    return ["O" if tag == "O" else tag.split("-", 1)[1] for tag in tags]


def allowed_transition_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return legal BIOES transition, start and end masks."""
    n = len(TAGS)
    trans = np.zeros((n, n), dtype=bool)
    start = np.zeros(n, dtype=bool)
    end = np.zeros(n, dtype=bool)

    def can_start(tag: str) -> bool:
        return tag == "O" or tag.startswith("B-") or tag.startswith("S-")

    def can_end(tag: str) -> bool:
        return tag == "O" or tag.startswith("E-") or tag.startswith("S-")

    def can_follow(left: str, right: str) -> bool:
        if left == "O" or left.startswith("E-") or left.startswith("S-"):
            return right == "O" or right.startswith("B-") or right.startswith("S-")
        left_prefix, left_class = left.split("-", 1)
        if left_prefix not in ("B", "I"):
            return False
        return right in (f"I-{left_class}", f"E-{left_class}")

    for i, left in enumerate(TAGS):
        start[i] = can_start(left)
        end[i] = can_end(left)
        for j, right in enumerate(TAGS):
            trans[i, j] = can_follow(left, right)
    return trans, start, end


def validate_bioes(tags: Sequence[str]) -> None:
    if not tags:
        raise ValueError("an empty tag sequence is invalid")
    trans, start, end = allowed_transition_mask()
    ids = [TAG_TO_ID[tag] for tag in tags]
    if not start[ids[0]]:
        raise ValueError(f"invalid BIOES start: {tags[0]}")
    for left, right in zip(ids, ids[1:]):
        if not trans[left, right]:
            raise ValueError(f"invalid BIOES transition: {TAGS[left]} -> {TAGS[right]}")
    if not end[ids[-1]]:
        raise ValueError(f"invalid BIOES end: {tags[-1]}")


def _stable_bucket(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=b"tocbibv1").digest()
    return int.from_bytes(digest, "big") % dimension


class FeatureEncoder:
    """Existing Rust-mirrored signals plus optional hashed Unicode char n-grams.

    Strings are deliberately not normalized: NFC/NFD and OCR corruption remain
    observable, matching the corpus no-normalization policy.  Hashing is stable
    across Python processes and machines.
    """

    def __init__(self, *, char_hash_dim: int = 0, char_ngram_min: int = 2, char_ngram_max: int = 5):
        if char_hash_dim < 0:
            raise ValueError("char_hash_dim must be non-negative")
        if not (1 <= char_ngram_min <= char_ngram_max):
            raise ValueError("invalid character n-gram range")
        self.char_hash_dim = int(char_hash_dim)
        self.char_ngram_min = int(char_ngram_min)
        self.char_ngram_max = int(char_ngram_max)
        self.hand_dim = len(HAND_FEATURES)
        self.n_features = self.hand_dim + self.char_hash_dim

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "academic-structure-features-v1",
            "hand_features": list(HAND_FEATURES),
            "char_hash": {
                "dimension": self.char_hash_dim,
                "minimum_n": self.char_ngram_min,
                "maximum_n": self.char_ngram_max,
                "hash": "blake2b-64-person=tocbibv1",
                "unicode_normalization": "none",
            },
        }

    def _char_features(self, text: str) -> dict[int, float]:
        if self.char_hash_dim == 0:
            return {}
        bounded = f"^{text}$"
        counts: collections.Counter[int] = collections.Counter()
        for n in range(self.char_ngram_min, self.char_ngram_max + 1):
            for i in range(max(0, len(bounded) - n + 1)):
                gram = bounded[i:i + n]
                counts[_stable_bucket(f"n{n}:{gram}", self.char_hash_dim)] += 1
        norm = math.sqrt(sum((1.0 + math.log(c)) ** 2 for c in counts.values())) or 1.0
        return {
            self.hand_dim + bucket: (1.0 + math.log(count)) / norm
            for bucket, count in counts.items()
        }

    def encode_document(self, document: object) -> list[dict[int, float]]:
        # GoldDocument is intentionally duck-typed to keep this module reusable
        # for private Clariden adapters without copying text into this repo.
        lines = list(getattr(document, "lines"))
        n_physical_lines = int(getattr(document, "n_physical_lines"))
        legacy = {
            "lines": [(line.abs_idx, line.text) for line in lines],
            "N": n_physical_lines,
        }
        base_rows = existing_features.doc_features(legacy)
        encoded: list[dict[int, float]] = []
        for line, base in zip(lines, base_rows):
            toc = span_signals.toc_signals(line.text)
            position = line.abs_idx / max(n_physical_lines, 1)
            values = dict(base)
            values.update(toc)
            values.update(
                bias=1.0,
                pos2=position * position,
                front_decay=math.exp(-6.0 * position),
                back_decay=math.exp(-6.0 * (1.0 - position)),
            )
            row = {
                index: float(values[name])
                for index, name in enumerate(HAND_FEATURES)
                if float(values[name]) != 0.0
            }
            row.update(self._char_features(line.text))
            encoded.append(row)
        return encoded


def document_tag_ids(document: object) -> np.ndarray:
    tags = classes_to_bioes([line.label for line in getattr(document, "lines")])
    validate_bioes(tags)
    return np.asarray([TAG_TO_ID[tag] for tag in tags], dtype=np.int64)


def deletion_class_from_tag_id(tag_id: int) -> str:
    tag = TAGS[int(tag_id)]
    return "O" if tag == "O" else tag.split("-", 1)[1]
