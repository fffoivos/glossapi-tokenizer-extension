#!/usr/bin/env python3
"""Gold-data and split contract for structural sequence-model promotion.

The historical STRUCT_2K labels were produced by language models and the large
artifact is absent.  They are useful silver training data if recovered, but this
validator never upgrades them to human gold.  Promotion requires a separately
locked, full-document, human-adjudicated test set.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "academic-structure-gold-v1"
LABELS = ("O", "BIB", "TOC", "UNKNOWN")
SPLITS = ("train", "validation", "test")
ANNOTATION_STATUSES = (
    "silver_llm",
    "human_single",
    "human_double",
    "human_adjudicated",
)


class ContractError(ValueError):
    """Raised when data could invalidate a model-comparison claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GoldLine:
    line_id: str
    abs_idx: int
    text: str
    label: str
    token_count: int
    is_running_prose: bool | None


@dataclass(frozen=True)
class GoldDocument:
    document_id: str
    work_id: str
    representation_id: str
    source: str
    split: str
    coverage: str
    n_physical_lines: int
    n_present_lines: int
    annotation_status: str
    annotator_ids: tuple[str, ...]
    adjudicator_id: str | None
    tokenizer_id: str
    tokenizer_revision: str
    lines: tuple[GoldLine, ...]

    @property
    def text_sha256(self) -> str:
        # Include physical coordinates so two differently-windowed views cannot
        # evade the exact-leak check merely by dropping blank lines.
        rows = [[line.abs_idx, line.text] for line in self.lines]
        return canonical_json_sha256(rows)

    @property
    def is_double_annotated(self) -> bool:
        return len(set(self.annotator_ids)) >= 2

    @property
    def labelled_line_fraction(self) -> float:
        """Present-line label coverage; missing labels are UNKNOWN, never O."""
        labelled = sum(line.label != "UNKNOWN" for line in self.lines)
        return labelled / self.n_present_lines

    @property
    def represented_line_fraction(self) -> float:
        return len(self.lines) / self.n_present_lines


def _parse_document(raw: Mapping[str, Any], row_number: int) -> GoldDocument:
    prefix = f"row {row_number}"
    _require(raw.get("schema_version") == SCHEMA_VERSION, f"{prefix}: unsupported schema_version")
    for field in ("document_id", "work_id", "representation_id", "source", "split", "coverage"):
        _require(isinstance(raw.get(field), str) and bool(raw[field].strip()), f"{prefix}: missing {field}")
    _require(raw["split"] in SPLITS, f"{prefix}: invalid split {raw['split']!r}")
    _require(isinstance(raw.get("n_physical_lines"), int) and raw["n_physical_lines"] > 0,
             f"{prefix}: n_physical_lines must be positive")
    _require(isinstance(raw.get("n_present_lines"), int) and raw["n_present_lines"] > 0,
             f"{prefix}: n_present_lines must be positive")
    _require(raw["n_present_lines"] <= raw["n_physical_lines"],
             f"{prefix}: present lines exceed physical lines")

    annotation = raw.get("annotation")
    _require(isinstance(annotation, Mapping), f"{prefix}: annotation provenance is required")
    status = annotation.get("status")
    _require(status in ANNOTATION_STATUSES, f"{prefix}: invalid annotation.status {status!r}")
    annotators = annotation.get("annotator_ids", [])
    _require(isinstance(annotators, list) and all(isinstance(x, str) and x for x in annotators),
             f"{prefix}: annotation.annotator_ids must be a string list")
    adjudicator = annotation.get("adjudicator_id")
    _require(adjudicator is None or (isinstance(adjudicator, str) and adjudicator),
             f"{prefix}: invalid adjudicator_id")
    if status == "human_adjudicated":
        _require(bool(adjudicator), f"{prefix}: adjudicated gold requires adjudicator_id")

    tokenizer = raw.get("tokenizer")
    _require(isinstance(tokenizer, Mapping), f"{prefix}: tokenizer provenance is required")
    tokenizer_id = tokenizer.get("id")
    tokenizer_revision = tokenizer.get("revision")
    _require(isinstance(tokenizer_id, str) and tokenizer_id, f"{prefix}: tokenizer.id is required")
    _require(isinstance(tokenizer_revision, str) and tokenizer_revision,
             f"{prefix}: tokenizer.revision is required")

    raw_lines = raw.get("lines")
    _require(isinstance(raw_lines, list) and raw_lines, f"{prefix}: non-empty lines are required")
    lines: list[GoldLine] = []
    previous_abs = -1
    line_ids: set[str] = set()
    for line_number, line in enumerate(raw_lines, 1):
        lp = f"{prefix}, line {line_number}"
        _require(isinstance(line, Mapping), f"{lp}: line must be an object")
        line_id = str(line.get("line_id", ""))
        _require(bool(line_id), f"{lp}: line_id is required")
        _require(line_id not in line_ids, f"{lp}: duplicate line_id {line_id!r}")
        line_ids.add(line_id)
        abs_idx = line.get("abs_idx")
        _require(isinstance(abs_idx, int) and abs_idx >= 0, f"{lp}: invalid abs_idx")
        _require(abs_idx > previous_abs, f"{lp}: abs_idx must be strictly increasing")
        previous_abs = abs_idx
        text = line.get("text")
        _require(isinstance(text, str), f"{lp}: text must be a string")
        _require(bool(text.strip()), f"{lp}: blank physical lines must be omitted")
        label = line.get("label")
        _require(label in LABELS, f"{lp}: invalid label {label!r}")
        token_count = line.get("token_count")
        _require(isinstance(token_count, int) and token_count >= 0, f"{lp}: invalid token_count")
        is_running_prose = line.get("is_running_prose")
        _require(is_running_prose is None or isinstance(is_running_prose, bool),
                 f"{lp}: is_running_prose must be boolean or null")
        _require(not is_running_prose or label == "O",
                 f"{lp}: only retained O lines may be running prose")
        _require(label != "UNKNOWN" or is_running_prose is None,
                 f"{lp}: UNKNOWN lines cannot assert running-prose status")
        lines.append(GoldLine(line_id, abs_idx, text, label, token_count, is_running_prose))
    _require(lines[-1].abs_idx < raw["n_physical_lines"],
             f"{prefix}: n_physical_lines does not cover the last line")

    return GoldDocument(
        document_id=raw["document_id"],
        work_id=raw["work_id"],
        representation_id=raw["representation_id"],
        source=raw["source"],
        split=raw["split"],
        coverage=raw["coverage"],
        n_physical_lines=raw["n_physical_lines"],
        n_present_lines=raw["n_present_lines"],
        annotation_status=status,
        annotator_ids=tuple(annotators),
        adjudicator_id=adjudicator,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        lines=tuple(lines),
    )


def read_gold(path: str | Path) -> list[GoldDocument]:
    documents: list[GoldDocument] = []
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if line.strip():
                raw = json.loads(line)
                _require(isinstance(raw, Mapping), f"row {row_number}: expected a JSON object")
                documents.append(_parse_document(raw, row_number))
    _require(bool(documents), "gold JSONL is empty")
    return documents


def _load_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == "academic-structure-split-v1",
             "unsupported split manifest")
    _require(isinstance(manifest.get("assignments"), Mapping), "split manifest lacks assignments")
    return manifest


def validate_gold(
    documents: Sequence[GoldDocument],
    policy: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
    for_promotion: bool = False,
) -> dict[str, Any]:
    """Validate all leakage/adjudication invariants and return a receipt.

    ``for_promotion=False`` still enforces identity and split safety but permits a
    small development corpus.  It never permits a silver record in the test
    split; the relaxed mode only waives sample-size/source-count thresholds.
    """
    _require(bool(documents), "no documents")
    ids: set[str] = set()
    representations: set[str] = set()
    work_splits: dict[str, str] = {}
    exact_hash_splits: dict[str, str] = {}
    test_by_source: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    status_counts: collections.Counter[str] = collections.Counter()
    double_test = 0
    tokenizer_pairs: set[tuple[str, str]] = set()
    coverage_counts: collections.Counter[str] = collections.Counter()
    represented_lines = labelled_lines = present_lines = 0

    assignments = split_manifest.get("assignments", {}) if split_manifest else {}
    manifest_inventory = sorted((doc.document_id, doc.work_id, doc.source) for doc in documents)
    if split_manifest is not None:
        _require(split_manifest.get("schema_version") == "academic-structure-split-v1",
                 "unsupported split manifest")
        _require(isinstance(assignments, Mapping), "split manifest lacks assignments")
        _require(
            split_manifest.get("inventory_sha256") == canonical_json_sha256(manifest_inventory),
            "locked split manifest inventory does not match document/work/source identities",
        )
    for doc in documents:
        _require(len(doc.lines) == doc.n_present_lines,
                 f"document {doc.document_id!r} omits present lines; encode their labels as UNKNOWN")
        _require(doc.document_id not in ids, f"duplicate document_id {doc.document_id!r}")
        ids.add(doc.document_id)
        _require(doc.representation_id not in representations,
                 f"duplicate representation_id {doc.representation_id!r}")
        representations.add(doc.representation_id)
        previous = work_splits.setdefault(doc.work_id, doc.split)
        _require(previous == doc.split,
                 f"work leakage: {doc.work_id!r} occurs in {previous!r} and {doc.split!r}")
        exact_previous = exact_hash_splits.setdefault(doc.text_sha256, doc.split)
        _require(exact_previous == doc.split,
                 f"exact-text leakage: document {doc.document_id!r} crosses splits")
        if split_manifest is not None:
            _require(doc.document_id in assignments,
                     f"document {doc.document_id!r} is absent from locked split manifest")
            _require(assignments[doc.document_id] == doc.split,
                     f"document {doc.document_id!r} disagrees with locked split manifest")
        split_counts[doc.split] += 1
        status_counts[doc.annotation_status] += 1
        coverage_counts[doc.coverage] += 1
        represented_lines += len(doc.lines)
        labelled_lines += sum(line.label != "UNKNOWN" for line in doc.lines)
        present_lines += doc.n_present_lines
        tokenizer_pairs.add((doc.tokenizer_id, doc.tokenizer_revision))
        if doc.split == "test":
            _require(doc.annotation_status == policy["test_annotation_status"],
                     f"test document {doc.document_id!r} is {doc.annotation_status!r}, not adjudicated gold")
            _require(doc.coverage == policy["test_coverage"],
                     f"test document {doc.document_id!r} does not have full-document coverage")
            _require(all(line.label != "UNKNOWN" for line in doc.lines),
                     f"test document {doc.document_id!r} contains UNKNOWN labels")
            _require(all(line.is_running_prose is not None for line in doc.lines),
                     f"test document {doc.document_id!r} lacks running-prose adjudication")
            _require(all(line.token_count >= 0 for line in doc.lines),
                     f"test document {doc.document_id!r} lacks token counts")
            test_by_source[doc.source] += 1
            double_test += int(doc.is_double_annotated)

    # Exact token comparisons require a single pinned tokenizer build.
    _require(len(tokenizer_pairs) == 1, f"mixed tokenizer provenance: {sorted(tokenizer_pairs)!r}")
    if policy.get("require_split_manifest") and for_promotion:
        _require(split_manifest is not None, "promotion requires a locked split manifest")
        _require(set(assignments) == ids,
                 "locked split manifest and gold JSONL must contain exactly the same documents")

    if for_promotion:
        required_sources = list(policy["required_sources"])
        total_test = split_counts["test"]
        _require(total_test >= int(policy["minimum_test_documents"]),
                 f"test set has {total_test} documents; need {policy['minimum_test_documents']}")
        for source in required_sources:
            _require(test_by_source[source] >= int(policy["minimum_test_documents_per_source"]),
                     f"test source {source!r} has {test_by_source[source]} documents; "
                     f"need {policy['minimum_test_documents_per_source']}")
        double_fraction = double_test / total_test if total_test else 0.0
        _require(double_fraction >= float(policy["minimum_double_annotated_fraction"]),
                 f"double-annotation fraction {double_fraction:.3f} is below contract")
    else:
        double_fraction = double_test / split_counts["test"] if split_counts["test"] else 0.0

    inventory = sorted(
        (doc.document_id, doc.work_id, doc.representation_id, doc.source, doc.split, doc.text_sha256)
        for doc in documents
    )
    return {
        "schema_version": "academic-structure-contract-receipt-v1",
        "status": "pass",
        "promotion_contract_enforced": for_promotion,
        "document_count": len(documents),
        "work_count": len(work_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "test_by_source": dict(sorted(test_by_source.items())),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "represented_present_line_fraction": represented_lines / present_lines,
        "known_label_present_line_fraction": labelled_lines / present_lines,
        "double_annotated_test_fraction": double_fraction,
        "tokenizer": {"id": next(iter(tokenizer_pairs))[0], "revision": next(iter(tokenizer_pairs))[1]},
        "inventory_sha256": canonical_json_sha256(inventory),
    }


def stable_split(work_id: str, seed: str, train: float, validation: float) -> str:
    """Assign a work group once, without looking at labels or model output."""
    digest = hashlib.sha256(f"{seed}\0{work_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def build_split_manifest(documents: Sequence[GoldDocument], split_policy: Mapping[str, Any]) -> dict[str, Any]:
    train = float(split_policy["train_fraction"])
    validation = float(split_policy["validation_fraction"])
    test = float(split_policy["test_fraction"])
    _require(math.isclose(train + validation + test, 1.0, abs_tol=1e-9),
             "split fractions must sum to one")
    seed = str(split_policy["seed"])
    # Balance work groups inside source strata. A work represented in multiple
    # sources remains indivisible and receives a composite stratum.
    sources_by_work: dict[str, set[str]] = collections.defaultdict(set)
    for doc in documents:
        sources_by_work[doc.work_id].add(doc.source)
    strata: dict[str, list[str]] = collections.defaultdict(list)
    for work_id, sources in sources_by_work.items():
        strata["+".join(sorted(sources))].append(work_id)
    work_assignments: dict[str, str] = {}
    for stratum, work_ids in sorted(strata.items()):
        ordered = sorted(
            work_ids,
            key=lambda work_id: hashlib.sha256(
                f"{seed}\0{stratum}\0{work_id}".encode("utf-8")
            ).digest(),
        )
        n = len(ordered)
        fractions = {"train": train, "validation": validation, "test": test}
        counts = {name: int(math.floor(n * fraction)) for name, fraction in fractions.items()}
        # Largest-remainder allocation keeps every stratum as close as possible
        # to the requested fractions without consulting labels.
        remainders = sorted(
            ((n * fractions[name] - counts[name], name) for name in fractions),
            key=lambda item: (-item[0], item[1]),
        )
        for _remainder, name in remainders[:n - sum(counts.values())]:
            counts[name] += 1
        boundaries = (counts["train"], counts["train"] + counts["validation"])
        for index, work_id in enumerate(ordered):
            work_assignments[work_id] = (
                "train" if index < boundaries[0]
                else "validation" if index < boundaries[1]
                else "test"
            )
    assignments = {doc.document_id: work_assignments[doc.work_id] for doc in documents}
    inventory = sorted((doc.document_id, doc.work_id, doc.source) for doc in documents)
    return {
        "schema_version": "academic-structure-split-v1",
        "seed": seed,
        "algorithm": "source-stratified-sha256-work-order-v1",
        "fractions": {"train": train, "validation": validation, "test": test},
        "inventory_sha256": canonical_json_sha256(inventory),
        "assignments": dict(sorted(assignments.items())),
    }


def _load_config(path: str | Path) -> Mapping[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(config.get("schema_version") == "academic-structure-sequence-eval-v1",
             "unsupported sequence evaluation config")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate gold and emit a receipt")
    validate.add_argument("--gold", required=True)
    validate.add_argument("--config", required=True)
    validate.add_argument("--split-manifest")
    validate.add_argument("--promotion", action="store_true")
    validate.add_argument("--output")
    split = sub.add_parser("make-split", help="make a deterministic work-group split manifest")
    split.add_argument("--gold", required=True, help="inventory/gold JSONL; labels are ignored")
    split.add_argument("--config", required=True)
    split.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    documents = read_gold(args.gold)
    if args.command == "validate":
        manifest = _load_manifest(args.split_manifest) if args.split_manifest else None
        receipt = validate_gold(
            documents,
            config["gold_contract"],
            split_manifest=manifest,
            for_promotion=args.promotion,
        )
        receipt["gold_sha256"] = sha256_file(args.gold)
        receipt["config_sha256"] = sha256_file(args.config)
        if args.split_manifest:
            receipt["split_manifest_sha256"] = sha256_file(args.split_manifest)
        payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0

    manifest = build_split_manifest(documents, config["split"])
    Path(args.output).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
