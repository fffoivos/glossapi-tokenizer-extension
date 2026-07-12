#!/usr/bin/env python3
"""Python↔Rust parity on a receipt-bound joint LLM-silver corpus.

The legacy `units/STRUCT_2K_gold.jsonl` path remains supported for historical
replay. The operational path consumes `struct2k.LLM_silver.jsonl` emitted by
the locked importer and checks its newly derived validation partition. Runtime
parity is an implementation-equivalence check, not held-out quality evidence,
and cannot itself authorize production.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import decode_spans as decode
import line_lr
import span_signals
import struct_lines


HERE = Path(__file__).resolve().parent
DEFAULT_BIN = (
    HERE.parent / "reference_detector" / "target" / "debug" / "reference_detect"
)
LABELS = {"O": 0, "BIB": 1, "TOC": 2}
SNAPSHOT_METHOD = "private_job_local_o_nofollow_copy_rehash_before_publish"


@dataclass(frozen=True)
class InputSnapshot:
    original: Path
    snapshot: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _hash_regular_nofollow(path: Path, *, label: str) -> str:
    """Hash one stable regular-file inode without following a final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"{label} must be a present non-symlink regular file"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_state = os.stat(path, follow_symlinks=False)
        stable = (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                path_state.st_dev,
                path_state.st_ino,
                path_state.st_size,
                path_state.st_mtime_ns,
                path_state.st_ctime_ns,
            )
        )
        if not stable:
            raise ValueError(f"{label} changed while it was being read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _snapshot_input(
    original: Path,
    destination: Path,
    *,
    label: str,
    executable: bool = False,
) -> InputSnapshot:
    if original.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    expected_sha256 = _hash_regular_nofollow(original, label=label)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(original, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        with (
            os.fdopen(os.dup(descriptor), "rb") as source,
            destination.open("xb") as target,
        ):
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        after = os.fstat(descriptor)
        path_state = os.stat(original, follow_symlinks=False)
        stable = (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                path_state.st_dev,
                path_state.st_ino,
                path_state.st_size,
                path_state.st_mtime_ns,
                path_state.st_ctime_ns,
            )
        )
        if not stable:
            raise ValueError(f"{label} changed while it was being snapshotted")
    finally:
        os.close(descriptor)
    os.chmod(destination, 0o500 if executable else 0o400)
    snapshot_sha256 = _hash_regular_nofollow(destination, label=f"{label} snapshot")
    if snapshot_sha256 != expected_sha256:
        raise ValueError(f"{label} snapshot differs from the verified input")
    return InputSnapshot(
        original=original.absolute(),
        snapshot=destination.absolute(),
        sha256=snapshot_sha256,
    )


def _verify_inputs_unchanged(snapshots: dict[str, InputSnapshot]) -> None:
    for label, item in snapshots.items():
        if (
            _hash_regular_nofollow(item.snapshot, label=f"{label} snapshot")
            != item.sha256
            or _hash_regular_nofollow(item.original, label=label) != item.sha256
        ):
            raise ValueError(f"{label} changed after the parity input snapshot")


def _write_receipt_atomic_no_clobber(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite parity receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite parity receipt: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def load_materialized(path: Path) -> SimpleNamespace:
    """Load the modern imported STRUCT-2K schema into the historical parity view."""
    docs = {}
    with path.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "academic-structure-gold-v1":
                raise ValueError(f"{path}:{row_number}: unsupported modern silver row")
            document_id = row.get("document_id")
            raw_lines = row.get("lines")
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id in docs
                or not isinstance(raw_lines, list)
                or not raw_lines
            ):
                raise ValueError(f"{path}:{row_number}: malformed/duplicate document")
            lines = []
            labels = []
            for raw in raw_lines:
                if (
                    not isinstance(raw, dict)
                    or not isinstance(raw.get("abs_idx"), int)
                    or not isinstance(raw.get("text"), str)
                    or raw.get("label") not in LABELS
                ):
                    raise ValueError(f"{path}:{row_number}: malformed line")
                lines.append((raw["abs_idx"], raw["text"]))
                labels.append(LABELS[raw["label"]])
            coordinates = [coordinate for coordinate, _text in lines]
            n_total = row.get("n_physical_lines")
            if (
                coordinates != sorted(set(coordinates))
                or not isinstance(n_total, int)
                or n_total <= coordinates[-1]
                or row.get("split") not in {"train", "validation"}
                or row.get("historical_split") != "train"
            ):
                raise ValueError(
                    f"{path}:{row_number}: split/coordinate boundary drift"
                )
            docs[document_id] = {
                "source": row.get("source"),
                "split": row["split"],
                "lines": lines,
                "labels": labels,
                "N": n_total,
            }
    if not docs:
        raise ValueError(f"{path}: no materialized documents")
    return SimpleNamespace(docs=docs)


def validate_source_receipt(
    receipt_path: Path, corpus_path: Path, split_path: Path
) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    exclusion = receipt.get("historical_partition_exclusion")
    assignments = split.get("assignments")
    split_counts = receipt.get("split_counts")
    observed_counts = (
        {
            partition: sum(value == partition for value in assignments.values())
            for partition in ("train", "validation")
        }
        if isinstance(assignments, dict)
        else None
    )
    _positive_integer(
        split_counts.get("validation") if isinstance(split_counts, dict) else None,
        label="source split_counts.validation",
    )
    if (
        receipt.get("schema_version") != "academic-structure-silver-contract-receipt-v1"
        or receipt.get("status") != "pass"
        or receipt.get("evidence_tier") != "LLM_silver"
        or receipt.get("production_eligible") is not False
        or receipt.get("silver_sha256") != sha256_file(corpus_path)
        or receipt.get("split_manifest_sha256") != sha256_file(split_path)
        or split.get("schema_version") != "academic-structure-split-v1"
        or split.get("inventory_sha256") != receipt.get("inventory_sha256")
        or not isinstance(assignments, dict)
        or len(assignments) != receipt.get("document_count")
        or set(assignments.values()) != {"train", "validation"}
        or split_counts != observed_counts
        or not isinstance(exclusion, dict)
        or exclusion.get("eligible_historical_train_documents")
        != receipt.get("document_count")
        or exclusion.get("historical_test_documents_excluded") != 608
        or exclusion.get("historical_test_rows_emitted") != 0
        or exclusion.get("historical_test_predictions_permitted") is not False
        or receipt.get("task_scope_counts", {}).get("bibliography_toc_windows")
        != receipt.get("document_count")
    ):
        raise ValueError(
            "modern joint source receipt does not prove 608-document exclusion"
        )
    materialized = receipt.get("materialized_artifacts", {})
    if (
        materialized.get("silver_filename") != corpus_path.name
        or materialized.get("split_manifest_filename") != split_path.name
        or len(
            {
                receipt_path.resolve().parent,
                corpus_path.resolve().parent,
                split_path.resolve().parent,
            }
        )
        != 1
    ):
        raise ValueError("modern joint source artifact path differs from its receipt")
    return receipt


def frozen_probabilities(data, model_path: Path, *, toc: bool) -> dict[str, np.ndarray]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    mu = np.asarray(model["mu"], dtype=np.float64)
    sd = np.asarray(model["sd"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    bias = _finite_number(model["bias"], label=f"{model_path} bias")
    if (
        not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(sd))
        or not np.all(sd > 0.0)
        or not np.all(np.isfinite(weight))
    ):
        raise ValueError(
            f"{model_path}: model arrays must be finite with positive scales"
        )
    result: dict[str, np.ndarray] = {}
    for doc_id, document in data.docs.items():
        rows = []
        for index, features in enumerate(line_lr.doc_features(document)):
            values = [features[name] for name in line_lr.FEATS]
            if toc:
                signals = span_signals.toc_signals(document["lines"][index][1])
                values.extend(signals[name] for name in span_signals.TOC_KEYS)
            rows.append(values)
        matrix = np.asarray(rows, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{model_path}: extracted features contain non-finite values"
            )
        scores = 1.0 / (1.0 + np.exp(-(((matrix - mu) / sd) @ weight + bias)))
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"{model_path}: Python probabilities contain non-finite values"
            )
        if toc:
            cut = min(300, int(0.30 * document["N"]))
            scores *= np.asarray(
                [line < cut for line, _text in document["lines"]], dtype=bool
            )
        result[doc_id] = scores
    return result


def run_rust_scores(
    binary: Path, docs: list[str], data, mode: str, work: Path
) -> list[list[float]]:
    input_path = work / f"{mode}.input.jsonl"
    output_path = work / f"{mode}.output.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for doc_id in docs:
            document = data.docs[doc_id]
            handle.write(
                json.dumps(
                    {
                        "present": [text for _line, text in document["lines"]],
                        "abs_idx": [line for line, _text in document["lines"]],
                        "n_total": document["N"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    subprocess.run(
        [
            str(binary),
            "--mode",
            mode,
            "--input",
            str(input_path),
            "--out-spans",
            str(work / f"{mode}.unused-spans.jsonl"),
            "--out-counters",
            str(output_path),
        ],
        check=True,
    )
    rows = [
        json.loads(line)["p"]
        for line in output_path.open(encoding="utf-8")
        if line.strip()
    ]
    if len(rows) != len(docs):
        raise AssertionError(
            f"{mode}: output rows {len(rows)} != input docs {len(docs)}"
        )
    return rows


def run_rust_spans(
    binary: Path, docs: list[str], data, mode: str, work: Path
) -> dict[str, list[tuple[int, int]]]:
    input_path = work / f"{mode}.docs.jsonl"
    spans_path = work / f"{mode}.spans.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for doc_id in docs:
            document = data.docs[doc_id]
            lines = [""] * document["N"]
            for line, text in document["lines"]:
                lines[line] = text
            handle.write(
                json.dumps({"id": doc_id, "text": "\n".join(lines)}, ensure_ascii=False)
                + "\n"
            )
    subprocess.run(
        [
            str(binary),
            "--mode",
            mode,
            "--input",
            str(input_path),
            "--text-field",
            "text",
            "--id-field",
            "id",
            "--out-spans",
            str(spans_path),
            "--out-counters",
            str(work / f"{mode}.counters.jsonl"),
        ],
        check=True,
    )
    result: dict[str, list[tuple[int, int]]] = {}
    expected_kind = "toc_span" if mode == "toc-spans" else "bib_span"
    for line in spans_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["kind"] == expected_kind:
            result.setdefault(row["doc_id"], []).append(
                (row["line_start"], row["line_end"])
            )
    counter_rows = [
        json.loads(line)
        for line in (work / f"{mode}.counters.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    if len(counter_rows) != len(docs) or any("error" in row for row in counter_rows):
        raise AssertionError(f"{mode}: incomplete/error counter coverage")
    if [str(row.get("doc_id")) for row in counter_rows] != docs:
        raise AssertionError(f"{mode}: counter document order/coverage mismatch")
    return result


def compare_head(
    binary: Path,
    docs: list[str],
    data,
    *,
    head: str,
    work: Path,
    tolerance: float,
    model_path: Path,
    smoother_path: Path,
) -> dict:
    toc = head == "toc"
    python_probabilities = frozen_probabilities(data, model_path, toc=toc)
    score_mode = "toc-score-lines" if toc else "score-lines"
    rust_probabilities = run_rust_scores(binary, docs, data, score_mode, work)

    max_difference = 0.0
    worst = None
    for doc_id, rust_scores in zip(docs, rust_probabilities):
        python_scores = python_probabilities[doc_id]
        if len(python_scores) != len(rust_scores):
            raise AssertionError((doc_id, len(python_scores), len(rust_scores)))
        for index, (python_score, rust_score) in enumerate(
            zip(python_scores, rust_scores)
        ):
            python_value = _finite_nonnegative(
                python_score, label=f"{head} Python probability"
            )
            rust_value = _finite_nonnegative(
                rust_score, label=f"{head} Rust probability"
            )
            if python_value > 1.0 or rust_value > 1.0:
                raise ValueError(f"{head} probabilities must be in [0, 1]")
            difference = abs(python_value - rust_value)
            if difference > max_difference:
                max_difference = difference
                worst = (doc_id, index, python_value, rust_value)
    if max_difference > tolerance:
        raise AssertionError(
            f"{head}: max probability difference {max_difference:.6g}; worst={worst}"
        )

    params = json.loads(smoother_path.read_text(encoding="utf-8"))[head]
    params = {key: params[key] for key in ("theta_hi", "theta_lo", "gap", "lmin")}
    span_mode = "toc-spans" if toc else "bib-spans"
    rust_spans = run_rust_spans(binary, docs, data, span_mode, work)
    mismatches = []
    for doc_id in docs:
        python_spans = sorted(
            decode.decode_doc(data.docs[doc_id], python_probabilities[doc_id], params)
        )
        actual = sorted(rust_spans.get(doc_id, []))
        if python_spans != actual:
            mismatches.append((doc_id, python_spans, actual))
    if mismatches:
        raise AssertionError(
            f"{head}: {len(mismatches)} span mismatches; first={mismatches[:3]}"
        )
    print(f"{head}: {len(docs)} docs; max |Δp|={max_difference:.6g}; spans identical")
    return {
        "documents": len(docs),
        "max_probability_difference": max_difference,
        "span_mismatches": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="modern imported struct2k.LLM_silver.jsonl; omit for legacy replay",
    )
    parser.add_argument(
        "--source-receipt",
        type=Path,
        help="required contract receipt for --corpus",
    )
    parser.add_argument(
        "--source-split-manifest",
        type=Path,
        help="required imported split manifest for --corpus",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test", "all"),
        help="default: validation for modern import, test for legacy replay",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=0,
        help="0 means every document in the partition",
    )
    parser.add_argument("--expected-documents", type=int)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-corpus-sha256")
    expected.add_argument("--expected-gold-sha256", help="deprecated legacy spelling")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError(f"refusing to overwrite parity receipt: {args.receipt}")
    tolerance = _finite_nonnegative(args.tolerance, label="parity tolerance")
    if args.documents < 0:
        raise ValueError("--documents must be zero or a positive integer")
    if args.expected_documents is not None:
        _positive_integer(args.expected_documents, label="--expected-documents")

    modern_flags = (
        args.corpus is not None,
        args.source_receipt is not None,
        args.source_split_manifest is not None,
    )
    if any(modern_flags) and not all(modern_flags):
        raise SystemExit(
            "--corpus, --source-receipt and --source-split-manifest must be supplied together"
        )
    modern = all(modern_flags)
    corpus_path = args.corpus if modern else Path(struct_lines.GOLD)
    expected_corpus_sha256 = args.expected_corpus_sha256 or args.expected_gold_sha256
    if (
        not isinstance(expected_corpus_sha256, str)
        or len(expected_corpus_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_corpus_sha256
        )
    ):
        raise ValueError("expected corpus SHA-256 must be lowercase 64-hex")

    original_models = {
        "bib": HERE / "span_line_lr_struct_model.json",
        "toc": HERE / "toc_line_lr_model.json",
        "smoother": HERE / "struct_smooth_params.json",
    }
    with tempfile.TemporaryDirectory(prefix="rust-parity-struct-") as temporary_name:
        temporary = Path(temporary_name)
        snapshots = {
            "corpus": _snapshot_input(
                corpus_path,
                temporary / "source" / corpus_path.name,
                label="parity corpus",
            ),
            "detector binary": _snapshot_input(
                args.binary,
                temporary / "bin" / "reference_detect",
                label="detector binary",
                executable=True,
            ),
            "bibliography model": _snapshot_input(
                original_models["bib"],
                temporary / "models" / original_models["bib"].name,
                label="bibliography model",
            ),
            "toc model": _snapshot_input(
                original_models["toc"],
                temporary / "models" / original_models["toc"].name,
                label="toc model",
            ),
            "smoother": _snapshot_input(
                original_models["smoother"],
                temporary / "models" / original_models["smoother"].name,
                label="smoother",
            ),
        }
        if modern:
            snapshots["source receipt"] = _snapshot_input(
                args.source_receipt,
                temporary / "source" / args.source_receipt.name,
                label="source receipt",
            )
            snapshots["source split manifest"] = _snapshot_input(
                args.source_split_manifest,
                temporary / "source" / args.source_split_manifest.name,
                label="source split manifest",
            )
        actual_corpus_sha256 = snapshots["corpus"].sha256
        if actual_corpus_sha256 != expected_corpus_sha256:
            raise SystemExit(
                "structural LLM-silver corpus hash mismatch: "
                f"expected {expected_corpus_sha256}, got {actual_corpus_sha256}"
            )

        if modern:
            source_receipt = validate_source_receipt(
                snapshots["source receipt"].snapshot,
                snapshots["corpus"].snapshot,
                snapshots["source split manifest"].snapshot,
            )
            data = load_materialized(snapshots["corpus"].snapshot)
            assignments = json.loads(
                snapshots["source split manifest"].snapshot.read_text(encoding="utf-8")
            )["assignments"]
            if set(assignments) != set(data.docs) or any(
                assignments[document_id] != data.docs[document_id]["split"]
                for document_id in data.docs
            ):
                raise ValueError(
                    "source split assignments differ from materialized document partitions"
                )
            partition = args.partition or "validation"
            if partition == "test":
                raise SystemExit(
                    "modern imported source physically contains no historical-test rows"
                )
            receipt_expected = (
                source_receipt.get("document_count")
                if partition == "all"
                else source_receipt.get("split_counts", {}).get(partition)
            )
            expected_documents = (
                args.expected_documents
                if args.expected_documents is not None
                else _positive_integer(
                    receipt_expected,
                    label=f"source receipt {partition} document count",
                )
            )
            semantics = (
                "derived_historical_train_validation_runtime_parity_not_quality_holdout"
            )
        else:
            original_gold = struct_lines.GOLD
            try:
                struct_lines.GOLD = str(snapshots["corpus"].snapshot)
                data = struct_lines.load()
            finally:
                struct_lines.GOLD = original_gold
            partition = args.partition or "test"
            expected_documents = (
                args.expected_documents if args.expected_documents is not None else 608
            )
            semantics = (
                "legacy_historical_test_runtime_parity_not_independent_quality_evidence"
            )

        expected_documents = _positive_integer(
            expected_documents, label="expected parity documents"
        )
        docs = [
            doc_id
            for doc_id, document in data.docs.items()
            if partition == "all" or document["split"] == partition
        ]
        if args.documents:
            docs = docs[: args.documents]
        if not docs:
            raise SystemExit(f"no {partition} parity documents found")
        if len(docs) != expected_documents:
            raise SystemExit(
                f"parity coverage {len(docs)} != required {expected_documents} for {partition}"
            )
        sources = collections.Counter(
            str(data.docs[doc_id].get("source")) for doc_id in docs
        )
        positive_docs = {
            "bib": sum(
                any(label == 1 for label in data.docs[doc_id]["labels"])
                for doc_id in docs
            ),
            "toc": sum(
                any(label == 2 for label in data.docs[doc_id]["labels"])
                for doc_id in docs
            ),
        }
        if not sources or any(value == 0 for value in positive_docs.values()):
            raise SystemExit(
                f"parity source/positive coverage is insufficient: sources={sources}, "
                f"positives={positive_docs}"
            )
        work = temporary / "work"
        work.mkdir(mode=0o700)
        heads = {
            "bib": compare_head(
                snapshots["detector binary"].snapshot,
                docs,
                data,
                head="bib",
                work=work,
                tolerance=tolerance,
                model_path=snapshots["bibliography model"].snapshot,
                smoother_path=snapshots["smoother"].snapshot,
            ),
            "toc": compare_head(
                snapshots["detector binary"].snapshot,
                docs,
                data,
                head="toc",
                work=work,
                tolerance=tolerance,
                model_path=snapshots["toc model"].snapshot,
                smoother_path=snapshots["smoother"].snapshot,
            ),
        }
        for head, result in heads.items():
            if _positive_integer(
                result.get("documents"), label=f"{head} parity documents"
            ) != len(docs):
                raise ValueError(f"{head} parity document coverage drift")
            mismatches = result.get("span_mismatches")
            if (
                isinstance(mismatches, bool)
                or not isinstance(mismatches, int)
                or mismatches != 0
            ):
                raise ValueError(f"{head} parity has invalid decoded-span mismatches")
            delta = _finite_nonnegative(
                result.get("max_probability_difference"),
                label=f"{head} maximum probability difference",
            )
            if delta > tolerance:
                raise ValueError(f"{head} probability delta exceeds parity tolerance")

        _verify_inputs_unchanged(snapshots)
        receipt = {
            "schema_version": "struct_rust_parity_receipt_v1",
            "status": "passed",
            "evidence_status": "LLM_silver",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_snapshot_method": SNAPSHOT_METHOD,
            "inputs_rehashed_before_publication": True,
            "binary": str(args.binary.absolute()),
            "binary_sha256": snapshots["detector binary"].sha256,
            "corpus": str(corpus_path.absolute()),
            "corpus_sha256": actual_corpus_sha256,
            "heldout_documents": len(docs),
            "evaluation_partition": partition,
            "partition_semantics": semantics,
            "historical_test_documents_loaded": 0 if modern else len(docs),
            "source_receipt_sha256": (
                snapshots["source receipt"].sha256 if modern else None
            ),
            "source_split_manifest_sha256": (
                snapshots["source split manifest"].sha256 if modern else None
            ),
            "source_document_counts": dict(sorted(sources.items())),
            "positive_document_counts": positive_docs,
            "tolerance": tolerance,
            "heads": heads,
            "model_sha256": {
                "bib": snapshots["bibliography model"].sha256,
                "toc": snapshots["toc model"].sha256,
                "smoother": snapshots["smoother"].sha256,
            },
        }
        _write_receipt_atomic_no_clobber(args.receipt, receipt)
    print(f"wrote parity receipt {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
