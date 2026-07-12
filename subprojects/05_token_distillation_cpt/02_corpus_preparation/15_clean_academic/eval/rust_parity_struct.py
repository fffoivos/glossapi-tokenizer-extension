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
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import decode_spans as decode
import line_lr
import span_signals
import struct_lines


HERE = Path(__file__).resolve().parent
DEFAULT_BIN = HERE.parent / "reference_detector" / "target" / "debug" / "reference_detect"
LABELS = {"O": 0, "BIB": 1, "TOC": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
                raise ValueError(f"{path}:{row_number}: split/coordinate boundary drift")
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


def validate_source_receipt(receipt_path: Path, corpus_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    exclusion = receipt.get("historical_partition_exclusion")
    if (
        receipt.get("schema_version")
        != "academic-structure-silver-contract-receipt-v1"
        or receipt.get("status") != "pass"
        or receipt.get("evidence_tier") != "LLM_silver"
        or receipt.get("production_eligible") is not False
        or receipt.get("silver_sha256") != sha256_file(corpus_path)
        or not isinstance(exclusion, dict)
        or exclusion.get("eligible_historical_train_documents")
        != receipt.get("document_count")
        or exclusion.get("historical_test_documents_excluded") != 608
        or exclusion.get("historical_test_rows_emitted") != 0
        or exclusion.get("historical_test_predictions_permitted") is not False
        or receipt.get("task_scope_counts", {}).get("bibliography_toc_windows")
        != receipt.get("document_count")
    ):
        raise ValueError("modern joint source receipt does not prove 608-document exclusion")
    materialized = receipt.get("materialized_artifacts", {})
    if (
        materialized.get("silver_filename") != corpus_path.name
        or receipt_path.resolve().parent != corpus_path.resolve().parent
    ):
        raise ValueError("modern joint source artifact path differs from its receipt")
    return receipt


def frozen_probabilities(data, model_path: Path, *, toc: bool) -> dict[str, np.ndarray]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    mu = np.asarray(model["mu"], dtype=np.float64)
    sd = np.asarray(model["sd"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    bias = float(model["bias"])
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
        scores = 1.0 / (1.0 + np.exp(-(((matrix - mu) / sd) @ weight + bias)))
        if toc:
            cut = min(300, int(0.30 * document["N"]))
            scores *= np.asarray([line < cut for line, _text in document["lines"]], dtype=bool)
        result[doc_id] = scores
    return result


def run_rust_scores(binary: Path, docs: list[str], data, mode: str, work: Path) -> list[list[float]]:
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
    rows = [json.loads(line)["p"] for line in output_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != len(docs):
        raise AssertionError(f"{mode}: output rows {len(rows)} != input docs {len(docs)}")
    return rows


def run_rust_spans(binary: Path, docs: list[str], data, mode: str, work: Path) -> dict[str, list[tuple[int, int]]]:
    input_path = work / f"{mode}.docs.jsonl"
    spans_path = work / f"{mode}.spans.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for doc_id in docs:
            document = data.docs[doc_id]
            lines = [""] * document["N"]
            for line, text in document["lines"]:
                lines[line] = text
            handle.write(json.dumps({"id": doc_id, "text": "\n".join(lines)}, ensure_ascii=False) + "\n")
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
            result.setdefault(row["doc_id"], []).append((row["line_start"], row["line_end"]))
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


def compare_head(binary: Path, docs: list[str], data, *, head: str, work: Path, tolerance: float) -> dict:
    toc = head == "toc"
    model = HERE / ("toc_line_lr_model.json" if toc else "span_line_lr_struct_model.json")
    python_probabilities = frozen_probabilities(data, model, toc=toc)
    score_mode = "toc-score-lines" if toc else "score-lines"
    rust_probabilities = run_rust_scores(binary, docs, data, score_mode, work)

    max_difference = 0.0
    worst = None
    for doc_id, rust_scores in zip(docs, rust_probabilities):
        python_scores = python_probabilities[doc_id]
        if len(python_scores) != len(rust_scores):
            raise AssertionError((doc_id, len(python_scores), len(rust_scores)))
        for index, (python_score, rust_score) in enumerate(zip(python_scores, rust_scores)):
            difference = abs(float(python_score) - float(rust_score))
            if difference > max_difference:
                max_difference = difference
                worst = (doc_id, index, float(python_score), float(rust_score))
    if max_difference >= tolerance:
        raise AssertionError(f"{head}: max probability difference {max_difference:.6g}; worst={worst}")

    params = json.loads((HERE / "struct_smooth_params.json").read_text(encoding="utf-8"))[head]
    params = {key: params[key] for key in ("theta_hi", "theta_lo", "gap", "lmin")}
    span_mode = "toc-spans" if toc else "bib-spans"
    rust_spans = run_rust_spans(binary, docs, data, span_mode, work)
    mismatches = []
    for doc_id in docs:
        python_spans = sorted(decode.decode_doc(data.docs[doc_id], python_probabilities[doc_id], params))
        actual = sorted(rust_spans.get(doc_id, []))
        if python_spans != actual:
            mismatches.append((doc_id, python_spans, actual))
    if mismatches:
        raise AssertionError(f"{head}: {len(mismatches)} span mismatches; first={mismatches[:3]}")
    print(f"{head}: {len(docs)} docs; max |Δp|={max_difference:.6g}; spans identical")
    return {"documents": len(docs), "max_probability_difference": max_difference, "span_mismatches": 0}


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
        "--partition",
        choices=("train", "validation", "test", "all"),
        help="default: validation for modern import, test for legacy replay",
    )
    parser.add_argument(
        "--documents", type=int, default=0, help="0 means every document in the partition"
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

    modern = args.corpus is not None
    if modern != (args.source_receipt is not None):
        raise SystemExit("--corpus and --source-receipt must be supplied together")
    corpus_path = args.corpus if modern else Path(struct_lines.GOLD)
    if not corpus_path.is_file():
        raise SystemExit(f"missing hydrated structural LLM-silver corpus: {corpus_path}")
    if not args.binary.is_file():
        raise SystemExit(f"missing Rust binary: {args.binary}")
    expected_corpus_sha256 = args.expected_corpus_sha256 or args.expected_gold_sha256
    actual_corpus_sha256 = sha256_file(corpus_path)
    if actual_corpus_sha256 != expected_corpus_sha256:
        raise SystemExit(
            "structural LLM-silver corpus hash mismatch: "
            f"expected {expected_corpus_sha256}, got {actual_corpus_sha256}"
        )
    if modern:
        source_receipt = validate_source_receipt(args.source_receipt, corpus_path)
        data = load_materialized(corpus_path)
        partition = args.partition or "validation"
        if partition == "test":
            raise SystemExit("modern imported source physically contains no historical-test rows")
        receipt_expected = (
            source_receipt.get("document_count")
            if partition == "all"
            else source_receipt.get("split_counts", {}).get(partition, -1)
        )
        expected_documents = (
            args.expected_documents
            if args.expected_documents is not None
            else int(receipt_expected)
        )
        semantics = "derived_historical_train_validation_runtime_parity_not_quality_holdout"
    else:
        data = struct_lines.load()
        partition = args.partition or "test"
        expected_documents = args.expected_documents or 608
        semantics = "legacy_historical_test_runtime_parity_not_independent_quality_evidence"
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
    sources = collections.Counter(str(data.docs[doc_id].get("source")) for doc_id in docs)
    positive_docs = {
        "bib": sum(any(label == 1 for label in data.docs[doc_id]["labels"]) for doc_id in docs),
        "toc": sum(any(label == 2 for label in data.docs[doc_id]["labels"]) for doc_id in docs),
    }
    if not sources or any(value == 0 for value in positive_docs.values()):
        raise SystemExit(
            f"parity source/positive coverage is insufficient: sources={sources}, "
            f"positives={positive_docs}"
        )
    with tempfile.TemporaryDirectory(prefix="rust-parity-struct-") as temporary:
        work = Path(temporary)
        heads = {
            "bib": compare_head(args.binary, docs, data, head="bib", work=work, tolerance=args.tolerance),
            "toc": compare_head(args.binary, docs, data, head="toc", work=work, tolerance=args.tolerance),
        }
    receipt = {
        "schema_version": "struct_rust_parity_receipt_v1",
        "status": "passed",
        "evidence_status": "LLM_silver",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256_file(args.binary),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": actual_corpus_sha256,
        "heldout_documents": len(docs),
        "evaluation_partition": partition,
        "partition_semantics": semantics,
        "historical_test_documents_loaded": 0 if modern else len(docs),
        "source_receipt_sha256": (
            sha256_file(args.source_receipt) if modern else None
        ),
        "source_document_counts": dict(sorted(sources.items())),
        "positive_document_counts": positive_docs,
        "tolerance": args.tolerance,
        "heads": heads,
        "model_sha256": {
            "bib": sha256_file(HERE / "span_line_lr_struct_model.json"),
            "toc": sha256_file(HERE / "toc_line_lr_model.json"),
            "smoother": sha256_file(HERE / "struct_smooth_params.json"),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote parity receipt {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
