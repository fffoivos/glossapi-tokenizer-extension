#!/usr/bin/env python3
"""Python↔Rust parity on the historical LLM-silver structural corpus.

The legacy filename is `units/STRUCT_2K_gold.jsonl`, but its annotations are
LLM-produced silver. The check compares per-line probabilities and decoded
absolute line spans; it cannot itself authorize production.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import decode_spans as decode
import line_lr
import span_signals
import struct_lines


HERE = Path(__file__).resolve().parent
DEFAULT_BIN = HERE.parent / "reference_detector" / "target" / "debug" / "reference_detect"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--documents", type=int, default=0, help="0 means every held-out test document")
    parser.add_argument("--expected-documents", type=int, default=608)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-corpus-sha256")
    expected.add_argument("--expected-gold-sha256", help="deprecated legacy spelling")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if not struct_lines.GOLD or not os.path.exists(struct_lines.GOLD):
        raise SystemExit(
            f"missing hydrated structural LLM-silver corpus: {struct_lines.GOLD}"
        )
    if not args.binary.is_file():
        raise SystemExit(f"missing Rust binary: {args.binary}")
    expected_corpus_sha256 = args.expected_corpus_sha256 or args.expected_gold_sha256
    actual_corpus_sha256 = sha256_file(Path(struct_lines.GOLD))
    if actual_corpus_sha256 != expected_corpus_sha256:
        raise SystemExit(
            "structural LLM-silver corpus hash mismatch: "
            f"expected {expected_corpus_sha256}, got {actual_corpus_sha256}"
        )
    data = struct_lines.load()
    docs = [doc_id for doc_id, document in data.docs.items() if document["split"] == "test"]
    if args.documents:
        docs = docs[: args.documents]
    if not docs:
        raise SystemExit("no test documents found")
    if len(docs) != args.expected_documents:
        raise SystemExit(f"held-out coverage {len(docs)} != required {args.expected_documents}")
    sources = collections.Counter(str(data.docs[doc_id].get("source")) for doc_id in docs)
    positive_docs = {
        "bib": sum(any(label == 1 for label in data.docs[doc_id]["labels"]) for doc_id in docs),
        "toc": sum(any(label == 2 for label in data.docs[doc_id]["labels"]) for doc_id in docs),
    }
    if not sources or any(value == 0 for value in positive_docs.values()):
        raise SystemExit(f"held-out source/positive coverage is insufficient: sources={sources}, positives={positive_docs}")
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
        "corpus": str(Path(struct_lines.GOLD).resolve()),
        "corpus_sha256": actual_corpus_sha256,
        "heldout_documents": len(docs),
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
