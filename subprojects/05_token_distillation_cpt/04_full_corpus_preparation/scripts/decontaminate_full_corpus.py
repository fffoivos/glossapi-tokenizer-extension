#!/usr/bin/env python3
"""Conservatively remove GreekMMLU leakage from canonical cleaned Parquet.

The benchmark query file is not trusted by filename.  A mandatory sidecar
manifest binds its SHA-256, immutable upstream revision and required splits.
Only three high-confidence rules remove a document:

* an exact long evaluation-prompt/question-plus-choices surface;
* an exact question followed nearby by the correct answer; or
* an aligned >=85% question-shingle match that also passes deterministic
  MinHash >=85% and has the correct answer immediately after the question.

Answer-only and isolated short n-gram matches are audit evidence, never drop
rules.  Input text is copied unchanged; every decision is written to a
text-hash-bound Parquet ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from finalization_io import (
    atomic_output_path,
    discover_parquet,
    parquet_file_receipt,
    read_json_object,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_atomic,
)


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
HEX_REVISION = re.compile(r"^[0-9a-f]{40}$")
GREEKMMLU_REPO_ID = "dascim/GreekMMLU"
GREEKMMLU_CONFIG = "All"
QUERY_SCHEMA = "greek-mcq-decontam-query-v1"
POLICY_VERSION = "greekmmlu_decontamination_v1"
DEFAULT_K = 8
DEFAULT_MIN_COVERAGE = 0.85
DEFAULT_MINHASH_THRESHOLD = 0.85
DEFAULT_MIN_MATCHED_GRAMS = 4
DEFAULT_MAX_GAP = 40
DEFAULT_MIN_PROMPT_TOKENS = 16
DEFAULT_MIN_QUESTION_TOKENS = 8
MINHASH_PERMUTATIONS = 64
MINHASH_PRIME = (1 << 61) - 1


def normalize_surface(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(text.casefold().split())


def tokenize(value: object) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize_surface(value)))


def kgrams(tokens: tuple[str, ...], k: int) -> Iterator[tuple[int, tuple[str, ...]]]:
    for index in range(max(0, len(tokens) - k + 1)):
        yield index, tokens[index : index + k]


def _shingle_hash(shingle: tuple[str, ...]) -> int:
    payload = "\0".join(shingle).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"grmmlu-v1").digest(), "big")


def _permutation_parameters() -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for index in range(MINHASH_PERMUTATIONS):
        digest = hashlib.sha256(f"greekmmlu-minhash-v1:{index}".encode("ascii")).digest()
        a = (int.from_bytes(digest[:8], "big") % (MINHASH_PRIME - 1)) + 1
        b = int.from_bytes(digest[8:16], "big") % MINHASH_PRIME
        result.append((a, b))
    return tuple(result)


PERMUTATIONS = _permutation_parameters()


def minhash_signature(shingles: Iterable[tuple[str, ...]]) -> tuple[int, ...]:
    hashes = {_shingle_hash(shingle) % MINHASH_PRIME for shingle in shingles}
    if not hashes:
        return ()
    return tuple(min(((a * value) + b) % MINHASH_PRIME for value in hashes) for a, b in PERMUTATIONS)


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def _contiguous_position(haystack: tuple[str, ...], needle: tuple[str, ...], start: int, end: int) -> int:
    if not needle:
        return -1
    lower = max(0, start)
    upper = min(len(haystack), end)
    for index in range(lower, max(lower, upper - len(needle) + 1)):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _choices(row: Mapping[str, Any]) -> list[str]:
    raw = _field(row, "choices", "options", "answer_choices")
    if isinstance(raw, Mapping):
        raw = raw.get("text") or raw.get("choices") or raw.get("label")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("GreekMMLU query row has a non-list choices field")
    return [str(value) for value in raw]


def _correct_answer(row: Mapping[str, Any], choices: list[str]) -> str:
    explicit = _field(row, "answer_text", "correct_answer_text")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    raw = _field(row, "answer_index", "answer", "correct_answer")
    if raw is None:
        return ""
    if isinstance(raw, str) and raw.strip() in choices:
        return raw.strip()
    try:
        index = int(raw)
    except (TypeError, ValueError):
        label = str(raw).strip().upper()
        index = ord(label) - ord("A") if len(label) == 1 and "A" <= label <= "Z" else -1
    return choices[index] if 0 <= index < len(choices) else ""


@dataclass(frozen=True)
class BenchmarkItem:
    index: int
    item_id: str
    split: str
    subject: str
    question_tokens: tuple[str, ...]
    answer_tokens: tuple[str, ...]
    question_grams: tuple[tuple[str, ...], ...]
    question_signature: tuple[int, ...]
    prompt_surfaces: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class BenchmarkIndex:
    items: tuple[BenchmarkItem, ...]
    qgram_index: Mapping[tuple[str, ...], tuple[tuple[int, int], ...]]
    prompt_anchor_index: Mapping[tuple[str, ...], tuple[tuple[int, str, tuple[str, ...]], ...]]
    k: int
    min_coverage: float
    minhash_threshold: float
    min_matched_grams: int
    max_gap_tokens: int


def benchmark_manifest_path(queries_jsonl: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return Path(str(queries_jsonl) + ".manifest.json")


def load_benchmark_index(
    queries_jsonl: Path,
    manifest_path: Path,
    *,
    k: int,
    min_coverage: float,
    minhash_threshold: float,
    min_matched_grams: int,
    max_gap_tokens: int,
) -> tuple[BenchmarkIndex, dict[str, Any]]:
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != "greekmmlu_query_manifest_v1":
        raise ValueError(f"{manifest_path}: unsupported GreekMMLU manifest schema")
    if str(manifest.get("benchmark_id", "")).casefold() != "greekmmlu":
        raise ValueError(f"{manifest_path}: benchmark_id must be greekmmlu")
    if manifest.get("dataset_repo_id") != GREEKMMLU_REPO_ID:
        raise ValueError(f"{manifest_path}: dataset_repo_id must be {GREEKMMLU_REPO_ID}")
    if manifest.get("dataset_config") != GREEKMMLU_CONFIG:
        raise ValueError(f"{manifest_path}: dataset_config must be {GREEKMMLU_CONFIG}")
    revision = str(manifest.get("dataset_revision", ""))
    if not HEX_REVISION.fullmatch(revision):
        raise ValueError(f"{manifest_path}: dataset_revision must be a full 40-hex commit")
    expected_hash = str(manifest.get("queries_sha256", ""))
    actual_hash = sha256_file(queries_jsonl)
    if expected_hash != actual_hash:
        raise ValueError(f"{queries_jsonl}: SHA-256 differs from the benchmark manifest")
    required_splits = {str(value) for value in manifest.get("required_splits", [])}
    if not required_splits:
        raise ValueError(f"{manifest_path}: required_splits must be non-empty")
    if manifest.get("default_split") != "":
        raise ValueError(f"{manifest_path}: default_split must be empty; row provenance is mandatory")
    frozen_observed = {str(value) for value in manifest.get("observed_splits", [])}
    if frozen_observed != required_splits:
        raise ValueError(f"{manifest_path}: observed_splits must exactly equal required_splits")
    frozen_rows = manifest.get("query_rows")
    if not isinstance(frozen_rows, int) or isinstance(frozen_rows, bool) or frozen_rows < 1:
        raise ValueError(f"{manifest_path}: query_rows must be a positive integer")
    rows: list[dict[str, Any]] = []
    with queries_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{queries_jsonl}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{queries_jsonl}:{line_number}: query must be an object")
            benchmark = str(row.get("benchmark", "")).casefold()
            if benchmark != "greekmmlu":
                continue
            if row.get("schema") != QUERY_SCHEMA:
                raise ValueError(f"{queries_jsonl}:{line_number}: unsupported GreekMMLU query schema")
            if row.get("dataset_repo_id") != GREEKMMLU_REPO_ID:
                raise ValueError(f"{queries_jsonl}:{line_number}: dataset_repo_id drift")
            if row.get("dataset_revision") != revision:
                raise ValueError(f"{queries_jsonl}:{line_number}: dataset_revision drift")
            if row.get("dataset_config") != GREEKMMLU_CONFIG:
                raise ValueError(f"{queries_jsonl}:{line_number}: dataset_config drift")
            rows.append(row)
    if not rows:
        raise ValueError(f"{queries_jsonl}: no GreekMMLU query rows")
    if len(rows) != frozen_rows:
        raise ValueError(
            f"{queries_jsonl}: GreekMMLU row count differs from manifest: {len(rows)} != {frozen_rows}"
        )

    items: list[BenchmarkItem] = []
    seen_ids: set[tuple[str, str]] = set()
    observed_splits: set[str] = set()
    for row_number, row in enumerate(rows):
        split = str(row.get("split") or "")
        if not split:
            raise ValueError(f"GreekMMLU query row {row_number}: split provenance is absent")
        observed_splits.add(split)
        item_id = str(row.get("example_id") or "")
        if not item_id:
            raise ValueError(f"GreekMMLU query row {row_number}: example_id is absent")
        if (split, item_id) in seen_ids:
            raise ValueError(f"duplicate GreekMMLU query identity: {split}/{item_id}")
        seen_ids.add((split, item_id))
        question = str(_field(row, "question", "prompt_question") or "")
        question_tokens = tokenize(question)
        choices = _choices(row)
        answer_tokens = tokenize(_correct_answer(row, choices))
        if not question_tokens or not answer_tokens:
            raise ValueError(f"GreekMMLU query {split}/{item_id}: question/correct answer is empty")
        question_grams = tuple(dict.fromkeys(gram for _, gram in kgrams(question_tokens, k)))
        surfaces: dict[str, tuple[str, ...]] = {}
        raw_surfaces = row.get("surfaces")
        if isinstance(raw_surfaces, Mapping):
            for kind in ("eval_prompt", "question_all_choices", "prompt"):
                if raw_surfaces.get(kind):
                    surfaces[kind] = tokenize(raw_surfaces[kind])
        explicit_prompt = _field(row, "eval_prompt", "prompt")
        if explicit_prompt:
            surfaces["eval_prompt"] = tokenize(explicit_prompt)
        if choices:
            surfaces.setdefault("question_all_choices", tokenize(question + "\n" + "\n".join(choices)))
        prompt_surfaces = tuple(
            sorted(
                (kind, tokens)
                for kind, tokens in surfaces.items()
                if len(tokens) >= DEFAULT_MIN_PROMPT_TOKENS
            )
        )
        items.append(
            BenchmarkItem(
                index=len(items),
                item_id=item_id,
                split=split,
                subject=str(row.get("subject") or ""),
                question_tokens=question_tokens,
                answer_tokens=answer_tokens,
                question_grams=question_grams,
                question_signature=minhash_signature(question_grams),
                prompt_surfaces=prompt_surfaces,
            )
        )
    if observed_splits != required_splits:
        raise ValueError(
            f"{queries_jsonl}: split set differs from manifest: "
            f"observed={sorted(observed_splits)} required={sorted(required_splits)}"
        )

    qindex: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    pindex: dict[tuple[str, ...], list[tuple[int, str, tuple[str, ...]]]] = defaultdict(list)
    for item in items:
        for offset, gram in kgrams(item.question_tokens, k):
            qindex[gram].append((item.index, offset))
        for kind, surface in item.prompt_surfaces:
            if len(surface) >= k:
                pindex[surface[:k]].append((item.index, kind, surface))
    index = BenchmarkIndex(
        items=tuple(items),
        qgram_index={key: tuple(value) for key, value in qindex.items()},
        prompt_anchor_index={key: tuple(value) for key, value in pindex.items()},
        k=k,
        min_coverage=min_coverage,
        minhash_threshold=minhash_threshold,
        min_matched_grams=min_matched_grams,
        max_gap_tokens=max_gap_tokens,
    )
    receipt = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "queries_path": str(queries_jsonl.resolve()),
        "queries_sha256": actual_hash,
        "dataset_repo_id": manifest.get("dataset_repo_id"),
        "dataset_revision": revision,
        "dataset_config": manifest.get("dataset_config"),
        "required_splits": sorted(required_splits),
        "observed_splits": sorted(observed_splits),
        "items": len(items),
    }
    return index, receipt


def _evidence(item: BenchmarkItem, method: str, **extra: Any) -> dict[str, Any]:
    return {
        "benchmark": "greekmmlu",
        "benchmark_item_id": item.item_id,
        "split": item.split,
        "subject": item.subject,
        "method": method,
        **extra,
    }


def match_document(text: str, index: BenchmarkIndex) -> tuple[str, str, list[dict[str, Any]]]:
    tokens = tokenize(text)
    if len(tokens) < index.k:
        return "keep", "no_high_confidence_match", []
    proposed_starts: dict[int, set[int]] = defaultdict(set)
    exact_evidence: list[dict[str, Any]] = []
    seen_prompt: set[tuple[int, str, int]] = set()
    for position, gram in kgrams(tokens, index.k):
        for item_index, question_offset in index.qgram_index.get(gram, ()):
            start = position - question_offset
            if start >= 0 and len(proposed_starts[item_index]) < 128:
                proposed_starts[item_index].add(start)
        for item_index, kind, surface in index.prompt_anchor_index.get(gram, ()):
            key = (item_index, kind, position)
            if key in seen_prompt:
                continue
            seen_prompt.add(key)
            if tokens[position : position + len(surface)] == surface:
                exact_evidence.append(
                    _evidence(index.items[item_index], "exact_prompt", surface=kind, token_start=position)
                )
    if exact_evidence:
        return "drop", "greekmmlu_exact_prompt", exact_evidence

    audit: list[dict[str, Any]] = []
    for item_index in sorted(proposed_starts):
        item = index.items[item_index]
        for start in sorted(proposed_starts[item_index])[:64]:
            question_end = start + len(item.question_tokens)
            if question_end > len(tokens):
                continue
            answer_position = _contiguous_position(
                tokens,
                item.answer_tokens,
                question_end,
                question_end + index.max_gap_tokens + len(item.answer_tokens) + 1,
            )
            if tokens[start:question_end] == item.question_tokens:
                if len(item.question_tokens) >= DEFAULT_MIN_QUESTION_TOKENS and answer_position >= 0:
                    evidence = _evidence(
                        item,
                        "exact_question_answer",
                        token_start=start,
                        answer_token_start=answer_position,
                    )
                    return "drop", "greekmmlu_exact_question_answer", [evidence]
                audit.append(
                    _evidence(item, "exact_question_without_answer", token_start=start)
                )
                continue
            if len(item.question_tokens) < 12 or len(item.question_grams) < index.min_matched_grams:
                continue
            window = tokens[start:question_end]
            window_grams = tuple(dict.fromkeys(gram for _, gram in kgrams(window, index.k)))
            question_set = set(item.question_grams)
            window_set = set(window_grams)
            matched = len(question_set & window_set)
            coverage = matched / len(question_set) if question_set else 0.0
            similarity = minhash_similarity(item.question_signature, minhash_signature(window_grams))
            if matched >= index.min_matched_grams and coverage >= index.min_coverage:
                evidence = _evidence(
                    item,
                    "aligned_ngram_minhash",
                    token_start=start,
                    matched_unique_grams=matched,
                    question_unique_grams=len(question_set),
                    question_coverage=round(coverage, 6),
                    minhash_similarity=round(similarity, 6),
                    answer_token_start=answer_position if answer_position >= 0 else None,
                )
                audit.append(evidence)
                if similarity >= index.minhash_threshold and answer_position >= 0:
                    return "drop", "greekmmlu_ngram_minhash_answer", [evidence]
    return "keep", "no_high_confidence_match", audit[:16]


def ledger_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("stable_uid", pa.string()),
            ("acquisition_source_id", pa.string()),
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("action", pa.string()),
            ("reason", pa.string()),
            ("benchmark_matches_json", pa.string()),
        ]
    )


_WORKER_INDEX: BenchmarkIndex | None = None


def _worker_init(index: BenchmarkIndex) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _process_file(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_name, relative, output_root, dropped_root, ledger_root = task
    input_path = Path(input_name)
    output_path = Path(output_root) / relative
    dropped_path = Path(dropped_root) / relative
    ledger_path = Path(ledger_root) / relative
    index = _WORKER_INDEX
    if index is None:
        raise RuntimeError("decontamination worker lacks benchmark index")
    parquet = pq.ParquetFile(input_path)
    required = {"stable_uid", "source_dataset", "source_doc_id", "acquisition_source_id", "text"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{input_path}: missing required columns: {sorted(missing)}")
    output_temp = atomic_output_path(output_path)
    dropped_temp = atomic_output_path(dropped_path)
    ledger_temp = atomic_output_path(ledger_path)
    output_writer = pq.ParquetWriter(output_temp, parquet.schema_arrow, compression="zstd")
    dropped_writer = pq.ParquetWriter(dropped_temp, parquet.schema_arrow, compression="zstd")
    action_writer = pq.ParquetWriter(ledger_temp, ledger_schema(), compression="zstd")
    counts = {"input": 0, "kept": 0, "dropped": 0, "audit_candidates": 0}
    try:
        for batch in parquet.iter_batches(batch_size=512, use_threads=False):
            rows = batch.to_pylist()
            kept_rows: list[dict[str, Any]] = []
            dropped_rows: list[dict[str, Any]] = []
            action_rows: list[dict[str, Any]] = []
            for row in rows:
                text = str(row.get("text") or "")
                actual_text_hash = sha256_text(text)
                claimed_text_hash = row.get("cleaned_text_sha256")
                if claimed_text_hash and str(claimed_text_hash) != actual_text_hash:
                    raise ValueError(
                        f"{input_path}: cleaned_text_sha256 drift for stable_uid={row['stable_uid']}"
                    )
                action, reason, evidence = match_document(text, index)
                text_hash = actual_text_hash
                action_rows.append(
                    {
                        "stable_uid": str(row["stable_uid"]),
                        "acquisition_source_id": str(row["acquisition_source_id"]),
                        "source_dataset": str(row["source_dataset"]),
                        "source_doc_id": str(row["source_doc_id"]),
                        "input_text_sha256": text_hash,
                        "action": action,
                        "reason": reason,
                        "benchmark_matches_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                    }
                )
                if action == "drop":
                    dropped_rows.append(row)
                    counts["dropped"] += 1
                else:
                    kept_rows.append(row)
                    counts["kept"] += 1
                    if evidence:
                        counts["audit_candidates"] += 1
                counts["input"] += 1
            if kept_rows:
                output_writer.write_table(pa.Table.from_pylist(kept_rows, schema=parquet.schema_arrow))
            if dropped_rows:
                dropped_writer.write_table(pa.Table.from_pylist(dropped_rows, schema=parquet.schema_arrow))
            if action_rows:
                action_writer.write_table(pa.Table.from_pylist(action_rows, schema=ledger_schema()))
    except BaseException:
        output_writer.close()
        dropped_writer.close()
        action_writer.close()
        output_temp.unlink(missing_ok=True)
        dropped_temp.unlink(missing_ok=True)
        ledger_temp.unlink(missing_ok=True)
        raise
    output_writer.close()
    dropped_writer.close()
    action_writer.close()
    os.replace(output_temp, output_path)
    os.replace(dropped_temp, dropped_path)
    os.replace(ledger_temp, ledger_path)
    return {
        "relative_path": relative,
        "counts": counts,
        "output": parquet_file_receipt(output_path, relative_to=Path(output_root)),
        "dropped": parquet_file_receipt(dropped_path, relative_to=Path(dropped_root)),
        "ledger": parquet_file_receipt(ledger_path, relative_to=Path(ledger_root)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cleaned canonical Parquet root")
    parser.add_argument("--output", type=Path, required=True, help="Decontaminated Parquet root")
    parser.add_argument("--dropped", type=Path, required=True, help="High-confidence removals")
    parser.add_argument("--ledger", type=Path, required=True, help="Per-document decision Parquet root")
    parser.add_argument("--manifest", type=Path, required=True, help="Immutable run manifest")
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--minhash-threshold", type=float, default=DEFAULT_MINHASH_THRESHOLD)
    parser.add_argument("--min-matched-grams", type=int, default=DEFAULT_MIN_MATCHED_GRAMS)
    parser.add_argument("--max-gap-tokens", type=int, default=DEFAULT_MAX_GAP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.k < DEFAULT_K:
        raise ValueError(f"production --k cannot be lower than {DEFAULT_K}")
    if args.min_matched_grams < DEFAULT_MIN_MATCHED_GRAMS:
        raise ValueError(
            f"production --min-matched-grams cannot be lower than {DEFAULT_MIN_MATCHED_GRAMS}"
        )
    if not 0 <= args.max_gap_tokens <= DEFAULT_MAX_GAP:
        raise ValueError(f"production --max-gap-tokens must be in [0,{DEFAULT_MAX_GAP}]")
    if not (0.85 <= args.min_coverage <= 1.0) or not (0.85 <= args.minhash_threshold <= 1.0):
        raise ValueError("production thresholds cannot be lower than 0.85")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {args.manifest}")
    benchmark_path = benchmark_manifest_path(args.queries_jsonl, args.benchmark_manifest)
    index, benchmark_receipt = load_benchmark_index(
        args.queries_jsonl,
        benchmark_path,
        k=args.k,
        min_coverage=args.min_coverage,
        minhash_threshold=args.minhash_threshold,
        min_matched_grams=args.min_matched_grams,
        max_gap_tokens=args.max_gap_tokens,
    )
    files = discover_parquet(args.input)
    for root in (args.output, args.dropped, args.ledger):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"refusing to write into non-empty output root: {root}")
        root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(path),
            path.relative_to(args.input).as_posix(),
            str(args.output),
            str(args.dropped),
            str(args.ledger),
        )
        for path in files
    ]
    if args.workers == 1:
        _worker_init(index)
        receipts = [_process_file(task) for task in tasks]
    else:
        try:
            context = mp.get_context("fork")
        except ValueError:  # pragma: no cover - Windows fallback
            context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(index,),
        ) as executor:
            receipts = list(executor.map(_process_file, tasks, chunksize=1))
    totals: dict[str, int] = defaultdict(int)
    for receipt in receipts:
        for key, value in receipt["counts"].items():
            totals[key] += int(value)
    payload = {
        "schema_version": "full_cpt_greekmmlu_decontamination_v1",
        "status": "completed",
        "completed_at": utc_now(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "dropped": str(args.dropped.resolve()),
        "ledger": str(args.ledger.resolve()),
        "benchmark": benchmark_receipt,
        "policy": {
            "policy_version": POLICY_VERSION,
            "normalization": "NFKC+strip_combining_marks+casefold+unicode_word_tokens_v1",
            "k": args.k,
            "min_coverage": args.min_coverage,
            "minhash_threshold": args.minhash_threshold,
            "minhash_permutations": MINHASH_PERMUTATIONS,
            "min_matched_grams": args.min_matched_grams,
            "max_gap_tokens": args.max_gap_tokens,
            "drop_rules": [
                "greekmmlu_exact_prompt",
                "greekmmlu_exact_question_answer",
                "greekmmlu_ngram_minhash_answer",
            ],
            "answer_only_action": "audit_only",
        },
        "workers": args.workers,
        "counts": dict(totals),
        "files": sorted(receipts, key=lambda row: row["relative_path"]),
    }
    if totals["input"] != totals["kept"] + totals["dropped"]:
        raise RuntimeError("decontamination row accounting does not reconcile")
    write_json_atomic(args.manifest, payload)
    print(json.dumps({"ok": True, "manifest": str(args.manifest), "counts": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
