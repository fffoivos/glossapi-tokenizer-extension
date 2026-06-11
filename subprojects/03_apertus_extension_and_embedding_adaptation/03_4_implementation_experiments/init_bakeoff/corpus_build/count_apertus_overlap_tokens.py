#!/usr/bin/env python3
"""Count token mass represented by the Apertus-overlap hard-drop overlay.

This answers a narrower question than the published dedup-audit report:
given the source text parquets and `cpt_final_overlay/apertus_overlap_drop_docs.parquet`,
how many tokenizer tokens are in the documents that CPT excluded because
Apertus had already seen them?

The script streams source parquets and keeps only a doc-key set in memory. It
does not write document text and does not affect training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def stable_doc_key(source_dataset: str, source_doc_id: str) -> str:
    try:
        from blake3 import blake3
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "blake3 is required when source parquets do not already contain doc_key; "
            "install it in the runtime or run inside the existing Clariden/uenv stack."
        ) from exc
    return blake3(f"{source_dataset}\0{source_doc_id}".encode("utf-8")).hexdigest()


@dataclass
class TextCounters:
    rows: int = 0
    chars: int = 0
    utf8_bytes: int = 0

    def add(self, text: str) -> None:
        self.rows += 1
        self.chars += len(text)
        self.utf8_bytes += len(text.encode("utf-8"))


@dataclass
class TokenCounters:
    tokens_no_eod: int = 0
    tokens_with_eod: int = 0

    def add(self, token_lengths: list[int], *, append_eod: bool) -> None:
        n = sum(token_lengths)
        self.tokens_no_eod += n
        self.tokens_with_eod += n + (len(token_lengths) if append_eod else 0)


@dataclass
class SourceCounters:
    total: TextCounters = field(default_factory=TextCounters)
    dropped: TextCounters = field(default_factory=TextCounters)
    fresh: TextCounters = field(default_factory=TextCounters)
    tokenizers: dict[str, TokenCounters] = field(default_factory=lambda: defaultdict(TokenCounters))


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(p) for p in matches)
            continue
        p = Path(pattern)
        if p.exists():
            paths.append(p)
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise SystemExit(f"No source parquets matched: {patterns}")
    return unique


def parse_tokenizer_specs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--tokenizer must be LABEL=PATH, got: {spec!r}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser()
        if not label:
            raise SystemExit(f"empty tokenizer label in {spec!r}")
        if label in out:
            raise SystemExit(f"duplicate tokenizer label: {label}")
        if not path.exists():
            raise SystemExit(f"tokenizer path for {label!r} does not exist: {path}")
        out[label] = path
    if not out:
        raise SystemExit("At least one --tokenizer LABEL=PATH is required")
    return out


def load_drop_keys(paths: list[Path]) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    import pyarrow.parquet as pq

    drop: set[tuple[str, str]] = set()
    by_source: dict[str, int] = defaultdict(int)
    stage_counts: dict[str, int] = defaultdict(int)
    input_rows = 0
    for path in paths:
        table = pq.read_table(path)
        cols = table.column_names
        if "doc_key" not in cols:
            raise SystemExit(f"{path} has no doc_key column; columns={cols}")
        if "source_dataset" not in cols:
            raise SystemExit(f"{path} has no source_dataset column; columns={cols}")
        data = table.to_pydict()
        stages = data.get("best_overlap_stage") or data.get("overlap_stage")
        for i, (doc_key, source_dataset) in enumerate(zip(data["doc_key"], data["source_dataset"], strict=True)):
            key = (str(source_dataset), str(doc_key))
            if key not in drop:
                by_source[str(source_dataset)] += 1
            drop.add(key)
            if stages is not None:
                stage_counts[str(stages[i])] += 1
            input_rows += 1
    meta = {
        "drop_input_rows": input_rows,
        "unique_drop_docs": len(drop),
        "drop_docs_by_source": dict(sorted(by_source.items())),
        "drop_docs_by_stage": dict(sorted(stage_counts.items())),
    }
    return drop, meta


def batch_token_lengths(tokenizer: Any, texts: list[str], encode_batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), encode_batch_size):
        chunk = texts[start : start + encode_batch_size]
        enc = tokenizer(
            chunk,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def get_column(data: dict[str, list[Any]], names: tuple[str, ...], *, required: bool = True) -> tuple[str | None, list[Any] | None]:
    for name in names:
        if name in data:
            return name, data[name]
    if required:
        raise KeyError(f"missing required column; tried {names}; available={sorted(data)}")
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-glob", action="append", required=True, help="Source parquet path/glob. Repeatable.")
    ap.add_argument("--drop-docs", action="append", required=True, help="Apertus overlap drop-doc parquet. Repeatable.")
    ap.add_argument("--tokenizer", action="append", required=True, help="Tokenizer as LABEL=PATH. Repeatable.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8192, help="Parquet rows per read batch.")
    ap.add_argument("--encode-batch-size", type=int, default=256, help="Texts per tokenizer batch.")
    ap.add_argument("--append-eod", action="store_true", help="Also report token counts with +1 EOD per document.")
    ap.add_argument(
        "--count-all-tokens",
        action="store_true",
        help="Tokenize every source row, not just dropped rows. Slower, but yields total/fresh token counts.",
    )
    args = ap.parse_args()

    t0 = time.time()
    source_paths = expand_inputs(args.source_glob)
    drop_paths = expand_inputs(args.drop_docs)
    tokenizer_specs = parse_tokenizer_specs(args.tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=== count_apertus_overlap_tokens ===", flush=True)
    print(f"source parquets: {len(source_paths)}", flush=True)
    print(f"drop parquets:   {len(drop_paths)}", flush=True)
    print(f"output_dir:      {args.output_dir}", flush=True)
    print(f"count_all:       {args.count_all_tokens}", flush=True)

    drop_keys, drop_meta = load_drop_keys(drop_paths)
    print(f"loaded drop docs: {len(drop_keys):,}", flush=True)

    from transformers import AutoTokenizer  # type: ignore
    import pyarrow.parquet as pq

    tokenizers = {}
    tokenizer_meta = {}
    for label, path in tokenizer_specs.items():
        print(f"loading tokenizer {label}: {path}", flush=True)
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        tokenizers[label] = tok
        tokenizer_meta[label] = {
            "path": str(path),
            "vocab_size": getattr(tok, "vocab_size", None),
            "class": tok.__class__.__name__,
        }

    counters: dict[str, SourceCounters] = defaultdict(SourceCounters)
    totals = SourceCounters()
    rows_seen = 0
    rows_dropped = 0
    rows_without_text = 0
    rows_without_key = 0

    for path_idx, path in enumerate(source_paths, start=1):
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        columns = ["text"]
        for candidate in ("doc_key", "source_dataset", "source_doc_id", "doc_id", "id"):
            if candidate in available:
                columns.append(candidate)
        columns = list(dict.fromkeys(columns))
        if "text" not in available:
            raise SystemExit(f"{path} has no text column")
        if "source_dataset" not in available:
            raise SystemExit(f"{path} has no source_dataset column")
        if "doc_key" not in available and not ({"source_doc_id"} & available):
            raise SystemExit(f"{path} needs doc_key or source_doc_id to match the overlay")

        print(f"[{path_idx}/{len(source_paths)}] {path}", flush=True)
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
            data = batch.to_pydict()
            _, source_col = get_column(data, ("source_dataset",))
            text_name, text_col = get_column(data, ("text",))
            doc_key_name, doc_key_col = get_column(data, ("doc_key",), required=False)
            _, source_doc_id_col = get_column(data, ("source_doc_id", "doc_id", "id"), required=False)

            dropped_texts_by_source: dict[str, list[str]] = defaultdict(list)
            all_texts_by_source: dict[str, list[str]] = defaultdict(list)

            for i in range(batch.num_rows):
                rows_seen += 1
                text = text_col[i] if text_col is not None else None
                if not isinstance(text, str) or not text:
                    rows_without_text += 1
                    continue
                source_dataset = str(source_col[i])
                if doc_key_col is not None:
                    doc_key = str(doc_key_col[i])
                elif source_doc_id_col is not None:
                    doc_key = stable_doc_key(source_dataset, str(source_doc_id_col[i]))
                else:
                    rows_without_key += 1
                    continue

                is_dropped = (source_dataset, doc_key) in drop_keys
                counters[source_dataset].total.add(text)
                totals.total.add(text)
                if is_dropped:
                    rows_dropped += 1
                    counters[source_dataset].dropped.add(text)
                    totals.dropped.add(text)
                    dropped_texts_by_source[source_dataset].append(text)
                else:
                    counters[source_dataset].fresh.add(text)
                    totals.fresh.add(text)

                if args.count_all_tokens:
                    all_texts_by_source[source_dataset].append(text)

            if args.count_all_tokens:
                for source_dataset, texts in all_texts_by_source.items():
                    for label, tok in tokenizers.items():
                        lengths = batch_token_lengths(tok, texts, args.encode_batch_size)
                        counters[source_dataset].tokenizers[f"{label}:total"].add(lengths, append_eod=args.append_eod)
                        totals.tokenizers[f"{label}:total"].add(lengths, append_eod=args.append_eod)

            for source_dataset, texts in dropped_texts_by_source.items():
                for label, tok in tokenizers.items():
                    lengths = batch_token_lengths(tok, texts, args.encode_batch_size)
                    counters[source_dataset].tokenizers[f"{label}:dropped"].add(lengths, append_eod=args.append_eod)
                    totals.tokenizers[f"{label}:dropped"].add(lengths, append_eod=args.append_eod)

        elapsed = time.time() - t0
        print(
            f"  progress rows={rows_seen:,} dropped={rows_dropped:,} "
            f"elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    by_source_rows: list[dict[str, Any]] = []
    for source, c in sorted(counters.items()):
        row: dict[str, Any] = {
            "source_dataset": source,
            "total_rows": c.total.rows,
            "dropped_rows": c.dropped.rows,
            "fresh_rows": c.fresh.rows,
            "total_chars": c.total.chars,
            "dropped_chars": c.dropped.chars,
            "fresh_chars": c.fresh.chars,
            "total_utf8_bytes": c.total.utf8_bytes,
            "dropped_utf8_bytes": c.dropped.utf8_bytes,
            "fresh_utf8_bytes": c.fresh.utf8_bytes,
            "dropped_row_share": c.dropped.rows / c.total.rows if c.total.rows else 0.0,
            "dropped_char_share": c.dropped.chars / c.total.chars if c.total.chars else 0.0,
        }
        for metric, tc in sorted(c.tokenizers.items()):
            row[f"{metric}_tokens_no_eod"] = tc.tokens_no_eod
            row[f"{metric}_tokens_with_eod"] = tc.tokens_with_eod
        by_source_rows.append(row)

    def text_counter_dict(c: TextCounters) -> dict[str, int]:
        return {"rows": c.rows, "chars": c.chars, "utf8_bytes": c.utf8_bytes}

    summary: dict[str, Any] = {
        "source_globs": args.source_glob,
        "source_parquets": [str(p) for p in source_paths],
        "drop_docs": [str(p) for p in drop_paths],
        "drop_meta": drop_meta,
        "tokenizers": tokenizer_meta,
        "append_eod": args.append_eod,
        "count_all_tokens": args.count_all_tokens,
        "rows_seen": rows_seen,
        "rows_dropped_matched_in_source": rows_dropped,
        "drop_docs_unmatched_in_source": max(len(drop_keys) - rows_dropped, 0),
        "rows_without_text": rows_without_text,
        "rows_without_key": rows_without_key,
        "totals": {
            "total": text_counter_dict(totals.total),
            "dropped": text_counter_dict(totals.dropped),
            "fresh": text_counter_dict(totals.fresh),
            "shares": {
                "dropped_rows": totals.dropped.rows / totals.total.rows if totals.total.rows else 0.0,
                "dropped_chars": totals.dropped.chars / totals.total.chars if totals.total.chars else 0.0,
                "dropped_utf8_bytes": (
                    totals.dropped.utf8_bytes / totals.total.utf8_bytes if totals.total.utf8_bytes else 0.0
                ),
            },
            "tokenizers": {
                metric: {
                    "tokens_no_eod": tc.tokens_no_eod,
                    "tokens_with_eod": tc.tokens_with_eod,
                }
                for metric, tc in sorted(totals.tokenizers.items())
            },
        },
        "wall_seconds": time.time() - t0,
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "by_source.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = sorted({k for row in by_source_rows for k in row})
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_source_rows)
    (args.output_dir / "by_source.json").write_text(
        json.dumps(by_source_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)
    print(f"wrote {args.output_dir / 'by_source.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
