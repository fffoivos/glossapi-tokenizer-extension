#!/usr/bin/env python3
"""Count ReTok new-token firings and snippet coverage for Token Distillation.

This is the CPU/I/O prepass that must run before any GPU Token Distillation
pilot. It scans the exact mixed JSONL stream in training order, tokenizes with
the extended tokenizer, and records actual emitted new-token IDs. Merge ancestry
and raw substring matches do not count.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import unicodedata as ud
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from tokenizers import Tokenizer


@dataclass
class TokenCoverage:
    extended_firings: int = 0
    docs_with_firing: int = 0
    snippets_seen: int = 0
    snippets: list[dict[str, object]] = field(default_factory=list)


def tokenizer_json_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "tokenizer.json"
    else:
        candidate = path
    if not candidate.is_file():
        raise SystemExit(f"tokenizer.json not found: {candidate}")
    return candidate


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, object]]]:
    for path in paths:
        with open_text(path) as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{line_no}: expected JSON object")
                yield path, line_no, row


def status_for(usable: int) -> tuple[str, str]:
    if usable >= 100:
        return "enough_100", "td_100"
    if usable >= 25:
        return "enough_25", "td_25"
    if usable >= 20:
        return "low_20_24", "keep_retok"
    if usable > 0:
        return "low_lt20", "keep_retok"
    return "zero", "inspect"


def reservoir_add(
    rng: random.Random,
    coverage: TokenCoverage,
    max_snippets: int,
    snippet: dict[str, object],
) -> None:
    coverage.snippets_seen += 1
    if len(coverage.snippets) < max_snippets:
        coverage.snippets.append(snippet)
        return
    idx = rng.randrange(coverage.snippets_seen)
    if idx < max_snippets:
        coverage.snippets[idx] = snippet


def encode_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def build_static_token_info(
    base_tokenizer: Tokenizer,
    student_tokenizer: Tokenizer,
    token_id: int,
) -> dict[str, object]:
    raw = student_tokenizer.id_to_token(token_id)
    decoded = student_tokenizer.decode([token_id], skip_special_tokens=False)
    base_ids = encode_ids(base_tokenizer, decoded) if decoded else []
    return {
        "new_token_id": token_id,
        "raw_token": raw,
        "token_string": decoded,
        "base_subtoken_ids": base_ids,
        "base_subtoken_len": len(base_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--new-id-start", type=int, default=131_072)
    parser.add_argument("--new-id-end", type=int, default=148_480)
    parser.add_argument("--target-extended-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--source-key", default="source")
    parser.add_argument("--doc-id-key", default="doc_id")
    parser.add_argument("--lang-key", default="lang")
    parser.add_argument("--snippet-token-radius", type=int, default=50)
    parser.add_argument("--snippets-per-token", type=int, default=100)
    parser.add_argument("--example-refs-per-token", type=int, default=5)
    parser.add_argument("--progress-token-interval", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--require-nfc", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.new_id_start >= args.new_id_end:
        raise SystemExit("--new-id-start must be smaller than --new-id-end")
    if args.target_extended_tokens <= 0:
        raise SystemExit("--target-extended-tokens must be positive")

    base_tokenizer = Tokenizer.from_file(str(tokenizer_json_path(args.base_tokenizer)))
    student_tokenizer = Tokenizer.from_file(str(tokenizer_json_path(args.student_tokenizer)))
    vocab_size = student_tokenizer.get_vocab_size(with_added_tokens=True)
    if args.new_id_end > vocab_size:
        raise SystemExit(f"new-id-end {args.new_id_end} exceeds student vocab {vocab_size}")

    out_dir = args.output_dir
    snippet_dir = out_dir / "td_snippet_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    snippet_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ids_of_interest = set(range(args.new_id_start, args.new_id_end))
    coverage = {token_id: TokenCoverage() for token_id in ids_of_interest}
    static_info = {
        token_id: build_static_token_info(base_tokenizer, student_tokenizer, token_id)
        for token_id in range(args.new_id_start, args.new_id_end)
    }

    docs_seen = 0
    docs_used = 0
    tokens_seen = 0
    chars_seen = 0
    non_nfc_docs = 0
    stopped_on_budget = False
    next_progress_tokens = args.progress_token_interval if args.progress_token_interval > 0 else None

    for path, line_no, row in iter_jsonl(args.input_jsonl):
        text = row.get(args.text_key)
        if not isinstance(text, str) or not text:
            continue
        docs_seen += 1
        chars_seen += len(text)
        if args.require_nfc and text != ud.normalize("NFC", text):
            non_nfc_docs += 1

        enc = student_tokenizer.encode(text, add_special_tokens=False)
        ids = enc.ids
        offsets = enc.offsets
        if not ids:
            continue

        remaining = args.target_extended_tokens - tokens_seen
        if remaining <= 0:
            stopped_on_budget = True
            break
        if len(ids) > remaining:
            ids = ids[:remaining]
            offsets = offsets[:remaining]
            stopped_on_budget = True

        docs_used += 1
        tokens_seen += len(ids)
        source = row.get(args.source_key)
        lang = row.get(args.lang_key)
        doc_id = row.get(args.doc_id_key) or f"{path.name}:{line_no}"
        seen_in_doc: set[int] = set()

        for token_index, token_id in enumerate(ids):
            if token_id not in ids_of_interest:
                continue
            cov = coverage[token_id]
            cov.extended_firings += 1
            seen_in_doc.add(token_id)

            start, end = offsets[token_index]
            if end <= start:
                continue
            left_i = max(0, token_index - args.snippet_token_radius)
            right_i = min(len(ids), token_index + args.snippet_token_radius + 1)
            char_start = offsets[left_i][0]
            char_end = offsets[right_i - 1][1]
            surface = text[start:end]
            snippet = {
                "new_token_id": token_id,
                "doc_ref": f"{path}:{line_no}",
                "doc_id": doc_id,
                "source": source,
                "lang": lang,
                "token_index": token_index,
                "char_start": start,
                "char_end": end,
                "snippet_char_start": char_start,
                "snippet_char_end": char_end,
                "surface": surface,
                "span_base_subtoken_ids": encode_ids(base_tokenizer, surface),
                "snippet_text": text[char_start:char_end],
            }
            reservoir_add(rng, cov, args.snippets_per_token, snippet)

        for token_id in seen_in_doc:
            coverage[token_id].docs_with_firing += 1

        if stopped_on_budget:
            break
        if next_progress_tokens is not None and tokens_seen >= next_progress_tokens:
            print(
                json.dumps(
                    {
                        "event": "td_coverage_progress",
                        "tokens_scanned": tokens_seen,
                        "target_extended_tokens": args.target_extended_tokens,
                        "docs_seen": docs_seen,
                        "docs_used": docs_used,
                        "chars_seen": chars_seen,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            while next_progress_tokens is not None and tokens_seen >= next_progress_tokens:
                next_progress_tokens += args.progress_token_interval

    if args.require_nfc and non_nfc_docs:
        raise SystemExit(
            f"Refusing to emit TD coverage: {non_nfc_docs} docs were not NFC. "
            "Run on the post-normalize corpus or pass --no-require-nfc for a diagnostic-only scan."
        )

    snippet_jsonl = snippet_dir / "snippets.jsonl"
    prepass_jsonl = out_dir / "td_coverage_prepass.jsonl"
    summary_path = out_dir / "td_coverage_summary.json"

    snippet_refs_by_token: dict[int, list[str]] = {token_id: [] for token_id in ids_of_interest}
    with snippet_jsonl.open("w", encoding="utf-8") as fh:
        for token_id in range(args.new_id_start, args.new_id_end):
            for i, snippet in enumerate(coverage[token_id].snippets):
                snippet_id = f"{token_id}:{i:04d}"
                snippet_refs_by_token[token_id].append(snippet_id)
                fh.write(json.dumps({"snippet_id": snippet_id, **snippet}, ensure_ascii=False) + "\n")

    status_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    with prepass_jsonl.open("w", encoding="utf-8") as fh:
        for token_id in range(args.new_id_start, args.new_id_end):
            cov = coverage[token_id]
            usable = len(cov.snippets)
            status, action = status_for(usable)
            status_counts[status] = status_counts.get(status, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
            row = {
                **static_info[token_id],
                "extended_firings": cov.extended_firings,
                "raw_surface_occurrences": None,
                "usable_snippets_25": min(usable, 25),
                "usable_snippets_100": min(usable, 100),
                "docs_with_firing": cov.docs_with_firing,
                "example_snippet_refs": snippet_refs_by_token[token_id][: args.example_refs_per_token],
                "status": status,
                "recommended_action": action,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_tokens = args.new_id_end - args.new_id_start
    enough_100 = status_counts.get("enough_100", 0)
    enough_25 = enough_100 + status_counts.get("enough_25", 0)
    low_lt25 = n_tokens - enough_25
    if enough_100 / n_tokens >= 0.90:
        recommended = "run_full_td_100"
    elif enough_25 / n_tokens >= 0.90:
        recommended = "run_td_25_with_flagged_tail"
    else:
        recommended = "do_not_launch_full_td_inspect_coverage"

    summary = {
        "inputs": [str(p) for p in args.input_jsonl],
        "base_tokenizer": str(args.base_tokenizer),
        "student_tokenizer": str(args.student_tokenizer),
        "new_id_start": args.new_id_start,
        "new_id_end": args.new_id_end,
        "target_extended_tokens": args.target_extended_tokens,
        "tokens_scanned": tokens_seen,
        "docs_seen": docs_seen,
        "docs_used": docs_used,
        "chars_seen": chars_seen,
        "stopped_on_budget": stopped_on_budget,
        "require_nfc": args.require_nfc,
        "non_nfc_docs": non_nfc_docs,
        "snippet_token_radius": args.snippet_token_radius,
        "snippets_per_token": args.snippets_per_token,
        "status_counts": status_counts,
        "action_counts": action_counts,
        "enough_100_fraction": enough_100 / n_tokens,
        "enough_25_fraction": enough_25 / n_tokens,
        "low_lt25_count": low_lt25,
        "recommended_next_step": recommended,
        "artifacts": {
            "coverage_jsonl": str(prepass_jsonl),
            "summary_json": str(summary_path),
            "snippet_index_jsonl": str(snippet_jsonl),
        },
        "counting_rule": "A firing means the extended tokenizer emitted the new token ID in the scanned training stream.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
