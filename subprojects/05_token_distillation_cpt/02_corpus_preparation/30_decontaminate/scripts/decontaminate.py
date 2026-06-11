#!/usr/bin/env python3
"""DCLM-style decontamination adapted for GreekMMLU, with proximity gate.

Reference: `mlfoundations/dclm` MMLU/HellaSwag contamination check
(Li et al. 2024, "DataComp-LM: In Search of the Next Generation of Training
Sets for Language Models", arXiv 2406.11794).

DCLM's MMLU rule (the baseline this adapts): a corpus document is contaminated
against an eval item iff the document contains both (a) a Q-side k-gram from
the item's question stem AND (b) at least one option-side k-gram from the
item's answer choices.

**Proximity gate** (this adaptation, beyond the DCLM paper rule): the option
side must occur NEAR and AFTER the Q-side match. Specifically, within
`--max-gap-tokens` tokens after the end of a Q-side k-gram match. This
removes false positives where the question stem appears in one passage and
an answer surface form appears in an unrelated passage of the same doc —
both of which can happen for general-knowledge Greek text without indicating
test-set leakage. Token-level matching is naturally noise-tolerant: the
Greek-NLP normalizer strips punctuation while keeping digits as tokens, so
formatting like "Q1: ... 1. Σωστό" becomes a sequence of word tokens with
short gaps, easily covered by a window of a few dozen tokens.

The Greek-language adaptation otherwise mirrors `scan_5b_decontamination.py`:

- Greek normalization (NFKC + strip combining marks + casefold + Unicode
  word tokens excluding underscore).
- Short-option fallback: for options shorter than K tokens (very common in
  Greek MCQs — T/F items have 1-token answers), we cannot form a k-gram
  from the option. Fall back to contiguous-token-subsequence presence of
  the option tokens within the proximity window after a Q-side match.

Three option-side rules emitted in parallel, all under the same proximity
gate so they're directly comparable:

- `correct_only` — option-side hit = the CORRECT option fired in-window
  after Q. This is the "doc contains Q AND the right answer adjacent"
  cheating signal we want for A-loaded items.
- `any` — option-side hit = at least one option fired in-window. DCLM
  paper rule plus proximity.
- `all` — option-side hit = every option fired in-window. Strictest;
  near-verbatim eval-item appearance.

Outputs:
- audit JSON: per-rule contamination counts, methodology metadata.
- per-item JSONL (correct_only rule): list of contaminating doc ids.

Stdlib only. No `datasets`, `numpy`, or `pandas`.

This script re-scans the corpus from scratch. It does NOT consume the
existing `hits.jsonl` from `scan_5b_decontamination.py`.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return " ".join(text.casefold().split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def normalize_doc_id(raw: Any, fallback: str) -> str:
    if raw is None:
        return fallback
    doc_id = str(raw)
    return doc_id if doc_id else fallback


def kgrams(tokens: list[str], k: int) -> Iterable[tuple[str, ...]]:
    if k <= 0 or len(tokens) < k:
        return
    for i in range(len(tokens) - k + 1):
        yield tuple(tokens[i : i + k])


def find_subsequence_position(haystack: list[str], needle: list[str], start: int = 0) -> int:
    """Return the position of the first occurrence of `needle` in `haystack[start:]`,
    or -1 if not found. Position is absolute (counts from start of haystack)."""
    n = len(needle)
    if n == 0 or start + n > len(haystack):
        return -1
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="greekmmlu")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--max-gap-tokens",
        type=int,
        default=50,
        help=(
            "Maximum token gap between the end of a Q-side k-gram match and the "
            "start of an option-side match, for options LONG enough to form a "
            "K-gram. Punctuation and most special chars are stripped during "
            "tokenization, so this gap is in normalized word tokens (numbers "
            "count as tokens). Default 50."
        ),
    )
    parser.add_argument(
        "--max-gap-tokens-short",
        type=int,
        default=5,
        help=(
            "Maximum token gap for options too SHORT for a K-gram — the "
            "standard/formulaic answers like True/False, Yes/No, single-token "
            "labels, letter combinations. These options are matched by "
            "contiguous-token-subsequence presence (substring fallback). A "
            "tight window is essential here: words like 'σωστο' or 'ναι' appear "
            "regularly in Greek text and a generous window would fire on "
            "coincidental nearby occurrences. The actual eval-scrape pattern "
            "for short answers is 'stem + brief connector + label', covered by "
            "a few tokens. Default 5."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["after", "any"],
        default="after",
        help=(
            "'after': option-side match must START at-or-after the end of a Q-side "
            "match position, within --max-gap-tokens. 'any': option-side match may "
            "occur before or after Q-side within --max-gap-tokens of either edge. "
            "Default 'after' (matches the user direction 2026-06-02)."
        ),
    )
    parser.add_argument("--progress-every-rows", type=int, default=50000)
    parser.add_argument(
        "--max-per-item-detail",
        type=int,
        default=50,
        help="Cap the number of contaminating doc ids stored per item in the detail JSONL.",
    )
    parser.add_argument(
        "--primary-rule",
        choices=[
            "correct_only",
            "q_plus_correct_and_wrong",
            "q_plus_correct_only",
            "any",
            "all",
        ],
        default="correct_only",
        help=(
            "Production rule. Default `correct_only` (union of q_plus_correct_only ∪ "
            "q_plus_correct_and_wrong): 92.4%% codex-judge precision on a 118-doc "
            "sample (2026-06-02, gpt-5.5, 250-pair audit). This is the rule used for "
            "actual corpus filtering — see --filter-output-clean. Other values are "
            "diagnostic-only and not used in production."
        ),
    )
    parser.add_argument(
        "--max-q-kgram-digit-fraction", type=float, default=0.5,
        help=(
            "Drop a Q-side k-gram if more than this fraction of its tokens are "
            "pure digits. Default 0.5 (>50%% digit tokens → drop). Prevents "
            "degenerate items like 'Ποιος είναι ο κανόνας στο μοτίβο 10, 9, 8, "
            "7, 6, 5, 4, 3;' from firing on any doc that contains the digit "
            "sequence in any language. Set to 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help=(
            "Multiprocessing pool size for the scan. Default 1 (serial). With "
            "n>1, the corpus is streamed line-by-line and per-doc work is "
            "fanned out across N worker processes. The Q-gram index is "
            "fork-shared (copy-on-write on Linux), so workers add no memory. "
            "Typical: set to $SLURM_CPUS_PER_TASK on Clariden."
        ),
    )
    parser.add_argument(
        "--mp-chunksize", type=int, default=1024,
        help="multiprocessing imap_unordered chunksize. Higher = less IPC overhead, "
             "less responsive progress prints. Default 1024.",
    )
    parser.add_argument(
        "--filter-output-clean", type=Path, default=None,
        help=(
            "If set, also stream the corpus and write a decontaminated copy (rows with "
            "doc_id NOT in the primary-rule contaminated set) to this path. This is the "
            "main production path."
        ),
    )
    parser.add_argument(
        "--filter-output-dropped", type=Path, default=None,
        help=(
            "If set together with --filter-output-clean, write the dropped rows (with "
            "an injected `_decontam_evidence` field) to this path."
        ),
    )
    return parser.parse_args()


def load_items(queries_jsonl: Path, benchmark: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with queries_jsonl.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{queries_jsonl}:{line_no}: invalid JSON ({exc})") from exc
            if row.get("benchmark") != benchmark:
                continue
            items.append(row)
    return items


def _digit_fraction(gram: tuple[str, ...]) -> float:
    """Fraction of tokens in `gram` that are pure ASCII digits ([0-9]+)."""
    if not gram:
        return 0.0
    return sum(1 for t in gram if t.isdigit()) / len(gram)


def build_item_grams(
    items: list[dict[str, Any]],
    k: int,
    max_q_kgram_digit_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build per-item k-gram sets and a per-input-issue count dict.

    Tracked input issues (counts, not failures — the items still flow through):
      - empty_question
      - empty_choices
      - missing_answer_index
      - answer_index_out_of_range
      - empty_option_tokens (any option text that normalizes to no tokens)
      - q_kgrams_all_digit_filtered (item had Q-grams but every one was
        dropped because it exceeded `max_q_kgram_digit_fraction`)

    `max_q_kgram_digit_fraction` drops Q-side k-grams whose digit-only token
    fraction strictly exceeds this. Default 0.5: any k-gram with > half
    digit tokens is dropped. Prevents items whose Q stem contains a long
    digit run (e.g. "Ποιος είναι ο κανόνας στο μοτίβο 10, 9, 8, 7, 6, 5, 4, 3;")
    from firing on any doc that happens to contain that digit sequence in any
    language.
    """
    items_with_grams: list[dict[str, Any]] = []
    issues: dict[str, int] = {
        "empty_question": 0,
        "empty_choices": 0,
        "missing_answer_index": 0,
        "answer_index_out_of_range": 0,
        "empty_option_tokens": 0,
        "q_kgrams_all_digit_filtered": 0,
    }
    for item in items:
        question_text = item.get("question") or ""
        q_tokens = tokenize(question_text)
        if not q_tokens:
            issues["empty_question"] += 1
        all_q_grams = set(kgrams(q_tokens, k))
        q_grams = {
            g for g in all_q_grams
            if _digit_fraction(g) <= max_q_kgram_digit_fraction
        }
        if all_q_grams and not q_grams:
            issues["q_kgrams_all_digit_filtered"] += 1

        choices_raw = item.get("choices") or []
        if not isinstance(choices_raw, list):
            raise RuntimeError(
                f"item {item.get('example_id')!r} has non-list 'choices' "
                f"(got {type(choices_raw).__name__}); cannot proceed"
            )
        choices = choices_raw
        if not choices:
            issues["empty_choices"] += 1
        option_tokens_list: list[list[str]] = []
        option_grams_list: list[set[tuple[str, ...]]] = []
        for choice in choices:
            t = tokenize(str(choice))
            if not t:
                issues["empty_option_tokens"] += 1
            option_tokens_list.append(t)
            option_grams_list.append(set(kgrams(t, k)))

        answer_index = item.get("answer_index")
        try:
            answer_index_int = int(answer_index) if answer_index is not None else None
        except (TypeError, ValueError):
            answer_index_int = None
        if answer_index_int is None:
            issues["missing_answer_index"] += 1
        elif not (0 <= answer_index_int < len(choices)):
            issues["answer_index_out_of_range"] += 1
            answer_index_int = None  # treat as missing for categorization

        items_with_grams.append(
            {
                "example_id": item.get("example_id"),
                "benchmark": item.get("benchmark"),
                "subject": item.get("subject"),
                "answer_index": answer_index_int,
                "n_choices": len(choices),
                "q_tokens": q_tokens,
                "q_grams": q_grams,
                "q_too_short_for_kgram": len(q_tokens) < k,
                "option_tokens": option_tokens_list,
                "option_grams": option_grams_list,
                "option_too_short_for_kgram": [len(t) < k for t in option_tokens_list],
            }
        )
    return items_with_grams, issues


def build_global_q_index(items_with_grams: list[dict[str, Any]]) -> dict[tuple[str, ...], list[int]]:
    q_gram_to_items: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)
    for item_idx, item in enumerate(items_with_grams):
        for g in item["q_grams"]:
            q_gram_to_items[g].append(item_idx)
    return q_gram_to_items


def find_q_match_positions(
    doc_tokens: list[str],
    q_gram_index: dict[tuple[str, ...], list[int]],
    k: int,
) -> dict[int, list[int]]:
    """For each item that has any Q-side match in this doc, return the list of
    start positions where the match occurred."""
    matches: dict[int, list[int]] = collections.defaultdict(list)
    if len(doc_tokens) < k:
        return matches
    for pos in range(len(doc_tokens) - k + 1):
        gram = tuple(doc_tokens[pos : pos + k])
        item_idxs = q_gram_index.get(gram)
        if item_idxs:
            for item_idx in item_idxs:
                matches[item_idx].append(pos)
    return matches


def option_hits_in_windows(
    doc_tokens: list[str],
    item: dict[str, Any],
    q_match_positions: list[int],
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
    direction: str,
) -> set[int]:
    """Return the set of option indices that fire within an allowed window
    relative to any of the item's Q-side match positions in this doc.

    Window size is per-option:
      - Options long enough to form a K-gram → `max_gap_tokens` (default 50).
      - Options too short for a K-gram (substring fallback path) →
        `max_gap_tokens_short` (default 5).

    The gap is the number of doc tokens between the END of the Q-match and the
    START of the option. So for direction='after' and gap = effective_gap, the
    option's first token can be in [q_end, q_end + effective_gap]. The substring
    check enforces this strictly; the k-gram check is slightly fuzzier for
    long options (any k-gram of the option may match anywhere in the window
    slice, which lets a chunk of the option be near Q even if the full option's
    notional start is a few tokens past the gap boundary). The fuzz is bounded
    by `opt_len - k` tokens and is only relevant for long options where that's
    a small fraction of `effective_gap`.
    """
    fired: set[int] = set()
    n_doc = len(doc_tokens)
    option_tokens = item["option_tokens"]
    option_grams = item["option_grams"]
    too_short = item["option_too_short_for_kgram"]
    n_choices = item["n_choices"]

    for opt_idx in range(n_choices):
        opt_tokens = option_tokens[opt_idx]
        opt_grams = option_grams[opt_idx]
        opt_len = len(opt_tokens)
        if opt_len == 0:
            continue

        effective_gap = max_gap_tokens_short if too_short[opt_idx] else max_gap_tokens

        for q_pos in q_match_positions:
            q_end = q_pos + k  # first doc index AFTER the Q-match

            # Valid option_start range for this q_pos:
            #   direction='after':  option_start in [q_end, q_end + effective_gap]
            #   direction='any':    option_start in [q_pos - opt_len - effective_gap, q_end + effective_gap]
            #     (the 'before' branch lets the option's end touch q_pos with up to
            #      effective_gap tokens of intervening text; combined with the
            #      'after' branch this is a single contiguous range, which also
            #      admits within-Q overlap as a side effect — acceptable for the
            #      'any' direction's intentionally permissive semantics.)
            if direction == "after":
                opt_start_lo = q_end
            else:  # "any"
                opt_start_lo = max(0, q_pos - opt_len - effective_gap)
            opt_start_hi_excl = q_end + effective_gap + 1  # exclusive
            # Clip to doc bounds: the option must fit entirely within doc_tokens.
            opt_start_hi_excl = min(opt_start_hi_excl, n_doc - opt_len + 1)
            if opt_start_lo >= opt_start_hi_excl:
                continue

            if too_short[opt_idx]:
                # Substring fallback: option appears as a contiguous subsequence
                # starting in [opt_start_lo, opt_start_hi_excl). Bounded strictly
                # by the option_start range.
                pos = find_subsequence_position(doc_tokens, opt_tokens, start=opt_start_lo)
                if pos != -1 and pos < opt_start_hi_excl:
                    fired.add(opt_idx)
                    break
            else:
                # K-gram check: any k-gram of the option appears in a window that
                # covers all valid option_start positions plus the k-gram tail.
                # For option_start = p in [opt_start_lo, opt_start_hi_excl), the
                # option's i-th k-gram (i in [0, opt_len - k]) sits at doc position
                # p + i, occupying [p + i, p + i + k - 1]. The rightmost k-gram
                # endpoint is (opt_start_hi_excl - 1) + (opt_len - k) + (k - 1)
                # = opt_start_hi_excl + opt_len - 2. The window slice must
                # therefore extend to opt_start_hi_excl + opt_len - 1 (exclusive).
                window_start = opt_start_lo
                window_end_excl = min(opt_start_hi_excl + opt_len - 1, n_doc)
                if window_end_excl - window_start < k:
                    continue
                window_grams = set(kgrams(doc_tokens[window_start:window_end_excl], k))
                if window_grams & opt_grams:
                    fired.add(opt_idx)
                    break
    return fired


def scan_corpus(
    corpus_jsonl: Path,
    items_with_grams: list[dict[str, Any]],
    q_gram_index: dict[tuple[str, ...], list[int]],
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
    direction: str,
    progress_every: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Walk the corpus once; emit per-(item, doc) evidence with proximity gate."""
    evidence: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    rows = 0
    bytes_read = 0
    last_progress = time.time()

    with corpus_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            rows += 1
            bytes_read += len(line)
            if rows % progress_every == 0:
                now = time.time()
                rate = progress_every / max(now - last_progress, 1e-9)
                print(
                    f"[scan] {rows} rows / {bytes_read/1e9:.2f} GB / {rate:.0f} rows/s",
                    flush=True,
                )
                last_progress = now

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = normalize_doc_id(row.get("doc_id"), f"row:{rows}")
            text = row.get("text") or ""
            doc_tokens = tokenize(text)
            if len(doc_tokens) < k:
                continue

            q_match_positions_per_item = find_q_match_positions(doc_tokens, q_gram_index, k)
            if not q_match_positions_per_item:
                continue

            for item_idx, positions in q_match_positions_per_item.items():
                item = items_with_grams[item_idx]
                fired = option_hits_in_windows(
                    doc_tokens, item, positions, k,
                    max_gap_tokens, max_gap_tokens_short, direction,
                )
                doc_dict = evidence[item_idx].setdefault(
                    doc_id, {"q_hit": False, "option_hits_by_idx": set()}
                )
                doc_dict["q_hit"] = True
                doc_dict["option_hits_by_idx"].update(fired)

    return evidence


# ----------------------- Multiprocessing scan path -----------------------
# Workers inherit these via fork (copy-on-write on Linux); _worker_init is
# the explicit setter when fork isn't used (e.g., 'spawn' start method).
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(items_with_grams, q_gram_index, k, max_gap_tokens,
                 max_gap_tokens_short, direction):
    _WORKER_STATE["items"] = items_with_grams
    _WORKER_STATE["q_gram_index"] = q_gram_index
    _WORKER_STATE["k"] = k
    _WORKER_STATE["max_gap"] = max_gap_tokens
    _WORKER_STATE["max_gap_short"] = max_gap_tokens_short
    _WORKER_STATE["direction"] = direction


def _scan_one_line(item: tuple[int, str]):
    """Worker: process one corpus line, return list of (item_idx, doc_id, fired_frozenset)."""
    row_no, line = item
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return []
    doc_id = normalize_doc_id(row.get("doc_id"), f"row:{row_no}")
    text = row.get("text") or ""
    k = _WORKER_STATE["k"]
    doc_tokens = tokenize(text)
    if len(doc_tokens) < k:
        return []
    q_match_positions_per_item = find_q_match_positions(
        doc_tokens, _WORKER_STATE["q_gram_index"], k,
    )
    if not q_match_positions_per_item:
        return []
    items = _WORKER_STATE["items"]
    max_gap = _WORKER_STATE["max_gap"]
    max_gap_short = _WORKER_STATE["max_gap_short"]
    direction = _WORKER_STATE["direction"]
    out = []
    for item_idx, positions in q_match_positions_per_item.items():
        item = items[item_idx]
        fired = option_hits_in_windows(
            doc_tokens, item, positions, k,
            max_gap, max_gap_short, direction,
        )
        out.append((item_idx, doc_id, frozenset(fired)))
    return out


def scan_corpus_parallel(
    corpus_jsonl: Path,
    items_with_grams: list[dict[str, Any]],
    q_gram_index: dict[tuple[str, ...], list[int]],
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
    direction: str,
    progress_every: int,
    n_workers: int,
    chunksize: int,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Multiprocessing version. Workers fork-share the read-only Q-gram index."""
    import multiprocessing as mp
    # Use 'fork' so the Q-gram index is inherited copy-on-write; 'spawn' would
    # re-pickle the whole index per worker (huge for a 5B-token corpus job).
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()  # fallback (macOS / Windows)
    evidence: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    rows = 0
    last_progress = time.time()
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(
            items_with_grams, q_gram_index, k,
            max_gap_tokens, max_gap_tokens_short, direction,
        ),
    ) as pool, corpus_jsonl.open(encoding="utf-8") as fh:
        for hits in pool.imap_unordered(_scan_one_line, enumerate(fh, start=1), chunksize=chunksize):
            rows += 1
            if rows % progress_every == 0:
                now = time.time()
                rate = progress_every / max(now - last_progress, 1e-9)
                print(
                    f"[scan] {rows} rows / {rate:.0f} rows/s (parallel n_workers={n_workers})",
                    flush=True,
                )
                last_progress = now
            if not hits:
                continue
            for item_idx, doc_id, fired in hits:
                doc_dict = evidence[item_idx].setdefault(
                    doc_id, {"q_hit": False, "option_hits_by_idx": set()}
                )
                doc_dict["q_hit"] = True
                doc_dict["option_hits_by_idx"].update(fired)
    return evidence


CATEGORY_NAMES = ("q_only", "q_plus_correct_only", "q_plus_wrong_only", "q_plus_correct_and_wrong")


def categorize_pair(answer_idx: int | None, opt_hits: set[int]) -> str:
    """Assign one of the four mutually exclusive categories to an (item, doc) pair.

    - `q_only`                     — Q stem matched, no option fired.
    - `q_plus_correct_only`        — Q + the correct option fired (no wrongs).
    - `q_plus_wrong_only`          — Q + at least one wrong option, NOT the correct one.
    - `q_plus_correct_and_wrong`   — Q + the correct option AND at least one wrong.
    """
    if not opt_hits:
        return "q_only"
    has_correct = answer_idx is not None and answer_idx in opt_hits
    has_wrong = bool(opt_hits - ({answer_idx} if answer_idx is not None else set()))
    if has_correct and has_wrong:
        return "q_plus_correct_and_wrong"
    if has_correct:
        return "q_plus_correct_only"
    return "q_plus_wrong_only"


def evaluate_audit(
    items_with_grams: list[dict[str, Any]],
    evidence: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Emit two views of the same data:

    1. Mutually exclusive `category_stats`: every (item, doc) pair where Q-side
       fires lands in exactly one of the four categories.
    2. Backward-compat `rule_stats` matching the DCLM-paper terminology:
       `correct_only` ∪ `any` ∪ `all` are union rules over the categories.
    """
    category_stats: dict[str, dict[str, Any]] = {
        name: {"contaminated_items": set(), "contaminated_pairs": 0}
        for name in CATEGORY_NAMES
    }
    rule_stats = {
        "correct_only": {"contaminated_items": set(), "contaminated_pairs": 0},
        "any": {"contaminated_items": set(), "contaminated_pairs": 0},
        "all": {"contaminated_items": set(), "contaminated_pairs": 0},
    }
    # Per-(item, doc) category, so the detail JSONL can emit a per-category
    # sample of doc ids per item.
    item_doc_category: dict[int, dict[str, str]] = collections.defaultdict(dict)

    for item_idx, item in enumerate(items_with_grams):
        n_choices = item["n_choices"]
        answer_idx = item["answer_index"]
        for doc_id, doc_ev in evidence.get(item_idx, {}).items():
            if not doc_ev["q_hit"]:
                continue
            opt_hits: set[int] = doc_ev["option_hits_by_idx"]
            category = categorize_pair(answer_idx, opt_hits)
            category_stats[category]["contaminated_items"].add(item_idx)
            category_stats[category]["contaminated_pairs"] += 1
            item_doc_category[item_idx][doc_id] = category

            # Backward-compat union rules.
            if opt_hits:
                rule_stats["any"]["contaminated_items"].add(item_idx)
                rule_stats["any"]["contaminated_pairs"] += 1
            if n_choices > 0 and len(opt_hits) == n_choices:
                rule_stats["all"]["contaminated_items"].add(item_idx)
                rule_stats["all"]["contaminated_pairs"] += 1
            if answer_idx is not None and answer_idx in opt_hits:
                rule_stats["correct_only"]["contaminated_items"].add(item_idx)
                rule_stats["correct_only"]["contaminated_pairs"] += 1

    return {
        "category_stats": category_stats,
        "rule_stats": rule_stats,
        "item_doc_category": item_doc_category,
    }


def main() -> None:
    args = parse_args()

    if not args.queries_jsonl.is_file():
        print(f"ERROR: queries JSONL not found: {args.queries_jsonl}", file=sys.stderr)
        sys.exit(2)
    if not args.corpus_jsonl.is_file():
        print(f"ERROR: corpus JSONL not found: {args.corpus_jsonl}", file=sys.stderr)
        sys.exit(2)
    if args.k < 1:
        print(f"ERROR: --k must be >= 1 (got {args.k})", file=sys.stderr)
        sys.exit(2)
    if args.max_gap_tokens < 0 or args.max_gap_tokens_short < 0:
        print(
            f"ERROR: --max-gap-tokens (got {args.max_gap_tokens}) and "
            f"--max-gap-tokens-short (got {args.max_gap_tokens_short}) must be >= 0",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.max_per_item_detail < 0:
        print(
            f"ERROR: --max-per-item-detail must be >= 0 (got {args.max_per_item_detail})",
            file=sys.stderr,
        )
        sys.exit(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[setup] loading queries from {args.queries_jsonl}", flush=True)
    items = load_items(args.queries_jsonl, args.benchmark)
    print(f"[setup] {len(items)} items for benchmark={args.benchmark}", flush=True)
    if not items:
        print(f"ERROR: no items for benchmark={args.benchmark}", file=sys.stderr)
        sys.exit(3)

    print(
        f"[setup] building per-item n-grams (K={args.k}, "
        f"max_q_kgram_digit_fraction={args.max_q_kgram_digit_fraction})",
        flush=True,
    )
    items_with_grams, input_issues = build_item_grams(
        items, args.k, args.max_q_kgram_digit_fraction,
    )
    n_q_too_short = sum(1 for it in items_with_grams if it["q_too_short_for_kgram"])
    n_opt_too_short_total = sum(sum(it["option_too_short_for_kgram"]) for it in items_with_grams)
    n_options_total = sum(it["n_choices"] for it in items_with_grams)
    print(
        f"[setup] items: {len(items_with_grams)}; "
        f"q-too-short: {n_q_too_short}; "
        f"options-too-short / total: {n_opt_too_short_total} / {n_options_total}",
        flush=True,
    )
    nonzero_issues = {k_name: v for k_name, v in input_issues.items() if v > 0}
    if nonzero_issues:
        print(f"[setup] input issues (counts, not failures): {nonzero_issues}", flush=True)

    print("[setup] building global Q-gram index", flush=True)
    q_gram_index = build_global_q_index(items_with_grams)
    print(f"[setup] q-gram index size: {len(q_gram_index)}", flush=True)

    print(
        f"[scan] scanning corpus {args.corpus_jsonl} "
        f"(max_gap_tokens={args.max_gap_tokens}, direction={args.direction})",
        flush=True,
    )
    t_scan_start = time.time()
    if args.n_workers <= 1:
        evidence = scan_corpus(
            args.corpus_jsonl,
            items_with_grams,
            q_gram_index,
            args.k,
            args.max_gap_tokens,
            args.max_gap_tokens_short,
            args.direction,
            args.progress_every_rows,
        )
    else:
        print(
            f"[scan] parallel mode: n_workers={args.n_workers} "
            f"chunksize={args.mp_chunksize}",
            flush=True,
        )
        evidence = scan_corpus_parallel(
            args.corpus_jsonl,
            items_with_grams,
            q_gram_index,
            args.k,
            args.max_gap_tokens,
            args.max_gap_tokens_short,
            args.direction,
            args.progress_every_rows,
            args.n_workers,
            args.mp_chunksize,
        )
    t_scan = time.time() - t_scan_start
    print(f"[scan] done in {t_scan:.1f} s", flush=True)

    print("[aggregate] categorizing (item, doc) pairs", flush=True)
    audit_results = evaluate_audit(items_with_grams, evidence)

    total_items = len(items_with_grams)
    generated_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Emit the headline category first so the JSON is easy to read by eye.
    CATEGORY_EMIT_ORDER = (
        "q_plus_correct_and_wrong",
        "q_plus_correct_only",
        "q_plus_wrong_only",
        "q_only",
    )
    category_summary: dict[str, Any] = {}
    for cat_name in CATEGORY_EMIT_ORDER:
        if cat_name not in audit_results["category_stats"]:
            continue
        stats = audit_results["category_stats"][cat_name]
        n_items = len(stats["contaminated_items"])
        category_summary[cat_name] = {
            "items_with_at_least_one_doc_in_category": n_items,
            "items_fraction": round(n_items / total_items, 6) if total_items else 0.0,
            "item_doc_pairs_in_category": stats["contaminated_pairs"],
        }
    # Catch any category not in the canonical order (forward-compat).
    for cat_name, stats in audit_results["category_stats"].items():
        if cat_name in category_summary:
            continue
        n_items = len(stats["contaminated_items"])
        category_summary[cat_name] = {
            "items_with_at_least_one_doc_in_category": n_items,
            "items_fraction": round(n_items / total_items, 6) if total_items else 0.0,
            "item_doc_pairs_in_category": stats["contaminated_pairs"],
        }

    rule_summary: dict[str, Any] = {}
    for rule_name, stats in audit_results["rule_stats"].items():
        n_items = len(stats["contaminated_items"])
        rule_summary[rule_name] = {
            "contaminated_items": n_items,
            "contaminated_items_fraction": round(n_items / total_items, 6) if total_items else 0.0,
            "contaminated_item_doc_pairs": stats["contaminated_pairs"],
        }

    PRIMARY_RULE_RATIONALE = {
        "correct_only": (
            "PRODUCTION RULE. Union of q_plus_correct_only ∪ q_plus_correct_and_wrong. "
            "92.4% codex-judge precision on a 118-doc sample (gpt-5.5, 250-pair audit, "
            "2026-06-02). Residual false positives are source-overlap on Greek regulation "
            "portals (opengov.gr, eur-lex, elinyae) — these docs contain the regulation "
            "text that becomes the eval question, which is appropriate to remove from the "
            "training corpus anyway. Chosen for the corpus filter step."
        ),
        "q_plus_correct_and_wrong": (
            "Diagnostic only. Strictest mutually-exclusive bucket: Q + correct option + "
            "at least one wrong option all fire in proximity windows. 98.5% codex-judge "
            "precision on 67 docs (sole outlier: opengov.gr regulation). Use to report "
            "the zero-FP cheating rate; not the filter rule."
        ),
        "q_plus_correct_only": (
            "Diagnostic only. Strict bucket: Q + correct option fired, no wrongs. 86.3% "
            "codex-judge precision (n=51)."
        ),
        "any": "DCLM paper rule. Diagnostic only; over-flags wrong-only / topic-overlap pages.",
        "all": "Strictest variant: every option fired. Subset of q_plus_correct_and_wrong.",
    }
    if args.primary_rule in ("q_plus_correct_and_wrong", "q_plus_correct_only"):
        primary_block = category_summary.get(args.primary_rule, {})
        primary_n = primary_block.get("items_with_at_least_one_doc_in_category", 0)
        primary_frac = primary_block.get("items_fraction", 0.0)
        primary_pairs = primary_block.get("item_doc_pairs_in_category", 0)
        primary_source = "results_by_category"
    else:
        primary_block = rule_summary.get(args.primary_rule, {})
        primary_n = primary_block.get("contaminated_items", 0)
        primary_frac = primary_block.get("contaminated_items_fraction", 0.0)
        primary_pairs = primary_block.get("contaminated_item_doc_pairs", 0)
        primary_source = "results_by_rule"
    headline = {
        "rule": args.primary_rule,
        "source": primary_source,
        "rationale": PRIMARY_RULE_RATIONALE.get(args.primary_rule, ""),
        "contaminated_items": primary_n,
        "contaminated_items_fraction": primary_frac,
        "item_doc_pairs": primary_pairs,
    }

    summary = {
        "schema": "greekmmlu-dclm-decontam-v3-categorized",
        "generated_utc": generated_utc,
        "benchmark": args.benchmark,
        "k": args.k,
        "max_gap_tokens": args.max_gap_tokens,
        "max_gap_tokens_short": args.max_gap_tokens_short,
        "direction": args.direction,
        "headline": headline,
        "inputs": {
            "queries_jsonl": str(args.queries_jsonl),
            "corpus_jsonl": str(args.corpus_jsonl),
        },
        "method": {
            "name": "DCLM-adapted MMLU contamination check (Greek-language adaptation, proximity-gated)",
            "rule": (
                "Q-side: any K-gram of the item's normalized question appears in the "
                "normalized doc tokens. If so, record the doc positions where it matched.\n"
                "Option-side: for each option and each Q-match position, check the "
                "option in a per-option window. The 'gap' is the number of doc tokens "
                "between the END of the Q-match and the START of the option. With "
                "--direction=after, option_start ∈ [Q_end, Q_end + effective_gap]. "
                "With --direction=any, option_start can also be effective_gap tokens "
                "before the Q (the before/after ranges merge with the within-Q overlap "
                "case included). For options shorter than K tokens, use substring "
                "(contiguous-token-subsequence) presence inside that range with the "
                "tight `--max-gap-tokens-short` window; otherwise use k-gram "
                "intersection inside the window, with `--max-gap-tokens`. K-gram check "
                "has a bounded fuzz of `opt_len - k` tokens past the gap boundary "
                "(the matched k-gram can be any of the option's chunks, so the implied "
                "option_start may sit slightly outside the strict range); this fuzz is "
                "negligible relative to the long-option window default of 50.\n"
                "Categorization: for each (item, doc) pair where Q-side fired, assign "
                "exactly one of {q_only, q_plus_correct_only, q_plus_wrong_only, "
                "q_plus_correct_and_wrong} based on which option indices fired."
            ),
            "categories_emitted": [
                "q_only: Q stem matched in doc, no option fired in window (often general-knowledge stem overlap; for A-loaded items, eval-format scaffold without a specific item)",
                "q_plus_correct_only: Q + the correct option fired in window (no wrongs); strongest direct cheating signal",
                "q_plus_wrong_only: Q + at least one wrong option fired (not the correct); eval scaffolding without the answer key, or wrong-answer discussion",
                "q_plus_correct_and_wrong: Q + correct + at least one wrong; near-verbatim eval-item scrape with options visible",
            ],
            "rules_emitted_for_backcompat": [
                "correct_only: q_plus_correct_only ∪ q_plus_correct_and_wrong (DCLM-adapted; the cheating signal proper)",
                "any: any of the three non-q_only categories (DCLM paper rule)",
                "all: every option fired (strictest)",
            ],
            "normalization": "NFKC + strip combining marks + casefold + Unicode word tokens excluding underscore",
            "tokenization_note": "punctuation is stripped during tokenization; digits remain as tokens; --max-gap-tokens is measured in normalized word tokens",
            "reference": "mlfoundations/dclm (Li et al. 2024, arXiv 2406.11794), MMLU/HellaSwag contamination check; proximity gate added per user direction 2026-06-02",
        },
        "totals": {
            "items": total_items,
            "items_q_too_short_for_kgram": n_q_too_short,
            "items_q_too_short_note": (
                "Items whose normalized question has fewer than K tokens cannot form "
                "a Q-side k-gram and are therefore invisible to this rule. They are "
                "counted here so the audit reader knows the unmeasurable subset."
            ),
            "options_too_short_for_kgram_total": n_opt_too_short_total,
            "options_total": n_options_total,
        },
        "input_issues": input_issues,
        "results_by_category": category_summary,
        "results_by_rule": rule_summary,
        "scan": {
            "wall_seconds": round(t_scan, 2),
            "total_seconds": round(time.time() - t_start, 2),
            "progress_every_rows": args.progress_every_rows,
        },
    }

    audit_path = args.output_dir / f"greekmmlu_dclm_audit_{generated_utc}.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    item_doc_category = audit_results["item_doc_category"]
    detail_path = args.output_dir / f"greekmmlu_dclm_per_item_categories_{generated_utc}.jsonl"
    # Sort the per-item records by example_id so the JSONL is deterministic and
    # easy to compare across runs / re-runs with different parameters.
    sorted_item_indices = sorted(
        (idx for idx, dt in item_doc_category.items() if dt),
        key=lambda idx: str(items_with_grams[idx].get("example_id") or ""),
    )
    with detail_path.open("w", encoding="utf-8") as out:
        for item_idx in sorted_item_indices:
            doc_to_cat = item_doc_category[item_idx]
            item = items_with_grams[item_idx]
            by_category: dict[str, list[str]] = collections.defaultdict(list)
            for doc_id, category in doc_to_cat.items():
                by_category[category].append(doc_id)
            out.write(
                json.dumps(
                    {
                        "example_id": item["example_id"],
                        "subject": item["subject"],
                        "answer_index": item["answer_index"],
                        "n_choices": item["n_choices"],
                        "total_q_firing_docs": len(doc_to_cat),
                        "categories": {
                            category: {
                                "n_docs": len(doc_ids),
                                "sample_doc_ids": sorted(set(doc_ids))[: args.max_per_item_detail],
                            }
                            for category, doc_ids in by_category.items()
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---- Build the FULL contaminated-doc set for the primary rule ----
    # (per_item JSONL caps sample_doc_ids; this file lists every doc_id.)
    RULE_TO_CATEGORIES = {
        "correct_only":             {"q_plus_correct_only", "q_plus_correct_and_wrong"},
        "q_plus_correct_and_wrong": {"q_plus_correct_and_wrong"},
        "q_plus_correct_only":      {"q_plus_correct_only"},
        "any":                      {"q_plus_correct_only", "q_plus_wrong_only", "q_plus_correct_and_wrong"},
    }
    rule_categories = RULE_TO_CATEGORIES.get(args.primary_rule)
    contaminated_doc_evidence: dict[str, list[dict]] = collections.defaultdict(list)
    if rule_categories is not None:
        for item_idx, doc_to_cat in item_doc_category.items():
            item = items_with_grams[item_idx]
            for doc_id, category in doc_to_cat.items():
                if category in rule_categories:
                    contaminated_doc_evidence[doc_id].append({
                        "example_id": item["example_id"],
                        "subject": item["subject"],
                        "category": category,
                    })
    contaminated_doc_ids = set(contaminated_doc_evidence.keys())
    contaminated_ids_path = args.output_dir / f"contaminated_doc_ids_{args.primary_rule}_{generated_utc}.txt"
    contaminated_ids_path.write_text(
        "".join(did + "\n" for did in sorted(contaminated_doc_ids)),
        encoding="utf-8",
    )
    print(
        f"[contaminated-ids] rule={args.primary_rule}  n_doc_ids={len(contaminated_doc_ids)}  "
        f"path={contaminated_ids_path}",
        flush=True,
    )

    # ---- Optional: stream the corpus a second time and write decontaminated copy ----
    filter_summary: dict[str, Any] | None = None
    if args.filter_output_clean is not None:
        if args.primary_rule == "all":
            print(
                "ERROR: --filter-output-clean requires --primary-rule != 'all' "
                "(the 'all' rule requires per-evidence inspection not implemented in filter mode).",
                file=sys.stderr,
            )
            sys.exit(4)
        args.filter_output_clean.parent.mkdir(parents=True, exist_ok=True)
        if args.filter_output_dropped is not None:
            args.filter_output_dropped.parent.mkdir(parents=True, exist_ok=True)
        t_filter_start = time.time()
        n_in = n_clean = n_drop = 0
        fdrop = (
            args.filter_output_dropped.open("w", encoding="utf-8")
            if args.filter_output_dropped is not None
            else None
        )
        try:
            with args.corpus_jsonl.open(encoding="utf-8") as fin, \
                 args.filter_output_clean.open("w", encoding="utf-8") as fclean:
                for line in fin:
                    if not line.strip():
                        continue
                    n_in += 1
                    d = json.loads(line)
                    did = normalize_doc_id(d.get("doc_id"), f"row:{n_in}")
                    if did in contaminated_doc_ids:
                        n_drop += 1
                        if fdrop is not None:
                            d["_decontam_evidence"] = {
                                "rule": args.primary_rule,
                                "matches": contaminated_doc_evidence[did],
                            }
                            fdrop.write(json.dumps(d, ensure_ascii=False) + "\n")
                    else:
                        n_clean += 1
                        fclean.write(line if line.endswith("\n") else line + "\n")
                    if n_in % max(args.progress_every_rows, 1) == 0:
                        print(
                            f"[filter] rows={n_in}  clean={n_clean}  dropped={n_drop}",
                            flush=True,
                        )
        finally:
            if fdrop is not None:
                fdrop.close()
        t_filter = time.time() - t_filter_start
        filter_summary = {
            "rule": args.primary_rule,
            "input_rows": n_in,
            "clean_rows": n_clean,
            "dropped_rows": n_drop,
            "clean_path": str(args.filter_output_clean),
            "dropped_path": (
                str(args.filter_output_dropped) if args.filter_output_dropped else None
            ),
            "wall_seconds": round(t_filter, 2),
        }
        # Patch the audit JSON in place so the audit has the filter outcome too.
        summary["filter"] = filter_summary
        audit_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[filter] done in {t_filter:.1f} s  input={n_in}  clean={n_clean}  dropped={n_drop}  "
            f"clean_path={args.filter_output_clean}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "audit_json": str(audit_path),
                "per_item_categories_jsonl": str(detail_path),
                "contaminated_doc_ids_txt": str(contaminated_ids_path),
                "filter": filter_summary,
                "k": args.k,
                "max_gap_tokens": args.max_gap_tokens,
                "max_gap_tokens_short": args.max_gap_tokens_short,
                "direction": args.direction,
                "headline": headline,
                "results_by_category": category_summary,
                "results_by_rule": rule_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
