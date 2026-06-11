#!/usr/bin/env python3
"""Codex review of v2 decontamination matches — per-(item, doc) pair, with
the algorithm's exact match positions wrapped inline in the doc body.

Why this exists (vs codex_judge_contamination.py):

  codex_judge_contamination.py samples docs across the four v2 categories
  and judges each doc holistically. That's good for validating the rule
  CHOICE on a stratified sample. It does NOT verify that the algorithm's
  specific MATCH inside the doc is real cheating — codex sees the doc
  and the item but has to find the match itself.

  This script:
    1. Iterates (item, doc) PAIRS, not docs.
    2. Re-derives the algorithm's exact match positions in the doc
       (Q-stem k-gram hits + per-option matches).
    3. Annotates the NFKC-normalized doc body inline with
         <match kind="q_stem">…</match>
         <match kind="correct_option">…</match>
         <match kind="wrong_option">…</match>
       so codex sees WHAT the algorithm matched, in context.
    4. Asks codex (gpt-5.5) to rate whether the match is real cheating.

Default: review EVERY (item, doc) pair under --rule (default `correct_only`).
For the future full-pipeline decontamination (HPLT + GlossAPI + replay +
code + math), set --n-sample 1000 to review a random sample — this is the
quality gate documented in ../README.md § "Review requirement".

Stdlib-only. CPU-only. Codex calls go through the OpenAI API (paid).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# Reuse the algorithm helpers from the production scanner.
from decontaminate import (  # noqa: E402
    normalize_text,
    kgrams,
    build_item_grams,
    _digit_fraction,
)


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
RULE_TO_CATEGORIES = {
    "correct_only":             {"q_plus_correct_only", "q_plus_correct_and_wrong"},
    "q_plus_correct_and_wrong": {"q_plus_correct_and_wrong"},
    "q_plus_correct_only":      {"q_plus_correct_only"},
    "any":                      {"q_plus_correct_only", "q_plus_wrong_only", "q_plus_correct_and_wrong"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries-jsonl", type=Path, required=True,
                   help="GreekMMLU queries JSONL (one item per line).")
    p.add_argument("--corpus-jsonl", type=Path, required=True,
                   help="Docs JSONL keyed by doc_id (subset is fine — e.g. hplt_b1_5b_dropped.jsonl).")
    p.add_argument("--per-item-jsonl", type=Path, required=True,
                   help="v2 audit per-item categories JSONL (greekmmlu_dclm_per_item_categories_*.jsonl).")
    p.add_argument("--output-jsonl", type=Path, required=True,
                   help="Where to write per-pair judgments (incremental, one line per call).")
    p.add_argument("--rule", default="correct_only",
                   choices=sorted(RULE_TO_CATEGORIES.keys()),
                   help="Which v2 rule to review. Default correct_only.")
    p.add_argument("--n-sample", type=int, default=0,
                   help="If >0, random-sample this many pairs from the rule's match set. "
                        "Default 0 = review ALL matches under --rule. "
                        "Set to 1000 for the full-pipeline quality gate.")
    p.add_argument("--seed", type=int, default=20260603,
                   help="Random seed for sampling.")
    p.add_argument("--k", type=int, default=8,
                   help="K-gram size (must match the scan that produced --per-item-jsonl).")
    p.add_argument("--max-gap-tokens", type=int, default=50)
    p.add_argument("--max-gap-tokens-short", type=int, default=5)
    p.add_argument("--direction", choices=["after", "any"], default="after")
    p.add_argument("--max-q-kgram-digit-fraction", type=float, default=0.5)
    p.add_argument("--benchmark", default="greekmmlu")
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--max-workers", type=int, default=6,
                   help="Concurrent codex exec calls. codex has API rate limits; "
                        "don't exceed 8 without monitoring.")
    p.add_argument("--retry", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=10.0)
    p.add_argument("--max-doc-chars", type=int, default=0,
                   help="Optional cap on annotated doc body chars (0 = no cap).")
    p.add_argument("--smoke", type=int, default=0,
                   help="If >0, only run this many pairs serially (smoke test).")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return (normalized_tokens, char_spans_in_normalized_text). The spans
    index into normalize_text(text), so annotations are inserted into the
    NORMALIZED doc body (lowercase, monotonic Greek)."""
    if not text:
        return [], []
    normalized = normalize_text(text)
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in TOKEN_RE.finditer(normalized):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def derive_match_positions(
    doc_text: str,
    item: dict[str, Any],
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
    direction: str,
    answer_idx: int,
) -> dict[str, Any]:
    """Re-derive Q-side + option-side match positions for one (item, doc) pair.

    Returns char-position spans into the NORMALIZED doc body, suitable for
    inline <match kind="…">…</match> annotation.
    """
    normalized = normalize_text(doc_text)
    doc_tokens, token_spans = tokenize_with_spans(doc_text)
    n_doc = len(doc_tokens)
    q_grams = item["q_grams"]
    # Q-side: scan once, record every position where a Q gram fires.
    q_match_positions: list[int] = []
    if doc_tokens and q_grams:
        for i in range(n_doc - k + 1):
            gram = tuple(doc_tokens[i:i + k])
            if gram in q_grams:
                q_match_positions.append(i)

    q_match_spans: list[tuple[int, int]] = []
    for q_pos in q_match_positions:
        cs = token_spans[q_pos][0]
        ce = token_spans[q_pos + k - 1][1]
        q_match_spans.append((cs, ce))

    # Option-side: for each option, walk every Q-match position and record
    # the FIRST hit position (matches the algorithm's early-break semantics).
    option_match_spans: list[dict[str, Any]] = []
    fired_options: set[int] = set()
    for opt_idx, opt_tokens in enumerate(item["option_tokens"]):
        if not opt_tokens:
            continue
        opt_len = len(opt_tokens)
        is_short = opt_len < k
        effective_gap = max_gap_tokens_short if is_short else max_gap_tokens
        opt_grams = item["option_grams"][opt_idx]
        found = False
        for q_pos in q_match_positions:
            if found:
                break
            q_end = q_pos + k
            opt_start_lo = max(0, q_end - effective_gap) if direction == "any" else q_end
            opt_start_hi_excl = q_end + effective_gap + 1
            if is_short:
                # Substring path: walk every candidate start, check if doc[start:start+opt_len] == opt_tokens.
                for pos in range(opt_start_lo, opt_start_hi_excl):
                    if pos < 0 or pos + opt_len > n_doc:
                        continue
                    if doc_tokens[pos:pos + opt_len] == opt_tokens:
                        cs = token_spans[pos][0]
                        ce = token_spans[pos + opt_len - 1][1]
                        option_match_spans.append({
                            "option_idx": opt_idx,
                            "is_correct": opt_idx == answer_idx,
                            "char_start": cs,
                            "char_end": ce,
                            "token_position": pos,
                            "matched_text": normalized[cs:ce],
                            "path": "substring",
                        })
                        fired_options.add(opt_idx)
                        found = True
                        break
            else:
                # k-gram path: window from opt_start_lo to opt_start_hi_excl + opt_len - 1.
                window_start = opt_start_lo
                window_end_excl = min(opt_start_hi_excl + opt_len - 1, n_doc)
                if window_end_excl - window_start < k:
                    continue
                for pos in range(window_start, window_end_excl - k + 1):
                    gram = tuple(doc_tokens[pos:pos + k])
                    if gram in opt_grams:
                        cs = token_spans[pos][0]
                        ce = token_spans[pos + k - 1][1]
                        option_match_spans.append({
                            "option_idx": opt_idx,
                            "is_correct": opt_idx == answer_idx,
                            "char_start": cs,
                            "char_end": ce,
                            "token_position": pos,
                            "matched_text": normalized[cs:ce],
                            "path": "kgram",
                        })
                        fired_options.add(opt_idx)
                        found = True
                        break

    return {
        "doc_normalized": normalized,
        "q_match_spans": q_match_spans,
        "q_match_token_positions": q_match_positions,
        "option_match_spans": option_match_spans,
        "fired_option_indices": sorted(fired_options),
    }


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or touching (start, end) intervals."""
    if not intervals:
        return []
    s_ints = sorted(intervals)
    merged: list[tuple[int, int]] = [s_ints[0]]
    for s, e in s_ints[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def annotate_doc(
    doc_normalized: str,
    q_spans: list[tuple[int, int]],
    option_spans: list[dict[str, Any]],
) -> str:
    """Insert `<match kind="…">…</match>` markers inline into the normalized
    doc body at the algorithm's exact match positions.

    Q-stem spans are MERGED across overlapping sliding-window k-gram hits so
    we don't emit dozens of redundant nested `<match kind="q_stem">` tags
    when the Q-gram fires many times consecutively.

    Q-stem and option spans can still overlap (e.g. a short correct-option
    token landing inside the Q-stem span). Markup nests cleanly because open
    tags sort before close tags at the same position.
    """
    merged_q = _merge_intervals(q_spans)
    events: list[tuple[int, int, str]] = []
    for s, e in merged_q:
        events.append((s, 0, '<match kind="q_stem">'))
        events.append((e, 1, "</match>"))
    for opt in option_spans:
        kind = "correct_option" if opt["is_correct"] else "wrong_option"
        events.append((opt["char_start"], 0, f'<match kind="{kind}">'))
        events.append((opt["char_end"], 1, "</match>"))
    # Sort by (position, open/close): open tags before close tags at same pos.
    events.sort(key=lambda x: (x[0], x[1]))
    parts: list[str] = []
    last = 0
    for pos, _, tag in events:
        parts.append(doc_normalized[last:pos])
        parts.append(tag)
        last = pos
    parts.append(doc_normalized[last:])
    return "".join(parts)


def build_prompt(
    item: dict[str, Any],
    pair: dict[str, Any],
    annotated_doc: str,
    match_info: dict[str, Any],
    max_doc_chars: int,
) -> str:
    """Per-(item, doc) pair codex prompt with match-annotated doc body."""
    choices_text = "\n".join(f"  {chr(0x391 + i)}. {c}" for i, c in enumerate(item["choices"]))
    answer_text = item["choices"][int(item["answer_index"])]

    doc_size_note = ""
    if max_doc_chars > 0 and len(annotated_doc) > max_doc_chars:
        annotated_doc = annotated_doc[:max_doc_chars] + "\n\n[TRUNCATED]"
        doc_size_note = f" (truncated at {max_doc_chars} chars)"

    n_q = len(match_info["q_match_spans"])
    fired_opts = match_info["fired_option_indices"]
    fired_summary = ", ".join(
        f"option {chr(0x391 + i)} ({'correct' if i == int(item['answer_index']) else 'wrong'})"
        for i in fired_opts
    ) or "(none)"

    return f"""## Task

You are auditing a Greek-language test-set decontamination pipeline. Our
algorithm flagged THIS document as containing a specific match against
THIS GreekMMLU benchmark item. The match positions are wrapped inline
in the document body below using `<match kind="...">…</match>` tags.
Your job is to rate whether the algorithm's specific match represents
real test-set contamination.

You are NOT being asked "does cheating exist anywhere in this doc?".
You are being asked: "is the algorithm's MARKED match a real
contamination signal?"

## Algorithm summary

- Q-side: K-gram (K=8) of the question stem matched somewhere in the doc.
- Option-side: within a per-option proximity window after each Q match,
  short formulaic options (T/F, letter labels) use a 5-token substring
  window; longer options use a 50-token k-gram window.
- Match wrapping: every Q-side hit is wrapped as
  `<match kind="q_stem">…</match>` and every fired option as
  `<match kind="correct_option">…</match>` or
  `<match kind="wrong_option">…</match>` at the algorithm's exact
  matched span.

For THIS (item, doc) pair the algorithm reports:
- Q-stem matches in doc: {n_q}
- Options fired (in proximity): {fired_summary}
- v2 category: {pair["v2_category"]}

## Corpus context

The document was drawn from the 5B-token Apertus Greek CPT training
pool (mostly `greek_hplt_clean60`, the Greek HPLT web crawl; the same
pool also includes math/code/replay slices). Decontamination removes
documents that let the model memorize a GreekMMLU question → answer
mapping during training.

NOTE: a document being the primary authoritative source of the
underlying Greek fact (regulation, constitutional text, religious
document) is NOT contamination — that is exactly what Greek CPT is
for. Distinguish these cases.

## Rating rubric

**Rating 5 — DEFINITE_CHEATING.** The marked match is inside clear
exam scaffolding (numbered/lettered options, answer key, MCQ
sequence). A model trained on this doc could memorize Q→A directly.

**Rating 4 — STRONG_CHEATING.** The marked match contains the Q stem
verbatim/close-paraphrase AND the correct answer adjacent, but the
format is not full exam scaffolding (could be a study note, flowing
Q+A, answer key without explicit option labels). Q→A still
directly recoverable.

**Rating 3 — SOURCE_OVERLAP.** The doc is an authoritative primary
source of the regulation/fact/text the question is about. The correct
answer is present but in non-exam form (regulatory clauses, official
prose). Training on this learns the underlying Greek fact from its
source, not a Q→A mapping.

**Rating 2 — TOPIC_OVERLAP.** The doc discusses the same topic as the
question but does NOT contain the question or the correct answer
verbatim/adjacent. The k-gram match is real overlap but doesn't
expose the Q→A mapping.

**Rating 1 — NO_OVERLAP.** The marked match is coincidental —
generic-vocabulary k-gram overlap, no semantic connection.

## The benchmark item

Subject: {item.get("subject", "")}
example_id: {item.get("example_id", "")}

Question:
{item["question"]}

Choices:
{choices_text}

Correct answer (index {item["answer_index"]}): {answer_text}

## The document (NFKC-normalized, lowercase, monotonic; matches wrapped inline)

Document length: {len(annotated_doc)} characters{doc_size_note}

[BEGIN DOCUMENT]
{annotated_doc}
[END DOCUMENT]

## Output

Return a single JSON object matching the output schema. The
`reasoning` field must specifically discuss the MARKED match (not
just the doc in general): is the Q-stem mark inside exam scaffolding?
is the option mark the actual answer key, or coincidental? Cite a
short Greek snippet from the marked region.
"""


def parse_tokens_from_stderr(stderr: str) -> dict[str, Any]:
    """codex prints token counts to stderr like 'tokens used: input X output Y'."""
    pat = re.compile(r"tokens used:\s*input\s*(\d+)\s*output\s*(\d+)", re.IGNORECASE)
    m = pat.search(stderr or "")
    if not m:
        return {}
    return {"tokens_in": int(m.group(1)), "tokens_out": int(m.group(2))}


def codex_exec_judge(
    prompt: str,
    model: str,
    effort: str,
    schema_path: Path,
    instructions_path: Path,
    out_file: Path,
    retries: int,
    retry_sleep: float,
) -> tuple[bool, dict[str, Any], str, dict[str, Any]]:
    """Submit one prompt to codex exec. Returns (ok, parsed_json, raw_stderr, usage)."""
    cmd = [
        "codex", "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--output-schema", str(schema_path),
        "-o", str(out_file),
        "-m", model,
        "-c", f"model_reasoning_effort={effort}",
        "-c", f"model_instructions_file={instructions_path}",
        "-",
    ]
    last_err = ""
    usage = {}
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=300, check=False,
            )
            last_err = r.stderr
            usage = parse_tokens_from_stderr(r.stderr)
            if r.returncode != 0:
                if attempt < retries:
                    time.sleep(retry_sleep)
                    continue
                return False, {}, r.stderr, usage
            if not out_file.exists():
                if attempt < retries:
                    time.sleep(retry_sleep)
                    continue
                return False, {}, "no output file written", usage
            try:
                parsed = json.loads(out_file.read_text())
            except json.JSONDecodeError as e:
                if attempt < retries:
                    time.sleep(retry_sleep)
                    continue
                return False, {}, f"json parse failed: {e}", usage
            return True, parsed, r.stderr, usage
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            return False, {}, "timeout", usage
    return False, {}, last_err, usage


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading queries: {args.queries_jsonl}", flush=True)
    raw_items = load_jsonl(args.queries_jsonl)
    raw_items = [it for it in raw_items if it.get("benchmark") == args.benchmark]
    print(f"[setup] {len(raw_items)} items for benchmark={args.benchmark}", flush=True)

    print(f"[setup] building items_with_grams (K={args.k}, max_q_kgram_digit_fraction={args.max_q_kgram_digit_fraction})", flush=True)
    items_with_grams, issues = build_item_grams(raw_items, args.k, args.max_q_kgram_digit_fraction)
    items_by_id = {it["example_id"]: it for it in items_with_grams}
    raw_items_by_id = {it["example_id"]: it for it in raw_items}

    print(f"[setup] loading corpus subset: {args.corpus_jsonl}", flush=True)
    corpus = load_jsonl(args.corpus_jsonl)
    corpus_by_id = {d.get("doc_id"): d for d in corpus}
    print(f"[setup] {len(corpus_by_id)} docs available", flush=True)

    print(f"[setup] loading per-item categories: {args.per_item_jsonl}", flush=True)
    per_item = load_jsonl(args.per_item_jsonl)

    target_cats = RULE_TO_CATEGORIES[args.rule]
    pairs: list[dict[str, Any]] = []
    for rec in per_item:
        for cat, info in rec["categories"].items():
            if cat not in target_cats:
                continue
            for doc_id in info["sample_doc_ids"]:
                pairs.append({
                    "example_id": rec["example_id"],
                    "subject": rec["subject"],
                    "answer_index": rec["answer_index"],
                    "v2_category": cat,
                    "doc_id": doc_id,
                })
    total_pairs = len(pairs)
    print(f"[setup] {total_pairs} (item, doc) pairs under rule={args.rule}", flush=True)

    if args.n_sample > 0 and args.n_sample < total_pairs:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
        pairs = pairs[:args.n_sample]
        print(f"[setup] random-sampled {len(pairs)} pairs (seed={args.seed})", flush=True)

    if args.smoke > 0:
        pairs = pairs[:args.smoke]
        print(f"[setup] smoke mode: {len(pairs)} pairs serially", flush=True)

    schema_path = SCRIPT_DIR / "codex_judge_output_schema.json"
    instructions_path = SCRIPT_DIR / "codex_judge_instructions.md"
    assert schema_path.exists(), schema_path
    assert instructions_path.exists(), instructions_path

    write_lock = threading.Lock()
    out_fh = args.output_jsonl.open("a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    n_failed = 0
    n_total = len(pairs)

    def run_one(idx: int, pair: dict[str, Any]) -> dict[str, Any]:
        nonlocal n_done, n_failed
        item_grams = items_by_id.get(pair["example_id"])
        raw_item = raw_items_by_id.get(pair["example_id"])
        doc = corpus_by_id.get(pair["doc_id"])
        rec: dict[str, Any] = {
            "idx": idx,
            "example_id": pair["example_id"],
            "doc_id": pair["doc_id"],
            "v2_category": pair["v2_category"],
            "subject": pair["subject"],
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
        }
        if item_grams is None or raw_item is None:
            rec.update(ok=False, error="item not found")
            return rec
        if doc is None:
            rec.update(ok=False, error="doc not found in --corpus-jsonl")
            return rec
        doc_text = doc.get("text") or ""
        match_info = derive_match_positions(
            doc_text, item_grams, args.k,
            args.max_gap_tokens, args.max_gap_tokens_short, args.direction,
            int(item_grams["answer_index"]),
        )
        if not match_info["q_match_spans"]:
            rec.update(
                ok=False,
                error="no q-stem match re-derived (algorithm parameter mismatch?)",
                fired_option_indices=match_info["fired_option_indices"],
            )
            return rec
        annotated = annotate_doc(
            match_info["doc_normalized"],
            match_info["q_match_spans"],
            match_info["option_match_spans"],
        )
        # build_prompt needs the original question + choices text (not the
        # tokenized grams), so pass the RAW item.
        prompt = build_prompt(raw_item, pair, annotated, match_info, args.max_doc_chars)
        out_dir = Path(tempfile.mkdtemp(prefix="codex_review_"))
        out_file = out_dir / "judge.json"
        t_call_start = time.time()
        ok, parsed, stderr, usage = codex_exec_judge(
            prompt, args.model, args.reasoning_effort,
            schema_path, instructions_path,
            out_file, args.retry, args.retry_sleep,
        )
        wall = time.time() - t_call_start
        rec.update(
            ok=ok,
            wall_seconds=round(wall, 2),
            judgment=parsed,
            usage=usage,
            url=doc.get("url"),
            n_q_match_spans=len(match_info["q_match_spans"]),
            fired_option_indices=match_info["fired_option_indices"],
        )
        if not ok:
            rec["error"] = stderr[-400:] if stderr else "unknown"
        try:
            out_file.unlink()
            out_dir.rmdir()
        except OSError:
            pass
        return rec

    n_workers = 1 if args.smoke > 0 else args.max_workers
    print(f"[run] firing {n_total} codex calls with max_workers={n_workers}, model={args.model}, effort={args.reasoning_effort}", flush=True)

    def emit(rec: dict[str, Any]) -> None:
        nonlocal n_done, n_failed
        n_done += 1
        if not rec.get("ok"):
            n_failed += 1
        with write_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
        eta = (time.time() - t0) / max(n_done, 1) * (n_total - n_done)
        rating = rec.get("judgment", {}).get("rating", "?")
        print(
            f"[run] {n_done}/{n_total}  ok={rec.get('ok')}  rating={rating}  "
            f"cat={rec.get('v2_category')}  secs={rec.get('wall_seconds', 0)}  eta≈{eta:.0f}s",
            flush=True,
        )

    if n_workers == 1:
        for idx, pair in enumerate(pairs):
            emit(run_one(idx, pair))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(run_one, i, p) for i, p in enumerate(pairs)]
            for fut in concurrent.futures.as_completed(futures):
                emit(fut.result())

    out_fh.close()
    print(f"[done] {n_total} pairs in {time.time()-t0:.0f}s, {n_failed} failed.", flush=True)
    print(f"[done] results: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
