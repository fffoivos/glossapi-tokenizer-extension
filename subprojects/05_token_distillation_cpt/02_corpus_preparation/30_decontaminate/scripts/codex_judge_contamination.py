#!/usr/bin/env python3
"""Codex-as-judge contamination rating for v2 DCLM hits.

Pipeline:

1. Read the v2 audit's per-item categories JSONL.
2. Stratified-sample N (item, doc) pairs across the four categories.
3. For each pair, build a prompt containing:
     - the algorithm context (DCLM-adapted Greek MCQ decontamination)
     - the corpus context (Greek HPLT web docs from the Apertus 5B CPT pool)
     - the audit goal (judge real test-set contamination vs source overlap vs topic overlap)
     - the 5-tier rating rubric
     - the question + choices + correct answer
     - the FULL document text
4. Call `codex exec` in parallel with --output-schema enforced; capture
   the model's JSON response per pair.
5. Write results incrementally to an output JSONL.
6. Print aggregate distribution by (v2 category × judge rating).

Per project memory:
- `feedback_codex_strip_instructions`: pass `-c model_instructions_file=<path>`
  and `--ignore-user-config` so we drop the codex agent prompt and
  $CODEX_HOME/AGENTS.md baggage.
- `feedback_judge_models_for_benchmarking`: gpt-5.5 is the production
  judge.

Stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-per-item-jsonl", type=Path, required=True,
                        help="v2 audit per-item JSONL (greekmmlu_dclm_per_item_categories_*.jsonl)")
    parser.add_argument("--corpus-jsonl", type=Path, required=True,
                        help="Corpus JSONL the v2 audit ran against (doc_id, text, url)")
    parser.add_argument("--queries-jsonl", type=Path, required=True,
                        help="Queries JSONL (question, choices, answer_index, example_id)")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--n-per-category", type=int, default=25,
                        help="How many pairs to sample from each v2 category (default 25, total 100). "
                             "Capped at availability per category.")
    parser.add_argument("--exclude-judged-jsonl", type=Path, default=None,
                        help="Optional prior judge_output JSONL. (example_id, doc_id) pairs already "
                             "judged there are excluded from sampling.")
    parser.add_argument("--n-by-category", default="",
                        help="Optional per-category counts overriding --n-per-category. Format: "
                             "'q_only=38,q_plus_correct_only=26,q_plus_wrong_only=38,q_plus_correct_and_wrong=48'. "
                             "Categories not listed fall back to --n-per-category.")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium",
                        help="codex -c reasoning_effort. 'medium' is a good default for judge work; "
                             "'low' for smoke tests; 'high' for the final pass.")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Concurrent codex exec calls (default 6). codex has API rate limits; "
                             "do not exceed 8 without monitoring for throttling.")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--smoke", type=int, default=0,
                        help="If >0, run only this many pairs serially as a smoke test.")
    parser.add_argument("--max-doc-chars", type=int, default=0,
                        help="Optional cap on document text length sent to judge. 0 = no cap.")
    parser.add_argument("--retry", type=int, default=2,
                        help="Retries on codex exec failure (default 2).")
    parser.add_argument("--retry-sleep", type=float, default=10.0,
                        help="Sleep seconds between retries (default 10).")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def stratified_sample(
    per_item: list[dict[str, Any]],
    n_per_category: int,
    seed: int,
    exclude: set[tuple[str, str]] | None = None,
    n_by_category: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Flatten v2 per-item JSONL into (item, doc, category) pairs and sample n per category.

    `exclude` is a set of (example_id, doc_id) pairs to skip (e.g. already judged).
    `n_by_category` overrides `n_per_category` for the named categories.
    """
    exclude = exclude or set()
    n_by_category = n_by_category or {}
    pairs_by_cat: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in per_item:
        for cat, info in rec["categories"].items():
            for doc_id in info["sample_doc_ids"]:
                if (rec["example_id"], doc_id) in exclude:
                    continue
                pairs_by_cat[cat].append({
                    "example_id": rec["example_id"],
                    "subject": rec["subject"],
                    "answer_index": rec["answer_index"],
                    "n_choices": rec["n_choices"],
                    "doc_id": doc_id,
                    "v2_category": cat,
                })
    rng = random.Random(seed)
    sampled = []
    for cat in ("q_only", "q_plus_correct_only", "q_plus_wrong_only", "q_plus_correct_and_wrong"):
        bucket = pairs_by_cat.get(cat, [])
        rng.shuffle(bucket)
        n_take = n_by_category.get(cat, n_per_category)
        sampled.extend(bucket[:n_take])
    return sampled


def build_prompt(item: dict[str, Any], doc: dict[str, Any], pair: dict[str, Any], max_doc_chars: int) -> str:
    """Assemble the per-pair judge prompt.

    Includes:
    - Algorithm context (DCLM-style + proximity gate + per-option window)
    - Corpus context (Greek HPLT from Apertus 5B CPT pool)
    - Audit goal
    - 5-tier rating rubric
    - The benchmark item and its choices
    - The full document text
    - The required JSON output (schema enforced separately by codex)
    """
    choices_text = "\n".join(f"  {chr(0x391 + i)}. {c}" for i, c in enumerate(item["choices"]))
    answer_text = item["choices"][int(item["answer_index"])]

    doc_text = doc.get("text") or ""
    doc_size_note = ""
    if max_doc_chars > 0 and len(doc_text) > max_doc_chars:
        doc_text = doc_text[:max_doc_chars] + "\n\n[TRUNCATED: original document is longer]"
        doc_size_note = f" (truncated at {max_doc_chars} chars)"

    return f"""## Task

You are auditing a Greek-language test-set decontamination pipeline. The
algorithm flagged one specific document as having an overlap with one
specific GreekMMLU benchmark question. Your job is to rate, on a 1-5
scale, how real the contamination signal is.

## Algorithm context

We built a DCLM-adapted scanner (after Li et al. 2024,
DataComp-LM) that:

- finds documents containing a k-gram of the question's normalized stem
  (Greek-aware: NFKC + strip combining marks + casefold + Unicode word
  tokens, k=8);
- then, within a proximity window after each Q-side match, checks
  whether the option texts (correct / wrong) appear — using a tight
  ~5-token window for short formulaic answers (Σωστό/Λάθος, ΝΑΙ/ΟΧΙ,
  letter labels) and a wider ~50-token window for substantive
  multi-token answers;
- categorizes the (item, doc) pair into one of four mutually exclusive
  buckets: `q_only`, `q_plus_correct_only`, `q_plus_wrong_only`,
  `q_plus_correct_and_wrong`.

The algorithm's signal is structural (Q present + which options
present, where). It cannot judge whether the document is an exam
scrape vs an authoritative source of the underlying Greek fact vs a
general-knowledge discussion of the topic. That is what we want from
you.

## Corpus context

The document was drawn from the Greek HPLT web crawl that we used as
training data for the Apertus-8B Greek continued-pretraining (the 5B
diagnostic Vanilla run). The point of decontamination is to decide
whether documents like this one let the model memorize GreekMMLU
question→answer mappings during training. If the model could memorize
the answer rather than learn it from underlying Greek text, that is
contamination.

NOTE: a document being the primary source of a Greek regulation,
constitutional text, or other authoritative fact is NOT the same as
test-set contamination. The model learning Greek regulations from
their source is exactly what Greek CPT is for. We need you to
distinguish these cases.

## Rating rubric

Use these definitions exactly:

**Rating 5 — DEFINITE_CHEATING**
The document presents the question and the correct answer in clear
exam scaffolding (numbered or lettered options, possibly an answer
key, possibly a sequence of similar multiple-choice items). A model
trained on this document could memorize the Q→A mapping directly.
The eval was scraped or republished here.

**Rating 4 — STRONG_CHEATING**
The document contains the question (verbatim or close paraphrase) AND
the correct answer adjacent to it, but the format is not full exam
scaffolding — it could be a study note, a flowing-prose Q+A, an answer
key without explicit option labels. The Q→A pairing is still directly
recoverable.

**Rating 3 — SOURCE_OVERLAP**
The document is an authoritative primary source of the fact, regulation,
constitutional text, or canonical historical/religious document the
question is asking about. The correct answer is recoverable from the
source text but in non-exam form (regulatory clauses, encyclopedic
prose, official document). A model trained on this learns the
underlying Greek fact from its source — which is what Greek CPT is for.

**Rating 2 — TOPIC_OVERLAP**
The document discusses the same topic as the question but does NOT
contain the question or the correct answer verbatim or adjacent. The
model can possibly reason out the answer if it understands the topic,
but cannot memorize a Q→A mapping from this document.

**Rating 1 — NO_OVERLAP**
The document and the question are essentially unrelated. The
n-gram-level match the algorithm found is coincidental — shared
generic vocabulary, no semantic overlap.

## The benchmark item

Subject: {item.get("subject", "")}
example_id: {item.get("example_id", "")}
v2 category for this match: {pair["v2_category"]}

Question:
{item["question"]}

Choices:
{choices_text}

Correct answer (index {item["answer_index"]}): {answer_text}

## The document

URL: {doc.get("url", "<unknown>")}
Document length: {len(doc_text)} characters{doc_size_note}

[BEGIN DOCUMENT]
{doc_text}
[END DOCUMENT]

## Output

Output ONLY a single JSON object on one line, conforming exactly to the
output schema you were given. No markdown fences, no preamble, no
commentary. The `reasoning` field must cite a specific Greek phrase
from the document and explain how it maps to your rating.
"""


def codex_exec_judge(
    prompt: str,
    model: str,
    reasoning_effort: str,
    schema_path: Path,
    instructions_path: Path,
    out_file: Path,
    retry: int,
    retry_sleep: float,
) -> tuple[dict[str, Any] | None, str, float]:
    """Run a single codex exec call. Returns (parsed_json_or_None, raw_stderr, wall_seconds)."""
    cmd = [
        "codex", "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--output-schema", str(schema_path),
        "-o", str(out_file),
        "-m", model,
        "-c", f"model_reasoning_effort={reasoning_effort}",
        "-c", f"model_instructions_file={instructions_path}",
        "-",
    ]
    last_err = ""
    last_secs = 0.0
    for attempt in range(retry + 1):
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=600,
            )
            last_secs = time.time() - t0
            last_err = proc.stderr
            if proc.returncode != 0:
                if attempt < retry:
                    time.sleep(retry_sleep * (attempt + 1))
                    continue
                return None, last_err, last_secs
            if not out_file.exists():
                if attempt < retry:
                    time.sleep(retry_sleep * (attempt + 1))
                    continue
                return None, last_err + "\n[no output file written]", last_secs
            raw = out_file.read_text(encoding="utf-8").strip()
            try:
                obj = json.loads(raw)
                return obj, last_err, last_secs
            except json.JSONDecodeError as exc:
                last_err = f"json parse error: {exc}; raw output head: {raw[:400]}\n" + last_err
                if attempt < retry:
                    time.sleep(retry_sleep * (attempt + 1))
                    continue
                return None, last_err, last_secs
        except subprocess.TimeoutExpired:
            last_err = "subprocess timeout after 600s"
            last_secs = 600.0
            if attempt < retry:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            return None, last_err, last_secs
        except Exception as exc:  # noqa: BLE001
            last_err = f"unexpected exception: {exc}"
            last_secs = time.time() - t0
            if attempt < retry:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            return None, last_err, last_secs
    return None, last_err, last_secs


def parse_tokens_from_stderr(stderr: str) -> dict[str, Any]:
    """Best-effort token-usage extraction from codex stderr."""
    out: dict[str, Any] = {}
    # codex prints lines like "tokens used: 1234" or "input_tokens=..."
    for line in stderr.splitlines():
        m = re.search(r"tokens used[:=]?\s*([0-9]+)", line, re.IGNORECASE)
        if m:
            out["tokens_used"] = int(m.group(1))
        m = re.search(r"input_tokens[:=]\s*([0-9]+)", line)
        if m:
            out["input_tokens"] = int(m.group(1))
        m = re.search(r"output_tokens[:=]\s*([0-9]+)", line)
        if m:
            out["output_tokens"] = int(m.group(1))
    return out


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    instructions_path = Path(__file__).parent / "codex_judge_instructions.md"
    schema_path = Path(__file__).parent / "codex_judge_output_schema.json"
    if not instructions_path.is_file():
        sys.exit(f"ERROR: missing instructions file: {instructions_path}")
    if not schema_path.is_file():
        sys.exit(f"ERROR: missing schema file: {schema_path}")

    print(f"[setup] loading v2 per-item JSONL: {args.v2_per_item_jsonl}", flush=True)
    per_item = load_jsonl(args.v2_per_item_jsonl)
    print(f"[setup] loading corpus: {args.corpus_jsonl}", flush=True)
    corpus_by_id = {d["doc_id"]: d for d in load_jsonl(args.corpus_jsonl)}
    print(f"[setup] loading queries: {args.queries_jsonl}", flush=True)
    queries_by_id = {q["example_id"]: q for q in load_jsonl(args.queries_jsonl)}

    exclude: set[tuple[str, str]] = set()
    if args.exclude_judged_jsonl:
        prior = load_jsonl(args.exclude_judged_jsonl)
        for r in prior:
            exclude.add((r["example_id"], r["doc_id"]))
        print(f"[setup] loaded {len(exclude)} prior-judged pairs to exclude", flush=True)
    n_by_category: dict[str, int] = {}
    if args.n_by_category:
        for tok in args.n_by_category.split(","):
            tok = tok.strip()
            if not tok:
                continue
            k, v = tok.split("=")
            n_by_category[k.strip()] = int(v)
        print(f"[setup] n_by_category override: {n_by_category}", flush=True)

    pairs = stratified_sample(per_item, args.n_per_category, args.seed,
                              exclude=exclude, n_by_category=n_by_category)
    if args.smoke > 0:
        pairs = pairs[: args.smoke]
    print(f"[setup] sampled {len(pairs)} pairs (smoke={args.smoke})", flush=True)
    by_cat = collections.Counter(p["v2_category"] for p in pairs)
    print(f"[setup] per-category breakdown: {dict(by_cat)}", flush=True)

    tmp_dir = Path("/tmp/codex_judge_outputs"); tmp_dir.mkdir(exist_ok=True)

    results_lock_path = args.output_jsonl.with_suffix(".lock")
    if args.output_jsonl.exists():
        print(f"[setup] output file already exists: {args.output_jsonl} (will append)", flush=True)

    def run_one(idx: int, pair: dict[str, Any]) -> dict[str, Any]:
        eid = pair["example_id"]; did = pair["doc_id"]
        item = queries_by_id.get(eid)
        doc = corpus_by_id.get(did)
        if not item or not doc:
            return {"_skipped": True, "reason": "missing item or doc", "example_id": eid, "doc_id": did}
        prompt = build_prompt(item, doc, pair, args.max_doc_chars)
        out_file = tmp_dir / f"out_{os.getpid()}_{idx}.json"
        if out_file.exists(): out_file.unlink()
        obj, stderr, secs = codex_exec_judge(
            prompt, args.model, args.reasoning_effort, schema_path, instructions_path,
            out_file, args.retry, args.retry_sleep,
        )
        usage = parse_tokens_from_stderr(stderr)
        rec: dict[str, Any] = {
            "example_id": eid,
            "doc_id": did,
            "v2_category": pair["v2_category"],
            "subject": pair["subject"],
            "url": doc.get("url"),
            "doc_len_chars": len(doc.get("text") or ""),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "wall_seconds": round(secs, 2),
            "usage": usage,
            "ok": obj is not None,
            "judgment": obj,
        }
        if obj is None:
            rec["stderr_tail"] = stderr[-2000:]
        # cleanup tmp file
        try: out_file.unlink()
        except FileNotFoundError: pass
        return rec

    # Sequential for smoke; parallel otherwise.
    n_workers = 1 if args.smoke > 0 else args.max_workers
    print(f"[run] firing {len(pairs)} codex calls with max_workers={n_workers}, model={args.model}, effort={args.reasoning_effort}", flush=True)

    completed = 0
    failed = 0
    t_start = time.time()
    with args.output_jsonl.open("a", encoding="utf-8") as fout:
        if n_workers == 1:
            for idx, pair in enumerate(pairs):
                rec = run_one(idx, pair)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                completed += 1
                if not rec.get("ok"): failed += 1
                print(f"[run] {completed}/{len(pairs)}  cat={pair['v2_category']}  ok={rec.get('ok')}  rating={rec.get('judgment',{}).get('rating') if rec.get('ok') else 'N/A'}  secs={rec.get('wall_seconds')}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(run_one, idx, pair): (idx, pair) for idx, pair in enumerate(pairs)}
                for fut in as_completed(futures):
                    idx, pair = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = {
                            "_skipped": True,
                            "reason": f"executor error: {exc}",
                            "example_id": pair["example_id"],
                            "doc_id": pair["doc_id"],
                            "v2_category": pair["v2_category"],
                        }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    completed += 1
                    if not rec.get("ok"): failed += 1
                    eta_s = (time.time() - t_start) * (len(pairs) - completed) / max(completed, 1)
                    print(f"[run] {completed}/{len(pairs)}  cat={pair['v2_category']}  ok={rec.get('ok')}  rating={rec.get('judgment',{}).get('rating') if rec.get('ok') else 'N/A'}  secs={rec.get('wall_seconds')}  eta≈{eta_s:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[done] {completed} pairs in {elapsed:.0f}s, {failed} failed.", flush=True)
    print(f"[done] results: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
