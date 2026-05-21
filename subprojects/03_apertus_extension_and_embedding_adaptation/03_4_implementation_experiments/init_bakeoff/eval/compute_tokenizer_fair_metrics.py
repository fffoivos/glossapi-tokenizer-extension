"""Tokenizer-fair primary metrics for the init bakeoff (cpt_plan v0.7 §5.1).

Per-token PPL is NOT comparable across Vanilla (vocab 131,072) and
ReTok/Centroid (vocab 148,480). The cross-tokenizer-fair metrics this
script computes:

  BPC (bits per byte)         — cleanest cross-tokenizer comparison
  NLL per Unicode character   — more interpretable for Greek/polytonic
  NLL per word                — human-facing language metric
  tokens per word             — tokenizer efficiency
  chars per token             — tokenizer efficiency
  compression ratio (= chars / tokens)
  STRR (Subword-Tokenization Recovery Rate)
                              — fraction of held-out words that
                                tokenize to a SINGLE token

[Cite: references/papers/apertus_2509.14233.pdf — pretraining-eval choice
       follows Apertus's tokenizer-fairness discussion; cpt_plan v0.7 §5.1]

Usage:
    python3 compute_tokenizer_fair_metrics.py \\
        --model-path /path/to/apertus-or-arm-checkpoint \\
        --eval-jsonl /path/to/held_out_greek.jsonl \\
        --output-json /path/to/metrics.json \\
        --max-context 4096

JSONL format expected at --eval-jsonl: one doc per line, fields:
    text     — the document text (required)
    source   — sub-corpus name (optional; used for per-source aggregation)
    register — register label (optional; used for per-register aggregation)
    doc_id   — stable identifier (optional)

Use --stats-only to skip the model forward pass (computes only the
tokenizer-intrinsic metrics: tokens/word, chars/token, STRR). Useful for
sanity-checking the eval set + tokenizer without an 8B model on disk.

Output JSON:
{
  "model_path": ..., "tokenizer_vocab_size": ...,
  "global": {bpc, nll_per_char, nll_per_word, tokens_per_word, ...},
  "per_source": {source_name: {...}},
  "per_register": {register: {...}},
  "strr_overall": ...,
  "n_docs": ..., "n_chars": ..., "n_bytes": ..., "n_words": ..., "n_tokens": ...
}
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


# Word splitter: whitespace + punctuation. Defensible across Greek
# (and Latin / Cyrillic) without language-specific resources. For
# stronger morphological splitting we'd use gr-nlp-toolkit / CLTK
# (v0.7 §7.3), but for STRR / NLL-per-word the simple split is fine
# and avoids a heavy dependency.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _doc_iter(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" not in row or not row["text"]:
                continue
            yield row


def _compute_text_stats(text: str) -> dict:
    return {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "words": _word_count(text),
    }


def _compute_strr(tokenizer, words: list[str]) -> tuple[int, int]:
    """Return (n_single_token, n_words). STRR = single / total."""
    n_single = 0
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            n_single += 1
    return n_single, len(words)


def _accumulate(agg: dict, stats: dict, n_tokens: int, nll_nats: float | None):
    agg["n_docs"] += 1
    agg["n_chars"] += stats["chars"]
    agg["n_bytes"] += stats["bytes"]
    agg["n_words"] += stats["words"]
    agg["n_tokens"] += n_tokens
    if nll_nats is not None:
        agg["nll_nats_total"] += nll_nats


def _finalize(agg: dict) -> dict:
    """From accumulators → final metrics."""
    out = {
        "n_docs": agg["n_docs"],
        "n_chars": agg["n_chars"],
        "n_bytes": agg["n_bytes"],
        "n_words": agg["n_words"],
        "n_tokens": agg["n_tokens"],
    }
    if agg["n_tokens"]:
        out["chars_per_token"] = agg["n_chars"] / agg["n_tokens"]
        out["compression_ratio"] = agg["n_chars"] / agg["n_tokens"]
    if agg["n_words"]:
        out["tokens_per_word"] = agg["n_tokens"] / agg["n_words"]
    if "nll_nats_total" in agg and agg["nll_nats_total"] > 0:
        nll = agg["nll_nats_total"]
        if agg["n_bytes"]:
            out["bpc_bits_per_byte"] = (nll / agg["n_bytes"]) / math.log(2)
        if agg["n_chars"]:
            out["nll_per_char"] = nll / agg["n_chars"]
        if agg["n_words"]:
            out["nll_per_word"] = nll / agg["n_words"]
        if agg["n_tokens"]:
            out["nll_per_token"] = nll / agg["n_tokens"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=str, required=True,
                    help="HF-format model dir (or just the tokenizer dir if --stats-only)")
    ap.add_argument("--eval-jsonl", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--max-context", type=int, default=4096,
                    help="truncate each doc to this many tokens before forward (matches bakeoff seq_length)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="optional cap for debugging")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--stats-only", action="store_true",
                    help="skip model load; compute only tokenizer-intrinsic metrics (tokens/word, chars/token, STRR)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    print(f"loaded tokenizer: vocab_size={tok.vocab_size}", file=sys.stderr)

    model = None
    if not args.stats_only:
        import torch
        from transformers import AutoModelForCausalLM
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        print(f"loading model {args.model_path} in {args.dtype} on {args.device} ...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype_map[args.dtype],
            device_map=args.device if args.device == "cuda" else None,
            trust_remote_code=True,
        )
        if args.device != "cuda":
            model = model.to(args.device)
        model.eval()

    # Aggregators
    global_agg = defaultdict(float)
    per_source = defaultdict(lambda: defaultdict(float))
    per_register = defaultdict(lambda: defaultdict(float))
    strr_words: list[str] = []
    n_truncated = 0
    n_tokens_dropped_to_truncation = 0

    t0 = time.time()
    n_seen = 0
    for row in _doc_iter(args.eval_jsonl):
        if args.max_docs and n_seen >= args.max_docs:
            break
        n_seen += 1
        text = row["text"]
        source = row.get("source") or "unknown"
        register = row.get("register") or "unknown"

        # Tokenize once + truncate to max_context.
        # CRITICAL: char/byte/word stats must be computed on the same prefix
        # that was actually tokenized and forwarded — NOT on the full text —
        # otherwise BPC + NLL/char divide prefix-only loss by full-document
        # denominators and the primary intrinsic metrics come out artificially
        # low (reviewer round-2 finding, High 5). For truncated docs we decode
        # the truncated ID list back to recover the exact scored substring.
        ids_full = tok.encode(text, add_special_tokens=False)
        if args.max_context and len(ids_full) > args.max_context:
            ids = ids_full[: args.max_context]
            scored_text = tok.decode(ids, skip_special_tokens=False)
            n_truncated += 1
            n_tokens_dropped_to_truncation += (len(ids_full) - len(ids))
        else:
            ids = ids_full
            scored_text = text
        n_tokens = len(ids)

        stats = _compute_text_stats(scored_text)

        # STRR words come from the full text (it's a tokenizer-only metric;
        # not coupled to the forward pass).
        if len(strr_words) < 200_000:
            strr_words.extend(_WORD_RE.findall(text)[:1000])

        # Forward (skip in --stats-only)
        nll_nats = None
        if model is not None:
            import torch
            if n_tokens >= 2:
                input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, labels=input_ids)
                # HF returns mean per-token NLL over n_tokens-1 positions
                n_predicted = n_tokens - 1
                nll_nats = float(out.loss.item()) * n_predicted

        _accumulate(global_agg, stats, n_tokens, nll_nats)
        _accumulate(per_source[source], stats, n_tokens, nll_nats)
        _accumulate(per_register[register], stats, n_tokens, nll_nats)

        if n_seen % 50 == 0:
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  doc {n_seen}: {rate:.2f}/s  source={source}", file=sys.stderr)

    # STRR over the collected word set
    print(f"computing STRR over {len(strr_words):,} sampled words ...", file=sys.stderr)
    n_single, n_total = _compute_strr(tok, strr_words)
    strr = (n_single / n_total) if n_total else 0.0

    report = {
        "model_path": args.model_path,
        "eval_jsonl": str(args.eval_jsonl),
        "max_context": args.max_context,
        "stats_only": bool(args.stats_only),
        "tokenizer_vocab_size": tok.vocab_size,
        "wall_seconds": time.time() - t0,
        "global": _finalize(global_agg),
        "per_source": {s: _finalize(a) for s, a in per_source.items()},
        "per_register": {r: _finalize(a) for r, a in per_register.items()},
        "strr": {
            "n_words_sampled": n_total,
            "n_single_token": n_single,
            "rate": strr,
        },
        "truncation": {
            "n_docs_truncated": n_truncated,
            "n_tokens_dropped": n_tokens_dropped_to_truncation,
            "fraction_truncated": (n_truncated / n_seen) if n_seen else 0.0,
            "note": "char/byte/word counts are computed on the scored prefix, NOT on full text. Look at this if it's > ~10 % of docs.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Pretty-print a compact table
    g = report["global"]
    print("\n=== tokenizer-fair metrics (global) ===")
    print(f"  vocab_size:        {report['tokenizer_vocab_size']:,}")
    print(f"  n_docs:            {g['n_docs']:,}")
    print(f"  n_chars/bytes/words/tokens: {g['n_chars']:,} / {g['n_bytes']:,} / {g['n_words']:,} / {g['n_tokens']:,}")
    print(f"  chars/token:       {g.get('chars_per_token', float('nan')):.4f}")
    print(f"  tokens/word:       {g.get('tokens_per_word', float('nan')):.4f}")
    print(f"  compression ratio: {g.get('compression_ratio', float('nan')):.4f}")
    print(f"  STRR:              {strr:.4f} ({n_single:,}/{n_total:,} single-token words)")
    if n_truncated:
        print(f"  truncated:         {n_truncated}/{n_seen} docs ({100*n_truncated/max(n_seen,1):.1f}%); {n_tokens_dropped_to_truncation:,} tokens dropped")
    if "bpc_bits_per_byte" in g:
        print(f"  BPC (bits/byte):   {g['bpc_bits_per_byte']:.4f}")
        print(f"  NLL/char:          {g['nll_per_char']:.4f}")
        print(f"  NLL/word:          {g['nll_per_word']:.4f}")
        print(f"  NLL/token:         {g['nll_per_token']:.4f}")
    print(f"\nwrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
