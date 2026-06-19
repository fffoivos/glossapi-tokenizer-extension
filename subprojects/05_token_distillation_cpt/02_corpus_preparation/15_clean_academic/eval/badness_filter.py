#!/usr/bin/env python3
"""Shared extraction-quality gate for annotation unit-building. Docs with greek_badness_score > 60 are
badly extracted (OCR garble / low Greek) and are dropped before annotation — no point spending Opus on
text that won't survive into the clean corpus. greek_phd + openarchives shards carry the score as a column;
kallipos (and any source lacking it) is scored on the fly with the same Rust scorer (glossapi_rs_noise),
so the gate is identical across sources."""
import tempfile, os
BADNESS_MAX = 60.0
BADNESS_KEY = "greek_badness_score"
_N = None


def _noise():
    global _N
    if _N is None:
        import glossapi_rs_noise as M
        _N = M
    return _N


def score_text(text):
    """Compute greek_badness_score for a doc's text (kallipos / any source missing the column)."""
    if not text or not text.strip():
        return None
    fd, p = tempfile.mkstemp(suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        r = _noise().score_markdown_file_detailed(p)
        return float(r[0]) if r else None
    finally:
        os.unlink(p)


def record_badness(rec, text=None):
    """Badness from a shard record's column if present, else computed from text."""
    v = rec.get(BADNESS_KEY) if isinstance(rec, dict) else None
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return score_text(text if text is not None else (rec.get("text") if isinstance(rec, dict) else None))


def is_bad(score):
    """True → drop. Missing score (None) is KEPT (can't assess → don't silently drop)."""
    return score is not None and score > BADNESS_MAX
