"""Apertus-parity PII masker (email / IP / IBAN) — stdlib only.

The masking loop and the email/IP regexes are VERBATIM from Apertus's
`swiss-ai/pretrain-data` `PIIFormatter`
(commit 8af990b9401101cf95acd02b066ed0c449789126,
 src/data_pipeline_pretrain/pipeline/formatters/pii_formatter.py).

The ONE deviation is IBAN. Apertus's `iban_regex` was tuned for ~22-char
space-grouped IBANs; on Greek 27-char IBANs it truncates (leaving a PII
fragment) or misses the compact form. Here IBAN uses per-country
exact-length matching (compact OR single-space-grouped) + ISO-7064 mod-97
validation, so Greek — and every EU — IBAN is caught cleanly with
near-zero false positives. Replacement tokens are unchanged and match the
reserved Apertus tokenizer tokens: `<email-pii>` / `<ip-pii>` / `<iban-pii>`.

Stdlib only (no datatrove, no python-stdnum) so it is portable and
testable; `pii_formatter.py` wraps this for the datatrove production run.
"""
from __future__ import annotations
import re

# ---- email / IP : verbatim from Apertus PIIFormatter ----
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)])"
)
IP_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
)

# ---- IBAN : the Greek fix (per-country exact length + mod-97) ----
# ISO 13616 total lengths (EU + Greek + replay-relevant countries).
IBAN_LEN = {
    "AT": 20, "BE": 16, "BG": 22, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18,
    "EE": 20, "ES": 24, "FI": 18, "FR": 27, "GB": 22, "GR": 27, "HR": 21, "HU": 28,
    "IE": 22, "IS": 26, "IT": 27, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27,
    "MT": 31, "NL": 18, "NO": 15, "PL": 28, "PT": 25, "RO": 24, "SE": 24, "SI": 19,
    "SK": 24,
}
# Generic candidate run: CC + 2 check digits + 11..30 alphanumerics, each
# optionally preceded by a single space (catches compact AND grouped forms).
IBAN_CAND_RE = re.compile(r"\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]){11,30}")


def _mod97_ok(compact: str) -> bool:
    """ISO 7064 mod-97-10: move first 4 chars to end, letters->A=10..Z=35, %97==1."""
    s = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in s)) % 97 == 1


def iban_valid(compact: str) -> bool:
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]+", compact):
        return False
    cc, L = compact[:2], len(compact)
    if cc in IBAN_LEN:
        if L != IBAN_LEN[cc]:
            return False
    elif not (15 <= L <= 34):
        return False
    return _mod97_ok(compact)


def _iban_from_candidate(cand: str) -> str | None:
    """Return the exact original (possibly space-grouped) IBAN substring inside a
    candidate run, or None. Handles over-extension into trailing uppercase."""
    compact = cand.replace(" ", "")
    if len(compact) < 15:
        return None
    cc = compact[:2]
    lengths = [IBAN_LEN[cc]] if cc in IBAN_LEN else range(min(34, len(compact)), 14, -1)
    for L in lengths:
        if L <= len(compact) and iban_valid(compact[:L]):
            taken = end = 0
            for i, ch in enumerate(cand):
                if ch != " ":
                    taken += 1
                end = i + 1
                if taken == L:
                    break
            return cand[:end]
    return None


def mask(text: str) -> tuple[str, dict]:
    """Mask email/IP/IBAN in `text`. Returns (masked_text, per-type unique counts).
    Mirrors Apertus's sequential findall + text.replace(unique_match, token),
    repeating the sequence to a fixed point so overlap/boundary exposure cannot
    leave PII for a second invocation."""
    counts = {"email": 0, "ip": 0, "iban": 0}
    seen: set[str] = set()
    while True:
        changed = False
        for m in EMAIL_RE.findall(text):
            if m and m not in seen:
                replaced = text.replace(m, "<email-pii>")
                if replaced == text:
                    continue
                seen.add(m)
                counts["email"] += 1
                text = replaced
                changed = True
        for m in IP_RE.findall(text):
            if m and m not in seen:
                replaced = text.replace(m, "<ip-pii>")
                if replaced == text:
                    continue
                seen.add(m)
                counts["ip"] += 1
                text = replaced
                changed = True
        # A candidate is deliberately capped at 34 compact characters. Two
        # concatenated valid IBANs can therefore expose the second only after
        # the first replacement; the fixed-point pass catches it immediately.
        for cand in IBAN_CAND_RE.findall(text):
            ib = _iban_from_candidate(cand)
            if ib and ib not in seen:
                replaced = text.replace(ib, "<iban-pii>")
                if replaced == text:
                    continue
                seen.add(ib)
                counts["iban"] += 1
                text = replaced
                changed = True
        if not changed:
            break
    return text, counts
