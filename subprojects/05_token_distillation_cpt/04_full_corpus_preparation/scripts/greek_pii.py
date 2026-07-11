#!/usr/bin/env python3
"""High-precision Greek direct-identifier masking for Phase 04.

The policy intentionally excludes generic person-name and address NER.  Bare
AFM/AMKA values are accepted only after their checksums pass; identity-card
numbers require an explicit label; phone numbers require Greek numbering form.
"""

from __future__ import annotations

import re
from collections import Counter


PHONE = re.compile(r"(?<![\d.])(?:\+?30|0030)?[\s.-]?((?:2\d|69)\d{2}[\s.-]?\d{6}|(?:2\d|69)\d{8})(?!\d)")
AFM_LABEL = re.compile(
    r"(?i)(?:Α\.?\s*Φ\.?\s*Μ\.?|ΑΦΜ|Αριθμ(?:ός|\.)?\s+Φορολογικού\s+Μητρώου)\s*[:#-]?\s*(\d{9})\b"
)
AMKA_LABEL = re.compile(r"(?i)(?:Α\.?\s*Μ\.?\s*Κ\.?\s*Α\.?|ΑΜΚΑ)\s*[:#-]?\s*(\d{11})\b")
NINE_DIGITS = re.compile(r"(?<!\d)(\d{9})(?!\d)")
ELEVEN_DIGITS = re.compile(r"(?<!\d)(\d{11})(?!\d)")
IDENTITY = re.compile(
    r"(?i)(?:Α\.?\s*Δ\.?\s*Τ\.?|ΑΔΤ|Δελτί(?:ο|ου)\s+Ταυτότητας|"
    r"αριθμ(?:ός|\.)?\s+(?:δελτίου\s+)?ταυτότητας|passport)\s*[:#-]?\s*"
    r"([A-ZΑ-Ω]{1,3}[\s-]?\d{5,10})\b"
)


def afm_valid(value: str) -> bool:
    if len(value) != 9 or not value.isdigit() or value == "000000000":
        return False
    digits = [int(character) for character in value]
    checksum = (sum(digits[index] * (2 ** (8 - index)) for index in range(8)) % 11) % 10
    return checksum == digits[8]


def luhn_valid(value: str) -> bool:
    if not value.isdigit():
        return False
    digits = [int(character) for character in value]
    total = sum(digits[-1::-2]) + sum(sum(divmod(digit * 2, 10)) for digit in digits[-2::-2])
    return total % 10 == 0


def amka_valid(value: str) -> bool:
    if len(value) != 11 or not value.isdigit():
        return False
    day, month = int(value[:2]), int(value[2:4])
    return 1 <= day <= 31 and 1 <= month <= 12 and luhn_valid(value)


def _validated_values(text: str, labelled: re.Pattern[str], bare: re.Pattern[str], validator) -> set[str]:
    values = {match.group(1) for match in labelled.finditer(text) if validator(match.group(1))}
    values.update(match.group(1) for match in bare.finditer(text) if validator(match.group(1)))
    return values


def mask_greek_identifiers(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    for value in sorted(_validated_values(text, AFM_LABEL, NINE_DIGITS, afm_valid), key=len, reverse=True):
        occurrences = text.count(value)
        if occurrences:
            text = text.replace(value, "<afm-pii>")
            counts["afm"] += occurrences
    for value in sorted(_validated_values(text, AMKA_LABEL, ELEVEN_DIGITS, amka_valid), key=len, reverse=True):
        occurrences = text.count(value)
        if occurrences:
            text = text.replace(value, "<amka-pii>")
            counts["amka"] += occurrences
    for match in list(IDENTITY.finditer(text))[::-1]:
        start, end = match.span(1)
        text = text[:start] + "<identity-pii>" + text[end:]
        counts["identity"] += 1
    for match in list(PHONE.finditer(text))[::-1]:
        start, end = match.span(0)
        text = text[:start] + "<phone-pii>" + text[end:]
        counts["phone"] += 1
    return text, dict(counts)

