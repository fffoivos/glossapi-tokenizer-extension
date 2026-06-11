#!/usr/bin/env python3
"""Build a candidate HPLT cleaning action manifest from a review bundle.

This is a non-destructive bridge from the first review bundle to the cleaning
schema. It does not edit corpus text. It writes candidate error IDs, normalized
actions, per-error indexes, and summary reports so examples can be retrieved and
reviewed by category.
"""

from __future__ import print_function

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import unicodedata


SCHEMA_VERSION = "hplt-cleaning-action-v1"
SOURCE_DATASET = "HPLT/ell_Grek_ge8_no_mt_clean60"

ERROR_NAMES = collections.OrderedDict([
    ("E001", "replacement/control/private-use characters"),
    ("E002", "escaped Unicode, HTML entities, percent-encoding residue"),
    ("E003", "CSS/JS/HTML/XML remnants"),
    ("E004", "URL/path/metadata-key dumps"),
    ("E005", "mojibake-like symbol and punctuation clutter"),
    ("E006", "top navigation/menu chrome"),
    ("E007", "footer/contact/social/copyright chrome"),
    ("E008", "cookie/login/newsletter overlays"),
    ("E009", "related-post/comment/share scaffolding"),
    ("E010", "heading/list-only snippets or truncated extraction"),
    ("E011", "multiple independent documents in one row"),
    ("E012", "repeated title/date/author metadata resets"),
    ("E013", "internal paragraph or sentence repetition loops"),
    ("E014", "localized gibberish or OCR-like noise islands"),
    ("E015", "low Greek body share or language drift"),
    ("E016", "SEO keyword lists or machine-generated register spam"),
    ("E017", "table/list extraction fragments without prose"),
    ("E018", "duplicated main body separated by boilerplate"),
    ("E019", "forum/comment/form scaffolding dominating content"),
    ("E020", "no main body: tiny fragments, index pages, orphan headings"),
])

DETECTOR_TO_ERROR_IDS = {
    "encoding_score": ["E001", "E002"],
    "markup_score": ["E003", "E004"],
    "symbol_score": ["E005"],
    "boilerplate_score": ["E006", "E007", "E008", "E009"],
    "archive_snippet_score": ["E010"],
    "catalog_list_score": ["E017"],
    "seo_spam_score": ["E016"],
    "form_comment_score": ["E019"],
    "no_main_body_score": ["E020"],
    "split_candidate_score": ["E011", "E012"],
    "internal_repetition_score": ["E013", "E018"],
    "badness_score": ["E014"],
    "lang_drift_score": ["E015"],
}

ERROR_TO_SCORE = {
    "E001": "encoding_score",
    "E002": "encoding_score",
    "E003": "markup_score",
    "E004": "markup_score",
    "E005": "symbol_score",
    "E006": "boilerplate_score",
    "E007": "boilerplate_score",
    "E008": "boilerplate_score",
    "E009": "boilerplate_score",
    "E010": "archive_snippet_score",
    "E011": "split_candidate_score",
    "E012": "split_candidate_score",
    "E013": "internal_repetition_score",
    "E014": "badness_score",
    "E015": "lang_drift_score",
    "E016": "seo_spam_score",
    "E017": "catalog_list_score",
    "E018": "internal_repetition_score",
    "E019": "form_comment_score",
    "E020": "no_main_body_score",
}

TOP_CHROME_PATTERNS = [
    "home", "menu", "search", "login", "register",
    "αρχικη σελιδα", "μενου", "αναζητηση", "συνδεση", "εγγραφη", "κατηγορι",
]
FOOTER_PATTERNS = [
    "copyright", "all rights reserved", "privacy", "terms", "contact",
    "facebook", "twitter", "instagram", "youtube", "rss",
    "πνευματικ", "δικαιωμα", "πολιτικη απορρητου", "οροι χρησης",
    "επικοινων", "ακολουθηστε",
]
COOKIE_LOGIN_PATTERNS = [
    "cookie", "cookies", "newsletter", "subscribe", "login", "register",
    "συνδεση", "εγγραφη", "ενημερωτικο δελτιο",
]
RELATED_COMMENT_PATTERNS = [
    "related", "read more", "share", "comment", "comments",
    "σχετικα αρθρα", "διαβαστε επισης", "διαβαστε ακομη", "κοινοποι",
    "μοιραστ", "σχολια", "αφηστε σχολιο",
]
ARCHIVE_SNIPPET_PATTERNS = [
    "archive for", "posts tagged", "posted in", "posted by", "leave a comment",
    "read more", "continue reading", "category archive", "tag archive",
    "δειτε επισησ", "δειτε ακομα", "δειτε ακομη", "περισσοτερα αρθρα",
    "συνεχιστε την αναγνωση", "τα πιο δημοφιλη", "διαβαζονται παντα", "πριν απο",
]
CATALOG_LIST_PATTERNS = [
    "προσθηκη στο καλαθι", "καλαθι", "μη διαθεσιμο", "top sellers",
    "αρχικη αξια", "εκπτωση", "αγορεσ", "προσφορεσ", "τιμη",
    "brand", "sku", "ml", "τεμ", "καινουριο",
]
SEO_SPAM_PATTERNS = [
    "watch movies", "watch series", "free streaming", "download",
    "putlocker", "tainies online", "ταινιεσ online", "greek subs",
    "magnet link", "adult", "torrent", "online free",
]
FORM_COMMENT_PATTERNS = [
    "φορμα επαφησ", "επαφη", "σχολιαστε", "σχολια", "leave a comment",
    "reply", "συνδεθειτε", "εγγραφειτε", "δεν επιτρεπεται σχολιασμοσ",
]
COMMENT_SUFFIX_PATTERNS = [
    "δεν υπαρχουν σχολια",
    "δημοσιευση σχολιου",
    "προσθηκη σχολιου",
    "σχολιαστε",
    "σχολια αναγνωστων",
    "πρεπει να συνδεθειτε",
    "πρεπει να ειστε μελοσ",
    "δεν σασ επιτρεπεται η υποβολη σχολιων",
    "για να σχολιασετε",
    "0 comments",
]
RELATED_SUFFIX_PATTERNS = [
    "διαβαστε επισησ",
    "διαβαστε ακομη",
    "δειτε επισησ",
    "δειτε ακομα",
    "δειτε ακομη",
    "τα πιο δημοφιλη",
    "διαβαζονται παντα",
]
MARKUP_PATTERNS = [
    r"<\/?\w+", r"class=", r"href=", r"src=", r"style=",
    r"\bfunction\b", r"\bvar\b", r"\bwindow\.", r"adsbygoogle",
    r"\biframe\b", r"\/iframe",
]
URL_PATTERNS = [r"https?://", r"\bwww\.", r"\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+"]
ENCODED_PATTERNS = [r"&(?:[a-zA-Z][a-zA-Z0-9]{1,16}|#[0-9]{2,7}|#x[0-9a-fA-F]{2,6});", r"\\u[0-9a-fA-F]{4}", r"%[0-9A-Fa-f]{2}"]
SPLIT_PATTERNS = [r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", r"δημοσιε", r"αναρτη", r"τιτλος", r"posted", r"published"]
GREEK_WORD_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώΪΫϊϋΐΰ]{2,}")
WORD_RE = re.compile(r"\w+", re.U)


def utc_timestamp():
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def stable_sha256(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError("Invalid JSON at %s:%s: %s" % (path, line_number, exc))


def detector_scores(record):
    scores = record.get("detector_scores") or {}
    return {str(key): float(value or 0.0) for key, value in scores.items()}


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def count_substrings(patterns, norm):
    return sum(norm.count(pattern) for pattern in patterns)


def greek_word_count(text):
    return len(GREEK_WORD_RE.findall(text))


def total_word_count(text):
    return len(WORD_RE.findall(text))


def compute_text_features(text):
    if not text:
        return {
            "archive_snippet_score": 0.0,
            "catalog_list_score": 0.0,
            "seo_spam_score": 0.0,
            "form_comment_score": 0.0,
            "no_main_body_score": 0.0,
        }
    norm = normalize_text(text)
    lines = [line.strip() for line in norm.splitlines() if line.strip()]
    nonempty = len(lines)
    words = max(1, total_word_count(text))
    greek_words = greek_word_count(text)
    greek_share = float(greek_words) / float(words)
    short_density = line_density(lines, 45)
    dash_line_density = float(sum(1 for line in lines if line.startswith("-"))) / float(max(1, nonempty))
    ellipsis_count = norm.count("...") + norm.count("[...]")
    euro_count = text.count("€")
    percent_count = text.count("%")
    url_count = regex_count(URL_PATTERNS, text)

    archive_hits = count_substrings(ARCHIVE_SNIPPET_PATTERNS, norm)
    catalog_hits = count_substrings(CATALOG_LIST_PATTERNS, norm)
    seo_hits = count_substrings(SEO_SPAM_PATTERNS, norm)
    form_hits = count_substrings(FORM_COMMENT_PATTERNS, norm)
    posted_hits = len(re.findall(r"\bposted\b|αναρτη|δημοσιε", norm, re.I))
    read_more_hits = norm.count("read more")
    list_marker_hits = sum(1 for line in lines if line.startswith("-") or re.match(r"^\d+[\).]", line))

    archive_score = clamp01(
        0.13 * archive_hits
        + 0.10 * ellipsis_count
        + 0.10 * posted_hits
        + 0.12 * read_more_hits
        + (0.25 if short_density >= 0.55 and archive_hits >= 2 else 0.0)
    )
    catalog_score = clamp01(
        0.11 * catalog_hits
        + 0.08 * euro_count
        + 0.04 * percent_count
        + 0.05 * url_count
        + 0.20 * dash_line_density
        + (0.25 if list_marker_hits >= 12 and catalog_hits >= 2 else 0.0)
    )
    seo_score = clamp01(
        0.16 * seo_hits
        + (0.20 if greek_share < 0.55 and seo_hits >= 2 else 0.0)
        + (0.10 if words > 1500 and greek_share < 0.65 else 0.0)
    )
    form_comment_score = clamp01(
        0.16 * form_hits
        + (0.20 if form_hits >= 2 and greek_words < 350 else 0.0)
        + (0.15 if "φόρμα" in text.casefold() or "form" in norm else 0.0)
    )
    no_main_body_score = clamp01(
        (0.35 if greek_words < 110 and (archive_hits or form_hits or catalog_hits or ellipsis_count) else 0.0)
        + (0.30 if nonempty <= 8 and short_density >= 0.60 else 0.0)
        + (0.20 if archive_score >= 0.45 and greek_words < 300 else 0.0)
        + (0.20 if catalog_score >= 0.45 and greek_words < 300 else 0.0)
        + (0.15 if seo_score >= 0.45 and greek_share < 0.55 else 0.0)
    )

    return {
        "archive_snippet_score": archive_score,
        "catalog_list_score": catalog_score,
        "seo_spam_score": seo_score,
        "form_comment_score": form_comment_score,
        "no_main_body_score": no_main_body_score,
        "greek_word_share": clamp01(greek_share),
        "short_line_density": clamp01(short_density),
        "list_marker_density": clamp01(dash_line_density),
    }


def infer_error_type_ids(record, threshold, scores=None, text=None, text_features=None):
    existing = record.get("error_type_ids")
    ids = set(str(item) for item in existing or [])

    scores = scores or detector_scores(record)
    text_features = text_features or {}
    ids.update(text_based_error_type_ids(record, scores, threshold, text, text_features))
    for detector, mapped_ids in DETECTOR_TO_ERROR_IDS.items():
        if scores.get(detector, 0.0) >= threshold:
            if detector == "markup_score" and text and not ids.intersection(set(["E003", "E004"])):
                continue
            if detector == "encoding_score" and text and not ids.intersection(set(["E001", "E002", "E005"])):
                continue
            if detector == "symbol_score" and text and scores.get(detector, 0.0) < 0.55:
                continue
            if detector == "boilerplate_score" and text and not ids.intersection(set(["E006", "E007", "E008", "E009"])):
                continue
            if detector == "boilerplate_score" and ids.intersection(set(["E006", "E007", "E008", "E009"])):
                continue
            if detector == "markup_score" and ids.intersection(set(["E003", "E004"])):
                continue
            if detector == "encoding_score" and ids.intersection(set(["E001", "E002", "E005"])):
                continue
            if detector == "split_candidate_score" and ids.intersection(set(["E011", "E012"])):
                continue
            ids.update(mapped_ids)
    return sorted(ids)


def normalize_action(record):
    action = record.get("action")
    if action:
        return action
    proposed = record.get("proposed_action")
    if proposed == "review_drop_candidate":
        return "drop_doc"
    if proposed == "review_split_candidate":
        return "split_doc"
    if proposed == "review_trim_candidate":
        return "trim_span"
    if proposed == "review_quarantine_candidate":
        return "quarantine"
    return "keep"


def refine_action(action, error_type_ids, chars_before, scores=None, text=None):
    ids = set(error_type_ids or [])
    chars = int(chars_before or 0)
    scores = scores or {}
    norm = normalize_text(text or "")
    archive_score = scores.get("archive_snippet_score", 0.0)
    catalog_score = scores.get("catalog_list_score", 0.0)
    form_score = scores.get("form_comment_score", 0.0)
    no_main_score = scores.get("no_main_body_score", 0.0)
    seo_score = scores.get("seo_spam_score", 0.0)
    badness_score = scores.get("badness_score", 0.0)
    greek_share = scores.get("greek_word_share", 0.0)
    suffix_window = norm[-1800:]
    has_comment_suffix_marker = any(pattern in suffix_window for pattern in COMMENT_SUFFIX_PATTERNS)
    has_related_suffix_marker = any(pattern in suffix_window for pattern in RELATED_SUFFIX_PATTERNS)
    archive_pattern_hits = count_substrings(ARCHIVE_SNIPPET_PATTERNS, norm)
    strong_spam = (
        "E016" in ids
        and (
            seo_score >= 0.65
            or (seo_score >= 0.45 and "E019" not in ids)
            or (seo_score >= 0.45 and greek_share < 0.70)
            or (count_substrings(SEO_SPAM_PATTERNS, norm) >= 2 and greek_share < 0.75)
        )
    )
    strong_no_main_body = (
        "E020" in ids
        and chars < 1200
        and (
            "text-missing" in norm
            or "search results" in norm
            or "found results" in norm
            or "συνεχιστε την αναγνωση" in norm
            or count_substrings(ARCHIVE_SNIPPET_PATTERNS, norm) >= 2
            or form_score >= 0.45
            or catalog_score >= 0.75
            or seo_score >= 0.45
        )
    )
    high_badness_drop = (
        "E014" in ids
        and badness_score >= 0.90
        and (
            "E005" in ids
            or ("E017" in ids and catalog_score >= 0.65)
            or ("E010" in ids and chars < 2500 and archive_score >= 0.55)
            or (greek_share < 0.55 and (archive_score >= 0.45 or catalog_score >= 0.45))
        )
    )
    coherent_short_article_tail = (
        chars >= 500
        and greek_share >= 0.82
        and badness_score < 0.45
        and catalog_score < 0.40
        and seo_score < 0.35
        and (has_comment_suffix_marker or has_related_suffix_marker)
        and (
            (
                "E019" in ids
                and form_score >= 0.45
                and archive_score < 0.45
            )
            or (
                "E010" in ids
                and "E020" in ids
                and archive_score < 0.45
                and archive_pattern_hits <= 2
            )
        )
    )
    coherent_short_body_without_structural_marker = (
        chars >= 500
        and greek_share >= 0.82
        and badness_score < 0.45
        and catalog_score < 0.40
        and seo_score < 0.35
        and archive_score < 0.45
        and archive_pattern_hits <= 2
        and not has_comment_suffix_marker
        and not has_related_suffix_marker
    )

    weak_single_catalog_or_product = (
        "E017" in ids
        and "E010" not in ids
        and "E016" not in ids
        and chars < 1800
        and archive_score < 0.20
        and catalog_score <= 0.60
        and form_score < 0.25
        and badness_score < 0.60
    )
    if weak_single_catalog_or_product:
        return "keep"

    long_comment_or_archive_tail = (
        "E019" in ids
        and chars >= 5000
        and greek_share >= 0.60
        and form_score >= 0.45
    )
    if long_comment_or_archive_tail:
        return "quarantine"

    recoverable_e014 = (
        action == "drop_doc"
        and "E014" in ids
        and "E005" not in ids
        and badness_score >= 0.75
        and not high_badness_drop
        and not strong_spam
        and no_main_score < 0.45
    )
    if recoverable_e014:
        if greek_share >= 0.85 and archive_score < 0.35 and catalog_score < 0.35 and chars >= 2500:
            return "normalize_or_trim_span"
        return "quarantine"

    if coherent_short_article_tail:
        if ids.intersection(set(["E007", "E009", "E019"])):
            return "trim_suffix"
        return "quarantine"
    if coherent_short_body_without_structural_marker:
        return "keep"

    if action == "drop_doc" and "E020" in ids and chars < 700 and no_main_score >= 0.35:
        if "E005" in ids or "E014" in ids or archive_score >= 0.20:
            return "drop_doc"
        return "quarantine"

    if strong_spam or strong_no_main_body or high_badness_drop:
        return "drop_doc"
    if "E020" in ids and chars < 1200:
        weak_short_only = (
            archive_score < 0.35
            and catalog_score < 0.35
            and form_score < 0.35
            and seo_score < 0.35
            and badness_score < 0.75
        )
        if weak_short_only:
            return "keep"
        return "quarantine"
    if "E016" in ids:
        return "quarantine"
    if action == "drop_doc" and ids and ids.issubset(set(["E003", "E004", "E005", "E015", "E017"])):
        return "quarantine"
    if ids.intersection(set(["E010", "E017"])) and ids.intersection(set(["E011", "E012"])):
        return "quarantine"
    if ids.intersection(set(["E010", "E017", "E020"])) and action in set(["keep", "trim_span"]):
        return "quarantine"
    if "E019" in ids and action == "keep":
        return "quarantine"
    return action


def read_doc_text(path, limit):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        if limit is None or limit < 0:
            return handle.read()
        return handle.read(limit)


def normalize_text(text):
    folded = text.casefold()
    decomposed = unicodedata.normalize("NFD", folded)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def regex_count(patterns, text, flags=re.I | re.M):
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, flags))
    return total


def line_density(lines, max_len):
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return 0.0
    short = sum(1 for line in nonempty if len(line.strip()) <= max_len)
    return float(short) / float(len(nonempty))


def text_based_error_type_ids(record, scores, threshold, text=None, text_features=None):
    text = text if text is not None else read_doc_text(record.get("doc_text_path"), 200000)
    if not text:
        return set()
    text_features = text_features or compute_text_features(text)
    norm = normalize_text(text)
    lines = [line.strip() for line in norm.splitlines() if line.strip()]
    top = "\n".join(lines[:30])
    bottom = "\n".join(lines[-30:])
    ids = set()

    if scores.get("encoding_score", 0.0) >= threshold:
        if "\ufffd" in text or any((ord(ch) < 32 and ch not in "\n\r\t") for ch in text):
            ids.add("E001")
        if regex_count(ENCODED_PATTERNS, text):
            ids.add("E002")
        if not ids:
            ids.add("E005")

    if scores.get("markup_score", 0.0) >= threshold:
        if regex_count(MARKUP_PATTERNS, text):
            ids.add("E003")
        if regex_count(URL_PATTERNS, text):
            ids.add("E004")

    if scores.get("symbol_score", 0.0) >= threshold:
        suspicious_symbols = sum(
            1 for ch in text
            if unicodedata.category(ch).startswith("S") and ch not in set("€$%&+-=/*#@")
        )
        if scores.get("symbol_score", 0.0) >= 0.55 or suspicious_symbols >= 5 or "\ufffd" in text:
            ids.add("E005")

    if scores.get("boilerplate_score", 0.0) >= threshold:
        top_hits = any(pattern in top for pattern in TOP_CHROME_PATTERNS)
        bottom_hits = any(pattern in bottom for pattern in FOOTER_PATTERNS)
        cookie_hits = any(pattern in norm for pattern in COOKIE_LOGIN_PATTERNS)
        related_hits = any(pattern in norm for pattern in RELATED_COMMENT_PATTERNS)
        if top_hits or line_density(lines[:30], 35) >= 0.70:
            ids.add("E006")
        if bottom_hits or line_density(lines[-30:], 35) >= 0.70:
            ids.add("E007")
        if cookie_hits:
            ids.add("E008")
        if related_hits:
            ids.add("E009")
    if text_features.get("archive_snippet_score", 0.0) >= threshold:
        ids.add("E010")
        if count_substrings(ARCHIVE_SNIPPET_PATTERNS, norm) >= 2:
            ids.add("E009")

    if text_features.get("catalog_list_score", 0.0) >= threshold:
        ids.add("E017")
        if regex_count(URL_PATTERNS, text) >= 2:
            ids.add("E004")

    if text_features.get("seo_spam_score", 0.0) >= threshold:
        ids.add("E016")
        if text_features.get("greek_word_share", 1.0) < 0.65:
            ids.add("E015")

    if text_features.get("form_comment_score", 0.0) >= threshold:
        ids.add("E019")
        if any(pattern in bottom for pattern in FORM_COMMENT_PATTERNS):
            ids.add("E007")

    if text_features.get("no_main_body_score", 0.0) >= threshold:
        ids.add("E020")

    if scores.get("split_candidate_score", 0.0) >= threshold:
        split_hits = regex_count(SPLIT_PATTERNS, norm)
        if split_hits >= 2:
            ids.update(["E011", "E012"])
        else:
            ids.add("E011")

    if scores.get("internal_repetition_score", 0.0) >= threshold:
        ids.update(["E013", "E018"])

    if scores.get("badness_score", 0.0) >= threshold:
        ids.add("E014")

    if scores.get("lang_drift_score", 0.0) >= threshold:
        ids.add("E015")

    return ids


def evidence_excerpt(record, error_type_ids, max_chars):
    text = read_doc_text(record.get("doc_text_path"), max_chars * 4)
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:max_chars]

    patterns = []
    if any(err in error_type_ids for err in ("E003", "E004")):
        patterns.extend([r"https?://", r"\bwww\.", r"class=", r"href=", r"<\/?\w+", r"\{|\}"])
    if any(err in error_type_ids for err in ("E006", "E007", "E008", "E009")):
        patterns.extend([
            r"cookie", r"newsletter", r"facebook", r"instagram", r"youtube",
            r"σχ[oό]λια", r"κοινοπο", r"επικοινων", r"αναζ[ηή]τη",
        ])
    if any(err in error_type_ids for err in ("E010", "E011", "E012", "E017", "E019", "E020")):
        patterns.extend([
            r"read more", r"posted", r"archive", r"posts tagged", r"leave a comment",
            r"δείτε", r"περισσότερα", r"φόρμα", r"σχολιά", r"προσφορ", r"έκπτωση",
            r"καλάθι", r"€",
        ])
    if any(err in error_type_ids for err in ("E011", "E012")):
        patterns.extend([r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", r"δημοσιε", r"αναρτη", r"τίτλος|τιτλος"])

    if patterns:
        combined = re.compile("|".join(patterns), re.I)
        hits = [line for line in lines if combined.search(line)]
        if hits:
            return "\n".join(hits[:5])[:max_chars]

    if any(err in error_type_ids for err in ("E006", "E007", "E008", "E009")):
        excerpt = "\n".join(lines[:8] + ["..."] + lines[-8:])
    else:
        excerpt = "\n".join(lines[:12])
    return excerpt[:max_chars]


def normalize_record(record, threshold, timestamp, excerpt_chars):
    source_doc_id = record.get("source_doc_id")
    text = read_doc_text(record.get("doc_text_path"), 200000)
    text_features = compute_text_features(text)
    scores = detector_scores(record)
    scores.update(text_features)
    error_type_ids = infer_error_type_ids(record, threshold, scores, text, text_features)
    char_count = record.get("chars_before", record.get("char_count"))
    action = refine_action(normalize_action(record), error_type_ids, char_count, scores, text)
    text_sha = record.get("text_sha256_before")
    if not text_sha and not record.get("text_truncated"):
        text_path = record.get("doc_text_path")
        text = read_doc_text(text_path, -1)
        text_sha = stable_sha256(text) if text else None

    return collections.OrderedDict([
        ("schema_version", SCHEMA_VERSION),
        ("source_doc_id", source_doc_id),
        ("parent_source_doc_id", record.get("parent_source_doc_id") or source_doc_id),
        ("derived_doc_id", record.get("derived_doc_id")),
        ("is_shadow_record", bool(record.get("is_shadow_record", False))),
        ("doc_key", record.get("doc_key")),
        ("source_dataset", record.get("source_dataset") or SOURCE_DATASET),
        ("url", record.get("url")),
        ("host", record.get("host")),
        ("crawl_id", record.get("crawl_id")),
        ("quality_bin", record.get("quality_bin")),
        ("text_sha256_before", text_sha),
        ("error_type_ids", error_type_ids),
        ("action", action),
        ("action_status", record.get("action_status") or "candidate"),
        ("confidence", record.get("confidence", record.get("max_artifact_score"))),
        ("reason_codes", record.get("reason_codes") or []),
        ("detector_scores", scores),
        ("span_ranges", record.get("span_ranges") or []),
        ("split_parts", record.get("split_parts") or []),
        ("text_sha256_after", record.get("text_sha256_after")),
        ("chars_before", char_count),
        ("chars_after", record.get("chars_after")),
        ("chars_removed", record.get("chars_removed")),
        ("tokens_before", record.get("tokens_before")),
        ("tokens_after", record.get("tokens_after")),
        ("tokens_removed", record.get("tokens_removed")),
        ("good_text_loss_estimate", record.get("good_text_loss_estimate")),
        ("sample_set", record.get("sample_set") or []),
        ("evidence_excerpt", record.get("evidence_excerpt") or evidence_excerpt(record, error_type_ids, excerpt_chars)),
        ("doc_text_path", record.get("doc_text_path")),
        ("review_label", record.get("review_label")),
        ("reviewer", record.get("reviewer")),
        ("review_notes", record.get("review_notes")),
        ("policy_id", record.get("policy_id") or "P0_candidate_review_manifest"),
        ("policy_version", record.get("policy_version") or timestamp),
        ("created_at_utc", record.get("created_at_utc") or timestamp),
    ])


def increment(counter, key):
    counter[str(key) if key is not None else "null"] += 1


def summarize(records):
    summary = {
        "total_records": len(records),
        "by_error_type": collections.Counter(),
        "by_action": collections.Counter(),
        "by_action_status": collections.Counter(),
        "by_sample_set": collections.Counter(),
        "by_quality_bin": collections.Counter(),
        "by_host": collections.Counter(),
        "no_candidate_error_type": 0,
    }
    for record in records:
        ids = record.get("error_type_ids") or []
        if not ids:
            summary["no_candidate_error_type"] += 1
        for err in ids:
            increment(summary["by_error_type"], err)
        increment(summary["by_action"], record.get("action"))
        increment(summary["by_action_status"], record.get("action_status"))
        increment(summary["by_quality_bin"], record.get("quality_bin"))
        increment(summary["by_host"], record.get("host"))
        for sample_set in record.get("sample_set") or []:
            increment(summary["by_sample_set"], sample_set)
    return summary


def score_for_error(record, error_id):
    score_name = ERROR_TO_SCORE.get(error_id)
    return float((record.get("detector_scores") or {}).get(score_name) or 0.0)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def jsonable_summary(summary, top_hosts):
    result = {}
    for key, value in summary.items():
        if isinstance(value, collections.Counter):
            result[key] = dict(value.most_common(top_hosts if key == "by_host" else None))
        else:
            result[key] = value
    return result


def write_markdown(path, args, records, summary, outputs, top_examples):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Candidate HPLT Cleaning Issue Summary\n\n")
        out.write("This report is non-destructive. It enriches the review bundle with candidate error IDs and normalized actions; it does not edit original HPLT rows.\n\n")
        out.write("## Inputs\n\n")
        out.write("- manifest: `%s`\n" % args.manifest)
        out.write("- threshold: `%.3f`\n" % args.threshold)
        out.write("- records: `%d`\n\n" % len(records))
        out.write("## Outputs\n\n")
        for name, value in outputs:
            out.write("- %s: `%s`\n" % (name, value))
        out.write("\n## Counts By Action\n\n")
        out.write("| Action | Count |\n| --- | ---: |\n")
        for action, count in summary["by_action"].most_common():
            out.write("| `%s` | %d |\n" % (action, count))
        out.write("\n## Candidate Error Type Counts\n\n")
        out.write("| ID | Candidate type | Count | Retrieval |\n| --- | --- | ---: | --- |\n")
        for error_id, count in summary["by_error_type"].most_common():
            out.write("| `%s` | %s | %d | `jq -c 'select(.error_type_ids[]? == \"%s\")' action_manifest.jsonl` |\n" % (
                error_id, ERROR_NAMES.get(error_id, ""), count, error_id))
        out.write("| none | no candidate ID above threshold | %d | `jq -c 'select((.error_type_ids | length) == 0)' action_manifest.jsonl` |\n" % summary["no_candidate_error_type"])
        out.write("\n## Top Hosts\n\n")
        out.write("| Host | Count |\n| --- | ---: |\n")
        for host, count in summary["by_host"].most_common(20):
            out.write("| `%s` | %d |\n" % (host, count))
        out.write("\n## Example Records By Candidate Type\n\n")
        for error_id in ERROR_NAMES:
            examples = top_examples.get(error_id) or []
            if not examples:
                continue
            out.write("### %s - %s\n\n" % (error_id, ERROR_NAMES[error_id]))
            out.write("Query: `jq -c 'select(.error_type_ids[]? == \"%s\")' action_manifest.jsonl`\n\n" % error_id)
            out.write("| Score | Action | Host | QBin | Source doc | Text path |\n| ---: | --- | --- | ---: | --- | --- |\n")
            for record in examples:
                out.write("| %.3f | `%s` | `%s` | %s | `%s` | `%s` |\n" % (
                    score_for_error(record, error_id),
                    record.get("action"),
                    record.get("host"),
                    record.get("quality_bin"),
                    record.get("source_doc_id"),
                    record.get("doc_text_path"),
                ))
            out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--top-examples-per-type", type=int, default=20)
    parser.add_argument("--excerpt-chars", type=int, default=1000)
    parser.add_argument("--top-hosts", type=int, default=100)
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    mkdir(args.output_dir)
    index_dir = os.path.join(args.output_dir, "issue_type_index")
    mkdir(index_dir)

    records = [normalize_record(record, args.threshold, timestamp, args.excerpt_chars) for record in read_jsonl(args.manifest)]
    action_manifest = os.path.join(args.output_dir, "action_manifest_%s.jsonl" % timestamp)
    with open(action_manifest, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    for error_id in ERROR_NAMES:
        path = os.path.join(index_dir, "%s.jsonl" % error_id)
        with open(path, "w", encoding="utf-8") as out:
            for record in records:
                if error_id in (record.get("error_type_ids") or []):
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

    no_id_path = os.path.join(index_dir, "NO_CANDIDATE_ERROR_TYPE.jsonl")
    with open(no_id_path, "w", encoding="utf-8") as out:
        for record in records:
            if not record.get("error_type_ids"):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records)
    top_examples = {}
    for error_id in ERROR_NAMES:
        selected = [record for record in records if error_id in (record.get("error_type_ids") or [])]
        selected.sort(key=lambda item: (score_for_error(item, error_id), float(item.get("confidence") or 0.0)), reverse=True)
        top_examples[error_id] = selected[:args.top_examples_per_type]

    summary_json = os.path.join(args.output_dir, "candidate_issue_summary_%s.json" % timestamp)
    payload = jsonable_summary(summary, args.top_hosts)
    payload["input_manifest"] = args.manifest
    payload["action_manifest"] = action_manifest
    payload["issue_type_index_dir"] = index_dir
    payload["threshold"] = args.threshold
    payload["top_examples_per_type"] = args.top_examples_per_type
    write_json(summary_json, payload)

    summary_md = os.path.join(args.output_dir, "candidate_issue_summary_%s.md" % timestamp)
    outputs = [
        ("action_manifest", action_manifest),
        ("summary_json", summary_json),
        ("issue_type_index_dir", index_dir),
    ]
    write_markdown(summary_md, args, records, summary, outputs, top_examples)

    print(json.dumps({
        "event": "complete",
        "records": len(records),
        "action_manifest": action_manifest,
        "summary_json": summary_json,
        "summary_md": summary_md,
        "issue_type_index_dir": index_dir,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
