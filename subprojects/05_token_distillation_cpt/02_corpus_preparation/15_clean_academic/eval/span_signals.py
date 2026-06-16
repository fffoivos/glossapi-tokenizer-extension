#!/usr/bin/env python3
"""Shared per-line signal primitives for the bibliography-span classifiers (line_lr, window_clf,
decode_spans, line_crf). Single source of truth, mirroring the Rust `reference_signals.rs` so the
interpretable line model can be ported to the crate. Regexes are the ones already calibrated in
iterate2.py / reference_signals.rs — do not fork them."""
import re

YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
YEAR_PAREN = re.compile(r"\((1[5-9]\d{2}|20\d{2})\)")
PAGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d")
PLACE_PUB = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]")
NAME = re.compile(r"[Α-ΩA-ZΆ-Ώἀ-῿][α-ωa-zά-ώϊϋἀ-῿]{2,}")  # incl. polytonic (Greek Extended)
AUTHOR = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+")  # Surname, F.
# Greek-style author: "Επώνυμο, Ό." or "Επώνυμο Ό." (Greek capital surname + Greek capital initial)
GREEK_AUTHOR = re.compile(r"[Α-ΩΆ-Ώἀ-῿][α-ωά-ώϊϋἀ-῿]+\s*,?\s*[Α-ΩΆ-Ώἀ-῿]\.")  # incl. polytonic
# Greek bibliographic edition/translation/volume markers — strong Greek reference-entry signal
GR_EDMARK = re.compile(r"\b(μτφρ|επιμ|εκδ|εκδόσεις|εκδ́σεις|τόμ|τχ|σσ|σελ|αρ\. φ|ό\.π)\b\.?|"
                       r"\b(Αθήνα|Θεσσαλονίκη|Αθήναι|Λευκωσία)\s*[:·,]")
EDITOR = re.compile(r"επιμ\.|\(επιμ|eds?\.|\(eds?\)|μτφρ\.")
URL = re.compile(r"https?://|doi\.|www\.|\.gr/|\.com/")
ATX_HEADER = re.compile(r"^\s{0,3}#{1,6}\s")
NUM_PREFIX = re.compile(r"^\s*(\[?\d{1,3}\]?[.)\]]|\d{1,3}\s)")  # vancouver [12] / 12. entry marker
LOC = ["σσ.", "σ.", "τ.", "τόμ.", "εκδ.", "εκδόσεις", "επιμ.", "μτφρ.", "vol.", "pp.", "eds.", "ed.", "et al"]
H_BIB = ["βιβλιογραφ", "αναφορ", "πηγ", "references", "bibliograph", "δικτυογραφ", "ελληνογλωσσ",
         "ξενογλωσσ", "ιστοσελιδ", "works cited", "further reading"]
H_CV = ["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "σεμιναρ",
        "συνεδρι", "καταλογος ανακοιν", "curriculum"]
H_APP = ["δραστηριοτ", "λεξεις-κλειδ", "λεξεις κλειδ", "απαντησ", "προαπαιτ", "ασκησ", "περιληψ",
         "συνοψ", "κριτηρι", "παραρτημα", "appendix"]


def _low(s):
    return s.lower()


def is_entry(l):
    """A line is a reference entry: a capitalized NAME token + (year | page-range | Surname,F. |
    Place:Publisher | locator). Mirrors beta_gate_model.rs::is_entry_line."""
    if not NAME.search(l):
        return False
    low = l.lower()
    return bool(YEAR.search(l) or PAGE.search(l) or AUTHOR.search(l) or PLACE_PUB.search(l)
                or GREEK_AUTHOR.search(l) or GR_EDMARK.search(l) or any(x in low for x in LOC))


_LAT = re.compile(r"[A-Za-z]")
_GRK = re.compile(r"[Α-Ωα-ωΆ-ώϊϋΐΰ]")


def latin_fraction(s):
    lat = len(_LAT.findall(s))
    gr = len(_GRK.findall(s))
    d = lat + gr
    return lat / d if d else 0.0


def is_bib_header(l):
    return bool(ATX_HEADER.search(l)) and any(x in l.lower() for x in H_BIB)


def header_family(l):
    """Return 'bib' | 'cv' | 'app' | '' for a header-ish line (used as deny/allow context)."""
    low = l.lower()
    if any(x in low for x in H_CV):
        return "cv"
    if any(x in low for x in H_APP):
        return "app"
    if any(x in low for x in H_BIB):
        return "bib"
    return ""


def line_signals(l):
    """Own-line scalar/binary signals for a single line (no neighbor context). Returns a dict; the
    line model adds context features (neighbor entry-density, position, header-distance) on top."""
    low = l.lower()
    txt = l.strip()
    return {
        "is_entry": 1.0 if is_entry(l) else 0.0,
        "author": 1.0 if AUTHOR.search(l) else 0.0,
        "greek_author": 1.0 if GREEK_AUTHOR.search(l) else 0.0,
        "gr_edmark": 1.0 if GR_EDMARK.search(l) else 0.0,
        "year_paren": 1.0 if YEAR_PAREN.search(l) else 0.0,
        "year_bare": 1.0 if YEAR.search(l) else 0.0,
        "page": 1.0 if PAGE.search(l) else 0.0,
        "place": 1.0 if PLACE_PUB.search(l) else 0.0,
        "locator": 1.0 if any(x in low for x in LOC) else 0.0,
        "editor": 1.0 if EDITOR.search(l) else 0.0,
        "url": 1.0 if URL.search(l) else 0.0,
        "num_prefix": 1.0 if NUM_PREFIX.search(l) else 0.0,
        "latin_frac": latin_fraction(l),
        "is_bib_header": 1.0 if is_bib_header(l) else 0.0,
        "is_header": 1.0 if ATX_HEADER.search(l) else 0.0,
        "short_line": 1.0 if len(txt) < 40 else 0.0,
        "very_short": 1.0 if len(txt) < 12 else 0.0,
    }


# stable feature order for the line model (own-line block); context features appended by line_lr.py
OWN_KEYS = ["is_entry", "author", "year_paren", "year_bare", "page", "place", "locator",
            "editor", "url", "num_prefix", "latin_frac", "is_bib_header", "is_header",
            "short_line", "very_short"]
