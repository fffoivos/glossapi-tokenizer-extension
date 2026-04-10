#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable


MAIN_LABELS = OrderedDict(
    [
        ("MT", "Machine translated or generated"),
        ("LY", "Lyrical"),
        ("SP", "Spoken"),
        ("ID", "Interactive discussion"),
        ("NA", "Narrative"),
        ("HI", "How-to / instructional"),
        ("IN", "Informational description / explanation"),
        ("OP", "Opinion"),
        ("IP", "Informational persuasion"),
    ]
)


# These are the current lower-level labels exposed on the Turku register
# abbreviations page. HPLT also carries additional older fine-grained labels
# from the CORE taxonomy; see LEGACY_EXTRA_SUBLABELS below.
SIMPLIFIED_SUBLABELS = OrderedDict(
    [
        ("it", {"label": "Interview", "parent": "SP", "tier": "simplified"}),
        ("os", {"label": "Other spoken", "parent": "SP", "tier": "simplified"}),
        ("ne", {"label": "News report", "parent": "NA", "tier": "simplified"}),
        ("sr", {"label": "Sports report", "parent": "NA", "tier": "simplified"}),
        ("nb", {"label": "Narrative blog", "parent": "NA", "tier": "simplified"}),
        ("on", {"label": "Other narrative", "parent": "NA", "tier": "simplified"}),
        ("re", {"label": "Recipe", "parent": "HI", "tier": "simplified"}),
        ("oh", {"label": "Other how-to", "parent": "HI", "tier": "simplified"}),
        ("en", {"label": "Encyclopedia article", "parent": "IN", "tier": "simplified"}),
        ("ra", {"label": "Research article", "parent": "IN", "tier": "simplified"}),
        (
            "dtp",
            {
                "label": "Description of a thing or person",
                "parent": "IN",
                "tier": "simplified",
            },
        ),
        ("fi", {"label": "FAQ about information", "parent": "IN", "tier": "simplified"}),
        ("lt", {"label": "Legal terms and conditions", "parent": "IN", "tier": "simplified"}),
        (
            "oi",
            {"label": "Other informational description", "parent": "IN", "tier": "simplified"},
        ),
        ("rv", {"label": "Review", "parent": "OP", "tier": "simplified"}),
        ("ob", {"label": "Opinion blog", "parent": "OP", "tier": "simplified"}),
        (
            "rs",
            {"label": "Denominational religious blog / sermon", "parent": "OP", "tier": "simplified"},
        ),
        ("av", {"label": "Advice", "parent": "OP", "tier": "simplified"}),
        ("oo", {"label": "Other opinion", "parent": "OP", "tier": "simplified"}),
        (
            "ds",
            {"label": "Description with intent to sell", "parent": "IP", "tier": "simplified"},
        ),
        (
            "ed",
            {"label": "News and opinion blog or editorial", "parent": "IP", "tier": "simplified"},
        ),
        ("oe", {"label": "Other informational persuasion", "parent": "IP", "tier": "simplified"}),
    ]
)


# HPLT's schema also includes older CORE fine labels. They coexist with the
# simplified labels above rather than replacing them.
LEGACY_EXTRA_SUBLABELS = OrderedDict(
    [
        ("cm", {"label": "Course materials", "parent": "IN", "tier": "legacy"}),
        ("dp", {"label": "Description of a person", "parent": "IN", "tier": "legacy"}),
        ("dt", {"label": "Description of a thing", "parent": "IN", "tier": "legacy"}),
        ("ib", {"label": "Information blog", "parent": "IN", "tier": "legacy"}),
        ("tr", {"label": "Technical report", "parent": "IN", "tier": "legacy"}),
        ("ha", {"label": "Historical article", "parent": "NA", "tier": "legacy"}),
        ("ma", {"label": "Magazine article", "parent": "NA", "tier": "legacy"}),
        ("pb", {"label": "Personal blog", "parent": "NA", "tier": "legacy"}),
        ("tb", {"label": "Travel blog", "parent": "NA", "tier": "legacy"}),
        ("fh", {"label": "FAQ about how-to", "parent": "HI", "tier": "legacy"}),
        ("ht", {"label": "How-to", "parent": "HI", "tier": "legacy"}),
        ("ts", {"label": "Technical support", "parent": "HI", "tier": "legacy"}),
        ("ol", {"label": "Other lyrical", "parent": "LY", "tier": "legacy"}),
        ("po", {"label": "Poem", "parent": "LY", "tier": "legacy"}),
        ("pr", {"label": "Prayer", "parent": "LY", "tier": "legacy"}),
        ("sl", {"label": "Song lyrics", "parent": "LY", "tier": "legacy"}),
        ("ad", {"label": "Advertisement", "parent": "IP", "tier": "legacy"}),
        ("pa", {"label": "Persuasive article or essay", "parent": "IP", "tier": "legacy"}),
        ("le", {"label": "Letter to editor", "parent": "IP", "tier": "legacy"}),
        ("df", {"label": "Discussion forum", "parent": "ID", "tier": "legacy"}),
        ("of", {"label": "Other forum", "parent": "ID", "tier": "legacy"}),
        ("qa", {"label": "Question / answer forum", "parent": "ID", "tier": "legacy"}),
        ("rr", {"label": "Reader / viewer responses", "parent": "ID", "tier": "legacy"}),
        ("fs", {"label": "Formal speech", "parent": "SP", "tier": "legacy"}),
        ("ta", {"label": "Transcript of video / audio", "parent": "SP", "tier": "legacy"}),
        ("tv", {"label": "TV / movie script", "parent": "SP", "tier": "legacy"}),
    ]
)


ALL_SUBLABELS = OrderedDict()
ALL_SUBLABELS.update(SIMPLIFIED_SUBLABELS)
ALL_SUBLABELS.update(LEGACY_EXTRA_SUBLABELS)

HPLT_SCHEMA_MAIN_ORDER = list(MAIN_LABELS.keys())
HPLT_SCHEMA_SUBLABEL_ORDER = list(ALL_SUBLABELS.keys())


def label_name(code: str) -> str:
    if code in MAIN_LABELS:
        return MAIN_LABELS[code]
    if code in ALL_SUBLABELS:
        return ALL_SUBLABELS[code]["label"]
    return code


def parent_label(code: str) -> str | None:
    if code in ALL_SUBLABELS:
        return str(ALL_SUBLABELS[code]["parent"])
    return None


def sorted_scores(web_register: dict[str, float] | None, codes: Iterable[str]) -> list[tuple[str, float]]:
    if not web_register:
        return []
    scored = [(code, float(web_register.get(code, 0.0))) for code in codes]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def top_main_label(web_register: dict[str, float] | None) -> tuple[str | None, float]:
    scores = sorted_scores(web_register, HPLT_SCHEMA_MAIN_ORDER)
    if not scores:
        return None, 0.0
    return scores[0]


def top_sub_label(web_register: dict[str, float] | None) -> tuple[str | None, float]:
    scores = sorted_scores(web_register, HPLT_SCHEMA_SUBLABEL_ORDER)
    if not scores:
        return None, 0.0
    return scores[0]

