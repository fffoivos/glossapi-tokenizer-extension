#!/usr/bin/env python3
"""Generate per-line feature fixtures from the Python extractor.

The Rust port is validated against these: for every line, the 35 deterministic
counts, the 34 line-shape values and the deterministic bibliography role must
match. Run this from `15_clean_academic/eval` (it imports `sequence_models`);
it needs only numpy, so it runs on a laptop as well as on the cluster.

    python3 bib_line_model/fixtures/generate_fixtures.py --out fixtures/lines.json

`--from-jsonl FILE --text-field text` widens the set with real corpus lines
once cluster data is at hand; the built-in seed set is a hand-picked spread of
the shapes seen in the Greek academic sources -- bibliography entries in several
citation styles, the prose confusers that share their vocabulary, table rows,
table-of-contents leaders, OCR wreckage and Unicode edge cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# real shapes observed in greek_phd / openarchives / kallipos / libduth
SEED_LINES: list[str] = [
    # --- bibliography entries, several styles ---
    "Alchian, A.A. and Demsetz, H. (1972). Production, information costs and economic organization. American Economic Review 62, pp.777-795.",
    "Barney, J. (1991). Firm resources and sustained competitive advantage. Journal of Management 17(1), pp.99120.",
    "- [1] J. Bakus, Can Programming Be Liberated from the von Neumann Style?, CACM , 21(8), 613-641, 1978.",
    "Δαγτόγλου Π.Δ., «Διοικητικό Δικονομικό Δίκαιο», Εκδ. Σάκκουλα, έκδ. 6η, 2014.",
    "Αγγελόπουλος, Η., Καραγιάννης, Π. & Φωκάς, Ε. (2002). «Αντιλήψεις διευθυντών σχολικών μονάδων Ν. Αχαΐας».",
    "IFLA (1963). International Conference on Cataloguing Principles (Paris: 1961). Report. London: International Federation of Library Associations.",
    "2 P.R. O'Connor: Absorption of subretinal fluid after external scleral buckling. Am J Ophthalmol 76:30-34,1973",
    "Διαθέσιμο στο: https://journal.educircle.gr/images/teuxos/2013/2/teyxos2.pdf (Ανακτήθηκε 04/12/2020)",
    "Zegers-Hochschild F, Adamson GD, Dyer S, et al. The International Glossary on Infertility. Hum Reprod. 2017;32(9):1786-1801. doi:10.1093/humrep/dex234",
    "Αγριαντώνη Χριστίνα, Οι απαρχές της εκβιομηχάνισης στην Ελλάδα τον 19ο αιώνα, Ιστορικό Αρχείο, Αθήνα, 1986.",
    # --- prose that shares the vocabulary (the confusers) ---
    "« Οι επιχειρήσεις δεν είναι τυχαίες συναθροίσεις ανεξαρτήτων παραγόντων » (Mintzberg, 2005, σελ. 334).",
    "Ο Oliver Williamson (1975; 1985) υιοθετώντας την οπτική του Coase θεωρεί ότι η επιχείρηση είναι μηχανισμός.",
    "- · Η επιχείρηση ως πλέγμα συμβάσεων ( nexus of contracts) (Alchian and Demsetz, 1972) ή χαλαρότερα.",
    "Σύμφωνα με τα διαθέσιμα στοιχεία όσον αφορά στην εικοσαετία 1990-2009, το μερίδιο των πετρελαιοειδών αυξήθηκε.",
    "3 Ν . 2121/1993, άρθρο 18 παρ . 3: « Εάν για την ελεύθερη αναπαραγωγή του έργου χρησιμοποιούνται τεχνικά μέσα ».",
    "Έχοντας ολοκληρώσει τη μελέτη του παρόντος κεφαλαίου, θα πρέπει να είστε σε θέση να αναφέρετε τους ορισμούς.",
    # --- headings ---
    "## ΒΙΒΛΙΟΓΡΑΦΙΑ",
    "## 16. References - Bibliography",
    "## Βιβλιογραφική Ανασκόπηση",
    "Α. ΕΛΛΗΝΙΚΗ",
    "ΒΙΒΛΙΟΓΡΑΦΙΑ - ΠΗΓΕΣ",
    "## Σε επιστημονικά περιοδικά:",
    # --- tables and table-of-contents ---
    "| Βιβλιογραφία .................... 325                    |",
    "| 164 | Ιωάννου , Κύρος     |   22 | Θεσσαλονίκη     |     | Απολυτήριον Γυμνασίου Αθηνών (1717) |",
    "| dc.classificationURI     | **N/A**-Πληροφόρηση                     |",
    # --- OCR wreckage and degenerate input ---
    "## ΛΙΟΓ ;",
    "## ✶✳✶ /a80❡/a114✐❣/a114❛❢➔ /a116♦✉ ♣/a114♦❜❧➔♠❛/a116♦❝",
    "\\_\\_\\_\\_\\_",
    "<!-- image -->",
    ".",
    "1001",
    # --- Unicode / whitespace edge cases the analyzer must survive ---
    "",
    "   ",
    "\ttab\tseparated\tvalues\t",
    "a b",                      # non-breaking space
    "Ｆｕｌｌｗｉｄｔｈ ２０２０",        # fullwidth forms
    "ΆΈΉΊΌΎΏ ᾳῃῳ ϊϋΐΰ",              # polytonic + diacritics
    "égale",                   # combining acute (NFD)
    "x" * 400,                       # long line
    "Smith, J. (2020). " * 20,       # repeated entry shape
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--from-jsonl")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    from sequence_models.bibliography_entry_dataset import FEATURE_NAMES
    from sequence_models.bibliography_positional_features import extract_positional_line
    from sequence_models.bibliography_role_features import LINE_SHAPE_NAMES, line_shape
    from sequence_models.deterministic_structure import analyze_bib_line

    lines = list(SEED_LINES)
    if args.from_jsonl:
        with open(args.from_jsonl, encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                doc = json.loads(raw)
                for text in str(doc.get(args.text_field, "")).split("\n"):
                    lines.append(text)
                    if len(lines) >= args.limit:
                        break
                if len(lines) >= args.limit:
                    break

    cases = []
    for text in lines:
        encoding = extract_positional_line(text)
        cases.append({
            "text": text,
            "counts": [int(v) for v in encoding.counts],
            "gap_summaries": [float(v) for v in encoding.gap_summaries],
            "shape": [float(v) for v in line_shape(text)],
            "bib_role": analyze_bib_line(text, 0).role.name,
        })

    payload = {
        "schema_version": "bib-line-feature-fixture-v1",
        "feature_names": list(FEATURE_NAMES),
        "shape_names": list(LINE_SHAPE_NAMES),
        "n_cases": len(cases),
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(f"wrote {len(cases)} cases -> {out}", file=sys.stderr)
    print(f"  {len(FEATURE_NAMES)} counts, {len(LINE_SHAPE_NAMES)} shape values", file=sys.stderr)


if __name__ == "__main__":
    main()
