# Eval B — β-gate confusion matrix (Opus 4.8 annotation vs detector)

n=64 sections · hallucination-flagged quotes=2 · inter-annotator κ (is_reference_list, 9 double)=1.00

## Confusion matrix (rows = detector gate, cols = Opus truth)

| | truth: bibliography | truth: NOT bib |
|---|---:|---:|
| **gate: bib (flag/drop)** | 25 (TP) | 7 (FP) |
| **gate: kept-non-bib** | 4 (FN) | 28 (TN) |

**precision=0.781  recall=0.862  F1=0.820  accuracy=0.828**

Precision = of sections the gate flags as bibliography, how many truly are. Recall = of true bibliographies, how many the gate flags.

## Error breakdown by sampling stratum

| stratum | FP | FN | ok |
|---|---:|---:|---:|
| bib_strong | 5 | 0 | 11 |
| bib_weak | 2 | 0 | 14 |
| kept_hasyear | 0 | 4 | 12 |
| kept_noyear | 0 | 0 | 16 |

## False positives (gate said bib, Opus says NOT) — over-removal risk

- `B0004` gate=`bib:header_stem` → Opus kind=**not_reference** style=na feat={'yc': 0, 'yd': 0.0, 'latin': 0.0, 'dash': 0.4, 'pos': 0.06}  «Ο συχνά χρησιμοποιούμενος όρος « Πηγές Ενέργειας » δεν ευσταθεί από επιστημονική»
- `B0051` gate=`bib:header_stem` → Opus kind=**not_reference** style=na feat={'yc': 0, 'yd': 0.0, 'latin': 0.78, 'dash': 0.0, 'pos': 0.2}  «| AcP                | Archiv für die civilistische Praxis (περιοδικό)          »
- `B0005` gate=`bib:citation_shape` → Opus kind=**not_reference** style=none feat={'yc': 1, 'yd': 0.25, 'latin': 0.46, 'dash': 1.0, 'pos': 0.78}  «Δραστηριότητες»
- `B0025` gate=`bib:citation_shape` → Opus kind=**not_reference** style=na feat={'yc': 2, 'yd': 0.5, 'latin': 1.0, 'dash': 0.0, 'pos': 0.36}  «Petrus  Grimani Dei  Gratia  Dux  Venetiarum  e.t.c  Nobilibus  et  Sapientibus »
- `B0033` gate=`bib:citation_shape` → Opus kind=**not_reference** style=footnote_humanities feat={'yc': 1, 'yd': 1.0, 'latin': 0.44, 'dash': 0.0, 'pos': 0.52}  «Ήρθε στο φως δίπλα στην δεξαμενή/impluvium (Soles & Davaras 1996, 189).»
- `B0046` gate=`bib:citation_shape` → Opus kind=**cv_publication_list** style=vancouver_numbered feat={'yc': 2, 'yd': 0.67, 'latin': 0.99, 'dash': 0.0, 'pos': 0.98}  «ΔΗΜΟΣΙΕΥΣΗ 2»
- `B0054` gate=`bib:header_stem` → Opus kind=**cv_publication_list** style=none feat={'yc': 5, 'yd': 1.0, 'latin': 1.0, 'dash': 1.0, 'pos': 0.2}  «ΚΑΤΑΛΟΓΟΣ ΞΕΝΟΓΛΩΣΣΩΝ ΑΝΑΚΟΙΝΩΣΕΩΝ ΣΕ ΔΙΕΘΝΗ ΣΥΝΕΔΡΙΑ»

## False negatives (gate kept, Opus says bibliography) — under-removal

- `B0019` gate=`kept-non-bib:unconfirmed` → Opus kind=**chapter_bibliography** style=chicago_turabian lang=greek feat={'yc': 1, 'yd': 1.0, 'latin': 0.0, 'dash': 0.0, 'pos': 0.77}  «Τάκης Καλογερόπουλος, Το λεξικό της ελληνικής μουσικής , τόμος 3, Γιαλλελής, Αθή»
- `B0022` gate=`kept-non-bib:unconfirmed` → Opus kind=**end_bibliography** style=chicago_turabian lang=mixed_greek_foreign feat={'yc': 6, 'yd': 1.5, 'latin': 0.08, 'dash': 0.0, 'pos': 0.85}  «Γ.Κ.  Σπυριδάκης, Ελληνικά  δημοτικά  τραγούδια (Εκλογή),  τ.  Α',  εν  Αθήναις »
- `B0048` gate=`kept-non-bib:unconfirmed` → Opus kind=**subdivided_sublist** style=chicago_turabian lang=greek feat={'yc': 28, 'yd': 1.75, 'latin': 0.0, 'dash': 0.0, 'pos': 0.85}  «Βερέμης  Θάνος  Μ.-  Κολιόπουλος  Ιωάννης  Σ., Νεότερη  Ελλάδα ,  Αθήνα,  Εκδόσε»
- `B0063` gate=`kept-non-bib:unconfirmed` → Opus kind=**subdivided_sublist** style=chicago_turabian lang=greek feat={'yc': 26, 'yd': 1.37, 'latin': 0.03, 'dash': 0.0, 'pos': 0.95}  «1. Ελληνική»

## Tuning signal — feature profile of the errors

- **FN (missed bib)** (n=4): kinds={'chapter_bibliography': 1, 'end_bibliography': 1, 'subdivided_sublist': 2}; styles={'chicago_turabian': 4}; detector saw year_count=0 in 0/4.
- **FP (false bib)** (n=7): kinds={'not_reference': 5, 'cv_publication_list': 2}; styles={'na': 3, 'none': 2, 'footnote_humanities': 1, 'vancouver_numbered': 1}; detector saw year_count=0 in 2/7.