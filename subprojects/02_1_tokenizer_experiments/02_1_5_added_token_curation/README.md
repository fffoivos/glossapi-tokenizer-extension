# 02.1.5 — Added-Token Curation

> **In one line:** a per-token keep/remove policy for extraction and encoding artefacts in the added vocab — policy only, no tokenizer edits — whose manifest was later consumed at *build* time so the 69 in-cutoff noise tokens never entered the shipped vocab at all.
> **Period:** policy dated 2026-05-17; committed 2026-05-18 (`7deea009`). **Status:** completed; the manifest is live input to the production build path.
> **Came from / led to:** [`../02_1_4_cutoff_analysis/`](../02_1_4_cutoff_analysis/README.md) → this → [`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/README.md) (consumes the manifest) → [`../../02_2_tokenizer_implementation/`](../../02_2_tokenizer_implementation/README.md)

## Why this existed

The cutoff decides *how many* added units to keep. It does not decide whether some of them are garbage. C3's cleaner let a small residue of mojibake, PDF font-glyph names, homoglyph substitutions and its own newline placeholders through into the BPE candidate pool. Those tokens are not content the model should learn or be able to emit — a dedicated token for `Tο` (Latin T, Greek ο) lets the model generate text that breaks every downstream Greek-aware system.

## History

**2026-05-17 — [`CURATION_REPORT.md`](CURATION_REPORT.md), revised the same day to widen the list after an audit discussion.** Six removal classes, defined as predicates over the glossary category so the rule set is reproducible:

| class | rule | count at 25,600 |
|---|---|---:|
| `latin1_utf8_mojibake` | whole `category = mojibake` (`ÉÉ`, `Ø`, `Ô`…) | 6 |
| `mixed_script_artifact` | whole `category = mixed_script_token` | 77 |
| `pdf_postscript_glyph` | whole `category = postscript_glyph` (`/Α`, `/η`, `/pi`…) | 14 |
| `cleaner_linenewline_placeholder` | whole `category = code_identifier` (`LINENEWLINE`, `NEWLINENEWLINE`) | 2 |
| `cleaner_linenewline_bpe_fragment` | `latin_acronym` ∈ {`LIN`, `ENEW`, `LINENEW`} | 3 |
| `cleaner_extraction_tag` | `latin_fragment` ∈ {`-missing`, `-decoded`} | 2 |
| **total** | | **104** (0.41 %) |

The report flags its own broadest call: class B mixes ~5 true homoglyph artefacts (where the leading `Ω` is U+2126 OHM SIGN, not the Greek letter) with ~72 punctuation+Greek BPE-boundary fragments (`,τι`, `«Η`, `/και`) that *do* occur in real text. The agreed policy removes both on the grounds that each surface form is too infrequent to earn a vocab slot and byte-fallback composition covers it — and the report spells out exactly how to implement the narrower "homoglyphs only" rule instead, using the `lang_bucket` field that already separates them.

**Scope shift.** The rules were authored against `02_1_4`'s 11,264 anchor, where 39 removals fall in cutoff. The ship cutoff turned out to be 17,408, where **69** fall in cutoff.

**2026-05-18 — the manifest became structural.** `02_1_7`'s builder walks C3's merge sequence, **skips** the 69 in-cutoff ids and **backfills** with the next valid merges. So the removal is not a runtime mask: the noise tokens are absent from the vocab entirely, vocab size and alignment are preserved, and `03_apertus_extension_and_embedding_adaptation` needs no "skip these 69" branch when initializing embeddings.

## Outcome

- **Shipped**: `manifests/removal_list.jsonl` (one row per removed token with id, decoded string, category, `lang_bucket`, `removal_class`, meaning snippet) and `manifests/decision_summary.json` (class counts, per-cutoff impact, rule predicates). Both git-tracked because a downstream build reads them.
- **Measured cost of curation: none.** In `02_1_7`'s curated-vs-raw comparison every metric is flat or marginally better after removal ([`../02_1_7_intrinsic_eval_sweep/REPORT.md`](../02_1_7_intrinsic_eval_sweep/REPORT.md) § Curated-arm delta; [`../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md) § Verification).
- **Explicitly not removable**: ~17 structural byte-fallback / NFD / URL-encoded artefacts inside the cutoff. The report also documents an inconsistency it chose not to paper over — `%CE` is tagged `encoding_artifact` (NOISE) while `%CF` is tagged `url_or_path` (USEFUL) despite being the same UTF-8-prefix family; neither is safely removable.
- **Forward path recorded, not taken**: every removal entry points at a cleaner pattern that should be fixed upstream so these never become BPE candidates again.

## Where things are

| What | Where |
|---|---|
| Reasoning, per-class keep/remove justifications | [`CURATION_REPORT.md`](CURATION_REPORT.md) |
| Machine-readable removal manifest (consumed at build time) | [`manifests/removal_list.jsonl`](manifests/removal_list.jsonl) |
| Class counts + rule predicates | [`manifests/decision_summary.json`](manifests/decision_summary.json) |
| Rule engine | `scripts/emit_removal_list.py` (idempotent, deterministic) |
| The 69 ids that were actually filtered | [`../02_1_7_intrinsic_eval_sweep/manifests/removal_mask_at_17408.jsonl`](../02_1_7_intrinsic_eval_sweep/manifests/removal_mask_at_17408.jsonl) |

`artifacts/keep_list.jsonl` (~25k rows) is gitignored and regenerable from the glossary.

## Working documents

- [`CURATION_REPORT.md`](CURATION_REPORT.md) — the policy record. Its "Implementation handoff" section offers two options (embedding-init masking vs. a pruned variant); neither was what shipped — `02_1_7` invented a third, skip-and-backfill, after two reviewer rounds rejected the pruned variant for breaking alignment and append-only.
