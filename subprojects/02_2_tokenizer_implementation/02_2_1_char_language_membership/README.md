# 02.2.1 — Char-language membership masks

> **In one line:** a strict-rule, CLDR-derived answer to "which scripts, families and languages admit this codepoint?", built for *rejection* rather than classification; it went from 54 language bits to 88 in two days of audit-driven releases and became the reference layer every downstream attribution joined against.
> **Period:** 2026-05-14 → 2026-05-15 (commits `002bddc5`, `0b20b96d`, `719d3834`, `0bbd93de`). **Status:** completed and consumed; last released version v3.3.3, schema v5, 29 script / 47 family / 88 language bits.
> **Came from / led to:** this → [`02_2_4`](../02_2_4_language_category_promotion/) PMI promotion (via [`02_2_2`](../02_2_2_vocab_lang_attribution/)) and → [`02_1_4_cutoff_analysis`](../../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/)

## Why this existed

The Apertus-8B-2509 tokenizer was not trained on Apertus data — it is Mistral-Nemo's `tekken` tokenizer inherited wholesale, and Mistral published only 11 of its "over 100" training languages ([`PLAN.md`](PLAN.md) § "How we approach the missing-tokenizer-training-data problem"). To reason about what the 131,072 base-vocab tokens *are*, the project needed a source-authoritative, dataset-free answer to what characters each language can legitimately produce. The design choice that shaped everything: the artifact never assigns a token to a language, it only says which categories are **excluded with confidence**. Apertus's documented pretrain mix stands in as a proxy for Mistral's unpublished list; the proxy is falsifiable by an audit that checks whether any script with real presence in the vocab is unmodelled.

## History

### 2026-05-14 — v2.2: one layer, strict rules (`002bddc5`, `0b20b96d`)

Created as `char_language_membership/` with a single language layer of 54 (language, script, encoding) triples, four deterministic closures on top of raw CLDR exemplars — script-compatibility filter, case, NFD, script-range fallback, plus a post-fallback NFD pass — and substrate (`N*`/`P*`/`S*`/`Z*`/`Cc`/`Cf`) set to all bits. Cyrillic and Arabic were **deliberately excluded from the script-range fallback** because their blocks carry extensions for out-of-scope languages; those codepoints fall through to zero bits on purpose ([`PLAN.md`](PLAN.md) § "Strict rule").

The same day's hardening commit took the count to 55, added `scripts/_common.py` and `scripts/query_codepoint.py`, and fixed a real consumer trap: the char parquet is **sparse**, so a naive lookup returning 0 for a missing codepoint false-rejects. `query_codepoint.py` reproduces the build-time substrate fallback so consumers get identical semantics. Masks were stored as fixed-width `binary(16)` rather than `uint64` because the audit already foreshadowed growth past 64 bits (`0b20b96d`). `validate.py` gained a phase-2 token audit gate asserting fewer than 50 fall-through tokens per *out-of-scope* script; in-scope gaps are reported but do not fail. Recorded gaps at that point: out-of-scope `Other` = 1, in-scope Cyrl 132, Latn 42, Arab 38 ([`TODO.md`](TODO.md)).

### 2026-05-15 — v3.0/v3.1: three layers (`719d3834`)

Renamed to `02_2_1_char_language_membership/` and rebuilt as the hierarchical design in [`PLAN_v3_HIERARCHICAL.md`](PLAN_v3_HIERARCHICAL.md): script / family / language masks in parallel, shipped as schema v4 with 22 script, 31 family and 55 language bits. The motivating complaint was that a bare-ASCII token like ` the` reported "compatible with 28 specific languages" when the honest statement is "Latin-script: yes, narrower: no signal". Two decisions were resolved and held: single-locale scripts get family bits anyway, for layer symmetry; and the script layer stays a **projection of language evidence, not a Unicode-script detector** — `ō`, admissible in no in-scope locale, correctly gets zero bits at every level. Twelve per-script research notes landed with it ([`notes/`](notes/)), covering Latin, Cyrillic, Greek, Arabic, CJK, Korean, Hebrew, Indic and the smaller scripts.

### 2026-05-15 — v3.2 → v3.3.3: five releases in one session (`0bbd93de`)

The PMI promotion consumer filed [`FEEDBACK_FROM_PMI_PROMOTION_CONSUMER_20260515.md`](FEEDBACK_FROM_PMI_PROMOTION_CONSUMER_20260515.md): correctness was "excellent — zero cross-script leakage", but **34 of the 87 well-sampled corpus keys had no char-tool mapping**, and the ISO-639-3 ↔ BCP-47 map was unpublished, forcing a hand-curated 50-entry dict in the consumer that had caused **four silent bugs** (`srp_Cyrl`, `lvs_Latn`, `ekk_Latn`, `cmn_Hani` — macrolanguage-vs-individual code confusion, ~4 B tokens, undetected for two days).

The response, all on 2026-05-15:

- **v3.2** (schema v5) — 18 new locales, a published `canonical_key_to_char_tool_code` map, `iso_639_3` aliases, and a per-token `category_or` column. The consumer deleted its 50-entry dict.
- **v3.3** — 7 new-script locales (Amharic, Khmer, Sinhala, Lao, Tibetan, Odia, Dhivehi; [`notes/New-scripts-v3.3.md`](notes/New-scripts-v3.3.md)). **Zero coverage gain**: Apertus byte-fragments all seven scripts, so every token there decodes as `partial_utf8`. Recorded as an Apertus-vocab fact, not a v3.3 defect — and the consumer's own "~2 pp" estimate was retracted as wrong.
- **v3.3.1** hotfix — closed the two manifest bugs the consumer found (`ell_Grek` missing from the map, so *Greek*, the project's anchor language, silently produced an empty masked set; and `arb_Arab` missing despite `ara_Arab` being present), added Urdu `ں` (U+06BA) via a new `extra_codepoints` field, and added a **build-time self-test** requiring every language's primary `(iso_639_3, script)` pair to resolve — so the silent-bug class cannot recur.
- **v3.3.2** — "Albanians and Romans": bits 85–87 add `sq`, `gsw` and `la`, resolving `als_Latn` from both its readings and `lat_Latn`. Classical Latin has an empty CLDR exemplar, so it was seeded by hand with A–Z, Æ/æ and macron-bearing forms ([`languages.yaml`](languages.yaml) line ~640).
- **v3.3.3** — the self-test broadened beyond the primary pair ([`scripts/build_char_language_bitmask.py`](scripts/build_char_language_bitmask.py) ~line 773).

The full round-trip, including the post-hotfix re-verification, is in [`v3_2_INTEGRATION_REPORT_20260515.md`](v3_2_INTEGRATION_REPORT_20260515.md). Its closing "85 language / 45 family / 29 script" is the v3.3.1 state; the shipped YAMLs end at **88 / 47 / 29**.

## Outcome

- **Coverage of the Apertus vocab by at least one language's PMI-promoted set**, the metric the consumer used to grade each release: 81.18 % (v3.1, 106,404 tokens) → 85.54 % (v3.2, 112,117) → 85.55 % (v3.3.1, 112,131; the +14 is exactly the Urdu fix) → **86.35 % (113,184 of 131,072)** in the committed rebuild ([`v3_2_INTEGRATION_REPORT_20260515.md`](v3_2_INTEGRATION_REPORT_20260515.md); final figure recomputed from [`../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/uncovered_tokens.tsv`](../02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/uncovered_tokens.tsv), 17,888 rows).
- **Unmapped corpus keys**: 34 → 7 after v3.3.1 → **5** in the shipped manifest (`gmh_Latn` and the four `und_*` keys — genuinely out of scope, no CLDR data or no identified language).
- **The correctness claim held under an independent test.** Across 87 languages the masked sets show exactly one cross-script overlap, `cmn_Hani` ↔ `jpn_Jpan`, which is linguistically correct (shared Han characters). Every other cross-script pair is zero ([`../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md`](../CHECKPOINT_LANGUAGE_ATTRIBUTION_20260515.md) § 5).
- **Consumed downstream**, and this is the strongest evidence it worked: `02_1_4_cutoff_analysis/scripts/classify_added_tokens.py` reads `artifacts/char_language_bitmask.parquet` + `artifacts/manifest.json` to classify every C3 added token as GREEK / USEFUL_STRUCTURAL / NOISE, feeding [`../../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/REPORT.md`](../../02_1_tokenizer_experiments/02_1_4_cutoff_analysis/REPORT.md) § 4.
- **The artifacts are not in the repo.** `artifacts/` is gitignored repo-wide (root `.gitignore`: `subprojects/**/artifacts/`, `*.parquet`), so both parquets and both manifests must be rebuilt from the YAMLs before any consumer runs. Only the sources of truth are tracked.
- **Left open** ([`TODO.md`](TODO.md)): vocabulary version pinning (the apply step hardcodes the Apertus snapshot path); the in-scope Cyrillic/Latin/Arabic fall-through gaps, which are correct-by-design rejections but closeable by adding locales; and a possible Rust port for hot-path lookup, never needed.

## Where things are

| Artifact | Path | Role |
| --- | --- | --- |
| Language bit assignments (88) | [`languages.yaml`](languages.yaml) | source of truth; bits are stable wire identifiers, never reassigned |
| Family / script bit assignments (47 / 29) | [`families.yaml`](families.yaml), [`scripts.yaml`](scripts.yaml) | same |
| Codepoint build | [`scripts/build_char_language_bitmask.py`](scripts/build_char_language_bitmask.py) | CLDR pull + closures + self-test |
| Token apply | [`scripts/apply_to_apertus_vocab.py`](scripts/apply_to_apertus_vocab.py) | per-token AND / OR across the 131,072-entry vocab |
| **Consumer entrypoint** | [`scripts/query_codepoint.py`](scripts/query_codepoint.py) | applies the sparse-table substrate fallback — do not re-implement it |
| Validation | [`scripts/validate.py`](scripts/validate.py) | phase-1 char checks, phase-2 token wire-format + recompute + out-of-scope audit gate |
| Built tables (not tracked) | `artifacts/{char,token}_language_bitmask.parquet`, `artifacts/{manifest,token_manifest}.json` | gitignored; rebuild before use |

## Working documents

- **Designs (both historical, both still describe live behaviour):** [`PLAN.md`](PLAN.md) — the v2.2 language-only design; the closures and substrate rule it defines are unchanged. [`PLAN_v3_HIERARCHICAL.md`](PLAN_v3_HIERARCHICAL.md) — the three-layer design; its counts (22/31/55) are the v3.0 design moment, superseded by 29/47/88.
- **Reviews and integration records:** [`FEEDBACK_FROM_PMI_PROMOTION_CONSUMER_20260515.md`](FEEDBACK_FROM_PMI_PROMOTION_CONSUMER_20260515.md) — the consumer's coverage complaint that drove v3.2/v3.3; [`v3_2_INTEGRATION_REPORT_20260515.md`](v3_2_INTEGRATION_REPORT_20260515.md) — verdict on v3.2 plus the post-v3.3.1 verification matrix.
- **Per-script research notes** ([`notes/`](notes/), 12 files): [`Latin.md`](notes/Latin.md), [`Latin-residue.md`](notes/Latin-residue.md), [`Cyrillic.md`](notes/Cyrillic.md), [`Greek.md`](notes/Greek.md), [`Arabic.md`](notes/Arabic.md), [`Chinese.md`](notes/Chinese.md), [`Japanese.md`](notes/Japanese.md), [`Korean.md`](notes/Korean.md), [`Hebrew.md`](notes/Hebrew.md), [`Indic.md`](notes/Indic.md), [`Smaller-scripts.md`](notes/Smaller-scripts.md), [`New-scripts-v3.3.md`](notes/New-scripts-v3.3.md). Each records sources consulted and whether the locale needed a `languages.yaml` change.
- **Status snapshot:** [`TODO.md`](TODO.md) — the v2-era done/open list; its "55 triples" reflects the state before the v3 releases.
