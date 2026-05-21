# Review integration — 2026-05-18 round 3

Third review pass, all four corrections integrated plus user's
explicit `greek_diacritic_policy=preserve` decision recorded.

## Verifications performed before editing the plan

| Reviewer claim | Verified | Source of truth |
|---|---|---|
| Current repo HEAD is `9a6b039` "Add firing count attribution workflow" | ✓ | `git log --oneline -1` |
| `git hash-object` of text_dedup.py is `6b9bfdb0bd9923349c348f80866c472101ab8fcf` | ✓ | `git hash-object glossapi_corpus_cli/text_dedup.py` (note: my earlier `sha1sum` differs because git prepends a `blob <size>\0` header) |
| Live `latest.json` points to `full_publish_aws_strip_20260328T101312Z`, not `exact_stage_20260413T025237Z` | ✓ | `/home/foivos/data/glossapi_work/hf_release_publish/dedup_metadata/latest.json` |
| `decision_rows = 717,265` (not 49,643,978) | ✓ | `final/run_summary.json` of the live publish bundle |
| Published bundle includes `near_candidate_pairs.parquet` | ✓ | `builder_metadata/near_candidate_pairs.parquet` exists |
| Manifest says `greek_diacritic_policy = strip` | ✓ | `builder_metadata/manifest.json` shows `"greek_diacritic_policy": "strip"`, with `exact_relaxed_version: "exact_relaxed_norm_v5_strip"` and `near_norm_version: "near_norm_v5_strip"` |
| Manifest has `candidate_score_floor = 0.85`, `num_perm = 128`, `shingle_token_5_v1` | ✓ | Same manifest |

My earlier characterization (49.6 M rows, "exact-stage only") was
from the **HF cache snapshot** stored locally, not from the live
filesystem state. That cache reflects an older publish. Going
forward, trust the live `dedup_metadata/latest.json` overlay, not
the HF cache snapshots.

## Apertus Greek encoding context (for the diacritic-policy decision)

The user asked me to check what encoding the Apertus tokenizer
uses for Greek. From `CLAUDE.md`'s empirical Apertus tokenizer
facts (verified at `~/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/`):

- **`normalizer: null`** — no Unicode normalization is applied at
  tokenization time.
- Single-token vocab hits: bare α (U+03B1), monotonic ά (U+03AC),
  final sigma ς, en-dash, em-dash, ellipsis …, smart quotes, NBSP.
- **No single-token merges for**: oxia ά (U+1F71), psili+oxia ἄ
  (U+1F04), tonos+ypogegrammeni ᾴ (U+1FB4), combining acute U+0301,
  NFD-decomposed α+◌́.

So Apertus's tokenizer treats monotonic ά (U+03AC) as a distinct
token from bare α (U+03B1). **Stripping diacritics in our dedup
would conflate texts that Apertus encodes differently** — exactly
the wrong behaviour for a "what did the model see" audit. User's
`preserve` choice is therefore correct on first-principles
grounds, not just preference.

## Findings and resolutions

### Finding 1 — Wrong commit pin

**Reviewer**: I claimed `edb98d6`; actual HEAD is `9a6b039`.

**Status**: **CORRECTED.** PLAN.md §3.1 now records:
- `text_dedup_commit = 9a6b039`
- `text_dedup_file_hash = 6b9bfdb0bd9923349c348f80866c472101ab8fcf`
  (git hash-object form, the canonical content hash)

Added a run-start verification step: halt with "policy drift
detected" if either pin differs at start of run, rather than
silently producing inconsistent hashes.

### Finding 2 — Cached exact-stage claim not confirmed on disk

**Reviewer**: I claimed `exact_stage_20260413T025237Z` was the
live published metadata. Live disk has it as an obsolete cache;
current `latest.json` points to `full_publish_aws_strip_20260328T101312Z`.

**Status**: **CORRECTED.** PLAN.md §6.9 (new section) now
references the live publish bundle by name with the correct
row counts: `717,265 decision / 674,411 kept / 42,854 dropped`.

### Finding 3 — Published bundle is NOT exact-stage only

**Reviewer**: My characterisation "exact-stage only, near-dup
deferred" was wrong. The current local publish bundle has
`near_candidate_pairs.parquet` with `candidate_score_floor = 0.85`,
`num_perm = 128`, `shingle_token_5_v1`.

**Status**: **CORRECTED.** PLAN.md §6.9 now states that
near-dup metadata IS in the bundle, with the exact
`candidate_score_floor` / `num_perm` / `shingle_version`
values listed. Builder *policy* is what's deferred (which docs to
drop), not the raw near-dup *candidates*.

### Finding 4 — Diacritic-policy mismatch risk

**Reviewer**: live `text_dedup.py` defaults to `preserve`, but the
published bundle was built with `strip`. Reusing the bundle's
relaxed/near metadata under a `preserve` audit policy would silently
produce inconsistent hashes.

**Status**: **CORRECTED + USER DECISION CAPTURED.**

**User decision (2026-05-18)**: `greek_diacritic_policy = preserve`.

PLAN.md §3.1 now:
- Records `preserve` as the audit's deliberate policy override
  (with rationale tied to Apertus's tokenizer behaviour — see
  "Apertus Greek encoding context" above).
- Manifest contents include `exact_relaxed_version =
  exact_relaxed_norm_v5_preserve` and `near_norm_version =
  near_norm_v5_preserve` — explicitly *not* matching the published
  bundle's `_strip` versions, so this is auditable from the
  manifest.

PLAN.md §6.9 (new) provides the **reuse decision matrix** the
reviewer asked for:

| C3-side stage | Published bundle version | Audit version | Reusable? |
|---|---|---|---|
| `strict_exact` | `exact_strict_norm_v1` | `exact_strict_norm_v1` | **YES** — policy-independent |
| `relaxed_exact` | `exact_relaxed_norm_v5_strip` | `exact_relaxed_norm_v5_preserve` | **NO** — different normalisation |
| `near` (sigs + bands) | `near_norm_v5_strip` | `near_norm_v5_preserve` | **NO** — same reason |

Plus a **coverage check** before reuse (verify C3-train GlossAPI
doc_ids are subset of published-bundle doc_ids), and a
**run-start checklist** matching the reviewer's exact guardrails
(verify commit + file-hash + bundle pointer + manifest fields;
record reuse decisions in `manifests/run_<ts>/reuse_log.json`).

### Reviewer's recommended addition — HF-release × Apertus secondary

**Reviewer**: "yes, add the optional HF-release × Apertus
measurement to §10, but keep it explicitly secondary to the
C3-train × Apertus audit."

**Status**: **INTEGRATED.** PLAN.md §10 now has the
HF-release × Apertus measurement as an explicitly-secondary
option, with separate `artifacts/secondary_hf_release/` output
directory so it doesn't get conflated with the load-bearing
CPT inputs. Triggered only if there's spare worker time after
the primary measurement completes.

## Net effect on the plan

After three review rounds, the audit's measurement contract is
tight enough to script:

- **Versioning**: commit hash + file hash pinned, verified at
  run-start, halt on drift.
- **Policy**: `greek_diacritic_policy = preserve`, deliberate
  override of the (mismatched) published-bundle policy of `strip`,
  rationale tied to Apertus's tokenizer behaviour.
- **Reuse boundary**: strict_exact ✓ (policy-independent);
  relaxed_exact + near ✗ (different policy → different hashes);
  HPLT-side and Apertus-side ✗ (no published bundle).
- **Secondary measurement**: HF-release × Apertus available as
  optional add-on, separate output namespace.

No further conceptual ambiguity. Scripting can start.

## Open items NOT addressed in this round

These are intentionally deferred until they become decision-relevant:

- **Whether to record `apertus_consumed_probability` per Apertus
  doc** as a literal column or derive it from a side table at join
  time. PLAN §2.1 currently says "per-doc rows carry a
  `consumed_probability` column"; impl detail to decide at
  scripting time.
- **HPLT-side run-start coverage check**: HPLT clean60 is NOT in
  the published bundle, so the published bundle's strict_exact
  memberships don't cover HPLT. The audit recomputes HPLT
  strict_exact fresh. Plan reflects this in §6.9 but doesn't
  break it out as a separate concern.
- **What `text_dedup.py` does to `unicodedata` normalisation at
  the strict_exact stage** when `greek_diacritic_policy =
  preserve`. The code uses NFC + whitespace-collapse before
  blake3 hashing; verified by reading
  `glossapi_corpus_cli/text_dedup.py` constants block but not by
  step-through. Worth re-verifying before scripting if it
  matters for our predicted strict_exact match rate against the
  published bundle.
