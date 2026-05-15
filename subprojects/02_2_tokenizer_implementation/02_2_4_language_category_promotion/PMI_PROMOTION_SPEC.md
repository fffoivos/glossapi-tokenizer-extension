# PMI promotion — implementation specification

> Companion to `METHODOLOGY.md` and `PLAN.md`. This is the
> build-spec for the **first concrete promotion pass**: multi-language
> Pointwise Mutual Information across the well-sampled subset of our
> tokenisation run, with **char-mask filtering toggled on and off** so
> the two outputs can be compared directly.

## Scope of this pass

- **Marginal scope** = canonical keys with `Σ_t H[L, t] > 1 × 10⁹` (the
  "≥ 1 B cap-hit" set). After adding `eng_Latn_fineweb_hq`, this is
  **87 keys** in the current run. The exact list is enumerated by the
  script at build time and pinned in the output manifest.
- **Targets to promote** = the same 87 keys (one promoted set per
  cap-hit key). Low-resource keys are deferred to a follow-up pass.
- **Vocabulary** = 131,072 tokens, the full Apertus base vocab.
- **Char masks** = char-tool schema v5 (current) from
  `02_2_1_char_language_membership/artifacts/`. Bit counts
  (currently 29 script / 45 family / 85 language) are read from
  the manifest, not hardcoded.
- **PMI base** = log₁₀ throughout for readability.

Two English keys (`eng_Latn` from clean-wikipedia and
`eng_Latn_fineweb_hq` from FineWeb-HQ) both clear the threshold. The
marginal counts en firings twice as a result, depressing PMI for both
en samples by `≈ log10((K-1)/K)` where K = 87 — a ~0.005 dex shift,
negligible at our δ ≥ 1.0 threshold. The double-count is recorded in
the manifest but not corrected (corpora are distinct, both legitimately
in scope).

## Inputs

| input | path | role |
| --- | --- | --- |
| `histogram_matrix.npz` | `02_2_2_vocab_lang_attribution/outputs/` | per (key × token) firing counts, shape (1934, 131072) |
| `token_language_bitmask.parquet` | `02_2_1_char_language_membership/artifacts/` | char masks per token (v5 schema or later, read from char-tool manifest) |
| `manifest.json` (char tool) | `02_2_1_char_language_membership/artifacts/` | language-bit ↔ ISO code map; `N_LANG_BITS` |

No other inputs. Deterministic transform.

## Algorithm

Refers back to `METHODOLOGY.md § Method W — W4 (PMI)` for derivation.
This section is purely operational.

```python
# Constants
ALPHA           = 0.5
DELTA           = 1.0
MIN_COUNT       = 100
MARGINAL_FLOOR  = 1_000_000_000      # keys with total >= this contribute to marginal
N_LANG_BITS     = manifest["levels"]["language"]["bits_used"]   # 55 in v4
V               = 131_072
UNKNOWN_STATUSES = {"partial_utf8", "byte_unmapped", "special"}
```

### Step A — choose marginal-contributing keys

```
marg_keys = [L  for L in canonical_keys
                if  sum(H[L]) >= MARGINAL_FLOOR]
```

Pinned in manifest as `marginal_keys: [...]`.

### Step B — compute per-language and marginal rates

For each marginal key L compute once:
```
total_L = sum(H[L])                                          # scalar
p_L[t]  = (H[L][t] + ALPHA) / (total_L + ALPHA * V)          # vector, length V
```

Marginal across the 87 keys:
```
count_marg[t]     = Σ_{L in marg_keys} H[L][t]               # vector, length V
total_marg        = Σ_{L in marg_keys} total_L
p_marg[t]         = (count_marg[t] + ALPHA) / (total_marg + ALPHA * V)
```

### Step C — compute PMI per (target_key × token)

```
pmi[L, t]  =  log10( p_L[t] / p_marg[t] )                    # vector per L
```

Bounded above by `log10(total_marg / total_L)` (token fires only in L).
For our scope: ceiling ≈ +2.0.

### Step D — char-admissibility mask (Variant A only)

Per target language code `c_L` (the ISO bit code for the target key's
language; e.g. `eng_Latn_fineweb_hq → en`, `deu_Latn → de`):

```
admissible_for_L[t]  =     ((bitmask_and[t] >> bit_of[c_L]) & 1) == 1
                       AND  popcount(bitmask_and[t])  <  N_LANG_BITS
                       AND  status[t]  not in  UNKNOWN_STATUSES
```

The language bit for each canonical key is derived from
`key.split("_")[0]` (ISO 639-3 code) cross-referenced against
the char-tool manifest's `languages` list. For sample keys like
`eng_Latn_fineweb_hq`, strip the suffix back to the canonical
language code (`en`).

Hard rule: keys whose language code is not in the char-tool's in-scope
locale list cannot be promoted under Variant A (no admissibility filter
defined). They get **only Variant B output** and a warning in the
manifest.

### Step E — apply thresholds and emit

Two outputs per target language L:

**Variant A — masked (char filter ON):**
```
promoted_A[L]  =  { t  :  admissible_for_L[t]
                         AND  H[L][t]  >=  MIN_COUNT
                         AND  pmi[L, t]  >=  DELTA }
```

**Variant B — unmasked (char filter OFF):**
```
promoted_B[L]  =  { t  :  H[L][t]  >=  MIN_COUNT
                         AND  pmi[L, t]  >=  DELTA }
```

Sort each by `H[L][t]` descending (most-firing first).

## Output layout

```
02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/
├── tables/
│   ├── <canonical_key>__masked.txt          # Variant A
│   ├── <canonical_key>__unmasked.txt        # Variant B
│   └── <canonical_key>__delta.txt           # B \ A — tokens in B but not A
├── per_token_pmi.parquet                    # full pmi[L, t] table for inspection
├── overlap_matrix.tsv                       # |A_i ∩ A_j| for every (i, j) of the 87 keys
├── summary.tsv                              # one row per target language
└── manifest.json                            # provenance, knobs, marginal-keys list
```

### txt-file format (per the existing main_token_sets convention)

```
# <key>__masked  — masked: bitmask_and has <lang_code>-bit set,
#                  popcount < N_LANG_BITS, status evaluable
#                  PMI >= 1.0  in marginal of 87 keys >= 1B firings
#                  count >= 100 in <key>
# N tokens, sorted by count desc
{ <id>: '<decoded>' }
{ <id>: '<decoded>' }
...
```

Decoded string is `repr()` quoted (Python-style) so unprintables /
internal-byte tokens render unambiguously.

### `summary.tsv` schema

| column | meaning |
| --- | --- |
| `target_key` | canonical key name |
| `lang_code` | ISO bit code used for masking (or `unmapped` if not in scope) |
| `total_L` | sum of firing counts for this key |
| `masked_count` | # tokens in Variant A |
| `unmasked_count` | # tokens in Variant B |
| `delta_count` | # tokens in B but not A (rejected by char mask) |
| `masked_mass_pct` | Σ counts of A / total_L |
| `unmasked_mass_pct` | Σ counts of B / total_L |
| `max_pmi` | highest PMI for any promoted token |
| `min_pmi` | lowest PMI for any promoted token (≥ δ) |
| `pmi_at_median` | median PMI among promoted tokens |

### `overlap_matrix.tsv` schema

Square table over the 87 target keys. Cell `(i, j)` is
`|promoted_A[L_i] ∩ promoted_A[L_j]|`. Diagonal is `|promoted_A[L_i]|`.
Useful for spotting closely-related-language overlap (e.g.
sister-Latin pairs should overlap appreciably; cross-script pairs
should overlap near 0).

### `manifest.json` schema

```json
{
  "schema_version": 1,
  "built_at": "2026-05-15T...",
  "inputs": {
    "histogram_matrix_md5": "...",
    "token_bitmask_md5":    "...",
    "char_tool_schema_version": 5
  },
  "parameters": {
    "alpha":           0.5,
    "delta":           1.0,
    "min_count":       100,
    "marginal_floor":  1_000_000_000,
    "pmi_base":        10
  },
  "marginal_keys":   ["aar_Latn?", "...", "deu_Latn", "eng_Latn", "eng_Latn_fineweb_hq", "..."],
  "marginal_total":  113_000_000_000,
  "double_counted_languages": ["en"],
  "target_keys":     [...],                 // same as marginal_keys for this pass
  "keys_without_lang_code_mapping": [],     // any target key whose ISO code isn't in char tool
  "outputs": {
    "tables_dir":   "tables/",
    "per_token_pmi_parquet": "per_token_pmi.parquet",
    "overlap_matrix_tsv":    "overlap_matrix.tsv",
    "summary_tsv":           "summary.tsv"
  }
}
```

## Build script

`02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/build.py`

CLI:

```
python3 build.py
   [--alpha 0.5]
   [--delta 1.0]
   [--min-count 100]
   [--marginal-floor 1_000_000_000]
   [--training-weights weights.json]
```

Algorithm structure:

```
1. Load histogram_matrix.npz, char-tool manifest, token_bitmask.parquet.
2. Compute totals per key; select keys >= marginal-floor → marg_keys.
3. Compute count_marg[t] = H[marg_keys].sum(axis=0)        # 131072 vector
   Compute total_marg = sum(total_L for L in marg_keys).
4. Compute p_marg[t] = (count_marg + α) / (total_marg + α·V).
5. For each L in marg_keys (target):
   5.1. Read H[L], total_L, derive lang_code from L's name.
   5.2. p_L[t] = (H[L] + α) / (total_L + α·V).
   5.3. pmi[t] = log10(p_L / p_marg).
   5.4. Masked promotion (Variant A):
        admissible = ((bm_and >> bit_L) & 1) == 1
                     AND popcount < N_LANG_BITS
                     AND status not in UNKNOWN
        promoted_A = where(admissible AND H[L] >= min_count AND pmi >= δ)
   5.5. Unmasked promotion (Variant B):
        promoted_B = where(H[L] >= min_count AND pmi >= δ)
   5.6. Write tables/<L>__masked.txt, tables/<L>__unmasked.txt,
        tables/<L>__delta.txt.
6. Build summary.tsv, overlap_matrix.tsv, per_token_pmi.parquet,
   manifest.json.
```

`per_token_pmi.parquet` columns (one row per (target_key, token)
where pmi ≥ 0 and H[L][t] ≥ 1; rows with pmi < 0 dropped to keep
file size sane):

| column | type |
| --- | --- |
| `target_key` | str |
| `token_id` | uint32 |
| `decoded` | str |
| `count_L` | uint64 |
| `pmi` | float64 |
| `admissible_for_L` | bool |
| `popcount_lang` | uint8 |

## Expected sanity points (validated by the script)

1. Substrate (`popcount == N_LANG_BITS`) tokens have `|pmi|` near zero for every
   key — within `±log10(2)` ≈ ±0.30. They do NOT appear in masked
   promotions; they MAY appear in unmasked promotions only if a single
   substrate token is dramatically over-represented in one key (e.g.
   if Wikipedia infobox-laden English has 2× the digit density). Flag
   any substrate token that gets promoted unmasked as "suspicious
   corpus-mix artifact".
2. `promoted_A[L] ⊆ promoted_B[L]` for every L (mask only removes
   tokens, never adds).
3. T0 tokens for L (where `bitmask_and == {L} only`) — if they fire
   ≥ min_count, they should land at very high PMI (chars rule out
   every other language → marginal share for that token is dominated
   by L). Sanity: all T0 tokens that fire ≥ min_count should be in
   `promoted_A[L]`.
4. English structural: T0 is empty for en. The en main set is built
   entirely from T2.

## Open knobs and design intent

This pass picks defaults; the **comparison harness** in `METHODOLOGY.md
§ Comparison plan` is what eventually decides the right values per
phase 3 diagnostic. The defaults here are designed to be a defensible
first cut, not a settled answer.

| knob | this pass | rationale |
| --- | --- | --- |
| `α` | 0.5 | mild Laplace; protects against zero-counts without distorting count ≥ 100 |
| `δ` | 1.0 | ≥ 10× over-represented vs marginal; round-number visible-in-data |
| `min_count` | 100 | evidence floor; orphans most of the long Zipfian tail with <1% mass loss |
| marginal floor | 1 × 10⁹ | "cap-hit" subset; 87 keys; stable marginal |
| variant | A + B both shipped | comparison is the deliverable |

A follow-up pass should sweep `δ ∈ {0.5, 1.0, 1.5, 2.0}` and
`min_count ∈ {10, 100, 1000}` against representative target keys
(say de, en, el, ja, ru) and report set-size + mass-coverage curves
for the comparison harness.

## Training-weighted PMI as a diagnostic column (not promotion)

In addition to the canonical count-pooled PMI used for promotion, the
per-token table emits a second PMI column:

```
p_training(t)        =  Σ_L  w_L · p_L(t)            for w_L  =  approximate Apertus
                                                          training share of L
pmi_training(t, L)   =  log10( p_L(t) / p_training(t) )
```

**Promotion is NOT done on this column.** It exists only to surface
tokens whose canonical and training-weighted PMIs disagree
substantially — those are interesting cases (typically: high-resource
languages where the training-weighted marginal absorbs more of their
mass, so PMI shrinks).

Important: the **canonical PMI uses a count-pooled marginal**
(`p_marg = Σ count_L / Σ total_L`), which is equivalent to a
language-mass-weighted average of per-language rates. The diagnostic
column with **equal weights** (`p_marg_training = (1/K) Σ p_L`) is a
**different formula** that gives uniform weight to each language
regardless of sample size. At our cap-hit scale (totals within ~4 %
of each other) the two are numerically close but not algebraically
identical.

**Weights are an open input** until verified:

- For this pass, the script defaults to *equal weights* — the
  diagnostic column then numerically approximates the canonical
  count-pooled PMI without being identical to it. It accepts an
  optional `--training-weights weights.json` whose contents are a
  `{canonical_key: float_share}` mapping. The float shares **must
  sum to approximately 1.0** (tolerance ±1e-6); if the supplied
  weights sum to less than 1.0, the script fails with a SystemExit
  rather than auto-normalising. Missing canonical keys default to
  weight 0. Once the sum check passes, the script divides by the
  sum so the final weights are exactly a probability distribution.
- TODO before relying on this column: source verified per-locale
  Apertus training shares (paper, model card, or FineWeb-2 /
  FineWeb-HQ / DCLM-Edu reported per-locale token counts as proxies).
- The script writes `weights_used.json` next to the output so future
  re-runs are reproducible against the same approximate shares.

## What this pass does *not* produce

- Per-language Beta-Binomial credibility intervals (the F3-with-CI
  variant from the methodology doc). Computable from the same data but
  not in this pass.
- The W1/W2/W3 weight variants. PMI (W4) only.
- Cross-source consistency check (e.g. `eng_Latn` vs
  `eng_Latn_fineweb_hq`) — possible from the per_token_pmi table by
  pairwise diff, but emitted as a follow-up.
- Promotion for the 1,842 non-cap-hit keys. Deferred.

## Estimated work

| item | time |
| --- | ---: |
| `build.py` core (PMI + thresholds + I/O) | 2 h |
| Variant A + B + delta files | 0.5 h |
| `overlap_matrix.tsv` + `summary.tsv` | 0.5 h |
| `per_token_pmi.parquet` + manifest | 0.5 h |
| Sanity-check assertions | 0.5 h |
| First-run audit (eyeball 5-10 target keys, validate against pairwise en/de) | 1 h |
| **total** | **~5 h** |

Storage estimate: `per_token_pmi.parquet` keeps only `pmi ≥ 0` rows.
At 87 keys × ~50 K positive-PMI tokens each × ~30 bytes/row ≈ 130 MB.
Sized to fit in the repo.

## Hand-off after this pass

The 87 `<key>__masked.txt` files become the candidate input for the
phase 3 embedding diagnostic (`03_1_greek_embedding_diagnostic` and
its non-Greek successors). The 87 `<key>__unmasked.txt` files plus
the `__delta.txt` files are the audit trail showing what the char
mask did and didn't catch — used to decide whether the mask is doing
useful work or whether PMI alone suffices for the consumer's purpose.

Decision point after this pass: for each phase 3 diagnostic, choose
masked or unmasked input — guided by the `__delta.txt` inspections and
the comparison harness from `METHODOLOGY.md`.
