# Ship Tokenizer Reconstruction — Two Apertus-Compatible Bundles

*2026-05-20 (updated post-review). **Both** the modern-only 148,480
variant and the composite 153,600 variant on disk had a HF wrapper
defect. Both have been rebuilt as Apertus-compatible bundles under
[`ship/`](ship/).*

## The two bundles

| Bundle | Vocab | Use for | Path |
|---|---:|---|---|
| **Modern-only** | **148,480 = 256 × 580** | The three-arm Vanilla / ReTok / Distillation init comparison | [`ship/apertus_greek_modern_only_148480/`](ship/apertus_greek_modern_only_148480/) |
| **Composite** | **153,600 = 256 × 600** | Downstream polytonic specialization arm (stacked on top of the modern winner) | [`ship/apertus_greek_extended_153600/`](ship/apertus_greek_extended_153600/) |

> **Both source variants on disk** (`c3_added_17408_curated_padded/`
> and `c3p_poly_added_5120/`) **emit a `tokenizer_config.json` with
> `tokenizer_class: TokenizersBackend`, which `AutoTokenizer.from_pretrained()`
> cannot load.** The underlying `tokenizer.json` (BPE / vocab /
> merges / pre-tokenizer / ByteLevel decoder) is structurally correct
> in both; only the lightweight HF wrapper config is wrong. The
> rebuild is a 3-file copy that swaps the wrapper for Apertus's
> canonical `PreTrainedTokenizerFast` config. **Build/verify with
> [`scripts/build_and_verify_ship_tokenizer.py`](scripts/build_and_verify_ship_tokenizer.py).**

## TL;DR (both bundles)

| | Modern-only 148,480 | Composite 153,600 |
|---|---|---|
| Total vocab | **148,480 = 256 × 580** ✓ | **153,600 = 256 × 600** ✓ |
| Base Apertus | 131,072 (ids 0..131,071, byte-identical) | 131,072 (ids 0..131,071, byte-identical) |
| Modern Greek C3 | +17,408 (ids 131,072..148,479, curated+backfilled, byte-identical to `c3_added_17408_curated_padded`) | +17,408 (same as left) |
| Ancient/polytonic | — | +5,120 (ids 148,480..153,599, byte-identical to `c3p_poly_added_5120`) |
| First-1000 ids preserved | ✓ | ✓ |
| Special tokens (unk=0, bos=1, eos=2, pad=3) | identical to Apertus ✓ | identical to Apertus ✓ |
| Front-end contract | `normalizer:null` / pre_tokenizer ByteLevel / decoder ByteLevel / model BPE — identical to Apertus ✓ | same ✓ |
| Loadable | `AutoTokenizer.from_pretrained()` → `PreTrainedTokenizerFast` ✓ | same ✓ |
| Polytonic-NT fertility win (60 → ? tokens) | 60 → 28 (-53.3 %) | 60 → 20 (-66.7 %) |
| `tokenizer.json` sha256 | (same as C3 ship source) | `b1eeb739a564b3abd33c1b85a16162b8284d98f9ab5d67528d3cbe8a82e9cbad` |

## 1. Why a reconstruction step was necessary

The polytonic extension's training pipeline emitted a
`tokenizer_config.json` with `tokenizer_class: TokenizersBackend` (its
internal naming for the custom builder). HuggingFace `AutoTokenizer`
doesn't know that class:

```
$ python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('.../c3p_poly_added_5120')"
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

The `tokenizer.json` inside the directory is **structurally fine** —
the BPE model, vocab, merges, pre-tokenizer regex, ByteLevel decoder,
and 1000 special tokens are all correctly set up and match Apertus's
front-end contract byte-for-byte. The bug is only in the lightweight
wrapper that HuggingFace reads to decide which Python class to wrap
the underlying tokenizer in.

The fix is to replace `tokenizer_config.json` with Apertus's
canonical one (which uses `tokenizer_class: PreTrainedTokenizerFast`,
the right class for byte-level BPE with a JSON model) and keep
everything else.

## 2. What was assembled

```
ship/apertus_greek_extended_153600/
├── tokenizer.json              # 19.4 MB; copied verbatim from c3p_poly_added_5120
├── tokenizer_config.json       # 173 KB; copied verbatim from swiss-ai/Apertus-8B-2509@3162c99
├── special_tokens_map.json     # 551 B; already identical between Apertus and polytonic
└── manifest.json               # provenance + SHA-256 of each file
```

Same idea, three lines:

```bash
cp <polytonic-run>/c3p_poly_added_5120/tokenizer.json            ship/apertus_greek_extended_153600/
cp <apertus-base>/tokenizer_config.json                          ship/apertus_greek_extended_153600/
cp <apertus-base>/special_tokens_map.json                        ship/apertus_greek_extended_153600/
```

## 3. End-to-end verification

The reconstruction script was followed by an end-to-end load + behavior
test. Output:

```
=== Loading rebuilt ship bundle via AutoTokenizer ===
  ✓ Loaded: class=PreTrainedTokenizerFast, vocab_size=153600
  bos=<s>(1) eos=</s>(2) pad=<pad>(3) unk=<unk>(0)
  ✓ Special-token ids match Apertus exactly (unk=0, bos=1, eos=2, pad=3)

=== Behavior on real text (fewer tokens = better compression) ===
  register                apertus  ext153600   Δ tokens  % saved
  modern Greek (web)           22         15         -7   +31.8%
  polytonic (NT)               60         20        -40   +66.7%
  Katharevousa                 44         15        -29   +65.9%
  academic                     33         15        -18   +54.5%
  legal                        44         28        -16   +36.4%
  English (control)            13         13         +0    +0.0%
  Russian (control)            16         16         +0    +0.0%

=== Special-token round-trip ===
  First 1000 ids: 0 mismatches  ✓

=== Sample new-id decode (the 5120 polytonic block) ===
  id=148480  decoded='καὶ'
  id=148481  decoded=' οὐδὲ'
  id=148482  decoded=' φη'
  id=149000  decoded='θις'
  id=150000  decoded='δέναι'
  id=152000  decoded=' Κατὰ'
  id=153599  decoded='οῦσιν'

✓ All checks pass.
```

Headlines:

- **Polytonic compression is the big win**: 60 → 20 tokens on a real Septuagint-style sentence is a 3× improvement. Katharevousa drops 65.9 %.
- **Modern Greek web also wins** at 31.8 % (consistent with the C3 paper's claim of ~30 % inference cost reduction on Greek workloads at this cutoff).
- **English and Russian don't budge**, which is what the multilingual-preservation hard constraint demands. The added units don't poison the other-language inference path.

## 4. Structural-integrity checklist (the "appropriately constructed in every sense" claim)

Each of the hard constraints from [`docs/GLOBAL_DECISIONS.md`](../../../docs/GLOBAL_DECISIONS.md) was checked against the rebuilt artifact:

| constraint | result |
|---|---|
| Match Apertus tokenization behavior | `normalizer / pre_tokenizer / decoder` are byte-identical to Apertus base ✓ |
| Preserve the fixed first 1,000 ids | 0 mismatches across ids 0..999 ✓ |
| Preserve special-token behavior | `added_tokens` list is identical (same 1,000 entries, same content, same metadata flags) ✓ |
| Preserve the regex split + `ByteLevel` regime | pre_tokenizer is `Sequence(Split(GPT-2-style regex) → ByteLevel)` ✓ |
| Final vocab size must remain divisible by 128 | 153,600 mod 128 = 0 ✓ |
| Final vocab divisible by 256 (project-stronger rule) | 153,600 mod 256 = 0 (and 153,600 = 256 × 600) ✓ |
| `tie_word_embeddings = false` (model-side) | model property — preserved by Apertus's config, unchanged here |
| Append-only extension (no renumbering) | id ranges 0..131,071 / 131,072..148,479 / 148,480..153,599 are strictly contiguous and append-only ✓ |
| Loadable via the standard HF path | `AutoTokenizer.from_pretrained('ship/apertus_greek_extended_153600')` → `PreTrainedTokenizerFast` ✓ |
| BPE merges count tracks added-id count | 269,443 (Apertus) → 286,851 (+17,408 for C3) → 291,971 (+5,120 for poly) ✓ |

## 5. Two things this artifact does NOT include yet

1. **The corresponding resized model checkpoint.** Adding 22,528 ids
   (17,408 + 5,120) means we have to resize Apertus-8B-2509's input
   embedding matrix `E` and LM head `U`, both [131,072, 4,096] →
   [153,600, 4,096]. That's ~22,528 × 8 KiB × 2 = ~360 MiB of new
   parameters across both matrices. The plan for *how* to initialize
   those new rows is the three-arm comparison in
   [`../experiments_plan.md`](../experiments_plan.md) §5
   (Vanilla / ReTok / Distillation). The resize itself is a one-liner
   (`model.resize_token_embeddings(153600)`), but the *content* of the
   new rows is the experimental question.
2. **A push to the HuggingFace Hub.** The ship bundle lives in this
   subproject for now. When it goes to CSCS storage,
   `/capstor/store/cscs/swissai/a0140/tokenizers/apertus_greek_extended_153600/`
   is the natural home. We can also push to a private HF repo
   (`fffoivos/apertus-greek-extended-tokenizer-153600`) so that the
   `swiss-ai/Apertus-8B-2509-greek-ext` model derivative can reference
   it by name.

## 6. Reproducibility

```bash
SHIP_DIR=/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600
APERTUS=/home/foivos/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/snapshots/3162c99675aa588097cecd4a24b9aa1f712af477
POLY=/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/variants/c3p_poly_added_5120

mkdir -p "$SHIP_DIR"
cp "$POLY/tokenizer.json"          "$SHIP_DIR/tokenizer.json"
cp "$APERTUS/tokenizer_config.json" "$SHIP_DIR/tokenizer_config.json"
cp "$APERTUS/special_tokens_map.json" "$SHIP_DIR/special_tokens_map.json"
```

Verification harness (run from `home`):

```bash
/home/foivos/.venvs/glossapi-merge-docling/bin/python3 -c "
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('$SHIP_DIR')
assert tk.vocab_size == 153600
assert tk.bos_token_id == 1 and tk.eos_token_id == 2 and tk.pad_token_id == 3 and tk.unk_token_id == 0
print(f'OK: class={type(tk).__name__} vocab={tk.vocab_size}')
"
```

SHA-256 (manifested in `manifest.json` after `cp`):

```
tokenizer.json              b1eeb739a564b3abd33c1b85a16162b8284d98f9ab5d67528d3cbe8a82e9cbad
tokenizer_config.json       ea64a17b41e1deaa7469212f413676129f33977ca3a48767f0ca68dc346df502
special_tokens_map.json     816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5
```
