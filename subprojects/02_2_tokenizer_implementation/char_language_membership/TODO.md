# TODO — char_language_membership

## v2 status

Built. See `PLAN.md` for the design and `manifest.json` for
build-time metadata.

### Done

- [x] 55 (language, script, encoding) triples in `languages.yaml`.
- [x] CLDR JSON sourcing (release-keyed cache, fail-fast on missing
      locale).
- [x] Strict-rule build: positive evidence only, four deterministic
      closures (script-filter, case, NFD, script-range fallback).
- [x] Auxiliary excluded (preserves `el` vs `el-polyton` encoding
      distinction).
- [x] Substrate = ALL_BITS (`N*`/`P*`/`S*`/`Z*`/`Cc`/`Cf`).
- [x] Apply script with substrate-aware fallback at apply time.
- [x] `validate.py` covering case/NFD/script-fallback/substrate/new
      locales/sister-language attribution.
- [x] Docs (PLAN, README) updated for (language, script, encoding)
      scope framing and Apertus-as-proxy assumption.

## Open

- [x] **Audit gate codified in validate.py** — phase-2 token check
      asserts <50 fall-through tokens per *out-of-scope* script.
      In-scope scripts with coverage gaps are reported but don't fail.
      Current state: out-of-scope `Other` = 1 (well under threshold).
      In-scope coverage gaps (reported, not failing):
      - **Cyrl 132** — Kazakh/Belarusian/Mongolian/Tatar Cyrillic
        extensions (ў ә ү қ ө ң ғ …) from out-of-scope languages.
        Could close by adding `be` (Belarusian), `kk` (Kazakh),
        `mn` (Mongolian) at bits 55–57.
      - **Latn 42** — Esperanto (ĉ ĝ ĥ ĵ ŝ ŭ), Japanese romaji (ō),
        ð/þ residue. Closeable with `eo`.
      - **Arab 38** — Pashto/Sindhi/Uyghur extensions
        (ښ ڤ ڠ ے ہ ں …). Closeable with `ps` / `sd` / `ug`.
- [x] **Polytonic Greek decomposition** — verified post-fallback NFD
      closure works: ᾳ ᾴ ᾲ ἀ are `el-polyton` only; α/ά are `el` +
      `el-polyton`; combining ypogegrammeni U+0345 inherits the
      `el-polyton` bit from its precomposed parents.
- [ ] **Vocabulary version pinning** — when Apertus ships an updated
      tokenizer snapshot, rerun apply and re-audit. Right now the
      apply hardcodes the snapshot path in the run command.

## Extension work (later)

- [ ] Append the next tier of FineWeb-2 languages if the audit
      surfaces a script with >50 fall-through tokens:
      Khmer (`km`), Lao (`lo`), Tibetan (`bo`), Mongolian (`mn`),
      Sinhala (`si`), Cherokee, Ethiopic, etc. Bits 55+, never reused.
- [ ] If a downstream consumer ever needs hot-path lookup over the
      codepoint table, port to Rust (the table is small enough that
      a `[u64; 0x110000]` lookup array fits in ~9 MB).
