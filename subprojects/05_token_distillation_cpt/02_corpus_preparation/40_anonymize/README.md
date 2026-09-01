# 40 — Anonymize (PII masking)

> **In one line:** an Apertus-parity email/IP/IBAN masker with a Greek-IBAN fix, a recall study that deliberately went *beyond* it and was not shipped, and — two months later — a full receipt-bound pipeline that re-published the whole 51.8 M-row v2 corpus anonymized.
> **Period:** masker work before 2026-06-11 (committed in the bulk checkpoint `a19c136f`); the `hf_v2_release/` pipeline written 2026-08-11; both entered the consolidation branch late (`hf_v2_release/` and the masker fixed-point change only on 2026-09-01, `2aec4a66`). **Status:** masker completed and in the CPT launch path; the v2 release pipeline is written and tested, with **no run receipt in this tree**.
> **Came from / led to:** [`../30_decontaminate`](../30_decontaminate/README.md) → this → sharding/tokenization in [`../../03_training_experiments`](../../03_training_experiments/README.md) and the release path in [`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md)

## Why this existed

The pipeline order fixed in [`../../ARCHIVE.md`](../../ARCHIVE.md) is *"clean, dedup-validate,
decontaminate, anonymize, shard"* — anonymization runs **after** decontamination, so masking
never hides a contamination match. The requirement was parity with the Apertus pretraining
data pipeline (so the masked tokens are the ones the tokenizer already reserves), plus
whatever Greek-specific correction that parity got wrong.

## History

### Before 2026-06-11 — the masker, and the one deliberate deviation

`pii_masker.py` takes the masking loop and the email/IP regexes **verbatim** from Apertus's
`swiss-ai/pretrain-data` `PIIFormatter` at commit `8af990b9401101cf95acd02b066ed0c449789126`.
The single deviation is IBAN: Apertus's `iban_regex` was tuned for ~22-character
space-grouped IBANs and, on Greek **27-character** IBANs, either truncates (leaving a PII
fragment) or misses the compact form. The replacement does per-country exact-length matching
(compact *or* single-space-grouped) plus ISO-7064 mod-97 validation. Replacement tokens are
unchanged — `<email-pii>` / `<ip-pii>` / `<iban-pii>` — because they are reserved Apertus
tokenizer tokens. Stdlib only, so it is portable and testable; `pii_formatter.py` wraps it
for datatrove and `anonymize.py` runs it format-agnostically (parquet or jsonl, text/id
fields as CLI arguments) on one CPU node, 64 datatrove tasks. Evidence: file docstrings and
[`scripts/run_anonymize.sbatch`](scripts/run_anonymize.sbatch), whose header argues the
compute choice explicitly (CPU only, no GPU; a Rust port judged not worth the parity risk).

### Before 2026-06-11 — the "above and beyond" recall study, not shipped

[`above_and_beyond/exploration/scan_hplt_greek_pii.py`](above_and_beyond/exploration/scan_hplt_greek_pii.py)
is labelled in its own docstring *"NOT a production detector. Investigative instrument for
the recall sweep."* It asked what a generic email/phone/IP detector **misses in Greek**:
checksum-validated ΑΦΜ / ΑΜΚΑ / ΑΔΤ, Greek IBANs, vehicle plates (handling the 14 Greek
glyphs with Latin homoglyphs), addresses, honorifics. Scanned 40,000 HPLT docs /
294,991,374 chars ([`above_and_beyond/exploration/dim2_hplt_pii_scan_counters.json`](above_and_beyond/exploration/dim2_hplt_pii_scan_counters.json)):
per 1,000 docs — honorific+name **818.8**, contact cue 48.3, email 40.8, phone 35.1,
address 19.4, plate-format 9.9, IP 7.0, valid ΑΦΜ **1.25**, valid IBAN 1.375, labelled ΑΔΤ
0.05. The structured Greek identifiers are rare; names/honorifics are ubiquitous and were
never in scope. Nothing from this study entered the production masker.

### 2026-08-11 — `hf_v2_release/`: the row-preserving anonymized v2 release

A separate, later program (file mtimes 14:56 → 17:37 on 2026-08-11) that applies the same
masker to the **whole** published corpus rather than to a training-run staging copy.
It is contract-driven: [`hf_v2_release/configs/release.json`](hf_v2_release/configs/release.json)
pins the input repository `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` at
revision `3f97cec48af502f4996cf8ff20b02660e2dd3d31`, **51,839,746 rows / 431 shards**, the
polytonic tokenizer (`fffoivos/apertus-tokenizer-extension`, vocab **148,992**), and the
whole dedup receipt (53,046,533 − 1,206,787 = 51,839,746; MinHash 5-token shingles, 128
permutations, 32 bands, verified Jaccard 0.85). The row policy is explicit: *"preserve every
row, row order, source identity, multiplicity, schema, and all non-text values."*
[`hf_v2_release/tests/test_release_contract.py`](hf_v2_release/tests/test_release_contract.py)
asserts the taxonomy is exactly 37 sources and that the expected per-source rows sum to
51,839,746, with HPLT at 48,629,460. Staged Slurm flow via
[`hf_v2_release/clariden/submit.sh`](hf_v2_release/clariden/submit.sh):
`prepare → canary-a (task 396) → canary-b (task 418) → transform → finalize → publish`, then
a later same-day `*_overlay` pair and `make_public.sbatch`, which makes the repository public
and ungated **and proves anonymous access** against pinned SHAs
(`987b8955…`, manifest `1343b49211…`).

### 2026-09-01 — recovered from an uncommitted working tree

`hf_v2_release/` and a modified `pii_masker.py` existed only as untracked/modified files in
a local worktree (`codex/bib-nextgen-lexicon-gate`) until commit `2aec4a66`. The masker
change makes `mask()` iterate **to a fixed point** instead of running one pass: a candidate
is capped at 34 compact characters, so two concatenated valid IBANs expose the second only
*after* the first replacement. Comment in the diff: *"overlap/boundary exposure cannot leave
PII for a second invocation."*

## Outcome

- **Shipped for the CPT launch:** mask email, IP and IBAN to the reserved single tokens,
  keeping Apertus parity on email/IP and the Greek-IBAN fix — recorded as a standing method
  decision in [`../../ARCHIVE.md`](../../ARCHIVE.md).
- **Deliberately not shipped:** Greek structured-ID and name/honorific detection. The scan
  measured how much is out there (818.8 honorific-name hits per 1,000 docs against 1.25 valid
  ΑΦΜ) and the decision was to keep production at Apertus parity.
- **Written, not evidenced as run:** the 2026-08-11 v2 release pipeline. There is no receipt,
  log or manifest for it anywhere in this directory ( `reports/` holds only `.gitkeep`), so
  whether the public anonymized revision was actually published cannot be established here.
- The fixed-point masking fix is the last substantive change to this stage and arrived
  outside git until the 2026-09-01 recovery.

## Where things are

| Path | What |
|---|---|
| [`scripts/pii_masker.py`](scripts/pii_masker.py) | The masker. Apertus-verbatim email/IP; per-country + mod-97 IBAN; fixed-point loop. |
| [`scripts/pii_formatter.py`](scripts/pii_formatter.py) | datatrove wrapper; writes `pii_count` / `pii_by_type` to per-doc metadata. |
| [`scripts/anonymize.py`](scripts/anonymize.py) · [`scripts/run_anonymize.sbatch`](scripts/run_anonymize.sbatch) | Format-agnostic runner + the one-node CPU job. |
| [`scripts/test_pii_local.py`](scripts/test_pii_local.py) | Local masker tests. |
| [`hf_v2_release/configs/release.json`](hf_v2_release/configs/release.json) | The pinned release contract (input revision, tokenizer, dedup receipt, 37-source taxonomy). |
| [`hf_v2_release/scripts/pipeline.py`](hf_v2_release/scripts/pipeline.py) | prepare / transform / finalize stages with per-file receipts. |
| [`hf_v2_release/scripts/publish_release.py`](hf_v2_release/scripts/publish_release.py) · [`ensure_public_release.py`](hf_v2_release/scripts/ensure_public_release.py) | Publish through a checked HF PR; then make public + ungated and prove anonymous access. |
| [`hf_v2_release/clariden/`](hf_v2_release/clariden) | The staged Slurm jobs and `submit.sh`. |

## Working documents

This stage carries no prose documents — the design rationale lives in script docstrings and
the sbatch headers, and the standing decisions are recorded upstream in
[`../../ARCHIVE.md`](../../ARCHIVE.md) ("Corpus-Prep Method Summary").
[`above_and_beyond/`](above_and_beyond) is historical exploration, not a production path.
