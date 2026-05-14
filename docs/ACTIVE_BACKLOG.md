# Active Backlog

## Tokenizer Critical Path

Anchor: [C3_CONVERGENCE.md](C3_CONVERGENCE.md). The only open
tokenizer-side decision is the C3 cutoff; everything below tracks the
work that produces that decision and then ships the extension.

1. Lock the held-out evaluation manifests:
- `GlossAPI` held-out
- `HPLT` held-out
- mixed `GlossAPI + HPLT` held-out
- `modern_greek_eval` (primary decision set)

2. Lock the literal Apertus tokenizer-replication checklist, including
   the exact tokenizer files and a toy extension proof.

3. Build Apertus-compatible merged variants of C3 at each cutoff in the
   frozen grid `{10240, 15360, 20480, 25600}`. Source tokenizer lives
   on the gcloud worker at
   `~/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/tokenizers/`.

4. Run the intrinsic metric bundle on every merged variant:
- `bytes_per_token`
- `tokens_per_byte`
- fertility
- added-token utilization rate
- vocabulary utilization rate
- unreachable added tokens
- byte-fallback rate

5. Run intrinsic + fertility on all four evaluation slices for each
   merged variant. Produce a single comparison table across the four
   cutoffs.

6. Diff C3's learned Greek-unit set against Apertus `model.vocab` and
   `model.merges` to characterize extension quality at each cutoff.

7. Identify the cutoff elbow and freeze the shipped size (already
   `128`-aligned for all four candidates).

8. Implement and test the merge-rule extension in
   `subprojects/02_2_tokenizer_implementation`:
- preserve the first `1000` ids exactly
- preserve special-token behavior
- preserve the regex split and byte-level regime
- non-Greek smoke test after extension

9. Hand off the shipped tokenizer to
   `subprojects/03_apertus_extension_and_embedding_adaptation` for
   embedding + `lm_head` initialization, frozen-base warmup, and full
   continued pretraining. Pre-extension diagnostic of how Apertus
   already represents Greek on E + U is complete under the sub-
   subproject `03_1_greek_embedding_diagnostic/`
   ([report](../subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md))
   and informs the init choice but does not commit to one.

## Archived (pre-convergence, retained for traceability)

These items were the multi-arm exploration that produced C3. They are
done and should not drive new execution. Kept here because some
referenced docs still describe them in detail.

- complete the live downstream continuation from the dedup-complete
  worker state; full-size mix build, tokenizer launch verification,
  E2E worker-run report — see
  `PIPELINE_E2E_*.md` family
- explicit pipeline E2E verification pass from dedup completion through
  tokenizer launch — closed
- near-dedup redesign vs HF/DataTrove MinHash — operational sidetrack;
  semantics frozen
- HPLT slice rebuild with the final filter path (`>=8` bins,
  exclude `Machine translated or generated`, real `Corpus.clean` gate,
  drop `greek_badness_score > 60`) — done
- export BPE-training text and freeze worker-side downstream builder
  inputs for the corpus views — done
- run the full four-arm tokenizer experiment matrix (`F1`, `F2`, `C1`,
  `C2`) — done; arms retained as analyzed baselines only
- compare arms on the shared evaluation bundle to pick the winning arm
  — done; C3 selected (see [C3_CONVERGENCE.md](C3_CONVERGENCE.md))
- builder/tokenizer efficiency sweep on worker hardware
  (`RAYON_NUM_THREADS`, batch size) — closed

## Dataset Operational Sidetrack

1. Replace the stopped score-only HPLT upload attempt with a rebuilt GCP-side slice that includes the real `Corpus.clean` gate.

2. Rerun the full prepared-source dataset with HPLT included, using the existing dataset scripts rather than inventing a new release path.

3. Keep tokenizer work moving off the prepared dataset without waiting for the HF upload to finish.

4. Refresh published `dedup_metadata` only after the intended dataset state is settled; do not block tokenizer work on that refresh.

5. Provision a separate cheap uploader instance for HF publication work.

6. Use the repo-owned uploader handoff under `ops/upload/` to stage the complete filtered HPLT source parquet slice on that uploader instance, without applying physical dataset deduplication to the published source parquets.

7. Stage the refreshed `dedup_metadata` bundle on that uploader instance so downstream builder-time dedup works after HF download.

8. Publish from that uploader instance with the official HF large-folder upload path through [publish_hf_release.py](/home/foivos/Projects/glossapi-tokenizer-extension/publish_hf_release.py), keeping this upload track fully independent of the tokenizer worker.

9. Verify the uploader instance is configured for the best officially recommended HF large-dataset path, including Xet-backed uploads when available, before the next publication run.

10. Keep the downstream builder/tokenizer efficiency work explicit:
- use [BUILDER_TOKENIZER_EFFICIENCY_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/BUILDER_TOKENIZER_EFFICIENCY_PLAN.md)
- preserve builder semantics while reducing unnecessary bundle loads
- benchmark tokenizer throughput on worker hardware before freezing runtime defaults

11. Treat the current worker-side source-only upload as an explicit temporary fallback:
- the intended permanent target is still the separate cheap uploader host
- while that host is unreachable, keep the worker upload low-priority and isolated from dedup
- once the cheap uploader host is reachable again, move the publication path back there

## Immediate Risks

- the exploratory HPLT review sample is not the same thing as the final upload-ready HPLT slice
- the exact tokenizer-replication spec is still missing some literal details and a proof-of-mechanism test
- the current HF uploader strategy is poor for observability and recovery on very large patches
- the old baseline workflow still exists for reference, so active work must stay within the new canonical files
- the live full-size downstream chain is still blocked on a long serial mix stage before tokenizer launch
- tokenizer training still lacks a trustworthy mid-run progress signal
- the cheap uploader host is still unreachable, so the worker-side source-only upload fallback must remain temporary
