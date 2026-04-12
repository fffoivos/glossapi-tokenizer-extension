# TODO

- rebuild the HPLT slice on a GCP worker with:
  - quality bins `>=8`
  - no `Machine translated or generated`
  - real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring
  - `greek_badness_score <= 60`
- freeze the exact HPLT filtering policy in explicit machine-readable form
- freeze whether HPLT `filter == keep` is required in the final slice
- freeze the exact HPLT fields preserved inside `source_metadata_json`
- rerun the full prepared-source dataset with the corrected HPLT slice included, using the existing dataset scripts
- rebuild the review sample under the current quality policy instead of using the stale exploratory sample
- keep this operational track separate from the tokenizer critical path
