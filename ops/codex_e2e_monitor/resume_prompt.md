Do one monitoring pass on the GlossAPI Greek tokenizer end-to-end chain.

Required behavior:
- read the canonical plan/state files first:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/README.md`
  - `/home/foivos/Projects/glossapi-tokenizer-extension/docs/PROJECT_INDEX.md`
  - `/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md`
  - `/home/foivos/Projects/glossapi-tokenizer-extension/docs/CURRENT_STATUS.md`
  - `/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md`
  - `/home/foivos/Projects/glossapi-tokenizer-extension/config/apertus_greek_extension.yaml`
- read the latest monitor note first:
  - `/home/foivos/data/glossapi_work/logs/codex_e2e_monitor/latest.md`
- consult recent note history when needed:
  - `/home/foivos/data/glossapi_work/logs/codex_e2e_monitor/notes`
- inspect the live GCP worker state
- check whether every stage is healthy and progressing
- use the plan plus real markers/logs to determine whether the chain is on the intended path
- if something failed or drifted, fix it and steer the process back onto the intended path
- do not stop at diagnosis if a concrete repair is possible

Minimum checks:
- corrected HPLT build health or completion
- HPLT integration watcher
- dedup watcher
- dedup overlay watcher
- tokenizer mix watcher
- tokenizer training launcher watcher
- actual discovery tokenizer jobs if they have already been launched

Artifacts to verify when relevant:
- `/home/foivos/data/glossapi_work/hf_release_publish_hplt_clean60/hplt_clean60_summary.json`
- `/home/foivos/data/glossapi_work/hf_release_publish_working/hplt_integration_summary.json`
- `/home/foivos/data/glossapi_work/analysis/dedup/text_publish/state/gcp_refresh_20260413/latest_success.json`
- `/home/foivos/data/glossapi_work/hf_release_publish_working/dedup_metadata/latest.json`
- `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413/glossapi_only/mix.parquet`
- `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413/glossapi_plus_hplt_70_30/mix.parquet`
- `/home/foivos/data/glossapi_work/tokenizer_training_runs_20260413/glossapi_only_50k/training_summary.json`
- `/home/foivos/data/glossapi_work/tokenizer_training_runs_20260413/glossapi_plus_hplt_70_30_50k/training_summary.json`

If everything is healthy:
- report the current stage
- give the next expected transition
- keep the run moving

If the chain is fully complete:
- report that explicitly
- list the final artifact paths
- do not restart completed stages

End every tick with a concise structured note using these exact headings:
- `Current Stage`
- `What I Checked`
- `Failures Found`
- `What I Changed`
- `Next Expected Transition`

If there were no failures or no changes, say `none`.

Use these evidence sources whenever relevant:
- worker repo code:
  `/home/foivos/Projects/glossapi-tokenizer-extension`
- worker dataset/builder code:
  `/home/foivos/Projects/glossapi-tokenizer-extension/glossapi_corpus_cli`
- service status, real processes, file markers, log freshness, parquet counts, and summary files

If you repair something:
- verify the repair with concrete evidence
- record exactly what changed in the final structured note
