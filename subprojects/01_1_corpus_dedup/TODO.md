# TODO

- document the dedup guarantees already provided by the upstream corpus dataset pipeline
- verify whether any lightweight downstream dedup annotations still remain in the builder flow
- record which dedup metadata artifacts downstream consumers are allowed to rely on
- avoid introducing a second independent local dedup pipeline for HPLT prep
- freeze and snapshot the current pathological exact-stage run before modifying code
- refactor exact-stage finalization so SQLite is no longer the heavy export engine
- add shard-level exact-stage outputs and resume markers
- resume the current run from saved state rather than restarting from raw input
- add medium-scale resume tests for the repaired exact-stage path
