# Bibliography role pipeline run — 2026-07-15

This directory is the durable local archive of the provenance manifests and
reports for the first complete heading → connector → structured-block run. It
contains metrics and hashes, not the model binaries.

Clariden artifact root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_role_pipeline_20260715`

The dual heading-review packet, pass receipts, and adjudicated overlay are also
available locally at:

`/Users/foivoskarounos-zamparloukos/presentations/train-apertus-with-glossapi/bibliography-role-pilot-20260715/heading-review-a3522be`

## Executed stages

| Stage | Slurm job | Clariden artifact | Train-only OOF result |
|---|---:|---|---|
| Contextual heading inventory | 2766168 | `heading_inventory_2776933_r1` | 24,616 candidates; trusted recall 1.0 |
| Source/fold/shape review selection | 2766183 | `heading_review_selection_a3522be_q500_r1` | 1,419 dual-reviewed cases |
| Heading table | 2766683 | `heading_table_6b28e4c_r1` | 939,014 lines; 2,137 trusted candidates |
| Heading expert | 2766684 | `heading_oof_6b28e4c_r1` | any-heading PR-AUC 0.9690; typed macro PR-AUC 0.7651 |
| Connector table | 2766807 | `connector_table_8f8d6ed_r1` | 8,040 reviewed lines; 876 trusted connectors |
| Connector expert | 2766811 | `connector_oof_8f8d6ed_r1` | connector PR-AUC 0.6453; conditional subtype PR-AUC 0.9709 |
| Fully reviewed block table | 2766827 | `block_table_8f8d6ed_r1` | 103 sequences; 4,382 lines; 67 gold blocks |
| Final structured model | 2767308 | `block_model_23b4034_r1` | precision passed; recall gate failed |

The two independent heading passes agreed exactly on 1,347/1,419 cases
(94.93%). The 72 disagreements remained masked. Validation was not opened in
any stage.

## Final verdict

The final grouped OOF structured result is research-complete but not approved
for corpus removal:

- line precision: 0.999431;
- line recall: 0.943087;
- character precision: 0.999698;
- character recall: 0.943466;
- trusted hard-stop crossings: 0;
- spurious blocks per reviewed zero-BIB sequence: 0.

The deployment gate requires at least 0.99 precision and 0.95 recall for both
line and character measures. `deployment_gate_passed` is therefore `false`.
The held-out validation set remains unopened.

## Important proposal diagnostic

The line expert is supervised by all reviewed lines. Proposal reachability is
tracked separately: radius 30 reaches 673/876 trusted continuation/filler
examples, radius 50 reaches 754, radius 100 reaches 808, and radius 200 reaches
867. Most radius-30 misses (191/203) are OpenArchives lines, heavily influenced
by a 1,713-line OCR/table bibliography. This evidence is why unreachable
examples were not silently dropped and why the normal proposal window was not
expanded to 200.
