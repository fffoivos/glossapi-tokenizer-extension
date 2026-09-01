# Analysis entry points

Only scripts that support the three canonical questions are copied here:

- `analyze_retention_snapshot.py` and `forecast_retention_slope.py` quantify
  the learning/forgetting trajectory;
- `analyze_per_document_endpoints.py` compares exact document-local endpoint
  measurements;
- `analyze_greekmmlu_answer_drift.py` measures checkpoint-level answer changes;
- `analyze_checkpoint_source_exposure.py` reconstructs source exposure at each
  checkpoint.

They are preserved byte-for-byte from the completed-run workspace. Raw CSCS
prediction payloads, packing catalogs and receipts remain under subproject 07
and the paths embedded in the canonical JSON payloads. The scripts are not a
second training or evaluation pipeline.
