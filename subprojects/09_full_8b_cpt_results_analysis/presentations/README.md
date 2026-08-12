# Canonical presentations

This directory intentionally contains four reports:

- `FULL8_RESULTS.html`: the complete standalone 8B trajectory;
- `FULL8_VS_0P5B.html`: the scale/data-order comparison;
- `CHECKPOINT_BEHAVIOR.html`: GreekMMLU answer behavior and source exposure;
- `NATIVE_GREEK_3CP_BENCHMARKS.html`: full and contamination-filtered native-
  Greek results at initialization, approximately 40B tokens and the endpoint.

Each HTML is self-contained. Its adjacent `.data.json` file is the exact
machine-readable payload used to render it. Earlier progress reports,
previous-run-anchored comparisons and synthetic drift demonstrations remain in
subproject 07 as historical working material and are not canonical results.

Regenerate the native-Greek benchmark report with:

```bash
python3 presentations/build_native_greek_3cp_benchmarks_report.py
```
