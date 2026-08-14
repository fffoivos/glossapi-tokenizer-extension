#!/usr/bin/env python3
"""Compare token-aligned D0 0.5B and sanitized full-8B native-Greek trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "d0_0p5b_vs_full8_native_greek_3cp_20260814"
PRESENTATION_DATA = ROOT / "presentations" / "NATIVE_GREEK_3CP_BENCHMARKS.data.json"
OUTPUT_JSON = ROOT / "D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.data.json"
OUTPUT_MD = ROOT / "D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md"

D0_CHECKPOINTS = [
    ("d0_iter_0000000", "0B", 0.0),
    ("d0_iter_0018944", "39.728B", 39.728447488),
    ("d0_iter_0038496", "80.732B", 80.731963392),
]
FULL8_CHECKPOINTS = [
    ("iter_0000000", "0B", 0.0),
    ("iter_0009536", "39.997B", 39.996882944),
    ("iter_0018284", "76.689B", 76.688654336),
]
BENCHMARKS = [
    ("greekmmlu", "GreekMMLU"),
    ("asep_mcqa", "ASEP MCQA"),
    ("demosqa", "DemosQA"),
    ("gpcr", "GPCR"),
    ("medical_mcqa", "Medical MCQA"),
    ("oyxoy_metaphor", "OYXOY metaphor"),
    ("oyxoy_nli", "OYXOY NLI binary"),
    ("oyxoy_wic", "OYXOY WiC"),
    ("oyxoy_wsd_definition", "OYXOY WSD"),
    ("oyxoy_nli_exact_set", "OYXOY NLI exact set"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def read_filtered(root: Path, checkpoints: list[tuple[str, str, float]]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for checkpoint, _, _ in checkpoints:
        path = root / checkpoint / "strict_filtered_metrics.csv"
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["subject"] == "__all__"]
        result[checkpoint] = {
            row["benchmark"]: {
                key: numeric(row.get(key))
                for key in ("n", "accuracy", "balanced_accuracy", "choice_nll", "correct_answer_bpb")
            }
            for row in rows
        }
    return result


def add_greekmmlu(
    d0: dict[str, dict[str, dict]], full8: dict[str, dict[str, dict]]
) -> None:
    iterations = ["0000000", "0018944", "0038496"]
    for (checkpoint, _, _), iteration in zip(D0_CHECKPOINTS, iterations, strict=True):
        path = EVIDENCE / "d0_greekmmlu" / f"iteration_{iteration}_receipt.json"
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "completed":
            raise ValueError(f"incomplete GreekMMLU receipt: {path}")
        metrics = receipt["metrics"]["decontaminated"]
        d0[checkpoint]["greekmmlu"] = {
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": None,
            "choice_nll": metrics["choice_nll"],
            "correct_answer_bpb": metrics["correct_answer_bpb"],
        }

    presentation = json.loads(PRESENTATION_DATA.read_text())
    greekmmlu = next(row for row in presentation["benchmarks"] if row["id"] == "greekmmlu")
    for index, (checkpoint, _, _) in enumerate(FULL8_CHECKPOINTS):
        full8[checkpoint]["greekmmlu"] = {
            "n": greekmmlu["n_filtered"],
            "accuracy": greekmmlu["accuracy_filtered"][index],
            "balanced_accuracy": None,
            "choice_nll": greekmmlu["choice_nll_filtered"][index],
            "correct_answer_bpb": greekmmlu["correct_answer_bpb_filtered"][index],
        }


def trajectory(values: list[float | None], lower_is_better: bool) -> dict:
    if any(value is None for value in values):
        return {"values": values, "best_checkpoint": None, "phase_deltas": [None, None]}
    concrete = [float(value) for value in values if value is not None]
    best = min(range(3), key=concrete.__getitem__) if lower_is_better else max(range(3), key=concrete.__getitem__)
    return {
        "values": concrete,
        "best_checkpoint": ["initial", "mid", "final"][best],
        "phase_deltas": [concrete[1] - concrete[0], concrete[2] - concrete[1]],
    }


def direction(delta: float | None, lower_is_better: bool) -> str | None:
    if delta is None:
        return None
    if delta == 0:
        return "tie"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "worsened"


def fmt(values: list[float | None], percentage: bool = False) -> str:
    if percentage:
        return " → ".join("—" if value is None else f"{100 * value:.2f}%" for value in values)
    return " → ".join("—" if value is None else f"{value:.4f}" for value in values)


def main() -> int:
    d0 = read_filtered(EVIDENCE / "d0_filtered", D0_CHECKPOINTS)
    full8 = read_filtered(EVIDENCE / "full8_filtered", FULL8_CHECKPOINTS)
    add_greekmmlu(d0, full8)

    rows = []
    nll_phase_agreement = [0, 0]
    nll_phase_comparable = [0, 0]
    nll_best_agreement = 0
    for benchmark, label in BENCHMARKS:
        d0_accuracy = trajectory([d0[cp][benchmark]["accuracy"] for cp, _, _ in D0_CHECKPOINTS], False)
        full8_accuracy = trajectory([full8[cp][benchmark]["accuracy"] for cp, _, _ in FULL8_CHECKPOINTS], False)
        d0_nll = trajectory([d0[cp][benchmark]["choice_nll"] for cp, _, _ in D0_CHECKPOINTS], True)
        full8_nll = trajectory([full8[cp][benchmark]["choice_nll"] for cp, _, _ in FULL8_CHECKPOINTS], True)
        phase_match = []
        for phase in range(2):
            left = direction(d0_nll["phase_deltas"][phase], True)
            right = direction(full8_nll["phase_deltas"][phase], True)
            match = None if left is None or right is None else left == right
            phase_match.append(match)
            if match is not None:
                nll_phase_comparable[phase] += 1
                nll_phase_agreement[phase] += int(match)
        best_match = (
            None
            if d0_nll["best_checkpoint"] is None or full8_nll["best_checkpoint"] is None
            else d0_nll["best_checkpoint"] == full8_nll["best_checkpoint"]
        )
        if best_match:
            nll_best_agreement += 1
        rows.append(
            {
                "benchmark": benchmark,
                "label": label,
                "n": int(d0[D0_CHECKPOINTS[0][0]][benchmark]["n"]),
                "d0_accuracy": d0_accuracy,
                "full8_accuracy": full8_accuracy,
                "d0_choice_nll": d0_nll,
                "full8_choice_nll": full8_nll,
                "choice_nll_phase_direction_match": phase_match,
                "choice_nll_best_checkpoint_match": best_match,
            }
        )

    evidence_paths = sorted(path for path in EVIDENCE.rglob("*") if path.is_file())
    payload = {
        "schema_version": "apertus_d0_0p5b_vs_full8_native_greek_3cp_v1",
        "status": "completed",
        "comparison_policy": {
            "primary_metric": "strict-filtered choice NLL; decontaminated choice NLL for GreekMMLU",
            "checkpoint_alignment": "initial, nearest saved checkpoint to 40B token slots, final",
            "replication_test": "exact agreement of improvement/worsening direction in each interval and exact best-checkpoint identity; no uncertainty threshold is imposed",
        },
        "checkpoints": {
            "d0_0p5b": [dict(id=cp, label=label, token_slots_b=tokens) for cp, label, tokens in D0_CHECKPOINTS],
            "full8": [dict(id=cp, label=label, token_slots_b=tokens) for cp, label, tokens in FULL8_CHECKPOINTS],
        },
        "data_and_model_differences": {
            "d0_0p5b_active_tokens": 80_729_939_067,
            "full8_active_tokens": 76_685_490_476,
            "full8_fewer_active_tokens": 4_044_448_591,
            "full8_postmask_dedup_dropped_documents": 2_386_676,
            "full8_masked_documents": 2_515_489,
            "d0_0p5b": {
                "geometry": "20 layers, hidden 1024, 16 attention heads, 4 KV heads",
                "embedding_initialization": "tied layer-7 Token Distillation with MSE plus auto-weighted CE",
                "peak_lr": 0.00015,
                "rope": "theta 500000, no scaling",
                "data": "pre-sanitation 80.730B-active-token schedule",
            },
            "full8": {
                "geometry": "32 layers, hidden 4096, 32 attention heads, 8 query groups",
                "embedding_initialization": "untied layer-11 Token Distillation plus polytonic output calibration",
                "peak_lr": 0.000055,
                "rope": "theta 500000, scaling factor 8",
                "data": "PII-masked and globally exact post-mask-deduplicated 76.685B-active-token schedule",
            },
        },
        "summary": {
            "choice_nll_initial_to_mid_direction_agreement": f"{nll_phase_agreement[0]}/{nll_phase_comparable[0]}",
            "choice_nll_mid_to_final_direction_agreement": f"{nll_phase_agreement[1]}/{nll_phase_comparable[1]}",
            "choice_nll_best_checkpoint_agreement": f"{nll_best_agreement}/9",
            "interpretation": "early mainstream-task learning partially replicates, but late checkpoint shape does not",
        },
        "benchmarks": rows,
        "evidence": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in evidence_paths
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    nll_rows = []
    accuracy_rows = []
    for row in rows:
        d0_nll = row["d0_choice_nll"]
        full8_nll = row["full8_choice_nll"]
        nll_rows.append(
            f"| {row['label']} | {fmt(d0_nll['values'])} | {d0_nll['best_checkpoint'] or '—'} | "
            f"{fmt(full8_nll['values'])} | {full8_nll['best_checkpoint'] or '—'} |"
        )
        accuracy_rows.append(
            f"| {row['label']} | {fmt(row['d0_accuracy']['values'], True)} | "
            f"{fmt(row['full8_accuracy']['values'], True)} |"
        )

    report = f"""# D0 0.5B versus sanitized 8B: native-Greek three-checkpoint replication

Date: 2026-08-14

## Verdict

The 0.5B D0 run **partially reproduces early learning, but does not reproduce the
8B late-training trajectory**. On strict-filtered choice NLL, the direction
from initialization to approximately 40B token slots agrees on
**{nll_phase_agreement[0]}/{nll_phase_comparable[0]}** comparable benchmarks.
From approximately 40B to the endpoint it agrees on only
**{nll_phase_agreement[1]}/{nll_phase_comparable[1]}**, and the exact
best-checkpoint identity agrees on **{nll_best_agreement}/9** benchmarks with
choice NLL.

The practical result is narrow but useful: a Mini checkpoint near 40B is a
credible screening point for whether mainstream Greek MCQ capability has
emerged. It is not a reliable proxy for whether the 8B model will peak there or
continue improving afterwards.

## Choice NLL trajectories

Lower is better. Each cell is initialization → approximately 40B → final. All
non-GreekMMLU rows use the strict contamination filter; GreekMMLU uses its
separate frozen 16,159-question decontaminated subset.

| Benchmark | D0 0.5B NLL | 0.5B best | Sanitized 8B NLL | 8B best |
| --- | --- | --- | --- | --- |
{chr(10).join(nll_rows)}

## Accuracy trajectories

Higher is better. Accuracy is secondary to choice NLL, particularly for the
imbalanced OYXOY binary tasks.

| Benchmark | D0 0.5B accuracy | Sanitized 8B accuracy |
| --- | --- | --- |
{chr(10).join(accuracy_rows)}

## What actually replicates

- GreekMMLU, ASEP, DemosQA, GPCR, Medical MCQA and OYXOY WSD all improve in
  choice NLL from initialization to approximately 40B at both scales.
- GPCR continues improving after 40B at both scales.
- Medical MCQA worsens slightly after 40B at both scales, making it the clearest
  replicated mid-run peak.
- The other late trajectories do not transfer. The 0.5B GreekMMLU, ASEP and WSD
  NLLs continue improving slightly, while their 8B counterparts worsen. DemosQA
  changes by very little in either model but with opposite exact signs.
- OYXOY metaphor, NLI and WiC begin from very different label-bias regimes at
  the two scales. Their raw accuracies and NLL shapes therefore should not be
  interpreted as a clean scale-replication result.

## Why a mismatch is unsurprising

This is a token-aligned scorer replication, not a controlled scale study:

- **Architecture:** Mini is 20 × 1,024 with tied embeddings; 8B is 32 × 4,096
  with untied embeddings.
- **Embedding adaptation:** Mini selected tied layer-7 TD with MSE plus
  auto-weighted CE. The 8B initialization used untied layer-11 TD plus separate
  polytonic output calibration.
- **Optimization:** both use WSD-10, but Mini peaks at `1.5e-4` and 8B at
  `5.5e-5`; their global-token batches are 2.097M and 4.194M respectively.
- **Data:** Mini consumed the pre-sanitation 80.730B-token schedule. The 8B run
  used PII masking followed by exact post-mask deduplication, dropping
  2,386,676 documents and reducing active tokens to 76.685B—a 4.044B-token
  difference.
- **RoPE:** both use theta 500,000 and 4,096 context, but Mini uses the native
  no-scaling geometry while the 8B recipe uses scaling factor 8.

These differences prevent a causal claim that model scale alone caused the
trajectory mismatch.

## Execution and evidence

- Clariden scoring jobs: `3079741` preserved 39 completed shards before its
  planned wall-time exit; resume job `3079936` completed the remaining 24,
  aggregated all 63, and finished in 13:06.
- Each checkpoint contains 83,970 newly scored examples across the frozen suite.
- The strict filter applied exactly 10,076 frozen exclusions at every checkpoint.
- D0 matrix receipt SHA-256:
  `86eecdefdcbb32717773a096f0afffb27ad907129f7e88e70afd09dda17c6849`.
- D0 contamination-filter receipt SHA-256:
  `6e8b3b07ad4c450685cf3c30d4f9562acc4307ded55d14779d2522571126b2af`.
- Machine-readable comparison: `{OUTPUT_JSON.name}`.

Greek Protipa Exams remains intentionally unscored because its owner-side
manual access gate was not available. It is not counted as evaluated.
"""
    OUTPUT_MD.write_text(report)
    print(json.dumps({"ok": True, "json": str(OUTPUT_JSON), "report": str(OUTPUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
