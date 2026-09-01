#!/usr/bin/env python3
"""Reduce the raw per-added-token audit payloads into the compact 09.3 result payload.

The raw payloads (three ~16 MB JSON files, one per checkpoint) stay on CSCS; this
script emits the aggregate tables plus hash pointers to them.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

POLY_START = 148_480
VOCAB = 148_992
CKPTS = [(400, "step400-tokens2B"), (9_536, "step9536-tokens40B"), (18_284, "main")]
CSCS_OUT = "/iopsstor/scratch/cscs/fffoivos/newtok_audit_20260823/out"


def bytes_to_unicode() -> dict[int, int]:
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(cs, bs))


_B2U = bytes_to_unicode()


def decode(surface: str) -> str:
    try:
        return bytes(bytearray(_B2U[ord(c)] for c in surface)).decode("utf-8", "replace")
    except KeyError:
        return surface


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_rows(payload: dict, group: str, min_occ: int = 4) -> list[dict]:
    rows = []
    for tid, value in payload["per_token"].items():
        token = int(tid)
        if group == "modern" and token >= POLY_START:
            continue
        if group == "polytonic" and token < POLY_START:
            continue
        if value.get("n_occ", 0) < min_occ:
            continue
        rows.append(value)
    return rows


def q(values, p):
    return float(np.quantile(values, p)) if len(values) else None


def summarise(rows: list[dict], update: int) -> dict:
    delta = np.array([r["mean_delta_logp"] for r in rows])
    frac_pos = np.array([r["frac_delta_positive"] for r in rows])
    h11 = np.array([r["mean_hidden_cos"] for r in rows if r.get("mean_hidden_cos") is not None])
    h30 = np.array([r["mean_hidden_cos_late"] for r in rows
                    if r.get("mean_hidden_cos_late") is not None])
    rank = np.array([r["echo"]["rank"] for r in rows if r.get("echo")])
    return {
        "update": update,
        "n_scored_tokens": len(rows),
        "delta_logp": {"p5": q(delta, .05), "p50": float(np.median(delta)), "p95": q(delta, .95)},
        "frac_tokens_nonpositive": float((delta <= 0).mean()),
        "frac_occurrences_positive_p50": float(np.median(frac_pos)),
        "hidden_cos_layer11": {"p5": q(h11, .05), "p50": float(np.median(h11))} if h11.size else None,
        "hidden_cos_layer30": {"p5": q(h30, .05), "p50": float(np.median(h30)),
                               "p95": q(h30, .95)} if h30.size else None,
        "echo_top1_rate": float((rank == 1).mean()) if rank.size else None,
        "echo_top10_rate": float((rank <= 10).mean()) if rank.size else None,
    }


def main(raw_dir: str, out_path: str) -> int:
    raw = Path(raw_dir)
    payloads = {u: json.loads((raw / f"audit_{name}.json").read_text(encoding="utf-8"))
                for u, name in CKPTS}
    coverage = json.loads((raw / "coverage.json").read_text(encoding="utf-8"))
    coverage.pop("zero_token_ids", None)
    terminal = payloads[18_284]["per_token"]

    trajectory = {g: [summarise(group_rows(payloads[u], g), u) for u, _ in CKPTS]
                  for g in ("modern", "polytonic")}

    peak_to_terminal = {}
    for group in ("modern", "polytonic"):
        peak = {r["surface"]: r for r in group_rows(payloads[9_536], group)}
        term = {r["surface"]: r for r in group_rows(payloads[18_284], group)}
        shared = sorted(set(peak) & set(term))
        change = np.array([term[s]["mean_delta_logp"] - peak[s]["mean_delta_logp"] for s in shared])
        worst = sorted(shared, key=lambda s: term[s]["mean_delta_logp"] - peak[s]["mean_delta_logp"])[:10]
        peak_to_terminal[group] = {
            "n_shared_tokens": len(shared),
            "frac_improved": float((change > 0).mean()),
            "frac_regressed": float((change < 0).mean()),
            "mean_change_nats": float(change.mean()),
            "change_p5": q(change, .05), "change_p95": q(change, .95),
            "largest_regressions": [
                {"surface_readable": decode(s),
                 "delta_at_9536": peak[s]["mean_delta_logp"],
                 "delta_at_18284": term[s]["mean_delta_logp"],
                 "change": term[s]["mean_delta_logp"] - peak[s]["mean_delta_logp"],
                 "n_occ": term[s]["n_occ"]}
                for s in worst],
        }

    by_pieces = collections.defaultdict(list)
    for tid, value in terminal.items():
        if int(tid) < POLY_START and value.get("n_occ", 0) >= 4:
            by_pieces[value["n_base_pieces"]].append(value["mean_delta_logp"])
    delta_by_pieces = [{"base_pieces": p, "n_tokens": len(v), "delta_logp_p50": float(np.median(v))}
                       for p, v in sorted(by_pieces.items()) if len(v) >= 20]

    peak_modern = {r["surface"]: r for r in group_rows(payloads[9_536], "modern")}
    bands, rows = [(4, 20, "4-19"), (20, 100, "20-99"), (100, 500, "100-499"),
                   (500, 10 ** 9, "500+")], []
    for lo, hi, label in bands:
        sel = [(v["n_occ"], v["mean_delta_logp"] - peak_modern[v["surface"]]["mean_delta_logp"])
               for v in group_rows(payloads[18_284], "modern")
               if lo <= v["n_occ"] < hi and v["surface"] in peak_modern]
        if sel:
            ch = np.array([s[1] for s in sel])
            rows.append({"occurrence_band": label, "n_tokens": len(sel),
                         "mean_change_nats": float(ch.mean()),
                         "frac_regressed": float((ch < 0).mean())})

    unscored = [int(t) for t, v in terminal.items() if v.get("n_occ", 0) == 0]
    payload = {
        "schema_version": "added_token_adaptation_v1",
        "meta": {
            "question": ("Did the 17,920 added Greek vocabulary entries adapt in the full-8B "
                         "CPT trajectory, and does that adaptation explain the GreekMMLU peak "
                         "at update 9,536?"),
            "base_vocab": 131_072, "added_modern": 17_408, "added_polytonic": 512,
            "vocab_size": VOCAB,
            "checkpoints": [{"update": u, "staged_as": n} for u, n in CKPTS],
            "occurrences_scored_per_checkpoint": payloads[18_284]["n_occurrences"],
            "documents_scored": payloads[18_284]["n_docs"],
            "unaligned_occurrences": payloads[18_284]["n_unaligned"],
            "probe_layers": {"token_distillation_layer": payloads[18_284]["layer"],
                             "late_layer": payloads[18_284]["late_layer"]},
            "corpus": ("cpt25b held-out sets (excluded from training, GreekMMLU-decontaminated, "
                       "PII-masked): val_historical_polytonic, val_forget_old_greek, val_greek_phd, "
                       "val_openarchives, val_non_hplt, val_hplt"),
            "design_note": ("Both arms score identical text under the same model, so the comparison "
                            "is paired; the corpus is additionally held out, so it is also clean."),
        },
        "coverage": coverage,
        "trajectory": trajectory,
        "peak_to_terminal": peak_to_terminal,
        "delta_logp_by_base_piece_count": delta_by_pieces,
        "peak_to_terminal_change_by_occurrence_band": rows,
        "unmeasurable_tokens": {
            "total_added": VOCAB - 131_072,
            "zero_scored_occurrences": len(unscored),
            "absent_from_corpus": coverage["n_zero"],
            "present_but_single_base_piece": len(unscored) - coverage["n_zero"],
            "note": ("A single-character added token whose base decomposition is one token has no "
                     "merged-vs-split contrast; the test is inapplicable to it, not failed."),
        },
        "raw_payload_pointers": [
            {"name": f"audit_{name}.json", "update": u,
             "cscs_path": f"{CSCS_OUT}/audit_{name}.json",
             "bytes": (raw / f"audit_{name}.json").stat().st_size,
             "sha256": sha256(raw / f"audit_{name}.json")}
            for u, name in CKPTS],
        "provenance": {
            "corpus_build_job": 3_162_848, "pipeline_smoke_job": 3_162_819,
            "throughput_probe_job": 3_162_887, "audit_job": 3_162_910,
            "audit_elapsed": "00:38:31", "audit_partition": "normal",
            "audit_walltime_granted": "02:00:00",
        },
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"ok": True, "out": out_path,
                      "bytes": Path(out_path).stat().st_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
