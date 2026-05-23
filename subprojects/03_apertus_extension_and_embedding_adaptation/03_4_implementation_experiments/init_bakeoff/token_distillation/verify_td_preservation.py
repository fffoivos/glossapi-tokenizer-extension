#!/usr/bin/env python3
"""Verify Token Distillation changed only the intended embedding rows.

This is the post-TD preservation gate for the Apertus adapter. It compares a
TD checkpoint against its ReTok reference checkpoint and enforces:

* state-dict keys and shapes are identical;
* every non-embedding tensor is unchanged within tolerance;
* input/output embedding rows outside the trained-token set are unchanged;
* trained new input/output rows are finite and moved.

The upstream TD loop already checks original rows before saving. This script is
the persisted artifact-level check we can run independently after the job.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import torch
from safetensors import safe_open


EMBED_KEYS = ("model.embed_tokens.weight", "lm_head.weight")
XIELU_PATTERN = re.compile(r"act_fn\.(alpha_p|alpha_n|beta|eps)")
QK_PATTERN = re.compile(r"self_attn\.(q_norm|k_norm)\.weight")


def read_weight_map(hf_dir: Path) -> Dict[str, str]:
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = hf_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError("found neither model.safetensors.index.json nor model.safetensors in %s" % hf_dir)


def load_tensor(hf_dir: Path, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    shard = hf_dir / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).detach().cpu()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError("shape mismatch: %s vs %s" % (tuple(a.shape), tuple(b.shape)))
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def load_trained_ids(manifest: Path) -> Set[int]:
    data = json.loads(manifest.read_text())
    ids = data.get("trained_token_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("manifest has no non-empty trained_token_ids: %s" % manifest)
    out = set()
    for value in ids:
        if not isinstance(value, int):
            raise ValueError("non-integer trained token id in %s: %r" % (manifest, value))
        out.add(value)
    return out


def sample_ids(ids: Iterable[int], limit: int) -> List[int]:
    out = sorted(set(ids))
    if limit <= 0 or len(out) <= limit:
        return out
    head = out[: limit // 2]
    tail = out[-(limit - len(head)) :]
    return head + tail


def verify(args: argparse.Namespace) -> Dict[str, object]:
    ref_map = read_weight_map(args.reference_hf_dir)
    td_map = read_weight_map(args.td_hf_dir)
    ref_keys = set(ref_map)
    td_keys = set(td_map)
    orig_only = sorted(ref_keys - td_keys)
    td_only = sorted(td_keys - ref_keys)
    if orig_only or td_only:
        raise AssertionError("state_dict keys differ: orig_only=%s td_only=%s" % (orig_only[:5], td_only[:5]))

    trained_ids = load_trained_ids(args.manifest)
    trained_idx = torch.tensor(sorted(trained_ids), dtype=torch.long)

    summary = {
        "reference_hf_dir": str(args.reference_hf_dir),
        "td_hf_dir": str(args.td_hf_dir),
        "manifest": str(args.manifest),
        "trained_token_count": len(trained_ids),
        "tol": args.tol,
        "non_embedding_max_abs_diff": 0.0,
        "xielu_max_abs_diff": 0.0,
        "qk_norm_max_abs_diff": 0.0,
        "embedding_preserved_rows_max_abs_diff": {},
        "trained_rows": {},
        "changed_non_embedding_over_tol": [],
        "shape_mismatches": [],
    }

    changed_non_embedding = []
    shape_mismatches = []

    for key in sorted(ref_keys):
        ref = load_tensor(args.reference_hf_dir, ref_map, key)
        td = load_tensor(args.td_hf_dir, td_map, key)
        if ref.shape != td.shape:
            shape_mismatches.append("%s: %s vs %s" % (key, tuple(ref.shape), tuple(td.shape)))
            continue

        if key not in EMBED_KEYS:
            diff = max_abs_diff(ref, td)
            summary["non_embedding_max_abs_diff"] = max(float(summary["non_embedding_max_abs_diff"]), diff)
            if XIELU_PATTERN.search(key):
                summary["xielu_max_abs_diff"] = max(float(summary["xielu_max_abs_diff"]), diff)
            if QK_PATTERN.search(key):
                summary["qk_norm_max_abs_diff"] = max(float(summary["qk_norm_max_abs_diff"]), diff)
            if diff > args.tol:
                changed_non_embedding.append([key, diff])
            continue

        if ref.ndim != 2:
            raise AssertionError("expected 2D embedding tensor for %s, got %s" % (key, tuple(ref.shape)))
        vocab = ref.shape[0]
        if max(trained_ids) >= vocab:
            raise AssertionError("trained ID outside vocab for %s: max=%d vocab=%d" % (key, max(trained_ids), vocab))

        preserve_mask = torch.ones(vocab, dtype=torch.bool)
        preserve_mask[trained_idx] = False
        preserved_diff = max_abs_diff(ref[preserve_mask], td[preserve_mask])
        summary["embedding_preserved_rows_max_abs_diff"][key] = preserved_diff
        if preserved_diff > args.tol:
            changed_non_embedding.append([key + "[preserved_rows]", preserved_diff])

        trained_ref = ref[trained_idx]
        trained_td = td[trained_idx]
        trained_delta = (trained_ref.float() - trained_td.float()).abs()
        if not torch.isfinite(trained_td.float()).all():
            raise AssertionError("non-finite trained rows in %s" % key)
        changed_rows = (trained_delta.max(dim=1).values > args.trained_change_tol).sum().item()
        summary["trained_rows"][key] = {
            "max_abs_delta": float(trained_delta.max().item()) if trained_delta.numel() else 0.0,
            "mean_abs_delta": float(trained_delta.mean().item()) if trained_delta.numel() else 0.0,
            "rows_changed_over_trained_change_tol": int(changed_rows),
            "sample_trained_ids": sample_ids(trained_ids, 20),
        }
        if changed_rows == 0:
            raise AssertionError("no trained rows moved above tolerance in %s" % key)

    summary["changed_non_embedding_over_tol"] = changed_non_embedding[:50]
    summary["changed_non_embedding_over_tol_count"] = len(changed_non_embedding)
    summary["shape_mismatches"] = shape_mismatches

    if shape_mismatches:
        raise AssertionError("shape mismatches: %s" % shape_mismatches[:5])
    if changed_non_embedding:
        raise AssertionError("preservation violations: %s" % changed_non_embedding[:5])
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-hf-dir", type=Path, required=True)
    ap.add_argument("--td-hf-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=0.0, help="max allowed diff for preserved tensors/rows")
    ap.add_argument("--trained-change-tol", type=float, default=0.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary = verify(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote: %s" % args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
