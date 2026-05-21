#!/usr/bin/env python3
"""Patch Apertus-specific tensors into converted Megatron checkpoints.

The public Apertus release is Hugging Face format, while the CSCS training path
uses Megatron checkpoints. Our HF -> Megatron loader can preserve the standard
transformer tensors through saver_core, but saver_core has no protocol slots for
Apertus extras:

  - xIELU learned activation parameters: alpha_p / alpha_n
  - QK-Norm weights: q_norm / k_norm

Current Megatron checkpoints do not serialize xIELU beta/eps; this script
checks the HF source values are still the Megatron defaults before accepting
that absence.

The default mode writes to --out-dir by hardlink-copying the checkpoint tree and
then atomically replacing patched rank files. That keeps the original converted
checkpoint intact for rollback and comparison.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class PatchSpec:
    hf_key: str
    megatron_key: str
    required_in_megatron: bool = True
    default_if_missing: Optional[float] = None
    missing_tolerance: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, type=Path, help="HF Apertus checkpoint directory to copy extras from.")
    parser.add_argument("--megatron-dir", required=True, type=Path, help="Converted Megatron checkpoint root or release dir.")
    parser.add_argument("--out-dir", type=Path, help="Write a patched copy here. Mutually exclusive with --in-place.")
    parser.add_argument("--in-place", action="store_true", help="Patch --megatron-dir directly.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --out-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Report diffs without writing patched checkpoints.")
    parser.add_argument("--no-qk-norm", action="store_true", help="Patch only xIELU tensors, not QK-Norm.")
    parser.add_argument("--max-current-diff", type=float, default=0.0, help="Fail if all current diffs are already <= this value.")
    return parser.parse_args()


def read_config(hf_dir: Path) -> dict:
    path = hf_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing HF config: {path}")
    return json.loads(path.read_text())


def read_weight_map(hf_dir: Path) -> Dict[str, str]:
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = hf_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(f"found neither model.safetensors.index.json nor model.safetensors in {hf_dir}")


class HFTensorReader:
    def __init__(self, hf_dir: Path):
        self.hf_dir = hf_dir
        self.weight_map = read_weight_map(hf_dir)

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"missing HF tensor: {key}")
        shard = self.hf_dir / self.weight_map[key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).detach().cpu()


def release_dir(root_or_release: Path) -> Path:
    if (root_or_release / "release").is_dir():
        return root_or_release / "release"
    if root_or_release.name == "release" and root_or_release.is_dir():
        return root_or_release
    raise FileNotFoundError(f"could not find release checkpoint under {root_or_release}")


def checkpoint_root(root_or_release: Path) -> Path:
    return root_or_release.parent if root_or_release.name == "release" else root_or_release


def rank_files(root_or_release: Path) -> List[Path]:
    release = release_dir(root_or_release)
    files = sorted(release.glob("mp_rank_*/model_optim_rng.pt"))
    if not files:
        raise FileNotFoundError(f"no mp_rank_*/model_optim_rng.pt files under {release}")
    return files


def link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_output_tree(source: Path, out_dir: Path, overwrite: bool) -> Path:
    source_root = checkpoint_root(source)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"--out-dir already exists: {out_dir}; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, out_dir, copy_function=link_or_copy)
    return out_dir


def build_specs(num_layers: int, include_qk_norm: bool) -> List[PatchSpec]:
    specs: List[PatchSpec] = []
    for layer in range(num_layers):
        specs.extend(
            [
                PatchSpec(
                    hf_key=f"model.layers.{layer}.mlp.act_fn.alpha_p",
                    megatron_key=f"decoder.layers.{layer}.mlp.activation_func.alpha_p",
                ),
                PatchSpec(
                    hf_key=f"model.layers.{layer}.mlp.act_fn.alpha_n",
                    megatron_key=f"decoder.layers.{layer}.mlp.activation_func.alpha_n",
                ),
                PatchSpec(
                    hf_key=f"model.layers.{layer}.mlp.act_fn.beta",
                    megatron_key=f"decoder.layers.{layer}.mlp.activation_func.beta",
                    required_in_megatron=False,
                    default_if_missing=0.5,
                    missing_tolerance=0.0,
                ),
                PatchSpec(
                    hf_key=f"model.layers.{layer}.mlp.act_fn.eps",
                    megatron_key=f"decoder.layers.{layer}.mlp.activation_func.eps",
                    required_in_megatron=False,
                    default_if_missing=-1.0e-6,
                    missing_tolerance=1.0e-6,
                ),
            ]
        )
        if include_qk_norm:
            specs.extend(
                [
                    PatchSpec(
                        hf_key=f"model.layers.{layer}.self_attn.q_norm.weight",
                        megatron_key=f"decoder.layers.{layer}.self_attention.q_layernorm.weight",
                    ),
                    PatchSpec(
                        hf_key=f"model.layers.{layer}.self_attn.k_norm.weight",
                        megatron_key=f"decoder.layers.{layer}.self_attention.k_layernorm.weight",
                    ),
                ]
            )
    return specs


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def validate_missing_default(spec: PatchSpec, hf_tensor: torch.Tensor) -> None:
    if spec.default_if_missing is None:
        return
    actual = hf_tensor.float()
    expected = torch.full_like(actual, float(spec.default_if_missing), dtype=torch.float32)
    diff = max_abs_diff(actual, expected)
    if diff > spec.missing_tolerance:
        raise AssertionError(
            f"Megatron checkpoint lacks {spec.megatron_key}, but HF {spec.hf_key} "
            f"is not default {spec.default_if_missing} (max abs diff {diff})"
        )


def patch_rank_file(
    path: Path,
    specs: List[PatchSpec],
    hf: HFTensorReader,
    dry_run: bool,
) -> Dict[str, Union[float, int]]:
    print(f"[patch_apertus_extras] loading {path}", flush=True)
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if "model" not in ckpt or not isinstance(ckpt["model"], dict):
        raise KeyError(f"{path} does not look like a Megatron checkpoint with a model state dict")
    model = ckpt["model"]

    patched = 0
    skipped_missing_defaults = 0
    current_max_diff = 0.0
    patched_max_diff = 0.0

    for spec in specs:
        hf_tensor = hf.get(spec.hf_key)
        if spec.megatron_key not in model:
            if spec.required_in_megatron:
                raise KeyError(f"missing Megatron tensor {spec.megatron_key} in {path}")
            validate_missing_default(spec, hf_tensor)
            skipped_missing_defaults += 1
            continue

        target = model[spec.megatron_key]
        replacement = hf_tensor.to(dtype=target.dtype).contiguous()
        before = max_abs_diff(target, replacement)
        current_max_diff = max(current_max_diff, before)
        if not dry_run:
            model[spec.megatron_key] = replacement
        patched += 1
        patched_max_diff = max(patched_max_diff, max_abs_diff(model[spec.megatron_key], replacement))

    if not dry_run:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"[patch_apertus_extras] wrote {path}", flush=True)

    return {
        "patched": patched,
        "skipped_missing_defaults": skipped_missing_defaults,
        "current_max_diff": current_max_diff,
        "patched_max_diff": patched_max_diff,
    }


def main() -> int:
    args = parse_args()
    if bool(args.out_dir) == bool(args.in_place):
        raise SystemExit("choose exactly one of --out-dir or --in-place")

    config = read_config(args.hf_dir)
    num_layers = int(config["num_hidden_layers"])
    specs = build_specs(num_layers=num_layers, include_qk_norm=not args.no_qk_norm)
    hf = HFTensorReader(args.hf_dir)

    target_root = args.megatron_dir
    if args.out_dir is not None and not args.dry_run:
        target_root = prepare_output_tree(args.megatron_dir, args.out_dir, args.overwrite)
    elif args.out_dir is not None and args.dry_run:
        target_root = args.megatron_dir

    files = rank_files(target_root)
    print("[patch_apertus_extras] HF dir:       ", args.hf_dir)
    print("[patch_apertus_extras] Megatron dir: ", args.megatron_dir)
    print("[patch_apertus_extras] Target dir:   ", target_root)
    print("[patch_apertus_extras] layers:       ", num_layers)
    print("[patch_apertus_extras] rank files:   ", len(files))
    print("[patch_apertus_extras] specs:        ", len(specs))
    print("[patch_apertus_extras] dry_run:      ", args.dry_run)

    totals = {
        "patched": 0,
        "skipped_missing_defaults": 0,
        "current_max_diff": 0.0,
        "patched_max_diff": 0.0,
    }
    for path in files:
        stats = patch_rank_file(path, specs, hf, dry_run=args.dry_run)
        totals["patched"] += int(stats["patched"])
        totals["skipped_missing_defaults"] += int(stats["skipped_missing_defaults"])
        totals["current_max_diff"] = max(float(totals["current_max_diff"]), float(stats["current_max_diff"]))
        totals["patched_max_diff"] = max(float(totals["patched_max_diff"]), float(stats["patched_max_diff"]))
        print(f"[patch_apertus_extras] stats {path.parent.name}: {stats}", flush=True)

    print("[patch_apertus_extras] TOTAL", totals)
    if args.max_current_diff and float(totals["current_max_diff"]) <= args.max_current_diff:
        raise AssertionError(
            f"current max diff {totals['current_max_diff']} <= --max-current-diff {args.max_current_diff}; "
            "nothing appears to need patching"
        )
    if not args.dry_run and float(totals["patched_max_diff"]) != 0.0:
        raise AssertionError(f"post-patch max diff is nonzero: {totals['patched_max_diff']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
