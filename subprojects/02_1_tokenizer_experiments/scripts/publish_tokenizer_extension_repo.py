#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer


@dataclass(frozen=True)
class ArmSpec:
    category: str
    name: str
    run_dir: Path


def parse_named_path(value: str, *, category: str) -> ArmSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH for {category} arm, got: {value}")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError(f"Missing arm name in: {value}")
    run_dir = Path(raw_path).expanduser().resolve()
    return ArmSpec(category=category, name=name, run_dir=run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package and publish the Apertus tokenizer extension comparison arms to a Hugging Face model repo.")
    parser.add_argument("--fresh-run", action="append", default=[], help="Fresh tokenizer arm as NAME=PATH.")
    parser.add_argument("--continuous-run", action="append", default=[], help="Continuous tokenizer arm as NAME=PATH.")
    parser.add_argument("--repo-id", default=None, help="Optional explicit Hugging Face repo id (owner/name).")
    parser.add_argument("--repo-slug", default="apertus-tokenizer-extension", help="Repo name when repo id is inferred from whoami().")
    parser.add_argument("--staging-dir", type=Path, required=True, help="Temporary staging directory to build the upload tree into.")
    parser.add_argument("--private", action="store_true", help="Create/upload as a private model repo.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and verify the staging tree without uploading.")
    return parser


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def require_files(run_dir: Path, names: list[str]) -> None:
    missing = [name for name in names if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {run_dir}: {missing}")


def copy_tree_contents(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_dir.iterdir()):
        dest = dest_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest)


def synthesize_special_tokens_map(summary: dict, arm_dir: Path) -> None:
    target = arm_dir / "special_tokens_map.json"
    if target.exists():
        return
    mapping = summary.get("special_tokens_map")
    if mapping:
        write_json(target, mapping)


def verify_loadable(arm_dir: Path, *, expected_vocab_size: int | None = None) -> dict:
    tok = AutoTokenizer.from_pretrained(arm_dir, use_fast=True, trust_remote_code=False)
    loaded_length = len(tok)
    model_vocab_size = int(tok.vocab_size)
    if expected_vocab_size is not None and model_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer at {arm_dir} loaded with model vocab size {model_vocab_size}, expected {expected_vocab_size}"
        )
    return {
        "model_vocab_size": model_vocab_size,
        "loaded_length": loaded_length,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "is_fast": bool(getattr(tok, "is_fast", False)),
    }


def stage_fresh_arm(spec: ArmSpec, stage_root: Path) -> dict:
    require_files(spec.run_dir, ["tokenizer.json", "tokenizer_config.json", "training_summary.json"])
    arm_dir = stage_root / "fresh" / spec.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "training_summary.json"]:
        shutil.copy2(spec.run_dir / name, arm_dir / name)
    summary = load_json(arm_dir / "training_summary.json")
    synthesize_special_tokens_map(summary, arm_dir)
    verification = verify_loadable(arm_dir, expected_vocab_size=int(summary["vocab_size_actual"]))
    return {
        "category": spec.category,
        "name": spec.name,
        "source_run_dir": str(spec.run_dir),
        "publish_dir": str(arm_dir.relative_to(stage_root)),
        "training_summary": {
            "vocab_size_requested": summary.get("vocab_size_requested"),
            "vocab_size_actual": summary.get("vocab_size_actual"),
            "runtime_seconds": summary.get("runtime_seconds"),
            "base_tokenizer": summary.get("base_tokenizer"),
        },
        "verification": verification,
    }


def stage_continuous_arm(spec: ArmSpec, stage_root: Path) -> dict:
    require_files(spec.run_dir, ["training_summary.json", "replication_check.json", "front_end_contract_check.json"])
    tokenizer_dir = spec.run_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory in {spec.run_dir}")
    arm_dir = stage_root / "continuous" / spec.name
    copy_tree_contents(tokenizer_dir, arm_dir)
    for name in ["training_summary.json", "replication_check.json", "front_end_contract_check.json"]:
        shutil.copy2(spec.run_dir / name, arm_dir / name)
    summary = load_json(arm_dir / "training_summary.json")
    verification = verify_loadable(arm_dir, expected_vocab_size=int(summary["target_vocab_size"]))
    return {
        "category": spec.category,
        "name": spec.name,
        "source_run_dir": str(spec.run_dir),
        "publish_dir": str(arm_dir.relative_to(stage_root)),
        "training_summary": {
            "target_vocab_size": summary.get("target_vocab_size"),
            "base_vocab_size": summary.get("base_vocab_size"),
            "added_token_count": summary.get("added_token_count"),
            "added_merge_count": summary.get("added_merge_count"),
            "runtime_seconds": summary.get("runtime_seconds"),
            "reference_tokenizer": summary.get("reference_tokenizer"),
        },
        "verification": verification,
    }


def build_readme(manifest: dict) -> str:
    fresh = manifest["arms"]["fresh"]
    continuous = manifest["arms"]["continuous"]
    lines = [
        "---",
        "language:",
        "- el",
        "tags:",
        "- tokenizer",
        "- bpe",
        "- greek",
        "- apertus",
        "---",
        "",
        "# Apertus Tokenizer Extension",
        "",
        "This repo packages the raw tokenizer arms used for the Apertus extension comparison work.",
        "",
        "Layout:",
        "- `fresh/*`: fresh-discovery tokenizers trained from the two corpus mixes",
        "- `continuous/*`: true append-only continuous-BPE extensions starting from Apertus",
        "",
        "Included arms:",
    ]
    for record in fresh + continuous:
        lines.append(f"- `{record['publish_dir']}`")
    lines.extend(
        [
            "",
            "Each arm directory is directly loadable with `transformers.AutoTokenizer.from_pretrained(...)`.",
            "Continuous arms also include `replication_check.json` and `front_end_contract_check.json`.",
            "",
            "The comparable merged cutoff grid is handled separately; this repo contains the four raw tokenizer arms only.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_remote_files(api: HfApi, repo_id: str, manifest: dict) -> None:
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    required = {"README.md", "manifest.json"}
    for record in manifest["arms"]["fresh"] + manifest["arms"]["continuous"]:
        publish_dir = record["publish_dir"]
        required.add(f"{publish_dir}/tokenizer.json")
        required.add(f"{publish_dir}/tokenizer_config.json")
        required.add(f"{publish_dir}/training_summary.json")
        if record["category"] == "fresh":
            required.add(f"{publish_dir}/special_tokens_map.json")
        else:
            required.add(f"{publish_dir}/replication_check.json")
            required.add(f"{publish_dir}/front_end_contract_check.json")
    missing = sorted(required - files)
    if missing:
        raise RuntimeError(f"Uploaded repo is missing expected files: {missing}")


def infer_repo_id(api: HfApi, repo_id: str | None, repo_slug: str) -> str:
    if repo_id:
        return repo_id
    who = api.whoami()
    owner = who.get("name") or who.get("user")
    if not owner:
        raise RuntimeError(f"Could not infer Hugging Face username from whoami(): {who}")
    return f"{owner}/{repo_slug}"


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args.fresh_run = [parse_named_path(value, category="fresh") for value in args.fresh_run]
    args.continuous_run = [parse_named_path(value, category="continuous") for value in args.continuous_run]
    if not args.fresh_run or not args.continuous_run:
        parser.error("At least one --fresh-run and one --continuous-run are required.")
    return args


def main() -> None:
    args = parse_args()
    stage_root = args.staging_dir.expanduser().resolve()
    reset_dir(stage_root)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_slug": args.repo_slug,
        "arms": {"fresh": [], "continuous": []},
    }

    for spec in args.fresh_run:
        manifest["arms"]["fresh"].append(stage_fresh_arm(spec, stage_root))
    for spec in args.continuous_run:
        manifest["arms"]["continuous"].append(stage_continuous_arm(spec, stage_root))

    write_json(stage_root / "manifest.json", manifest)
    (stage_root / "README.md").write_text(build_readme(manifest) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "stage_root": str(stage_root), "manifest": manifest}, ensure_ascii=False, indent=2))
        return

    api = HfApi()
    repo_id = infer_repo_id(api, args.repo_id, args.repo_slug)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_root),
        commit_message="Upload fresh and continuous Apertus tokenizer extension arms",
    )
    verify_remote_files(api, repo_id, manifest)
    result = {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "stage_root": str(stage_root),
        "manifest": manifest,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
