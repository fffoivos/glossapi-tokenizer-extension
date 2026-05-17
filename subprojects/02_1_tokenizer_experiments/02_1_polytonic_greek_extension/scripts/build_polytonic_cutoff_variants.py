#!/usr/bin/env python3
"""Build 256-aligned cutoff variants from a full polytonic continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_merge(merge):
    if isinstance(merge, str):
        parts = merge.split(" ", 1)
        return parts if len(parts) == 2 else None
    if isinstance(merge, list) and len(merge) == 2:
        return merge
    return None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer-dir", type=Path, required=True, help="C3 curated padded tokenizer directory")
    parser.add_argument("--full-tokenizer-dir", type=Path, required=True, help="full 5,120-token continuation tokenizer directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=512)
    parser.add_argument("--max-added", type=int, default=5120)
    parser.add_argument("--variant-prefix", default="c3p_poly_added")
    args = parser.parse_args()

    base_tok = load_json(args.base_tokenizer_dir / "tokenizer.json")
    full_tok = load_json(args.full_tokenizer_dir / "tokenizer.json")
    base_vocab = base_tok["model"]["vocab"]
    full_vocab = full_tok["model"]["vocab"]
    base_merges = base_tok["model"]["merges"]
    full_merges = full_tok["model"]["merges"]
    base_size = len(base_vocab)
    full_size = len(full_vocab)
    if full_size < base_size + args.max_added:
        raise SystemExit(f"full tokenizer too small: {full_size} < {base_size + args.max_added}")
    if full_merges[: len(base_merges)] != base_merges:
        raise SystemExit("full tokenizer does not preserve base merge prefix")
    base_by_id = {v: k for k, v in base_vocab.items()}
    full_by_id = {v: k for k, v in full_vocab.items()}
    for idx in range(base_size):
        if base_by_id[idx] != full_by_id.get(idx):
            raise SystemExit(f"base vocab mismatch at id {idx}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for added in range(0, args.max_added + 1, args.step):
        final_size = base_size + added
        if final_size % 256 != 0:
            raise SystemExit(f"unaligned cutoff {added}: final vocab {final_size}")
        name = f"{args.variant_prefix}_{added:04d}"
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if added == 0:
            tokenizer_json = base_tok
            added_merges = []
        else:
            added_merges = full_merges[len(base_merges) : len(base_merges) + added]
            new_vocab = dict(base_vocab)
            next_id = base_size
            for merge in added_merges:
                parts = split_merge(merge)
                if parts is None:
                    raise SystemExit(f"bad merge format at cutoff {added}: {merge!r}")
                result = parts[0] + parts[1]
                new_vocab[result] = next_id
                next_id += 1
            if len(new_vocab) != final_size:
                raise SystemExit(f"vocab size mismatch for {name}: {len(new_vocab)} != {final_size}")
            tokenizer_json = json.loads(json.dumps(full_tok))
            tokenizer_json["model"]["vocab"] = new_vocab
            tokenizer_json["model"]["merges"] = base_merges + added_merges
        (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for sibling in args.base_tokenizer_dir.iterdir():
            if sibling.name == "tokenizer.json" or not sibling.is_file():
                continue
            shutil.copy2(sibling, out_dir / sibling.name)
        tok_sha = sha256_path(out_dir / "tokenizer.json")
        manifest = {
            "variant_id": name,
            "base_tokenizer_dir": str(args.base_tokenizer_dir),
            "full_tokenizer_dir": str(args.full_tokenizer_dir),
            "base_vocab_size": base_size,
            "polytonic_added_count": added,
            "final_vocab_size": final_size,
            "alignment_128": final_size % 128 == 0,
            "alignment_256": final_size % 256 == 0,
            "base_tokenizer_sha256": sha256_path(args.base_tokenizer_dir / "tokenizer.json"),
            "full_tokenizer_sha256": sha256_path(args.full_tokenizer_dir / "tokenizer.json"),
            "tokenizer_sha256": tok_sha,
            "added_merge_count": len(added_merges),
            "variant_dir": str(out_dir),
        }
        write_json(out_dir / "manifest.json", manifest)
        variants.append(manifest)
    write_json(args.output_dir / "variants_manifest.json", {"variants": variants})
    print(json.dumps({"output_dir": str(args.output_dir), "variants": len(variants)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
