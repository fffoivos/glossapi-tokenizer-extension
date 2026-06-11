#!/usr/bin/env python3
"""Cache and verify native-Greek benchmark sources.

This is intentionally a CPU/network helper. It does not evaluate models and
should not be submitted to GPU nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(__file__).with_name("native_greek_benchmark_registry.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/native_greek_eval_cache"),
        help="Ignored local cache root, relative to repo root unless absolute.",
    )
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument(
        "--include-diagnostic",
        action="store_true",
        help="Also cache explicit MT diagnostic datasets listed outside the native headline registry.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit 0 after writing the manifest even when gated/unavailable datasets are recorded.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                cwd=Path(__file__).resolve().parent,
            ).strip()
        )
    except Exception:
        return Path.cwd()


def _load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()  # nosec - provenance checksum, not security.
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_datasets_import():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Run with e.g.:\n"
            "  uv run --with datasets --with pandas --with pyarrow "
            "subprojects/03_apertus_extension_and_embedding_adaptation/"
            "03_4_implementation_experiments/init_bakeoff/eval/"
            "cache_native_greek_benchmarks.py"
        ) from exc


def _split_candidates(spec: dict[str, Any]) -> list[str | None]:
    split = spec.get("split")
    if split in (None, "default"):
        return [None, "train", "test", "validation"]
    if split == "test_or_train":
        return ["test", "validation", "train", None]
    if split == "test_or_default":
        return ["test", "validation", None, "train"]
    return [split, None]


def _cache_hf_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    _ensure_datasets_import()
    from datasets import get_dataset_config_names, load_dataset

    dataset_id = spec["source"]
    requested_config = spec.get("config")
    config = requested_config
    if not config:
        try:
            configs = get_dataset_config_names(dataset_id)
        except Exception:
            configs = []
        if len(configs) == 1 and configs[0] != "default":
            config = configs[0]

    last_error: str | None = None
    for split in _split_candidates(spec):
        try:
            kwargs: dict[str, Any] = {}
            if split:
                kwargs["split"] = split
            ds = load_dataset(dataset_id, config, **kwargs)
            if isinstance(ds, dict):
                selected_name = "test" if "test" in ds else next(iter(ds))
                selected = ds[selected_name]
                split_used = selected_name
            else:
                selected = ds
                split_used = split or getattr(selected, "split", None) or "default"
            return {
                "status": "cached",
                "source": dataset_id,
                "config": config,
                "split": str(split_used),
                "rows": int(len(selected)),
                "columns": list(selected.column_names),
                "cache_files": [entry.get("filename") for entry in selected.cache_files],
            }
        except Exception as exc:  # gated repos and unusual builders are recorded.
            last_error = repr(exc)
    return {
        "status": "blocked",
        "source": dataset_id,
        "config": config,
        "error": last_error or "unknown load failure",
    }


def _cache_git(spec: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    target = cache_dir / "git" / spec["id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        if target.exists():
            shutil.rmtree(target)
        subprocess.check_call(["git", "clone", spec["source"], str(target)])
    subprocess.check_call(["git", "-C", str(target), "fetch", "--all", "--tags"])
    revision = spec.get("revision")
    if revision:
        subprocess.check_call(["git", "-C", str(target), "checkout", revision])
    commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    return {
        "status": "cached",
        "source": spec["source"],
        "path": str(target),
        "revision": revision,
        "commit": commit,
    }


def _cache_zenodo_zip(spec: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    target_dir = cache_dir / "zenodo" / spec["id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / Path(spec["source"].split("/")[-2]).name
    if not zip_path.exists():
        with urllib.request.urlopen(spec["source"]) as response:
            zip_path.write_bytes(response.read())
    checksum = spec.get("checksum")
    md5 = _md5(zip_path)
    if checksum and checksum.startswith("md5:") and md5 != checksum.split(":", 1)[1]:
        return {
            "status": "blocked",
            "source": spec["source_record"],
            "path": str(zip_path),
            "error": f"md5 mismatch: expected {checksum}, got md5:{md5}",
        }
    extract_dir = target_dir / "extracted"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    files = sorted(str(path.relative_to(extract_dir)) for path in extract_dir.rglob("*") if path.is_file())
    return {
        "status": "cached",
        "source": spec["source_record"],
        "path": str(zip_path),
        "sha256": _sha256(zip_path),
        "md5": md5,
        "extract_dir": str(extract_dir),
        "file_count": len(files),
        "sample_files": files[:20],
    }


def cache_one(spec: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    started = time.time()
    try:
        if spec["source_type"] == "hf_dataset":
            result = _cache_hf_dataset(spec)
        elif spec["source_type"] == "git":
            result = _cache_git(spec, cache_dir)
        elif spec["source_type"] == "zenodo_zip":
            result = _cache_zenodo_zip(spec, cache_dir)
        else:
            result = {"status": "blocked", "error": f"unsupported source_type={spec['source_type']}"}
    except Exception as exc:
        result = {"status": "blocked", "error": repr(exc)}
    result.update(
        {
            "id": spec["id"],
            "name": spec["name"],
            "runner": spec.get("runner"),
            "headline_bucket": spec.get("headline_bucket"),
            "registry_status": spec.get("status"),
            "elapsed_seconds": round(time.time() - started, 2),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    registry = _load_registry(args.registry)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else args.repo_root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in registry["benchmarks"]:
        print(f"[cache] {spec['id']} ({spec['source_type']})", flush=True)
        result = cache_one(spec, cache_dir)
        print(f"  -> {result['status']}", flush=True)
        results.append(result)

    if args.include_diagnostic:
        for item in registry.get("excluded_from_headline_mt_diagnostics", []):
            source = item["source"]
            if "/" not in source:
                continue
            spec = {
                "id": item["id"],
                "name": item["id"],
                "source_type": "hf_dataset",
                "source": source,
                "split": "test_or_train",
                "runner": "diagnostic_only",
                "headline_bucket": "excluded_mt_diagnostic",
                "status": "diagnostic_only",
            }
            print(f"[cache:diagnostic] {source}", flush=True)
            results.append(cache_one(spec, cache_dir))

    manifest = {
        "schema": "native-greek-cache-manifest-v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registry": str(args.registry.resolve()),
        "cache_dir": str(cache_dir),
        "results": results,
    }
    manifest_path = cache_dir / "native_greek_benchmark_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {manifest_path}")

    blocked = [item for item in results if item["status"] != "cached"]
    if blocked:
        print("\nBlocked/unavailable:")
        for item in blocked:
            print(f"- {item['id']}: {item.get('error', item['status'])}")
        if not args.allow_blocked:
            sys.exit(1)


if __name__ == "__main__":
    main()
