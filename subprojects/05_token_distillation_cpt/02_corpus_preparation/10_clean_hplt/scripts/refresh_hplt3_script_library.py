#!/usr/bin/env python3
"""Refresh or verify the HPLT 3 script-library checksums.

This helper clones the inspected public HPLT/WDS repos into a temporary
directory and computes fingerprints for the cleaning/filtering scripts listed
in `hplt3_script_library/script_checksums.sha256`. It is metadata-only: it does
not vendor upstream code into this project.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


REPOS = [
    {
        "name": "warc2text-runner",
        "url": "https://github.com/hplt-project/warc2text-runner.git",
        "ref": "0b88ccc8f439d4375d880ff2ca2b385b903c00ff",
        "paths": [
            "three/README.MD",
            "one/run_warc2text.sh",
            "robotstxt/filter_robotstxt.sh",
            "robotstxt/warc2requesturls.sh",
            "three/ia_transfer_content_enc/warcio-warc2text.py",
            "three/langdiff/add_htmllang_tld.sh",
            "three/lid_thresholding/stage2outputs_lid_thresh.ipynb",
        ],
    },
    {
        "name": "monotextor-slurm",
        "url": "https://github.com/hplt-project/monotextor-slurm.git",
        "branch": "v2.0",
        "ref": "63c6d8f422f505c3b8b793793c26868d8e06e1cf",
        "label": "v2.0-63c6d8f422f505c3b8b793793c26868d8e06e1cf",
        "paths": [
            "00.merge-batching.sh",
            "01.merge-text-meta",
            "02.split-lang",
            "scripts/split-lang.py",
            "09.robotstxt",
            "scripts/robots2tsv.py",
            "10.dedup.sh",
            "10.index",
            "10.dedup",
            "20.processing.sh",
            "20.processing",
            "scripts/annotate.py",
            "30.clean.sh",
            "30.clean",
        ],
    },
    {
        "name": "web-docs-scorer",
        "url": "https://github.com/pablop16n/web-docs-scorer.git",
        "ref": "743d13184d704b264556c805512fb252e4f4a2b9",
        "paths": [
            "src/docscorer/docscorer.py",
            "src/docscorer/configuration.py",
            "src/docscorer/cli.py",
            "src/docscorer/scorers/lang_scorer.py",
            "src/docscorer/scorers/repeated_scorer.py",
            "src/docscorer/scorers/url_scorer.py",
            "src/docscorer/scorers/short_segments_score.py",
            "src/docscorer/scorers/long_texts_scorer.py",
            "src/docscorer/scorers/punct_scorer.py",
            "src/docscorer/scorers/numbers_scorer.py",
            "src/docscorer/scorers/singular_chars_scorer.py",
            "src/docscorer/scorers/informativeness_scorer.py",
        ],
    },
]


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("Command failed: %s\n%s" % (" ".join(cmd), proc.stderr.strip()))
    return proc.stdout.strip()


def clone_repo(repo, root):
    dest = root / repo["name"]
    if dest.exists():
        shutil.rmtree(dest)
    cmd = ["git", "clone", "--quiet", "--depth", "1"]
    if repo.get("branch"):
        cmd.extend(["--branch", repo["branch"]])
    cmd.extend([repo["url"], str(dest)])
    run(cmd)
    head = run(["git", "rev-parse", "HEAD"], cwd=dest)
    if head != repo["ref"]:
        raise RuntimeError("%s checked out %s, expected %s" % (repo["name"], head, repo["ref"]))
    return dest


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lines(work_dir):
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    lines = [
        "# Source snapshots refreshed by scripts/refresh_hplt3_script_library.py.",
        "# Format: sha256  repo@ref:path",
        "",
    ]
    for repo in REPOS:
        dest = clone_repo(repo, work_dir)
        ref_label = repo.get("label") or repo["ref"]
        for rel in repo["paths"]:
            path = dest / rel
            if not path.exists():
                raise RuntimeError("Missing expected upstream path: %s:%s" % (repo["name"], rel))
            lines.append("%s  %s@%s:%s" % (sha256(path), repo["name"], ref_label, rel))
        lines.append("")
    return lines


def normalize_checksum_text(text):
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            continue
        lines.append(line.rstrip())
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/hplt_refs_refresh"))
    parser.add_argument("--checksum-file", type=Path, default=Path("hplt3_script_library/script_checksums.sha256"))
    parser.add_argument("--write", action="store_true", help="Rewrite the checksum file instead of only verifying it.")
    args = parser.parse_args()

    lines = build_lines(args.work_dir)
    new_text = "\n".join(lines).rstrip() + "\n"
    if args.write:
        args.checksum_file.write_text(new_text, encoding="utf-8")
        print("wrote", args.checksum_file)
        return 0

    expected = normalize_checksum_text(args.checksum_file.read_text(encoding="utf-8"))
    actual = normalize_checksum_text(new_text)
    if expected != actual:
        print("checksum mismatch for", args.checksum_file, file=sys.stderr)
        return 1
    print("hplt3 script library checksums ok:", len(actual), "files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
