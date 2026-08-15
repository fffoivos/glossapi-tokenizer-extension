#!/usr/bin/env python3
"""Patch the pinned bakeoff launcher for CSCS uenv v10 Slurm mounting."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = 'uenv run "$UENV_IMAGE" --view=default -- \\' + "\n" + '            srun '
NEW = 'srun --uenv="$UENV_IMAGE" --view=default '


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(NEW) == 2 and OLD not in text:
        print("uenv v10 srun patch already installed")
        return
    observed = text.count(OLD)
    if observed != 2:
        raise SystemExit(f"expected exactly two uenv-run/srun launch anchors, found {observed}")
    patched = text.replace(OLD, NEW)
    if patched.count(NEW) != 2 or OLD in patched:
        raise SystemExit("uenv v10 srun patch postcondition failed")
    path.write_text(patched, encoding="utf-8")
    print("installed uenv v10 srun patch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()
    patch(args.trainer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
