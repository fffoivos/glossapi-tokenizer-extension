#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from freeze_experiment_contract import validate_static


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paths = [
        ROOT / "configs/experiment_a_recipe.json",
        ROOT / "configs/experiment_b_recipe.json",
        ROOT / "configs/allocation_plan.json",
    ]
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validate_static(*values)
    print(json.dumps({"ok": True, "validated": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
