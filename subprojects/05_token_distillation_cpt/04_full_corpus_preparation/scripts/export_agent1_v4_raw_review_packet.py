#!/usr/bin/env python3
"""Export the Agent 1 v4 18x20 raw-document review packet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from agent1_v4_raw_review import PacketBlockedError, materialize_raw_review_packet


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / "configs" / "sources.json")
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument(
        "--roster", type=Path, default=root / "configs" / "agent1_v3_candidate_roster.json"
    )
    parser.add_argument(
        "--policy", type=Path, default=root / "configs" / "agent1_v4_raw_review_policy.json"
    )
    parser.add_argument("--seed", required=True, help="32-byte lowercase hexadecimal seed")
    parser.add_argument(
        "--prompt", type=Path, default=root / "configs" / "agent1_v4_terra_review_prompt.md"
    )
    parser.add_argument(
        "--response-schema",
        type=Path,
        default=root / "schemas" / "agent1_v4_terra_review_response.schema.json",
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        materialize_raw_review_packet(
            sources_path=args.sources,
            acquisition_receipt=args.acquisition_receipt,
            roster_path=args.roster,
            policy_path=args.policy,
            seed_hex=args.seed,
            prompt_path=args.prompt,
            response_schema_path=args.response_schema,
            code_commit=args.code_commit,
            output=args.output,
        )
    except PacketBlockedError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
