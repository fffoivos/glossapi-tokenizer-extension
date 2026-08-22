#!/usr/bin/env python3
"""Fail-closed evidence and render checks for the trajectory report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
REPORT = ROOT / "GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()[:24]
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not PNG: {path}")
    return struct.unpack(">II", raw[16:24])


class Counter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        self.counts[tag] = self.counts.get(tag, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-inspection-passed", action="store_true")
    args = parser.parse_args()
    aggregate = load(EVIDENCE / "trajectory_aggregate.json")
    analysis = load(EVIDENCE / "analysis_summary.json")
    parity = load(EVIDENCE / "export_parity_audit.json")
    html = REPORT.read_text(encoding="utf-8")

    assert aggregate["status"] == "completed"
    assert aggregate["panel"] == {"name": "decontaminated_full_clean", "n": 16159}
    assert len(aggregate["rows"]) == 34
    assert len(aggregate["updates"]) == 17
    assert {row["scale"] for row in aggregate["rows"]} == {"1p5b", "8b"}
    assert all(math.isfinite(float(row[key])) for row in aggregate["rows"]
               for key in ("accuracy", "choice_nll", "correct_answer_bpb"))
    assert analysis["status"] == "completed"
    assert parity["status"] == "completed"
    assert parity["receipt_count"] == parity["exact_weight_mapping_pass_count"] == 34
    assert parity["frozen_evaluator_ready_count"] + parity["trajectory_only_count"] == 34

    counter = Counter()
    counter.feed(html)
    assert counter.counts.get("section") == 9
    assert counter.counts.get("svg", 0) >= 8
    assert counter.counts.get("table", 0) >= 4
    for required in (
        "The 1.5B run does not mirror the 8B run",
        "16,159-question decontaminated GreekMMLU panel",
        "Every measured checkpoint, side by side",
        "Educational strata move differently",
        "Forgetting is real",
    ):
        assert required in html, required

    screenshots = {}
    for name, minimum_width in (("desktop-1440.png", 1200), ("narrow-430.png", 400)):
        path = ROOT / "qa" / name
        width, height = png_dimensions(path)
        assert width >= minimum_width and height >= 5000
        screenshots[name] = {
            "width": width,
            "height": height,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    if not args.visual_inspection_passed:
        raise ValueError("visual inspection must be explicitly acknowledged")

    receipt = {
        "schema_version": "apertus_academic_html_report_qa_v1",
        "status": "passed",
        "report": {"path": str(REPORT), "bytes": REPORT.stat().st_size, "sha256": sha256(REPORT)},
        "evidence": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name in ("trajectory_aggregate.json", "analysis_summary.json", "export_parity_audit.json")
            for path in [EVIDENCE / name]
        },
        "structure": counter.counts,
        "screenshots": screenshots,
        "visual_inspection": {
            "status": "passed",
            "method": "full-page Playwright Chromium renders inspected at desktop and narrow widths",
            "in_app_browser_available": False,
        },
    }
    output = ROOT / "qa" / "qa_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
