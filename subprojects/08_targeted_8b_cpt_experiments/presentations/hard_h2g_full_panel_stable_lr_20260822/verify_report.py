#!/usr/bin/env python3
"""Fail closed on evidence, HTML structure, and complete rendered layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "HARD_H2G_FULL_PANEL_AND_STABLE_LR_20260822.html"
EVIDENCE = ROOT / "evidence" / "analysis.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Counter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}
    def handle_starttag(self, tag, attrs) -> None:
        self.counts[tag] = self.counts.get(tag, 0) + 1


def png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()[:24]
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not PNG: {path}")
    return struct.unpack(">II", raw[16:24])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-inspection-passed", action="store_true")
    args = parser.parse_args()
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["panel"] == {"name": "full_public", "n": 16632, "dtype": "float32"}
    assert len(data["decayed_full_panel"]) == 17
    assert len(data["stable_full_panel"]) == len(data["paired"]) == 4
    assert data["optimizer_integrity"]["stable_skipped"] == 0
    assert data["optimizer_integrity"]["stable_nonfinite"] == 0
    assert data["legacy_replication"]["replication_decision"]["replicated"] is False
    assert set(data["validation"]["paired_branch"]) == {
        "hplt", "openarchives", "greek_phd", "english", "de", "ru", "zh", "code", "old_greek"
    }
    raw = REPORT.read_text(encoding="utf-8")
    counter = Counter(); counter.feed(raw)
    assert counter.counts.get("section") == 7
    assert counter.counts.get("svg", 0) >= 4
    assert counter.counts.get("table", 0) >= 4
    for phrase in (
        "Did cooldown create the GreekMMLU peak?",
        "All validation panels, across the complete horizon",
        "The extension 3,219→3,694 remains unauthorized and was not launched",
        "outside the preregistered ±1.0 pp band",
        "Parity scope",
    ):
        assert phrase in raw, phrase
    if not args.visual_inspection_passed:
        raise ValueError("visual inspection must be acknowledged")
    screenshots = {}
    for name, min_width, min_height in (("desktop-1440.png", 1400, 5500), ("narrow-430.png", 430, 6000)):
        path = ROOT / "qa" / name
        width, height = png_size(path)
        assert width >= min_width and height >= min_height
        screenshots[name] = {"width": width, "height": height, "bytes": path.stat().st_size, "sha256": sha256(path)}
    receipt = {
        "schema_version": "apertus_academic_html_report_qa_v1",
        "status": "passed",
        "report": {"path": str(REPORT), "bytes": REPORT.stat().st_size, "sha256": sha256(REPORT)},
        "evidence": {"path": str(EVIDENCE), "bytes": EVIDENCE.stat().st_size, "sha256": sha256(EVIDENCE)},
        "structure": counter.counts,
        "screenshots": screenshots,
        "visual_inspection": {"status": "passed", "method": "full-page Playwright Chromium renders inspected at desktop and narrow widths"},
    }
    output = ROOT / "qa" / "qa_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
