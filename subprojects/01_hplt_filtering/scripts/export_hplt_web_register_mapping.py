#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hplt_web_register import ALL_SUBLABELS, LEGACY_EXTRA_SUBLABELS, MAIN_LABELS, SIMPLIFIED_SUBLABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HPLT web-register label mappings in JSON and Markdown.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported artifacts.")
    return parser.parse_args()


def write_markdown(output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# HPLT Web-Register Label Map")
    lines.append("")
    lines.append("This map keeps the labels in full words rather than code abbreviations.")
    lines.append("")
    lines.append("Important note: the public HPLT schema is mixed.")
    lines.append("- The current Turku abbreviations page exposes a simplified two-level taxonomy.")
    lines.append("- HPLT's own dataset schema also carries older CORE fine labels.")
    lines.append("- That means labels like `nb` and `dtp` coexist with more specific older labels such as `pb`, `tb`, `dp`, and `dt`.")
    lines.append("")
    lines.append("## Main Registers")
    lines.append("")
    lines.append("| Code | Full label |")
    lines.append("| --- | --- |")
    for code, label in MAIN_LABELS.items():
        lines.append(f"| `{code}` | {label} |")

    lines.append("")
    lines.append("## Simplified Lower-Level Labels")
    lines.append("")
    lines.append("| Code | Full label | Parent |")
    lines.append("| --- | --- | --- |")
    for code, meta in SIMPLIFIED_SUBLABELS.items():
        parent = MAIN_LABELS[str(meta["parent"])]
        lines.append(f"| `{code}` | {meta['label']} | {parent} |")

    lines.append("")
    lines.append("## Additional Legacy Fine Labels Present In HPLT Schema")
    lines.append("")
    lines.append("| Code | Full label | Parent |")
    lines.append("| --- | --- | --- |")
    for code, meta in LEGACY_EXTRA_SUBLABELS.items():
        parent = MAIN_LABELS[str(meta["parent"])]
        lines.append(f"| `{code}` | {meta['label']} | {parent} |")

    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("- HPLT 3.0 README schema: https://huggingface.co/datasets/HPLT/HPLT3.0/blob/main/README.md")
    lines.append("- Turku abbreviations page: https://turkunlp.org/register-annotation-docs/abbreviations")
    lines.append("- CORE Table 1: https://link.springer.com/article/10.1007/s10579-022-09624-1/tables/1")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "hplt_web_register_mapping.json"
    md_path = args.output_dir / "hplt_web_register_mapping.md"

    payload = {
        "main_labels": MAIN_LABELS,
        "simplified_sublabels": SIMPLIFIED_SUBLABELS,
        "legacy_extra_sublabels": LEGACY_EXTRA_SUBLABELS,
        "all_sublabels": ALL_SUBLABELS,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path)
    print(json.dumps({"mapping_json": str(json_path), "mapping_md": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

