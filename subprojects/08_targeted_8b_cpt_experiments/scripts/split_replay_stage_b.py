#!/usr/bin/env python3
"""Split the final Stage-B replay stream into foreign and Old-Greek pools."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


OLD_GREEK_SOURCE = "greek_replay_apertus_original"
EXPECTED_SPLITTER_SHA256 = "06f244dd4e0d44f8352af14601768385d77fd35362b3bececda72c01de28f7aa"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--post-greekmmlu-receipt", type=Path, required=True)
    parser.add_argument("--post-native-corpus-receipt", type=Path, required=True)
    parser.add_argument("--post-native-scan-receipt", type=Path, required=True)
    parser.add_argument("--historical-splitter", type=Path, required=True)
    parser.add_argument("--foreign-output", type=Path, required=True)
    parser.add_argument("--old-greek-output", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.foreign_output, args.old_greek_output, args.output_receipt):
        require(not path.exists(), f"immutable replay-split output exists: {path}")
    upstream = read_json(args.input_receipt)
    require(upstream.get("schema_version") == "apertus_hard_h_to_g_stage_b_stream_v1", "replay Stage-B receipt schema drift")
    require(upstream.get("status") == "passed" and upstream.get("stream") == "replay_selected", "replay Stage-B receipt identity drift")
    require(upstream.get("mode") == "apply", "external replay Stage-B was not applied")
    input_binding = file_binding(args.input_jsonl)
    require(upstream["output"]["sha256"] == input_binding["sha256"], "replay Stage-B input binding drift")
    expected_rows = int(upstream["output"]["rows"])

    post_greekmmlu = read_json(args.post_greekmmlu_receipt)
    require(post_greekmmlu.get("schema_version") == "apertus_fresh_greekmmlu_stream_scan_v1", "post-Stage-B GreekMMLU schema drift")
    require(post_greekmmlu.get("status") == "passed" and post_greekmmlu.get("stream") == "replay_selected_post", "post-Stage-B GreekMMLU identity drift")
    require(post_greekmmlu.get("audit_only") is True and int(post_greekmmlu.get("counts", {}).get("item_doc_pairs", -1)) == 0, "post-Stage-B GreekMMLU audit is not clean")
    require(post_greekmmlu.get("input", {}).get("sha256") == input_binding["sha256"], "post-Stage-B GreekMMLU input drift")

    post_corpus = read_json(args.post_native_corpus_receipt)
    require(post_corpus.get("schema_version") == "apertus_replay_training_scan_corpus_receipt_v1", "post-Stage-B native corpus schema drift")
    require(post_corpus.get("status") == "passed", "post-Stage-B native corpus did not pass")
    require(post_corpus.get("input", {}).get("sha256") == input_binding["sha256"], "post-Stage-B native corpus input drift")
    require(post_corpus.get("zero_greekmmlu_receipt") == file_binding(args.post_greekmmlu_receipt), "post-Stage-B native corpus/GreekMMLU binding drift")

    post_native = read_json(args.post_native_scan_receipt)
    require(post_native.get("schema_version") == "apertus_native_suite_training_scan_exclusions_v1", "post-Stage-B native scan schema drift")
    require(post_native.get("status") == "passed", "post-Stage-B native scan did not pass")
    require(post_native.get("corpus_manifest") == post_corpus.get("manifest"), "post-Stage-B native scan corpus drift")
    require(int(post_native.get("counts", {}).get("strong_match_rows", -1)) == 0, "post-Stage-B native scan has strong matches")
    require(int(post_native.get("counts", {}).get("excluded_documents", -1)) == 0, "post-Stage-B native scan has excluded documents")
    require(args.historical_splitter.is_file(), "historical replay splitter missing")
    require(file_binding(args.historical_splitter)["sha256"] == EXPECTED_SPLITTER_SHA256, "historical replay splitter SHA drift")
    args.foreign_output.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    descriptors: list[int] = []
    for output in (args.foreign_output, args.old_greek_output):
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
        descriptors.append(descriptor)
        temporaries.append(Path(name))
    counts: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = {}
    digests = {"foreign": hashlib.sha256(), "old_greek": hashlib.sha256()}
    sizes = Counter()
    try:
        with args.input_jsonl.open(encoding="utf-8") as source, \
                os.fdopen(descriptors[0], "wb") as foreign, os.fdopen(descriptors[1], "wb") as old_greek:
            for line_number, line in enumerate(source, 1):
                require(line.strip(), f"blank replay row at line {line_number}")
                row = json.loads(line)
                text = row.get("text")
                replay_source = row.get("source")
                if not replay_source and isinstance(row.get("metadata"), dict):
                    replay_source = row["metadata"].get("source")
                if not replay_source:
                    replay_source = row.get("source_dataset")
                require(isinstance(text, str) and text, f"invalid replay text at line {line_number}")
                require(isinstance(replay_source, str) and replay_source, f"missing replay source at line {line_number}")
                require(row.get("doc_id") not in (None, ""), f"missing replay doc_id at line {line_number}")
                encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                role = "old_greek" if replay_source == OLD_GREEK_SOURCE else "foreign"
                handle = old_greek if role == "old_greek" else foreign
                handle.write(encoded)
                digests[role].update(encoded)
                sizes[role] += len(encoded)
                counts["input_rows"] += 1
                counts[f"{role}_rows"] += 1
                counts[f"{role}_utf8_text_bytes"] += len(text.encode("utf-8"))
                bucket = per_source.setdefault(replay_source, Counter())
                bucket["rows"] += 1
                bucket["utf8_text_bytes"] += len(text.encode("utf-8"))
            foreign.flush(); os.fsync(foreign.fileno()); old_greek.flush(); os.fsync(old_greek.fileno())
        require(counts["input_rows"] == expected_rows, "replay split row-count drift")
        require(counts["foreign_rows"] > 0 and counts["old_greek_rows"] > 0, "replay split produced an empty pool")
        require(counts["foreign_rows"] + counts["old_greek_rows"] == expected_rows, "replay split accounting drift")
        for temporary, output in zip(temporaries, (args.foreign_output, args.old_greek_output), strict=True):
            os.link(temporary, output)
            temporary.unlink()
    except BaseException:
        for descriptor in descriptors:
            try: os.close(descriptor)
            except OSError: pass
        for temporary in temporaries: temporary.unlink(missing_ok=True)
        raise
    output_bindings = {
        role: {
            "path": str(path.resolve()),
            "bytes": sizes[role],
            "sha256": digests[role].hexdigest(),
            "rows": counts[f"{role}_rows"],
        }
        for role, path in (("foreign", args.foreign_output), ("old_greek", args.old_greek_output))
    }
    payload = {
        "schema_version": "apertus_hard_h_to_g_replay_split_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "input": input_binding,
        "input_receipt": file_binding(args.input_receipt),
        "post_greekmmlu_receipt": file_binding(args.post_greekmmlu_receipt),
        "post_native_corpus_receipt": file_binding(args.post_native_corpus_receipt),
        "post_native_scan_receipt": file_binding(args.post_native_scan_receipt),
        "historical_splitter": file_binding(args.historical_splitter),
        "old_greek_source": OLD_GREEK_SOURCE,
        "counts": dict(counts),
        "per_source": {name: dict(values) for name, values in sorted(per_source.items())},
        "outputs": output_bindings,
        "invariants": {
            "historical_source_partition_rule_reproduced": True,
            "row_order_preserved_within_each_output": True,
            "row_multiplicity_preserved": True,
            "non_text_lineage_fields_preserved": True,
            "additional_deduplication": False,
            "post_stage_b_greekmmlu_zero_matches": True,
            "post_stage_b_native_suite_zero_matches": True,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
