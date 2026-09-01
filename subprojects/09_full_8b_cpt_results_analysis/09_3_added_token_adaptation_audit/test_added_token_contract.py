#!/usr/bin/env python3
"""Regression tests for the 09.3 added-token adaptation result payload.

These lock the load-bearing claims in RESULTS.md so they cannot drift silently.
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class AddedTokenContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(
            (ROOT / "presentations" / "ADDED_TOKEN_ADAPTATION.data.json").read_text(
                encoding="utf-8"
            )
        )

    def test_schema_and_scope(self) -> None:
        self.assertEqual(self.payload["schema_version"], "added_token_adaptation_v1")
        meta = self.payload["meta"]
        self.assertEqual(meta["vocab_size"], 148_992)
        self.assertEqual(meta["added_modern"] + meta["added_polytonic"], 17_920)
        self.assertEqual([c["update"] for c in meta["checkpoints"]], [400, 9_536, 18_284])
        self.assertEqual(meta["probe_layers"]["token_distillation_layer"], 11)

    def test_alignment_was_exact(self) -> None:
        # a non-zero unaligned count would invalidate every merged-vs-split delta
        self.assertEqual(self.payload["meta"]["unaligned_occurrences"], 0)

    def test_adaptation_is_monotone_across_the_trajectory(self) -> None:
        for group in ("modern", "polytonic"):
            rows = self.payload["trajectory"][group]
            deltas = [r["delta_logp"]["p50"] for r in rows]
            with self.subTest(group=group):
                self.assertEqual(deltas, sorted(deltas))
                echo = [r["echo_top1_rate"] for r in rows]
                self.assertEqual(echo, sorted(echo))

    def test_no_added_token_is_net_negative_from_the_peak_onward(self) -> None:
        for group in ("modern", "polytonic"):
            for row in self.payload["trajectory"][group]:
                if row["update"] >= 9_536:
                    with self.subTest(group=group, update=row["update"]):
                        self.assertEqual(row["frac_tokens_nonpositive"], 0.0)

    def test_terminal_beats_peak_per_token(self) -> None:
        # the decoupling claim: GreekMMLU peaks at 9,536 but tokens keep improving
        for group in ("modern", "polytonic"):
            block = self.payload["peak_to_terminal"][group]
            with self.subTest(group=group):
                self.assertGreater(block["frac_improved"], 0.5)
                self.assertGreater(block["mean_change_nats"], 0.0)

    def test_late_layer_declines_while_distillation_layer_holds(self) -> None:
        rows = self.payload["trajectory"]["modern"]
        l11 = [r["hidden_cos_layer11"]["p50"] for r in rows]
        l30 = [r["hidden_cos_layer30"]["p50"] for r in rows]
        l30_p5 = [r["hidden_cos_layer30"]["p5"] for r in rows]
        self.assertLess(abs(l11[-1] - l11[0]), 0.02)   # layer 11 holds
        self.assertLess(l30[-1], l30[0])               # layer 30 declines
        self.assertLess(l30_p5[-1], l30_p5[0])         # and its tail declines faster
        self.assertGreater(l30[0] - l30[-1], 0.0)
        self.assertGreater(l30_p5[0] - l30_p5[-1], l30[0] - l30[-1])

    def test_unmeasurable_tokens_are_accounted_for(self) -> None:
        block = self.payload["unmeasurable_tokens"]
        self.assertEqual(
            block["absent_from_corpus"] + block["present_but_single_base_piece"],
            block["zero_scored_occurrences"],
        )

    def test_raw_payload_pointers_are_hash_bound(self) -> None:
        pointers = self.payload["raw_payload_pointers"]
        self.assertEqual(len(pointers), 3)
        for pointer in pointers:
            with self.subTest(name=pointer["name"]):
                self.assertTrue(SHA_RE.fullmatch(pointer["sha256"]))
                self.assertGreater(pointer["bytes"], 1_000_000)
                self.assertTrue(pointer["cscs_path"].startswith("/iopsstor/"))

    def test_manifest_digests_match_committed_files(self) -> None:
        manifest = json.loads(
            (ROOT / "evidence" / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8")
        )
        for artifact in manifest["artifacts"]:
            path = ROOT / artifact["path"]
            with self.subTest(path=artifact["path"]):
                self.assertTrue(path.is_file(), path)
                self.assertEqual(path.stat().st_size, artifact["bytes"])
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                self.assertEqual(digest.hexdigest(), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
