"""Merge the eng_Latn_fineweb_hq.npy single-source rerun into the canonical
histogram_matrix.npz as a new row, keeping the existing eng_Latn (wiki-only)
row untouched.

After this:
  H has shape (1934, 131072) with the new row appended.
  canonical_keys gets "eng_Latn_fineweb_hq" appended.
  lang_metadata.json gets a new entry pointing at the new run.
  outputs/histogram_matrix.npz is rewritten atomically (tmp → mv).

The original eng_Latn row stays at index 1275 (wiki-only) for backward compat.
Re-running this script is idempotent — it skips if "eng_Latn_fineweb_hq" is
already present and the row matches the .npy.

Usage:
  python3 merge_eng_fineweb_hq.py \
      --npy /path/to/eng_Latn_fineweb_hq.npy \
      --summary /path/to/eng_Latn_fineweb_hq.summary.json
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
NEW_KEY = "eng_Latn_fineweb_hq"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, type=Path,
                    help="Pulled-from-worker eng_Latn_fineweb_hq.npy.")
    ap.add_argument("--summary", required=True, type=Path,
                    help="Pulled-from-worker eng_Latn_fineweb_hq.summary.json.")
    args = ap.parse_args()

    new_row = np.load(args.npy).astype(np.int64)
    summary = json.loads(args.summary.read_text())
    assert new_row.shape == (131072,), f"unexpected new_row shape {new_row.shape}"
    assert new_row.sum() > 0, "new_row is all zeros — refusing to merge"

    npz_path = OUTPUTS / "histogram_matrix.npz"
    z = np.load(npz_path, allow_pickle=True)
    H = z["H"]
    keys = list(z["canonical_keys"])

    if NEW_KEY in keys:
        idx = keys.index(NEW_KEY)
        if np.array_equal(H[idx], new_row):
            print(f"[merge] {NEW_KEY} already present and identical — nothing to do")
            return
        # Overwrite the existing row in place (rare; happens if we re-run the worker)
        H_new = H.copy()
        H_new[idx] = new_row
        new_keys = keys
        print(f"[merge] {NEW_KEY} already present, updating row {idx}")
    else:
        H_new = np.vstack([H, new_row.reshape(1, -1)])
        new_keys = keys + [NEW_KEY]
        print(f"[merge] appending {NEW_KEY} at row {len(keys)}")

    # Atomic write
    with tempfile.NamedTemporaryFile(dir=OUTPUTS, delete=False, suffix=".npz") as tmp:
        np.savez(tmp.name, H=H_new, canonical_keys=np.array(new_keys, dtype=object))
        tmp_path = Path(tmp.name)
    shutil.move(tmp_path, npz_path)
    print(f"[merge] wrote {npz_path}  shape={H_new.shape}  keys={len(new_keys)}")

    # Update lang_metadata.json
    meta_path = OUTPUTS / "lang_metadata.json"
    meta = json.loads(meta_path.read_text())
    src_used = summary.get("sources_used", [])
    new_entry = {
        "name": "<single-source: en from epfml/FineWeb-HQ>",
        "family": meta.get("eng_Latn", {}).get("family", "<unknown>"),
        "iso_639_3": "eng",
        "script_iso15924": "Latn",
        "row_index": len(new_keys) - 1 if NEW_KEY not in keys else keys.index(NEW_KEY),
        "sample_tokens_total": int(new_row.sum()),
        "vocab_entries_fired": int((new_row > 0).sum()),
        "vocab_entries_fired_geq_10":  int((new_row >= 10).sum()),
        "vocab_entries_fired_geq_100": int((new_row >= 100).sum()),
        "wall_seconds": summary.get("wall_seconds", None),
        "sources_in_map": ["fineweb_hq"],
        "sources_contributed": src_used,
    }
    meta[NEW_KEY] = new_entry
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[merge] updated {meta_path} with {NEW_KEY} entry "
          f"(sample={new_entry['sample_tokens_total']:,}, "
          f"fired={new_entry['vocab_entries_fired']:,})")


if __name__ == "__main__":
    main()
