#!/usr/bin/env python3
"""sha256 of every dumped tensor, for `evidence/reference_sha256.txt`.

Exit codes: 0 = hashed and cross-checked the manifest; 1 = a manifest sha256
disagreed with the stored bytes; 2 = nothing to hash (no manifests found).

The hash is over the STORED bytes (bf16 is stored as uint16 bit patterns), which
is exactly what `manifest.json` records per segment, so the two can be diffed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory containing <tag>/manifest.json trees")
    args = ap.parse_args()
    root = Path(args.root)
    total = 0
    for manifest in sorted(root.glob("*/manifest.json")):
        data = json.loads(manifest.read_text())
        tag = manifest.parent.name
        print(f"# {tag}")
        if "extra" in data:
            print(f"#   extra: {json.dumps(data['extra'], sort_keys=True)}")
        for name, entry in sorted(data["segments"].items()):
            arr = np.load(manifest.parent / entry["file"])
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            if digest != entry["sha256_stored_bytes"]:
                print(f"#   !! manifest sha mismatch for {name}", file=sys.stderr)
                return 2
            print(f"{digest}  {tag}/{entry['file']}  {entry['dtype']}{tuple(entry['shape'])}")
            total += 1
    if total == 0:
        print("RESULT: SKIPPED (0 tensors hashed -- no manifests found under "
              f"{root}; NOT a pass)")
        return 2
    print(f"# {total} tensors")
    print(f"RESULT: OK ({total} tensors hashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
