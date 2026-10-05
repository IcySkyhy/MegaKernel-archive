#!/usr/bin/env python3
"""Build a synthetic "kernel dump" that ZEROS the columns the official kernel
never wrote, to demonstrate the `--mask-policy` knob of `compare_dumps.py`.

    python3 tools/zero_unscored.py --src reference/layer3_chunk_m64 \
        --dst /tmp/dump_zeroed_unscored --segments qsa.index_logits

Why this exists: `qsa.index_logits` is a [rows, seq_len//4] fp32 buffer, but the
official indexer only writes `column < visible_blocks`
(vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:101-107). The reference dump marks
the rest with `-inf`; a real NPU kernel is at least as likely to leave 0 there.
Under the default `--mask-policy strict` that scores as FAIL, which is why
`--mask-policy ignore-unscored` exists and is documented in the report itself.
This tool manufactures exactly that case so `tools/make_evidence.sh` can show
strict=FAIL / ignore-unscored=PASS with the same numbers a reviewer can reproduce.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="reference dump directory")
    ap.add_argument("--dst", required=True, help="output dump directory")
    ap.add_argument("--segments", default="qsa.index_logits",
                    help="comma-separated segments to zero-out")
    ap.add_argument("--value", type=float, default=0.0,
                    help="value written where the reference is non-finite")
    ap.add_argument("--also-perturb", default="", help="extra segments to nudge by 1 ulp")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    targets = {s.strip() for s in args.segments.split(",") if s.strip()}
    extra = {s.strip() for s in args.also_perturb.split(",") if s.strip()}

    man = json.loads((src / "manifest.json").read_text())
    for name, entry in sorted(man["segments"].items()):
        arr = np.load(src / entry["file"])
        if name in targets and arr.dtype in (np.float32, np.float16):
            arr = np.where(np.isfinite(arr), arr, np.asarray(args.value, arr.dtype))
        if name in extra and arr.dtype == np.uint16:  # bf16 bit patterns
            arr = arr.copy()
            arr[0, 0] = np.uint16((int(arr[0, 0]) + 1) & 0xFFFF)
        np.save(dst / entry["file"], arr)
    man["extra"] = dict(man.get("extra", {}))
    man["extra"]["synthetic"] = f"unscored columns zeroed in {sorted(targets)}"
    (dst / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[zero_unscored] {src} -> {dst}: zeroed non-finite in {sorted(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
