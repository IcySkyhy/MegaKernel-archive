#!/usr/bin/env python3
"""Peak process memory per reference tag -- measured, not asserted.

    /workspace/venvs/baseline/bin/python3 tools/measure_peak_rss.py

Round-3 review caught that README §0 row 4 said "实测峰值进程内存 < 6 GiB" while
nothing in the package measured RSS and no evidence file recorded it: the claim
happened to be true (the reviewer measured 5.35 GiB themselves) but it was
unfalsifiable as archived. This is the measurement, and its output is archived to
`evidence/peak_rss.txt`.

Method: spawn `run_reference.py` per tag and read the CHILD's `ru_maxrss` from
`os.wait4` (Linux: KiB). Reading `RUSAGE_CHILDREN` would give a running maximum
over all children so far, which cannot be attributed per tag -- hence `wait4`.

Exit codes (project rule 2026-09-26; `docs/17` §8.3): 0 = measured; 2 = nothing
measured. On 2 the archived `evidence/peak_rss.txt` is deliberately NOT written
(stderr only) so a failed manual run cannot destroy a real reading.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable

TAGS = [
    "--layer 0 --m 1 --warmup 8 --input embed --pending none --tag layer0_decode_m1",
    "--layer 0 --m 1 --warmup 8 --input embed --pending synth --tag layer0_decode_m1_pending",
    "--layer 3 --m 1 --warmup 2100 --input synthetic --pending synth --tag layer3_decode_m1",
    "--layer 0 --m 64 --input embed --pending synth --tag layer0_chunk_m64",
    "--layer 3 --m 64 --input synthetic --pending none --tag layer3_chunk_m64",
    "--layer 3 --m 64 --input synthetic --seed 1 --pending none --tag layer3_chunk_m64_seed1",
]


DISCARD = "reference/.rss_discard"


def run(spec: str) -> tuple[str, float | None]:
    """-> (tag, peak RSS in GiB or None). The dumps go to a throwaway dir: only
    the RSS numbers are the product, and re-running would clobber reference/."""
    tag = spec.split("--tag ")[1].split()[0]
    cmd = [PY, "run_reference.py", *spec.split(), "--out", DISCARD]
    env = dict(os.environ, M39_THREADS=os.environ.get("M39_THREADS", "8"))
    proc = subprocess.Popen(cmd, cwd=HERE, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, rusage = os.wait4(proc.pid, 0)
    if status != 0:
        return tag, None
    return tag, rusage.ru_maxrss / (1024 * 1024)  # KiB -> GiB


def main() -> int:
    out = HERE / "evidence" / "peak_rss.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    load = open("/proc/loadavg").read().split()[:3]
    lines = [
        "# peak process memory per tag (child ru_maxrss via os.wait4)",
        "# measured by tools/measure_peak_rss.py; the weights are read from the",
        "# real checkpoint, so this is what 'a layer fits in memory' must cite",
        f"# M39_THREADS={os.environ.get('M39_THREADS', '8')}  loadavg={' '.join(load)}",
        "#",
        "# THIS FILE IS A MEASUREMENT, NOT A DETERMINISTIC ARTIFACT: the per-tag",
        "# values move by ~0.05 GiB between runs (allocator/load), so -- unlike the",
        "# other evidence files -- it is NOT expected to be byte-identical across",
        "# runs. Valid for the collection above; quote it as a level (~5.4 GiB),",
        "# and quote the min/max below rather than a single tag's number.",
        "",
    ]
    peaks: list[tuple[str, float]] = []
    for spec in TAGS:
        tag, gib = run(spec)
        if gib is None:
            lines.append(f"{tag}: FAILED to run")
            continue
        peaks.append((tag, gib))
        lines.append(f"{tag}: {gib:.2f} GiB")
    import shutil

    shutil.rmtree(HERE / DISCARD, ignore_errors=True)
    if not peaks:
        # A FAILED run must NOT touch the archive: running this by hand in a
        # broken environment would otherwise replace a real reading in
        # evidence/peak_rss.txt with "SKIPPED" (round-4 review nit-3).
        # The diagnosis goes to stderr; the file keeps whatever it had.
        for line in lines:
            print(line, file=sys.stderr)
        print("RESULT: SKIPPED (0 tags measured) -- evidence/peak_rss.txt "
              "left untouched", file=sys.stderr)
        return 2
    worst = max(peaks, key=lambda kv: kv[1])
    lines += [
        "",
        f"max over {len(peaks)}/{len(TAGS)} tags: {worst[1]:.2f} GiB ({worst[0]})",
        f"min: {min(g for _, g in peaks):.2f} GiB",
        f"RESULT: OK ({len(peaks)}/{len(TAGS)} tags measured)",
    ]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
