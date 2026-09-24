"""Run one decode-kernel variant benchmark in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

# Make sure local imports resolve from repo root when invoked anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental.variant_sources import build_variant_source  # noqa: E402


def _bench_variant(
    variant: str,
    tokens: int,
    warmup: int,
    runs: int,
    prompt: str,
) -> dict[str, float | int | str]:
    """Compile one variant and benchmark decode-only token throughput."""
    # Reduce noisy HF logs in this subprocess.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # Import after env setup.
    if str(REPO_ROOT / "csrc" / "megakernel") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "csrc" / "megakernel"))
    import megakernel_decode as mk  # noqa: E402

    # Patch source provider so existing compile path uses our variant source.
    variant_src = build_variant_source(variant)
    original_get_cuda_source = mk._get_cuda_source
    original_load_inline = mk.load_inline

    def _patched_get_cuda_source(filename: str) -> str:
        if filename == "fused_decode_ldg.cu":
            return variant_src
        return original_get_cuda_source(filename)

    safe_variant = "".join(ch if ch.isalnum() else "_" for ch in variant)

    def _patched_load_inline(*args, **kwargs):
        # Keep each variant in its own extension module to avoid cache aliasing.
        kw = dict(kwargs)
        base_name = kw.get("name", "megakernel_decode")
        kw["name"] = f"{base_name}_{safe_variant}"
        return original_load_inline(*args, **kw)

    mk._get_cuda_source = _patched_get_cuda_source
    mk.load_inline = _patched_load_inline
    mk._decode_kernel = None

    gen = mk.MegakernelGenerator(max_seq_len=2048)
    input_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)

    def run_once() -> None:
        gen.decoder.reset()
        for tid in input_ids[:-1]:
            gen.decoder.decode_step(tid)
        tok = input_ids[-1]
        for _ in range(tokens):
            tok = gen.decoder.decode_step(tok)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    durations: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    mean_s = statistics.mean(durations)
    min_s = min(durations)
    max_s = max(durations)

    return {
        "variant": variant,
        "tokens": tokens,
        "warmup": warmup,
        "runs": runs,
        "mean_s": mean_s,
        "min_s": min_s,
        "max_s": max_s,
        "tok_s": tokens / mean_s,
        "ms_tok": (mean_s * 1000.0) / tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    result = _bench_variant(
        variant=args.variant,
        tokens=args.tokens,
        warmup=args.warmup,
        runs=args.runs,
        prompt=args.prompt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
