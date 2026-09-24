"""Position-dependent throughput benchmark for the optimized megakernel."""
import sys
import time

import torch

sys.path.insert(0, "../02_megakernel")
from model import Decoder


def main():
    print("Loading megakernel...")
    decoder = Decoder()

    print("\nWarming up...")
    decoder.generate("Hello", max_tokens=10)

    print("\nPosition-dependent throughput:")
    print(f"{'Tokens':>8}  {'tok/s':>8}  {'ms/tok':>8}")
    print("-" * 30)

    for n_tokens in [10, 50, 100, 200]:
        times = []
        for _ in range(3):
            decoder.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            decoder.generate("Hello", max_tokens=n_tokens)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times)
        tps = n_tokens / avg
        ms = avg * 1000 / n_tokens
        print(f"{n_tokens:>8}  {tps:>8.1f}  {ms:>8.2f}")

    weight_mb = 1192
    bw_gbs = 936
    theoretical_ms = weight_mb / bw_gbs
    theoretical_tps = 1000 / theoretical_ms
    print(f"\nTheoretical max: {theoretical_tps:.0f} tok/s ({theoretical_ms:.2f} ms/tok)")
    print("(RTX 3090: 936 GB/s, 1192 MB per step)")


if __name__ == "__main__":
    main()
