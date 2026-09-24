# P32/D128 all-active-opcode micro ablation

## Complete decode-step result

H100, real Llama-3.2-1B weights, batch 1, BF16 storage and FP32
accumulation, prompt 32, output 128. Each row is the median of three
fresh-process means; each process uses three warmups and ten timed
generations. Timing has no timeline, profiler, or device-side timing record.

| Variant | Trial means (us/token) | Median (us/token) | Tokens/s | Delta vs matched VM |
|---|---|---:|---:|---:|
| Persistent VM matched control | 9 controls: 982.921--983.161 | 983.046 | 1017.25 | baseline |
| Original 13-page direct-TK PDL | 1010.835, 1011.015, 1010.966 | 1010.966 | 989.15 | +2.84% |
| All-active micro, store release | 1243.396, 1243.493, 1243.376 | 1243.396 | 804.25 | +26.48% |
| Same micro functions, 128-KiB dynamic SMEM | 1243.304, 1243.234, 1243.504 | 1243.304 | 804.31 | +26.47% |

Micro is 22.99% slower than the original direct chain. Big-SMEM is 0.007%
faster than normal-SMEM micro, i.e. no measurable admission effect in the
safe store-release configuration.

Every decode forward has 81 kernel nodes and 80 programmatic edges. There are
127 separate per-position CUDA Graphs; 128 decode steps are not combined into
one graph.

## What the result does and does not isolate

The original plan preserved the direct chain's early PDL release. That version
became incorrect when reduced resources enabled true adjacent-grid
co-residence. Pairwise tests could pass while the 16-layer chain failed.
Consumer-wide waits, loader waits, and CTA-local gating did not repair it.

The final correct body candidate therefore releases opcodes 1/2/4/5/6 only
after their last output store. Consequently, `micro` versus `original` is not
a pure microtile-compute comparison: it also includes the loss of early
cross-kernel load overlap. The earlier roughly 925-us body plateau came from an
unsafe early-release diagnostic and is not a valid performance result.

The useful conclusion is narrower:

- all-opcode resource reduction makes co-residence feasible;
- under conservative store release, co-residence itself changes complete-step
  latency by approximately zero;
- recovering the persistent-VM gap requires a correct early-release/prefetch
  protocol, not admission alone.

## Correctness

- Final 16-layer small-SMEM body passes graph-versus-eager and
  graph-versus-extracted-TK checks at positions 32, 95, and 158.
- The short complete-path candidate deterministically emits
  `[14924, 198, 59]`.
- The persistent atomic reference is itself nondeterministic in free-running
  generation, including P32/D3 in some runs. D128 token hashes are therefore
  retained as diagnostics rather than used as a bitwise oracle.
- No arithmetic, storage type, accumulation type, or output work was omitted.

## Resources

| Opcode | Operation | Warps/CTA | Registers/thread | Stack | Spills |
|---:|---|---:|---:|---:|---:|
| 1 | RMSNorm + QKV + RoPE/KV append | 20 | 36 | 0 B | 0 B |
| 2 | GQA attention | 8 | 107 | 0 B | 0 B |
| 4 | O projection + residual | 20 | 37 | 0 B | 0 B |
| 5 | RMSNorm + up/gate + SiLU | 20 | 47 | 0 B | 0 B |
| 6 | down projection + residual | 20 | 37 | 0 B | 0 B |
| 7 | RMSNorm + LM head | 20 | 35 | 0 B | 0 B |

All entries use 10,240 bytes static SMEM. Normal micro requests 82,944 bytes
dynamic SMEM, for 93,184 bytes per CTA. The big-SMEM control requests 131,072
bytes dynamic, for 141,312 bytes per CTA.

With H100 register allocation rounded to 256 registers per warp, every
normal-micro adjacent pair in `1 -> 2 -> 4 -> 5 -> 6 -> 1` and `6 -> 7`
fits within 64 warps, 65,536 registers, and 228 KiB SMEM per SM. Two big-SMEM
CTAs require 282,624 bytes and cannot co-reside.

The final six entry functions have no SASS `LDL`/`STL`. Opcode 2 originally
used a 16-byte local pointer array in `store_4_rows`; replacing it with direct
shared-address calculation removed the stack without changing arithmetic.

## Evidence

The files below are in the
[blog-v1.0 evidence release](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/releases/tag/blog-v1.0)
under `all-micro/`:

- Formal JSONs and stdout/stderr: `e2e/final/`
- Formal execution log: `e2e/formal-run.log`
- Fixed-position correctness:
  `correctness/final-nostack-all-store-pos32-158.json`
- Short-path correctness:
  `correctness/final-nostack-all-store-p32d3.json`
- CUDA 12.8 ptxas log: `build/all-micro-final-nostack-cuda128.log`
- Scoped final SASS: `build/all-micro-final-nostack.sass`

The paper and blog were not modified.
