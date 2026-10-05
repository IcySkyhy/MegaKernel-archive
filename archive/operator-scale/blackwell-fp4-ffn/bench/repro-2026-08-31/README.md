# Re-measurement, 2026-08-31

Raw stdout from a clean rebuild of the benchmarks behind this repo's headline numbers,
with the controls and checks that were run beside them. Nothing here is edited except
two log headers that named an unrelated internal document.

| file | what it is |
| --- | --- |
| `chain_dense_vs_sparse_3090lock.log` | dense vs 2:4-sparse residual double GEMM, 4096 square, three interleaved same-session rounds at a 3090 MHz lock |
| `fused_chain_4096.log` | the fused single-kernel chain at `d=8` and `d=2`, interleaved against dense and sparse in the same session |
| `ffn_block_2700_and_3090.log` | the FP4 FFN block at 2700 and 3090 MHz locks, 10 and 300 iterations, three runs each |
| `ffn_no_clock_lock.log` | the same FFN block with no clock lock at the stock 300 W cap |
| `contention.log` | A/B/A showing what one competing GPU process does to these timings |
| `sparse_layout_probe.log` | host-only check of the analytic sm120 sparse LayoutA/LayoutE byte formulas against the real cute layout objects, aligned and padded shapes |

`fused_chain_4096.log` names two claims in its header, the 862 d=8 row and the 873
deep-C row. Its rounds run the deep-C epilogue (`sC=3 sD=1 C=16b`) and read 857.0 to
862.3 at 4096 square, so the 873 is not reproduced here and no round in this folder
reports it. The 878 fp4-residual row needs `-DC_FP4` and no such round was run.

Each log records the driver version, the clock lock, the sustained SM clock, power draw
and temperature alongside the result, because at this level none of the numbers mean
anything without them.

Toolchain: nvcc 13.1.115 front end, CUTLASS 4.4.2 at `4ca61d06` with
`cute_scaledbasis_eq.patch` applied, and a 13.3 ptxas spliced in via `--dryrun` for the
sparse and fused paths (recipe in the top-level README).
