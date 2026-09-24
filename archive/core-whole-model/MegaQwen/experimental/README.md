# Decode Kernel Experiments

This directory is isolated from production decode code.

## Run Variant Benchmarks

```bash
uv run experimental/benchmark_decode_variants.py
```

Optional flags:

```bash
uv run experimental/benchmark_decode_variants.py --tokens 100 --runs 5 --warmup 2 --variants baseline fastmath blocks128 blocks128_fastmath
```

Results are saved to:

- `experimental/results/decode_variant_results.json`
