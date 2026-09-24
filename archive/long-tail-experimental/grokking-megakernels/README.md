# Grokking Megakernels - Companion Code

Educational code for the book *Grokking Megakernels*. This repo walks through building a CUDA megakernel for Qwen3-0.6B inference, from separate kernels to a fully fused forward pass.

This is a teaching repo. It is composed of distilled, annotated code from multiple sources, restructured to match the book's chapter progression. The original implementations live here:

- **MegaQwen** (Elliot Arledge, RTX 3090): https://github.com/Infatoshi/megaqwen
- **Qwen Megakernel** (AlpinDale, RTX 5090): https://github.com/AlpinDale/qwen_megakernel
- **Hazy Research Megakernels** (Stanford, H100): https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

## Structure

```
01_separate_kernels/      Chapters 1-2: The problem
  rmsnorm.cu                Standalone RMSNorm kernel
  matvec.cu                 Standalone matrix-vector kernel
  benchmark_launches.py     Measure kernel launch overhead

02_megakernel/            Chapters 3-4: The solution
  config.cuh                Model constants (Qwen3-0.6B)
  rmsnorm.cuh               Fused RMSNorm
  matvec.cuh                Fused matrix-vector with __ldg
  rope.cuh                  Fused RoPE
  attention.cuh             Online softmax attention
  megakernel.cu             Full cooperative megakernel
  model.py                  Weight loading + decode API
  generate.py               Text generation
  verify.py                 Correctness vs HuggingFace
  benchmark.py              Throughput measurement

03_optimized/             Chapters 5-6: Optimization
  megakernel_optimized.cu   Block divergence + L2 prefetch
  benchmark.py              Position-dependent throughput

references/               Chapters 7-10: Further reading
  BLACKWELL.md              AlpinDale's RTX 5090 work
  HAZY_RESEARCH.md          Instruction-interpreter model
```

## Requirements

- NVIDIA GPU with compute capability 8.6+ (RTX 3090, A100, etc.)
- CUDA 12+
- Python 3.10+

## Setup

```bash
git clone https://github.com/Infatoshi/grokking-megakernels.git
cd grokking-megakernels
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers
```

## Quick Start

Step 1: See why separate kernels are slow.

```bash
cd 01_separate_kernels
python benchmark_launches.py
```

Step 2: Run the fused megakernel.

```bash
cd 02_megakernel
python generate.py "The capital of France is"
```

Step 3: Verify correctness against HuggingFace.

```bash
python verify.py
```

Step 4: Benchmark the optimized version.

```bash
cd 03_optimized
python benchmark.py
```

## Credits

- Elliot Arledge: MegaQwen cooperative megakernel (RTX 3090)
- AlpinDale: Persistent kernel port (RTX 5090)
- Benjamin Spector, Jordan Juravsky, Stuart Sul, et al. (Hazy Research): Instruction-interpreter megakernels (H100)
