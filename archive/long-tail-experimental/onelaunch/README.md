# onelaunch

A from-scratch **fused decode megakernel** for batch-1 LLM inference. The whole
transformer decode step runs as a handful of Triton kernels captured into **one
CUDA-graph launch**, so the GPU streams weights back-to-back at memory bandwidth
with no launch gaps and no HBM round-trips between ops. Built on an RTX 4090,
checked argmax-identical to a hand-rolled PyTorch oracle at every step, and
compared honestly against vLLM.

Batch-1 decode is weight-bandwidth-bound: every projection is a matrix times a
vector, the tensor cores are idle, and latency is `weight_bytes / bandwidth`.
The only job is to get out of the memory bus's way.

## Results (batch 1, ctx 512, bf16)

**Dense — Qwen2.5-1.5B-Instruct, RTX 4090**

| | ms/token | tok/s |
|---|---|---|
| eager PyTorch | 18.98 | 53 |
| torch.compile + CUDA graph | 5.69 | 176 |
| vLLM 0.11 (SOTA) | 4.41 | 227 |
| **fused megakernel** | **3.72** | **269** |
| memory-bandwidth floor | 3.06 | 330 |

**MoE — OLMoE-1B-7B-0924-Instruct, RTX 4090** (8 of 64 experts per token)

| | ms/token | tok/s |
|---|---|---|
| eager PyTorch | 19.53 | 51 |
| vLLM 0.11 (SOTA) | 3.70 | 270 |
| **fused MoE megakernel** | **3.28** | **305** |
| active-weight floor | 2.34 | 431 |

**Tensor-parallel — dense Qwen, 2× H100 SXM (NVLink)** — a *negative* result,
kept because it's true: splitting the weights halved the per-GPU compute
(2.82 → 2.06 ms) but the per-layer all-reduce added 0.92 ms of latency-bound
collective cost, so TP=2 (2.98 ms) came out no faster than TP=1 (2.82 ms).
Tensor-parallelism buys batch-1 latency only when per-GPU weight streaming
dominates the fixed all-reduce latency — larger models or slower per-GPU
bandwidth. A small model on a fast card is neither.

## Layout

```
src/onelaunch/
  model_ref.py        dense decode oracle (plain PyTorch, verified vs HF)
  kernels.py          Triton: rmsnorm+QKV GEMV, fused RoPE+KV-write,
                      split-K flash-decode attention, swiglu, gemv
  model_fused.py      eager fused decode (readable reference path)
  model_graph.py      CUDA-graph-captured fused decode  <- the dense number
  tune_gemv.py        per-shape GEMV block autotuning micro-bench

  model_ref_moe.py    OLMoE decode oracle (QK-norm, top-8 router, fused experts)
  kernels_moe.py      Triton: QK-norm+RoPE+write, indirect-indexed expert
                      SwiGLU + gated down-combine
  model_graph_moe.py  CUDA-graph-captured fused MoE decode  <- the MoE number

  model_tp.py         Megatron column/row sharding + single-GPU correctness sim
  model_tp_dist.py    distributed TP decode (torchrun + NCCL)

  bench_vllm.py       vLLM SOTA baseline (needs the vllm extra)
results/              recorded ladders (baselines*.json)
```

## Run

```bash
pip install -e .

# dense
python -m onelaunch.model_ref     # oracle: argmax matches Hugging Face
python -m onelaunch.model_graph            # correctness vs oracle
python -m onelaunch.model_graph --bench    # latency ladder

# MoE
python -m onelaunch.model_graph_moe
python -m onelaunch.model_graph_moe --bench

# tensor-parallel (2 GPUs, ideally NVLink) -- see RUN_TP.md
python -m onelaunch.model_tp               # sharding correctness on 1 GPU
torchrun --nproc_per_node=2 -m onelaunch.model_tp_dist --bench

# SOTA baseline
pip install -e '.[vllm]'
python -m onelaunch.bench_vllm --model Qwen/Qwen2.5-1.5B-Instruct
```

## What's borrowed

The megakernel idea — fusing a whole forward pass into one persistent kernel —
is [Mirage/MPK](https://arxiv.org/abs/2506.21335) and
[Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles);
the two-stage decode attention is
[flash-decoding](https://pytorch.org/blog/flash-decoding/); the tensor-parallel
sharding is [Megatron-LM](https://arxiv.org/abs/1909.08053);
[vLLM](https://arxiv.org/abs/2309.06180) is the SOTA baseline and NVIDIA's CUDA
graphs do half the work. What's here is the from-scratch Triton, the
oracle-checked correctness at every step, the indirect-indexed MoE experts, and
a tensor-parallel experiment reported at the value it returned.

Write-up: **One launch per token** (on my blog — the narrative version, with an
interactive latency ladder).
