# Megakernel Decode: Progress Log

## Overview

Single fused CUDA kernel for Qwen3.5-0.8B hybrid model (24 layers: 18 DeltaNet + 6 Full Attention). Skips per-operator launch overhead by running all layers in one kernel, keeping weights in registers/shared memory.

## Files Changed

### New file
- **`ggml/src/ggml-cuda/fused-megakernel.cu`** (~49KB) — The fused kernel and C++ launcher

### Modified files
- **`ggml/include/ggml.h`** — Added `GGML_OP_FUSED_MEGAKERNEL_DECODE` op enum + `ggml_fused_megakernel_decode()` API
- **`ggml/src/ggml.c`** — Op name/symbol registration + `ggml_fused_megakernel_decode()` implementation (packs 64-bit pointers into op_params)
- **`ggml/src/ggml-cuda/ggml-cuda.cu`** — Forward declaration + dispatch in `ggml_cuda_compute_forward()` + `supports_op`
- **`ggml/src/ggml-cpu/ggml-cpu.c`** — Stub no-op in `ggml_compute_forward()` + `n_tasks=1`
- **`ggml/src/ggml-cpu/ggml-cpu.cpp`** — `supports_op` returns `false`
- **`src/models/qwen35.cpp`** — Megakernel path guarded by `LLAMA_CUDA_MEGAKERNEL=1` env var; builds flat tensor pointer array per layer

## Architecture

```
llama_cli / llama_bench
  -> qwen35.cpp graph (env LLAMA_CUDA_MEGAKERNEL=1 triggers fast path)
    -> ggml_fused_megakernel_decode() packs all ptrs into op_params
      -> ggml_cuda_op_fused_megakernel_decode() [C++ host]
        -> dequant_to_f16() [host-to-device dequantization of all model weights on first call]
        -> decode_kernel<<<num_SMs, 512>>>() [single fused CUDA kernel]
          - embed lookup -> 24 layers (DeltaNet or FullAttn) -> final norm -> lm_head
```

## Kernel Structure (decode_kernel)

Single grid of SM-count blocks (68 on RTX 3080 Laptop GPU), 512 threads each.

**Layers loop (24 iterations):**
- DeltaNet (type 0): RMSNorm -> QKV proj + Z/Beta/Alpha proj -> Conv1d -> Recurrence (gated linear attention) -> Out proj + residual -> Post-norm -> Gate+Up SiLU MLP -> Down proj + residual
- Full Attention (type 1): RMSNorm -> Q/K/V proj -> QK Norm + RoPE -> Online softmax attention with KV cache -> O proj + residual -> Post-norm -> Gate+Up SiLU MLP -> Down proj + residual

**Final:** RMSNorm -> LM head projection (split across blocks) + repetition penalty

**Synchronization:** Custom `AtomicGridSync` using global memory atomics for cross-block barriers between phases.

## Current Status

### Working
- Op plumbing (ggml.h, ggml.c, ggml-cuda.cu, ggml-cpu.c, qwen35.cpp)
- Model loads, kernel launches without CUDA errors
- Correct inference on `llama-cli` — produces valid output (e.g. "The capital of France is Paris")
- Host pointer support in `dequant_to_f16` — handles CPU-resident tensors by copying to GPU

### Known Issues
1. **llama-bench crashes** — Fixed in latest commit. Static weight pointer caches (`d_embed_weight`, `d_layer_weights`, etc.) persisted across model reloads but pointed to stale GPU memory. Now detecting model tensor pointer changes and invalidating all caches.
2. **Memory leak on model reload** — Old `cudaMalloc`ed weight buffers are not freed when model changes (minor, only affects `llama-bench`-style repeated loading).
3. **V_TRANS (v_cache transposed layout)** — Full attention supports both transposed and contiguous V layouts; probed via tensor stride comparison in `qwen35.cpp`.

### Notes
- Hardcoded for Qwen3.5-0.8B (hidden=1024, layers=24, vocab=248320).
- DeltaNet recurrence uses persistent float32 state matrix (heads x key_dim x value_dim).
- KV cache is updated in-place for each position.
- Grid sync uses two global memory ints (counter + generation), reset before each kernel launch.
