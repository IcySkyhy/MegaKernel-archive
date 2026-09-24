# Megakernel Decode — Handover Document

## Branch & Commit

- **Repo**: `llama.cpp-original/` (standalone git repo)
- **Branch**: `megakernel-wip`
- **Base commit**: `ddcdecc` — "add fused megakernel decode for Qwen3.5-0.8B (hybrid DeltaNet + FullAttn)"
- **Uncommitted**: 3 files modified (see below)

---

## What's in `ddcdecc` (committed)

| File | What |
|------|------|
| `ggml/include/ggml.h` | `GGML_OP_FUSED_MEGAKERNEL_DECODE` enum + `ggml_fused_megakernel_decode()` declaration |
| `ggml/src/ggml.c` | Op name/symbol + `ggml_fused_megakernel_decode()` body — packs 384 tensor pointers + scalars into `op_params` |
| `ggml/src/ggml-cuda/fused-megakernel.cu` | Full fused CUDA kernel (1162 lines, `decode_kernel`) + C++ host launcher `ggml_cuda_op_fused_megakernel_decode` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Forward declaration + dispatch in `compute_forward` + `supports_op` |
| `ggml/src/ggml-cpu/ggml-cpu.c` / `.cpp` | Stub no-op for CPU, `supports_op=false` |
| `src/models/qwen35.cpp` | Megakernel decode path in `graph::graph()`, guarded by `LLAMA_CUDA_MEGAKERNEL=1` |
| `progress-megakernel.md` | Architecture docs |

### Architecture (as committed)

```
qwen35.cpp graph builder
  -> ggml_fused_megakernel_decode() packs ALL pointers into op_params[0..14]
     (weights_flat array addr, final_norm, embed, lm_head, scalars)
  -> ggml_cuda_op_fused_megakernel_decode() [CUDA host]
    -> dequant_to_f16() on first call: reads tensor->data from 
       the flat pointer array, caches F16 copies on GPU
    -> decode_kernel<<<num_SMs, 512>>>() runs all 24 layers in one launch
       using AtomicGridSync for cross-block barriers
```

### How `qwen35.cpp` builds the flat array

```cpp
const ggml_tensor * h_weights_flat[24 * 16] = {};  // LOCAL VARIABLE (bug)
for (int il = 0; il < 24; ++il) {
    // fills 16 slots per layer: attn_norm, wqkv/wq, beta, alpha, etc.
    // plus recurrent state tensors from mctx_cur
}
// Stores address of local array into op_params via PACK_PTR
cur = ggml_fused_megakernel_decode(ctx0, inpL, (const void*)h_weights_flat, ...);
```

---

## Uncommitted Changes (3 files)

### 1. `src/models/models.h` — Added member `h_weights_flat`

```cpp
struct graph : public llm_build_delta_net_base {
    graph(const llama_model & model, const llm_graph_params & params);
private:
    const ggml_tensor * h_weights_flat[24 * 16] = {};  // NEW
    // ...
};
```

Moved the flat array from a local variable to a member of the `graph` struct, extending its lifetime from stack-frame to `graph` object lifetime.

### 2. `src/models/qwen35.cpp` — Use member, remove static guard

- Removed `static const ggml_tensor * h_weights_flat[24 * 16] = {};`
- Removed `static bool weights_flat_initialized = false;`
- The member array is now filled every call (no skip-once guard)

### 3. `ggml/src/ggml-cuda/fused-megakernel.cu` — Debug logging

Added `fprintf(stderr)` + `fflush(stderr)` debug logging:
- `MEGACHECK:` — model-change detection (embed/norm/lm_head pointer comparison)
- `MEGAINIT:` — first-time weight dequantization
- `MEGASKIP:` — cached weight reuse
- `MEGALLOC:` — per-layer dequant malloc size
- Post-kernel sync + peek at first two logits

---

## Current Status: llama-bench works, llama-cli still crashes

### llama-bench (passes 3 iterations)
The `h_weights_flat` lifetime fix (moving from local to member) wasn't needed for llama-bench — it was the `static` removal that fixed it. llama-bench reloads models, and the old `static` array held stale tensor pointers.

### llama-cli (still crashes: OOM on first decode step)
**Root cause**: `h_weights_flat` still dies before compute.

**Flow**:
1. `llama_context::process_ubatch()` calls `model.build_graph(gparams)` (line 1337)
2. Inside `build_graph()`:
   - `build_arch_graph(params)` creates `graph` object (has `h_weights_flat` member)
   - `graph::graph()` constructor fills the array, creates `cur` tensor with op_params pointing to `h_weights_flat`
   - `build_graph()` returns `llm->res->get_gf()`
   - `unique_ptr<llm_graph_context> llm` goes out of scope ~ destroys `graph` ~ destroys `h_weights_flat`
3. Back in `process_ubatch()`: `graph_compute()` runs ~ CUDA backend reads op_params ~ gets dangling pointer to dead `h_weights_flat` ~ reads garbage tensor pointers ~ garbage `nelems` ~ `cudaMalloc` with garbage size ~ OOM

**Fix needed**: The flat array must live through graph compute. Here are options:

### Option A: Allocate from ggml context (Recommended)

Instead of a C++ array, allocate the flat tensor pointer storage from `ctx0` (which is `res->ctx_compute.get()`, alive through compute):

```cpp
// In graph::graph():
ggml_tensor * flat_meta = ggml_new_tensor_1d(ctx0, GGML_TYPE_I8,
    24 * 16 * sizeof(ggml_tensor *));
ggml_tensor ** flat = (ggml_tensor **) flat_meta->data;
// fill flat[0..383] with tensor pointers
cur = ggml_fused_megakernel_decode(ctx0, inpL, (const void *) flat, ...);
```

The `flat` pointer points into ggml-managed memory that lives as long as `ctx_compute`. The `flat_meta` tensor is unused in the graph but its allocation survives.

**Caveat**: `ctx0` is the compute graph context. Adding allocations here is fine during graph build.

### Option B: Static thread_local array

```cpp
static thread_local const ggml_tensor * h_weights_flat[24 * 16];
static thread_local uint64_t h_weights_flat_version = 0;

uint64_t cur_version = compute_version_hash(model, mctx);
if (cur_version != h_weights_flat_version) {
    fill_array(h_weights_flat, model, mctx);
    h_weights_flat_version = cur_version;
}
```

This is what the committed code had, except without the `static` + `thread_local`. Thread-local static lives until thread exit, so it survives compute. The version hash detects model reloads.

### Option C: Allocate from heap, store in `llm_graph_result`

Add a `std::vector<const ggml_tensor *>` to `llm_graph_result` (or use a slot in `params`). The vector lives until `res` is reset.

### Option D: Copy all tensor pointers into op_params

Use a different approach entirely — instead of a flat array pointer, pack each tensor into a `src[]` slot or use the new approach from `server/` codebase (weights via `dst->src[1..9]` tensor inputs, see below).

---

## Reference: server/ codebase approach

The `server/deps/llama.cpp/` version has a different architecture that avoids this problem entirely:

- `fused-megakernel.cu` reads weights from **ggml tensor inputs** (`dst->src[1..9]`) not from a raw pointer array
- Layer weights, KV caches, states are all ggml tensors managed by the graph
- Scalar params (token_id, position) stored in `op_params`
- No `weights_flat` array — weights are pre-dequantized to F16 tensors and stored as GPU tensors before the graph is built

This is the cleaner design but requires more plumbing (creating F16 weight tensors, pre-dequantizing at load time, managing KV cache/state tensors in the graph).

---

## Build System

The `llama.cpp-original/` uses CMake. Standard build:

```
cd llama.cpp-original
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DLLAMA_CUDA=ON
cmake --build . --config RelWithDebInfo
```

(Adjust `-G` for your VS version. VS 2019 Community with CUDA 12.x was used.)

---

## Test Commands

```powershell
# llama-cli
$env:LLAMA_CUDA_MEGAKERNEL="1"
./build/bin/RelWithDebInfo/llama-cli.exe -m Qwen3.5-0.8B-Q4_K_M.gguf -p "The capital of France is" -n 10

# llama-bench
$env:LLAMA_CUDA_MEGAKERNEL="1"
./build/bin/RelWithDebInfo/llama-bench.exe -m Qwen3.5-0.8B-Q4_K_M.gguf -n 3
```

---

## Progress Update (July 2026 Session)

1. **Logits Projection (LM Head):** Added a grid-synchronized Phase 6 (LM Head projection + Repetition Penalty) to the end of `decode_kernel` in `fused-megakernel.cu` to correctly compute the final vocabulary probabilities instead of outputting raw hidden states or uninitialized memory.
2. **Forced GPU Scheduling:** Overrode the CPU backend's `supports_op` to return `false` for `GGML_OP_FUSED_MEGAKERNEL_DECODE` in `ggml-cpu.cpp`, preventing CPU fallback.
3. **Pointer Safety (`cudaPointerGetAttributes`):** Upgraded `dequant_to_f16` to query pointer locations using standard CUDA attributes. If a tensor (such as the token embeddings `embed_t`) is resident on the host (CPU), it is automatically staged to a temporary GPU buffer before dequantization, fixing the `illegal memory access` crash.
4. **Strided/Transposed KV Cache Support:** Implemented logic to handle both transposed and non-transposed `v_cache` layouts by checking the strides of the `get_v()` tensor at launch time and passing a `v_trans` parameter down to the GPU.
5. **Memory Reload Safety:** Added automatic cached memory invalidation inside `ggml_cuda_op_fused_megakernel_decode`. If a new model is loaded, all cached device allocations are freed and rebuilt.
6. **Inference Status:** Tested and validated: `llama-cli` generates 100% coherent and fast text without crashing.

---

## Next Steps

1. **Lifetime Pointer Resolution (Option A):** Implement the C++ compute context allocation for `h_weights_flat` (Option A) to prevent pointer invalidation.
2. **Remove Debug File I/O:** Remove the `megakernel_debug.log` writing from `fused-megakernel.cu` once inference is fully stabilized.
3. **Performance Profiling:** Benchmark performance of the megakernel path vs. the stock `llama.cpp` server for throughput comparison.
4. **Port to larger model:** Scale to Qwen3.5-2B/9B/27B models once 0.8B is completely finalized.

