// int4 head GEMV — draft token head reading the int4-packed lm_head (128MB) instead of the bf16 embedding
// (513MB), ~4x less memory. The draft's head_tok is memory-bound on the vocab weight, so this ~4x's it.
// xn [DIM] bf16 (already rmsnorm'd) -> logits [VOCAB] f32 via the same contiguous int4 layout as the megakernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

__global__ void head_gemv(const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ scales,
                          const __nv_bfloat16* __restrict__ xn, float* __restrict__ logits,
                          int DIM, int VOCAB, float inv_ls) {
  const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + wib;
  if (row >= VOCAB) return;
  const int ncols = DIM >> 3;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * ncols;
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (DIM >> 7);
  float acc = 0.f;
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col]; const float sc = __bfloat162float(srow[col >> 4]);
    const int4 xr = __ldg(reinterpret_cast<const int4*>(&xn[col << 3]));   // 8 bf16 = one 128-bit load
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
    #pragma unroll
    for (int nib = 0; nib < 8; nib++) acc += (float((w >> (4 * nib)) & 0xF) - 8.0f) * sc * __bfloat162float(xb[nib]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) logits[row] = acc * inv_ls;
}

torch::Tensor head_launch(torch::Tensor Wq, torch::Tensor scales, torch::Tensor xn, double logits_scaling) {
  const int VOCAB = Wq.size(0), DIM = Wq.size(1) * 8;
  auto logits = torch::empty({VOCAB}, xn.options().dtype(torch::kFloat32));
  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr<at::BFloat16>());
  float* lg = logits.data_ptr<float>();
  const int nthreads = 256, wpb = nthreads >> 5, grid = (VOCAB + wpb - 1) / wpb;
  head_gemv<<<grid, nthreads, 0, at::cuda::getCurrentCUDAStream()>>>(wq, sc, x, lg, DIM, VOCAB, 1.0f / logits_scaling);
  return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("head", &head_launch, "int4 head GEMV -> logits [VOCAB]"); }
