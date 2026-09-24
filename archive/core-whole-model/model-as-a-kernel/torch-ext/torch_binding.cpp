#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include <vector>

#include "../mak_cuda/mak_kernels.h"
#include "registration.h"
#include "torch_binding.h"

namespace {

void check_prog(const at::Tensor& prog) {
  TORCH_CHECK(prog.is_cuda(), "prog must be CUDA");
  TORCH_CHECK(prog.scalar_type() == at::kLong, "prog must be int64");
  TORCH_CHECK(prog.dim() == 2 && prog.size(1) == 16 && prog.is_contiguous(),
              "prog must be a contiguous [n_phases, 16] tensor");
}

void check_barrier(const at::Tensor& barrier) {
  TORCH_CHECK(barrier.is_cuda() && barrier.scalar_type() == at::kInt &&
                  barrier.numel() >= 34,
              "barrier must be a CUDA int32 tensor with >= 2 elements");
}

struct LaunchShape {
  int smem;
  int ring_stages;
};

// Dynamic shared = bf16 staging + the cp.async weight ring (one stage is
// 8 warps x 32 lanes x 16B = 4KB); the ring depth adapts to the device.
// Batched decode fully stages [B][K] for the small-K projections, so the
// cap admits panels up to ~96KB; whether that many blocks stay co-resident
// is settled by the occupancy query, not this bound.
LaunchShape shape_for(int64_t stage_elems) {
  TORCH_CHECK(stage_elems > 0 && stage_elems <= 49152,
              "staging size out of range");
  const int stage_bytes = (int)(stage_elems * 2);
  const int rs = mak_ring_stages_impl(stage_bytes);
  return {stage_bytes + rs * 4096, rs};
}

}  // namespace

int64_t mak_num_blocks(at::Tensor prog, int64_t max_k) {
  const c10::cuda::CUDAGuard guard(prog.device());
  return (int64_t)mak_grid_blocks_impl(shape_for(max_k).smem);
}

int64_t mak_batch_maxb() { return (int64_t)mak_batch_maxb_impl(); }

double mak_bw_per_sm(at::Tensor probe) {
  const c10::cuda::CUDAGuard guard(probe.device());
  int dev = 0;
  cudaGetDevice(&dev);
  int clk_khz = 0, bus_bits = 0, sms = 1;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (sms < 1) sms = 1;
  const double gbps =
      2.0 * (double)clk_khz * 1e3 * ((double)bus_bits / 8.0) / 1e9;
  return gbps / (double)sms;
}

void mak_run(at::Tensor prog, at::Tensor barrier, int64_t pos,
             int64_t step_slot, int64_t max_k) {
  check_prog(prog);
  check_barrier(barrier);
  const c10::cuda::CUDAGuard guard(prog.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int* bar = barrier.data_ptr<int>();
  const LaunchShape sh = shape_for(max_k);
  const cudaError_t err = mak_launch_mega(
      reinterpret_cast<const long long*>(prog.data_ptr<int64_t>()),
      (int)prog.size(0), (int)pos, 1, (int)step_slot, -1, 1, sh.ring_stages,
      bar, mak_grid_blocks_impl(sh.smem), sh.smem, stream.stream(), nullptr);
  TORCH_CHECK(err == cudaSuccess, "mak_run launch failed: ",
              cudaGetErrorString(err));
}

void mak_run_steps(at::Tensor prog, at::Tensor barrier, int64_t pos0,
                   int64_t steps, int64_t slot0, int64_t max_k) {
  check_prog(prog);
  check_barrier(barrier);
  const c10::cuda::CUDAGuard guard(prog.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int* bar = barrier.data_ptr<int>();
  const long long* pp =
      reinterpret_cast<const long long*>(prog.data_ptr<int64_t>());
  const int n = (int)prog.size(0);
  const LaunchShape sh = shape_for(max_k);
  const int grid = mak_grid_blocks_impl(sh.smem);
  for (int64_t i = 0; i < steps; ++i) {
    const cudaError_t err =
        mak_launch_mega(pp, n, (int)(pos0 + i), 1, (int)(slot0 + i), 0, 1,
                        sh.ring_stages, bar, grid, sh.smem, stream.stream(),
                        nullptr);
    TORCH_CHECK(err == cudaSuccess, "mak_run_steps launch failed: ",
                cudaGetErrorString(err));
  }
}

void mak_run_seq(at::Tensor prog, at::Tensor barrier, int64_t pos0,
                 int64_t steps, int64_t slot0, int64_t prompt_len,
                 int64_t max_k, int64_t chunk_m) {
  check_prog(prog);
  check_barrier(barrier);
  TORCH_CHECK(steps >= 1, "steps must be >= 1");
  TORCH_CHECK(chunk_m >= 1 && chunk_m <= 8, "chunk_m must be in [1, 8]");
  const c10::cuda::CUDAGuard guard(prog.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int* bar = barrier.data_ptr<int>();
  const LaunchShape sh = shape_for(max_k * chunk_m);
  const cudaError_t err = mak_launch_mega(
      reinterpret_cast<const long long*>(prog.data_ptr<int64_t>()),
      (int)prog.size(0), (int)pos0, (int)steps, (int)slot0, (int)prompt_len,
      (int)chunk_m, sh.ring_stages, bar, mak_grid_blocks_impl(sh.smem),
      sh.smem, stream.stream(), nullptr);
  TORCH_CHECK(err == cudaSuccess, "mak_run_seq launch failed: ",
              cudaGetErrorString(err));
}

// B <= 8 runs the fused shared-staging kernel (input staged [B][K]); larger
// B runs the global-scratch kernel, which does not stage the input in
// shared and so is bounded only by the register accumulators.
// stage_elems is the largest shared panel the batch program stages: for the
// fused path max_k*B, for the wide path the max B*K over the projections
// that stay fused (the rest transform into global scratch and stage nothing).
void mak_run_batch(at::Tensor prog, at::Tensor barrier, at::Tensor pos_b,
                   int64_t steps, int64_t slot0, int64_t max_k,
                   int64_t kv_bstride, int64_t stage_elems) {
  check_prog(prog);
  check_barrier(barrier);
  const int maxb = mak_batch_maxb();
  TORCH_CHECK(pos_b.is_cuda() && pos_b.scalar_type() == at::kInt &&
                  pos_b.is_contiguous() && pos_b.numel() >= 1 &&
                  pos_b.numel() <= maxb,
              "pos_b must be a contiguous CUDA int32 tensor with 1..",
              maxb, " elements");
  TORCH_CHECK(steps >= 1, "steps must be >= 1");
  const int B = (int)pos_b.numel();
  const c10::cuda::CUDAGuard guard(prog.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int* bar = barrier.data_ptr<int>();
  const long long* pp =
      reinterpret_cast<const long long*>(prog.data_ptr<int64_t>());
  // The fully-fused path holds [B][K] for every projection and up to 8
  // accumulators, so it serves B up to min(8, 16384 / max_k); wider batches
  // take the acc[NB] kernel, where large-K projections spill to scratch.
  const int64_t fused_cap = std::min<int64_t>(8, 16384 / max_k);
  const LaunchShape sh = shape_for(stage_elems);
  cudaError_t err;
  if ((int64_t)B <= fused_cap) {
    err = mak_launch_mega_batch(
        pp, (int)prog.size(0), (int)steps, (int)slot0, sh.ring_stages, bar,
        mak_grid_blocks_impl(sh.smem), sh.smem, pos_b.data_ptr<int>(), B,
        (long long)kv_bstride, stream.stream());
  } else {
    err = mak_launch_mega_batch_big(
        pp, (int)prog.size(0), (int)steps, (int)slot0, sh.ring_stages, bar,
        mak_grid_blocks_big_impl(sh.smem), sh.smem, pos_b.data_ptr<int>(), B,
        (long long)kv_bstride, stream.stream());
  }
  TORCH_CHECK(err == cudaSuccess, "mak_run_batch launch failed: ",
              cudaGetErrorString(err));
}

void mak_run_ts(at::Tensor prog, at::Tensor barrier, int64_t pos,
                int64_t step_slot, int64_t max_k, at::Tensor ts) {
  check_prog(prog);
  check_barrier(barrier);
  TORCH_CHECK(ts.is_cuda() && ts.scalar_type() == at::kLong &&
                  ts.numel() >= prog.size(0) + 1,
              "ts must be a CUDA int64 tensor with n_phases+1 elements");
  const c10::cuda::CUDAGuard guard(prog.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int* bar = barrier.data_ptr<int>();
  const LaunchShape sh = shape_for(max_k);
  const cudaError_t err = mak_launch_mega(
      reinterpret_cast<const long long*>(prog.data_ptr<int64_t>()),
      (int)prog.size(0), (int)pos, 1, (int)step_slot, -1, 1, sh.ring_stages,
      bar, mak_grid_blocks_impl(sh.smem), sh.smem, stream.stream(),
      reinterpret_cast<long long*>(ts.data_ptr<int64_t>()));
  TORCH_CHECK(err == cudaSuccess, "mak_run_ts launch failed: ",
              cudaGetErrorString(err));
}

at::Tensor mak_run_phased(at::Tensor prog, int64_t pos, int64_t step_slot,
                          bool timed, int64_t max_k) {
  check_prog(prog);
  const c10::cuda::CUDAGuard guard(prog.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const long long* pp =
      reinterpret_cast<const long long*>(prog.data_ptr<int64_t>());
  const int n = (int)prog.size(0);
  const LaunchShape sh = shape_for(max_k);
  const int grid = mak_grid_blocks_impl(sh.smem);
  if (timed) {
    std::vector<float> ms((size_t)n, 0.f);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int p = 0; p < n; ++p) {
      cudaEventRecord(a, stream.stream());
      const cudaError_t err = mak_launch_phase(pp, p, (int)pos,
                                               (int)step_slot,
                                               sh.ring_stages, grid, sh.smem,
                                               stream.stream());
      TORCH_CHECK(err == cudaSuccess, "mak_run_phased launch failed: ",
                  cudaGetErrorString(err));
      cudaEventRecord(b, stream.stream());
      cudaEventSynchronize(b);
      cudaEventElapsedTime(&ms[p], a, b);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return torch::tensor(ms, torch::dtype(torch::kFloat32));
  }
  for (int p = 0; p < n; ++p) {
    const cudaError_t err = mak_launch_phase(pp, p, (int)pos, (int)step_slot,
                                             sh.ring_stages, grid, sh.smem,
                                             stream.stream());
    TORCH_CHECK(err == cudaSuccess, "mak_run_phased launch failed: ",
                cudaGetErrorString(err));
  }
  return torch::empty({0}, torch::dtype(torch::kFloat32));
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("mak_run(Tensor prog, Tensor barrier, int pos, int step_slot, "
          "int max_k) -> ()");
  ops.def("mak_run_steps(Tensor prog, Tensor barrier, int pos0, int steps, "
          "int slot0, int max_k) -> ()");
  ops.def("mak_run_seq(Tensor prog, Tensor barrier, int pos0, int steps, "
          "int slot0, int prompt_len, int max_k, int chunk_m) -> ()");
  ops.def("mak_run_phased(Tensor prog, int pos, int step_slot, bool timed, "
          "int max_k) -> Tensor");
  ops.def("mak_num_blocks(Tensor prog, int max_k) -> int");
  ops.def("mak_bw_per_sm(Tensor probe) -> float");
  ops.def("mak_run_ts(Tensor prog, Tensor barrier, int pos, int step_slot, "
          "int max_k, Tensor ts) -> ()");
  ops.def("mak_run_batch(Tensor prog, Tensor barrier, Tensor pos_b, "
          "int steps, int slot0, int max_k, int kv_bstride, "
          "int stage_elems) -> ()");
  // no tensor argument, so register a device-agnostic implementation
  ops.def("mak_batch_maxb() -> int", &mak_batch_maxb);

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("mak_run", torch::kCUDA, &mak_run);
  ops.impl("mak_run_steps", torch::kCUDA, &mak_run_steps);
  ops.impl("mak_run_seq", torch::kCUDA, &mak_run_seq);
  ops.impl("mak_run_phased", torch::kCUDA, &mak_run_phased);
  ops.impl("mak_num_blocks", torch::kCUDA, &mak_num_blocks);
  ops.impl("mak_bw_per_sm", torch::kCUDA, &mak_bw_per_sm);
  ops.impl("mak_run_ts", torch::kCUDA, &mak_run_ts);
  ops.impl("mak_run_batch", torch::kCUDA, &mak_run_batch);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)



