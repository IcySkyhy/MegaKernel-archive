/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM120 architecture.

    This example demonstrates a simple way to instantiate and run a blockscaled NVFP4 GEMM on the NVIDIA Blackwell SM120 architecture.
    This kernel is optimized for the GeForce RTX 50 series GPUs.

    The Blackwell SM120 CUTLASS kernel uses the new Block Scaled Tensor Core MMA Instructions (mma.sync.aligned.block_scale).
    NVFP4 MMA has 2x throughput compared to MXFP8 MMA and 4x throughput compared to Ada Tensor Core FP8 MMA.
    (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    This kernel leverages:
    1. Warp-Specialized persistent kernel design that supports both cooperative and ping-pong kernel schedule introduced in Hopper.
    2. The new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).
    3. Block Scaled Tensor Core MMA Instructions
    4. Epilogue Optimization

    Note that GeForce RTX 50 series GPUs do not support:
    1. Multicast feature of TMA load. Cluster shape has to be 1x1x1.
    2. Dynamic datatypes.

    Usage:

      $ ./examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm --m=2048 --n=2048 --k=2048
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
#ifndef TBM
#define TBM 128
#endif
#ifndef TBN
#define TBN 128
#endif
#ifndef TBK
#define TBK 128
#endif
using ThreadBlockShape    = Shape<Int<TBM>,Int<TBN>,Int<TBK>>;              // Threadblock's tile size (macro-tunable)
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
uint64_t seed;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool verify_ref;

  float alpha, beta;
  int iterations;
  int m, n, k;

  Options():
    help(false),
    verify_ref(true),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    int v = 1; cmd.get_cmd_line_argument("verify", v); verify_ref = (v != 0);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "79a_blackwell_geforce_nvfp4_bf16_gemm\n\n"
      << "  Blackwell NVFP4 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;
  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  initialize_block(block_A.host_view(), seed + 2021);
  initialize_block(block_B.host_view(), seed + 2022);
  initialize_block(block_C.host_view(), seed + 2023);
  initialize_block(block_SFA.host_view(), seed + 2024);
  initialize_block(block_SFB.host_view(), seed + 2025);

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
}

// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      {options.alpha, options.beta},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };

  return arguments;
}

bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                 // ElementAccumulator
      decltype(tensor_A),                 // TensorA
      decltype(tensor_SFA),               // TensorSfA
      decltype(tensor_B),                 // TensorB
      decltype(tensor_SFB)                // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D)                    // TensorD
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // Comparison
  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  return passed;
}

// ===================== NVFP4 device quantizer (keystone for the FFN activation seam) =====================
using ElementAData = typename ElementA::DataType;          // float_e2m1_t, packed 2 per byte
using ElementASF   = typename ElementA::ScaleFactorType;   // float_ue4m3_t (NVFP4 block scale)
using ElementBData = typename ElementB::DataType;
using ElementBSF   = typename ElementB::ScaleFactorType;

// One thread per (row m, 16-element K-block). RowMajor A means the 16 e2m1 codes are physically
// contiguous (8 bytes) and owned by this thread alone, so the subbyte writes are race free.
// Convention matches gett.hpp: store SFA = quantize(amax/6); encode element = e2m1(x / dequant(SFA)).
__global__ void quantize_A_kernel(const float* __restrict__ src,
                                  ElementAData* a_ptr, ElementASF* sf_ptr,
                                  LayoutA layoutA, LayoutSFA layoutSFA,
                                  int M, int K) {
  int m  = blockIdx.y * blockDim.y + threadIdx.y;
  int kb = blockIdx.x * blockDim.x + threadIdx.x;
  int K16 = K >> 4;
  if (m >= M || kb >= K16) return;
  int k0 = kb << 4;
  float vals[16]; float amax = 0.f;
  #pragma unroll
  for (int v = 0; v < 16; ++v) { float x = src[(size_t)m*K + k0 + v]; vals[v] = x; amax = fmaxf(amax, fabsf(x)); }
  float pvscale = amax * (1.0f / 6.0f);                 // fp_max(e2m1) = 6
  ElementASF q = static_cast<ElementASF>(pvscale);      // fp32 -> ue4m3
  float deq = static_cast<float>(q);
  float rcp = deq > 0.f ? (1.0f / deq) : 0.f;
  auto aT  = make_tensor(cute::recast_ptr<ElementAData>(a_ptr), layoutA);
  auto sfT = make_tensor(cute::make_gmem_ptr(sf_ptr), layoutSFA);
  sfT(m, k0, 0) = q;
  #pragma unroll
  for (int v = 0; v < 16; ++v) aT(m, k0 + v, 0) = static_cast<ElementAData>(vals[v] * rcp);
}

// Host NVFP4 quantizer for operand B (ColumnMajor [N,K]). Weights are quantized once offline;
// host single-thread sidesteps the subbyte race ColumnMajor packing would create on device.
static void quantize_B_host(const std::vector<float>& hB, int N, int K) {
  auto bT  = make_tensor(make_iterator(block_B.host_data()), layout_B);
  auto sfT = make_tensor(block_SFB.host_data(), layout_SFB);
  int K16 = K >> 4;
  for (int n = 0; n < N; ++n) for (int kb = 0; kb < K16; ++kb) {
    int k0 = kb << 4; float amax = 0.f;
    for (int v = 0; v < 16; ++v) amax = std::max(amax, std::fabs(hB[(size_t)n*K + k0 + v]));
    float pvscale = amax * (1.0f / 6.0f);
    ElementBSF q = static_cast<ElementBSF>(pvscale);
    float deq = static_cast<float>(q); float rcp = deq > 0.f ? 1.0f / deq : 0.f;
    sfT(n, k0, 0) = q;
    for (int v = 0; v < 16; ++v) bT(n, k0 + v, 0) = static_cast<ElementBData>(hB[(size_t)n*K + k0 + v] * rcp);
  }
}

// Keystone probe: device-quantize A + host-quantize B from fp32 sources, run the FP4 GEMM, compare
// to an fp32 reference of the ORIGINAL data. The SF layout is correct by construction (the same
// layout object is handed to the GEMM), so this validates the quant math + subbyte packing.
int quant_probe(Options &options) {
  using namespace cute;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  int M = options.m, N = options.n, K = options.k;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  layout_A = make_layout(make_shape(M, K, 1), stride_A);
  layout_B = make_layout(make_shape(N, K, 1), stride_B);
  layout_C = make_layout(make_shape(M, N, 1), stride_C);
  layout_D = make_layout(make_shape(M, N, 1), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  std::vector<float> hA((size_t)M*K), hB((size_t)N*K);
  std::mt19937 rng(1234); std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : hA) x = dist(rng);
  for (auto& x : hB) x = dist(rng);

  for (int i = 0; i < (int)size(layout_C); ++i) block_C.host_data()[i] = ElementC(0);
  block_C.sync_device();

  float* dA = nullptr; CUDA_CHECK(cudaMalloc(&dA, sizeof(float)*(size_t)M*K));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float)*(size_t)M*K, cudaMemcpyHostToDevice));
  dim3 blk(32, 8), grd((unsigned)((K/16 + 31)/32), (unsigned)((M + 7)/8));
  quantize_A_kernel<<<grd, blk>>>(dA, block_A.device_data(), block_SFA.device_data(), layout_A, layout_SFA, M, K);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

  quantize_B_host(hB, N, K); block_B.sync_device(); block_SFB.sync_device();

  options.alpha = 1.f; options.beta = 0.f;
  Gemm gemm; auto arguments = args_from_options(options);
  size_t ws = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  block_D.sync_host();

  auto dT = make_tensor(make_iterator(block_D.host_data()), layout_D);
  double num = 0.0, den = 0.0, maxrel = 0.0;
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    double ref = 0.0;
    for (int k = 0; k < K; ++k) ref += (double)hA[(size_t)m*K+k] * (double)hB[(size_t)n*K+k];
    double got = (double)(float)dT(m, n, 0);
    double d = got - ref; num += d*d; den += ref*ref;
    double rel = std::fabs(d) / (std::fabs(ref) + 1e-6);
    if (rel > maxrel) maxrel = rel;
  }
  double frob = std::sqrt(num / (den + 1e-30));
  printf("QUANT_PROBE M=%d N=%d K=%d  rel_frob=%.4f  max_elem_rel=%.3f  %s\n",
         M, N, K, frob, maxrel, (frob < 0.15 ? "PASS" : "FAIL"));
  cudaFree(dA);
  return 0;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  cudaDeviceSynchronize();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.verify_ref) {
    result.passed = verify(options);
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
    if (!result.passed) {
      exit(-1);
    }
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    // re-warm after the CPU-side verify so the GPU is at steady clock, not cool-boost
    for (int w = 0; w < 100; ++w) { CUTLASS_CHECK(gemm.run()); }
    cudaDeviceSynchronize();
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);


    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
#ifdef DIAG
  printf("DIAG ElementD=%s  mainloop_stages=%d  mainloop_smem=%zu  epilogue_smem=%zu  total_smem=%zu\n",
         (cutlass::sizeof_bits<ElementD>::value==16 ? "bf16" : "fp4"),
         (int)CollectiveMainloop::DispatchPolicy::Stages,
         sizeof(typename CollectiveMainloop::SharedStorage),
         sizeof(typename CollectiveEpilogue::SharedStorage),
         sizeof(typename GemmKernel::SharedStorage));
  return 0;
#endif

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit for SM120 support,
  // or CUDA 12.9 or higher for SM121 support.
  // Must have compute capability at least 120.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer for SM120 support." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#elif defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer for SM121 support." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#endif

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 120 or 121)." << std::endl;
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#ifdef QUANT_PROBE
  quant_probe(options);
#else
  run<Gemm>(options);
#endif
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
