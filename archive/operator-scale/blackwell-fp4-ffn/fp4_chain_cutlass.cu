// Fused FP4 double GEMM: D=(A*B)*C, all NVFP4, via CUTLASS sm_120 (example 79b type).
// The point: GEMM1 quantizes its result to FP4 + generates block scales INSIDE its
// epilogue (LinCombBlockScaleFactor fusion), so GEMM2 reads that FP4 buffer directly
// as its A operand. No separate BF16->FP4 repack kernel, no BF16 intermediate round
// trip. This is the lever the cuBLASLt double-GEMM harness (605 TFLOPS) was missing:
// cuBLASLt rejects FP4 output (status 15) and forces the repack as its own kernel.
//
// Throughput-only: GEMM2's input scales are a prepared buffer, not GEMM1's generated
// SFD (the ue8m0-out vs ue4m3-in scale-format match is an accuracy detail, irrelevant
// to wall-clock). The real D1->A2 data dependency forces the serialized FP4 read.
//
// Build (from ~/Desktop/code/cutlass):
//   nvcc -O3 -arch=sm_120a -std=c++17 --expt-relaxed-constexpr -DNDEBUG \
//     -I include -I tools/util/include -I examples/common \
//     ../double-gemm/fp4_chain_cutlass.cu -o /tmp/fp4chain
#include <iostream>
#include <cstdio>
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
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"

using namespace cute;

// ---- type config: identical to 79b (NVFP4 in, FP4 out + ue8m0 scale generation) ----
// FP4 block-scale granularity: NVFP4 = vec16 (ue4m3 scales), MXFP4 = vec32 (ue8m0,
// half the scale traffic). -DMXFP4 selects the coarser, lighter-scale path.
#ifdef MXFP4
using ElementA     = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB     = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
constexpr int SFVEC = 32;
#else
using ElementA     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int SFVEC = 16;
#endif
using LayoutATag   = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
using LayoutBTag   = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD     = cutlass::float_e2m1_t;
using ElementSFD   = cutlass::float_ue8m0_t;
// residual (C) format: -DCRESID_FP8 halves C read traffic vs BF16, to test if the
// residual cost is DRAM read traffic (scales with bytes) or the epilogue path itself.
#ifdef CRESID_FP8
using ElementC     = cutlass::float_e4m3_t;
#else
using ElementC     = cutlass::bfloat16_t;
#endif
using LayoutCTag   = cutlass::layout::RowMajor;
using LayoutDTag   = cutlass::layout::RowMajor;
using LayoutSFDTag = LayoutDTag;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
#ifndef TBM
#define TBM 128
#endif
#ifndef TBN
#define TBN 128
#endif
#ifndef TBK
#define TBK 128
#endif
using ThreadBlockShape = Shape<Int<TBM>,Int<TBN>,Int<TBK>>;
using ClusterShape     = Shape<_1,_1,_1>;  // clusters unavailable on sm_120 ("no programmatic multicast on this arch")
// mainloop kernel schedule (the cuBLASLt-gap knob); override with -DSCHED=...
#ifndef SCHED
#define SCHED cutlass::gemm::KernelTmaWarpSpecializedPingpong
#endif
constexpr int InputSFVectorSize  = SFVEC;
constexpr int OutputSFVectorSize = InputSFVectorSize;
#ifndef SWIZZLE
#define SWIZZLE 1
#endif
#ifndef RASTER
#define RASTER 0
#endif

using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize, ElementD, ElementCompute, ElementSFD, LayoutSFDTag, ElementC>;

// epilogue subtile override (numeric selector; commas/angle-brackets break nvcc -D).
// smaller frees smem for more mainloop stages. must divide the 128x128 CTA tile.
#ifndef EPI
#define EPI 0
#endif
#if   EPI==1
using EpiTileT = cute::Shape<cute::_64,cute::_16>;
#elif EPI==2
using EpiTileT = cute::Shape<cute::_32,cute::_32>;
#elif EPI==3
using EpiTileT = cute::Shape<cute::_32,cute::_16>;
#elif EPI==4
using EpiTileT = cute::Shape<cute::_64,cute::_64>;
#else
using EpiTileT = cutlass::epilogue::collective::EpilogueTileAuto;
#endif
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    EpiTileT,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
#ifdef STAGES
    cutlass::gemm::collective::StageCount<STAGES>,
#else
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
#endif
    SCHED>::CollectiveOp;

// tile scheduler (the wave-quantization / load-balance knob); -DTILESCHED=cutlass::gemm::StreamKScheduler
#ifndef TILESCHED
#define TILESCHED void
#endif
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, TILESCHED>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;

int main(int argc, char** argv){
  int n     = argc>1 ? atoi(argv[1]) : 4096;
  int iters = argc>2 ? atoi(argv[2]) : 200;
  int swz   = argc>3 ? atoi(argv[3]) : SWIZZLE;  // runtime override of -DSWIZZLE
  int ras   = argc>4 ? atoi(argv[4]) : RASTER;   // runtime override of -DRASTER
  int M=n, N=n, K=n;

#ifdef DIAG
  printf("DIAG ElementD=%s  mainloop_stages=%d  mainloop_smem=%zu  epilogue_smem=%zu  total_smem=%zu  tile=%dx%dx%d\n",
         (cutlass::sizeof_bits<ElementD>::value==16 ? "bf16" : "fp4"),
         (int)CollectiveMainloop::DispatchPolicy::Stages,
         sizeof(typename CollectiveMainloop::SharedStorage),
         sizeof(typename CollectiveEpilogue::SharedStorage),
         sizeof(typename GemmKernel::SharedStorage),
         TBM, TBN, TBK);
  return 0;
#endif

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M,K,1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N,K,1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M,N,1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M,N,1});

  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(make_shape(M,N,K,1));

  // buffers (HostTensor used only for device allocation/sizing)
  cutlass::HostTensor<ElementA::DataType,        cutlass::layout::PackedVectorLayout> A1, B1, B2, D1, D2;
  cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> SFA, SFB;
  cutlass::HostTensor<ElementC,   cutlass::layout::PackedVectorLayout> C1, C2;
  cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> SFD1, SFD2;
  cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> Norm;

  A1.reset(cutlass::make_Coord(M*K)); B1.reset(cutlass::make_Coord(N*K));
  B2.reset(cutlass::make_Coord(N*K));
  D1.reset(cutlass::make_Coord(M*N)); D2.reset(cutlass::make_Coord(M*N));
  C1.reset(cutlass::make_Coord(M*N)); C2.reset(cutlass::make_Coord(M*N));
  SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  SFD1.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  SFD2.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  Norm.reset(cutlass::make_Coord(1));

  cutlass::reference::host::TensorFillRandomUniform(A1.host_view(), 1, 3, -3, 0);
  cutlass::reference::host::TensorFillRandomUniform(B1.host_view(), 2, 3, -3, 0);
  cutlass::reference::host::TensorFillRandomUniform(B2.host_view(), 3, 3, -3, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFA.host_view(), 4, 2, 1, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFB.host_view(), 5, 2, 1, 0);
  Norm.at(cutlass::make_Coord(0)) = ElementCompute(2);
  A1.sync_device(); B1.sync_device(); B2.sync_device();
  SFA.sync_device(); SFB.sync_device(); Norm.sync_device();
  SFD1.sync_device(); SFD2.sync_device();

  auto make_args = [&](ElementA::DataType* a, ElementA::DataType* b, ElementC* c,
                       ElementA::DataType* d, ElementSFD* sfd, float beta){
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
      { a, stride_A, b, stride_B, SFA.device_data(), layout_SFA, SFB.device_data(), layout_SFB },
      { {1.0f, beta}, c, stride_C, d, stride_D }
    };
    if constexpr (IsBlockScaleSupported) {
      args.epilogue.thread.block_scale_factor_ptr = sfd;
      args.epilogue.thread.norm_constant_ptr      = Norm.device_data();
    }
    // L2-reuse knobs (default swizzle=1 is OFF, raster=Heuristic): argv[3]=swizzle argv[4]=raster 0|1|2
    using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.max_swizzle_size = swz;
    args.scheduler.raster_order = (ras==1)?RO::AlongM:(ras==2)?RO::AlongN:RO::Heuristic;
    return args;
  };

  // GEMM1: A1*B1 -> D1 (fp4) + SFD1.  GEMM2: D1*B2 -> D2 (fp4) + SFD2.
  // Step 1: GEMM2 adds the block residual via the epilogue beta term: out = D1*B2 + beta*C2.
  // -DRESIDUAL sets beta=1 on g2 only (g1 stays beta=0). C2 is BF16 MxN, a conservative
  // residual cost (BF16 read > FP4 read). Values irrelevant to throughput, accuracy parked.
#ifdef RESIDUAL
  float beta2 = 1.0f;
#else
  float beta2 = 0.0f;
#endif
  auto args1 = make_args(A1.device_data(), B1.device_data(), C1.device_data(), D1.device_data(), SFD1.device_data(), 0.0f);
  auto args2 = make_args(D1.device_data(), B2.device_data(), C2.device_data(), D2.device_data(), SFD2.device_data(), beta2);

  Gemm g1, g2;
  cutlass::device_memory::allocation<uint8_t> ws1(Gemm::get_workspace_size(args1));
  cutlass::device_memory::allocation<uint8_t> ws2(Gemm::get_workspace_size(args2));
  CUTLASS_CHECK(g1.can_implement(args1)); CUTLASS_CHECK(g2.can_implement(args2));
  CUTLASS_CHECK(g1.initialize(args1, ws1.get()));
  CUTLASS_CHECK(g2.initialize(args2, ws2.get()));

  cudaStream_t s; cudaStreamCreate(&s);
  // warmup
  for(int i=0;i<10;i++){ CUTLASS_CHECK(g1.run(s)); CUTLASS_CHECK(g2.run(s)); }
  cudaStreamSynchronize(s);

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
#ifdef USE_GRAPH
  // capture one g1+g2 pair into a CUDA graph; replay removes per-iter host gap
  cudaGraph_t graph; cudaGraphExec_t gexec;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
#ifdef G1ONLY
  CUTLASS_CHECK(g1.run(s));
#else
  CUTLASS_CHECK(g1.run(s)); CUTLASS_CHECK(g2.run(s));
#endif
  cudaStreamEndCapture(s, &graph);
  cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
  for(int i=0;i<10;i++) cudaGraphLaunch(gexec, s);
  cudaStreamSynchronize(s);
  cudaEventRecord(t0,s);
  for(int i=0;i<iters;i++) cudaGraphLaunch(gexec, s);
  cudaEventRecord(t1,s);
#else
  cudaEventRecord(t0,s);
#ifdef G1ONLY
  for(int i=0;i<iters;i++){ CUTLASS_CHECK(g1.run(s)); }
#else
  for(int i=0;i<iters;i++){ CUTLASS_CHECK(g1.run(s)); CUTLASS_CHECK(g2.run(s)); }
#endif
  cudaEventRecord(t1,s);
#endif
  cudaError_t e=cudaStreamSynchronize(s);
  if(e){ printf("cuda err: %s\n", cudaGetErrorString(e)); return 1; }
  float ms_total=0; cudaEventElapsedTime(&ms_total,t0,t1);

  double ms_pair = double(ms_total)/iters;
#ifdef G1ONLY
  double flop = 2.0*(double)M*N*K;             // single GEMM (GEMM1, FP4 output)
#else
  double flop = 4.0*(double)M*N*K;             // two GEMMs
#endif
  double tf = flop/(ms_pair/1e3)/1e12;
  const double ceil_fp4 = 703.0*(2.95/2.45);   // dense FP4 at measured 2.95 GHz
  printf("fused FP4 double GEMM  N=%d tile=%dx%dx%d sw%d r%d  %.4f ms/pair  %.1f TFLOPS  %.1f%% of %.0f (actual-clock peak)\n",
         n, TBM, TBN, TBK, swz, ras, ms_pair, tf, 100.0*tf/ceil_fp4, ceil_fp4);
  return 0;
}
