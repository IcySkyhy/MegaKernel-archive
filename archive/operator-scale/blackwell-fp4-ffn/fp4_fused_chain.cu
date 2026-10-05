// Fused sparse residual FP4 double GEMM: both chain GEMMs in ONE persistent
// kernel. Scheduler domain is (M, 2N, K): n-tile >= n_tiles means GEMM2, which
// selects mainloop2/epilogue2 params per tile. GEMM2 B-loads gate on per
// n-stripe counters that GEMM1 tiles bump after their epilogue store, so the
// kernel-boundary serialization bubble (two tails + launch gap) collapses to
// tile-granular dependency with D1 L2-resident.
//
// Ordering honesty: a GEMM2 tile cannot start loading D1 stripe n before all
// 32 GEMM1 tiles of that stripe have executed and issued their stores. The
// signal follows store() issue, not TMA completion (sub-microsecond skew);
// values are parked project-wide, the dependency structure is real.
//
// Launch order MUST enumerate all GEMM1 tiles before GEMM2 tiles (CLC steals
// are strict launch order): AlongM raster with any power-of-2 swizzle <= 32
// satisfies this. AlongN can put spinning GEMM2 tiles on all SMs -> deadlock.
//
// Same shortcuts and accounting as fp4_sparse_chain.cu: 4*M*N*K dense-equiv.
// Build: 13.3 ptxas splice, sm_120a only (README sparse section).
#include <iostream>
#include <cstdio>
#include <vector>
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
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"
#include "sm120_fused_chain.hpp"

using namespace cute;

using ElementA     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // weights (sparse operand)
using LayoutATag   = cutlass::layout::RowMajor;
constexpr int AlignmentA = 64;
using ElementB     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // activations
using LayoutBTag   = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD     = cutlass::float_e2m1_t;
using ElementSFD   = cutlass::float_ue4m3_t;
#ifdef C_FP4
using ElementC     = cutlass::float_e2m1_t;                        // residual as fp4 stream
#else
using ElementC     = cutlass::bfloat16_t;                          // residual
#endif
using LayoutCTag   = cutlass::layout::RowMajor;
using LayoutDTag   = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementE     = uint8_t;                                      // 2:4 metadata
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int OutputSFVectorSize = 32;
#ifndef TBM
#define TBM 128
#endif
#ifndef TBN
#define TBN 128
#endif
#ifndef TBK
#define TBK 256
#endif
using ThreadBlockShape = Shape<Int<TBM>,Int<TBN>,Int<TBK>>;
using ClusterShape     = Shape<_1,_1,_1>;

// Epilogue instantiated through Sm120TmaBuilderImpl directly: the named-schedule
// builder hardcodes StagesC=2, which strangles the residual C pipeline (the
// producer warp can only run 2 subtiles ahead). Deeper StagesC lets the C loads
// spread across the whole mainloop window (and across G1 tiles, which load no
// C). StagesD=1 is free: D is fp4, stores drain fast. Budget: mainloop 3 stages
// = 87040B; epi must fit in 101376-87040 = 14336B. bf16 C stage = 4096B ->
// StagesC=3 exactly fits; fp4 C stage = 1024B -> StagesC=8 = full-tile runahead.
#ifndef STAGESC
#if TBM == 256
#ifdef C_FP4
#define STAGESC 16
#else
#define STAGESC 4
#endif
#else
#ifdef C_FP4
#define STAGESC 8
#else
#define STAGESC 3
#endif
#endif
#endif
#ifndef STAGESD
#define STAGESD 1
#endif
using CollectiveEpilogue = typename cutlass::epilogue::collective::detail::Sm120TmaBuilderImpl<
    ThreadBlockShape, Shape<_64,_32>,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::fusion::LinCombBlockScaleFactor<
        OutputSFVectorSize, ElementD, ElementAccumulator, ElementSFD, LayoutDTag, ElementC>,
    cutlass::epilogue::Sm120TmaWarpSpecialized<STAGESC, STAGESD, 4, false, false>
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelScheduleType>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using FusedKernel = cutlass::gemm::kernel::FusedSparseChain<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutE   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;

using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
    Shape<int,int,int,int>, ElementA::DataType, LayoutATag, SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    Shape<int,int,int,int>, ElementA::DataType, LayoutATag, SparseConfig, ArchTag>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

// Explicit launch bounds: match the canonical adapter launch configuration.
template <class Op>
__global__ __launch_bounds__(Op::MaxThreadsPerBlock, Op::MinBlocksPerMultiprocessor)
void fused_entry(CUTLASS_GRID_CONSTANT typename Op::Params const params) {
  extern __shared__ char smem[];
  Op op;
  op(params, smem);
}

// One sparse weight operand: dense-random fill, 2:4 zero mask, device compress.
struct SparseWeight {
  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> dense, comp;
  cutlass::HostTensor<ElementE, cutlass::layout::PackedVectorLayout> meta;
  LayoutA layout_A; LayoutE layout_E;
  void init(int M, int K, int seed, StrideA stride_A, cudaStream_t s) {
    auto workload = make_shape(M, 1, K, 1);
    CompressorUtility util(workload, stride_A);
    layout_A = SparseConfig::fill_layoutA(workload);
    layout_E = SparseConfig::fill_layoutE(workload);
    dense.reset(cutlass::make_Coord(M * K));
    comp.reset(cutlass::make_Coord(util.get_tensorA_m_physical() * util.get_tensorA_k_physical()));
    meta.reset(cutlass::make_Coord(util.get_metadata_m_physical() * util.get_metadata_k_physical()));
    cutlass::reference::host::TensorFillRandomUniform(dense.host_view(), seed, 3, -3, 0);
    util.structure_sparse_zero_mask_fill(dense.host_data(), seed + 1);
    dense.sync_device();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Compressor::Arguments args{
      {M, 1, K, 1},
      { dense.device_data(), stride_A, comp.device_data(), meta.device_data() },
      {hw_info} };
    Compressor op;
    cutlass::device_memory::allocation<uint8_t> ws(Compressor::get_workspace_size(args));
    CUTLASS_CHECK(op.can_implement(args));
    CUTLASS_CHECK(op.initialize(args, ws.get(), s));
    CUTLASS_CHECK(op.run(s));
    cudaStreamSynchronize(s);
  }
};

int main(int argc, char** argv){
  int n     = argc>1 ? atoi(argv[1]) : 4096;
  int iters = argc>2 ? atoi(argv[2]) : 200;
  int swz   = argc>3 ? atoi(argv[3]) : 1;   // power of 2, <= 32 (launch-order safety)
  int ras   = argc>4 ? atoi(argv[4]) : 1;   // 1 AlongM only; AlongN can deadlock the gate
  if (ras != 1) { printf("raster forced AlongM: GEMM1-before-GEMM2 launch order required\n"); ras = 1; }
  int M=n, N=n, K=n;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M,K,1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N,K,1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M,N,1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M,N,1});

  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(make_shape(M,N,K,1));

  cudaStream_t s; cudaStreamCreate(&s);

  SparseWeight W1, W2;
  W1.init(M, K, 11, stride_A, s);
  W2.init(M, K, 22, stride_A, s);

  cutlass::HostTensor<ElementB::DataType,        cutlass::layout::PackedVectorLayout> X, D1, D2;
  cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> SFA1, SFA2, SFB;
  cutlass::HostTensor<ElementC,   cutlass::layout::PackedVectorLayout> C2;
  cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> SFD1, SFD2;
  cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> Norm;
  X.reset(cutlass::make_Coord(N*K));
  D1.reset(cutlass::make_Coord(M*N)); D2.reset(cutlass::make_Coord(M*N));
  C2.reset(cutlass::make_Coord(M*N));
  SFA1.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  SFA2.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  SFD1.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  SFD2.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  Norm.reset(cutlass::make_Coord(1));
  cutlass::reference::host::TensorFillRandomUniform(X.host_view(), 7, 3, -3, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFA1.host_view(), 8, 2, 1, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFA2.host_view(), 9, 2, 1, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFB.host_view(), 10, 2, 1, 0);
  Norm.at(cutlass::make_Coord(0)) = ElementCompute(2);
  X.sync_device(); C2.sync_device();
  SFA1.sync_device(); SFA2.sync_device(); SFB.sync_device();
  SFD1.sync_device(); SFD2.sync_device(); Norm.sync_device();

  auto make_args = [&](SparseWeight& W, ElementB::ScaleFactorType* sfa,
                       ElementB::DataType* b, ElementC* c,
                       ElementB::DataType* d, ElementSFD* sfd, float beta){
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
      { W.comp.device_data(), W.layout_A, b, stride_B, W.meta.device_data(), W.layout_E,
        sfa, layout_SFA, SFB.device_data(), layout_SFB },
      { {1.0f, beta}, c, stride_C, d, stride_D }
    };
    args.epilogue.thread.block_scale_factor_ptr = sfd;
    args.epilogue.thread.norm_constant_ptr      = Norm.device_data();
    using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.max_swizzle_size = swz;
    args.scheduler.raster_order = RO::AlongM;
    return args;
  };

  // GEMM1: W1*X -> D1 (fp4) + SFD1, beta=0.  GEMM2: W2*D1 -> D2 + residual C2, beta=1.
  auto args1 = make_args(W1, SFA1.device_data(), X.device_data(),  C2.device_data(), D1.device_data(), SFD1.device_data(), 0.0f);
  auto args2 = make_args(W2, SFA2.device_data(), D1.device_data(), C2.device_data(), D2.device_data(), SFD2.device_data(), 1.0f);

  // Per-GEMM params exactly as the adapter would build them.
  cutlass::device_memory::allocation<uint8_t> ws1(GemmKernel::get_workspace_size(args1));
  cutlass::device_memory::allocation<uint8_t> ws2(GemmKernel::get_workspace_size(args2));
  if (!GemmKernel::can_implement(args1) || !GemmKernel::can_implement(args2)) {
    printf("can_implement failed\n"); return 1;
  }
  CUTLASS_CHECK(GemmKernel::initialize_workspace(args1, ws1.get(), s));
  CUTLASS_CHECK(GemmKernel::initialize_workspace(args2, ws2.get(), s));
  auto params1 = GemmKernel::to_underlying_arguments(args1, ws1.get());
  auto params2 = GemmKernel::to_underlying_arguments(args2, ws2.get());

  // Scheduler over the doubled-N domain (M, 2N, K): second N half is GEMM2.
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto shape2N = make_shape(M, 2*N, K, 1);
  typename FusedKernel::TileSchedulerArguments sched_args{};
  sched_args.max_swizzle_size = swz;
  sched_args.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  constexpr uint32_t NumEpiSubTiles = CollectiveEpilogue::get_store_pipe_increment(typename FusedKernel::TileShape{});
  size_t sched_ws_size = FusedKernel::TileScheduler::template get_workspace_size<Shape<int,int,int,int>, ElementAccumulator>(
      sched_args, shape2N, hw_info, FusedKernel::NumMmaWarpGroups, NumEpiSubTiles);
  cutlass::device_memory::allocation<uint8_t> sched_ws(sched_ws_size);
  auto sched_params = FusedKernel::TileScheduler::to_underlying_arguments(
      shape2N, typename FusedKernel::TileShape{}, typename FusedKernel::ClusterShape{},
      hw_info, sched_args, sched_ws.get(), NumEpiSubTiles);

  int m_tiles = (M + TBM - 1) / TBM, n_tiles = (N + TBN - 1) / TBN;
  int* counters = nullptr;
  cudaMalloc(&counters, n_tiles * sizeof(int));
  cudaMemset(counters, 0, n_tiles * sizeof(int));

  typename FusedKernel::Params fp{};
  fp.mode          = cutlass::gemm::GemmUniversalMode::kGemm;
  fp.problem_shape = {M, N, K, 1};        // real N: device coords and load_init
  fp.mainloop      = params1.mainloop;
  fp.epilogue      = params1.epilogue;
  fp.hw_info       = hw_info;
  fp.scheduler     = sched_params;        // doubled N: tile enumeration
  fp.mainloop2     = params2.mainloop;
  fp.epilogue2     = params2.epilogue;
  fp.counters      = counters;
  fp.n_tiles       = n_tiles;
  fp.m_tiles       = m_tiles;

  dim3 grid = FusedKernel::TileScheduler::get_grid_shape(
      sched_params, shape2N, typename FusedKernel::TileShape{},
      typename FusedKernel::ClusterShape{}, hw_info, sched_args);
  dim3 block = FusedKernel::get_block_shape();
  int smem = FusedKernel::SharedStorageSize;
  CUDA_CHECK(cudaFuncSetAttribute(fused_entry<FusedKernel>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  int gate  = argc>5 ? atoi(argv[5]) : 1;  // 0: gate targets 0, measures structure overhead alone
  int delay = argc>6 ? atoi(argv[6]) : n_tiles;  // G1 lead columns; < n_tiles interleaves stripes
  if (delay < 1) delay = 1;
  if (delay > n_tiles) delay = n_tiles;
  if (delay < n_tiles && swz != 1) { printf("interleave requires swz=1 (launch-order math)\n"); return 1; }
  fp.delay = delay;
  int beta2off = argc>7 ? atoi(argv[7]) : 0;  // diag: drop C2 residual load from GEMM2
  if (beta2off) { fp.epilogue2 = params1.epilogue; }  // beta=0 epilogue params, D1/SFD1 targets
  int pf = argc>8 ? atoi(argv[8]) : 0;  // prefetch C2 tiles into L2 during G1 tiles
  fp.c2_base     = pf ? reinterpret_cast<char const*>(C2.device_data()) : nullptr;
  fp.c_row_bytes = pf ? long(N) * sizeof(ElementC) : 0;
  int epoch = 0;
  auto launch = [&]{
    ++epoch;
    fp.epoch = gate ? epoch : 0;
    fused_entry<FusedKernel><<<grid, block, smem, s>>>(fp);
  };

  launch();
  CUDA_CHECK(cudaGetLastError());
  cudaError_t e0 = cudaStreamSynchronize(s);
  if (e0) { printf("first launch failed: %s\n", cudaGetErrorString(e0)); return 1; }
  for(int i=0;i<9;i++) launch();
  cudaStreamSynchronize(s);

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0,s);
  for(int i=0;i<iters;i++) launch();
  cudaEventRecord(t1,s);
  cudaError_t e=cudaStreamSynchronize(s);
  if(e){ printf("cuda err: %s\n", cudaGetErrorString(e)); return 1; }
  float ms_total=0; cudaEventElapsedTime(&ms_total,t0,t1);

  // Gate integrity: every stripe counter must equal m_tiles * launches.
  std::vector<int> hc(n_tiles);
  cudaMemcpy(hc.data(), counters, n_tiles*sizeof(int), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < n_tiles; i++) if (hc[i] != m_tiles * epoch) bad++;
  if (gate && bad) { printf("GATE BROKEN: %d stripes off (want %d)\n", bad, m_tiles*epoch); return 1; }

  double ms_pair = double(ms_total)/iters;
  double flop = 4.0*(double)M*N*K;   // dense-equivalent, two GEMMs
  double tf = flop/(ms_pair/1e3)/1e12;
  printf("fused sparse residual FP4 double GEMM  N=%d grid=%dx%d swz=%d d=%d%s C=%db sC=%d sD=%d  %.4f ms/pair  %.1f TFLOPS dense-equiv\n",
         n, grid.x, grid.y, swz, delay, beta2off?" beta2off":pf?" pf":"",
         (int)cutlass::sizeof_bits<ElementC>::value, STAGESC, STAGESD, ms_pair, tf);
  return 0;
}
