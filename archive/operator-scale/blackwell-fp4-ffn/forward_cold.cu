// Sparse residual FP4 double GEMM: D2 = W2*(W1*X) + C, weights 2:4 structured
// sparse, all NVFP4, via CUTLASS sm120 blockscaled sparse (example 80b type).
// The dense chain (fp4_chain_cutlass.cu) tops out near 80% of the dense wall;
// this crosses it with the OMMA.SF.SP 4X instruction (marketed 1406 sparse TOPS).
//
// Orientation: weights are the A operand (the sparse side in CUTLASS sm120),
// activations are B. GEMM1 emits FP4 D1 + scales in its epilogue; GEMM2 reads
// D1's buffer directly as its B operand (bytes are bytes, contiguous, aligned).
// Throughput-only, same shortcuts as the dense chain: GEMM2's input scales are
// a prepared buffer, values and scale-format matching are accuracy details.
// Residual via GEMM2's epilogue beta (bf16 C, the conservative read cost).
// Weight compression (compressor kernel) runs offline, untimed: weights are
// static in inference, compression is a load-time cost.
//
// TOOLKIT: CUDA 13.1 ptxas rejects the sm120 sparse FP4 MMA. Build with a
// 13.3+ ptxas spliced into the 13.1 pipeline (see README sparse section).
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
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"

using namespace cute;

using ElementA     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // weights (sparse operand)
using LayoutATag   = cutlass::layout::RowMajor;
constexpr int AlignmentA = 64;
using ElementB     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // activations
using LayoutBTag   = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD     = cutlass::float_e2m1_t;
using ElementSFD   = cutlass::float_ue4m3_t;
using ElementC     = cutlass::bfloat16_t;                          // residual
#ifdef COLMAJOR_CD   // 80b's original C/D layout; row-major measured faster (see README)
using LayoutCTag   = cutlass::layout::ColumnMajor;
using LayoutDTag   = cutlass::layout::ColumnMajor;
#else
using LayoutCTag   = cutlass::layout::RowMajor;
using LayoutDTag   = cutlass::layout::RowMajor;
#endif
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

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
    cutlass::epilogue::fusion::LinCombBlockScaleFactor<
        OutputSFVectorSize, ElementD, ElementAccumulator, ElementSFD, LayoutDTag, ElementC>
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelScheduleType>::CollectiveOp;

// tile scheduler: -DTILESCHED=cutlass::gemm::StreamKScheduler for tail-wave recovery
#ifndef TILESCHED
#define TILESCHED void
#endif
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, TILESCHED>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
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

// One sparse weight operand: dense-random fill, 2:4 zero mask, device compress.
struct SparseWeight {
  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> dense, comp;
  cutlass::HostTensor<ElementE, cutlass::layout::PackedVectorLayout> meta;
  LayoutA layout_A; LayoutE layout_E;
  void init(int M, int K, int seed, StrideA stride_A, cudaStream_t s) {
    auto workload = make_shape(M, 1, K, 1);  // (M, N-unused, K, L) for the compressor
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
  int swz   = argc>3 ? atoi(argv[3]) : 1;   // scheduler max swizzle
  int ras   = argc>4 ? atoi(argv[4]) : 0;   // raster: 0 heuristic, 1 AlongM, 2 AlongN
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

  // weights: NBUF independent GEMM pairs, compressed offline. Cycle them during
  // timing so 2*NBUF compressed weights exceed L2 (honest L=8-style cold path).
  constexpr int NBUF=4;
  SparseWeight W1[NBUF], W2[NBUF];
  for(int j=0;j<NBUF;j++){ W1[j].init(M,K,11+2*j,stride_A,s); W2[j].init(M,K,22+2*j,stride_A,s); }

  // activations, intermediates, residual, scales
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
    args.scheduler.raster_order = (ras==1)?RO::AlongM:(ras==2)?RO::AlongN:RO::Heuristic;
    return args;
  };

  // GEMM1: W1*X -> D1 (fp4) + SFD1, beta=0.  GEMM2: W2*D1 -> D2 + residual C2, beta=1.
  Gemm* g1[NBUF]; Gemm* g2[NBUF];
  cutlass::device_memory::allocation<uint8_t>* ws1[NBUF]; cutlass::device_memory::allocation<uint8_t>* ws2[NBUF];
  for(int j=0;j<NBUF;j++){
    auto a1=make_args(W1[j],SFA1.device_data(),X.device_data(),C2.device_data(),D1.device_data(),SFD1.device_data(),0.f);
    auto a2=make_args(W2[j],SFA2.device_data(),D1.device_data(),C2.device_data(),D2.device_data(),SFD2.device_data(),1.f);
    g1[j]=new Gemm(); g2[j]=new Gemm();
    ws1[j]=new cutlass::device_memory::allocation<uint8_t>(Gemm::get_workspace_size(a1));
    ws2[j]=new cutlass::device_memory::allocation<uint8_t>(Gemm::get_workspace_size(a2));
    CUTLASS_CHECK(g1[j]->initialize(a1,ws1[j]->get())); CUTLASS_CHECK(g2[j]->initialize(a2,ws2[j]->get()));
  }

  for(int i=0;i<10;i++){ int j=i%NBUF; CUTLASS_CHECK(g1[j]->run(s)); CUTLASS_CHECK(g2[j]->run(s)); }
  cudaStreamSynchronize(s);

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0,s);
  for(int i=0;i<iters;i++){ int j=i%NBUF; CUTLASS_CHECK(g1[j]->run(s)); CUTLASS_CHECK(g2[j]->run(s)); }
  cudaEventRecord(t1,s);
  cudaError_t e=cudaStreamSynchronize(s);
  if(e){ printf("cuda err: %s\n", cudaGetErrorString(e)); return 1; }
  float ms_total=0; cudaEventElapsedTime(&ms_total,t0,t1);

  double ms_pair = double(ms_total)/iters;
  double flop = 4.0*(double)M*N*K;   // dense-equivalent, two GEMMs
  double tf = flop/(ms_pair/1e3)/1e12;
  printf("sparse residual FP4 double GEMM L2-cold x%d N=%d tile=%dx%dx%d %.4f ms/pair %.1f TFLOPS dense-equiv\n",
         NBUF,n,TBM,TBN,TBK,ms_pair,tf);
  return 0;
}
