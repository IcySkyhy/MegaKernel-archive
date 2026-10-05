// Sparse nvf4 GEMM variant probe for sm120 — the forward/dAct rate (goal 1400).
// Baseline: TBM=256, TB=(256,128,256), coop, auto stages (2,2,2) => 1271 @8192.
// Knobs per binary:
//   -DSTM=  tile M (128|256|384)
//   -DSTN=  tile N (128|256)
//   -DSTK=  tile K (256|512)
//   -DCARVE=extra carveout bytes on top of epilogue (forces asymmetric B15 stages / StagesE=0)
//   -DPHASE=phase-stagger the two consumer WGs (generated project-local collective)
//   -DHASH=print raw output hash; baseline and PHASE must match bit-exactly
//   -DGEMM_ONLY=skip compressor/backprop composite benches (fast variant sweeps)
//   -maxrregcount=N (ptxas reg cap)
// build_sparse.sh sparse_probe.cu /tmp/sX -DSTM=256 ... ; run: /tmp/sX S iters
#include <cstdio>
#include <cstdint>
#include <functional>
#include <cuda_bf16.h>
#include <chrono>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
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
#ifdef PHASE
#include "sm120_sparse_phase.hpp"
#endif
using namespace cute;

using EA=cutlass::float_e2m1_t;
#ifdef MXF4
using ESF=cutlass::float_ue8m0_t;
using ElementA=cutlass::mx_float4_t<EA>; using LayoutATag=cutlass::layout::RowMajor;
using ElementB=cutlass::mx_float4_t<EA>; using LayoutBTag=cutlass::layout::ColumnMajor;
#else
using ESF=cutlass::float_ue4m3_t;
using ElementA=cutlass::nv_float4_t<EA>; using LayoutATag=cutlass::layout::RowMajor;
using ElementB=cutlass::nv_float4_t<EA>; using LayoutBTag=cutlass::layout::ColumnMajor;
#endif
constexpr int AA=64, AB=32;
#ifdef CVOID
using ElementC=void; using CStorage=cutlass::bfloat16_t;
#else
using ElementC=cutlass::bfloat16_t; using CStorage=ElementC;
#endif
#ifdef D8
using ElementD=cutlass::float_e4m3_t;
#else
#ifdef D4RAW
using ElementD=cutlass::float_e2m1_t;
#else
using ElementD=cutlass::bfloat16_t;
#endif
#endif
using LayoutCTag=cutlass::layout::RowMajor;
#ifdef COLD
using LayoutDTag=cutlass::layout::ColumnMajor;
#else
using LayoutDTag=cutlass::layout::RowMajor;
#endif
constexpr int AD=128/cutlass::sizeof_bits<ElementD>::value;
#ifdef CVOID
constexpr int AC=1;
#else
constexpr int AC=128/cutlass::sizeof_bits<ElementC>::value;
#endif
using ElementE=uint8_t; using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using SpOp=cutlass::arch::OpClassBlockScaledSparseTensorOp;
#ifdef ACC24
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120;
#else
#ifdef MXF4
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedMxf4Sm120;
#else
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
#endif
#endif

#ifndef STM
#define STM 256
#endif
#ifndef STN
#define STN 128
#endif
#ifndef STK
#define STK 256
#endif
#ifndef CARVE
#define CARVE 0
#endif
using TB=Shape<Int<STM>,Int<STN>,Int<STK>>; using CL=Shape<_1,_1,_1>;

#ifdef EPIM
using EpiTile=Shape<Int<EPIM>,Int<EPIN>>;
#else
using EpiTile=cutlass::epilogue::collective::EpilogueTileAuto;
#endif
using SpEpi=typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,SpOp,TB,CL,EpiTile,Acc,Acc,
  ElementC,LayoutCTag,AC,ElementD,LayoutDTag,AD,
  cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120>::CollectiveOp;
#ifdef MANSTAGE
using StagePolicy=cutlass::gemm::collective::StageCount<MANSTAGE>;
#else
using StagePolicy=cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename SpEpi::SharedStorage)+CARVE>;
#endif
using SpMain=typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag,SpOp,ElementA,LayoutATag,AA,ElementB,LayoutBTag,AB,Acc,TB,CL,
  StagePolicy, SpSched>::CollectiveOp;
#if defined(SCHED) || defined(OVSTAGE)
// Rebind the built CollectiveMma to a manual dispatch policy while retaining
// the builder-generated copies/layouts. OVSTAGE attacks the 1-CTA/SM wall by
// shrinking TBM=128 from its auto 3 A/B/E stages to 2; pair with reg cap <=85.
namespace cutlass::gemm::collective{
template<class> struct RebindMma;
template<class DP,class... R> struct RebindMma<CollectiveMma<DP,R...>>{
  template<class NDP> using with=CollectiveMma<NDP,R...>; };
}
using DP0=typename SpMain::DispatchPolicy;
#ifdef OVSTAGE
#ifdef OVSE
#ifdef OVSB
constexpr int OSA=OVSTAGE, OSB=OVSB, OSE=OVSE;
#else
constexpr int OSA=OVSTAGE, OSB=OVSTAGE, OSE=OVSE;
#endif
#else
constexpr int OSA=OVSTAGE, OSB=OVSTAGE, OSE=OVSTAGE;
#endif
#else
constexpr int OSA=DP0::StagesA, OSB=DP0::StagesB, OSE=DP0::StagesE;
#endif
#ifdef SCHED
constexpr int OSS=SCHED;
#else
constexpr int OSS=2;
#endif
using DPn=cutlass::gemm::MainloopSm120TmaWarpSpecializedSparseBlockScaled<OSA,OSB,OSE,OSS,CL>;
using SpMainX=typename cutlass::gemm::collective::RebindMma<SpMain>::template with<DPn>;
#else
using SpMainX=SpMain;
#endif
#ifdef ATOMM
namespace cutlass::gemm::collective{
template<class> struct RebindTiled;
template<class DP,class TS,class EPA,class LPA,class EPB,class SPB,class TM,
         class GCA,class SLA,class SCA,class TA,class GCB,class SLB,class SCB,class TBX>
struct RebindTiled<CollectiveMma<DP,TS,EPA,LPA,EPB,SPB,TM,GCA,SLA,SCA,TA,GCB,SLB,SCB,TBX>>{
  using Perm=decltype(cute::make_tile(TM{}.template permutation_mnk<0>(),TM{}.template permutation_mnk<1>(),TM{}.template permutation_mnk<2>()));
  using NTM=cute::TiledMMA<typename TM::Atom,cute::Layout<cute::Shape<cute::Int<ATOMM>,cute::Int<ATOMN>,cute::_1>>,Perm>;
  using type=CollectiveMma<DP,TS,EPA,LPA,EPB,SPB,NTM,GCA,SLA,SCA,TA,GCB,SLB,SCB,TBX>;
};
}
using SpMainY=typename cutlass::gemm::collective::RebindTiled<SpMainX>::type;
#else
using SpMainY=SpMainX;
#endif
#ifdef PHASE
namespace cutlass::gemm::collective {
template<class> struct RebindPhase;
template<int SA,int SB,int SE,int SS,class CS,class... R>
struct RebindPhase<CollectiveMma<
    MainloopSm120TmaWarpSpecializedSparseBlockScaled<SA,SB,SE,SS,CS>,R...>> {
  using type=PhaseCollectiveMma<SA,SB,SE,SS,CS,R...>;
};
}
using SpMainZ=typename cutlass::gemm::collective::RebindPhase<SpMainY>::type;
#else
using SpMainZ=SpMainY;
#endif
#ifndef TILESCHED
#define TILESCHED void
#endif
using SpK=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,SpMainZ,SpEpi,TILESCHED>;
using SpGemm=cutlass::gemm::device::GemmUniversalAdapter<SpK>;
using StrideA=typename SpGemm::GemmKernel::StrideA; using LayoutA=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutA;
using StrideB=typename SpGemm::GemmKernel::StrideB; using StrideC=typename SpGemm::GemmKernel::StrideC; using StrideD=typename SpGemm::GemmKernel::StrideD;
using LayoutE=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutE;
using SpCfg=typename SpGemm::GemmKernel::CollectiveMainloop::SparseConfig;
using CompU=cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg>;
using CompK=cutlass::transform::kernel::StructuredSparseCompressor<Shape<int,int,int,int>,EA,LayoutATag,SpCfg,ArchTag>;
using Comp=cutlass::transform::device::TransformUniversalAdapter<CompK>;

#ifdef OCC2
template<class Op>
__global__ __launch_bounds__(Op::MaxThreadsPerBlock,2)
void occ2_entry(CUTLASS_GRID_CONSTANT typename Op::Params const params){
  extern __shared__ char smem[]; Op op; op(params,smem);
}
#endif


// Fast activation-grad 2:4 compressor: e2m1 magnitude = low 3 bits as uint.
// Thread handles 32 contiguous fp4 along k: reads uint4 (16B), writes uint2 (8B)
// compressed + uint32 (4B) selector meta. Pure memory-shuffle, no smem needed.
__global__ void fast_compress24(const uint4* __restrict__ src, uint2* __restrict__ dst,
                                uint32_t* __restrict__ meta, size_t nthr){
  size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(g>=nthr) return;
  uint4 v=src[g];
  uint32_t w[4]={v.x,v.y,v.z,v.w};
  uint32_t out[2]={0,0}; uint32_t sel=0;
  #pragma unroll
  for(int gi=0; gi<8; ++gi){                    // 8 groups of 4 nibbles
    uint32_t word=w[gi>>1]; int sh=(gi&1)*16;
    uint32_t q=(word>>sh)&0xFFFF;               // 4 nibbles
    uint32_t n0=q&0xF,n1=(q>>4)&0xF,n2=(q>>8)&0xF,n3=(q>>12)&0xF;
    uint32_t m0=n0&7,m1=n1&7,m2=n2&7,m3=n3&7;   // e2m1 magnitude
    // pick two largest magnitudes (stable): indices i>j
    int a=0; uint32_t ma=m0;
    if(m1>ma){ma=m1;a=1;} if(m2>ma){ma=m2;a=2;} if(m3>ma){ma=m3;a=3;}
    int b=(a==0)?1:0; uint32_t mb=(b==0)?m0:m1;
    if(a!=1&&1!=b&&m1>mb){mb=m1;b=1;}
    if(a!=2&&2!=b&&m2>mb){mb=m2;b=2;}
    if(a!=3&&3!=b&&m3>mb){mb=m3;b=3;}
    int lo=a<b?a:b, hi=a<b?b:a;
    uint32_t klo=(q>>(lo*4))&0xF, khi=(q>>(hi*4))&0xF;
    out[gi>>2] |= (klo|(khi<<4)) << ((gi&3)*8); // 2 nibbles per group
    sel |= ((uint32_t)(lo|(hi<<2))) << (gi*4);  // 4-bit selector per group
  }
  dst[g]=make_uint2(out[0],out[1]);
  meta[g]=sel;
}

static float time_ms(std::function<void(cudaStream_t)> run, cudaStream_t s, int it){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for(int i=0;i<15;i++) run(s); cudaStreamSynchronize(s);
  cudaEventRecord(a,s); for(int i=0;i<it;i++) run(s); cudaEventRecord(b,s); cudaStreamSynchronize(s);
  float ms=0; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}

int main(int argc,char**argv){
  int S=argc>1?atoi(argv[1]):8192, it=argc>2?atoi(argv[2]):100;
  int swz=argc>3?atoi(argv[3]):1; int rast=argc>4?atoi(argv[4]):0;  // rast 0=heur 1=M 2=N
  int M=S,N=S,K=argc>5?atoi(argv[5]):S; // optional deeper contraction K
  cudaStream_t s; cudaStreamCreate(&s);
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  auto sA=cutlass::make_cute_packed_stride(StrideA{},{M,K,1});
  auto sB=cutlass::make_cute_packed_stride(StrideB{},{N,K,1});
  auto sC=cutlass::make_cute_packed_stride(StrideC{},{M,N,1});
  auto sD=cutlass::make_cute_packed_stride(StrideD{},{M,N,1});
  using Cfg=typename SpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto lSFA=Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB=Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto wl=make_shape(M,1,K,1); CompU util(wl,sA);
  LayoutA lA=SpCfg::fill_layoutA(wl); LayoutE lE=SpCfg::fill_layoutE(wl);
  cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> Wdense,Wcomp,B;
  cutlass::HostTensor<ElementD,cutlass::layout::PackedVectorLayout> D;
  cutlass::HostTensor<ElementE,cutlass::layout::PackedVectorLayout> meta;
  cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> SFA,SFB;
  cutlass::HostTensor<CStorage,cutlass::layout::PackedVectorLayout> C;
  Wdense.reset(cutlass::make_Coord(M*K));
  Wcomp.reset(cutlass::make_Coord(util.get_tensorA_m_physical()*util.get_tensorA_k_physical()));
  meta.reset(cutlass::make_Coord(util.get_metadata_m_physical()*util.get_metadata_k_physical()));
  B.reset(cutlass::make_Coord(N*K)); D.reset(cutlass::make_Coord(M*N)); C.reset(cutlass::make_Coord(M*N));
  SFA.reset(cutlass::make_Coord(size(filter_zeros(lSFA)))); SFB.reset(cutlass::make_Coord(size(filter_zeros(lSFB))));
  cutlass::reference::host::TensorFillRandomUniform(Wdense.host_view(),11,3,-3,0);
  util.structure_sparse_zero_mask_fill(Wdense.host_data(),12);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(),7,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(SFA.host_view(),8,2,1,0);
  cutlass::reference::host::TensorFillRandomUniform(SFB.host_view(),10,2,1,0);
  Wdense.sync_device(); B.sync_device(); C.sync_device(); SFA.sync_device(); SFB.sync_device();

  typename Comp::Arguments cargs{{M,1,K,1},{Wdense.device_data(),sA,Wcomp.device_data(),meta.device_data()},{hw}};
  Comp cop; cutlass::device_memory::allocation<uint8_t> cws(Comp::get_workspace_size(cargs));
  CUTLASS_CHECK(cop.can_implement(cargs)); CUTLASS_CHECK(cop.initialize(cargs,cws.get(),s));
  CUTLASS_CHECK(cop.run(s)); cudaStreamSynchronize(s);

#if defined(NOC) || defined(CVOID)
  ElementC* cptr=nullptr;
#else
  ElementC* cptr=C.device_data();
#endif
  typename SpGemm::Arguments spa{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
    {Wcomp.device_data(),lA,B.device_data(),sB,meta.device_data(),lE,SFA.device_data(),lSFA,SFB.device_data(),lSFB},
    {{1.f,0.f},cptr,sC,D.device_data(),sD}};
  using RO=cutlass::gemm::kernel::detail::RasterOrderOptions;
  spa.scheduler.max_swizzle_size=swz;
  spa.scheduler.raster_order = rast==1?RO::AlongM : rast==2?RO::AlongN : RO::Heuristic;
  SpGemm sg; cutlass::device_memory::allocation<uint8_t> sws(SpGemm::get_workspace_size(spa));
  if(sg.can_implement(spa)!=cutlass::Status::kSuccess){ printf("tile %dx%dx%d carve %d: can_implement FALSE\n",STM,STN,STK,CARVE); return 1; }
  CUTLASS_CHECK(sg.initialize(spa,sws.get()));

#ifdef OCC2
  auto kp=sg.params();
  dim3 occgrid=SpK::get_grid_shape(kp), occblock=SpK::get_block_shape();
  CUDA_CHECK(cudaFuncSetAttribute(occ2_entry<SpK>,cudaFuncAttributeMaxDynamicSharedMemorySize,SpK::SharedStorageSize));
  auto runsp=[&](cudaStream_t st){ occ2_entry<SpK><<<occgrid,occblock,SpK::SharedStorageSize,st>>>(kp); };
#else
  auto runsp=[&](cudaStream_t st){ CUTLASS_CHECK(sg.run(st)); };
#endif
#ifdef HASH
  // A phase-scheduling change can race silently and still benchmark faster.
  // Emit a raw-output hash so baseline and PHASE binaries must be bit-exact.
  runsp(s); cudaStreamSynchronize(s); D.sync_host();
  const unsigned char* hp=reinterpret_cast<const unsigned char*>(D.host_data());
  size_t hn=(size_t)M*N*cutlass::sizeof_bits<ElementD>::value/8;
  uint64_t hh=1469598103934665603ull;
  for(size_t i=0;i<hn;i++){ hh^=hp[i]; hh*=1099511628211ull; }
  printf("output hash: %016llx (%zu bytes)\n",(unsigned long long)hh,hn);
#endif
  float sp = time_ms(runsp, s, it);
  double g1=2.0*M*N*K/1e12;
  printf("sparse tile %dx%dx%d carve %d StagesA/B/E %d/%d/%d swz=%d rast=%d S=%d : %.4f ms  %.1f TFLOPS-de\n",
    STM,STN,STK,CARVE,
    (int)SpGemm::GemmKernel::CollectiveMainloop::DispatchPolicy::StagesA,
    (int)SpGemm::GemmKernel::CollectiveMainloop::DispatchPolicy::StagesB,
    (int)SpGemm::GemmKernel::CollectiveMainloop::DispatchPolicy::StagesE,
    swz,rast,S,sp,g1/(sp/1e3));

#ifndef GEMM_ONLY
  // ---- custom fast 2:4 compressor: standalone rate ----
  // Dense fp4 grad source S*S/2 bytes; compressed S*S/4 bytes; meta S*S/8 bytes.
  uint4* gsrc; uint2* gdst; uint32_t* gmeta; size_t nthr=(size_t)M*K/32;
  cudaMalloc(&gsrc,(size_t)M*K/2); cudaMalloc(&gdst,(size_t)M*K/4); cudaMalloc(&gmeta,(size_t)M*K/8);
  cudaMemset(gsrc,0x5A,(size_t)M*K/2);
  int cthr=256; size_t cblk=(nthr+cthr-1)/cthr;
  float fc = time_ms([&](cudaStream_t st){ fast_compress24<<<(int)cblk,cthr,0,st>>>(gsrc,gdst,gmeta,nthr); }, s, it);
  double cbytes=(double)M*K/2 + (double)M*K/4 + (double)M*K/8;
  printf("  fast_compress24: %.4f ms  (%.0f GB/s;  CUTLASS compressor for same tensor: measure below)\n", fc, cbytes/(fc/1e3)/1e9);
  float cm = time_ms([&](cudaStream_t st){ CUTLASS_CHECK(cop.run(st)); }, s, it);
  printf("  CUTLASS compress: %.4f ms\n", cm);

  // ---- ALL-SPARSE L=8 backward: 32 sparse GEMMs + 16 fast grad-compresses ----
  // dW = dD . act^T contracts over batch: 2:4-prune the GRAD (sparse A operand),
  // activations stay dense B, dW output dense. All 32 backward GEMMs then run at
  // the sparse rate. Grad compresses are fresh each step (2/layer): serial vs
  // overlapped on a 2nd stream.
  {
    const int NG=32, NCMP=16;
    cudaStream_t sc; cudaStreamCreate(&sc);
    double bflop=(double)NG*g1;
    auto bench=[&](std::function<void()> body)->double{
      for(int w=0;w<4;w++) body();
      cudaDeviceSynchronize();
      auto t0=std::chrono::steady_clock::now();
      for(int r=0;r<it;r++) body();
      cudaDeviceSynchronize();
      return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/it;
    };
    double msg=bench([&]{ for(int i=0;i<NG;i++) runsp(s); });
    double mss=bench([&]{ for(int i=0;i<NG;i++){ runsp(s); if(i<NCMP) fast_compress24<<<(int)cblk,cthr,0,s>>>(gsrc,gdst,gmeta,nthr);} });
    double mso=bench([&]{ for(int i=0;i<NG;i++){ runsp(s); if(i<NCMP) fast_compress24<<<(int)cblk,cthr,0,sc>>>(gsrc,gdst,gmeta,nthr);} });
    printf("\n=== ALL-SPARSE L=8 backward (32 sparse GEMMs + 16 fast grad-compress) @ S=%d ===\n", S);
    printf("  GEMM-only                 : %.4f ms/backward   %6.1f TFLOPS backprop\n", msg, bflop/(msg/1e3));
    printf("  + 16 fast-compress SERIAL : %.4f ms/backward   %6.1f TFLOPS backprop\n", mss, bflop/(mss/1e3));
    printf("  + 16 fast-compress OVERLAP: %.4f ms/backward   %6.1f TFLOPS backprop  (hidden %.0f%%)\n",
           mso, bflop/(mso/1e3), (mss-mso)/(mss-msg+1e-9)*100.0);
    // FULL L=8 forward in the same accounting: 16 sparse GEMMs (D1,D2 per layer)
    double msf=bench([&]{ for(int i=0;i<16;i++) runsp(s); });
    printf("  FULL L=8 forward (16 sparse GEMMs): %.4f ms  %6.1f TFLOPS\n", msf, 16.0*g1/(msf/1e3));
  }
#endif
  return 0;
}
