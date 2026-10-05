// Dense fp4 (nvf4) GEMM variant probe for sm120 — the backward-dW binding wall.
// The trainer/bwd_probe dense GEMM uses 128x128x128 + KernelScheduleAuto (693
// TFLOPS @8192, 3090 lock). This sweeps schedule (Cooperative vs Pingpong) x
// tile (M,N,K) x reg cap, one variant per binary:
//   build_sparse.sh dense_probe.cu /tmp/dN -DDTM=128 -DDTN=128 -DDTK=128 [-DPING] [-maxrregcount=N]
// run: /tmp/dN S iters
#include <cstdio>
#include <functional>
#include <cuda_bf16.h>
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
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"
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
using ElementD=cutlass::bfloat16_t; using LayoutDTag=cutlass::layout::RowMajor;
using ElementC=cutlass::bfloat16_t; using LayoutCTag=cutlass::layout::RowMajor;
using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using DnOp=cutlass::arch::OpClassBlockScaledTensorOp;

#ifndef DTM
#define DTM 128
#endif
#ifndef DTN
#define DTN 128
#endif
#ifndef DTK
#define DTK 128
#endif
using DnTB=Shape<Int<DTM>,Int<DTN>,Int<DTK>>; using CL=Shape<_1,_1,_1>;

#ifdef MXF4
#ifdef PING
using DnSched=cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf4Sm120;
#else
using DnSched=cutlass::gemm::KernelTmaWarpSpecializedMxf4Sm120;
#endif
#else
#ifdef PING
using DnSched=cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
#else
#ifdef AUTOSCHED
using DnSched=cutlass::gemm::collective::KernelScheduleAuto;
#else
using DnSched=cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;
#endif
#endif
#endif

using DnEpi=typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,DnOp,DnTB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
  ElementC,LayoutCTag,128/16,ElementD,LayoutDTag,128/16,
  cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using DnMain=typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag,DnOp,ElementA,LayoutATag,32,ElementB,LayoutBTag,32,Acc,DnTB,CL,
  cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename DnEpi::SharedStorage)>,
  DnSched>::CollectiveOp;
using DnK=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,DnMain,DnEpi,void>;
using DnGemm=cutlass::gemm::device::GemmUniversalAdapter<DnK>;
using DnCfg=typename DnGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static float time_ms(std::function<void(cudaStream_t)> run, cudaStream_t s, int it){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for(int i=0;i<15;i++) run(s); cudaStreamSynchronize(s);
  cudaEventRecord(a,s); for(int i=0;i<it;i++) run(s); cudaEventRecord(b,s); cudaStreamSynchronize(s);
  float ms=0; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}

int main(int argc,char**argv){
  int S=argc>1?atoi(argv[1]):8192, it=argc>2?atoi(argv[2]):100;
  int swz=argc>3?atoi(argv[3]):1; int rast=argc>4?atoi(argv[4]):0;
  int M=S,N=S,K=S;
  cudaStream_t s; cudaStreamCreate(&s);
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> DA,DB;
  cutlass::HostTensor<ElementD,cutlass::layout::PackedVectorLayout> DD;
  cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> DSFA,DSFB;
  auto dlSFA=DnCfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto dlSFB=DnCfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  DA.reset(cutlass::make_Coord(M*K)); DB.reset(cutlass::make_Coord(N*K)); DD.reset(cutlass::make_Coord(M*N));
  DSFA.reset(cutlass::make_Coord(size(filter_zeros(dlSFA)))); DSFB.reset(cutlass::make_Coord(size(filter_zeros(dlSFB))));
  cutlass::reference::host::TensorFillRandomUniform(DA.host_view(),21,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(DB.host_view(),22,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(DSFA.host_view(),23,2,1,0);
  cutlass::reference::host::TensorFillRandomUniform(DSFB.host_view(),24,2,1,0);
  DA.sync_device(); DB.sync_device(); DSFA.sync_device(); DSFB.sync_device();

  auto run_mnk=[&](int m,int n,int k)->float{
    auto dsA=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideA{},{m,k,1});
    auto dsB=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideB{},{n,k,1});
    auto dsC=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideC{},{m,n,1});
    auto dsD=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideD{},{m,n,1});
    auto lSFA=DnCfg::tile_atom_to_shape_SFA(make_shape(m,n,k,1));
    auto lSFB=DnCfg::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
    typename DnGemm::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
      {DA.device_data(),dsA,DB.device_data(),dsB,DSFA.device_data(),lSFA,DSFB.device_data(),lSFB},
      {{1.f,0.f},(ElementC*)nullptr,dsC,DD.device_data(),dsD}};
    using RO=cutlass::gemm::kernel::detail::RasterOrderOptions;
    a.scheduler.max_swizzle_size=swz;
    a.scheduler.raster_order = rast==1?RO::AlongM : rast==2?RO::AlongN : RO::Heuristic;
    static DnGemm g; static cutlass::device_memory::allocation<uint8_t>* ws=nullptr;
    if(!ws) ws=new cutlass::device_memory::allocation<uint8_t>(DnGemm::get_workspace_size(a));
    if(g.can_implement(a)!=cutlass::Status::kSuccess) return -1.f;
    CUTLASS_CHECK(g.initialize(a,ws->get()));
    return time_ms([&](cudaStream_t st){ CUTLASS_CHECK(g.run(st)); }, s, it);
  };

  double g1=2.0*M*N*K/1e12;
  float f  = run_mnk(M,N,K);
  float h  = run_mnk(M,N/2,K);
  const char* sch =
#ifdef PING
    "ping";
#else
#ifdef AUTOSCHED
    "auto";
#else
    "coop";
#endif
#endif
  printf("tile %dx%dx%d %s swz=%d rast=%d S=%d : full %.4f ms %.1f TFLOPS | half-N %.4f ms %.1f TFLOPS-de\n",
         DTM,DTN,DTK,sch,swz,rast,S, f, f>0?g1/(f/1e3):0.0, h, h>0?g1/(h/1e3):0.0);
  return 0;
}
