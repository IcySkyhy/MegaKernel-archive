// Honest training-step throughput probe. Measures, at one square size S, the
// rates that a REAL fp4 training step depends on (not the inference-forward-only
// numbers). Two output-dtype paths via -DBF16OUT:
//   default : ElementD = fp4 + block-scale SF quant epilogue (chain-style output)
//   BF16OUT : ElementD = bf16, plain linear-combination epilogue (requant path)
// Reports: sparse GEMM rate (weight = sparse A operand, forward),
//          dense  GEMM rate (no sparse operand, backward dW = g @ act^T),
//          weight compression cost (per weight, PER STEP in training).
// Build: 13.3 ptxas splice (build_sparse.sh).
#include <iostream>
#include <cstdio>
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
using namespace cute;

using EA=cutlass::float_e2m1_t; using ESF=cutlass::float_ue4m3_t;
using ElementA=cutlass::nv_float4_t<EA>; using LayoutATag=cutlass::layout::RowMajor;
using ElementB=cutlass::nv_float4_t<EA>; using LayoutBTag=cutlass::layout::ColumnMajor;
constexpr int AA=64, AB=32;
#ifdef BF16OUT
using ElementC=cutlass::bfloat16_t; using ElementD=cutlass::bfloat16_t;
#else
using ElementC=cutlass::bfloat16_t; using ElementD=EA;
#endif
using LayoutCTag=cutlass::layout::RowMajor; using LayoutDTag=cutlass::layout::RowMajor;
constexpr int AD=128/cutlass::sizeof_bits<ElementD>::value, AC=128/cutlass::sizeof_bits<ElementC>::value;
using ElementE=uint8_t; using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using SpOp=cutlass::arch::OpClassBlockScaledSparseTensorOp;
using DnOp=cutlass::arch::OpClassBlockScaledTensorOp;
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int SFVEC=16;
#ifndef TBM
#define TBM 256
#endif
using TB=Shape<Int<TBM>,_128,_256>; using CL=Shape<_1,_1,_1>;

// ---------- sparse GEMM (weight = sparse A) ----------
#ifdef BF16OUT
using SpEpi=typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,SpOp,TB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
  ElementC,LayoutCTag,AC,ElementD,LayoutDTag,AD,
  cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120>::CollectiveOp;
#else
using SpEpi=typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,SpOp,TB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
  ElementC,LayoutCTag,AC,ElementD,LayoutDTag,AD,
  cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
  cutlass::epilogue::fusion::LinCombBlockScaleFactor<SFVEC,ElementD,Acc,ESF,LayoutDTag,ElementC>>::CollectiveOp;
#endif
using SpMain=typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag,SpOp,ElementA,LayoutATag,AA,ElementB,LayoutBTag,AB,Acc,TB,CL,
  cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename SpEpi::SharedStorage)>,
  SpSched>::CollectiveOp;
using SpK=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,SpMain,SpEpi,void>;
using SpGemm=cutlass::gemm::device::GemmUniversalAdapter<SpK>;
using StrideA=typename SpGemm::GemmKernel::StrideA; using LayoutA=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutA;
using StrideB=typename SpGemm::GemmKernel::StrideB; using StrideC=typename SpGemm::GemmKernel::StrideC; using StrideD=typename SpGemm::GemmKernel::StrideD;
using LayoutE=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutE;
using SpCfg=typename SpGemm::GemmKernel::CollectiveMainloop::SparseConfig;
using CompU=cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg>;
using CompK=cutlass::transform::kernel::StructuredSparseCompressor<Shape<int,int,int,int>,EA,LayoutATag,SpCfg,ArchTag>;
using Comp=cutlass::transform::device::TransformUniversalAdapter<CompK>;
using SfdCfg=cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVEC>;

// ---------- dense GEMM (both operands dense: backward dW), own 128-cube tile ----------
using DnTB=Shape<_128,_128,_128>;
using DnEpi=typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,DnOp,DnTB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
  ElementC,LayoutCTag,AC,cutlass::bfloat16_t,LayoutDTag,128/16,
  cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using DnMain=typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag,DnOp,ElementA,LayoutATag,32,ElementB,LayoutBTag,32,Acc,DnTB,CL,
  cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename DnEpi::SharedStorage)>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using DnK=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,DnMain,DnEpi,void>;
using DnGemm=cutlass::gemm::device::GemmUniversalAdapter<DnK>;
using DnSFA=typename DnGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

int main(int argc,char**argv){
  int S=argc>1?atoi(argv[1]):8192, it=argc>2?atoi(argv[2]):200;
  int M=S,N=S,K=S;
  cudaStream_t s; cudaStreamCreate(&s);
  auto sA=cutlass::make_cute_packed_stride(StrideA{},{M,K,1});
  auto sB=cutlass::make_cute_packed_stride(StrideB{},{N,K,1});
  auto sC=cutlass::make_cute_packed_stride(StrideC{},{M,N,1});
  auto sD=cutlass::make_cute_packed_stride(StrideD{},{M,N,1});
  using Cfg=typename SpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto lSFA=Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB=Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto lSFD=SfdCfg::tile_atom_to_shape_SFD(make_shape(M,N,K,1));
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  // sparse weight
  auto wl=make_shape(M,1,K,1); CompU util(wl,sA);
  LayoutA lA=SpCfg::fill_layoutA(wl); LayoutE lE=SpCfg::fill_layoutE(wl);
  cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> Wdense,Wcomp,B;
  cutlass::HostTensor<ElementD,cutlass::layout::PackedVectorLayout> D;
  cutlass::HostTensor<ElementE,cutlass::layout::PackedVectorLayout> meta;
  cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> SFA,SFB,SFD;
  cutlass::HostTensor<ElementC,cutlass::layout::PackedVectorLayout> C;
  cutlass::HostTensor<Acc,cutlass::layout::PackedVectorLayout> Norm;
  Wdense.reset(cutlass::make_Coord(M*K));
  Wcomp.reset(cutlass::make_Coord(util.get_tensorA_m_physical()*util.get_tensorA_k_physical()));
  meta.reset(cutlass::make_Coord(util.get_metadata_m_physical()*util.get_metadata_k_physical()));
  B.reset(cutlass::make_Coord(N*K)); D.reset(cutlass::make_Coord(M*N)); C.reset(cutlass::make_Coord(M*N));
  SFA.reset(cutlass::make_Coord(size(filter_zeros(lSFA)))); SFB.reset(cutlass::make_Coord(size(filter_zeros(lSFB))));
  SFD.reset(cutlass::make_Coord(size(filter_zeros(lSFD)))); Norm.reset(cutlass::make_Coord(1));
  cutlass::reference::host::TensorFillRandomUniform(Wdense.host_view(),11,3,-3,0);
  util.structure_sparse_zero_mask_fill(Wdense.host_data(),12);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(),7,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(SFA.host_view(),8,2,1,0);
  cutlass::reference::host::TensorFillRandomUniform(SFB.host_view(),10,2,1,0);
  Norm.at(cutlass::make_Coord(0))=2.f;
  Wdense.sync_device(); B.sync_device(); C.sync_device(); SFA.sync_device(); SFB.sync_device(); SFD.sync_device(); Norm.sync_device();

  // ---- compression cost (per weight, per step in training) ----
  typename Comp::Arguments cargs{{M,1,K,1},{Wdense.device_data(),sA,Wcomp.device_data(),meta.device_data()},{hw}};
  Comp cop; cutlass::device_memory::allocation<uint8_t> cws(Comp::get_workspace_size(cargs));
  CUTLASS_CHECK(cop.can_implement(cargs)); CUTLASS_CHECK(cop.initialize(cargs,cws.get(),s));
  for(int i=0;i<10;i++) CUTLASS_CHECK(cop.run(s)); cudaStreamSynchronize(s);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a,s); for(int i=0;i<it;i++) CUTLASS_CHECK(cop.run(s)); cudaEventRecord(b,s); cudaStreamSynchronize(s);
  float cms=0; cudaEventElapsedTime(&cms,a,b); cms/=it;

  // ---- sparse GEMM ----
  typename SpGemm::Arguments spa{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
    {Wcomp.device_data(),lA,B.device_data(),sB,meta.device_data(),lE,SFA.device_data(),lSFA,SFB.device_data(),lSFB},
    {{1.f,0.f},C.device_data(),sC,D.device_data(),sD}};
#ifndef BF16OUT
  spa.epilogue.thread.block_scale_factor_ptr=SFD.device_data();
  spa.epilogue.thread.norm_constant_ptr=Norm.device_data();
#endif
  SpGemm sg; cutlass::device_memory::allocation<uint8_t> sws(SpGemm::get_workspace_size(spa));
  CUTLASS_CHECK(sg.can_implement(spa)); CUTLASS_CHECK(sg.initialize(spa,sws.get()));
  for(int i=0;i<20;i++) CUTLASS_CHECK(sg.run(s)); cudaStreamSynchronize(s);
  cudaEventRecord(a,s); for(int i=0;i<it;i++) CUTLASS_CHECK(sg.run(s)); cudaEventRecord(b,s); cudaStreamSynchronize(s);
  float sms=0; cudaEventElapsedTime(&sms,a,b); sms/=it;

  // ---- dense GEMM (backward dW: both operands dense activations) ----
  cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> DA,DB;
  cutlass::HostTensor<cutlass::bfloat16_t,cutlass::layout::PackedVectorLayout> DD;
  cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> DSFA,DSFB;
  auto dsA=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideA{},{M,K,1});
  auto dsB=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideB{},{N,K,1});
  auto dsC=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideC{},{M,N,1});
  auto dsD=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideD{},{M,N,1});
  auto dlSFA=DnSFA::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto dlSFB=DnSFA::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  DA.reset(cutlass::make_Coord(M*K)); DB.reset(cutlass::make_Coord(N*K)); DD.reset(cutlass::make_Coord(M*N));
  DSFA.reset(cutlass::make_Coord(size(filter_zeros(dlSFA)))); DSFB.reset(cutlass::make_Coord(size(filter_zeros(dlSFB))));
  cutlass::reference::host::TensorFillRandomUniform(DA.host_view(),1,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(DB.host_view(),2,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(DSFA.host_view(),3,2,1,0);
  cutlass::reference::host::TensorFillRandomUniform(DSFB.host_view(),4,2,1,0);
  DA.sync_device(); DB.sync_device(); DSFA.sync_device(); DSFB.sync_device();
  typename DnGemm::Arguments dna{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
    {DA.device_data(),dsA,DB.device_data(),dsB,DSFA.device_data(),dlSFA,DSFB.device_data(),dlSFB},
    {{1.f,0.f},(cutlass::bfloat16_t*)nullptr,dsC,DD.device_data(),dsD}};
  DnGemm dg; cutlass::device_memory::allocation<uint8_t> dws(DnGemm::get_workspace_size(dna));
  cutlass::Status st=dg.can_implement(dna);
  float dms=-1;
  if(st==cutlass::Status::kSuccess){
    CUTLASS_CHECK(dg.initialize(dna,dws.get()));
    for(int i=0;i<20;i++) CUTLASS_CHECK(dg.run(s)); cudaStreamSynchronize(s);
    cudaEventRecord(a,s); for(int i=0;i<it;i++) CUTLASS_CHECK(dg.run(s)); cudaEventRecord(b,s); cudaStreamSynchronize(s);
    cudaEventElapsedTime(&dms,a,b); dms/=it;
  }

  double gf=2.0*M*N*K;
  printf("S=%d  out=%s  tile=%dx128x256\n", S,
#ifdef BF16OUT
    "bf16",
#else
    "fp4+SF",
#endif
    TBM);
  printf("  sparse GEMM (weight=A): %.4f ms  %.1f TFLOPS dense-equiv\n", sms, gf/(sms/1e3)/1e12);
  if(dms>0) printf("  dense  GEMM (bwd dW) : %.4f ms  %.1f TFLOPS\n", dms, gf/(dms/1e3)/1e12);
  else      printf("  dense  GEMM (bwd dW) : can_implement=false\n");
  printf("  weight compress/call  : %.4f ms  (%.1f%% of a sparse GEMM)\n", cms, 100.0*cms/sms);
  return 0;
}
