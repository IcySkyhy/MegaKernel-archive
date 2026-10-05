// End-to-end fp4 TRAINING-STEP throughput probe for the residual double GEMM.
// Grounds the "730-930 projection" in measured component GEMMs, and measures the
// structured-sparse dW ceiling (a half-N dense GEMM = the exact compute a
// tile-shared 2:4 mask lets the dW GEMM do).
//
// Per layer the step is 6 GEMMs, each 2*S^3 FLOPs (M=N=K=batch=S):
//   fwd:  D1 = W1 . X          (sparse, weight = A)
//         D2 = W2 . D1 + X     (sparse, residual)
//   bwd:  dW2 = dD2 . D1^T     (weight is the OUTPUT -> dense, OR half-N if tile-shared 2:4)
//         dD1 = W2^T . dD2     (weight is the operand -> sparse IFF transposable 2:4)
//         dW1 = dD1 . X^T      (dense, or half-N)
//         dX  = W1^T . dD1     (sparse IFF transposable 2:4)
// Building blocks measured: sparse GEMM (fwd/dAct), dense GEMM full (dW dense
// ceiling), dense GEMM half-N (dW tile-shared-mask 2x ceiling), weight compress.
// Build: build_sparse.sh bwd_probe.cu /tmp/bwd -DTBM=256
#include <iostream>
#include <cstdio>
#include <functional>
#include <chrono>
#include <vector>
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
using ElementC=cutlass::bfloat16_t; using ElementD=cutlass::bfloat16_t;   // requant path: bf16 out
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

// ---------- sparse GEMM (weight = sparse A), bf16 out ----------
using SpEpi=typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag,SpOp,TB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
  ElementC,LayoutCTag,AC,ElementD,LayoutDTag,AD,
  cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120>::CollectiveOp;
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

// ---------- dense GEMM (both operands dense: backward dW), 128-cube tile ----------
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

// deployable activation requant: bf16 -> fp4 + per-16 ue4m3 SF. Memory-bound
// (read N*K bf16, write N*K/2 fp4 + SF). One thread per (n, 16-k block).
__global__ void requant_bf16_fp4(const __nv_bfloat16* __restrict__ src,
                                 uint8_t* __restrict__ dst, ESF* __restrict__ sf, int N, int K){
  int Kb=K/16; size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(g>=(size_t)N*Kb) return;
  int n=g/Kb, k0=(g%Kb)*16;
  const uint4* pr=reinterpret_cast<const uint4*>(src+(size_t)n*K+k0);   // 2x uint4 = 16 bf16
  __nv_bfloat16 xb[16];
  reinterpret_cast<uint4*>(xb)[0]=pr[0]; reinterpret_cast<uint4*>(xb)[1]=pr[1];
  float x[16], amax=0.f;
  #pragma unroll
  for(int v=0;v<16;v++){ x[v]=__bfloat162float(xb[v]); amax=fmaxf(amax,fabsf(x[v])); }
  float pv=amax*(1.f/6.f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.f/deq:0.f;
  sf[g]=q;
  uint32_t w[2]={0,0};
  #pragma unroll
  for(int v=0;v<16;v++){ uint32_t c=EA(x[v]*rcp).raw()&0xF; w[v>>3]|=c<<((v&7)*4); }
  reinterpret_cast<uint2*>(dst)[g]=make_uint2(w[0],w[1]);
}

static float time_ms(std::function<void(cudaStream_t)> run, cudaStream_t s, int it){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for(int i=0;i<20;i++) run(s); cudaStreamSynchronize(s);
  cudaEventRecord(a,s); for(int i=0;i<it;i++) run(s); cudaEventRecord(b,s); cudaStreamSynchronize(s);
  float ms=0; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b); return ms/it;
}

int main(int argc,char**argv){
  int S=argc>1?atoi(argv[1]):8192, it=argc>2?atoi(argv[2]):200;
  int M=S,N=S,K=S;
  cudaStream_t s; cudaStreamCreate(&s);
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  // ---- sparse GEMM (weight = sparse A) ----
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
  cutlass::HostTensor<ElementC,cutlass::layout::PackedVectorLayout> C;
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

  typename SpGemm::Arguments spa{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
    {Wcomp.device_data(),lA,B.device_data(),sB,meta.device_data(),lE,SFA.device_data(),lSFA,SFB.device_data(),lSFB},
    {{1.f,0.f},C.device_data(),sC,D.device_data(),sD}};
  SpGemm sg; cutlass::device_memory::allocation<uint8_t> sws(SpGemm::get_workspace_size(spa));
  CUTLASS_CHECK(sg.can_implement(spa)); CUTLASS_CHECK(sg.initialize(spa,sws.get()));

  // ---- dense GEMM: build args for a given (m,n,k) ----
  cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> DA,DB;
  cutlass::HostTensor<cutlass::bfloat16_t,cutlass::layout::PackedVectorLayout> DD;
  cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> DSFA,DSFB;
  DA.reset(cutlass::make_Coord(M*K)); DB.reset(cutlass::make_Coord(N*K)); DD.reset(cutlass::make_Coord(M*N));
  auto mk_dlSFA=[&](int m,int n,int k){ return DnSFA::tile_atom_to_shape_SFA(make_shape(m,n,k,1)); };
  auto mk_dlSFB=[&](int m,int n,int k){ return DnSFA::tile_atom_to_shape_SFB(make_shape(m,n,k,1)); };
  { auto a=mk_dlSFA(M,N,K), b=mk_dlSFB(M,N,K);
    DSFA.reset(cutlass::make_Coord(size(filter_zeros(a)))); DSFB.reset(cutlass::make_Coord(size(filter_zeros(b)))); }
  cutlass::reference::host::TensorFillRandomUniform(DA.host_view(),1,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(DB.host_view(),2,3,-3,0);
  cutlass::reference::host::TensorFillRandomUniform(DSFA.host_view(),3,2,1,0);
  cutlass::reference::host::TensorFillRandomUniform(DSFB.host_view(),4,2,1,0);
  DA.sync_device(); DB.sync_device(); DSFA.sync_device(); DSFB.sync_device();

  auto run_dense=[&](int m,int n,int k)->float{
    auto dsA=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideA{},{m,k,1});
    auto dsB=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideB{},{n,k,1});
    auto dsC=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideC{},{m,n,1});
    auto dsD=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideD{},{m,n,1});
    auto dlSFA=mk_dlSFA(m,n,k), dlSFB=mk_dlSFB(m,n,k);
    typename DnGemm::Arguments dna{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
      {DA.device_data(),dsA,DB.device_data(),dsB,DSFA.device_data(),dlSFA,DSFB.device_data(),dlSFB},
      {{1.f,0.f},(cutlass::bfloat16_t*)nullptr,dsC,DD.device_data(),dsD}};
    static DnGemm dg; static cutlass::device_memory::allocation<uint8_t>* dws=nullptr;
    if(!dws) dws=new cutlass::device_memory::allocation<uint8_t>(DnGemm::get_workspace_size(dna));
    if(dg.can_implement(dna)!=cutlass::Status::kSuccess) return -1.f;
    CUTLASS_CHECK(dg.initialize(dna,dws->get()));
    return time_ms([&](cudaStream_t st){ CUTLASS_CHECK(dg.run(st)); }, s, it);
  };

  // ---- requant buffers (bf16 activation -> fp4 + SF) ----
  __nv_bfloat16* rq_src; uint8_t* rq_dst; ESF* rq_sf;
  cudaMalloc(&rq_src,(size_t)N*K*2); cudaMalloc(&rq_dst,(size_t)N*K/2); cudaMalloc(&rq_sf,(size_t)N*K/16*sizeof(ESF));
  cudaMemset(rq_src,0,(size_t)N*K*2);
  int rq_thr=256; size_t rq_g=(size_t)N*(K/16); int rq_blocks=(rq_g+rq_thr-1)/rq_thr;

  // ---- measure the building blocks ----
  float sp = time_ms([&](cudaStream_t st){ CUTLASS_CHECK(sg.run(st)); }, s, it);            // sparse fwd / dAct
  float dn = run_dense(M,N,K);                                                              // dense dW (full)
  float dh = run_dense(M,N/2,K);                                                            // dW tile-shared 2:4 ceiling (half N)
  float cm = time_ms([&](cudaStream_t st){ CUTLASS_CHECK(cop.run(st)); }, s, it);           // weight compress / call
  float rq = time_ms([&](cudaStream_t st){ requant_bf16_fp4<<<rq_blocks,rq_thr,0,st>>>(rq_src,rq_dst,rq_sf,N,K); }, s, it); // activation requant / pass

  double g1=2.0*M*N*K/1e12;   // one full GEMM, dense-equiv TFLOP
  printf("\n=== components @ S=%d (tile %dx128x256, requant/bf16-out path) ===\n", S, TBM);
  printf("  sparse GEMM (fwd/dAct)   : %.4f ms  %.1f TFLOPS\n", sp, g1/(sp/1e3));
  printf("  dense  GEMM (dW, full)   : %.4f ms  %.1f TFLOPS\n", dn, g1/(dn/1e3));
  printf("  dense  GEMM (dW, half-N) : %.4f ms  %.1f TFLOPS dense-equiv  <- structured-sparse dW 2x ceiling\n", dh, g1/(dh/1e3));
  printf("  weight compress / call   : %.4f ms\n", cm);
  printf("  activation requant / pass: %.4f ms  (%.0f GB/s, bf16 read + fp4 write)\n", rq, (2.0*M*K + 0.5*M*K)/(rq/1e3)/1e9);

  // ---- compose the per-layer STEP (6 GEMMs = 6*g1 dense-equiv TFLOP) ----
  auto step=[&](const char* name,double gemm_ms,double extra_ms){
    double t=gemm_ms+extra_ms; printf("  %-46s %.3f ms  %.1f TFLOPS\n", name, t, 6*g1/(t/1e3)); };
  printf("\n=== composed training STEP / layer (6 GEMMs, dense-equiv) ===\n");
  printf("  [GEMM-only (pure tensor-core rate)]\n");
  step("plain 2:4 (bwd all dense)",            2*sp+4*dn, 0);
  step("transposable 2:4 (dAct sparse, dW dense)", 4*sp+2*dn, 0);
  step("+ structured-sparse dW (half-N, block mask)", 4*sp+2*dh, 0);
  printf("  [+ amortized compress (recompress every 4 steps => 1 compress/step)]\n");
  step("transposable 2:4 + amortized",         4*sp+2*dn, 1*cm);
  step("+ structured dW + amortized",          4*sp+2*dh, 1*cm);
  printf("  [DEPLOYABLE-serial: + amortized compress + 4 activation requant passes/layer]\n");
  step("transposable 2:4 (deployable, serial requant)",  4*sp+2*dn, 1*cm+4*rq);
  step("+ structured dW (deployable, serial requant)",    4*sp+2*dh, 1*cm+4*rq);

  // ---- overlap test: can requant hide behind a compute-bound GEMM? ----
  // requant alone needs ~800 GB/s of the 896 wall, so concurrent with a GEMM
  // (which uses ~125 GB/s) they contend. Measure: 4 GEMMs serial vs 4 GEMMs on
  // stream A with 16 requant passes on stream B concurrently.
  cudaStream_t s2; cudaStreamCreate(&s2);
  auto run4=[&](bool overlap)->float{
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    for(int i=0;i<10;i++){ CUTLASS_CHECK(sg.run(s)); if(overlap) requant_bf16_fp4<<<rq_blocks,rq_thr,0,s2>>>(rq_src,rq_dst,rq_sf,N,K); }
    cudaStreamSynchronize(s); cudaStreamSynchronize(s2);
    cudaEventRecord(a,s);
    for(int i=0;i<it;i++){ CUTLASS_CHECK(sg.run(s)); if(overlap) requant_bf16_fp4<<<rq_blocks,rq_thr,0,s2>>>(rq_src,rq_dst,rq_sf,N,K); }
    cudaEventRecord(b,s); cudaStreamSynchronize(s); cudaStreamSynchronize(s2);
    float ms=0; cudaEventElapsedTime(&ms,a,b); return ms/it;
  };
  float g_only=run4(false), g_plus=run4(true);
  printf("\n=== requant/GEMM overlap (1 sparse GEMM +/- 1 concurrent requant on stream B) ===\n");
  printf("  GEMM alone            : %.4f ms\n", g_only);
  printf("  GEMM + concurrent rq  : %.4f ms  (serial-sum would be %.4f; hidden fraction %.0f%%)\n",
         g_plus, g_only+rq, 100.0*(1.0-(g_plus-g_only)/rq));

  // ---- FILL-WALL BREAKER: co-schedule N independent dense-fp4 dW GEMMs on N streams ----
  // At S=1024 one dW GEMM = 64 CTAs (128^3 tiles) on 70 SMs -> half-idle machine.
  // The dW GEMMs of the backward are INDEPENDENT given stored activations, so
  // co-scheduling floods the SMs and lifts aggregate throughput toward the fp4 peak.
  const int NSMAX=16; cudaStream_t st[NSMAX];
  for(int i=0;i<NSMAX;i++) cudaStreamCreate(&st[i]);
  auto csA=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideA{},{M,K,1});
  auto csB=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideB{},{N,K,1});
  auto csC=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideC{},{M,N,1});
  auto csD=cutlass::make_cute_packed_stride(typename DnGemm::GemmKernel::StrideD{},{M,N,1});
  auto clSFA=mk_dlSFA(M,N,K), clSFB=mk_dlSFB(M,N,K);
  typename DnGemm::Arguments cna{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
    {DA.device_data(),csA,DB.device_data(),csB,DSFA.device_data(),clSFA,DSFB.device_data(),clSFB},
    {{1.f,0.f},(cutlass::bfloat16_t*)nullptr,csC,DD.device_data(),csD}};
  DnGemm* dg[NSMAX]; cutlass::device_memory::allocation<uint8_t>* dws[NSMAX];
  for(int i=0;i<NSMAX;i++){ dg[i]=new DnGemm(); dws[i]=new cutlass::device_memory::allocation<uint8_t>(DnGemm::get_workspace_size(cna)); CUTLASS_CHECK(dg[i]->initialize(cna,dws[i]->get())); }
  printf("\n=== fill-wall breaker: N concurrent dense-fp4 dW GEMMs @ S=%d (each %.4f ms alone) ===\n", S, dn);
  for(int ns : {1,2,3,4,6,8,16}){
    for(int w=0;w<8;w++){ for(int i=0;i<ns;i++) CUTLASS_CHECK(dg[i]->run(st[i])); } cudaDeviceSynchronize();
    auto t0=std::chrono::steady_clock::now();
    for(int r=0;r<it;r++) for(int i=0;i<ns;i++) CUTLASS_CHECK(dg[i]->run(st[i]));
    cudaDeviceSynchronize();
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/it;
    double tf=(double)ns*g1/(ms/1e3);   // aggregate dense-equiv TFLOPS across the ns concurrent GEMMs
    printf("  %2d concurrent : %.4f ms/round  %6.1f TFLOPS aggregate  (%.1f/GEMM)\n", ns, ms, tf, tf/ns);
  }

  // ---- FULL L=8 fp4 BACKWARD, actually scheduled, aggregate backprop TFLOPS ----
  // Per layer the backward is 4 GEMMs: dW2,dW1 (dense, weight=OUTPUT) + dD1,dX
  // (sparse IFF the mask is transposable, weight=operand). For L=8 that is 16
  // dense dW GEMMs + 16 sparse dActivation GEMMs. Schedule dense on the concurrent
  // stream pool (fills the SMs at small S) and sparse on its own stream; the two
  // pools overlap. Report the measured aggregate rate over all 32 GEMMs.
  {
    const int L=8, NDW=2*L, NDA=2*L;               // 16 dense dW + 16 sparse dActivation
    auto sched=[&](int ndense_streams)->double{
      for(int w=0;w<6;w++){
        for(int i=0;i<NDW;i++)  CUTLASS_CHECK(dg[i%ndense_streams]->run(st[i%ndense_streams]));
        for(int i=0;i<NDA;i++)  CUTLASS_CHECK(sg.run(s));
      }
      cudaDeviceSynchronize();
      auto t0=std::chrono::steady_clock::now();
      for(int r=0;r<it;r++){
        for(int i=0;i<NDW;i++)  CUTLASS_CHECK(dg[i%ndense_streams]->run(st[i%ndense_streams]));
        for(int i=0;i<NDA;i++)  CUTLASS_CHECK(sg.run(s));
      }
      cudaDeviceSynchronize();
      return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/it;
    };
    printf("\n=== FULL L=8 fp4 backward (16 dense dW + 16 sparse dActivation), transposable-2:4 ===\n");
    double bflop = (double)(NDW+NDA)*g1;           // dense-equiv TFLOP of the whole backward
    for(int nds : {1,2,4}){
      double ms=sched(nds);
      printf("  dense dW on %2d stream(s): %.4f ms/backward   %6.1f TFLOPS backprop  (%d GEMMs)\n",
             nds, ms, bflop/(ms/1e3), NDW+NDA);
    }
  }
  return 0;
}
