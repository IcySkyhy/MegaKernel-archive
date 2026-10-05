// fp4_nlayer_deploy.cu -- the NUMERICALLY-VALID N-layer residual double-GEMM
// forward, and its real deployable TFLOPS.
//
// Design (2026-07-04). The fp4->fp4 fused chain is numerically valid ONLY with a
// column-major fp4 intermediate (epilogue col-store SF == mainloop SFB layout,
// proven in layout_probe.cu; probe_chain CHAIN D2 rel_frob 0.10). But the
// col-major sub-byte fp4 EPILOGUE store costs 30-40% (nlayer_valid 627/729 vs
// proxy 1044/1019) -- fusing the requant into the epilogue is SLOWER, not faster.
// Column-major *bf16* output is free (1202/1248 == row-major). So the fast valid
// path is: sparse fp4 GEMMs with col-major *bf16* output at ~1240, and a
// dedicated coalesced device kernel that requantizes bf16 -> fp4 + SFB between
// GEMMs (memory-bound, ~1/8 of a GEMM's traffic). The bf16 residual stream keeps
// the chain faithful (raw fp4 residual is unscaled => invalid).
//
// Per layer l (all activation buffers col-major [S,S] bf16, index (row,col)=(feature,sample)):
//   Xq,SFBx = requant(Xres[l])           ; fp4 B for GEMM1
//   D1      = W1_l . Xq        (bf16)     ; GEMM1, beta=0
//   Aq,SFBa = requant(D1)                 ; fp4 B for GEMM2
//   Xres[l+1] = W2_l . Aq + Xres[l] (bf16); GEMM2, C=Xres[l] beta=1 (residual)
// Weights: fp32 -> 2:4 mask -> nvfp4 (vec32) -> compressed sparse A operand.
// Validated vs an fp32 host reference (dequantized-fp4 weights, full-precision
// activations): relPJ = the fp4 activation-requant error, the deployable number.
//
// Build: 13.3 ptxas splice (build_sparse.sh). -DTBM=256 for >=6144.
//   bash build_sparse.sh fp4_nlayer_deploy.cu /tmp/nlayer_deploy -DTBM=256
//   /tmp/nlayer_deploy <S> <iters> <L> [validate]
#include <iostream>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
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
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"
using namespace cute;

template <typename T> auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }  // subbyte-safe

using EA=cutlass::float_e2m1_t; using ESF=cutlass::float_ue4m3_t;
using ElementA=cutlass::nv_float4_t<EA>; using LayoutATag=cutlass::layout::RowMajor;
using ElementB=cutlass::nv_float4_t<EA>; using LayoutBTag=cutlass::layout::ColumnMajor;
constexpr int AA=64, AB=32;
using ElementC=cutlass::bfloat16_t; using ElementD=cutlass::bfloat16_t;
using LayoutCTag=cutlass::layout::ColumnMajor; using LayoutDTag=cutlass::layout::ColumnMajor;
constexpr int AD=128/cutlass::sizeof_bits<ElementD>::value, AC=128/cutlass::sizeof_bits<ElementC>::value;
using ElementE=uint8_t; using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using SpOp=cutlass::arch::OpClassBlockScaledSparseTensorOp;
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int SFVEC=32;                        // sparse nvf4 sm120 SF vector size (measured)
#ifndef TBM
#define TBM 256
#endif
using TB=Shape<Int<TBM>,_128,_256>; using CL=Shape<_1,_1,_1>;

// sparse fp4 GEMM, col-major bf16 output (no SF fusion: plain linear-combination epilogue)
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
using BlkCfg=typename SpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFB=decltype(BlkCfg::tile_atom_to_shape_SFB(make_shape(0,0,0,1)));
using CompU=cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg>;
using CompK=cutlass::transform::kernel::StructuredSparseCompressor<Shape<int,int,int,int>,EA,LayoutATag,SpCfg,ArchTag>;
using Comp=cutlass::transform::device::TransformUniversalAdapter<CompK>;

// -------- device requant: col-major bf16 [K,N] (index [row=k,col=n]=mem[k+n*K]) --------
//   -> fp4 B operand (via layoutB, indexed (n,k)) + SFB (via layoutSFB, indexed (n,k0)).
// Quantize along k (the consuming GEMM's contraction), 32-blocks. One thread owns an
// No transpose: src is col-major [K,N] (k-contiguous) and the GEMM's fp4 B operand
// has stride (N,1) => element (n,k) at e2m1-linear n*N+k (also K-CONTIGUOUS), so
// byte(n,k)=(n*N+k)/2 (even k = low nibble). One WARP per COLUMN n streams the whole
// column: each lane reads a bf16x2 (float via __nv_bfloat162) so a warp covers 64
// contiguous k per step (128B coalesced load); per 32-k block a 16-lane amax-reduce
// gives the ue4m3 SF; 16 contiguous bytes written per block => coalesced, race-free.
// One THREAD per (n, 32-k block): reads its 32 contiguous k as 4x uint4 (128B,
// coalesced across threads), thread-local amax -> ue4m3 SF, packs 32 fp4 into a
// uint4 (16B) written coalesced. No transpose (B is k-contiguous), no shuffles.
template<class LSF>
__global__ void requant_kernel(const __nv_bfloat16* __restrict__ src,
                               uint8_t* __restrict__ b_bytes, ESF* sf_ptr, LSF layoutSFB,
                               int N, int K){
  int Kb = K / SFVEC;
  size_t g = (size_t)blockIdx.x*blockDim.x + threadIdx.x;   // = n*Kb + kb
  if (g >= (size_t)N*Kb) return;
  int n = g / Kb, k0 = (g % Kb) * SFVEC;
  const uint4* pr = reinterpret_cast<const uint4*>(src + (size_t)k0 + (size_t)n*K);  // 128-bit loads
  __nv_bfloat16 xb[SFVEC];
  #pragma unroll
  for (int i=0;i<SFVEC/8;++i) reinterpret_cast<uint4*>(xb)[i] = pr[i];   // 4x uint4 = 32 bf16
  float x[SFVEC]; float amax=0.f;
  #pragma unroll
  for (int v=0;v<SFVEC;++v){ x[v]=__bfloat162float(xb[v]); amax=fmaxf(amax, fabsf(x[v])); }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT = make_tensor(cute::make_gmem_ptr(sf_ptr), layoutSFB); sfT(n, k0, 0)=q;
  uint32_t w[SFVEC/8]={0};
  #pragma unroll
  for (int v=0;v<SFVEC;++v){ uint32_t c=EA(x[v]*rcp).raw()&0xF; w[v>>3] |= c << ((v&7)*4); }
  reinterpret_cast<uint4*>(b_bytes)[((size_t)n*N + k0)/32] = make_uint4(w[0],w[1],w[2],w[3]);
}

// host nvfp4 quantizer for weights (row-major [M,K], vec32 along k). Writes e2m1
// dense (for the compressor) + SFA (layout) + Wsim col-major [M,K] (fp32 deq, for ref).
template<class TV, class TSF>
static void hquant_w(const std::vector<float>& W, int M, int K, TV&& vT, TSF&& sfT, std::vector<float>& Wsim_col){
  for (int m=0;m<M;++m) for (int k0=0;k0<K;k0+=SFVEC){
    float amax=0.f; for (int v=0;v<SFVEC;++v) amax=std::max(amax,std::fabs(W[(size_t)m*K+k0+v]));
    float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
    sfT(m,k0,0)=q;
    for (int v=0;v<SFVEC;++v){ float qv=float(EA(W[(size_t)m*K+k0+v]*rcp)); vT(m,k0+v,0)=EA(qv); Wsim_col[(size_t)m + (size_t)(k0+v)*M]=qv*deq; }
  }
}

int main(int argc,char**argv){
  int S    = argc>1?atoi(argv[1]):6144;
  int iters= argc>2?atoi(argv[2]):100;
  int L    = argc>3?atoi(argv[3]):8;
  int validate = argc>4?atoi(argv[4]):0;   // 1 = host fp32 ref check (use small S)
  int M=S,N=S,K=S;
  cudaStream_t s; cudaStreamCreate(&s);

  auto sA=cutlass::make_cute_packed_stride(StrideA{},{M,K,1});
  auto sB=cutlass::make_cute_packed_stride(StrideB{},{N,K,1});
  auto sC=cutlass::make_cute_packed_stride(StrideC{},{M,N,1});
  auto sD=cutlass::make_cute_packed_stride(StrideD{},{M,N,1});
  auto lSFA=BlkCfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB=BlkCfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  auto wl=make_shape(M,1,K,1); CompU util(wl,sA);
  LayoutA lA=SpCfg::fill_layoutA(wl); LayoutE lE=SpCfg::fill_layoutE(wl);
  int m_phys=util.get_tensorA_m_physical(), k_phys=util.get_tensorA_k_physical();
  int me_phys=util.get_metadata_m_physical(), ke_phys=util.get_metadata_k_physical();

  // ---- weights per layer: fp32 -> 2:4 mask -> nvfp4 -> compress ----
  std::mt19937 rng(1234); std::normal_distribution<float> g(0.f,1.f);
  struct Wt { cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> comp;
              cutlass::HostTensor<ElementE,cutlass::layout::PackedVectorLayout> meta;
              cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> sfa;
              std::vector<float> sim_col; };
  std::vector<Wt> W1(L), W2(L);
  // variance-preserving init: with 2:4 (K/2 live) a GEMM keeps var(D)~0.5 var(X),
  // so the residual forward is STABLE over depth (unscaled random weights explode
  // exponentially and saturate the ue4m3 block scale => meaningless relPJ).
  float wscale = 1.0f/std::sqrt((float)K);
  auto build_w=[&](Wt& w, int seed){
    std::vector<float> Wf((size_t)M*K); std::mt19937 r(seed);
    for (auto& x:Wf) x=g(r)*wscale;
    for (size_t m=0;m<(size_t)M;++m) for (int k=0;k<K;k+=4){ float* p=&Wf[m*K+k];
      float a=std::fabs(p[0])+std::fabs(p[1]), b=std::fabs(p[2])+std::fabs(p[3]);
      if(a>=b){p[2]=p[3]=0.f;}else{p[0]=p[1]=0.f;} }        // aligned 2:4
    cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> dense; dense.reset(cutlass::make_Coord(M*K));
    w.sfa.reset(cutlass::make_Coord(size(filter_zeros(lSFA)))); w.sim_col.assign((size_t)M*K,0.f);
    auto vT=make_tensor(make_iterator(dense.host_data()), make_layout(make_shape(M,K,1), sA));
    auto sfT=make_tensor(w.sfa.host_data(), lSFA);
    hquant_w(Wf,M,K,vT,sfT,w.sim_col);
    dense.sync_device(); w.sfa.sync_device();
    w.comp.reset(cutlass::make_Coord(m_phys*k_phys)); w.meta.reset(cutlass::make_Coord(me_phys*ke_phys));
    typename Comp::Arguments ca{{M,1,K,1},{dense.device_data(),sA,w.comp.device_data(),w.meta.device_data()},{hw}};
    Comp cop; cutlass::device_memory::allocation<uint8_t> cws(Comp::get_workspace_size(ca));
    CUTLASS_CHECK(cop.can_implement(ca)); CUTLASS_CHECK(cop.initialize(ca,cws.get(),s)); CUTLASS_CHECK(cop.run(s));
    cudaStreamSynchronize(s);
  };
  for (int l=0;l<L;++l){ build_w(W1[l],100+l); build_w(W2[l],500+l); }

  // ---- activation buffers (col-major [S,S] bf16) + fp4 B operands + SFB ----
  cutlass::HostTensor<cutlass::bfloat16_t,cutlass::layout::PackedVectorLayout> Xres0,Xres1,D1bf;
  cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> Xq,Aq;
  cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> SFBx,SFBa;
  Xres0.reset(cutlass::make_Coord(M*N)); Xres1.reset(cutlass::make_Coord(M*N)); D1bf.reset(cutlass::make_Coord(M*N));
  Xq.reset(cutlass::make_Coord(N*K)); Aq.reset(cutlass::make_Coord(N*K));
  SFBx.reset(cutlass::make_Coord(size(filter_zeros(lSFB)))); SFBa.reset(cutlass::make_Coord(size(filter_zeros(lSFB))));

  // input activation X0 (fp32) -> Xres0 bf16 (col-major)
  std::vector<float> X0((size_t)M*N);
  for (auto& x:X0) x=g(rng);
  for (int j=0;j<N;++j) for (int i=0;i<M;++i) Xres0.host_data()[(size_t)i+(size_t)j*M]=cutlass::bfloat16_t(X0[(size_t)i+(size_t)j*M]);
  Xres0.sync_device();
  cutlass::bfloat16_t* Xres[2]={Xres0.device_data(),Xres1.device_data()};

  // ---- GEMM argument factory (col-major bf16 out) ----
  auto make_gemm=[&](Wt& w, EA* b, ESF* sfb, cutlass::bfloat16_t* c, cutlass::bfloat16_t* d, float beta){
    typename SpGemm::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
      {w.comp.device_data(),lA,b,sB,w.meta.device_data(),lE,w.sfa.device_data(),lSFA,sfb,lSFB},
      {{1.f,beta},c,sC,d,sD}};
    return a; };
  // Pre-initialize one adapter per (layer,GEMM). Shape+pointers are fixed for the
  // lifetime (Xq/Aq/D1bf reused every layer, Xres ping-pongs 2 fixed buffers, only
  // the weights differ) so can_implement/initialize run ONCE, out of the timed loop.
  std::vector<SpGemm> g1(L), g2(L);
  std::vector<cutlass::device_memory::allocation<uint8_t>> ws1(L), ws2(L);
  for (int l=0;l<L;++l){
    auto a1=make_gemm(W1[l],Xq.device_data(),SFBx.device_data(),nullptr,D1bf.device_data(),0.f);
    auto a2=make_gemm(W2[l],Aq.device_data(),SFBa.device_data(),Xres[l%2],Xres[(l+1)%2],1.f);
    ws1[l]=cutlass::device_memory::allocation<uint8_t>(SpGemm::get_workspace_size(a1));
    ws2[l]=cutlass::device_memory::allocation<uint8_t>(SpGemm::get_workspace_size(a2));
    CUTLASS_CHECK(g1[l].can_implement(a1)); CUTLASS_CHECK(g1[l].initialize(a1,ws1[l].get(),s));
    CUTLASS_CHECK(g2[l].can_implement(a2)); CUTLASS_CHECK(g2[l].initialize(a2,ws2[l].get(),s));
  }

  int rqThreads=256; dim3 rqB(rqThreads);
  dim3 rqG(((size_t)N*(K/SFVEC) + rqThreads-1)/rqThreads);   // one thread per (n, 32-k block)
  auto Blayout = make_layout(make_shape(N,K,1),sB);

  // one forward pass over L layers (only kernel launches, no host-side setup)
  auto forward=[&](){
    for (int l=0;l<L;++l){
      requant_kernel<<<rqG,rqB,0,s>>>((const __nv_bfloat16*)Xres[l%2], (uint8_t*)Xq.device_data(), SFBx.device_data(), lSFB, N, K);
      CUTLASS_CHECK(g1[l].run(s));
      requant_kernel<<<rqG,rqB,0,s>>>((const __nv_bfloat16*)D1bf.device_data(), (uint8_t*)Aq.device_data(), SFBa.device_data(), lSFB, N, K);
      CUTLASS_CHECK(g2[l].run(s));
    }
  };

  forward(); cudaError_t e0=cudaStreamSynchronize(s);
  if(e0){ printf("first forward failed: %s\n",cudaGetErrorString(e0)); return 1; }

  // ---- requant self-check: dequant Xq (from a fresh requant of Xres0) vs X0 ----
  if (validate){
    requant_kernel<<<rqG,rqB,0,s>>>((const __nv_bfloat16*)Xres0.device_data(), (uint8_t*)Xq.device_data(), SFBx.device_data(), lSFB, N, K);
    cudaStreamSynchronize(s); Xq.sync_host(); SFBx.sync_host();
    auto bh = make_tensor(make_iterator(Xq.host_data()), Blayout);
    auto sh = make_tensor(SFBx.host_data(), lSFB);
    double num=0,den=0;
    for (int n=0;n<N;++n) for (int k=0;k<K;++k){
      float v = float(EA(bh(n,k,0))) * float(sh(n,(k/SFVEC)*SFVEC,0));
      float r = X0[(size_t)k + (size_t)n*M];       // src(k,n) = X0 col-major (k,n)
      double d=v-r; num+=d*d; den+=(double)r*r;
    }
    printf("REQUANT self-check: relPJ(deq Xq vs X0) = %.4f\n", std::sqrt(num/(den+1e-30)));
  }

  // ---- validation vs fp32 host reference ----
  if (validate){
    cutlass::bfloat16_t* fin=Xres[L%2];
    std::vector<cutlass::bfloat16_t> hout((size_t)M*N);
    cudaMemcpy(hout.data(),fin,(size_t)M*N*sizeof(cutlass::bfloat16_t),cudaMemcpyDeviceToHost);
    // fp32 ref: col-major, dequant-fp4 weights, full-precision activations, real requant-free chain
    std::vector<float> Xr((size_t)M*N), D1r((size_t)M*N), D2r((size_t)M*N);
    for (size_t i=0;i<(size_t)M*N;++i) Xr[i]=X0[i];
    for (int l=0;l<L;++l){
      const auto& w1=W1[l].sim_col; const auto& w2=W2[l].sim_col;
      for (int m=0;m<M;++m) for (int n=0;n<N;++n){ double acc=0;
        for (int k=0;k<K;++k) acc += (double)w1[(size_t)m+(size_t)k*M]*Xr[(size_t)k+(size_t)n*M];
        D1r[(size_t)m+(size_t)n*M]=(float)acc; }
      for (int m=0;m<M;++m) for (int n=0;n<N;++n){ double acc=0;
        for (int k=0;k<K;++k) acc += (double)w2[(size_t)m+(size_t)k*M]*D1r[(size_t)k+(size_t)n*M];
        D2r[(size_t)m+(size_t)n*M]=(float)acc + Xr[(size_t)m+(size_t)n*M]; }
      Xr.swap(D2r);
    }
    double num=0,den=0;
    for (size_t i=0;i<(size_t)M*N;++i){ double d=(double)float(hout[i])-Xr[i]; num+=d*d; den+=Xr[i]*Xr[i]; }
    printf("VALIDATE S=%d L=%d  relPJ (fp4 chain vs fp32) = %.4f\n", S,L,std::sqrt(num/(den+1e-30)));
  }

  // ---- timing ----
  for(int i=0;i<3;i++) forward(); cudaStreamSynchronize(s);
  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0,s); for(int i=0;i<iters;i++) forward(); cudaEventRecord(t1,s);
  cudaError_t e=cudaStreamSynchronize(s);
  if(e){ printf("cuda err: %s\n",cudaGetErrorString(e)); return 1; }
  float ms_tot=0; cudaEventElapsedTime(&ms_tot,t0,t1);
  double ms_fwd=(double)ms_tot/iters, ms_layer=ms_fwd/L;
  double flop=(double)L*4.0*(double)M*N*K, tf=flop/(ms_fwd/1e3)/1e12;
  printf("DEPLOY valid N-layer fwd  S=%d L=%d tile=%dx128x256  %.4f ms/fwd  %.4f ms/layer  %.1f TFLOPS dense-equiv (incl requant)\n",
         S,L,TBM,ms_fwd,ms_layer,tf);
  return 0;
}
