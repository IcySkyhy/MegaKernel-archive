// fp4_nlayer_mb.cu -- the VALID N-layer forward (fp4_nlayer_deploy.cu numerics,
// UNCHANGED) run as MB independent microbatches on MB concurrent streams.
//
// Why: the deploy chain is strictly serial per microbatch (requant -> GEMM1 ->
// requant -> GEMM2), so the memory-bound requant (~0.8ms/layer @8192) and the
// inter-kernel dependency bubbles (the ~19% chain tax) are EXPOSED. A second
// (third) independent microbatch on its own stream fills both: its requant
// overlaps our GEMM (compute-bound, DRAM ~57%) and its GEMM CTAs pack into our
// tail waves. Per-microbatch math is bit-identical to the serial deploy run --
// same kernels, same launch order, same buffers-per-mb -- so relPJ is EXACTLY
// the deploy number (0.239 @ L=8, fp4 noise floor). Throughput is honest
// deployable inference/training rate: real serving batches ARE independent.
//
// Build: bash build_sparse.sh fp4_nlayer_mb.cu /tmp/nlayer_mb -DTBM=256
// Run:   /tmp/nlayer_mb <S> <iters> <L> [validate] [MB]
#include <iostream>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
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
// Activation-stream storage types (traffic levers; numerics measured by relPJ):
//   default        : D1 bf16, residual bf16   (the certified deploy numbers)
//   -DD1E4M3       : GEMM1 output + requant2 source in fp8 e4m3 (halves that read)
//   -DRESE4M3      : residual stream (GEMM2 C/D + requant1 source) in fp8 e4m3
#ifdef D1E4M3
using ED1=cutlass::float_e4m3_t;
#else
using ED1=cutlass::bfloat16_t;
#endif
#ifdef RESE4M3
using ERES=cutlass::float_e4m3_t;
#else
using ERES=cutlass::bfloat16_t;
#endif
using LayoutCTag=cutlass::layout::ColumnMajor; using LayoutDTag=cutlass::layout::ColumnMajor;
using ElementE=uint8_t; using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using SpOp=cutlass::arch::OpClassBlockScaledSparseTensorOp;
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int SFVEC=32;
#ifndef TBM
#define TBM 256
#endif
using TB=Shape<Int<TBM>,_128,_256>; using CL=Shape<_1,_1,_1>;

template<class EC, class ED>
struct GemmFor {
  static constexpr int AC_=128/cutlass::sizeof_bits<EC>::value, AD_=128/cutlass::sizeof_bits<ED>::value;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,SpOp,TB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
    EC,LayoutCTag,AC_,ED,LayoutDTag,AD_,
    cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120>::CollectiveOp;
  using Main=typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,SpOp,ElementA,LayoutATag,AA,ElementB,LayoutBTag,AB,Acc,TB,CL,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epi::SharedStorage)>,
    SpSched>::CollectiveOp;
  using Krn=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,void>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Krn>;
};
using G1G=typename GemmFor<ED1,ED1>::Gemm;   // GEMM1: C unused (beta=0), D=ED1
using G2G=typename GemmFor<ERES,ERES>::Gemm; // GEMM2: C=residual in, D=residual out
using SpGemm=G1G;                            // mainloop-side aliases (identical for both)
using StrideA=typename SpGemm::GemmKernel::StrideA; using LayoutA=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutA;
using StrideB=typename SpGemm::GemmKernel::StrideB; using StrideC=typename SpGemm::GemmKernel::StrideC; using StrideD=typename SpGemm::GemmKernel::StrideD;
using LayoutE=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutE;
using SpCfg=typename SpGemm::GemmKernel::CollectiveMainloop::SparseConfig;
using BlkCfg=typename SpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFB=decltype(BlkCfg::tile_atom_to_shape_SFB(make_shape(0,0,0,1)));
using CompU=cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg>;
using CompK=cutlass::transform::kernel::StructuredSparseCompressor<Shape<int,int,int,int>,EA,LayoutATag,SpCfg,ArchTag>;
using Comp=cutlass::transform::device::TransformUniversalAdapter<CompK>;

// device requant: identical numerics to fp4_nlayer_deploy.cu (see there for
// layout proof); templated on the activation storage type (bf16 or fp8 e4m3).
template<class SRCT, class LSF>
__global__ void requant_kernel(const SRCT* __restrict__ src,
                               uint8_t* __restrict__ b_bytes, ESF* sf_ptr, LSF layoutSFB,
                               int N, int K){
  int Kb = K / SFVEC;
  size_t g = (size_t)blockIdx.x*blockDim.x + threadIdx.x;   // = n*Kb + kb
  if (g >= (size_t)N*Kb) return;
  int n = g / Kb, k0 = (g % Kb) * SFVEC;
  const uint4* pr = reinterpret_cast<const uint4*>(src + (size_t)k0 + (size_t)n*K);
  SRCT xb[SFVEC];
  #pragma unroll
  for (int i=0;i<(int)(SFVEC*sizeof(SRCT)/16);++i) reinterpret_cast<uint4*>(xb)[i] = pr[i];
  float x[SFVEC]; float amax=0.f;
  #pragma unroll
  for (int v=0;v<SFVEC;++v){ x[v]=float(xb[v]); amax=fmaxf(amax, fabsf(x[v])); }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT = make_tensor(cute::make_gmem_ptr(sf_ptr), layoutSFB); sfT(n, k0, 0)=q;
  uint32_t w[SFVEC/8]={0};
  #pragma unroll
  for (int v=0;v<SFVEC;++v){ uint32_t c=EA(x[v]*rcp).raw()&0xF; w[v>>3] |= c << ((v&7)*4); }
  reinterpret_cast<uint4*>(b_bytes)[((size_t)n*N + k0)/32] = make_uint4(w[0],w[1],w[2],w[3]);
}

// host nvfp4 weight quantizer: identical to fp4_nlayer_deploy.cu
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
  int S    = argc>1?atoi(argv[1]):8192;
  int iters= argc>2?atoi(argv[2]):100;
  int L    = argc>3?atoi(argv[3]):8;
  int validate = argc>4?atoi(argv[4]):0;
  int MB   = argc>5?atoi(argv[5]):2;
  int M=S,N=S,K=S;

  std::vector<cudaStream_t> st(MB);
  for (int b=0;b<MB;++b) cudaStreamCreate(&st[b]);
  cudaStream_t s = st[0];

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

  std::mt19937 rng(1234); std::normal_distribution<float> g(0.f,1.f);
  struct Wt { cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> comp;
              cutlass::HostTensor<ElementE,cutlass::layout::PackedVectorLayout> meta;
              cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> sfa;
              std::vector<float> sim_col; };
  std::vector<Wt> W1(L), W2(L);
  float wscale = 1.0f/std::sqrt((float)K);
  auto build_w=[&](Wt& w, int seed){
    std::vector<float> Wf((size_t)M*K); std::mt19937 r(seed);
    for (auto& x:Wf) x=g(r)*wscale;
    for (size_t m=0;m<(size_t)M;++m) for (int k=0;k<K;k+=4){ float* p=&Wf[m*K+k];
      float a=std::fabs(p[0])+std::fabs(p[1]), b=std::fabs(p[2])+std::fabs(p[3]);
      if(a>=b){p[2]=p[3]=0.f;}else{p[0]=p[1]=0.f;} }
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

  // ---- per-microbatch activation state ----
  struct MBuf {
    cutlass::HostTensor<ERES,cutlass::layout::PackedVectorLayout> Xres0,Xres1;
    cutlass::HostTensor<ED1,cutlass::layout::PackedVectorLayout> D1bf;
    cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> Xq,Aq;
    cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> SFBx,SFBa;
    ERES* Xres[2];
    std::vector<float> X0;
  };
  std::vector<MBuf> mb(MB);
  for (int b=0;b<MB;++b){
    MBuf& u=mb[b];
    u.Xres0.reset(cutlass::make_Coord(M*N)); u.Xres1.reset(cutlass::make_Coord(M*N)); u.D1bf.reset(cutlass::make_Coord(M*N));
    u.Xq.reset(cutlass::make_Coord(N*K)); u.Aq.reset(cutlass::make_Coord(N*K));
    u.SFBx.reset(cutlass::make_Coord(size(filter_zeros(lSFB)))); u.SFBa.reset(cutlass::make_Coord(size(filter_zeros(lSFB))));
    u.X0.assign((size_t)M*N,0.f);
    std::mt19937 xr(9000+b);
    for (auto& x:u.X0) x=g(xr);
    for (int j=0;j<N;++j) for (int i=0;i<M;++i) u.Xres0.host_data()[(size_t)i+(size_t)j*M]=ERES(u.X0[(size_t)i+(size_t)j*M]);
    u.Xres0.sync_device();
    u.Xres[0]=u.Xres0.device_data(); u.Xres[1]=u.Xres1.device_data();
  }

  auto make_g1=[&](Wt& w, EA* bb, ESF* sfb, ED1* d){
    typename G1G::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
      {w.comp.device_data(),lA,bb,sB,w.meta.device_data(),lE,w.sfa.device_data(),lSFA,sfb,lSFB},
      {{1.f,0.f},nullptr,sC,d,sD}};
    return a; };
  auto make_g2=[&](Wt& w, EA* bb, ESF* sfb, ERES* c, ERES* d){
    typename G2G::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
      {w.comp.device_data(),lA,bb,sB,w.meta.device_data(),lE,w.sfa.device_data(),lSFA,sfb,lSFB},
      {{1.f,1.f},c,sC,d,sD}};
    return a; };
  // adapters per (layer, microbatch): pointers fixed for the lifetime
  std::vector<G1G> g1((size_t)L*MB); std::vector<G2G> g2((size_t)L*MB);
  std::vector<cutlass::device_memory::allocation<uint8_t>> ws1((size_t)L*MB), ws2((size_t)L*MB);
  for (int l=0;l<L;++l) for (int b=0;b<MB;++b){
    MBuf& u=mb[b]; size_t i=(size_t)l*MB+b;
    auto a1=make_g1(W1[l],u.Xq.device_data(),u.SFBx.device_data(),u.D1bf.device_data());
    auto a2=make_g2(W2[l],u.Aq.device_data(),u.SFBa.device_data(),u.Xres[l%2],u.Xres[(l+1)%2]);
    ws1[i]=cutlass::device_memory::allocation<uint8_t>(G1G::get_workspace_size(a1));
    ws2[i]=cutlass::device_memory::allocation<uint8_t>(G2G::get_workspace_size(a2));
    CUTLASS_CHECK(g1[i].can_implement(a1)); CUTLASS_CHECK(g1[i].initialize(a1,ws1[i].get(),s));
    CUTLASS_CHECK(g2[i].can_implement(a2)); CUTLASS_CHECK(g2[i].initialize(a2,ws2[i].get(),s));
  }

  int rqThreads=256; dim3 rqB(rqThreads);
  dim3 rqG(((size_t)N*(K/SFVEC) + rqThreads-1)/rqThreads);
  auto Blayout = make_layout(make_shape(N,K,1),sB);
  (void)Blayout;

  // one forward pass of microbatch b on its stream (identical sequence to deploy)
  auto forward=[&](int b){
    MBuf& u=mb[b];
    for (int l=0;l<L;++l){
      size_t i=(size_t)l*MB+b;
      requant_kernel<<<rqG,rqB,0,st[b]>>>((const ERES*)u.Xres[l%2], (uint8_t*)u.Xq.device_data(), u.SFBx.device_data(), lSFB, N, K);
      CUTLASS_CHECK(g1[i].run(st[b]));
      requant_kernel<<<rqG,rqB,0,st[b]>>>((const ED1*)u.D1bf.device_data(), (uint8_t*)u.Aq.device_data(), u.SFBa.device_data(), lSFB, N, K);
      CUTLASS_CHECK(g2[i].run(st[b]));
    }
  };

  for (int b=0;b<MB;++b) forward(b);
  cudaError_t e0=cudaDeviceSynchronize();
  if(e0){ printf("first forward failed: %s\n",cudaGetErrorString(e0)); return 1; }

  if (validate){
    // per-mb host fp32 reference (dequant-fp4 weights, full-precision acts)
    for (int b=0;b<MB && b<2;++b){
      MBuf& u=mb[b];
      ERES* fin=u.Xres[L%2];
      std::vector<ERES> hout((size_t)M*N);
      cudaMemcpy(hout.data(),fin,(size_t)M*N*sizeof(ERES),cudaMemcpyDeviceToHost);
      std::vector<float> Xr=u.X0, D1r((size_t)M*N), D2r((size_t)M*N);
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
      for (size_t i2=0;i2<(size_t)M*N;++i2){ double d=(double)float(hout[i2])-Xr[i2]; num+=d*d; den+=Xr[i2]*Xr[i2]; }
      printf("VALIDATE mb=%d S=%d L=%d  relPJ (fp4 chain vs fp32) = %.4f\n", b,S,L,std::sqrt(num/(den+1e-30)));
    }
  }

  // warm
  for(int i=0;i<3;i++) for (int b=0;b<MB;++b) forward(b);
  cudaDeviceSynchronize();
  // timed: all MB streams run concurrently
  auto h0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++) for (int b=0;b<MB;++b) forward(b);
  cudaError_t e=cudaDeviceSynchronize();
  auto h1=std::chrono::steady_clock::now();
  if(e){ printf("cuda err: %s\n",cudaGetErrorString(e)); return 1; }
  double ms_tot=std::chrono::duration<double,std::milli>(h1-h0).count();
  double ms_fwd=ms_tot/iters/MB, ms_layer=ms_fwd/L;
  double flop=(double)L*4.0*(double)M*N*K, tf=flop/(ms_fwd/1e3)/1e12;
  const char* d1t=sizeof(ED1)==1?"fp8":"bf16"; const char* rest=sizeof(ERES)==1?"fp8":"bf16";
  printf("MB-OVERLAP valid N-layer fwd  S=%d L=%d MB=%d tile=%dx128x256 D1=%s res=%s  %.4f ms/fwd(eff)  %.4f ms/layer  %.1f TFLOPS dense-equiv (incl requant)\n",
         S,L,MB,TBM,d1t,rest,ms_fwd,ms_layer,tf);
  return 0;
}
