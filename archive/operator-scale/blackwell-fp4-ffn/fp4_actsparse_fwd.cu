// fp4_actsparse_fwd.cu -- ACTIVATION-sparse N-layer residual double-GEMM forward.
//
// Role swap vs fp4_nlayer_deploy.cu: the ACTIVATIONS are the 2:4(pair-4:8)
// sparse A operand (gated+compressed at runtime), the WEIGHTS are the dense
// fp4 B operand -- FULL weight capacity retained (no weight pruning at all).
// This is the "maintain sparse, keep general capability" arm of the vision
// brief: sparsity becomes a *model nonlinearity* (exact structured top-2-of-4
// pair gate) instead of a weight constraint.
//
// Layer l (all activations ROW-major [S samples x K feats], bf16):
//   A1,meta1,SFA1 = gate_quant_compress(Xres[l])   ; one fused pass
//   D1            = A1 . W1_l   (bf16 row-major)   ; GEMM1 beta=0
//   A2,meta2,SFA2 = gate_quant_compress(D1)
//   Xres[l+1]     = A2 . W2_l + Xres[l]            ; GEMM2 C=residual beta=1
// Chain symmetry: row-major D IS the next gate pass's input layout; the
// col-major/SFB-axis problem of the weight-sparse chain does not exist here.
//
// Gate semantics (EXACT, replicated in the host reference): per 8 logical
// fp4-pair chunk keep the 2 of 4 byte-pairs with the largest |a|+|b| (float,
// pre-quantization, stable ties) -> HW pair-granular 4:8 (contract in
// compress_equiv.cu). SF block-32 amax over the 16 SURVIVORS.
//
// Validation prints TWO relPJ numbers @small S:
//   vs GATED fp32 ref   -> quantization-only error (target ~fp4 floor 0.24@L8)
//   vs UNGATED fp32 ref -> the gate's approximation cost (accuracy-phase input)
//
// Build: bash build_sparse.sh fp4_actsparse_fwd.cu /tmp/actsp -DTBM=256
// Run:   /tmp/actsp <S> <iters> <L> [validate]
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

template <typename T> auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

using EA=cutlass::float_e2m1_t; using ESF=cutlass::float_ue4m3_t;
using ElementA=cutlass::nv_float4_t<EA>; using LayoutATag=cutlass::layout::RowMajor;
using ElementB=cutlass::nv_float4_t<EA>; using LayoutBTag=cutlass::layout::ColumnMajor;
constexpr int AA=64, AB=32;
using ElementC=cutlass::bfloat16_t; using ElementD=cutlass::bfloat16_t;
using LayoutCTag=cutlass::layout::RowMajor; using LayoutDTag=cutlass::layout::RowMajor;
constexpr int AD=128/cutlass::sizeof_bits<ElementD>::value, AC=128/cutlass::sizeof_bits<ElementC>::value;
using ElementE=uint8_t; using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using SpOp=cutlass::arch::OpClassBlockScaledSparseTensorOp;
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int SFVEC=32;
#ifndef TBM
#define TBM 256
#endif
using TB=Shape<Int<TBM>,_128,_256>; using CL=Shape<_1,_1,_1>;

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

// ---- fused gate + quantize + 4:8 pair-compress + SFA emit -----------------
// One thread = 32 logical fp4 along k of one activation row m.
// Emits: 8 compressed value bytes (LayoutA), 2 metadata bytes (LayoutE
// contract, cf fast24_vec.inc), 1 SFA byte (atom-layout tensor).
template<class LSF>
__global__ void gate_quant_compress(const __nv_bfloat16* __restrict__ src,
                                    uint8_t* __restrict__ comp, uint8_t* __restrict__ meta,
                                    ESF* __restrict__ sfa, LSF layoutSFA,
                                    int M, int K){
  int Kb=K/SFVEC;
  size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if (g>=(size_t)M*Kb) return;
  int m=g/Kb, seg=g%Kb, k0=seg*SFVEC;
  const uint4* pr=reinterpret_cast<const uint4*>(src+(size_t)m*K+k0);
  __nv_bfloat16 xb[SFVEC];
  #pragma unroll
  for(int i=0;i<SFVEC/8;++i) reinterpret_cast<uint4*>(xb)[i]=pr[i];
  float x[SFVEC];
  #pragma unroll
  for(int v=0;v<SFVEC;++v) x[v]=__bfloat162float(xb[v]);

  // gate: 4 chunks of 4 pairs; keep top-2 pairs by |a|+|b| (stable, float)
  // -DGATE01: keep pairs (0,1) always -- deterministic diagnostic that removes
  // gate-decision divergence so relPJ isolates pure quantization+layout.
  int keep[8]; uint8_t nib[4]; float amax=0.f;
  #pragma unroll
  for(int c=0;c<4;++c){
    const float* p=x+8*c;
    float sc[4]; int a=0,b=1;
    #pragma unroll
    for(int j=0;j<4;++j) sc[j]=fabsf(p[2*j])+fabsf(p[2*j+1]);
#ifdef GATE01
    a=0; b=1; (void)sc;
#else
    { float sa=sc[0]; a=0;
      if(sc[1]>sa){sa=sc[1];a=1;} if(sc[2]>sa){sa=sc[2];a=2;} if(sc[3]>sa){sa=sc[3];a=3;}
      b=(a==0)?1:0; float sb=sc[b];
      #pragma unroll
      for(int j=0;j<4;++j) if(j!=a&&j!=b&&sc[j]>sb){sb=sc[j];b=j;}
    }
#endif
    int lo=a<b?a:b, hi=a<b?b:a;
    keep[2*c]=8*c+2*lo; keep[2*c+1]=8*c+2*hi;      // even-element index of each kept pair
    nib[c]=uint8_t(lo|(hi<<2));
    float m1=fmaxf(fabsf(p[2*lo]),fabsf(p[2*lo+1])), m2=fmaxf(fabsf(p[2*hi]),fabsf(p[2*hi+1]));
    amax=fmaxf(amax,fmaxf(m1,m2));
  }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT=make_tensor(cute::make_gmem_ptr(sfa),layoutSFA); sfT(m,k0,0)=q;
  // pack survivors: chunk c -> bytes 2c,2c+1; byte = (even k -> lo nibble)
  uint32_t o0=0,o1=0;
  #pragma unroll
  for(int i=0;i<8;++i){
    uint32_t b8=(EA(x[keep[i]]*rcp).raw()&0xF) | ((EA(x[keep[i]+1]*rcp).raw()&0xF)<<4);
    if(i<4) o0|=b8<<(8*i); else o1|=b8<<(8*(i-4));
  }
  reinterpret_cast<uint2*>(comp)[g]=make_uint2(o0,o1);   // K_al==K (K%256==0)
  size_t M_alE=size_t((M+127)/128*128);
  size_t eb=size_t(m%128)*16+size_t(m/128)*2048+size_t(k0/256)*(M_alE*16)+size_t(k0%256)/16;
  meta[eb]=uint8_t(nib[0]|(nib[1]<<4)); meta[eb+1]=uint8_t(nib[2]|(nib[3]<<4));
}

// host weight quantizer: dense fp4 B col-major (K,N: column n = output n's
// inputs contiguous) + SFB block-32 along K, and the dequant sim for the ref.
template<class TV, class TSF>
static void hquant_wB(const std::vector<float>& W, int N, int K, TV&& vT, TSF&& sfT, std::vector<float>& Wsim){
  for (int n=0;n<N;++n) for (int k0=0;k0<K;k0+=SFVEC){
    float amax=0.f; for (int v=0;v<SFVEC;++v) amax=std::max(amax,std::fabs(W[(size_t)n*K+k0+v]));
    float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
    sfT(n,k0,0)=q;
    for (int v=0;v<SFVEC;++v){ float qv=float(EA(W[(size_t)n*K+k0+v]*rcp)); vT(n,k0+v,0)=EA(qv); Wsim[(size_t)n*K+k0+v]=qv*deq; }
  }
}

// host replica of the device gate (float scores, stable ties) for the ref
static void host_gate(std::vector<float>& X, int Mm, int Kk){
  for (size_t m=0;m<(size_t)Mm;++m) for (int c0=0;c0<Kk;c0+=8){
    float* p=&X[m*Kk+c0];
    float sc[4]; for(int j=0;j<4;++j) sc[j]=std::fabs(p[2*j])+std::fabs(p[2*j+1]);
    int a=0, b=1;
#ifdef GATE01
    (void)sc;
#else
    float sa=sc[0];
    if(sc[1]>sa){sa=sc[1];a=1;} if(sc[2]>sa){sa=sc[2];a=2;} if(sc[3]>sa){sa=sc[3];a=3;}
    b=(a==0)?1:0; float sb=sc[b];
    for(int j=0;j<4;++j) if(j!=a&&j!=b&&sc[j]>sb){sb=sc[j];b=j;}
#endif
    for(int j=0;j<4;++j) if(j!=a&&j!=b){ p[2*j]=0.f; p[2*j+1]=0.f; }
  }
}

int main(int argc,char**argv){
  int S    = argc>1?atoi(argv[1]):8192;
  int iters= argc>2?atoi(argv[2]):100;
  int L    = argc>3?atoi(argv[3]):8;
  int validate = argc>4?atoi(argv[4]):0;
  int M=S,N=S,K=S;
  if (K%256 || M%128){ printf("need K%%256==0 && M%%128==0\n"); return 1; }
  cudaStream_t s; cudaStreamCreate(&s);
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  auto sA=cutlass::make_cute_packed_stride(StrideA{},{M,K,1});
  auto sB=cutlass::make_cute_packed_stride(StrideB{},{N,K,1});
  auto sC=cutlass::make_cute_packed_stride(StrideC{},{M,N,1});
  auto sD=cutlass::make_cute_packed_stride(StrideD{},{M,N,1});
  auto lSFA=BlkCfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB=BlkCfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto wl=make_shape(M,1,K,1);
  cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg> util(wl,sA);
  LayoutA lA=SpCfg::fill_layoutA(wl); LayoutE lE=SpCfg::fill_layoutE(wl);
  int m_phys=util.get_tensorA_m_physical(), k_phys=util.get_tensorA_k_physical();
  int me_phys=util.get_metadata_m_physical(), ke_phys=util.get_metadata_k_physical();

  std::mt19937 rng(1234); std::normal_distribution<float> g(0.f,1.f);
  float wscale=1.0f/std::sqrt((float)K);

  struct Wt { cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> vals;
              cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> sfb;
              std::vector<float> sim; };
  std::vector<Wt> W1(L), W2(L);
  auto build_w=[&](Wt& w, int seed){
    std::vector<float> Wf((size_t)N*K); std::mt19937 r(seed);
    for (auto& x:Wf) x=g(r)*wscale;
    w.vals.reset(cutlass::make_Coord(N*K)); w.sfb.reset(cutlass::make_Coord(size(filter_zeros(lSFB))));
    w.sim.assign((size_t)N*K,0.f);
    auto vT=make_tensor(make_iterator(w.vals.host_data()), make_layout(make_shape(N,K,1),sB));
    auto sfT=make_tensor(w.sfb.host_data(), lSFB);
    hquant_wB(Wf,N,K,vT,sfT,w.sim);
    w.vals.sync_device(); w.sfb.sync_device();
  };
  for (int l=0;l<L;++l){ build_w(W1[l],100+l); build_w(W2[l],500+l); }

  // activations
  cutlass::HostTensor<cutlass::bfloat16_t,cutlass::layout::PackedVectorLayout> Xres0,Xres1,D1;
  Xres0.reset(cutlass::make_Coord(M*N)); Xres1.reset(cutlass::make_Coord(M*N)); D1.reset(cutlass::make_Coord(M*N));
  std::vector<float> X0((size_t)M*K);
  { std::mt19937 xr(9001); for (auto& x:X0) x=g(xr);
    for (size_t i=0;i<(size_t)M*K;++i) Xres0.host_data()[i]=cutlass::bfloat16_t(X0[i]);
    Xres0.sync_device(); }
  cutlass::bfloat16_t* Xres[2]={Xres0.device_data(),Xres1.device_data()};

  // two compressed-activation slots (X-side and D1-side), reused across layers
  struct ASlot { cutlass::device_memory::allocation<uint8_t> comp, meta; 
                 cutlass::device_memory::allocation<ESF> sfa; };
  ASlot A1{ {size_t(m_phys)*k_phys/2}, {size_t(me_phys)*ke_phys}, {size_t(size(filter_zeros(lSFA)))} };
  ASlot A2{ {size_t(m_phys)*k_phys/2}, {size_t(me_phys)*ke_phys}, {size_t(size(filter_zeros(lSFA)))} };
  cudaMemset(A1.comp.get(),0,size_t(m_phys)*k_phys/2); cudaMemset(A1.meta.get(),0,size_t(me_phys)*ke_phys);
  cudaMemset(A2.comp.get(),0,size_t(m_phys)*k_phys/2); cudaMemset(A2.meta.get(),0,size_t(me_phys)*ke_phys);

  auto make_gemm=[&](ASlot& a, Wt& w, cutlass::bfloat16_t* c, cutlass::bfloat16_t* d, float beta){
    typename SpGemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
      {reinterpret_cast<EA*>(a.comp.get()),lA,
       reinterpret_cast<EA*>(w.vals.device_data()),sB,
       a.meta.get(),lE,
       a.sfa.get(),lSFA, w.sfb.device_data(),lSFB},
      {{1.f,beta},c,sC,d,sD}};
    return args; };
  std::vector<SpGemm> g1(L), g2(L);
  std::vector<cutlass::device_memory::allocation<uint8_t>> ws1(L), ws2(L);
  for (int l=0;l<L;++l){
    auto a1=make_gemm(A1,W1[l],nullptr,D1.device_data(),0.f);
    auto a2=make_gemm(A2,W2[l],(cutlass::bfloat16_t*)Xres[l%2],(cutlass::bfloat16_t*)Xres[(l+1)%2],1.f);
    ws1[l]=cutlass::device_memory::allocation<uint8_t>(SpGemm::get_workspace_size(a1));
    ws2[l]=cutlass::device_memory::allocation<uint8_t>(SpGemm::get_workspace_size(a2));
    CUTLASS_CHECK(g1[l].can_implement(a1)); CUTLASS_CHECK(g1[l].initialize(a1,ws1[l].get(),s));
    CUTLASS_CHECK(g2[l].can_implement(a2)); CUTLASS_CHECK(g2[l].initialize(a2,ws2[l].get(),s));
  }

  int thr=256; dim3 gcG(((size_t)M*(K/SFVEC)+thr-1)/thr), gcB(thr);
  auto forward=[&](){
    for (int l=0;l<L;++l){
      gate_quant_compress<<<gcG,gcB,0,s>>>((const __nv_bfloat16*)Xres[l%2],A1.comp.get(),A1.meta.get(),A1.sfa.get(),lSFA,M,K);
      CUTLASS_CHECK(g1[l].run(s));
      gate_quant_compress<<<gcG,gcB,0,s>>>((const __nv_bfloat16*)D1.device_data(),A2.comp.get(),A2.meta.get(),A2.sfa.get(),lSFA,M,K);
      CUTLASS_CHECK(g2[l].run(s));
    }
  };

  forward();
  cudaError_t e0=cudaDeviceSynchronize();
  if(e0){ printf("first forward failed: %s\n",cudaGetErrorString(e0)); return 1; }

  if (validate){
    std::vector<cutlass::bfloat16_t> hout((size_t)M*N);
    cudaMemcpy(hout.data(),Xres[L%2],(size_t)M*N*sizeof(cutlass::bfloat16_t),cudaMemcpyDeviceToHost);
    // ref A: gated fp32 chain (device semantics minus quantization)
    // ref B: ungated fp32 chain (the gate's approximation cost)
    std::vector<float> Xa=X0, Xb=X0, D1r((size_t)M*N), D2r((size_t)M*N);
    for (int pass=0; pass<2; ++pass){
      std::vector<float>& Xr = pass? Xb : Xa;
      for (int l=0;l<L;++l){
        std::vector<float> Xg=Xr;
        if (pass==0) host_gate(Xg,M,K);
        const auto& w1=W1[l].sim; const auto& w2=W2[l].sim;
        for (int m=0;m<M;++m) for (int n=0;n<N;++n){ double acc=0;
          for (int k=0;k<K;++k) acc+=(double)Xg[(size_t)m*K+k]*w1[(size_t)n*K+k];
          D1r[(size_t)m*N+n]=(float)acc; }
        std::vector<float> Dg=D1r;
        if (pass==0) host_gate(Dg,M,K);
        for (int m=0;m<M;++m) for (int n=0;n<N;++n){ double acc=0;
          for (int k=0;k<K;++k) acc+=(double)Dg[(size_t)m*K+k]*w2[(size_t)n*K+k];
          D2r[(size_t)m*N+n]=(float)acc + Xr[(size_t)m*N+n]; }
        Xr.swap(D2r);
      }
      double num=0,den=0;
      for (size_t i=0;i<(size_t)M*N;++i){ double d=(double)float(hout[i])-Xr[i]; num+=d*d; den+=Xr[i]*Xr[i]; }
      printf("VALIDATE S=%d L=%d  relPJ vs %s fp32 ref = %.4f\n", S,L, pass?"UNGATED":"GATED", std::sqrt(num/(den+1e-30)));
    }
  }

  // standalone gate-compress rate
  { cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    for(int i=0;i<10;i++) gate_quant_compress<<<gcG,gcB,0,s>>>((const __nv_bfloat16*)Xres[0],A1.comp.get(),A1.meta.get(),A1.sfa.get(),lSFA,M,K);
    cudaStreamSynchronize(s);
    cudaEventRecord(a,s);
    for(int i=0;i<50;i++) gate_quant_compress<<<gcG,gcB,0,s>>>((const __nv_bfloat16*)Xres[0],A1.comp.get(),A1.meta.get(),A1.sfa.get(),lSFA,M,K);
    cudaEventRecord(b,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=50;
    double gb=((double)M*K*2 + (double)m_phys*k_phys/2 + (double)me_phys*ke_phys + (double)M*K/32)/1e9;
    printf("gate_quant_compress: %.4f ms (%.0f GB/s)\n", ms, gb/(ms/1e3));
  }

  for(int i=0;i<3;i++) forward(); cudaDeviceSynchronize();
  auto h0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++) forward();
  cudaError_t e=cudaDeviceSynchronize();
  auto h1=std::chrono::steady_clock::now();
  if(e){ printf("cuda err: %s\n",cudaGetErrorString(e)); return 1; }
  double ms_fwd=std::chrono::duration<double,std::milli>(h1-h0).count()/iters, ms_layer=ms_fwd/L;
  double flop=(double)L*4.0*(double)M*N*K, tf=flop/(ms_fwd/1e3)/1e12;
  printf("ACT-SPARSE valid N-layer fwd  S=%d L=%d tile=%dx128x256 (dense weights)  %.4f ms/fwd  %.4f ms/layer  %.1f TFLOPS dense-equiv (incl gate+compress)\n",
         S,L,TBM,ms_fwd,ms_layer,tf);
  return 0;
}
