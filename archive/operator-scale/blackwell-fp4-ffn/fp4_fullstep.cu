// fp4_fullstep.cu -- CHAINED full-training-step (fwd+bwd) N-layer residual
// double-GEMM harness. The bridging artifact both campaigns flagged as missing:
// fwd (872.8 nlayer_mb) and bwd (1196 bwd_1000) existed only as separate
// probes; this runs the dependency-honest 6-GEMM/layer step with REAL data
// flowing producer->consumer (no synthetic buffer cycling).
//
// Formulation: ACTIVATION-sparse (fp4_actsparse_fwd.cu arm -- full weight
// capacity, sparsity = exact structured top-2-of-4-pair gate as a model
// nonlinearity). This makes ALL SIX GEMMs per layer the SAME CUTLASS type
// (sparse A x dense fp4 B, bf16 row-major C/D, runtime beta):
//
//   fwd  (per layer l, activations row-major [S samples x F feats]):
//     A1,mX = gate_quant_compress(X_l)         ; sparse A + gate mask emit
//     A1T   = tquantT(X_l, mX)                 ; B-form (K=samples) for dW1, L2-hot
//     D1    = A1 . W1_l                        ; GEMM (beta=0)
//     A2,mD = gate_quant_compress(D1)
//     A2T   = tquantT(D1, mD)                  ; B-form for dW2
//     X_l+1 = A2 . W2_l + X_l                  ; GEMM (C=residual, beta=1)
//   loss grad: G_L = X_L                       ; (loss = 1/2||X_L||^2)
//   bwd  (per layer l = L-1..0, grads row-major [S x F]):
//     Gc_s  = gqc_T(G)                         ; grad gated 4:8 along SAMPLES
//     dW2^T = Gc_s . A2T_l                     ; GEMM (sink)
//     Gc_f  = gqc(G)                           ; grad gated 4:8 along FEATS
//     dA2   = Gc_f . W2T_l                     ; GEMM (W2T = indep. quant of W2^T)
//     dc_s  = gqc_T(dA2, maskIn=mD)            ; STE mask fused into the read
//     dW1^T = dc_s . A1T_l                     ; GEMM (sink)
//     dc_f  = gqc(dA2, maskIn=mD)
//     dA1   = dc_f . W1T_l                     ; GEMM
//     G'    = mX (.) dA1 + G                   ; STE + residual (elementwise)
//
// Accounting (matches campaign convention): TFLOPS dense-equiv over the LOGICAL
// math = 6L GEMMs x 2*S^3, wall includes every requant/gate/compress/elementwise
// (optimizer/allreduce excluded -- that's the mult32 track's domain).
// Weights stationary: W and W^T quantized independently at init (a real trainer
// re-emits both post-optimizer; mult32 measured that snapshot cost ~cheap/L2-hot).
//
// Validation (dual-relPJ, actsparse pattern; host = fp32 arithmetic, replicated
// gates/masks, NO activation-quant sim):
//   fwd:  relPJ vs GATED ref (quant noise) + vs UNGATED ref (gate cost)
//   bwd:  dX0/dW1T/dW2T vs GATED-GRAD ref (quant+wiring noise)
//                       vs UNGATED-GRAD ref (the grad-gate approximation cost --
//                        the number bwd_1000's batch-prune arm never measured)
// -DGATE01 pins every gate to pairs(0,1): removes gate-decision divergence
// (device gates score bf16, host scores fp32) => wiring-exactness acceptance gate.
//
// Build: bash build_sparse.sh fp4_fullstep.cu /tmp/fullstep -DTBM=256
//        [-DGROUPS=2|4|8] [-DGRAPH] [-DGATE01] [-DD1E4M3]
//        (+ -Xcompiler=-fopenmp -lgomp to parallelize the host ref)
// GROUPS filters compact block-permutation weights (default 1, dense); GRAPH removes
// fine-grained launch tax. D1E4M3 is a certified negative traffic arm (slower here).
// Run:   /tmp/fullstep <S> <iters> <L> [validate] [MB]
#include <iostream>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>
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
// Native sm120 paired fp4 convert (mult32 461d012 donor; bit-exact vs EA(x).raw()
// per pack_probe.cu): a -> upper nibble, b -> lower nibble.
__device__ __forceinline__ uint32_t cvtE2M1x2(float a, float b){
  uint16_t r;
  asm("{\n\t.reg .b8 t;\n\tcvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n\tcvt.u16.u8 %0, t;\n\t}"
      : "=h"(r) : "f"(a), "f"(b));
  return r;
}
using ElementA=cutlass::nv_float4_t<EA>; using LayoutATag=cutlass::layout::RowMajor;
using ElementB=cutlass::nv_float4_t<EA>; using LayoutBTag=cutlass::layout::ColumnMajor;
constexpr int AA=64, AB=32;
using ElementC=cutlass::bfloat16_t; using ElementD=cutlass::bfloat16_t;
#ifdef D1E4M3
using ED1=cutlass::float_e4m3_t; // traffic lever: first projection output and next fused-quant source
#else
using ED1=cutlass::bfloat16_t;
#endif
using LayoutCTag=cutlass::layout::RowMajor; using LayoutDTag=cutlass::layout::RowMajor;
using LayoutD1Tag=cutlass::layout::ColumnMajor;
using Acc=float;
using ArchTag=cutlass::arch::Sm120;
using SpOp=cutlass::arch::OpClassBlockScaledSparseTensorOp;
using SpSched=cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int SFVEC=32;
#ifndef TBM
#define TBM 256
#endif
#ifndef GROUPS
#define GROUPS 1
#endif
static_assert(GROUPS == 1 || GROUPS == 2 || GROUPS == 4 || GROUPS == 8, "GROUPS must be 1/2/4/8");
using TB=Shape<Int<TBM>,_128,_256>; using CL=Shape<_1,_1,_1>;

template<class EC, class ED, class LC=LayoutCTag, class LD=LayoutDTag>
struct GemmFor {
  static constexpr int AC=128/cutlass::sizeof_bits<EC>::value, AD=128/cutlass::sizeof_bits<ED>::value;
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,SpOp,TB,CL,cutlass::epilogue::collective::EpilogueTileAuto,Acc,Acc,
    EC,LC,AC,ED,LD,AD,
    cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120>::CollectiveOp;
  using Main=typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,SpOp,ElementA,LayoutATag,AA,ElementB,LayoutBTag,AB,Acc,TB,CL,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epi::SharedStorage)>,
    SpSched>::CollectiveOp;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,void>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using SpGemm=typename GemmFor<ElementC,ElementD>::Gemm;
#ifdef D1E4M3
using G1G=typename GemmFor<ED1,ED1,LayoutD1Tag,LayoutD1Tag>::Gemm;
#else
using G1G=typename GemmFor<ED1,ED1>::Gemm;
#endif
using StrideD1=typename G1G::GemmKernel::StrideD;
using StrideC1=typename G1G::GemmKernel::StrideC;
using StrideA=typename SpGemm::GemmKernel::StrideA; using LayoutA=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutA;
using StrideB=typename SpGemm::GemmKernel::StrideB; using StrideC=typename SpGemm::GemmKernel::StrideC; using StrideD=typename SpGemm::GemmKernel::StrideD;
using LayoutE=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutE;
using SpCfg=typename SpGemm::GemmKernel::CollectiveMainloop::SparseConfig;
using BlkCfg=typename SpGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ---- unified fused gate + quantize + 4:8 pair-compress + SFA emit ----------
// One thread = 32 logical fp4 along k of one OPERAND row m.
// TR:   operand is the transpose of storage (storage R=K rows x C=M cols,
//       row-major; elem(m,k)=src[k*pitch+m], pitch=M). TR=false: pitch=K.
// MIN:  apply STE mask before gating (mask nibbles in STORAGE orientation:
//       byte at (row, col/8), pair (col%8)/2 kept iff == lo || == hi).
// MOUT: emit this pass's gate decisions (same storage-orientation layout;
//       only meaningful for TR=false where operand==storage).
template<bool TR, bool MIN, bool MOUT, class LSF>
__global__ void gqc_k(const __nv_bfloat16* __restrict__ src, int pitch,
                      uint8_t* __restrict__ comp, uint8_t* __restrict__ meta,
                      ESF* __restrict__ sfa, LSF layoutSFA,
                      const uint8_t* __restrict__ mIn, uint8_t* __restrict__ mOut,
                      int M, int K){
  int Kb=K/SFVEC;
  size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if (g>=(size_t)M*Kb) return;
  int m=g/Kb, seg=g%Kb, k0=seg*SFVEC;
  float x[SFVEC];
  if (!TR){
    const uint4* pr=reinterpret_cast<const uint4*>(src+(size_t)m*pitch+k0);
    __nv_bfloat16 xb[SFVEC];
    #pragma unroll
    for(int i=0;i<SFVEC/8;++i) reinterpret_cast<uint4*>(xb)[i]=pr[i];
    #pragma unroll
    for(int v=0;v<SFVEC;++v) x[v]=__bfloat162float(xb[v]);
    if (MIN){
      #pragma unroll
      for(int c=0;c<4;++c){
        uint8_t nb=mIn[(size_t)m*(pitch/8)+(k0/8)+c];
        int lo=nb&3, hi=(nb>>2)&3;
        #pragma unroll
        for(int j=0;j<4;++j) if(j!=lo&&j!=hi){ x[8*c+2*j]=0.f; x[8*c+2*j+1]=0.f; }
      }
    }
  } else {
    #pragma unroll
    for(int v=0;v<SFVEC;++v){
      int k=k0+v;
      float val=__bfloat162float(src[(size_t)k*pitch+m]);
      if (MIN){
        uint8_t nb=mIn[(size_t)k*(pitch/8)+(m/8)];
        int lo=nb&3, hi=(nb>>2)&3, pj=(m%8)/2;
        if (pj!=lo && pj!=hi) val=0.f;
      }
      x[v]=val;
    }
  }
  // gate: 4 chunks of 4 pairs; keep top-2 pairs by |a|+|b| (stable, float)
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
    keep[2*c]=8*c+2*lo; keep[2*c+1]=8*c+2*hi;
    nib[c]=uint8_t(lo|(hi<<2));
    float m1=fmaxf(fabsf(p[2*lo]),fabsf(p[2*lo+1])), m2=fmaxf(fabsf(p[2*hi]),fabsf(p[2*hi+1]));
    amax=fmaxf(amax,fmaxf(m1,m2));
  }
  if (MOUT){
    uint8_t* mo=mOut+(size_t)m*(K/8)+(k0/8);
    #pragma unroll
    for(int c=0;c<4;++c) mo[c]=nib[c];
  }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT=make_tensor(cute::make_gmem_ptr(sfa),layoutSFA); sfT(m,k0,0)=q;
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

// ---- smem-tile-transposed gate+quant+compress (replaces gqc_k<true,...>) ----
// Tile: KT=128 samples x MT=64 feats. Coalesced uint4 row loads -> padded smem
// -> per-thread 32-sample column reads. One block = 256 threads = 64 feats x 4 segs.
constexpr int KT=128, MT=64, SMP=66; // pad 64->66 (stride 33 words, conflict-free)
template<bool MIN, class LSF>
__global__ void gqcT_s(const __nv_bfloat16* __restrict__ src, int F,
                       uint8_t* __restrict__ comp, uint8_t* __restrict__ meta,
                       ESF* __restrict__ sfa, LSF layoutSFA,
                       const uint8_t* __restrict__ mIn,
                       int M, int K){          // operand: M=feats rows, K=samples
  __shared__ __nv_bfloat16 sm[KT][SMP];
  __shared__ uint8_t mk[KT][MT/8];
  int k0t=blockIdx.x*KT, m0t=blockIdx.y*MT;
  int tid=threadIdx.x;
  #pragma unroll
  for (int i=0;i<4;++i){                       // 1024 uint4 = 8192 bf16
    int idx=tid+i*256, row=idx/(MT/8), c8=idx%(MT/8);
    uint4 v=reinterpret_cast<const uint4*>(src+(size_t)(k0t+row)*F+m0t+c8*8)[0];
    uint32_t* d=reinterpret_cast<uint32_t*>(&sm[row][c8*8]);   // 132B row stride: 4B-aligned
    d[0]=v.x; d[1]=v.y; d[2]=v.z; d[3]=v.w;
  }
  if (MIN){
    int row=tid/2, half=tid%2;
    reinterpret_cast<uint32_t*>(&mk[row][half*4])[0]=
      reinterpret_cast<const uint32_t*>(mIn+(size_t)(k0t+row)*(F/8)+m0t/8+half*4)[0];
  }
  __syncthreads();
  int ml=tid/4, sl=tid%4;                      // feat-local, seg-local
  int m=m0t+ml, k0=k0t+sl*SFVEC;
  float x[SFVEC];
  #pragma unroll
  for (int v=0;v<SFVEC;++v){
    float val=__bfloat162float(sm[sl*SFVEC+v][ml]);
    if (MIN){
      uint8_t nb=mk[sl*SFVEC+v][ml/8];
      int lo=nb&3, hi=(nb>>2)&3, pj=(ml%8)/2;
      if (pj!=lo && pj!=hi) val=0.f;
    }
    x[v]=val;
  }
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
    keep[2*c]=8*c+2*lo; keep[2*c+1]=8*c+2*hi;
    nib[c]=uint8_t(lo|(hi<<2));
    float m1=fmaxf(fabsf(p[2*lo]),fabsf(p[2*lo+1])), m2=fmaxf(fabsf(p[2*hi]),fabsf(p[2*hi+1]));
    amax=fmaxf(amax,fmaxf(m1,m2));
  }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT=make_tensor(cute::make_gmem_ptr(sfa),layoutSFA); sfT(m,k0,0)=q;
  uint32_t o0=0,o1=0;
  #pragma unroll
  for(int i=0;i<8;++i){
    uint32_t b8=(EA(x[keep[i]]*rcp).raw()&0xF) | ((EA(x[keep[i]+1]*rcp).raw()&0xF)<<4);
    if(i<4) o0|=b8<<(8*i); else o1|=b8<<(8*(i-4));
  }
  reinterpret_cast<uint2*>(comp)[(size_t)m*(K/SFVEC)+k0/SFVEC]=make_uint2(o0,o1);
  size_t M_alE=size_t((M+127)/128*128);
  size_t eb=size_t(m%128)*16+size_t(m/128)*2048+size_t(k0/256)*(M_alE*16)+size_t(k0%256)/16;
  meta[eb]=uint8_t(nib[0]|(nib[1]<<4)); meta[eb+1]=uint8_t(nib[2]|(nib[3]<<4));
}

// ---- smem-tile-transposed dense T-form quant (replaces strided tqT_k) -------
template<class LSF>
__global__ void tqT_s(const __nv_bfloat16* __restrict__ src, const uint8_t* __restrict__ mask,
                      uint8_t* __restrict__ vals, ESF* __restrict__ sfb, LSF layoutSFB,
                      int S, int F){
  __shared__ __nv_bfloat16 sm[KT][SMP];
  __shared__ uint8_t mk[KT][MT/8];
  int k0t=blockIdx.x*KT, n0t=blockIdx.y*MT;
  int tid=threadIdx.x;
  #pragma unroll
  for (int i=0;i<4;++i){
    int idx=tid+i*256, row=idx/(MT/8), c8=idx%(MT/8);
    uint4 v=reinterpret_cast<const uint4*>(src+(size_t)(k0t+row)*F+n0t+c8*8)[0];
    uint32_t* d=reinterpret_cast<uint32_t*>(&sm[row][c8*8]);
    d[0]=v.x; d[1]=v.y; d[2]=v.z; d[3]=v.w;
  }
  { int row=tid/2, half=tid%2;
    reinterpret_cast<uint32_t*>(&mk[row][half*4])[0]=
      reinterpret_cast<const uint32_t*>(mask+(size_t)(k0t+row)*(F/8)+n0t/8+half*4)[0]; }
  __syncthreads();
  int nl=tid/4, sl=tid%4;
  int n=n0t+nl, k0=k0t+sl*SFVEC;
  float x[SFVEC]; float amax=0.f;
  #pragma unroll
  for (int v=0;v<SFVEC;++v){
    float val=__bfloat162float(sm[sl*SFVEC+v][nl]);
    uint8_t nb=mk[sl*SFVEC+v][nl/8];
    int lo=nb&3, hi=(nb>>2)&3, pj=(nl%8)/2;
    if (pj!=lo && pj!=hi) val=0.f;
    x[v]=val; amax=fmaxf(amax,fabsf(val));
  }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT=make_tensor(cute::make_gmem_ptr(sfb),layoutSFB); sfT(n,k0,0)=q;
  uint32_t o[4];
  #pragma unroll
  for(int i=0;i<4;++i){
    uint32_t w=0;
    #pragma unroll
    for(int b=0;b<4;++b){
      uint32_t b8=(EA(x[8*i+2*b]*rcp).raw()&0xF) | ((EA(x[8*i+2*b+1]*rcp).raw()&0xF)<<4);
      w|=b8<<(8*b);
    }
    o[i]=w;
  }
  reinterpret_cast<uint4*>(vals+((size_t)n*S+k0)/2)[0]=make_uint4(o[0],o[1],o[2],o[3]);
}

// ---- dense T-form quant: gated activation -> fp4 dense B (col-major, K=samples)
// One thread = one feature n x 32 samples k0..k0+31. Reads STORAGE row-major
// [S x F] strided; applies the fwd gate mask; SFB block-32 along K=samples.
template<class LSF>
__global__ void tqT_k(const __nv_bfloat16* __restrict__ src, const uint8_t* __restrict__ mask,
                      uint8_t* __restrict__ vals, ESF* __restrict__ sfb, LSF layoutSFB,
                      int S, int F){
  int Kb=S/SFVEC;
  size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if (g>=(size_t)F*Kb) return;
  int n=g/Kb, seg=g%Kb, k0=seg*SFVEC;
  float x[SFVEC]; float amax=0.f;
  #pragma unroll
  for(int v=0;v<SFVEC;++v){
    int k=k0+v;
    float val=__bfloat162float(src[(size_t)k*F+n]);
    uint8_t nb=mask[(size_t)k*(F/8)+(n/8)];
    int lo=nb&3, hi=(nb>>2)&3, pj=(n%8)/2;
    if (pj!=lo && pj!=hi) val=0.f;
    x[v]=val; amax=fmaxf(amax,fabsf(val));
  }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT=make_tensor(cute::make_gmem_ptr(sfb),layoutSFB); sfT(n,k0,0)=q;
  uint32_t o[4];
  #pragma unroll
  for(int i=0;i<4;++i){
    uint32_t w=0;
    #pragma unroll
    for(int b=0;b<4;++b){
      uint32_t b8=(EA(x[8*i+2*b]*rcp).raw()&0xF) | ((EA(x[8*i+2*b+1]*rcp).raw()&0xF)<<4);
      w|=b8<<(8*b);
    }
    o[i]=w;
  }
  reinterpret_cast<uint4*>(vals+((size_t)n*S+k0)/2)[0]=make_uint4(o[0],o[1],o[2],o[3]);
}

// ---- STE + residual: G' = mask (.) dA1 + G  (one thread = 8 elems = 1 mask byte)
__global__ void resid_k(const __nv_bfloat16* __restrict__ dA1, const uint8_t* __restrict__ mask,
                        const __nv_bfloat16* __restrict__ G, __nv_bfloat16* __restrict__ Gn,
                        size_t total8){
  size_t g=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if (g>=total8) return;
  uint4 a=reinterpret_cast<const uint4*>(dA1)[g], r=reinterpret_cast<const uint4*>(G)[g];
  __nv_bfloat16 xa[8], xg[8];
  reinterpret_cast<uint4*>(xa)[0]=a; reinterpret_cast<uint4*>(xg)[0]=r;
  uint8_t nb=mask[g]; int lo=nb&3, hi=(nb>>2)&3;
  #pragma unroll
  for(int j=0;j<4;++j){
    bool kept=(j==lo)||(j==hi);
    float v0=kept?__bfloat162float(xa[2*j]):0.f, v1=kept?__bfloat162float(xa[2*j+1]):0.f;
    xa[2*j]  =__float2bfloat16(v0+__bfloat162float(xg[2*j]));
    xa[2*j+1]=__float2bfloat16(v1+__bfloat162float(xg[2*j+1]));
  }
  reinterpret_cast<uint4*>(Gn)[g]=reinterpret_cast<uint4*>(xa)[0];
}

// ============== FUSED TILE KERNELS (one read, all consumers) =================
// Shared device helpers: gate decision per 8-chunk + fp4 pack.
__device__ __forceinline__ void gate8(const float* p, int& lo, int& hi){
#ifdef GATE01
  lo=0; hi=1; (void)p;
#else
  float sc[4]; int a=0,b=1;
  #pragma unroll
  for(int j=0;j<4;++j) sc[j]=fabsf(p[2*j])+fabsf(p[2*j+1]);
  float sa=sc[0];
  if(sc[1]>sa){sa=sc[1];a=1;} if(sc[2]>sa){sa=sc[2];a=2;} if(sc[3]>sa){sa=sc[3];a=3;}
  b=(a==0)?1:0; float sb=sc[b];
  #pragma unroll
  for(int j=0;j<4;++j) if(j!=a&&j!=b&&sc[j]>sb){sb=sc[j];b=j;}
  lo=a<b?a:b; hi=a<b?b:a;
#endif
}
// gate+quant+compress 32 values of operand row m at k0 -> comp/meta/SFA writes.
// Two-pass, fully predicated (no dynamic register indexing -> no local-mem
// spill), native paired e2m1 cvt (no software converter SM-throttle).
template<class LSF>
__device__ __forceinline__ void gqc_emit(const float* x, int m, int k0,
    uint8_t* comp, uint8_t* meta, ESF* sfa, LSF& layoutSFA, int M, int K){
  int lo[4],hi[4]; float amax=0.f;
  #pragma unroll
  for(int c=0;c<4;++c){
    const float* p=x+8*c;
    gate8(p,lo[c],hi[c]);
    float mj, ac=0.f;
    #pragma unroll
    for(int j=0;j<4;++j){
      mj=fmaxf(fabsf(p[2*j]),fabsf(p[2*j+1]));
      ac=fmaxf(ac,(j==lo[c]||j==hi[c])?mj:0.f);
    }
    amax=fmaxf(amax,ac);
  }
  float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
  auto sfT=make_tensor(cute::make_gmem_ptr(sfa),layoutSFA); sfT(m,k0,0)=q;
  uint32_t o0=0,o1=0;
  #pragma unroll
  for(int c=0;c<4;++c){
    const float* p=x+8*c;
    uint32_t bj[4];
    #pragma unroll
    for(int j=0;j<4;++j) bj[j]=cvtE2M1x2(p[2*j+1]*rcp,p[2*j]*rcp);
    uint32_t blo=bj[0], bhi=bj[0];
    #pragma unroll
    for(int j=1;j<4;++j){ blo=(lo[c]==j)?bj[j]:blo; bhi=(hi[c]==j)?bj[j]:bhi; }
    uint32_t hw=blo|(bhi<<8);
    if(c<2) o0|=hw<<(16*c); else o1|=hw<<(16*(c-2));
  }
  reinterpret_cast<uint2*>(comp)[(size_t)m*(K/SFVEC)+k0/SFVEC]=make_uint2(o0,o1);
  size_t M_alE=size_t((M+127)/128*128);
  size_t eb=size_t(m%128)*16+size_t(m/128)*2048+size_t(k0/256)*(M_alE*16)+size_t(k0%256)/16;
  meta[eb]=uint8_t(lo[0]|(hi[0]<<2)|((lo[1]|(hi[1]<<2))<<4));
  meta[eb+1]=uint8_t(lo[2]|(hi[2]<<2)|((lo[3]|(hi[3]<<2))<<4));
}
// mask nib -> keep test for element pair-index pj
__device__ __forceinline__ bool mkeep(uint8_t nb, int pj){ int lo=nb&3, hi=(nb>>2)&3; return pj==lo||pj==hi; }

// K_fwd: X tile -> (A-compressed feats-gated + mask emit) AND (T-form dense fp4)
template<bool COL, class SRC, class LSFA, class LSFB>
__global__ void k_fwd(const SRC* __restrict__ src, int F,
                      uint8_t* __restrict__ comp, uint8_t* __restrict__ meta, ESF* __restrict__ sfa, LSFA layoutSFA,
                      uint8_t* __restrict__ mOut,
                      uint8_t* __restrict__ tvals, ESF* __restrict__ tsfb, LSFB layoutSFB,
                      int M, int K){            // M=samples, K=feats (square: also S)
  using SMT=std::conditional_t<(sizeof(SRC)==1),__half,__nv_bfloat16>;
  __shared__ SMT sm[KT][SMP];
  __shared__ uint8_t mk[KT][MT/8];
  int k0t=blockIdx.x*KT, m0t=blockIdx.y*MT;     // k0t: sample base; m0t: feat base
  int tid=threadIdx.x;
  if constexpr(COL){
    // Column-major fp8 D1: 16 contiguous samples per load, transpose into [sample][feat] smem.
    #pragma unroll
    for(int i=0;i<2;++i){
      int idx=tid+i*256, fl=idx/(KT/16), s16=idx%(KT/16);
      const uint16_t* pw=reinterpret_cast<const uint16_t*>(src+(size_t)(m0t+fl)*F+k0t+s16*16);
      #pragma unroll
      for(int j=0;j<8;++j){
        __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)pw[j],__NV_E4M3);
        sm[s16*16+2*j][fl]=*reinterpret_cast<__half*>(&h.x);
        sm[s16*16+2*j+1][fl]=*reinterpret_cast<__half*>(&h.y);
      }
    }
  } else {
    #pragma unroll
    for (int i=0;i<4;++i){
      int idx=tid+i*256, row=idx/(MT/8), c8=idx%(MT/8);
      const SRC* p=src+(size_t)(k0t+row)*F+m0t+c8*8;
      uint4 v=reinterpret_cast<const uint4*>(p)[0];
      uint32_t* d=reinterpret_cast<uint32_t*>(&sm[row][c8*8]);
      d[0]=v.x; d[1]=v.y; d[2]=v.z; d[3]=v.w;
    }
  }
  __syncthreads();
  // phase 1: direct (operand rows = samples, contraction = feats)
  { int sl=tid/2, fl=tid%2;                     // 128 samples x 2 feat-segs
    int m=k0t+sl, k0=m0t+fl*SFVEC;
    float x[SFVEC];
    #pragma unroll
    for(int v=0;v<SFVEC;++v) x[v]=float(sm[sl][fl*SFVEC+v]);
    // gate decisions + mask emit (global + smem), then full emit
    uint8_t* mo=mOut+(size_t)m*(K/8)+(k0/8);
    #pragma unroll
    for(int c=0;c<4;++c){
      int lo,hi; gate8(x+8*c,lo,hi);
      uint8_t nb=uint8_t(lo|(hi<<2));
      mo[c]=nb; mk[sl][fl*4+c]=nb;
    }
    gqc_emit(x,m,k0,comp,meta,sfa,layoutSFA,M,K);
  }
  __syncthreads();
  // phase 2: T-form dense (feat-major, K=samples), gated by phase-1 mask
  { int nl=tid/4, sl2=tid%4;                    // 64 feats x 4 sample-segs
    int n=m0t+nl, k0=k0t+sl2*SFVEC;
    float x[SFVEC]; float amax=0.f;
    #pragma unroll
    for(int v=0;v<SFVEC;++v){
      int kl=sl2*SFVEC+v;
      float val=float(sm[kl][nl]);
      if (!mkeep(mk[kl][nl/8],(nl%8)/2)) val=0.f;
      x[v]=val; amax=fmaxf(amax,fabsf(val));
    }
    float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
    auto sfT=make_tensor(cute::make_gmem_ptr(tsfb),layoutSFB); sfT(n,k0,0)=q;
    uint32_t o[4];
    #pragma unroll
    for(int i=0;i<4;++i){
      uint32_t w=0;
      #pragma unroll
      for(int b=0;b<4;++b) w|=cvtE2M1x2(x[8*i+2*b+1]*rcp,x[8*i+2*b]*rcp)<<(8*b);
      o[i]=w;
    }
    reinterpret_cast<uint4*>(tvals+((size_t)n*M+k0)/2)[0]=make_uint4(o[0],o[1],o[2],o[3]);
  }
}

// K_bwdG: (dA1_prev, G_prev, mX_prev) -> Gnext inline (bf16-rounded, materialized)
//         -> Gc_f (feats-gated) + Gc_s (samples-gated). dA1==null => Gnext=G.
template<class LSF>
__global__ void k_bwdG(const __nv_bfloat16* __restrict__ dA1, const __nv_bfloat16* __restrict__ G,
                       const uint8_t* __restrict__ mX, __nv_bfloat16* __restrict__ Gn, int F,
                       uint8_t* __restrict__ compF, uint8_t* __restrict__ metaF, ESF* __restrict__ sfaF,
                       uint8_t* __restrict__ compS, uint8_t* __restrict__ metaS, ESF* __restrict__ sfaS,
                       LSF layoutSFA, int M, int K){
  __shared__ __nv_bfloat16 sm[KT][SMP];
  int k0t=blockIdx.x*KT, m0t=blockIdx.y*MT;
  int tid=threadIdx.x;
  #pragma unroll
  for (int i=0;i<4;++i){
    int idx=tid+i*256, row=idx/(MT/8), c8=idx%(MT/8);
    size_t off=(size_t)(k0t+row)*F+m0t+c8*8;
    uint4 vg=reinterpret_cast<const uint4*>(G+off)[0];
    __nv_bfloat16 xg[8]; reinterpret_cast<uint4*>(xg)[0]=vg;
    if (dA1){
      uint4 va=reinterpret_cast<const uint4*>(dA1+off)[0];
      __nv_bfloat16 xa[8]; reinterpret_cast<uint4*>(xa)[0]=va;
      uint8_t nb=mX[(size_t)(k0t+row)*(F/8)+(m0t+c8*8)/8];
      #pragma unroll
      for(int j=0;j<4;++j){
        bool kept=mkeep(nb,j);
        float v0=kept?__bfloat162float(xa[2*j]):0.f, v1=kept?__bfloat162float(xa[2*j+1]):0.f;
        xg[2*j]  =__float2bfloat16(v0+__bfloat162float(xg[2*j]));
        xg[2*j+1]=__float2bfloat16(v1+__bfloat162float(xg[2*j+1]));
      }
    }
    reinterpret_cast<uint4*>(Gn+off)[0]=reinterpret_cast<uint4*>(xg)[0];
    uint32_t* d=reinterpret_cast<uint32_t*>(&sm[row][c8*8]);
    d[0]=reinterpret_cast<uint32_t*>(xg)[0]; d[1]=reinterpret_cast<uint32_t*>(xg)[1];
    d[2]=reinterpret_cast<uint32_t*>(xg)[2]; d[3]=reinterpret_cast<uint32_t*>(xg)[3];
  }
  __syncthreads();
  { int sl=tid/2, fl=tid%2;                     // feats-gated (direct)
    int m=k0t+sl, k0=m0t+fl*SFVEC;
    float x[SFVEC];
    #pragma unroll
    for(int v=0;v<SFVEC;++v) x[v]=__bfloat162float(sm[sl][fl*SFVEC+v]);
    gqc_emit(x,m,k0,compF,metaF,sfaF,layoutSFA,M,K);
  }
  { int nl=tid/4, sl2=tid%4;                    // samples-gated (T)
    int n=m0t+nl, k0=k0t+sl2*SFVEC;
    float x[SFVEC];
    #pragma unroll
    for(int v=0;v<SFVEC;++v) x[v]=__bfloat162float(sm[sl2*SFVEC+v][nl]);
    gqc_emit(x,n,k0,compS,metaS,sfaS,layoutSFA,M,K);
  }
}

// K_bwdD: (dA2, mD) -> dc_f (feats-gated) + dc_s (samples-gated), STE mask inline
template<class LSF>
__global__ void k_bwdD(const __nv_bfloat16* __restrict__ dA2, const uint8_t* __restrict__ mD, int F,
                       uint8_t* __restrict__ compF, uint8_t* __restrict__ metaF, ESF* __restrict__ sfaF,
                       uint8_t* __restrict__ compS, uint8_t* __restrict__ metaS, ESF* __restrict__ sfaS,
                       LSF layoutSFA, int M, int K){
  __shared__ __nv_bfloat16 sm[KT][SMP];
  __shared__ uint8_t mk[KT][MT/8];
  int k0t=blockIdx.x*KT, m0t=blockIdx.y*MT;
  int tid=threadIdx.x;
  #pragma unroll
  for (int i=0;i<4;++i){
    int idx=tid+i*256, row=idx/(MT/8), c8=idx%(MT/8);
    uint4 v=reinterpret_cast<const uint4*>(dA2+(size_t)(k0t+row)*F+m0t+c8*8)[0];
    uint32_t* d=reinterpret_cast<uint32_t*>(&sm[row][c8*8]);
    d[0]=v.x; d[1]=v.y; d[2]=v.z; d[3]=v.w;
  }
  { int row=tid/2, half=tid%2;
    reinterpret_cast<uint32_t*>(&mk[row][half*4])[0]=
      reinterpret_cast<const uint32_t*>(mD+(size_t)(k0t+row)*(F/8)+m0t/8+half*4)[0]; }
  __syncthreads();
  { int sl=tid/2, fl=tid%2;
    int m=k0t+sl, k0=m0t+fl*SFVEC;
    float x[SFVEC];
    #pragma unroll
    for(int v=0;v<SFVEC;++v){
      float val=__bfloat162float(sm[sl][fl*SFVEC+v]);
      if(!mkeep(mk[sl][fl*4+v/8],(v%8)/2)) val=0.f;
      x[v]=val;
    }
    gqc_emit(x,m,k0,compF,metaF,sfaF,layoutSFA,M,K);
  }
  { int nl=tid/4, sl2=tid%4;
    int n=m0t+nl, k0=k0t+sl2*SFVEC;
    float x[SFVEC];
    #pragma unroll
    for(int v=0;v<SFVEC;++v){
      int kl=sl2*SFVEC+v;
      float val=__bfloat162float(sm[kl][nl]);
      if(!mkeep(mk[kl][nl/8],(nl%8)/2)) val=0.f;
      x[v]=val;
    }
    gqc_emit(x,n,k0,compS,metaS,sfaS,layoutSFA,M,K);
  }
}

// host weight quantizer (donor: fp4_actsparse_fwd.cu)
template<class TV, class TSF>
static void hquant_wB(const std::vector<float>& W, int N, int K, TV&& vT, TSF&& sfT, std::vector<float>& Wsim){
  for (int n=0;n<N;++n) for (int k0=0;k0<K;k0+=SFVEC){
    float amax=0.f; for (int v=0;v<SFVEC;++v) amax=std::max(amax,std::fabs(W[(size_t)n*K+k0+v]));
    float pv=amax*(1.0f/6.0f); ESF q=(ESF)pv; float deq=(float)q; float rcp=deq>0.f?1.0f/deq:0.f;
    sfT(n,k0,0)=q;
    for (int v=0;v<SFVEC;++v){ float qv=float(EA(W[(size_t)n*K+k0+v]*rcp)); vT(n,k0+v,0)=EA(qv); Wsim[(size_t)n*K+k0+v]=qv*deq; }
  }
}

// host gate replica; records decisions (nib per (row, col-octet)) when rec!=null
static void host_gate(std::vector<float>& X, int Mm, int Kk, std::vector<uint8_t>* rec=nullptr){
  if (rec) rec->assign((size_t)Mm*Kk/8,0);
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
    int lo=a<b?a:b, hi=a<b?b:a;
    if (rec) (*rec)[m*(Kk/8)+c0/8]=uint8_t(lo|(hi<<2));
    for(int j=0;j<4;++j) if(j!=a&&j!=b){ p[2*j]=0.f; p[2*j+1]=0.f; }
  }
}
// host gate along SAMPLES (transposed view of row-major [S x F])
static void host_gateT(std::vector<float>& X, int S, int F){
  for (int n=0;n<F;++n) for (int c0=0;c0<S;c0+=8){
    float sc[4]; for(int j=0;j<4;++j) sc[j]=std::fabs(X[(size_t)(c0+2*j)*F+n])+std::fabs(X[(size_t)(c0+2*j+1)*F+n]);
    int a=0,b=1;
#ifdef GATE01
    (void)sc;
#else
    float sa=sc[0];
    if(sc[1]>sa){sa=sc[1];a=1;} if(sc[2]>sa){sa=sc[2];a=2;} if(sc[3]>sa){sa=sc[3];a=3;}
    b=(a==0)?1:0; float sb=sc[b];
    for(int j=0;j<4;++j) if(j!=a&&j!=b&&sc[j]>sb){sb=sc[j];b=j;}
#endif
    for(int j=0;j<4;++j) if(j!=a&&j!=b){ X[(size_t)(c0+2*j)*F+n]=0.f; X[(size_t)(c0+2*j+1)*F+n]=0.f; }
  }
}
static void host_mask_apply(std::vector<float>& X, const std::vector<uint8_t>& nibs, int S, int F){
  for (size_t s=0;s<(size_t)S;++s) for (int c0=0;c0<F;c0+=8){
    uint8_t nb=nibs[s*(F/8)+c0/8]; int lo=nb&3, hi=(nb>>2)&3;
    for(int j=0;j<4;++j) if(j!=lo&&j!=hi){ X[s*F+c0+2*j]=0.f; X[s*F+c0+2*j+1]=0.f; }
  }
}
// C[m][n] = sum_k A[m][k]*B[n*K+k]  (B col-major sim)
static void host_gemm(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K){
  #pragma omp parallel for
  for (int m=0;m<M;++m) for (int n=0;n<N;++n){ double acc=0;
    for (int k=0;k<K;++k) acc+=(double)A[(size_t)m*K+k]*B[(size_t)n*K+k];
    C[(size_t)m*N+n]=(float)acc; }
}
// C[m][n] = sum_k A[k][m]*B[k][n]   (A^T . B, both row-major [K x *])
static void host_gemmTA(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, int M, int N, int K){
  #pragma omp parallel for
  for (int m=0;m<M;++m) for (int n=0;n<N;++n){ double acc=0;
    for (int k=0;k<K;++k) acc+=(double)A[(size_t)k*M+m]*B[(size_t)k*N+n];
    C[(size_t)m*N+n]=(float)acc; }
}
static double relpj(const std::vector<float>& ref, const cutlass::bfloat16_t* dev, size_t n){
  double num=0,den=0;
  for (size_t i=0;i<n;++i){ double d=(double)float(dev[i])-ref[i]; num+=d*d; den+=ref[i]*ref[i]; }
  return std::sqrt(num/(den+1e-30));
}

int main(int argc,char**argv){
  int S    = argc>1?atoi(argv[1]):8192;
  int iters= argc>2?atoi(argv[2]):30;
  int L    = argc>3?atoi(argv[3]):8;
  int validate = argc>4?atoi(argv[4]):0;
  int MB   = argc>5?atoi(argv[5]):1;
  int M=S,N=S,K=S;
  constexpr int G=GROUPS;
  if (K%256 || M%128 || S%G || (S/G)%256){
    printf("need S%%256==0, S%%GROUPS==0, and (S/GROUPS)%%256==0\n"); return 1;
  }
  int F=S/G; // contiguous feature partitions; one diagonal F x F weight block per group
  cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  // Full tensor layouts stay unchanged: fused quantizers still emit every activation/gradient.
  // Group GEMMs consume aligned subviews. Compact block weights use the F x F B layouts.
  auto sA=cutlass::make_cute_packed_stride(StrideA{},{M,K,1});
  auto sB=cutlass::make_cute_packed_stride(StrideB{},{N,K,1});
  auto sBG=cutlass::make_cute_packed_stride(StrideB{},{F,F,1});
  auto sC=cutlass::make_cute_packed_stride(StrideC{},{M,N,1});
  auto sD=cutlass::make_cute_packed_stride(StrideD{},{M,N,1});
  auto sC1=cutlass::make_cute_packed_stride(StrideC1{},{M,N,1});
  auto sD1=cutlass::make_cute_packed_stride(StrideD1{},{M,N,1});
  auto lSFA=BlkCfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB=BlkCfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto lSFBG=BlkCfg::tile_atom_to_shape_SFB(make_shape(M,F,F,1));
  auto wl=make_shape(M,1,K,1);
  cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg> util(wl,sA);
  LayoutA lA=SpCfg::fill_layoutA(wl); LayoutE lE=SpCfg::fill_layoutE(wl);
  int m_phys=util.get_tensorA_m_physical(), k_phys=util.get_tensorA_k_physical();
  int me_phys=util.get_metadata_m_physical(), ke_phys=util.get_metadata_k_physical();

  std::normal_distribution<float> g(0.f,1.f);
  float wscale=1.0f/std::sqrt((float)F); // fan-in preserving for a 1/G-density block row

  // Compact diagonal weight blocks: index l*G+g. Forward and transpose forms are
  // independently quantized, exactly as the dense harness did. GROUPS=1 is unchanged.
  struct Wt { cutlass::HostTensor<EA,cutlass::layout::PackedVectorLayout> vals;
              cutlass::HostTensor<ESF,cutlass::layout::PackedVectorLayout> sfb;
              std::vector<float> sim; };
  std::vector<Wt> W1(L*G), W2(L*G), W1T(L*G), W2T(L*G);
  auto quant_into=[&](Wt& w, const std::vector<float>& Wf){
    w.vals.reset(cutlass::make_Coord((size_t)F*F));
    w.sfb.reset(cutlass::make_Coord(size(filter_zeros(lSFBG))));
    w.sim.assign((size_t)F*F,0.f);
    auto vT=make_tensor(make_iterator(w.vals.host_data()), make_layout(make_shape(F,F,1),sBG));
    auto sfT=make_tensor(w.sfb.host_data(), lSFBG);
    hquant_wB(Wf,F,F,vT,sfT,w.sim);
    w.vals.sync_device(); w.sfb.sync_device();
  };
  for (int l=0;l<L;++l) for (int gg=0;gg<G;++gg){
    int wi=l*G+gg;
    std::vector<float> Wf((size_t)F*F), WfT((size_t)F*F);
    { std::mt19937 r(100+l*G+gg); for (auto& x:Wf) x=g(r)*wscale; }
    for (int n=0;n<F;++n) for (int k=0;k<F;++k) WfT[(size_t)k*F+n]=Wf[(size_t)n*F+k];
    quant_into(W1[wi],Wf); quant_into(W1T[wi],WfT);
    { std::mt19937 r(500+l*G+gg); for (auto& x:Wf) x=g(r)*wscale; }
    for (int n=0;n<F;++n) for (int k=0;k<F;++k) WfT[(size_t)k*F+n]=Wf[(size_t)n*F+k];
    quant_into(W2[wi],Wf); quant_into(W2T[wi],WfT);
  }

  size_t compB=size_t(m_phys)*k_phys/2, metaB=size_t(me_phys)*ke_phys, sfaN=size_t(size(filter_zeros(lSFA)));
  size_t matB=(size_t)M*N*sizeof(cutlass::bfloat16_t), maskB=(size_t)M*K/8;
  size_t tvalB=(size_t)N*K/2, sfbN=size_t(size(filter_zeros(lSFB)));

  struct ASlot { cutlass::device_memory::allocation<uint8_t> comp, meta;
                 cutlass::device_memory::allocation<ESF> sfa; };
  struct TSlot { cutlass::device_memory::allocation<uint8_t> vals;
                 cutlass::device_memory::allocation<ESF> sfb; };

  // per-microbatch buffer set + pre-initialized GEMMs
  struct MBSet {
    cudaStream_t s, sc;
    std::vector<cudaEvent_t> ev_a1t, ev_a2t, ev_mx, ev_md, ev_gs, ev_da2, ev_dcs;
    cutlass::device_memory::allocation<cutlass::bfloat16_t> Xres0,Xres1,G0,G1v,dA2,dA1,dW1s,dW2s;
    cutlass::device_memory::allocation<ED1> D1;
    std::vector<TSlot> A1T,A2T;
    std::vector<cutlass::device_memory::allocation<uint8_t>> mX,mD;
    ASlot A1c,A2c,GcS[2],GcF[2],DcS[2],DcF[2];
    std::vector<G1G> g1; std::vector<SpGemm> g2,gw2,ga2,gw1,ga1; // each indexed l*G+group
    std::vector<cutlass::device_memory::allocation<uint8_t>> ws;
  };
  std::vector<MBSet> mb(MB);
  auto alloc_aslot=[&](ASlot& a){
    a.comp=cutlass::device_memory::allocation<uint8_t>(compB);
    a.meta=cutlass::device_memory::allocation<uint8_t>(metaB);
    a.sfa =cutlass::device_memory::allocation<ESF>(sfaN);
    cudaMemset(a.comp.get(),0,compB); cudaMemset(a.meta.get(),0,metaB);
  };
  std::mt19937 xr(9001); std::vector<float> X0((size_t)M*K);
  for (auto& x:X0) x=g(xr);
  std::vector<cutlass::bfloat16_t> X0h((size_t)M*K);
  for (size_t i=0;i<(size_t)M*K;++i) X0h[i]=cutlass::bfloat16_t(X0[i]);

  for (int b=0;b<MB;++b){
    MBSet& q=mb[b];
    cudaStreamCreate(&q.s); cudaStreamCreate(&q.sc);
    auto mkev=[&](std::vector<cudaEvent_t>& v){ v.resize(L); for(auto& e:v) cudaEventCreateWithFlags(&e,cudaEventDisableTiming); };
    mkev(q.ev_a1t); mkev(q.ev_a2t); mkev(q.ev_mx); mkev(q.ev_md); mkev(q.ev_gs); mkev(q.ev_da2); mkev(q.ev_dcs);
    auto Am=[&](cutlass::device_memory::allocation<cutlass::bfloat16_t>& a){ a=cutlass::device_memory::allocation<cutlass::bfloat16_t>((size_t)M*N); };
    Am(q.Xres0);Am(q.Xres1);Am(q.G0);Am(q.G1v);Am(q.dA2);Am(q.dA1);Am(q.dW1s);Am(q.dW2s);
    q.D1=cutlass::device_memory::allocation<ED1>((size_t)M*N);
    cudaMemset(q.dW1s.get(),0,matB); cudaMemset(q.dW2s.get(),0,matB); // off-block gradients are structural zeros
    cudaMemcpy(q.Xres0.get(),X0h.data(),matB,cudaMemcpyHostToDevice);
    q.A1T.resize(L); q.A2T.resize(L); q.mX.resize(L); q.mD.resize(L);
    for (int l=0;l<L;++l){
      q.A1T[l].vals=cutlass::device_memory::allocation<uint8_t>(tvalB);
      q.A1T[l].sfb =cutlass::device_memory::allocation<ESF>(sfbN);
      q.A2T[l].vals=cutlass::device_memory::allocation<uint8_t>(tvalB);
      q.A2T[l].sfb =cutlass::device_memory::allocation<ESF>(sfbN);
      q.mX[l]=cutlass::device_memory::allocation<uint8_t>(maskB);
      q.mD[l]=cutlass::device_memory::allocation<uint8_t>(maskB);
    }
    alloc_aslot(q.A1c); alloc_aslot(q.A2c);
    for (int i=0;i<2;++i){ alloc_aslot(q.GcS[i]); alloc_aslot(q.GcF[i]); alloc_aslot(q.DcS[i]); alloc_aslot(q.DcF[i]); }

    q.g1.resize(L*G); q.g2.resize(L*G); q.gw2.resize(L*G); q.ga2.resize(L*G); q.gw1.resize(L*G); q.ga1.resize(L*G);
    cutlass::bfloat16_t* Xr[2]={q.Xres0.get(),q.Xres1.get()};
    auto init=[&](auto& gm, auto args){
      using GG=std::decay_t<decltype(gm)>;
      q.ws.emplace_back(GG::get_workspace_size(args));
      CUTLASS_CHECK(gm.can_implement(args)); CUTLASS_CHECK(gm.initialize(args,q.ws.back().get(),q.s));
    };
    auto meta_off=[&](int m0,int k0){
      size_t Mal=size_t((M+127)/128*128);
      return size_t(m0%128)*16+size_t(m0/128)*2048+size_t(k0/256)*(Mal*16)+size_t(k0%256)/16;
    };
    // Projection/activation-gradient block: A[:,gF:(g+1)F] x compact W[g],
    // writing D[:,gF:(g+1)F] through the original full-row strides.
    auto mkf=[&](ASlot& a, int gg, Wt& w, cutlass::bfloat16_t* c,
                 cutlass::bfloat16_t* d, float beta){
      int k0=gg*F, n0=gg*F;
      EA* av=reinterpret_cast<EA*>(a.comp.get()+size_t(k0)/4);
      uint8_t* ae=a.meta.get()+meta_off(0,k0);
      ESF* as=a.sfa.get()+(size_t)lSFA(0,k0,0);
      typename SpGemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,F,F,1},
        {av,lA, w.vals.device_data(),sBG, ae,lE, as,lSFA, w.sfb.device_data(),lSFBG},
        {{1.f,beta},c?c+n0:nullptr,sC,d+n0,sD}};
      return args;
    };
    auto mkf1=[&](ASlot& a, int gg, Wt& w, ED1* d){
      int k0=gg*F, n0=gg*F;
      EA* av=reinterpret_cast<EA*>(a.comp.get()+size_t(k0)/4);
      uint8_t* ae=a.meta.get()+meta_off(0,k0);
      ESF* as=a.sfa.get()+(size_t)lSFA(0,k0,0);
#ifdef D1E4M3
      ED1* dout=d+(size_t)n0*M;
#else
      ED1* dout=d+n0;
#endif
      typename G1G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,F,F,1},
        {av,lA, w.vals.device_data(),sBG, ae,lE, as,lSFA, w.sfb.device_data(),lSFBG},
        {{1.f,0.f},nullptr,sC1,dout,sD1}};
      return args;
    };
    // Weight-gradient block: A[gF:(g+1)F,:] x Tform[gF:(g+1)F,:],
    // writing only the corresponding diagonal F x F block of the full dW sink.
    auto mkdw=[&](ASlot& a, int gg, TSlot& bt, cutlass::bfloat16_t* d){
      int m0=gg*F, n0=gg*F;
      EA* av=reinterpret_cast<EA*>(a.comp.get()+size_t(m0)*K/4);
      uint8_t* ae=a.meta.get()+meta_off(m0,0);
      ESF* as=a.sfa.get()+(size_t)lSFA(m0,0,0);
      EA* bv=reinterpret_cast<EA*>(bt.vals.get()+size_t(n0)*K/2);
      ESF* bs=bt.sfb.get()+(size_t)lSFB(n0,0,0);
      typename SpGemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{F,F,K,1},
        {av,lA, bv,sB, ae,lE, as,lSFA, bs,lSFB},
        {{1.f,0.f},nullptr,sC,d+(size_t)m0*N+n0,sD}};
      return args;
    };
    for (int l=0;l<L;++l) for (int gg=0;gg<G;++gg){
      int wi=l*G+gg;
      init(q.g1[wi],  mkf1(q.A1c,gg,W1[wi],q.D1.get()));
      init(q.g2[wi],  mkf(q.A2c,gg,W2[wi], Xr[l%2],Xr[(l+1)%2],1.f));
      init(q.gw2[wi], mkdw(q.GcS[l%2],gg,q.A2T[l],q.dW2s.get()));
      init(q.ga2[wi], mkf(q.GcF[l%2],gg,W2T[wi],nullptr,q.dA2.get(),0.f));
      init(q.gw1[wi], mkdw(q.DcS[l%2],gg,q.A1T[l],q.dW1s.get()));
      init(q.ga1[wi], mkf(q.DcF[l%2],gg,W1T[wi],nullptr,q.dA1.get(),0.f));
    }
  }

  int thr=256; dim3 gcG((unsigned)(((size_t)M*(K/SFVEC)+thr-1)/thr)), gcB(thr);
  dim3 gT((unsigned)(K/KT),(unsigned)(M/MT));
  dim3 grR((unsigned)(((size_t)M*N/8+thr-1)/thr));
  // Fused-kernel step. Critical path on s; dW GEMMs on sc (off-path).
  // k_fwd = gqc+tqT (one X read). k_bwdG = resid+gqc+gqcT (one G read).
  // k_bwdD = gqc_m+gqcT_m (one dA2 read). Slots double-buffered by parity.
  auto step=[&](MBSet& q){
    cutlass::bfloat16_t* Xr[2]={q.Xres0.get(),q.Xres1.get()};
    cutlass::bfloat16_t* Gr[2]={q.G0.get(),q.G1v.get()};
    for (int l=0;l<L;++l){
      k_fwd<false><<<gT,256,0,q.s>>>((const __nv_bfloat16*)Xr[l%2],K,
        q.A1c.comp.get(),q.A1c.meta.get(),q.A1c.sfa.get(),lSFA, q.mX[l].get(),
        q.A1T[l].vals.get(),q.A1T[l].sfb.get(),lSFB, M,K);
      for(int gg=0;gg<G;++gg) CUTLASS_CHECK(q.g1[l*G+gg].run(q.s));
#ifdef D1E4M3
      k_fwd<true><<<gT,256,0,q.s>>>((const ED1*)q.D1.get(),K,
#else
      k_fwd<false><<<gT,256,0,q.s>>>((const ED1*)q.D1.get(),K,
#endif
        q.A2c.comp.get(),q.A2c.meta.get(),q.A2c.sfa.get(),lSFA, q.mD[l].get(),
        q.A2T[l].vals.get(),q.A2T[l].sfb.get(),lSFB, M,K);
      for(int gg=0;gg<G;++gg) CUTLASS_CHECK(q.g2[l*G+gg].run(q.s));
    }
    for (int l=L-1;l>=0;--l){
      // G(l) = mX[l+1] (.) dA1(l+1) + G(l+1)  [inline], + both gated-compressed forms
      if (l+2<L) cudaStreamWaitEvent(q.s,q.ev_dcs[l+2],0);   // WAR: gw2(l+2) read GcS[l%2]
      k_bwdG<<<gT,256,0,q.s>>>(
        l==L-1?nullptr:(const __nv_bfloat16*)q.dA1.get(),
        (const __nv_bfloat16*)(l==L-1?Xr[L%2]:Gr[(L-l)%2]), // dL/dX_L aliases X_L: no 134MB copy
        l==L-1?nullptr:q.mX[l+1].get(),
        (__nv_bfloat16*)Gr[(L-1-l)%2], K,
        q.GcF[l%2].comp.get(),q.GcF[l%2].meta.get(),q.GcF[l%2].sfa.get(),
        q.GcS[l%2].comp.get(),q.GcS[l%2].meta.get(),q.GcS[l%2].sfa.get(),
        lSFA, M,K);
      cudaEventRecord(q.ev_gs[l],q.s);
      cudaStreamWaitEvent(q.sc,q.ev_gs[l],0);
      for(int gg=0;gg<G;++gg) CUTLASS_CHECK(q.gw2[l*G+gg].run(q.sc)); // dW2^T blocks (sink, off-path)
      cudaEventRecord(q.ev_dcs[l],q.sc);
      for(int gg=0;gg<G;++gg) CUTLASS_CHECK(q.ga2[l*G+gg].run(q.s)); // dA2 blocks
      if (l+2<L) cudaStreamWaitEvent(q.s,q.ev_a2t[l+2],0);   // WAR: gw1(l+2) read DcS[l%2]
      k_bwdD<<<gT,256,0,q.s>>>((const __nv_bfloat16*)q.dA2.get(),q.mD[l].get(),K,
        q.DcF[l%2].comp.get(),q.DcF[l%2].meta.get(),q.DcF[l%2].sfa.get(),
        q.DcS[l%2].comp.get(),q.DcS[l%2].meta.get(),q.DcS[l%2].sfa.get(),
        lSFA, M,K);
      cudaEventRecord(q.ev_da2[l],q.s);
      cudaStreamWaitEvent(q.sc,q.ev_da2[l],0);
      for(int gg=0;gg<G;++gg) CUTLASS_CHECK(q.gw1[l*G+gg].run(q.sc)); // dW1^T blocks (sink, off-path)
      cudaEventRecord(q.ev_a2t[l],q.sc);
      for(int gg=0;gg<G;++gg) CUTLASS_CHECK(q.ga1[l*G+gg].run(q.s)); // dA1 blocks
    }
    resid_k<<<grR,gcB,0,q.s>>>((const __nv_bfloat16*)q.dA1.get(),q.mX[0].get(),
      (const __nv_bfloat16*)Gr[(L-1)%2],(__nv_bfloat16*)Gr[L%2],(size_t)M*N/8);  // dX0
    cudaEventRecord(q.ev_mx[0],q.sc); cudaStreamWaitEvent(q.s,q.ev_mx[0],0);     // join
  };

  step(mb[0]);
  cudaError_t e0=cudaDeviceSynchronize();
  if(e0){ printf("first step failed: %s\n",cudaGetErrorString(e0)); return 1; }

  if (validate){
    MBSet& q=mb[0];
    size_t MN=(size_t)M*N;
    std::vector<cutlass::bfloat16_t> hXL(MN), hGf(MN), hW1(MN), hW2(MN);
    cudaMemcpy(hXL.data(), (L%2)?q.Xres1.get():q.Xres0.get(), matB, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGf.data(), (L%2)?q.G1v.get():q.G0.get(),      matB, cudaMemcpyDeviceToHost);
    cudaMemcpy(hW1.data(), q.dW1s.get(), matB, cudaMemcpyDeviceToHost);
    cudaMemcpy(hW2.data(), q.dW2s.get(), matB, cudaMemcpyDeviceToHost);

    // Host block-GEMM replicas. They become the original full GEMMs at GROUPS=1.
    auto hgw=[&](const std::vector<float>& A, const std::vector<Wt>& W, int l, std::vector<float>& C){
      C.assign((size_t)S*S,0.f);
      #pragma omp parallel for collapse(2)
      for(int gg=0;gg<G;++gg) for(int m=0;m<S;++m){
        const auto& B=W[l*G+gg].sim; int b=gg*F;
        for(int n=0;n<F;++n){ double acc=0;
          for(int k=0;k<F;++k) acc+=(double)A[(size_t)m*S+b+k]*B[(size_t)n*F+k];
          C[(size_t)m*S+b+n]=(float)acc;
        }
      }
    };
    auto hgdw=[&](const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C){
      C.assign((size_t)S*S,0.f);
      #pragma omp parallel for collapse(2)
      for(int gg=0;gg<G;++gg) for(int m=0;m<F;++m){
        int b=gg*F;
        for(int n=0;n<F;++n){ double acc=0;
          for(int k=0;k<S;++k) acc+=(double)A[(size_t)k*S+b+m]*B[(size_t)k*S+b+n];
          C[(size_t)(b+m)*S+b+n]=(float)acc;
        }
      }
    };

    // host fwd: gated chain (records masks + per-layer gated activations) + ungated chain
    std::vector<std::vector<float>> Xg_l(L), Dg_l(L);
    std::vector<std::vector<uint8_t>> mXh(L), mDh(L);
    std::vector<float> X(X0), Xu(X0), tmp(MN);
    for (int l=0;l<L;++l){
      Xg_l[l]=X; host_gate(Xg_l[l],M,K,&mXh[l]);
      hgw(Xg_l[l],W1,l,tmp);
#ifdef D1E4M3
      for(auto& x:tmp) x=float(ED1(x));
#endif
      Dg_l[l]=tmp; host_gate(Dg_l[l],M,K,&mDh[l]);
      hgw(Dg_l[l],W2,l,tmp);
      for (size_t i=0;i<MN;++i) X[i]+=tmp[i];
    }
    { std::vector<float> t2(MN);
      for (int l=0;l<L;++l){ hgw(Xu,W1,l,t2);
#ifdef D1E4M3
        for(auto& x:t2) x=float(ED1(x));
#endif
        hgw(t2,W2,l,tmp);
        for (size_t i=0;i<MN;++i) Xu[i]+=tmp[i]; } }
    printf("VALIDATE fwd  S=%d L=%d  relPJ vs GATED ref = %.4f   vs UNGATED ref = %.4f\n",
           S,L, relpj(X,hXL.data(),MN), relpj(Xu,hXL.data(),MN));

    // host bwd: pass 0 = GATED-grad ref (replicates the 4:8 grad gates in fp32),
    //           pass 1 = UNGATED-grad ref (grad gates off; STE masks stay)
    for (int pass=0; pass<2; ++pass){
      std::vector<float> G(X), dW1r, dW2r, dA2(MN), dD1, dA1(MN), Gs, Gf, d1s, d1f;
      for (int l=L-1;l>=0;--l){
        Gs=G; if(!pass) host_gateT(Gs,M,N);
        Gf=G; if(!pass) host_gate(Gf,M,N);
        if (l==0){ hgdw(Gs,Dg_l[l],dW2r); }
        hgw(Gf,W2T,l,dA2);
        dD1=dA2; host_mask_apply(dD1,mDh[l],M,N);
        d1s=dD1; if(!pass) host_gateT(d1s,M,N);
        d1f=dD1; if(!pass) host_gate(d1f,M,N);
        if (l==0){ hgdw(d1s,Xg_l[l],dW1r); }
        hgw(d1f,W1T,l,dA1);
        host_mask_apply(dA1,mXh[l],M,N);
        for (size_t i=0;i<MN;++i) G[i]+=dA1[i];      // G' = mX (.) dA1 + G
      }
      printf("VALIDATE bwd  vs %s-grad ref:  dX0 relPJ = %.4f   dW1T = %.4f   dW2T = %.4f\n",
             pass?"UNGATED":"GATED", relpj(G,hGf.data(),MN), relpj(dW1r,hW1.data(),MN), relpj(dW2r,hW2.data(),MN));
    }
  }

  // component microbenches (rates on mb[0] buffers)
  { MBSet& q=mb[0]; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto rate=[&](const char* name, auto&& fn, double gb){
      for(int i=0;i<5;i++) fn(); cudaStreamSynchronize(q.s);
      cudaEventRecord(a,q.s);
      for(int i=0;i<30;i++) fn();
      cudaEventRecord(b,q.s); cudaStreamSynchronize(q.s);
      float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=30;
      printf("  %-26s %.4f ms  (%.0f GB/s)\n", name, ms, gb/(ms/1e3));
    };
    double srcGB=(double)M*K*2/1e9, cGB=((double)compB+metaB+(double)M*K/32)/1e9;
    rate("k_fwd (gqc+tqT fused)",[&]{ k_fwd<false><<<gT,256,0,q.s>>>((const __nv_bfloat16*)q.Xres0.get(),K,
      q.A1c.comp.get(),q.A1c.meta.get(),q.A1c.sfa.get(),lSFA,q.mX[0].get(),
      q.A1T[0].vals.get(),q.A1T[0].sfb.get(),lSFB,M,K); }, srcGB+cGB+(double)maskB/1e9+(double)tvalB/1e9);
    rate("k_bwdG (resid+2xgqc)",[&]{ k_bwdG<<<gT,256,0,q.s>>>((const __nv_bfloat16*)q.dA1.get(),
      (const __nv_bfloat16*)q.G0.get(),q.mX[0].get(),(__nv_bfloat16*)q.G1v.get(),K,
      q.GcF[0].comp.get(),q.GcF[0].meta.get(),q.GcF[0].sfa.get(),
      q.GcS[0].comp.get(),q.GcS[0].meta.get(),q.GcS[0].sfa.get(),lSFA,M,K); }, 3*srcGB+2*cGB+(double)maskB/1e9);
    rate("k_bwdD (2xgqc masked)",[&]{ k_bwdD<<<gT,256,0,q.s>>>((const __nv_bfloat16*)q.dA2.get(),q.mD[0].get(),K,
      q.DcF[0].comp.get(),q.DcF[0].meta.get(),q.DcF[0].sfa.get(),
      q.DcS[0].comp.get(),q.DcS[0].meta.get(),q.DcS[0].sfa.get(),lSFA,M,K); }, srcGB+2*cGB+(double)maskB/1e9);
    rate("resid (STE+add)",[&]{ resid_k<<<grR,gcB,0,q.s>>>((const __nv_bfloat16*)q.dA1.get(),q.mX[0].get(),
      (const __nv_bfloat16*)q.G0.get(),(__nv_bfloat16*)q.G1v.get(),(size_t)M*N/8); }, 3*srcGB+(double)maskB/1e9);
  }

  // Timed full steps. Optional CUDA graphs remove the O(L*G) host-launch tax
  // exposed by fine-grained block filtering; all data/event dependencies remain identical.
#ifdef GRAPH
  std::vector<cudaGraph_t> graphs(MB); std::vector<cudaGraphExec_t> graph_execs(MB);
  for(int b=0;b<MB;++b){
    cudaStreamBeginCapture(mb[b].s,cudaStreamCaptureModeGlobal);
    step(mb[b]);
    CUDA_CHECK(cudaStreamEndCapture(mb[b].s,&graphs[b]));
    CUDA_CHECK(cudaGraphInstantiate(&graph_execs[b],graphs[b],nullptr,nullptr,0));
  }
  auto run_step=[&](int b){ CUDA_CHECK(cudaGraphLaunch(graph_execs[b],mb[b].s)); };
#else
  auto run_step=[&](int b){ step(mb[b]); };
#endif
  for(int i=0;i<3;i++) for (int b=0;b<MB;++b) run_step(b);
  cudaDeviceSynchronize();
  auto h0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++) for (int b=0;b<MB;++b) run_step(b);
  cudaError_t e=cudaDeviceSynchronize();
  auto h1=std::chrono::steady_clock::now();
  if(e){ printf("cuda err: %s\n",cudaGetErrorString(e)); return 1; }
  double ms_step=std::chrono::duration<double,std::milli>(h1-h0).count()/iters/MB;
  double flop=(double)L*6.0*2.0*(double)M*N*K, tf=flop/(ms_step/1e3)/1e12;
  double actual_tf=tf/G;
  printf("FULL-STEP fwd+bwd chained  S=%d L=%d MB=%d GROUPS=%d W-density=%.3f tile=%dx128x256  %.4f ms/step  %.4f ms/layer  %.1f TFLOPS dense-equiv / %.1f actual (6 block-filtered GEMMs/layer, incl gate/quant/compress/STE)\n",
         S,L,MB,G,1.0/G,TBM,ms_step,ms_step/L,tf,actual_tf);
  return 0;
}
