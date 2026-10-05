// Throughput-audited fp32/TF32 trainer: 8-layer residual double-GEMM, flipped identity.
// Same math/trajectory as train_flip_ref.cu; fixes measured throughput bugs (07-05 audit):
//   1. loss_grad did 1M atomicAdds into ONE float (1.24ms, 29% of step) -> block-reduced (~35us).
//      Also fixes fp32 atomic-noise in the reported loss.
//   2. grad-norm clip via cublasSnrm2 forced a device->host sync EVERY step -> device-side
//      sumsq + clip scale folded into adamw (no sync, no extra Sscal pass over G).
//   3. bwd dW GEMMs run on a second stream, overlapped with the dH/dX chain (1024^2 GEMMs are
//      only 64 CTAs on 70 SMs -> two concurrent GEMMs fill the machine).
//   4. l=0 dX GEMM skipped (gradient w.r.t. the INPUT, never consumed).
//   5. pj_err block-reduced (eval-only, untimed, but was 1.24ms/call).
// Build: nvcc -O3 -arch=sm_120a train_flip_opt.cu -lcublas -lcurand -o opt
// Run:   opt S M steps lr w2rnd gamma sdrop w2amp homo clip cos L fp4eval qat prec
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <chrono>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define CK(x) do{ auto e=(x); if(e){printf("ERR %s:%d %d\n",__FILE__,__LINE__,(int)e); exit(1);} }while(0)

__global__ void flip_target(const float* X0, float* Y, int S, int M, float t){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;   // Y = (1-t)*X0 + t*flip(X0)
  long n = (long)S*M;
  if(i<n){ int m = i/S, j = i%S; Y[i] = t*X0[(long)m*S + (S-1-j)] + (1.f-t)*X0[i]; }
}
// dY = 2(X8-Y)/n and loss += sum d^2/n, block-reduced (one atomic per block, not per element).
__global__ void loss_grad(const float* X8, const float* Y, float* dY, float* loss, long n, float inv_n,
                          void* dY16, int h16){  // optional fused 16-bit grad out (prec>=2)
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  float c = 0.f;
  if(i<n){ float d = X8[i]-Y[i]; float gv = 2.f*d*inv_n; dY[i] = gv; c = d*d*inv_n;
           if(dY16){ if(h16) ((__half*)dY16)[i]=__float2half_rn(gv); else ((__nv_bfloat16*)dY16)[i]=__float2bfloat16_rn(gv); } }
  __shared__ float sh[256];
  int t = threadIdx.x; sh[t] = c; __syncthreads();
  for(int o=128;o>0;o>>=1){ if(t<o) sh[t]+=sh[t+o]; __syncthreads(); }
  if(t==0) atomicAdd(loss, sh[0]);
}
// grid-stride sum of squares (device-side grad norm; kills the per-step snrm2 host sync)
__global__ void sumsq(const float* g, float* out, long n){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x*blockDim.x;
  float s = 0.f;
  for(long j=i;j<n;j+=stride){ float v=g[j]; s += v*v; }
  __shared__ float sh[256];
  int t = threadIdx.x; sh[t] = s; __syncthreads();
  for(int o=128;o>0;o>>=1){ if(t<o) sh[t]+=sh[t+o]; __syncthreads(); }
  if(t==0) atomicAdd(out, sh[0]);
}
__global__ void sumsq16(const __nv_bfloat16* g, float* out, long n){ // bf16-G variant, 2-wide loads
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x*blockDim.x;
  float s = 0.f;
  for(long j=i;j<n/2;j+=stride){ __nv_bfloat162 v2 = ((const __nv_bfloat162*)g)[j];
    float a=__bfloat162float(v2.x), b=__bfloat162float(v2.y); s += a*a+b*b; }
  __shared__ float sh[256];
  int t = threadIdx.x; sh[t] = s; __syncthreads();
  for(int o=128;o>0;o>>=1){ if(t<o) sh[t]+=sh[t+o]; __syncthreads(); }
  if(t==0) atomicAdd(out, sh[0]);
}
__global__ void adamw(float* w, const float* g, float* m, float* v, long n,
                      float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
                      long per_layer, float gamma, const float* gn2, float clip,
                      void* wb, int h16){  // wb: fused 16-bit weight mirror (prec>=2), free vs a separate cast pass
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  if(i<n){
    float sc = 1.f;                            // clip scale from device-side norm^2 (broadcast read)
    if(clip>0.f){ float g2 = *gn2; if(g2>clip*clip) sc = clip*rsqrtf(g2); }
    float lrl = lr * (gamma==1.f ? 1.f : powf(gamma, (float)(i/(2*per_layer))));
    float gi = g[i]*sc;
    float mi = m[i] = b1*m[i] + (1.f-b1)*gi;
    float vi = v[i] = b2*v[i] + (1.f-b2)*gi*gi;
    float mh = mi*bc1, vh = vi*bc2;
    float wn = w[i] - lrl*(mh/(sqrtf(vh)+eps) + wd*w[i]);
    w[i] = wn;
    if(wb){ if(h16) ((__half*)wb)[i] = __float2half_rn(wn); else ((__nv_bfloat16*)wb)[i] = __float2bfloat16_rn(wn); }
  }
}
// AdamW with compressed optimizer state, quant fused IN-KERNEL (no extra launches/passes):
//   mq: m stored bf16.  vq: v stored block-16 amax-scaled fp8 e4m3 (vq=8) or fp4 e2m1 (vq=4,
//   nibble-packed, unsigned since v>=0). Scale = blk amax/{448,6} in fp32 (0.25B/elem).
//   Traffic/elem fp32-state: 30B -> mq+vq8: 20.5B -> mq+vq4: 19.8B; V mem 64MB -> 12/9MB.
__global__ void adamw_q(float* w, const float* g, __nv_bfloat16* m, unsigned char* v, float* vsc, long n,
                        float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
                        const float* gn2, float clip, void* wb, int h16, int vq){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  if(i>=n) return;
  int lane = threadIdx.x & 31, g16 = lane & 15;          // 16-lane group == one scale block
  long blk = i >> 4;
  float sc = 1.f;
  if(clip>0.f){ float g2 = *gn2; if(g2>clip*clip) sc = clip*rsqrtf(g2); }
  float gi = g[i]*sc;
  // decode v
  float vs = vsc[blk], vi;
  if(vq==4){ unsigned char b = v[i>>1]; int nib = (i&1)? (b>>4):(b&15);
             const float grid[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
             vi = grid[nib&7]*vs; }                       // unsigned e2m1 (sign bit unused, v>=0)
  else     { vi = (float)(*(__nv_fp8_e4m3*)&v[i]) * vs; }
  float mi = __bfloat162float(m[i]);
  mi = b1*mi + (1.f-b1)*gi;
  vi = b2*vi + (1.f-b2)*gi*gi;
  m[i] = __float2bfloat16_rn(mi);
  float mh = mi*bc1, vh = vi*bc2;
  float wn = w[i] - lr*(mh/(sqrtf(vh)+eps) + wd*w[i]);
  w[i] = wn;
  if(wb){ if(h16) ((__half*)wb)[i] = __float2half_rn(wn); else ((__nv_bfloat16*)wb)[i] = __float2bfloat16_rn(wn); }
  // encode v: group amax via shuffle (lanes are block-aligned: TPB%32==0, n%32==0)
  float amax = vi;
  #pragma unroll
  for(int o=1;o<16;o<<=1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if(vq==4){
    float s2 = amax/6.f; float r = s2>0?1.f/s2:0.f;
    float x = vi*r; int nib=0; float bd=1e30f;
    const float grid[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    #pragma unroll
    for(int k=0;k<8;k++){ float d=fabsf(x-grid[k]); if(d<bd){bd=d;nib=k;} }
    int other = __shfl_xor_sync(0xffffffffu, nib, 1);
    if(!(i&1)) v[i>>1] = (unsigned char)((nib&15) | ((other&15)<<4));
    if(g16==0) vsc[blk] = s2;
  } else {
    float s2 = amax/448.f; float r = s2>0?1.f/s2:0.f;
    __nv_fp8_e4m3 q(vi*r); v[i] = *(unsigned char*)&q;
    if(g16==0) vsc[blk] = s2;
  }
}
// Adam-mini (arXiv:2406.16793, ported from gridworld multiply32_fp8 kernels.cuh:1037):
// ONE fp32 v per K-row (contiguous S-chunk of col-major W = all outputs fed by one input
// feature): v_row = b2*v_row + (1-b2)*mean(g_row^2). m per-element fp32 or bf16 (mq16).
// One 256-thread block per row (S=1024 -> 4 g's cached in registers between phases).
// Keeps OUR trajectory conventions: bias correction + GLOBAL-norm clip via gn2 (not per-row).
__global__ void adam_mini(float* w, const float* g, const __nv_bfloat16* g16, void* m, float* msc,
                          float* vrow, int rowlen,
                          float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
                          const float* gn2, float clip, void* wb, int h16, int mq16){
  int k = blockIdx.x; int tid = threadIdx.x;
  long base = (long)k*rowlen;
  float sc = 1.f;
  if(clip>0.f){ float g2 = *gn2; if(g2>clip*clip) sc = clip*rsqrtf(g2); }
  float gr[4]; float sumsq_r = 0.f;                      // rowlen <= 4*blockDim
  int nper = (rowlen + blockDim.x-1)/blockDim.x;
  for(int c=0;c<nper;c++){ int j = tid + c*blockDim.x;
    float gv = 0.f;
    if(j<rowlen) gv = (g16? __bfloat162float(g16[base+j]) : g[base+j])*sc;
    gr[c] = gv; sumsq_r += gv*gv; }
  __shared__ float sh[32];
  // block reduce
  float s = sumsq_r;
  for(int o=16;o>0;o>>=1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if((tid&31)==0) sh[tid>>5] = s; __syncthreads();
  if(tid<32){ float x = (tid < (blockDim.x>>5))? sh[tid]:0.f;
    for(int o=16;o>0;o>>=1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if(tid==0){ float vn = b2*vrow[k] + (1.f-b2)*(x/(float)rowlen); vrow[k]=vn; sh[0]=vn; } }
  __syncthreads();
  float vh = sh[0]*bc2; float inv = 1.f/(sqrtf(vh)+eps);
  for(int c=0;c<nper;c++){ int j = tid + c*blockDim.x; if(j>=rowlen) continue;
    long i = base+j;
    float mi;                                            // mq16: 0=fp32, 1=bf16, 2=fp4 e2m1 blk16
    if(mq16==2){ unsigned char b = ((unsigned char*)m)[i>>1]; int nib=(i&1)?(b>>4):(b&15);
      const float grid[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
      float mag = grid[nib&7]*msc[i>>4]; mi = (nib&8)? -mag: mag; }
    else if(mq16==1) mi = __bfloat162float(((__nv_bfloat16*)m)[i]);
    else mi = ((float*)m)[i];
    mi = b1*mi + (1.f-b1)*gr[c];
    if(mq16==2){
      float amag = fabsf(mi);
      float amax = amag;                                 // lanes cover consecutive j (16-block aligned)
      #pragma unroll
      for(int o=1;o<16;o<<=1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      float s2 = amax/6.f; float r = s2>0?1.f/s2:0.f;
      const float grid[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
      float x = amag*r; int idx=0; float bd=1e30f;
      #pragma unroll
      for(int t2=0;t2<8;t2++){ float d=fabsf(x-grid[t2]); if(d<bd){bd=d;idx=t2;} }
      int nib = idx | ((mi<0.f)?8:0);
      int other = __shfl_xor_sync(0xffffffffu, nib, 1);
      if(!(i&1)) ((unsigned char*)m)[i>>1] = (unsigned char)((nib&15)|((other&15)<<4));
      if((tid&15)==0) msc[i>>4] = s2;
      mi = ((nib&8)?-grid[idx]:grid[idx])*s2;            // use the DEQUANTIZED m (state==update view)
    }
    else if(mq16==1) ((__nv_bfloat16*)m)[i] = __float2bfloat16_rn(mi);
    else ((float*)m)[i] = mi;
    float wn = w[i] - lr*(mi*bc1*inv + wd*w[i]);
    w[i] = wn;
    if(wb){ if(h16) ((__half*)wb)[i] = __float2half_rn(wn); else ((__nv_bfloat16*)wb)[i] = __float2bfloat16_rn(wn); }
  }
}
// fp32 -> 16-bit cast (bf16 or fp16), float4/x4 vectorized (memory-bound; n multiple of 4 here)
__global__ void cast16(const float* __restrict__ in, void* __restrict__ outv, long n, int h16){
  long i = (blockIdx.x*(long)blockDim.x + threadIdx.x)*4;
  long stride = (long)gridDim.x*blockDim.x*4;
  for(long j=i;j<n;j+=stride){
    float4 f = *reinterpret_cast<const float4*>(in+j);
    if(h16){
      __half* out = (__half*)outv;
      __half2 a = {__float2half_rn(f.x), __float2half_rn(f.y)};
      __half2 b = {__float2half_rn(f.z), __float2half_rn(f.w)};
      *reinterpret_cast<__half2*>(out+j)   = a;
      *reinterpret_cast<__half2*>(out+j+2) = b;
    } else {
      __nv_bfloat16* out = (__nv_bfloat16*)outv;
      __nv_bfloat162 a = {__float2bfloat16_rn(f.x), __float2bfloat16_rn(f.y)};
      __nv_bfloat162 b = {__float2bfloat16_rn(f.z), __float2bfloat16_rn(f.w)};
      *reinterpret_cast<__nv_bfloat162*>(out+j)   = a;
      *reinterpret_cast<__nv_bfloat162*>(out+j+2) = b;
    }
  }
}
// NVFP4 round: e2m1 value grid {0,.5,1,1.5,2,3,4,6}, per-16-block ue4m3 scale.
__device__ float e2m1(float x){
  float a=fabsf(x), s=copysignf(1.f,x);
  const float g[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float best=0.f,bd=1e30f;
  #pragma unroll
  for(int i=0;i<8;i++){float d=fabsf(a-g[i]); if(d<bd){bd=d;best=g[i];}}
  return s*best;
}
__device__ float ue4m3(float x){ // 8 exp bits e4m3-ish positive scale, coarse
  if(x<=0) return 0.f;
  int e; float m=frexpf(x,&e);      // x=m*2^e, m in [.5,1)
  float q=roundf(m*8.f)/8.f;        // 3 mantissa bits
  return ldexpf(q,e);
}
// in-place NVFP4 quantize a matrix W[R x C] row-major, blocks of 16 along C.
__global__ void fp4_quant(float* W, int R, int C){
  int r=blockIdx.x; int kb=threadIdx.x; int nb=C/16;
  if(r>=R||kb>=nb) return;
  int k0=kb*16; float amax=0.f;
  for(int v=0;v<16;v++) amax=fmaxf(amax,fabsf(W[(long)r*C+k0+v]));
  float sf=ue4m3(amax/6.f); float rcp=sf>0?1.f/sf:0.f;
  for(int v=0;v<16;v++){ long i=(long)r*C+k0+v; W[i]=e2m1(W[i]*rcp)*sf; }
}
__global__ void set_identity(float* X, int S){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  long n = (long)S*S;
  if(i<n){ int m=i/S, j=i%S; X[i] = (j==m)?1.f:0.f; }
}
__global__ void pj_err(const float* P, float* err, int S){ // {||P-J||_F^2, tr(P), tr(JP)}, block-reduced
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  long n = (long)S*S;
  float e0=0.f, e1=0.f, e2=0.f;
  if(i<n){ int m=i/S, j=i%S; float t = (j==S-1-m)?1.f:0.f; float d=P[i]-t; e0=d*d;
           if(j==m) e1=P[i]; if(j==S-1-m) e2=P[i]; }
  __shared__ float sh[3][256];
  int t = threadIdx.x; sh[0][t]=e0; sh[1][t]=e1; sh[2][t]=e2; __syncthreads();
  for(int o=128;o>0;o>>=1){ if(t<o){ sh[0][t]+=sh[0][t+o]; sh[1][t]+=sh[1][t+o]; sh[2][t]+=sh[2][t+o]; } __syncthreads(); }
  if(t==0){ atomicAdd(err, sh[0][0]); atomicAdd(err+1, sh[1][0]); atomicAdd(err+2, sh[2][0]); }
}

// All GEMMs route here: fp32 buffers always; computeType picks the math path.
// prec=2 -> CUBLAS_COMPUTE_32F_FAST_16BF: cuBLAS converts fp32->bf16 inputs on the fly,
// accumulates fp32 (2x TF32 tensor peak on sm120, same API, no extra buffers).
static cublasComputeType_t g_ct = CUBLAS_COMPUTE_32F;
static inline cublasStatus_t gm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m, int n, int k, const float* al, const float* A, int lda,
    const float* B, int ldb, const float* be, float* C, int ldc){
  return cublasGemmEx(h, ta, tb, m, n, k, al, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                      be, C, CUDA_R_32F, ldc, g_ct, CUBLAS_GEMM_DEFAULT);
}
// prec=2/3 hot path: REAL 16-bit A/B (bf16 or fp16 mult), fp32 accumulate + fp32 C (32-bit out).
static cudaDataType g_t16 = CUDA_R_16BF;
static inline cublasStatus_t gm16(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m, int n, int k, const float* al, const void* A, int lda,
    const void* B, int ldb, const float* be, float* C, int ldc){
  return cublasGemmEx(h, ta, tb, m, n, k, al, A, g_t16, lda, B, g_t16, ldb,
                      be, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
// 16-bit C variant (still fp32 accumulate): kills the separate cast16 pass when the
// consumer only ever reads the 16-bit mirror anyway (grad chain, fwd intermediate H).
static inline cublasStatus_t gmC16(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m, int n, int k, const float* al, const void* A, int lda,
    const void* B, int ldb, const float* be, void* C, int ldc){
  return cublasGemmEx(h, ta, tb, m, n, k, al, A, g_t16, lda, B, g_t16, ldb,
                      be, C, g_t16, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

int main(int argc, char** argv){
  int S     = argc>1 ? atoi(argv[1]) : 1024;
  int M     = argc>2 ? atoi(argv[2]) : 1024;
  int steps = argc>3 ? atoi(argv[3]) : 3000;
  float lr  = argc>4 ? atof(argv[4]) : 1e-3f;
  int w2rnd = argc>5 ? atoi(argv[5]) : 0;   // 1: random W2 init (break layer symmetry)
  float gamma = argc>6 ? atof(argv[6]) : 1.f; // per-layer lr scale gamma^l
  float sdrop = argc>7 ? atof(argv[7]) : 0.f; // stochastic depth drop prob (annealed if <0: |sdrop|)
  float w2amp = argc>8 ? atof(argv[8]) : 1.f; // W2 init std amplifier on last layer
  float homo  = argc>9 ? atof(argv[9]) : 0.f; // >0: homotopy, target ramps I->J over homo*steps
  float clip  = argc>10? atof(argv[10]): 0.f; // >0: global grad-norm clip threshold
  int   cos   = argc>11? atoi(argv[11]): 0;   // 1: cosine decay lr->lr/20 after warmup
  int   L     = argc>12? atoi(argv[12]): 8;
  int   prec  = argc>15? atoi(argv[15]): 0;   // 0=fp32, 1=TF32, 2=bf16-in/fp32-acc, 3=fp16-in/fp32-acc
  int   vopt  = argc>16? atoi(argv[16]): 0;   // optimizer state: 0=fp32 m+v, 1=bf16m+fp8v(blk16),
                                              // 2=bf16m+fp4v(blk16), 3=mini(fp32 row-v, fp32 m),
                                              // 4=mini+bf16 m, 5=mini+fp4 m(blk16)
  int   gq    = argc>17? atoi(argv[17]): 0;   // 1: dW GEMMs write bf16 G direct (prec>=2 + mini only)
  int warm = 100;
  float b1=0.9f, b2=0.95f, eps=1e-8f, wd=0.0f;
  printf("S=%d M=%d L=%d steps=%d lr=%g homo=%g clip=%g cos=%d qat=%d prec=%d vopt=%d opt=1\n",
         S, M, L, steps, lr, homo, clip, cos, (argc>14?atoi(argv[14]):0), prec, vopt);

  cudaStream_t s1, s2;                        // s1 = main chain, s2 = bwd dW GEMMs (overlap)
  CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
  cudaEvent_t eDY, eDH, eDW2, eDW1, eGN, eAd[8];
  CK(cudaEventCreateWithFlags(&eDY,  cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&eDH,  cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&eDW2, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&eDW1, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&eGN,  cudaEventDisableTiming));
  for(int l=0;l<8;l++) CK(cudaEventCreateWithFlags(&eAd[l], cudaEventDisableTiming));
  cublasHandle_t hA, hB;                      // one handle per stream (no per-call stream churn)
  CK(cublasCreate(&hA)); CK(cublasSetStream(hA, s1));
  CK(cublasCreate(&hB)); CK(cublasSetStream(hB, s2));
  CK(cublasSetMathMode(hA, prec==1 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  CK(cublasSetMathMode(hB, prec==1 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  g_ct = prec==1 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F; // eval/recovery path stays fp32-ish
  curandGenerator_t gen; CK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CK(curandSetPseudoRandomGeneratorSeed(gen, 42));
  CK(curandSetStream(gen, s1));

  long ws = (long)S*S;                       // one weight matrix
  long xs = (long)S*M;                       // one activation
  float *W, *G, *Mm=0, *Vv=0;                // 2L matrices each, slab
  CK(cudaMalloc(&W, 2L*L*ws*4)); CK(cudaMalloc(&G, 2L*L*ws*4));
  long nW = 2L*L*ws;
  __nv_bfloat16 *Mq=0; unsigned char *Vq=0; float *Vs=0, *Vrow=0;  // compressed opt state
  if(vopt==0){ CK(cudaMalloc(&Mm,nW*4)); CK(cudaMalloc(&Vv,nW*4));
               CK(cudaMemset(Mm,0,nW*4)); CK(cudaMemset(Vv,0,nW*4)); }
  else if(vopt<=2){ CK(cudaMalloc(&Mq,nW*2)); CK(cudaMemset(Mq,0,nW*2));
               long vb = vopt==2 ? nW/2 : nW;
               CK(cudaMalloc(&Vq,vb)); CK(cudaMemset(Vq,0,vb));
               CK(cudaMalloc(&Vs,(nW/16)*4)); CK(cudaMemset(Vs,0,(nW/16)*4)); }
  else {       long nrow = nW/S;                                  // K-rows of length S
               CK(cudaMalloc(&Vrow,nrow*4)); CK(cudaMemset(Vrow,0,nrow*4));
               if(vopt==4)     { CK(cudaMalloc(&Mq,nW*2)); CK(cudaMemset(Mq,0,nW*2)); }
               else if(vopt==5){ CK(cudaMalloc(&Mq,nW/2)); CK(cudaMemset(Mq,0,nW/2));
                                 CK(cudaMalloc(&Vs,(nW/16)*4)); CK(cudaMemset(Vs,0,(nW/16)*4)); } // Vs reused as m-scales
               else            { CK(cudaMalloc(&Mm,nW*4)); CK(cudaMemset(Mm,0,nW*4)); } }
  __nv_bfloat16 *Gb=0;                                            // bf16 gradient slab (gq=1)
  if(gq){ if(prec<2 || vopt<3){ printf("gq needs prec>=2 and mini\n"); return 1; }
          CK(cudaMalloc(&Gb,nW*2)); }
  // init: W1_l ~ N(0, 1/S), W2_l = 0   (W1 at slab[2l], W2 at slab[2l+1])
  CK(cudaMemset(W,0,2L*L*ws*4));
  for(int l=0;l<L;l++){
    CK(curandGenerateNormal(gen, W + (2*l)*ws, ws, 0.f, 1.f/sqrtf((float)S)));
    // w2rnd: 0 zero-init all W2; 1 random all W2; 2 random ONLY layer 0 (break symmetry)
    if(w2rnd==1 || (w2rnd==2 && l==0)){
      float amp = (l==L-1 || w2rnd==2) ? w2amp : 1.f;
      CK(curandGenerateNormal(gen, W + (2*l+1)*ws, ws, 0.f, amp/sqrtf((float)S)));
    }
  }
  srand(1234);

  float *X[9], *Hh[8], *dY, *dH, *scratch, *lossd, *errd, *gn2d;
  for(int l=0;l<=L;l++) CK(cudaMalloc(&X[l], xs*4));
  for(int l=0;l<L;l++)  CK(cudaMalloc(&Hh[l], xs*4));
  CK(cudaMalloc(&dY, xs*4)); CK(cudaMalloc(&dH, xs*4));
  CK(cudaMalloc(&scratch, xs*4));
  CK(cudaMalloc(&lossd, 4)); CK(cudaMalloc(&errd, 12)); CK(cudaMalloc(&gn2d, 4));
  int fp4eval = argc>13? atoi(argv[13]) : 0;   // recovery pass simulates NVFP4 GEMM operands
  int qat     = argc>14? atoi(argv[14]) : 0;   // quantization-aware training (fp4 fwd + STE)
  float *Wq1=0,*Wq2=0,*Aq=0,*Hq=0;
  if(fp4eval || qat){ CK(cudaMalloc(&Wq1,ws*4)); CK(cudaMalloc(&Wq2,ws*4)); CK(cudaMalloc(&Aq,ws*4)); CK(cudaMalloc(&Hq,ws*4)); }
  float *Wqs=0, *Aqs=0;                          // QAT: per-layer quantized W (STE) + activations
  if(qat){ CK(cudaMalloc(&Wqs, 2L*L*ws*4)); CK(cudaMalloc(&Aqs, (long)L*xs*4)); fp4eval=1; }
  // prec=2: bf16 mirrors of every GEMM operand (weights kept fresh by fused adamw write)
  int h16 = (prec==3);                           // 2=bf16, 3=fp16 (both fp32-acc)
  if(h16) g_t16 = CUDA_R_16F;
  __nv_bfloat16 *Wb=0, *Xb[8]={0}, *Hb[8]={0}, *dYb=0, *dHb=0, *dYs=0;  // 2-byte elems either way
  if(prec>=2){
    if(qat){ printf("prec=2 + qat unsupported\n"); return 1; }
    CK(cudaMalloc(&Wb, 2L*L*ws*2));
    for(int l=0;l<L;l++){ CK(cudaMalloc(&Xb[l], xs*2)); CK(cudaMalloc(&Hb[l], xs*2)); }
    CK(cudaMalloc(&dYb, xs*2)); CK(cudaMalloc(&dHb, (long)L*xs*2));  // dHb: per-layer slab (decouple dW1<->next-dHb)
    if(gq) CK(cudaMalloc(&dYs, (long)L*xs*2));                       // per-layer dY snapshot (decouple dW2<->cast16)
  }
  int nb_c = 512;                                // cast16 grid (grid-stride, vec4)
  if(prec>=2) cast16<<<nb_c,256>>>(W, Wb, 2L*L*ws, h16); // seed bf16 weights (adamw keeps them fresh)
  int qgW=S, qtW=S/16, qgA=M, qtA=S/16;         // quant launch dims: W [SxS], act [MxS]col=row[MxS]
  float one=1.f, zero=0.f;
  int TPB=256; long nb_x=(xs+TPB-1)/TPB;
  int nb_ss = 2048;                             // sumsq grid (grid-stride over 2L*ws)

  double t_prev = now_ms(), train_ms = 0.0; bool timing = false;   // clean training-only wall (excludes eval/print/init)
  const float PI = 3.14159265f;
  for(int step=1; step<=steps; step++){
    float lrt;
    if(step<warm) lrt = lr*(float)step/warm;
    else if(cos){ float pr=(float)(step-warm)/(steps-warm); lrt = lr*(0.05f+0.95f*0.5f*(1.f+cosf(PI*pr))); }
    else lrt = lr;
    // fresh data
    CK(curandGenerateNormal(gen, X[0], xs, 0.f, 1.f));
    float tt = homo>0.f ? fminf(1.f, (float)step/(homo*steps)) : 1.f;
    flip_target<<<nb_x,TPB,0,s1>>>(X[0], scratch, S, M, tt);   // scratch = Y
    // fwd (stochastic depth: drop branch w.p. p, scale kept by 1/(1-p))
    float p = sdrop<0.f ? -sdrop*fmaxf(0.f, 1.f-2.f*(float)step/steps) : sdrop;
    int alive[8]; float keep = 1.f - p;
    for(int l=0;l<L;l++) alive[l] = (p==0.f) || ((float)rand()/RAND_MAX >= p);
    float kscale = (p==0.f) ? 1.f : 1.f/keep;
    for(int l=0;l<L;l++){
      const float* W1 = W+(2*l)*ws; const float* W2 = W+(2*l+1)*ws;
      CK(cudaStreamWaitEvent(s1, eAd[l], 0));   // layer-l weights fresh (prev step's adamw chunk, on s2)
      CK(cudaMemcpyAsync(X[l+1], X[l], xs*4, cudaMemcpyDeviceToDevice, s1));
      if(!alive[l]) continue;
      if(prec>=2){ // 16-bit multiply, fp32 accumulate; H lives ONLY as 16-bit (GEMM writes it direct)
        const __nv_bfloat16* Wb1 = Wb+(2*l)*ws; const __nv_bfloat16* Wb2 = Wb+(2*l+1)*ws;
        cast16<<<nb_c,TPB,0,s1>>>(X[l], Xb[l], xs, h16);
        CK(gmC16(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &one, Wb1, S, Xb[l], S, &zero, Hb[l], S));
        CK(gm16(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &kscale, Wb2, S, Hb[l], S, &one, X[l+1], S));
      } else if(!qat){
        CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &one, W1, S, X[l], S, &zero, Hh[l], S));
        CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &kscale, W2, S, Hh[l], S, &one, X[l+1], S));
      } else { // fp4 forward, master weights fp32 (STE). Store quantized W (Wqs) + act (Aqs) for bwd.
        float* Wq1s=Wqs+(2*l)*ws; float* Wq2s=Wqs+(2*l+1)*ws; float* Aql=Aqs+(long)l*xs;
        CK(cudaMemcpyAsync(Wq1s,W1,ws*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<qgW,qtW,0,s1>>>(Wq1s,S,S);
        CK(cudaMemcpyAsync(Wq2s,W2,ws*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<qgW,qtW,0,s1>>>(Wq2s,S,S);
        CK(cudaMemcpyAsync(Aql,X[l],xs*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<qgA,qtA,0,s1>>>(Aql,M,S);
        CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &one, Wq1s, S, Aql, S, &zero, Hh[l], S));
        fp4_quant<<<qgA,qtA,0,s1>>>(Hh[l],M,S);   // quantize intermediate (GEMM2 fp4 input); STE in bwd
        CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &kscale, Wq2s, S, Hh[l], S, &one, X[l+1], S));
      }
    }
    // loss + dY (block-reduced)
    CK(cudaMemsetAsync(lossd,0,4,s1));
    loss_grad<<<nb_x,TPB,0,s1>>>(X[L], scratch, dY, lossd, xs, 1.f/xs, prec>=2?(void*)dYb:0, h16);
    // bwd: dH/dX chain on s1, dW GEMMs overlapped on s2.
    //   dW2 = dY @ H^T ; dH = W2^T dY ; dW1 = dH @ X^T ; dY += W1^T dH
    //   Hazards: dW2 reads dY before s1 updates it in place; next layer's dH write waits dW1's read.
    for(int l=L-1;l>=0;l--){
      const float* W1 = W+(2*l)*ws; const float* W2 = W+(2*l+1)*ws;
      float* dW1 = G+(2*l)*ws; float* dW2 = G+(2*l+1)*ws;
      // QAT: use the fp4 operands from forward (STE), grads accumulate to fp32 master.
      const float* bW1 = qat ? Wqs+(2*l)*ws   : W1;
      const float* bW2 = qat ? Wqs+(2*l+1)*ws : W2;
      const float* bX  = qat ? Aqs+(long)l*xs : X[l];
      if(gq) CK(cudaMemcpyAsync(dYs + (long)l*xs, dYb, xs*2, cudaMemcpyDeviceToDevice, s1)); // snapshot dY(l) for dW2
      CK(cudaEventRecord(eDY, s1));                     // dY(l) final (+ dYs[l] copy done, gq)
      CK(cudaStreamWaitEvent(s2, eDY, 0));
      if(alive[l]){
        if(gq)          CK(gmC16(hB, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &kscale, dYs+(long)l*xs, S, Hb[l], S, &zero, Gb+(2*l+1)*ws, S));
        else if(prec>=2) CK(gm16(hB, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &kscale, dYb, S, Hb[l], S, &zero, dW2, S));
        else            CK(gm(hB, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &kscale, dY, S, Hh[l], S, &zero, dW2, S));
      } else CK(cudaMemsetAsync(gq?(void*)(Gb+(2*l+1)*ws):(void*)dW2,0,gq?ws*2:ws*4,s2));
      CK(cudaEventRecord(eDW2, s2));
      if(alive[l]){
        if(prec>=2){ // dH consumed only as 16-bit -> GEMM writes dHb direct (fp32 acc inside)
          CK(gmC16(hA, CUBLAS_OP_T, CUBLAS_OP_N, S, M, S, &kscale, Wb+(2*l+1)*ws, S, dYb, S, &zero, dHb+(long)l*xs, S));
        } else CK(gm(hA, CUBLAS_OP_T, CUBLAS_OP_N, S, M, S, &kscale, bW2, S, dY, S, &zero, dH, S));
      }
      CK(cudaEventRecord(eDH, s1));
      CK(cudaStreamWaitEvent(s2, eDH, 0));
      if(alive[l]){
        if(gq)          CK(gmC16(hB, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &one, dHb+(long)l*xs, S, Xb[l], S, &zero, Gb+(2*l)*ws, S));
        else if(prec>=2) CK(gm16(hB, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &one, dHb+(long)l*xs, S, Xb[l], S, &zero, dW1, S));
        else            CK(gm(hB, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &one, dH, S, bX, S, &zero, dW1, S));
      } else CK(cudaMemsetAsync(gq?(void*)(Gb+(2*l)*ws):(void*)dW1,0,gq?ws*2:ws*4,s2));
      CK(cudaEventRecord(eDW1, s2));
      if(!gq) CK(cudaStreamWaitEvent(s1, eDW2, 0));     // (gq: dYs snapshot removes this WAR hazard)
      if(l>0 && alive[l]){                              // l=0: dX is grad w.r.t. the INPUT, unused -> skip
        if(prec>=2){
          CK(gm16(hA, CUBLAS_OP_T, CUBLAS_OP_N, S, M, S, &one, Wb+(2*l)*ws, S, dHb+(long)l*xs, S, &one, dY, S));
        } else CK(gm(hA, CUBLAS_OP_T, CUBLAS_OP_N, S, M, S, &one, bW1, S, dH, S, &one, dY, S));
      }
      if(prec>=2 && l>0) cast16<<<nb_c,TPB,0,s1>>>(dY, dYb, xs, h16);  // refresh bf16 dY for next layer
      if(!gq) CK(cudaStreamWaitEvent(s1, eDW1, 0));     // (gq: per-layer dHb slab removes this WAR hazard)
    }
    if(gq) CK(cudaStreamWaitEvent(s1, eDW1, 0));        // gq: sumsq/adamw must wait ALL dW GEMMs (last=dW1 l=0 on s2)
    // global grad-norm clip, device-side: sumsq -> scale folded into adamw (no host sync)
    if(clip>0.f){
      CK(cudaMemsetAsync(gn2d,0,4,s1));
      if(gq) sumsq16<<<nb_ss,TPB,0,s1>>>(Gb, gn2d, 2L*L*ws);
      else   sumsq<<<nb_ss,TPB,0,s1>>>(G, gn2d, 2L*L*ws);
    }
    CK(cudaEventRecord(eGN, s1));
    // adamw: per-layer chunks on s2, overlapped with NEXT step's forward (fwd layer l waits eAd[l]).
    // Same math as one launch: gn2d is complete before any chunk (eGN), chunks are disjoint slices.
    float bc1 = 1.f/(1.f-powf(b1,step)), bc2 = 1.f/(1.f-powf(b2,step));
    CK(cudaStreamWaitEvent(s2, eGN, 0));
    for(int l=0;l<L;l++){
      float lrl = lrt * (gamma==1.f ? 1.f : powf(gamma,(float)l));
      long off = (2*l)*ws;
      void* wbl = Wb ? (void*)(Wb+off) : (void*)0;
      if(vopt==0)
        adamw<<<(2*ws+TPB-1)/TPB,TPB,0,s2>>>(W+off, G+off, Mm+off, Vv+off, 2*ws,
                                             lrl, b1, b2, eps, wd, bc1, bc2, ws, 1.f, gn2d, clip, wbl, h16);
      else if(vopt<=2)
        adamw_q<<<(2*ws+TPB-1)/TPB,TPB,0,s2>>>(W+off, G+off, Mq+off, Vq+(vopt==2?off/2:off), Vs+off/16, 2*ws,
                                               lrl, b1, b2, eps, wd, bc1, bc2, gn2d, clip, wbl, h16, vopt==2?4:8);
      else{
        void* mp = vopt==3 ? (void*)(Mm+off) : vopt==4 ? (void*)(Mq+off) : (void*)(((unsigned char*)Mq)+off/2);
        adam_mini<<<2*ws/S,TPB,0,s2>>>(W+off, G+off, gq?Gb+off:0, mp, vopt==5?Vs+off/16:0, Vrow+off/S, S,
                                       lrl, b1, b2, eps, wd, bc1, bc2, gn2d, clip, wbl, h16,
                                       vopt==3?0:vopt==4?1:2);
      }
      CK(cudaEventRecord(eAd[l], s2));
    }

    if(step%100==0 || step==1){
      cudaDeviceSynchronize(); double t_end = now_ms();     // drain training work, mark interval end
      if(timing) train_ms += t_end - t_prev;                 // accumulate clean training-only wall
      float lh; CK(cudaMemcpy(&lh, lossd, 4, cudaMemcpyDeviceToHost));
      float pj = -1.f, ab[2] = {0,0};
      if(S<=2048){ // P recovery every log point: feed I, recover P (relPJ vs true flip J, untimed)
        // only valid when M==S; guard
        if(M==S){
          set_identity<<<(ws+TPB-1)/TPB,TPB,0,s1>>>(X[0], S);
          for(int l=0;l<L;l++){
            const float* W1 = W+(2*l)*ws; const float* W2 = W+(2*l+1)*ws;
            CK(cudaMemcpyAsync(X[l+1], X[l], ws*4, cudaMemcpyDeviceToDevice, s1));
            if(!fp4eval){ // exact fp32 chain
              CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, W1, S, X[l], S, &zero, Hh[l], S));
              CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, W2, S, Hh[l], S, &one, X[l+1], S));
            } else { // NVFP4-simulated operands; residual stream stays fp32
              CK(cudaMemcpyAsync(Wq1,W1,ws*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<S,S/16,0,s1>>>(Wq1,S,S);
              CK(cudaMemcpyAsync(Wq2,W2,ws*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<S,S/16,0,s1>>>(Wq2,S,S);
              CK(cudaMemcpyAsync(Aq,X[l],ws*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<S,S/16,0,s1>>>(Aq,S,S);
              CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, Wq1, S, Aq, S, &zero, Hh[l], S));
              CK(cudaMemcpyAsync(Hq,Hh[l],ws*4,cudaMemcpyDeviceToDevice,s1)); fp4_quant<<<S,S/16,0,s1>>>(Hq,S,S);
              CK(gm(hA, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, Wq2, S, Hq, S, &one, X[l+1], S));
            }
          }
          CK(cudaMemsetAsync(errd,0,12,s1));
          pj_err<<<(ws+TPB-1)/TPB,TPB,0,s1>>>(X[L], errd, S);
          float eh[3]; CK(cudaMemcpy(eh, errd, 12, cudaMemcpyDeviceToHost));
          pj = sqrtf(eh[0])/sqrtf((float)S);
          ab[0] = eh[1]/S; ab[1] = eh[2]/S;  // P ~ a*I + b*J coefficients
        }
      }
      printf("step %6d  loss %.6e  relPJ %s%.4f  a %.3f b %.3f  train_ms %.1f  ms/step %.3f\n", step, lh,
             pj<0?"(skip) ":"", pj<0?0.f:pj, ab[0], ab[1], train_ms, timing?(float)(train_ms/(step-1)):0.f);
      fflush(stdout);
      timing = true;                                          // steady-state clock (skip step-1 CUDA/cuBLAS init)
      cudaDeviceSynchronize(); t_prev = now_ms();              // resume AFTER eval+print (both excluded from train_ms)
    }
  }
  cudaError_t e = cudaDeviceSynchronize();
  printf("done err=%d\n", (int)e);
  return 0;
}
