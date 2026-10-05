// mma_roof.cu — pure-issue roofline for sm120 block-scaled fp4 MMA (sparse+dense).
// Measures the TRUE tensor instruction-rate ceiling of this silicon at locked
// clocks: all operands live in registers, zero memory traffic, NACC independent
// accumulator chains per warp so throughput (not latency) binds.
//
//   sparse instr = mma.sync.kind::mxf4nvf4.sp::ordered_metadata.block_scale
//                  .scale_vec::4X.m16n8k128  -> 2*16*8*128 = 32768 FLOPs-de
//   dense  instr = mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//                  -> 2*16*8*64 = 16384 FLOPs-de
//
// Build (needs 13.3 ptxas splice):
//   ./build_sparse.sh mma_roof.cu /tmp/mma_roof
// Run: /tmp/mma_roof [ctas=70] [warps=8] [nacc=4] [iters=2000000] [reps=5]
// Prints TFLOPS-de and FLOPs/clk/SM (clock64-based, clock-lock independent).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} }while(0)

template<int NACC, bool SPARSE>
__global__ void roof(float* out, long long* cyc, int iters)
{
  // benign constants: fp4 zeros, canonical 2:4 metadata (keep 0,1), ue4m3 scale=1.0
  const uint32_t a0=0u,a1=0u,a2=0u,a3=0u;
  const uint32_t b0=0u,b1=0u,b2=0u,b3=0u;
  const uint32_t e =0x44444444u;
  const uint32_t sfa=0x38383838u, sfb=0x38383838u;

  float d[NACC][4];
  #pragma unroll
  for(int j=0;j<NACC;j++){ d[j][0]=0.f; d[j][1]=0.f; d[j][2]=0.f; d[j][3]=0.f; }

  long long t0 = clock64();
  for(int i=0;i<iters;i++){
    #pragma unroll
    for(int j=0;j<NACC;j++){
      if (SPARSE) {
        asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},"
          "{%16}, 0x0,"
          "{%17},{%18,%19},{%20},{%21,%22};\n"
          : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
            "r"(b0),"r"(b1),"r"(b2),"r"(b3),
            "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),
            "r"(e),
            "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
            "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
      } else {
        asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
          "{%14},{%15,%16},{%17},{%18,%19};\n"
          : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
            "r"(b0),"r"(b1),
            "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),
            "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
            "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
      }
    }
  }
  long long t1 = clock64();

  float s=0.f;
  #pragma unroll
  for(int j=0;j<NACC;j++) s += d[j][0]+d[j][1]+d[j][2]+d[j][3];
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  out[tid] = s;
  if (threadIdx.x==0) cyc[blockIdx.x] = t1-t0;
}

template<bool SPARSE>
static void run_one(int ctas,int warps,int nacc,int iters,int reps)
{
  int threads = warps*32;
  float* out; long long* cyc;
  CK(cudaMalloc(&out, sizeof(float)*ctas*threads));
  CK(cudaMalloc(&cyc, sizeof(long long)*ctas));
  long long* hcyc = (long long*)malloc(sizeof(long long)*ctas);

  auto launch=[&](int it){
    switch(nacc){
      case 1: roof<1,SPARSE><<<ctas,threads>>>(out,cyc,it); break;
      case 2: roof<2,SPARSE><<<ctas,threads>>>(out,cyc,it); break;
      case 4: roof<4,SPARSE><<<ctas,threads>>>(out,cyc,it); break;
      case 6: roof<6,SPARSE><<<ctas,threads>>>(out,cyc,it); break;
      case 8: roof<8,SPARSE><<<ctas,threads>>>(out,cyc,it); break;
      case 12: roof<12,SPARSE><<<ctas,threads>>>(out,cyc,it); break;
      default: printf("nacc %d unsupported\n",nacc); exit(1);
    }
  };
  launch(10000); CK(cudaDeviceSynchronize()); // warm

  cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
  float best_ms=1e30f; long long best_cyc=0;
  for(int r=0;r<reps;r++){
    CK(cudaEventRecord(ev0));
    launch(iters);
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    float ms; CK(cudaEventElapsedTime(&ms,ev0,ev1));
    CK(cudaMemcpy(hcyc,cyc,sizeof(long long)*ctas,cudaMemcpyDeviceToHost));
    long long mx=0; for(int i=0;i<ctas;i++) if(hcyc[i]>mx) mx=hcyc[i];
    if(ms<best_ms){ best_ms=ms; best_cyc=mx; }
  }
  double instr_per_warp = (double)iters*nacc;
  double flops_de = (SPARSE?32768.0:16384.0)*instr_per_warp*warps*ctas;
  double tflops = flops_de/(best_ms*1e-3)/1e12;
  // per-SM normalize: assume ctas>=SM count and 1 CTA/SM when ctas==SMs
  double fpc_sm = (SPARSE?32768.0:16384.0)*instr_per_warp*warps/(double)best_cyc;
  printf("%s ctas=%d warps=%d nacc=%d iters=%d : %.3f ms  %.1f TFLOPS-de  %.1f FLOPs-de/clk/SM(cta)\n",
         SPARSE?"SPARSE k128":"DENSE  k64 ",ctas,warps,nacc,iters,best_ms,tflops,fpc_sm);
  fflush(stdout);
  cudaFree(out); cudaFree(cyc); free(hcyc);
}

int main(int argc,char**argv)
{
  int ctas  = argc>1?atoi(argv[1]):70;
  int warps = argc>2?atoi(argv[2]):8;
  int nacc  = argc>3?atoi(argv[3]):4;
  int iters = argc>4?atoi(argv[4]):2000000;
  int reps  = argc>5?atoi(argv[5]):5;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device %s SMs=%d\n",p.name,p.multiProcessorCount);
  run_one<true >(ctas,warps,nacc,iters,reps);
  run_one<false>(ctas,warps,nacc,iters,reps);
  return 0;
}
