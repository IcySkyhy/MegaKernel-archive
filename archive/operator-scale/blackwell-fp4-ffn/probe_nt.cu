// Probe: bf16 nt GEMM (dW shape: C[S,S] = A[S,M] * B[S,M]^T) algo choices.
// GemmEx heuristic picked 64x64_32x6 (44 TFLOPS). Try explicit algos + cublasLt.
#include <cstdio>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define CK(x) do{ auto e=(x); if(e){printf("ERR %d @%d\n",(int)e,__LINE__);} }while(0)

int main(){
  int S=1024, M=1024;
  __nv_bfloat16 *A,*B; float *C;
  cudaMalloc(&A,(long)S*M*2); cudaMalloc(&B,(long)S*M*2); cudaMalloc(&C,(long)S*S*4);
  cudaMemset(A,0,(long)S*M*2); cudaMemset(B,0,(long)S*M*2);
  float one=1.f, zero=0.f;
  cublasHandle_t h; cublasCreate(&h);
  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  double fl = 2.0*S*S*M;

  // GemmEx algo sweep
  for(int algo=-1; algo<=24; algo++){
    cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, S,S,M, &one,
        A, CUDA_R_16BF, S, B, CUDA_R_16BF, S, &zero, C, CUDA_R_32F, S,
        CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t)algo);
    if(st) continue;
    cudaEventRecord(t0);
    for(int i=0;i<50;i++) cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, S,S,M, &one,
        A, CUDA_R_16BF, S, B, CUDA_R_16BF, S, &zero, C, CUDA_R_32F, S,
        CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t)algo);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    printf("GemmEx algo %2d : %7.1f us  %6.1f TFLOPS\n", algo, ms*1000/50, fl/(ms/50)/1e9);
  }

  // cublasLt heuristic top-8
  cublasLtHandle_t lt; cublasLtCreate(&lt);
  cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t tA=CUBLAS_OP_N, tB=CUBLAS_OP_T;
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
  cublasLtMatrixLayout_t la,lb,lc;
  cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, S, M, S);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, S, M, S);   // B given pre-op-T: S x M, ld S
  cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, S, S, S);
  cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
  size_t wsz = 32<<20; void* wbuf; cudaMalloc(&wbuf, wsz);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
  cublasLtMatmulHeuristicResult_t res[8]; int nres=0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &nres);
  printf("Lt heuristics: %d\n", nres);
  for(int i=0;i<nres;i++){
    cudaEventRecord(t0);
    for(int k=0;k<50;k++)
      cublasLtMatmul(lt, op, &one, A, la, B, lb, &zero, C, lc, C, lc, &res[i].algo, wbuf, wsz, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    printf("Lt algo #%d : %7.1f us  %6.1f TFLOPS  (ws %zu)\n", i, ms*1000/50, fl/(ms/50)/1e9, res[i].workspaceSize);
  }
  return 0;
}
