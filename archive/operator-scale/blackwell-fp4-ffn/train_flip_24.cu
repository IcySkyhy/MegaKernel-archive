// fp32 reference: 8-layer residual double-GEMM trained on flipped identity.
// X_{l+1} = X_l + W2_l @ (W1_l @ X_l), target Y[j,m] = X0[S-1-j, m].
// Column-major everywhere (cuBLAS native): X [S features x M batch].
// Purpose: validate task + init + AdamW hyperparams before the fp4 pipeline.
// Build: nvcc -O3 -arch=sm_120a train_flip_ref.cu -lcublas -lcurand -o ref
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <chrono>
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define CK(x) do{ auto e=(x); if(e){printf("ERR %s:%d %d\n",__FILE__,__LINE__,(int)e); exit(1);} }while(0)

__global__ void flip_target(const float* X0, float* Y, int S, int M, float t){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;   // Y = (1-t)*X0 + t*flip(X0)
  long n = (long)S*M;
  if(i<n){ int m = i/S, j = i%S; Y[i] = t*X0[(long)m*S + (S-1-j)] + (1.f-t)*X0[i]; }
}
__global__ void loss_grad(const float* X8, const float* Y, float* dY, float* loss, long n, float inv_n){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  if(i<n){ float d = X8[i]-Y[i]; dY[i] = 2.f*d*inv_n; atomicAdd(loss, d*d*inv_n); }
}
__global__ void adamw(float* w, const float* g, float* m, float* v, long n,
                      float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
                      long per_layer, float gamma){ // lr scale gamma^l, l = i/per_layer/2
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  if(i<n){
    float lrl = lr * (gamma==1.f ? 1.f : powf(gamma, (float)(i/(2*per_layer))));
    float gi = g[i];
    float mi = m[i] = b1*m[i] + (1.f-b1)*gi;
    float vi = v[i] = b2*v[i] + (1.f-b2)*gi*gi;
    float mh = mi*bc1, vh = vi*bc2;
    w[i] -= lrl*(mh/(sqrtf(vh)+eps) + wd*w[i]);
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
// 2:4-prune W in place (magnitude), along the CONTRACTION = in-features.
// W is column-major [So out-feat (stride 1) x Si in-feat (stride So)] (cuBLAS native).
// blk: mask granularity along out-features. blk=1 = per-row 2:4 (NVIDIA standard);
// blk=B = a block of B consecutive out-features shares one 2:4 in-feature pattern
// (still a valid per-row mask for the sparse tensor core, but it lets the dW GEMM
// skip half its N-columns per M-tile => the structured-sparse dW 2x). Saliency =
// sum |W| over the block. grid=(Si/4, So/blk), one thread per (in-group, out-block).
__global__ void prune24(float* W, int So, int Si, int blk){
  long g = blockIdx.x, b = blockIdx.y;
  int j0 = g*4, i0 = b*blk;
  if(j0+3>=Si) return;
  float mag[4]={0,0,0,0};
  for(int r=0;r<blk && i0+r<So;r++){ long i=i0+r; for(int t=0;t<4;t++) mag[t]+=fabsf(W[i+(long)(j0+t)*So]); }
  int s0=0; for(int t=1;t<4;t++) if(mag[t]<mag[s0]) s0=t;
  int s1=(s0==0)?1:0; for(int t=0;t<4;t++) if(t!=s0 && mag[t]<mag[s1]) s1=t;
  for(int r=0;r<blk && i0+r<So;r++){ long i=i0+r; W[i+(long)(j0+s0)*So]=0.f; W[i+(long)(j0+s1)*So]=0.f; }
}
// Transposable 2:4 (Hubara et al.): per 4x4 block (4 out x 4 in), keep the
// row-sum-2 AND col-sum-2 pattern maximizing retained |W| (enumerate all 90,
// via the 6^4 row-choice combos filtered to col-sum 2). Guarantees 2:4 along BOTH
// axes => W AND W^T are 2:4 => the dActivation GEMMs (W^T @ g) stay sparse.
__global__ void prune24_trans(float* W, int So, int Si){
  long bi = blockIdx.y*4, bj = blockIdx.x*4;
  if(bj+3>=Si || bi+3>=So) return;
  float a[4][4];
  for(int r=0;r<4;r++) for(int c=0;c<4;c++) a[r][c]=fabsf(W[(bi+r)+(long)(bj+c)*So]);
  const int ch[6]={0x3,0x5,0x9,0x6,0xA,0xC};   // the six 2-of-4 keep-masks (bit c = keep col c)
  float best=-1; int bm[4]={0x3,0xC,0x3,0xC};
  for(int r0=0;r0<6;r0++)for(int r1=0;r1<6;r1++)for(int r2=0;r2<6;r2++)for(int r3=0;r3<6;r3++){
    int m[4]={ch[r0],ch[r1],ch[r2],ch[r3]}; int ok=1;
    for(int c=0;c<4;c++){ int cs=((m[0]>>c)&1)+((m[1]>>c)&1)+((m[2]>>c)&1)+((m[3]>>c)&1); if(cs!=2){ok=0;break;} }
    if(!ok) continue;
    float sum=0; for(int r=0;r<4;r++) for(int c=0;c<4;c++) if((m[r]>>c)&1) sum+=a[r][c];
    if(sum>best){ best=sum; for(int r=0;r<4;r++) bm[r]=m[r]; }
  }
  for(int r=0;r<4;r++) for(int c=0;c<4;c++) if(!((bm[r]>>c)&1)) W[(bi+r)+(long)(bj+c)*So]=0.f;
}
__global__ void set_identity(float* X, int S){
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  long n = (long)S*S;
  if(i<n){ int m=i/S, j=i%S; X[i] = (j==m)?1.f:0.f; }
}
__global__ void pj_err(const float* P, float* err, int S){ // {||P-J||_F^2, tr(P), tr(JP)}
  long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
  long n = (long)S*S;
  if(i<n){ int m=i/S, j=i%S; float t = (j==S-1-m)?1.f:0.f; float d=P[i]-t; atomicAdd(err, d*d);
           if(j==m) atomicAdd(err+1, P[i]);
           if(j==S-1-m) atomicAdd(err+2, P[i]); }
}

int main(int argc, char** argv){
  int S     = argc>1 ? atoi(argv[1]) : 1024;
  int M     = argc>2 ? atoi(argv[2]) : 1024;
  int steps = argc>3 ? atoi(argv[3]) : 3000;
  float lr  = argc>4 ? atof(argv[4]) : 1e-3f;
  int w2rnd = argc>5 ? atoi(argv[5]) : 0;   // 1: random W2 init (break layer symmetry)
  float gamma = argc>6 ? atof(argv[6]) : 1.f; // per-layer lr scale gamma^l
  float sdrop = argc>7 ? atof(argv[7]) : 0.f; // stochastic depth drop prob (annealed to 0 by mid-training if <0: |sdrop|)
  float w2amp = argc>8 ? atof(argv[8]) : 1.f; // W2 init std amplifier on last layer
  float homo  = argc>9 ? atof(argv[9]) : 0.f; // >0: homotopy, target ramps I->J over homo*steps
  float clip  = argc>10? atof(argv[10]): 0.f; // >0: global grad-norm clip threshold
  int   cos   = argc>11? atoi(argv[11]): 0;   // 1: cosine decay lr->lr/20 after warmup
  int   L     = argc>12? atoi(argv[12]): 8;
  int   prec  = argc>15? atoi(argv[15]): 0;   // 0=fp32 IEEE (CUDA cores), 1=fp32 TF32 (tensor cores)
  int   blk   = argc>16? atoi(argv[16]): 0;   // 2:4 sparsity block: 0=dense fp4, 1=per-row 2:4, B=block-B 2:4
  int warm = 100;
  float b1=0.9f, b2=0.95f, eps=1e-8f, wd=0.0f;
  printf("S=%d M=%d L=%d steps=%d lr=%g homo=%g clip=%g cos=%d qat=%d prec=%d blk=%d\n",
         S, M, L, steps, lr, homo, clip, cos, (argc>14?atoi(argv[14]):0), prec, blk);

  cublasHandle_t h; CK(cublasCreate(&h));
  CK(cublasSetMathMode(h, prec==1 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  curandGenerator_t gen; CK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CK(curandSetPseudoRandomGeneratorSeed(gen, 42));

  long ws = (long)S*S;                       // one weight matrix
  long xs = (long)S*M;                       // one activation
  float *W, *G, *Mm, *Vv;                    // 2L matrices each, slab
  CK(cudaMalloc(&W, 2L*L*ws*4)); CK(cudaMalloc(&G, 2L*L*ws*4));
  CK(cudaMalloc(&Mm,2L*L*ws*4)); CK(cudaMalloc(&Vv,2L*L*ws*4));
  CK(cudaMemset(Mm,0,2L*L*ws*4)); CK(cudaMemset(Vv,0,2L*L*ws*4));
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

  float *X[9], *Hh[8], *dY, *dH, *scratch, *lossd, *errd;
  for(int l=0;l<=L;l++) CK(cudaMalloc(&X[l], xs*4));
  for(int l=0;l<L;l++)  CK(cudaMalloc(&Hh[l], xs*4));
  CK(cudaMalloc(&dY, xs*4)); CK(cudaMalloc(&dH, xs*4));
  CK(cudaMalloc(&scratch, xs*4));
  CK(cudaMalloc(&lossd, 4)); CK(cudaMalloc(&errd, 12));
  int fp4eval = argc>13? atoi(argv[13]) : 0;   // recovery pass simulates NVFP4 GEMM operands
  int qat     = argc>14? atoi(argv[14]) : 0;   // quantization-aware training (fp4 fwd + STE)
  float *Wq1=0,*Wq2=0,*Aq=0,*Hq=0;
  if(fp4eval || qat){ CK(cudaMalloc(&Wq1,ws*4)); CK(cudaMalloc(&Wq2,ws*4)); CK(cudaMalloc(&Aq,ws*4)); CK(cudaMalloc(&Hq,ws*4)); }
  float *Wqs=0, *Aqs=0;                          // QAT: per-layer quantized W (STE) + activations
  if(qat){ CK(cudaMalloc(&Wqs, 2L*L*ws*4)); CK(cudaMalloc(&Aqs, (long)L*xs*4)); fp4eval=1; }
  int qgW=S, qtW=S/16, qgA=M, qtA=S/16;         // quant launch dims: W [SxS], act [MxS]col=row[MxS]
  float one=1.f, zero=0.f;
  int TPB=256; long nb_x=(xs+TPB-1)/TPB, nb_w=(ws+TPB-1)/TPB;

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
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
    flip_target<<<nb_x,TPB>>>(X[0], scratch, S, M, tt);   // scratch = Y
    // fwd (stochastic depth: drop branch w.p. p, scale kept by 1/(1-p))
    float p = sdrop<0.f ? -sdrop*fmaxf(0.f, 1.f-2.f*(float)step/steps) : sdrop;
    int alive[8]; float keep = 1.f - p;
    for(int l=0;l<L;l++) alive[l] = (p==0.f) || ((float)rand()/RAND_MAX >= p);
    float kscale = (p==0.f) ? 1.f : 1.f/keep;
    for(int l=0;l<L;l++){
      const float* W1 = W+(2*l)*ws; const float* W2 = W+(2*l+1)*ws;
      CK(cudaMemcpyAsync(X[l+1], X[l], xs*4, cudaMemcpyDeviceToDevice));
      if(!alive[l]) continue;
      if(!qat){
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &one, W1, S, X[l], S, &zero, Hh[l], S));
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &kscale, W2, S, Hh[l], S, &one, X[l+1], S));
      } else { // fp4 forward, master weights fp32 (STE). Store quantized W (Wqs) + act (Aqs) for bwd.
        float* Wq1s=Wqs+(2*l)*ws; float* Wq2s=Wqs+(2*l+1)*ws; float* Aql=Aqs+(long)l*xs;
        CK(cudaMemcpyAsync(Wq1s,W1,ws*4,cudaMemcpyDeviceToDevice)); if(blk<0) prune24_trans<<<dim3(S/4,S/4),1>>>(Wq1s,S,S); else if(blk>0) prune24<<<dim3(S/4,S/blk),1>>>(Wq1s,S,S,blk); fp4_quant<<<qgW,qtW>>>(Wq1s,S,S);
        CK(cudaMemcpyAsync(Wq2s,W2,ws*4,cudaMemcpyDeviceToDevice)); if(blk<0) prune24_trans<<<dim3(S/4,S/4),1>>>(Wq2s,S,S); else if(blk>0) prune24<<<dim3(S/4,S/blk),1>>>(Wq2s,S,S,blk); fp4_quant<<<qgW,qtW>>>(Wq2s,S,S);
        CK(cudaMemcpyAsync(Aql,X[l],xs*4,cudaMemcpyDeviceToDevice)); fp4_quant<<<qgA,qtA>>>(Aql,M,S);
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &one, Wq1s, S, Aql, S, &zero, Hh[l], S));
        fp4_quant<<<qgA,qtA>>>(Hh[l],M,S);   // quantize intermediate (GEMM2 fp4 input); STE in bwd
        CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, M, S, &kscale, Wq2s, S, Hh[l], S, &one, X[l+1], S));
      }
    }
    // loss + dY
    CK(cudaMemset(lossd,0,4));
    loss_grad<<<nb_x,TPB>>>(X[L], scratch, dY, lossd, xs, 1.f/xs);
    // bwd
    for(int l=L-1;l>=0;l--){
      const float* W1 = W+(2*l)*ws; const float* W2 = W+(2*l+1)*ws;
      float* dW1 = G+(2*l)*ws; float* dW2 = G+(2*l+1)*ws;
      if(!alive[l]){ CK(cudaMemsetAsync(dW1,0,ws*4)); CK(cudaMemsetAsync(dW2,0,ws*4)); continue; }
      // dW2 = dY @ H^T ; dH = W2^T dY ; dW1 = dH @ X^T ; dY += W1^T dH
      // QAT: use the fp4 operands from forward (STE), grads accumulate to fp32 master.
      const float* bW1 = qat ? Wqs+(2*l)*ws   : W1;
      const float* bW2 = qat ? Wqs+(2*l+1)*ws : W2;
      const float* bX  = qat ? Aqs+(long)l*xs : X[l];
      CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &kscale, dY, S, Hh[l], S, &zero, dW2, S));
      CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, S, M, S, &kscale, bW2, S, dY, S, &zero, dH, S));
      CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, S, S, M, &one, dH, S, bX, S, &zero, dW1, S));
      CK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, S, M, S, &one, bW1, S, dH, S, &one, dY, S));
    }
    // global grad-norm clip (scale G in place before Adam moments)
    if(clip>0.f){
      float gn; CK(cublasSnrm2(h, (int)(2*L*ws), G, 1, &gn));
      if(gn>clip){ float sc=clip/gn; CK(cublasSscal(h, (int)(2*L*ws), &sc, G, 1)); }
    }
    // adamw
    float bc1 = 1.f/(1.f-powf(b1,step)), bc2 = 1.f/(1.f-powf(b2,step));
    adamw<<<(2L*L*ws+TPB-1)/TPB,TPB>>>(W, G, Mm, Vv, 2L*L*ws, lrt, b1, b2, eps, wd, bc1, bc2, ws, gamma);

    if(step%100==0 || step==1){
      cudaDeviceSynchronize(); double t_end = now_ms();     // drain training work, mark interval end
      if(timing) train_ms += t_end - t_prev;                 // accumulate clean training-only wall
      float lh; CK(cudaMemcpy(&lh, lossd, 4, cudaMemcpyDeviceToHost));
      float pj = -1.f, ab[2] = {0,0};
      if(S<=2048){ // P recovery every log point: feed I, recover P (relPJ vs true flip J, untimed)
        // only valid when M==S; guard
        if(M==S){
          set_identity<<<(ws+TPB-1)/TPB,TPB>>>(X[0], S);
          for(int l=0;l<L;l++){
            const float* W1 = W+(2*l)*ws; const float* W2 = W+(2*l+1)*ws;
            CK(cudaMemcpyAsync(X[l+1], X[l], ws*4, cudaMemcpyDeviceToDevice));
            if(!fp4eval){ // exact fp32 chain
              CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, W1, S, X[l], S, &zero, Hh[l], S));
              CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, W2, S, Hh[l], S, &one, X[l+1], S));
            } else { // NVFP4-simulated operands; residual stream stays fp32
              CK(cudaMemcpyAsync(Wq1,W1,ws*4,cudaMemcpyDeviceToDevice)); if(blk<0) prune24_trans<<<dim3(S/4,S/4),1>>>(Wq1,S,S); else if(blk>0) prune24<<<dim3(S/4,S/blk),1>>>(Wq1,S,S,blk); fp4_quant<<<S,S/16>>>(Wq1,S,S);
              CK(cudaMemcpyAsync(Wq2,W2,ws*4,cudaMemcpyDeviceToDevice)); if(blk<0) prune24_trans<<<dim3(S/4,S/4),1>>>(Wq2,S,S); else if(blk>0) prune24<<<dim3(S/4,S/blk),1>>>(Wq2,S,S,blk); fp4_quant<<<S,S/16>>>(Wq2,S,S);
              CK(cudaMemcpyAsync(Aq,X[l],ws*4,cudaMemcpyDeviceToDevice)); fp4_quant<<<S,S/16>>>(Aq,S,S);
              CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, Wq1, S, Aq, S, &zero, Hh[l], S));
              CK(cudaMemcpyAsync(Hq,Hh[l],ws*4,cudaMemcpyDeviceToDevice)); fp4_quant<<<S,S/16>>>(Hq,S,S);
              CK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, S, S, S, &one, Wq2, S, Hq, S, &one, X[l+1], S));
            }
          }
          CK(cudaMemset(errd,0,12));
          pj_err<<<(ws+TPB-1)/TPB,TPB>>>(X[L], errd, S);
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
