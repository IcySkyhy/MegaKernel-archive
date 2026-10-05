// fast24_vec.inc — vectorized pair-granular 4:8 compressor for sm120 NVFP4.
// One thread handles 32 logical fp4 values (16 packed bytes), emits 8 value
// bytes in LayoutA and 2 metadata bytes in LayoutE. M%128==0, K%256==0.

__device__ __forceinline__ uint8_t fast24_pair_score(uint8_t p) {
  return uint8_t((p & 7u) + ((p >> 4) & 7u));
}

__device__ __forceinline__ uint8_t fast24_chunk(const uint8_t* pr, uint8_t* out) {
  int nz = 0, only = -1;
  #pragma unroll
  for (int j = 0; j < 4; ++j) if (pr[j] & 0x77u) { ++nz; only = j; }
  int lo = 0, hi = 0;
  if (nz == 1) { lo = (only == 3) ? 0 : only; hi = 3; }
  else if (nz >= 2) {
    int a=-1,b=-1,sa=-1,sb=-1;
    #pragma unroll
    for(int j=0;j<4;++j){
      int s=int(fast24_pair_score(pr[j]));
      if(s>sa){b=a;sb=sa;a=j;sa=s;} else if(s>sb){b=j;sb=s;}
    }
    lo=a<b?a:b; hi=a<b?b:a;
  }
  out[0]=nz?pr[lo]:0; out[1]=nz?pr[hi]:0;
  return uint8_t(lo | (hi<<2));
}

__global__ void fast_compress24_pair_kernel(const uint4* __restrict__ dense,
                                             uint2* __restrict__ comp,
                                             uint8_t* __restrict__ meta,
                                             int M, int K) {
  size_t seg_per_row=size_t(K)/32, nthr=size_t(M)*seg_per_row;
  size_t g=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(g>=nthr) return;
  int m=int(g/seg_per_row), seg=int(g-size_t(m)*seg_per_row);
  uint4 vin=dense[g]; uint32_t w[4]={vin.x,vin.y,vin.z,vin.w};
  uint8_t pr[16];
  #pragma unroll
  for(int q=0;q<4;q++){
    #pragma unroll
    for(int b=0;b<4;b++) pr[4*q+b]=uint8_t(w[q]>>(8*b));
  }
  uint8_t packed[8],nib[4];
  #pragma unroll
  for(int c=0;c<4;c++) nib[c]=fast24_chunk(pr+4*c,packed+2*c);
  uint32_t o0=0,o1=0;
  #pragma unroll
  for(int b=0;b<4;b++){o0|=uint32_t(packed[b])<<(8*b); o1|=uint32_t(packed[4+b])<<(8*b);}
  comp[g]=make_uint2(o0,o1);
  int k=seg*32; size_t M_alE=size_t((M+127)/128*128);
  size_t eb=size_t(m%128)*16+size_t(m/128)*2048+size_t(k/256)*(M_alE*16)+size_t(k%256)/16;
  meta[eb]=uint8_t(nib[0]|(nib[1]<<4)); meta[eb+1]=uint8_t(nib[2]|(nib[3]<<4));
}

static void fast_compress24_launch(const uint8_t* denseA,uint8_t* Acomp,uint8_t* E,
                                   int M,int K,cudaStream_t s){
  size_t n=size_t(M)*K/32; int t=256,b=int((n+t-1)/t);
  fast_compress24_pair_kernel<<<b,t,0,s>>>(reinterpret_cast<const uint4*>(denseA),
      reinterpret_cast<uint2*>(Acomp),E,M,K);
}
