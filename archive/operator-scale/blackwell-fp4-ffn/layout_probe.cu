// Decisive layout comparison: does GEMM1's col-major output SF layout match
// GEMM2's input SFB layout? If equal, a col-major-D fused chain is directly
// numerically valid (no relayout). Prints SFB, SFD(K-major), SFD(MN-major).
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
using namespace cute;

template<int VS>
void run(int M,int N,int K){
  using InCfg  = cutlass::detail::Sm1xxBlockScaledConfig<VS>;
  using OutK   = cutlass::detail::Sm1xxBlockScaledOutputConfig<VS, cute::UMMA::Major::K>;
  using OutMN  = cutlass::detail::Sm1xxBlockScaledOutputConfig<VS, cute::UMMA::Major::MN>;
  auto sfb   = InCfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto sfd_k = OutK::tile_atom_to_shape_SFD(make_shape(M,N,K,1));
  auto sfd_mn= OutMN::tile_atom_to_shape_SFD(make_shape(M,N,K,1));
  printf("--- VS=%d  M=%d N=%d K=%d ---\n", VS, M, N, K);
  printf("SFB          : "); print(sfb);    printf("  size=%d\n",(int)size(filter_zeros(sfb)));
  printf("SFD (K-major): "); print(sfd_k);  printf("  size=%d\n",(int)size(filter_zeros(sfd_k)));
  printf("SFD (MNmajor): "); print(sfd_mn); printf("  size=%d\n",(int)size(filter_zeros(sfd_mn)));
  // For the chain, GEMM2's B = D1 (which is [M,N] from GEMM1). GEMM2 sees B as
  // [K2,N2] with K2 = M(gemm1). So the relevant SFB is over (N2,K2) = (N,M).
  auto sfb_chain = InCfg::tile_atom_to_shape_SFB(make_shape(N,M,M,1)); // N2=N, K2=M
  printf("SFB_chain(N,M over K2=M): "); print(sfb_chain); printf("  size=%d\n",(int)size(filter_zeros(sfb_chain)));
  // element-by-element index compare: does SFD_MN[(m,n)] land where SFB_chain reads (n2=n, k2=m)?
  int mism=0, checked=0;
  auto Lmn = sfd_mn;              // indexed (m,n,0)
  auto Lsfb= sfb_chain;          // indexed (n,m,0)  [ (N2,K2) ]
  for(int m=0;m<M && mism<8;m++) for(int n=0;n<N && mism<8;n++){
    int i_mn  = Lmn(m,n,0);
    int i_sfb = Lsfb(n,m,0);
    checked++;
    if(i_mn != i_sfb){ if(mism<8) printf("  MISMATCH (m=%d,n=%d): SFD_MN=%d SFB=%d\n",m,n,i_mn,i_sfb); mism++; }
  }
  printf("index-equal(SFD_MN[m,n]==SFB[n,m]): %s (%d checked, %d mism)\n\n",
         mism? "NO":"YES", checked, mism);
}

int main(){
  run<16>(512,512,512);
  run<16>(256,256,256);
  run<32>(512,512,512);
  return 0;
}
