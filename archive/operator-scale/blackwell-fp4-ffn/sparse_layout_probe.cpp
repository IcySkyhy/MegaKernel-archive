// sparse_layout_probe.cpp — HOST-ONLY (no GPU needed): proves the analytic byte
// formulas for CUTLASS sm120 nvf4 sparse LayoutA (compressed values) and LayoutE
// (metadata) against the real cute layout objects, for aligned + padded shapes.
// These formulas are the contract fast_compress24 (bwd_1000.cu) must write to.
//
// Build (CUDA toolkit present; CUDA 13.x also needs the bundled cccl headers):
//   g++ -O2 -std=c++17 -I /usr/local/cuda/include -I /usr/local/cuda/include/cccl \
//       -I $CUT/include sparse_layout_probe.cpp -o /tmp/slp && /tmp/slp
// Also builds without a CUDA install by stubbing crt/host_defines.h, cuda_runtime.h
// and vector_types.h and supplying pip's nvidia-cuda-runtime headers plus cccl.
// Last run 2026-08-31 on CUDA 13.1 + CUTLASS 4.4.2: ALL LAYOUT FORMULAS HOLD, all
// shapes ok including padded M=200/K=320; atom_bytes=2048, AlignK=64, KMajor=1.
// Output is in bench/repro-2026-08-31/sparse_layout_probe.log.
//
// Facts encoded (sm120 nvf4 sparse, A = K-major/RowMajor, from
// cutlass/gemm/collective/builders/sm1xx_sparse_config.inl + sm90_sparse_gemm_compressor.hpp):
//   IsF4: ElementAMma=sparse_elem<4,uint8>, ElementEMma=sparse_elem<16,uint8>
//   chunk = 8 logical e2m1 along K = 4 pairs (a pair = one packed uint8: lo nibble
//   = even k, hi nibble = odd k). HW keeps 2 of 4 PAIRS per chunk (pairwise 4:8),
//   NOT 2 of 4 individual elements.
//   Compressed A: row m starts at byte m*K_aligned/4; kept pair j of row m is the
//   byte at (m*K_al + 4j)/4  i.e. simply packed pairs in order, 2 bytes per chunk.
//   Metadata E: TensorEAtom (128m x 256k logical, k-fastest inside atom), atoms
//   tile M-first then K: byte(m,k) = (m%128)*16 + (m/128)*2048
//                                  + (k/256)*(M_alE*16) + (k%256)/16
//   nibble = ((k%16)/8); nibble value = sel0 | sel1<<2, sel = kept PAIR indices
//   (0..3, ascending). Corner cases (compressor kernel):
//     - 1 nonzero pair at index 3   -> phys pairs (zero, pair), sels (0,3)
//     - 1 nonzero pair at index i<3 -> phys pairs (pair, zero), sels (i,3)
//     - 0 nonzero pairs             -> nothing written to values, sels (0,0)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "cutlass/gemm/collective/builders/sm1xx_sparse_config.inl"
#include "cute/pointer_sparse.hpp"

using namespace cute;
using ElementAMma = cute::sparse_elem<4, uint8_t>;
using ElementEMma = cute::sparse_elem<16, uint8_t>;
using Cfg = cutlass::Sm1xxGemmSparseConfig<ElementAMma, cutlass::layout::RowMajor, ElementEMma>;

static int fails = 0;
#define CHECK_EQ(a, b, ...) do { if ((long long)(a) != (long long)(b)) { \
  if (fails < 10) { printf("FAIL "); printf(__VA_ARGS__); printf("  got %lld want %lld\n", (long long)(a), (long long)(b)); } \
  fails++; } } while (0)

template <class LA, class LE>
static void probe(int M, int K, LA lA, LE lE) {
  const long M_alA = ((M + Cfg::TensorAAlignmentM{} - 1) / Cfg::TensorAAlignmentM{}) * Cfg::TensorAAlignmentM{};
  const long K_alA = ((K + Cfg::TensorAAlignmentK{} - 1) / Cfg::TensorAAlignmentK{}) * Cfg::TensorAAlignmentK{};
  const long M_alE = ((M + 127) / 128) * 128;
  const long K_alE = ((K + 255) / 256) * 256;
  (void)M_alA;
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      // LayoutA: logical offset in e2m1 units; physical byte = offset/4
      // (sparse_elem<4,uint8>: 4 logical elems share 1 stored byte = 2 kept nibbles).
      long offA = lA(m, k, 0);                        // flat k auto-split (2, K/2) colex
      long wantA = (long)m * K_alA + k;
      CHECK_EQ(offA, wantA, "A M=%d K=%d m=%d k=%d", M, K, m, k);
      // LayoutE: logical offset in E units; byte = offset/16, chunk nibble = (off%16)/8
      long offE = lE(m, k, 0);
      long wantE = (long)(m % 128) * 256 + (long)(m / 128) * 32768
                 + (long)(k / 256) * (M_alE * 256) + (k % 256);
      CHECK_EQ(offE, wantE, "E M=%d K=%d m=%d k=%d", M, K, m, k);
      // byte-level restatement used by fast_compress24:
      long byteE = offE / 16;
      long wantByteE = (long)(m % 128) * 16 + (long)(m / 128) * 2048
                     + (long)(k / 256) * (M_alE * 16) + (long)(k % 256) / 16;
      CHECK_EQ(byteE, wantByteE, "Ebyte M=%d K=%d m=%d k=%d", M, K, m, k);
    }
  }
  // sizes
  long bytesA_total = M_alA * K_alA / 4;
  long bytesE_total = (M_alE / 128) * (K_alE / 256) * 2048;
  printf("  M=%5d K=%5d  -> A: M_al=%ld K_al=%ld bytes=%ld   E: M_al=%ld K_al=%ld bytes=%ld  %s\n",
         M, K, M_alA, K_alA, bytesA_total, M_alE, K_alE, bytesE_total, fails ? "FAIL" : "ok");
}

int main() {
  printf("sm120 nvf4 sparse config constants:\n");
  printf("  LogicalElemsAPerChunk=%d PhysicalElemsAPerChunk=%d ElemsARawPerMmaRaw=%d\n",
         (int)Cfg::LogicalElemsAPerChunk{}, (int)Cfg::PhysicalElemsAPerChunk{}, (int)Cfg::ElemsARawPerElementAMmaRaw{});
  printf("  TensorEAtomM=%d TensorEAtomK=%d atom_cosize(logical)=%d atom_bytes=%d\n",
         (int)Cfg::TensorEAtomM{}, (int)Cfg::TensorEAtomK{}, (int)cosize(Cfg::TensorEAtom{}),
         (int)cosize(Cfg::TensorEAtom{}) / 16);
  printf("  TensorAAlignmentM=%d TensorAAlignmentK=%d (logical, incl 2x sparsity)  KMajor=%d\n",
         (int)Cfg::TensorAAlignmentM{}, (int)Cfg::TensorAAlignmentK{}, (int)Cfg::IsKMajor);

  struct { int M, K; } shapes[] = {
    {256, 512},      // aligned small
    {8192, 8192},    // production S
    {200, 320},      // padded both dims (if alignment permits) — exercises round-up
    {1024, 1024},    // trainer S
  };
  for (auto s : shapes) {
    auto wl = make_shape(s.M, 1, s.K, 1);
    probe(s.M, s.K, Cfg::fill_layoutA(wl), Cfg::fill_layoutE(wl));
  }
  if (fails) { printf("TOTAL FAILURES: %d\n", fails); return 1; }
  printf("ALL LAYOUT FORMULAS HOLD\n");
  return 0;
}
