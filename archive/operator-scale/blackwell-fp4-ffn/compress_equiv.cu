// compress_equiv.cu — BOX-READY (needs GPU): byte-for-byte validation of the
// custom fast_compress24 (bwd_1000.cu) against the CUTLASS generic compressor,
// via an analytic HOST reference of the sm120 nvf4 sparse LayoutA/LayoutE
// contract. The contract itself was proven on host without a GPU
// (sparse_layout_probe.cpp, ALL LAYOUT FORMULAS HOLD, 2026-07-10).
//
// Build on box:  ./build_sparse.sh compress_equiv.cu /tmp/ceq && /tmp/ceq [M] [K]
// Then:          #define FAST_COMPRESS_INC "your_kernel.inc" variant to also
//                memcmp fast_compress24's buffers vs the generic compressor.
//
// ============================ THE CONTRACT ==================================
// (sm1xx_sparse_config.inl IsF4 + sm90_sparse_gemm_compressor.hpp, verified)
//
// GRANULARITY: nvf4 sparse is PAIR-granular 4:8, NOT elementwise 2:4.
//   A raw unit = uint8 = 2 packed e2m1 (even k -> lo nibble, odd k -> hi).
//   A chunk = 8 logical e2m1 = 4 raw pairs; HW keeps 2 of 4 PAIRS.
//   CUTLASS's structure_sparse_zero_mask_fill zeroes whole pairs (proven).
//   => any pruner feeding this engine must prune PAIRS (|lo|,|hi| together),
//      and accuracy sims must model pairwise 4:8, not elementwise 2:4.
//
// COMPRESSED A (LayoutA, K-major/RowMajor, logical offset o=m*K_al+k, byte=o/4):
//   row m starts at byte m*K_al/4 (K_al = round_up(K,64) logical).
//   chunk c of row m = bytes [m*K_al/4 + 2c, +1]: the two KEPT pairs in
//   ascending original pair-index order.
//
// METADATA E (LayoutE, TensorEAtom 128m x 256k, atoms M-first then K):
//   byte(m,k)  = (m%128)*16 + (m/128)*2048 + (k/256)*(M_alE*16) + (k%256)/16
//   nibble     = (k%16)/8      (chunk c -> nibble c%2; lo nibble = even chunk)
//   nibble val = sel0 | (sel1<<2)   sel = kept pair indices (0..3, ascending)
//   M_alE = round_up(M,128), K_alE = round_up(K,256).
//
// CORNER CASES (compressor kernel, lines 459-475):
//   1 nonzero pair at idx 3  -> phys (zero, pair), sels (0,3)
//   1 nonzero pair at idx<3  -> phys (pair, zero), sels (idx,3)
//   0 nonzero pairs          -> phys untouched (zero-init!), sels stay 0 -> (0,0)
//   nonzero = (byte & 0x77) != 0   (sign bits masked: -0 counts as zero)
// NOTE: because the all-zero chunk leaves values UNWRITTEN, both compared
// buffers must be cudaMemset(0) first or memcmp is meaningless.
// ============================================================================
#include <iostream>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
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

// ---- same type block as bwd_probe.cu (must stay identical: defines SpCfg) ----
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
using TB=Shape<_256,_128,_256>; using CL=Shape<_1,_1,_1>;
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
using StrideA=typename SpGemm::GemmKernel::StrideA;
using LayoutA=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutE=typename SpGemm::GemmKernel::CollectiveMainloop::LayoutE;
using SpCfg=typename SpGemm::GemmKernel::CollectiveMainloop::SparseConfig;
using CompU=cutlass::transform::kernel::StructuredSparseCompressorUtility<Shape<int,int,int,int>,EA,LayoutATag,SpCfg>;
using CompK=cutlass::transform::kernel::StructuredSparseCompressor<Shape<int,int,int,int>,EA,LayoutATag,SpCfg,ArchTag>;
using Comp=cutlass::transform::device::TransformUniversalAdapter<CompK>;

#ifdef FAST_COMPRESS_INC
#include FAST_COMPRESS_INC   // must define: void fast_compress24_launch(const uint8_t* denseA, uint8_t* Acomp, uint8_t* E, int M, int K, cudaStream_t s);
#endif

// ---------------- analytic host reference (the proven contract) --------------
static void host_compress_ref(const uint8_t* dense /*M x K/2 bytes, row-major*/,
                              uint8_t* Acomp, uint8_t* E, int M, int K) {
  const long K_alA = (K + 63) / 64 * 64;
  const long M_alE = (long)(M + 127) / 128 * 128;
  for (int m = 0; m < M; m++) {
    const uint8_t* row = dense + (size_t)m * (K / 2);
    for (int c = 0; c < K / 8; c++) {
      uint8_t pr[4]; int sel[2] = {0, 0}; uint8_t phys[2] = {0, 0}; int cnt = 0;
      for (int j = 0; j < 4; j++) pr[j] = row[4 * c + j];
      for (int j = 0; j < 4 && cnt < 2; j++)
        if (pr[j] & 0x77) { sel[cnt] = j; phys[cnt] = pr[j]; cnt++; }
      if (cnt == 1 && sel[0] == 3) { phys[1] = phys[0]; phys[0] = 0; sel[0] = 0; sel[1] = 3; }
      else if (cnt == 1)           { phys[1] = 0; sel[1] = 3; }
      else if (cnt == 0)           { sel[0] = 0; sel[1] = 0; }         // values untouched
      Acomp[(size_t)m * (K_alA / 4) + 2 * c]     = phys[0];
      Acomp[(size_t)m * (K_alA / 4) + 2 * c + 1] = phys[1];
      const int k = 8 * c;
      size_t eb = (size_t)(m % 128) * 16 + (size_t)(m / 128) * 2048
                + (size_t)(k / 256) * (M_alE * 16) + (k % 256) / 16;
      uint8_t nib = (uint8_t)(sel[0] | (sel[1] << 2));
      E[eb] |= (k % 16) / 8 ? (nib << 4) : nib;
    }
  }
}

static int diff_report(const char* tag, const uint8_t* a, const uint8_t* b, size_t n) {
  size_t bad = 0, first = (size_t)-1;
  for (size_t i = 0; i < n; i++) if (a[i] != b[i]) { if (!bad) first = i; bad++; }
  if (bad) printf("  %s MISMATCH: %zu/%zu bytes differ, first @%zu (ref %02x vs gpu %02x)\n",
                  tag, bad, n, first, a[first], b[first]);
  else printf("  %s: identical (%zu bytes)\n", tag, n);
  return bad != 0;
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 1024, K = argc > 2 ? atoi(argv[2]) : 1024;
  if (M % 128 || K % 256) { printf("use M%%128==0, K%%256==0\n"); return 2; }
  cudaStream_t s; cudaStreamCreate(&s);
  cutlass::KernelHardwareInfo hw; hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  auto sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto wl = make_shape(M, 1, K, 1); CompU util(wl, sA);
  // physical sizes: tensorA counts are in EA ELEMENTS (2 per byte); metadata in uint8.
  const size_t nAelem = (size_t)util.get_tensorA_m_physical() * util.get_tensorA_k_physical();
  const size_t nA = nAelem / 2;                                   // bytes
  const size_t nE = (size_t)util.get_metadata_m_physical() * util.get_metadata_k_physical();
  const long K_alA = (K + 63) / 64 * 64, M_alE = (long)(M + 127) / 128 * 128, K_alE = ((long)K + 255) / 256 * 256;
  printf("M=%d K=%d  Acomp=%zu B (expect %ld)  E=%zu B (expect %ld)\n", M, K,
         nA, (long)M * K_alA / 4, nE, (M_alE / 128) * (K_alE / 256) * 2048);

  cutlass::HostTensor<EA, cutlass::layout::PackedVectorLayout> Wd, Wc;
  cutlass::HostTensor<ElementE, cutlass::layout::PackedVectorLayout> Me;
  Wd.reset(cutlass::make_Coord(M * K));
  Wc.reset(cutlass::make_Coord((int)nAelem));
  Me.reset(cutlass::make_Coord((int)nE));
  cutlass::reference::host::TensorFillRandomUniform(Wd.host_view(), 11, 3, -3, 0);
  util.structure_sparse_zero_mask_fill(Wd.host_data(), 12);   // valid PAIRWISE 4:8

  // hand-crafted corner rows (row 0, first 4 chunks): [0001],[0100],[0000],[1010] pair-patterns
  {
    uint8_t* raw = reinterpret_cast<uint8_t*>(Wd.host_data());   // K/2 bytes per row
    uint8_t pat[4][4] = {{0,0,0,0x21},{0,0x35,0,0},{0,0,0,0},{0x11,0,0x42,0}};
    for (int c = 0; c < 4; c++) for (int j = 0; j < 4; j++) raw[4 * c + j] = pat[c][j];
  }

  cudaMemset(Wc.device_data(), 0, nA); cudaMemset(Me.device_data(), 0, nE);
  Wd.sync_device();
  typename Comp::Arguments cargs{{M, 1, K, 1}, {Wd.device_data(), sA, Wc.device_data(), Me.device_data()}, {hw}};
  Comp cop; cutlass::device_memory::allocation<uint8_t> cws(Comp::get_workspace_size(cargs));
  CUTLASS_CHECK(cop.can_implement(cargs)); CUTLASS_CHECK(cop.initialize(cargs, cws.get(), s));
  CUTLASS_CHECK(cop.run(s)); cudaStreamSynchronize(s);
  Wc.sync_host(); Me.sync_host();

  std::vector<uint8_t> refA(nA, 0), refE(nE, 0);
  host_compress_ref(reinterpret_cast<const uint8_t*>(Wd.host_data()), refA.data(), refE.data(), M, K);

  int bad = 0;
  printf("generic CUTLASS compressor vs analytic host reference:\n");
  bad |= diff_report("values (LayoutA)", refA.data(), reinterpret_cast<uint8_t*>(Wc.host_data()), nA);
  bad |= diff_report("metadata (LayoutE)", refE.data(), reinterpret_cast<uint8_t*>(Me.host_data()), nE);

#ifdef FAST_COMPRESS_INC
  cutlass::device_memory::allocation<uint8_t> fA(nA), fE(nE);
  cudaMemset(fA.get(), 0, nA); cudaMemset(fE.get(), 0, nE);
  fast_compress24_launch(reinterpret_cast<const uint8_t*>(Wd.device_data()), fA.get(), fE.get(), M, K, s);
  cudaStreamSynchronize(s);
  std::vector<uint8_t> hA(nA), hE(nE);
  cudaMemcpy(hA.data(), fA.get(), nA, cudaMemcpyDeviceToHost);
  cudaMemcpy(hE.data(), fE.get(), nE, cudaMemcpyDeviceToHost);
  printf("fast_compress24 vs analytic host reference:\n");
  bad |= diff_report("values (LayoutA)", refA.data(), hA.data(), nA);
  bad |= diff_report("metadata (LayoutE)", refE.data(), hE.data(), nE);
#else
  printf("(rebuild with -DFAST_COMPRESS_INC='\"fast24.inc\"' to also check fast_compress24)\n");
#endif
  printf(bad ? "RESULT: FAIL\n" : "RESULT: PASS\n");
  return bad;
}
