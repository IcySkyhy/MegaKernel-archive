// Chain-semantics probe for the fp4 training pipeline. Answers, with real
// numerics vs host reference (fp4_sparse_chain.cu types, but semantics live):
//   1. col-major C/D + OutputSFVectorSize=16 fp4-D: compiles? correct?
//   2. is SFD (col-major D, vec16) readable through layout_SFB by the next
//      GEMM (the chain link GEMM2.B := D1, GEMM2.SFB := SFD1)?
//   3. LinCombBlockScaleFactor's SF formula + norm_constant semantics.
//   4. 2:4 pattern granularity for e2m1 (arbitrary 2-of-4 vs aligned pairs):
//      prints structure_sparse_zero_mask_fill's pattern + tests a chosen mask.
//   5. -DPERF: wall rates at big shapes for this exact config.
// Build: 13.3 ptxas splice (see build_sparse.sh). TBM=128 numerics, 256 perf.
#include <iostream>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"

using namespace cute;

using ElementA     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag   = cutlass::layout::RowMajor;
constexpr int AlignmentA = 64;
using ElementB     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag   = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD     = cutlass::float_e2m1_t;
using ElementSFD   = cutlass::float_ue4m3_t;
using ElementC     = cutlass::bfloat16_t;
using LayoutCTag   = cutlass::layout::ColumnMajor;   // semantic chaining needs col-major
using LayoutDTag   = cutlass::layout::ColumnMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementE     = uint8_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
constexpr int OutputSFVectorSize = 32;               // match mainloop SFB vec (measured: sparse nvf4 uses vec32)
#ifndef TBM
#define TBM 128
#endif
#ifndef TBN
#define TBN 128
#endif
#ifndef TBK
#define TBK 256
#endif
using ThreadBlockShape = Shape<Int<TBM>,Int<TBN>,Int<TBK>>;
using ClusterShape     = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120,
    cutlass::epilogue::fusion::LinCombBlockScaleFactor<
        OutputSFVectorSize, ElementD, ElementAccumulator, ElementSFD, LayoutDTag, ElementC>
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelScheduleType>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutE   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;

using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
    Shape<int,int,int,int>, ElementA::DataType, LayoutATag, SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    Shape<int,int,int,int>, ElementA::DataType, LayoutATag, SparseConfig, ArchTag>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

using EA  = cutlass::float_e2m1_t;
using ESF = cutlass::float_ue4m3_t;

template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }  // subbyte-safe

static float q_e2m1(float x){ return float(EA(x)); }               // fp32 -> e2m1 -> fp32
static float q_ue4m3(float x){ return float(ESF(x)); }

// host quantize: 16-block along K, SF = ue4m3(amax/6). src row-major [R x K].
// writes val = e2m1(x/deq(SF)). Returns dequantized sim into `sim` (exact model).
template <class TensorV, class TensorSF>
void hquant(const std::vector<float>& src, int R, int K, TensorV&& vT, TensorSF&& sfT,
            std::vector<float>* sim, bool trans /*write vT(k-index major?) false: vT(r,k)*/){
  constexpr int VS = 32;   // sparse nvf4 sm120 SF vector size (measured)
  for (int r = 0; r < R; ++r)
    for (int k0 = 0; k0 < K; k0 += VS) {
      float amax = 0.f;
      for (int v = 0; v < VS; ++v) amax = std::max(amax, std::fabs(src[(size_t)r*K + k0 + v]));
      float sf = q_ue4m3(amax / 6.f);
      float rcp = sf > 0.f ? 1.f/sf : 0.f;
      sfT(r, k0, 0) = ESF(sf);
      for (int v = 0; v < VS; ++v) {
        float x = src[(size_t)r*K + k0 + v];
        float qv = q_e2m1(x * rcp);
        vT(r, k0 + v, 0) = EA(qv);
        if (sim) (*sim)[(size_t)r*K + k0 + v] = qv * sf;
      }
    }
  (void)trans;
}

int main(int argc, char** argv){
  int S     = argc>1 ? atoi(argv[1]) : 512;
  int iters = argc>2 ? atoi(argv[2]) : 200;
  int M=S, N=S, K=S;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M,K,1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N,K,1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M,N,1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M,N,1});
  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  auto layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(make_shape(M,N,K,1));

  printf("layout_SFB: "); print(layout_SFB); printf("\n");
  printf("layout_SFD: "); print(layout_SFD); printf("\n");
  printf("sizes: SFB %d SFD %d\n", (int)size(filter_zeros(layout_SFB)), (int)size(filter_zeros(layout_SFD)));

  // ---- 2:4 pattern granularity probe ----
  {
    auto workload = make_shape(64, 1, 64, 1);
    auto sa = cutlass::make_cute_packed_stride(StrideA{}, {64,64,1});
    CompressorUtility util(workload, sa);
    std::vector<EA> ones((size_t)64*64, EA(1.f));
    util.structure_sparse_zero_mask_fill(ones.data(), 123);
    printf("mask_fill row0 (48 elems as 0/1): ");
    for (int k=0;k<48;k++) printf("%d", float(ones[k])!=0.f ? 1:0);
    printf("\nmask_fill row1: ");
    for (int k=0;k<48;k++) printf("%d", float(ones[(size_t)64+k])!=0.f ? 1:0);
    printf("\n");
  }

  cudaStream_t s; cudaStreamCreate(&s);
  std::mt19937 rng(7); std::normal_distribution<float> g(0.f, 1.f);

  // ---- weights: fp32 -> pairwise 2:4 mask (keep best aligned pair of each 4) -> quantize ----
  std::vector<float> W1(M*(size_t)K), W2(M*(size_t)K), Xf(K*(size_t)N);
  for (auto& x : W1) x = g(rng) * 0.5f;
  for (auto& x : W2) x = g(rng) * 0.5f;
  for (auto& x : Xf) x = g(rng);
  auto mask24 = [&](std::vector<float>& Wv){
    for (size_t r = 0; r < (size_t)M; ++r)
      for (int k = 0; k < K; k += 4) {
        float* p = &Wv[r*K + k];
        float p0 = std::fabs(p[0]) + std::fabs(p[1]), p1 = std::fabs(p[2]) + std::fabs(p[3]);
        if (p0 >= p1) { p[2] = p[3] = 0.f; } else { p[0] = p[1] = 0.f; }
      }
  };
  mask24(W1); mask24(W2);

  cutlass::HostTensor<EA, cutlass::layout::PackedVectorLayout> W1d, W2d, Xd, D1, D2;
  cutlass::HostTensor<ESF, cutlass::layout::PackedVectorLayout> SFA1, SFA2, SFBX, SFD1, SFD2;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> Cb;
  cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> Norm;
  W1d.reset(cutlass::make_Coord(M*K)); W2d.reset(cutlass::make_Coord(M*K));
  Xd.reset(cutlass::make_Coord(K*N));
  D1.reset(cutlass::make_Coord(M*N)); D2.reset(cutlass::make_Coord(M*N));
  SFA1.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  SFA2.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  SFBX.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  SFD1.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  SFD2.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  Cb.reset(cutlass::make_Coord(M*N));
  Norm.reset(cutlass::make_Coord(2)); // [0]=1.0 used, [1]=2.0 for formula probe
  Norm.at(cutlass::make_Coord(0)) = 1.0f; Norm.at(cutlass::make_Coord(1)) = 2.0f;

  std::vector<float> W1s(M*(size_t)K), W2s(M*(size_t)K), Xs(K*(size_t)N);
  { // A tensors: row-major over (m,k); SFA layout indexed (m,k,l)
    auto w1T = make_tensor(make_iterator(W1d.host_data()), make_layout(make_shape(M,K,1), stride_A));
    auto sf1 = make_tensor(SFA1.host_data(), layout_SFA);
    hquant(W1, M, K, w1T, sf1, &W1s, false);
    auto w2T = make_tensor(make_iterator(W2d.host_data()), make_layout(make_shape(M,K,1), stride_A));
    auto sf2 = make_tensor(SFA2.host_data(), layout_SFA);
    hquant(W2, M, K, w2T, sf2, &W2s, false);
    // B tensor: col-major (k,n) with blocks along k; hquant iterates rows=n over K
    auto xT  = make_tensor(make_iterator(Xd.host_data()), make_layout(make_shape(N,K,1), stride_B));
    auto sfx = make_tensor(SFBX.host_data(), layout_SFB);
    hquant(Xf, N, K, xT, sfx, &Xs, false);  // Xf[n*K+k] = X(k,n) sample-major ✓
  }
  // C = 0.25 * gaussian, col-major (m,n)
  std::vector<float> Cf(M*(size_t)N);
  { auto cT = make_tensor(make_iterator(Cb.host_data()), make_layout(make_shape(M,N,1), stride_C));
    for (int nn=0; nn<N; ++nn) for (int mm=0; mm<M; ++mm){ float v = 0.25f*g(rng); Cf[(size_t)nn*M+mm]=v; cT(mm,nn,0)=ElementC(v);} }
  W1d.sync_device(); W2d.sync_device(); Xd.sync_device(); Cb.sync_device();
  SFA1.sync_device(); SFA2.sync_device(); SFBX.sync_device(); SFD1.sync_device(); SFD2.sync_device();
  Norm.sync_device();

  // ---- compress W1, W2 ----
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto workload = make_shape(M, 1, K, 1);
  CompressorUtility util(workload, stride_A);
  LayoutA layout_A = SparseConfig::fill_layoutA(workload);
  LayoutE layout_E = SparseConfig::fill_layoutE(workload);
  cutlass::HostTensor<EA, cutlass::layout::PackedVectorLayout> W1c, W2c;
  cutlass::HostTensor<ElementE, cutlass::layout::PackedVectorLayout> E1, E2;
  W1c.reset(cutlass::make_Coord(util.get_tensorA_m_physical()*util.get_tensorA_k_physical()));
  W2c.reset(cutlass::make_Coord(util.get_tensorA_m_physical()*util.get_tensorA_k_physical()));
  E1.reset(cutlass::make_Coord(util.get_metadata_m_physical()*util.get_metadata_k_physical()));
  E2.reset(cutlass::make_Coord(util.get_metadata_m_physical()*util.get_metadata_k_physical()));
  auto compress = [&](EA* dense_dev, EA* comp_dev, ElementE* meta_dev){
    typename Compressor::Arguments cargs{ {M,1,K,1}, { dense_dev, stride_A, comp_dev, meta_dev }, {hw_info} };
    Compressor op;
    cutlass::device_memory::allocation<uint8_t> ws(Compressor::get_workspace_size(cargs));
    CUTLASS_CHECK(op.can_implement(cargs));
    CUTLASS_CHECK(op.initialize(cargs, ws.get(), s));
    CUTLASS_CHECK(op.run(s));
    cudaStreamSynchronize(s);
  };
  compress(W1d.device_data(), W1c.device_data(), E1.device_data());
  compress(W2d.device_data(), W2c.device_data(), E2.device_data());

  auto run_gemm = [&](EA* a, ESF* sfa, LayoutA lA, ElementE* e, EA* b, ESF* sfb,
                      ElementC* c, float beta, EA* d, ESF* sfd, float* norm){
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
      { a, lA, b, stride_B, e, layout_E, sfa, layout_SFA, sfb, layout_SFB },
      { {1.0f, beta}, c, stride_C, d, stride_D }
    };
    args.epilogue.thread.block_scale_factor_ptr = sfd;
    args.epilogue.thread.norm_constant_ptr      = norm;
    Gemm gemm;
    cutlass::device_memory::allocation<uint8_t> ws(Gemm::get_workspace_size(args));
    CUTLASS_CHECK(gemm.can_implement(args));
    CUTLASS_CHECK(gemm.initialize(args, ws.get()));
    CUTLASS_CHECK(gemm.run(s));
    cudaError_t err = cudaStreamSynchronize(s);
    if (err) { printf("gemm err: %s\n", cudaGetErrorString(err)); exit(1); }
  };

#ifdef PERF
  // wall rate of this exact config (col-major fp4-D vec16), GEMM1-only + pair
  for (int i=0;i<10;i++) run_gemm(W1c.device_data(), SFA1.device_data(), layout_A, E1.device_data(),
                                  Xd.device_data(), SFBX.device_data(), Cb.device_data(), 0.f,
                                  D1.device_data(), SFD1.device_data(), Norm.device_data());
  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0,s);
  for (int i=0;i<iters;i++)
    run_gemm(W1c.device_data(), SFA1.device_data(), layout_A, E1.device_data(),
             Xd.device_data(), SFBX.device_data(), Cb.device_data(), 0.f,
             D1.device_data(), SFD1.device_data(), Norm.device_data());
  cudaEventRecord(t1,s); cudaStreamSynchronize(s);
  float ms=0; cudaEventElapsedTime(&ms,t0,t1); ms/=iters;
  printf("PERF colD-fp4-vec16 single sparse GEMM S=%d tile=%dx%dx%d beta=0: %.4f ms  %.1f TFLOPS dense-equiv\n",
         S, TBM, TBN, TBK, ms, 2.0*M*N*K/(ms/1e3)/1e12);
  return 0;
#else
  // ---- GEMM1: D1 = W1 * X (beta=0, norm=1) ----
  run_gemm(W1c.device_data(), SFA1.device_data(), layout_A, E1.device_data(),
           Xd.device_data(), SFBX.device_data(), Cb.device_data(), 0.f,
           D1.device_data(), SFD1.device_data(), Norm.device_data());
  D1.sync_host(); SFD1.sync_host();

  // host reference: R1 = W1s @ Xs   (exact quantized-operand sim, fp64 accum)
  std::vector<float> R1((size_t)M*N);
  for (int mm=0; mm<M; ++mm) for (int nn=0; nn<N; ++nn){
    double acc=0; for (int kk=0; kk<K; ++kk) acc += (double)W1s[(size_t)mm*K+kk]*Xs[(size_t)nn*K+kk];
    R1[(size_t)nn*M+mm] = (float)acc;   // col-major store
  }
  // dequant D1 via chain-link read: value через col-major (m,n); SF via layout_SFB(n=col, k=m)
  auto d1T  = make_tensor(make_iterator(D1.host_data()), make_layout(make_shape(M,N,1), stride_D));
  auto sfd1_as_sfb = make_tensor(SFD1.host_data(), layout_SFB);
  std::vector<float> D1h((size_t)M*N);
  double num=0, den=0;
  for (int nn=0; nn<N; ++nn) for (int mm=0; mm<M; ++mm){
    float v = float(EA(d1T(mm,nn,0))) * float(sfd1_as_sfb(nn,mm,0));
    D1h[(size_t)nn*M+mm] = v;
    double d = v - R1[(size_t)nn*M+mm]; num += d*d; den += (double)R1[(size_t)nn*M+mm]*R1[(size_t)nn*M+mm];
  }
  printf("CHAIN-LINK D1 (deq via layout_SFB) vs ref: rel_frob=%.4f  %s\n",
         std::sqrt(num/(den+1e-30)), std::sqrt(num/(den+1e-30)) < 0.08 ? "PASS" : "FAIL");

  // DIAG: which layout did the store actually use? read SFD1 via K-major output
  // config (idx m,n) and via MN-major output config (== SFB, idx n,m).
  {
    using OutK  = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize, cute::UMMA::Major::K>;
    using OutMN = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize, cute::UMMA::Major::MN>;
    auto lK  = OutK ::tile_atom_to_shape_SFD(make_shape(M,N,K,1));
    auto lMN = OutMN::tile_atom_to_shape_SFD(make_shape(M,N,K,1));
    auto sfK  = make_tensor(SFD1.host_data(), lK);
    auto sfMN = make_tensor(SFD1.host_data(), lMN);
    double nk=0,dk=0,nm=0,dm=0;
    for (int nn=0; nn<N; ++nn) for (int mm=0; mm<M; ++mm){
      double r = R1[(size_t)nn*M+mm];
      float vk = float(EA(d1T(mm,nn,0))) * float(sfK (mm,nn,0)); // K-major idx (m,n)
      float vm = float(EA(d1T(mm,nn,0))) * float(sfMN(mm,nn,0)); // MN-major idx (m,n)
      nk+=(vk-r)*(vk-r); dk+=r*r; nm+=(vm-r)*(vm-r); dm+=r*r;
    }
    printf("DIAG D1 via SFD(K-major idx m,n) : rel_frob=%.4f\n", std::sqrt(nk/(dk+1e-30)));
    printf("DIAG D1 via SFD(MNmajor idx m,n) : rel_frob=%.4f\n", std::sqrt(nm/(dm+1e-30)));
  }

  // ---- GEMM2: D2 = W2 * D1 + C (B := D1 buffer, SFB := SFD1 buffer!) ----
  run_gemm(W2c.device_data(), SFA2.device_data(), layout_A, E2.device_data(),
           D1.device_data(), SFD1.device_data(), Cb.device_data(), 1.f,
           D2.device_data(), SFD2.device_data(), Norm.device_data());
  D2.sync_host(); SFD2.sync_host();

  // host ref chain: R2 = W2s @ D1h + C
  std::vector<float> R2((size_t)M*N);
  for (int mm=0; mm<M; ++mm) for (int nn=0; nn<N; ++nn){
    double acc=0; for (int kk=0; kk<K; ++kk) acc += (double)W2s[(size_t)mm*K+kk]*D1h[(size_t)nn*M+kk];
    R2[(size_t)nn*M+mm] = (float)acc + Cf[(size_t)nn*M+mm];
  }
  auto d2T  = make_tensor(make_iterator(D2.host_data()), make_layout(make_shape(M,N,1), stride_D));
  auto sfd2_as_sfb = make_tensor(SFD2.host_data(), layout_SFB);
  num=0; den=0; double worst=0;
  for (int nn=0; nn<N; ++nn) for (int mm=0; mm<M; ++mm){
    float v = float(EA(d2T(mm,nn,0))) * float(sfd2_as_sfb(nn,mm,0));
    double d = v - R2[(size_t)nn*M+mm]; num += d*d; den += (double)R2[(size_t)nn*M+mm]*R2[(size_t)nn*M+mm];
    if (std::fabs(d) > worst) worst = std::fabs(d);
  }
  printf("CHAIN D2 = W2*(W1*X)+C (real scales end to end): rel_frob=%.4f worst=%.3f  %s\n",
         std::sqrt(num/(den+1e-30)), worst, std::sqrt(num/(den+1e-30)) < 0.12 ? "PASS" : "FAIL");

  // ---- norm_constant semantics: rerun GEMM1 with norm=2, test hypotheses ----
  run_gemm(W1c.device_data(), SFA1.device_data(), layout_A, E1.device_data(),
           Xd.device_data(), SFBX.device_data(), Cb.device_data(), 0.f,
           D1.device_data(), SFD1.device_data(), Norm.device_data()+1);
  D1.sync_host(); SFD1.sync_host();
  double n_h1=0, n_h2=0, dsum=0;
  for (int nn=0; nn<N; ++nn) for (int mm=0; mm<M; ++mm){
    float raw = float(EA(d1T(mm,nn,0))) * float(sfd1_as_sfb(nn,mm,0));
    double r = R1[(size_t)nn*M+mm];
    double d1 = raw - r;          // H1: dequant = v*SF (norm folded INTO stored SF)
    double d2 = 2.0*raw - r;      // H2: dequant = v*SF*norm (SF divided by norm)
    n_h1 += d1*d1; n_h2 += d2*d2; dsum += r*r;
  }
  printf("norm=2 formula: H1(v*SF) rel=%.4f  H2(v*SF*norm) rel=%.4f\n",
         std::sqrt(n_h1/dsum), std::sqrt(n_h2/dsum));
  return 0;
#endif
}
