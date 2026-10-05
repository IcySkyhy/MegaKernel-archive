#!/usr/bin/env python3
# Generates sm120_fused_chain.hpp for fp4_fused_chain.cu: copies CUTLASS's
# sm120 sparse blockscaled kernel and applies anchored patches (fails loudly
# on anchor drift). Both chain GEMMs run in one persistent kernel: scheduler
# domain (M, 2N, K), second N half is GEMM2 with its own mainloop/epilogue
# params, GEMM2 B-loads gate on per n-stripe counters bumped by GEMM1 tiles.
# usage: make_fused_kernel.py [cutlass_root] [out.hpp]
import sys, os

cutlass = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/code/cutlass")
out     = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__) or ".", "sm120_fused_chain.hpp")
src = open(os.path.join(cutlass,
    "include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp")).read()

def rep(old, new, count=1):
    global src
    found = src.count(old)
    assert found == count, f"anchor count {found} != {count}: {old[:70]!r}"
    src = src.replace(old, new, count)

# 1. Rename class, drop the GemmUniversal partial-specialization machinery.
rep("""class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cutlass::detail::is_asymmetric_dma_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                        KernelTmaWarpSpecializedCooperativeSparseSm120> ||
    cutlass::detail::is_asymmetric_dma_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                        KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120>>>
{""",
"""class FusedSparseChain
{""")

# 2. Params: second GEMM param set + tile-granular gate state.
rep("""    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };""",
"""    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
    // Fused chain extras: second GEMM param set + tile-granular gate state.
    MainloopParams mainloop2{};
    EpilogueParams epilogue2{};
    int* counters{nullptr};  // per n-stripe GEMM1 tile counts, monotonic across launches
    int epoch{0};            // launch ordinal; gate target = m_tiles * epoch
    int n_tiles{0};          // n-stripes per GEMM (scheduler domain covers 2*n_tiles)
    int m_tiles{0};          // m-tiles per n-stripe
    int delay{0};            // G1 lead in columns; n_tiles = pure phases, 1 = tight interleave
    char const* c2_base{nullptr};  // GEMM2 residual: L2 prefetch target (row-major bf16)
    long c_row_bytes{0};           // C2 row pitch in bytes; 0 disables prefetch
  };""")

# 3. Device helpers before operator().
rep("""  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {""",
"""  // Map a doubled-N scheduler column to (is_gemm2, effective tile). Launch
  // order is column-major (AlongM, swz 1): d head columns are G1 stripes
  // 0..d-1, then G1/G2 alternate (G1 first), d tail columns are the last G2
  // stripes. G2 stripe i sits 2d+1 columns after its producer G1 stripe i,
  // so producers always precede consumers in CLC launch order (deadlock-free
  // for d >= 1). d = n_tiles degenerates to all-G1-then-all-G2.
  CUTLASS_DEVICE static cute::tuple<bool, typename TileScheduler::WorkTileInfo>
  remap(typename TileScheduler::WorkTileInfo const& w, int n_tiles, int d) {
    auto t = w;
    int c = t.N_idx;
    bool g2;
    int col;
    if (c < d)                    { g2 = false; col = c; }
    else if (c < 2 * n_tiles - d) { int r = c - d; g2 = (r & 1); col = (r >> 1) + (g2 ? 0 : d); }
    else                          { g2 = true;  col = c - n_tiles; }
    t.N_idx = col;
    return cute::make_tuple(g2, t);
  }

  // Spin until GEMM1 finished the n-stripe this GEMM2 tile reads (D1 producer gate).
  CUTLASS_DEVICE static void
  gate_wait(int const* counter, int target) {
    while (__ldcg(counter) < target) { __nanosleep(128); }
    __threadfence();
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {""")

# 4. Prefetch both TMA descriptor sets.
rep("""      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);""",
"""      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop2);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue2);""")

# 5. Outer epilogue pair; epi-load-needed if either needs it (must agree across warps).
rep("""    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();""",
"""    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
    CollectiveEpilogue collective_epilogue2(params.epilogue2, shared_storage.tensors.epilogue);
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed()
                           || collective_epilogue2.is_producer_load_needed();""")

# 6. Second load_init.
rep("""    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);""",
"""    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    auto load_inputs2 = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop2);""")

# 7. Coord decode remap: LoadMK then LoadNK (identical 10-space text, sequential).
coord10_old = """          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);"""
coord10_new = """          auto [g2v, wt] = remap(work_tile_info, params.n_tiles, params.delay);
          auto m_coord = idx2crd(wt.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(wt.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(wt.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);"""
rep(coord10_old, coord10_new, 2)

# 8. load_MK param + input select.
rep("""          collective_mainloop.load_MK(
            params.mainloop,""",
"""          collective_mainloop.load_MK(
            (g2v ? params.mainloop2 : params.mainloop),""")
rep("""            mainloop_pipe_producer_state_mk,
            load_inputs,""",
"""            mainloop_pipe_producer_state_mk,
            (g2v ? load_inputs2 : load_inputs),""")

# 9. Gate + select on the B loader (GEMM2 reads D1 here).
rep("""          collective_mainloop.load_NK(
            params.mainloop,""",
"""          if (g2v) {
            if (lane_idx == 0) { gate_wait(&params.counters[wt.N_idx], params.m_tiles * params.epoch); }
            __syncwarp();
          }
          collective_mainloop.load_NK(
            (g2v ? params.mainloop2 : params.mainloop),""")
rep("""            mainloop_pipe_producer_state_nk,
            load_inputs,""",
"""            mainloop_pipe_producer_state_nk,
            (g2v ? load_inputs2 : load_inputs),""")

# 10. Epi-load warp: coord remap (12-space indent).
rep("""            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);""",
"""            auto [g2v, wt] = remap(work_tile_info, params.n_tiles, params.delay);
            auto m_coord = idx2crd(wt.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(wt.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(wt.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);""")

# 11. Consumer shadow pair FIRST (context-anchored: the bare 6-space line is a
# substring of the 8-space LoadMN line, so ordering + context disambiguate).
rep("""      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);""",
"""      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
      CollectiveEpilogue collective_epilogue2(params.epilogue2, shared_storage.tensors.epilogue);""")

# 12. LoadMN shadow pair (newline pins the 8-space indent).
rep("""
        CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);""",
"""
        CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
        CollectiveEpilogue collective_epilogue2(params.epilogue2, shared_storage.tensors.epilogue);""")

# 13. Epi C load: G1 tiles may prefetch the matching C2 tile (measured loss on
# GB203, runtime-off by default); guard keeps beta=0 tiles off the pipeline;
# instance select per tile.
rep("""            epi_load_pipe_producer_state =
            collective_epilogue.load(""",
"""            if (!g2v && params.c_row_bytes != 0) {
              // This warp is otherwise idle on GEMM1 tiles: prefetch the C2
              // tile of the same (m, stripe), consumed by GEMM2 ~2d+1 columns
              // later. Converts its once-per-byte DRAM reads into L2 hits.
              char const* base = params.c2_base
                               + size_t(wt.M_idx) * 128 * params.c_row_bytes
                               + size_t(wt.N_idx) * 256;
              CUTLASS_PRAGMA_UNROLL
              for (int r = 0; r < 4; r++) {
                char const* p = base + size_t(r * 32 + lane_idx) * params.c_row_bytes;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(p + 128));
              }
            }
            if ((g2v ? collective_epilogue2 : collective_epilogue).is_producer_load_needed()) {
            epi_load_pipe_producer_state =
            (g2v ? collective_epilogue2 : collective_epilogue).load(""")
rep("""              work_tile_info.reduction_subtile_idx()
            );
          }""",
"""              work_tile_info.reduction_subtile_idx()
            );
            }
          }""")

# 14. Consumer coord remap (8-space indent).
rep("""        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);""",
"""        auto [g2v, wt] = remap(work_tile_info, params.n_tiles, params.delay);
        auto m_coord = idx2crd(wt.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(wt.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(wt.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);""")

# 15. mma mainloop param select.
rep("""            shared_storage.tensors.mainloop,
            params.mainloop,""",
"""            shared_storage.tensors.mainloop,
            (g2v ? params.mainloop2 : params.mainloop),""")

# 16. Store instance select.
rep("""          collective_epilogue.store(""",
"""          (g2v ? collective_epilogue2 : collective_epilogue).store(""")

# 17. GEMM1 completion signal.
rep("""          do_store_tail = true;
        }

        // Get next work tile""",
"""          do_store_tail = true;
        }

        if (!g2v && mma_thread_idx == 0) {
          // GEMM1 tile complete: publish to its n-stripe counter.
          // Signal follows store() return; TMA store issue skew is accepted
          // (throughput benchmark, values parked project-wide).
          __threadfence();
          atomicAdd(&params.counters[wt.N_idx], 1);
        }

        // Get next work tile""")

open(out, 'w').write(src)
print(f"wrote {out}")
