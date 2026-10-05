#!/usr/bin/env python3
"""Generate an anchored project-local sm120 sparse collective phase probe.

The stock cooperative mainloop runs both 128-thread consumer warpgroups in
lockstep: copy K fragment, issue MMA, repeat.  This probe leaves WG0 on that
path but makes WG1 preload all K fragments before issuing its MMAs.  For the
champion K tile (two MMA K fragments), that phase-staggers copy traffic against
MMA issue without changing accumulation order, shared-memory pipeline release,
or the final all-consumer correctness barrier.

Usage: make_sparse_phase.py [cutlass_root] [out.hpp]
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
cutlass = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/code/cutlass")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "sm120_sparse_phase.hpp")
path = os.path.join(cutlass, "include/cutlass/gemm/collective/sm120_blockscaled_sparse_mma_tma.hpp")
src = open(path).read()


def rep(old, new, count=1):
    global src
    found = src.count(old)
    assert found == count, f"anchor count {found} != {count}: {old[:90]!r}"
    src = src.replace(old, new, count)


# Turn CUTLASS's partial specialization into a separately named collective.
# The original template parameter list remains intact, so a small rebind in the
# benchmark can instantiate this with every builder-generated copy/layout type.
rep("""struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedSparseBlockScaled<StagesA, StagesB, StagesE, SchedulerPipelineStageCount, ClusterShape>,
    TileShape_,
    ElementPairA_,
    LayoutPairsA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {""",
"""struct PhaseCollectiveMma {""")

# Stock loop: both WGs copy then issue the same K fragment in lockstep.  Probe:
# mma_thread_idx is threadIdx % 256, hence each 128-thread consumer WG follows a
# uniform branch.  One WG preloads both fragments while the other begins MMA;
# no fragment storage is added and each WG's accumulation order is unchanged.
rep("""      // Load A/B/E/SFA/SFB, then do gemm.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        // Copy smem->rmem for A/B/E operand
        copy_transform_A(_, k_block);
        copy_transform_B(_, k_block);
        copy_E(_, k_block);

        // Copy smem->rmem for SFA/SFB operand
        copy_SFA(_, k_block);
        copy_SFBs(k_block);

        // Gemm
        cute::gemm(tiled_mma,
                  make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrE(_,_,k_block)),
                  make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)),
                  accum);

      });""",
"""      auto phase_load = [&] (auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
        copy_transform_A(_, k_block);
        copy_transform_B(_, k_block);
        copy_E(_, k_block);
        copy_SFA(_, k_block);
        copy_SFBs(k_block);
      };
      auto phase_mma = [&] (auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
        cute::gemm(tiled_mma,
                  make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrE(_,_,k_block)),
                  make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)),
                  accum);
      };

      // Consumer WG with local thread ids 0..127 preloads both K fragments;
      // the other WG retains stock copy->MMA interleaving.  Branches are
      // warpgroup-uniform, as required by the 128-thread UMMA instructions.
      if (thread_idx < 128) {
        for_each(make_int_sequence<K_BLOCK_MAX>{}, phase_load);
        for_each(make_int_sequence<K_BLOCK_MAX>{}, phase_mma);
      }
      else {
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          phase_load(k_block);
          phase_mma(k_block);
        });
      }""")

with open(out, "w") as f:
    f.write(src)
print(f"generated {out}")
