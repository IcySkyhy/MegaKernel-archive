//===- SplitLoopTransformOps.td - Split-loop schedule as transform ops ----===//
//
// The rewrites behind stages S1-S3 of the split-loop work (SplitLoops/ir/tail/make_s*.py) and
// behind the fused variant (SplitLoops/both), expressed as transform dialect operations. Each op
// checks the facts its rewrite relies on and fails with a reason instead of rewriting when they
// do not hold. Numeric choices (tile, block size, slab size, band depth, backpressure distance)
// are transform params, so a schedule can take them from transform.tune.knob and a tuner can
// choose them later.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_SPLIT_LOOP_TRANSFORM_OPS
#define ENZYMEXLA_SPLIT_LOOP_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def SplitLoopNormalizePingPongOp
    : Op<Transform_Dialect, "split_loop.normalize_pingpong",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Re-roll a host loop that ping-pongs one kernel into one launch per half-step";
  let description = [{
    Matches the `scf.while` that Polygeist produces for

        for (i = 0; i < n / 2; i++) { kern(a, b); CUDA_ERRCK; kern(b, a); CUDA_ERRCK; }

    that is: a do-while whose body launches two `enzymexla.gpu_wrapper`s that are the same kernel
    with the two buffers swapped, each followed by a `cudaGetLastError` check, and whose exit
    status is only used to report an error and exit.

    It replaces the loop by

        for (s = 0; s < 2 * tripcount; s++) { src, dst = s even ? (a, b) : (b, a); kern(src, dst) }
        if (cudaGetLastError() != 0) { report and exit, as the loop did }

    i.e. the launch error is checked once after the loop instead of after every launch (the
    control carry is specialized for the error-free path). The data carry is the ping-pong of the
    two buffers with period 2.

    Returns the new step loop and the launch inside it. Fails, naming the part that did not
    match, if the loop is not of this form.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$step_loop,
                      TransformHandleTypeInterface:$launch);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
}

def SplitLoopFusePairsOp
    : Op<Transform_Dialect, "split_loop.fuse_pairs",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Fuse each pair of half-steps into one kernel that keeps the middle state in shared memory";
  let description = [{
    `step_loop` is the loop `split_loop.normalize_pingpong` returns: step s launches the kernel
    with (src, dst) = s even ? (a, b) : (b, a). The op generates, from that kernel's body, a kernel
    that runs two consecutive steps, and replaces the step loop by

        for (p = 0; p < P; p++) fused(p even ? (a, b) : (b, a))  // pair p = steps 2p, 2p+1
        for (s = 2P; s < 2 * tripcount; s++) kern(s even ? (a, b) : (b, a))   // the stock steps left

    The fused kernel tiles the cells into `tile_x` x `tile_y` x `tile_z` blocks of `threads`
    threads (x = thread index, y = row within a plane, z = plane; a plane is grid.x rows). A block
    runs the original body on every cell of its tile plus a one-cell halo, keeps the values the
    first step would store into the tile in shared memory, synchronizes, then runs the original
    body on the tile's cells reading those values, and stores to dst as the second step does. Its
    result is in dst, not in src as after the original two steps, so pair p reads the buffer pair
    p - 1 wrote: the buffers still alternate, with period two pairs.

    Legality, checked from the kernel body (it fails naming what does not hold):
      - the body is in scatter form: every load reads the thread's own cell (one field each), and
        every field that is loaded and stored is stored exactly once, unconditionally, to a fixed
        neighbour at most one cell away in each dimension. So which slots a step writes does not
        depend on data, and the slots nobody writes are the same in every step;
      - the body has no calls, atomics, loops or other side effects, so it can be run redundantly
        for the halo;
      - the intermediate (one value per written field and tile cell) fits in the shared memory
        a block can have on the target (99 KiB on sm_120; above the 48 KiB of static shared
        memory, the GPU lowering places it in dynamic shared memory and opts the kernel in);
        `threads` is at most 1024.

    Exactness does not depend on host facts by default:
      - a slot of the middle state that no cell writes keeps the middle buffer's own value, so the
        kernel reads it from dst, the buffer the original second step would have read;
      - the original loop leaves the last step in a and the one before in b. Fused pairs leave
        the step before that in b, so the last one or two pairs run as the stock launches, and
        both buffers end exactly as in the source loop (P is the largest even number below the
        trip count).
    `assume` names host facts that make this cheaper; the op records them on the fused launch:
      - "unwritten_slots_equal": both buffers hold the same values in the slots no launch writes
        (LBM: MAIN_initialize in main.cc initializes both grids alike). Those slots are then read
        from src, which the halo already reads.
      - "dst_dead_after_loop": after the loop only a, the buffer holding the result, is read
        (LBM: main.cc copies srcGrid back and frees both). Then every pair is fused except, when
        the trip count is odd, the last.

    Returns the pair loop and the fused launch. The pair loop has the shape of a step loop, so
    `split_loop.fission`, `skew` and `capture` apply to it; the fused launch records its tile
    layers and reach for `fission`.
  }];
  let arguments = (ins TransformHandleTypeInterface:$step_loop,
                       TransformParamTypeInterface:$tile_x,
                       TransformParamTypeInterface:$tile_y,
                       TransformParamTypeInterface:$tile_z,
                       TransformParamTypeInterface:$threads,
                       OptionalAttr<StrArrayAttr>:$assume);
  let results = (outs TransformHandleTypeInterface:$pair_loop,
                      TransformHandleTypeInterface:$fused_launch);
  let assemblyFormat = [{
    $step_loop `tile` `=` `[` $tile_x `,` $tile_y `,` $tile_z `]` `threads` `=` $threads
    (`assume` `=` $assume^)? attr-dict `:` functional-type(operands, results)
  }];
}

def SplitLoopFissionOp
    : Op<Transform_Dialect, "split_loop.fission",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Cut a launch into slab launches along its outermost grid dimension";
  let description = [{
    `launch` is an `enzymexla.gpu_wrapper` whose body is one 2-D `affine.parallel` over
    (rows, threads), rows = grid.x * grid.y. The op replaces it by a loop over
    rows / `slab_rows` slabs; slab i launches the same body over `slab_rows` rows with every buffer
    pointer the body uses shifted by i slabs, and grid.y = `slab_rows` / grid.x. The kernel code
    is unchanged.

    Legality, checked from the body's accesses (all affine, through `pointer2memref` views of
    pointers defined outside the launch):
      - no atomics (blocks of one launch do not communicate);
      - rows are uniformly strided in bytes, and a row's threads stay inside the row;
      - accesses group into fields; each field is stored at most once, so writes are injective;
      - loads and stores use different pointers (the buffers are assumed not to overlap, which the
        host guarantees for LBM; the runtime guard is not emitted yet);
      - with R the largest write reach plus read reach plus a row's width, two rows further apart
        than R / row stride never touch the same byte, and `slab_rows` must be at least that.
        Then a slab of one step only depends on the slabs i-1..i+1 of the previous step, which is
        what the skew and the graph edges assume.
    The evidence is recorded on the slab loop (`split_loop.*` attributes).

    A launch made by `split_loop.fuse_pairs` is cut along its tile layers instead: `slab_rows`
    must be a whole number of tile layers, slab i launches the layers it covers by adding its
    first layer to the block index (the kernel's boundary tests use absolute cells, so its
    pointers cannot be shifted), and the reach is the one `fuse_pairs` recorded.
  }];
  let arguments = (ins TransformHandleTypeInterface:$launch,
                       TransformParamTypeInterface:$slab_rows);
  let results = (outs TransformHandleTypeInterface:$slab_loop,
                      TransformHandleTypeInterface:$slab_launch);
  let assemblyFormat = [{
    $launch `slab_rows` `=` $slab_rows attr-dict `:` functional-type(operands, results)
  }];
}

def SplitLoopSkewOp
    : Op<Transform_Dialect, "split_loop.skew",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Reorder (step, slab) launches into time-skewed bands";
  let description = [{
    `step_loop` is a loop over time steps whose body is side-effect-free setup followed by a loop
    over slabs produced by `split_loop.fission` (which established that a slab depends only on the
    neighbouring slabs of the previous step). The op finds that slab loop itself: a handle to it
    would point inside the step loop this op consumes. It replaces the nest by

        for (b = 0; b < H; b += band)            // bands of `band` steps
          for (w = 0; w < nslab + band - 1; w++)  // waves
            for (j = 0; j < min(band, H - b); j++) // steps of the band
              if (0 <= w - j < nslab) body(step = b + j, slab = w - j)

    Wave w runs step j on slab w - j, so every dependence points to an earlier wave or to a
    lower step of the same wave. Returns the band loop.
  }];
  let arguments = (ins TransformHandleTypeInterface:$step_loop,
                       TransformParamTypeInterface:$band);
  let results = (outs TransformHandleTypeInterface:$band_loop);
  let assemblyFormat = [{
    $step_loop `band` `=` $band attr-dict `:` functional-type(operands, results)
  }];
}

def SplitLoopCaptureOp
    : Op<Transform_Dialect, "split_loop.capture",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Run a band loop's launches as a replayed CUDA graph";
  let description = [{
    `band_loop` comes from `split_loop.skew` with an even band (the graph bakes in the buffer
    parity). The op routes every launch to a stream the IR creates, captures one full band into a
    graph, launches it once per full band, and runs a leftover partial band as plain launches on
    the stream.

    `edges = "chain"`: capture with the stream's own order (a total order, stage S2).
    `edges = "dag"`: bracket every captured launch with `slrt_pre(j, i)` / `slrt_post(j, i)`
    (SplitLoops/routeb/slrt_lite.cc), which replace the stream order by the producer edges
    (j-1, i-1..i+1), the slab-order edge (j, i-1) and the capacity edge (j+1, i-`backpressure`)
    (stage S3).
  }];
  let arguments = (ins TransformHandleTypeInterface:$band_loop,
                       StrAttr:$edges,
                       Optional<TransformParamTypeInterface>:$backpressure);
  let results = (outs);
  let assemblyFormat = [{
    $band_loop `edges` `=` $edges (`backpressure` `=` $backpressure^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

#endif // ENZYMEXLA_SPLIT_LOOP_TRANSFORM_OPS
