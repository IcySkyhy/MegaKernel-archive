// pairwise_prune.cuh — HW-honest pruning kernels for the nvf4 sparse engine.
//
// WHY THIS EXISTS (2026-07-10 laptop archaeology, box offline):
// sm120 nvf4 sparse (Sm1xxGemmSparseConfig IsF4) is PAIR-granular 4:8:
// a chunk = 8 consecutive logical e2m1 along the CONTRACTION = 4 pairs
// (uint8 raw units, even/odd k share a byte); the HW keeps 2 of 4 PAIRS and
// metadata is two 2-bit PAIR selectors per chunk. Elementwise 2:4 masks — what
// train_flip_24.cu's prune24 (per-row) and prune24_trans (transposable) model —
// are NOT expressible by this engine unless the kept elements happen to form
// whole pairs. => All existing flip relPJ numbers for "2:4 weights"
// (per-row 0.34, transposable 0.40, block collapse 0.82) measured a mask family
// the tensor core cannot run. Re-gate with these kernels.
//
// For the backprop-1000 path (dW = dD . act^T, K = batch): dD pruned "2:4 along
// batch" must actually keep 2-of-4 PAIRS of adjacent batch samples per 8.
//
// fp32 simulation kernels (train_flip_24.cu style, W in-place):
#pragma once

// Per-row pairwise 4:8 along the contraction (= in-features for weights,
// = batch for dD): per 8 consecutive Si-elements, keep the 2 pairs with the
// largest |a|+|b|, zero the other 2 pairs. grid: dim3(So, Si/8), 1 thread.
__global__ void prune48_pair(float* W, int So, int Si) {
  int r = blockIdx.x, c8 = blockIdx.y * 8;
  float* p = W + (size_t)r * Si + c8;
  float mag[4];
  for (int j = 0; j < 4; j++) mag[j] = fabsf(p[2 * j]) + fabsf(p[2 * j + 1]);
  // find the two smallest-magnitude pairs, zero them
  int lo0 = 0, lo1 = 1;
  if (mag[lo1] < mag[lo0]) { int t = lo0; lo0 = lo1; lo1 = t; }
  for (int j = 2; j < 4; j++) {
    if (mag[j] < mag[lo0]) { lo1 = lo0; lo0 = j; }
    else if (mag[j] < mag[lo1]) { lo1 = j; }
  }
  p[2 * lo0] = 0.f; p[2 * lo0 + 1] = 0.f;
  p[2 * lo1] = 0.f; p[2 * lo1 + 1] = 0.f;
}

// Transposable pairwise 4:8 (needed if BOTH W and W^T ride the sparse engine):
// pair granularity on BOTH axes means the mask lives on 2x2 element blocks and
// the constraint becomes 2-of-4 BLOCKS per 8x2 stripe in each direction — i.e.
// block-transposable 2:4 over 2x2 super-elements. Reuse prune24_trans's 4x4
// search on the 2x2-block-reduced matrix: caller reduces W to B[So/2][Si/2]
// (sum of |.| over each 2x2 block), runs the existing transposable searcher on
// B, then expands the mask (keep/kill whole 2x2 blocks). No new kernel here;
// see train_flip_24.cu prune24_trans + this expansion note. Expect accuracy
// strictly <= elementwise transposable (mask family is smaller) — MEASURE.
