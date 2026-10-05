// fast24.inc — fast GPU compressor that emits the EXACT sm120 nvf4 sparse
// CUTLASS LayoutA (values) + LayoutE (metadata), so its output feeds the sparse
// GEMM directly. Validated byte-for-byte against the analytic host reference
// (== the generic CUTLASS compressor) by compress_equiv.cu.
//
// Role: PRUNE+PACK. Input = dense e2m1 (row-major, K/2 bytes/row, a byte = one
// pair of 2 packed e2m1 = one 4:8 pair-unit). Per chunk of 4 pairs it keeps the
// 2 largest-magnitude pairs (|lo|+|hi|, e2m1 magnitude = low 3 bits) and packs
// them in ascending pair-index order — on a pre-pruned input (exactly <=2
// nonzero pairs/chunk) this reduces to the reference's first-2-nonzero packing,
// so the same kernel is both the validation subject and the backprop path.
//
// One thread per chunk (m,c). Acomp writes are to distinct bytes (no race).
// Metadata E: chunks 2t and 2t+1 share one byte (lo/hi nibble) -> atomicOr on
// the enclosing 32-bit word (E is memset 0 by the caller).
//
// defines: void fast_compress24_launch(const uint8_t* denseA, uint8_t* Acomp,
//                                       uint8_t* E, int M, int K, cudaStream_t s);

__global__ void fast24_pack(const uint8_t* __restrict__ dense,
                            uint8_t* __restrict__ Acomp,
                            uint32_t* __restrict__ E32,
                            int M, int K, long K_alA, long M_alE) {
  long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long chunks = (long)M * (K / 8);
  if (g >= chunks) return;
  int cpr = K / 8;                 // chunks per row
  int m = g / cpr, c = g - (long)m * cpr;

  const uint8_t* pr = dense + (size_t)m * (K / 2) + 4 * c;  // 4 pair-bytes
  // magnitude of each pair = sum of the two e2m1 magnitudes (low 3 bits/nibble)
  uint8_t b0 = pr[0], b1 = pr[1], b2 = pr[2], b3 = pr[3];
  bool nz0 = (b0 & 0x77) != 0, nz1 = (b1 & 0x77) != 0,
       nz2 = (b2 & 0x77) != 0, nz3 = (b3 & 0x77) != 0;
  int mag0 = (b0 & 7) + ((b0 >> 4) & 7), mag1 = (b1 & 7) + ((b1 >> 4) & 7),
      mag2 = (b2 & 7) + ((b2 >> 4) & 7), mag3 = (b3 & 7) + ((b3 >> 4) & 7);
  // zero-out magnitude of already-zero pairs so they never win the top-2
  if (!nz0) mag0 = -1; if (!nz1) mag1 = -1; if (!nz2) mag2 = -1; if (!nz3) mag3 = -1;
  int mag[4] = {mag0, mag1, mag2, mag3};
  uint8_t val[4] = {b0, b1, b2, b3};
  bool nz[4] = {nz0, nz1, nz2, nz3};

  // pick the two largest-magnitude pairs; keep only if actually nonzero.
  int a = 0; for (int j = 1; j < 4; j++) if (mag[j] > mag[a]) a = j;
  int b = (a == 0) ? 1 : 0; for (int j = 0; j < 4; j++) if (j != a && mag[j] > mag[b]) b = j;
  int lo = a < b ? a : b, hi = a < b ? b : a;   // ascending index

  int sel[2] = {0, 0}; uint8_t phys[2] = {0, 0};
  int cnt = (nz[lo] ? 1 : 0) + (nz[hi] ? 1 : 0);
  if (cnt == 2)      { sel[0] = lo;  phys[0] = val[lo]; sel[1] = hi; phys[1] = val[hi]; }
  else if (cnt == 1) {
    int s1 = nz[lo] ? lo : hi; uint8_t v1 = val[s1];
    if (s1 == 3)     { phys[0] = 0;  sel[0] = 0;  phys[1] = v1; sel[1] = 3; }
    else             { phys[0] = v1; sel[0] = s1; phys[1] = 0;  sel[1] = 3; }
  }
  // cnt==0: phys/sel stay 0 (Acomp memset 0, so leaving untouched is equivalent)

  // ---- LayoutA (values): row m at byte m*K_alA/4, chunk c at +2c ----
  size_t ab = (size_t)m * (K_alA / 4) + 2 * c;
  Acomp[ab]     = phys[0];
  Acomp[ab + 1] = phys[1];

  // ---- LayoutE (metadata) ----
  int k = 8 * c;
  size_t eb = (size_t)(m % 128) * 16 + (size_t)(m / 128) * 2048
            + (size_t)(k / 256) * (M_alE * 16) + (size_t)(k % 256) / 16;
  uint8_t nib = (uint8_t)(sel[0] | (sel[1] << 2));
  uint32_t packed = (uint32_t)((k % 16) / 8 ? (nib << 4) : nib) << ((eb & 3) * 8);
  atomicOr(&E32[eb >> 2], packed);
}

void fast_compress24_launch(const uint8_t* denseA, uint8_t* Acomp, uint8_t* E,
                            int M, int K, cudaStream_t s) {
  long K_alA = (K + 63) / 64 * 64;
  long K_alE = ((long)K + 255) / 256 * 256;
  long M_alE = (long)(M + 127) / 128 * 128;
  long chunks = (long)M * (K / 8);
  int thr = 256;
  long blk = (chunks + thr - 1) / thr;
  // E is OR-accumulated (two chunks share a metadata byte), so it MUST be zero
  // before every call; on changing gradients a stale nonzero selector would
  // silently corrupt an emptied chunk. Zero it here so the caller cannot forget.
  cudaMemsetAsync(E, 0, (size_t)M_alE * K_alE / 16, s);
  fast24_pack<<<(unsigned)blk, thr, 0, s>>>(denseA, Acomp, (uint32_t*)E, M, K, K_alA, M_alE);
}
