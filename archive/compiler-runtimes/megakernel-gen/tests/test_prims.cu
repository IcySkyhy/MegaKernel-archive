// ============================================================================
// Exhaustive tests for the runtime primitives.
//
// A dequant has a tiny input domain -- 256 byte values crossed with a range of
// scales -- so it can be tested completely rather than sampled.  Bit tricks are
// exactly the kind of code where a wrong nibble is invisible in an end-to-end
// logit and wrong for one input in sixteen.
// ============================================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "mk/common.cuh"
#include "mk/gemv.cuh"
#include "mk/attn.cuh"

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("cuda %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

// the eight e2m1 magnitudes, by code
static const float E2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__global__ void k_dequant_all(float* out) {
    // one thread per 32-bit word of four packed byte values
    const unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 h[4];
    mk::mxfp4_8(w, h);
    for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(h[i]);
        out[(size_t)w * 8 + 2 * i + 0] = f.x;
        out[(size_t)w * 8 + 2 * i + 1] = f.y;
    }
}

static int test_dequant_exhaustive() {
    // every byte value appears in every nibble position across these words
    const size_t NW = 1u << 16;              // all 16-bit patterns, twice over
    float* d = nullptr; CK(cudaMalloc(&d, NW * 8 * 4));
    k_dequant_all<<<NW / 256, 256>>>(d);
    CK(cudaDeviceSynchronize());
    std::vector<float> h(NW * 8);
    CK(cudaMemcpy(h.data(), d, NW * 8 * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (unsigned w = 0; w < NW; ++w) {
        for (int j = 0; j < 8; ++j) {
            const unsigned nib = (w >> (4 * j)) & 0xF;
            const float want = (nib & 8 ? -1.f : 1.f) * E2M1[nib & 7];
            const float got = h[(size_t)w * 8 + j];
            // -0.0 and +0.0 are the same value
            if (!(got == want || (want == 0.f && got == 0.f))) {
                if (bad < 8)
                    printf("  mxfp4 word %08x nibble %d: got %g want %g\n", w, j, got, want);
                ++bad;
            }
        }
    }
    cudaFree(d);
    printf("[%s] mxfp4 dequant: %zu nibbles, %d wrong\n", bad ? "FAIL" : " ok ", NW * 8, bad);
    return bad;
}

__global__ void k_scale(float* out) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    out[s] = mk::mx_scale(s);
}

static int test_scale() {
    float* d = nullptr; CK(cudaMalloc(&d, 256 * 4));
    k_scale<<<1, 256>>>(d);
    CK(cudaDeviceSynchronize());
    std::vector<float> h(256);
    CK(cudaMemcpy(h.data(), d, 256 * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int s = 1; s < 255; ++s) {                 // 0 and 255 are the special codes
        const float want = ldexpf(1.f, s - 127);
        if (h[s] != want) { if (bad < 4) printf("  scale %d: got %g want %g\n", s, h[s], want); ++bad; }
    }
    cudaFree(d);
    printf("[%s] e8m0 scale: 254 codes, %d wrong\n", bad ? "FAIL" : " ok ", bad);
    return bad;
}

__global__ void k_int4_all(float* out, float z) {
    const unsigned w = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 h[4];
    mk::int4_8(w, z, h);
    for (int t = 0; t < 4; ++t) {
        float2 f = __half22float2(h[t]);
        out[(size_t)w * 8 + t + 0] = f.x;      // values 0..3
        out[(size_t)w * 8 + t + 4] = f.y;      // values 4..7
    }
}

static int test_int4_exhaustive() {
    const size_t NW = 1u << 16;
    float* d = nullptr; CK(cudaMalloc(&d, NW * 8 * 4));
    std::vector<float> h(NW * 8);
    int bad = 0;
    for (float z : {0.f, 7.f, 8.f, 15.f}) {
        k_int4_all<<<NW / 256, 256>>>(d, z);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h.data(), d, NW * 8 * 4, cudaMemcpyDeviceToHost));
        for (unsigned w = 0; w < NW; ++w)
            for (int j = 0; j < 8; ++j) {
                const float want = (float)((w >> (4 * j)) & 0xF) - z;
                const float got = h[(size_t)w * 8 + j];
                if (got != want) {
                    if (bad < 8)
                        printf("  int4 word %08x value %d zp %g: got %g want %g\n", w, j, z, got, want);
                    ++bad;
                }
            }
    }
    cudaFree(d);
    printf("[%s] int4 dequant: %zu values x 4 zero points, %d wrong\n",
           bad ? "FAIL" : " ok ", NW * 8, bad);
    return bad;
}

// gemv against a straightforward CPU implementation
template<int K, int R>
__global__ void k_gemv(const __nv_bfloat16* W, const float* xs, float* out) {
    float acc[R];
    mk::gemv_dense<__nv_bfloat16, K, R>(W, K, xs, threadIdx.x & 31, acc);
    if ((threadIdx.x & 31) == 0)
        for (int r = 0; r < R; ++r) out[r] = acc[r];
}

template<int K, int R>
static int test_gemv(const char* name) {
    std::vector<float> hx(mk::pad33_size(K), 0.f), hw(K * R);
    for (int i = 0; i < K; ++i) hx[(i >> 5) * 33 + (i & 31)] = sinf(i * 0.37f);
    std::vector<__nv_bfloat16> hwb(K * R);
    for (int i = 0; i < K * R; ++i) { hw[i] = cosf(i * 0.11f); hwb[i] = __float2bfloat16(hw[i]); }
    __nv_bfloat16* dW; float *dx, *dout;
    CK(cudaMalloc(&dW, K * R * 2)); CK(cudaMalloc(&dx, hx.size() * 4)); CK(cudaMalloc(&dout, R * 4));
    CK(cudaMemcpy(dW, hwb.data(), K * R * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    k_gemv<K, R><<<1, 32>>>(dW, dx, dout);
    CK(cudaDeviceSynchronize());
    std::vector<float> got(R);
    CK(cudaMemcpy(got.data(), dout, R * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < R; ++r) {
        double ref = 0, mag = 0;
        for (int k = 0; k < K; ++k) {
            const double t = (double)__bfloat162float(hwb[r * K + k]) * hx[(k >> 5) * 33 + (k & 31)];
            ref += t; mag += fabs(t);
        }
        // Judge against the magnitude of the TERMS, not of the sum: a dot
        // product of 4096 terms of size ~1 that lands near zero has cancelled,
        // and a relative-to-result tolerance would be measuring the
        // conditioning of the test case rather than the kernel.
        const double err = fabs(ref - got[r]) / fmax(mag, 1e-9);
        if (err > 1e-6) { printf("  %s row %d: got %g want %g (err/|terms| %.2e)\n",
                                 name, r, got[r], ref, err); ++bad; }
    }
    cudaFree(dW); cudaFree(dx); cudaFree(dout);
    printf("[%s] gemv_dense %s\n", bad ? "FAIL" : " ok ", name);
    return bad;
}

// ---------------------------------------------------------------- attn_reduce
// The cross-split softmax combine is block-cooperative (a block per output
// chunk, its warps over the split axis), so it can only be tested by launching
// it the way the megakernel calls it.  The reference is the same arithmetic in
// double precision, including the empty-split and attention-sink cases.
template<int HD, int SPLITS, bool SINKS>
__global__ void k_attn_reduce(const float* pml, const float* psum,
                              const float* sinks, float* out, int nq) {
    extern __shared__ float scr[];
    mk::attn_reduce<float, HD, SPLITS, SINKS>(pml, psum, sinks, out, nq, scr);
}

template<int HD, int SPLITS, bool SINKS>
static int test_attn_reduce(const char* name, int nq, int empty_from) {
    const int np = nq * SPLITS;
    std::vector<float> pml(2 * np), psum((size_t)np * HD), sk(nq);
    srand(7);
    auto rnd = [] { return (float)rand() / RAND_MAX * 2.f - 1.f; };
    for (int h = 0; h < nq; ++h) {
        sk[h] = rnd();
        for (int s = 0; s < SPLITS; ++s) {
            // splits past `empty_from` carry no keys at all: (-inf, 0)
            const bool empty = s >= empty_from;
            pml[2 * (h * SPLITS + s)]     = empty ? -INFINITY : rnd() * 4.f;
            pml[2 * (h * SPLITS + s) + 1] = empty ? 0.f : (float)(rand() % 16 + 1);
            for (int d = 0; d < HD; ++d)
                psum[(size_t)(h * SPLITS + s) * HD + d] = empty ? 0.f : rnd();
        }
    }
    float *dp, *ds, *dk, *dout;
    CK(cudaMalloc(&dp, pml.size() * 4));   CK(cudaMalloc(&ds, psum.size() * 4));
    CK(cudaMalloc(&dk, sk.size() * 4));    CK(cudaMalloc(&dout, (size_t)nq * HD * 4));
    CK(cudaMemcpy(dp, pml.data(), pml.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, psum.data(), psum.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, sk.data(), sk.size() * 4, cudaMemcpyHostToDevice));
    // fewer blocks than items, so the multi-item path is exercised too; the
    // scratch is 32 floats per warp
    const int NT = 256;
    k_attn_reduce<HD, SPLITS, SINKS><<<7, NT, ((NT / 32) * 32 + 2) * 4>>>(dp, ds, dk, dout, nq);
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)nq * HD);
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int h = 0; h < nq && bad < 5; ++h) {
        double M = -INFINITY;
        for (int s = 0; s < SPLITS; ++s) M = fmax(M, (double)pml[2 * (h * SPLITS + s)]);
        if (SINKS) M = fmax(M, (double)sk[h]);
        for (int d = 0; d < HD; ++d) {
            double L = SINKS ? exp((double)sk[h] - M) : 0.0, acc = 0.0;
            if (M != -INFINITY) {
                for (int s = 0; s < SPLITS; ++s) {
                    const double f = exp((double)pml[2 * (h * SPLITS + s)] - M);
                    L += (double)pml[2 * (h * SPLITS + s) + 1] * f;
                    acc += (double)psum[(size_t)(h * SPLITS + s) * HD + d] * f;
                }
            }
            const double ref = (M == -INFINITY || L == 0.0) ? 0.0 : acc / L;
            const double err = fabs(ref - got[(size_t)h * HD + d]) / fmax(fabs(ref), 1e-3);
            if (err > 2e-5) {
                printf("  %s h=%d d=%d: got %g want %g\n", name, h, d,
                       got[(size_t)h * HD + d], ref); ++bad;
            }
        }
    }
    cudaFree(dp); cudaFree(ds); cudaFree(dk); cudaFree(dout);
    printf("[%s] attn_reduce %s\n", bad ? "FAIL" : " ok ", name);
    return bad;
}

int main() {
    int bad = 0;
    bad += test_dequant_exhaustive();
    bad += test_scale();
    bad += test_int4_exhaustive();
    bad += test_gemv<2880, 1>("K=2880 R=1 (tail path)");
    bad += test_gemv<2880, 4>("K=2880 R=4 (tail path)");
    bad += test_gemv<4096, 2>("K=4096 R=2 (no tail)");
    bad += test_gemv<1024, 8>("K=1024 R=8");
    bad += test_attn_reduce<128, 32, false>("HD=128 splits=32", 12, 32);
    bad += test_attn_reduce<64, 128, true>("HD=64 splits=128 sinks", 9, 100);
    bad += test_attn_reduce<128, 8, false>("HD=128 splits=8 all-empty head", 4, 0);
    printf("%s\n", bad ? "FAILURES" : "all primitive tests passed");
    return bad ? 1 : 0;
}
