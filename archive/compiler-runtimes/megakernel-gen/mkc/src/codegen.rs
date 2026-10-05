//! CUDA emission.
//!
//! The compiler does not emit arithmetic.  Every gemv core, norm, rotation and
//! softmax lives in the hand-written, individually tested runtime library under
//! `runtime/include/mk`.  What is emitted here is the *composition*: which stage
//! runs when, what a work item is, where the barriers fall, how the shared arena
//! is carved up, and the constants.  A generated kernel can therefore be wrong
//! in its schedule but not in its numerics.
//!
//! EXTENDING
//!   * a new weight format -> `quant.rs` only (plus a core in mk/gemv.cuh)
//!   * a new stage         -> a template below, a `StageKind`, and one line in
//!                            `CALLS`; the planner already carries it through
//!   * a new architecture  -> usually nothing here at all; `arch.rs` fills in
//!                            the IR and the templates read it
//!
//! The CUDA is written as literal templates with `@NAME@` holes rather than a
//! sequence of `writeln!` calls, so that what the compiler emits is legible as
//! CUDA in this file.

use crate::hw::Hw;
use crate::ir::*;
use crate::plan::{Plan, StageKind};
use crate::quant::Mat;
use std::fmt::Write as _;

/// Substitute `@NAME@` holes.  Longest-first so `@R@` never eats `@ROWS@`.
/// Substitute `@NAME@` holes in a template.
///
/// Longest key first, and that is load-bearing: `@SGATE@` and `@SGATEVAL@` both
/// appear in the shared-expert template, and filling the short one first would
/// leave `VAL@` behind.  A hole that no filler replaces survives into the
/// generated source, where it is a compile error at best and a comment at
/// worst -- `tests/run_tests.sh` greps the emitted tree for `@NAME@` so that
/// cannot pass unnoticed.
fn fill(t: &str, kv: &[(&str, String)]) -> String {
    let mut keys: Vec<&(&str, String)> = kv.iter().collect();
    keys.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    let mut s = t.to_string();
    for (k, v) in keys {
        s = s.replace(&format!("@{}@", k), v);
    }
    s
}

pub struct Gen<'a> {
    pub m: &'a Model,
    pub p: &'a Plan,
    /// read by emitters that need machine constants (L2 size, arch)
    #[allow(dead_code)]
    pub hw: &'a Hw,
    pub ctx: usize,
}

// ============================================================ stage templates

const T_EMBED: &str = r#"
// ---- embedding lookup -------------------------------------------------
__device__ __forceinline__ void st_embed(const Globals& g, int pos) {
    const int tok = g.tokens[pos];
    for (int i = blockIdx.x*M::NT + threadIdx.x; i < M::H; i += gridDim.x*M::NT)
        g.rt.x[i] = @EMBED@;
}
"#;

const T_QKV: &str = r#"
// ---- qkv projection ---------------------------------------------------
// R CONSECUTIVE rows per warp.  An earlier version paired row d with row
// d + head_dim/2 so a warp held both halves of a rotary pair -- but the rotation
// now happens in the attention stage, where the consuming warp already holds the
// head, and pairing distant rows made each warp open two DRAM streams instead of
// one.  Contiguous rows measured 1641 -> 2600 GB/s on the same arithmetic.
//
// q and k are left RAW here; their per-head norm and rotation happen in the
// attention stage, which removes a whole grid barrier from every layer.
__device__ __forceinline__ void st_qkv(const Globals& g, const LayerW& lw, int pos,
                                       float* xs, float* scr) {
    MK_NORM<M::H, M::AT, M::NT, M::ATTN_NORM_KIND>(g.rt.x, lw.attn_norm, nullptr,
                                                   M::NORM_EPS, xs, scr);
    constexpr int R = @R@;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int rg = mk::ws_lo(M::QKVD/R) + wid; rg < mk::ws_hi(M::QKVD/R); rg += M::NW) {
        const int n0 = rg*R;
        float acc[R];
        @GEMV@
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const int n = n0 + r;
                const float v = mk::ActRound<M::AT>::r(acc[r]@QBIAS@);
                if (n < M::QD)                 g.rt.qraw[n] = v;
                else if (n < M::QD + M::KVD)   g.rt.kraw[n - M::QD] = v;
                else {
                    const int j = n - M::QD - M::KVD;
                    lw.vcache[((size_t)(j/M::HD)*M::MAXCTX + pos)*M::HD + (j%M::HD)]
                        = mk::from_f<M::AT>(v);
                }
            }
        }
    }
}
"#;

const T_HEAD_PREP: &str = r#"
// ---- per-head prologue ------------------------------------------------
// The warp about to consume a head already holds it, so the q/k norm costs one
// warp reduction and (for head dims that are a multiple of 64) the rotation
// costs no shuffle at all.
__device__ __forceinline__ void head_prep(const float* __restrict__ raw,
                                          const float* __restrict__ nw,
                                          const M::AT* __restrict__ cosr,
                                          const M::AT* __restrict__ sinr,
                                          int head, int lane, float v[M::HD/32],
                                          float* sh) {
    constexpr int E = M::HD/32;
    #pragma unroll
    for (int e = 0; e < E; ++e) v[e] = raw[(size_t)head*M::HD + e*32 + lane];
@QKNORM@
@ROPE@
    (void)nw; (void)sh;
}
"#;

const T_ATTN: &str = r#"
// ---- attention --------------------------------------------------------
// Work item = (query head, key split); one warp owns one.  The KV cache is laid
// out [kv_head][pos][head_dim], so a head's whole history is one contiguous run
// and the @QM@ query heads sharing a kv head hit the same lines in L2, not DRAM.
//
// The current position is handled INLINE, from the raw k still in scratch, so
// the cache is read only for positions written by earlier tokens and the write
// of K[pos] races with nothing.  That is what lets the q/k norm and the rotation
// live here, in the registers of the warp that consumes them, instead of costing
// their own grid barrier.
__device__ __forceinline__ void st_attn(const Globals& g, const LayerW& lw,
                                        int L, int pos, float* xs) {
    constexpr int E = M::HD/32;
    constexpr int NITEM = M::NQ * M::SPLITS;
    float* const sh = xs + (size_t)(threadIdx.x >> 5)*M::HD;   // per-warp rope scratch
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int win = M::WINDOW[L];
    const int lo  = win ? max(0, pos - win + 1) : 0;
    const int hi  = pos;                       // cached keys are [lo, pos)
    const int per = (hi - lo + M::SPLITS - 1) / M::SPLITS;
    const M::AT* cosr = g.rt.cosd + (size_t)pos*(M::RD/2);
    const M::AT* sinr = g.rt.sind + (size_t)pos*(M::RD/2);
    for (int item = mk::ws_lo(NITEM) + w; item < mk::ws_hi(NITEM); item += M::NW) {
        const int h = item % M::NQ, sp = item / M::NQ;
        const int kvh = h / M::QM;
        const int b0 = lo + sp*per, b1 = min(hi, lo + (sp+1)*per);
        float qv[E];
        head_prep(g.rt.qraw, @QNORM@, cosr, sinr, h, lane, qv, sh);
        float m_ = -INFINITY, l_ = 0.f, o_[E];
        #pragma unroll
        for (int e = 0; e < E; ++e) o_[e] = 0.f;

        // --- the current position, from registers, never from the cache
        if (sp == 0) {
            float kv_[E];
            head_prep(g.rt.kraw, @KNORM@, cosr, sinr, kvh, lane, kv_, sh);
            if (h % M::QM == 0) {          // exactly one writer per kv head in the grid
                M::AT* kc = lw.kcache + ((size_t)kvh*M::MAXCTX + pos)*M::HD;
                #pragma unroll
                for (int e = 0; e < E; ++e) kc[e*32+lane] = mk::from_f<M::AT>(kv_[e]);
            }
            float sd = 0.f;
            #pragma unroll
            for (int e = 0; e < E; ++e) sd = fmaf(kv_[e], qv[e], sd);
            sd = mk::warp_sum(sd) * M::ATTN_SCALE;
@SOFTCAP1@            m_ = sd; l_ = 1.f;
            const M::AT* vc = lw.vcache + ((size_t)kvh*M::MAXCTX + pos)*M::HD;
            #pragma unroll
            for (int e = 0; e < E; ++e) o_[e] = mk::to_f<M::AT>(vc[e*32+lane]);
        }

        // --- cached history, AU keys at a time
        //
        // One key per iteration gives the warp 2*E loads in flight and then
        // stalls on a warp_sum and an exp before it may issue the next: the
        // stage measured 138 GB/s that way on a 48-layer MoE, 17% of the token
        // for 2 MB a layer.  Loading AU keys' K AND V up front turns that into
        // 2*AU*E independent loads; the online softmax that consumes them is
        // unchanged and still runs in a fixed order, so the result is the same
        // bits every time.  AU costs 2*AU*E live registers, which is why it is
        // small.
        constexpr int AU = 2;
        const M::AT* kb = lw.kcache + (size_t)kvh*M::MAXCTX*M::HD;
        const M::AT* vb = lw.vcache + (size_t)kvh*M::MAXCTX*M::HD;
        int t = b0;
        for (; t + AU <= b1; t += AU) {
            float kbuf[AU][E], vbuf[AU][E];
            #pragma unroll
            for (int u = 0; u < AU; ++u)
                #pragma unroll
                for (int e = 0; e < E; ++e) {
                    kbuf[u][e] = mk::to_f<M::AT>(kb[(size_t)(t+u)*M::HD + e*32+lane]);
                    vbuf[u][e] = mk::to_f<M::AT>(vb[(size_t)(t+u)*M::HD + e*32+lane]);
                }
            #pragma unroll
            for (int u = 0; u < AU; ++u) {
                float sd = 0.f;
                #pragma unroll
                for (int e = 0; e < E; ++e) sd = fmaf(kbuf[u][e], qv[e], sd);
                sd = mk::warp_sum(sd) * M::ATTN_SCALE;
@SOFTCAP2@                const float mn = fmaxf(m_, sd);
                const float cr = __expf(m_-mn), pe = __expf(sd-mn);
                l_ = l_*cr + pe;
                #pragma unroll
                for (int e = 0; e < E; ++e) o_[e] = o_[e]*cr + pe*vbuf[u][e];
                m_ = mn;
            }
        }
        for (; t < b1; ++t) {
            float sd = 0.f;
            #pragma unroll
            for (int e = 0; e < E; ++e)
                sd = fmaf(mk::to_f<M::AT>(kb[(size_t)t*M::HD + e*32+lane]), qv[e], sd);
            sd = mk::warp_sum(sd) * M::ATTN_SCALE;
@SOFTCAP2@            const float mn = fmaxf(m_, sd);
            const float cr = __expf(m_-mn), pe = __expf(sd-mn);
            l_ = l_*cr + pe;
            #pragma unroll
            for (int e = 0; e < E; ++e)
                o_[e] = o_[e]*cr + pe*mk::to_f<M::AT>(vb[(size_t)t*M::HD + e*32+lane]);
            m_ = mn;
        }
@EPILOGUE@    }
}
"#;

const T_ATTN_EPI_SPLIT: &str = r#"        const int slot = h*M::SPLITS + sp;
        if (lane == 0) { g.rt.pml[2*slot] = m_; g.rt.pml[2*slot+1] = l_; }
        #pragma unroll
        for (int e = 0; e < E; ++e) g.rt.psum[(size_t)slot*M::HD + e*32+lane] = o_[e];
"#;

const T_ATTN_EPI_LOCAL: &str = r#"        // SPLITS == 1: the softmax is entirely local, so there is no reduce stage
@SINK@        #pragma unroll
        for (int e = 0; e < E; ++e)
            g.rt.attno[(size_t)h*M::HD + e*32+lane] = mk::ActRound<M::AT>::r(o_[e]/l_);
"#;

const T_ATTN_SINK_LOCAL: &str = r#"        {
            const float sk = lw.sinks[h];
            const float M2 = fmaxf(m_, sk);
            const float sc = __expf(m_ - M2);
            l_ = l_*sc + __expf(sk - M2);
            #pragma unroll
            for (int e = 0; e < E; ++e) o_[e] *= sc;
        }
"#;

const T_ATTN_REDUCE: &str = r#"
__device__ __forceinline__ void st_attn_reduce(const Globals& g, const LayerW& lw, float* xs) {
    // the arena is free here: attention has finished with its rope scratch and
    // the output projection has not yet staged the attention output into it
    mk::attn_reduce<M::AT, M::HD, M::SPLITS, @SINKS@>(
        g.rt.pml, g.rt.psum, @SINKPTR@, g.rt.attno, M::NQ, xs);
}
"#;

const T_OPROJ: &str = r#"
// ---- output projection + residual -------------------------------------
__device__ __forceinline__ void st_oproj(const Globals& g, const LayerW& lw, float* xs) {
    mk::vec_to_smem<M::QD, M::NT>(g.rt.attno, xs);
    constexpr int R = @R@;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int rg = mk::ws_lo(M::H/R) + wid; rg < mk::ws_hi(M::H/R); rg += M::NW) {
        const int n0 = rg*R;
        float acc[R];
        @GEMV@
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v = mk::ActRound<M::AT>::r(acc[r]@OBIAS@);
                @WRITE@
            }
        }
    }
}
"#;

const T_MLP1: &str = r#"
// ---- dense MLP, gate/up fused -----------------------------------------
// gu is stored with gate and up INTERLEAVED (row 2c gate, 2c+1 up), so one warp
// produces both halves of a channel and applies the activation immediately --
// the intermediate never leaves registers.
__device__ __forceinline__ void st_mlp1(const Globals& g, const LayerW& lw,
                                        float* xs, float* scr) {
    MK_NORM<M::H, M::AT, M::NT, M::FFN_NORM_KIND>(g.rt.x, lw.ffn_norm, nullptr,
                                                  M::NORM_EPS, xs, scr);
    constexpr int R = @R@, CH = R/2;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int it = mk::ws_lo(M::INTER/CH) + wid; it < mk::ws_hi(M::INTER/CH); it += M::NW) {
        const size_t row = (size_t)R*it;
        float acc[R];
        @GEMV@
        if (lane == 0) {
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                const float gv = mk::ActRound<M::AT>::r(acc[2*c]);
                const float uv = mk::ActRound<M::AT>::r(acc[2*c+1]);
                g.rt.act[it*CH + c] = @ACT@;
            }
        }
    }
}
"#;

const T_MLP2: &str = r#"
__device__ __forceinline__ void st_mlp2(const Globals& g, const LayerW& lw, float* xs) {
    mk::vec_to_smem<M::INTER, M::NT>(g.rt.act, xs);
    constexpr int R = @R@;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int rg = mk::ws_lo(M::H/R) + wid; rg < mk::ws_hi(M::H/R); rg += M::NW) {
        const int n0 = rg*R;
        float acc[R];
        @GEMV@
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) @WRITE2@
        }
    }
}
"#;

const T_ROUTER: &str = r#"
// ---- router -----------------------------------------------------------
// One expert per BLOCK, not per warp.  Every block recomputing all @NEXP@ gate
// logits turns a few hundred kB into tens of MB of L2 traffic, and was measured
// at a third of the whole layer in the project this generalises.
__device__ __forceinline__ void st_router(const Globals& g, const LayerW& lw,
                                          float* xs, float* scr) {
    MK_NORM<M::H, M::AT, M::NT, M::FFN_NORM_KIND>(g.rt.x, lw.ffn_norm, nullptr,
                                                  M::NORM_EPS, xs, scr);
    for (int e = blockIdx.x; e < M::NEXP; e += gridDim.x) {
        const float acc = mk::gemv_dense_block<M::AT, M::H>(lw.router_w + (size_t)e*M::H, xs, scr);
        if (threadIdx.x == 0)
            g.rt.gate[e] = mk::ActRound<M::AT>::r(acc@RBIAS@);
    }
}

// Recomputed per block from the gate logits already in L2: a few hundred bytes,
// far cheaper than the extra launch and barrier a dedicated stage would cost.
__device__ __forceinline__ void st_topk(const Globals& g, int* seidx, float* sew) {
    mk::topk_warp<M::NEXP, M::TOPK, @SCORE@, @SAT@, @NORMK@>(
        g.rt.gate, @CORR@, seidx, sew, M::ROUTED_SCALE);
}
"#;

const T_MOE1: &str = r#"
// ---- expert gate/up ---------------------------------------------------
// xs still holds ffn_norm(x) from the router stage: a grid barrier does not
// clear shared memory, and st_topk only reads the gate logits.  Not recomputing
// that norm is a fusion that only exists inside one kernel.
__device__ __forceinline__ void st_moe1(const Globals& g, const LayerW& lw,
                                        float* xs, const int* seidx) {
    constexpr int R = @R@, CH = R/2;
    constexpr int NITEM = M::TOPK * (M::INTER/CH);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int it = mk::ws_lo(NITEM) + wid; it < mk::ws_hi(NITEM); it += M::NW) {
        const int slot = it/(M::INTER/CH), cg = it%(M::INTER/CH);
        const size_t row = (size_t)seidx[slot]*(2*M::INTER) + (size_t)R*cg;
        float acc[R];
        @GEMV@
        if (lane == 0) {
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                const float gv = mk::ActRound<M::AT>::r(acc[2*c]@EBIAS0@);
                const float uv = mk::ActRound<M::AT>::r(acc[2*c+1]@EBIAS1@);
                g.rt.act[slot*M::INTER + cg*CH + c] = @ACT@;
            }
        }
    }
}
"#;

const T_MOE2: &str = r#"
// ---- expert down-projection -------------------------------------------
// The @TOPK@ expert gemvs are interleaved into ONE group loop, not run back to
// back.  They read the same output rows from different expert matrices, so
// interleaving multiplies the loads in flight per warp by the fusion degree at
// no cost in item count -- and at batch 1, loads in flight per warp IS the
// bandwidth.  The degree is chosen by the planner against the register budget.
__device__ __forceinline__ void st_moe2(const Globals& g, const LayerW& lw, float* as,
                                        const int* seidx, const float* sew) {
    constexpr int PAD = mk::pad33_size(M::INTER);
    for (int s = 0; s < M::TOPK; ++s) {
        for (int i = threadIdx.x; i < M::INTER; i += M::NT)
            as[s*PAD + mk::pad33(i)] = g.rt.act[s*M::INTER + i];
        for (int q = threadIdx.x; q < M::GI; q += M::NT) as[s*PAD + q*33 + 32] = 0.f;
    }
    __syncthreads();
    constexpr int R = @R@, F = MOE2_FUSE, NG = M::TOPK / F;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const float* xp[M::TOPK];
    #pragma unroll
    for (int s = 0; s < M::TOPK; ++s) xp[s] = as + s*PAD;
    for (int rg = mk::ws_lo(M::H/R) + wid; rg < mk::ws_hi(M::H/R); rg += M::NW) {
        const int n0 = rg*R;
        float acc[M::TOPK][R];
        #pragma unroll
        for (int gi = 0; gi < NG; ++gi) {
@GEMVMULTI@        }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float tot = 0.f;
                #pragma unroll
                for (int s = 0; s < M::TOPK; ++s) tot += sew[s]*(acc[s][r]@DBIAS@);
                g.rt.x[n0+r] = mk::from_f<M::AT>(mk::to_f<M::AT>(g.rt.x[n0+r])
                                                 + mk::ActRound<M::AT>::r(tot));
            }
        }
    }
}
"#;

const T_POSTNORM: &str = r#"
// ---- @NAME@ ----------------------------------------------------------
// Gemma normalises a sublayer's OUTPUT before the residual add.  The norm needs
// the whole vector, so the sublayer wrote to a temporary and this stage folds it
// in.  Every block recomputes the reduction from L2 rather than synchronising on
// one producer -- H floats per block is far cheaper than a second barrier.
__device__ __forceinline__ void st_@NAME@(const Globals& g, const LayerW& lw, float* scr) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < M::H; i += M::NT) { const float v = g.rt.tmp[i]; ss = fmaf(v, v, ss); }
    ss = mk::block_sum(ss, scr);
    const float inv = rsqrtf(ss / M::H + M::NORM_EPS);
    for (int i = mk::ws_lo(M::H) + threadIdx.x; i < mk::ws_hi(M::H); i += M::NT) {
        const float w = lw.@W@[i];
        const float v = @APPLY@;
        g.rt.x[i] = mk::from_f<M::AT>(mk::to_f<M::AT>(g.rt.x[i]) + v);
    }
}
"#;

const T_SHARED: &str = r#"
// ---- shared expert ----------------------------------------------------
// Runs BEFORE the routed experts, because both consume the ffn_norm(x) that is
// still sitting in the shared arena and both add into the same residual.  Its
// output is scaled by a sigmoid gate, computed once per block from the same
// normalised activation -- a 1 x H dot, cheaper than a barrier.
__device__ __forceinline__ void st_shmlp1(const Globals& g, const LayerW& lw,
                                          float* xs, float* scr) {
@SGATE@    constexpr int R = @R1@, CH = R/2;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int it = mk::ws_lo(M::SINTER/CH) + wid; it < mk::ws_hi(M::SINTER/CH); it += M::NW) {
        const size_t row = (size_t)R*it;
        float acc[R];
        @GEMV1@
        if (lane == 0) {
            #pragma unroll
            for (int c = 0; c < CH; ++c) {
                const float gv = mk::ActRound<M::AT>::r(acc[2*c]);
                const float uv = mk::ActRound<M::AT>::r(acc[2*c+1]);
                g.rt.act[it*CH + c] = @ACT@;
            }
        }
    }
}

__device__ __forceinline__ void st_shmlp2(const Globals& g, const LayerW& lw, float* xs0) {
    // its own arena region: the routed experts still need ffn_norm(x) in xs0
    float* xs = xs0 + SHARED_ARENA_OFF;
    mk::vec_to_smem<M::SINTER, M::NT>(g.rt.act, xs);
    const float sg = @SGATEVAL@;
    constexpr int R = @R2@;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int rg = mk::ws_lo(M::H/R) + wid; rg < mk::ws_hi(M::H/R); rg += M::NW) {
        const int n0 = rg*R;
        float acc[R];
        @GEMV2@
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r)
                g.rt.x[n0+r] = mk::from_f<M::AT>(mk::to_f<M::AT>(g.rt.x[n0+r])
                                                 + mk::ActRound<M::AT>::r(sg*acc[r]));
        }
    }
}
"#;

const T_LMHEAD: &str = r#"
// ---- final norm + lm head ---------------------------------------------
// The single largest contiguous stream in the model, and the only stage long
// enough to reach steady-state bandwidth.
__device__ __forceinline__ void st_lmhead(const Globals& g, float* xs, float* scr) {
    MK_NORM<M::H, M::AT, M::NT, M::FINAL_NORM_KIND>(g.rt.x, g.fnorm, nullptr,
                                                    M::NORM_EPS, xs, scr);
    constexpr int R = @R@, NR = M::VOCAB / R;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int rg = mk::ws_lo(NR) + wid; rg < mk::ws_hi(NR); rg += M::NW) {
        const int n0 = rg*R;
        float acc[R];
        @GEMV@
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < R; ++r) g.logits[n0+r] = @CAP@;
        }
    }
    // vocab tail, one thread per row
    for (int n = NR*R + blockIdx.x*M::NT + threadIdx.x; n < M::VOCAB; n += gridDim.x*M::NT) {
        float a = 0.f;
        for (int k = 0; k < M::H; ++k)
            a = fmaf(mk::to_f<M::AT>(g.unembed[(size_t)n*M::H+k]), xs[mk::pad33(k)], a);
        g.logits[n] = mk::ActRound<M::AT>::r(a);
    }
}
"#;

const T_KERNEL: &str = r#"
// ======================================================================
// The megakernel: one cooperative launch runs all @NL@ layers, the LM head and
// the sampler.  Stages inside a layer are genuinely dependent -- every block
// needs the whole hidden state to normalise it -- so they are separated by grid
// barriers, and everything that is NOT a real dependence has been arranged out
// of existence by the planner.
//
// The grid must be exactly the resident set: a persistent kernel with grid-wide
// barriers deadlocks if one block is not resident, so the host derives it from
// cudaOccupancyMaxActiveBlocksPerMultiprocessor rather than guessing.
//
// The second __launch_bounds__ argument is the occupancy the plan was made
// under.  Without it ptxas takes as many registers as it likes -- 128 on this
// kernel, with no spills and no need, when its hungriest stage wants 64 -- and
// silently halves the resident warp count that every single-wave stage's
// bandwidth depends on.  `mkc build` backs the number off until the assembler
// accepts it without spilling.
// planned for @BPSDESC@ at @REGS@ registers
extern "C" __global__ __launch_bounds__(M::NT@MINB@) void mk_megakernel(Globals g) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ char smem[];
    float* xs    = reinterpret_cast<float*>(smem);
    float* scr   = xs + ARENA_FLOATS;
    int*   seidx = reinterpret_cast<int*>(scr + M::NW);
    float* sew   = reinterpret_cast<float*>(seidx + (M::TOPK ? M::TOPK : 1));

    // one %globaltimer read per barrier, from block 0 only: it costs nothing and
    // it is how the compiler checks its cost model against the machine
    int _pi = 0;
    const bool _pm = (g.prof != nullptr) && (blockIdx.x == 0) && (threadIdx.x == 0);
    #define MK_MARK() do { if (_pm) g.prof[_pi] = mk::gtime(); ++_pi; } while(0)
    MK_MARK();

    const int pos = *g.dpos;          // device-side: no host round trip per token
    st_embed(g, pos);
    grid.sync(); MK_MARK();

    for (int L = 0; L < M::NL; ++L) {
        // a REFERENCE, not a copy: LayerW is a dozen pointers, and copying it
        // pins ~26 registers live across the whole layer body
        const LayerW& lw = g.layers[L];
@BODY@    }
    st_lmhead(g, xs, scr);
    grid.sync(); MK_MARK();

@SAMPLE@    if (blockIdx.x == 0 && threadIdx.x == 0) *g.dpos = pos + 1;
    #undef MK_MARK
}
"#;

const T_SAMPLE: &str = r#"    // greedy sampling, on device, so a decode step needs no host round trip
    if (g.greedy) {
        unsigned long long best = 0;      // (order-preserving float bits << 32) | ~index
        for (int i = blockIdx.x*M::NT + threadIdx.x; i < M::VOCAB; i += gridDim.x*M::NT) {
            unsigned int u = __float_as_uint(g.logits[i]);
            u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            const unsigned long long c = ((unsigned long long)u << 32) | (unsigned)(~i);
            if (c > best) best = c;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) {
            const unsigned long long v = __shfl_xor_sync(0xffffffffu, best, o);
            if (v > best) best = v;
        }
        if ((threadIdx.x & 31) == 0) atomicMax(g.dargmax, best);
        grid.sync();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // during prefill the next token is already known; writing the argmax
            // there would silently replace the prompt with the model's guess
            if (pos + 1 >= g.n_prompt)
                g.tokens[pos+1] = (int)(~(unsigned)(*g.dargmax & 0xffffffffu));
            *g.dargmax = 0;
        }
    }
"#;

// ============================================================ the emitter

impl<'a> Gen<'a> {
    fn at(&self) -> &'static str {
        self.m.act_dtype.c_type()
    }

    /// Row blocking the planner chose for a stage.
    fn r_of(&self, k: StageKind, dflt: usize) -> usize {
        self.p
            .layer
            .iter()
            .chain(self.p.tail.iter())
            .find(|s| s.kind == k)
            .map(|s| s.r)
            .unwrap_or(dflt)
    }

    fn mat(&self, role: &'a str, q: &'a Quant, ksym: &'a str, k: usize, rows: usize) -> Mat<'a> {
        Mat { role, q, k, ksym, rows, at: self.at() }
    }

    /// The activation, with the roundings where the reference puts them: gate and
    /// up are materialised in the activation dtype, the non-linearity is applied
    /// there, and the product is rounded again.
    fn act_expr(&self, g: &str, u: &str) -> String {
        let r = |f: &str| {
            format!("mk::ActRound<M::AT>::r(mk::ActRound<M::AT>::r({f}({g})) * ({u}))")
        };
        match self.m.ffn.act() {
            Act::Silu => r("mk::act_silu"),
            Act::Gelu => r("mk::act_gelu"),
            Act::GeluTanh => r("mk::act_gelu_tanh"),
            Act::Relu2 => r("mk::act_relu2"),
            Act::SwigluClamped { alpha, limit } => {
                format!("mk::swiglu_clamped({g}, {u}, {alpha:.6}f, {limit:.6}f)")
            }
        }
    }

    // ---------------------------------------------------------------- model.h
    pub fn model_h(&self) -> String {
        let m = self.m;
        let p = self.p;
        let mut s = format!(
            "// generated by mkc {} -- do not edit\n// model: {}  arch: {}\n\
             #pragma once\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n\
             #include <cuda_fp8.h>\n#include <cstdint>\n\nnamespace M {{\n\
             using AT = {};   // activation / dense-weight element\n",
            crate::VERSION, m.name, m.arch, self.at()
        );
        let qm = m.attn.n_heads / m.attn.n_kv_heads;
        let inter = m.ffn.intermediate();
        for (k, v) in [
            ("H", m.hidden), ("NL", m.n_layers), ("VOCAB", m.vocab),
            ("NQ", m.attn.n_heads), ("NKV", m.attn.n_kv_heads), ("HD", m.attn.head_dim),
            ("QM", qm), ("QD", m.q_dim()), ("KVD", m.kv_dim()), ("QKVD", m.qkv_dim()),
            ("RD", m.rope.rotary_dim), ("INTER", inter),
            ("SINTER", match &m.ffn { Ffn::Moe { shared_intermediate, .. } => *shared_intermediate, _ => 0 }.max(1)),
            // the kv cache must outlast the planning context: a benchmark
            // prefills `ctx` tokens and then decodes hundreds more past it
            ("MAXCTX", (self.ctx + (self.ctx / 2).max(512)).next_power_of_two().max(1024)),
            ("NT", p.nt), ("NW", p.nt / 32), ("SPLITS", p.attn_splits),
            ("GK", m.hidden / 32), ("GI", inter / 32),
        ] {
            let _ = writeln!(s, "constexpr int {:<8} = {};", k, v);
        }
        // what this kernel was PRICED for: the schedule is only as good as the
        // machine model behind it, and running it on a different card is not
        // wrong but is no longer the compiler's claim
        let _ = writeln!(s, "constexpr const char* PLANNED_FOR = \"{}\";", self.hw.device);
        let f = |k: &str, v: f32| format!("constexpr float {:<11}= {:.9e}f;\n", k, v);
        s.push_str(&f("ATTN_SCALE", m.attn.scale));
        s.push_str(&f("NORM_EPS", m.attn_norm.eps));
        s.push_str(&f("SOFTCAP", m.attn.logit_softcap.unwrap_or(0.0)));
        s.push_str(&f("LOGIT_CAP", m.logit_softcap.unwrap_or(0.0)));
        s.push_str(&f("EMBED_SCALE", m.embed_scale.unwrap_or(1.0)));
        let kind = |n: &Norm| match n.kind {
            NormKind::Rms => "0 /*RMS*/",
            NormKind::RmsOnePlus => "1 /*RMS_1P*/",
            NormKind::Layer => "2 /*LAYER*/",
        };
        for (k, v) in [
            ("ATTN_NORM_KIND", kind(&m.attn_norm)),
            ("FFN_NORM_KIND", kind(&m.ffn_norm)),
            ("FINAL_NORM_KIND", kind(&m.final_norm)),
        ] {
            let _ = writeln!(s, "constexpr int   {:<15}= {};", k, v);
        }
        let _ = writeln!(s, "constexpr int   TIED_EMBED     = {};", m.tie_embeddings as i32);
        match m.moe() {
            Some(c) => {
                let _ = writeln!(s, "constexpr int   NEXP = {};\nconstexpr int   TOPK = {};",
                                 c.n_experts, c.top_k);
                let _ = writeln!(s, "constexpr float ROUTED_SCALE = {:.9e}f;", c.routed_scale);
            }
            None => {
                let _ = writeln!(s, "constexpr int   NEXP = 0;\nconstexpr int   TOPK = 0;");
                let _ = writeln!(s, "constexpr float ROUTED_SCALE = 1.f;");
            }
        }
        let _ = write!(s, "__device__ __constant__ int WINDOW[NL] = {{");
        for (i, w) in m.attn.window.iter().enumerate() {
            let _ = write!(s, "{}{}", if i > 0 { "," } else { "" }, w.unwrap_or(0));
        }
        let _ = writeln!(s, "}};\n}} // namespace M");
        s
    }

    // ---------------------------------------------------------------- kernel.h
    fn structs(&self) -> String {
        let m = self.m;
        let at = self.at();
        let mut s = String::from(
            "// One layer's device pointers.  Filled by the loader; the kernel takes a\n\
             // reference, never a copy -- copying pins every pointer in a register for\n\
             // the whole layer body.\nstruct LayerW {\n    const float* attn_norm;\n",
        );
        s.push_str(&self.mat("qkv", &m.q_attn, "M::H", m.hidden, m.qkv_dim()).fields(m.act_dtype));
        if m.attn.qkv_bias { s.push_str("    const float* qkv_b;\n"); }
        if m.attn.q_norm.is_some() { s.push_str("    const float* q_norm;\n    const float* k_norm;\n"); }
        if m.attn.sinks { s.push_str("    const float* sinks;\n"); }
        s.push_str(&self.mat("o", &m.q_attn, "M::QD", m.q_dim(), m.hidden).fields(m.act_dtype));
        if m.attn.o_bias { s.push_str("    const float* o_b;\n"); }
        if m.post_attn_norm { s.push_str("    const float* post_attn_norm;\n"); }
        s.push_str("    const float* ffn_norm;\n");
        if m.post_ffn_norm { s.push_str("    const float* post_ffn_norm;\n"); }
        let _ = write!(s, "    {at}* kcache;\n    {at}* vcache;\n");
        let inter = m.ffn.intermediate();
        match &m.ffn {
            Ffn::Dense { .. } => {
                s.push_str(&self.mat("gu", &m.q_ffn, "M::H", m.hidden, 2 * inter).fields(m.act_dtype));
                s.push_str(&self.mat("dn", &m.q_ffn, "M::INTER", inter, m.hidden).fields(m.act_dtype));
            }
            Ffn::Moe { cfg, .. } => {
                let _ = write!(s, "    const {at}* router_w;\n    const float* router_b;\n");
                if cfg.router_correction_bias { s.push_str("    const float* router_cb;\n"); }
                if let Ffn::Moe { shared_intermediate: si, .. } = &m.ffn {
                    if *si > 0 {
                        s.push_str(&self.mat("sgu", &m.q_ffn, "M::H", m.hidden, 2 * si).fields(m.act_dtype));
                        s.push_str(&self.mat("sdn", &m.q_ffn, "M::SINTER", *si, m.hidden).fields(m.act_dtype));
                        if m.weights.contains_key("l0.sgate_w") {
                            let _ = write!(s, "    const {at}* sgate_w;\n");
                        }
                    }
                }
                s.push_str(&self.mat("egu", &m.q_ffn, "M::H", m.hidden, 2 * inter).fields(m.act_dtype));
                s.push_str(&self.mat("edn", &m.q_ffn, "M::INTER", inter, m.hidden).fields(m.act_dtype));
                s.push_str("    const float* egu_b;\n    const float* edn_b;\n");
            }
        }
        s.push_str("};\n\n");
        let _ = write!(s,
            "// Per-token scratch, all in device memory so the kernel takes no host input.\n\
             struct Rt {{\n    {at}*  x;        // residual stream, H\n\
             \x20   float* qraw;    // QD, pre-norm pre-rope\n\
             \x20   float* kraw;    // KVD\n\
             \x20   float* attno;   // QD, attention output\n\
             \x20   float* psum;    // NQ*SPLITS*HD partial outputs\n\
             \x20   float* pml;     // NQ*SPLITS*2 running (max, sum)\n\
             \x20   float* act;     // MLP activation\n\
             \x20   float* gate;    // router logits\n\
             \x20   float* tmp;     // sublayer output, when a post-norm follows\n\
             \x20   float* sgate;   // shared-expert sigmoid gate\n\
             \x20   const {at}* cosd;\n    const {at}* sind;\n}};\n\n\
             struct Globals {{\n    const LayerW* layers;\n    Rt rt;\n\
             \x20   const float* fnorm;\n    const {at}* embed;\n    const {at}* unembed;\n\
             \x20   float* logits;\n\
             \x20   int*   tokens;   // [max_len] input ids; greedy mode appends\n\
             \x20   int*   dpos;     // device-side position: no host round trip\n\
             \x20   int    n_prompt; // greedy mode never overwrites a prompt token\n\
             \x20   unsigned long long* dargmax;\n    int    greedy;\n\
             \x20   // optional per-stage timeline, written by block 0 only\n\
             \x20   unsigned long long* prof;\n}};\n\n");
        s
    }

    pub fn kernel_h(&self) -> String {
        let nstage = self.p.layer.iter().filter(|x| x.barrier_after).count();
        let mut s = format!(
            "// generated by mkc {} -- shared between the kernel and its host runtime\n\
             #pragma once\n#include \"model.h\"\n#include <cstdint>\n\n",
            crate::VERSION
        );
        s.push_str(&self.structs());
        let _ = writeln!(s, "constexpr int ARENA_FLOATS = {};", self.p.arena_floats);
        let _ = writeln!(s, "constexpr int PLAN_BPS     = {};", self.p.blocks_per_sm);
        let _ = writeln!(s, "constexpr int MOE2_FUSE    = {};", self.p.moe2_fuse.max(1));
        let _ = writeln!(s, "constexpr int SHARED_ARENA_OFF = {};", crate::plan::shared_arena_offset(self.m));
        let _ = writeln!(s, "#define MK_STAGES_PER_LAYER {}", nstage);
        // embed, the layer stages, the lm head, the sampler
        // one 256-byte rounding per `take` in the runtime's scratch arena
        let _ = writeln!(s, "#define MK_SCRATCH_SLOTS 32");
        let _ = writeln!(s, "#define MK_PROF_SLOTS {}", nstage * self.m.n_layers + 8);
        // from plan::smem_bytes, so the launch and the occupancy ceiling agree
        let _ = writeln!(s, "constexpr size_t SMEM_BYTES = {};", self.p.smem_bytes);
        let _ = writeln!(s, "extern \"C\" __global__ void mk_megakernel(Globals g);");
        s
    }
}

// ============================================================ stage bodies
impl<'a> Gen<'a> {
    fn t_embed(&self) -> String {
        let e = if self.m.embed_scale.is_some() {
            "mk::from_f<M::AT>(mk::ActRound<M::AT>::r(\n            \
             mk::to_f<M::AT>(g.embed[(size_t)tok*M::H + i]) * M::EMBED_SCALE))"
        } else {
            "g.embed[(size_t)tok*M::H + i]"
        };
        fill(T_EMBED, &[("EMBED", e.into())])
    }

    fn t_qkv(&self) -> String {
        let m = self.m;
        let r = self.r_of(StageKind::QkvRope, 2);
        let mat = self.mat("qkv", &m.q_attn, "M::H", m.hidden, m.qkv_dim());
        fill(T_QKV, &[
            ("R", r.to_string()),
            ("GEMV", mat.gemv("lw.", "n0", "R", "xs", "acc")),
            ("QBIAS", if m.attn.qkv_bias { " + lw.qkv_b[n]".into() } else { String::new() }),
        ])
    }

    fn t_head_prep(&self) -> String {
        let m = self.m;
        let qk = if m.attn.q_norm.is_some() {
            "    mk::head_rmsnorm<M::HD, M::AT>(v, nw, M::NORM_EPS, lane);".to_string()
        } else {
            String::new()
        };
        let rope = if !m.rope.halves {
            "    mk::rope_warp_interleaved<M::HD, M::RD, M::AT>(v, cosr, sinr, lane);".to_string()
        } else if m.rope.rotary_dim % 64 == 0 {
            "    mk::rope_warp<M::HD, M::RD, M::AT>(v, cosr, sinr, lane);".to_string()
        } else {
            format!(
                "    // head_dim {} is not a multiple of 64: the rotary partner lives in\n\
                 \x20   // another lane, so the rotation goes through shared memory\n\
                 \x20   mk::rope_warp_smem<M::HD, M::RD, M::AT>(v, cosr, sinr, lane, sh);",
                m.attn.head_dim
            )
        };
        fill(T_HEAD_PREP, &[("QKNORM", qk), ("ROPE", rope)])
    }

    fn t_attn(&self) -> String {
        let m = self.m;
        let cap = |ind: &str| match m.attn.logit_softcap {
            Some(_) => format!("{ind}sd = mk::apply_softcap(sd, M::SOFTCAP);\n"),
            None => String::new(),
        };
        let epi = if self.p.attn_splits == 1 {
            fill(T_ATTN_EPI_LOCAL, &[(
                "SINK",
                if m.attn.sinks { T_ATTN_SINK_LOCAL.to_string() } else { String::new() },
            )])
        } else {
            T_ATTN_EPI_SPLIT.to_string()
        };
        let mut s = fill(T_ATTN, &[
            ("QM", (m.attn.n_heads / m.attn.n_kv_heads).to_string()),
            ("QNORM", if m.attn.q_norm.is_some() { "lw.q_norm".into() } else { "nullptr".into() }),
            ("KNORM", if m.attn.k_norm.is_some() { "lw.k_norm".into() } else { "nullptr".into() }),
            ("SOFTCAP1", cap("            ")),
            ("SOFTCAP2", cap("            ")),
            ("EPILOGUE", epi),
        ]);
        if self.p.attn_splits > 1 {
            s.push_str(&fill(T_ATTN_REDUCE, &[
                ("SINKS", m.attn.sinks.to_string()),
                ("SINKPTR", if m.attn.sinks { "lw.sinks".into() } else { "nullptr".into() }),
            ]));
        }
        s
    }

    fn t_oproj(&self) -> String {
        let m = self.m;
        let mat = self.mat("o", &m.q_attn, "M::QD", m.q_dim(), m.hidden);
        fill(T_OPROJ, &[
            ("R", self.r_of(StageKind::OProj, 1).to_string()),
            ("GEMV", mat.gemv("lw.", "n0", "R", "xs", "acc")),
            ("OBIAS", if m.attn.o_bias { " + lw.o_b[n0+r]".into() } else { String::new() }),
            ("WRITE", if m.post_attn_norm {
                "g.rt.tmp[n0+r] = v;".into()
            } else {
                "g.rt.x[n0+r] = mk::from_f<M::AT>(mk::to_f<M::AT>(g.rt.x[n0+r]) + v);".to_string()
            }),
        ])
    }

    fn t_mlp(&self) -> String {
        let m = self.m;
        let inter = m.ffn.intermediate();
        let r1 = (self.r_of(StageKind::Mlp1, 2).max(2)) & !1;
        let gu = self.mat("gu", &m.q_ffn, "M::H", m.hidden, 2 * inter);
        let dn = self.mat("dn", &m.q_ffn, "M::INTER", inter, m.hidden);
        let mut s = fill(T_MLP1, &[
            ("R", r1.to_string()),
            ("GEMV", gu.gemv("lw.", "row", "R", "xs", "acc")),
            ("ACT", self.act_expr("gv", "uv")),
        ]);
        s.push_str(&fill(T_MLP2, &[
            ("R", self.r_of(StageKind::Mlp2, 1).to_string()),
            ("GEMV", dn.gemv("lw.", "n0", "R", "xs", "acc")),
            ("WRITE2", if m.post_ffn_norm {
                "g.rt.tmp[n0+r] = mk::ActRound<M::AT>::r(acc[r]);".into()
            } else {
                "g.rt.x[n0+r] = mk::from_f<M::AT>(mk::to_f<M::AT>(g.rt.x[n0+r])\n                                                 + mk::ActRound<M::AT>::r(acc[r]));".to_string()
            }),
        ]));
        s
    }

    fn t_moe(&self) -> String {
        let m = self.m;
        let inter = m.ffn.intermediate();
        let cfg = m.moe().unwrap();
        let r1 = (self.r_of(StageKind::Moe1, 2).max(2)) & !1;
        let r2 = self.r_of(StageKind::Moe2, 1);
        let egu = self.mat("egu", &m.q_ffn, "M::H", m.hidden, 2 * inter);
        let edn = self.mat("edn", &m.q_ffn, "M::INTER", inter, m.hidden);
        let mut s = fill(T_ROUTER, &[
            ("NEXP", cfg.n_experts.to_string()),
            ("RBIAS", if cfg.router_bias { " + lw.router_b[e]".into() } else { String::new() }),
            ("SCORE", if cfg.score == RouterScore::Sigmoid { "mk::SCORE_SIGMOID".into() }
                      else { "mk::SCORE_SOFTMAX".to_string() }),
            ("SAT", cfg.softmax_after_topk.to_string()),
            ("NORMK", cfg.norm_topk.to_string()),
            ("CORR", if cfg.router_correction_bias { "lw.router_cb".into() } else { "nullptr".into() }),
        ]);
        s.push_str(&fill(T_MOE1, &[
            ("R", r1.to_string()),
            ("GEMV", egu.gemv("lw.", "row", "R", "xs", "acc")),
            ("EBIAS0", if cfg.expert_bias { "   + lw.egu_b[row + 2*c]".into() } else { String::new() }),
            ("EBIAS1", if cfg.expert_bias { " + lw.egu_b[row + 2*c+1]".into() } else { String::new() }),
            ("ACT", self.act_expr("gv", "uv")),
        ]));
        s.push_str(&fill(T_MOE2, &[
            ("R", r2.to_string()),
            ("TOPK", cfg.top_k.to_string()),
            ("GEMVMULTI", edn.gemv_multi(
                "lw.", "(size_t)seidx[gi*F+t]*M::H + n0", "F", "R", "xp[gi*F+t]", "&acc[gi*F]")),
            ("DBIAS", if cfg.expert_bias {
                " + lw.edn_b[(size_t)seidx[s]*M::H + n0 + r]".into() } else { String::new() }),
        ]));
        s
    }

    fn t_postnorm(&self, name: &str, w: &str) -> String {
        // RmsOnePlus (Gemma) is the only kind that reaches here
        let apply = if self.m.attn_norm.kind == NormKind::RmsOnePlus {
            "mk::ActRound<M::AT>::r((g.rt.tmp[i] * inv) * (1.f + w))"
        } else {
            "mk::ActRound<M::AT>::r(w * mk::ActRound<M::AT>::r(g.rt.tmp[i] * inv))"
        };
        fill(T_POSTNORM, &[("NAME", name.into()), ("W", w.into()), ("APPLY", apply.into())])
    }

    fn t_shared(&self) -> String {
        let m = self.m;
        let si = match &m.ffn { Ffn::Moe { shared_intermediate, .. } => *shared_intermediate, _ => 0 };
        let r1 = (self.r_of(StageKind::SharedMlp1, 2).max(2)) & !1;
        let r2 = self.r_of(StageKind::SharedMlp2, 1);
        let gu = self.mat("sgu", &m.q_ffn, "M::H", m.hidden, 2 * si);
        let dn = self.mat("sdn", &m.q_ffn, "M::SINTER", si, m.hidden);
        let gated = m.weights.contains_key("l0.sgate_w");
        fill(T_SHARED, &[
            ("R1", r1.to_string()),
            ("R2", r2.to_string()),
            ("GEMV1", gu.gemv("lw.", "row", "R", "xs", "acc")),
            ("GEMV2", dn.gemv("lw.", "n0", "R", "xs", "acc")),
            ("ACT", self.act_expr("gv", "uv")),
            ("SGATE", if gated {
                "    // one sigmoid gate for the whole shared expert\n\
                 \x20   {\n\
                 \x20       const float a = mk::gemv_dense_block<M::AT, M::H>(lw.sgate_w, xs, scr);\n\
                 \x20       if (threadIdx.x == 0) g.rt.sgate[0] = 1.f/(1.f + __expf(-a));\n\
                 \x20   }\n".to_string()
            } else { "    (void)scr;\n".to_string() }),
            ("SGATEVAL", if gated { "g.rt.sgate[0]".into() } else { "1.f".to_string() }),
        ])
    }

    fn t_lmhead(&self) -> String {
        let m = self.m;
        let r = self.r_of(StageKind::LmHead, 4);
        let lm = self.mat("__lm", &m.q_lm, "M::H", m.hidden, m.vocab);
        // the LM head lives in Globals, not LayerW, so its pointer is spelled out
        let gemv = lm.gemv("", "n0", "R", "xs", "acc").replace("__lm_w", "g.unembed");
        let cap = match m.logit_softcap {
            Some(_) => "M::LOGIT_CAP*tanhf(mk::ActRound<M::AT>::r(acc[r])/M::LOGIT_CAP)",
            None => "mk::ActRound<M::AT>::r(acc[r])",
        };
        fill(T_LMHEAD, &[("R", r.to_string()), ("GEMV", gemv), ("CAP", cap.into())])
    }

    /// The device call for one planned stage, and whether it ends at a barrier.
    /// EXTENDING: a new stage is one arm here plus its template.
    fn stage_call(k: StageKind) -> &'static str {
        match k {
            StageKind::Embed => "st_embed(g, pos);",
            StageKind::QkvRope => "st_qkv(g, lw, pos, xs, scr);",
            StageKind::Attn => "st_attn(g, lw, L, pos, xs);",
            StageKind::AttnReduce => "st_attn_reduce(g, lw, xs);",
            StageKind::OProj => "st_oproj(g, lw, xs);",
            StageKind::PostAttnNorm => "st_post_attn(g, lw, scr);",
            StageKind::PostFfnNorm => "st_post_ffn(g, lw, scr);",
            StageKind::Router => "st_router(g, lw, xs, scr);",
            StageKind::TopK => "st_topk(g, seidx, sew);",
            StageKind::SharedMlp1 => "st_shmlp1(g, lw, xs, scr);",
            StageKind::SharedMlp2 => "st_shmlp2(g, lw, xs);",
            StageKind::Moe1 => "st_moe1(g, lw, xs, seidx);",
            StageKind::Moe2 => "st_moe2(g, lw, xs, seidx, sew);",
            StageKind::Mlp1 => "st_mlp1(g, lw, xs, scr);",
            StageKind::Mlp2 => "st_mlp2(g, lw, xs);",
            StageKind::LmHead => "st_lmhead(g, xs, scr);",
        }
    }

    pub fn stages_cuh(&self) -> String {
        let mut s = format!(
            "// generated by mkc {} -- do not edit\n#pragma once\n\
             #include \"kernel.h\"\n#include \"mk/common.cuh\"\n#include \"mk/gemv.cuh\"\n\
             #include \"mk/norm.cuh\"\n#include \"mk/rope.cuh\"\n#include \"mk/attn.cuh\"\n\
             #include \"mk/moe.cuh\"\n#include <cooperative_groups.h>\n\
             namespace cg = cooperative_groups;\n\n\
             // LayerNorm needs a mean, which only the two-pass form computes.\n\
             #define MK_NORM {}\n",
            crate::VERSION,
            if self.m.attn_norm.kind == NormKind::Layer { "mk::norm_to_smem_slow" }
            else { "mk::norm_to_smem" }
        );
        s.push_str(&self.t_embed());
        s.push_str(&self.t_qkv());
        s.push_str(&self.t_head_prep());
        s.push_str(&self.t_attn());
        s.push_str(&self.t_oproj());
        if self.m.post_attn_norm { s.push_str(&self.t_postnorm("post_attn", "post_attn_norm")); }
        if self.p.layer.iter().any(|s| s.kind == StageKind::SharedMlp1) {
            s.push_str(&self.t_shared());
        }
        s.push_str(&if self.m.is_moe() { self.t_moe() } else { self.t_mlp() });
        if self.m.post_ffn_norm { s.push_str(&self.t_postnorm("post_ffn", "post_ffn_norm")); }
        s.push_str(&self.t_lmhead());
        s
    }

    pub fn megakernel_cu(&self) -> String {
        // the per-layer body is the planner's stage list, in order
        let mut body = String::new();
        for st in &self.p.layer {
            let call = Self::stage_call(st.kind);
            if st.barrier_after {
                let _ = writeln!(body, "        {:<44} grid.sync(); MK_MARK();", call);
            } else {
                let _ = writeln!(body, "        {:<44} // block-local: no barrier", call);
            }
        }
        format!(
            "// generated by mkc {} -- do not edit\n#include \"stages.cuh\"\n{}",
            crate::VERSION,
            fill(T_KERNEL, &[
                ("NL", self.m.n_layers.to_string()),
                ("MINB", match self.p.launch_min_blocks {
                    Some(b) => format!(", {}", b),
                    None => String::new(),
                }),
                ("BPSDESC", match self.p.launch_min_blocks {
                    Some(b) => format!("{} blocks/SM", b),
                    None => "whatever occupancy the assembler chooses".to_string(),
                }),
                ("REGS", self.p.predicted_regs.to_string()),
                ("BODY", body),
                ("SAMPLE", T_SAMPLE.to_string()),
            ])
        )
    }

    /// One plain kernel per stage.  A megakernel is opaque to `ncu` -- you cannot
    /// point a profiler at a stage inside one -- so the compiler also emits each
    /// stage as an ordinary launch.  They are what the cost model is validated
    /// against, and what `ncu` answers "bandwidth or issue bound?" for.
    pub fn stages_cu(&self) -> String {
        let mut names: Vec<StageKind> = vec![StageKind::Embed];
        names.extend(self.p.layer.iter().map(|s| s.kind).filter(|k| *k != StageKind::TopK));
        names.push(StageKind::LmHead);
        let mut s = format!(
            "// generated by mkc {} -- standalone stage kernels, for profiling\n\
             #include \"stages.cuh\"\n\n",
            crate::VERSION
        );
        for k in &names {
            // moe1 and moe2 need the router's smem state and the expert choice
            let pre = match k {
                StageKind::Moe1 => "st_router(g, lw, xs, scr); __syncthreads(); st_topk(g, seidx, sew); ",
                StageKind::Moe2 => "st_topk(g, seidx, sew); ",
                _ => "",
            };
            let _ = write!(s,
"extern \"C\" __global__ __launch_bounds__(M::NT) void kst_{name}(Globals g, int L) {{
    extern __shared__ char smem[];
    float* xs    = reinterpret_cast<float*>(smem);
    float* scr   = xs + ARENA_FLOATS;
    int*   seidx = reinterpret_cast<int*>(scr + M::NW);
    float* sew   = reinterpret_cast<float*>(seidx + (M::TOPK ? M::TOPK : 1));
    const int pos = *g.dpos;
    const LayerW& lw = g.layers[L];
    {pre}{call}
    (void)xs; (void)scr; (void)seidx; (void)sew; (void)lw; (void)pos;
}}
", name = k.short(), pre = pre, call = Self::stage_call(*k));
        }
        let _ = writeln!(s, "extern \"C\" const char* const MK_STAGE_NAMES[] = {{");
        for k in &names { let _ = writeln!(s, "    \"{}\",", k.short()); }
        let _ = writeln!(s, "    nullptr }};");
        let _ = writeln!(s, "extern \"C\" void* const MK_STAGE_FNS[] = {{");
        for k in &names { let _ = writeln!(s, "    (void*)kst_{},", k.short()); }
        let _ = writeln!(s, "    nullptr }};");
        s
    }
}
