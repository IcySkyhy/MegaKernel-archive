// Architecture-invariant transcendentals: every operation is an IEEE 754
// primitive with explicit rounding intrinsics (no per-target
// re-contraction), so results are bit-identical on every CUDA
// architecture; replaces libdevice expf/cosf/sinf/powf/rsqrtf, which are
// not specified to match across architectures. Hot functions run in
// double-float (Dekker/Knuth fp32 pairs, ~49-bit significand) on the
// full-rate fp32 fma units instead of 1/64-rate consumer fp64.
#pragma once

namespace detm {

// ---- double-float (fp32 pair) primitives; value = h + l ----------------
struct df32 {
  float h, l;
};

__device__ __forceinline__ df32 dtwo_sum(float a, float b) {
  const float s = __fadd_rn(a, b);
  const float bb = __fsub_rn(s, a);
  const float e =
      __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
  return {s, e};
}

__device__ __forceinline__ df32 dquick_sum(float a, float b) {  // |a|>=|b|
  const float s = __fadd_rn(a, b);
  return {s, __fsub_rn(b, __fsub_rn(s, a))};
}

__device__ __forceinline__ df32 dtwo_prod(float a, float b) {
  const float p = __fmul_rn(a, b);
  return {p, __fmaf_rn(a, b, -p)};
}

__device__ __forceinline__ df32 df_mul(df32 a, df32 b) {
  df32 p = dtwo_prod(a.h, b.h);
  p.l = __fmaf_rn(a.h, b.l, __fmaf_rn(a.l, b.h, p.l));
  return dquick_sum(p.h, p.l);
}

__device__ __forceinline__ df32 df_add(df32 a, df32 b) {
  const df32 s = dtwo_sum(a.h, b.h);
  const float t = __fadd_rn(__fadd_rn(a.l, b.l), s.l);
  return dquick_sum(s.h, t);
}

__device__ __forceinline__ double exp_d(double x) {
  // Cody-Waite reduction: x = k*ln2 + r, |r| <= ln2/2, exp(r) by Taylor.
  const double INV_LN2 = 1.4426950408889634074;
  const double LN2_HI = 6.9314718036912381649e-01;
  const double LN2_LO = 1.9082149292705877000e-10;
  const double kd = x * INV_LN2;
  const int k = (int)(kd + (kd >= 0.0 ? 0.5 : -0.5));
  double r = fma((double)-k, LN2_HI, x);
  r = fma((double)-k, LN2_LO, r);
  // Taylor to degree 12: term 13 is r^13/13! ~ 2e-16 relative at |r| <=
  // ln2/2, ample for fp32 use.
  double p = 2.0876756987868099e-09;            // 1/12!
  p = fma(p, r, 2.5052108385441719e-08);        // 1/11!
  p = fma(p, r, 2.7557319223985893e-07);        // 1/10!
  p = fma(p, r, 2.7557319223985888e-06);        // 1/9!
  p = fma(p, r, 2.4801587301587302e-05);        // 1/8!
  p = fma(p, r, 1.9841269841269841e-04);        // 1/7!
  p = fma(p, r, 1.3888888888888889e-03);        // 1/6!
  p = fma(p, r, 8.3333333333333333e-03);        // 1/5!
  p = fma(p, r, 4.1666666666666664e-02);        // 1/4!
  p = fma(p, r, 1.6666666666666666e-01);        // 1/3!
  p = fma(p, r, 5.0e-01);                       // 1/2!
  p = fma(p, r, 1.0);                           // 1/1!
  p = fma(p, r, 1.0);                           // 1/0!
  return ldexp(p, k);                           // exact scaling by 2^k
}

// exp in fp32 double-float: three-term Cody-Waite ln2 split (k*A exact
// for |k| <= 152; A+B+C = ln2 to the last double bit), fp32 Horner for
// degrees 12..6 (tail below 2^-30 relative), double-float Horner for
// 5..0, one rounding at the collapse, exact power-of-two scaling.
__device__ __forceinline__ float expf_det(float xf) {
  if (xf < -104.0f) return 0.f;
  if (xf > 88.8f) return __int_as_float(0x7f800000);
  const float INV_LN2 = 0x1.715476p+0f;
  const float LN2_A = 0x1.62ep-1f;
  const float LN2_B = 0x1.0bfbe8p-15f;
  const float LN2_C = 0x1.cf78p-40f;
  const float kf = rintf(__fmul_rn(xf, INV_LN2));
  const int k = (int)kf;
  // r = x - k*ln2: the A term cancels exactly, B through an exact product
  // pair, C folded into the low word.
  const float t1 = __fmaf_rn(-kf, LN2_A, xf);
  const df32 pb = dtwo_prod(kf, LN2_B);
  df32 r = dtwo_sum(t1, -pb.h);
  r.l = __fmaf_rn(-kf, LN2_C, __fsub_rn(r.l, pb.l));
  r = dquick_sum(r.h, r.l);

  // fp32 tail, degrees 12..6
  float q = 0x1.1eed8ep-29f;                 // 1/12!
  q = __fmaf_rn(q, r.h, 0x1.ae6456p-26f);    // 1/11!
  q = __fmaf_rn(q, r.h, 0x1.27e4fcp-22f);    // 1/10!
  q = __fmaf_rn(q, r.h, 0x1.71de3ap-19f);    // 1/9!
  q = __fmaf_rn(q, r.h, 0x1.a01a02p-16f);    // 1/8!
  q = __fmaf_rn(q, r.h, 0x1.a01a02p-13f);    // 1/7!
  q = __fmaf_rn(q, r.h, 0x1.6c16c2p-10f);    // 1/6!
  // double-float head, degrees 5..0
  df32 p = {q, 0.f};
  p = df_add(df_mul(p, r), {0x1.111112p-7f, -0x1.dddddep-32f});   // 1/5!
  p = df_add(df_mul(p, r), {0x1.555556p-5f, -0x1.555556p-30f});   // 1/4!
  p = df_add(df_mul(p, r), {0x1.555556p-3f, -0x1.555556p-28f});   // 1/3!
  p = df_add(df_mul(p, r), {0x1p-1f, 0.f});                       // 1/2!
  p = df_add(df_mul(p, r), {0x1p+0f, 0.f});                       // 1/1!
  p = df_add(df_mul(p, r), {0x1p+0f, 0.f});                       // 1/0!

  const float v = __fadd_rn(p.h, p.l);  // the single rounding
  const int k1 = k / 2, k2 = k - k1;
  const float s1 = __int_as_float((unsigned)(k1 + 127) << 23);
  const float s2 = __int_as_float((unsigned)(k2 + 127) << 23);
  return __fmul_rn(__fmul_rn(v, s1), s2);
}

__device__ __forceinline__ double log_d(double x) {
  // x = m * 2^e with m in [sqrt(1/2), sqrt(2)); log(m) by atanh series.
  int e;
  double m = frexp(x, &e);                      // exact decomposition
  if (m < 0.70710678118654752440) { m = m + m; e -= 1; }
  const double s = (m - 1.0) / (m + 1.0);
  const double z = s * s;
  double p = 1.0 / 15.0;
  p = fma(p, z, 1.0 / 13.0);
  p = fma(p, z, 1.0 / 11.0);
  p = fma(p, z, 1.0 / 9.0);
  p = fma(p, z, 1.0 / 7.0);
  p = fma(p, z, 1.0 / 5.0);
  p = fma(p, z, 1.0 / 3.0);
  p = fma(p, z, 1.0);
  const double lm = 2.0 * (s * p);
  const double LN2_HI = 6.9314718036912381649e-01;
  const double LN2_LO = 1.9082149292705877000e-10;
  return fma((double)e, LN2_HI, fma((double)e, LN2_LO, lm));
}

// theta^(-2d/D) for rope, matching 1/powf(theta, 2d/D) semantics.
__device__ __forceinline__ float inv_freq_det(float theta, int d, int D) {
  const double y = -((double)(2 * d)) / (double)D;
  return (float)exp_d(y * log_d((double)theta));
}

__device__ __forceinline__ float tanhf_det(float x) {
  // (e^2x - 1) / (e^2x + 1) over IEEE primitives; saturates where
  // fp32 tanh is exactly +-1
  if (x > 9.02f) return 1.f;
  if (x < -9.02f) return -1.f;
  const float t = expf_det(__fmul_rn(2.f, x));
  return __fdiv_rn(__fsub_rn(t, 1.f), __fadd_rn(t, 1.f));
}

__device__ __forceinline__ float rsqrtf_det(float x) {
  return (float)(1.0 / sqrt((double)x));        // IEEE sqrt and divide
}

__device__ __forceinline__ void sincosf_det(float af, float* so, float* co) {
  // Reduction mod pi/2, two-term Cody-Waite in fp64: for |a| < 2^20 the
  // reduction error is < 1e-10. Polynomials run in double-float on the
  // fp32 units, cutting the fp64 chain from ~18 operations to 4.
  const double x = (double)af;
  const double TWO_OVER_PI = 0.63661977236758134308;
  const double PIO2_HI = 1.5707963267948965580e+00;
  const double PIO2_LO = 6.1232339957367660359e-17;
  const double kd = x * TWO_OVER_PI;
  const long long k = (long long)(kd + (kd >= 0.0 ? 0.5 : -0.5));
  double rd = fma((double)-k, PIO2_HI, x);
  rd = fma((double)-k, PIO2_LO, rd);
  const float rh = (float)rd;
  const float rl = (float)(rd - (double)rh);
  const df32 r = {rh, rl};
  const df32 z = df_mul(r, r);

  // sin(r)/r as a polynomial in z: fp32 tail (degrees 6..3), df head
  float sq = 0x1.612462p-33f;                 // 1/13!
  sq = __fmaf_rn(sq, z.h, -0x1.ae6456p-26f);  // -1/11!
  sq = __fmaf_rn(sq, z.h, 0x1.71de3ap-19f);   // 1/9!
  sq = __fmaf_rn(sq, z.h, -0x1.a01a02p-13f);  // -1/7!
  df32 sp = {sq, 0.f};
  sp = df_add(df_mul(sp, z), {0x1.111112p-7f, -0x1.dddddep-32f});  // 1/5!
  sp = df_add(df_mul(sp, z), {-0x1.555556p-3f, 0x1.555556p-28f});  // -1/3!
  sp = df_add(df_mul(sp, z), {0x1p+0f, 0.f});
  sp = df_mul(sp, r);
  const float sinr = __fadd_rn(sp.h, sp.l);

  // cos(r) in z: fp32 tail (degrees 7..3), df head
  float cq = -0x1.93974ap-37f;                // -1/14!
  cq = __fmaf_rn(cq, z.h, 0x1.1eed8ep-29f);   // 1/12!
  cq = __fmaf_rn(cq, z.h, -0x1.27e4fcp-22f);  // -1/10!
  cq = __fmaf_rn(cq, z.h, 0x1.a01a02p-16f);   // 1/8!
  cq = __fmaf_rn(cq, z.h, -0x1.6c16c2p-10f);  // -1/6!
  df32 cp = {cq, 0.f};
  cp = df_add(df_mul(cp, z), {0x1.555556p-5f, -0x1.555556p-30f});  // 1/4!
  cp = df_add(df_mul(cp, z), {-0x1p-1f, 0.f});                     // -1/2!
  cp = df_add(df_mul(cp, z), {0x1p+0f, 0.f});
  const float cosr = __fadd_rn(cp.h, cp.l);

  float s, c;
  switch ((int)(k & 3)) {
    case 0: s = sinr;  c = cosr;  break;
    case 1: s = cosr;  c = -sinr; break;
    case 2: s = -sinr; c = -cosr; break;
    default: s = -cosr; c = sinr; break;
  }
  *so = s;
  *co = c;
}

}  // namespace detm
