#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M14 GDN layer kernel 数值交叉校验（numpy **独立**实现，消费 M14_DUMP=1 的 dump）

用法（在 dump 目录里运行）：
    M14_DUMP=1 <build>/m14_gdn_layer            # 落盘：m14_params.txt + m14_param_*.bin
                                                #       + <tag>.txt / <tag>_ws.bin / <tag>_{cs,ssm}_out.bin
    /usr/local/python3.12.13/bin/python3 check_ref.py [dump_dir]     # 默认 '.'

与 kernel 内建 C++ 判据的关系：本脚本不共用任何 C++ 代码，用 numpy 从 **dump 出的原始字节**
独立重算 GDN 链（S1 Add+RMSNorm#1 → S2 in_proj bf16 GEMM → S3 conv+SiLU/l2norm/gating →
S4 递推 → S5 RMSNormGated → S6 out_proj → S7 Add+RMSNorm#2），并按与 host 相同的判据口径比对。

## manifest 契约（host H_DumpStep / H_DumpParams 写出）

m14_params.txt：共享参数/输入（固定文件名，只写一次）
    tensor=<name> file=<file> dtype=<bf16|f32> shape=<d0>[,<d1>...]
    names：w_in[16480,2560] w_out[2560,6144] conv_w[4,10240] conv_bias[10240] alog[64] dtbias[64]
           gamma1[2560] gamma2[2560] gamma_g[128] ssm_state_init[48,128,128] conv_state_init[3,10240]
           x_in_s0[2560] res_in_s0[2560] x_in_s1[2560] res_in_s1[2560] x_qkvzba_host[16480]

<tag>.txt：一次 kernel 启动后的现场
    tag=<tag> case=<case> step=<n> m=1 slice_mode=<0|1> stage_limit=<n> ws_bytes=<n>
    tensor=<name> ws_off=<byteoffset> bytes=<n> dtype=<...> shape=<...>   # <tag>_ws.bin 内的切片（行 0）
    tensor=conv_state file=<tag>_cs_out.bin ...   # 常驻 state（in-place 更新后）
    tensor=ssm_state  file=<tag>_ssm_out.bin ...
    workspace 张量：x_norm res1 qkvzba q k v g beta o opin opout y_final res2
    （slice_mode=1 时 qkvzba 指向 m14_param_xqkvzba.bin——kernel 的 S3 直接消费该 host 缓冲）

## 判据（与 host 内建判据同口径；M62 按「参考的输入从哪来」重命名并分族）

**族 H「host 全链」——参考的输入全部是 host 声明量**（dump 的 params + host 侧公式重算），不含任何
设备字节。覆盖：S1（fp32 逐位 + x_norm bf16 网格）与 S2（bf16 网格 ≤1 ulp），即「声明输入 → S2 出口」；
`hostchain` 用例（slice_mode=2）另把 **S3→S7 也放在 host 声明输入之下跑完**（参考在每个 bf16 出口都用
**自己**的舍入值续算），覆盖「S2 出口 → S7」这条组合链。

**族 D「分段（设备锚）」——S3–S7 的输入取设备自己的 bf16 出口字节**（qkvzba / opin / opout），段间由
参考自携带 fp32 状态。它覆盖「给定设备上游字节，本段算得对不对」，**不是**端到端。自指形态 = **S1**
（设备产物入参）——其合法性依赖「被共享的那个出口字节自身已被一条外部规则钉住的判据覆盖」：
* S2 出口（`qkvzba`）：满足 —— 族 H 的 `①S2` 网格判据直接判它；
* S5/S6 出口（`opin`/`opout`）：**在 `chain` 这一份 dump 之内**没有任何 host 输入判据覆盖它们（族 D
  的每条都 re-anchor）⇒ 这两处的 S1 是**未被覆盖的**自指；**跨用例看**，`hostchain` 档的 `①S5`/`①S6`/
  `①S7` 正是覆盖这两个出口的 host 声明输入判据（但那是**另一次运行**，故不能给 `chain` 这一份 dump 免票）。
  （M62 r1 P2-2 订正：原句写成「S5/S6 出口不满足」而未限定「同一 dump 内」，与 §5.2 及 hostchain 的实际
  覆盖自相矛盾。）
> 注意：M59 审计把本文件所有标「全链」的行都记成 S1，**对 S1/S2 与 slice 用例并不成立**（它们的参考
> 输入就是 host 声明量）。M62 已改为按上面的家族显式命名，不再用「全链」这个会误导的词。

**族 P「传播（报告项，不判定）」——纯参考链（无锚点、参考自舍入）与设备的偏差**。**不可复现**：参考
缺设备 S2/S6 的 fp32 L0C 累加值（实测 2/16480 个 S2 出口元素差 1 bf16 ulp），而 conv 段近相消把这点
差放大（实测 y_final 到 128 bf16 ulp）。故只报数不判定；缺的那个输入与它挡住的错误类别见 README §5.2。

判据口径：

* bf16 出口（x_norm/qkvzba/opin/opout/y_final）：与参考的 bf16 位型距离 ≤1 ulp（另报逐位一致占比）；
* fp32 出口（res1/res2/q/k/v/g/β/o/ssm_state）：|got − exp| ≤ 1e-5·|exp| + 1e-6（另报「最差容差占用」）；
* conv_state：**位级一致**（原位更新是纯 bf16 搬移），另做「new[0]=old[1]/new[1]=old[2]/new[2]=x」自洽性检查；
* ssm_state：除容差判据外，另**报告** |exp|≥1e-6 子集上的位级一致率 / ≤1ulp 占比（残差来自 fp32 累加次序与
  e^g 的 exp 近似，host 参考用 float64 作 oracle 无法逐位复现——README §5 有说明）。

## 外部 pin（**规则**的出处；不是设备实现）

| 段 | 规则 | 外部 pin（`文件:符号`） |
|---|---|---|
| S1/S7 | Add+RMSNorm（fp32 残差；NR rsqrt） | `ops-nn/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h`::`CalculateXAdd` / `CalculateSquareReduceSum` / `ComputeRstdNewtonRaphson` / `CalculateY` |
| S2/S6 | bf16 GEMM：`C = round_bf16(A·Wᵀ)`，**舍入规则 = RNE（CAST_RINT）** | **规则源（仓外官方 API 文档）**：`asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.md` §`quantPre` 的 `QuantMode_t::F322BF16 // Float32_2_BFloat16，cast mode 为 CAST_RINT 模式`（`DataCopy_L0CToGM.md` 同表）；**官方算子的同一决策点**：`ops-nn/conv/common/op_kernel/arch35/conv_instr_impl.h`::`GetQuantPreFp32`（`OutputT=bfloat16_t ⇒ QuantMode_t::F322BF16`）；**本仓实现触点（同源转录，不是规则源）**：`m14_gdn_layer.asc`::`Cube::Bf16Gemm::CopyOut` 的 `fp.quantPre = QuantMode_t::F322BF16`；**结构样例（教学样例，非算子规格）**：`asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_basic_api_high_performance/mmad.asc`::`KernelMmad`（`AscendC::Mmad` / `AscendC::Fixpipe`） |
| S3 | conv1d(K=4)+SiLU；q/k l2norm（仅 q ×1/√128）；g/β | `ops-transformer/mamba/causal_conv1d`（arch35 环形 conv）；`vllm-ascend/vllm_ascend/ops/triton/fla/sigmoid_gating.py`（softplus/gating）；`vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`（q/k l2norm 与 q 的 1/√128） |
| S4 | 递推 decay→delta→outer→matvec | `ops-transformer/attention/recurrent_gated_delta_rule`（arch35 `vf_vec_mul_mat.h` / `vf_outer_add.h`） |
| S5 | per-head RMSNormGated（`norm_before_gate=True` + sigmoid） | `vllm/vllm/model_executor/layers/layernorm.py::RMSNormGated`；`output_gate_type="sigmoid"`（checkpoint `config.json`） |

**「同源转录」（correlated transcription）声明**：本脚本的 numpy 参考与 `m14_gdn_layer.asc` 的 host C++
参考（`H_RefNorm`/`H_RefGemm`/`H_RefProlog`/`H_RefRecur`/`H_RefGated`）**是同一作者按同一份读法的两次
转写** —— 两条实现、**不是两个独立来源**（与 donor 的关系同理）。⇒ 两者一致只排除「转写笔误」，
**不构成规则级的独立见证**；规则级的独立性由上面那列外部 pin 承担。

## 退出码与覆盖范围（三态）

`0` = 比过且判定项全 PASS；`1` = 比过且有判定项 FAIL；`2` = **没得比**（输入缺失，**或本批 dump 一条
判定项都没产出**——两种都打印 `RESULT: SKIPPED` / `NO-CRITERIA`，绝不发合格证）。汇总把 **判定项 / 报告项 / 未覆盖（跳过）** 分三栏打印，三栏都由
**同一份门控条件**产出（不是手写数字）。另跑 `咬合自检`（负向对照）：①篡改 1 个 bf16 位 ⇒ 网格判据必须
FAIL（证明判据是活的）；②把全体元素统一 ±1 ulp ⇒ 网格判据**仍 PASS**，定量证明它对「≤1 ulp 的规则级
错误」无感（这是族 D 咬不住的那一类，见 README §5.2）。

精度约定：fp32 段的标量乘法按 float32 逐次舍入（与设备逐条向量指令同序），归约（平方和/l2norm/
递推内积）用 float64 作 oracle（差 ~1e-7，远小于 1e-5 容差与 bf16 ulp）。
"""
import glob
import os
import sys

import numpy as np

HIDDEN = 2560
IN_N = 16480
Q_DIM = 2048
K_DIM = 2048
V_DIM = 6144
Z_DIM = 6144
GATE = 48
Q_OFF = 0
K_OFF = 2048
V_OFF = 4096
Z_OFF = 10240
B_OFF = 16384
A_OFF = 16432
CH = 10240
KW = 4
ST = 3
HEADS = 48
NK = 16
HEAD = 128
VGROUP = 3
BLK = 128            # prolog block = 128 通道
NBLK = CH // BLK     # 80
RTOL = 1e-5
ATOL = 1e-6
QSCALE = np.float32(0.08838834764831845)   # 1/sqrt(128)


# ------------------------------------------------------------------
# dump 读取
# ------------------------------------------------------------------

def parse_manifest(path):
    hdr, tens = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0].startswith('tensor='):
                name = parts[0][len('tensor='):]
                d = {}
                for p in parts[1:]:
                    k, v = p.split('=', 1)
                    d[k] = v
                tens[name] = d
            else:
                k, v = line.split('=', 1)
                hdr[k] = v
    return hdr, tens


class Dump:
    def __init__(self, root, path):
        self.root = root
        self.hdr, self.tens = parse_manifest(path)
        self.cache = {}
        # ws 切片张量（只有 ws_off、没有 file=）落在本 tag 的 workspace 文件里
        self.wsfile = os.path.splitext(os.path.basename(path))[0] + '_ws.bin'

    def raw(self, name):
        if name not in self.tens:
            raise KeyError(name)
        if name in self.cache:
            return self.cache[name]
        d = self.tens[name]
        dt = {'bf16': '<u2', 'f32': '<f4'}[d['dtype']]
        shape = tuple(int(x) for x in d['shape'].split(','))
        n = int(np.prod(shape))
        arr = np.fromfile(os.path.join(self.root, d.get('file', self.wsfile)), dtype=dt)
        if 'ws_off' in d:
            arr = arr[int(d['ws_off']) // np.dtype(dt).itemsize:]
        arr = arr[:n].reshape(shape)
        self.cache[name] = arr
        return arr

    def f32(self, name):
        a = self.raw(name)
        return bf16_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)

    def __getitem__(self, name):
        return self.raw(name)


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return r.astype(np.uint16)


def ulp_dist_bf16(a, b):
    """bf16 位型距离（符号-幅值 → 单调整数）。"""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    ia = np.where(a & 0x8000, -(a & 0x7FFF), a)
    ib = np.where(b & 0x8000, -(b & 0x7FFF), b)
    return np.abs(ia - ib)


# ------------------------------------------------------------------
# 判据（返回 bool）+ 计数
# ------------------------------------------------------------------

class Stat:
    """判定项 / 报告项 / 未覆盖（跳过）三态计数。

    三态的**计数与其门控条件同源**：`gate()` 既做门控又记「未覆盖」，不在别处手写数字。
    """

    def __init__(self):
        self.n_judge = 0
        self.n_pass = 0
        self.n_report = 0
        self.n_skipped = 0
        self.skipped = []

    def judge(self, tag, ok, detail):
        self.n_judge += 1
        if ok:
            self.n_pass += 1
        print('[m14-ref] %-36s %s %s' % (tag, 'PASS' if ok else 'FAIL', detail))
        return ok

    def report(self, tag, detail):
        self.n_report += 1
        print('[m14-ref] %-36s (报告) %s' % (tag, detail))

    def skip(self, tag, reason):
        self.n_skipped += 1
        self.skipped.append('%s — %s' % (tag, reason))

    def gate(self, cond, tag, reason):
        """门控：cond 为假 ⇒ 记为「未覆盖」并返回 False（判定项不产出）。"""
        if not cond:
            self.skip(tag, reason)
        return bool(cond)

    def banner(self):
        """收尾横幅。**判定项 0 ⇒ 不发合格证**（返回 False，调用方据此走 rc=2）。

        「通过了」与「根本没检查」必须在表面就分辨得出（tower 规则 2026-09-26；M62 r1 P2-3）。
        """
        if self.n_judge == 0:
            print('[m14-ref] ===== NO-CRITERIA（判定项 0：无判定力，不发合格证）=====')
        else:
            print('[m14-ref] ===== %s（判定项 %d）=====' %
                  ('ALL PASS' if self.n_pass == self.n_judge else 'FAILURES PRESENT', self.n_judge))
        print('[m14-ref] 覆盖三态：判定项 %d（PASS %d / FAIL %d）| 报告项 %d | 未覆盖(跳过) %d' %
              (self.n_judge, self.n_pass, self.n_judge - self.n_pass, self.n_report, self.n_skipped))
        for s in self.skipped:
            print('[m14-ref]   未覆盖：%s' % s)
        return self.n_judge > 0 and self.n_pass == self.n_judge


def cmp_bf16_grid(st, tag, got, exp, note=''):
    got = np.asarray(got, dtype=np.uint16).reshape(-1)
    exp = np.asarray(exp, dtype=np.uint16).reshape(-1)
    d = ulp_dist_bf16(got, exp)
    exact = int(np.count_nonzero(d == 0))
    bad = int(np.count_nonzero(d > 1))
    worst = int(d.max()) if d.size else 0
    ok = (bad == 0)
    return st.judge(tag, ok, 'bf16 网格 ≤1ulp: maxUlp %d, bit 一致 %.4f%%, 超限 %d/%d %s' %
                    (worst, 100.0 * exact / d.size, bad, d.size, note))


def cmp_f32_rel(st, tag, got, exp, note=''):
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    exp = np.asarray(exp, dtype=np.float64).reshape(-1)
    a = np.abs(got - exp)
    util = a / (RTOL * np.abs(exp) + ATOL)
    worst = float(util.max()) if util.size else 0.0
    ok = bool(np.all(util <= 1.0))
    return st.judge(tag, ok, 'rel 1e-5+1e-6: 最差容差占用 %.3f, maxAbs %.3e %s' % (worst, float(a.max()), note))


def cmp_f32_bit(st, tag, got, exp):
    got = np.asarray(got, dtype=np.float32).reshape(-1)
    exp = np.asarray(exp, dtype=np.float32).reshape(-1)
    bad = int(np.count_nonzero(got != exp))
    return st.judge(tag, bad == 0, 'fp32 逐位一致: 不同 %d/%d' % (bad, got.size))


def cmp_u16_bit(st, tag, got, exp):
    got = np.asarray(got, dtype=np.uint16).reshape(-1)
    exp = np.asarray(exp, dtype=np.uint16).reshape(-1)
    bad = int(np.count_nonzero(got != exp))
    return st.judge(tag, bad == 0, 'bf16 位级一致: 不同 %d/%d' % (bad, got.size))


def report_state_bit(st, tag, got, exp):
    got = np.asarray(got, dtype=np.float32).reshape(-1)
    exp = np.asarray(exp, dtype=np.float64).reshape(-1)
    gb = got.view(np.uint32)
    eb = exp.astype(np.float32).view(np.uint32)
    d = np.abs(np.where(gb & 0x80000000, -(gb & 0x7FFFFFFF).astype(np.int64), gb.astype(np.int64)) -
               np.where(eb & 0x80000000, -(eb & 0x7FFFFFFF).astype(np.int64), eb.astype(np.int64)))
    big = np.abs(exp) >= 1e-6
    nb = int(np.count_nonzero(big))
    exact = int(np.count_nonzero(d == 0))
    bexact = int(np.count_nonzero((d == 0) & big))
    within1 = int(np.count_nonzero((d <= 1) & big))
    st.report(tag, '逐位一致 %.4f%%; |exp|>=1e-6 子集: 位级一致 %.4f%% / ≤1ulp %.4f%% / max ulp %d' %
              (100.0 * exact / d.size, 100.0 * bexact / max(nb, 1), 100.0 * within1 / max(nb, 1),
               int(d.max()) if d.size else 0))


# ------------------------------------------------------------------
# 参考实现（numpy 独立；见文件头精度约定）
# ------------------------------------------------------------------

def ref_norm(x_bits, res_bits, res_f32, gamma_bits, rows=1):
    """Add + RMSNorm：返回 (resOut fp32[N], y bf16[N])（N = rows*HIDDEN，m=1 只用行 0）。"""
    xs = bf16_to_f32(x_bits).astype(np.float32).reshape(rows, HIDDEN).copy()
    if res_bits is not None:
        xs += bf16_to_f32(res_bits).astype(np.float32).reshape(rows, HIDDEN)
    else:
        xs += np.asarray(res_f32, dtype=np.float32).reshape(rows, HIDDEN)
    ss = np.sum(xs.astype(np.float64) ** 2, axis=1) / HIDDEN
    rstd = (1.0 / np.sqrt(ss + 1e-6)).astype(np.float32)
    g = bf16_to_f32(gamma_bits).astype(np.float32)
    y = (xs * rstd[:, None]).astype(np.float32) * g[None, :]
    return xs.astype(np.float32).reshape(-1), f32_to_bf16(y).reshape(-1)


def ref_gemm(a_bits, w_bits, k, n, m=1, chunk=2048):
    """C = round_bf16(A·Wᵀ)：float64 累加（oracle），W 为 [N,K] 行主序。"""
    a = bf16_to_f32(a_bits).astype(np.float64).reshape(m, k)
    w = bf16_to_f32(w_bits).reshape(n, k)
    out = np.empty((m, n), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[:, s:e] = (a @ w[s:e].astype(np.float64).T).astype(np.float32)
    return f32_to_bf16(out).reshape(-1)


def ref_prolog(xq_bits, cs_bits, w_bits, bias_bits, alog, dtbias):
    """S3：conv1d(K=4)+bias+SiLU → q/k l2norm（q ×1/√128）→ g/β；conv_state 原位左移。
    返回 dict(q,k,v,g,beta,cs)。"""
    xq = bf16_to_f32(xq_bits).astype(np.float32).reshape(-1)
    cs = bf16_to_f32(cs_bits).astype(np.float32).reshape(ST, CH)
    w = bf16_to_f32(w_bits).astype(np.float64).reshape(KW, CH)
    bias = bf16_to_f32(bias_bits).astype(np.float64)
    q = np.zeros((NK, HEAD), dtype=np.float32)
    kk = np.zeros((NK, HEAD), dtype=np.float32)
    v = np.zeros((HEADS, HEAD), dtype=np.float32)
    g = np.zeros((HEADS, 8), dtype=np.float32)
    beta = np.zeros((HEADS, 8), dtype=np.float32)
    for b in range(NBLK):
        c0 = b * BLK
        acc = bias[c0:c0 + BLK].copy()
        acc += w[0, c0:c0 + BLK] * cs[0, c0:c0 + BLK].astype(np.float64)
        acc += w[1, c0:c0 + BLK] * cs[1, c0:c0 + BLK].astype(np.float64)
        acc += w[2, c0:c0 + BLK] * cs[2, c0:c0 + BLK].astype(np.float64)
        acc += w[3, c0:c0 + BLK] * xq[c0:c0 + BLK].astype(np.float64)
        y = acc.astype(np.float32)
        s = (y / (1.0 + np.exp(-y))).astype(np.float32)     # SiLU（fp32）
        if b < 16:
            r = np.float32(1.0 / np.sqrt(np.sum(s.astype(np.float64) ** 2) + 1e-6)) * QSCALE
            q[b] = (s * r).astype(np.float32)
        elif b < 32:
            r = np.float32(1.0 / np.sqrt(np.sum(s.astype(np.float64) ** 2) + 1e-6))
            kk[b - 16] = (s * r).astype(np.float32)
        else:
            h = b - 32
            v[h] = s
            a = np.float64(bf16_to_f32(xq_bits[A_OFF + h]))
            bb = np.float64(bf16_to_f32(xq_bits[B_OFF + h]))
            x = a + np.float64(dtbias[h])
            sp = np.log(1.0 + np.exp(x)) if x <= 20.0 else x
            g[h, 0] = np.float32(-np.exp(np.float64(alog[h])) * sp)
            beta[h, 0] = np.float32(1.0 / (1.0 + np.exp(-bb)))
    cs_new = np.empty_like(cs_bits)
    cs_new[0] = cs_bits[1]
    cs_new[1] = cs_bits[2]
    cs_new[2] = np.asarray(xq_bits, dtype=np.uint16).reshape(-1)[:CH]
    return {'q': q, 'k': kk, 'v': v, 'g': g, 'beta': beta, 'cs': cs_new}


def ref_recur(q, k, v, g, beta, ssm_in):
    """S4：per value head hv（hk = hv//3），状态原位递推 → (ssm_out, o)。"""
    ssm = np.array(ssm_in, dtype=np.float64).reshape(HEADS, HEAD, HEAD)
    o = np.empty((HEADS, HEAD), dtype=np.float32)
    for h in range(HEADS):
        hk = h // VGROUP
        eg = np.exp(np.float64(g[h, 0]))
        kh = k[hk].astype(np.float64)
        qh = q[hk].astype(np.float64)
        for i in range(HEAD):
            row = ssm[h, i] * eg
            w = float(row @ kh)
            d = np.float64(beta[h, 0]) * (np.float64(v[h, i]) - w)
            row += kh * d
            ssm[h, i] = row
            o[h, i] = np.float32(float(row @ qh))
    return ssm, o


def ref_gated(o, z_bits, gamma_g_bits):
    """S5：per head(128) RMSNormGated（norm_before_gate + sigmoid）。"""
    out = np.empty((HEADS, HEAD), dtype=np.uint16)
    gg = bf16_to_f32(gamma_g_bits).astype(np.float32)
    for h in range(HEADS):
        oh = np.asarray(o, dtype=np.float32)[h]
        var = np.sum(oh.astype(np.float64) ** 2) / HEAD
        rstd = np.float32(1.0 / np.sqrt(var + 1e-6))
        z = bf16_to_f32(np.asarray(z_bits, dtype=np.uint16).reshape(-1)[Z_OFF + h * HEAD: Z_OFF + (h + 1) * HEAD])
        z = z.astype(np.float32)
        sig = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        y = ((oh * rstd).astype(np.float32) * gg).astype(np.float32) * sig
        out[h] = f32_to_bf16(y)
    return out.reshape(-1)


# ------------------------------------------------------------------
# 单 tag 校验
# ------------------------------------------------------------------

def _diag_f32(got, exp):
    a = np.abs(np.asarray(got, dtype=np.float64).reshape(-1) - np.asarray(exp, dtype=np.float64).reshape(-1))
    e = np.abs(np.asarray(exp, dtype=np.float64).reshape(-1))
    return 'maxAbs %.3e maxRel %.3e' % (float(a.max()), float((a / (e + 1e-6)).max()))


def _diag_bf16(got, exp):
    d = ulp_dist_bf16(np.asarray(got, dtype=np.uint16).reshape(-1), np.asarray(exp, dtype=np.uint16).reshape(-1))
    return 'maxUlp %d bit 一致 %.4f%%' % (int(d.max()), 100.0 * np.count_nonzero(d == 0) / d.size)


def _res2_judge(st, tag, got_res2, exp_res2, slice_mode):
    """S7 res2：mode 0（输入 opout 逐位一致）走逐位判据；host 全链降为报告项（原因见下）。"""
    if slice_mode == 0:
        return cmp_f32_bit(st, tag, got_res2, exp_res2)
    nbad = int(np.count_nonzero(np.asarray(got_res2, dtype=np.float32).reshape(-1) !=
                                np.asarray(exp_res2, dtype=np.float32).reshape(-1)))
    st.report(tag, 'res2 位型不同 %d/%d（源: 参考侧 opout 的 bf16 网格 ≤1 ulp，不是 S7 的算术问题；'
                   'S7 算术精确性由 ② 的隔离项以逐位判据给出）'
              % (nbad, np.asarray(exp_res2).size))
    return True


def check_tag(st, params, tag, d, ref_state, dev_state_in, pure_state, slice_mode, stage_limit):
    """ref_state = (cs, ssm) ①族参考侧携带状态；dev_state_in = (cs, ssm) 设备侧上一步输出（隔离用）；
    pure_state = (cs, ssm) ③族纯参考链自己的携带状态（与 ① 分开，否则 chain2 的 ③ 会混入设备锚）。

    slice_mode：0 = chain（设备整链；①S3–S7 的参考输入 = 设备 bf16 出口字节 ⇒「分段, 设备锚」）
                1 = slice（host 任意 qkvzba → device S3→S5；AIC 不参与）
                2 = hostchain（host 声明输入 → device S3→S7；参考侧无任何设备锚 ⇒「host 全链」）
    """
    got = {}
    for name in ('x_norm', 'res1', 'qkvzba', 'q', 'k', 'v', 'g', 'beta', 'o', 'opin', 'opout', 'y_final', 'res2',
                 'conv_state', 'ssm_state'):
        try:
            got[name] = d.raw(name)
        except KeyError:
            got[name] = None
    xin = params['x_in_s%d' % int(d.hdr['step'])]
    resin = params['res_in_s%d' % int(d.hdr['step'])]
    g1 = params['gamma1']
    g2 = params['gamma2']
    cs_ref, ssm_ref = ref_state
    cs_dev_in, ssm_dev_in = dev_state_in
    ok = True

    # S3 的输入：mode 0 = 设备 workspace 的 qkvzba（bf16 锚点）；mode 1/2 = host 声明字节。
    qkv_src = got['qkvzba'].reshape(-1)
    fam = ' (分段, 设备锚)' if slice_mode == 0 else ' (host 全链)'
    # host 声明输入 → S1 / S2：mode 0 时它们是 ①S1/S2 的判据值；mode 2 时它们同时是 S7 残差与 S3 输入
    res1_host, xnorm_host = ref_norm(xin, resin, None, g1)
    qkv_ref_host = ref_gemm(xnorm_host, params['w_in'], HIDDEN, IN_N)

    # ---- 覆盖登记（三态分栏由这些条件产出；下面的判据用的是**同一批布尔值**，无手写数字）----
    cov_s1s2 = st.gate(slice_mode == 0, tag + ' ①S1/S2 (host 全链)',
                       'slice_mode=%d：device 不跑 S1/S2（S2 出口由 host 直供）' % slice_mode)
    cov_s2own = st.gate(slice_mode == 2, tag + ' ①S2 出口自洽',
                        '仅 hostchain（其余 mode 的 S3 输入不是「参考自算的 S2 出口」）')
    cov_s3 = st.gate(stage_limit >= 3, tag + ' ①S3', 'stage_limit=%d' % stage_limit)
    cov_s4 = st.gate(stage_limit >= 4, tag + ' ①S4', 'stage_limit=%d' % stage_limit)
    cov_s5 = st.gate(stage_limit >= 5, tag + ' ①S5', 'stage_limit=%d' % stage_limit)
    cov_s6 = st.gate(slice_mode != 1 and stage_limit >= 6, tag + ' ①S6',
                     'slice_mode=1：device 不跑 S6' if slice_mode == 1 else 'stage_limit=%d' % stage_limit)
    cov_s7 = st.gate(slice_mode != 1 and stage_limit >= 7, tag + ' ①S7',
                     'slice_mode=1：device 不跑 S7' if slice_mode == 1 else 'stage_limit=%d' % stage_limit)

    # ================= ① 参考链 =================
    if cov_s1s2:
        ok &= cmp_f32_bit(st, tag + ' S1 res1 (host 全链)', got['res1'].reshape(-1), res1_host)
        ok &= cmp_bf16_grid(st, tag + ' S1 x_norm (host 全链)', got['x_norm'].reshape(-1), xnorm_host)
        if stage_limit >= 2:
            ok &= cmp_bf16_grid(st, tag + ' S2 qkvzba (host 全链)', got['qkvzba'].reshape(-1), qkv_ref_host)
    if cov_s2own:
        # mode 2：S3 的输入 = host 参考链自己的 S2 出口 ⇒ 与本脚本独立重算的 ref_gemm 应逐位一致
        # （注意：这是 C++ host 参考与 numpy 参考之间的**同源转录**交叉核对，不是规则级独立见证）
        ok &= cmp_u16_bit(st, tag + ' host 声明 S2 出口 == check_ref 独立重算', qkv_src, qkv_ref_host)
    if slice_mode != 0 and cov_s3:
        # 物质见证：device 侧把消费到的 q|k|v 前缀原地移进 conv_state[2]（落盘字节）⇒ 与 host 输入逐字节比。
        # M62 订正：原判据两侧取同一条 host 指针（恒真项，自指形态 S4「同一份值算两遍」）⇒ 已换成这条。
        # 标签带上 mode（M62 r1 P2-4：与 kernel 侧一致，且不再用 slice: 前缀去指 mode 2；case 名已在 tag 里）
        ok &= cmp_u16_bit(st, tag + ' (mode %d) device 消费的 q|k|v == host 输入字节' % slice_mode,
                          got['conv_state'].reshape(-1)[2 * CH:3 * CH], qkv_src[:CH])

    cur_cs, cur_ssm = cs_ref, ssm_ref
    if cov_s3:
        r = ref_prolog(qkv_src, cur_cs, params['conv_w'], params['conv_bias'], params['alog'], params['dtbias'])
        cur_cs = r['cs']
        ok &= cmp_f32_rel(st, tag + ' S3 q' + fam, got['q'].reshape(-1), r['q'].reshape(-1))
        ok &= cmp_f32_rel(st, tag + ' S3 k' + fam, got['k'].reshape(-1), r['k'].reshape(-1))
        ok &= cmp_f32_rel(st, tag + ' S3 v' + fam, got['v'].reshape(-1), r['v'].reshape(-1))
        ok &= cmp_f32_rel(st, tag + ' S3 g' + fam, got['g'].reshape(-1)[0::8], r['g'].reshape(-1)[0::8])
        ok &= cmp_f32_rel(st, tag + ' S3 beta' + fam, got['beta'].reshape(-1)[0::8], r['beta'].reshape(-1)[0::8])
        ok &= cmp_u16_bit(st, tag + ' S3 conv_state' + fam, got['conv_state'].reshape(-1), cur_cs.reshape(-1))
        if cov_s4:
            cur_ssm, o = ref_recur(r['q'], r['k'], r['v'], r['g'], r['beta'], cur_ssm)
            ok &= cmp_f32_rel(st, tag + ' S4 o' + fam, got['o'].reshape(-1), o.reshape(-1))
            ok &= cmp_f32_rel(st, tag + ' S4 ssm_state' + fam, got['ssm_state'].reshape(-1), cur_ssm.reshape(-1))
            report_state_bit(st, tag + ' S4 ssm_state 位级统计' + fam, got['ssm_state'], cur_ssm)
            if cov_s5:
                # mode 0（锚）：参考的 o 由设备 qkvzba 续算、z 取设备 qkvzba；
                # mode 1/2（host）：参考的 o 与 z 都来自 host 声明输入 ⇒ 这条即是「S2 出口→S5」的组合判据
                opin_r = ref_gated(o.astype(np.float32), qkv_src, params['gamma_g'])
                ok &= cmp_bf16_grid(st, tag + ' S5 opin = RMSNormGated' + fam, got['opin'].reshape(-1), opin_r)
                if cov_s6:
                    # mode 0：S6 的输入取设备 opin（锚）；mode 1/2：取参考自己的 opin ⇒ 覆盖 S5 出口的组合
                    opin_in = got['opin'].reshape(-1) if slice_mode == 0 else opin_r
                    opout_r = ref_gemm(opin_in, params['w_out'], V_DIM, HIDDEN)
                    ok &= cmp_bf16_grid(st, tag + ' S6 out_proj' + fam, got['opout'].reshape(-1), opout_r)
                    if cov_s7:
                        opout_in = got['opout'].reshape(-1) if slice_mode == 0 else opout_r
                        r2, y2 = ref_norm(opout_in, None, res1_host.astype(np.float32), g2)
                        ok &= _res2_judge(st, tag + ' S7 res2 (fp32)' + fam, got['res2'].reshape(-1), r2, slice_mode)
                        ok &= cmp_bf16_grid(st, tag + ' S7 y_final' + fam, got['y_final'].reshape(-1), y2)

    # ================= ② 逐段隔离（用设备自己上一段的输出当本段输入）=================
    if slice_mode == 0 and stage_limit >= 1:
        res1_i, xnorm_i = ref_norm(xin, resin, None, g1)
        ok &= cmp_f32_bit(st, tag + ' S1 res1 (隔离)', got['res1'].reshape(-1), res1_i)
        ok &= cmp_bf16_grid(st, tag + ' S1 x_norm (隔离)', got['x_norm'].reshape(-1), xnorm_i)
        if stage_limit >= 2:
            # 下游用**设备的** x_norm（锚）⇒ 这条把 S2 的 GEMM 与 S1 的误差隔离开
            qkv_i = ref_gemm(got['x_norm'].reshape(-1), params['w_in'], HIDDEN, IN_N)
            ok &= cmp_bf16_grid(st, tag + ' S2 in_proj (隔离)', got['qkvzba'].reshape(-1), qkv_i)
    if cov_s3:
        # S3 的输入 = 设备实际的 qkvzba（mode 0）或 host 声明字节（mode 1/2）
        r = ref_prolog(qkv_src, cs_dev_in, params['conv_w'], params['conv_bias'], params['alog'], params['dtbias'])
        ok &= cmp_f32_rel(st, tag + ' S3 q (隔离)', got['q'].reshape(-1), r['q'].reshape(-1))
        ok &= cmp_f32_rel(st, tag + ' S3 k (隔离)', got['k'].reshape(-1), r['k'].reshape(-1))
        ok &= cmp_f32_rel(st, tag + ' S3 v (隔离)', got['v'].reshape(-1), r['v'].reshape(-1))
        ok &= cmp_f32_rel(st, tag + ' S3 g (隔离)', got['g'].reshape(-1)[0::8], r['g'].reshape(-1)[0::8])
        ok &= cmp_f32_rel(st, tag + ' S3 beta (隔离)', got['beta'].reshape(-1)[0::8], r['beta'].reshape(-1)[0::8])
        ok &= cmp_u16_bit(st, tag + ' S3 conv_state (隔离, 位级)', got['conv_state'].reshape(-1), r['cs'].reshape(-1))
        # 自洽性： new[0]=old[1], new[1]=old[2], new[2]=x[q|k|v]
        sh = got['conv_state'].reshape(-1)
        cons = (np.array_equal(sh[0:CH], cs_dev_in.reshape(-1)[CH:2 * CH]) and
                np.array_equal(sh[CH:2 * CH], cs_dev_in.reshape(-1)[2 * CH:3 * CH]) and
                np.array_equal(sh[2 * CH:3 * CH], qkv_src[:CH].astype(np.uint16)))
        ok &= st.judge(tag + ' conv_state 移位一致性', bool(cons), 'new[0]=old[1],new[1]=old[2],new[2]=x')
        if cov_s4:
            ssm, o = ref_recur(got['q'].astype(np.float32), got['k'].astype(np.float32), got['v'].astype(np.float32),
                               got['g'].astype(np.float32), got['beta'].astype(np.float32), ssm_dev_in)
            ok &= cmp_f32_rel(st, tag + ' S4 o (隔离)', got['o'].reshape(-1), o.reshape(-1))
            ok &= cmp_f32_rel(st, tag + ' S4 ssm_state (隔离)', got['ssm_state'].reshape(-1), ssm.reshape(-1))
            report_state_bit(st, tag + ' S4 ssm_state 位级统计(隔离)', got['ssm_state'], ssm)
            if cov_s5:
                opin = ref_gated(got['o'].astype(np.float32), qkv_src, params['gamma_g'])
                ok &= cmp_bf16_grid(st, tag + ' S5 opin (隔离)', got['opin'].reshape(-1), opin)
                if cov_s6:
                    opout = ref_gemm(got['opin'].reshape(-1), params['w_out'], V_DIM, HIDDEN)
                    ok &= cmp_bf16_grid(st, tag + ' S6 out_proj (隔离)', got['opout'].reshape(-1), opout)
                    if cov_s7:
                        r2, y2 = ref_norm(got['opout'].reshape(-1), None, got['res1'].astype(np.float32), g2)
                        ok &= cmp_f32_bit(st, tag + ' S7 res2 (隔离, fp32)', got['res2'].reshape(-1), r2)
                        ok &= cmp_bf16_grid(st, tag + ' S7 y_final (隔离)', got['y_final'].reshape(-1), y2)

    # ================= ③ 传播（族 P；报告项，不判定；只在 mode 0 有独立含义）=================
    # 从 host 声明输入出发、参考在**每个 bf16 出口都用自己算出的舍入值**续算（无任何设备锚）——
    # 这就是「从 host 输入重算的全链」，但它**不可复现**：参考缺设备 S2/S6 的 fp32 L0C 累加值
    # （实测 2/16480 个 S2 出口元素差 1 bf16 ulp），conv 近相消把这点差放大到下游。故只报数。
    p_cs, p_ssm_in = pure_state
    if slice_mode == 0:
        p_res1, _p_xnorm = ref_norm(xin, resin, None, g1)
        pr = ref_prolog(qkv_ref_host, p_cs, params['conv_w'], params['conv_bias'], params['alog'],
                        params['dtbias'])
        p_cs = pr['cs']
        p_ssm, p_o = ref_recur(pr['q'], pr['k'], pr['v'], pr['g'], pr['beta'], p_ssm_in)
        p_ssm_in = p_ssm
        p_opin = ref_gated(p_o.astype(np.float32), qkv_ref_host, params['gamma_g'])
        p_opout = ref_gemm(p_opin, params['w_out'], V_DIM, HIDDEN)
        _p_r2, p_y = ref_norm(p_opout, None, p_res1.astype(np.float32), g2)
        st.report(tag + ' P: S3 k (纯参考链)', _diag_f32(got['k'].reshape(-1), pr['k'].reshape(-1)))
        st.report(tag + ' P: S4 ssm_state (纯参考链)', _diag_f32(got['ssm_state'].reshape(-1), p_ssm.reshape(-1)))
        st.report(tag + ' P: S6 opout (纯参考链)', _diag_bf16(got['opout'].reshape(-1), p_opout))
        st.report(tag + ' P: S7 y_final (纯参考链)', _diag_bf16(got['y_final'].reshape(-1), p_y))
    return ok, (cur_cs, cur_ssm), (p_cs, p_ssm_in)


def selfcheck(st, d):
    """咬合自检（负向对照；报告项，不计入判定）：

    ① 篡改 1 个元素的 2 ulp（网格判据**必须** FAIL）⇒ 证明判据是活的；
    ② 把**全体**元素统一 +1 ulp ⇒ 网格判据**仍 PASS** ⇒ 定量证明它对「≤1 ulp 的规则级偏移」无感。
    ② 正是族 D 咬不住的那一类（M59 记的 S1-条件），必须写进 README 而不是靠读者自己推。
    """
    x = np.asarray(d.raw('x_norm'), dtype=np.uint16).reshape(-1).copy()
    mut = x.copy()
    mut[0] = np.uint16((int(mut[0]) + 2) & 0xFFFF)          # +2 ulp（同号幅值域）
    d1 = int(ulp_dist_bf16(mut, x).max())
    st.report('咬合自检 ①：篡改 1 元素 +2 ulp', 'maxUlp %d ⇒ 网格判据%s（必须咬得住）'
              % (d1, 'FAIL' if d1 > 1 else 'PASS=判据失效'))
    xf = x.astype(np.int32)
    up = np.where(xf & 0x8000 != 0, xf - 1, xf + 1).astype(np.uint16)
    d2 = int(ulp_dist_bf16(up, x).max())
    st.report('咬合自检 ②：全体 +1 ulp（负向对照）', 'maxUlp %d ⇒ 网格判据仍 PASS ⇒ **咬不住** ≤1 ulp 的规则级偏移'
              % d2)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    st = Stat()
    pfile = os.path.join(root, 'm14_params.txt')
    if not os.path.exists(pfile):
        print('[m14-ref] RESULT: SKIPPED — 找不到 %s（先跑 M14_DUMP=1 的 kernel）' % pfile)
        return 2
    try:
        params = Dump(root, pfile)
    except Exception as e:      # noqa: BLE001
        print('[m14-ref] RESULT: SKIPPED — params manifest 解析失败: %s' % e)
        return 2

    tags = []
    n_bad_manifest = 0
    for path in sorted(glob.glob(os.path.join(root, '*.txt'))):
        base = os.path.basename(path)
        if base == 'm14_params.txt' or base.endswith('_meta.txt'):
            continue
        try:
            d = Dump(root, path)
        except Exception as e:      # noqa: BLE001
            print('[m14-ref] SKIP %s（manifest 解析失败: %s）' % (base, e))
            n_bad_manifest += 1
            continue
        tags.append(d)
    if not tags:
        print('[m14-ref] RESULT: SKIPPED — 未找到任何 tag manifest（先跑 M14_DUMP=1 的 kernel）')
        return 2
    print('[m14-ref] 发现 %d 个 tag manifest（解析失败跳过 %d 个）；覆盖范围见末尾三态汇总' %
          (len(tags), n_bad_manifest))

    # 咬合自检（负向对照）：用 chain 档的 x_norm；没有就如实记「未覆盖」
    sc = next((d for d in tags if d.hdr['case'] == 'chain' and 'x_norm' in d.tens), None)
    if sc is not None:
        selfcheck(st, sc)
    else:
        st.skip('咬合自检', '未找到 chain 档（自检需要它的 x_norm dump）')

    # 按 case 分组、按 step 排序；跨 step 携带参考状态与设备状态
    by_case = {}
    for d in tags:
        by_case.setdefault(d.hdr['case'], []).append(d)
    for case in sorted(by_case.keys()):
        ds = sorted(by_case[case], key=lambda x: int(x.hdr['step']))
        slice_mode = int(ds[0].hdr['slice_mode'])
        stage_limit = int(ds[0].hdr['stage_limit'])
        fam = '分段(设备锚)' if slice_mode == 0 else 'host 全链'
        print('[m14-ref] ---- case %s (%d step%s, slice_mode=%d stage_limit=%d): ① 参考链=%s / ② 隔离 ----' %
              (case, len(ds), 's' if len(ds) > 1 else '', slice_mode, stage_limit, fam))
        cs_ref = params.raw('conv_state_init')
        ssm_ref = params.f32('ssm_state_init')
        cs_dev = cs_ref
        ssm_dev = ssm_ref
        cs_pure = cs_ref      # ③族纯参考链自己的携带状态（与 ① 分离）
        ssm_pure = ssm_ref
        for d in ds:
            tag = d.hdr['tag']
            ok, (cs_ref, ssm_ref), (cs_pure, ssm_pure) = check_tag(
                st, params, tag, d, (cs_ref, ssm_ref), (cs_dev, ssm_dev), (cs_pure, ssm_pure),
                slice_mode, stage_limit)
            if not ok:
                print('[m14-ref] %s 存在 FAIL 项（见上）' % tag)
            cs_dev = d.raw('conv_state')
            ssm_dev = d.f32('ssm_state')
    good = st.banner()
    if st.n_judge == 0:
        # 一条判定项都没产出 ⇒ 与「比过且通过」用退出码区分（0 = 通过 / 1 = 有差异 / 2 = 没得判）
        print('[m14-ref] RESULT: SKIPPED（判定项 0：本批 dump 没产出任何判定项，不发合格证；'
              '报告项 %d，未覆盖 %d）' % (st.n_report, st.n_skipped))
        return 2
    print('[m14-ref] RESULT: %s（判定项 %d，报告项 %d，未覆盖 %d）' %
          ('OK' if good else 'DIFFERENCES', st.n_judge, st.n_report, st.n_skipped))
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
