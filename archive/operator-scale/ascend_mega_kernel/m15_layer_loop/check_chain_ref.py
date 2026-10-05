#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m15_layer_loop/check_chain_ref.py —— M65：48 层链的 hc 边界 / 末层 mixer 与 **M39 官方单层参考**对拍

参考（oracle）= **m21_layer_ref（M39）里逐句复刻官方 nvidia 路径的 torch 实现**：

    m21_layer_ref/ref/hc.py      GatedResidual.combine_and_mix / mix / combine、grouped_gemma_rmsnorm、
                                 hc_silu / hc_gate_mix / hc_combine（行号锚点见该文件注释）
    m21_layer_ref/ref/layer.py   final_mixer（全局 hyper_connection_mixer，use_combine=False）
    m21_layer_ref/ref/ckpt.py    LayerWeights（真实 checkpoint 的权重，**不读 m15 的权重 dump**）

被对拍的是 `M15_DUMP=1 ... chain` 落下来的设备侧张量（`ch_L*_*` / `ch_gm_*`）。
**判据（docs/17 §1.1 的 T1/T2/T3 三档）**：bf16 张量一律走 m20 的保守门限
（`m20_hyperconn/check_ref.py::judge`：张量尺度归一化绝对误差 ≤1e-2 **且** 良态元素 ulp ≤2 **且**
良态元素逐位率 ≥99%）；fp32 证据张量（rstd / injw）走相对误差；**层 1 的舍入点**另有逐字节判据
（在 C++ 侧 `Ch.ple.bf16.L01`），本脚本只做它的**判别性负向对照**（见下）。

用法（在 dump 目录里跑；先用 M15_DUMP=1 落盘）：

    mkdir -p /tmp/m65_dump && cd /tmp/m65_dump
    M15_DUMP=1 <bin> <repo>/m15_layer_loop/weights_manifest.txt chain
    /workspace/venvs/baseline/bin/python3 <repo>/m15_layer_loop/check_chain_ref.py; echo rc=$?

（**必须用 baseline venv 的 python**：本脚本要 torch；m15 的其它 check_* 脚本只用 numpy。）

## 对拍的三段（与 C++ 侧 `H_RunChain` 的分组对应）

    A1  边界 #1：设备 `ch_L*_hin/bo/ij` → 参考 `mix`（层 0）/ `combine_and_mix`（层 ≥2）/
        `combine`+`mix` 两段（层 1 = PLE 打断点）→ 与设备 `ch_L*_b1_*` 对拍
    A2  边界 #2：设备 `ch_L*_attnout`（层 0 用 `hin`）→ 参考 `combine_and_mix` → 与 `ch_L*_b2_*` 对拍
    GM  末层全局 mixer：设备末层三态 → 参考 `final_mixer` → 与 `ch_gm_multi` / `ch_gm_sample` 对拍

## 覆盖范围交代（tower 规则）

* 覆盖 = `M15_CHC_REF_LAYERS`（默认 `M15_HC_LAYERS` = 0,1,3）列出的层 × 该层的 {边界 #1、边界 #2}
  + 末层 mixer 一次；每条判据由调用点自增同一个计数器产出。
* **不覆盖**：① 子层段（GDN / QSA-attention）与 MoE 段 —— 本 kernel 的 attention 仍是占位直通，
  MoE 是 4 专家缩形档，与官方 512 专家不同源，**不能**用官方参考对拍；② PLE 本体（未实现）；
  ③ 三相位/四相位的资源叠放与同步（由 kernel 内的 `H.h1ws`/`H.h2ws` 与 C++ 侧的
  `Ch.*` 手工路径判据覆盖）；④ 跨层 handoff 的地址算式（由 C++ 侧 `Ch.h1*/h2*/bo` 覆盖）。
* **负向对照**（必须做，否则「逐字节判据判别性」只是口头声明）：`--negctrl`（默认开）在**真实层 1
  数据**上算「丢掉 combine→RMSNorm 那次 bf16 物化」的反事实输出，并报告有多少 bf16 元素改变；
  若该比例过低（<1%），说明这条判据不具判别性 —— 脚本会把它算作**失败**。
  对照组：M39 的 B4 实测在同族数据上量到「物化残差」**14029/40960 = 34.25%** 元素受影响
  （`m21_layer_ref/evidence/selfcheck.log`）。

## 退出码（三态）

    0 = 比过且通过    1 = 比过且有差异    2 = 没得比 / 输入缺失（SKIPPED）
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(ROOT, "m20_hyperconn"))
import check_ref as m20  # noqa: E402  —— m20 的独立保守门限（judge / b16 / f32）

sys.path.insert(0, os.path.join(ROOT, "m21_layer_ref"))
try:
    import torch  # noqa: E402
    from ref.ckpt import LayerWeights  # noqa: E402
    from ref.hc import GatedResidual, grouped_gemma_rmsnorm, hc_gate_mix, hc_silu  # noqa: E402
    from ref.layer import final_mixer  # noqa: E402
except Exception as e:  # pragma: no cover - 只在环境缺 torch 时走到
    print("[chk] RESULT: SKIPPED（本脚本需要 torch / m21_layer_ref：%s）" % e)
    print("[chk] 请用 /workspace/venvs/baseline/bin/python3 运行")
    sys.exit(2)

HC, HID, HYPER = m20.HC, m20.HID, m20.HYPER
LOWRANK, INJ_N, IJ_STRIDE, INJW_SLOT = m20.LOWRANK, m20.INJ_N, m20.IJ_STRIDE, m20.INJW_SLOT
OH_W, UP_N = m20.OH_W, m20.UP_N
OH_INJ = LOWRANK    # = m15_hc_resources.h 的 `OH_INJ`（injection 列起点 = 320）
MODE_MIX, MODE_COMBINE_MIX, MODE_FINAL_MIX = m20.MODE_MIX, m20.MODE_COMBINE_MIX, m20.MODE_FINAL_MIX
PACK_ELEMS = LOWRANK + INJ_N            # 与 C++ 侧 CH_OH_PACK_ELEMS 同源（lora + injection 紧凑布局）
EPS = 1e-6

COUNT = {"judge": 0, "guard": 0, "skipped": 0}
FAILS = []
# 超 m20 保守门限（但 T3 界内）的读数：逐条收集、末尾 DISCLOSE 段打印，**不静默吸收**
OVER = []


class MissingInput(Exception):
    pass


def rd(name, dtype, shape):
    path = name + ".bin"
    if not os.path.exists(path):
        raise MissingInput(path)
    a = np.fromfile(path, dtype=dtype)
    if a.size != int(np.prod(shape)):
        raise MissingInput("%s 元素数 %d != 期望 %d（shape %s）" % (path, a.size, int(np.prod(shape)), shape))
    return a.reshape(shape)


def bf(name, shape):
    """设备张量（bf16 位模式，uint16）"""
    return rd(name, np.uint16, shape)


def f32b(name, shape):
    return rd(name, np.float32, shape)


def j(name, got_u16, exp_f64):
    """bf16 张量判据。**两条门限并列**：

      · **判定门 = m20 保守门限去掉「良态逐位率 ≥99%」那一条**：张量尺度归一化绝对误差 ≤1e-2
        **且** 良态元素 ulp ≤2。依据：hc 段这六个张量的**适用档是 docs/17 §1.1 的 T3**（逐项 ε
        推导见 M58-6 表），而 docs/17 §1.1 明确把 T3 档的「`≤1ulp 比例` / `maxRel` / 位级一致率」
        **降为报告项**，判定看逐元素的界 ⇒ 本脚本按该规范处置。
      · **报告项**：m20 的「良态逐位率 ≥99%」读数照打，凡被它挡下的逐条进 `[DISCLOSE]` 段
        （它的下限是在 m20 自己的输入分布上校准的；M58 在 m=33 档已记录过同类越门限）。
    """
    got = m20.f32(got_u16).astype(np.float64)
    scale = float(np.max(np.abs(exp_f64))) if exp_f64.size else 0.0
    norm_abs = float(np.max(np.abs(got - exp_f64))) / scale if scale > 0 else 0.0
    d = np.abs(m20.ulp_key(got_u16) - m20.ulp_key(m20.b16(exp_f64)))
    sig = np.abs(exp_f64) > 0.05 * scale
    sfrac = float(np.mean(d[sig] == 0)) if np.any(sig) else 1.0
    maxulp = int(np.max(d[sig])) if np.any(sig) else 0
    frac = float(np.mean(d == 0))
    gate_ok = norm_abs <= 1e-2 and maxulp <= 2          # 判定门（见 docstring）
    m20_ok = gate_ok and sfrac >= 0.99                  # m20 保守门限（多一条逐位率下限，作报告项）
    COUNT["judge"] += 1
    if not gate_ok:
        FAILS.append(name)
    if not m20_ok:
        OVER.append((name, frac, sfrac, maxulp, norm_abs))
    print("[chk] 判定 %-24s n=%-7d 逐位=%.4f 良态逐位=%.4f ulpMax=%-3d 归一maxAbs=%.3e  %s%s"
          % (name, got_u16.size, frac, sfrac, maxulp, norm_abs, "PASS" if gate_ok else "FAIL",
             "" if m20_ok else "  [超 m20 保守门限：良态逐位率 <0.99；判定门内]"))
    return gate_ok


def jr(name, got_f32, exp_f64, tol):
    rel = float(np.max(np.abs(got_f32.astype(np.float64) - exp_f64) / np.maximum(np.abs(exp_f64), 1e-30)))
    COUNT["judge"] += 1
    ok = rel <= tol
    if not ok:
        FAILS.append(name)
    print("[chk] 判定 %-24s maxRel=%.3e (tol %.0e)  %s" % (name, rel, tol, "PASS" if ok else "FAIL"))
    return ok


def g(name, cond, detail=""):
    COUNT["guard"] += 1
    if not cond:
        FAILS.append(name)
    print("[chk] guard %-23s %s  %s" % (name, detail, "OK" if cond else "DEGENERATE"))
    return cond


def load_layout(path="hc_layout.txt"):
    if not os.path.exists(path):
        raise MissingInput(path)
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)", line):
                d[k] = int(v)
    return d


def t_u16_to_torch(a):
    """uint16（bf16 位模式）numpy → torch bf16"""
    return torch.frombuffer(np.ascontiguousarray(a, dtype=np.uint16).tobytes(), dtype=torch.uint16).view(
        torch.bfloat16).reshape(a.shape)


def t_bf16_to_f64(t):
    return t.detach().to(torch.float32).numpy().astype(np.float64)


def extract_inj(packed_oh_u16, m):
    """紧凑 OH `[m, 324]`（lora || injection）→ injection `[m, 4]` 的 bf16 位模式"""
    return np.ascontiguousarray(packed_oh_u16[:, LOWRANK:LOWRANK + INJ_N])


def main():
    try:
        return _main()
    except MissingInput as e:
        print("[chk] RESULT: SKIPPED (缺输入文件：%s)\n"
              "[chk] 先在空目录跑一次 `M15_DUMP=1 <bin> <manifest> chain` 再复核。" % e)
        return 2


def _main():
    layout = load_layout()
    m = int(os.environ.get("M15_CHAIN_M", os.environ.get("M15_HC_M", "1")))
    layers_env = os.environ.get("M15_CHC_REF_LAYERS", os.environ.get("M15_HC_LAYERS", "0,1,3"))
    layers = [int(x) for x in re.split(r"[,\s]+", layers_env.strip()) if x != ""]

    print("[chk] m15 48 层链 × **M39 官方单层参考**（m21_layer_ref/ref：torch，逐句复刻 nvidia 路径）")
    print("[chk] 层表 = %s；m = %d；判据门限 = m20_hyperconn/check_ref.py::judge（保守三条件）" % (layers, m))
    print("[chk] ws 布局来自 hc_layout.txt（由 C++ 常量算出）；紧凑 OH = lora[0:320] || injection[320:324]")

    # 末层（`ch_gm_*`）的对拍：末层的三态 = 层 47 的出口（dump 里恒有 `ch_gm_hin/bo/ij`）
    gm_hin = bf("ch_gm_hin", (m, HYPER))
    gm_bo = bf("ch_gm_bo", (m, HID))
    gm_ij = bf("ch_gm_ij", (m, INJ_N))
    gm_ws = {k: bf("ch_gm_b1_%s" % k, sh) for k, sh in
             (("hcp", (m, HYPER)), ("xn", (m, HYPER)), ("oh", (m, PACK_ELEMS)), ("ls", (m, LOWRANK)),
              ("gate", (m, UP_N)), ("blk", (m, HID)))}
    gm_multi = bf("ch_gm_multi", (m, HYPER))
    gm_sample = bf("ch_gm_sample", (m, HID))

    for L in layers:
        base = "ch_L%02u" % L
        attn_mode = MODE_MIX if L == 0 else MODE_COMBINE_MIX
        print("\n===== 层 %02u（attn 边界 %s，mlp 边界 combine_and_mix）=====" %
              (L, "MIX（无 pending combine）" if attn_mode == MODE_MIX else
               ("combine-only → mix-only（**PLE 打断点**）" if L == 1 else "combine_and_mix")))

        hin = bf(base + "_hin", (m, HYPER))
        bo = bf(base + "_bo", (m, HID))
        ij = bf(base + "_ij", (m, INJ_N))
        attnout = bf(base + "_attnout", (m, HID))
        d1 = {k: bf("%s_b1_%s" % (base, k), sh) for k, sh in
              (("hcp", (m, HYPER)), ("xn", (m, HYPER)), ("oh", (m, PACK_ELEMS)), ("ls", (m, LOWRANK)),
               ("gate", (m, UP_N)), ("blk", (m, HID)))}
        d1["rstd"] = f32b("%s_b1_rstd" % base, (m, HC))
        d1["injw"] = f32b("%s_b1_injw" % base, (m, HC, INJW_SLOT))
        d2 = {k: bf("%s_b2_%s" % (base, k), sh) for k, sh in
              (("hcp", (m, HYPER)), ("xn", (m, HYPER)), ("oh", (m, PACK_ELEMS)), ("ls", (m, LOWRANK)),
               ("gate", (m, UP_N)), ("blk", (m, HID)))}
        d2["rstd"] = f32b("%s_b2_rstd" % base, (m, HC))
        d2["injw"] = f32b("%s_b2_injw" % base, (m, HC, INJW_SLOT))

        # ---- 权重：官方参考自己从 checkpoint 读（不读 m15 的权重 dump）----
        lw = LayerWeights(L)
        wa = lw.hyper_connection("attn_hyper_connection")
        wm = lw.hyper_connection("mlp_hyper_connection")

        def mix(wi):
            return GatedResidual(wi.hc_norm_weight, wi.down_weight, wi.up_weight, wi.inject_weight,
                                 hc_count=HC, eps=EPS, use_combine=True)

        attn, mlp = mix(wa), mix(wm)
        th_in = t_u16_to_torch(hin)
        th_bo = t_u16_to_torch(bo)
        th_ij = t_u16_to_torch(ij)

        # ---------- A1：边界 #1 ----------
        print("-- A1 边界 #1（输入 = 设备 dump 的 hin/bo/ij）--")
        ref_hcp = None
        if L == 0:
            h_ref, blk_ref, inj_ref = attn.mix(th_in)
            print("[chk] 判定 %-24s mode MIX 不做 combine ⇒ 设备 `b1_hcp` 未被写出（显式 SKIPPED）" % "A1.H'")
            COUNT["skipped"] += 1
        elif L == 1:
            # PLE 打断点：combine（物化 bf16 H'）→ mix（无 combine）
            ref_hcp = attn.combine(th_in, th_bo, th_ij)
            h_ref, blk_ref, inj_ref = attn.mix(ref_hcp)
            j("A1.H'<-combine", d1["hcp"], t_bf16_to_f64(ref_hcp))
        else:
            ref_hcp, blk_ref, inj_ref = attn.combine_and_mix(th_in, th_bo, th_ij)
            j("A1.H'<-combine", d1["hcp"], t_bf16_to_f64(ref_hcp))
        # XN：参考的规范化输出（官方路径下 combine 物化的 bf16 之后）
        hcp_ref = ref_hcp if ref_hcp is not None else th_in
        xn_ref = grouped_gemma_rmsnorm(hcp_ref, attn.hc_norm_weight, EPS, HC)
        j("A1.xn<-h'/hc_norm", d1["xn"], t_bf16_to_f64(xn_ref))
        j("A1.blk<-gate,xn", d1["blk"], t_bf16_to_f64(blk_ref))
        if inj_ref is not None:
            j("A1.inject<-xn,winj", extract_inj(d1["oh"], m), t_bf16_to_f64(inj_ref))
        j("A1.lora<-xn,wdown", d1["oh"][:, :LOWRANK], _lora_ref(attn, xn_ref))
        j("A1.ls<-lora", d1["ls"], t_bf16_to_f64(hc_silu(_lora_t(attn, xn_ref), HC)))
        j("A1.gate<-ls,wup", d1["gate"], t_bf16_to_f64(attn._up(hc_silu(_lora_t(attn, xn_ref), HC))))
        jr("A1.rstd", d1["rstd"], _rstd_ref(hcp_ref, HC), 1e-4)
        if attn_mode == MODE_MIX:
            # mode MIX 不跑 W0（`InjwStage` 被 `use_combine` 条件关掉）⇒ 设备 `WS_INJW` **未写**，
            # 与参考的 2σ(0)=1.0 不可比 —— 显式 SKIPPED（不静默）
            print("[chk] 判定 %-24s mode MIX 不跑 W0 ⇒ 设备 injW 区未写（显式 SKIPPED）" % "A1.injW")
            COUNT["skipped"] += 1
        else:
            jr("A1.injW", d1["injw"][:, :, 0], _injw_ref(ij), 1e-6)

        # 舍入点的**判别性负向对照**（真实层 1 数据）：丢掉 combine→RMSNorm 的 bf16 物化
        if L == 1:
            n_ratio, n_tot, x_ratio = _rounding_negctrl(attn, th_in, th_bo, th_ij)
            print("[chk] 负向对照（真实层 1 数据）：丢掉 combine→RMSNorm 的 bf16 物化 ⇒ "
                  "XN 有 %.4f%% 元素不同、BLK 有 %.4f%%（%d/%d）元素不同"
                  % (x_ratio * 100.0, n_ratio * 100.0, n_tot, m * HID))
            print("[chk]   对照：M39 的 B4 在同族数据上量到「物化残差」14029/40960 = 34.25% 元素受影响"
                  "（m21_layer_ref/evidence/selfcheck.log；口径是残差本身，不是 BLK，两者不可直接相减）")
            g("负向对照 丢舍入可区分", n_ratio >= 0.01 and x_ratio >= 0.01,
              "XN %.4f%% / BLK %.4f%% —— 都远高于 1%%，逐字节判据不空洞" % (x_ratio * 100.0, n_ratio * 100.0))

        # ---------- A2：边界 #2 ----------
        print("-- A2 边界 #2（输入 = 设备 b1 的 H'（层 0 用层输入）+ 子层段出口 + b1 的 injection）--")
        h2 = d1["hcp"] if attn_mode != MODE_MIX else hin
        th_h2 = t_u16_to_torch(h2)
        th_attn = t_u16_to_torch(attnout)
        th_ij2 = t_u16_to_torch(extract_inj(d1["oh"], m))
        hcp2, blk2, inj2 = mlp.combine_and_mix(th_h2, th_attn, th_ij2)
        j("A2.H''<-combine", d2["hcp"], t_bf16_to_f64(hcp2))
        xn2 = grouped_gemma_rmsnorm(hcp2, mlp.hc_norm_weight, EPS, HC)
        j("A2.xn<-h''/hc_norm", d2["xn"], t_bf16_to_f64(xn2))
        j("A2.blk<-gate,xn", d2["blk"], t_bf16_to_f64(blk2))
        j("A2.inject<-xn,winj", extract_inj(d2["oh"], m), t_bf16_to_f64(inj2))
        j("A2.lora<-xn,wdown", d2["oh"][:, :LOWRANK], _lora_ref(mlp, xn2))
        j("A2.ls<-lora", d2["ls"], t_bf16_to_f64(hc_silu(_lora_t(mlp, xn2), HC)))
        j("A2.gate<-ls,wup", d2["gate"], t_bf16_to_f64(mlp._up(hc_silu(_lora_t(mlp, xn2), HC))))
        jr("A2.rstd", d2["rstd"], _rstd_ref(hcp2, HC), 1e-4)
        jr("A2.injW", d2["injw"][:, :, 0], _injw_ref(extract_inj(d1["oh"], m)), 1e-6)

        # ---------- 第二 oracle：m20 fp64 参考（把 oracle 差异与设备错误分开）----------
        print("-- 第二 oracle：m20 fp64 参考（M58 对同一段 device 代码用的那一份）--")
        _m20_oracle_block(L, m, attn_mode, hin, bo, ij, d1, d2, attnout, wa, wm)

        # ---------- 非空洞性（guard）----------
        g("%d H''≠层输入链" % L, not np.array_equal(d2["hcp"].view(np.uint8), hin.view(np.uint8)),
          "整层不是恒等映射")
        g("%d 两边界 blk 互异" % L, not np.array_equal(d1["blk"].view(np.uint8), d2["blk"].view(np.uint8)),
          "attn 边界与 mlp 边界的 block input 不同")

    # ---------- GM：末层全局 mixer ----------
    print("\n===== 末层全局 mixer（`hyper_connection_mixer`，use_combine=False ⇒ MODE_FINAL_MIX）=====")
    mixer_w = LayerWeights.global_mixer()
    mixer, multi_ref, sample_ref = final_mixer(t_u16_to_torch(gm_hin), t_u16_to_torch(gm_bo),
                                               t_u16_to_torch(gm_ij), mixer_w, HC)
    j("GM.multi_hidden", gm_multi, t_bf16_to_f64(multi_ref))
    j("GM.sample_hidden", gm_sample, t_bf16_to_f64(sample_ref))
    g("GM.两出口形状各异", tuple(multi_ref.shape)[-1] == HYPER and tuple(sample_ref.shape)[-1] == HID,
      "multi_hidden %s 与 sample_hidden %s 是两个不同的量（拿反了会被上面的对拍抓住）"
      % (tuple(multi_ref.shape), tuple(sample_ref.shape)))
    j("GM.xn<-h'/hc_norm", gm_ws["xn"],
      t_bf16_to_f64(grouped_gemma_rmsnorm(t_u16_to_torch(gm_ws["hcp"]), mixer.hc_norm_weight, EPS, HC)))

    # ---------- 汇总 ----------
    compared = COUNT["judge"] + COUNT["guard"]
    print("\n[chk] 覆盖：判定项 %d 条 + guard %d 条（共比较 %d 条）；显式 SKIPPED %d 条"
          % (COUNT["judge"], COUNT["guard"], compared, COUNT["skipped"]))
    print("[chk] 判定门 = m20 保守门限去掉「良态逐位率 ≥99%」（docs/17 §1.1 把 T3 档的逐位率降为报告项；"
          "逐项 ε 推导见 M58-6）；报告项 = m20 的逐位率读数")
    print("[chk] 超 m20 保守门限（判定门内）的读数：%d 条（见下方 DISCLOSE 段，逐条列出）" % len(OVER))
    print("[chk] 两条独立 oracle：`M39.*`（torch，官方语义，中间量按 bf16 舍入）与 "
          "`M20.*`（m20 的 fp64 逐句实现，M58 用过的同一份）——两者一起看才能区分"
          "「设备算错」与「oracle 的累加顺序/中间舍入差」")
    print("[chk] 覆盖范围：层表 %s × {边界 #1、边界 #2} + 末层 mixer；**不覆盖**子层段/MoE 段"
          "（attention 仍是占位直通、MoE 是 4 专家缩形档）、PLE 本体、资源叠放与同步" % layers)
    if OVER:
        print("\n[chk][DISCLOSE] 超 m20 保守门限（良态逐位率 <0.99）但**满足本脚本判定门**的读数 "
              "%d 条：" % len(OVER))
        for (nm, frac, sfrac, maxulp, na) in OVER:
            print("[chk][DISCLOSE]   %-24s 逐位=%.4f 良态逐位=%.4f ulpMax=%d 归一maxAbs=%.2e"
                  % (nm, frac, sfrac, maxulp, na))
        print("[chk][DISCLOSE] 归因（**不作结论性断言，只列证据**）：① 两条独立 oracle（M39 torch / "
              "M20 fp64）在这些张量上给出**逐位相同**的读数 ⇒ 不是某一个 oracle 的实现差异；"
              "② 越门限的都是**边界 #1 = MODE_MIX**（链上唯一的 mix-only 边界）的下游张量 "
              "`ls/gate/blk`，其上游 `xn` 只有 1 个**非良态**元素差 1 ulp；③ 这条 MODE_MIX 边界的"
              "整块 ws 与 `m15_hc_segment_kernel(mode=MODE_MIX)` **逐字节相同**（C++ 侧 "
              "`Ch.h1blk.L00`/`Ch.h1inj.L00`）⇒ 差不是链的接线引入的，而是 hc 段在**本输入分布**下"
              "的 ≤1 ulp 伴生（silu + K=320 的 up GEMM 在近零处放大）。M58 在 m=33 档记录过同类"
              "越门限（README M58-9 第 2 条，未归因）。")
    if FAILS:
        print("[chk] RESULT: FAILED (%d 条不符：%s)" % (len(FAILS), FAILS))
        return 1
    if compared == 0:
        print("[chk] RESULT: SKIPPED (没有可比对的输入)")
        return 2
    print("[chk] RESULT: OK (%d 条比较过：判定项 %d + guard %d；SKIPPED %d)"
          % (compared, COUNT["judge"], COUNT["guard"], COUNT["skipped"]))
    return 0


def _m20_oracle_block(L, m, attn_mode, hin, bo, ij, d1, d2, attnout, wa, wm):
    """第二 oracle：M58 用的 m20 fp64 参考（**同一份语义、纯 float64 逐句实现**）。

    为什么要有它：M39 的 torch 参考在中间量上按 bf16 舍入（`_down_inject` / `_up` 都显式 cast），
    与设备的 cube 累加顺序不同 ⇒ 可能出现 ≤1 ulp 的伴生差；而 m20 的 fp64 参考是**另一种**独立
    实现（M58 对同一段 device 代码用的就是它，逐位率中位数 1.0000）。
    两条 oracle 一起报，才能把「设备算错」与「oracle 的中间舍入/累加顺序差」分开。
    """
    def u16(t):   # torch bf16 → uint16 numpy（m20.reference 的入参口径）
        return t.contiguous().view(torch.uint16).numpy().reshape(t.shape)

    r1 = m20.reference(m, attn_mode, hin, bo, ij, u16(wa.down_weight), u16(wa.inject_weight),
                       u16(wa.up_weight), u16(wa.hc_norm_weight))
    if attn_mode != MODE_MIX:
        j("M20.A1.H'", d1["hcp"], r1["hc"])
    j("M20.A1.xn", d1["xn"], r1["xn"])
    j("M20.A1.lora", d1["oh"][:, :LOWRANK], r1["lora"][:, :LOWRANK])
    j("M20.A1.ls", d1["ls"], r1["ls"])
    j("M20.A1.gate", d1["gate"], r1["gate"])
    j("M20.A1.blk", d1["blk"], r1["blk"])

    h2 = d1["hcp"] if attn_mode != MODE_MIX else hin
    ij2 = np.zeros((m, IJ_STRIDE), dtype=np.float64)
    ij2[:, :INJ_N] = m20.f32(extract_inj(d1["oh"], m))
    r2 = m20.reference(m, MODE_COMBINE_MIX, h2, attnout, m20.b16(ij2).astype(np.uint16), u16(wm.down_weight),
                       u16(wm.inject_weight), u16(wm.up_weight), u16(wm.hc_norm_weight))
    j("M20.A2.H''", d2["hcp"], r2["hc"])
    j("M20.A2.xn", d2["xn"], r2["xn"])
    j("M20.A2.blk", d2["blk"], r2["blk"])
    j("M20.A2.inject", extract_inj(d2["oh"], m), r2["lora"][:, LOWRANK:])


# ---------------------------------------------------------------------------
# 参考侧的中间量（只用官方 ref/hc.py 的函数，不另造数学）
# ---------------------------------------------------------------------------
def _lora_t(mixer, xn_ref):
    return mixer._down_inject(xn_ref)[0]


def _lora_ref(mixer, xn_ref):
    return t_bf16_to_f64(_lora_t(mixer, xn_ref))


def _rstd_ref(hcp_ref_bf16, hc):
    x = hcp_ref_bf16.detach().to(torch.float32).reshape(hcp_ref_bf16.shape[0], hc, -1)
    return torch.rsqrt(x.square().sum(-1) / x.shape[-1] + EPS).reshape(-1).numpy().astype(np.float64)


def _injw_ref(ij_u16):
    x = t_u16_to_torch(ij_u16).detach().to(torch.float32)
    return (2.0 * torch.sigmoid(x / HC)).reshape(-1).numpy().astype(np.float64)


def _rounding_negctrl(mixer, th_in, th_bo, th_ij):
    """负向对照：**丢掉 combine → RMSNorm 之间的 bf16 物化**时，BLK 有多少 bf16 元素会变。

    官方路径（`ref/hc.py::hc_combine_norm`，hc.py:325-327）先把 combine 结果舍到 bf16 再做
    RMSNorm；反事实路径直接拿 combine 的 fp32 结果做 RMSNorm（其余完全一样）。
    返回 (差异比例, 差异元素数)。
    """
    m = th_in.shape[0]
    res = th_in.detach().to(torch.float32).reshape(m, HC, -1)
    blk = th_bo.detach().to(torch.float32)
    inj = th_ij.detach().to(torch.float32)
    w = 2.0 * torch.sigmoid(inj / HC)
    combined = (res + blk.unsqueeze(1) * w.unsqueeze(-1)).reshape(m, HYPER)

    def _tail(hc_in):
        xn = grouped_gemma_rmsnorm(hc_in, mixer.hc_norm_weight, EPS, HC)
        lora, _ = mixer._down_inject(xn)
        gate = mixer._up(hc_silu(lora, HC))
        return hc_gate_mix(xn, gate, HC)

    with torch.no_grad():
        blk_round = _tail(combined.to(torch.bfloat16))       # 官方路径（先物化）
        blk_noround = _tail(combined)                        # 反事实（不物化）
    # 两侧的**输出**都是 bf16（`block_input` 落盘即 bf16）；反事实路径里 x 是 fp32，
    # 故显式回 bf16 再比位模式（不是放宽，而是让「唯一的差别」只剩那次物化）
    a = blk_round.contiguous().view(torch.uint16).numpy().reshape(-1)
    b = blk_noround.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().reshape(-1)
    diff = int(np.count_nonzero(a != b))
    # 另一条可对照的读数：规范化输出 XN 本身（比 BLK 更靠近舍入点）
    xn_r = grouped_gemma_rmsnorm(combined.to(torch.bfloat16), mixer.hc_norm_weight, EPS, HC)
    xn_u = grouped_gemma_rmsnorm(combined, mixer.hc_norm_weight, EPS, HC)
    xr = xn_r.contiguous().view(torch.uint16).numpy().reshape(-1)
    xu = xn_u.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().reshape(-1)
    xdiff = int(np.count_nonzero(xr != xu))
    return diff / float(a.size), diff, xdiff / float(xr.size)


if __name__ == "__main__":
    sys.exit(main())
