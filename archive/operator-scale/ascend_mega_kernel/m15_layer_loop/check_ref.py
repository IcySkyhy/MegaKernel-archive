#!/usr/bin/env python3
"""M25：48 层循环骨架的 **独立** numpy 交叉校验（消费 M15_DUMP=1 的 dump + dump_manifest.txt）。

与 C++ host 判据（m15_layer_loop.asc 内建）**不共用任何代码**，只用 numpy + 标准库。
它覆盖 C++ 侧无法自证的四件事，每件事都能独立把「循环骨架」判死：

  1. 层类型模式（读真实 config.json 的 layer_types）→ 期望的 full_attention 层集合；
     再核对 dump 里的行为：attention 层必须「hout 逐字节 == hin」（占位直通），
     GDN 层必须「hout != hin」。**内核派发错了（层类型取错）这里立刻 FAIL**。
  2. 残差流交接：Ver A 逐层 y[L] == x[L+1]；Ver B 逐 token 逐层同款（144 条）。
  3. conv_state 位级（不依赖参考实现，纯 device 自洽）：
     new[0] == old[1]、new[1] == old[2]、new[2] == qkvzba[q|k|v]（逐字节），
     跨 token 的 old 取上一 token 同层的 dump → **状态确实跨 token 常驻并被移位**。
  4. 小权重确实是 checkpoint 对应层的切片：conv1d（转置后）、A_log、dt_bias、norm.weight
     与 /workspace/... 的原始 safetensors **逐字节**比对（用 tools/weights/safetensors_reader.py）。
     **权重取错层 / 转置搞反 / bf16→fp32 上采样错，这里立刻 FAIL**。

用法（dump 目录 = cwd）：
    mkdir -p /tmp/m15_dump && cd /tmp/m15_dump
    M15_DUMP=1 <repo>/m15_layer_loop/build/m15_layer_loop <repo>/m15_layer_loop/weights_manifest.txt all
    /usr/local/python3.12.13/bin/python3.12 <repo>/m15_layer_loop/check_ref.py .
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "weights"))

MODEL_DIR = os.environ.get("M15_MODEL_DIR", "/workspace/Qwen3.8-Flash-Next-MXFP4")
HIDDEN, CH, KW, ST, HEAD_D, HEADS = 2560, 10240, 4, 3, 128, 48

PASS = 0
FAIL = 0
REPORT = 0


def judge(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def report(name, detail):
    global REPORT
    REPORT += 1
    print(f"  [report] {name} {detail}")


# ---------------- dump 装载 ----------------
def load_manifest(dump_dir):
    """返回 {name: {"bytes":int, "dtype":str, "shape":str, ...}}"""
    out = {}
    path = os.path.join(dump_dir, "dump_manifest.txt")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("tensor="):
                continue
            kv = {}
            head, _, rest = line.partition(" ")
            kv["name"] = head[len("tensor="):]
            for tok in rest.split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            out[kv["name"]] = kv
    return out


class Dump:
    def __init__(self, dump_dir, man):
        self.dir = dump_dir
        self.man = man

    def has(self, name):
        return name in self.man

    def raw(self, name):
        return np.fromfile(os.path.join(self.dir, name + ".bin"), dtype=np.uint8)

    def bf16(self, name):
        raw = self.raw(name)
        return raw.view(np.uint16).reshape(-1)

    def f32(self, name):
        return self.raw(name).view(np.float32).reshape(-1)

    def bf16_2d(self, name, rows, cols):
        return self.bf16(name).reshape(rows, cols)


def bits_eq(a, b):
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def ulp_close(a, b, maxulp=1):
    """bf16 位型（uint16）按 ulp 距离比较"""
    ai = a.astype(np.int32)
    bi = b.astype(np.int32)
    d = np.abs(ai - bi)
    return int(d.max()), float((d <= maxulp).mean() * 100.0)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------- 1. 层类型模式 ----------------
def check_layer_kinds(man, dump, n_layers=(48,), steps=(3,)):
    print("\n== 1. 层类型模式（config.json layer_types）与内核派发 ==")
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))["text_config"]
    kinds = list(cfg["layer_types"])
    interval = int(cfg["full_attention_interval"])
    assert len(kinds) == int(cfg["num_hidden_layers"])
    attn = {i for i, k in enumerate(kinds) if k == "full_attention"}
    judge("cfg.interval", interval == 4, f"full_attention_interval={interval}")
    judge("cfg.pattern", all((i in attn) == ((i + 1) % interval == 0) for i in range(len(kinds))),
          f"{len(kinds)} 层：{len(attn)} full_attention")

    n_layers = max(n_layers)
    for L in range(n_layers):
        hi, ho = f"L{L:02d}_hin", f"L{L:02d}_hout"
        if not (dump.has(hi) and dump.has(ho)):
            continue
        a, b = dump.bf16(hi), dump.bf16(ho)
        same = bits_eq(a, b)
        expect_same = L in attn
        judge(f"kind.L{L:02d}", same == expect_same,
              f"期望={'直通' if expect_same else 'GDN 计算'}，实际={'直通' if same else '计算'}")
    return attn


# ---------------- 2. 残差流交接 ----------------
def check_handoff(dump, n_layers, steps):
    print("\n== 2. 残差流交接（层间 / token 间 GM 交接逐字节）==")
    # Ver A：y[L] == x[L+1]
    for L in range(n_layers - 1):
        a, b = f"L{L:02d}_hout", f"L{L + 1:02d}_hin"
        if dump.has(a) and dump.has(b):
            judge(f"A.handoff.L{L:02d}->L{L + 1:02d}", bits_eq(dump.bf16(a), dump.bf16(b)))
    # Ver B：每 token 内逐层 + 层 0 输入 == h0[t]
    for t in range(steps):
        h0 = f"h0_t{t}"
        if dump.has(h0):
            judge(f"B.entry.t{t}", bits_eq(dump.bf16(h0), dump.bf16(f"B_t{t}_L00_hin")))
        for L in range(n_layers - 1):
            a, b = f"B_t{t}_L{L:02d}_hout", f"B_t{t}_L{L + 1:02d}_hin"
            if dump.has(a) and dump.has(b):
                judge(f"B.handoff.t{t}.L{L:02d}", bits_eq(dump.bf16(a), dump.bf16(b)))


# ---------------- 3. conv_state 位级（device 自洽） ----------------
def check_conv_state(dump, n_layers, steps):
    print("\n== 3. conv_state 位级：移位自洽 + 跨 token 常驻 ==")
    prev = {}
    for t in range(steps):
        for L in range(n_layers):
            nm = f"B_t{t}_L{L:02d}_cs"
            if not dump.has(nm):
                continue
            cs = dump.bf16_2d(nm, ST, CH)
            qkv = dump.bf16(f"B_t{t}_L{L:02d}_qkvzba")[:CH]
            judge(f"cs.cur.t{t}.L{L:02d}", bits_eq(cs[2], qkv))     # 新行 = 本 token 的 q|k|v
            if L in prev:
                judge(f"cs.shift0.t{t}.L{L:02d}", bits_eq(cs[0], prev[L][1]))
                judge(f"cs.shift1.t{t}.L{L:02d}", bits_eq(cs[1], prev[L][2]))
            prev[L] = cs
    # 非空洞性：最后一个 token 的 cs/ssm 与首 token 不同（状态确实在演化）
    for L in list(prev.keys()):
        first = f"B_t0_L{L:02d}_cs"
        last = f"B_t{steps - 1}_L{L:02d}_cs"
        if dump.has(first) and dump.has(last):
            judge(f"cs.evolved.L{L:02d}", not bits_eq(dump.bf16(first), dump.bf16(last)))


# ---------------- 4. 小权重 == checkpoint 对应层切片 ----------------
def check_weights(dump, n_layers):
    print("\n== 4. 小权重 == checkpoint 对应层切片（逐字节）==")
    from safetensors_reader import ShardReader
    rdr = ShardReader(MODEL_DIR)
    for L in range(n_layers):
        nm = f"W_L{L:02d}_convw"
        if not dump.has(nm):
            continue
        ck = rdr.load(f"model.language_model.layers.{L}.linear_attn.conv1d.weight")   # [CH,1,KW] bf16
        ck = np.asarray(ck).reshape(CH, KW).astype(np.uint16)
        want = ck.T.reshape(-1)          # [KW,CH]（转置：convW[j*CH+ch] = ck[ch][j]）
        judge(f"W.L{L:02d}.convw", bits_eq(dump.bf16(nm), want))

        a = np.asarray(rdr.load(f"model.language_model.layers.{L}.linear_attn.A_log")).reshape(-1).astype(np.uint16)
        al = dump.f32(f"W_L{L:02d}_alog")
        ok = al.size == 64 and bool(np.array_equal(al[:HEADS], bf16_to_f32(a))) and bool((al[HEADS:] == 0).all())
        judge(f"W.L{L:02d}.alog", ok, "(bf16→fp32 上采样 + 尾零)")

        d = np.asarray(rdr.load(f"model.language_model.layers.{L}.linear_attn.dt_bias")).reshape(-1).astype(np.uint16)
        dl = dump.f32(f"W_L{L:02d}_dtb")
        judge(f"W.L{L:02d}.dtbias", dl.size == 64 and bool(np.array_equal(dl[:HEADS], bf16_to_f32(d))))

        g = np.asarray(rdr.load(f"model.language_model.layers.{L}.linear_attn.norm.weight")).reshape(-1).astype(np.uint16)
        judge(f"W.L{L:02d}.gammaG", bits_eq(dump.bf16(f"W_L{L:02d}_gg"), g))

        cb = dump.bf16(f"W_L{L:02d}_convb")
        judge(f"W.L{L:02d}.convbias_zero", bool((cb == 0).all()), "(checkpoint 无 conv bias → 全零)")
        # gamma1/gamma2 是合成的（checkpoint 无该形状层级 norm）→ 只报告，不判定
        report(f"W.L{L:02d}.g1_g2", "合成 gamma（checkpoint 无 [2560] 层级 norm，见 README §缺口）")


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    n_layers = int(os.environ.get("M15_REF_LAYERS", "48"))
    steps = int(os.environ.get("M15_REF_STEPS", "3"))
    man = load_manifest(dump_dir)
    dump = Dump(dump_dir, man)
    print(f"[check_ref] dump 目录 {dump_dir}：{len(man)} 个张量；层数 {n_layers}，token {steps}")

    check_layer_kinds(man, dump, [n_layers], [steps])
    check_handoff(dump, n_layers, steps)
    check_conv_state(dump, n_layers, steps)
    check_weights(dump, n_layers)

    print(f"\n[check_ref] ===== 判定项 {PASS + FAIL} 条：PASS {PASS} / FAIL {FAIL}（另有 {REPORT} 条参考项）=====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
