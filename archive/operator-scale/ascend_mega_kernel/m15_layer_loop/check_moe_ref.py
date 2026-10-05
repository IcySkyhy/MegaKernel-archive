#!/usr/bin/env python3
"""M40/M90：融合 per-layer kernel 的 **MoE 段** 独立 numpy 交叉校验。

## M90 的改动：参考链**重指向**（本 mission 的 task 1）

原来这里 `import m13_moe_layer/check_ref.py`「复用 m13 的参考链」。M90 按 M77 §5.1 的建议
**改指向 `m17_moe_real/check_ref.py` 的 numpy/double 独立参考链**：m17 与 m15 的 MoE 段同源
（都是 m13 段），规模抬到 512/10 后需要同一套真实规模参考；而 m17 那条链自带**量化规则见证**
（M60 的 `W0`–`W6`）与 4 条负向自检，且它的 `quant_hw` / `dequant` / `tol_*` 是设备路径的镜像。

重指向后本文件仍然 `import tools/golden/moe_block_ref.py`（权威编解码 + 路由语义），并且：

* **路由参考不走 m17**：`docs/17` §7.1 登记的 FTZ 建模在
  `moe_block_ref.py::ftz_f32` / `router_topk` 里（三个 materialize 点位 FTZ#1/#2/#3），
  而 `m17_moe_real/check_ref.py` 的路由**没有**建模 FTZ（它的 `e2e_chain` 与分段判据直接
  `np.exp` 后 `argsort`）。真档上不建模 FTZ 会让 top-10 集合分叉 ⇒ 本文件的路由判据一律走
  `MB.router_topk`。**m17 那一侧未改**（不在本 mission 的写权限内），已报塔。
* **T3 的 ε 不新拍**：全部沿用 m17/M60 已归档的推导（`EPS_GEMV` / `EPS_FAST` /
  `eps_cube(k)` / `EPS_FOLD`），只把**网格项**按 `docs/17` §1.1 的值写明。
* 每条判据在名字里带 **T1 / T3** 档位标签（`docs/17` §1.1 的分档）。

## M96 的改动：修掉 M90 改写引入的三处**口径漂移**（本 mission）

M90 的改写把几对「**静态上界/槽位几何**」与「**运行期/有效**」的量绑到了同一个裸符号上，
于是 `26d644b..0bd80c6` 之后本脚本在 main 上**直接崩**（`ValueError: cannot reshape array of
size 32 into shape (20)`，rc=1），且即使不崩也判不出东西。三处都已在**本文件**内修好：

1. **H(= down 输入) scale 的「槽宽」被当成「有效前缀」**：`M17R.dequant(..., scale_stride)`
   吃的是**槽宽**（`M15M::DN_SCALE_STRIDE` = 32 B/专家行），M90 传了 `INTER / GROUP` = 20。
   现在用限定命名空间的 `DN_SCALE_STRIDE` / `DN_SCALE_EFF` 两个常量把概念分开（见下方常量块）。
2. **运行期 topk 被当成模板上界**：meta 只给 `topk_max`（= `M15M::TOPK_MAX` = 4，**模板上界**），
   M90 拿它当运行期 topk（缩形档实际 = 2）⇒ 比 dump 里实际的路由多读 2 条 `0xCD` 污染 id。
   现在 `TOPK`（运行期，与 harness 的 `M15_TOPK` 同源、默认 2）与 `TOPK_TMPL`（定尺用上界）分开。
3. **`t3` 的 1-D / 2-D 形状口径**：参考算成 `(1, N)`（布局表声明 `1xN`）而设备读是一维 ⇒
   `router_logits` / `GU_shd` / `Y_shd` 三条 T3 变成 shape FAIL（数值其实一致，3 条/层）。
   `t3` 保持**形状严格**（不许静默广播），改成在设备读处补成 `(1, N)`。

修完在 main tip 的 dump 上读数为 **128 条判定项 + 20 条非空洞性，0 FAIL**（rc=0）。
`--negctl` 是本文件自带的负向对照：把上面这条判据的**取数环节**在内存里弄坏，判据必须变红
（正对照必须绿），见 `negctl()` 的 docstring。

## 判据分档（`docs/17-verification-standard.md` §1.1）

* **T1 整数域/位域**（逐位/逐字节）：`topk_ids`、`expert_token_counts`、`expert_offsets`、
  `perm_src_token`、`perm_expert`、`inv_slot`、`w_tk_packed`、`x_sorted`、
  `A_qx`/`A_scale`/`H_qx`/`H_scale`（routed + 共享）、`res1`、`res2`。
* **T3 推导界**：`router_logits`（含 `Exp`，触发①）、`topk_weights`（`Exp` + 除法，触发①）、
  `GU`/`Y`（cube 累加，触发③）、`H`/`shared`（超越函数）、`routed`（长 fp32 折叠）、`moe`。

## ulp / 网格项口径（引用时必须写明量 + `e` 约定，`docs/17` §1.1 该段末条）

`docs/17` §1.1 的 `BfUlp(v) = 2^(e-7)`（`e` = **binade 指数**，`2^e ≤ |v| < 2^(e+1)`）是**一个格点间距**。
m17 的 `spacing_bf16` 给的是 `2^(e-8)` = **0.5·BfUlp** = **最大 RNE 舍入误差**（它的 docstring 称
「网格步长」，是命名/单位差）。本文件因此把两件事分开写：

* 参考**为实值**时用 `0.5·BfUlp`（= `M17.tol_bf16(..., nulp=1.0, ...)`）；
* 参考**本身已落在 bf16 格点**时用 `1.0·BfUlp`（= `M17.tol_bf16(..., nulp=2.0, ...)`）。

`--selftest` 会把这两个量并排打印出来。

## 参考的输入从哪来（`docs/17` §1.3 三分法）

设备路径的参考吃的输入逐条归类如下（**这份表放在本脚本的 docstring 里**，因为
`m15_layer_loop/README.md` 不在本 mission 的写权限内）：

| prov | 类别 | 本脚本里的具体输入 | 覆盖它的判据 |
|---|---|---|---|
| **①** | 声明输入 | `L*_layer_in.bin` / `L*_layer_out.bin`（host 落的层入口/出口）；`--model-dir` 的真实 checkpoint 权重（按张量名 pread，与 m15 的 manifest 路径独立） | — |
| **②** | 上游输出（**设备**产物） | `x_norm`（S1）；`topk_ids` / `perm_*` / `expert_token_counts` / `expert_offsets`（S2/S3，作本文件索引判据的**被判量**，不是参考输入）；`A_qx`/`A_scale`（S5）与 `h_swiglu`（S7）作量化字节判据的参考输入；`GU`（S6）作 SwiGLU 判据的输入；`h_qx`/`h_scale`（S7）作 Y 判据的输入；`Y`（S8）、`w_tk_packed`、`sgate`、`Y_shd` 作 combine 判据的输入 | 逐条点名（均为**本文件**的判据，逐 op 口径：每段只比本段残差）：`A_qx`/`A_scale` → `T1 A_qx/A_scale 逐字节`；`GU` → `T3 GU slot{e}`；`h_swiglu` → `T1 H_qx/H_scale 逐字节` 与 `T3 H_swiglu slot{e}`；`h_qx`/`h_scale` → `T1 H_qx/H_scale 逐字节` 与 `T3 Y slot{e}`；`Y` → `T3 Y slot{e}` 与 `T3 routed`；`w_tk_packed` → `T1 w_tk_packed 低16位`；`sgate`/`Y_shd` → `T3 shared_output` / `T3 moe_output`。**`x_norm` 本文件不覆盖 —— 见下** |
| **③** | 被判量自身的产物 / 与其共享同一推导链 | 无：所有判据都是「device 字节 vs 独立复算（从 `x_norm` 或 ① 出发）」，没有拿被判量自己的产物当参考输入 | — |

> **口径（与 `tools/golden/moe_real_accept.py` 一致）**：本节只说「哪一类输入喂给哪一些判据」；
> **数字一律以运行时打印为准**，文档不复写会漂的计数。
> 本脚本是**设备路径**（PENDING），它的 ③ = 空集；这一点由下面这一行机器可读标记承载，
> `tools/golden/moe_real_accept.py run` 会**真的读它**并与主腿的 ③ 集合分开比对
> （本行被写坏 / 被删 ⇒ 那条判据 FAIL ⇒ `run` 退 1）：
> `<!-- M90-CHAIN-IDS[devpath]: - -->`（`-` = 空集）

* **登记的覆盖缺口（如实写，不粉饰）**：`x_norm` 是 ② 类输入，而**本文件没有一条判据独立复算
  S1 的 RMSNorm**（它直接吃 device 的 `x_norm` 去算 router）。按 `docs/17` §1.3，② 类输入
  必须在 README 里被**点名判据**覆盖才合法 —— 该覆盖在 `m15_layer_loop` 的层链参考
  （`m15_layer_loop/check_ref.py`，M25 的 Ver B / 残差流交接，见 `m15_layer_loop/README.md` §5.5）
  那一侧，而**那不在本 mission 的写权限内**，故此处只登记缺口并报塔，**不**自行声明它已被覆盖。
* 参考的**规则**来源（§1.2）在 m17 的量化器镜像上（`M17R.quant_hw`），其 caveat 见
  `tools/golden/moe_real_accept.py` 的 docstring 与 `tools/golden/README.md`（同源转录，非独立第二来源）。

## 设备路径与 SKIPPED

本脚本消费的是**融合 kernel 落的整块 MoE 段 ws + `moe_layout.txt`**。512/10 的 device 段
（`m15_moe_resources.h` 的规模常量 + `.asc` 的 host arena）在另一条 mission 上，因此：
无 dump / layout 规模与 512/10 档不符时，本脚本**打 SKIPPED 并退 2**（不发合格证）。

用法：
    # 设备 path（需要先跑融合 kernel 落 L*_moe_ws.bin 与 moe_layout.txt）
    /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_moe_ref.py <dumpdir> \
        --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4
    # 主机侧自检（无 device）：证明重指向是**行为保持**的 swap + 打印 ulp 口径
    #   ⚠ 需要先 `python3 tools/golden/gen_dataset.py --groups real` 生成真实档（bins 不入仓）；
    #     未生成时本模式打 SKIPPED 并退 2，不抛异常。
    /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_moe_ref.py --selftest
    # 负向对照（第五变体纪律）：把取数环节在内存里弄坏 ⇒ 同一份 dump 上判据必须变红
    /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_moe_ref.py <dumpdir> \
        --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4 --negctl
    # 运行期 topk 与产生 dump 的 harness 对齐（harness 的开关同名，默认 2）
    M15_TOPK=2 /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_moe_ref.py <dumpdir> \
        --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4

退出码：0 = 比过且通过 / 1 = 比过有差异 / 2 = 没得比（SKIPPED）。
"""
import argparse
import importlib.util
import os
import re
import sys

# 不写 __pycache__：本脚本 import 的 m13/m17 check_ref 与 tools/golden 都在**别人的 scope**里，
# 跑一次就留一个 .pyc 目录不合适（本 mission 只改 m15_layer_loop/check_moe_ref.py 与 tools/golden/**）。
sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "golden"))
sys.path.insert(0, os.path.join(ROOT, "tools", "weights"))

import moe_block_ref as MB  # noqa: E402


def _load_py(name, path):
    """按路径 import（m13/m17 都有自己的 `check_ref.py`，不能靠 sys.path 撞名）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 参考链 = m17 的 numpy/double 独立链（量化器 + T3 界的来源）。**只读 import。**
M17R = _load_py("m90_m15_m17ref", os.path.join(ROOT, "m17_moe_real", "check_ref.py"))
# m13 的旧链只在 `--selftest` 里用来证明重指向是行为保持的（不参与判据）。
M13R = _load_py("m90_m15_m13ref", os.path.join(ROOT, "m13_moe_layer", "check_ref.py"))

from safetensors_reader import ShardReader  # noqa: E402

HIDDEN, INTER, GROUP = 2560, 640, 32
GU_N = 2 * INTER
GUN = GU_N

# 设备 dump 里 H（= down 输入）scale 的**两个宽度** —— 它们**不是**同一个概念（M96）：
#   * **槽宽**（每专家行的字节数）= `M15M::DN_SCALE_STRIDE` = 32。它是设备侧的**布局常量**：
#     `m15_layer_loop/m15_moe_resources.h` §0 定义它，`m15_layer_loop/m15_moe_host.h` 的
#     `MoeLayoutRows()` 用同一常量算出布局行（`H_scale` bytes=32），kernel 侧按它步进写
#     （`lift_moe_segment.py`：`ws + WS_HS + off[e] * DN_SCALE_STRIDE`）。
#   * **有效前缀**（每行真正有值的 e8m0 组数）= INTER / GROUP = 20（32 B 的槽里只有前 20 B 有效，
#     其余是 `aclrtMemset(0xCD)` 污染，见 `[check_moe_ref]` 的 T4 位置正确性判据）。
# `M17R.dequant(packed, scale, rows, k, packed_stride, scale_stride)` 的第 6 个参数吃的是**槽宽**
# （m17 自己的设备路径同样传 `DN_SCALE_STRIDE`，见 `m17_moe_real/check_ref.py` 里 `check_layer`
# 的 `hd = dequant(hq[ex], hs[ex], t, INTER, INTER // 2, DN_SCALE_STRIDE)`）。
# ⇒ **把「有效前缀」当行距传**会把 32 B 的槽按 20 B 的行距 reshape ⇒ 直接 ValueError。
# M90 的改写正是丢了这个 32（`26d644b` 版传的是 `M13R.dequant_device(..., 32, INTER)`）—— M96 根因。
DN_SCALE_STRIDE = 32                    # 槽宽（M15M::DN_SCALE_STRIDE；与 GROUP 同值纯属巧合）
DN_SCALE_EFF = INTER // GROUP           # 20：**有效前缀**，只用于「每行取多少组」的切片

# E / M_MAX 由 `moe_layout.txt` 的 meta 行给出（512/10 档就靠这条；不再硬编码 4/2）。
# **topk 也有两个量**（同样是 M96 的坑，M90 的改写把它们绑到了同一个裸 `TOPK`）：
#   * `TOPK_TMPL` = meta 行的 `topk_max` = **模板上界**（`M15M::TOPK_MAX`）：设备的 buffer 按它定尺
#     （`x_sorted` / `h_swiglu` / `inv_slot` / `perm_*` 的行数都是 M_MAX*TOPK_MAX），meta 只给这一个；
#   * `TOPK` = **运行期 top-k**：路由、`topk_ids`/`topk_weights` 的条数、`inv_slot` 的 m*topk 用它。
#     它与 harness 的 `M15_TOPK` 同源（`m15_layer_loop/m15_layer_loop.asc`：`O.topk = H_EnvU32(
#     "M15_TOPK", 2u)`，越界回落 2），本文件默认也取 2、可由同一环境变量覆盖。同值还有两处仓内
#     见证：`m15_layer_loop/slice_layer_manifest.py:81` 的 `TOPK_MOE = 2`（MoE 段的 checkpoint 切片
#     按 2 个专家算），以及 dump 自己（`expert_offsets` 的前缀和 Σt_e = m×2）。
# 用模板上界当运行期 topk ⇒ 比 dump 里实际的路由多读 2 条 0xCD 污染 id，判据整片错位（M96 登记）。
# **同名符号警告**：裸 `TOPK` 在别的命名空间里**不是同一个量** —— `m17_moe_real/check_ref.py:80`
# 的模块级 `TOPK = 10`（真实档的 top-10）。本文件只用**自己的** `TOPK`，从不写 `M17R.TOPK`；
# 同理 `DN_SCALE_STRIDE` 在 M13M/M15M 头文件与 m17 里各有一份（当前都是 32），本文件用
# `_assert_scale_slot()` 把它钉在**被读的布局表**上，而不是靠名字相同。
E, TOPK, TOPK_TMPL, MMAX = 4, 2, 4, 64
CKPT_PREFIX = "model.language_model.layers"
DUMP = "."


def bf_ulp(x):
    """`docs/17` §1.1 的格点间距：`BfUlp(v) = 2^(e-7)`，`e` = binade 指数。"""
    a = np.abs(np.asarray(x, dtype=np.float64))
    return np.where(a > 0, 2.0 ** (np.floor(np.log2(np.where(a > 0, a, 1.0))) - 7.0), 0.0)


# ---------------------------------------------------------------------------
# 布局表与 ws 切片
# ---------------------------------------------------------------------------
def load_layout(path):
    tensors, meta = {}, {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("meta "):
                for k, v in re.findall(r"(\w+)=(\S+)", line):
                    meta[k] = v
                continue
            kv = dict(re.findall(r"(\w+)=([^\s]+)", line))
            tail = re.search(r"shape=(\S.*)$", line)
            if tail:
                kv["shape"] = tail.group(1).strip()
            kv["ws_off"] = int(kv["ws_off"])
            kv["bytes"] = int(kv["bytes"])
            kv["stride"] = int(kv["stride"])
            kv["count"] = int(kv["count"])
            tensors[kv["name"]] = kv
    return tensors, meta


_DT = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "U8": np.uint8}


def ts(ws, layout, name):
    """按布局表切出一块张量。mode=compact 切出 [count, bytes]（每专家一行）；offsets 由 ws 读。"""
    e = layout[name]
    dt = _DT[e["dtype"]]
    if e["mode"] == "contig":
        b = ws[e["ws_off"]:e["ws_off"] + e["bytes"]]
        return np.frombuffer(b, dtype=dt).copy()
    # mode=compact：紧凑专家槽 Σt_e —— 第 e 个专家的行起点 = ws_off + offs[e]*bytes，
    # offs[] 是 ws 偏移 stride 处的 int32[NUM_EXPERTS+1]（S3 的前缀和，与 kernel 同一来源）
    offs = np.frombuffer(ws[e["stride"]:e["stride"] + 4 * (E + 1)], dtype=np.int32)
    rows = []
    for i in range(e["count"]):
        o = e["ws_off"] + int(offs[i]) * e["bytes"]
        rows.append(np.frombuffer(ws[o:o + e["bytes"]], dtype=dt).copy())
    return np.stack(rows)


def bf16(bits):
    return MB.bf16_bits_to_f32(np.asarray(bits, dtype=np.uint16))


def t3(results, tag, dev, ref, budget, note):
    """T3：`|out − ref| ≤ budget` 逐元素（`docs/17` §1.1 的 T3 行）。"""
    dev = np.asarray(dev, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if dev.shape != ref.shape:
        results.append((tag, False, f"shape {dev.shape} vs {ref.shape}"))
        return
    b = np.broadcast_to(np.asarray(budget, dtype=np.float64), dev.shape).astype(np.float64)
    floor = 1e-6 * float(np.abs(ref).max()) if ref.size else 0.0
    b = np.maximum(b, max(floor, 1e-30))
    ad = np.abs(dev - ref)
    nonfin = ~np.isfinite(dev) | ~np.isfinite(ref)
    nbad = int(nonfin.sum() + np.sum(~nonfin & (ad > b)))
    with np.errstate(invalid="ignore"):
        ratio = float(np.nanmax(np.where(nonfin, np.inf, ad / b)))
    results.append((tag, nbad == 0,
                    f"n={dev.size} 越界={nbad} | |Δ| max={ad.max() if ad.size else 0.0:.3e} "
                    f"| 最差占预算={ratio:.3f} | {note}"))


def t1(results, tag, got, want, what="元素"):
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        results.append((tag, False, f"shape {got.shape} vs {want.shape}"))
        return
    nbad = int((got != want).sum())
    results.append((tag, nbad == 0, f"{nbad}/{got.size} {what}不符; n={got.size}"))


# ---------------------------------------------------------------------------
# 主校验
# ---------------------------------------------------------------------------
def check_layer(L, ws, layout, rdr, results, refs, nonvac):
    tag = f"L{L:02d}"
    g = lambda n: ts(ws, layout, n)  # noqa: E731

    # ---- 真实 checkpoint 权重（按张量名独立 pread；不经过 m15 的 manifest）----
    p = f"{CKPT_PREFIX}.{L}"
    router_w = bf16(rdr.load(f"{p}.mlp.gate.weight", slice(0, E)))
    sgate_w = bf16(rdr.load(f"{p}.mlp.shared_expert_gate.weight"))[0]
    wgu = rdr.load(f"{p}.mlp.experts.gate_up_proj", slice(0, E))
    sgu = rdr.load(f"{p}.mlp.experts.gate_up_proj.weight_scale", slice(0, E))
    wdn = rdr.load(f"{p}.mlp.experts.down_proj", slice(0, E))
    sdn = rdr.load(f"{p}.mlp.experts.down_proj.weight_scale", slice(0, E))
    wgs = rdr.load(f"{p}.mlp.shared_expert.gate_proj.weight")
    wgn = rdr.load(f"{p}.mlp.shared_expert.gate_proj.weight_scale")
    wus = rdr.load(f"{p}.mlp.shared_expert.up_proj.weight")
    wun = rdr.load(f"{p}.mlp.shared_expert.up_proj.weight_scale")
    wds = rdr.load(f"{p}.mlp.shared_expert.down_proj.weight")
    wdn_s = rdr.load(f"{p}.mlp.shared_expert.down_proj.weight_scale")
    assert wgu.shape == MB.expert_weight_shapes(E)["experts.gate_up_proj"], wgu.shape
    assert wgs.shape == MB.shared_weight_shapes()["shared_expert.gate_proj"], wgs.shape

    # ---- 段输入锚点：S1 的输出（device 自己的 x_norm，bf16）----
    x_norm_bits = g("x_norm").reshape(1, HIDDEN)
    x_norm = bf16(x_norm_bits)
    li_p = os.path.join(DUMP, f"{tag}_layer_in.bin")
    lo_p = os.path.join(DUMP, f"{tag}_layer_out.bin")
    have_layer_io = os.path.exists(li_p) and os.path.exists(lo_p)
    layer_in = np.frombuffer(open(li_p, "rb").read(), dtype=np.uint16) if have_layer_io else None
    layer_out = np.frombuffer(open(lo_p, "rb").read(), dtype=np.uint16) if have_layer_io else None

    # ---- 1. router：T1 索引（路由参考走 MB = docs/17 §7.1 的 FTZ 实现）+ T3 数值 ----
    logits_ref, ids_ref, w_ref = MB.router_topk(x_norm.astype(np.float32),
                                                router_w.astype(np.float32), TOPK)
    logits_dev = g("router_logits")[:E].reshape(1, E)   # 布局表声明 1xE；参考 lg64 也是 (1, E)
    ids_dev = g("topk_ids")[:TOPK]
    w_dev = g("topk_weights")[:TOPK]
    t1(results, f"{tag} T1 topk_ids 精确相等（FTZ 由 MB.router_topk 建模）",
       ids_dev, ids_ref[0], "id")
    # T3：logits 的 fp64 独立复算（不经过 MB），ε = EPS_GEMV·Σ|a·w|
    x64 = x_norm.astype(np.float64)
    w64 = router_w.astype(np.float64)
    lg64 = x64 @ w64.T
    lg64 = lg64 - lg64.max(axis=1, keepdims=True)
    t3(results, f"{tag} T3 router_logits（含 Exp/max-shift）", logits_dev, lg64,
       M17R.tol_l1(lg64, np.abs(x64) @ np.abs(w64).T, M17R.EPS_GEMV), "ε_gemv·Σ|a·w|")
    t3(results, f"{tag} T3 topk_weights（Exp + 除法 ⇒ §1.1 触发①）", w_dev, w_ref[0],
       M17R.tol_bf16(w_ref[0], 1.0, M17R.EPS_FAST * np.abs(w_ref[0])),
       "0.5·BfUlp + EPS_FAST·|ref|（m17 单位 nulp=1.0）")

    # ---- 2. 索引生成：T1 逐位 ----
    perm = MB.moe_permute(ids_ref, E)
    total = int(perm["expert_token_counts"].sum())
    t1(results, f"{tag} T1 expert_token_counts 精确",
       g("expert_token_counts")[:E], perm["expert_token_counts"], "count")
    t1(results, f"{tag} T1 expert_offsets 精确",
       g("expert_offsets")[:E + 1], perm["expert_offsets"], "offset")
    t1(results, f"{tag} T1 perm_src_token 精确",
       g("perm_src_token")[:total], perm["perm_src_token"], "slot")
    t1(results, f"{tag} T1 perm_expert 精确",
       g("perm_expert")[:total], perm["perm_expert"], "slot")
    # inv_slot 与 perm/topk_ids 互洽（紧凑槽：inv 存的是计数排序里的产出位置）
    # 融合路的这一份只跑 decode m=1（`m15_layer_loop/README.md` §8 第 9 条：prefill 是另一个入口）
    m_tok = 1
    inv = g("inv_slot")[:m_tok * TOPK]
    off = perm["expert_offsets"]
    inv_ref = np.full(m_tok * TOPK, -1, dtype=np.int32)
    for i in range(total):
        e = int(perm["perm_expert"][i])
        t = int(perm["perm_src_token"][i])
        k = int(np.where(ids_ref[t] == e)[0][0])
        inv_ref[t * TOPK + k] = i     # 紧凑行号（见 lift_moe_segment.py 规则 6a）
    t1(results, f"{tag} T1 inv_slot 与 perm 互洽（紧凑行号）", inv, inv_ref, "元素")
    wtk = g("w_tk_packed")
    wtk_ref = MB.f32_to_bf16_bits(w_ref[0]).astype(np.int32)
    t1(results, f"{tag} T1 w_tk_packed 低16位 = bf16(weights)",
       wtk[:TOPK] & 0xFFFF, wtk_ref, "bit")

    # ---- 3. permute：T1 逐位 ----
    xsorted = g("x_sorted").reshape(MMAX * TOPK_TMPL, HIDDEN)[:total]   # 定尺用**模板上界**
    t1(results, f"{tag} T1 x_sorted == gather(x_norm, perm_src) 逐位",
       xsorted, x_norm_bits[perm["perm_src_token"]], "bf16")

    # ---- 4. A 侧量化器：T1 逐字节（硬件规范，独立 numpy）----
    aqx = g("A_qx")
    asc = g("A_scale")
    aq_bad = as_bad = aq_n = as_n = 0
    aqx_shd = g("A_qx_shd")
    asc_shd = g("A_scale_shd")
    for idx, row in enumerate(xsorted):
        e = int(perm["perm_expert"][idx])
        pq, sq = M17R.quant_hw(np.ascontiguousarray(bf16(row).astype(np.float32)).reshape(1, HIDDEN))
        aq_bad += int((pq[0] != aqx[e][:HIDDEN // 2]).sum())
        as_bad += int((sq[0] != asc[e][:HIDDEN // GROUP]).sum())
        aq_n += pq[0].size
        as_n += sq[0].size
    pq, sq = M17R.quant_hw(x_norm.astype(np.float32))
    aq_bad += int((pq[0] != aqx_shd[:HIDDEN // 2]).sum())
    as_bad += int((sq[0] != asc_shd[:HIDDEN // GROUP]).sum())
    aq_n += pq[0].size
    as_n += sq[0].size
    results.append((f"{tag} T1 A_qx 逐字节 vs numpy(硬件规范)", aq_bad == 0,
                    f"{aq_bad}/{aq_n} 字节不符"))
    results.append((f"{tag} T1 A_scale 逐字节 vs numpy(硬件规范)", as_bad == 0,
                    f"{as_bad}/{as_n} 字节不符"))
    pq_g, sq_g = MB.quantize_ocp(x_norm.astype(np.float32))
    refs.append((f"{tag} A_qx_shd/golden(floor 序列) 规范",
                 f"{int((pq_g[0] != aqx_shd[:HIDDEN // 2]).sum())}/{pq_g[0].size}"))
    refs.append((f"{tag} A_scale_shd/golden(floor 序列) 规范",
                 f"{int((sq_g[0] != asc_shd[:HIDDEN // GROUP]).sum())}/{sq_g[0].size}"))

    # ---- 5. gate_up GEMM / SwiGLU / H 量化 / down GEMM（T3 推导界）----
    gu_dev = bf16(g("GU")).reshape(E, GU_N)
    h_dev = bf16(g("h_swiglu")).reshape(MMAX * TOPK_TMPL, INTER)[:total]  # 定尺用**模板上界**
    hqx = g("H_qx")
    hs = g("H_scale")
    y_dev = bf16(g("Y")).reshape(E, HIDDEN)
    hq_bad = hs_bad2 = hq_n = hs_n = 0
    for e in range(E):
        t = int(perm["expert_token_counts"][e])
        if t == 0:
            continue
        s = int(off[e])
        adq = M17R.dequant(aqx[e], asc[e], t, HIDDEN, HIDDEN // 2, scale_stride=HIDDEN // GROUP)
        w = M17R.dequant(wgu[e], sgu[e], GU_N, HIDDEN, HIDDEN // 2, scale_stride=HIDDEN // GROUP)
        gu_ref = adq @ w.T
        t3(results, f"{tag} T3 GU slot{e}（cube 累加）", gu_dev[e].reshape(t, GU_N), gu_ref,
           M17R.tol_bf16(gu_ref, 1.0, M17R.eps_cube(HIDDEN) * (np.abs(adq) @ np.abs(w).T)),
           "ε_cube(2560)·Σ|a·w|")
        # SwiGLU 的输入取 device 自己的 bf16 GU（逐 op 口径：每段残差只含本段）
        gd = gu_dev[e].reshape(t, GU_N)[:, :INTER].astype(np.float64)
        ud = gu_dev[e].reshape(t, GU_N)[:, INTER:].astype(np.float64)
        h_ref = (gd / (1.0 + np.exp(-gd))) * ud
        t3(results, f"{tag} T3 H_swiglu slot{e}（超越函数）", h_dev[s:s + t], h_ref,
           M17R.tol_bf16(h_ref, 1.0, M17R.EPS_FAST * np.abs(h_ref)), "0.5·BfUlp + EPS_FAST·|ref|")
        hb = np.ascontiguousarray(h_dev[s:s + t].astype(np.float32))
        ph, sh = M17R.quant_hw(hb)
        hq_bad += int((ph != hqx[e][:t * (INTER // 2)].reshape(t, INTER // 2)).sum())
        # 逐字节判据同样按**槽宽**取数：每行是 DN_SCALE_STRIDE B 的槽，有效前缀只取前 DN_SCALE_EFF 个
        hs_rows = hs[e].reshape(-1, DN_SCALE_STRIDE)[:t, :DN_SCALE_EFF]
        hs_bad2 += int((sh != hs_rows).sum())
        hq_n += ph.size
        hs_n += sh.size
        # scale_stride = **槽宽**（DN_SCALE_STRIDE=32），不是有效前缀（INTER//GROUP=20）—— 见文件头的常量说明。
        # 布局表按 M15M::DN_SCALE_STRIDE 给每个专家定尺 ⇒ 一个专家的槽必须装得下它的 t 行
        # （m>1 且 t_e>1 时布局表要把 H_scale 的 bytes 抬到 TOTAL_MAX 行；现在只跑 m=1）。
        assert t * DN_SCALE_STRIDE <= hs[e].size, (
            f"专家 {e} 的 t={t} 行在 {hs[e].size} B 的 H_scale 槽里放不下（槽宽 {DN_SCALE_STRIDE}）")
        hd = M17R.dequant(hqx[e], hs[e], t, INTER, INTER // 2, scale_stride=DN_SCALE_STRIDE)
        # 权重侧没有槽对齐：down 权重的 scale 行距**就是**有效前缀 K/GROUP（m17_moe_real/README.md 的 32/20 说明）
        wd = M17R.dequant(wdn[e], sdn[e], HIDDEN, INTER, INTER // 2, scale_stride=INTER // GROUP)
        y_ref = hd @ wd.T
        t3(results, f"{tag} T3 Y slot{e}（cube 累加）", y_dev[e].reshape(t, HIDDEN), y_ref,
           M17R.tol_bf16(y_ref, 1.0, M17R.eps_cube(INTER) * (np.abs(hd) @ np.abs(wd).T)),
           "ε_cube(640)·Σ|a·w|")
    results.append((f"{tag} T1 H_qx 逐字节 vs numpy(硬件规范)", hq_bad == 0,
                    f"{hq_bad}/{hq_n} 字节不符"))
    results.append((f"{tag} T1 H_scale 逐字节 vs numpy(硬件规范)", hs_bad2 == 0,
                    f"{hs_bad2}/{hs_n} 字节不符"))

    # ---- 6. unpermute 加权折叠 + combine（T3）----
    # 紧凑槽：第 e 个专家占紧凑行 [off[e], off[e]+t_e)；本档 m=1 ⇒ 每专家 1 行
    y_flat = np.zeros((E * MMAX, HIDDEN), dtype=y_dev.dtype)
    for e in range(E):
        assert int(off[e + 1]) - int(off[e]) == int(perm["expert_token_counts"][e]), \
            "offsets 与 counts 不自洽"
        for j in range(int(perm["expert_token_counts"][e])):
            y_flat[int(off[e]) + j] = y_dev[e]
    routed_ref = np.zeros((1, HIDDEN), dtype=np.float64)
    l1r = np.zeros((1, HIDDEN), dtype=np.float64)
    for k in range(TOPK):
        wv = float(bf16(np.array([wtk[k] & 0xFFFF], dtype=np.uint16))[0])
        yy = y_flat[inv_ref.reshape(1, TOPK)[0, k]].astype(np.float64)
        routed_ref[0] += wv * yy
        l1r[0] += abs(wv) * np.abs(yy)
    t3(results, f"{tag} T3 routed_output（加权折叠）", bf16(g("routed_output")).reshape(1, HIDDEN),
       routed_ref, M17R.tol_bf16(routed_ref, 1.0, M17R.EPS_FOLD * l1r),
       "0.5·BfUlp + EPS_FOLD·Σ|w·Y|")

    # ---- 7. 共享专家（独立 double 链，T1 量化字节 + T3 数值）----
    gu_shd_dev = bf16(g("GU_shd")).reshape(GU_N)
    adq_s = M17R.dequant(aqx_shd, asc_shd, 1, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    wg = M17R.dequant(wgs, wgn, INTER, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    wu = M17R.dequant(wus, wun, INTER, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    gu_s_ref = adq_s @ np.concatenate([wg, wu], axis=0).T
    # 判据的形状口径：布局表声明的是 `1xN`（参考也算成 (1, N)），设备读是一维 ⇒ 这里补成 (1, N)，
    # `t3` 保持**形状严格**（不许 1-D/2-D 静默广播）—— M90 改写这里漏了，3 条 T3 变成 shape FAIL。
    t3(results, f"{tag} T3 GU_shd", gu_shd_dev.reshape(1, GU_N), gu_s_ref,
       M17R.tol_bf16(gu_s_ref, 1.0, M17R.eps_cube(HIDDEN) *
                     (np.abs(adq_s) @ np.abs(np.concatenate([wg, wu], axis=0)).T)),
       "ε_cube(2560)·Σ|a·w|")
    gg = gu_shd_dev[:INTER].astype(np.float64)
    uu = gu_shd_dev[INTER:].astype(np.float64)
    h_shd_dev = bf16(g("h_swiglu_shd")).reshape(INTER)
    h_s_ref = (gg / (1.0 + np.exp(-gg))) * uu
    t3(results, f"{tag} T3 H_shd", h_shd_dev, h_s_ref,
       M17R.tol_bf16(h_s_ref, 1.0, M17R.EPS_FAST * np.abs(h_s_ref)), "0.5·BfUlp + EPS_FAST·|ref|")
    pq2, sq2 = M17R.quant_hw(np.ascontiguousarray(h_shd_dev.astype(np.float32)).reshape(1, INTER))
    hq2 = g("H_qx_shd")
    hs2 = g("H_scale_shd")
    t1(results, f"{tag} T1 H_shd_qx 逐字节 vs numpy(硬件规范)", pq2[0], hq2[:INTER // 2], "byte")
    t1(results, f"{tag} T1 H_shd_scale 逐字节 vs numpy(硬件规范)", sq2[0],
       hs2.reshape(-1, DN_SCALE_STRIDE)[0, :DN_SCALE_EFF], "byte")
    # 共享专家同上：H_scale_shd 是 DN_SCALE_STRIDE B 的槽（布局表 1x32），行距 = 槽宽
    hd2 = M17R.dequant(hq2, hs2, 1, INTER, INTER // 2, scale_stride=DN_SCALE_STRIDE)
    wd2 = M17R.dequant(wds, wdn_s, HIDDEN, INTER, INTER // 2, scale_stride=INTER // GROUP)
    y_shd_ref = hd2 @ wd2.T
    y_shd_dev = bf16(g("Y_shd")).reshape(HIDDEN)
    t3(results, f"{tag} T3 Y_shd", y_shd_dev.reshape(1, HIDDEN), y_shd_ref,
       M17R.tol_bf16(y_shd_ref, 1.0, M17R.eps_cube(INTER) * (np.abs(hd2) @ np.abs(wd2).T)),
       "ε_cube(640)·Σ|a·w|")

    # combine（S9b）：参考按 **kernel 的序列** 复算 —— `moe = bf16(routed + 未取整的 fp32(g·Y_shd))`
    # （m13 `CombineStage`：Add 用的是 Mul 的 fp32 结果，不是 shared 的 bf16 值）。
    r_dev = bf16(g("routed_output")).reshape(1, HIDDEN).astype(np.float64)
    y_shd_d = bf16(g("Y_shd")).reshape(1, HIDDEN).astype(np.float64)
    gv = 1.0 / (1.0 + np.exp(-float(np.float32(g("sgate")[0]))))
    gated = gv * y_shd_d
    shared_dev = bf16(g("shared_output")).reshape(1, HIDDEN).astype(np.float64)
    t3(results, f"{tag} T3 shared_output", shared_dev, gated,
       M17R.tol_bf16(gated, 2.0, M17R.EPS_FAST * np.abs(gated)),
       "1.0·BfUlp（参考在格点上）+ EPS_FAST·|ref|")
    moe_ref = M17R.bf16_round(r_dev + gated)
    t3(results, f"{tag} T3 moe_output（routed + 未取整 gated）",
       bf16(g("moe_output")).reshape(1, HIDDEN), moe_ref,
       M17R.tol_bf16(moe_ref, 2.0, 2.0 * 2.0 ** -24 * (np.abs(r_dev) + np.abs(gated))
                     + M17R.EPS_FAST * np.abs(gated)),
       "1.0·BfUlp（参考在格点上）+ 2·2^-24·Σ|加数| + EPS_FAST·|gated|")

    # ---- 8. S1/S10：res1 位级、res2 位级、层出口 ----
    res1 = g("res1")
    res2 = g("res2")
    if have_layer_io:
        t1(results, f"{tag} T1 res1 == fp32(层输入)（S1 零残差）位级",
           res1.view(np.uint32), bf16(layer_in).astype(np.float32).view(np.uint32), "bit")
    t1(results, f"{tag} T1 res2 == fp32(moe_out + res1) 位级",
       res2.view(np.uint32),
       np.float32(bf16(g("moe_output")).reshape(HIDDEN) + res1).view(np.uint32), "bit")
    if have_layer_io:
        # 注：融合形态下 S10 的出口**不**落在 ws 的 WS_YFINAL，而是层间残差流缓冲 yLayer
        # （lift_moe_segment.py 规则 5 的接口改造），所以 ws 里的 y_final 区间按设计是旧的；
        # 层出口本身在 host 侧由 Ver M1 的 M.ym 判据逐字节对准。这里只用 numpy 侧见证出口。
        finite = bool(np.all((layer_out & 0x7F80) != 0x7F80))
        nzn = int((layer_out != 0).sum())
        results.append((f"{tag} T4 层出口非零且有限（dump 位置正确）", finite and nzn > 0,
                        f"{nzn}/{HIDDEN} 非零"))
        moe_bits = MB.f32_to_bf16_bits(bf16(g("moe_output")))
        results.append((f"{tag} T4 层出口 != moe_output（S10 归一化确实作用）",
                        bool(not np.array_equal(layer_out, moe_bits)),
                        f"{int((layer_out != moe_bits).sum())}/{HIDDEN} 不同"))

    # ---- 9. 非空洞性（docs/17 §4）----
    nz = int((xsorted != 0).sum())
    nonvac.append((f"{tag} x_sorted 非全零（非空洞）", nz > 0, f"{nz}/{xsorted.size} 个非零"))
    nonvac.append((f"{tag} topk_ids 落在 [0,E) 且互不相同",
                   bool(np.all((ids_dev >= 0) & (ids_dev < E))
                        and len(set(ids_dev.tolist())) == TOPK), f"{ids_dev.tolist()}"))
    nonvac.append((f"{tag} counts 之和 == m*topk", int(perm["expert_token_counts"].sum()) == total,
                   f"Σt_e={int(perm['expert_token_counts'].sum())}"))
    nonvac.append((f"{tag} logits 非常数（输入敏感）",
                   float(logits_dev.max() - logits_dev.min()) > 1e-6,
                   f"span={float(logits_dev.max() - logits_dev.min()):.3e}"))
    nonvac.append((f"{tag} 路由参考非退化（活跃专家数可变）",
                   0 < int((perm["expert_token_counts"] > 0).sum()) < E,
                   f"活跃 {int((perm['expert_token_counts'] > 0).sum())}/{E}"))


# ---------------------------------------------------------------------------
# --selftest：证明「重指向」是行为保持的 swap（无 device）
# ---------------------------------------------------------------------------
def selftest() -> int:
    """主机侧自检：① 两条参考链在同一批真实数据上逐字节一致（swap 不改结论）；
    ② 并排打印 ulp / 网格项口径（`docs/17` §1.1 要求写明量 + `e` 约定）。

    用的是 `tools/golden/data/real`（E=512/topK=10/m=8）已生成的输入行，**不需要 device**。
    """
    import json
    td = os.path.join(ROOT, "tools", "golden", "data", "real")
    if not os.path.exists(os.path.join(td, "manifest.json")):
        print(f"[selftest] RESULT: SKIPPED（缺 {td}/manifest.json）—— 退出码 2")
        return 2
    man = json.load(open(os.path.join(td, "manifest.json")))
    qd = os.path.join(td, "qaware", "floor")
    qman = json.load(open(os.path.join(qd, "manifest.json")))
    need = [os.path.join(td, d["tensor"]) for d in man["files"]] + \
           [os.path.join(qd, d["tensor"]) for d in qman["files"]]
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        print(f"[selftest] RESULT: SKIPPED（真实档的 .bin 未生成：缺 {len(miss)}/{len(need)} 个，"
              f"例如 {os.path.relpath(miss[0], ROOT)}）—— 先跑 "
              f"`python3 tools/golden/gen_dataset.py --groups real`；退出码 2")
        return 2
    T = {d["tensor"]: MB.read_bin(os.path.join(td, d["tensor"]), d) for d in man["files"]}
    Q = {d["tensor"]: MB.read_bin(os.path.join(qd, d["tensor"]), d) for d in qman["files"]}
    xs = np.ascontiguousarray(np.asarray(T["x_sorted.bin"], dtype=np.float32))
    h = np.ascontiguousarray(np.asarray(Q["h_swiglu.bin"], dtype=np.float32))
    print(f"[selftest] 真实档输入：x_sorted {xs.shape} / h_swiglu {h.shape}（E=512 topK=10 m=8）")

    # ① 量化器：m17 的 quant_hw 与 m13 的 quant_mxfp4_hw 在同一批行上必须逐字节一致
    for nm, arr in (("A(x_sorted)", xs), ("H(h_swiglu)", h)):
        p17, s17 = M17R.quant_hw(arr)
        p13, s13 = M13R.quant_mxfp4_hw(arr)
        same = bool(np.array_equal(p17, p13) and np.array_equal(s17, s13))
        print(f"[selftest] ①1 quant_hw(m17) vs quant_mxfp4_hw(m13) on {nm}: "
              f"{'逐字节一致' if same else '不一致'} "
              f"(nibble 不符 {int((p17 != p13).sum())}/{p17.size}，"
              f"scale 不符 {int((s17 != s13).sum())}/{s17.size})")
        if not same:
            print("[selftest] RESULT: FAIL（两条链在同一批数据上分叉 ⇒ 重指向不是行为保持的 swap）")
            return 1
    # ② 反量化：m17 的 dequant 与 m13 的 dequant_device 必须逐元素一致
    e_pack = np.asarray(T["experts.gate_up_proj.bin"], dtype=np.uint8)
    e_scan = np.asarray(T["experts.gate_up_proj.weight_scale.bin"], dtype=np.uint8)
    d17 = M17R.dequant(e_pack[0], e_scan[0], GUN, HIDDEN, HIDDEN // 2, HIDDEN // GROUP)
    d13 = M13R.dequant_device(e_pack[0], e_scan[0], GUN, HIDDEN // 2, HIDDEN // GROUP, HIDDEN)
    same_d = bool(np.array_equal(d17, d13))
    print(f"[selftest] ①2 dequant(m17) vs dequant_device(m13) on experts[0]: "
          f"{'逐元素一致' if same_d else '不一致'}（最大差 "
          f"{float(np.abs(d17 - d13).max()) if not same_d else 0.0}）")
    if not same_d:
        print("[selftest] RESULT: FAIL（反量化链分叉）")
        return 1
    # ② ulp / 网格项口径并排
    probe = float(np.abs(np.asarray(T["router_logits.bin"], dtype=np.float32)).max())
    print(f"[selftest] ② ulp 口径（以 |v|≈{probe:.4g} 为例）："
          f"BfUlp = 2^(e-7) = {float(bf_ulp(np.array([probe]))[0]):.3e}（格点间距，docs/17 §1.1）；"
          f"m17 spacing_bf16 = 2^(e-8) = {float(M17R.spacing_bf16(np.array([probe]))[0]):.3e}"
          f"（= 0.5·BfUlp = 最大 RNE 舍入误差）")
    print("[selftest] ③ 路由参考 = MB.router_topk（docs/17 §7.1 的 FTZ 实现；m17 的路由未建模 FTZ）")
    print("[selftest] RESULT: OK（重指向为行为保持的 swap；device 路径需 B/C 落地后才有 dump）")
    return 0


def _assert_scale_slot(layout):
    """把「H scale 的槽宽」钉在**被读的那份布局表**上（常量名带命名空间，防「裸符号静默绑到别处」）。

    布局表由 `m15_layer_loop/m15_moe_host.h` 的 `MoeLayoutRows()` 用 `M15M::DN_SCALE_STRIDE` 算出；
    M95 正在重写 `m15_moe_resources.h` 的 §4 布局 —— 一旦该常量变了，这里必须**先炸**，
    而不是拿一个过期行距去静默读错数据。
    """
    for nm in ("H_scale", "H_scale_shd"):
        got = int(layout[nm]["bytes"])
        assert got == DN_SCALE_STRIDE, (
            f"布局表 {nm} 每行 {got} B != M15M::DN_SCALE_STRIDE={DN_SCALE_STRIDE}"
            f"（定尺处：m15_layer_loop/m15_moe_host.h 的 MoeLayoutRows()）—— "
            f"本文件读 H scale 的**行距**（不是有效前缀）必须跟着改")
    assert DN_SCALE_EFF <= DN_SCALE_STRIDE and DN_SCALE_STRIDE % GROUP == 0, \
        f"槽宽 {DN_SCALE_STRIDE} 装不下有效前缀 {DN_SCALE_EFF}（GROUP={GROUP}）"


def _run_dump(dump_dir, model_dir):
    """跑一遍 dump 上的全部判据，返回 `(rc, 判定项, 非空洞项)`（打印与调用方式无关）。"""
    global DUMP, E, TOPK, TOPK_TMPL, MMAX
    DUMP = dump_dir
    lay_p = os.path.join(DUMP, "moe_layout.txt")
    if not os.path.exists(lay_p):
        print(f"[check_moe_ref] RESULT: SKIPPED（缺 {lay_p}；融合 kernel 先落一次 dump）"
              f" —— 退出码 2；未比较任何判据")
        return 2, [], []
    layout, meta = load_layout(lay_p)
    E = int(meta["num_experts"])
    # meta 只给**模板上界**；运行期 topk 与 harness 同源（`m15_layer_loop.asc` 的 `M15_TOPK`，默认 2）
    TOPK_TMPL = int(meta["topk_max"])
    TOPK = int(os.environ.get("M15_TOPK", "2"))
    MMAX = int(meta["m_max"])
    _assert_scale_slot(layout)
    print(f"[check_moe_ref] 布局表 {len(layout)} 个张量，ws {int(meta['ws_bytes'])} B/层；"
          f"E={E} 运行期 topk={TOPK}（M15_TOPK）topk 模板上界={TOPK_TMPL} M_MAX={MMAX}"
          f"（参考链 = m17_moe_real/check_ref.py）")
    if not 1 <= TOPK <= TOPK_TMPL:
        print(f"[check_moe_ref] RESULT: FAIL（运行期 topk={TOPK} 不在 1..模板上界 {TOPK_TMPL} 内；"
              f"与产生该 dump 的 harness 的 M15_TOPK 对齐）—— 退出码 1")
        return 1, [], []
    layers = sorted(int(m.group(1)) for m in
                    (re.match(r"L(\d+)_moe_ws\.bin$", f) for f in os.listdir(DUMP)) if m)
    if not layers:
        print(f"[check_moe_ref] RESULT: SKIPPED（{DUMP} 下没有 L*_moe_ws.bin）—— 退出码 2")
        return 2, [], []
    print(f"[check_moe_ref] dump 层：{layers}")
    rdr = ShardReader(model_dir)
    results, refs, nonvac = [], [], []
    for L in layers:
        ws = open(os.path.join(DUMP, f"L{L:02d}_moe_ws.bin"), "rb").read()
        assert len(ws) == int(meta["ws_bytes"]), f"ws 大小 {len(ws)} != {meta['ws_bytes']}"
        check_layer(L, ws, layout, rdr, results, refs, nonvac)

    print("\n[check_moe_ref] ===== 非空洞性 / 位置正确性（T4 guard）=====")
    nbad_nv = 0
    for t, ok, info in nonvac:
        print(f"[check_moe_ref]   {t:56s} {'PASS' if ok else 'FAIL'}  ({info})")
        nbad_nv += 0 if ok else 1
    print("\n[check_moe_ref] ===== 判定项（T1 逐位 / T3 推导界 / T4）=====")
    nbad = 0
    for t, ok, info in results:
        if not ok:
            print(f"[check_moe_ref]   {t:56s} FAIL  ({info})")
        nbad += 0 if ok else 1
    print(f"[check_moe_ref] 判定项 {len(results)} 条：{len(results) - nbad} PASS / {nbad} FAIL"
          f"（另有非空洞性 {len(nonvac)} 条：{len(nonvac) - nbad_nv} PASS / {nbad_nv} FAIL）")
    print("\n[check_moe_ref] ===== 参考项（非判定；两套 MXFP4 量化规范的逐字节差异）=====")
    for t, info in refs:
        print(f"[check_moe_ref]   {t:56s} {info}")
    njudged = len(results)
    if njudged == 0:
        print("\n[check_moe_ref] RESULT: SKIPPED (0 条判定项被比较) —— 退出码 2")
        return 2, results, nonvac
    if (nbad + nbad_nv) == 0:
        print(f"\n[check_moe_ref] RESULT: OK ({njudged} 条判定项 + {len(nonvac)} 条非空洞全部比较通过)")
        return 0, results, nonvac
    print(f"\n[check_moe_ref] RESULT: FAIL ({nbad}/{njudged} 条判定项越界) —— 退出码 1")
    return 1, results, nonvac


# ---------------------------------------------------------------------------
# --negctl：M96 的负向对照（第五变体纪律「把被测对象弄坏，判据必须变红」）
# ---------------------------------------------------------------------------
def _negctl_variants():
    """每个变体 = (名字, 安装函数)。安装函数只改**本进程内存**里的取数环节：
    不写任何文件（仓库文件、dump、/tmp 副本都不动），返回还原函数。"""
    real = M17R.dequant

    def _install(fn):
        M17R.dequant = fn
        return lambda: setattr(M17R, "dequant", real)

    def v1_stride_from_prefix(packed, scale, rows, k, packed_stride=None, scale_stride=None):
        # 把 H scale 的**槽宽**按另一个候选值（有效前缀 20）传 —— M90 回归的本体
        return real(packed, scale, rows, k, packed_stride,
                    DN_SCALE_EFF if scale_stride == DN_SCALE_STRIDE else scale_stride)

    def v2_slot_tail(packed, scale, rows, k, packed_stride=None, scale_stride=None):
        # 取数区段**偏移**改错：取槽尾 DN_SCALE_EFF 字节，而不是槽头
        a = np.asarray(scale)
        if scale_stride != DN_SCALE_STRIDE:
            return real(packed, scale, rows, k, packed_stride, scale_stride)
        a = a.reshape(-1, DN_SCALE_STRIDE)[:, DN_SCALE_STRIDE - DN_SCALE_EFF:].reshape(-1)
        return real(packed, a, rows, k, packed_stride, DN_SCALE_EFF)

    def v3_slot_reversed(packed, scale, rows, k, packed_stride=None, scale_stride=None):
        # 取数区段**方向**改错：槽内逆序后再取前缀
        a = np.asarray(scale)
        if scale_stride == DN_SCALE_STRIDE:
            a = a.reshape(-1, DN_SCALE_STRIDE)[:, ::-1].reshape(-1)
        return real(packed, a, rows, k, packed_stride, scale_stride)

    return [
        ("V1 H scale 行距传「有效前缀」20（M90 回归本体）", lambda: _install(v1_stride_from_prefix)),
        ("V2 槽内取**尾** 20 B（取数区段偏移错）", lambda: _install(v2_slot_tail)),
        ("V3 槽内**逆序**（取数区段方向错）", lambda: _install(v3_slot_reversed)),
        ("V4 运行期 topk 按模板上界取（多读 0xCD 污染 id）", None),
    ]


def negctl(dump_dir, model_dir) -> int:
    """M96 负向对照：把**取数环节**在内存里弄坏，判据必须变红。

    正对照：不注入 ⇒ 必须 rc=0 且 0 FAIL（否则变体的「红」不能归因于变异）。
    每个变体的判据：出现 ≥1 条判定项/非空洞 FAIL，或直接抛异常（结构性错误），或 rc≠0。
    """
    rows = []

    def _run():
        import io
        import contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc, res, nv = _run_dump(dump_dir, model_dir)
            return rc, res, nv, None
        except Exception as e:      # 结构性变异（例如 32 B 的槽按 20 B 行距 reshape）会直接抛
            return 1, [], [], f"{type(e).__name__}: {e}"

    def _fmt(name, rc, res, nv, err):
        nf = sum(0 if ok else 1 for _, ok, _ in res)
        nvf = sum(0 if ok else 1 for _, ok, _ in nv)
        red = (err is not None) or rc != 0 or (nf + nvf) > 0
        detail = err or (f"rc={rc}；判定项 {len(res)} 条 {nf} FAIL / 非空洞 {len(nv)} 条 {nvf} FAIL")
        print(f"[negctl]   {name:42s} {'红（咬住）' if red else '★绿（没咬住）'}  {detail}")
        rows.append((name, red))
        return red

    print("[negctl] 正对照 + 4 个变体；同一份 dump、同一条判据路径；变异只在内存里")
    rc0, res0, nv0, err0 = _run()
    nf0 = sum(0 if ok else 1 for _, ok, _ in res0)
    nvf0 = sum(0 if ok else 1 for _, ok, _ in nv0)
    pos_green = err0 is None and rc0 == 0 and nf0 + nvf0 == 0
    print(f"[negctl]   {'正对照（不注入，必须绿）':42s} {'绿' if pos_green else '★红'}  "
          f"rc={rc0}；判定项 {len(res0)} 条 {nf0} FAIL / 非空洞 {len(nv0)} 条 {nvf0} FAIL"
          + (f"；{err0}" if err0 else ""))
    for name, install in _negctl_variants():
        if install is None:                       # V4：只动环境变量（harness 的同一个开关）
            old = os.environ.get("M15_TOPK")
            os.environ["M15_TOPK"] = str(TOPK_TMPL)
            try:
                rc, res, nv, err = _run()
            finally:
                os.environ.pop("M15_TOPK", None)
                if old is not None:
                    os.environ["M15_TOPK"] = old
        else:
            restore = install()
            try:
                rc, res, nv, err = _run()
            finally:
                restore()
        _fmt(name, rc, res, nv, err)
    if not pos_green:
        print("[negctl] RESULT: FAIL（正对照不绿 ⇒ 变体的红不能归因于变异）—— 退出码 1")
        return 1
    if not all(red for _, red in rows):
        missed = ", ".join(n for n, red in rows if not red)
        print(f"[negctl] RESULT: FAIL（有变体没被咬住：{missed}）—— 退出码 1")
        return 1
    print(f"[negctl] RESULT: OK（正对照绿 + {len(rows)} 个变体全部变红）—— 退出码 0")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?")
    ap.add_argument("--model-dir", default="/workspace/Qwen3.8-Flash-Next-MXFP4")
    ap.add_argument("--selftest", action="store_true",
                    help="主机侧自检（无 device）：证明重指向是行为保持的 swap")
    ap.add_argument("--negctl", action="store_true",
                    help="M96 负向对照：把 H scale 的取数环节在内存里弄坏，判据必须变红")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.dump_dir:
        print("[check_moe_ref] RESULT: SKIPPED（既没给 dump_dir 也没给 --selftest）—— 退出码 2")
        return 2
    if args.negctl:
        return negctl(args.dump_dir, args.model_dir)
    return _run_dump(args.dump_dir, args.model_dir)[0]


if __name__ == "__main__":
    sys.exit(main())
