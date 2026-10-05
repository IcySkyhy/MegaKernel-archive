#!/usr/bin/env python3.12
"""M88 —— attention prolog 的 host 侧参考（numpy / float64），**规则逐条钉在 vLLM 官方源码上**。

本文件是 M82 显式 descope 的「attention 前端 prolog」（`m15_layer_loop/README.md` Part D · M82-6 第 1 项）
的第一段落盘：**先把语义变成可测的，再写 AscendC**。

覆盖两条路（**不得假定同源**，本文件对二者分别给出依据）：
  A. 主干 attention：`qkv_proj` bf16 GEMM → split(q | gate | k | v) → q/k **GemmaRMSNorm(256, eps 1e-6)**
     → partial NeoX RoPE(64) →（gate 不 norm 不 rope）
  B. indexer：`index_qk_proj` bf16 GEMM → split(idx q[4×128] | raw k[128]) → idx q 走
     **GemmaRMSNorm(128) + 同一张 cos/sin 表上的 partial NeoX RoPE(64)**；raw k **不 norm 不 rope**。

每条规则的 `文件:符号` 依据（**N2 外部权威**，docs/17 §1.2；仓外工件 = /workspace/vllm）：

| # | 规则 | 依据（file:symbol） |
|---|---|---|
| R1 | `qkv_proj` 输出宽度 = `nH*(1+gate)*hd + 2*nKV*hd` | `vllm/model_executor/models/qwen3_next.py:427-434`（`Qwen3NextAttention._project_qkv_gate`） |
| R2 | split 顺序 `[q|gate] , k , v`；q 与 gate **按头交织**（头 h 的 512 列 = q[256] ‖ gate[256]） | 同上 `:428-434` + `vllm/models/qwen4_exp/nvidia/qsa.py:505-506` |
| R3 | q/k norm = **GemmaRMSNorm**（`(1+w)`，不是朴素 `w`） | `nvidia/qsa.py:350-351`；语义 `vllm/model_executor/layers/layernorm.py:140-168`（`weight = self.weight + 1.0`）；别名 `vllm/model_executor/models/qwen3_next.py:31`（`GemmaRMSNorm as Qwen3NextRMSNorm`） |
| R4 | Qwen4Exp 是**文本-only**：三个 MRoPE 轴 position id 恒等相同 | `nvidia/model.py:846-852`（`positions.unsqueeze(0).expand(3,-1)`） |
| R5 | 文本-only ⇒ MRoPE 的三轴置换在数值上恒等 ⇒ **退化为普通 NeoX partial RoPE** | `vllm/model_executor/layers/rotary_embedding/mrope.py:236-247`（`apply_interleaved_rope` 在 `x[0]==x[1]==x[2]` 时返回 `x[0]`）；`mrope.py:374-423`（`cos_sin_cache[positions]` 后仅做三轴选择） |
| R6 | RoPE 类别 = `MRotaryEmbedding`（**不是** `MRotaryEmbeddingInterleaved`） | `rotary_embedding/__init__.py:62-71,110-121`（`rope_type='default'` + `mrope_section` ⇒ `MRotaryEmbedding`）；`MRotaryEmbeddingInterleaved` 只在 `scaling_type=='openpangu'` 分支（`:333`） |
| R7 | `rotary_dim = int(head_dim * partial_rotary_factor)` = 256×0.25 = **64** | `rotary_embedding/__init__.py:68-71`；本 checkpoint `text_config.rope_parameters.partial_rotary_factor = 0.25` |
| R8 | inv_freq / cos / sin 表公式；表按 **query dtype(bf16)** 落地 | `rotary_embedding/base.py:80-99`（`_compute_inv_freq` / `_compute_cos_sin_cache`）、`:105-125`（`_match_cos_sin_cache_dtype` → `.to(dtype=query.dtype)`） |
| R9 | NeoX 配对 `(j, j+32)`；`o1=x1·cos−x2·sin`、`o2=x2·cos+x1·sin`；`[64,256)` 直通 | `rotary_embedding/common.py:134-173`（`ApplyRotaryEmb.forward_static`，`is_neox_style=True`）；`mrope.py:414-434`（rot/pass 切分在 `:rotary_dim`） |
| R10 | indexer：`index_qk_proj` 输出 = `(nH_idx + nKV_idx) * idx_head_dim` = (4+1)×128 = **640** | `nvidia/indexer_qsa.py:131-137` |
| R11 | indexer q/k norm = `GemmaRMSNorm(128, eps=rms_norm_eps)` | `nvidia/indexer_qsa.py:138-145`；fused 版实证 `nvidia/ops/qsa_pre_indexer.py:69`（`weight = load(...) + 1.0`） |
| R12 | indexer 复用**主 rotary 的 cos/sin 表**（stride = 64 = rotary_dim），只旋 128 维的**前 64 维** | `nvidia/ops/qsa_pre_indexer.py:165,174-186`（`cos_sin_stride = D//2`）、`:31-32`（`HALF=D//2, QUARTER=D//4`）、`:72-79`（rot/直通切分与配对）、`:430`（`cos_sin_cache.shape[-1]*2 == head_dim`） |
| R13 | indexer raw k **不 norm 不 rope**，原样进 raw ring | `nvidia/ops/qsa_pre_indexer.py:367-372`（`k = tl.load(k_ptr...); tl.store(state_row+dims, k)`） |

**基准性质声明（docs/17 §7，硬要求）**：本文件是**自建参考**，它对齐的是 vLLM 的**前端算子语义**；
它**不是**「与官方端到端输出对齐」。本工程**不做**打分/topk/expand，**任何地方不得写
「m=4097 对齐官方输出」** —— 官方 QSA 的 `indexer_budget=2048` 是 token 预算，每 token 只 attend
约一半历史（`/workspace/ascend_mega_kernel/docs/17-verification-standard.md:281`）。

用法：
  # 1) 生成 device 侧的输入与期望（真实 checkpoint 权重 + 确定性输入）
  /usr/local/python3.12.13/bin/python3.12 oracle_attn_prolog.py gen
  # 2) 复算 capacity/常量自检（不依赖 checkpoint）
  /usr/local/python3.12.13/bin/python3.12 oracle_attn_prolog.py selfcheck
"""

import json
import math
import pathlib
import struct
import sys

import numpy as np

# ============================================================
# 0. 形状 / 常量（与 m15_attn_prolog.h 的 M15AP::* 逐项一致，改一处必须改两处）
# ============================================================
HIDDEN = 2560
NH = 24               # config.num_attention_heads
HD = 256              # config.head_dim
NKV = 2               # config.num_key_value_heads
IDX_NH = 4            # config.indexer_n_heads
IDX_D = 128           # config.indexer_head_dim
ROT = 64              # int(256 * partial_rotary_factor=0.25)
HALF = ROT // 2       # 32
EPS = 1e-6            # config.rms_norm_eps
THETA = 1e7           # rope_parameters.rope_theta
QW = NH * HD          # 6144
IDX_Q = IDX_NH * IDX_D  # 512
IDX_W = IDX_Q + IDX_D   # 640

# 段布局（bf16 元素下标；与 m15_attn_prolog.h 的 M15AP::*_OFF 逐项一致）
Y0_QG = 0
Y0_K = QW * 2         # 12288  （q|gate 交织区之后是 k）
Y0_V = Y0_K + NKV * HD          # 12800
Y0_IDX = Y0_V + NKV * HD        # 13312
Y0_N = Y0_IDX + (IDX_Q + IDX_D)  # 13952

OUT_Q = 0
OUT_GATE = QW                   # 6144
OUT_K = OUT_GATE + QW           # 12288
OUT_V = OUT_K + NKV * HD        # 12800
OUT_QIDX = OUT_V + NKV * HD     # 13312
OUT_KRAW = OUT_QIDX + IDX_Q     # 13824
OUT_N = OUT_KRAW + IDX_D        # 13952

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST = HERE.parent.parent / "weights_manifest.txt"


# ============================================================
# 1. 与 kernel 逐位一致的 bf16 取整（RNE；m19_qsa_indexer/check_ref.py:83-88 同款）
# ============================================================
def bf16_round(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
    return (r << 16).view(np.float32)


def bf16_ulp(x):
    """bf16 的**格点间距** = 2^(e-7)，e = binade 指数（docs/17 §1.1 口径：2^e ≤ |v| < 2^(e+1)）。"""
    a = np.abs(np.asarray(x, dtype=np.float64))
    e = np.floor(np.log2(np.maximum(a, np.finfo(np.float32).tiny)))
    return 2.0 ** (e - 7.0)


# ============================================================
# 2. 基础算子（每条对应 R# 表里的一行）
# ============================================================
def gemma_norm_f64(x64, w, eps=EPS):
    """R3/R11：y = x * rsqrt(mean_{head}(x^2) + eps) * (1 + w)。x64 为 float64 的整头向量。"""
    rstd = 1.0 / math.sqrt(float(np.mean(x64 * x64)) + eps)
    return x64 * rstd * (1.0 + w.astype(np.float64))


def cos_sin_table(pos, rot=ROT, theta=THETA):
    """R8：cos/sin 表。vLLM 在 fp32 里算表，然后按 query dtype 落成 bf16。

    返回 (cos_bf16_as_f64, sin_bf16_as_f64)，各 32 个数（rot/2）。
    """
    d = np.arange(0, rot, 2, dtype=np.float32)
    inv_freq = np.float32(1.0) / (np.float32(theta) ** (d / np.float32(rot)))
    t = np.float32(pos)
    freqs = (t * inv_freq).astype(np.float32)
    cos = bf16_round(np.cos(freqs.astype(np.float64)).astype(np.float32)).astype(np.float64)
    sin = bf16_round(np.sin(freqs.astype(np.float64)).astype(np.float32)).astype(np.float64)
    return cos, sin


def rope_neox_partial(y_bf16, cos, sin, rot=ROT):
    """R9/R12：NeoX partial RoPE。y_bf16 是**已落到 bf16 格点**的整头向量（float64 存值）。"""
    out = np.array(y_bf16, dtype=np.float64, copy=True)
    h = rot // 2
    a = y_bf16[:h]
    b = y_bf16[h:rot]
    out[:h] = a * cos - b * sin
    out[h:rot] = b * cos + a * sin
    return out


# ============================================================
# 3. 参考流水线（两条路）
# ============================================================
def main_prolog(x_bf16, Wq, Wk, Wv, wqn, wkn, pos, mode=0):
    """主干：返回 (y0[Y0_N], out[OUT_N], terms_y0[Y0_N], terms_out[OUT_N])。

    mode（负向对照用，**方向级**改错；0=正确）：
      1 = 朴素 RMSNorm（乘 w，不乘 (1+w)）—— R3 的反向
      2 = RoPE 符号反向（o1 = x1·c + x2·s）—— R9 的反向
      3 = 完全不做 RoPE —— R9 的删除
    """
    x64 = x_bf16.astype(np.float64)
    xf = x_bf16.astype(np.float32)

    ex = {}

    def gemm(W, tag):
        """bf16 GEMM：fp32 累加（设备侧 mmad 的语义）+ 落 bf16。

        额外返回 fp64 精确值 `ex[tag]`（T2' 判"翻转元素是否真的贴格点中点"要用它）。
        """
        acc = xf @ W.T.astype(np.float32)      # [N]
        ex[tag] = x64 @ W.T.astype(np.float64)
        return bf16_round(acc).astype(np.float64), np.abs(xf @ np.abs(W.T.astype(np.float32)))

    qg, t_qg = gemm(Wq, "qg")
    kv, t_kv = gemm(Wk, "k")
    vv, t_vv = gemm(Wv, "v")

    y0 = np.zeros(Y0_N, dtype=np.float64)
    y0[Y0_QG:Y0_QG + QW * 2] = qg
    y0[Y0_K:Y0_K + NKV * HD] = kv
    y0[Y0_V:Y0_V + NKV * HD] = vv

    ty0 = np.zeros(Y0_N, dtype=np.float64)
    ty0[Y0_QG:Y0_QG + QW * 2] = t_qg
    ty0[Y0_K:Y0_K + NKV * HD] = t_kv
    ty0[Y0_V:Y0_V + NKV * HD] = t_vv

    out = np.zeros(OUT_N, dtype=np.float64)
    normonly = np.zeros(OUT_N, dtype=np.float64)   # 诊断：norm 之后、rope 之前（bf16 格点）
    terms = np.zeros(OUT_N, dtype=np.float64)
    cos, sin = cos_sin_table(pos)

    def do_norm(xh, w, mm):
        """xh: fp64 整头向量（bf16 值）；返回**落 bf16 后**的整头向量（fp64 存值）与 Σ|terms|。"""
        if mm == 1:
            rstd = 1.0 / math.sqrt(float(np.mean(xh * xh)) + EPS)
            y = xh * rstd * w.astype(np.float64)
            t = np.abs(y)
        else:
            rstd = 1.0 / math.sqrt(float(np.mean(xh * xh)) + EPS)
            y = xh * rstd * (1.0 + w.astype(np.float64))
            t = np.abs(y)
        return bf16_round(y.astype(np.float32)).astype(np.float64), t

    def do_rope(yb, mm):
        if mm == 3:
            return yb, np.zeros_like(yb)
        c, s = cos, sin
        r = np.array(yb, dtype=np.float64, copy=True)
        h = ROT // 2
        a = yb[:h]
        b = yb[h:ROT]
        if mm == 2:
            r[:h] = a * c + b * s
            r[h:ROT] = b * c - a * s
        else:
            r[:h] = a * c - b * s
            r[h:ROT] = b * c + a * s
        t = np.zeros_like(yb)
        t[:h] = np.abs(a * c) + np.abs(b * s)
        t[h:ROT] = np.abs(b * c) + np.abs(a * s)
        return bf16_round(r.astype(np.float32)).astype(np.float64), t

    for h in range(NH):
        base = h * (2 * HD)
        qh = y0[Y0_QG + base: Y0_QG + base + HD]
        gh = y0[Y0_QG + base + HD: Y0_QG + base + 2 * HD]
        out[OUT_GATE + h * HD: OUT_GATE + (h + 1) * HD] = gh           # gate 不 norm 不 rope
        yn, tn = do_norm(qh, wqn, mode)
        normonly[OUT_Q + h * HD: OUT_Q + (h + 1) * HD] = yn
        yr, tr = do_rope(yn, mode)
        out[OUT_Q + h * HD: OUT_Q + (h + 1) * HD] = yr
        terms[OUT_Q + h * HD: OUT_Q + (h + 1) * HD] = tn + tr
        terms[OUT_GATE + h * HD: OUT_GATE + (h + 1) * HD] = np.abs(gh)
        # gate 的 terms：它是 GEMM 输出的直接抄写，Σ|terms| 取 |gh| 的 GEMM 累加量
        terms[OUT_GATE + h * HD: OUT_GATE + (h + 1) * HD] = t_qg[base + HD: base + 2 * HD]

    for h in range(NKV):
        kh = y0[Y0_K + h * HD: Y0_K + (h + 1) * HD]
        out[OUT_V + h * HD: OUT_V + (h + 1) * HD] = y0[Y0_V + h * HD: Y0_V + (h + 1) * HD]
        terms[OUT_V + h * HD: OUT_V + (h + 1) * HD] = t_vv[h * HD: (h + 1) * HD]
        yn, tn = do_norm(kh, wkn, mode)
        normonly[OUT_K + h * HD: OUT_K + (h + 1) * HD] = yn
        yr, tr = do_rope(yn, mode)
        out[OUT_K + h * HD: OUT_K + (h + 1) * HD] = yr
        terms[OUT_K + h * HD: OUT_K + (h + 1) * HD] = tn + tr

    return y0, out, ty0, terms, ex, normonly


def indexer_prolog(x_bf16, Wqk, wqn, wkn, pos, mode=0):
    """indexer：返回 (y0[640], out_qidx[512], out_kraw[128], terms_qidx[512], exact, normonly[512])。"""
    xf = x_bf16.astype(np.float32)
    acc = xf @ Wqk.T.astype(np.float32)
    qk = bf16_round(acc).astype(np.float64)
    t_qk = np.abs(xf @ np.abs(Wqk.T.astype(np.float32)))
    exact = x_bf16.astype(np.float64) @ Wqk.T.astype(np.float64)

    cos, sin = cos_sin_table(pos)
    qidx = np.zeros(IDX_Q, dtype=np.float64)
    terms = np.zeros(IDX_Q, dtype=np.float64)
    normonly = np.zeros(IDX_Q, dtype=np.float64)
    for h in range(IDX_NH):
        xh = qk[h * IDX_D: (h + 1) * IDX_D]
        if mode == 1:
            rstd = 1.0 / math.sqrt(float(np.mean(xh * xh)) + EPS)
            y = xh * rstd * wqn.astype(np.float64)
        else:
            rstd = 1.0 / math.sqrt(float(np.mean(xh * xh)) + EPS)
            y = xh * rstd * (1.0 + wqn.astype(np.float64))
        tn = np.abs(y)
        yb = bf16_round(y.astype(np.float32)).astype(np.float64)
        # ⚠ **无条件填 normonly**（r4 修正）：这一段原来只在 `if mode == 3:` 里填，而 `cmd_gen`
        #   调 indexer_prolog 用的是默认 `mode=0` ⇒ `exp_normonly_*.bin` 的 qidx 段**恒为全零**，
        #   于是每次 runs=prolog 都打印的「qidx 不同 512/512」是 **oracle 数据缺口**、不是设备读数
        #   （拿设备的非零 qidx 比全零）。⇒ 该诊断行对 q/k 成立、对 qidx **不成立**。
        normonly[h * IDX_D: (h + 1) * IDX_D] = yb
        if mode == 3:
            qidx[h * IDX_D: (h + 1) * IDX_D] = yb
            terms[h * IDX_D: (h + 1) * IDX_D] = tn
            continue
        hh = ROT // 2
        r = np.array(yb, dtype=np.float64, copy=True)
        a = yb[:hh]
        b = yb[hh:ROT]
        if mode == 2:
            r[:hh] = a * cos + b * sin
            r[hh:ROT] = b * cos - a * sin
        else:
            r[:hh] = a * cos - b * sin
            r[hh:ROT] = b * cos + a * sin
        tr = np.zeros_like(yb)
        tr[:hh] = np.abs(a * cos) + np.abs(b * sin)
        tr[hh:ROT] = np.abs(b * cos) + np.abs(a * sin)
        qidx[h * IDX_D: (h + 1) * IDX_D] = bf16_round(r.astype(np.float32)).astype(np.float64)
        terms[h * IDX_D: (h + 1) * IDX_D] = tn + tr
    # R13：raw k 原样，不 norm 不 rope
    kraw = qk[IDX_Q: IDX_Q + IDX_D]
    return qk, qidx, kraw, terms, exact, normonly


# ============================================================
# 4. manifest / 权重读取（**与设备同一来源**：manifest 的 file/offset/rows/cols）
# ============================================================
def load_manifest(path=MANIFEST):
    model_dir, num_layers, kinds, tensors = None, 0, [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("model_dir="):
            model_dir = line.split("=", 1)[1].strip()
        elif line.startswith("num_layers="):
            num_layers = int(line.split("=", 1)[1])
        elif line.startswith("layer_kinds="):
            kinds = line.split("=", 1)[1].split(",")
        elif line.startswith("tensor "):
            d = dict(kv.split("=", 1) for kv in line.split()[1:])
            tensors.append(d)
    return model_dir, num_layers, kinds, tensors


def read_bf16(model_dir, t):
    path = pathlib.Path(model_dir) / t["file"]
    with open(path, "rb") as f:
        f.seek(int(t["offset"]))
        buf = f.read(int(t["bytes"]))
    assert int(t["bytes"]) == int(t["rows"]) * int(t["cols"]) * 2, t
    a = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
    a = (a << 16).view(np.float32).reshape(int(t["rows"]), int(t["cols"]))
    return a


def weights_for(tensors, layer):
    want = {
        "attn_q_proj": (12288, 2560), "attn_k_proj": (512, 2560), "attn_v_proj": (512, 2560),
        "attn_q_norm": (1, 256), "attn_k_norm": (1, 256),
        "attn_idx_qk_proj": (640, 2560), "attn_idx_q_norm": (1, 128), "attn_idx_k_norm": (1, 128),
    }
    found = {}
    for t in tensors:
        if int(t["layer"]) == layer and t["role"] in want:
            assert (int(t["rows"]), int(t["cols"])) == want[t["role"]], (t["role"], t["rows"], t["cols"])
            found[t["role"]] = t
    missing = set(want) - set(found)
    assert not missing, f"layer {layer} 缺 role: {sorted(missing)}"
    return found


# ============================================================
# 5. gen：把 device 侧需要的输入与期望写成 bin
# ============================================================
def dump_bf16(path, a):
    """写 bf16（**高 16 位**）。⚠ M85 的教训：取低位会写出整片 0，判据全部「PASS 但空过」。"""
    r = bf16_round(np.asarray(a, dtype=np.float32))
    hi = np.asarray(r, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    hi.astype(np.uint16).tofile(path)


def load_dump_bf16(path):
    """读回 bf16 文件为 float64 存值（判据的比对面一律走这条路，保证比的是**落盘字节**）。"""
    a = np.fromfile(path, dtype=np.uint16).astype(np.uint32)
    return (a << np.uint32(16)).view(np.float32).astype(np.float64)


def dump_f32(path, a):
    np.asarray(a, dtype=np.float32).tofile(path)


def cmd_gen(layer=3):
    model_dir, _nl, _kinds, tensors = load_manifest()
    W = weights_for(tensors, layer)
    Wq = read_bf16(model_dir, W["attn_q_proj"])
    Wk = read_bf16(model_dir, W["attn_k_proj"])
    Wv = read_bf16(model_dir, W["attn_v_proj"])
    wqn = read_bf16(model_dir, W["attn_q_norm"])[0]
    wkn = read_bf16(model_dir, W["attn_k_norm"])[0]
    Wqk = read_bf16(model_dir, W["attn_idx_qk_proj"])
    wqln = read_bf16(model_dir, W["attn_idx_q_norm"])[0]
    wkln = read_bf16(model_dir, W["attn_idx_k_norm"])[0]

    DATA.mkdir(parents=True, exist_ok=True)
    # 确定性输入（N1：输入隔离 —— 由 host 生成，不来自任何设备产物）
    rng = np.random.default_rng(8800)
    x0 = bf16_round((rng.standard_normal(HIDDEN) * 0.5).astype(np.float32))
    x1 = bf16_round((rng.standard_normal(HIDDEN) * 0.5).astype(np.float32))
    assert not np.array_equal(x0, x1)
    dump_bf16(DATA / "x0.bin", x0)
    dump_bf16(DATA / "x1.bin", x1)

    pos = 4096            # pos 4096 = "开放组"档（docs/17:286）；也覆盖 (pos+1)%4!=0
    # cos/sin 表（**host 预生成，设备不算超越函数**；R8）：[4097][64] bf16，行 = [cos(32)|sin(32)]
    tab = np.zeros((4097, ROT), dtype=np.float32)
    for p in range(4097):
        c, s = cos_sin_table(p)
        tab[p, :HALF] = c
        tab[p, HALF:] = s
    dump_bf16(DATA / "cos_sin.bin", tab.reshape(-1))
    tf = load_dump_bf16(DATA / "cos_sin.bin")
    print(f"[gen] cos/sin 表 4097×64 bf16 = {(DATA / 'cos_sin.bin').stat().st_size} B，"
          f"非零 {int((tf != 0).sum())}/{tf.size}，distinct={len(np.unique(tf))}")
    assert (tf != 0).sum() > tf.size // 2 and len(np.unique(tf)) > 256

    meta = {"layer": layer, "pos": pos, "n_inputs": 2, "y0_n": Y0_N, "out_n": OUT_N,
            "hidden": HIDDEN, "nh": NH, "hd": HD, "nkv": NKV, "idx_nh": IDX_NH,
            "idx_d": IDX_D, "rot": ROT, "eps": EPS, "theta": THETA,
            "offsets": {"y0_qg": Y0_QG, "y0_k": Y0_K, "y0_v": Y0_V, "y0_idx": Y0_IDX,
                        "out_q": OUT_Q, "out_gate": OUT_GATE, "out_k": OUT_K, "out_v": OUT_V,
                        "out_qidx": OUT_QIDX, "out_kraw": OUT_KRAW}}
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))

    for name, x in (("x0", x0), ("x1", x1)):
        y0, out, ty0, terms, ex, normonly = main_prolog(x, Wq, Wk, Wv, wqn, wkn, pos)
        qk, qidx, kraw, tq, ex_idx, idxnorm = indexer_prolog(x, Wqk, wqln, wkln, pos)
        normonly[OUT_QIDX:OUT_QIDX + IDX_Q] = idxnorm
        y0[Y0_IDX:Y0_IDX + IDX_Q + IDX_D] = qk
        ty0[Y0_IDX:Y0_IDX + IDX_Q + IDX_D] = np.abs(x.astype(np.float64) @
                                                    np.abs(Wqk.T.astype(np.float64)))
        out[OUT_QIDX:OUT_QIDX + IDX_Q] = qidx
        out[OUT_KRAW:OUT_KRAW + IDX_D] = kraw
        terms[OUT_QIDX:OUT_QIDX + IDX_Q] = tq
        terms[OUT_KRAW:OUT_KRAW + IDX_D] = ty0[Y0_IDX + IDX_Q: Y0_IDX + IDX_Q + IDX_D]
        dump_bf16(DATA / f"exp_y0_{name}.bin", y0)
        dump_bf16(DATA / f"exp_out_{name}.bin", out)
        dump_bf16(DATA / f"exp_normonly_{name}.bin", normonly)
        # ⚠ **这段自检必须读「落盘的字节」，不能读内存里的数组**（M85 的教训：内存非零、文件全 0）。
        #   而且它必须在**每次 dump 之后**跑 —— r4 我第一次修 P2-1 时正是栽在「为了做正对照把缺口改回去
        #   跑了一次 gen（那次 dump 出了全 0 的文件），随后只把代码恢复、没有重跑 gen」⇒ 磁盘上留下的是
        #   坏数据而代码是好的（半刷新）。读文件才能咬住这种形态。
        nf = load_dump_bf16(DATA / f"exp_normonly_{name}.bin")
        grid = [(OUT_Q, QW, "q"), (OUT_K, NKV * HD, "k"), (OUT_QIDX, IDX_Q, "qidx")]
        for off, cnt, tag in grid:
            seg = nf[off:off + cnt]
            nzc = int((seg != 0).sum())
            print(f"[gen] normonly.{tag}（**读落盘文件**）: 非零 {nzc}/{cnt}")
            assert nzc > cnt // 2, (
                f"normonly.{tag} 整段为 0 ⇒ 诊断行会拿设备读数去比全零（假信号）。"
                "这正是 r4 复审 P2-1 抓到的缺口；注意本断言读的是**落盘字节**。")
        ey = np.zeros(Y0_N, dtype=np.float64)
        ey[Y0_QG:Y0_QG + QW * 2] = ex["qg"]
        ey[Y0_K:Y0_K + NKV * HD] = ex["k"]
        ey[Y0_V:Y0_V + NKV * HD] = ex["v"]
        ey[Y0_IDX:Y0_IDX + IDX_W] = ex_idx
        dump_f32(DATA / f"exp_y0_exact_{name}.bin", ey)
        dump_f32(DATA / f"exp_terms_y0_{name}.bin", ty0)
        dump_f32(DATA / f"exp_terms_out_{name}.bin", terms)
        # 非空洞性（docs/17 §4）：**比对面取自落盘字节**，不是内存里的数组
        # （M85 的 `pack_bf16` 取低位 ⇒ 文件整片 0 而内存非零，判据全部空过）
        outf = load_dump_bf16(DATA / f"exp_out_{name}.bin")
        y0f = load_dump_bf16(DATA / f"exp_y0_{name}.bin")
        nz = int((outf != 0).sum())
        nzy = int((y0f != 0).sum())
        print(f"[gen] {name}: y0 非零 {nzy}/{Y0_N}，out 非零 {nz}/{OUT_N}，"
              f"distinct={len(np.unique(outf))}，|out|max={np.abs(outf).max():.4g}")
        assert nzy > Y0_N // 2 and nz > OUT_N // 2, "落盘的期望输出整片为 0 ⇒ 判据会空过"
        assert len(np.unique(outf)) > 1024, "落盘的期望输出退化（取值过少）"
    # 输入敏感性：两次输入的期望必须显著不同
    a = load_dump_bf16(DATA / "exp_out_x0.bin")
    b = load_dump_bf16(DATA / "exp_out_x1.bin")
    diff = int((a != b).sum())
    print(f"[gen] 输入敏感性：exp_out(x0) vs exp_out(x1) 不同元素 {diff}/{OUT_N}")
    assert diff > OUT_N // 2
    print(f"[gen] OK → {DATA}")


def cmd_selfcheck():
    """不依赖 checkpoint 的常量自检（与头文件逐条对照）。"""
    ok = True

    def g(tag, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[selfcheck] {tag}: {'PASS' if cond else 'FAIL'}")

    g("q+gate 宽度 = 2*nh*hd = 12288", QW * 2 == 12288)
    g("idx 宽度 = (4+1)*128 = 640", IDX_Q + IDX_D == 640)
    g("ROT = int(256*0.25) = 64", ROT == int(HD * 0.25))
    g("HALF = ROT/2 = 32", HALF == 32)
    g("mrope_section 和 == ROT/2", sum([11, 11, 10]) == HALF)
    g("Y0_N == 13952", Y0_N == 13952)
    g("OUT_N == 13952", OUT_N == 13952)
    cos, sin = cos_sin_table(0)
    g("pos 0 ⇒ cos 全 1 / sin 全 0", np.allclose(cos, 1.0) and np.allclose(sin, 0.0))
    c1, s1 = cos_sin_table(1)
    # 表按 bf16 落地 ⇒ 单侧误差上界 = 0.5*ulp(cos(1.0)) < 2^-9；这里取 4*2^-9 的宽松上限
    tol = 4.0 * 2.0 ** -9
    g("pos 1 ⇒ 第一对 (j=0, inv_freq=1) 与 cos/sin(1.0) 相差 < 4*2^-9",
      abs(float(c1[0]) - math.cos(1.0)) < tol and abs(float(s1[0]) - math.sin(1.0)) < tol)
    print(f"[selfcheck] {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "gen":
        cmd_gen(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == "selfcheck":
        sys.exit(cmd_selfcheck())
    else:
        raise SystemExit(f"unknown cmd {cmd}")
