#!/usr/bin/env python3
"""M25：48 层循环的权重 manifest 生成器（从真实 checkpoint 的 safetensors 头取 offset/bytes）。

产出 `m15_layer_loop/weights_manifest.txt`（纯 key=value 行，便于 C++ 直接解析）：
    model_dir=...
    num_layers=48
    layer_kinds=linear_attention,linear_attention,linear_attention,full_attention,...
    tensor layer=L role=... file=... offset=... bytes=... dtype=BF16 rows=... cols=...

M25 只给 **linear_attention（GDN）** 层写张量行（full_attention 层没有 linear_attn 权重）；
**M40 追加 MoE 段**（S1-S10，见 m15_moe_host.h）：MoE 是**每一层都有**的（`mlp.*`），
所以 GDN 行与 MoE 行分别记录、互不影响 —— attention 层只有 MoE 行。
**M58 追加 hc 段**（每层两个边界）、**M65 追加全局 mixer**（虚拟层号 = num_layers）、
**M82 追加 attention 段**（`ATTN_ROLES`，只对 12 个 full_attention 层：q/k/v/o_proj、
q/k_norm（主 attention 的 head_dim=256），以及 indexer 的 index_qk_proj + q/k_layernorm）。

MoE 段的**路由规模是缩形档**（m13 段的可编译规模）：`NUM_EXPERTS_MOE=4`、`TOPK_MOE=2`。
checkpoint 每层有 512 个 routed 专家，`role=moe_*` 只切**前 4 个专家**的连续字节
（专家在张量最外维 → 前 4 个专家的字节段是连续的，offset 不变、bytes = 4×单专家字节）。
这是已知且必须披露的规模差（原因与后续项见 README「已知限制」第 1 条）。

**M93 追加真规模段**（`MOE512_ROLES`，role 前缀 `moe512_`，**纯追加在文件尾**）：全
`config.num_experts`（=512）个专家的连续字节段 + 真 topk（=10）。它是**加性**的 ——
`role=moe_*` 的每个 E=4 行逐字节不变（`diff` 只有新增行），因为现消费者
`m15_layer_resources.h` / `m15_layer_loop.asc` 的 device 槽仍按 4 专家编译
（`NUM_EXPERTS 4→512` / `TOPK_MAX 4→10` / `MOE_W_STRIDE` 属后续 MoE-A/MoE-B）。
真规模字节数与 HBM 预算见 `tools/weights/moe_real_scale_audit.py` +
`tools/weights/evidence/moe_real_scale.txt`。

层类型直接来自 config.json 的 `layer_types`（host 侧会用该字段逐层核对 3:1 模式推导）。

用法（仓库根目录）：
    /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/slice_layer_manifest.py \
        --model-dir /workspace/Qwen3.8-Flash-Next-MXFP4 \
        --out m15_layer_loop/weights_manifest.txt
    ... --check        # 只校验已生成的 manifest（不写盘）
    ... --moe512-neg-experts 4   # 负向对照：真规模段按 4 专家算 ⇒ 算式核对 FAIL、rc=1

只用 numpy + 标准库（复用 tools/weights/safetensors_reader.py，不依赖 torch）。
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "weights"))

from safetensors_reader import ShardReader  # noqa: E402

# role → checkpoint 张量名后缀（model.language_model.layers.{L}.<suffix>）
ROLES = [
    ("in_proj_qkv", "linear_attn.in_proj_qkv.weight"),   # [10240, 2560] → q|k|v 通道
    ("in_proj_z", "linear_attn.in_proj_z.weight"),       # [6144, 2560]  → z（output gate）
    ("in_proj_b", "linear_attn.in_proj_b.weight"),       # [48, 2560]    → β
    ("in_proj_a", "linear_attn.in_proj_a.weight"),       # [48, 2560]    → a（decay 输入）
    ("out_proj", "linear_attn.out_proj.weight"),         # [2560, 6144]
    ("conv1d", "linear_attn.conv1d.weight"),             # [10240, 1, 4]（layer kernel 内 host 转置）
    ("A_log", "linear_attn.A_log"),                      # [48] bf16 → fp32[64]
    ("dt_bias", "linear_attn.dt_bias"),                  # [48] bf16 → fp32[64]
    ("norm", "linear_attn.norm.weight"),                 # [128] = RMSNormGated 的 gamma
]

# ---- M82：attention（full_attention）层的 `self_attn` 张量（**只对 12 个 full_attention 层**）----
# (role, suffix, [期望 shape], 切片后 rows, 切片后 cols)
# 形状是 checkpoint 实测（`ShardReader.info()`；M82 在 layer 3 上核过）：
#   q_proj [12288, 2560] = q (24 头 × 256) | output_gate (24 × 256) 两段行拼接
#   k_proj / v_proj [512, 2560] = 2 个 KV 头 × 256
#   o_proj [2560, 6144]（消费 24×256 的 attention 输出）
#   q_norm / k_norm [256]（主 attention 的 head_dim=256 的 RMSNorm，eps 1e-6）
#   indexer: index_qk_proj [640, 2560] = 4×128 q | 128 raw k；q/k_layernorm [128]（GemmaRMSNorm）
# **全量切片**（不做转换；分量切分由 host/kernel 侧按行/列偏移做）。
ATTN_ROLES = [
    ("attn_q_proj", "self_attn.q_proj.weight", [12288, 2560], 12288, 2560),
    ("attn_k_proj", "self_attn.k_proj.weight", [512, 2560], 512, 2560),
    ("attn_v_proj", "self_attn.v_proj.weight", [512, 2560], 512, 2560),
    ("attn_o_proj", "self_attn.o_proj.weight", [2560, 6144], 2560, 6144),
    ("attn_q_norm", "self_attn.q_norm.weight", [256], 1, 256),
    ("attn_k_norm", "self_attn.k_norm.weight", [256], 1, 256),
    ("attn_idx_qk_proj", "self_attn.indexer.index_qk_proj.weight", [640, 2560], 640, 2560),
    ("attn_idx_q_norm", "self_attn.indexer.q_layernorm.weight", [128], 1, 128),
    ("attn_idx_k_norm", "self_attn.indexer.k_layernorm.weight", [128], 1, 128),
]

# ---- M40：MoE 段的 checkpoint 张量（每层都有）----
# (role, suffix, [期望 shape], 切片后 rows, 切片后 cols)
# rows/cols 描述的是**切片后**的形状（写进 manifest，host 侧只用来核对字节数/拼接行数）
NUM_EXPERTS_MOE = 4
TOPK_MOE = 2
INTER = 640
GU_N = 2 * INTER
GROUP = 32
MOE_ROLES = [
    ("moe_router_w", "mlp.gate.weight", [512, 2560], NUM_EXPERTS_MOE, 2560),
    ("moe_sgate_w", "mlp.shared_expert_gate.weight", [1, 2560], 1, 2560),
    ("moe_experts_gate_up", "mlp.experts.gate_up_proj", [512, GU_N, 2560 // 2], NUM_EXPERTS_MOE,
     GU_N * (2560 // 2)),
    ("moe_experts_gate_up_scale", "mlp.experts.gate_up_proj.weight_scale", [512, GU_N, 2560 // GROUP],
     NUM_EXPERTS_MOE, GU_N * (2560 // GROUP)),
    ("moe_experts_down", "mlp.experts.down_proj", [512, 2560, INTER // 2], NUM_EXPERTS_MOE,
     2560 * (INTER // 2)),
    ("moe_experts_down_scale", "mlp.experts.down_proj.weight_scale", [512, 2560, INTER // GROUP],
     NUM_EXPERTS_MOE, 2560 * (INTER // GROUP)),
    ("moe_shared_gate", "mlp.shared_expert.gate_proj.weight", [INTER, 2560 // 2], INTER, 2560 // 2),
    ("moe_shared_gate_scale", "mlp.shared_expert.gate_proj.weight_scale", [INTER, 2560 // GROUP], INTER,
     2560 // GROUP),
    ("moe_shared_up", "mlp.shared_expert.up_proj.weight", [INTER, 2560 // 2], INTER, 2560 // 2),
    ("moe_shared_up_scale", "mlp.shared_expert.up_proj.weight_scale", [INTER, 2560 // GROUP], INTER,
     2560 // GROUP),
    ("moe_shared_down", "mlp.shared_expert.down_proj.weight", [2560, INTER // 2], 2560, INTER // 2),
    ("moe_shared_down_scale", "mlp.shared_expert.down_proj.weight_scale", [2560, INTER // GROUP], 2560,
     INTER // GROUP),
]

# ---- M93：MoE **真规模**（全 512 专家）段（**加性**，不动上面任何一行）----
# 上面 `MOE_ROLES` 的 4 专家缩形档是当前 device 槽能承载的规模（`m15_moe_resources.h` 的
# `NUM_EXPERTS = 4`，其 `MOE_W_STRIDE` 按 4 专家算）。checkpoint 每层有 512 个 routed 专家
# （`config.json` 的 `text_config.num_experts`），top-k 是 **10**（`num_experts_per_tok`）。
# 本段把**全 512 专家**的连续字节段以**新 role（`moe512_*`）**落进 manifest，供后续
# MoE-A/MoE-B 扩槽时直接消费 —— offset/bytes 与真实文件头逐值相符（见 `tools/weights/
# moe_real_scale_audit.py` 与 `tools/weights/evidence/moe_real_scale.txt`）。
#
# 为什么可以直接整段落：**专家在张量最外维**（shape[0] = 512）⇒ 前 512 个专家就是整张量，
# offset 不变、bytes = 该张量头部全量（`data_end - data_begin`），**不需要任何在线重排**。
# 共享专家（`moe_shared_*`）本来就**没有**专家维，现有 12 行里的那 6 行已经是全量，本段不重复。
MOE512_ROLES = [
    ("moe512_router_w", "gate.weight"),
    ("moe512_experts_gate_up", "experts.gate_up_proj"),
    ("moe512_experts_gate_up_scale", "experts.gate_up_proj.weight_scale"),
    ("moe512_experts_down", "experts.down_proj"),
    ("moe512_experts_down_scale", "experts.down_proj.weight_scale"),
]

# ---- M58：hc（hyper-connection）段的 checkpoint 张量（每层两个边界：attn_hc / mlp_hc）----
# (role, suffix, [期望 shape], 切片后 rows, 切片后 cols)
# **全量切片**（不做任何转换）：hc 权重在 checkpoint 里是 bf16 且按「行主序原始 layout」被 kernel
# 直接消费（m20 的权重契约：`wDown`/`wUp`/`hcNorm` 与 checkpoint 字节一一对应）。
# `*_inj` 在 checkpoint 里只有 INJ_N=4 行，而 device 槽要 ≥16 行可读（cube 的 N 分形以 16 行为
# 单位）⇒ **host 侧把行 [4,16) 置零**后再写槽（m20 §5.1 的 host 契约；不改变 checkpoint 的前
# 4 行）。manifest 里记录的是 **checkpoint 侧的 4 行**（也就是 host 要 pread 的字节数）。
HC_HYPER = 10240
HC_LOWRANK = 320
HC_INJ_ROWS = 4
_HC_SUFFIX = {
    "attn_hc": "attn_hyper_connection",
    "mlp_hc": "mlp_hyper_connection",
}
HC_ROLES = []
for _pfx, _mod in _HC_SUFFIX.items():
    HC_ROLES += [
        (f"{_pfx}_norm", f"{_mod}.hc_norm.weight", [HC_HYPER], 1, HC_HYPER),
        (f"{_pfx}_down", f"{_mod}.input_mix_weight_down.weight", [HC_LOWRANK, HC_HYPER], HC_LOWRANK,
         HC_HYPER),
        (f"{_pfx}_up", f"{_mod}.input_mix_weight_up.weight", [HC_HYPER, HC_LOWRANK], HC_HYPER,
         HC_LOWRANK),
        (f"{_pfx}_inj", f"{_mod}.block_inject_weight.weight", [HC_INJ_ROWS, HC_HYPER], HC_INJ_ROWS,
         HC_HYPER),
    ]

# ---- M65：末层之后的**全局 mixer**（`hyper_connection_mixer`，3 个张量）----
# 它不属于任何一层（checkpoint 里挂在 `model.language_model.hyper_connection_mixer.*`），
# manifest 里用 **layer = num_layers（48）** 这个虚拟层号登记（host 按它装到第 49 个 hc 权重槽）。
# 三个角色的名字与 m15_layer_resources.h 的 HW_ROLE_GM_* 一一对应。
# **没有 `block_inject_weight`**（`V-N:model.py:612` 显式丢弃）⇒ host 侧把注入区 16 行置零。
GM_ROLES = [
    ("gmixer_norm", "hyper_connection_mixer.hc_norm.weight", [HC_HYPER], 1, HC_HYPER),
    ("gmixer_down", "hyper_connection_mixer.input_mix_weight_down.weight", [HC_LOWRANK, HC_HYPER],
     HC_LOWRANK, HC_HYPER),
    ("gmixer_up", "hyper_connection_mixer.input_mix_weight_up.weight", [HC_HYPER, HC_LOWRANK],
     HC_HYPER, HC_LOWRANK),
]

# ---- M100：PLE 段的 checkpoint 张量（**只挂 0-based 层 1**）----
# 语义权威 = `m15_layer_loop/ple/PLE_SPEC.md` §2（形状逐值与 checkpoint 实测相符）。
# (role, suffix, [期望 shape], 切片后 rows, 切片后 cols)；全部**全量切片**（不做转换）：
#   key_proj [10240,2560] / value_proj [2560,2560]（checkpoint **无** kv_proj，见 PLE_SPEC D1）
#   conv1d   [10240,1,4]（kernel 侧按 tap-major 消费：wtap[k][c] = w[c,k]）
#   norm_key / norm_query / norm_conv [10240]（Gemma 约定 (1+w)，逐元素）
#   3 个 int64 标量 buffer（layer_multipliers[3] / ngram_heads_vocab_sizes[16] / offsets[16]）
PLE_LAYER = 1            # 0-based；config `ple_layer_ids=[2]` 是 1-based，checkpoint 名也是 layers.1.ple.*
PLE_HYPER = 10240        # hc_count(4) × hidden_size(2560)
PLE_HE = 2560            # ple_embed_dim
PLE_KW = 4               # ple_conv_kernel_size
PLE_HEAD_DIM = 160       # ple_embed_dim / ngram_heads(16) = 2560/16
PLE_TABLE_SHARDS = 128   # config split_ngram_parts
PLE_TABLE_ROWS = 2500012  # padded_rows / 128（PLE_SPEC §2.1）
PLE_ROLES = [
    ("ple_key_proj", "ple.key_proj.weight", [PLE_HYPER, PLE_HE], PLE_HYPER, PLE_HE),
    ("ple_value_proj", "ple.value_proj.weight", [PLE_HE, PLE_HE], PLE_HE, PLE_HE),
    ("ple_conv1d", "ple.conv1d.weight", [PLE_HYPER, 1, PLE_KW], PLE_HYPER, PLE_KW),
    ("ple_norm_key", "ple.norm_key.weight", [PLE_HYPER], 1, PLE_HYPER),
    ("ple_norm_query", "ple.norm_query.weight", [PLE_HYPER], 1, PLE_HYPER),
    ("ple_norm_conv", "ple.norm_conv.weight", [PLE_HYPER], 1, PLE_HYPER),
    ("ple_lm", "ple.ple_embedding.layer_multipliers", [3], 1, 3),
    ("ple_sz", "ple.ple_embedding.ngram_heads_vocab_sizes", [16], 1, 16),
    ("ple_of", "ple.ple_embedding.ngram_heads_offsets", [16], 1, 16),
]

# 与 m15 资源表/层 kernel 的契约（改这里必同步改 m15_gdn_resources.h / m15_loop_layout.h）
HIDDEN = 2560
IN_N = 16480          # in_proj 行数 = qkv 10240 | z 6144 | b 48 | a 48
OUT_N, OUT_K = 2560, 6144
CH, KW = 10240, 4

DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "I32": 4, "U32": 4, "F64": 8,
            "I64": 8}   # M100：PLE 的 3 个确定性标量 buffer 是 int64


def prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def build(model_dir, out_path, check_only=False, moe512_experts=None):
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tcfg = cfg["text_config"]
    n_layers = int(tcfg["num_hidden_layers"])
    interval = int(tcfg["full_attention_interval"])
    kinds = list(tcfg["layer_types"])
    assert len(kinds) == n_layers, f"layer_types 长度 {len(kinds)} != num_hidden_layers {n_layers}"
    for i, k in enumerate(kinds):   # 3:1 模式核对：每 interval 层的最后一个是 full_attention
        expect = "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        assert k == expect, f"config layer_types[{i}]={k} 与 3:1 模式（{expect}）不符"
    n_gdn = kinds.count("linear_attention")
    n_attn = kinds.count("full_attention")

    rdr = ShardReader(model_dir)
    lines = [
        "# m15 48 层循环权重 manifest —— 由 slice_layer_manifest.py 生成，勿手改",
        f"model_dir={model_dir}",
        f"num_layers={n_layers}",
        "layer_kinds=" + ",".join(kinds),
    ]
    rows_total = {}
    attn_rows_total = {}
    moe4_bytes = {}          # M93：E=NUM_EXPERTS_MOE 缩形档的 (L, role) -> 切片字节数
    for L in range(n_layers):
        in_rows = 0
        for role, suffix in ROLES:
            if kinds[L] != "linear_attention":
                break
            name = f"model.language_model.layers.{L}.{suffix}"
            if name not in set(rdr.names()):
                raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
            info = rdr.info(name)
            shape = list(info.shape)
            dt = info.st_dtype
            nbytes = int(info.data_end) - int(info.data_begin)
            assert nbytes == prod(shape) * DT_BYTES[dt], f"{name} 字节数 {nbytes} 与 shape 不符"
            rows = shape[0] if len(shape) > 0 else 1
            cols = prod(shape[1:]) if len(shape) > 1 else 1
            if role == "in_proj_qkv":
                assert shape == [10240, HIDDEN], shape
            elif role == "in_proj_z":
                assert shape == [6144, HIDDEN], shape
            elif role in ("in_proj_b", "in_proj_a"):
                assert shape == [48, HIDDEN], shape
            elif role == "out_proj":
                assert shape == [OUT_N, OUT_K], shape
            elif role == "conv1d":
                assert shape == [CH, 1, KW], shape
            elif role in ("A_log", "dt_bias"):
                assert shape == [48], shape
            elif role == "norm":
                assert shape == [128], shape
            lines.append(
                f"tensor layer={L} role={role} file={os.path.basename(str(info.file))} "
                f"offset={int(info.data_begin)} bytes={nbytes} dtype={dt} rows={rows} cols={cols}"
            )
            rows_total[role] = rows_total.get(role, 0) + rows

        # ---- M82：attention（full_attention）层张量（只对 12 个 full_attention 层）----
        if kinds[L] == "full_attention":
            for role, suffix, want_shape, srows, scols in ATTN_ROLES:
                name = f"model.language_model.layers.{L}.{suffix}"
                if name not in set(rdr.names()):
                    raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
                info = rdr.info(name)
                shape = list(info.shape)
                assert shape == want_shape, f"{name} shape {shape} != {want_shape}"
                dt = info.st_dtype
                nbytes = int(info.data_end) - int(info.data_begin)
                assert nbytes == prod(shape) * DT_BYTES[dt], f"{name} 字节数 {nbytes} 与 shape 不符"
                slice_bytes = srows * scols * DT_BYTES[dt]
                assert slice_bytes == nbytes, f"{name} 切片 {slice_bytes} != 全量 {nbytes}"
                lines.append(
                    f"tensor layer={L} role={role} file={os.path.basename(str(info.file))} "
                    f"offset={int(info.data_begin)} bytes={slice_bytes} dtype={dt} rows={srows} cols={scols}"
                )
                attn_rows_total[role] = attn_rows_total.get(role, 0) + srows

        # ---- M40：MoE 段张量（每一层都有，含 full_attention 层）----
        for role, suffix, want_shape, srows, scols in MOE_ROLES:
            name = f"model.language_model.layers.{L}.{suffix}"
            if name not in set(rdr.names()):
                raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
            info = rdr.info(name)
            shape = list(info.shape)
            assert shape == want_shape, f"{name} shape {shape} != {want_shape}"
            dt = info.st_dtype
            nbytes = int(info.data_end) - int(info.data_begin)
            assert nbytes == prod(shape) * DT_BYTES[dt], f"{name} 字节数 {nbytes} 与 shape 不符"
            per_row = (prod(shape[1:]) if len(shape) > 1 else 1) * DT_BYTES[dt]   # 单行字节数
            # 切片：专家在**最外维** → 前 srows 行是连续字节段（offset 不变）
            slice_bytes = srows * per_row
            assert slice_bytes <= nbytes, f"{name} 切片字节数 {slice_bytes} > 张量 {nbytes}"
            moe4_bytes[(L, role)] = slice_bytes
            lines.append(
                f"tensor layer={L} role={role} file={os.path.basename(str(info.file))} "
                f"offset={int(info.data_begin)} bytes={slice_bytes} dtype={dt} rows={srows} cols={scols}"
            )

        # ---- M58：hc 段张量（每一层都有两个边界：attn_hc / mlp_hc）----
        for role, suffix, want_shape, srows, scols in HC_ROLES:
            name = f"model.language_model.layers.{L}.{suffix}"
            if name not in set(rdr.names()):
                raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
            info = rdr.info(name)
            shape = list(info.shape)
            assert shape == want_shape, f"{name} shape {shape} != {want_shape}"
            dt = info.st_dtype
            nbytes = int(info.data_end) - int(info.data_begin)
            assert nbytes == prod(shape) * DT_BYTES[dt], f"{name} 字节数 {nbytes} 与 shape 不符"
            # 全量切片（hc 权重没有专家维，故 srows*scols == prod(shape)）
            slice_bytes = srows * scols * DT_BYTES[dt]
            assert slice_bytes == nbytes, f"{name} 切片 {slice_bytes} != 全量 {nbytes}"
            lines.append(
                f"tensor layer={L} role={role} file={os.path.basename(str(info.file))} "
                f"offset={int(info.data_begin)} bytes={slice_bytes} dtype={dt} rows={srows} cols={scols}"
            )

    # in_proj 拼接行数（每层）= 16480
    per_layer_in = sum(rows_total[r] for r in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")) // n_gdn
    if per_layer_in != IN_N:
        raise SystemExit(f"[FAIL] in_proj 拼接行数 {per_layer_in} != {IN_N}")

    # ---- M65：末层全局 mixer 的 3 个张量（虚拟层号 = n_layers）----
    for role, suffix, want_shape, srows, scols in GM_ROLES:
        name = f"model.language_model.{suffix}"
        if name not in set(rdr.names()):
            raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
        info = rdr.info(name)
        shape = list(info.shape)
        assert shape == want_shape, f"{name} shape {shape} != {want_shape}"
        dt = info.st_dtype
        nbytes = int(info.data_end) - int(info.data_begin)
        assert nbytes == prod(shape) * DT_BYTES[dt], f"{name} 字节数 {nbytes} 与 shape 不符"
        slice_bytes = srows * scols * DT_BYTES[dt]
        assert slice_bytes == nbytes, f"{name} 切片 {slice_bytes} != 全量 {nbytes}"
        lines.append(
            f"tensor layer={n_layers} role={role} file={os.path.basename(str(info.file))} "
            f"offset={int(info.data_begin)} bytes={slice_bytes} dtype={dt} rows={srows} cols={scols}"
        )

    # ---- M93：MoE **真规模**（全 n_experts 专家）段 ----
    # **纯追加在文件尾**：上面 role=moe_* 的 4 专家缩形档一行未动（`diff` 只有 `>` 行）。
    # 每条 tensor 行的 offset/bytes 直接取自真实文件头（`data_begin` / `data_end-data_begin`），
    # 并用「单专家字节 × n_experts == 头部全量」这条算式交叉核对 —— 把专家数当 4（或 stride
    # 算错）会在这里被咬住（负向对照见 `tools/weights/evidence/m93_negative_controls.txt`）。
    n_experts_cfg = int(tcfg["num_experts"])
    topk_cfg = int(tcfg["num_experts_per_tok"])
    n_exp = n_experts_cfg if moe512_experts is None else int(moe512_experts)
    # 单专家字节的**独立来源**：由 config 的几何量（hidden/inter/group）算出的期望值。
    # 它与「头部 shape」是两个来源 —— 头部 shape 算错、或 stride 写错，两边就会不一致。
    hidden_c = int(tcfg["hidden_size"])
    inter_c = int(tcfg["moe_intermediate_size"])
    group_c = int(cfg["quantization_config"]["group_size"])
    one_expect = {
        "moe512_router_w": hidden_c * 2,
        "moe512_experts_gate_up": GU_N * (hidden_c // 2),
        "moe512_experts_gate_up_scale": GU_N * (hidden_c // group_c),
        "moe512_experts_down": hidden_c * (inter_c // 2),
        "moe512_experts_down_scale": hidden_c * (inter_c // group_c),
    }
    if (hidden_c, inter_c, group_c) != (HIDDEN, INTER, GROUP):
        raise SystemExit(f"[FAIL] config 几何量 (hidden={hidden_c}, inter={inter_c}, group={group_c}) "
                         f"!= 本文件常量 ({HIDDEN}, {INTER}, {GROUP})")
    lines.append(f"# ---- M93 MoE 真规模段：全 {n_experts_cfg} 专家（config text_config.num_experts="
                 f"{n_experts_cfg}、topk={topk_cfg}）；上面的 `role=moe_*` 是 "
                 f"{NUM_EXPERTS_MOE}/{TOPK_MOE} 缩形档，本段**不替换**它们 ----")
    lines.append(f"moe512_num_experts={n_experts_cfg}")
    lines.append(f"moe512_topk={topk_cfg}")
    lines.append(f"moe512_intermediate={INTER}")
    moe512_routed_total = 0
    moe512_routed_layer = None
    for L in range(n_layers):
        routed_this = 0
        for role, suffix in MOE512_ROLES:
            name = f"model.language_model.layers.{L}.mlp.{suffix}"
            if name not in set(rdr.names()):
                raise SystemExit(f"[FAIL] checkpoint 缺张量 {name}")
            info = rdr.info(name)
            shape = list(info.shape)
            dt = info.st_dtype
            nbytes = int(info.data_end) - int(info.data_begin)
            assert nbytes == prod(shape) * DT_BYTES[dt], f"{name} 字节数 {nbytes} 与 shape 不符"
            # C5：专家在最外维，且维长必须等于 config 的 num_experts
            if shape[0] != n_experts_cfg:
                raise SystemExit(f"[FAIL] {name} shape[0]={shape[0]} != config num_experts "
                                 f"{n_experts_cfg}（专家维不在最外维？）")
            one = prod(shape[1:]) * DT_BYTES[dt]          # 单专家连续字节段（来源 = 头部 shape）
            full = one * n_exp                            # 全 n_exp 专家（专家在最外维 ⇒ 连续）
            # C4b：单专家字节必须 == config 几何量算出的期望（第二个来源）
            if one != one_expect[role]:
                raise SystemExit(
                    f"[FAIL] M93 {role}@{L}: 头部算出的单专家 {one} B != config 几何量期望 "
                    f"{one_expect[role]} B（stride / shape 读法有误）")
            # C3/C6：算式必须与真实文件头逐值相符；n_exp 被改错即在此 FAIL
            if full != nbytes:
                raise SystemExit(
                    f"[FAIL] M93 真规模 {role}@{L}: 单专家 {one} B × {n_exp} = {full} B "
                    f"!= 头部全量 {nbytes} B（把专家数当 {n_exp}（真值 {n_experts_cfg}）"
                    f"或 stride 算错，本条即 FAIL）")
            # C7：与上面的 4 专家缩形档的倍数关系（512/4 = 128）；NUM_EXPERTS_MOE 被改错即 FAIL
            base = moe4_bytes.get((L, role.replace("moe512_", "moe_", 1)))
            if base is None:
                raise SystemExit(f"[FAIL] M93 找不到 {role}@{L} 对应的 E={NUM_EXPERTS_MOE} 缩形档行")
            if n_experts_cfg % NUM_EXPERTS_MOE != 0 or base * (n_experts_cfg // NUM_EXPERTS_MOE) != full:
                raise SystemExit(
                    f"[FAIL] M93 {role}@{L}: E={NUM_EXPERTS_MOE} 档 {base} B × "
                    f"({n_experts_cfg}/{NUM_EXPERTS_MOE}) != 真规模 {full} B")
            if role != "moe512_router_w":
                routed_this += full
            lines.append(
                f"tensor layer={L} role={role} file={os.path.basename(str(info.file))} "
                f"offset={int(info.data_begin)} bytes={full} dtype={dt} rows={n_exp} cols={prod(shape[1:])}"
            )
        if L == 0:
            moe512_routed_layer = routed_this
        elif moe512_routed_layer != routed_this:
            raise SystemExit(f"[FAIL] M93 层 {L} 的 routed 字节 {routed_this} != 层 0 的 "
                             f"{moe512_routed_layer}（逐层不一致）")
        moe512_routed_total += routed_this
    lines.append(f"moe512_routed_layer_bytes={moe512_routed_layer}")
    lines.append(f"moe512_routed_total_bytes={moe512_routed_total}")
    lines.append(f"moe512_layers={n_layers}")

    # ---- M100：PLE 段（**只挂 0-based 层 1**；config 的 `ple_layer_ids=[2]` 是 1-based）----
    # 语义权威 = `m15_layer_loop/ple/PLE_SPEC.md` §2（9 个小张量 + 128 个分片表）。
    # **纯追加在文件尾**：上面每一条 role 行一字未动（`diff` 只有新增行）。
    # 表的分片形状 [2500012,160] 与 `padded = ceil(total/128)*128` 的推导复核：
    #   total = Σ size[h] = 320001446 ⇒ padded = 320001536 ⇒ padded/128 = 2500012（PLE_SPEC §2.1）
    if len(kinds) <= PLE_LAYER or kinds[PLE_LAYER] != "linear_attention":
        raise SystemExit(f"[FAIL] PLE 层 {PLE_LAYER} 的 kind = "
                         f"{kinds[PLE_LAYER] if len(kinds) > PLE_LAYER else 'N/A'} != linear_attention")
    ple_prefix = f"model.language_model.layers.{PLE_LAYER}."
    lines.append(f"# ---- M100 PLE 段：只挂 0-based 层 {PLE_LAYER}（config ple_layer_ids="
                 f"{tcfg.get('ple_layer_ids')} 是 1-based）；上面的 role 一行未动，本段纯追加 ----")
    ple_info = {}
    for role, suffix, want_shape, srows, scols in PLE_ROLES:
        name = ple_prefix + suffix
        if name not in set(rdr.names()):
            raise SystemExit(f"[FAIL] checkpoint 缺 PLE 张量 {name}")
        info = rdr.info(name)
        ple_info[role] = info
        shape = list(info.shape)
        if shape != want_shape:
            raise SystemExit(f"[FAIL] {name} shape {shape} != {want_shape}（PLE_SPEC §2）")
        dt = info.st_dtype
        nbytes = int(info.data_end) - int(info.data_begin)
        if nbytes != prod(shape) * DT_BYTES[dt]:
            raise SystemExit(f"[FAIL] {name} 字节数 {nbytes} 与 shape 不符")
        slice_bytes = srows * scols * DT_BYTES[dt]
        if slice_bytes != nbytes:
            raise SystemExit(f"[FAIL] {name} 切片 {slice_bytes} != 全量 {nbytes}")
        lines.append(
            f"tensor layer={PLE_LAYER} role={role} file={os.path.basename(str(info.file))} "
            f"offset={int(info.data_begin)} bytes={slice_bytes} dtype={dt} rows={srows} cols={scols}"
        )
    ple_tbl_rows = 0
    for k in range(PLE_TABLE_SHARDS):
        name = ple_prefix + f"ple.ple_embedding.ngram_embedding.shard_{k}.weight"
        if name not in set(rdr.names()):
            raise SystemExit(f"[FAIL] checkpoint 缺 PLE 表分片 {name}")
        info = rdr.info(name)
        shape = list(info.shape)
        if shape != [PLE_TABLE_ROWS, PLE_HEAD_DIM]:
            raise SystemExit(f"[FAIL] {name} shape {shape} != [{PLE_TABLE_ROWS}, {PLE_HEAD_DIM}]")
        dt = info.st_dtype
        nbytes = int(info.data_end) - int(info.data_begin)
        if nbytes != PLE_TABLE_ROWS * PLE_HEAD_DIM * DT_BYTES[dt]:
            raise SystemExit(f"[FAIL] {name} 字节数 {nbytes} 与 shape 不符")
        if k == 0:
            ple_tbl_off0, ple_tbl_file0 = int(info.data_begin), os.path.basename(str(info.file))
        # 每片一张**独立张量**（不一定一片一文件：PLE_SPEC §2.1 实测 shard_0/shard_1 同在
        # model-00005、shard_2..5 同在 model-00006）⇒ 逐片记 file + data_begin。
        lines.append(
            f"tensor layer={PLE_LAYER} role=ple_tbl_shard_{k:03d} "
            f"file={os.path.basename(str(info.file))} offset={int(info.data_begin)} "
            f"bytes={nbytes} dtype={dt} rows={PLE_TABLE_ROWS} cols={PLE_HEAD_DIM}"
        )
        ple_tbl_rows += shape[0]
    if ple_tbl_rows != PLE_TABLE_SHARDS * PLE_TABLE_ROWS:
        raise SystemExit(f"[FAIL] PLE 表总行数 {ple_tbl_rows} != {PLE_TABLE_SHARDS} × {PLE_TABLE_ROWS}")
    # 表行数的**独立来源**（PLE_SPEC §2.1 的 padding 公式）：Σ size[h] 与 offset 前缀和由
    # checkpoint 的 2 个 int64 buffer 直接读出（不经 shape），padded = ceil(total/128)*128
    # 必须等于上面逐片累加出来的行数 —— 两来源不一致即 FAIL（shape 读错 / 分片数写错）。
    def _raw(role):
        info = ple_info[role]
        nb = int(info.data_end) - int(info.data_begin)
        with open(str(info.file), "rb") as fh:
            fh.seek(int(info.data_begin))
            b = fh.read(nb)
        if len(b) != nb:
            raise SystemExit(f"[FAIL] 读 {role} 短读 {len(b)}/{nb}")
        return b
    _sz = struct.unpack("<16q", _raw("ple_sz"))
    _of = struct.unpack("<16q", _raw("ple_of"))
    if _of[0] != 0 or any(_of[h + 1] != _of[h] + _sz[h] for h in range(15)):
        raise SystemExit(f"[FAIL] PLE offsets 不是 sizes 的前缀和：sizes={_sz} offsets={_of}")
    if any(s <= 0 for s in _sz):
        raise SystemExit(f"[FAIL] PLE vocab_sizes 含非正项：{_sz}")
    _total = sum(_sz)
    _padded = ((_total + PLE_TABLE_SHARDS - 1) // PLE_TABLE_SHARDS) * PLE_TABLE_SHARDS
    if _padded != ple_tbl_rows:
        raise SystemExit(f"[FAIL] PLE padding 复核：ceil(Σsize={_total}/{PLE_TABLE_SHARDS})×"
                         f"{PLE_TABLE_SHARDS} = {_padded} != 分片表总行数 {ple_tbl_rows}")
    lines.append(f"ple_table_rows_total={_total}")
    lines.append(f"ple_layer_multipliers=" + ",".join(str(v) for v in struct.unpack(
        "<3q", _raw("ple_lm"))))
    lines.append(f"ple_layer={PLE_LAYER}")
    lines.append(f"ple_table_shards={PLE_TABLE_SHARDS}")
    lines.append(f"ple_table_rows_per_shard={PLE_TABLE_ROWS}")
    lines.append(f"ple_table_rows_padded={ple_tbl_rows}")
    lines.append(f"ple_table_shard0_file={ple_tbl_file0}")
    lines.append(f"ple_table_shard0_offset={ple_tbl_off0}")

    text = "\n".join(lines) + "\n"
    if check_only:
        with open(out_path) as f:
            old = f.read()
        if old != text:
            raise SystemExit(f"[FAIL] {out_path} 与 checkpoint 现况不一致（重新生成）")
        print(f"[ok] manifest 与 checkpoint 一致：{out_path}")
        return

    with open(out_path, "w") as f:
        f.write(text)
    print(f"[ok] 写出 {out_path}")
    print(f"     层数 {n_layers}（{n_gdn} GDN + {n_attn} attention，interval={interval}）")
    n_moe = sum(1 for l in lines if ' role=moe_' in l)
    n_hc = sum(1 for l in lines if ' role=attn_hc_' in l or ' role=mlp_hc_' in l)
    n_gm = sum(1 for l in lines if ' role=gmixer_' in l)
    # M82：attention 的 9 个 role 以 `attn_` 开头，但 `attn_hc_*`（hc 边界）也以 `attn_` 开头
    # ⇒ 计数必须**按匹配片段排除**，不能只按前缀（否则 hc 的 4 个 role × 48 层会被算进来）
    n_attn = sum(1 for l in lines if l.startswith('tensor ') and ' role=attn_' in l
                 and ' role=attn_hc_' not in l)
    n_gdn_rows = sum(1 for l in lines if l.startswith('tensor ') and ' role=moe_' not in l
                     and ' role=moe512_' not in l
                     and ' role=attn_hc_' not in l and ' role=mlp_hc_' not in l
                     and ' role=gmixer_' not in l and ' role=attn_' not in l)
    # M93：真规模段单独计数（role 前缀 `moe512_`，与上面 4 专家档的 `moe_` 无子串关系）
    n_moe512 = sum(1 for l in lines if l.startswith('tensor ') and ' role=moe512_' in l)
    # M100：PLE 段单独计数（role 前缀 `ple_`；9 个小张量 + 128 个表分片，只挂层 1）
    n_ple = sum(1 for l in lines if l.startswith('tensor ') and ' role=ple_' in l)
    print(f"     张量行 {sum(1 for l in lines if l.startswith('tensor '))} 条"
          f"（GDN {n_gdn_rows} + attention {n_attn} + MoE {n_moe} + hc {n_hc} + 全局 mixer {n_gm}"
          f" + MoE 真规模 {n_moe512} + PLE {n_ple}）；"
          f"in_proj 每层拼接 {per_layer_in} 行 = qkv 10240 | z 6144 | b 48 | a 48")
    print(f"     PLE 段（M100，**加性**，只挂 0-based 层 {PLE_LAYER}）：{len(PLE_ROLES)} 个 role"
          f"（key/value_proj + conv1d + 3 个 norm + 3 个 int64）+ {PLE_TABLE_SHARDS} 个表分片"
          f"（每片 [{PLE_TABLE_ROWS},{PLE_HEAD_DIM}] BF16）= {n_ple} 行；"
          f"表 padded 行 {ple_tbl_rows}（Σsize {_total} ⇒ ceil/128×128）、每片 "
          f"{PLE_TABLE_ROWS} 行；上面 role 一行未动")
    print(f"     attention 段规模（M82）：{len(ATTN_ROLES)} 个 role × {n_attn // max(len(ATTN_ROLES), 1)} 层"
          f" = {n_attn} 行（q_proj 12,288×2560 = q 6,144 | gate 6,144；k/v 各 512×2560；"
          f"o_proj 2560×6144；q/k_norm 各 [256]；indexer index_qk 640×2560 + q/k_layernorm [128]）")
    print(f"     MoE 段规模：routed 专家 {NUM_EXPERTS_MOE}（checkpoint 512 的前 {NUM_EXPERTS_MOE} 个）"
          f"、topk {TOPK_MOE}、intermediate {INTER}；每层 12 个张量行")
    print(f"     hc 段规模：每层 2 个边界（attn_hc / mlp_hc）× 4 个张量 = {n_hc} 行；"
          f"边界宽 HYPER={HC_HYPER}、低秩 {HC_LOWRANK}、注入行 {HC_INJ_ROWS}（device 槽按 16 行、尾部置零）")
    print(f"     全局 mixer（M65）：layer={n_layers} × {n_gm} 个张量行"
          f"（norm/down/up；**无 block_inject_weight**，host 侧注入区置零）")
    print(f"     MoE 真规模段（M93，**加性**）：{len(MOE512_ROLES)} 个新 role（moe512_*）× {n_layers} 层"
          f" = {n_moe512} 行；全 {n_experts_cfg} 专家、topk {topk_cfg}；"
          f"routed 每层 {moe512_routed_layer} B × {n_layers} = {moe512_routed_layer * n_layers} B"
          f"（{moe512_routed_layer * n_layers / 1024 ** 3:.3f} GiB）；"
          f"上面的 role=moe_*（E={NUM_EXPERTS_MOE}）一行未动")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/Qwen3.8-Flash-Next-MXFP4")
    ap.add_argument("--out", default=os.path.join(HERE, "weights_manifest.txt"))
    ap.add_argument("--check", action="store_true", help="只校验已生成的 manifest，不写盘")
    ap.add_argument("--moe512-neg-experts", type=int, default=None, metavar="N",
                    help="负向对照：M93 真规模段按 N 个专家算（真值 = config num_experts）；"
                         "算式核对必 FAIL、rc=1")
    args = ap.parse_args()
    build(args.model_dir, args.out, args.check, args.moe512_neg_experts)


if __name__ == "__main__":
    main()
