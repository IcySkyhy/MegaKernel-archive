# Hyper-connection / PLE / QSA-indexer 规格（以 checkpoint 为准）

> M31 survey（2026-09-26，agent-hyperconn，只读）。触发：M25 发现 checkpoint 无标准 `input_layernorm`/`post_attention_layernorm`，
> 层边界实际是 hyper-connection。本文给出三套结构的**精确数学 + 张量形状 + 访存预算 + 与现有 m13/m14/m15 约定的差异清单**。
> 权威实现抄自本机 vLLM 源码 `/workspace/vllm/vllm/models/qwen4_exp/`（含 NVIDIA/AMD 两个变体），并与
> checkpoint 的 `config.json` / `model.safetensors.index.json` / safetensors header **逐项对账**（§11 记录出入）。

## 0. 引用记号

| 记号 | 文件 |
|---|---|
| `V:` | `/workspace/vllm/vllm/models/qwen4_exp/`（相对路径），如 `V:nvidia/model.py:506` |
| `V-C:` / `V-A:` / `V-N:` | 同目录下 `common/` / `amd/` / `nvidia/` |
| `W:` | `/workspace/vllm/vllm/`（vLLM 主仓相对路径） |
| `CKPT:` | `/workspace/Qwen3.8-Flash-Next-MXFP4/`，如 `CKPT:config.json#text_config.hc_count` |
| `IDX:` | `CKPT:model.safetensors.index.json` 的 `weight_map`（1898 张量） |
| `HDR:` | 直接读 safetensors header（前 8 字节 = u64 LE header 长度，其后 JSON 含 dtype/shape）实测，**不加载权重** |
| `R:` | `/workspace/ascend_mega_kernel/`（本仓相对路径） |

复现命令（HDR/IDX 全部只读，numpy 只在 `/usr/local/python3.12.13/bin/python3` 下）：

```python
import json, struct
idx = json.load(open('CKPT/model.safetensors.index.json'))['weight_map']
with open('CKPT/model-00005-of-00131.safetensors','rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(n))          # {name: {dtype, shape, data_offsets}}
```

## 1. 结论摘要（TL;DR）

1. **层边界不是 `h = h + f(norm(h))`，也不是 M25 描述的"单路 `[1,2560]` 双缓冲"。**
   48 层每层有 `attn_hyper_connection` + `mlp_hyper_connection` 两个 **GatedResidual mixer**（`V-N:model.py:267-274`），
   层间流动的是 **4 流残差 `[T, 10240]` bf16**（`hc_count=4` × `hidden_size=2560`，HC 外层 / HS 内层 layout，
   `V-C:hyperconnection.py:14-16`），且 vLLM 用**延迟 combine**：层 L 的 mlp 输出不立即加回，而是成为层 L+1 的
   `prev_block_output`，与 L+1 的 mixer 输入 RMSNorm 融合（`V-N:model.py:327-331` + `V-N:ops/hc.py:348-392`）。
   因此**跨层边界要携带 3 个张量**：`hidden_states [T,10240]` + `pending block_output [T,2560]` + `injection [T,4]`。
2. **`input_layernorm`/`post_attention_layernorm` 的替代物就是两个 mixer 的 `hc_norm`**：
   分组 Gemma-RMSNorm，`weight [10240]`、4 组 × 2560 独立归一化（`hc_per_branch_norm=True`，`V-N:model.py:265,435`），
   eps `1e-6`。**全局的 `model.norm` 也不存在** —— 最终归一化由 `hyper_connection_mixer.hc_norm` 兼任（`V-N:model.py:577-583`）。
   > 这直接回答了 M25 的待裁决项：不是"post-norm 还是 pre-norm"之争，而是"层边界 = 4 流 mixer"这一结构性差异。
3. **hyper-connection 在 decode m=1 是纯带宽账：每 token 每 step 1.3048 GB（权重），占全模型权重流 15.7%**，
   等效 1.52 ms @0.86 TB/s（M25 实测等效带宽），即 GDN 段（5.115 ms / 36 层）的 29%。
   逐层 13.5 MB×2 = 26.9 MB，是 **MoE 活跃 expert 流量（31.4 MB/层）的 86%**、GDN attention（115.9 MB/层）的 23%。
   > 交叉验证：用本数字模型反推 M25 的 141.4 µs/层 ⇒ 等效 0.816 TB/s，与"M25 报的 0.86 TB/s ≈ 名义 1.6 TB/s 的 55%"一致（§5.4）。
4. **PLE 是设计硬约束，不是"另一个小 op"**：ngram 表 **128 个分片 / 95.37 GiB**（不是 mission 里写的 100 片），
   占整模型 56.2%，**HBM（128 GiB）和本容器 host 内存（cgroup 32 GB）都装不下**。vLLM 的默认答案是
   **pinned host memory + UVA 稀疏行查找 + 提前一层的异步 prefetch**（`W:config/engram.py:38` `cpu_offload=True` 默认开启，
   `W:config/vllm.py:1394-1399` 对 `Qwen4Exp*` 自动创建 `EngramConfig`）。每 token 只查 **16 行 × 320 B = 5 KiB**。
   PLE 只在 **0-based layer 1** 生效（=1-based layer 2），一次/step。
5. **QSA indexer 比 docs/11 现有描述小得多也简单得多**：每层只需重读 **3.28 MB**（`index_qk_proj [640,2560]` + 2×`[128]`），
   `indexer_budget=2048` 是 **token top-k**，block top-k = 2048/4 = **512**；打分是 `Σ_{h=0..3} ReLU(q_h·k_pooled)`，
   **无 `1/√d` 缩放、无 per-head 权重 W**；K 侧是 **4:1 无权重均值池化**后 RMSNorm+RoPE。
   选择缓冲区宽 **2051（+1 计数列 = 2052）**，不是 2048。
6. **PLE 的哈希系数与词表布局可以从 vLLM 代码 bit-exact 复算出来，且与 checkpoint 完全一致**（§6.4 实测验证）。
   这意味着 Ascend 侧**不必从 checkpoint 读 `layer_multipliers`/`ngram_heads_*`**，可编译期常量固化。

---

## 2. Checkpoint 事实表（HDR/IDX 实测）

### 2.1 全局

| 量 | 值 | 来源 |
|---|---|---|
| 总张量数 | 1898 | IDX |
| 总字节 | 182,234,382,328 B = **169.72 GiB** | HDR 聚合 |
| 分片数 | 131 | `CKPT/` 目录 |
| `text_config.hidden_size` | 2560 | `CKPT:config.json#text_config.hidden_size` |
| `hc_count` / `hc_lowrank` | **4 / 320** | idem |
| `num_hidden_layers` | 48 | idem |
| `layer_types` | 48 项，`full_attention` 在 0-based **3,7,11,…,47**（12 层），其余 36 层 `linear_attention` | idem |
| `ple_layer_ids` | **[2]（1-based）** ⇒ 0-based layer **1** | idem + `V-C:config.py:128-138`（注释明写 1-based）+ `V-N:model.py:195`（`layer_idx + 1 in ple_layer_ids`） |
| `full_attention_interval` | 4 | idem |
| `mtp_num_hidden_layers` / `mtp.layer_types` | 1 / `["full_attention"]` | idem |
| MXFP4 量化范围 | 240 个 `quantized_tensors`，**全部是 MoE expert / shared expert**；`hyper_connection`/`ple`/`indexer`/`self_attn`/`linear_attn` **均不在其中**（bf16） | `CKPT:config.json#quantization_config.quantized_tensors` |
| 无 `input_layernorm` / `post_attention_layernorm` / `model.norm` | IDX 中匹配数为 **0 / 0 / 0** | IDX grep |

**⚠ 对 mission 文本的两处更正**（tower 在 inbox 里也问了第一条）：

- `ple.ple_embedding.ngram_embedding.shard_N.weight` 是 **128 片（N=0..127）**，不是 100 片。与 `split_ngram_parts=128` 一致（§7.4）。
- `ple_layer_ids=[2]` 是 **1-based** 口径，指向 0-based layer 1；checkpoint 张量名 `model.language_model.layers.1.ple.*` 是佐证。
  层派发表请用 **0-based = 1**（`V-N:model.py:195,201-207`）。

### 2.2 每层张量清单（体积）

| 类别 | 张量数 | 字节 | GiB | 占比 |
|---|---|---|---|---|
| **PLE（仅 layer 1：128 分片表 + 9 个小张量）** | **137** | **102,466,171,160** | **95.429 + 0.061** | **56.26%** |
| MoE routed experts（48 层，MXFP4 U8） | 240 | 64,298,680,320 | 59.883 | 35.28% |
| MTP | 31 | 5,214,301,696 | 4.856 | 2.86% |
| GDN `linear_attn`（36 层） | 324 | 4,173,020,928 | 3.886 | 2.29% |
| `embed_tokens` / `lm_head`（各 `[248320,2560]` bf16，未 tie） | 1+1 | 1,271,398,400 ×2 | 1.184 ×2 | 0.70 ×2 |
| **hyper-connection（48 层 + 全局 mixer）** | **384 + 3** | **1,268,121,600 + 13,127,680** | **1.181 + 0.012** | **0.70%** |
| QSA `self_attn`（12 层，含 indexer） | 72 + 36 | 1,195,388,928 + 39,327,744 | 1.113 + 0.037 | 0.66% |
| vision（`model.visual.*`） | 333 | 897,862,112 | 0.836 | 0.49% |
| MoE shared expert（48 层） | 336 | 125,583,360 | 0.117 | 0.07% |

> 张量数合计 137+240+31+324+2+387+108+333+336 = **1898** ✔（PLE 的 137 已含 128 分片与 9 个小张量；HC 的 387 = 48×8+3；QSA 的 108 = 12×6 + 12×3）。

单层体积（HDR 聚合）：**GDN 层 1415.75 MiB = 1.3826 GiB**（`linear_attn` 110.55 + HC 25.20 + MoE 1280.0 + 5.0）
/ **QSA 层 1403.33 MiB = 1.3695 GiB**（`self_attn` 95.0 + indexer 3.13 + HC 25.20 + MoE 1280.0 + 5.0）。
> **口径声明**：tower 早前在 inbox 交接里给的「1.40 GiB（full-attention 层）/ 1.42 GiB（GDN 层）」为粗算，
> `docs/01-environment.md:172` 与 `docs/04-summary.md:14` 现表述为「单层 ~1.4 GiB」——**两者都以本节为准**：
> full-attention(QSA) 层 **1.3695 GiB**、GDN 层 **1.3826 GiB**。本文数字为逐张量聚合口径，与 169.72 GiB 总量自洽
> （48 层 66.4 GiB + PLE 95.43 + MTP 4.86 + embed/lm_head 2.37 + vision 0.84 ≈ 169.8 GiB）。

### 2.3 Hyper-connection 张量形状（HDR 实测）

| 张量名（`model.language_model.layers.{0..47}.` 前缀） | dtype | shape | 字节 | 数量 |
|---|---|---|---|---|
| `attn_hyper_connection.hc_norm.weight` | BF16 | `[10240]` | 20,480 | 48 |
| `attn_hyper_connection.input_mix_weight_down.weight` | BF16 | `[320, 10240]` | 6,553,600 | 48 |
| `attn_hyper_connection.input_mix_weight_up.weight` | BF16 | `[10240, 320]` | 6,553,600 | 48 |
| `attn_hyper_connection.block_inject_weight.weight` | BF16 | `[4, 10240]` | 81,920 | 48 |
| `mlp_hyper_connection.*`（同上四件） | BF16 | 同上 | 同上 | 48 |
| `model.language_model.hyper_connection_mixer.hc_norm.weight` | BF16 | `[10240]` | 20,480 | 1 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | BF16 | `[320, 10240]` | 6,553,600 | 1 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_up.weight` | BF16 | `[10240, 320]` | 6,553,600 | 1 |

要点：

- `hc_norm.weight` 是 **`[10240]` 而不是 `[2560]`** ⇒ 证实 `hc_per_branch_norm=True`，即 4 个 stream 各自独立 RMSNorm，
  但保留**逐元素** affine（`V-C:hyperconnection.py:161-172`；`V-N:ops/hc.py:36-37` 注释 "a [DIM] affine follows the grouped checkpoint layout"）。
- **全局 mixer 没有 `block_inject_weight`**（3 个张量而不是 4 个）—— 与 `use_combine=False` 对应（`V-N:model.py:439`），
  最后一个 mixer 不再产生 injection。IDX 中确实只有 3 个 `hyper_connection_mixer.*`。
- `hc_lowrank=320` 体现在 down/up 的内维：`input_mix_weight_down [320,10240]`、`input_mix_weight_up [10240,320]`。
  运行时把 down(320) + inject(4) **拼成一条 merged linear**：`pad_size = (-(320+4)) % 16 = 12`，合并输出 **320+4+12 = 336 行**
  （`V-N:hyperconnection.py:95-107`，`V-N:model.py:143-155` 的 `orig_to_new_stacked` 映射，`V-N:model.py:651-655` 的 `packed_modules_mapping`）。
  即 checkpoint 的 324 行在运行时会**多读 12 行 padding = 245,760 B/mixer**（占该 GEMM 3.6%）。Ascend 侧可以不复用 padding（§9）。
- MTP 层同样带 HC（`mtp.layers.0.attn_hyper_connection.*` + `mlp_hyper_connection.*`，各 4 件 bf16，形状同上）
  与 `mtp.hyper_connection_mixer.*`（3 件）；另有 `mtp.pre_fc_norm_hidden.weight [10240]` —— **MTP 消费的正是 10240 宽的 HC 多流态**
  （`V-N:model.py:584-589`）。MTP **不含 PLE**（`V-N:mtp.py:212-219` 使 `layer_idx+1` 落到 `ple_layer_ids` 外）。

---

## 3. Hyper-connection 精确数学

两个变体，同一份数学：

- **参考实现（纯 torch，最易读）**：`V-C:hyperconnection.py` `GatedResidual`（`class` 在 :140，`mix` 在 :203-222，`combine` 在 :224-242）。
- **NVIDIA 生产实现（Triton + vLLM Linear）**：`V-N:hyperconnection.py:50-196`，胶水核在 `V-N:ops/hc.py`。
  **数学与参考实现一致**，差别只在：(a) 合并 down+inject 成一条 GEMM；(b) 把 "combine 上一层 block output + 本层 RMSNorm" 融成一个核
  `_hc_combine_norm_kernel`（`V-N:ops/hc.py:273-392`），并在融合点**先把 combine 结果舍入到 bf16 再做 RMSNorm**（`V-N:ops/hc.py:325-327`）。
- **AMD 变体**与 NVIDIA 逐行同构（`V-A:hyperconnection.py:50-198`），仅注释与 `pad_size` 说明措辞不同。

### 3.1 记号

```
HC = hc_count = 4          H = hidden_size = 2560        C = HC*H = 10240
R  = hc_lowrank = 320      eps = rms_norm_eps = 1e-6
h          : [T, C]  bf16   层间的 4 流残差（HC 外层 / HS 内层：reshape 成 [T, HC, H] 后 stream s 占列 [s*H, (s+1)*H)）
block_out  : [T, H]  bf16   子层（attention 或 mlp）的输出
inj_logits : [T, HC] bf16   injection 未激活的 logits
W_norm     : [C]     bf16   hc_norm.weight（逐元素 affine，按 4 组分别归一化）
W_dn       : [R, C]  bf16   input_mix_weight_down
W_up       : [C, R]  bf16   input_mix_weight_up
W_inj      : [HC, C] bf16   block_inject_weight
```

### 3.2 逐句伪码（生产路径，`V-N:hyperconnection.py:128-196`）

```python
def grouped_gemma_rmsnorm(x[C], W_norm[C], eps, HC):     # V-N:ops/hc.py:13-78
    # 4 个 group，每组 GROUP_DIM = C//HC = 2560 各自归约；affine 是逐元素 [C]
    for s in range(HC):
        xg = x[s*H : (s+1)*H]                            # [H]
        rrms = rsqrt(mean(xg*xg) + eps)                  # 组内 RMS
        y[s*H:(s+1)*H] = xg * rrms * (1 + W_norm[s*H:(s+1)*H])   # Gemma 风格 (1+w)
    return y                                             # [C]  (保持 bf16)

def mixer_mix(h[C]):
    # --- ① 归一化（分组，per-stream）---
    xn = grouped_gemma_rmsnorm(h, W_norm, eps, HC)        # [C]
    if use_combine:
        # --- ② down 与 inject 合并的一条 GEMM：[C] -> [R + HC + pad] ---
        dn_inj = linear(xn, stack([W_dn, W_inj], dim=0)) # [R + HC + pad] = [320+4+12=336]
        lora      = dn_inj[0:R]                          # [320]
        inj_logits= dn_inj[R:R+HC]                       # [4]
    else:                                                # 最后一个 mixer
        lora = linear(xn, W_dn); inj_logits = None
    # --- ③ silu（注意：先除以 HC 再 silu）---
    lora = (lora / HC) * sigmoid(lora / HC)              # [320]   V-N:ops/hc.py:100-101
    # --- ④ up GEMM ---
    gate = linear(lora, W_up)                            # [10240]
    # --- ⑤ 门控均值：把 4 个 stream 压回 [H] 作为子层输入 ---
    for j in range(H):
        acc = sum(sigmoid(gate[s*H+j]) * xn[s*H+j] for s in range(HC))
        block_input[j] = acc / HC                        # [2560]  V-N:ops/hc.py:151-156
    return h, block_input, inj_logits

def mixer_combine(h[C], block_out[H], inj_logits[HC]|None):
    # 逐元素：out[s*H+j] = h[s*H+j] + block_out[j] * inj[s]      inj 缺省时为 1
    for s in range(HC):
        w = 1.0 if inj_logits is None else 2*sigmoid(inj_logits[s] / HC)   # [0,2]
        out[s*H:(s+1)*H] = h[s*H:(s+1)*H] + block_out * w
    return out                                           # [C]
```

**融合版本**（层间实际走这条，`combine_and_mix`，`V-N:ops/hc.py:273-392`）：

```python
def combine_and_mix(h[C], prev_block_out[H], prev_inj[HC]|None):
    for s in range(HC):
        for j in range(H):
            w   = 1.0 if prev_inj is None else 2*sigmoid(prev_inj[s]/HC)
            out = bf16(h[s*H+j] + w * prev_block_out[j])   # ← 先舍入回 bf16（v:325-327）
            h_new[s*H+j] = out                             # 落 GM
            rrms = rsqrt(mean(out[s*H:(s+1)*H]**2) + eps)  # 本 stream 的 RMS
            xn[s*H+j] = out * rrms * (1 + W_norm[s*H+j])   # ← 再逆量化性归一化（v:332-345）
    ... 然后接 mixer_mix 的 ②~⑤ 步（用 xn）
    return h_new, block_input, inj_logits
```

参考实现 `V-C:hyperconnection.py:203-242` 与上式等价（`mean(dim=-2)` ↔ `acc/HC`；`2*sigmoid(...)` ↔ 同）。
**唯一实质差异**：融合核在 combine→RMSNorm 边界插了一次 bf16 舍入；若做 bit-exact 对拍必须复刻这一步。

### 3.3 `hc_count` / `hc_lowrank` 在每个 weight 上的体现

| 参数 | 体现在哪 | 值 |
|---|---|---|
| `hc_count=4` | `hc_norm.weight` 长度 = 4×2560 = **10240**（4 组各自归一化） | 10240 |
| | `block_inject_weight` 的输出行数 = **4**（每个 stream 一个标量门） | 4 |
| | 残差流宽度、`input_mix_weight_*` 的 K 维 = **10240** | 10240 |
| | 门控均值 `/HC`、silu 前 `/HC`、inject 的 `2*sigmoid(x/HC)` 三处除以 HC | 4 |
| | `injection [T,4]` 这个额外跨层张量 | 4 |
| `hc_lowrank=320` | `input_mix_weight_down` 行数（输出内维）= **320** | 320 |
| | `input_mix_weight_up` 列数（输入内维）= **320** | 320 |
| | 低秩瓶颈宽度 = 320 = 10240/32 | 320 |
| `hc_count*hidden_size` | `input_mix_weight_*` 的 C 维 = **10240** | 10240 |
| 派生 | 运行时 merged linear 输出 = 320 + 4 + pad(12) = **336** | 336 |

### 3.4 归一化位置总表（回答 M25 的 `input_layernorm` 缺口）

| 原模型（Qwen3 类） | 本 checkpoint 的替代物 | 形状 / 组 | 每 step 实例数 | 来源 |
|---|---|---|---|---|
| `input_layernorm` | `attn_hyper_connection.hc_norm` | `[10240]`，4×2560 分组 Gemma-RMSNorm | 48 | `V-N:model.py:267-274,310-314` |
| `post_attention_layernorm` | `mlp_hyper_connection.hc_norm` | 同上 | 48 | `V-N:model.py:327-329` |
| `model.norm`（最终） | `hyper_connection_mixer.hc_norm` | 同上，`use_combine=False` | 1 | `V-N:model.py:437-441,579-583` |
| GDN 内部 per-head 归一 | `linear_attn.norm.weight` | `[128]`（m9 prolog 的 RMSNormGated，36 层） | 36 | `HDR:` 实测；`R:docs/10-gdn-analysis.md:7-9` |
| QSA 内部 q/k 归一 | `self_attn.q_norm/k_norm.weight` | `[256]`（逐 head，12 层） | 12 | `HDR:` |
| QSA indexer 归一 | `indexer.q_layernorm/k_layernorm.weight` | `[128]` | 13（12+MTP） | `HDR:`；`V-N:indexer_qsa.py:131-145` |
| PLE 内部三重归一 | `ple.norm_{key,query,conv}.weight` | `[10240]`，4×2560 分组 | 1 | `HDR:`；`V-N:ple_layer.py:116-124` |
| — | `lm_head` 之前**没有**额外 norm | — | — | IDX 无 `model.norm` |

---

## 4. 48 层完整数据流图

### 4.1 骨架（`V-N:model.py:489-590`）

```
hidden_states = embed_tokens(input_ids)                    # [T,2560] bf16
hidden_states = hidden_states.repeat(1, hc_count)          # [T,10240] —— 入口把单路复制成 4 流（V-N:model.py:506）

block_output = None ; injection = None                     # 延迟 combine 的 pending 状态
for L in 0..47:
    prefetch PLE(L+1)                                      # 提前一层发起（V-N:model.py:527-534）
    hidden_states, block_output, injection = layer_L(
        hidden_states, prev_block_output=block_output, prev_injection=injection, ...)

# 最后一层之后（末 rank）
multi_hidden, sample_hidden, _ = hyper_connection_mixer.combine_and_mix(hidden_states, block_output, injection)
#   multi_hidden  [T,10240] ← 物化后的多流态（MTP 用，V-N:model.py:584-589）
#   sample_hidden [T,2560]  ← 最终 mixer 的门控均值，直接进 lm_head（V-N:model.py:590, 678-683）
logits = lm_head(sample_hidden)
```

`Qwen4ExpDecoderLayer.forward`（`V-N:model.py:276-331`）每层内部：

```
if layer_idx == 1:                                          # ple_layer_ids=[2] → 0-based 1
    if prev_block_output is not None:                       # PLE 要写多流态，必须先物化 pending combine
        hidden = attn_hc.combine(hidden, prev_block_output, prev_injection)
        prev_block_output = prev_injection = None
    hidden = ple(hidden, input_ids, query_start_loc, ngram_context)   # 就地返回 [T,10240]

# ---- attn 段 ----
(hidden, block_input, injection) = attn_hc.combine_and_mix(hidden, prev_block_output, prev_injection)   # [T,10240],[T,2560],[T,4]
attn_out = linear_attn(block_input)   # 或 self_attn(block_input)（含 QSA indexer）, [T,2560]
# ---- mlp 段 ----
(hidden, block_input, injection) = mlp_hc.combine_and_mix(hidden, attn_out, injection)
mlp_out  = moe(block_input)                                  # 48 层全部是 MoE（§2.2）
return hidden, mlp_out, injection                            # 交给下一层
```

### 4.2 残差流宽度与跨层张量

| 张量 | 形状 | 字节/stream/层界 | 说明 |
|---|---|---|---|
| `hidden_states`（4 流残差） | `[T, 10240]` bf16 | 20,480 B/token | 层界**唯一**的长寿状态；入口由单路 repeat 得到 |
| `block_output`（pending 子层输出） | `[T, 2560]` bf16 | 5,120 B/token | 延迟 combine 的负载 |
| `injection` | `[T, 4]` bf16 | 8 B/token | pending 的门控标量 |

⇒ 每个层界跨 GM 的活跃数据 ≈ **25.6 KB/token**（不含双缓冲）。48 层界 ≈ 1.23 MB/token；
相比权重流（8.27 GB/token，§6）可忽略，但**UB/L1 尺寸与同步图必须按 3 张量 × 双缓冲规划**（§9）。

### 4.3 每层每 token 的 HC 调用次数

| op | 形状 | 每 mixer | 每层(2 mixer) | 48 层 + final |
|---|---|---|---|---|
| `combine_norm`（融合 combine + 分组 RMSNorm） | `[1,10240]`→`[1,10240]` ×2 出 | 1 | 2 | 97 |
| down+inject merged GEMM | `[1,10240] × [10240,336]` | 1 | 2 | 97 |
| silu（含 `/HC`） | `[1,320]` | 1 | 2 | 97 |
| up GEMM | `[1,320] × [320,10240]` | 1 | 2 | 97 |
| gate_mix（4 流门控均值） | `[1,10240] ×2` → `[1,2560]` | 1 | 2 | 97 |

> 第一个 `attn_hc` 的 `prev_block_output is None`，走 `mix()` 分支（无 combine 段），其余 95 个 mixer 都走 `combine_and_mix`。
> 层 0→1 因 PLE 会**打断融合**，多一次独立的 `combine` 全程 pass（`V-N:model.py:293-297`）。

### 4.4 `hc_count=4` 的"每层只有一次残差加"直觉是错的

- 常规 pre-norm 层：`h = h + f(norm(h))`，2 次加法/层（attn、mlp 各一次）。
- HC 层：**每个 mixer 都做一次 combine**（把 pending 的 2560 宽输出按 4 个标量门广播加回 10240 宽状态），
  即 2 次 combine/层，但 combine 的**数据宽度是 10240**（不是 2560），且带 4 个 sigmoid 门。
- 加上 `hc_norm` 也是 10240 宽、`gate_mix` 要读两份 10240 —— **HC 每层的激活访存量约为常规层的 4 倍**（见 §5.1）。

---

## 5. decode m=1 的 HC 额外 FLOPs 与访存字节

### 5.1 单个 mixer（m=1，只算权重流）

| 项 | 权重字节 | FLOPs（2·MAC） |
|---|---|---|
| `hc_norm.weight [10240]` | 20,480 | ~4.1e4（4×2560 组内归约 + 归一化 + affine） |
| down+inject merged `[336,10240]` | 6,881,280 | 2·10240·336 = 6.881e6 |
| silu `[320]` | 0 | ~1.0e3 |
| up `[10240,320]` | 6,553,600 | 2·320·10240 = 6.554e6 |
| gate_mix（读 xn + gate 各 10240） | 0（激活 40,960） | ~1.0e5 |
| **权重合计** | **13,455,360 B = 12.832 MiB** | **≈1.35e7** |

若按 checkpoint 原始 324 行（不补 pad）：13,209,600 B = 12.598 MiB；全局 mixer（`use_combine=False`）：13,127,680 B = 12.520 MiB。

激活（每 mixer）：norm 读+写 40,960 + gate 读 40,960 + combine 读残差/写新态 40,960 + block 5,120 + inj 8 ≈ **128 KB**（125 KiB）。

### 5.2 每 step 总计（1 token / 48 层）

```
HC 权重 = 96 × 13,455,360 B + 13,127,680 B = 1,304,842,240 B = 1.2152 GiB = 1.3048 GB
HC 激活 ≈ 96 × 128 KB + 层界物化 ≈ 12.4 MB        （小 1 个数量级）
```

| 带宽假设 | HC 单独耗时 | 每层 |
|---|---|---|
| 1.6 TB/s（名义） | **0.816 ms** | 25.2 µs |
| 1.15 TB/s（名义 72%） | 1.135 ms | 35.0 µs |
| 0.86 TB/s（M25 实测口径） | **1.517 ms** | 46.9 µs |

算术强度 ≈ 13.5 MFLOP / 13.5 MB ≈ **1.0 FLOP/byte** ⇒ m=1 下 100% 受 HBM 带宽约束，**字节就是唯一货币**。
（这同时说明：把 HC 权重常驻 L2/HBM 不产生收益——每 token 都要全读一遍；只有**降低字节**（量化）才有收益，§9。）

### 5.3 与 GDN / attention / MoE 主体对比（每 token / step，m=1）

| 项 | MiB/层 | 层数 | GiB/token | 占比 | 备注 |
|---|---|---|---|---|---|
| GDN `linear_attn` | 110.55 | 36 | 3.8865 | **50.9%** | 全部权重每 token 全读（稠密投影） |
| QSA `self_attn` + indexer | 98.13 | 12 | 1.1500 | **15.1%** | q_proj 60 + o_proj 30 + k/v 5 + indexer 3.13 |
| MoE 活跃（10 routed expert + shared + router） | 29.90 | 48 | 1.4016 | **18.3%** | 10/512 expert × 2.49 MiB + shared 2.5 + gate 2.5 |
| **hyper-connection（2 mixer）** | **25.66** | **48** | **1.2030** | **15.7%** | 本 mission 新增项 |
| **合计** | | | **7.6410** | 100% | |
| PLE（仅 layer 1，一次） | 62.64 | 1 | 0.0612 | 0.8% | 另加 95.37 GiB 表的常驻问题（§6.3） |
| **含 PLE 总计** | | | **7.7022 GiB = 8.270 GB** | | |

关键相对量：

- **HC 每层 25.66 MiB ≈ MoE 活跃流量（29.90 MiB）的 86%，≈ GDN attention（110.55 MiB）的 23%。**
  即每层"多出一个 MoE 的带宽开销"，这是本 mission 最重要的新事实。
- HC 的权重只占整模型 0.70%，但因**每 token 全读**，在 decode 带宽账上占 15.7% —— 典型的"小权重、高占用"。
- 端到端推算（纯权重下限）：总量 8.270 GB/token

| 带宽 | ms/token | tok/s |
|---|---|---|
| 1.6 TB/s | 5.17 | 193 |
| 1.15 TB/s | 7.19 | 139 |
| 0.86 TB/s | 9.62 | 104 |

### 5.4 与 M25 实测的交叉验证（重要，说明本预算可信）

M25 ver C：36 GDN 层 × 141.4 µs = 5.115 ms/step、AIC MTE2 占 70.3%（~0.86 TB/s）。

用本表的字节模型反推：`36 × 110.55 MiB = 3.8865 GiB = 4.173 GB`，`4.173 GB / 5.115 ms = 0.816 TB/s`
⇒ 预测每层 `110.55 MiB / 0.816 TB/s = 142.2 µs`，**实测 141.4 µs，误差 0.6%**。
结论：**本表的每类字节预算与实际硬件效率自洽**，可以直接用于立项估算；
按 0.816 TB/s，全模型 48 层（含 HC 与 MoE）≈ **10.1 ms/token（99 tok/s）**，HC 占其中 1.60 ms（15.8%）。

---

## 6. PLE（n-gram / Engram 嵌入层）精确语义

实现：`V-N:ple_layer.py`（`Qwen4ExpPLELayer`，:66）、`V-N:ngram_embedding.py`（`Qwen4ExpNGramEmbedding`，:43）、
`V-N:ops/ple.py`（三个 Triton 核 + 包装）、`V-C:ple.py`、`V-C:ngram_embedding.py`（device / pinned-host 两种存储）。
AMD 变体 `V-A:ple_layer.py` 数学相同（但 key/value 分开，见 §11）。

### 6.1 挂载点与调用频率

- **0-based layer 1**（=1-based layer 2）。构造条件 `(self.layer_idx + 1) in ple_layer_ids`（`V-N:model.py:193-207`），
  checkpoint 张量名 `model.language_model.layers.1.ple.*` 独立佐证。
- 每 forward **一次**；其余 47 层 `self.ple is None`，零开销。
- **prefetch 提前一层**：层 k 的 n-gram id 计算与行查找在层 k-1 运行时已在 side-stream 发起（`V-N:model.py:468-487,515-534`），
  用来掩盖 pinned-host 查找的延迟。
- MTP 层**不挂 PLE**（`V-N:mtp.py:212-219` 传入 `num_hidden_layers + idx` 作 layer_idx）。
- PP>1 不支持（`V-N:model_state.py:35-41`）。
- PLE 在层 1、不是层 0 ⇒ **实际总是存在 pending combine**，`attn_hc.combine` 的提前物化必然发生（`V-N:model.py:290-297`）。

### 6.2 精确伪码（decode，T=1 token）

常量（`CKPT:config.json#text_config`）：

```
N=ngram_size=3   P=heads_per_ngram=8   G=(N-1)*P=16   ple_embed_dim He=2560
head_dim He/G = 160     ngram_vocab_size_base=20,000,000
make_ngram_vocab_size_divisible_by=128   split_ngram_parts=128
ple_conv_kernel_size=4   conv dilation = N = 3   state_len = (4-1)*3 = 9
eos_token_id=248044   rms_norm_eps=1e-6
```

```python
# ---------- ① n-gram id（V-N:ops/ple.py:25-150）----------
# ngram_context[r] = [computed-2, computed-1] 两个 token（不足则 EOS）；V-N:model_state.py:43-93
c    = t - query_start_loc[r]           # 本 step 内的 chunk 位置
cur  = input_ids[t]
prev1= cur-1 的 token，且 chunk 内 c>=1，否则 ngram_context[r, NGRAM_CONTEXT_LEN-1+c]   # c=0 → ctx[r,1]
prev2= cur-2 的 token，且 chunk 内 c>=2，否则 ngram_context[r, NGRAM_CONTEXT_LEN-2+c]   # c=0 → ctx[r,0]；c=1 → ctx[r,1]
#   ★ D3（M89 更正，2026-09-27）：上下文列**随 c 变**，源码 `V-N:ops/ple.py:79-91` 是
#     `ctx_col = NGRAM_CONTEXT_LEN - shift + chunk_pos`（shift=1→prev1、shift=2→prev2）；
#     eager 路同一口径 `V-N:ngram_embedding.py:306-350`（context = cat([ngram_context, packed])，
#     列号 `adjusted_columns = c + ngram_size - 1`（ngram_size = NC+1 ⇒ = c + NGRAM_CONTEXT_LEN），
#     取 `shifted[shift][r, adjusted_columns]` = `context[c + NGRAM_CONTEXT_LEN - shift]`）。
#     ⇒ 原稿把 prev2 的上下文列写死成 `ctx[r,0]` 是**结构性简化**，**c=1 的 prev2 实际取 `ctx[r,1]`**（详见 §6.2.1）
# EOS 回退：一旦 backward 遇到 EOS，更老的位置全部替换成 EOS（v:78-98 "crossed"）
mixed = cur  * m[0]
mixed ^= prev1 * m[1]          # 只对 ngram_order > 1 的 head 生效
mixed ^= prev2 * m[2]          # 只对 ngram_order > 2 的 head 生效
#   m = layer_multipliers : int64[3]（巨大奇数哈希乘子，见 §6.4）
#   head g 的 ngram_order = g // P + 2  (v:74)
#   ⇒ g=0..7 是 bigram(cur,prev1)，g=8..15 是 trigram(cur,prev1,prev2)
for g in 0..15:
    ids[t,g] = floor_mod(mixed, size[g]) + offset[g]     # 非负余数；v:104-106
# ids : int64[T,16]，值域 [0, 320,001,446)

# ---------- ② 行查找（稀疏 gather）----------
emb16 = table[ids]                      # [T,16,160] bf16；table = [320001536,160] bf16（128 个 checkpoint 分片在加载时拼成一条参数，V-C:ngram_embedding.py:326-333）
emb   = emb16.flatten(-2)               # [T,2560]

# ---------- ③ key/value 投影（TP-replicated, disable_tp=True）----------
kv    = W_kv @ emb                      # W_kv = [12800,2560] = merge(key_proj[10240,2560] , value_proj[2560,2560])
key, value = kv[:, :10240], kv[:, 10240:]      # [T,10240], [T,2560]
#   ★ D1（M89 更正，2026-09-27）：checkpoint **没有** `ple.kv_proj` —— `IDX:` 的 `weight_map` 里该键不存在，
#     只有 `ple.key_proj.weight` + `ple.value_proj.weight`；融合发生在**装载期**（`V-N:ple_layer.py:107-115`
#     的 `MergedColumnParallelLinear(2560, [10240, 2560], disable_tp=True)`，映射表 `V-N:model.py:153-154`
#     把 key→`kv_proj` 分片 0、value→分片 1）⇒ **key 在前 10240 列、value 在后 2560 列，顺序不可反**：
#     反了张量形状照样合法（都是 [12800,2560]）但**静默读错**。详见 §6.2.1 与 §11 #3

# ---------- ④ gate + 分组归一（融合核 V-N:ops/ple.py:185-240；逐 (token, stream) 一个 program）----------
for s in 0..3:                                        # stream s 覆盖列 [s*2560,(s+1)*2560)
    k = key[t, s*2560:(s+1)*2560]
    q = hidden[t, s*2560:(s+1)*2560]                  # ★ query 就是当前 4 流残差本身
    k_n = bf16( k * rsqrt(mean(k^2)+1e-6) * (1 + norm_key  [s*2560+·]) )
    q_n = bf16( q * rsqrt(mean(q^2)+1e-6) * (1 + norm_query[s*2560+·]) )
    # ★ D2 同族补正（M89，2026-09-27）：上游**逐中间张量边界**都回 bf16（`V-N:ops/ple.py:218-229`
    #   的注释 "Match eager materialization at each intermediate tensor boundary"），原稿只写了首尾两步。
    #   逐字链（`V-N:ops/ple.py:221-232`）：
    prod = bf16( k_n * q_n )                          # 组内逐元素积也物化
    dot  = bf16( Σ fp32(prod) )                       # ★ 组内点积 → 标量（先回 bf16）
    d    = bf16( fp32(dot) / sqrt(2560) )             # 缩放后再回 bf16
    g    = bf16( sigmoid( sign(d) * bf16(sqrt(max(|d|, 1e-6))) ) )   # ★ 值域 (0,1)
    v    = value[t, 0:2560]                           # ★ 4 个 stream 共享同一个 value
    gated [t, s*2560:(s+1)*2560] = bf16( g * v )
    convin[t, s*2560:(s+1)*2560] = bf16( gated * rsqrt(mean(gated^2)+1e-6) * (1 + norm_conv[s*2560+·]) )

# ---------- ⑤ 膨胀深度卷积 + 状态更新 + 残差加（V-N:ops/ple.py:437-462）----------
# taps: lag 9,6,3,0（dilation=3, kernel=4, 感受野 9；w = conv1d.weight.squeeze(1) : bf16[10240,4]）
acc[j]          = Σ_{k=0..3} w[c,k] * (conv_state[c, j+3k] if j+3k<=8 else convin[q_start+j+3k-9])
conv[j]         = bf16( acc[j] )                                # F.conv1d 先物化输出 dtype（源码 :438）
y[j]            = fp32(conv[j]) * sigmoid(fp32(conv[j]))        # SiLU，中间量在 fp32
conv_output[j]  = bf16( y[j] )                                  # ★ D2（M89 更正，2026-09-27）：原稿漏了这一取整
gated_output[t] = bf16( gated[t] + conv_output[t] )             # 就地累加（gated = ④ 的输出；源码 :441-449）
out[t]          = bf16( fp32(hidden[t]) + fp32(gated_output[t]) )   # ★ 残差加，返回给模型当新的 hidden_states
#   源码逐字（V-N:ops/ple.py:437-462）：conv = acc.to(dtype) → y = conv*sigmoid(conv) → conv_output = bf16(y)
#   → ple_output = bf16(residual + conv_output) → ple_output = fp32(outer_residual) + fp32(ple_output) → store(bf16)
#   （自带注释："F.conv1d materializes its output dtype before SiLU" / "Preserve the original eager operation boundaries"）
```

**与 HC 的耦合（M25 规格的关键点）**：PLE 输出直接加到 **10240 宽的多流态**上（不是 2560），
并且把"当前多流态"当 **query**（说明 PLE 不是纯加性 embedding，而是随残差态变化的门控修正）。
NVIDIA 版把残差加**融进卷积核**（返回已含残差的 `[T,10240]`）；AMD 版返回 gate+conv，由调用方再加（`V-A:ple_layer.py:1125`、`V-A:model.py:298-303`）——**两种约定不要混用**。

### 6.2.1 与上游源码对账发现的结构性简化（D1/D2/D3；M89，2026-09-27）

> **时点与口径**：本节三条来自 **2026-09-27** 的实测对账 —— M85 按人类要求「与官方 vllm qwen4 exp 对齐」复核 PLE 规格时记下 **D1/D2**（出处：`.tower/comms/inbox/20260927-agent-ple-tower-review-request-m85-ple-kernel-11-11-tip-6e5d010.md`），`reviewer-m85` 的 r1 审查件追加 **D3**（出处：`.tower/comms/reviews/review-feat-m85-ple-kernel-spec-pin-and-standalone-i-reviewer-m85-r1.md`）；M89 逐条**重开上游源码复核**后回写。
> **不改写历史痕迹**（照 `docs/05` §11.5 的纪律：判据 = 命令 + 跑的 commit + 当场读数；本节改的是「我们对上游的转述」，与某个 commit 无关，故记**时点 + 上游 `文件:符号`**）：每行都保留**原稿写的是什么**与**当时的取舍**（取舍一栏是 **M89 的推断，非 M31 自述**：M89 只复核到"原稿这么写了"，没有据以推断动机的直接材料）。
> 上游权威 = 本机 `/workspace/vllm/vllm/models/qwen4_exp/`（`V-N:` = `nvidia/`、`V-C:` = `common/`、`V-A:` = `amd/`，记号见 §0）。
> 人类总口径：「**尽量和官方 vllm 的 qwen4 exp 对齐**，要求权重加载就能直接用、尽量少的**在线**格式转换」⇒ 凡涉及**字节布局 / 权重切片 / 舍入口径**的一律以上游为准。

| # | §6.2 原稿（写的什么 + 当时的取舍） | 上游 `文件:符号`（M89 逐字复核） | 差异与更正 |
|---|---|---|---|
| **D1** | ③ 只写 `W_kv = merge(key_proj, value_proj) → [T,12800]`。**没说**这条融合张量在 checkpoint 里不存在、没说融合发生在**装载期**、也没说 **key 在前**。当时的取舍（**M89 的推断，非 M31 自述**）：伪码只对数学负责，权重组织归 §11 #3 | `V-N:ple_layer.py:107-115`（`MergedColumnParallelLinear(2560, [10240, 2560], bias=False, disable_tp=True)`，输出尺寸 = `[hc_hidden_size=10240, hidden_size=2560]`）；`V-N:model.py:153-154`（`"ple.key_proj" → ("ple.kv_proj", 0)`、`"ple.value_proj" → ("ple.kv_proj", 1)`）；`CKPT:` 的 `weight_map` | checkpoint **没有** `ple.kv_proj`（`IDX:` 实测该键不存在，只有 `ple.key_proj.weight` + `ple.value_proj.weight`）；融合是**装载期**行为，且 **key 占前 10240 列、value 占后 2560 列，顺序不可反** —— 反了形状照样合法（都是 `[12800,2560]`）⇒ **静默读错**。⇒ ③ 已补注；§11 #3 只登记了**命名**差异，本条补的是**字节序契约** |
| **D2** | ⑤ 原稿 `y = silu(bf16(Σ…))` → `gated_output = bf16(gated + y)` → `out = bf16(hidden + gated_output)`：**只保留"卷积累加后一次 bf16"**，SiLU 之后**不再取整**。当时的取舍（**M89 的推断，非 M31 自述**）：按"每个中间张量边界"复刻舍入，但漏了 silu 输出那一处 | `V-N:ops/ple.py:437-462`：`conv = acc.to(dtype)` → `y = conv*sigmoid(conv)` → `conv_output = …to(dtype)` → `ple_output = (residual + conv_output).to(dtype)` → `ple_output = outer_residual.to(fp32) + ple_output.to(fp32)` → `tl.store`（bf16） | **少一次 bf16 取整**：源码在 **SiLU 之后**把 `y` 物化成 `conv_output = bf16(y)` 再加到 `gated` 上。⇒ ⑤ 已补 `conv_output[j] = bf16(y[j])`。**同族补正**：④ 的 gate 链在上游同样**逐中间边界**物化（`V-N:ops/ple.py:218-229` 自带注释 "Match eager materialization at each intermediate tensor boundary"，`prod`/`dot`/`d`/`magnitude`/`g` 各回一次 bf16），原稿只写了首尾两步 —— ④ 亦已按源码逐字改写（这一条是 M89 复核 D2 时在同一条源码里发现的，**不是** M85 的 D 系列编号）|
| **D3** | ① 原稿 `prev2= … 否则 ngram_context[r,0]`：把 prev2 的上下文列写成**与 c 无关的固定列 0**（prev1 那行写 `ctx[r,1]` 恰好对，因为 prev1 只在 c=0 时取上下文）。当时的取舍（**M89 的推断，非 M31 自述**）：按"chunk 内 c>=2 用本 step 的 token、否则吃 context"平铺，未逐 shift 展开 `ctx_col` | triton `V-N:ops/ple.py:79-91`（`ctx_col = NGRAM_CONTEXT_LEN - shift + chunk_pos`）；eager `V-N:ngram_embedding.py:306-350`（`context = cat([ngram_context, packed], -1)`、`adjusted_columns = columns + ngram_size - 1`、取 `shifted[shift][r, adjusted_columns]` = `context[c + NGRAM_CONTEXT_LEN - shift]`） | 上下文列**随 c 变**：`ctx_col = NGRAM_CONTEXT_LEN - shift + c`（shift=1→prev1、shift=2→prev2）。**c=1 的 prev2 取 `ctx[r,1]`**，`ctx[r,0]` 只在 c=0 成立。⇒ ① 已按公式改写 |

**这三条为什么必须回写**（都有 M85 的实测 / r1 评审在案，不是纸面抠字）：

- **D3**：M85 的 PLE kernel **第一版**在 `c=1` 取错了 n-gram 上下文列（SPEC / kernel / 自查参考三处同错）。其 r1 评审用 triton 与 eager 两条上游路的 numpy port 对拍，指出 160 个 id 里有 **16 个**与它们不同（`t=1` 与 `t=7` 的 `g=8..15` —— 正是两个 `c=1` token 的 trigram 头）；因为 device 与自查参考**同错**，判据 1 对该 case 是**空洞通过**（PASS 不代表对）。
- **D2 同族**：同一份 r1 评审独立抓到 kernel **漏了 gate 点积那一步的 bf16 物化**（Reduce 结果直接进 `Muls`）；评审用 **4000 组随机 `(k_n,q_n)`** 量到该处不一致率 d **28%** / g **约 2%** —— 即"中间边界少一次取整"换了输入就现。
- **D1**：决定 Ascend 侧自建 PLE 表的**字节序**（key 前 value 后）。

**本 mission 只负责文档侧更正**；kernel / SPEC 侧的收口归 M85 的后续 commit（不在本 mission 范围）。

### 6.3 权重体积、每 token 开销、常驻问题

**权重（HDR 实测，137 个张量 = 128 分片 + 9 个小张量）**

| 张量 | dtype | shape | 字节 |
|---|---|---|---|
| `ple_embedding.ngram_embedding.shard_0..127.weight` | BF16 | `[2500012, 160]` each | 800,003,840 each |
| `ple.key_proj.weight` | BF16 | `[10240, 2560]` | 52,428,800 |
| `ple.value_proj.weight` | BF16 | `[2560, 2560]` | 13,107,200 |
| `ple.conv1d.weight` | BF16 | `[10240, 1, 4]` | 81,920 |
| `ple.norm_key/query/conv.weight` | BF16 | `[10240]` ×3 | 20,480 ×3 |
| `ple_embedding.layer_multipliers` | I64 | `[3]` | 24 |
| `ple_embedding.ngram_heads_vocab_sizes` | I64 | `[16]` | 128 |
| `ple_embedding.ngram_heads_offsets` | I64 | `[16]` | 128 |

- **ngram 表 = 320,001,536 × 160 × 2 B = 102,400,491,520 B = 95.37 GiB**（128 × 762.94 MiB），占整模型 56.2%。
- 其余 PLE 权重 = **65,679,640 B = 62.64 MiB**（含 3 个 int64 buffer 的 280 B）。

**每 token（decode）开销**

| 项 | 字节 | 说明 |
|---|---|---|
| 行查找 payload | **5,120 B**（16 行 × 320 B） | 32B 扇区粒度下实测放大 1.15–1.6×（约 5.9–7.2 KiB） |
| `key_proj` | 52,428,800 | m=1 GEMV，权重全读 |
| `value_proj` | 13,107,200 | |
| `conv1d` + 3 个 norm | 81,920 + 61,440 | |
| **权重合计** | **65,679,640 B = 62.64 MiB** | 占全 step 权重流的 **0.8%** |
| 激活 | ≈130 KB | key/value/gated/conv_in 各 10240×2 B 起 |
| **short-conv 状态** | **184,320 B = 180 KiB / 序列** | `(9, 10240)` bf16；TP-replicated；独立于 GDN 的 MambaSpec |

**是否需要全量 shard 驻留 HBM？不需要，而且默认根本不驻留 device。** 两条路径（`V-N:ngram_embedding.py:210-226`）：

1. **Pinned host（默认）**：`W:config/vllm.py:1394-1399` 只要 architecture 是 `Qwen4Exp*` 且 `ple_layer_ids` 非空，就自动创建 `EngramConfig()`，
   而 **`EngramConfig.cpu_offload` 默认 `True`**（`W:config/engram.py:38`）⇒ 选用 `Qwen4ExpPLEPinnedHostEmbedding`，
   把 95.37 GiB 放在 **pinned host RAM**，通过 UVA 按行读（`V-C:ngram_embedding.py:385-441`），
   每 step 只有 ~5 KiB 过 host↔device 链路，并由提前一层的异步 prefetch 掩盖（`V-C:ngram_embedding.py:502-516`）。
   需要 `is_uva_available()`（`V-C:ngram_embedding.py:403-404`）。
2. **Device 常驻（`EngramConfig(cpu_offload=False)`）**：`Qwen4ExpPLEDeviceEmbedding` 把 95.37 GiB 全放 device（`V-C:ngram_embedding.py:323-352`），
   这时**才**需要全部 320,001,536 行常驻；每 step 流量仍是 ~5 KiB。

> **对本项目的硬约束**：本机 NPU 有 128 GiB HBM，host 有 754 GB 物理内存，但**容器 cgroup 内存上限 32 GB**（`R:docs/01-environment.md:165`），
> `/dev/shm` 16 GB。⇒ **95.37 GiB 的表两处都放不下**，vLLM 的两条路径在本容器里都不能直接照搬。
> 可行的方向只有"磁盘驻留 + 稀疏行查找"（`R:docs/01-environment.md:173` 已记录该结论）。这是**立项前必须与用户确认的阻塞项**（§10）。
> 另：`EngramConfig.verify_model_config` 要求 `current_platform.is_cuda_alike()`（`W:config/engram.py:77-87`），
> 在 Ascend NPU 上会直接抛异常 ⇒ Ascend 移植必须绕开 EngramConfig（自己申请 pinned/host 表）。

### 6.4 n-gram 索引算术的 bit-exact 复算（实测验证）

`layer_multipliers` 与 `ngram_heads_vocab_sizes/offsets` 在 vLLM 里是**确定性生成**的（`V-N:ngram_embedding.py:101-138`），
不是从 checkpoint 学来的。我用代码里的常数复算，与 checkpoint buffer **逐值相等**：

```
max_multiplier = ((1<<63)-1)//248320 = 37,143,089,710,272 ; half_bound = 18,571,544,855,136
base_seed      = seed + 10007*ple_dense_layer_id = 1234 + 0        (config 无 seed ⇒ 默认 1234，V-N:ngram_embedding.py:176)
m[i] = 2*(splitmix64(base_seed + 0x9E3779B97F4A7C15*(i+1)) % half_bound) + 1
     ⇒ [23703573157769, 20109073645365, 8052911324071]              ≡ CKPT 实测值 ✔

size[h]   = nth_prime_after(20,000,000-1, h+1)
          = [20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,
             20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171]  ≡ CKPT 实测值 ✔
offset[h] = Σ_{j<h} size[j] = [0,20000003,40000026,60000059,80000106,100000165,120000228,
             140000297,160000374,180000455,200000548,220000655,240000802,260000955,280001114,300001275] ≡ ✔
total = 320,001,446 ; padded = ceil(total/128)*128 = 320,001,536 ; shard rows = 320,001,536/128 = 2,500,012 ≡ shard shape [2500012,160] ✔
```

**结论**：(a) 索引算术可在 Ascend 侧用编译期常量固化，无需读 checkpoint 的 3 个 buffer；
(b) checkpoint 与 vLLM main 在 PLE 索引语义上完全一致（无版本漂移）；(c) `split_ngram_parts=128` 必须来自 `text_config`，
因为 `Qwen4ExpTextConfig` **没有声明**该字段（`V-C:config.py:39-54`），代码用 `getattr(config,"split_ngram_parts",512)` 兜底
（`V-N:ngram_embedding.py:169`）；若回落到 512，分片行数会变成 625,003 而加载失败。

---

## 7. QSA indexer 精确算法

实现：`V-N:indexer_qsa.py`（`QSAIndexer`，:89）、`V-N:qsa.py`（`Qwen4ExpQSAAttention`）、`V-N:ops/qsa_indexer.py`（打分/top-k/展开）、
`V-N:ops/qsa_pre_indexer.py`（融合 Q/K 预处理：norm+RoPE+4:1 池化+落 cache，:196-400）、`V-C:qsa_cache.py`（两个 paged cache）。
消费端 `V-N:ops/qsa.py:594-758` `qsa_sparse_paged_attention`。

### 7.1 触发条件与层派发

- `layer_types` 里 12 层是 `full_attention`（0-based 3,7,…,47）。这些层**实际是 QSA 稀疏检索注意力**，
  因为判断是 `layer_type == QSA_LAYER_TYPE or getattr(config,"indexer_n_heads",None) is not None`（`V-N:model.py:216-236`），
  而本 checkpoint 设置了全套 indexer 字段。
- `indexer_n_heads=4`、`indexer_kv_heads=1`（MQA，**代码硬性要求 =1**，`V-C:config.py:157-158`）、`indexer_head_dim=128`。
- 每 QSA 层一次/forward ⇒ **12 次/step**（+ MTP draft 1 次）。

### 7.2 精确算法

```
常量：token_topk=indexer_budget=2048 ; R=indexer_compress_ratio=4 ; block_topk = token_topk/R = 512
      HQ=4, HK=1, D=128 ; rotary_dim = head_dim*partial_rotary_factor = 256*0.25 = 64
      eps=1e-6 ; indexer cache dtype = BF16（indexer_kv_dtype="auto"→bf16, V-N:indexer_qsa.py:151-163）

# ---- ① 投影：与主 QKV 并行，读同一份 hidden ----
proj = index_qk_proj(hidden)          # ReplicatedLinear 2560 -> (HQ+HK)*D = 640, bias=False  (V-N:indexer_qsa.py:131-137)
q    = proj[:512].reshape(T,4,128)
raw_k= proj[512:640]                  # [T,128]，未归一化、未 RoPE

# ---- ② q 路径：Gemma-RMSNorm(128) → MRoPE(前 64 维, 3 轴 interleaved) ----
q = gemma_rmsnorm(q, q_layernorm, eps)      # 逐 head 128 维，x·rsqrt(mean(x²)+eps)·(1+w)
q = rope_mrope(q, positions)                # 前 64/128 维；mrope_section=[11,11,10]（和 32 = 64/2）
q = q.to(bf16)

# ---- ③ k 路径：4:1 无权重均值池化 → 归一化 → RoPE → 写压缩 cache ----
#  池化是【先池化后归一化】，RoPE 位置取【该组第一个 token】的位置
pooled[c] = (Σ_{i=0..3} raw_k[4c+i]) / 4                    # fp32 累加 → bf16 → fp32（贴未融合路径的舍入）
k_comp    = gemma_rmsnorm(pooled, k_layernorm, eps)
k_comp    = rope_mrope(k_comp, pos = end_pos - 4 + 1)
compressed_key_cache[c] = k_comp                            # paged，1 行 / 4 token
raw_key_cache[logical_position % ring_capacity] = raw_k      # ring，供跨 chunk 的组尾使用
# ★ 跨 chunk 的组：成员 i 的位置 >= chunk 起点则读 raw 内存，否则读 ring（V-N:qsa_pre_indexer.py:236-265）
# ★ 只有【完整组】才写压缩 cache，写点在 (logical_position+1) % 4 == 0（V-C:qsa_cache.py:159-187）

# ---- ④ 打分：paged MQA + ReLU + 4 head 求和（V-N:ops/qsa_indexer.py:94-107）----
visible_blocks[row] = max(0, min( (logical_position+1)//4, seq_len//4 ))    # V-C:qsa_cache.py:256-275
score[row,c] = Σ_{h=0..3} max(0, Σ_{d=0..127} q[row,h,d] * k_comp_cache[c,d])   # fp32，c < visible_blocks
# ★ 无 1/sqrt(D) 缩放、无 per-head 权重 W（QSA 与 lightning_indexer 最大的区别）

# ---- ⑤ block top-k = 512（V-N:ops/qsa_indexer.py:471-499）----
block_indices = persistent_topk(logits, lengths=visible_blocks, k=512)

# ---- ⑥ 展开 + 因果尾巴 → 打包选择缓冲（V-N:qsa_indexer.py:221-276）----
OUTPUT_WIDTH = token_topk + R - 1 = 2051 ; PACKED_WIDTH = 2052   # 末列 = 有效计数，不是索引
expanded = min(visible_blocks[row],512) * 4
tail_start = ((logical_position+1)//4)*4 ; tail = (logical_position+1) - tail_start   # ∈[0,3]
for col in 0..2050:
    if col < expanded:               token = block_indices[row, col//4]*4 + (col%4)
    elif col-expanded < min(tail,3): token = tail_start + (col-expanded)
    else:                            token = -1                                     # padding
out[row,2051] = expanded + tail                                                     # 循环上界

# ---- ⑦ 主注意力（V-N:ops/qsa.py:594-758）----
# 顺序：indexer 先跑（更新两侧 cache 并产出索引）→ 主 KV cache 写入 → 稀疏 attention
# valid_count = indices[row,2051]；logical_token → (page, offset) → 主 paged cache gather
# score = (q·k)/sqrt(256) → online softmax → PV → out = attn_out * sigmoid(output_gate)   # GROUP_SIZE=12
```

### 7.3 权重（HDR 实测）、访存、cache 需求

| 张量（`model.language_model.layers.{3,7,…,47}.self_attn.indexer.`） | dtype | shape | 字节 |
|---|---|---|---|
| `index_qk_proj.weight` | BF16 | `[640, 2560]` | 3,276,800 |
| `q_layernorm.weight` | BF16 | `[128]` | 256 |
| `k_layernorm.weight` | BF16 | `[128]` | 256 |
| **每层合计** | | | **3,277,312 B = 3.125 MiB** |

- 13 个实例（12 层 + MTP）= 42,605,056 B = 40.63 MiB。
- **每 step 重读 12 × 3.277 MB = 39.33 MB**（≈37.5 MiB），约占全 step 权重流 **0.5%**；
  相对 QSA 层 `self_attn`（95.0 MiB）只加 **3.2%**。m=1 下算术强度同样 ≈1 FLOP/byte（每层 3.28 MFLOP）。
- 无任何 indexer 张量被量化（249 个 `weight_scale` 全属 MoE），也**没有** `W`/`ape`/额外 scale 张量：39 = 13×3，不多不少。

**两侧 paged cache（`V-C:qsa_cache.py:756-870`）**

| cache | 结构 | 字节 | 容量规则 |
|---|---|---|---|
| raw key ring（压缩器的输入暂存） | `CircularBufferSpec(block_size=capacity, num_kv_heads=1, head_size=140, dtype=bf16)` | **280 B/行** | 1 物理块/请求终生；`capacity = 4*ceil((4+num_spec)/4)`（num_spec=0 → 4 行 = **1,120 B/请求/层**） |
| compressed key cache | `MLAAttentionSpec(block_size=cache_config.block_size, num_kv_heads=1, head_size=128, tokens_per_state=4)` | 256 B/行，**64 B / token-of-context / 层** | paged，1 行 / 4 token；行数 = `cdiv(max_len, block_size)*block_size/4` |

- 行宽 140 = 128 key(bf16) + `ceil(128/4)*4=128` 元素里额外存的 **3 个 int64 MRoPE 位置**（`V-C:qsa_cache.py:814-826`）。
- 12 层合计：**768 B / token-of-context**（bf16）；262144 ctx 时 ≈ **201.3 MB** —— 这是每 step indexer 扫描的全部代价。
- 选择缓冲 `topk_indices_buffer [max_num_batched_tokens, 2052] int32`（`V-N:qsa.py:410-424`）⇒ **8,208 B / token-of-batch-capacity / 层**
  （8192 token 批 → 67.2 MiB/层，12 层 806 MiB）；decode logits 暂存 `[rows, max_model_len/4]` fp32（262144 ctx 时 256 KiB/行）。
- 两侧 cache 都不随请求数预分配，交给 vLLM 的 KV-cache 管理器；`prefix_cacheable=False`。

### 7.4 调用链

```
Qwen4ExpDecoderLayer.forward                     V-N:model.py:316-322
└─ Qwen4ExpQSAAttention.forward                  V-N:qsa.py:500-526
   ├─ qkv_proj → _project_qkv_gate               V-N:qsa.py:505-511
   ├─ indexer.index_qk_proj(hidden)              V-N:qsa.py:514   （留在 eager break 之外）
   └─ _run_qsa(...)  @eager_break_during_capture V-N:qsa.py:444-498
      ├─ QSAIndexer.forward(projected_qk, positions, topk_buf)   V-N:qsa.py:472-476
      ├─ do_kv_cache_update(...)                                  V-N:qsa.py:480-486
      └─ forward_qsa(...) → qsa_sparse_paged_attention             V-N:qsa.py:487-498
```

- indexer 严格**先于**主 KV 写与稀疏 attention；尾部 token 因此在稀疏核运行时已可见。
- **不预计算**：top-k 每 step 重算；只有 MTP 的 `skip_topk` 会复用 step-0 的索引（`V-N:indexer_qsa.py:249-251,385-388`）。
- metadata 由每 cache group 一个 `QSAMetadataBuilder` 生成，`_cudagraph_support = UNIFORM_BATCH`（`V-C:qsa_cache.py:596-646,648-721`）。

### 7.5 Ascend donor 盘点（实测本机 `/workspace`）

**没有** Qwen4Exp QSA indexer 的现成 NPU 实现：`vllm-ascend` 全仓 grep `qwen4|qsa` 为零。但有很接近的 donor：

| donor | 覆盖 QSA 的哪一段 | 差距 |
|---|---|---|
| `ops-transformer/attention/pool_key_indexer`（**950 已支持**） | 打分→block top-k→展开+尾巴→`sparse_indices [T1, topk+pool-1]`（默认 `topk=2048`、`pool_size`、`N2=1`、−1 padding，**形状与 QSA 的 2051 完全对应**） | 它有 per-head 权重 `W` 与 `1/√d` 缩放（top-k 对正缩放不变，`W` 需去掉）；`pool_key` 是**输入**——池化/norm/RoPE/ring 在它之外；无末列计数；未进 950 CI 白名单 |
| `ops-transformer/attention/lightning_indexer` | 打分 + top-k（paged PA_BSND，`N2=1`） | 无池化、有 per-head `W`；CI 里只有 `quant_lightning_indexer` 启用 |
| `ops-transformer/attention/key_pool` / `compressor` | 池化 + paged state cache（含未完成组） | QSA 是"先均值池化后 norm"且**无门控、无 apex**；`key_pool` 是"先 norm 后 softmax 加权池化" |
| `ops-transformer/attention/sparse_flash_attention` | 消费端 = `qsa_sparse_paged_attention` | **已在 vllm-ascend 的 DSA 链路上量产**（`vllm_ascend/attention/sfa_v1.py:604-638`，含 MTP `skip_topk` 复用逻辑 `:1568-1580`） |

结论：**"打分→top-k→展开→稀疏 gather" 半段有可靠 donor；"投影 + 4:1 均值池化 + RMSNorm/RoPE + raw ring" 半段只有拼件，需要自研**
（`R:docs/11-attn-analysis.md:88-90` 已列的 `posembedding`/`kv_rms_norm_rope_cache` 可复用）。

**顺带发现，`docs/11` 有两处需要更正**（M31 登记；**M89 更新状态**）：

1. **（M89 已更正，2026-09-27）** raw key cache 被写成 `[tokens,128]`；实际是**每请求一个环**，容量
   `4*ceil((4+num_spec)/4)` 行、**行宽 140 元素**（128 个 key + 3 个 int64 MRoPE 位置；`V-C:qsa_cache.py:814-826`），
   `num_spec=0` 时 4 行 = **1,120 B/请求/层**。M31 登记时引的是 `R:docs/11-attn-analysis.md:25`，**行号已漂移**。
   该形态在改前 base 上的**匹配片段数**（判据命令 + 跑的 commit + 当场读数，按片段不按行 —— §11.5 口径）：
   `git show 1c7eb5e:docs/11-attn-analysis.md | grep -oE 'tokens,128' | wc -l` → **1**（在 §3.1）；
   `… | grep -oE '每 token 128 维' | wc -l` → **1**（在 §0 的双 cache 行）。**两处都已按「行宽 140 / 每请求一个环」改写**
   （改后同一条命令可按 §11.5 在 tip 上复跑核对；本节读数取自 base `1c7eb5e`）。
2. **（仍开放；不在 M89 范围）** `docs/11` 的 donor 表漏了 **`pool_key_indexer`** —— 它才是与 QSA 打分/展开段数值上最接近的 Ascend 算子
   （950 已支持、`topk` 默认同为 2048、`sparse_indices [T1, topk+pool-1]` 形状与 QSA 的 2051 对应）。
   M31 登记时引的 `R:docs/11-attn-analysis.md:92` 行号同样已漂移，**以内容为准**。

---

## 8. 差异清单：现有 m13/m14/m15 层边界约定 vs checkpoint 实际结构

### 8.1 先纠正对现有约定的描述

本仓**实际写下的**约定并不是字面的 `h = h + f(norm(h))`：
`m13/m14` 的层链是 **S1 = Add+RMSNorm(层输入, res)**、**S7/S10 = Add+RMSNorm(子层输出, 层输入)**，
即"**层的出口是归一化后的值**"，另配一条 fp32 残差旁路（`R:m6_rmsnorm/README.md:6-9,12,29-35`；
`R:m13_moe_layer/README.md:11,21`；`R:m14_gdn_layer/README.md:14,20`；`R:m13_moe_layer.asc` 文件头段序注释（原 `:7,:20`）；
`R:m14_gdn_layer.asc` 文件头段序注释（原 `:7,:16`））。"`[1,2560]` bf16 残差流双缓冲、层间只通过它交接"这一**字面**表述出现在
M25 的 `m15_layer_loop`（分支 `feat/48-layer-loop-skeleton`，`2b9cb36` 合入 main）——在 `m15_layer_loop/README.md`（M25 写作时口径 `:8,60`）与 `m15_layer_loop.asc` 文件头段序注释与层循环入口（原 `:8,:448`）。（行号随文件变动，以符号/内容为准。）
M25 的 TowerFinding 已把缺口记全：`.tower/comms/findings/20260926-agent-loop-improve-m13-m14-add-rmsnorm-checkpoint-2560-norm-hyper-connection.md:13,21,25,28`。

**所以差异有三层**，不要只改一层：
① 层出口语义（normalized vs raw residual）—— M25 已知；
② **残差流宽度与**流数（1×2560 → 4×2560，且 4 个 stream 是独立归一化/独立门控的实体）—— M25 只是提示可能，本 mission 确证；
③ **归一化的位置与形状**（层边界 norm 不是 `[2560]`，是 mixer 里的 `[10240]` 分组 norm，且融合在 combine 里）。

### 8.2 失效假设逐条

| # | 失效假设（现有约定） | 出处 | checkpoint 实际 | 影响 |
|---|---|---|---|---|
| 1 | 层界残差流 = 单路 `[m,2560]` bf16 | `R:m13_moe_layer/README.md:11,23-24`；`R:m13_resources.h:42`（`HIDDEN=2560`）；`R:m14_resources.h:47`；`wt-25/m15_layer_loop/README.md:8,60` | **`[m,10240]` = 4×2560**，4 组各自归一化（`V-N:model.py:265,506`；`HDR:attn_hyper_connection.hc_norm.weight=[10240]`） | 所有以 `HIDDEN` 为**层界宽**的常量/表述；DMA Block1 长度；buffer 数量 |
| 2 | 层边界 op = "Add+RMSNorm(gamma[2560])" | `R:m6_rmsnorm/README.md:1,29-35`；`R:docs/12-layer-integration.md:16` | mixer 的 `combine_norm`：`[10240]` 状态 + `[2560]` block output + `[4]` injection → `[10240]`，且 RMSNorm 是**分组（4×2560）Gemma 风格 `(1+w)`** | m6 **不再是层边界 op**（其 RMSNorm 归约数学仍可复用为 mixer 内的一个子步骤） |
| 3 | 层内 gamma 是 host 合成的 `[2560]` | `R:docs/12-layer-integration.md:54`"小改 D"；`R:m13_moe_layer/README.md:204-205`（数据集缺口）；M25 finding:22 | checkpoint **不存在**任何 `[2560]` 层界 gamma；只有 `hc_norm[10240]` | m13/m14 的 golden 数据集需重建（gamma1/gamma2 换成 hc_norm） |
| 4 | 零残差占位 `UB_ZEROS_B16 bf16[HIDDEN]` | `R:m13_resources.h:118` | 层界输入是**真实的 4 流残差**（入口才由 embed 复制 4 份，`V-N:model.py:506`）；只有"首层 attn_hc 无 pending combine"这一种零值是真实的 | S1 的 zero-residual 语义只在特定位置成立 |
| 5 | 层间**只**传 1 个张量 | `R:docs/12-layer-integration.md:28,30`；`wt-25/README.md:8` | 必须传 **3 个**：`[T,10240]` 状态 + `[T,2560]` pending block output + `[T,4]` injection（`V-N:model.py:276-331`） | 层 kernel 的 I/O 签名、双缓冲数量、cross-layer 依赖 DAG 全变 |
| 6 | 子层输出可以直接加回残差 | `R:docs/12-layer-integration.md:30` | 延迟 combine：层 L 的 mlp 输出要等到 L+1 的 mixer 才加回，且融合进 L+1 的输入 RMSNorm（`V-N:model.py:327-331` + `V-N:ops/hc.py:348-392`） | 段边界/同步图必须支持"跨层携带 pending 值"；层 kernel 不能自闭合 |
| 7 | 残差加是**逐元素**无门的 `h + f(") ` | `R:m6_rmsnorm/README.md:6-9` | 4 个 **stream 级标量门**：`2*sigmoid(inj[s]/4) ∈ (0,2)`，每个 stream 一个不同权重（`V-N:ops/hc.py:218-228`） | 需要新 kernel（`hc_combine`/`hc_combine_norm`）；也是 vLLM 把 combine 与 RMSNorm 融合的原因 |
| 8 | 子层输入 = norm(h) | `R:docs/12-layer-integration.md:16,28` | 子层输入 = **4 个 stream 的门控加权均值**（`Σ_s sigmoid(gate_s)·xn_s / 4`，`V-N:ops/hc.py:151-156`） | 子层前需要一个 10240→2560 的归约（新 op），不是简单 copy |
| 9 | fp32 残差旁路（resOut fp32 ↔ 下一实例 bf16）是层界的核心矛盾 | `R:docs/12-layer-integration.md:30,54`（"m6 实例间矛盾/小改 D"） | HC 的残差是 bf16 的 `[10240]`，且**融合核在 combine→RMSNorm 之间插了一次 bf16 舍入**（`V-N:ops/hc.py:325-327`） | 小改 D 的动机在 HC 路径上消失；取而代之的精度契约是"combine 后先舍回 bf16 再归一化" |
| 10 | 层类型三选一：MoE / GDN / attention | `R:docs/12-layer-integration.md:8-10` | 每层**都**是 MoE（48/48，HDR 实测）；QSA 的 12 个 `full_attention` 实为稀疏检索；**layer 1 额外多一个 PLE 前置段** | 层派发表要加 PLE 特例（0-based 1）；attention 不能按 dense 规划 |
| 11 | attention 层的 `self_attn` 是稠密 GQA | `R:docs/11-attn-analysis.md:5,36-42` | 12 层是 QSA：额外 3.125 MiB/层 indexer + 2052 宽索引缓冲 + 双侧 paged cache（§7.3） | QSA 段的资源/带宽预算需单列（本表 §7.3 已给） |
| 12 | 权重预算 = 层内 op 权重之和 | `R:docs/10-gdn-analysis.md:14` | 每层还要 +HC 25.20 MiB（+3.9%）+ MoE shared/gate 5.0 MiB；且 48 层全 MoE | 见 §5.3 的完整表；docs/10:14 的"122MB/层"只是 GDN attention 段 |

### 8.3 仍然有效的判据（不受 HC 影响）

| 模块 | 仍然有效 |
|---|---|
| m13 MoE | S2 router（GEMV + softmax(max-shift) + Sort32 top-k + renorm，`asc:8-9`）、S3 计数排序索引胶水、S4 permute、S5/S7 MXFP4 VEC 量化（SwiGLU + e8m0）、S6/S8 分组 MXFP4 GEMM、S9 unpermute/combine —— 它们只依赖"喂进来的 `[m,2560]` 向量"，与层界语义无关；682 条 kernel 判据 + 118 条 numpy 校验锚在 device 字节上而非层界语义（`R:m13_moe_layer/README.md:204-205`，M25 finding:22）。**注意**：expert GEMM 的 K=2560 **不变**（`R:m13_resources.h:237,241,249` 的 `SZ_AQ/SZ_GU/SZ_Y`），只有**边界**行要改 |
| m14 GDN | S2 `in_proj`（K=2560, N=16480）、S6 `out_proj`（K=6144, N=2560）形状与 checkpoint 一致（`R:docs/10-gdn-analysis.md:7`）；S3 m9 prolog（conv1d+l2norm+gating）、S4 m4 递推（in-place ssm_state）、S5 m12 RMSNormGated（**逐 head 128 维**，与层界宽度无关）；114 条 kernel 判据 + 103 条 numpy 校验 |
| m12 / m5 / m9 / m4 | 与层界无关：m12 的 norm 维是 128、宽度 6144（`R:m12_rmsnorm_gated/README.md:8-10,46-49`）；m5 是 `1280→640→320/20` 的 MoE 内部（`R:m5_swiglu_quant/README.md:20-25`）；m9/m4 只有 op 内双缓冲 |
| m6 内部 | RMSNorm 归约（二分折叠求和 + NR rsqrt）与 `(x·rstd)·gamma` 数学可复用为 mixer 内的子步骤；只有 `[m,2560]` 的 IO 契约失效 |
| 全局设计规则 | `R:docs/05-megakernel-design.md:90`（单一编译期资源表的"模块只声明 footprint"原则）、CrossCore flagId/pipe 规则、同 pipe PipeBarrier 规则、mode-0/2 语义、48 层 host 循环骨架本身（M25 finding:28 已预判） |
| m15 结构 | host 48 次启动、3:1 层派发骨架、`LayerSlot()` 权重/状态槽位、层间零 CrossCore。**要改的是残差流平面与 kernel 出口语义**，不是循环骨架 |

---

## 9. 对 mega kernel 的修订建议

### 9.1 docs/12 §4 全局资源表需要改的项（逐行）

| docs/12 行 | 现值 | 建议改成 |
|---|---|---|
| L16（算件表 m6 行） | `m6 rmsnorm \| x/res/gamma bf16 → y bf16 + resOut fp32` | 新增两行：**`hc_combine_norm`**（`[m,10240]` state + `[m,2560]` block + `[m,4]` inj → `[m,10240]` 物化态 + `[m,10240]` 归一化态；bf16，`(1+w)` 分组 4×2560）与 **`hc_gate_mix`**（`[m,10240]` xn × `[m,10240]` gate → `[m,2560]`）。m6 降级为"分组 RMSNorm 的归约核可复用" |
| L28（GM 平面 MoE） | `x_res[层输入] → m6#1 → x_norm → ... → m6#2 → y + resOut(fp32)` | 改为：`state[10240] + pending block[2560] + inj[4]` → `hc_combine_norm` → `state'[10240]` + `xn[10240]` → `merged down+inject GEMM(10240→324)` → `silu(/4)` → `up GEMM(320→10240)` → `hc_gate_mix` → `block_input[2560]` → 子层 → 新的 `pending block/mlp_out + inj` |
| L30（GDN 层链） | `x/res → m6 → ... → out_proj → m6 残差。**m6 实例间矛盾**：resOut fp32 vs bf16` | 重写整条：`(state[10240], pending[2560], inj[4]) → attn_hc.combine_and_mix → block_input[2560] → in_proj → m9 → m4 → m12 → out_proj → mlp_hc.combine_and_mix → block_input2[2560] → MoE → (state[10240], mlp_out[2560], inj2[4])`。删掉"m6 实例间矛盾/小改 D"，换成新精度契约"**combine 结果先舍回 bf16 再 RMSNorm**" |
| L34（AIC BufferID） | `0-1=A(+scale) L1 ping/pong；2-3=B；4-5=L0A/L0B；6=L0C；7-10=第二组 GEMM ping/pong（跨段权重预取）；11-13=FIXP→UB；14-27=扩展` | 保留分配，但**第二组 GEMM ping/pong 的用途要写明包含 2 个 mixer GEMM**（每层 4 个 GEMM：down/inject-merged、up，各 ×2 mixer）。**L1 预取滚动窗的流量预算每层 +26.9 MB**（GDN 层从 115.9 → 142.8 MB，+23%） |
| L36（AIV BufferID） | `0-7=通用行 ping-pong 4 对…；19-23=预取 staging + gamma fp32 预转 + rstd scratch；24-27=保留` | "通用行 ping-pong 窗"的**单行宽度从 2560 变 10240**（4×）。`19-23` 的 gamma 预转槽要覆盖 3 类新 gamma（`attn_hc_norm[10240]`、`mlp_hc_norm[10240]`、最终 mixer `[10240]`），建议每层只预转当前需要的 1 个（共 20 KB，bf16 直读即可） |
| L38（flagId） | `0-3=MoE 段边界；4-7=GDN/其它段边界；8-11=段内 mode-2 流式对；12-15=保留` | 同步点数从"每层 ~8"涨到"Moe 8 + GDN 7 + HC 10"≈ **25/层**。**12-15 保留组改为 HC 段专用**；旋转槽复核："每 id 每层 ≈2 次 ≪ 15 上限"在新计数下变为 ≈6-7 次，仍合规但要重算 |
| L40（UB/AIV 预算） | `PERSIST @0 ~12KB；SEG-VEC 窗 @16384 ~112KB；SEG-GDN @16384 ~138KB；余 ~40KB 机动` | **必须加一个 SEG-HC 窗**。裸算 `[10240]` bf16 的 combine_norm 需 res+out+y+w ≈ **85 KB**；但**因为 `hc_norm` 的分组边界（4×2560）恰好等于 combine 的 stream 结构，可以按 stream 窜流处理**，每 stream 5 张 `[2560]`（res/block/out/y/w）+ 双缓冲 ≈ **50 KB**。强烈建议按 stream 切（同时把 L0C 的 `[m,10240]` fp32 压力消掉）。另外 `[10240]` bf16 scratch（gate 与 xn）双缓冲 = **40 KB** |
| L42（L1/AIC） | `A 区 @0 ×2、B 区 @262144 ×2；@~330KB 起 ~180KB=权重预取滚动窗` | 新增两个 GEMM 的 tile 规划：**down+inject**：A `[m,10240]`（m=1 时 20 KB/行），B `[324,10240]` = 6.34 MB（不补 pad；vLLM 补到 336=6.88 MB）→ B 必须分 K/N tile 流送（如 N=168 × K=512 bf16 = 172 KB）；**up**：A `[m,320]`，B `[10240,320]` = 6.55 MB → K 只有 320，按 N（输出 10240 维）tile；L0C 对 up 的输出 `[m,10240]` fp32 在 m=16 时 = 640 KB **超 L0C ⇒ 必须 N-tile** |
| L54（小改清单 D） | `m6 残差输入 fp32 支持` | 在 HC 路径上**取消**（残差是 bf16 `[10240]`）；改为"D′：实现 `hc_combine_norm` 的融合（combine + 分组 RMSNorm，含 bf16 舍入点）"与"D″：实现 `hc_gate_mix`（4 流门控均值 + `/4`） |
| L67 | `48 层 host 循环前单 layer kernel 先验收` | 仍有效；但**单 layer kernel 的 I/O 签名改为 3 张量**（§8.2 #5） |

### 9.2 新增权重槽 / UB / L1 预算（可直接抄进资源表）

| 项 | 值 | 说明 |
|---|---|---|
| 层界残差流 | `[m, 10240]` bf16 = **20,480 B/token** | 双缓冲 = 40,960 B/token |
| 层界附带张量 | pending block `[m,2560]` bf16 = 5,120 B + inj `[m,4]` | 双缓冲各一份 |
| 每层 HC 权重 | **25.20 MiB**（= 2 × 12.60 MiB，checkpoint 原始 324 行）；执行时按 vLLM 补 pad 为 25.66 MiB | 4 张量 × 2 mixer = 8 个权重张量/层 |
| HC 权重槽总数 | 48 层 × 8 + 最终 mixer 3 = **387** | 形状只有 4 种（`[10240]`/`[320,10240]`/`[10240,320]`/`[4,10240]`）⇒ icache 与描述符可复用 |
| 每层权重合计 | GDN 层 **1415.75 MiB = 1.3826 GiB** / QSA 层 **1403.33 MiB = 1.3695 GiB**（HDR；含 MoE 1280.0 + shared/gate 5.0） | HC 是**净增量**：+25.20 MiB/层（checkpoint 口径）/ +25.66 MiB/层（补 pad 口径），× 48 层 = +1.18/1.20 GiB（§2.2 口径声明） |
| PLE 权重槽（仅 layer 1） | 表 **95.37 GiB**（磁盘/host 驻留，不入 HBM 常驻预算）+ **62.64 MiB** 小张量（进 HBM） | 需 16 行 × 320 B 的 gather 路径 |
| PLE 状态 | `[10240, 9]` bf16 = **180 KiB / 序列**，TP-replicated | 独立于 GDN 的 MambaSpec |
| QSA indexer 权重 | 3.125 MiB/层 × 12 | 进 HBM |
| QSA 压缩 KV | **768 B / token-of-context**（12 层合计，bf16） | 262144 ctx ⇒ 201.3 MB |
| QSA raw ring | **1,120 B / 请求 / 层**（num_spec=0） | 1 物理块/请求终生 |
| QSA 选择缓冲 | **8,208 B / token-of-batch-capacity / 层**（`[T,2052]` int32） | 8192 批 ⇒ 67.2 MiB/层 ⇒ 12 层 806 MiB |
| MTP 预留 | MTP 层权重 **4.856 GiB**（含 HC 8 件 + mixer 3 件，形状同主干；expert 为 **bf16 未量化**：`gate_up_proj [512,1280,2560]` bf16 = 3200 MiB、`down_proj [512,2560,640]` bf16 = 1600 MiB） | 暂不实现但资源表需留位；`mtp.pre_fc_norm_hidden.weight [10240]` 说明 MTP 消费 10240 宽多流态 |

### 9.3 同步图是否需要变化

**需要，但不是推倒重来。**

1. **段边界数量增加**：每层 +10 个 HC 段内边界（2 mixer × 5 op），flagId 旋转槽压力从 ~8/层 涨到 ~25/层。
   保留的 12-15 组改给 HC 用即可（rotate-4 仍够：25/4 ≈ 6-7 次/id ≪ 15）。
2. **跨层依赖变成"链式延迟"**：`pending block_output` 必须跨层界存活到下一层的 mixer。
   这**不破坏**"段边界严格串行"的前提（仍然逐层串行），但层 kernel 的**入口要等上一层的 3 个出口**，
   且 `combine_and_mix` 天然把"上一段的收尾"与"本段的开头"融在同一个核里 ⇒ 可以把两层间的同步点从 2 个降到 1 个
   （这是延迟 combine 的真实收益来源，mega kernel 应当照搬而不是拆开）。
3. **PLE 特例必须显式建模**：层 1 的 `attn_hc` 必须先做独立 `combine`（多 1 个全程 pass + 1 个同步点），
   因为 PLE 要往 10240 态上直接加。建议：把 PLE 段插在"层 0 出口 combine"与"层 1 mixer"之间，单独一组 flagId。
4. **Pipe 类匹配规则**（`.tower/comms/inbox/20260926-tower-all-crosscore-pipe-aic-mte3.md`）：
   HC 段里 AIV 侧 set 挂 `PIPE_MTE3`、AIC 侧只能 `PIPE_S/MTE1/MTE2/FIX/M`，写同步表时逐点核对。
5. **同 pipe 背靠背复用同一 buffer 必须 `PipeBarrier`**（同上广播）：HC 的 `[10240]` scratch 复用点很多，
   这是最容易踩的一条。

### 9.4 实现者最容易写错的三处（本 mission 额外发现，vLLM 里三处除以 HC 的语义各不相同）

```python
silu 分支        : lora_s = (lora / HC); lora_s = lora_s * sigmoid(lora_s)      # V-N:ops/hc.py:100-101  ← 除在 silu 的实参内
gate_mix 分支    : out = (Σ_s sigmoid(gate[s]) * xn[s]) / HC                     # V-N:ops/hc.py:151-156  ← 先 sigmoid 求和再除
combine 分支     : w = 2.0 * sigmoid(inj[s] / HC)                                # V-N:ops/hc.py:226      ← 除在 sigmoid 实参内且带系数 2
```

三者**不可互换**（`silu(x/4) ≠ silu(x)/4`，`2·σ(x/4) ≠ σ(x)`）。参考实现 `V-C:hyperconnection.py:212-216,237-240` 逐字一致。

### 9.5 降本选项（供立项时取舍）

| 选项 | 收益 | 风险 |
|---|---|---|
| HC 权重从 bf16 降到 FP8/MXFP4（1.18 GiB → ~0.6/~0.3 GiB） | 每 token 省 0.65–0.9 GB ⇒ **0.8–1.1 ms @0.816 TB/s**（HC 段降 55–75%） | gate/sigmoid 链对量化误差敏感；checkpoint 未量化，需离线重压 + 对拍 |
| 不补 `_input_mix_padding`（324 行而非 336） | 每 mixer 省 245,760 B ⇒ 每 token 省 23.6 MB ⇒ 29 µs | 无（Ascend 侧本就不需要 16 行对齐） |
| down 与 inject 拆成两条 GEMM（320 行 + 4 行） | 同上省 padding，且 4 行 GEMV 可并入 `hc_combine` 的尾段 | 多 1 个段边界 |
| 把 HC 权重常驻 L2 | **无收益** | 算术强度 ≈1 FLOP/B，每 token 必全读；常驻只省 L2 未命中延迟 |

---

## 10. 未取证 / 存疑（单独成节）

按"影响立项决策的程度"排序。凡标注**阻塞**的，建议在动工前与用户/集成侧裁决。

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| 1 | **PLE 95.37 GiB 表在本环境如何落地** | **阻塞，未取证** | cgroup 内存上限 32 GB（`R:docs/01-environment.md:165`）、`/dev/shm` 16 GB（实测）、HBM 128 GiB —— 都装不下。vLLM 的两条路径（pinned host / device 常驻）都要 95.37 GiB 连续空间。**未做任何磁盘 mmap 随机读的延迟/带宽实测**，也未给出可行方案。`R:docs/01-environment.md:173` 只记录了"只能稀疏行查找 + 磁盘驻留"的结论。**这是全模型能否跑起来的前置问题**，建议单独立项 |
| 2 | **HC 段在 Ascend 上的实测带宽效率** | 存疑 | 本报告的 0.816 TB/s 是用 M25 的 GDN（K=2560、N=16480 的**宽** GEMM）反推的。HC 的 down 是 `N=336`、up 是 `K=320` 的**极瘦** GEMM，tile 利用率低，MTE2 效率**可能明显低于**宽 GEMM ⇒ 实际 HC 段耗时可能**高于** 1.52 ms。**未做实测**。建议先做单个 mixer 的 micro-benchmark（1 个 GDN 层 + 1 对 mixer 对比） |
| 3 | `hc_combine_norm` 的 bf16 舍入点是否必须复刻 | 未取证 | `V-N:ops/hc.py:325-327` 在 combine→RMSNorm 之间插了一次 bf16 舍入；参考 torch 实现（`V-C:hyperconnection.py`）没有显式这一步（依赖 tensor dtype）。若验收判据要求 bit-exact 对拍则必须复刻；否则可省。**未做数值实验** |
| 4 | 跨层携带的 `injection` 的正式 dtype | 存疑 | `V-N:ops/hc.py:248-249` 断言 shape `[N,hc_count]`，值域是 `2σ(x/4)`；`V-N:model.py` 未显式声明其 dtype（由 `MergedColumnParallelLinear` 输出决定，bf16）。工程上按 bf16 即可，严格一致性需实验 |
| 5 | PLE 行查找的实际访存放大与固定延迟 | 未取证 | 表加载后是连续的 `[320001536,160]` 参数（不是分片布局）；我按 32B 扇区/128B cacheline 估 1.15–1.6×（≈5.9–7.2 KiB/token），**未实测**。另外 host→NPU 的链路类型（PCIe / C2C）未取证，决定每次 gather 的固定延迟；vLLM 靠"提前一层 prefetch"掩盖，Ascend 要用什么机制掩盖**未设计** |
| 6 | `indexer_budget=2048` 的官方语义 | 存疑（按代码） | 代码里 `token_topk = indexer_budget = 2048`、`block_topk = token_topk/ratio = 512`（`V-N:indexer_qsa.py:117-118`、`V-N:ops/qsa_indexer.py:446,480`）。**未取证**该字段在模型设计文档里的定义；本报告一律按代码行为（≤2048 个 token 被选中） |
| 7 | QSA 的稀疏收益（省了多少 attention 带宽/KV 读） | 未取证 | 本报告只给了 indexer 自身与两侧 cache 的体积/字节；attention 本体的节省需要序列长度分布假设，属 `R:docs/11-attn-analysis.md` / M17 范围 |
| 8 | PLE 在 **prefill**（T 大）时的代价 | 未取证 | 只算了 m=1。prefill 时是 `16T` 行随机 gather + `[T,12800]` GEMM，代价结构完全不同；另外 `V-C:ngram_embedding.py:190-199` 的 `_shift_apply` 逐 shift gather 也是 O(T) |
| 9 | `ngram_context` 的边界语义 | 转述，未亲自复核 | `V-N:model_state.py:43-93`（`all_token_ids[num_computed-2/-1]`，越界填 EOS）。我通过子代理报告获得，未逐行复核首 token / 请求切换的边界 |
| 10 | vision tower 的处理 | 未分析 | `model.visual.*` 333 张量 / 0.836 GiB + `preprocessor_config.json` + `image_token_id=248056`。mega kernel 若只做文本，需要在层派发外显式排除并说明 |
| 11 | 全 48 层都是 MoE 的判据 | 按实测，未读 config 默认值 | `CKPT:config.json#text_config` **没有** `decoder_sparse_step`；`V-N:model.py:243-245` 的判据是 `absolute_layer_id % config.decoder_sparse_step == 0`。我用 HDR 实测"48/48 层都有 `mlp.experts.*`"下结论，未查 `Qwen3NextConfig` 的默认值 |
| 12 | `hc_per_branch_norm=True` 是代码假设还是模型设计 | 风险低 | checkpoint **没有**该字段；`V-N:model.py:265,435` 硬编码 `True`。形状 `hc_norm.weight=[10240]`（而非 `[2560]`）唯一确定了 `norm_size = HC*H`，但"逐元素 affine + 4 组独立归约"这一组合仍是从 vLLM 代码得到的语义 |
| 13 | MTP 的 PLE/位置编码细节 | 未分析 | 只确认了 MTP 不挂 PLE、消费 `[10240]` 多流态、expert 为 bf16。MTP 的其余 31 个张量与调度细节未展开（mission 范围外） |

---

## 11. vLLM main 与 checkpoint 的出入记录

**结论：核心数学与张量集合逐项一致，无一处会导致加载/语义错误。** 以下 6 条是需要**显式记录**的差异或易踩点。

| # | 项 | vLLM main | checkpoint | 裁决 |
|---|---|---|---|---|
| 1 | PLE 分片数 | 代码推导 `ceil(padded_total/split_ngram_parts) = ceil(320001536/128) = 2,500,012`（`V-N:ngram_embedding.py:200-201,422-425`） | 128 片，每片 `[2500012,160]` | **一致**。但 `split_ngram_parts` **不是** `Qwen4ExpTextConfig` 的声明字段（`V-C:config.py:39-54`），代码默认 **512**（`V-N:ngram_embedding.py:169`）。若该字段丢失，分片行数变 625,003 且加载报 shape mismatch ⇒ **Ascend 侧应把它当必需字段校验** |
| 2 | PLE 哈希/词表 buffer | 由 `seed=1234`（config 无 `seed`）+ `ple_dense_layer_id=0` 确定性生成（`V-N:ngram_embedding.py:101-138,173-199`） | 实测 `layer_multipliers`/`vocab_sizes`/`offsets` 与生成值**逐值相等** | **一致（bit-exact，§6.4 实测）** ⇒ 可编译期固化，不必读这三个 buffer |
| 3 | PLE 投影的权重组织 | NVIDIA 变体把 `key_proj(10240,2560) + value_proj(2560,2560)` **合并**成一条 `kv_proj`（`_EXTRA_WEIGHTS_MAPPER`，`V-N:model.py:153-154`，`MergedColumnParallelLinear(2560,[10240,2560])`，`V-N:ple_layer.py:107-115`） | checkpoint 里是 `ple.key_proj.weight` + `ple.value_proj.weight` **两个独立张量**，无 `ple.kv_proj` | **两种约定都存在**：AMD 变体保留两个独立 `ReplicatedLinear` 且 mapper 无 PLE 条目（`V-A:ple_layer.py:520-533`，`V-A/model.py:144-155`），与 checkpoint 名字 1:1。**Ascend 侧建议直接按 checkpoint 原名**（key/value 分开）以省一次拼接与一致性风险 |
| 4 | HC 的 down/inject 组织 | NVIDIA/AMD 都把两者 `MergedColumnParallelLinear` 合并，并补 **12 行 padding** 到 336（`V-N:hyperconnection.py:95-107`） | **分开**：`input_mix_weight_down [320,10240]` + `block_inject_weight [4,10240]` | checkpoint 不必对齐 padding；Ascend 侧用 320+4=324 行（省 245,760 B/mixer，§9.5） |
| 5 | QSA 层的识别方式 | `layer_type == "qwen_sparse_attention"` **或** `indexer_n_heads is not None`（`V-N:model.py:216-236`）；`V-C:config.py:22-26` 注释说明"老 checkpoint 标为 `full_attention` 并在整个模型上设置 indexer 字段" | `layer_types` 里 **12 层是 `full_attention`**（无 `qwen_sparse_attention`），但设置了全套 indexer 字段 | **一致（走第二条分支）**。⚠ **地雷**：只按 `qwen_sparse_attention` 字符串识别的实现会把这 12 层误判为稠密 GQA。Ascend 层派发必须用"indexer 字段存在"作为判据 |
| 6 | PLE 量化方法选择 | `Qwen4ExpPLEEmbeddingMethod.from_quant_config`（`V-C:ngram_embedding.py:165-206`）对既非 `None`、非 `Fp8Config`、非 ModelOpt 的 quant config 抛 `NotImplementedError` | 顶层 `quantization_config.quant_method = "ascend"`，且 `quantized_tensors` 240 项**全不含 PLE/indexer/HC/attention** | vLLM main **没有** `ascend` quant method ⇒ 本 checkpoint 在 vLLM main 上本就无法直接加载。Ascend 侧需自己实现 MXFP4 quant method，**并让 PLE 表/投影、HC、indexer、self_attn、linear_attn 全部走 bf16 直通**（只有 240 个 MoE 张量是 MXFP4） |

补充（非差异，但影响移植）：

- `EngramConfig.verify_model_config` 要求 `current_platform.is_cuda_alike()`（`W:config/engram.py:77-87`），
  且 `EngramConfig` 由 `W:config/vllm.py:1394-1399` 自动创建 ⇒ **Ascend 上引用 EngramConfig 会直接抛异常**，
  PLE 的 host 驻留机制必须另起一套（§6.3）。
- `engram_config` 的 `dp_shared_memory`/`use_thp`/`embedding_across_dp` 是 95 GiB 表的三种省钱手段
  （`W:config/engram.py:41-56`）：`embedding_across_dp` 可把表按 TP×DP 分片，`dp_shared_memory` 让同机多 DP 副本共享一份，
  `use_thp` 用大页减少 TLB 压力。**这些是 Ascend 侧设计 host 驻留方案时值得照抄的三个思路**（未取证本机是否支持）。
- AMD 变体的 PLE 残差加在**调用方**（`hidden = hidden + ple(...)`，`V-A:model.py:298-303`），NVIDIA 融在卷积核里
  （`V-N:ple_layer.py:428`）。两者数学等价但**舍入点不同**；选一条并固定。

---

## 附：一页速查（给实现者）

```
hc_count=4  hidden_size=2560  hc_lowrank=320  eps=1e-6
层界残差流    [T,10240] bf16（HC 外层/HS 内层），双缓冲 40,960 B/token
层界附带      pending block [T,2560] bf16 + injection [T,4]
每层 mixer    2 个（attn_hyper_connection, mlp_hyper_connection）+ 全局 hyper_connection_mixer(1, use_combine=False)
mixer 5 op    combine_norm → [down(→320)|inject(→4)] → silu(/4) → up(320→10240) → gate_mix(/4)
三处 /HC      silu 实参内、gate_mix 求和后、2·σ(inj/4)
归一化        hc_norm [10240]，4 组 ×2560 独立 Gemma-RMSNorm (1+w)
MoE           48/48 层；top-10 + 1 shared；expert MXFP4 U8，另 240 张量量化
GDN           36 层 (0,1,2,4,...)；QSA 12 层 (3,7,...,47，看 indexer 字段而非 layer_type)
PLE           0-based layer 1；128 片 × [2500012,160] bf16 = 95.37 GiB（host/磁盘，非 HBM）；
              16 行 gather/token、kv_proj 2560→(10240+2560)、conv1d [10240,1,4] dilation3、state [10240,9]/序列
QSA indexer   13 实例 × 3 张量；index_qk_proj [640,2560]；token_topk=2048、block_topk=512；
              打分 Σ_h ReLU(q_h·k_pooled)（无缩放无 W）；选择缓冲 [T,2052] int32
每 token 权重流 8.270 GB（GDN 50.9% / MoE 18.3% / HC 15.7% / QSA 15.1% / PLE 0.8%）
等效带宽 0.816 TB/s（M25 反推，误差 0.6%）⇒ ≈10.1 ms/token（99 tok/s），HC 占 1.60 ms
```
