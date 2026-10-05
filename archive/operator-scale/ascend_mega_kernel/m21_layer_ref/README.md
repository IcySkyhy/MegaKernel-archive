# M21 — 官方 Qwen4Exp 单层 CPU 参考通路（M39）

> 目标：为 **L2（层/整网级）验收** 提供一条**可执行**的参考通路——用**真实 checkpoint 权重**在 CPU 上跑**一层**官方
> qwen4_exp decoder layer，dump 每一段的参考激活，并给出一键对拍脚本。
> 语义权威 = 本机官方 vLLM 源码 `/workspace/vllm/vllm/models/qwen4_exp/`（`docs/17-verification-standard.md` §5）。
> 本 mission **不跑 NPU**、不做 kernel、不实现 PLE。
>
> **行号基准**：本文与 `ref/*.py` 里的所有 `文件:行号` 都指 **`/workspace/vllm` main `8a2364605c`**
> （round-1 review 的基准 commit）。上游一旦前进，行号需要重新核对。`selfcheck.py` 的 I 段
> （`tools/check_anchors.py`）会检查每条锚点能解析到真实文件且行号不越界，**但不能**判断该行内容是否
> 仍是注释所说的语义 —— 后者只能人读。

## 0. 结论速览（TL;DR）

| # | 问题 | 结论 |
|---|---|---|
| 1 | 能否直接 `import` 官方层实现（只注入 `sys.path`，不装 vllm 包）？ | **不能**。`import vllm.*` 会执行 `vllm/__init__.py`，从 `nvidia/model.py` 出发的静态闭包触及 **1045 个 vllm 模块 / 121 个第三方 distribution**，baseline venv 里缺 **84 个**（第一个就死在 `regex`，`vllm/utils/platform_utils.py:10`）。**并且 `/workspace/vllm` 从未编译过**（`vllm/*.so` 数量 = 0），所以即使把 84 个 python 依赖全装上，`vllm._C` / `_custom_ops` 仍然 import 不了。逐条清单见 `evidence/import_probe.txt`，复现命令 `/workspace/venvs/baseline/bin/python3 -m ref.official`。 |
| 2 | 有没有能直接跑的官方实现？ | **有一小块**：`vllm/models/qwen4_exp/common/hyperconnection.py` 只 import torch（:32-36），用**文件路径加载**可以绕过 `vllm/__init__.py` 执行成功，作为 HC mixer 的**官方程式化裁判（oracle）**。 |
| 3 | 那参考通路是什么？ | 按官方源码**逐句复刻**的 torch 参考，**每条语句旁标注 `文件:行号`**；优先级：官方 `nvidia/ops/*.py` 的 Triton 内核体内数学（生产路径）> 官方测试里的纯 torch 参考（`tests/models/qwen4_exp/test_hc_ops.py`、`test_qsa_reference.py`）> `common/*.py` 的 torch 参考实现。 |
| 4 | 一层跑得动吗？ | 跑得动。单层权重 1.37~1.38 GiB；**峰值进程内存 ≈5.4 GiB**（`cgroup` 上限 32 GiB；量法见 §13 的 `evidence/peak_rss.txt`，读每个子进程的 `maxRSS`）。**耗时只当量级**（host 墙钟 + 共享机器，**单值不构成证据**）：单层 m=1/m=64 都是秒级（`M39_THREADS=8`）；**区间与采集时的负载见 §13 的 `evidence/rebuild_timing*.txt`**。 |
| 5 | dump 了什么？ | 每段激活 + 层界 3 张量：`attn_hc.*` / `mlp_hc.*` / GDN 段 / QSA 段 / MoE 段 / 层界输入输出。每 tag 29~32 个 tensor，6 个 tag 共 **183** 个；逐段清单与计数口径见 §5。 |
| 6 | QSA 段有可执行参考吗？ | **下游 4 步有第二实现（不是"有 oracle"）**：`ref/qsa_official_ref.py` 逐字复制官方 `test_qsa_reference.py:93-234` 的 5 个纯 torch 参考函数，`selfcheck.py` **K0–K5** 在真实权重上跑通（**K5** 用 `_qsa_select_paged_reference` 端到端比 `visible_blocks→打分→top-k`；K1 logits 相对误差 ≤9e-8、K2/K3/K5 集合逐行一致、K4 ≤2 bf16 ulp）。**输入全部取自 `ref/qsa.py` 自己的中间量 ⇒ 不是独立数据 oracle，前端（投影/norm/RoPE/池化/o_proj）仍只有行号核对**，见 §10 Ver-K。 |
| 7 | 与 docs/14 对账？ | 逐项一致；**docs/14 §10 第 3 条（`hc_combine_norm` 的 bf16 舍入点是否必须复刻）已用实测回答：必须**（B4 实测 **14029/40960 = 34.25%** 物化残差元素受影响，见 `evidence/selfcheck.log` 里的 B4 行；分母 40960 = m=4 的 4×10240，与 B 段的 `args.m` 一致）。另有 4 条补充/更正见 §8。 |

## 1. 为什么整网 CPU 参考不可行（限定边界）

* 整模型 **169.72 GiB**，其中 PLE ngram 表 **95.37 GiB**（`docs/14` §2.1/§2.2）。容器 cgroup 内存上限 **32 GiB**，
  `/dev/shm` 16 GiB ⇒ **表装不下**，且本 mission 明确 **PLE 段不实现**（用户已裁）。
* 48 层权重 **66.4 GiB** 也超过 cgroup 上限 ⇒ 只能逐层流式。
* 因此本参考**只做一层**，且**不做跨层数值连续**：层界输入由 §4 的输入模式生成，不是"真跑出来的前一层输出"
  （`--input chain:N` 除外，见 §4）。
* 本 harness **不产生 logits**，也不声称能替代 `docs/17` §1 的 L2 判据（logits 相对误差 / argmax 一致率）；
  它提供的是 L2 的**输入侧参考**：官方语义下每一条段级激活的权威值。

## 2. 目录与角色

| 路径 | 角色 | 上游来源 |
|---|---|---|
| `ref/ckpt.py` | 逐层权重装载（safetensors header + offset，只读需要的 tensor） | 本 mission 新写；读取手法复用 `tools/weights/safetensors_reader.py`（M8） |
| `tools/safetensors_reader.py` | **逐字复制的上游 reader**（sha256 pin 在 `selfcheck.py`，见 §9） | M8 |
| `tools/torch_reader.py` | M39 的 torch 适配层（`TorchShardReader` 子类，零改动上游文件） | — |
| `tools/check_anchors.py` | 锚点检查：**归属唯一** + 行号不越界 + 打印所在符号；自带负向对照（`--selftest`） | — |
| `tools/time_rebuild.sh` | 重建耗时的**区间**测量（host 墙钟 + 共享机器，单值不算证据） | — |
| `tools/measure_peak_rss.py` | 每个 tag 的子进程**峰值内存**测量（`ru_maxrss`），归档 `evidence/peak_rss.txt` | — |
| `tools/zero_unscored.py` | 造「未打分列填 0」的合成 dump，演示 `--mask-policy` | — |
| `ref/mxfp4.py` | E2M1 + E8M0 group32 反量化 | checkpoint `README.quant.md` + `tools/golden/moe_block_ref.py`（M26） |
| `ref/hc.py` | hyper-connection mixer（5 个 op + `GatedResidual`） | `qwen4_exp/nvidia/ops/hc.py`、`nvidia/hyperconnection.py` |
| `ref/gdn.py` | GDN（`linear_attention`）段 | `mamba/gdn/qwen_gdn_linear_attn.py`、`third_party/flash_linear_attention/ops/fused_recurrent.py` |
| `ref/qsa.py` | QSA（`full_attention`）段 + indexer | `qwen4_exp/nvidia/qsa.py`、`indexer_qsa.py`、`ops/qsa*.py`、`common/qsa_cache.py` |
| `ref/moe.py` | MoE 段（router + routed experts + shared expert） | `models/qwen3_next.py`、`fused_moe/*`、`qwen2_moe.py` |
| `ref/layer.py` | 层骨架（延迟 combine / 3 张量层界） | `qwen4_exp/nvidia/model.py:276-331` |
| `ref/official.py` | 直接 import 探测 + 官方 `common/hyperconnection.py` 加载器 | — |
| `ref/qsa_official_ref.py` | **逐字复制**官方 `test_qsa_reference.py:93-234` 的 5 个纯 torch 参考函数（QSA 段的第二实现，见 §10 Ver-K） | 官方测试 |
| `run_reference.py` | 生成 dump 到 `reference/<tag>/` | — |
| `compare_dumps.py` | **一键对拍**，按 `docs/17` 输出分级报告 | — |
| `selfcheck.py` | harness 自身的 51 条判据（A–K 段；其中 I2 是锚点检查器自己的负向对照） | — |
| `evidence/` | sha256、探测输出、日志 | — |

## 3. 权重装载约定（一层 = 29 个 checkpoint tensor）

前缀：`model.language_model.layers.<L>.`（vLLM 侧映射见 `nvidia/model.py:657-658`）。

**融合约定（必须在装载时做，checkpoint 里没有融合后的名字）**

| checkpoint tensor | 装进哪个模块 | 融合规则 + 出处 |
|---|---|---|
| `linear_attn.in_proj_qkv.weight [10240,2560]` + `in_proj_z.weight [6144,2560]` | `in_proj_qkvz` `[16384,2560]` | 顺序 q,k,v,z；`qwen3_5.py:219-226` |
| `linear_attn.in_proj_b.weight [48,2560]` + `in_proj_a.weight [48,2560]` | `in_proj_ba` `[96,2560]` | 顺序 b,a；`qwen3_5.py:223-224` |
| `linear_attn.conv1d.weight [10240,1,4]` | `conv_w` `[10240,4]` | `weight.view(size(0), size(2))`；`qwen_gdn_linear_attn.py:1311-1314` |
| `linear_attn.A_log [48]` BF16 | `A_log` **fp32** | vLLM 强制 fp32（`qwen_gdn_linear_attn.py:473-478`），装载时精确加宽 |
| `self_attn.q_proj [12288,2560]` + `k_proj [512,2560]` + `v_proj [512,2560]` | `qkv_proj` `[13312,2560]` | q,k,v 顺序；`qkv` 再按 `[12288, 512, 512]` 切 |
| `mlp.experts.gate_up_proj` U8 `[512,1280,1280]` + `.weight_scale` U8 `[512,1280,80]` | 每专家 `[1280,2560]` fp32 | 行 `[0,640)`=gate、`[640,1280)`=up；`fused_moe/routed_experts.py:936-941` |
| `mlp.experts.down_proj` U8 `[512,2560,320]` + `.weight_scale` `[512,2560,20]` | 每专家 `[2560,640]` fp32 | 组 32 沿**最后一维**（in_features）；`README.quant.md:5,33-35` |

**MXFP4 反量化**（`ref/mxfp4.py`）：低 nibble = 偶数 in 索引、高 nibble = 奇数（`README.quant.md:7`）；
E2M1 OCP 表 `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`（符号 = bit3）；E8M0 `scale = 2**(byte-127)`。
E2M1 的 8 个幅值 × 2 的幂**都能被 bf16 精确表示**，所以反量化在 bf16/fp32 下都无损。

**内存策略**：MoE routed experts **保持 packed**，按需逐专家反量化（LRU 上限 40 个专家 ≈ 长驻 314 MB）。
一层 gate_up 全量反量化要 6.7 GB，没必要——m=1 只用到 top-10。

## 4. 层界输入约定

`Qwen4ExpDecoderLayer.forward` 的输入是 **3 张量**（`nvidia/model.py:276-286`）：

```
hidden_states     [T, 10240] bf16   4 流残差（HC 外层 / HS 内层）
prev_block_output [T, 2560]  bf16   上一子层待 combine 的输出（延迟 combine）
prev_injection    [T, 4]     bf16   上一子层的 per-stream 门控 logits
```

多流态的入口唯一来源是 `embed_tokens(x).repeat(1, hc_count)`（`nvidia/model.py:506`）。
`run_reference.py --input` 提供三种：

| 模式 | 含义 | 正确性 |
|---|---|---|
| `embed`（layer 0 默认） | `embed_tokens(ids).repeat(1,4)` | **与官方完全一致**（layer 0 上游无 PLE） |
| `synthetic`（layer≥1 默认） | `randn(0,sigma)` 由 `(seed, layer, m)` 决定 | 合成；仅保证量级（embed 激活 ~0.05） |
| `chain:N` | 真跑 layer 0..N-1（**跳过 PLE**）再取输出 | 等于"关掉 PLE 的模型"的 layer N 输入，**不是**真值 |

`--pending none|synth` 选择是否提供 pending 张量：`none` 是 layer 0 的真实状态（此时第一个 mixer 走 `mix()` 分支），
`synth` 额外覆盖 `attn_hc.combine_and_mix()` 分支。
`--warmup N` 先跑 N 个 token 不进 dump（**一次 T=N 的调用**，不是 N 次 m=1），
用来把 GDN 的 `conv_state`/`ssm_state` 和 QSA 的两侧 cache 推到**非退化**状态
（否则首 token 的 `ssm_state ≡ 0`、`visible_blocks ≡ 0`，dump 没有检验价值）。
`layer3_decode_m1` 用 `--warmup 2100` ⇒ dump 位置 2100、`visible_blocks = 525 > block_topk = 512`，
**同时覆盖** block top-k 的截断路径（详见 §11 第 8 条已关闭）。

## 5. tensor 级契约（dump 清单）

全部落盘为 `.npy`（bf16 存为 uint16 位型）+ `manifest.json`（含 dtype/shape/sha256）。

| dump 名 | 形状 | dtype | 来自 |
|---|---|---|---|
| `layer.input.hidden` | `[m,10240]` | bf16 | 层界 |
| `layer.input.prev_block_output` / `.prev_injection` | `[m,2560]` / `[m,4]` | bf16 | 层界（pending synth 时才有） |
| `attn_hc.hidden` | `[m,10240]` | bf16 | `hc_combine_norm` 的**物化输出**（第一个返回值） |
| `attn_hc.block_input` | `[m,2560]` | bf16 | `hc_gate_mix` 输出 = attn 段输入 |
| `attn_hc.injection` | `[m,4]` | bf16 | merged down+inject 的 inj 分片 |
| `attn.out` | `[m,2560]` | bf16 | GDN 或 QSA 段输出 = 下一 mixer 的 block_output |
| `gdn.mixed_qkvz` / `gdn.ba` | `[m,16384]` / `[m,96]` | bf16 | `in_proj_*` 输出 |
| `gdn.conv_out` | `[m,10240]` | bf16 | causal conv1d(4) + silu |
| `gdn.q` `gdn.k` `gdn.v` | `[m,16,128]`×2 / `[m,48,128]` | bf16 | conv 切分后（q/k 已 l2norm，q 已乘 1/√128） |
| `gdn.g` `gdn.beta` | `[m,48]` | fp32 | `-exp(A_log)·softplus(a+dt_bias)` / `sigmoid(b)` |
| `gdn.core_out` | `[m,48,128]` | bf16 | 递推输出（**RMSNormGated 之前**） |
| `gdn.normed` | `[m,48,128]` | fp32 | RMSNormGated 输出（sigmooid gate） |
| `layer.state.conv_state` | `[10240,3]` | bf16 | mamba conv state（**dump 后**） |
| `layer.state.ssm_state` | `[48,128,128]` | fp32 | `h[v,k]`（**dump 后**，in-place 更新） |
| `qsa.q` `qsa.k` `qsa.v` `qsa.gate` | `[m,24,256]` / `[m,2,256]` / `[m,2,256]` / `[m,24,256]` | bf16 | q/k 已 GemmaRMSNorm+partial RoPE；**gate 未归一化** |
| `qsa.index_q` | `[m,4,128]` | bf16 | indexer q（norm+RoPE 后） |
| `qsa.index_k` | `[m,128]` | bf16 | indexer 原始 k（池化输入） |
| `qsa.compressed_key` | `[m//4,128]` | bf16 | 压缩 cache 的完整组（4:1 均值池化→norm→RoPE@组首） |
| `qsa.index_logits` | `[m, seq_len//4]`（`layer3_decode_m1` 为 `[1,525]`） | fp32 | `Σ_h relu(k·q_h)`，**无 1/√d、无 per-head W**；覆盖**全部可见压缩块**（不是只 512 个），未参与的列填 `-inf` |
| `qsa.block_indices` | `[m,512]` | int32 | block top-k（未选中填 -1） |
| `qsa.token_indices` | `[m,2052]` | int32 | 展开+因果尾；**末列 2051 = 有效计数** |
| `qsa.attn_out` | `[m,24,256]` | bf16 | 稀疏 attention 输出（已 bf16 舍入后乘 sigmoid(gate)） |
| `mlp_hc.*` | 同 `attn_hc.*` | bf16 | 第二个 mixer |
| `moe.block_input` | `[m,2560]` | bf16 | MoE 输入 |
| `moe.router_logits` | `[m,512]` | bf16 | `GateLinear` 输出（本配置 `out_dtype=None` → bf16） |
| `moe.topk_ids` / `moe.topk_weights` | `[m,10]` | int32 / **fp32** | top-10 + renorm（`norm_topk_prob` 缺省 True） |
| `moe.routed_out` | `[m,2560]` | bf16 | `Σ w_k·Expert_k(x)` |
| `moe.shared_out` | `[m,2560]` | bf16 | `sigmoid(W_sg·x)·SharedMLP(x)` |
| `moe.out` | `[m,2560]` | bf16 | `routed + shared`（`moe_runner.py:785-788`） |
| `layer.out.hidden` / `.block_output` / `.injection` | `[m,10240]` / `[m,2560]` / `[m,4]` | bf16 | 交给下一层的 3 张量 |

**表的计数口径**（避免三种数法混用）：上表是**清单行数 30 行**（`mlp_hc.*` 一行代表 3 个 tensor、
`layer.out.*` 一行代表 3 个）。实际 tensor 数按 `manifest.json` 数：

| tag | 总 tensor | `attn_hc` | `mlp_hc` | `gdn` | `qsa` | `moe` | `layer.*` | `attn` |
|---|---|---|---|---|---|---|---|---|
| `layer0_*` | 30 / 32 | 3 | 3 | 10 | — | 7 | 6 (+2 pending) | 1 |
| `layer3_*` | 29~31 | 3 | 3 | — | 11 | 7 | 6 (+2 pending) | 1 |

（`layer.*` 含 `layer.input.hidden`、`layer.out.{hidden,block_output,injection}`、
`layer.state.{conv_state,ssm_state}`；`--pending synth` 的 tag 多 2 个 `layer.input.prev_*`。
6 个 tag 合计 **183** 个 tensor。）

### ⚠️ 官方稀疏输出使用禁令（`docs/17` §7）

`qsa.attn_out` / `attn.out` / `layer.out.*`（凡经过 QSA 层的）都是 **「官方稀疏输出」**：
QSA 的 `token_topk = int(config.indexer_budget) = 2048`（`nvidia/indexer_qsa.py:117`，配合
`nvidia/ops/qsa_indexer.py:472-499` 的 block top-k）意味着每行 query **只 attend 至多 2048 个 token**，
context 更长时**必然丢弃一部分历史**。
因此：

* **不要**拿这些 dump 当「稠密 attention 参考」用。它们是与官方**稀疏路径**对齐的激活；
  一个正确实现稠密 attention 的 kernel 在长 context 下与它们**必然不同**，且差异随 context 增长。
* 做 QSA 段的对拍时，必须**同一条稀疏选择**（同一个 `qsa.token_indices`）后再比 attention 输出；
  跨选择比 attention 输出没有意义。
* `qsa.index_logits` / `qsa.block_indices` / `qsa.token_indices` 描述的是**选择过程本身**，
  它们是 QSA 段最靠前的可对齐产物，也是唯一能定位「选错了」的判据。

## 6. 一键对拍用法

```bash
PY=/workspace/venvs/baseline/bin/python3
cd m21_layer_ref

# 1) 生成参考（一次性；见 tools/make_reference.sh）
bash tools/make_reference.sh

# 2) 我们的 kernel dump 目录（含 manifest.json） vs 参考
$PY compare_dumps.py --ref reference/layer0_decode_m1 --test /path/to/our_dump

# 3) 或直接喂设备导出的裸 bin
$PY compare_dumps.py --ref reference/layer0_decode_m1 \
    --test-bin attn_hc.block_input=/tmp/hc_out.bin:bf16:1,2560 \
    --test-bin moe.topk_ids=/tmp/ids.bin:int32:1,10

# 4) L1 档（容差按 MXFP4 量化预算，而不是 L0 的位级/≤1ulp）
$PY compare_dumps.py --ref reference/layer0_chunk_m64 --test ... --profile mx-budget

# 5) 非空洞性：同参数换输入种子，证明 dump 会动
$PY compare_dumps.py --ref reference/layer3_chunk_m64 \
    --ref2 reference/layer3_chunk_m64_seed1 --nonhollow-only

# 6) 掩码口径：官方 kernel 只写 column < visible_blocks 的列，其余是未初始化 buffer
#    （`ops/qsa_indexer.py:101-107`）。kernel 在这些列 dump 0 是常见写法，
#    strict 会判 FAIL；用 ignore-unscored 把这些位置从所有判据里剔除（剔除个数单列）。
$PY compare_dumps.py --ref reference/layer3_chunk_m64 --test /path/to/dump \
    --segments qsa.index_logits --mask-policy ignore-unscored

# 7) 把 markdown 报告落盘（--md-out 写完整报告；--json-out 写结构化结果）
$PY compare_dumps.py --ref reference/layer0_decode_m1 --test /path/to/dump \
    --md-out evidence/our_kernel_report.md --json-out evidence/our_kernel_report.json
```

需要 kernel dump 侧提供 `manifest.json`（`{"segments": {"<name>": {"file","dtype","shape"}}}`），
或者用 `--test-bin NAME=PATH:DTYPE:shape,shape,...` 手工映射；名字不同时用 `--map`。

**给 kernel 侧的最小接入方式**（3 行，无需改 kernel）：在 dump 目录里放一个 `manifest.json`：

```python
import json, numpy as np, pathlib, hashlib
segs = {}
for name, (fname, dtype, shape) in SPEC.items():          # SPEC 由你按 dump 约定写死
    a = np.fromfile(fname, dtype=np.uint16 if dtype == "bfloat16" else np.dtype(dtype))
    segs[name] = {"file": fname, "dtype": dtype, "shape": list(shape),
                  "sha256_stored_bytes": hashlib.sha256(a.tobytes()).hexdigest()}
pathlib.Path("manifest.json").write_text(json.dumps({"segments": segs}))
```

**本仓现有的 kernel dump 名字对照**（供 `--map` / `--test-bin` 使用）：

| 参考段 | m13（MoE 层） | m14（GDN 层） |
|---|---|---|
| `moe.block_input` | `<case>_mode0_x_norm_device.bin` | — |
| `moe.router_logits` | `<case>_mode0_router_logits_device.bin`（bf16） | — |
| `moe.topk_ids` | `<case>_mode0_topk_ids_device.bin`（int32） | — |
| `moe.topk_weights` | `<case>_mode0_topk_weights_device.bin`（bf16，参考为 fp32） | — |
| `gdn.conv_out` / `gdn.core_out` | — | `<tag>_ws.bin` 按 `m14_params.txt` 的 `ws_off` 切片 |
| `layer.state.ssm_state` | — | `<tag>_ssm_out.bin`（fp32） |
| `layer.state.conv_state` | — | `<tag>_cs_out.bin`（bf16） |

> `moe.topk_weights` 的口径差异要注意：**参考是 fp32**（`topk_weights` 缓冲是 fp32，
> `fused_topk_router.py:92-94`），而 m13 kernel 内是 bf16 打包（`w_tk_packed`）。
> 对拍时要么把参考转 bf16，要么在 `--tolerance` 里给这一段单独放宽。

**报告结构（`docs/17` 强制）**

* **§1 判定项**：每段 `max_abs_err ≤ atol + rtol·ref_max_abs` 且 `max_ulp ≤ max_ulp`
  且非有限掩码一致 → PASS/FAIL，最后给总数。
* **§2 报告项**（**不进入 PASS/FAIL 计数**）：位级一致个数/比例、≤1ulp 比例、max 相对误差、显著元素数。
* **§2b argmax / top-k 一致率**（`docs/17` §1 L2 指标②，**报告项**）：对打分张量
  （默认 `moe.router_logits`、`qsa.index_logits`）逐行给 argmax 一致率、top-10 集合完全一致率、
  top-10 元素重合率。`--agreement-rows` 可改。
* 每个误差数字都写明是 **max**（不是首次不匹配）、**绝对/相对**、以及**哪一档输入**（manifest 的 `extra`）。
* **ULP 只在"显著元素"上计算**：`|值| ≥ 2^-8·本张量 max|ref|`。低于此阈值的元素是噪声尺度，
  ulp 距离没有意义（近零处 1 ulp 的绝对差极小但"ulp 数"可以很大），只由绝对预算判。
  被排除的元素个数在报告里单列。这是**必须明说的口径**，否则 ulp 数字不可解释。

行内排序后再比较的段（上游 top-k 顺序未定义）：`qsa.block_indices` / `qsa.token_indices`。
上游 top-k 的输出顺序是**未定义**的（`tests/kernels/test_top_k_per_row.py` 只比排序后的集合），
所以这两段只能当**多重集**比。`compare_dumps.py` 默认对它们逐行排序后再比（`--sort-rows` 可改/清空）。

**退出码（项目规则 2026-09-26；`docs/17` §8.3「审校脚本改用**三态退出码**」，原文：`2` = **没得比 / 输入缺失**（打印 `RESULT: SKIPPED`，绝不发合格证））**

本目录所有审校脚本都是三态，头注释里写明：

| 退出码 | 含义 |
|---|---|
| `0` | **比过且通过**（OK 文案里带实际比较计数，例如 `RESULT: OK (183 tensors hashed)`、`RESULT: OK (522/522 anchors verified…)`） |
| `1` | **比过且有差异/越界**（真判据失败） |
| `2` | **没得比 / 输入缺失**（`RESULT: SKIPPED`，**不是** OK）：`compare_dumps.py` 无共同段、`hash_dumps.py` 无 manifest、`check_anchors.py` 有 UNVERIFIED、`selfcheck.py` 0 条判据、`ref/official.py` 找不到 `/workspace/vllm` |

各脚本的负向对照（喂空输入必须不给合格证）已在开发时跑过，读数见 §10 Ver-I2 / 本节的说明。

**`--mask-policy`（非有限值口径）**

| 取值 | 语义 | 何时用 |
|---|---|---|
| `strict`（默认） | 参考侧非有限（`-inf`）必须对应测试侧非有限；不匹配算 **FAIL** | 测试侧也按「未计算 → 非有限」的约定 dump |
| `ignore-unscored` | 对 `--unscored-segments`（默认 `qsa.index_logits`）**参考侧非有限的位置**从数值、掩码、ulp 三类判据里全部剔除，个数在报告项 `ignored_unscored` 单列 | 测试侧在这些未初始化 buffer 列里 dump 0 或任意值 |

依据：官方 indexer 只写 `column < visible_blocks`（`ops/qsa_indexer.py:101-107`），
其余列是 buffer 里未被写过的内容，**语义上未定义**，不该被当判据。
`tools/zero_unscored.py` 造合成 dump、`evidence/compare_strict_mask.md` 与
`evidence/compare_ignore_unscored.md` 是这条口径的正反例（strict FAIL / ignore-unscored PASS）。

**三档容差（profile）**

| profile | 用途 | 判据 | 依据 |
|---|---|---|---|
| `l0`（默认） | **T1 / T2 数据**：整数域、以及"参考能完整建模每一步舍入"的浮点 | `max_abs_err ≤ atol + rtol·ref_max_abs` 且 `max_ulp ≤ 1`（bf16） | `docs/17` §1.1 T1/T2 |
| `mx-budget` | MoE 段（激活也走 MXFP4）——L1 量化预算口径 | bf16 rtol `0.25` | `docs/17` §1.1：e2m1 3-bit 尾数 half-ulp = `2^-4`，两个操作数 → `2·2^-4 = 1/8`；组尺度上界 `|δ| ≤ max|x|/8` |
| `t3` | **T3 数据**（见下表）：超越函数 / 跨 tile·online 重标定 / mmad 累加 | **逐元素** `\|out−ref\| ≤ rtol·\|ref\| + 0.5·ulp(out)`；`rtol` 默认 `2^-12`（来源见 `compare_dumps.py` 的 `T3_RTOL` 注释：参考侧 `4·2^-24`＋接收侧 `2n·2^-24`，n=2048）；超界数与"其中相消区"数在判定项表里附带；`≤1ulp 比例`/`maxRel` 降为报告项 | `docs/17` §1.1 T3 |
| （任何 profile 下） | `int32`/`int64`/`uint8` 段 | `atol = rtol = 0` **逐位**，任何 profile 都不降档 | `docs/17` §1.1 护栏 1：T1 不许降档 |

### 各段的档位声明（`docs/17` §1.1 要求"用哪一档必须写明理由"）

**为什么几乎全是 T3**：这张卡的 kernel 只要用到 mmad/cube 累加、`Exp`/`Rsqrt` 近似、或 online softmax 重标定，
就命中 §1.1 的触发条件①②③ —— 而这三种在这 48 层的每一段里都存在。**用 l0 的 `max_ulp=1` 去判这些段
会假 FAIL**（F3/K4 实测两侧顺序差异本身就有 2 bf16 ulp）。

| 段 | 档 | 触发的条件 + 理由 |
|---|---|---|
| `qsa.attn_out`、`attn.out`、`layer.out.*`（QSA 层） | **T3** | ① softmax 含 `Exp`；② kernel 是 online softmax + 逐 tile 重标定（`ops/qsa.py:162-179`）；③ PV 的 mmad 累加 |
| `qsa.index_logits` | **T3** | ②③：打分是 `tl.dot(keys, query)` 后按 `columns` tile 累加 + `tl.sum(axis=2)`（`qsa_indexer.py:94-100`）。**注意**：参考侧没有超越函数，`R._qsa_mqa_paged_reference` 也不含 —— 这一档是给**接收方 kernel** 的，不是给参考的 |
| `moe.router_logits` | **T3** | ③ `GateLinear` 是 GEMM（mmad / cube 累加，K=2560） |
| `moe.routed_out`、`moe.shared_out`、`moe.out`、`mlp_hc.block_input` | **T3** + L1 预算 | ③ 两段专家 GEMM；且**我们的 kernel 走 W4A4**，量化误差要用 `--profile mx-budget` 看 |
| `gdn.g`、`gdn.beta` | **T3** | ① `softplus`/`exp`/`sigmoid` 都是超越函数（`fused_recurrent.py:326-329`） |
| `gdn.*` 的其余浮点段（`conv_out`/`q`/`k`/`v`/`core_out`/`normed`）、`attn_hc.*`/`mlp_hc.*` 的浮点段 | **T3** | ③ 都含 GEMM（`in_proj`/`out_proj`/低秩 up·down）；`gdn.q/k` 另含 `Rsqrt`（l2norm）→ ① |
| `qsa.q`/`k`/`v`/`gate`/`index_q`/`compressed_key` | **T3** | ① GemmaRMSNorm 的 `Rsqrt`；③ QK/RoPE 与 indexer 投影的 mmad |
| `qsa.block_indices`、`qsa.token_indices`、`moe.topk_ids` | **T1** | 索引域，`atol=rtol=0` 逐位；top-k 顺序未定义，按**集合**比（`--sort-rows`） |
| `layer.state.conv_state` / `ssm_state` | **T3** | ③ 递推是逐 head 的 128×128 累加，且随 token 演化 |

**接收方 kernel 的作者必须做的**：T3 的 `ε` 里"接收侧"那一半（`n·2^-24` + 自家 `Exp`/`Rsqrt` 的**官方精度规格**）
只有你知道 —— 用 `--tolerance` 传自己的 `rtol`，不要沿用 `2^-12` 这个按 n=2048 估的默认值。
`compare_dumps.py` 的 `T3_RTOL` 注释里写清了哪一半是参考侧的责任、哪一半是接收侧的。

> ⚠️ `mx-budget` 的数字**故意宽松**，它的作用是分辨"坏了"还是"在量化预算内"，**不是**精度背书。
> 真正的量化预算应当由 `m13`/M26 golden 的 PSNR/误差上界给出，本文件只提供口径位置。

## 7. 实际用到的官方源码行号清单

`ref/*.py` 里每条语句旁都有 `path:line`。汇总（行号为 `/workspace/vllm` 下，**main `8a2364605c`**）。
当前共 **522 条锚点 / 88 个不同的锚点目标路径**（解析到 **39 个不同文件**；工具汇总行直接打印这三个数，`--verbose` 可逐条数），由 `tools/check_anchors.py` 机械检查（`selfcheck.py` I 段），
结果 `evidence/anchors.txt`。检查器报三个数：
**verified against the intended file / UNVERIFIED / out of range**（当前 `522 / 0 / 0`）。

> **符号优先（项目规则 2026-09-26；`docs/17` §8.2「docs 里"裸行号"代码引用的清扫（写作规则落地）」）**：下表的行号是**查阅提示**，**符号名才是稳定引用** ——
> 行号会随上游漂移（本轮之前就漂过一次）。`tools/check_anchors.py --verbose` 会为每条锚点打印
> 它落在哪个符号里（如 `hc.py:45 -> _grouped_gemma_rmsnorm_kernel`），
> 用 `python3 tools/check_anchors.py --verbose | grep hc.py` 即可自查。
> 检查器能保证「引用存在 + 归属唯一 + 落在哪个符号里」，**不能**保证「该行内容就是注释所说的语义」——后者需要人读。
> 上一轮 review 正是因为 QSA 段有 15 处锚点越界（8 处超过文件长度）才判 P1；已全部修正，
> 并加了这条自检防止回退。

**QSA 段锚点的修正记录（round-1 review 的 P1）**：以下 7 处曾在**文件末尾之外**，已按下表改正
（语义未变，只有引用错）。下表左列是**故意保留的错误引用**，`tools/check_anchors.py` 用
`anchor-check:off/on` 标记跳过它。

<!-- anchor-check:off -->
| 原（错） | 现（正确） |
|---|---|
| `qsa_indexer.py:686-699` / `:686-692` | `:94-100`（decode 打分）／`:206-208`（prefill） |
| `qsa_indexer.py:772-813` / `:772` / `:809-813` | `:221-276`（计数列 `:268-276`） |
| `qsa_pre_indexer.py:553-589` / `:568-571` / `:573` | `:335-343` / `:266-269` / `:271` |
| `indexer_qsa.py:471-499` | 删除（那是 `ops/qsa_indexer.py:471-499`） |
| `ops/qsa.py:1091-1126` / `:1110-1118` | `:181-214` / `:196-205` |
| `fused_qk_norm_rope.py:278-285` / `:285-286` | `:68-73`（norm）／`:125-126`（旋转） |
| `qsa_indexer.py:537-538` / `:57-58` / `:784-789` | `:536` / `:56-57`（prefill `:163-164`）／`:248-253` |
| `nvidia/qsa.py:331-332` / `:367-381` | `:525`（o_proj）／`:472-498`（indexer→KV 写顺序） |
| `fla.cpp:1667` | `:1670`（CPU l2norm eps=1e-5；另一处 `:1838`） |
<!-- anchor-check:on -->

**HC（`ref/hc.py`）**
`nvidia/ops/hc.py` 内核：`12-52` grouped_gemma_rmsnorm / `55-78` wrapper / `81-105` hc_silu /
`125-160` hc_gate_mix / `188-232` hc_combine / `272-345` hc_combine_norm（**`325-327` 是 bf16 舍入点**）、
`348-392` wrapper；`nvidia/hyperconnection.py:50-196`（`95-107` pad 到 336 行、`128-151` mix、`153-188` combine_and_mix、`190-196` combine）；
`common/hyperconnection.py:55-87`（GroupedGemmaRMSNorm）、`203-222`（mix）、`224-242`（combine）；
`tests/models/qwen4_exp/test_hc_ops.py:34-38,47-52,62-68,76-80,94-106,124-126`。

**GDN（`ref/gdn.py`）**
`mamba/gdn/qwen_gdn_linear_attn.py:396-404,419-427,468-481,483-507,564-612,632-641,803-838,845-857,940-946,1012-1053,1311-1314,1670-1681`；
`mamba/gdn/base.py:25-48`；`mamba/mamba_utils.py:279-300`（state 形状）；
`mamba/ops/cpu/causal_conv1d.py:33-82,118-147`；`layers/layernorm.py:216-314`（RMSNormGated）；
`third_party/flash_linear_attention/ops/fused_recurrent.py:296-341`（decode 内核，**这是递推的规范**）、
`ops/fused_gdn_prefill_post_conv.py:84-91,128-149`（l2norm eps=1e-6、gating）、
`ops/chunk.py:32-77`（chunk 流水线）；`tests/kernels/mamba/cpu/test_cpu_gdn_ops.py:92-187`（纯 torch 递推参考）。

**QSA（`ref/qsa.py`）**
`nvidia/qsa.py:257-351,434-442,444-498,500-526,182-249,525`；
`nvidia/indexer_qsa.py:35-60,117-118,131-137,151-163,323-330,342-372`；
`nvidia/ops/qsa.py:32-233,401-433,594-758,635-671,181-214`；
`nvidia/ops/qsa_indexer.py:19-107,221-276,471-499,536`；
`nvidia/ops/qsa_pre_indexer.py:10-80,32-53,204-343,266-269,271,335-343,346-372`；
`common/qsa_cache.py:253-275,756-870,838-856`；
`model_executor/layers/fused_qk_norm_rope.py:80-152`；`models/qwen3_next.py:387-447`；
`layers/rotary_embedding/base.py:80-102`；`nvidia/model.py:216-236,846-852`；
`tests/models/qwen4_exp/test_qsa_reference.py:93-105,108-125,128-164,167-195,198-234,1077`。

**MoE（`ref/moe.py`）**
`models/qwen3_next.py:131-270`；`qwen4_exp/nvidia/model.py:159-170,240-245`；
`layers/fused_moe/router/gate_linear.py:18,192-249`；`layers/fused_moe/router/fused_topk_router.py:80-124`；
`csrc/libtorch_stable/moe/topk_softmax_kernels.cu:113-134,497-543,581-592,845-847`；
`layers/fused_moe/runner/moe_runner.py:785-788`；`models/qwen2_moe.py:91-123`；`layers/activation.py:116-144`；
`layers/fused_moe/routed_experts.py:936-941,1106-1111`。

**层骨架（`ref/layer.py`）**：`qwen4_exp/nvidia/model.py:276-331`（`287-288` 断言、`308-314` 融合分支、
`316-322` attn 派发、`326-331` 第二 mixer + MoE + 返回）、`437-441,489-590,657-658`。

**checkpoint**：`README.quant.md:4-9,33-35`；`config.json#text_config`。

## 8. 与 docs/14、docs/15 的对账

逐项复核（`selfcheck.py` D/E 段自动检查）：

| docs 结论 | 复核结果 |
|---|---|
| docs/14 §2.3 HC 张量形状 4 件 ×2 mixer + 全局 3 件 | ✅ 逐 shape 一致（D1/D2） |
| docs/14 §3.2/§3.3 三处 `/HC` 语义（silu 实参内、gate_mix 求和后、`2σ(inj/4)`） | ✅ 与 `hc.py:100,156,226` 逐字一致，不可互换 |
| docs/14 §3.2 combine 后**先舍回 bf16 再 RMSNorm** | ✅ `hc.py:325-327`；且**实测可观测**（§10 B4） |
| docs/14 §4.1 延迟 combine、层界 3 张量 | ✅ `model.py:308-314,326-331`；`return hidden, mlp_out, injection` |
| docs/14 §4.1 layer 1 因 PLE 打断融合、多一次独立 combine | ✅ `model.py:290-297`（本 mission 不实现 PLE，只保留该分支的调用点说明） |
| docs/14 §6.1 PLE 在 0-based layer 1、`ple_layer_ids=[2]` 是 1-based | ✅ `config.json` + `model.py:195`（D5） |
| docs/14 §7.2 indexer 打分 `Σ_h relu(q_h·k)`，无 1/√d、无 W | ✅ 内核 `qsa_indexer.py:94-100` 确认；**注意**：官方测试参考 `test_qsa_reference.py:103` **除**了 `sqrt(128)`（见 §10 Ver QSA） |
| docs/14 §7.2 选择缓冲宽 2051（+1 计数列） | ✅ `qsa_indexer.py:268-276`（计数列在 `:273-276`） |
| docs/14 §7.2 `full_attention` 12 层实为 QSA（看 indexer 字段） | ✅ `model.py:216-220`（D4） |
| docs/14 §9.4 三处除法 | ✅ 同上 |
| docs/15 §2.3 (a)~(f) HC 数学 | ✅ 逐式一致；`hc.py:325-327` 的位级要求也被实测证实 |
| docs/10 §1 GDN `q,k=l2norm(eps 1e-6)，q×=1/√128` 与递推顺序 | ✅ `fused_recurrent.py:317-335`；与 m9 的编译期常量一致 |

**4 条补充 / 需要更正的地方（已发 tower）**

1. **docs/14 §10 第 3 条（阻塞级存疑）已被实测关闭**：融合核在 combine→RMSNorm 之间插入的 bf16 舍入
   （`hc.py:327`）在 layer 0 的真实权重上让 **34.25%（14029/40960）** 的物化残差元素发生改变 ⇒
   位级 L0 对拍 **必须复用这一步**；不复用的话 `layer.out.hidden` 至少差 1 个 bf16 ulp，并经
   下游 10240→320→10240 链放大。docs/15 §2.3 已经把它写成"位级可比性要求"，两处现在一致。
2. **"官方 vLLM qwen4_exp 实现"不是单一实现**：`common/hyperconnection.py`（纯 torch）与
   `nvidia/ops/hc.py`（Triton，生产路径）**不是位级等价**——差别有两处，都能复现：
   (a) 归一化归约顺序 `square().mean(-1)`（`common/hyperconnection.py:83`，`GroupedGemmaRMSNorm.forward`）vs `sum(x*x)/GROUP_DIM`（`hc.py:45`，`_grouped_gemma_rmsnorm_kernel`）⇒ xn 差 **1 个 bf16 ulp**；
   (b) 融合 combine 的早舍入（hc.py:327）。实测 `common.mix()` vs `nvidia` 路径的 block_input
   max 差 **0.5 个 bf16 ulp @ 张量最大幅值**（放大自 (a)）。
   ⇒ `docs/17` §1 L2 说"参考 = 官方 vLLM 的 qwen4_exp 实现"时，**必须钉住变体**；
   位级判据只能挂 `nvidia/`，`common/` 只能当 ≤1ulp 级 oracle。

3. **docs/14 §7.2 步骤 ⑦ 的注释 `# GROUP_SIZE=12` 有误导**：`GROUP_SIZE = q.shape[1]//k_cache.shape[2] = 24/2`
   是 **GQA 组头数**（`ops/qsa.py:670-671`），与输出门控无关；输出门控是逐 (row, head, dim) 逐元素的
   `sigmoid`（`ops/qsa.py:196-205`）。docs/14 正文没错，只是注释挂错了位置。
4. **docs/14 §7.2 没写 QSA 输出门的**来源**：它是 `q_proj` 每头 `[q|gate]` 的**后半**（`qwen3_next.py:427-435`，
   布局见 `fused_qk_norm_rope.py:156`），且在 fused kernel 里**原样拷贝、不做归一**（`fused_qk_norm_rope.py:129-136`）。
   实现者极易顺手给 gate 也加个 norm——那样就错了。

另有一条低优先：GDN 的 `l2norm eps` 在官方两个后端里不一致——CUDA/Triton 硬编码 **1e-6**
（`fused_recurrent.py:318-319`），CPU C++ 端口硬编码 **1e-5**（`csrc/cpu/sgl-kernels/fla.cpp:1670`）。
本参考与 m9 都取 **1e-6**。若将来拿 vLLM 的 CPU 路径做参考会在 `gdn.q/gdn.k` 上看到系统偏差。

## 9. 与上游的逐文件差异表（`docs/17` §3.4）

### 9.1 `tools/safetensors_reader.py` —— **逐字复制**（不是改写）

| 项 | 值 |
|---|---|
| 上游 | `tools/weights/safetensors_reader.py`（本仓 M8，agent-weights，268 行） |
| 本副本 | 268 行，**sha256 与上游完全相同** = `72b93ca2f2727c68cad77d128c3bc9338921b9dd3311256b7ed4c226bc515abd` |
| 机械校验 | `selfcheck.py` J1（本副本 vs 上游 sha256）+ J2（vs `selfcheck.py` 里的 `PINNED_READER_SHA256`）；`evidence/selfcheck.log` |
| 上游变更是否流入 | ✅ 把上游文件重新 `cp` 过来即可；J1 会在上游前进后立刻报 FAIL，提示重抄并更新 pin |

> 上一轮 review 抓到这里的差异表**不实**：旧版本写"只加了 `load_torch`/`load_torch_f32`，其余逐字相同"，
> 实际与上游差 **340 行**，而且**丢掉了上游「分片还在下载时只索引已落地张量」的行为**
> （上游 `SafetensorsFile.complete()` / `tensor_names(complete_only)` / `ShardReader.incomplete`）。
> 现在改成**真·逐字复制**：M39 需要的东西（torch tensor）全部放在 `tools/torch_reader.py` 的
> `TorchShardReader` 子类里，上游文件一字未改，所以差异表只有一行且可机械校验。
> 顺带补上了上游的完整性行为：`ref/ckpt.py:_get_reader` 现在会在**装载时**检查
> `ShardReader.incomplete`，分片没下完就立刻报错（不再是跑到一半才炸）。
> 本机 checkpoint 131 个分片全部完整，实测 `incomplete == {}`。

### 9.2 改写/新写的文件

| 本目录文件 | 上游来源 | 差异 | 上游变更是否自动流入 |
|---|---|---|---|
| `tools/torch_reader.py` | — | **新增**；`TorchShardReader(ShardReader)` 只加 `load_torch` / `load_torch_f32` / `file_size`（bf16 = 位重解释，不引入舍入） | n/a |
| `tools/check_anchors.py` | — | **新增**；扫 `ref/*.py`、`README.md`、`selfcheck.py`、`tools/*.py` 里的 `path:line`，断言文件存在且行号不越界 | n/a |
| `tools/zero_unscored.py` | — | **新增**；造「未打分列填 0」的合成 dump，用于演示 `--mask-policy` | n/a |
| `tools/hash_dumps.py`、`tools/make_*.sh`、`tools/zero_unscored.py` | — | **新增**（构建/证据脚本） | n/a |
| `ref/qsa_official_ref.py` | `tests/models/qwen4_exp/test_qsa_reference.py:93-234` | **逐字复制**（5 个纯 torch 参考函数，一字未改）；sha256 由 `QSA_REFERENCE_BODY_SHA256` pin，`selfcheck.py` K0 每次重新抽取比对 | ✅ 把上游源文件重抽一次并更新 pin（K0 会先报 FAIL） |
| `ref/hc.py` | `qwen4_exp/nvidia/ops/hc.py` | **重写**为 torch（Triton → torch），逐 `tl.*` 语句映射并带行号；合并 down+inject 的 336 行 padding 简化为 324 行（见 §10 Ver-B） | ❌ 否 |
| `ref/gdn.py` | `mamba/gdn/qwen_gdn_linear_attn.py` + `third_party/flash_linear_attention/ops/fused_recurrent.py` + `mamba/ops/cpu/causal_conv1d.py` | **重写**为 torch；chunk 流水线 → 顺序递推；conv 用 `F.conv1d(groups=dim)`（照抄官方 CPU 参考） | ❌ 否 |
| `ref/qsa.py` | `qwen4_exp/nvidia/{qsa,indexer_qsa,ops/qsa,ops/qsa_indexer,ops/qsa_pre_indexer}.py` + `common/qsa_cache.py` | **重写**为 torch（**该段官方没有可执行 eager 路径**）；paged cache → 单序列稠密；Triton 打分/top-k/展开按内核语义复刻 | ❌ 否 |
| `ref/moe.py` | `models/qwen3_next.py` + `fused_moe/*` + `models/qwen2_moe.py` | **重写**为 torch；`FusedMoE` 的 fused kernel → 专家主序显式循环；router softmax/topk/renorm 按 `topk_softmax_kernels.cu` 语义 | ❌ 否 |
| `ref/mxfp4.py` | — | **新增**；E2M1/E8M0 group32 反量化（checkpoint `README.quant.md` + 本仓 M26 golden 交叉确认） | n/a |
| `ref/ckpt.py`、`ref/layer.py`、`ref/official.py`、`run_reference.py`、`compare_dumps.py`、`selfcheck.py` | `qwen4_exp/nvidia/model.py:276-331`（仅 `ref/layer.py` 的层骨架） | **新增**；`ref/layer.py` 逐句照抄层 forward 并**删掉 PLE 分支**（mission 边界） | ❌ 否 |

**声明**：除 `tools/safetensors_reader.py`（逐字复制、sha256 pin）外，以上全部为改写或新写，
上游（vLLM main）的后续修改**不会**流入；同步方式只有人工 diff 并更新 `ref/*.py` 里的锚点。

## 10. Ver X 证明什么 / 不证明什么（`docs/17` §3.5）

* **Ver-A HC op 级**（`selfcheck.py` A1–A7）：证明 `ref/hc.py` 的 5 个 op 与 **vLLM 自己的 torch 参考**
  （`test_hc_ops.py`）在随机输入上**逐位一致**（7 项全部 0 ulp，含 unit-injection 的精确路径）。
  **不证明**：不证明它们等于 Triton 内核的位模式（Triton 归约顺序不同，见 Ver-B）。
* **Ver-B HC oracle**（B1a–B4）：证明 `ref/hc.py` 与**真跑起来的官方 `common/hyperconnection.py`**
  在真实权重上 ≤1 bf16 ulp @ 张量最大幅值；并**给出差异的原因**（B1c：换成 `mean(-1)` 就与官方逐位一致）。
  **不证明**与 `common/` 位级一致——它做不到，因为归约顺序不同。
* **Ver-C MXFP4**（C1/C2）：证明 `ref/mxfp4.py` 的归一化/E8M0/表与 M26 numpy golden **逐值相等**（层 0 真实字节）。
  **不证明**：不证明 nibble 顺序的"物理"正确性（那只能靠 device GEMM A/B；见 §11 已知限制）。
* **Ver-D 契约**（D1–D6）：证明装载的权重形状与 docs/14 §2.3 表逐项一致，且 layer 3 走 QSA 分支。
  **不证明**文档表本身正确（只证明代码与文档一致）。
* **Ver-E 非空洞性**（E1–E6）：证明改 1 个输入元素 → 24/24 段 dump 全变；层界状态确实演化
  （`ssm_state` 跨 step 变化 7.3e-2）；`conv_state` 只由**最近 3 列**决定而 `ssm_state` 依赖**全部历史**
  （E5b：只扰动第 0 列 → conv 逐位不变、ssm 必变，这是**真判据**，不是恒真 guard）；
  所有 float 段非零且非恒定。
  **不证明**数值正确——非空洞只排除"恒输出常数"这一类假通过。
* **Ver-F/G/H QSA 段**（F1–F4、H1–H4、G1–G2）：证明 `token_indices` 计数列与完整组数符合规范；
  稀疏 attention 的"按 KV 头向量化"实现与**独立的逐头重算**≤1 bf16 ulp 一致（F3）；
  packed selection 的计数列/因果性/尾巴结构不变式（F4）；
  block top-k 截断 + 计数列 = 2048+tail（H1–H3）+ logits dump 覆盖全部可见块（H4）；全局最终 mixer 的 3 张量契约（G1–G2）。
  **不证明**：QSA 段**没有**与官方 CUDA 内核的位级对拍（本机无 GPU + `triton` 运行环境不完整）；
  它是"按内核语义复刻 + 与官方测试参考的语义一致"，属于 **≤1ulp 类**，不是位级类。
  QSA 的 RoPE/打分累积顺序未与内核逐位核对（`ref/qsa.py` 头部 D5）。
* **Ver-文档对账**（§8 表 + selfcheck D 段）：证明**本实现用到的**每条结论都能指到源码行。
  **不证明** docs/14/docs/15 的全部结论——只覆盖本 mission 触达的部分。
* **Ver-I 锚点存在性**（I1–I2 + `tools/check_anchors.py`）：证明 `ref/*.py`、`README.md`、`selfcheck.py`、
  `tools/*.py` 里的 **522 条 `文件:行号` 锚点**都能解析到**唯一的、作者想要的那个**文件、且行号不越界。
  三个结果**分开报**：**522/522 verified against the intended file，0 UNVERIFIED，0 out of range**。
  * **为什么必须分开报**：round-1 的检查器只有"越界/不越界"两态，而它的裸文件名消歧会把
    `hc.py` 解析到 `models/hy_v4/nvidia/hc.py`、`model.py` 解析到 `models/deepseek_v32/nvidia/model.py`
    —— **82/514 条锚点当时是拿"另一个同名文件"做的越界判定并静默通过**（74 条是 `hc.py`）。
    `0 out of range` 因此**不等于**"锚点指到了目标文件"。现在裸文件名只走显式 `BASENAME_MAP` 或
    "全树唯一同名"，多候选一律记 **UNVERIFIED 且脚本非零退出**（reviewer 要求的形态）。
  * **可复用的教训**：**凡按裸名解析，都必须有歧义清单，并把"猜出来的"结果作为独立结论报出来。**
    因为错的原因而通过，比直接失败更危险 —— 同一条毛病在 round-1 已经以 `ops/qsa.py` 静默命中 amd 变体
    （1122 行）的形式出现过一次，当时是我自己抓到的；这次是 reviewer 在更大尺度上抓到的。
    **结论：一个检查器需要被另一个独立手段复核**，所以 I2 让检查器跑自己的负向对照。
  * **符号优先（项目规则 2026-09-26；`docs/17` §8.2）**：锚点按 mission 要求保留行号，但 `--verbose` 会为每条锚点打印
    **它落在哪个符号里**（例：`hc.py:45 -> _grouped_gemma_rmsnorm_kernel`、`hc.py:327 -> _hc_combine_norm_kernel`），
    行号只是查阅提示、**符号才是稳定引用**（行号基准 commit 见文首）。
  * **覆盖范围自报（项目规则 2026-09-26 / `docs/17` §9.2「统计/校验工具必须交代自己真正的覆盖范围」）**：检查器每次输出都打印
    **扫了多少文件、匹配器认哪种写法**，并把**匹配器没认出来的 `token:NNN`** 单列成 `NOT-CHECKED`
    —— 因为"0 问题"若没有这条，只等于"我的匹配器看不见问题"。这条在 M51 的符号枚举器上真实发生过
    （它的正则要求带扩展名，于是看不见 234 处简写形态却只报 56 处）。当前 **0 条 NOT-CHECKED**；
<!-- anchor-check:off -->
    本轮它抓出过两条**真引用**被写法漏掉（`ref/hc.py` 里的 `Qwen4ExpDecoderLayer.forward:313-314`
    与 §8 的 `common:83` —— 这两个 token **故意保留不带扩展名的原样**，用 `anchor-check:off` 跳过、
    以免它们又被自己的检查器判成 NOT-CHECKED），已改成**文件限定 + 符号**的形式并纳入检查：
    `model.py:313-314`（`Qwen4ExpDecoderLayer.forward`）与
    `common/hyperconnection.py:83`（`GroupedGemmaRMSNorm.forward`）。
<!-- anchor-check:on -->
  * **I2 负向对照**（`check_anchors.py --selftest`，**7** 个用例全过）：构造"只在错误的同名文件里存在"的锚点，
    确认它被报成 **OUT OF RANGE against the mapped file**，而不是通过；再构造一个没有候选的裸名，
    确认报 **UNVERIFIED**；最后一条是**已知盲区**对照：一条**故意不带扩展名**的引用必须出现在
    `NOT-CHECKED` 列表里而不是被当成通过（该用例的原文见 `NEGATIVE_CASES`，这里的描述被
    `anchor-check:off` 跳过）。用例原文见 `check_anchors.NEGATIVE_CASES`
    （下面引用的几个是**故意写错**的，所以被 `anchor-check:off` 跳过）：
<!-- anchor-check:off -->
    `causal_conv1d.py:600`（错的那个 1288 行、对的那个 147 行）、`hc.py:700`（映射到 504 行的那个）、
    `nonexistent_twin_name.py:1`
<!-- anchor-check:on -->
  * **不证明**「该行内容就是注释所说的语义」—— 检查器只查引用存在性与归属，语义要人读。
    指向本仓的锚点写 `R:`、checkpoint 写 `CKPT:`；根目录缺失时记 SKIP（不是 FAIL），
    所以把 `m21_layer_ref/` 单独拷到别处跑也不会误报。

* **Ver-J reader 逐字同步**（J1–J3）：证明 `tools/safetensors_reader.py` 与上游
  `tools/weights/safetensors_reader.py` **sha256 相同**（并 pin 在 `selfcheck.py`），
  M39 的 torch 适配只在 `tools/torch_reader.py` 子类里。
  **不证明**上游 reader 本身正确——那是 M8 的验收范围。

* **Ver-K QSA 段的官方第二实现**（K0–K5）：`ref/qsa_official_ref.py` 是**逐字复制**的官方
  `tests/models/qwen4_exp/test_qsa_reference.py:93-234`（5 个纯 torch 参考函数，sha256 pin 在
  `QSA_REFERENCE_BODY_SHA256`，K0 每次重新从上游源文件抽一次比对）——round-1 review 指出 QSA 是
  唯一没有可执行 oracle 的段，这就是补上的那一个。在 T=256 的真实权重跑上：
  K1 indexer logits 与官方参考（乘回它在 `:103` 独有的 `1/sqrt(128)`）**相对误差 ≤ 9e-8**；
  K2 block top-k **集合逐行一致**（256/256）；
  K3 展开+因果尾 **token 集合逐行一致**（256/256；官方 helper 会把 -1 压到尾部且没有计数列，
  与内核 `:268-276` 的排布不同，集合才是共同不变量）；
  K4 稀疏 attention 与官方参考 **≤2 bf16 ulp**（官方返回门控前的输出，测试在 `test_qsa_reference.py:1077`
  处乘 `sigmoid(gate)`）；
  K5 用 `_qsa_select_paged_reference` 把 `visible_blocks → 打分 → top-k` 整链端到端比一遍（256/256 集合一致），
  这是唯一一处两侧**各自独立**计算 `visible_blocks` 的检查。
  **不证明**：这一条仍不等于「与 CUDA 内核位级一致」——官方参考本身也只是语义参考（例如它用
  `softmax` 而非 kernel 的 exp2 online softmax，且它的分数带 1/√d 缩放）。它证明的是
  **`ref/qsa.py` 与官方作者的 torch 语义一致**，把 QSA 从"只能靠行号核对"提升到"下游 4 步有第二实现"。

  **覆盖边界（必须知道，别把"有第二实现"读成"有 oracle"）**：K1–K5 喂给官方参考的**输入**全部取自
  `ref/qsa.py` 自己的中间量（`qsa.index_q` / `qsa.compressed_key` / `qsa.q` / `qsa.k` / `qsa.v` /
  `qsa.token_indices` / `qsa.gate`），官方代码只负责这些输入**下游**的数学。因此 K 段覆盖的是
  **「打分 → block top-k → 展开+因果尾 → 稀疏 attention(+门控)」这 4 步**；它**不是**独立数据 oracle
  （不产生自己的输入），也**没有**第二实现的是 QSA 前端：主 qkv 投影与 `[q|gate]` 切分、
  QK-GemmaRMSNorm+partial RoPE、indexer 投影与 q 的 norm+RoPE、压缩键的 4:1 均值池化+bf16 舍入+norm+RoPE、
  以及 `o_proj`。这些部分仍只有「行号核对」这一层（D 段契约 + I 段锚点）。
  `visible_blocks` 公式由 **K5** 覆盖 —— 那是唯一一处**两侧各自独立算**这个公式（K1 是把我算好的
  `visible` 传进去），`_qsa_select_paged_reference` 因此被真正调用了一次（round-2 review 指出它当时 0 次调用）。
## 11. 已知限制

1. **整网 CPU 参考不可行**（§1）；也不产生 logits，因此**不覆盖 `docs/17` §1 L2 的三条指标**（logits 相对误差 /
   argmax 一致率 / 生成序列一致率）。L2 的这三条要等 per-layer 算子接进 vllm-ascend 之后用框架侧通路做。
2. **PLE 段不实现**（用户已裁；表 95.37 GiB 也不在 scope）。⇒ 0-based layer 1 的层界输入**不可能**是本参考生成的，
   该层的 `--input synthetic` 只是量级正确的合成值。
3. **层界输入不是真值**，除非 `--input chain:N` 且接受"PLE 关闭"这个前提。
4. **QSA 段的存储是单序列稠密**，不是 paged；`block_table` / `token_to_req` / metadata builder 都没实现。
   单请求、无投机 token 时这是精确重索引（见 `ref/qsa.py` D1–D3），**多请求/投机解码不支持**。
4b. **QSA 的输出是「官方稀疏输出」，不得当稠密判据**（`docs/17` §7；禁令原文见 §5 末尾）：
    `attn.out` / `layer.out.*` 是与官方稀疏路径对齐的激活，长 context 下正确实现稠密 attention 的 kernel
    与它**必然不同**；对拍必须先对齐 `qsa.token_indices` 再比 attention 输出。
5. **MoE 段没有官方可跑路径**：checkpoint 的 `quant_method="ascend"` 在 vLLM main 不存在（docs/14 §11 第 6 条）。
   本参考 = "权重按 checkpoint 精确反量化 + fp32 激活"的全精度链；我们 kernel 的 W4A4 激活量化误差
   要用 `--profile mx-budget` 读。
6. **QSA 段的 RoPE 用 fp32 演算 + bf16 cos/sin cache**（`ref/qsa.py` D5）。内核里部分中间量可能是 bf16，
   所以 QSA 段统计口径是"≤1ulp 类"而非位级。要位级需要逐内核复刻 `tl.*` 的 dtype 提升规则。
7. **`ref/mxfp4.py` 的 nibble 顺序只能靠文档 + 本仓 golden 交叉确认**（checkpoint `README.quant.md:7` 与
   `R:tools/golden/moe_block_ref.py:175-180` 一致），**没有** device GEMM 的 A/B 实证。若字节顺序反了，
   MoE 段会整体错但与自身自洽——这是一个**尚未被独立证明**的前提（`docs/14` §11 也把它列为易踩点）。
8. ~~QSA 的 top-k 截断路径没有归档 dump~~ —— **已关闭**：`layer3_decode_m1` 改为 warmup 2100
   （dump 位置 2100，`visible_blocks = 525 > 512`），归档覆盖截断；`selfcheck.py` H1–H3 也独立验证了它。
9. **GDN 用顺序递推**，不是 chunked 因子分解。两者数学等价（vLLM 自己的 CPU 测试 `test_cpu_gdn_ops.py:308-367` 互证），
   但 fp 舍入路径不同 ⇒ 与官方 chunk 内核不是位级一致（≤1ulp 类）。
10. **性能无关**（用户已裁）：耗时**区间**见 §13 的 `evidence/rebuild_timing.txt` / `rebuild_timing_with_evidence.txt`
    （host 墙钟 + 共享机器 ⇒ 单值不构成证据，项目规则 2026-09-26 / `docs/17` §9.1）。不设线程上限时 torch 会用到 ~15 个核；
    共用机器上可 `M39_THREADS=4 …` 限流（`tools/make_reference.sh` 固定为 8，以保持与归档证据同一配置）。
11. **`evidence/` 里的 dump 与 README 的 sha256 必须同 commit**（`docs/17` §2 第 6 条）。

## 12. 依赖清单（最终实际用到的）

| 依赖 | 版本 | 用途 |
|---|---|---|
| `/workspace/venvs/baseline/bin/python3` | CPython 3.12.13 | 手动 venv（M30 装好） |
| `torch` | 2.10.0+cpu | 全部参考数学 |
| `numpy` | 2.5.1 | `.npy` 读写、MXFP4 字节处理、sha256 |
| 标准库 | — | `hashlib`, `json`, `mmap`, `struct`, `ast`, `importlib`, `re`, `warnings` |

**没有**安装 vllm 包、**没有**新增任何 pip 依赖、**没有**建新 venv、**没有**跑 NPU。
`vllm` 源码只以两种方式被使用：(a) 只读引用（人读 + 行号标注）；(b) 文件路径加载 `common/hyperconnection.py`
（它只 import torch）。

## 13. 证据清单（`evidence/`）

| 文件 | 内容 |
|---|---|
| `import_probe.txt` | 直接 import 探测输出（逐 target 失败原因）+ 121 个第三方 distribution 的可用性 + 官方 `common/hyperconnection.py` 加载成功的证据 + `/workspace/vllm` 未编译（`*.so` 数 = 0） |
| `reference_sha256.txt` | **183 个 dump tensor 的 sha256**（与各 `manifest.json` 内记录互检，脚本不一致就报错退出） |
| `reference_sha256.first_run.txt` | 第一次独立 build 的 sha256，作为确定性基线 |
| `selfcheck.log` | **51 条自检**的完整输出（A–K 段，0 FAIL；含 I2 的锚点检查器负向对照） |
| `anchors.txt` | `tools/check_anchors.py` 的输出：**522 anchors / 522 verified against the intended file / 0 UNVERIFIED / 0 out of range**（round-1 P1 + round-2 P2-4 的机械防线） |
| `rebuild_timing.txt` | 6 个 tag **dump 阶段**耗时的 5 次连跑区间（含当时 loadavg） |
| `rebuild_timing_with_evidence.txt` | 同上再叠加 `make_evidence.sh` 的 3 次连跑区间。**host 墙钟 + 共享机器 ⇒ 只当区间读**（项目规则 2026-09-26 / `docs/17` §9.1） |
| `peak_rss.txt` | 6 个 tag 的**子进程峰值内存**（`os.wait4` 读 `ru_maxrss`，`tools/measure_peak_rss.py` 产出，含采集时 `loadavg`）。§0 第 4 行的"≈5.4 GiB"引这一份。**它是一次测量、不是确定性产物**（逐 tag 会在 ~0.05 GiB 内浮动，跑两次不会逐字节相同），所以要引**量级与 min/max**、不要引单 tag 数字 —— 文件头已写明这一点 |
| `nonhollow.md` | 换 seed 的非空洞报告（27/27 段随输入变化；非有限对已跳过、个数单列，不再出现 `nan`） |
| `compare_positive.md` | 对拍工具自检·正例（参考 vs 自己 → 3/3 PASS） |
| `compare_negative.md` | 对拍工具自检·反例（参考 vs 换 pending 的跑 → 3/3 FAIL，含 usage/ulp 数字） |
| `compare_strict_mask.md` | 掩码口径·strict：把「官方未打分列」填 0 的合成 dump → **1/1 FAIL**（`--mask-policy` 的正例） |
| `compare_ignore_unscored.md` | 掩码口径·`--mask-policy ignore-unscored`：同一 dump → **1/1 PASS**，`ignored_unscored = 528` |
| `reference_build.log` | 生成参考集的完整命令日志（含每次 run 的参数与耗时） |

**确定性已实测**：多次完全独立的 build（不同进程）产出的 183 个 tensor 的 sha256 **逐位相同**
（`tools/make_evidence.sh` 步骤 3 会与基线 diff，diff 非空即失败退出）。
**归档里不再有"活值"（round-2 review 的 P2-6）**：三处都改掉了 ——
① `manifest.json` 的 `extra` 只含确定性参数（wall time 移到 stdout → `reference_build.log`），
所以报告里打印的 `manifest sha256[:16]` 不再随重建漂移；
② 合成的 mask 证据用固定相对路径 `reference/.tmp_zeroed`（用完 `rm -rf`），不再把 `mktemp` 的
`/tmp/tmp.XXXX` 写进证据；
③ `check_anchors.py --selftest` 的负向对照用例改用包内固定路径 `.anchor_selftest_case.md`
（原先 `tempfile` 的随机路径会落进 `selfcheck.log`）；
④ **不再打印绝对 checkout 路径**（round-3 P2-3）：`check_anchors` 的锚点源标签改成**相对包根**的短名
（`src_name()`），`run_reference.py` 的 `segments -> <dir>` 改成相对 `HERE` 的路径
—— 否则把 `m21_layer_ref/` 换到别的目录复现，`selfcheck.log` / `reference_build.log` 就不同；
⑤ `run_reference.py` 那行 `wall clock ... not archived` **与事实相反**（这行正被 `make_reference.sh`
`tee` 进已归档的 `reference_build.log`，round-3 P2-4），已改成 `NOT a manifest field
(this stdout line IS tee'd into evidence/reference_build.log)`。

**已核（两种方式）**：
1. **同目录**连跑两次 `bash tools/make_evidence.sh` → `compare_positive.md` / `compare_negative.md` /
   `compare_strict_mask.md` / `compare_ignore_unscored.md` / `nonhollow.md` / `anchors.txt` /
   `selfcheck.log` / `reference_sha256.txt` **逐字节相同**；
2. **换目录**（把整个包 `cp -r` 到 `/tmp` 再跑 `make_evidence.sh`）→ 同样这 8 个文件与源目录
   **逐字节相同**（round-3 P2-3 修之前 `selfcheck.log` 与 `reference_build.log` 会因绝对路径而不同）。

**`make_evidence.sh` 不重写的文件**：`reference_sha256.first_run.txt` 是**冻结基线**（首次构建时抄一份，
按设计不随本次重建更新 —— 现在它与 `reference_sha256.txt` **逐字节相同**，因为 manifest 里已经没有活值了；
它被排除是因为"冻结"，**不是**因为有抖动）；`rebuild_timing*.txt` 由 `time_rebuild.sh` 单独产出。

**另外按设计每次都会变、因此只能引量级/区间（不能引单值）的有两个**：`peak_rss.txt`（RSS 是一次测量，
逐 tag 在 ~0.05 GiB 内浮动）与 **`reference_build.log`**（每轮都带 wall time，文件内已逐行标注
`qualitative only, NOT a manifest field`）。两者的文件头都写明了这一点。
复现：
```bash
bash tools/make_evidence.sh && cp -r evidence /tmp/ev_a
bash tools/make_evidence.sh && diff -r --exclude='rebuild_timing*.txt' --exclude='reference_sha256.first_run.txt' /tmp/ev_a evidence
```
这是 `docs/17` §4「确定性前提要实测」要求的正面证据。
固定 `M39_THREADS=8`（`tools/make_reference.sh` 已写死）；实测 8 线程 vs 默认线程只有
`moe.routed_out` 一个段会变（CPU GEMM 分块），其余 182 个逐位不变 ⇒ 该段的位级比较按 l0 容差判。

**`.npy` payload 不入库**（见 `m21_layer_ref/.gitignore`），入库的是 `manifest.json`（含每段
dtype/shape/sha256 + 运行参数）与上面的 `reference_sha256.txt`。这与本仓既有约定一致
（`m13_moe_layer/evidence/` 也是日志 + manifest，不含二进制），且重建是确定性的。

**重建耗时**：**host 墙钟、共享机器，只能当区间读**（项目规则 2026-09-26 / `docs/17` §9.1「host 墙钟计时在共享卡上不构成证据」（原文：**只报一个数就是不合格的证据**））。
原始记录在 `evidence/rebuild_timing.txt` 与 `evidence/rebuild_timing_with_evidence.txt`，
两份都带采集时的 `loadavg`。实测（`M39_THREADS=8`，机器当时 load 13~30，即**并非静场**）：

| 阶段 | 5/3 次连跑的区间 | spread |
|---|---|---|
| 6 个 tag 的 dump（`run_reference.py` ×6） | **39.0 – 67.2 s**（两次会话：39.5–67.2 与 39.0–54.5） | 1.40 – 1.70× |
| + `make_evidence.sh`（自检 + 对拍 + 探针 + sha256） | **88.6 – 130.3 s** | 1.47× |

**跨会话还会更宽**：上一轮 review 在同机上量到 **约 180 s**（它采集时的负载更高）。
⇒ 引用时必须给区间与当时的负载背景；**不要写单值**（我上一轮写的"约 15 s / 2 分钟 / 3–4 分钟"三个数
就是错的，被 review 抓到 —— 它们分别是 `quick` 模式量级、旧观测、和更早的估算）。
项目规则 2026-09-26（`docs/17` **§9.1**）：**host 墙钟在共享机器上不构成证据**，抖动大于待断言差异时不得下结论。

docstring 里引用的文件都真实存在：唯一曾引用不存在的 `evidence/determinism.diff`（通过即删除）
已改成明确写出这个行为 —— 见 `reference/README.md`。

复现：
```bash
cd m21_layer_ref
bash tools/make_reference.sh   # 重建 reference/ + evidence/（耗时见 evidence/rebuild_timing.txt）
bash tools/make_evidence.sh    # 只刷 evidence/（不重跑 dump）
```
