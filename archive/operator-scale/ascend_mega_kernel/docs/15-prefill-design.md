# Prefill m=4097 契约与设计（逐段形状 · 算法选型 · donor 地图 · decode/prefill 共用裁决）

> **⚠ 状态横幅（M103 加，2026-09-27，钉在 `20bd20d`；本横幅不改写正文任何一行）**
> 本文**正文**（`## 0.` 到 `## 10.`）写于 **`20bd20d` 之前**，早于这五个已合入的里程碑：
> **M40**（MoE 段融合进 per-layer kernel）、**M58**（hc 边界融合，两相位 → 四相位）、
> **M91**（MoE 段流式 router + 紧凑专家槽）、**M97**（attention 前端 prolog 接进相位 A）、
> **M98**（**主 KV 几何更正**：页 16,384 → 32,768 B、每层 4,210,688 → 8,421,376 B）。
> ⇒ **正文里所有 `文件:行号` 引用在当前树上已失效**（行号随后续合入漂移；其中主 KV 的**旧几何**
> 还会算出**恰好一半**的 stride，而且是"写错地方也不越界"的静默错）。
> 正文的**形状表、FLOP/时间估算与裁决（§5 方案 B）仍然有价值**，但**每一个引用位点都必须重取**。
> **现状、五段逐段缺口、以及文件级拆分清单，一律以文末 `## M103 重盘（@20bd20d）` 一节为权威** ——
> 该节引用为 2026-09-27 现取，行号一律标注基准 commit，并给出可直接照抄成 scope 的文件级 glob。


> 调研：M33（agent-prefill，2026-09-26，**只读 mission**）。权威规格＝官方 vLLM `qwen4_exp`
> 实现（`/workspace/vllm/vllm/models/qwen4_exp/`，main v0.30.1rc0-189）+ checkpoint
> `/workspace/Qwen3.8-Flash-Next-MXFP4/config.json`。所有结论标注来源（`文件:行号` 或 config 字段）；
> 未取证/存疑集中在 §9。
> 约束基线：docs/05 §2/§6（mix(1,2) 全核、只基础 API、禁 TPipe/TBuf/TQue/AllocTensor、
> BufferID/CrossCore/地址静态自管理、尽量不挂 PIPE_S）、docs/05 §6.2（CrossCore pipe 类必须匹配核型）、
> docs/12（段序与资源表草案）、docs/10（GDN 数学）、docs/11（QSA/KV）。
> 硬件锚点（本文件所有时间估算的基准）：CUBE 实测 **BF16 ≈ 368 TFLOPS / FP16 ≈ 373 TFLOPS**
> （`cannbot-knowledge/knowledge/ops/ascendc/concepts/roofline_model.md:26-28`，在 950PR 上实测），
> MXFP4 = 4×FP16 同频（docs/03:27），HBM 名义 1.6 TB/s / 实测 ~1.1 TB/s
> （`cannbot-knowledge/.../target_ascend950pr.md:81-82`），L2 128MB / 读带宽 5.28 TB/s（同文件 :85），
> L1 512KB、L0C 256KB、UB 248KB（同文件 :40/:37），本机 28 AIC + 56 AIV（docs/05 §2）。
> **本文件不写代码、不提交变更。**

---

## 0. 结论速览（先读这 8 条）

1. **prefill 的瓶颈结构与 decode 完全不同**：decode 是"权重带宽支配、cube 几乎空转"
   （m15 实测 141.4µs/层、AIC MTE2 占 70.3%、MAC 11.3%，`wt-25/m15_layer_loop/README.md:137-141`）；
   prefill m=4097 下**除 MoE 外全部转向算力支配**（§6.4 的 AI 表）。同一层 kernel 里两条路径的
   资源瓶颈相反，这是"共用一个 kernel"裁决的关键输入。
2. **GDN 用 chunked scan + WY 表示，BT=64**：官方三处实现（CANN `chunk_gated_delta_rule`、
   vllm-ascend Triton+FLA、CANN `chunk_kda_fwd`）**全部 BT=64**，其中 GDN 版硬编码
   （`ops-transformer/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:120`
   `int64_t c = 64;`，vllm-ascend `vllm_ascend/ops/triton/fla/chunk.py:51`，FLA
   `vllm/third_party/flash_linear_attention/ops/utils.py:31` `FLA_CHUNK_SIZE`）。BT=128 只有 KDA 家族
   声明支持，且 950 的特化路径仍以 `chunkSize == 64` 为门（`chunk_kda_fwd/op_host/arch35/
   chunk_kda_fwd_tiling_impl.h:29`）。→ **首版 BT=64，m=4097 = 64×64 + 1（65 个 chunk，尾块 1 行）**。
3. **intra-chunk 用 mmad，triangular inverse 用 RegBase VF**：`A=kβ·kᵀ`、`w=T@(βe^g k)`、`u=T@(βv)`、
   三个扫描 matmul 用 mmad；`(I+L)⁻¹` 用 32×32 对角块寄存器前代 + 下左块 2 次 mmad
   （donor `stage1_arch35.h:710-728` + `vf/...stage1_vf.h:87-128`，`INVERSE_SHAPE=32`）。
4. **QSA 在 m=4097 下几乎不省算力**（按 §6.3 计数：稠密 causal ≈ 8.39M token-pair，QSA ≈ 6.3M
   token-pair，**仅 ~1.33×**）：`indexer_budget=2048` 只覆盖 512 个 4-token 块，而 4097 长度的
   prompt 有 1025 个块 → 位置 4096 的 query 只看得到 2048 个历史 token。省下的 ≈51 GFLOP/层
   （×12 层 = 618 GFLOP）只相当于全 pass 算力下限的 **~1.1%**（attention 段自身的 8%）。
   → **prefill 首期走稠密 causal + gather 预留**（与 docs/11 §5（遗留事项 1）裁决一致）；但两件事必须注意：
   (a) indexer 的**缓存填充**（raw key ring + compressed key cache）在 prefill 必须做，否则后续
   decode 无 cache 可用（M35 的范围切分建议）；(b) **稠密替代与官方 QSA 数值不等价**，
   验收 golden 的口径必须二选一（见 §9 存疑 3，**已 escalate**）。
5. **MoE 在 prefill 仍是带宽支配**：每层专家权重 1.26GB(4bit) + 78.6MB scale（按 config 形状算），
   4097 token × top-10 = 40970 槽位摊到 512 专家 ≈ **80 槽/专家**，专家内 N/K 复用的 AI 只有 ~150
   FLOP/B，低于 MXFP4 ridge（~919 FLOP/B）→ 每层 ≈1.28ms 搬运（权重 1.34GB + 激活 ~0.70GB）
   是下限，而算力只需 ~0.21ms（§6.3）。全 pass 48 层 ≈ **61ms**，是 prefill 的最大单项。
   **MoE 是 prefill 里唯一"等 m 变大才能变算力支配"的段**（crossover ≈ 每专家 400 槽，
   即 m ≈ 20480）。
6. **共用裁决（§5）：推荐方案 B ——"单 TU、两个入口符号（decode / prefill）、一套编译期静态资源表"**，
   host 按 m 选符号，kernel 内**没有任何按 m 的运行时分叉**。
   理由：norm / proj / RMSNormGated 等段可零改动复用；而 m=1 的 head 条带递推与 prefill 的
   chunk 扫描在**每核工作项映射、UB 布局、串行临界路径**上完全不同，硬塞进同一段序只会两头受害。
   方案 A（段内按 m 分支）在 5 个维度全面最差（UB 必须叠放、同步图拓扑表达不了、icache 最差）；
   方案 C（两套独立 kernel/TU）调试成本最低但资源表会漂移，且 decode 的 5 个算件+段序要复制成两份。
7. **不返工清单**：§5.3 给出 14 条 decode 侧**现在就要遵守**的接口约束（可逐条勾选），
   其中第 1/3/4/8 条是"今天不改、prefill 期就要重写"的高危项。
8. **依赖**：可并行做的段（GDN chunk 扫描、MoE permute/GMM、HC、PLE）与必须等 M24/M27/M31 的段
   （attention core、L0 装载几何、层边界/hc 规格）见 §7。

---

## 1. 尺寸与常量总表（m=4097 的形状计算基准）

来源：`/workspace/Qwen3.8-Flash-Next-MXFP4/config.json`（字段名逐字），共享权重形状经
safetensors 头核对（由 §4 各 donor 调研一并核过）。

| 量 | 值 | 出处 |
|---|---|---|
| 层数 / 类型 | 48 = 36 `linear_attention` + 12 `full_attention`（层号 3,7,…,47） | `config.json: layer_types`, `full_attention_interval=4` |
| hidden | 2560 | `hidden_size` |
| head_dim / q 头 / kv 头 | 256 / 24 / 2（GQA g=12） | `head_dim`,`num_attention_heads`,`num_key_value_heads` |
| q_proj 宽度 | 24×256×2 = 12288（q|gate 各 6144） | `attn_output_gate` 缺省 True，`qwen3_next.py:307` |
| rope | partial 0.25 → 64/256；θ=1e7；`mrope_interleaved=true` | `partial_rotary_factor`,`rope_parameters` |
| QSA indexer | 4 头×128 + 1 KV 头；budget 2048 token；compress 4 → 512 块 | `indexer_*` |
| GDN | q/k 16 头×128；v 48 头×128（48=3×16，v 头 hv 用 key 头 hv//3）；conv K=4 over 10240ch | `linear_*`；`docs/10:7-10` |
| GDN 投影 | in_proj 2560→16480（q2048|k2048|v6144|z6144|b48|a48）；out_proj 6144→2560 | `docs/10:7`；`m15_layer_loop/README.md:66-67` |
| MoE | 512 路由专家 top-10 + 1 共享专家；intermediate 640；共享 expert 640 | `num_experts`,`num_experts_per_tok`,`*_intermediate_size` |
| MoE 权重（量化） | `mlp.experts.gate_up_proj`（**fused**）+ `experts.down_proj` + shared 三个，MXFP4 e2m1+e8m0 group32 | `quantization_config.quantized_tensors`,`format` |
| 未量化 | 全部 attention/GDN 投影、router、ngram、MTP 保持 bf16；`quantize_linear_attn=false` | 同上列表（不含 linear_attn 键）+ 该字段 |
| HC | `hc_count=4`、`hc_lowrank=320` → `hyper_hidden_size = 4×2560 = 10240`；`hc_per_branch_norm` 硬编码 True | `hc_*`；`nvidia/model.py:265,435` |
| PLE | `ple_layer_ids=[2]`（1-based → 0-based 层 1）、`ple_embed_dim=2560`、`ple_conv_kernel_size=4`、`ngram_size=3`（兼作 dilation） | `ple_*`；`nvidia/model.py:195` |
| ngram 表 | 16 头（=2×8）×160 = 2560；词表 320,001,536 行 → **95.37 GiB** | `heads_per_ngram`,`ngram_vocab_size_base`,`make_ngram_vocab_size_divisible_by`；§4.5 |
| vocab / lm_head | 248320 × 2560 | `vocab_size` |
| m | 4097 = 64×64 + 1 | mission 指定 |

**注意两处容易抄错的地方**：(a) `output_gate_type="sigmoid"` 是 GDN 的 **RMSNormGated 输出门**
（`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:483-495`），**不是 MoE router**——
router 是 softmax（§2.6）；(b) `ple_layer_ids=[2]` 是 1-based，PLE 挂在 **0-based 层 1**
（`nvidia/model.py:195` `if (self.layer_idx + 1) in ple_layer_ids`，checkpoint 里只有
`layers.1.ple.*`）。

---

## 2. 逐段形状与分块策略（m=4097）

### 2.1 层的调用图（权威 = 官方 vLLM，`nvidia/model.py`）

一层的执行序（行号见括号，全部来自 `vllm/models/qwen4_exp/nvidia/model.py`）：

```
1  [仅层 1] PLE: ngram id → gather → kv_proj → ple_gate → dilated dwconv → 加进 [m,10240] 残差  (:290-306)
2  attn_hc.combine_and_mix(hidden, prev_mlp_out, prev_inj)                    (:309-314)
   2a 组内 GemmaRMSNorm(4 段 × 2560) → 2b W_down+inj [336,10240] → 2c silu(/4)
   2d W_up [10240,320] → 2e sigmoid 门 → 平均成 [m,2560] 的 block_input
3  ATTENTION(block_input)   —— 分支：linear_attention → GDN；full_attention → QSA   (:316-324)
4  mlp_hc.combine_and_mix(hidden, attn_out, inj)   → block_input2 [m,2560]          (:327-329)
5  MoE(block_input2) → mlp_out（**不立即相加**，交给下一层的第 2 步）                  (:330-331)
6  [层 47 后] hyper_connection_mixer.combine_and_mix（含最终 RMSNorm）→ sample_hidden [m,2560] (:579-583)
7  lm_head → logits [m,248320]                                                    (:840-841)
```

**关键结构性事实**：残差流是 `[m, 10240]`（4 条流），只有交给 attention/MoE 的 `block_input` 是
`[m,2560]`；HC 的 combine 是**延迟一拍**的（层 L 的 combine 消费层 L-1 的 mlp 输出）；因此
kernel 边界若按"层"切，每层要多吃一个 `[m,2560]` 的 pending 张量（或等价地把 combine 放进被调用层）。
**m=1 与 m=4097 在这段代码里没有任何 `if m==1` 分支**（调研结论：`model.py:276-331/489-590`
无 m 分支；唯一 prefill/decode 分叉在 PLE 短卷积 `ple_layer.py:193-334` 与 QSA/GDN 的后端配置）。

### 2.2 embedding / ngram / PLE / lm_head

| 段 | 输入 → 输出 | m=4097 字节 | 分块策略 |
|---|---|---|---|
| embedding | ids[4097] → [4097,2560] bf16 | 21.0 MB | 行并行；`repeat(1, hc_count)` 直接扩到 [4097,10240]（`model.py:506`，**无 /4 缩放**） |
| ngram id | ids + `ngram_context[1,2]` + `query_start_loc` → ids[4097,16] int64 | 0.5 MB | 纯逐 token（§4.5）；`crossed` 掩码是 2 步依赖链，按 token 并行 |
| ngram gather | ids → [4097,16,160] → flatten [4097,2560] bf16 | 读 21 MB + 写 21 MB | **按 160 维行 gather（随机跨 95.37GiB 表）**，是纯延迟/带宽问题；预取编排可抄 `model.py:516-534` |
| PLE（层 1） | emb[4097,2560] + hidden[4097,10240] → hidden' [4097,10240] | 84 MB×若干 | 逐 token；dilated dwconv（dilation 3、4 tap、10240 ch）在 prefill 只需 `mode="prefill"` + 无初态（`ops/ple.py:318-688`，可丢掉 state writeback） |
| final norm | 融进 `hyper_connection_mixer.combine_and_mix`（**模型里没有独立的 `model.norm`**）| — | 与 HC 同一段 |
| lm_head | [4097,2560] × [248320,2560] → logits[4097,248320] bf16 | 权重 1.27GB，**logits 2.03GB** | M 行分块（N=248320 必须切 N；logits 2GB 是全 pass 最大的单体缓冲） |

lm_head 算力 2·4097·248320·2560 = **5.21 TFLOP**，AI ≈ 1570 FLOP/B → 算力支配；但 2.03GB 的
logits 写出本身就是 1.27ms@1.6TB/s，且必须在显存里放得下。

### 2.3 hyper-connection（每层 2 个模块 + 全局 1 个）

数学（`nvidia/ops/hc.py`，逐式核对）：

```
(a) 组内 GemmaRMSNorm：X[m,10240] 按 2560 分组 ×4 归一，scale = (1+w[s*2560+d])   hc.py:13-78
(b) [lora|inj|pad] = Xn @ W_{down+inj}^T     W:[336,10240]（320 lora + 4 inj + 12 pad）
(c) lora = (lora/4) * sigmoid(lora/4)                                            hc.py:82-122
(d) gate = lora @ W_up^T                     W:[10240,320]
(e) block_input[t,d] = (1/4) Σ_s sigmoid(gate[t,s*2560+d]) * Xn[t,s*2560+d]      hc.py:126-185
(f) combine: out = X + 2*sigmoid(inj_prev/4)*block_output（**先舍入到 bf16**，hc.py:327）
             然后对 out 再做一次 (a) 的组内 norm
```

m=4097 下的资源：`W_down+inj` 6.88 MB、`W_up` 6.55 MB（每模块 13.43MB）。
FLOPs = 13.43 MFLOP/token/模块 → **每层 26.9 MFLOP/token = 110 GFLOP**，全 pass 48 层 + 全局 1 个
≈ **14.5 TFLOP**（≈ MoE 全 pass 的算力，见 §6.3）。
分块：按 token 行分块即可（**无 token 间交互**），唯一跨 token 状态是"延迟一拍的 pending 张量"。
注意 (f) 的 `bf16(out)` 中间舍入是**位级可比性要求**（`hc.py:325-327` 注释即为此）。

### 2.4 GDN 段（36 层）——prefill 形态

**输入**：`x_norm [4097, 2560] bf16`（= HC 的 block_input），`conv_state [10240, 3] bf16`（初态在
chunk 前 3 行，全新 prefill 全零），`ssm_state [48,128,128] fp32`（in-place RMW）。

按 chunk（BT=64，共 65 个 = 64 满 + 尾 1）拆：

| 子段 | 形状（每 chunk，C=有效 token 数） | 计算 | 落地核 |
|---|---|---|---|
| P0 in_proj | A[C,2560]×W[16480,2560] → qkvzba[C,16480] | mmad（N 切 103 tile） | AIC |
| P1 conv1d+SiLU | qkvzba 前 10240 通道 depthwise K=4（前缀 = conv_state 末 3 行）→ SiLU | 逐元素（VEC） | AIV |
| P2 l2norm | q/k 各 16 头×128：`x/√(Σx²+1e-6)`；q 再 ×1/√128 | 归约+逐元素（VEC） | AIV |
| P3 gating | g=−exp(A_log)·softplus(a+dt_bias) fp32（β=1,thr=20）；β=sigmoid(b) → 槽位布局 [C,48]（stride-8 契约） | 逐元素（VEC） | AIV |
| P4 chunk-local cumsum | `g` 沿 chunk 内 cumsum fp32 → `gc`；`γ[i,j]=exp(gc_i−gc_j)`（严格下三角，含对角差异） | Hillis-Steele 扫描（VF，单寄存器装 64 元素） | AIV |
| P5 A 矩阵 | `kk=k·kᵀ`（C×C×128）→ `attn=−β_i·kk_ij·γ_ij` 严格下三角 | mmad + VF 乘 | AIC+AIV |
| P6 T=(I+L)⁻¹ | 32×32 对角块寄存器前代 + 下左块 2 次 mmad | VF + mmad | AIV+AIC |
| P7 WY | `w=T@(β e^{gc} k)`  [C,128]（每 key 头）<br>`u=T@(β v)` [C,128]（每 value 头，用其 key 头的 T） | mmad | AIC |
| P8 扫描（串行） | `v_new=(u−k_cumdecay@h)·e^{gc_last−gc}`；`o_inter=q'e^{gc}@h`；`h=e^{gc_last}h+kᵀv_new` | 3 次 mmad/chunk/头 | AIC（h 在片上） |
| P9 intra 输出 | `o=o_inter+tril(QKᵀ·γ)·v_new`（accum） | mmad + VF mask | AIC+AIV |
| P10 RMSNormGated | `o[C,6144]`+z 段 → per-head(128) RMSNorm ×γ × sigmoid(z) → `opin[C,6144]` bf16 | VEC | AIV |
| P11 out_proj | `[C,6144]×[2560,6144]ᵀ` → `[C,2560]` | mmad | AIC |

形状（整层，m=4097，bf16 除注明）：

| 张量 | 形状 | 字节 |
|---|---|---|
| x_norm | [4097,2560] | 21.0 MB |
| qkvzba | [4097,16480] | 135.0 MB |
| q', kg, k_cumdecay | 各 [4097,16,128] | 各 16.8 MB |
| u, v_new, o_inter | 各 [4097,48,128] | 各 50.3 MB |
| g/β | [4097,48] fp32 ×2 | 3.1 MB |
| gc（cumsum）| [4097,16] fp32（或每 chunk [65,16,64]） | 0.26 MB |
| A/T | 每 (key 头, chunk) [64,64]（T 存 bf16 供 3 个 v 头复用） | 16 头×65×2×(64²)=17.0 MB（bf16 后 8.5MB） |
| opin | [4097,6144] | 50.3 MB |
| opout | [4097,2560] | 21.0 MB |
| **ssm_state** | [48,128,128] fp32 in-place | 3.15 MB |
| conv_state | [10240,3] bf16 in-place | 60 KB |

**分块策略（与官方差异点）**：官方把 stage1/stage2/stage3 做成**三段 + 一个 `SyncAll`**
（`chunk_gated_delta_rule_arch35.h:177-183`），stage2 按 `Nv` 分核、**h 落 GM 每 chunk
RMW + Fixpipe atomicAdd**（`stage2_arch35.h:148-179`、`matmul_basic.h:188-194`），
每层 GM 往返约 400MB（docs/10:21）。我们的改动（与 docs/10 §5 一致）：
(i) 段边界用 **mode-0 barrier** 取代 `SyncAll`；(ii) **h 按头常驻片上**跨 65 chunk，砍掉这 400MB；
(iii) l2norm+gating 融进 P4 之前（官方 GDN chunk op 不含 l2norm，见 §9 存疑 1）；
(iv) chunk 组流水穿过串行扫描（stage1 of chunk c+1 与 stage2 of chunk c 并行）。

### 2.5 full attention 段（12 层，实为 QSA 稀疏检索）

prefill 数据流（官方顺序，`nvidia/qsa.py:444-526`）：

```
1 qkv_proj → [m,12800] → q[24,256] / gate[24,256] / k[2,256] / v[2,256]（q/k 各自 RMSNorm256 + RoPE 前 64 维）
2 index_qk_proj [2560→640] → q_idx[4,128] / raw_k[1,128]
3 indexer：q_idx = GemmaRMSNorm(128)+RoPE(64) ；
   raw key ring 写（[blocks, num_states, 1, 128] bf16）；
   每满 4 token 的组：mean-pool 4 个 raw key → GemmaRMSNorm → 用**组首 token 位置**做 RoPE
   → 写 compressed key cache（[blocks, page_size, 1, 128]，bf16）
4 打分：score(row) = Σ_{4 头} ReLU(q_idx·compressed_K)（**无 softmax**，topk 不变性允许省 /√d）
   候选块 b 可见 ⟺ 4b+3 ≤ qpos（严格因果，块粒度）
5 topk：block_topk = 2048/4 = 512 块/行 → expand 成 token 索引，拼 ≤3 个 causal-tail 尾部 token，
   无效位填 −1，末列写有效数 → **packed [m, 2052] int32**（`indexer_qsa.py:182-195`、`ops/qsa_indexer.py:221-276`）
6 主 KV 写（paged，`do_kv_cache_update`）——**必须在 gather 之前**（当前 chunk 的尾部 token 要从主 KV 读）
7 稀疏核心：gather ≤2051 token + online fp32 softmax + ×sigmoid(gate)（`ops/qsa.py:594-758`）
8 o_proj
```

**m=4097 的关键量化结论**（§6.4 有完整计数）：
- 块数 = ⌈4097/4⌉ = 1025，预算 512 块 → 每行只能选到约一半的可见块；
- 稠密 causal 的 token-pair 数 = m(m+1)/2 = **8.39M**；QSA 的 ≈ Σ_i min(2048, i+1) = **6.3M**；
  **省幅只有 1.33×**（长上下文（≥64K）才是 28× 级收益，docs/11:28）。
- 但 indexer 的**缓存填充是 prefill 不可省的义务**：raw ring + compressed cache 是 decode 的输入。
  → prefill 首期：做 §2.5 的 1-6（含缓存写），第 7 步用**稠密 causal FA**（gather 接口留 nullptr
  语义），indexer 打分/topk 可延后（与 docs/11 §5（遗留事项 1）的裁决一致，互相印证）。

分块：prep（1-3）按 token 行分块（RoPE/norm 逐 token）；主 KV 写按 16-token 页（block_size 取 16，
docs/11 §3.1）逐 token 落页；核心按 FA 的 m-tile 流水（config4：sOuter128/sInner128，gS1-merge
g=12 进 M 维，docs/11 §4.2）。causal 的上界随 m-tile 行变化，尾部 tile 大量跳过。

### 2.6 MoE 段（48 层，全部层都有）

官方 vllm-ascend 路径（`vllm_ascend/ops/fused_moe/*`）的算子序即为契约：

```
router: moe_gating_top_k(x[m,2560], k=10, renorm=0, norm_type=0)
        = softmax(512) → top-10 → 再除以 top-10 权重和（Python 侧 renorm，device_op.py:1263-1264）
init_routing_v2(x, topk_ids[m,10], expert_num=512, active_num=m*10, expert_tokens_num_type=1 /*count*/)
        → sorted_hidden[m*10,2560] + expanded_row_idx[m*10] + expert_tokens[512] int64(**count 模式**)
        （dropless：无 capacity padding，README:172,261）
GMM#1: npu_grouped_matmul_swiglu_quant_v2(x=A_qx, weight=[W13 融合], group_list=counts,
        dequant_mode=2, quant_mode=2) → H_qx[m*10,320] u8 + H_scale[m*10,10,2] e8m0
GMM#2: npu_grouped_matmul(x=[H_qx], weight=[W2], per_token_scale=H_scale, split_item=2)
        → Y[m*10,2560] bf16
unpermute: npu_moe_token_unpermute(Y, sorted_indices=|expanded_row_idx|, probs=topk_weights)
        → routed[m,2560]
combine: routed + sigmoid(sgate(x)) * shared_expert(x)
```

形状（m=4097）：`S = m·topk = 40970` 槽位；**每专家期望 80.0 槽，二项 σ=8.94，512 专家里的最热专家
约 108-116 槽**（§4.3 计算）。→ **BASE_M=64 的 m13 单 tile 假设会被一半以上的专家打破**。
分块策略：MoE 不按 token 分块（官方 op 内部把拼接后的 M 切 workspace 段循环，
`grouped_matmul_swiglu_quant_v2_utils_kernel.h:132-142` 的 `WorkSpaceSplitConfig`），
我们的 prefill 版必须把 GEMM item 改成 `(expert, m-tile, n-tile)` 三维工作项。

---

## 3. 算法选型与依据

### 3.1 GDN：chunk 公式选型

**选型：chunked delta rule（WY 表示）+ chunk 间 h 串行扫描**，不用逐 token 递推
（逐 token 递推在 m=4097 下 = 4097 步串行 ×48 头，实测 decode 单步递推已是 µs 级、串行 4097 步
会到几十 ms，且 cube 完全用不上）。公式（逐字对齐官方 golden
`chunk_gated_delta_rule/tests/pytest/chunk_gated_delta_rule_golden.py:99-149`）：

```
scale = 1/√128
v_beta = v·β ; k_beta = k·β
g = chunk_local_cumsum(g)                                    # chunk 内 fp32
decay_mask = tril(exp(g_i − g_j))                            # 含对角（i ≥ j）
L = +(k_beta @ k^T) ⊙ decay_mask      → masked_fill(含对角的上三角：i ≤ j 全清) ⇒ L 严格下三角
T = (I + L)^{-1}                      # 块前代，见下
u = T @ v_beta
k_cumdecay = T @ (k_beta · e^g)
逐 chunk（串行）:
  v_prime = k_cumdecay @ h
  v_new   = u − v_prime
  o_inter = (q·e^g) @ h
  o       = o_inter + tril(QK^T ⊙ decay_mask) @ v_new      # 含对角（i ≥ j 保留）；本项 Q 只用 scale，不带 e^g
  h       = h·e^{g_last} + (k·e^{g_last−g})^T @ v_new       # h 保持 fp32
```

> **权威口径（M49 订正，2026-09-26）**：上面这块伪码的**唯一权威是官方 golden**——`ops-transformer/attention/chunk_gated_delta_rule/tests/pytest/chunk_gated_delta_rule_golden.py:99-149`（本文件第 264 行自己引的就是它）。M34 对账时发现原文三处与 golden 不一致、**下游照字面实现会错**，此处已按 golden 订正；**若本文件与 golden 再有出入，一律以 golden 为准**（M34 的核 `m18_gdn_prefill` 及其 README §10 对账表同此口径）。
>
> **三处订正**（依据 golden 行号）：
> ① **负号与 `(I+A)^{-1}` 不能连用**。golden ∶117 取负 → ∶118-121 fla 逐行前代 → ∶122 加 `I`，三步合起来等于 **(I+L)^{-1}**（`L = +(k_beta@k^T)⊙decay_mask` 严格下三角）；原文的 `(I+A)^{-1}`（A = −L）算的是 **(I−L)^{-1} = I+L+L²+…**，**是另一个矩阵**。随机数据 BT=2/3/5/8 实测：`max|golden − (I+L)^{-1}| ≤ 4.0e-15` vs `max|golden − (I−L)^{-1}| = 0.95 ~ 1.5e+2`。⇒ 已改成 `L = +(…)`、`T = (I+L)^{-1}`（与 vllm-ascend `fla/wy_fast.py` 及本核写法同，三处一致）。
> ② **`decay_mask` 的对角在 L 处必须清零**。golden 里同一个 `decay_mask`（∶116）配**两个不同**的 mask：L 用 `triu(diagonal=0)`（∶110-114，连对角一起清 ⇒ L 严格下三角），输出项用 `triu(diagonal=1)`（∶131-134，只清严格上三角 ⇒ **含对角**）。原文只写「masked_fill(上三角)」会被读成严格上三角，从而在 L 对角留下 `−β_i(k_i·k_i)`，**WY 直接坏掉**。⇒ 两处已分别写明「含对角的上三角：i ≤ j 全清」与「含对角（i ≤ j 保留）」。
> ③ **`e^ĝ` 只能乘一次**。`o_inter` 项的 Q 是 `q·e^ĝ`（golden ∶141），而 `tril(QKᵀ⊙decay_mask)` 项的 Q **只有 scale、不带 `e^ĝ`**（golden ∶138 的 `q_i` 在 ∶100 只乘过 scale）。把 `q·e^ĝ` 复用到第二项会得到 `e^{2ĝ_i−ĝ_j}`（多乘一个 `e^{ĝ_i}`）——M34 实现时正踩过此坑（o 误差 1e-2 量级）。⇒ 已加行内注释。
>
> **M34 核对为一致、未改**：`v_prime = k_cumdecay @ h; v_new = u − v_prime`（h **未衰减**，`e^ĝ` 在 `k_cumdecay` 里）⇔ 本核 `d = u − w·S₀`、`w = T(β⊙eg⊙k)`，与 golden ∶139-140 同序；`h = h·e^{g_last} + (k·e^{g_last−g})ᵀ@v_new` ⇔ 本核 `S₁ = egL(S₀ + kᵀ(d⊙ig))`，代数等价（只是把衰减放在 `k` 还是放在 `d` 的结合顺序不同）。

**chunk 大小候选与 UB 预算**（这是本节的量化核心）：

| 候选 | UB 需求（AIV 单核） | 扫描串行步数 | intra 计算量（∝BT） | 依据 |
|---|---|---|---|---|
| **BT=64（选）** | A/T 矩阵 [64,64] fp32 = 16KB；γ 16KB；q'/k/v chunk 段各 64×128×4 = 32KB；h slab 128×128 fp32 = **64KB**；合计 ~160KB | **65 步**（4097） | 1× | 三处官方实现全部 BT=64；`INVERSE_SHAPE=32`、`INVERSE_COUNT=5`、`UB_REST_BYTES=140KB` 都按此规模定（`stage1_arch35.h:27-30`） |
| BT=128 | A 128×128 fp32 = **64KB**（γ 再加 64KB）→ 与 h slab(64KB) + q/k/v 段(64KB) 相加超 248KB；需砍并行度或分半处理 | 33 步 | 2× | `chunk_kda_fwd` 声明 64/128（README:146），但 950 特化路径门是 `chunkSize==64`（`chunk_kda_fwd_tiling_impl.h:29`）；GDN 版无 128 先例 |

→ **BT=64**。BT=128 只在"扫描串行临界路径实测支配"时作为备选（扫描步数减半、intra 翻倍，
intra 是可并行的、扫描不可并行 → 若 65 步 × 每步开销成为瓶颈，128 值得回测）。
注意 **m=4097 的尾块只有 1 行**（4097 = 64×64+1），官方 op 用 `validLenBatch_` 处理尾块
（`stage1_arch35.h:252-263`），我们同理需要"有效长度"掩码；且尾块 1 行时 A/T 退化为 1×1，
`(I+L)⁻¹` 的 32×32 分块代码必须能用掩码退化（不能靠 if 分支绕开，因为区块划分是静态的）。

**intra-chunk 走 mmad 还是 vf？——混合，按运算类型分**：

| 运算 | 选 | 依据 |
|---|---|---|
| `kk = k@kᵀ`（64×64×128）、`QKᵀ`（64×64×128） | **mmad** | K=128 满 tile，官方 stage1/stage3 都在 AIC 上用 mmad（`stage1_arch35.h:288-290`） |
| `w = T@(βe^g k)`、`u = T@(βv)`、`v_new ᵀ@kg`、`q'@h` | **mmad** | 同上，`matmul_basic.h`；K=64/128 都够 tile |
| `(I+L)⁻¹` | **VF 前代 + mmad 补下左块** | 官方 32×32 单寄存器前代（`vf/chunk_gated_delta_rule_stage1_vf.h:87-128`，`for i in 1..32` 内层 `MulAddDst`）+ 下左块两次 mmad（`stage1_arch35.h:710-728`：`−A₂₂⁻¹A₂₁A₁₁⁻¹`） |
| cumsum / γ=exp / β 广播 / decay 掩码 / q' 缩放 | **RegBase VF**（`__VEC_SCOPE__`） | 官方 VF（`stage1_vf.h:27-77` Hillis-Steele 扫描 + `Exp`），且 docs/05 §6.1 已裁决 AIV 侧位宽转换/逐行归约一律 RegBase VF |
| P1 conv1d / P2 l2norm / P3 gating / P10 RMSNormGated | **VF** | 沿用 m9/m4/m12 既有实现 |

**h 的存放与 dtype**：官方 fp32 state 是 **950-only** 能力（`chunk_gated_delta_rule_tiling.cpp:366-369`
硬拒非 DAV_3510 的 fp32 initial_state），且官方做法是 **fp32 权威 + bf16 镜像喂 cube**
（`stage2_arch35.h:203-208`、`chunk_kda_fwd_common.h:91`）。我们建议照抄这个组合：
**权威 h 用 fp32**（保精度、与 decode 的 ssm_state 同 dtype），**喂 mmad 的 B 操作数用 bf16 镜像**
（cube 对 bf16 友好、fp32 的 cube 峰值只有 24 TFLOPS ≈ bf16 的 1/15，见 roofline:28）。

**扫描临界路径的量化**：每 chunk-头 3 次 mmad = 3×64×128×128×2 = **6.29 MFLOP**；
48 头 / 28 AIC ≈ 2 头/核 → 65×2 = 130 个串行步；单核 cube 13.1 TFLOPS → 0.48µs/步的纯算力，
但每步有固定的 L1 装载/Fixpipe/同步开销（官方 op 的 cube 每次 `IterateAll` 有 ~44-69µs 的
per-call floor 报告，见 `cannbot-knowledge/.../ol_246_cube_offload_loses_small_contraction_dim.md:29`
——那是高阶 API 路径；我们走基础 `LoadData`+`Mmad` 不受其限，但**必须实测标定**）。
保守估 1-2µs/步 → **130-260µs/层的扫描段**，与官方 stage2"是整 op 最慢段"的判断一致
（`stage2_arch35.h:119-141` 每核只有 2 头、却要串行 64 步）。

### 3.2 attention：causal mask 与 QSA 选带如何与 decode 共用 KV/压缩缓存

**(a) causal mask**：稀疏核心**内部不含任何 causal mask**——因果性全部由 indexer 的
packed indices 预烘焙（`visible_blocks = min((pos+1)//4, seq//4)` + 尾部 ≤3 个 token，
`common/qsa_cache.py:268-275`、`ops/qsa_indexer.py:242-260`）。所以"用稠密 causal 替代 QSA"
不能改核心，而要在**同一层里换掉第 5/7 步**：生成一份等价的稠密 causal 索引（或让核心支持
一个 `causal=true + s2 上界` 模式）。后者才是 donor 形态（`flash_attn` 的 config4 prefill 就是
sOuter/sInner + 三角掩码，docs/11 §2.1）。**建议：稠密路径直接走 flash_attn 的 ND config4
流水（sOuter128/sInner128 + gS1-merge），不经过 packed indices**；packed 路径作为 Phase-2
在同一个核心入口后面接上（两条路径共享 Q/K/V 的 L1 装载与 CV 交接代码）。

**(b) KV 缓存共用**：decode 与 prefill 必须写**同一份** paged 主 KV、同一份 raw key ring、同一份
compressed key cache（否则 decode 读不到 prefill 的上下文）。契约（docs/11 §3.1 + §4.1 实测）：

| cache | 布局 | dtype | 谁写 | prefill 的写法 |
|---|---|---|---|---|
| 主 KV | `[blocks, 2, block_size, 256]`（vLLM: `kv_cache.transpose(1,2).split(256,-1)`，`qsa.py:215`）；block_size 取 **16** | bf16 | 前 3 步之后、gather 之前 | 逐 token 按 slot_mapping 落页；**必须早于核心**（本 chunk 尾部 token 是核心的输入） |
| raw key ring | `[blocks, num_states, 1, 128(+12 bf16 位置)]`，容量 = `CR·cdiv(CR+num_spec, CR)` | bf16 | pre-indexer | 每 token 一行；只保留请求后缀 |
| compressed key | `[blocks, block_size//4, 1, 128]` | bf16（可选 fp8） | pre-indexer | **每满 4 token 的组写 1 行**（边界 token 位置，`(pos+1)%4==0`），K=组内 4 行 raw key 的**均值**→GemmaRMSNorm→用组首位置 RoPE |

4097 长度 → compressed 行数 = 1024（位置 3,7,…,4095），位置 4096 因 `(4096+1)%4=1` 属"开放组"，
只作为 causal 尾部 token 出现——**这个 off-by-one 必须写进我们的实现契约**。

**(c) 与 decode 共用的最小接口**：`topk_indices_buffer` 必须与 decode 完全相同
（`[max_num_batched_tokens, 2052] int32`，末列 = 有效数 = 核心的循环上界，
`nvidia/qsa.py:410-424`）。**注意**：NVIDIA 版有末列计数、AMD 版没有（`[m,2051]`）——我们跟
NVIDIA 版（权威规格）。

### 3.3 MoE：permute + grouped GEMM 在 m=4097 下

**契约（官方 vllm-ascend 路径，即最终要接入的目标）**：
`group_list` 用 **count 模式 `int64[512]`**（不是 offsets），dropless 无 padding；
MXFP4 激活量化 group 32、scale e8m0、`floor(log2(amax)) − 2`（与 m5/m13 已实现的一致，
`grouped_matmul_swiglu_quant_v2/README.md:162-183` 的 `emax(FP4)=2`）。
GMM#1 是**融合算子**：grouped matmul → dequant → split(gate,up) → SwiGLU → MXFP4 量化一次完成，
输入是**融合的 w13**（与 checkpoint 的 `experts.gate_up_proj` 一一对应 → 不需要 host 重组）。

**m>1 的改造清单（相对 m13，六个硬阻塞，全部由 `M_MAX=64` 派生）**（引用以**符号**为准；括注行号是 M33 写作时的口径，随代码变动）：
1. GEMM item 单 M tile、无 m 循环（`m13_moe_layer.asc` 的 `MXFP4GemmItem::Run` 里 `curM` 直用 + 单次 L0C→L0C 累加，原 `:161-165`）；
2. 每专家固定行槽 `NUM_EXPERTS*M_MAX` 的 GM 布局（`m13_resources.h` 的槽尺寸常量 `SZ_AQ/SZ_GU/SZ_Y` 等，原 `:237-249`）→ 必须换成
   Σt_e 紧凑布局（否则 t_e>64 直接越界写下一槽）；
3. 路由数组按 `TOTAL_MAX = M_MAX*TOPK_MAX = 256` 定长（`m13_resources.h` 的 `UB_IG_*` 槽常量，原 `:156-162`）→ 40970 槽位；
4. 计数排序只在 AIV0 单核标量（`m13_moe_layer.asc` 的 `MoeLayerChain::ProcessAiv` 里 S2/S3 的 `isPrimary` 分支，原 `:1833-1838`）；
5. router 是单核 GEMV + 16 行 UB 块（`m13_resources.h` 的 `RT_RB` / `UB_RT_*` 常量，原 `:138-143`）→ 实测单核 0.066ms@m=1、
   1.47ms@m=64（`m7_router_topk/README.md` §已知边界），线性外推到 4097 ≈ **94ms**，必须多核；
6. host 侧 `m > M_MAX` 直接拒跑（`m13_moe_layer.asc` 的 `H_LoadDataset` 里 `dataset envelope violated` 检查，原 `:2174-2177`）。

**分块策略建议**：不按 token 分块，按 `(专家, m-tile, n-tile)` 三维工作项静态切核；
每专家的 t_e 期望 80、最热 ~116 → **m-tile = 32 或 64、每专家 2-4 个 m-tile**。
N 方向：gate_up N=1280（N%128==0 ✓，NZ 权重要求），down N=2560；K：gate_up K=2560、
down K=640（MIN_K_SIZE=32 ✓、K 必须偶数 ✓，`README.md:440-443`）。
router 必须改成多核（官方 `moe_gating_top_k` 的 regbase 无分组路径 `MoeGatingTopKWithoutGroupRegbase`
是 donor，`moe_gating_top_k_apt.cpp:57`）。permute/计数排序建议**改用初始化路由 v4 的
`expandedTopkWeightOut` 协议**（`moe_init_routing_v4/README.md:9-12`，950-only），
它同时产出排序后的 token 与 topk 权重，取代 m13 现在的 `w_tk_packed` 标量协议。

---

## 4. donor 地图（逐段：文件:行号 + 可抄度评级）

评级口径：**A** = 基础 API 结构可直接抄（含 tiling/同步/寄存器写法）；**B** = 语义/公式可抄、
外壳（TQue/TPipe/Matmul 高阶）必须重写；**C** = 只能当规格/数据流参考。
（所有 donor 事实来自 `docs/09` 与本次四个并行调研的 file:line 核对。）

### 4.1 GDN（prefill）——最佳 donor：`ops-transformer/attention/chunk_gated_delta_rule`

| 子段 | donor | 行号 | 评级 |
|---|---|---|---|
| 三段式 op 骨架 / MIX_AIC_1_2 / 段划分 | `op_kernel/arch35/chunk_gated_delta_rule_arch35.h` + `..._apt.cpp` | `:177-256`（三段与 SyncAll）、apt `:30`（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`） | **A**（把 SyncAll 换成 mode-0 barrier） |
| BT=64 / p=2 / stage1 并行度 4 / mask 数 4 | `op_host/chunk_gated_delta_rule_tiling.cpp` | `:60-62,120-124` | **A** |
| stage1 分核（`totalChunk=nv*numChunk`，每核连续切片，`paraNum=4`） | `op_kernel/arch35/stage1_arch35.h` | `:214-244` | **A** |
| 32×32 前代 inverse（单寄存器 VF） | `op_kernel/arch35/vf/chunk_gated_delta_rule_stage1_vf.h` | `:87-128` | **A** |
| 下左块 2 次 mmad（`−A₂₂⁻¹A₂₁A₁₁⁻¹`） | `stage1_arch35.h` | `:710-728` | **A** |
| γ = exp(gc_i−gc_j) 掩码（Hillis-Steele 扫描 + 行循环） | `.../vf/..._stage1_vf.h` | `:27-77` | **A** |
| WY（gbk=−βe^g k → T@gbk；vβ → T@vβ） | `stage1_arch35.h` | `:503-616` | **A**（注意缓冲名 `gCumExp` 实际装 log 域 cumsum，exp 在 stage2/3 才做，`:341`） |
| 扫描 three-matmul（`CalVPrime/CalAttnInter/CalStateNew`）+ accumulator 标志 | `stage2_arch35.h` | `:197-221,261-280` | **A** |
| fp32 state 的 bf16 镜像（喂 cube） | `stage2_arch35.h` | `:159-168,203-208` | **A**（**建议照抄**） |
| 基础 API mmad 封装（L0C 64KB、L1A@0/L1B@32KB、Nd2Nz、accum→AtomicAdd） | `op_kernel/arch35/chunk_gated_delta_rule_matmul_basic.h` | `:25-36,131-144,188-194` | **A**（这是我们"基础 API 手写 cube"最直接的模板） |
| g/β 的 stride 采集（`GCopyInWithStride`/`BetaCopyInWithStride`） | `stage1_arch35.h` | `:636-666` | **A** |
| intra 输出掩码（`exp((g_i−g_j)·mask)` → ×scale → ×mask → 累加） | `stage3_arch35.h` + `vf/..._stage3_vf.h` | `:97-129`、`:26-62` | **A** |
| 五段式融合单 kernel（另一套分解，可对比取舍） | `attention/chunk_kda_fwd/`（`op_host/arch35/chunk_kda_fwd_tiling_impl.h:29-42`、`docs/ChunkKdaFwd算子设计介绍.md:16-20`） | — | **B**（含 `computeGateInPrepare`/`fusePostWu`/`denseFwdH` 等开关，可借"把 gating 融进 prepare"的思路） |
| Triton 侧 16×16→64×64 的 solve 合并 | `vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py` | `:22,203,331-402`（`LARGE_BLOCK_T = 608*2`） | **C**（Triton，只借分块思想） |
| prefill 主机侧调度语义（BT=64 / 工作集） | `vllm-ascend/vllm_ascend/ops/gdn.py` + `gdn_attn_builder.py` | `gdn.py:77,98,477-480,602-617`；`gdn_attn_builder.py:45` | **C**（host 语义） |
| conv1d prefill（环形 UB + 状态回写 + SiLU 融合） | `ops-transformer/mamba/causal_conv1d/`：`causal_conv1d_fn.h:275-325`（InitRing）、`:188-193`（WriteBackState）、`tiling.cpp:805-842`（模式推断）、`:369-373`（K∈{2,3,4}） | — | **A**（AIV-only、`__ubuf__` 裸指针、环形缓冲，与本项目风格最接近） |
| decode 递推（寄存器 VF 风格参考） | `attention/recurrent_gated_delta_rule/`（AIV-only，`..._apt.cpp:35`） | — | **A**（m4 已用） |
| l2norm 融合 | **GDN 侧没有**：`chunk_gated_delta_rule` golden 的 `use_qk_l2norm_in_kernel` 参数未实现（`chunk_gated_delta_rule_golden.py:54`），l2norm 只在 vllm-ascend csrc `npu_fused_rearrange_qkv_l2norm` / `npu_causal_conv1d_qkv` 里，且明确"只对 decode non-spec 路径有效"（`vllm_ascend/ops/gdn.py:477-480`） | — | **C**（结论：**l2norm 要我们自己融进 P2**） |

### 4.2 full attention / QSA

| 子段 | donor | 行号 | 评级 |
|---|---|---|---|
| FA prefill 主流水（D=256 ND config4：sOuter128/sInner128） | `ops-transformer/attention/flash_attn/op_kernel/flash_attn.cpp` + `_block_cube_nd.h` + `_kernel_dn.h` | `.cpp:83,92`；cube_nd `:101,314,406`；kernel_dn `:221-273,434-456` | **A**（**零 TQue/TBuf**，全 `Mutex::Lock/Unlock<PIPE_*>(BUFFER_ID)` + CrossCore，正是我们的范式） |
| decode m=1 split-K flash-decode + AIV combine | 同上 `_block_vec_flashdecode.h` | `:413-513` | **A**（M24 在用） |
| FIXP L0C→UB 双 AIV 直写（`dualDstCtl=1`） | `flash_attn_block_cube_nd.h` | `:314,406` | **A** |
| paged KV 布局契约/block table/cost model | `attention/fused_infer_attention_score/`（`fia_tiling_nonquant_gqa.cpp:145`；KV 布局三种模式见 docs/11 §2.2） | — | **B**（D=256 落 BASEAPI 兜底模板，vec 侧混 TQue → 只抄契约） |
| RMSNorm+RoPE+KV cache 写融合（attention prolog） | `posembedding/kv_rms_norm_rope_cache/`（含 `_pa` 分页变体） | — | **A** |
| partial rotary（64/256，neox + interleaved 两版） | `posembedding/inplace_partial_rotary_mul/` | — | **A** |
| **QSA 稀疏核心（token 级 gather + paged KV）** | `attention/sparse_flash_attention/`（`sparse_block_size=1`、`sparse_mode=3` rightDownCausal、`layout_kv="PA_BSND"`）；入口 `op_kernel/sparse_flash_attention.cpp`、tiling `op_host/sparse_flash_attention_tiling.cpp`、schema `torch_extension/sparse_flash_attention.py:35-45` | — | **B**（比 `block_sparse_attention`/`generic_block_sparse_attention` 更贴近——后两者吃块级 mask 而非 token 索引） |
| **QSA indexer（打分+topk，压缩比 4）** | `attention/lightning_indexer_v2/`（`cmp_ratio`、`mask_mode=3`）；`op_kernel/lightning_indexer_v2.cpp`、`op_host/lightning_indexer_v2_tiling.cpp`、`lightning_indexer_v2_metadata/`（AICPU 负载均衡） | — | **B**（`w≡1` 即 QSA 的 4 头均匀求和；Ascend 侧调用签名 `vllm_ascend/device/device_op.py:618-629`，`sparse_count=2048, sparse_mode=3`） |
| 压缩 key cache 写入 | `attention/indexer_quant_cache/` | — | **B**（本模型 bf16 用不到量化，但结构可借） |
| **golden（authoritative）** | `/workspace/vllm/tests/models/qwen4_exp/test_qsa_reference.py`（`_qsa_mqa_paged_reference:93-105`、`_qsa_relative_topk_reference:108-125`、`_expand_qsa_indices_reference:128-164`、`_qsa_select_paged_reference:167-195`、`_qsa_sparse_paged_attention_reference:198-234`）；pre-indexer golden `test_qsa_pre_indexer.py:271-305` | — | **A**（规格） |
| 公式最清楚的参考实现（算法本体） | `qwen4_exp/amd/ops/qsa.py:19-113`（逐行循环版本）、`amd/indexer_qsa.py:148-184,206-254` | — | **B**（注意 AMD 与 NVIDIA 有行为差：AMD 无末列计数、gate 在外、bf16-only） |

### 4.3 MoE

| 子段 | donor | 行号 | 评级 |
|---|---|---|---|
| router softmax+topk+renorm（regbase） | `moe/moe_gating_top_k/op_kernel/arch35/moe_gating_top_k_without_group_regbase.h`（`:36,48,55,67,78`）+ dispatch `moe_gating_top_k_apt.cpp:57` | — | **B**（数学可 lift，外壳是 TPipe） |
| 计数排序 → 排序后行号 | `moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_counting_sort_unfull_load.h`（`:35,538,577`）；full-load 变体 `moe_v3_counting_sort_full_load_unquantized.h` | — | **B**（**SIMT**（`__simt_vf__`）实现，P0 代码要重写，语义可抄） |
| permute/gather | `moe/moe_token_permute_with_routing_map/op_kernel/arch35/gather_v2_simd_two_dim.h`（`:37,148`） | — | **A/B**（行 gather 循环可抄） |
| **MXFP4 量化融进 gather**（A 侧） | `moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_gather_mxfp4_quant.h`（`vfMxfp4ComputeMaxExp:26`、`:95` scale、`:178` data） | — | **A**（纯 `__simd_vf__`，与本项目 m5 三段式同构） |
| topk 权重 permute（取代 `w_tk_packed`） | `moe/moe_init_routing_v3/.../moe_v3_topk_weight_out.h`；v4 明确协议 `moe_init_routing_v4/README.md:9-12` | — | **A**（建议采纳） |
| **GMM#1 融合 gate_up→SwiGLU→MX 量化** | `gmm/grouped_matmul_swiglu_quant_v2/`：原核 `op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_mid.h:19-20`（Matmul 高阶）、Tensor-API 路径 `op_kernel/arch35/..._mxfp4_weight_nz.h:19,32-177`（Cgmct/Blaze）、tiling 常量 `op_host/op_tiling/arch35/..._weight_quant_tiling.cpp:31-53`、Tensor-API 门限 `..._basic_api_tiling.cpp:31-36,287-310` | — | **C→B**（**全部走框架**；我们的 shape（avgM/group≈80 < 512）官方也落回 template 2 = Matmul 高阶。**唯一 mmad 级 MX 模板是 A8W4 的** `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h:55,80-84,136-220`，且无 A4W4 分支 → 要自己接 A4 的 dtype/trait） |
| GMM#2（down） | `gmm/grouped_matmul/op_kernel/arch35/...`（tensor-api mx kernel `gqmm_tensor_api_mx_kernel.h:27-89`；mmad 级同上 basic_api 文件） | — | **C→B** |
| SwiGLU + MXFP4 量化（VF） | `ops-nn/activation/swiglu_group_quant/op_kernel/arch35/swiglu_group_quant_base.h`（`VFSwiGlu`/`VFComputeMaxExpMXFP4Vf`/…）、`gmm/.../weight_quant_basic_block/gmmsq_quant_mx_vf.h:59,112,154,212,242` | — | **A**（可抄度最高的一段） |
| unpermute / finalize routing | `moe/moe_token_unpermute_with_routing_map/op_kernel/..._not_pad.h:21`；`moe/moe_finalize_routing_v2/op_kernel/moe_finalize_routing_v2_apt.cpp:15-19` + `arch35/*.h` | — | **B**（无 arch35 的 plain 版只有循环骨架可抄；v2 是 RegBase + 密集 SetFlag → 同步要重写） |
| M-split workspace 结构（"官方怎么吃下 m=40970"） | `gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_utils_kernel.h:117-142`（`WorkSpaceSplitConfig`、`VecConfig`） | — | **B** |
| 专家槽位静态分组全流程 | `mc2/mega_moe/`（docs/05 §9 已列） | — | **B** |
| 小改：graded「t_e ≤ 64」等现状 | 本仓库 `m13_moe_layer/README.md:87,192-202`、`m3_grouped_gemm/README.md:113`、`m8_permute/README.md:125`、`m11_bf16_gemm/README.md:117` | — | 现状 |

### 4.4 HC / PLE / ngram

| 子段 | donor | 行号 | 评级 |
|---|---|---|---|
| HC 全部逐元素数学（norm/silu/gate_mix/combine） | `qwen4_exp/nvidia/ops/hc.py`：`:13-78`（组内 GemmaRMSNorm）、`:82-122`（silu /4）、`:126-185`（gate 平均）、`:188-392`（combine + norm） | — | **B**（Triton → RegBase VF 重写；公式与 bf16 舍入点可逐行照抄） |
| HC 两个 skinny GEMM `[336,10240]` / `[10240,320]` | 无 Ascend donor；vLLM CUDA 侧调优参考 `qwen4_exp/nvidia/low_latency_gemm.py:73-78,131-143` | — | **自研**（= bf16 GEMM 变体；N=320/336 → 可用 m11 结构，注意 pad 12 行可折掉，N 取 324） |
| **注意**：`ops-transformer/mhc/` **不是本模型的 HC donor** | mHC（Manifold-Constrained HC，Sinkhorn 双随机矩阵）≠ 本模型的 gated-residual HC（arXiv 2409.19606）。证据：`mhc/mhc_post/op_host/op_api/mhc_post.h:25`、`mhc/mhc_pre/README.md:22-46`（`n·D → n+n+n²` = 24 输出，而我们是 320+4） | — | **C**（仅作 950 上多流残差的 kernel 形态参考；`mhc_post_regbase.h:61-68` 的 TQue 版可读，`mhc_pre_cube_compute.h:223,240,261` 是**基础 API 手写 LoadData+Mmad** 的好样板） |
| PLE ple_gate（3 次 2560 维归约 + BMW 门） | `qwen4_exp/nvidia/ops/ple.py:185-281` | — | **B**（纯向量，公式可抄；注意每步都舍入到 bf16） |
| PLE 膨胀 depthwise conv（dilation 3、K=4、10240ch） | `qwen4_exp/nvidia/ops/ple.py:318-688`（prefill 分支 `:665-688`）；状态写回 `:489-569` | — | **B**（prefill 一次性场景可丢 state） |
| ngram id 哈希（16 头 = 8 bigram + 8 trigram，XOR + 取模 + 偏移） | `qwen4_exp/nvidia/ops/ple.py:24-181`；torch 参考 `nvidia/ngram_embedding.py:244-351` | — | **B**（**注意 int64 乘加在 Ascend 向量单元上要拆 32 位**；三张元数据表 `ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}` 直接从 checkpoint 读，**不要自己实现质数搜索**） |
| ngram 表本身 95.37GiB | 磁盘驻留 + 行 gather + 跨层预取编排（`nvidia/model.py:516-534`、`ple_layer.py:154-167`） | — | **自研**（唯一必须新设计的段） |
| 层边界 / 权重打包 glue | `_EXTRA_WEIGHTS_MAPPER`（`nvidia/model.py:94-156`）、PLE 128 分片→行映射（`nvidia/ngram_embedding.py:385-448`） | — | **C**（host glue，必须重实现） |

---

## 5. 关键裁决：decode 与 prefill 是否共用同一个 per-layer kernel

### 5.1 三个方案

- **方案 A（单入口 + 段内按 m 分支）**：一个 `m15_gdn_layer_kernel`，S1–S7 每段内部写
  `if (m == 1) {现 decode 实现} else {prefill 实现}`。
- **方案 B（推荐：单 TU、两个入口符号、一套静态资源表）**：同一个头文件/编译单元里放
  `gdn_layer_decode_kernel()` 与 `gdn_layer_prefill_kernel()`（各自把段序 `__noinline__` 成独立函数），
  共用一份编译期资源表（BufferID/flagId/UB 窗/L1 区/GM 偏移）与共用的无分支算件（norm、bf16 GEMM、
  RMSNormGated、MX 量化），host 按 m 选符号。**kernel 内无任何按 m 的运行时分叉。**
- **方案 C（两套独立 kernel，各自 TU 与资源表）**：prefill 另起一个工程文件，decode 保持 m14/m15 原样，
  host 侧分派。

### 5.2 五个维度对比

| 维度 | A（段内分支） | **B（双入口+单资源表）** | C（两套 kernel） |
|---|---|---|---|
| **UB/L1 占用** | ❌ 最差。UB 是编译期静态分配：decode 的 h slab（64KB）+ prefill 的 A/T/γ/chunk 缓冲必须**同时满足**，而 m14 现布局已用到 227KB/248KB（余 26KB，`m14_gdn_layer/README.md:41`）→ 只剩"同址叠放"一条路，而叠放恰是 m13 实测的两条坑（同 UB 区被不同 BufferID 保护会失效；同 pipe drain 后立刻 acquire 自锁，`m14_gdn_layer/README.md:45-48`）。L1 同理（decode 的 A/B ping-pong 与 prefill 的 h 镜像+K 分块窗要共存，512KB 会超） | ✅ 两条路径**互斥执行**，只需两份互斥的窗口布局各自 ≤248KB 并断言（无需叠放、无需求和）。L1 同理：按 mode 选一整套 A/B/h 分区 | ✅ 同 B（每套自带布局） |
| **icache**（AIV 16KB 贴边，docs/12 §7） | ❌ 两个分支交织进同一热函数，取指体量叠加；编译器未必 outline | ✅ 每条路径各自成函数，一次 launch 只取一套；TU 变大不影响取指局部性 | ✅ 同 B |
| **同步表复杂度** | ❌ 最差。两种形态的同步图**拓扑不同**：decode 是"段边界 4 个跨核点"（m14 表 4-15 全占），prefill 是"chunk 循环内每 chunk 的 AIV↔AIC + 段边界"。一张表表达两种拓扑不可行；且"相邻同步点 flagId 必须不同"要跨分支成立 → 需全表重规划 | ✅ flagId 窗口**按 mode 分区**（互斥执行 ⇒ id 可跨 mode 复用），头文件里分两节登记，各自满足"相邻不同 id" | ✅ 同 B，但两份表在不同文件 → 容易漂移（"每 id ≤15 次"这类全局计数要人工对齐） |
| **后续调试成本** | ❌ 最高。UB 叠放与跨分支 flagId 是长期风险源；一个 mode 的改动可能破坏另一个 mode 的叠放假设 | ⚠️ 中。两条路径可独立验收（decode 保持 m15 Ver A/B/C 基线；prefill 用 m=4097 golden 对食），但改共用算件要同时过两套判据——这是**收益**（回归立刻暴露） | ✅ 最低（互不影响），代价见下 |
| **与 mix(1,2) 全核用满的契合度** | ❌ 最差。同一段序里 AIC 与 AIV 的两种负载形态都活，无法把空闲核让给别的段（decode 时 AIV 实测只占 VEC 1.1%/MTE2 3.2%，`wt-25/.../README.md:137-139`；prefill 时 AIV 又要跑 l2norm/gating/cumsum/inverse/掩码） | ✅ 每条路径可独立把空闲核重分配（decode 把 AIV 让给权重预取；prefill 把 AIC 让给 chunk 组流水） | ✅ 同 B |
| **维护面/漂移面**（附加维度） | ✅ 单份代码 | ⚠️ 共用算件单份 + 两条段序两份 | ❌ decode 的 5 个算件 + 完整段序要被复制成两份；仓库已有教训：m15 明确记录"复制改造的代价是上游 m14 变更不会自动流入"（`m15_layer_loop/README.md:219-221`），m13/m14 也同款。再加一份整层 kernel 会让同步面翻倍 |

**裁决：方案 B。** 附加硬约束（写进资源表头文件）：
1. mode 由 **host 选入口符号**决定，**不允许**在 kernel 内由运行时 m 推导（tiling/布局是编译期的）；
2. 资源表按 mode 分节，每节必须自带 `static_assert(footprint ≤ 248KB)`（UB）与 L1 断言；
3. 文档写明"**flagId/BufferID 命名空间按 mode 划分、两 mode 互斥执行**"——这是 id 可复用的前提，
   也是 B 相对 C 唯一需要额外小心的地方。

### 5.3 【为了不返工】decode 侧现在就必须遵守的接口约束清单

逐条可勾选。标注 **★** 的是"今天不改、prefill 期必然重写"的高危项。

- [ ] **★1** GEMM item 从 `RunTile(nBlock)` 升级为 `RunTile(mTile, nBlock)` 二维工作项：L0C 累加与
      Fixpipe 都按 **m-tile** 收尾；`calcM = max(curM,2)` 的 quirk 保留但对每个 m-tile 生效
      （现状见 `m14_gdn_layer/README.md` 的 GEMM item 说明与 `m13_moe_layer.asc` 的 `MXFP4GemmItem::Run`；**以符号为准**，M33 写作时的行号口径是 `:104`、`:161-165`，行号随代码变动）。
- [ ] **2** 所有 GEMM 的 A/B 操作数统一以 `(M,K)×(K,N)` 描述并**允许 K 尾块**；不要再引入新的
      "N 必须整除 16480" 之类约束（K=640/2560/6144 都是 64 的整数倍，prefill 不需要 K 尾块，
      但需要 N/K 两侧都能被循环而不是被静态断言锁死）。
- [ ] **★3** MoE 的"每专家固定行槽 `NUM_EXPERTS * M_MAX`"布局改为 **Σt_e 紧凑布局**，
      并冻结 `group_list` 用 **count 模式 `int64[512]`** 的契约（官方 `moe_init_routing_v2`
      `expert_tokens_num_type=1`）。否则 t_e>64 直接越界写下一槽（`m13_resources.h` 的槽尺寸常量 `SZ_AQ/SZ_GU/SZ_Y`，原 `:237-249`）。
- [ ] **★4** 路由数组 `TOTAL_MAX = M_MAX*TOPK_MAX = 256` 定长 → 改为按 `active_num = m*topk` 分配；
      S3 计数排序从"单核标量"改成多核（prefill 有 40970 槽位，单核 94ms 不可接受，
      见 §3.3 第 5 条）；建议直接采纳 `moe_init_routing_v4` 的 `expandedTopkWeightOut` 协议
      （`moe_init_routing_v4/README.md:9-12`），取代 `w_tk_packed` 标量索引读。
- [ ] **★5** kernel 签名保留 `m`（token 数）与 `topk` 作为**运行期参数**，不要像 m15 那样把它们
      退化成常量（`m15_layer_loop/README.md:12`）——prefill 的 tile 数、每专家 token 数全靠它。
- [ ] **6** `ws` workspace 一律用 host 指针 + **编译期偏移表**，偏移表按 prefill 最大规模定尺寸
      （现 5.26MB/层，prefill 需要 ~100MB 级），不要把"只对 m=1 成立"的常量写死。
- [ ] **7** UB 布局写成**窗口表 + 每 mode 变体**，每个 mode 独立断言峰值 ≤248KB（不是求和），
      并保留 32B 对齐与 SIMT 40KB 预留的判断（docs/05 §6.1）。
- [ ] **8** flagId / BufferID 命名空间**按 mode 分节登记**，头文件显式声明"两 mode 互斥、id 可复用"；
      跨核同步的 pipe 类继续遵守 AIC∈{S,MTE1,MTE2,FIX,M} / AIV∈{S,V,MTE2,MTE3}（docs/05 §6.2）。
- [ ] **9** `GetBlockNum() * 2`（mix(1,2) 下 AIV 视角拿到的是 AIC 数）的修正在两条路径都要保留
      （`m14_gdn_layer/README.md:109`）。
- [ ] **10** GDN 的 **g/β stride-8 槽位契约**与 `{H,1,0,0}` 直读方式冻结；prefill 按 chunk 读 `[C,48]`
      时**沿用同一布局**（不要为 prefill 另立一套 → 否则 S3/S4 的产出/消费要对两份契约各写一遍）。
- [ ] **11** `conv_state` / `ssm_state` 的 GM 布局、in-place RMW 语义、"进入下一步之前的状态"定义不变
      ——prefill 跑完写入的 state 必须能被 decode 直接续用（同一个 state，不是两套）。
- [ ] **★12** 层边界（残差出口/入口）合约先与 M31 的 HC 规格统一再冻结；停止使用 m14/m15 的
      "层输出 = 归一化值 + 合成 gamma1/gamma2 + 全零 conv_bias" 占位
      （`m15_layer_loop/README.md:172-173,225-232`），否则 prefill 会把错的边界再固化一层。
- [ ] **★13** attention 侧：packed indices 采用 **NVIDIA 契约 `[m,2052] int32`（末列=有效数）**；
      **KV 写必须早于核心 gather**；主 KV（`[blocks,2,block_size,256]`，block_size=16）/raw key ring/
      compressed key cache 三套布局现在就定死（即使 attention 仍是占位），因为 prefill 与 decode
      必须写同一份 cache。
- [ ] **14** 所有新增的跨核交接都在头文件登记 flagId 与 pipe，禁止使用 `PIPE_ALL`；同 pipe 背靠背复用
      一律 `PipeBarrier<PIPE_X>`（docs/05 §6.2）。

---

## 6. 资源预算（m=4097）

> 口径声明：以下 FLOPs/bytes 都是**按 config 形状与官方数据流手算的上界**（无实测 kernel 可依），
> 时间下限用 §1 的实测峰值（BF16 368 TFLOPS、MXFP4 按 4× = 1.47 PFLOPS；若 368 TFLOPS 是满配
> 36 子系统测得，则本机 28 AIC 应乘 28/36 = 0.78 → BF16 ≈ 286 TFLOPS，估算再乘 1.28）。
> **这些是 roofline 下限，不是预期性能**；预期性能按经验的 40-70% 效率折算（§6.5）。

### 6.1 每段的 MAC 与搬运字节（每层）

| 段（每层） | MAC（G） | 权重 MB | 激活 MB（含中间量往返） | 关键张量 |
|---|---|---|---|---|
| GDN in_proj | 172.8 | 84.4 | 156（读 x_norm 21 + 写 qkvzba 135） | qkvzba [4097,16480] |
| GDN chunk core（含 in-proj 之外的 mmad） | 15.6 | — | 135 读 + 135~200 写 | q'/kg/k_cumdecay/u/v_new/o |
| GDN out_proj | 64.5 | 31.5 | 71（读 opin 50 + 写 opout 21） | opin [4097,6144] |
| GDN 其余 VEC 段（conv/l2norm/gating/cumsum/RMSNormGated） | — | — | ~500 | — |
| attention 投影（qkv+gate / o / index_qk） | 210.8 | 100.3 | ~250 | qkv [4097,13312] |
| attention indexer（打分+topk+缓存写） | 2.2 | 3.3 | ~15（compressed cache 0.25 + q_idx 4.2） | packed [4097,2052] i32 = 33.6MB |
| attention 核心（稠密 causal） | 103.1 | — | ~60（Q 常驻 + KV 页读写） | — |
| MoE 路由 | 5.4 | 2.6 | 22 | logits [4097,512] fp32 |
| MoE permute+MX 量化 | — | — | 262（读 x 21 + 写 A_qx 52.4 + scale 3.3 + 读回…） | sorted [40970,2560] |
| MoE GMM#1（gate_up+SwiGLU+量化） | 67.1(MX) | 839+39 | 52.4+3.3 读 / 13.1+0.8 写 | H_qx [40970,320] |
| MoE GMM#2（down） | 67.1(MX) | 419+39 | 13.1 读 / 209.8 写 | Y [40970,2560] bf16 |
| MoE unpermute+combine+shared | 13.4 | 2.6 | 210 读 + 42 写 | routed [4097,2560] |
| HC ×2 模块 | 55.0 | 26.9 | **融合后 ~336**（未融合 ~1340，见下） | 残差流 [4097,10240] = 84MB |
| LM head | 2605.0 | 1270 | 2064（logits 2.03GB） | logits [4097,248320] |

**HC 的搬运量是全 pass 的头号风险**：天真实现（每次 norm/silu/gate_mix 都从 GM 读一遍 84MB 的
残差流）→ 每模块约 6 次读写 84MB ≈ **670MB/模块 → 每层 ~1.34GB → 全 pass ≈ 64GB**，
与 MoE 权重全量同级。把 HC 按 **m-tile ≤ 8-12 行**处理（8 行 × 10240 × 2B = 164KB < 248KB UB）→
整段驻 UB，中间量不再往返 GM：每层只读两次残差流 168MB + 写 168MB + 权重 26.9MB ≈ **363MB/层
→ 17.4GB/pass（↓3.7×）**。这是 prefill 设计里第一个必须落实的"小 m-tile"约束，
也直接决定 M36 的 tiling。

### 6.2 片上占用（单个 GDN prefill chunk / 单个 HC m-tile）

| 资源 | 预算 | 依据 |
|---|---|---|
| UB / AIV | h 权威 fp32 slab [128,128] = **64KB**（1 value 头/AIV）+ h 的 bf16 镜像 staging 32KB + q/k/v chunk 段（64×128×4 = 32KB ×2 乒乓）+ A/T 16KB + γ 16KB + 归约/掩码 scratch ≈ **200KB / 248KB** | 官方 stage2 fp32 路径的 `inQueue = max(C,Dv)·Dk·4 = 64KB`、`tmpBuff ≈ 64KB`（`stage2_arch35.h:81-111`），与本预算同量级 |
| L1 / AIC | 扫描用的 h bf16 镜像 2 头 ×32KB = 64KB + K 分块后的 A 窗（64×320×2=40KB ×2 乒乓）+ B 权重窗（160×320×2=100KB ×2）≈ **344KB / 512KB**；剩余留权重预取滚动窗 | donor 的 L1 分配：`L1A@0`、`L1B@32KB`（`chunk_gated_delta_rule_matmul_basic.h:25-36`）；docs/12 §4 已预留 @330KB 起的预取窗 |
| L0A/L0B | 各 64KB（+4KB MX）；K 分块 320/128 → 一次装 128×128 与 128×128 fractal | donor `matmul_basic.h` |
| L0C | **64KB/tile**（16384 fp32）×2-4 乒乓 ≤ 256KB；`u`/`o_inter` 的累加都在 L0C 内完成，不落 GM | donor 固定 64KB（`matmul_basic.h:25-36`）的保守取法；L0C 有 256KB，prefill 可放大到 128×256 |
| GM（每层中间量，可驻 L2 更好） | in_proj 之后的 qkvzba 135MB + chunk 中间量 ~200MB（若按 §2.4 的 h 常驻方案，官方那 421MiB 可压到 ~250MB）| 官方同 shape 的 workspace 实测 **≈421MiB**（`tiling.cpp:133-156` 的逐项公式），其中 `kCumDecay 54 / vInner fp32 108 + bf16 54 / qPrime 54 / attnInter 54 / kg 54 / qkt 27 / stage1 temp 11.25 MB` |
| 常驻 state | ssm_state 3.15MB/层、conv_state 60KB/层（36 层 = 115.5MB） | 与 decode 同一份（`m15_loop_layout.h` 已按 3.21MB/层分配） |

> **⚠ 更正（2026-09-27，M120；针对上表 L1 行里「A 窗」的搬运前提）**
>
> 上表 L1 行的 **A 窗（`64×320×2 = 40KB ×2` 乒乓）** 原被读作「AIV 把 A/Γ 从 UB 直接写进这一窗」
> （`docs/19` §4.4 的「A / Γ」行就是按这个接的）。M115/B1 的 finding 曾把该前提判成**不成立** —— 依据是
> 四条基础 API 文档里那句逐字 NOTE「本接口为软件仿真实现，是在Matmul高阶API的基础上……需要先使用
> `REGISTER_MATMUL` 注册高阶API」。
>
> **更正**：那句 NOTE 只覆盖 **GM 中转那一条分支**。同一批官方文档另有一节
> 「新增UB到L1 Buffer搬运数据通路」，逐字写「**CV融合算子中，UB向L1 Buffer搬运数据不再需要通过GM中转**」，
> 并给出两条**不需要 Matmul 注册**的形态：**C API `asc_copy_ub2l1`**（「无需配置编译选项」；另见
> 「C API直接使用对应硬件接口，不能套用基础API兼容路径的注册要求」）与**基础 API + 编译选项
> `-DENABLE_CV_COMM_VIA_SSBUF=true`**。本机 CANN 9.1.0 的头文件里这两条都在，且一条只含目标 API 的最小探针
> **编得过**（host 侧，零设备；逐字引文、`文件:行` 与读数见 `docs/evidence/ub_to_l1/README.md` §1.1/§5）。
>
> **对本节的影响（如实划界，不改容量口径）**：
> - **容量数字不动**：A 窗 40KB×2 与 L1 合计 ≈344KB/512KB 是**容量**口径，不随搬运通路变；
> - **搬运代价按所选通路记**：走 **GM 中介** ⇒ 多一块 GM scratch，且操作数**多一次 GM 往返**（§6.1 的搬运字节
>   与 §6.3 的带宽线要按此上修，Wave C 的峰值分节据此选口径）；走**硬通道** ⇒ 无 GM 往返，但要 AIV/AIC 成对同步；
> - （M120 原写法，记作历史）~~**选哪条 = 未确定**：本 mission **零设备档**，真机读数一条都没有 ⇒ 候选清单、
>   各自代价与「需要什么实验」见 `docs/evidence/ub_to_l1/README.md` §2/§3。**这一前提不是"已被否"，
>   而是"有分支、待一次真机实验定"**~~ ⇒ **已由 M135 按 M121 真机读数更正为"硬通道可用"，见下「M135 更正」块**；
> - B1（M115）现行实现走的是 GM 中介（分支 `feat/m115-b1-gdn-prefill-chunk-scan`，commit `8ca32c5`/`1f7b630`；
>   该分支未合入 main），其报告自述 GM 往返偏高即源于此。
> - （M120 原写法，记作历史）~~**真机裁决在哪 = M121**：塔已另立 **M121**（UB→L1 硬通道最小真机实验；分支
>   `feat/m121-ub2l1-hard-channel-real-machine-pro`，wt-121）⇒ 本节的「选哪条」**以其读数为准复核**，
>   在此之前两条都只是候选（本 mission 零设备，未替它预判）~~ ⇒ **M121 已合入 main，裁决与读数见下**。
>
> **⚠ M135 更正（2026-10-04；把上面两条"待一次真机实验定 / 以其读数为准复核"的占位换成已裁决结论）**
>
> 触发：**M121**（UB→L1 硬通道最小真机实验，已合入 main；交付物 `probe_ub2l1/`）在 950 真机上把
> 「AIV 在 UB 里算出的操作数直接喂同核组 AIC 的 mmad」实测了一遍。本节「选哪条」按它的读数**改为**：
>
> - **硬通道可用**（限本机 CANN 9.1.0 + `__mix__(1,2)` + 本探针形态）：M121 读数逐字 ——
>   「C API `asc_copy_ub2l1` 在本机 9.1.0 真机可用」（`probe_ub2l1/README.md:21`）、
>   「基础 API `DataCopy(L1, UB)` 在本机 9.1.0 真机可用」（`probe_ub2l1/README.md:22`，带宏与不带宏读数
>   逐位相同）；往返档 `same=8192 diff=0`；AIC 把它当 mmad 操作数真做了一次收缩，误差分桶
>   「[0e+00,1e-06)=1536」（`probe_ub2l1/README.md:174`，另见同表 `:172-173`）。
> - **带宽（每 AIV 核，设备侧 SYS_CNT 计时）**：64KB/次 → C API「230.75」/ 基础 API「231.24」GB/s
>   （`probe_ub2l1/README.md:256-258`，汇总行 `probe_ub2l1/README.md:27`）。
> - **同步**：AIC 消费 AIV 刚写进 L1 的数据要用 **mode 2 成对会合**；AIV 侧那次会合不能省 —— M121 逐字
>   「去掉 AIV 那一侧会合，AIC 直接挂死（rc=124，两支都复现）」（`probe_ub2l1/README.md:10`，
>   设备侧见证 `probe_ub2l1/README.md:296-304`）。⇒ 上面
>   M120 块里「走硬通道 ⇒ 要 AIV/AIC 成对同步」这一条**得到真机见证**；「走 GM 中介 ⇒ 多一次 GM 往返」
>   仍成立（B1 走的就是这条），但**不再是本形态下被真机证实过的那条**。
> - **ND→NZ 仍须在 UB 内自己做**（官方口径「硬件本身不支持该能力」），与上面第 ③ 条候选一致
>   （`probe_ub2l1/README.md:26`）。
>
> **限度（不得读成比读数更宽的结论）**：以上只覆盖 **本机 CANN 9.1.0 + `__mix__(1,2)` + 本探针形态**
> （AIV 产出 A/Γ、AIC 当 mmad 操作数消费、64KB/次档）。M121 自己声明**不主张**「官方样例可在 9.1.0
> 直接跑通」（官方样例声明 ≥9.2.0，本机 9.1.0 整编不通过；`probe_ub2l1/README.md:31`），且对
> 「换别的 mix 配比 / 别的工具链版本会怎样」**未覆盖、不外推**（`probe_ub2l1/README.md:130`）；
> 小块不要按 231 GB/s 外推（`probe_ub2l1/README.md:270-272`）。本文件据此**只**改
> 「本仓 `__mix__(1,2)` 形态下硬通道可用」这一层，**不**把 L1 容量口径（A 窗 40KB×2）或 GM 中介那条候选
> 写成失效。
>
> **基准声明（不混基准）**：本块新增的 `probe_ub2l1/**` 行号取该目录的**入库版本**（M121 合入 main 的交付物，
> 晚于本文件顶部状态横幅声明的正文基准 `20bd20d`）。本文件声明的 `20bd20d` 基准约束的是正文原有的
> `文件:行`；本块的行号**另注基准**，两者不互相换算。

### 6.3 每层时间下限（算力 vs 带宽两条线）

| 段 | 算力下限（每层） | 带宽下限（每层，权重+激活 @1.6TB/s） | **谁支配** | 层数 | 小计 |
|---|---|---|---|---|---|
| GDN | 505.9 GFLOP / 368T = **1.37ms** | (115.9+~1000)MB = 0.70ms | **算力**（in_proj AI≈1437 FLOP/B ≫ ridge 248） | 36 | **49.3ms** |
| attention | 632 GFLOP / 368T = **1.72ms** | (103+~330)MB = 0.27ms | **算力**（核心 AI≈6000 FLOP/B） | 12 | **20.6ms** |
| MoE | 306 GFLOP(MX) / 1.47P = **0.21ms** | (1342+700)MB = **1.28ms** | **带宽**（AI≈150 FLOP/B ≪ MXFP4 ridge 919） | 48 | **61.4ms** |
| HC（2 模块/层） | 110 GFLOP / 368T = **0.30ms** | (26.9+336)MB = 0.23ms | 算力（略，AI≈550 FLOP/B，融合后） | 48 | **14.4ms** |
| LM head | 5210 GFLOP / 368T = **14.2ms** | (1270+2064)MB = 2.08ms | **算力** | 1 | **14.2ms** |

**全 pass（4097 token）下限汇总**：49.3 + 20.6 + **61.4（MoE，带宽支配）** + 14.4 + 14.2 ≈ **160ms**。
口径：取"每层两条线的较大者再按层数相加"，等价于**假设段间与层间零重叠**（per-layer kernel +
段边界 barrier 的形态下这是正确的下限）。若未来做整网级流水（算力与权重搬运跨段重叠），
理论下限可降到 ~100ms，但那超出 per-layer kernel 形态。另注：MoE 的 61.4ms 是**权重+激活总字节
÷ 名义 1.6TB/s**，而 m15 实测 AIC MTE2 只跑到 0.86TB/s（55%，`wt-25/.../README.md:139-141`）
→ 仅这一项就可能上修到 ~110ms，把全 pass 下限推到 ~210ms。

**带宽↔算力的切换点（mission 的问题）**：
- **MoE 是唯一不切换的**：每专家 t_e ≤ 116 槽时 AI≈150 FLOP/B，远低于 MXFP4 ridge 919；
  要变成算力支配需要 **t_e ≈ 400+**，即 `m ≈ 400×512/10 ≈ 20480 token`（或 topk/专家数变化）。
  → 在 m=4097 下 MoE 永远是权重带宽题（**这也解释了 4097 与 20480 的 MoE 段策略必须分开设计**）。
- **其余全部从带宽支配翻转为算力支配**：GDN in_proj 在 decode 是纯权重流（84.4MB / 单 token，
  AI≈6 FLOP/B），prefill 变成 AI≈1437；attention prolog、HC、LM head 同理。
- **GDN 的 chunk 扫描段是纯算力+串行**（每层 15.6 GMAC，但只有 48 头 × 65 chunk 的并行度，
  串行步 130/核）→ 它是 prefill 里**唯一"串行临界路径"段**，也可能是实际支配项（§3.1 的
  130-260µs/层估算 ≈ 全 pass 5-9ms，相对 GDN 的 1.37ms/层下限是 10-20%，量级不可忽略 → 必须实测）。

### 6.4 转换点的 AI 表（供后续优化判定用）

| 段 | AI（FLOP/B，prefill） | 对应 ridge | 结论 |
|---|---|---|---|
| GDN in_proj/out_proj | ~1400 | bf16 248 | 算力支配 |
| GDN chunk core | ~>2000（本地复用高） | bf16 248 | 算力 + 串行 |
| attention 投影 | ~1450 | bf16 248 | 算力支配 |
| attention 核心 | ~6000 | bf16 248 | 算力支配 |
| MoE（routed+shared） | **~150** | MXFP4 919 | **带宽支配** |
| HC（融合后） | ~550 | bf16 248 | 算力支配 |
| LM head | ~1570 | bf16 248 | 算力支配 |
| ngram gather | 0（纯 gather） | — | 延迟/随机带宽（95.37GiB 表） |

### 6.5 预期与对照

- 下限 **160ms / 4097 token ≈ 25.6k tok/s**（理想 100% 峰值）；把 MoE 按实测 55% 带宽效率折算
  （61.4→110ms）后下限 ≈ **210ms ≈ 19.5k tok/s**；再按整体 40-70% 效率折算 → **300-530ms
  ≈ 7.7-13.6k tok/s**。
- 对照 decode 实测：5.115ms/token = 196 tok/s（`wt-25/.../README.md:133`）→ prefill 的吞吐优势
  应有 **~40-70×**；若实测低于 20×，说明某个段（最可能是 MoE 权重带宽或 GDN 串行扫描）严重失速。
- 对照 m15 的实测带宽效率：AIC MTE2 等效 **0.86 TB/s = 名义 1.6TB/s 的 55%**
  （`wt-25/.../README.md:139-141`）——prefill 的 MoE 段与 decode 的 GDN 段是**同一个瓶颈**
  （HBM→L1 的权重流），所以"提升 MTE2 效率"在 prefill 上的收益与 decode 同源、可复用同一条优化线。

---

## 7. 风险与依赖清单

### 7.1 必须等上游落地（否则会返工）

> **合并状态对账（2026-09-26，M70）**：下表的状态声明已**逐行**按 git 核对（`git merge-base --is-ancestor <分支 tip> main` + `git log -1 <merge>`）：M24/M25/M27/M29 均已合入（各引**真实合入 commit**），M31 已放弃；第 6 行（M30/M37）本就没有状态声明、按原样保留（两者亦均在 main 上：M30 随 `9287912`、M37 分支 tip `7a0af64` 是 main 的祖先）。触发：finding `20260926-agent-m27backfill-bug-docs-15-prefill-design-md-7-1-m27-active-m66.md`（M27 行）＋ M67 复审裁定 (A)（同表其余行）。

| 依赖 | 等什么 | 影响我们的哪一段 | 并行替代方案 |
|---|---|---|---|
| **M24** attention decode FA core（**已随 `b680b0e` 合入 main**） | Q/K/V 的 L1 装载结构、CV 交接（FIXP→UB 双 AIV、P→L1 回流）、dense causal core 的 config 选择 | §2.5 的 attention 核心（12 层） | prep（qkv proj / norm+rope / KV 写）可先做，核心等 M24 |
| **M27** L0 装载几何标定（**已随 `eeab864` 合入 main**，分支 `feat/l0-load-geometry-calibration` 的 tip = `cdbfa20`） | `LoadL0_2D/3D/MX` 的偏移与 stride 真实语义（docs/05 §6.1 的"参数静默不生效"通用判据） | GDN chunk 段的 L1→L0 分块窗（§6.2）与 MoE 的 MX scale 装载 | 可以先按 m11 已验证的 (BASE_M/K/N) 组合做，不做新几何 |
| **M31** HC/PLE/ngram 规格 `docs/14`（M31 本身**已放弃**：分支 `feat/qwen3-8-hyperconnection-ple-indexer-spec` **未合入 main**；规格内容由 M38 落库，随 `ac762d8` 合入 main） | HC 的 `336 pad` 取舍、层边界合约、PLE 状态语义、ngram 表驻留 | §2.3、§6.1 的 HC 融合约束；§5.3 清单第 12 条 | 本文件 §2.3 是独立核对结果，可交叉验证；候选上限（m-tile ≤8-12 驻 UB）已可直接用 |
| **M25** 48 层循环（**已随 `2b9cb36` 合入 main**） | 层间残差流的 m 行分配（现在 `h[2]` 只分配 M_MAX 行、只用行 0）、层边界值定义 | 整网级 prefill 流水 | 单层 kernel 的 prefill 化不受阻 |
| **M29** MoE 真实 MXFP4 权重（**已随 `95ff3f5` 合入 main**） | 权重 1.34GB/层的装载与预取路径（host 逐层 H2D 不现实） | §2.6 的 GMM 段 | permute/路由/量化可先做（不需要权重） |
| **M30** 基线栈 + **M37** vllm-ascend 接入方案 | 端到端 reference（prefill 的整网 golden） | 验收判据 | 逐段 numpy/torch golden 可先建（M26/M32 在管） |

### 7.2 可以立刻并行做的（无依赖）

GDN prefill chunk 扫描（**M34**，本文件 §2.4/§3.1 已给足：BT=64、段序 P0–P11、h 存 UB fp32 + bf16 镜像进 L1、
donor 的 `matmul_basic.h` mmad 封装与 32×32 前代）、QSA indexer 前端（**M35**，§2.5 的 cache 填充部分）、
HC mixer（**M36**，§2.3 + §6.1 的 m-tile 约束）、PLE/ngram（独立，但表驻留要系统侧配合）、
MoE 的路由多核化与 permute 多核化（§3.3 的六条改造）。

### 7.3 风险登记

| # | 风险 | 量级 | 缓解 |
|---|---|---|---|
| R1 | **GDN 串行扫描临界路径**：48 头 / 28 AIC ≈ 2 头/核，65 chunk×2 = 130 个串行步；若每步 >3µs → >400µs/层（全 pass >14ms） | 高（是 prefill 里唯一的串行瓶颈） | h 常驻片上（不往返 GM）；chunk 组流水（stage1 of c+1 与 stage2 of c 并行）；必要时回测 BT=128（33 步）；先做**每步开销标定**实验 |
| R2 | **MoE 权重带宽**：m=4097 下每专家仅 ~80 槽，AI≈150 FLOP/B ≪ MXFP4 ridge；61ms 下限 ×(1/0.55 实测效率) ≈ 110ms = 全 pass 最大项 | 高 | 提升 MTE2 效率（与 decode 同一个收益点）；增大 batch（t_e→400+ 才翻转为算力支配）；per-expert 权重 L2 驻留策略（128MB L2 vs 1.34GB 权重 → 只能小部分驻留） |
| R3 | **HC 激活搬运**：天真实现 ~64GB/pass（与 MoE 权重同级） | 高 | 强制 m-tile ≤8-12 行（8×10240×2B=164KB<248KB UB），整段驻 UB；写进 M36 的契约 |
| R4 | **ngram 表 95.37GiB 无法驻显存**（128GB HBM 中权重已占 ~170GB 级别的全量） | 中（只影响层 1） | 磁盘/mmap 驻留 + 行 gather + 跨层预取（官方 `cpu_offload` 路径即此形态）；docs/01:173 已记 |
| R5 | **lm_head logits [4097,248320] bf16 = 2.03GB** 单缓冲 | 中 | 按 m-tile 分块出 logits；或裁决"只对最后 k 个 token 出"（需与集成侧对齐） |
| R6 | 方案 B 的 flagId/UB 按 mode 复用前提（互斥执行）被误用 | 中 | 写进资源表注释 + review 检查项；为每条路径加独立的 `static_assert` |
| R7 | **数值口径**：GDN 的 ssm_state 位级不可复现（m14 已裁 1e-5 相对，最差占用 0.332，`m14_gdn_layer/README.md:185-189`）；prefill 的 h 是更长 fp32 累加链 + 逐 chunk 衰减 → 误差累积更难界定 | 中高（可能是 prefill 验收的最大争议） | 在 M34 阶段就定口径：逐 chunk 与 fp32 参考对食 + 全链 1e-5 相对；必要时对 h 引入 fp64 host oracle 逐 chunk 锚定 |
| R8 | **尾块/边界**：m=4097 = 64×64+1（尾块 1 行）；且 vLLM 的 prefill 是 **chunked**（一个 chunk 可以任意长、可带初态，`qwen_gdn_linear_attn.py:1509-1521` 传 `prefill_query_start_loc`/`prefill_has_initial_state`/`chunk_indices`）→ 我们的 kernel 必须接受"任意 m + 初态" | 中 | 契约里写死 `m ∈ [1, M_MAX_CHUNK]`、tail 用有效长度掩码（官方 `validLenBatch_` 做法）；不要把 m=4097 当特例 |
| R9 | **QSA 稠密替代与 golden 不等价**（见 §9 存疑 3） | **高（阻塞验收口径）** | 需 tower 裁决：m=4097 的 golden 是官方 QSA 输出（→ 必须做 indexer 打分/topk）还是自建稠密 causal 参考 |

---

## 8. vllm-ascend 接入点（tower 方向指令要求）

> 方向：最终形态是**我们的 kernel 作为 vllm-ascend 自定义算子**，被 vLLM ≥0.29 的 `qwen4_exp`
> 模型实现调用；算子边界尽量对应 `qwen4_exp` 里的一个模块级算子。

### 8.1 现有注册机制（接入的落点）

- **Python 侧**：`vllm_ascend/ops/register_custom_ops.py` 的
  `direct_register_custom_op(op_name=..., op_func=..., fake_impl=..., mutates_args=[...],
  dispatch_key="PrivateUse1")`（该文件 :222-298 已注册 `maybe_chunk_residual`/`quantize`/`muls_add` 等）。
- **C++ 侧**：`csrc/` 下的算子经 `torch.ops._C_ascend.<name>` 暴露（现有例：
  `npu_fused_qkvzba_split_gating`、`npu_causal_conv1d_custom`、`npu_recurrent_gated_delta_rule`、
  `npu_grouped_matmul_swiglu_quant_v2`）。**我们的 kernel 走 C++ 侧更适合**（`<<<>>>` 直调 +
  host tiling），Python 侧只留一个薄 wrapper。

### 8.2 逐段的接入位置（谁是调用方 / 入出参 / 有无现成槽位）

| 我们的段 | vllm-ascend 里的现成槽位 | 建议的 op 边界 | 备注 |
|---|---|---|---|
| GDN prefill | `vllm_ascend/ops/gdn.py` 的 `num_prefills > 0` 分支（`:613-621`）→ 现在调 `chunk_gated_delta_rule(..., use_qk_l2norm_in_kernel=True)`，其内部是 Triton 六步链 + 两个 AscendC 算子（`ops/triton/fla/chunk.py:60-134`） | **先按"prefill chunk op"粒度**：一个 op 顶掉 chunk.py 的六步（cumsum/kkt/solve_tril/w·u/fwd_h/fwd_o）；验收后再考虑吞并 `npu_causal_conv1d_custom` 与 `npu_rms_norm_gated` | ✅ 槽位现成；入参需含 `q,k,v,g,beta,initial_state,cu_seqlens/lens`（注意 vllm-ascend 侧 state 是 `[N,H,K,V]`，`gdn.py:602` 做了 `transpose(-1,-2)`） |
| GDN decode | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule`（`gdn.py:542,571,662`） | 整层替换（我们已有的 m4/m14 路径） | ✅ 槽位现成 |
| QSA indexer | vllm-ascend **无** qwen4_exp 的 QSA 实现（`qwen4_exp` 本身还不被支持） | op 1 = pre-indexer（q norm+rope、raw ring 写、compressed cache 写）；op 2 = 打分+topk+expand（产出 `[m,2052] i32`） | ❌ 无槽位，要新增；CANN 侧有 `npu_lightning_indexer(_quant)` 可作 operator-level 参考/对照 |
| QSA 稀疏核心 | 同上 | op 3 = sparse/dense 二选一的注意力核心（入参含 packed indices / 或 nullptr 走 dense） | ❌ 无槽位；dense 路径也应注册成同一 op 的 mode，避免上层改两次 |
| MoE | `vllm_ascend/ops/fused_moe/*`（`moe_mlp.py:166-180` 调 `npu_grouped_matmul_swiglu_quant_v2`） | op 1 = permute+MX 量化+GMM1+SwiGLU（= 官方 `GroupedMatmulSwigluQuantV2` 的融合粒度）；op 2 = GMM2+unpermute+combine | ✅ 槽位现成（但 m>64 的形态要替换掉现有调用） |
| HC | **无**（`vllm_ascend/ops/mhc.py` 是别的模型的 mHC，**不是**本模型的 gated-residual HC，勿混用） | op = `combine_and_mix`（含组内 GemmaRMSNorm + 两个 skinny GEMM + sigmoid 门平均） | ❌ 新增 |
| PLE / ngram | 无 | op = ngram id + 行 gather；PLE 的 conv gate 段 | ❌ 新增；表的驻留策略需要 host 侧资源管理（官方 `engram_config cpu_offload`） |

### 8.3 对算子接口的三条硬要求

1. **权重按 checkpoint 原始排布消费**（用户既定要求）：MoE 的 `experts.gate_up_proj` 融合张量、
   GDN in_proj 的 `q|k|v|z|b|a` 行拼接、HC 的 down+inj 打包都直接吃；唯一建议偏离的是 HC 的
   **`336 = 324 + 12 pad`**（`nvidia/hyperconnection.py:95`）——那是 CUDA cublas 的启发式而非数学
   要求，我们按 324（或对齐到 16 的倍数）解，**需 M31 与集成侧确认**。
2. **in-place state 必须申报**：`direct_register_custom_op(..., mutates_args=[...])`（或跨 C++ 侧
   显式声明），涉及 `ssm_state`、`conv_state`、主 KV、raw key ring、compressed key cache、
   PLE conv state。否则 vLLM 的 graph capture / 别名分析会出错。
3. **必须只处理"一个 chunk"**：vLLM 的 prefill 是 **chunked prefill**，
   `QwenGatedDeltaNetAttention` 的 prefill 分支拿到的是 `prefill_query_start_loc` +
   `prefill_has_initial_state` + `chunk_indices/chunk_offsets`（`qwen_gdn_linear_attn.py:1509-1521`）。
   → 我们的 GDN/attention kernel 的输入契约必须是"**带初态、长度 m 可以是 1..M_MAX 的任意值**"，
   而不是"整段 4097 一次性"。m=4097 是验收口径，不是接口假设（清单第 5、11 条与此呼应）。

---

## 9. 未取证 / 存疑（单独成节，禁止当事实引用）

1. **l2norm 的位置两栈不一致**：vLLM-nvidia 在 chunk op **外**做
   （`qwen_gdn_linear_attn.py:1520` `use_qk_l2norm_in_kernel=False`，warmup 注释 `:1093` 同）；
   vllm-ascend 在 chunk op **内**做（`vllm_ascend/ops/gdn.py:615` `use_qk_l2norm_in_kernel=True`，
   且 `:477-480` 明说融合的 rearrange+l2norm "只对 decode-only non-spec 路径有效"）；
   而 CANN 官方的 `chunk_gated_delta_rule` **没有实现** l2norm（golden 的该 flag 未实现，
   `chunk_gated_delta_rule_golden.py:54`）。数学等价，但**我们的实现位置必须选定并与 golden 声明一致**；
   本文件建议融进 P2（prolog 段）。
2. **368 TFLOPS 的 AIC 数口径未确认**：roofline 文档只写"950PR / NPU1 957b 实测"
   （`roofline_model.md:26-28`）。若那是 36 子系统的满配值，本机 28 AIC 应乘 0.78 →
   §6 的所有算力下限要 ×1.28。**实测标定（一个 8192³ bf16 matmul）即可关闭此条**。
3. **【高优先，阻塞验收】QSA 稠密替代与官方 golden 不等价**：`indexer_budget=2048` 只覆盖
   512 个 4-token 块，而 4097 长度的 prompt 有 1025 个块；位置 4096 的 query 只能看到 2048 个
   历史 token（不是全部 4096）。→ **稠密 causal 的输出与官方 QSA 输出必然不同**，
   而 docs/11 §5（遗留事项 1）的裁决是"m=4097 验收首期走稠密路径"。两者无法同时成立，必须二选一：
   (a) m=4097 的 golden 定义为自建稠密 causal 参考（则 indexer 打分/topk 可以延后，但**上游
   cache 填充仍必须做**）；(b) golden 用官方 QSA（则 indexer 是 m=4097 的必做项）。
   **建议 escalate 给 tower**（已在本 mission 的汇报中提出）。
4. **MoE 是否 48 层全覆盖**：`nvidia/model.py:243-245` 的 `is_moe_layer` 依赖 `decoder_sparse_step`
   取模，而 config.json 里**没有该字段**（未确认 `Qwen3NextConfig` 的默认值是 1 还是别的）。
   若默认非 1，则部分层是 dense MLP（`Qwen3NextMLP`），MoE 的层数假设要改。
5. **GDN chunk 扫描每步固定开销未实测**（§3.1 的 130-260µs/层是推测）；官方高阶 API 的
   `IterateAll` 有 44-69µs/call 的 floor 报告（`cannbot-knowledge/.../ol_246_...md:29`），
   我们走基础 `LoadData`+`Mmad` 是否同样有此 floor **未知，必须标定**。
6. **MXFP4 的 4× 是同频理论值**（docs/03:27），A4W4 在本机的实际可达率（含 MX scale 装载开销）
   未实测；不过 MoE 段本来就被带宽支配，不影响结论方向。
7. **ngram 表 95.37GiB 的裁剪可行性未取证**（能否只对 prefill 的 token 做稀疏行预取、
   是否必须整表可用）。
8. **HC 的 `336` pad**（上面 §8.3 第 1 条）与 vLLM CUDA 侧的 `low_latency_gemm` 调优形状
   （`(336,10240)`）是否属于数值契约的一部分——若 golden 只是数学等价，我们可改用 324。
9. **lm_head 是否需要全 m 行的 logits**（2.03GB / 7.1ms）未裁决。
10. **M34/M35/M36 已经开工**，本文件的部分结论（尤其 §5.1 的 h 存放方案、§6.1 的 HC m-tile 约束、
    §2.5 的 indexer 范围切分）是在它们开工后发布的，需要它们反向确认是否已被采纳。

---

## 10. 主要来源索引

**规格（权威）**：`/workspace/Qwen3.8-Flash-Next-MXFP4/config.json`；
`/workspace/vllm/vllm/models/qwen4_exp/{config.py,nvidia/model.py,nvidia/qsa.py,nvidia/indexer_qsa.py,
nvidia/ops/{qsa.py,hc.py,ple.py},nvidia/{hyperconnection.py,ple_layer.py,ngram_embedding.py,
low_latency_gemm.py,model_state.py},common/{hyperconnection.py,qsa_cache.py,ngram_embedding.py},amd/*}`；
`/workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`；
`/workspace/vllm/vllm/model_executor/models/qwen3_next.py`；`/workspace/vllm/vllm/model_executor/layers/
fused_moe/*`；`/workspace/vllm/third_party/flash_linear_attention/ops/{chunk.py,utils.py}`；
golden `/workspace/vllm/tests/models/qwen4_exp/{test_qsa_reference.py,test_qsa_pre_indexer.py}`。

**donor（官方算子）**：`/workspace/ops-transformer/attention/{chunk_gated_delta_rule,chunk_kda_fwd,
recurrent_gated_delta_rule,flash_attn,fused_infer_attention_score,sparse_flash_attention,
lightning_indexer_v2,indexer_quant_cache}/`；`.../mamba/causal_conv1d/`；`.../gmm/{grouped_matmul,
grouped_matmul_swiglu_quant_v2,grouped_matmul_finalize_routing}/`；`.../moe/{moe_gating_top_k,
moe_init_routing_v3,moe_init_routing_v4,moe_token_permute_with_routing_map,moe_token_unpermute*,
moe_finalize_routing_v2}/`；`.../posembedding/{kv_rms_norm_rope_cache,inplace_partial_rotary_mul}/`；
`.../mhc/`（形态参考，非本模型 HC）；`/workspace/ops-nn/activation/swiglu_group_quant/`；
`/workspace/ops-nn/norm/add_rms_norm*/`。

**vllm-ascend（接入目标）**：`/workspace/vllm-ascend/vllm_ascend/ops/{gdn.py,gdn_attn_builder.py,
register_custom_ops.py,fused_moe/*}`；`vllm_ascend/ops/triton/fla/{chunk.py,cumsum.py,
chunk_scaled_dot_kkt.py,solve_tril.py,wy_fast.py}`；`vllm_ascend/device/device_op.py`；
`csrc/{attention,gmm,moe,online_mxfp4_gemm}/`。

**本仓库**：docs/05（约束与资源）、docs/10（GDN 数学与 prefill chunk 扫描）、docs/11（QSA/KV）、
docs/12（段序与资源表草案）、docs/09（MoE donor 地图）、docs/01（环境/表规模）、
`m13_moe_layer/README.md`（m≤64 的六条硬阻塞）、`m14_gdn_layer/README.md`（GDN 层 S1–S7 与同步表）、
`m15_layer_loop/README.md`（48 层循环接口 + msprof 基线）、`m4/m9/m11/m7/README.md`。

**平台**：`cannbot-knowledge/knowledge/ops/ascendc/concepts/roofline_model.md`（CUBE/VEC 峰值、
ridge、L2 48MB 说法与 docs/03 的 128MB 有出入，本文件按 docs/03 的 128MB 用）、
`cannbot-knowledge/knowledge/common/platforms/concepts/target_ascend950pr.md`（UB/L1/L0C/L2/带宽）、
docs/03 §1.2/§1.3/§6（Cube:Vector=8:1、MXFP4=4×、128MB L2、CrossCore 约束）。

---

## M103 重盘（@20bd20d）

> **这是什么**：M103（`agent-prefillplan`，2026-09-27，**只读 survey —— 本 mission 未改任何
> `m15_*` 源文件，本轮只新增本节与顶部横幅**）。任务 = 在**当前 main** 上把 `m=4097` prefill 的
> 现状逐段重盘，并给出**可直接派单的文件级拆分**。
>
> **基准 commit = `20bd20d`**（`git rev-parse --short HEAD`；本节末「M103-8」给了全部复现命令与读数）。
>
> **引用口径**：本节所有 `文件:符号` **现取于 `20bd20d`**；凡给行号处**一律标注该 commit**。
> 行号只作定位辅助 —— **符号名才是锚点**（本仓被"钉 main / 钉未合入分支"咬过多次，见 `docs/17 §9.6`）。
>
> **本节的边界**：只覆盖「现状 + 缺口 + 文件级拆分」。正文（`## 0.`–`## 10.`）已经做过的
> 形状表 / FLOP / 时间估算**不在这里重复**；正文结论凡与本节的现况不符，**以本节为准**。
> 未确定的部分集中在本节 **M103-6**，**不许当事实引用**。
>
> **两个在途 mission 的状态（按 `docs/17 §9.6` 标注）**：**M100**（分支
> `feat/m100-ple-segment-wiring-into-the-layer-k`，**未合入**）正在改
> `m15_layer_loop/m15_layer_kernel.h` / `m15_layer_loop/m15_layer_loop.asc` 与一个新文件
> `m15_ple_wire.h`（均在 `m15_layer_loop/` 下；后者当前树上**不存在**）；**M101**（分支
> `feat/m101-attention-core-segment-lift-and-o-p`，**未合入**）新增了 `m15_attn_core*`
> 三个文件（独立 target）。**本节对这两个分支的任何描述都只是"当时读到的 diff 概要"，不是既成事实。**

### M103-0. 速览（6 条）

1. **正文（M33）的方向性结论仍成立**，但它的 `文件:行` 引用**全部失效**（写于 M40/M58/M91/M97/M98
   之前）。正文 §3.3 的六条 MoE 硬阻塞里：**2 条已变、1 条换了形态、3 条仍成立**（见 M103-1.3）。
2. **M33 之后出现三条它没预见的结构性障碍**，它们决定拆分方式：
   - **N1 — attention 相位 A 是 `replace` 而不是 `augment`**：`A.apW != nullptr` 时
     `M15L_FusedBody` 走 attention 臂，**不再写 `subOut`**（hc 形态下 `subOut = A.hcAttnOut`
     = hc(H2) 的 BO）⇒ 子层出口**没有生产者**，而下游 MoE 照跑、**不报错、只数值错**。
     M97 已把它登记在 `m15_layer_loop/evidence/attn_wire/README.md` §5 第 8 项。
   - **N2 — attention core 的既有资产是 decode 形状**：M101（未合入）机械 lift 的
     `m10_attn_decode` 段是 Q `[2][16][256]` + 14 split × 2 KV 头 + split-K + `Combine()`，
     **不是** prefill 的 m-tile causal FA；且它自带 35 处 cross-core 调用 / 11 个 id（**含 mode 4**），
     与 `m15_layer_resources.h` 现有的 (核型, mode) 分节冲突，融合前必须**重号**。
   - **N3 — GDN 的 prefill 资产是 AIC 空转的 fork**：`m18_gdn_prefill/` 已有完整 BT=64
     chunk 扫描（已合入 main），但它是**纯 AIV/fp32、AIC 空转、独立工程、与 m15 零接线**，
     且 README §4.7 自述 **m=4097 单层 15.51 ms**（外推 36 层 ≈559 ms），比正文 §6.3 的
     GDN 算力下限（1.37 ms/层）高约一个量级 ⇒ **GDN 选型（复用 m18 的 VF vs 走 mmad）需要先裁决**。
3. **五个段的 prefill 就绪度差别很大**：hc 的 AIV 段**已按运行期 `m` 参数化**（只被 `M_MAX=64`
   包络卡住）；GDN **完全没有 chunk 扫描**；attention 的 prolog 与 cache 填**只在探针尺度**
   （≤8 行）存在；MoE 仍是 **E=4 / topk=4 缩形档**且 AIC 只用 5/28；PLE 只有**空操作挂载点**。
4. **必须串行的只有"挂载点"那 4 个文件**（`m15_layer_resources.h` / `m15_layer_kernel.h` /
   `m15_layer_loop.asc` / `m15_loop_layout.h`），且要**等 M100 合入**。其余 5 条段体可以
   **纯新文件并行**（M103-2）。
5. **`m=4097` 不能对官方输出**：唯一官方 CPU 参考（`m21_layer_ref/`，= M39 交付物）归档档只有
   m=1 与 m=64，且它的 attention 输出是官方**稀疏**结果；`docs/17 §7` 已禁止把它当稠密判据。
   逐段判据要自建或复用现有 numpy 参考（M103-4）。
6. **KV/cache 侧容量已经按 prefill 定尺**（`PREFILL_BLOCKS=257`、`KV_LAYER_STRIDE=8,421,376`）
   ⇒ prefill 的 KV 工作主要是**写入几何与时序**，且**只能用 `M15KV_KV_*` 宏**（M103-5）。

### M103-1. 五段逐段现状（每条给 `文件:符号`；行号钉 `20bd20d`）

#### M103-1.1 GDN 段（36 层）

| 项 | `m=1`（decode）现状 | `m=4097` 缺什么 |
|---|---|---|
| 入口 | `m15_layer_loop/m15_gdn_layer.h` 的 `class GdnLayerChain`：`Init(const GdnLayerPtrs&)` / `ProcessAiv()` / `ProcessAic()`；段序由 `stageLimit` 门控，S1/S7=`NormStage<false/true>`、S2=`Cube::Bf16Gemm<IN_K,IN_OUT>`、S3=`Prolog::GdnProlog`、S4=`Recur::GdnDecodeRec`、S5=`Gated::RmsNormGatedStage`、S6=`Bf16Gemm<OUT_K,OUT_N>`。**没有名为 `S1..S7` 的函数** —— 段序即 `ProcessAiv`/`ProcessAic` 里的顺序 | — |
| 有 chunk 扫描吗 | **没有**。全文件搜 `chunk\|cumsum\|WY\|inverse\|scan`（大小写不敏感），命中的 `chunk` 全部是"64-lane 向量寄存器分块"（`CHUNKS_H`、`CHUNK_PER_HEAD_`）；**无 cumsum、无 A/T 矩阵、无 `(I+L)^{-1}`、无跨 chunk 状态扫描** | 整套 P4–P9（正文 §2.4 的表）要从零建 |
| `m` 用在哪 | 只有两处随 `m` 变：`NormStage::Run` 的 `for (row = bid; row < M; row += nAiv)`；`Bf16Gemm::RunTile` 的 `CeilDiv(mTotal, BASE_M)` | 其余按单 token 或 `M_MAX` 行定尺 |
| 单 token 定尺 | `m15_layer_loop/m15_gdn_resources.h` 的 `SZ_Q / SZ_K / SZ_V / SZ_G / SZ_BETA / SZ_O`（S3/S4 中间量只有**一份 token**）、`UB_RC_G / UB_RC_EG / UB_RC_BETA / UB_RC_END`；`GdnProlog::Init` 的 `xGm_/qGm_/vGm_/gGm_` 按单行长度 `SetGlobalBuffer`（**无行 stride**）；`Recur::GdnDecodeRec` 的 `stateGm_` 无 T 维 | 每 chunk 的 `[C,16,128]` / `[C,48,128]` 等要新开 |
| 包络 | `M15G::M_MAX = 64`、`BASE_M = 64`（单 tile）、`CHAIN_M = 1`（**作标识符** `\bCHAIN_M\b` 在 `m15_layer_loop/` 下只命中定义处 **1** 行；作**字面子串**（如 `M15_CHAIN_M`）命中 **16** 行 —— 两条命令与读数见 M103-8 C6） | UB 余量只剩 **227,328 → 248 KB 之间约 26 KB**（`UB_RC_END = UB_RC_BETA + M_MAX*32`，注释自述 `UB_RC_END = 227328`）—— **几乎没有余量**给 chunk 扫描的开销 |
| m-tile 循环 | `Bf16Gemm::RunTile` 里 `for (mBlock = 0; mBlock < mLoop; ++mBlock)` **存在**，`CopyInA` 用 `srcDValue = K` 作行距、`calcM = max(curM, 2)`（3510 Nd2Nz "行数为 1 不切 NZ" 的契约） | 契约注释写死「`mLoop` **必须为 1**（m ≤ BASE_M = 64）」；**没有 m>1 的档**（两个 host launcher 都把 m 钉成 1）⇒ 能否直接跑 65 个 m-tile **未确定**（M103-6 第 3 条） |
| 状态 | `conv_state` planar bf16 `[ST=3][CH=10240]`，`GdnProlog::CopyOutBlock` 每次 launch 左移一位（**结构上写死单 token**）；`ssm_state` fp32 `[48,128,128]`，`GdnHeadRecurrence` **每 head 一次递推** | 两者都要改成"带初态、长度 m 任意"的 chunk 语义（正文 §5.3 ★11） |
| 相位边界 | 相位 A 两侧是 `FLAG_HC0_BOUND_AIV` / `FLAG_HC1_BOUND_AIV`；`m15_gdn_resources.h` 自述"本段实际占用 4–15 全部槽位，只剩 0–3 空" | m>1 改了行切分 ⇒ `GdnLayerChain::ProcessAiv` 里"m=1 时只有 AIV0 有活、其余立即越过 S1"那套**全体 AIV 到齐**的论证要重述 |

**可复用资产**：`m18_gdn_prefill/m18_gdn_prefill.asc` 已有完整 BT=64 扫描
（`CumSumExpVF / GammaStrictVF / GammaInclVF / GemmAA / GemmBT / GemmTA / TrilSolveVF /
StateScaleVF` + `GdnPrefillChunkScan::ProcessHead`），**已合入 main**。它的形态与限制：
**fp32-only、单层、AIV-only、AIC 空转**，UB 自用 `UB_TOTAL`（其头内注释自述 225.5 KB，
`static_assert(UB_TOTAL <= 248 * 1024)`），BufferID 只用 0–3、**无 cross-core flag**；
`m18_gdn_prefill/README.md` §8 自述四条限制：`fp32 only` / `conv1d、l2norm、gating 在核外` /
`无跨 chunk、跨核流水` / `单 head 状态 64 KB 常驻 UB`。
**它的 numpy fp64 逐句参考 `m18_gdn_prefill/check_ref.py` 可以直接当 m=4097 的 GDN 段判据**
（README §4.7 记录 target 档 = 48 head × 65 chunk 已跑过）。

#### M103-1.2 full attention 段（12 层）

| 子段 | `m=1` 现状 | `m=4097` 缺什么 |
|---|---|---|
| 相位 A 占位直通 | `m15_layer_loop/m15_attn_layer.h` 的 `m15_attn_passthrough_body(xIn, yOut, bytes)`：`CHUNK_BLOCKS = 16`（512 B/AIV），`start = bid * CHUNK_BLOCKS`，`if (start < nBlk)`；挂载点是 `M15L_FusedBody` 的 `m15_attn_passthrough_body(subIn, subOut, HIDDEN * 2u * A.m)`。**它是 m=1 定尺的**：56 个 AIV × 16 块 = 896 块 = 5.6 行（28 AIC 时）⇒ **`m ≥ 6` 起它只搬前 896 块，且不报错、不回绕**。今天不可达（host 把 m 钉 1），prefill 一上就踩 | 要么按行重排 AIV 分派，要么被真 attention 臂取代 |
| 前端 prolog | `m15_layer_loop/m15_attn_prolog_probe.h` 的 `m15_attn_prolog_probe_body(...)`（M88；M97 已把它接进相位 A）：AIV 按 **`bid` → head** 硬分派（`bid < 24` → q 头 `bid`；`24/25` → k 头 0/1；`26` → v；`27..30` → indexer 4 头；`31` → raw k），**一趟只处理一行**；AIC 按 `for (g = bid; g < GEMM_NBLOCKS; g += nCores)` 分 N 块，A 操作数**按 2 行**给（3510 m=1 quirk；`m15_attn_prolog_host.h` 的 `H_ApLaunch` 按 `apX + xRow*AP_HIDDEN*2` **逐行传指针**） | (a) `(row × head)` 二维工作项；(b) AIC 加 m-tile；(c) `LayerArgs::apPos` 是**单个 uint32** ⇒ 需要 per-row 位置；(d) `apX / apY0 / apOut` 三块平面是**单行**（host 用 `Y0_N * 2u` 之类按 2 行分配） |
| 三套 cache 的布局 | **已冻结**，唯一权威 `m15_layer_loop/m15_attn_kv.h`（物理字节布局 + `M15KV_KV_*` 寻址宏 + paged 契约 + off-by-one 门控 + 容量 `static_assert`）；容量**已按 prefill 定尺**：`PREFILL_M=4097`、`PREFILL_BLOCKS=257`、`KV_LAYER_STRIDE=8,421,376`、`KV_PLANE_BYTES=101,056,512` | 无（容量够；但主 KV 几何刚被 M98 更正，见 M103-5） |
| cache 填数学 | `m15_layer_loop/m15_attn_cache.h`（M98）已有 raw ring 落环、compressed 池化（fp32 累加、结果落 bf16）+ 组首位置 RoPE、主 KV 落页。**但它是独立启动的探针档**：`AC_MAX_ROWS = 8`、`AC_MAX_GROUPS = 2`、单核 AIV 串行、BufferID 10/11/12、**无 cross-core**。**`M15AC::` 在 `m15_layer_loop/` 的 `.h`/`.asc` 里被引用 0 次**（命令见 M103-8 C2，范围含 `m15_attn_cache.h` 自身之外的全部 `.h` 与 `.asc`） | (a) 接进 `M15L_FusedBody` 的相位 A；(b) 从 8 行探针尺度变 chunk 尺度；(c) `m15_layer_loop/evidence/attn_cache/README.md` §5 自述"真实规模档（真实权重切片 + 真实 4097 全上下文）未跑" |
| 稀疏核心 / 打分 topk | **未做**（M97 的 `m15_layer_loop/evidence/attn_wire/README.md` §5 第 4 项） | 稠密 causal FA 核心（m-tile）；正文 §2.5 的裁决是首期走稠密、indexer 打分/topk 可延后，**但 cache 填充不可省** |
| core 的既有资产 | M101（**未合入**）在 `m15_attn_core.h` 等三个文件里机械 lift 了 `m10_attn_decode` 的 device 段，**形态是 decode**（见 N2）；正文 §3.2 建议 prefill 走 `flash_attn` 的 ND config4（`sOuter128 / sInner128` + `gS1-merge`） | prefill core 是一个**新文件**，不是改 M101 的产物 |
| `o_proj` + `×sigmoid(gate)` | 未做 | 与 core 同批 |

#### M103-1.3 MoE 段（48 层）—— 正文 §3.3 六条硬阻塞逐条现况

| # | 正文 §3.3 说的 | 现况 | 锚点（`20bd20d`） |
|---|---|---|---|
| 1 | GEMM item 单 M tile、无 m 循环 | **已变**：work item 已是 `(expert, mTile, nBlock)`，`MoeLayerChain::ProcessAic` 里有 `mTiles = (t + BASE_M - 1) / BASE_M` 与 `rowBase`（shared 用 `mt*BASE_M`，routed 用 `offGm.GetValue(slot) + mt*BASE_M` 前缀和）；`MXFP4GemmItem::Run(nBlock, rows)` 仍是单 tile | `m15_layer_loop/m15_moe_layer.h` 的 `MoeLayerChain::ProcessAic`、`MXFP4GemmItem::Run` |
| 2 | 每专家固定行槽 `NUM_EXPERTS * M_MAX` | **已变（M91）**：`SZ_AQ / SZ_AS / SZ_GU / SZ_HQ / SZ_HS / SZ_Y` 改为**紧凑 `Σt_e` 上界 `TOTAL_MAX`**（`m15_moe_resources.h` 的对应常量，注释自述"padded 定尺是 41× 的浪费"）。**容量墙仍在**：`TOTAL_MAX = M_MAX * TOPK_MAX = 256` | `m15_layer_loop/m15_moe_resources.h::TOTAL_MAX` |
| 3 | 路由数组 `TOTAL_MAX` 定长 | **仍成立**：`UB_IG_SRC / UB_IG_EXP` 为 `i32[TOTAL_MAX]`，`SZ_PERM / SZ_INV / SZ_WTK` 同为 `TOTAL_MAX` / `M_MAX*TOPK_MAX` 行，并有 `static_assert` 钉着 | 同上 |
| 4 | 计数排序只在 AIV0 单核标量 | **仍成立**：`MoeLayerChain::ProcessAiv` 里 `isPrimary = (bid == 0)`，`router.Run()` 与 `idxGen.Run()` 都在 `if (isPrimary)` 内；`IndexGenStage::Run` 是 `for s < M*TOPK` 标量 | 同上 |
| 5 | router 单核 GEMV + 固定 UB 行块 | **仍成立**：`RouterStage::Run(subLimit)` **不收 bid/nAiv**，行块 `RT_RB = 8`，专家按 `RT_EGRP = 8` 流式（**时间维切，不跨核切 N**） | 同上 |
| 6 | host 侧 `m > M_MAX` 直接拒跑 | **形态已变**：m15 树里没有那句（原句只在 `m13_moe_layer/m13_moe_layer.asc`）。现在 `m15_layer_loop/m15_moe_host.h` 的 `H_LaunchFused`/`H_LaunchMoeOnly` **硬编码 `m = 1u`**；`m15_layer_loop/m15_layer_loop.asc` 对 `hcM / chainM > M15H::M_MAX` 做 clamp-to-1 + WARN | `m15_layer_loop/m15_moe_host.h`、`m15_layer_loop/m15_layer_loop.asc`（`20bd20d` 第 93/96/145–148/168–172 行是那两处 clamp） |

**另外四条 prefill 会立刻撞上的（正文没写）**：

- **路由规模仍是 E=4 / topk=4 缩形档**（`NUM_EXPERTS = 4`、`TOPK_MAX = 4`，`m15_moe_resources.h`）。
  M91 把 **UB 常驻量与 `NUM_EXPERTS` 解耦**（`UB_RT_*` 在 E=4 与 E=512 两档读数相同，依据
  `m15_layer_loop/moe_relift/m91_README.md` 的 rescale 表与 `m15_layer_loop/moe_relift/check_stream.py` 的 B1），
  但 `NUM_EXPERTS 4→512`、`TOPK_MAX 4→10` 属于**规模常量重定尺**，M91 自己写明"归 MoE-A"。
- **AIC 并行度**：`for (nb = bid; nb < GU_NBLK; nb += numBlocks)`，而 `GU_NBLK = 5`、`DN_NBLK = 10`
  < 28 ⇒ 只有 `bid < 5`（down 为 `bid < 10`）的 AIC 分到活。这就是 `m15_layer_loop/README.md`
  §8 第 12 条记的 **+51% 退化（39.06 → 58.96 µs/层）** 的根因。prefill 必须把它打平
  —— 正文 §3.3 的"按 `(专家, m-tile, n-tile)` 三维静态切核"正是这件事。
- **MoE 权重槽表是 E=4 定尺**：`m15_layer_resources.h` 的 `MW_WGU_BYTES = NUM_EXPERTS * GU_N * (HIDDEN/2)`
  （注释自述 6,553,600 B）⇒ E=512 时同式给 **838,860,800 B/层**（≈128×），`MOE_W_STRIDE` 与
  host arena 必须一起重定尺。48 层只算 routed gate_up 就是 ~40 GB（全 MoE 权重 ~64 GB 量级，
  仍在"权重常驻"的可行域内，但**槽表要一次改到位**）。这会牵动 `m15_layer_loop/slice_layer_manifest.py`、
  `m15_layer_loop/weights_manifest.txt` 与 `tools/weights/`（**不在 M103 的阅读范围**）。
- **`m15_layer_loop/m15_moe_layer.h` 是机械生成物**：由 `m15_layer_loop/lift_moe_segment.py` 从 m13
  抽 device 段 + 6 类替换 + `assert_upstream_vf()`，`--check` 可复核 ⇒ **手改它会破坏 `--check`**。
  `m15_layer_loop/m15_gdn_layer.h` 同理（`lift_hc_segment.py` 系同族，README §8 第 11 条）。

#### M103-1.4 hc 段（每层 2 个边界）

- **AIV 侧已经是按运行期 `m` 参数化的**：`m15_layer_loop/m15_hc_layer.h` 里五个段的 item 网格都是
  `const uint32_t nItems = p.m * NCH_*;` + `for (uint32_t i = bid; i < nItems; i += nAiv)`
  （S1 用 `NCH_D`、S4 用 `NCH_R`、gate 用 `NCH_H`）⇒ **hc 的结构不缺行循环**。
- 真正的墙是**包络**：`m15_layer_loop/m15_hc_resources.h` 的 `M15H::M_MAX = 64`、
  `BASE_M = M_MAX`（单 m-tile）、全部 `SZ_* = M_MAX * …`、GM 视图长度，以及 UB 里的
  **`UB_IWTAB_SLOTS = M_MAX * HC`**（injection 权重表按行暂存，`INJW_SLOT = 8`）。
  M_MAX=64 时该表占 `UB_IWTAB_SLOTS * INJW_SLOT * 4 = 8,192 B`；**把包络抬到 4097 会变成
  524,416 B > 248 KB** ⇒ **hc 不能靠"抬包络"吃下 m=4097，必须按 m-tile（正文 §6.1 的 R3 建议
  ≤8–12 行）分块**。这是本节给 R3 补的算术出处。
- 余量充足：`UB_PEAK`（`= UB_BYTES_USED`，注释自述 89,472）对 248 KB；`L1_PEAK` 对 512 KB。

#### M103-1.5 PLE 段（层 1）

- 现状 = **打断点的结构**：`m15_layer_loop/m15_layer_kernel.h` 的 `LayerArgs::hcPleBreak` +
  `M15L_PlePhasePlaceholder()`（**显式空操作**，文件内注释明写"它的输入与输出逐字节相同，语义是
  「PLE 缺席」，**不是**「等价于 PLE」"）+ 两条相位边界 `FLAG_PLE_IN_BOUND_AIV` /
  `FLAG_PLE_OUT_BOUND_AIV`（复用 hc 的 id 8/9，登记在 `m15_layer_resources.h`）。
- **PLE 本体未实现**，卡在 ngram 表 95.37 GiB 与 cgroup 32 GiB（`docs/14 §10` 第 1 条是阻塞项）。
- **M100 正在返工（未合入）**：`git diff --stat main...feat/m100-ple-segment-wiring-into-the-layer-k`
  读数为 20 文件 `+6,578 / −18`，其中 `m15_layer_loop/m15_layer_kernel.h` +279（`LayerArgs` 追加
  17 个 PLE 字段、新增 `namespace M15L::PLEW`、`T_MAX = M15H::M_MAX = 64`）、
  `m15_layer_loop/m15_layer_loop.asc` +612、新文件 `m15_ple_wire.h` +789（`m15_layer_loop/` 下）、
  `m15_layer_loop/m15_hc_host.h` +68。
  ⇒ 两个直接影响：① `LayerArgs` 尾部与 `M15L_LAYER_HC_ARGS_DECL/FILL` 是 **M100 与 prefill
  的同一处冲突面**；② M100 的 PLE 平面仍按 `T_MAX = M_MAX = 64` 定尺 ⇒ **它不交付 PLE 的 prefill，
  也不构成 prefill 的既成事实**。（M100 的实现细节 M103 未逐行读，**未确定**。）

### M103-2. 文件级拆分与依赖（**本节的核心交付，可直接照抄成 scope 派单**）

#### M103-2.0 两条总原则

- **P1｜把融合 TU 当串行资源，Wave B 一律不碰它**（塔裁 2026-09-27，复审 r1）：
  `m15_layer_loop/m15_layer_loop.asc` 是唯一的单 TU，所有接线都过它，且 M97 曾声明"include 清单由
  M97 独占"、M100（未合入）现在也在改它。**⇒ 每条 Wave B 的独立验证路一律放进自己的目录**
  （照 `m18_gdn_prefill/`、`m10_attn_decode/` 的先例：自己的 `CMakeLists.txt` + 自己的 `.asc`
  （含 `main()`）+ 自己的 host 头 + 自己的 README + 自己的 `evidence/`），而
  **段体的设备头仍交付到 `m15_layer_loop/`** —— 那才是要被融合进层 kernel 的东西。
  **Wave B 一律不动 `m15_layer_loop/CMakeLists.txt`**（它现在归在飞的 M101 手里）；
  把各段 target 注册进那个文件、以及把段体接进融合 TU，都是**后续串行 mission（Wave C/D）**的事。
  这一条同时回答了"并行度"与"独立 target 放哪"两个问题。
- **P2｜`m15_gdn_layer.h` 与 `m15_moe_layer.h` 不动**：它们是 `lift_*.py` 的机械产物（`--check` 可复核），
  且代表 **decode 段**；prefill 是**另一个入口符号**（正文 §5 的方案 B：单 TU、两入口、kernel 内无
  m 分叉）。改它们会让 `--check` 与 decode 的既有验收基线一起变。

#### M103-2.1 Wave A —— **必须串行，1 条 mission，且要等 M100 合入**

**为什么串行**：这 4 个文件是全部段体的公共装配面；两方同时改必然冲突。

| 文件 | Wave A 要动什么 |
|---|---|
| `m15_layer_loop/m15_layer_resources.h` | ① 把 `ENTRY_GDN_PREFILL` / `ENTRY_ATTN_PREFILL` 从"符号常量"变成**真入口登记**；② 新增 prefill 的 **flagId 分节**（N2 里 attention core 的 35 处调用 / 11 个 id、含 **mode 4**，必须逐个登记并给相邻性 `static_assert`；注意现有分节只覆盖 mode 0/2）；③ 新增 prefill 的 **BufferID 分节**；④ 预留 prefill 的 UB/L1/L0C **峰值断言位**（数值等 Wave C 填）；⑤ `LayerArgs`/挂载所需的 `#include` 依赖顺排 |
| `m15_layer_loop/m15_layer_kernel.h` | ① `LayerArgs` 增加 prefill 字段：`posArr`（per-row 位置）或 `posBase + m`、`slotMapping`（若采纳）、`layerK`（attention 层序号）、`attnKv / attnComp / attnRing / attnPack` 四个基址、MoE 的 `counts / expertOffsets`；② 新增 `M15L_LAYER_PREFILL_ARGS_DECL/FILL` 宏与两个 `__global__ __mix__(1,2)` 入口；③ 处理 N1（`subOut` 的生产者） |
| `m15_layer_loop/m15_layer_loop.asc` | ① `#include` 新增的宿主头（1–2 行；`20bd20d` 的 include 块在 59–70 行、host 头在 826–846 行）；② `Ctx`/`Opts` 加 prefill 平面与 `runs=` 档；③ **`.asc` 的 include 清单是共享热文件，归 Wave A 独占** |
| `m15_layer_loop/m15_loop_layout.h` | 激活平面按 m 定尺：`H_ROWS_BYTES = M_MAX * HIDDEN * 2`（`20bd20d` 第 96 行）现在只够 64 行 = `64 × 2560 × 2` = **327,680 B**；4097 行需要 `4097 × 2560 × 2` = **20,976,640 B/平面**。`ws` / `hcWs` 的 prefill 定尺同理。**同一族的手算式都写在旁边**（复审 r1 的 P2-1 就是这一族的手算错：初稿写成 20,979,200，多算 2,560 B = 半行） |

**Wave A 的第二个 commit（A-2）｜MoE E=512 槽表重定尺 —— `MW_*` / `MOE_W_STRIDE` 的唯一 owner 是 Wave A**
（塔裁 2026-09-27，复审 r1：同一批数字曾被本节与 B4 同时指派两个落点，是自相矛盾）。
Wave A 在 `m15_layer_resources.h` 的 `MW_*` 数值区（`MW_ROUTER_BYTES` / `MW_WGU_BYTES` /
`MW_SGU_BYTES` / … / `MOE_W_STRIDE`）按 `NUM_EXPERTS = 512`、`TOPK_MAX = 10` 重算；
**B4 只读、不重复定义**（见 M103-2.2 里 B4 的那一行）。
算式与两档读数（供 A-2 对照）：`MW_WGU_BYTES = NUM_EXPERTS × GU_N × (HIDDEN/2)` ⇒
E=4：`4 × 1280 × 1280` = **6,553,600 B**（与 `m15_layer_resources.h` 注释自述值一致）；
E=512：`512 × 1280 × 1280` = **838,860,800 B**（= 128×）。
**输入已经现成**：`m15_layer_loop/moe_relift/m91_README.md`
的 rescale 表（E=4 档 / E=512 假设档 / padded 档三列，并声明 E=512 紧凑档的 6 项与 `WS_BYTES`
已与另一套独立复算逐个相等）。**注意**：A-2 只改**设备侧槽表**；host arena /
`m15_layer_loop/slice_layer_manifest.py` / `m15_layer_loop/weights_manifest.txt` / `tools/weights/`
那部分**不在 M103 的阅读范围，需要另立 mission 且排在 Wave A 之后**（因为同一个
`m15_layer_resources.h` 是单写者文件）。

**Wave A 的 scope（可直接抄；唯一权威版见 M103-2.6）**：
```
m15_layer_loop/m15_layer_resources.h
m15_layer_loop/m15_layer_kernel.h
m15_layer_loop/m15_layer_loop.asc
m15_layer_loop/m15_loop_layout.h
# 注：m15_layer_loop/CMakeLists.txt **不在此列** —— Wave B 一律不碰它；各段 target 的注册
#     与段体接进融合 TU，归后续的串行融合 mission（Wave C/D）。
```
并**显式声明**：`m15_layer_loop/m15_layer_resources.h` 是**单写者文件**，任何时刻只允许一条 mission
拥有它（含 A-2 与后续的 E=512 host arena 重定尺）。

#### M103-2.2 Wave B —— **可并行，5 条，文件互不相交**

> **关于本小节里"新建文件"的写法**：新文件**尚未存在**，而 `docs/scan_doc_refs.py` 的路径式引用检查
> 会把它们判 out-of-range（M103 初稿实测 **15 条 gating**）⇒ 新建文件一律写成 **stem glob**
> （`<stem>*` 或 `<dir>/**`），段体头的裸文件名在正文里给出、并注明都位于 `m15_layer_loop/` 下。
> 这与逐字列出**语义等价**，且已用 fnmatch 逐对核过「B1–B5 两两相交 = 0」（M103-8 C8）。

**B1 — GDN prefill 段**

| 项 | 内容 |
|---|---|
| 段体头（交付到 `m15_layer_loop/`） | `m15_gdn_prefill.h`（设备段；**从第一天就是可被融合 TU include 的形态**，见 M103-2.7） |
| 独立验证路（自己的目录 `m15_gdn_prefill/`） | `CMakeLists.txt` + `m15_gdn_prefill.asc`（含 `main()`）+ `m15_gdn_prefill_host.h` + `README.md` + `evidence/` |
| 允许改（**独占**） | `m15_layer_loop/m15_gdn_resources.h`（新增 prefill 段窗常量；**不做** decode 段任何改动） |
| **不许碰** | `m15_layer_loop/m15_gdn_layer.h`、`m15_layer_loop/m15_layer_kernel.h`、`m15_layer_loop/m15_layer_resources.h`、`m15_layer_loop/m15_layer_loop.asc`、`m15_layer_loop/m15_loop_layout.h`、`m15_layer_loop/CMakeLists.txt` |
| donor / 依据 | `m18_gdn_prefill/m18_gdn_prefill.asc`（算法已证、fp32 VF、AIC 空转）+ 正文 §2.4 的 P0–P11 段序 + §3.1 的选型表（intra 走 mmad、`(I+L)^{-1}` 走 32×32 寄存器前代）+ `chunk_gated_delta_rule` 的 `matmul_basic.h`（外部快照，在 `/workspace/ops-transformer/` 下） |
| 范围提示 | **选型（VF vs mmad）要先裁决**（M103-6 第 1 条）：若走 mmad，本条的规模与 UB 预算都变；若先复用 m18 的 VF 路线，则本条的第一步是"把 m18 的段搬进 m15 的资源/同步语法" |
| scope | 见 M103-2.6（**唯一权威**） |

**B2 — attention 前端（per-row prolog）+ cache 填**

| 项 | 内容 |
|---|---|
| 段体头（交付到 `m15_layer_loop/`） | `m15_attn_prefill.h` |
| 独立验证路（自己的目录 `m15_attn_prefill/`） | `CMakeLists.txt` + `m15_attn_prefill.asc`（含 `main()`）+ `m15_attn_prefill_host.h` + `README.md` + `evidence/` |
| 允许改（**独占**） | `m15_layer_loop/m15_attn_cache.h`（把 `AC_MAX_ROWS = 8` 的探针尺度改成 chunk 尺度；它现在只被自己的 host 头调用）、`m15_layer_loop/m15_attn_prolog.h`（`AP_*` 形状常量是它的权威 —— **含 `apX` 多行化所需的常量**） |
| **不许碰** | M88 的探针实现 `m15_layer_loop/m15_attn_prolog_probe.h`（只读复用它的 `AivNormRope`）、`m15_layer_loop/m15_attn_kv.h`（布局权威，只读）、Wave A 的 4 个文件、`m15_layer_loop/CMakeLists.txt` |
| donor / 依据 | `m15_layer_loop/m15_attn_cache.h`（M98 的池化/落盘设备段）、`m15_layer_loop/evidence/attn_prolog/oracle_attn_prolog.py`（R1–R13 表，本仓内） |
| 交付义务 | 接进融合 kernel 时**必须一并给 `subOut` 一个生产者**（N1）；本 mission 可以先把"臂"做完（与 M97 的 `runs=attnwire` 同型），但要在**自己的 `README.md`** 显式写明未接链 |
| scope | 见 M103-2.6（**唯一权威**） |

**B3 — attention 稠密 causal core（m-tile FA）**

> **改名（复审 r1 的 P1-1）**：本 wave 的段体头从原稿的 `m15_attn_prefill_core*` **改名为
> `m15_attn_fa_core*`**。原因：B2 的 stem glob 若写成 `m15_layer_loop/m15_attn_prefill*`，
> 会把 `m15_attn_prefill_core.h` / `m15_attn_prefill_core_host.h` 一起吃掉（fnmatch 实证 True/True，
> 见 M103-8 C9）⇒ 两条并行 mission 的 scope 相交、`TowerPlan` 直接派不出去。改名后前缀互不相同，
> 逐对核过 0 相交（C8）。

| 项 | 内容 |
|---|---|
| 段体头（交付到 `m15_layer_loop/`） | `m15_attn_fa_core.h`（**新名**） |
| 独立验证路（自己的目录 `m15_attn_fa_core/`） | `CMakeLists.txt` + `m15_attn_fa_core.asc`（含 `main()`）+ `m15_attn_fa_core_host.h` + `README.md` + `evidence/` |
| 允许改 | **无**（全新文件；**尤其不要改 M101 的 `m15_attn_core*` 三个文件** —— 那是 M101 的交付物，未合入且会被它继续改） |
| donor / 依据 | `flash_attn` 的 ND config4（`sOuter128 / sInner128` + `gS1-merge`，外部快照在 `/workspace/ops-transformer/attention/flash_attn/` 下）+ 正文 §3.2 的"稠密路径直接走 FA config4 流水、不经过 packed indices" + §4.2 的 donor 表；结构可参考 M101 的 mmad / FIXP / `dualDstCtl` 用法（**读**，不写） |
| 交付义务 | 自带 flagId 清单（mode,id）与 UB/L1/L0C 峰值（见 M103-2.7 的"融合清单"）；同步只用核内 buffer id + 核间 `CrossCoreSetFlag/WaitFlag`（用户原话：不用 set/wait flag 系列；除值依赖外不用 `PIPE_S`） |
| scope | 见 M103-2.6（**唯一权威**） |

**B4 — MoE prefill 段**

| 项 | 内容 |
|---|---|
| 段体头（交付到 `m15_layer_loop/`） | `m15_moe_prefill.h` + `m15_moe_prefill_res.h`（后者**只放 prefill 段自用**的常量） |
| 独立验证路（自己的目录 `m15_moe_prefill/`） | `CMakeLists.txt` + `m15_moe_prefill.asc`（含 `main()`）+ `m15_moe_prefill_host.h` + `README.md` + `evidence/` |
| 允许改 | **无** |
| **`MW_*` / `MOE_W_STRIDE` 的取值来源（塔裁 2026-09-27，复审 r1 的 P1-2）** | **唯一 owner 是 Wave A 的 `m15_layer_resources.h`**（A-2 按 E=512/TOPK=10 重定尺）。B4 **只消费、不重复定义**：`m15_moe_prefill_res.h` 里**不得**再出现第二份 `MW_*` / `MOE_W_STRIDE`（如需派生量，必须写成对 Wave A 常量的引用式）。原稿"E=512 槽表全部落在自己的 `_res.h`、Wave A 只保留指针+分支"的说法**已作废**（同一批数字曾被指派两个落点，是自相矛盾） |
| **不许碰** | `m15_layer_loop/m15_moe_layer.h`（lift 产物）、`m15_layer_loop/m15_moe_resources.h`（decode 档）、Wave A 的 4 个文件、`m15_layer_loop/CMakeLists.txt` |
| donor / 依据 | `m15_layer_loop/m15_moe_layer.h`（紧凑 Σt_e 寻址已是接口，见 M103-1.3 #1/#2）、`m15_layer_loop/moe_relift/m91_README.md`（E=512 rescale 表 + 规模墙清单）、`m7_router_topk/`（512 专家 / top-10 流式 router）、`m22_router512/`（`group_list` count 模式 `int64[512]` 契约）、`moe_init_routing_v4`（外部快照，`expandedTopkWeightOut` 协议） |
| 交付义务 | ① **AIC 打平**（现在只有 5/28 个 AIC 有活，见 M103-1.3）；② router 与计数排序**多核化**（现在单核 AIV0）；③ 缓冲区按 `active_num` 而不是 `M_MAX*TOPK_MAX` 定尺 |
| scope | 见 M103-2.6（**唯一权威**） |

**B5 — hc prefill / 行分块**

| 项 | 内容 |
|---|---|
| 段体头（交付到 `m15_layer_loop/`） | `m15_hc_prefill.h` |
| 独立验证路（自己的目录 `m15_hc_prefill/`） | `CMakeLists.txt` + `m15_hc_prefill.asc`（含 `main()`）+ `m15_hc_prefill_host.h` + `README.md` + `evidence/` |
| 允许改（**需塔先裁决，见下**） | 方案 α：**不改**任何既有 hc 文件 —— `m15_hc_prefill.h` 自带按 m-tile 的行循环，逐 tile 调现有 `M15H::HyperConnOp` 的段函数；方案 β：改 `m15_layer_loop/m15_hc_layer.h` + `m15_layer_loop/m15_hc_resources.h`（抬包络 + 加 AIC 的 m-tile 循环），此时这两个文件**归 B5 独占** |
| **不许碰** | `m15_layer_loop/m15_hc_host.h`（**M100 正在改它**，未合入 ⇒ host 侧接线归 Wave D）、Wave A 的 4 个文件、`m15_layer_loop/CMakeLists.txt` |
| donor / 依据 | `m15_layer_loop/m15_hc_layer.h`（AIV 段已 `nItems = p.m * NCH_*` 参数化）+ 正文 §6.1 的 R3（m-tile ≤8–12 行）+ M103-1.4 的算术（`UB_IWTAB_SLOTS = M_MAX * HC` 是抬包络的硬墙） |
| 塔的裁决点 | **改不改 `m15_layer_loop/m15_hc_layer.h`？** 若 B5 与任何 hc 相关 mission 并行，建议取方案 α（纯新文件），把"抬包络"留给后续独立 mission |
| scope | 见 M103-2.6（**唯一权威**；α / β 两版） |

**Wave B 的共同契约（写进每条 mission 的 scope 说明，避免接线期返工）**：

1. **段体不得吃 `LayerArgs`**（那是 Wave A 的文件，B 不能改）。设备段签名只吃**指针 + 标量**
   （`__gm__ uint8_t*` 平面 + `m` + `pos` 基址/表首址），或提供 `__aicore__ inline` body 让 Wave A/C 包壳。
2. **同步形态**（用户原话）：核内 pipeline 用 **buffer id**，核间用 **`CrossCoreSetFlag/WaitFlag`**；
   **不用 set flag/wait flag 系列**；**除有值依赖的场景外不用 `PIPE_S`**。
3. **资源常量自带**：UB 窗 / L1 区 / L0C / BufferID / flagId **全部写在各自的 `_res.h` 或头文件里**，
   **不得**写进 `m15_layer_resources.h`（那归 Wave A/C）；峰值断言写在自己的头里。
   **例外（塔裁）**：MoE 的 `MW_*` / `MOE_W_STRIDE` 全是 Wave A 的，B4 只读（见 B4 那一行）。
4. **交付一份"融合清单"**（格式与用途见 M103-2.7）：UB 字节区间、BufferID、flagId 的 `(mode, id)`
   列表、要接的挂载点 ⇒ 供 Wave C 做全局分节与"每 mode 独立断言峰值 ≤ 248 KB / L1 ≤ 512 KB /
   L0C ≤ 256 KB"。
5. **`m` 的语义**：`m ∈ [1, M_PREFILL]`，**4097 是验收档、不是接口假设**（正文 §8.3 第 3 条同口径：
   vLLM 的 prefill 是 chunked，kernel 必须接受"任意 m + 初态"）。
6. **KV/cache 寻址一律经 `m15_layer_loop/m15_attn_kv.h` 的 `M15KV_KV_*` 宏**，不许自带第二份数字
   （B2/B3 强制；见 M103-5）。
7. **不碰 `m15_layer_loop/CMakeLists.txt`**（塔裁，见 M103-2.0 的 P1）：独立验证路自带
   `CMakeLists.txt`，放在自己的目录里。原稿"它是 Wave B 唯一的共享文件、各 wave 在末尾追加
   `add_executable`"的说法**已作废** —— 按塔裁，Wave B 与那个文件零接触。

#### M103-2.3 Wave C —— **串行，1 条**（峰值与分节收口）

- 把 B1–B5 的实数填进 `m15_layer_loop/m15_layer_resources.h` 的 prefill 分节：每个 mode 的峰值断言、
  flagId 相邻性 + 每 id 用量（硬件 4 bit 计数器 ≤15）、BufferID 登记。
- 在 `m15_layer_loop/m15_layer_kernel.h` 完成挂载（`if constexpr (PREFILL)` 分支或独立入口符号）。
- **C 的第一件事必须是算这笔账**（M103-6 第 2 条）：相位 A 的 UB 窗对 GDN-prefill 与 attention-prefill
  是**互斥**的（KIND_GDN / KIND_ATTN 不同层），故 prefill 的 UB 峰值 = `max(GDN-prefill 窗,
  attention-prefill 窗, hc 窗, MoE 窗)`。**m18 的 chunk 扫描自用 225.5 KB，只剩约 22.5 KB 给
  PERSIST（gamma/norm）—— 能不能装下没算过。**
- scope glob：`m15_layer_loop/m15_layer_resources.h`、`m15_layer_loop/m15_layer_kernel.h`（都排在 Wave A 之后）

#### M103-2.4 Wave D —— **串行，可与 C 合并**（host 侧接线）

- `m15_layer_loop/m15_layer_loop.asc` 的 `Ctx`/`Opts`/`main` 分派与权重装载。
- `m15_layer_loop/m15_chain_host.h` 的 `H_LaunchChainLayer`（**`20bd20d` 上把 `m` 硬编码成 `1u`**）、
  `m15_layer_loop/m15_moe_host.h`、`m15_layer_loop/m15_hc_host.h` 的 m 传参
  （**注意 `m15_hc_host.h` 同时被 M100 改**）。
- 激活平面按 m 的 H2D 装载与 dump 口径（`m15_layer_loop/m15_layer_loop.asc` 的 `H_DumpWsRow`
  现在只 dump 行 0）。

#### M103-2.5 依赖图

```
M100 合入 ──► Wave A（挂载点/契约；含 A-2 的 MoE E=512 槽表）
                    │
                    ├─► Wave B1..B5（5 条并行；各自独立 target；文件互不相交）
                    │        └──────────────┐
                    └──────────────────────┴─► Wave C（峰值/分节收口 + 挂载）
                                                  └─► Wave D（host 接线）
                                                        └─► 端到端 m=4097 验收（m=1 同档回归）

先决裁决①（GDN 选型：复用 m18 的 VF vs 走 mmad）──── 决定 B1 的规模与 B1 的 UB 预算
先决裁决②（判据口径：自建稠密 causal 参考）───────── 决定 Wave D 的验收形态
```

**关键性质**：**B1–B5 不依赖 Wave A 落地**（它们只依赖"资源常量头 + 自己的段接口"）⇒
塔可以在 M100 合入前就让它们开工，只要约定 P2 与 Wave B 的 7 条共同契约。
**反过来，Wave A 必须在 M100 合入之后**（同一处 `LayerArgs` 尾部与同一个 `.asc`）。

#### M103-2.6 【唯一权威】可直接贴进 `TowerPlan` 的 scope 清单 + 与在飞 mission 的相撞矩阵

> **写法说明（与复审给的 YAML 逐条等价，只换了拼写）**：新建文件**尚未存在**，把全路径写进
> `docs/*.md` 会被 `docs/scan_doc_refs.py` 的路径式引用检查判 out-of-range（M103 初稿实测
> **15 条 gating**，rc=1）⇒ 新建文件一律写成 **stem glob**（`<stem>*`）或**目录 glob**（`<dir>/**`）。
> 已用 fnmatch 核过 **B1–B5 两两相交 = 0**、且每条 wave 自己的文件被自己的 scope 全覆盖 = 0 漏
> （M103-8 C8）。`evidence/.gitignore` **显式列出**：picomatch 的 `**` 默认**不匹配点文件**
> （`dot:false`）—— 本机无 node，**这条语义取自塔裁与 picomatch 默认，M103 未能在本地跑 picomatch 实证**；
> 两种语义下显式列出都不冲突。

```
# Wave A（串行，须等 M100 合入；m15_layer_resources.h 为单写者）
m15_layer_loop/m15_layer_resources.h
m15_layer_loop/m15_layer_kernel.h
m15_layer_loop/m15_layer_loop.asc
m15_layer_loop/m15_loop_layout.h
# 注：m15_layer_loop/CMakeLists.txt 不在任何 wave 的 scope（塔裁，见 M103-2.0 的 P1）

# Wave B1 — GDN prefill
m15_layer_loop/m15_gdn_prefill*
m15_layer_loop/m15_gdn_resources.h
m15_gdn_prefill/**
m15_gdn_prefill/evidence/.gitignore

# Wave B2 — attention 前端 prolog + cache 填
m15_layer_loop/m15_attn_prefill*
m15_layer_loop/m15_attn_cache.h
m15_layer_loop/m15_attn_prolog.h
m15_attn_prefill/**
m15_attn_prefill/evidence/.gitignore

# Wave B3 — attention 稠密 causal core（已改名到独立前缀，见 P1-1）
m15_layer_loop/m15_attn_fa_core*
m15_attn_fa_core/**
m15_attn_fa_core/evidence/.gitignore

# Wave B4 — MoE prefill（E=512 槽表数值来自 Wave A，本 wave 只读不定义）
m15_layer_loop/m15_moe_prefill*
m15_moe_prefill/**
m15_moe_prefill/evidence/.gitignore

# Wave B5 — hc prefill（方案 α：纯新文件）
m15_layer_loop/m15_hc_prefill*
m15_hc_prefill/**
m15_hc_prefill/evidence/.gitignore
# 方案 β 追加（会与任何 hc mission 相撞，塔需裁决）：
# m15_layer_loop/m15_hc_layer.h
# m15_layer_loop/m15_hc_resources.h
```

> **取证位置的替代写法**：若塔倾向沿用本仓既有约定（`git ls-files m15_layer_loop/evidence | wc -l` 在
> `20bd20d` 上 = **87** 个 tracked 文件、且 `m15_layer_loop/.gitignore` 不忽略它们），把每条 wave 的
> 两行 `evidence` 改成 `m15_layer_loop/evidence/<name>/**` + `m15_layer_loop/evidence/<name>/.gitignore`
> 即可 —— 两种写法都与其它 wave 不相交（`<name>` 取 `gdn_prefill` / `attn_prefill` / `attn_fa_core` /
> `moe_prefill` / `hc_prefill`）。

**Wave B × 在飞 mission 的相撞矩阵**（在飞 = 分支存在、**未合入**；按 `docs/17 §9.6` 标注状态）：

| Wave | 撞 M100（`m15_layer_kernel.h` / `.asc` / `m15_ple_wire.h` / `m15_hc_host.h` / `m15_chain_host.h`） | 撞 M101（`m15_layer_loop/CMakeLists.txt` / `m15_attn_core*`） | 撞 M104（`m18_gdn_prefill/**`；分支 `feat/m104-gdn-prefill-scan-selection-measurem`，M103 读时**0 提交**） |
|---|---|---|---|
| A | **撞** `m15_layer_kernel.h` + `m15_layer_loop.asc`（须等 M100 合入） | 无 | 无 |
| B1 | 无 | 无（Wave B 不碰 CMakeLists） | 无**写**重叠；但 donor = `m18_gdn_prefill/**` 正是 M104 的靶子（**移动靶**） |
| B2 | 无 | 无（`m15_attn_core*` 已避开） | 无 |
| B3 | 无 | 同上 | 无 |
| B4 | 无 | 同上 | 无 |
| B5 | 无（已避开 `m15_hc_host.h`） | 同上 | 无 |
| D | **撞** `m15_hc_host.h` + `m15_chain_host.h` | 无 | 无 |

⇒ **Wave B 五条均不撞** M100/M101/M104 的既有写面；**唯一共享文件 `m15_layer_loop/CMakeLists.txt` 已按塔裁
从 Wave B 的接触面里彻底移除**。
**注意 M104 与先决裁决① 的关系**：M104 的分支名是 "gdn prefill scan selection measurement"，
**很可能就是裁决①（GDN 选型：复用 m18 的 VF vs 走 mmad）的执行者** ⇒ 派 B1 之前应先看 M104 的结论，
否则 B1 会与它撞在同一个选型问题上（M103 读时 M104 尚无提交，**未确定**）。

#### M103-2.7 Wave B 的三条硬交付要求（防「建好但从未集成」的 fork 堆积）

**为什么单列这一节**：本 mission 在 M103-1.1 点过 —— `m18_gdn_prefill/` 是一个**已合入 main、
算法已证、但从未接进 m15** 的独立 fork（M103 在 `m15_layer_loop/` 下搜 m18 的符号未见接线）。
Wave B 若照"先建独立工程、以后再融"的老路走，很容易再堆出四个同型 fork。⇒ 以下三条**写进每条
Wave B 的交付判据**（不是建议）：

1. **段体头从第一天就是"可被融合 TU include"的形态**：① include guard；② **无 `main()`**
   （`main()` / host 判据只出现在该 wave 自己的独立 `.asc` 里）；③ **不用匿名 `namespace` 遮蔽**
   —— 用具名 `namespace`（融合 TU 里已有多段同名工具，匿名 namespace 会造成同名重定义或遮蔽）；
   ④ 设备代码只依赖设备头（`kernel_operator.h` 等）+ 自家的 `_res.h`，**不得**反向依赖 host 头；
   ⑤ 资源全部**编译期静态**：BufferID / flagId / UB 窗 / L1 区写在自己的头里，并带 `static_assert`
   峰值断言（用户原话：所有 buffer id / cross core id / 地址都由我们自己编译期静态分配）。
2. **交付一份「融合清单」**（一页，固定小标题，随 README 入库）：
   (a) **挂载点**：要接进 `m15_layer_kernel.h` 的哪个相位 / 哪个入口符号、消费与生产哪些 GM 平面；
   (b) **`LayerArgs` 需要的字段**（Wave A 照它加字段）；
   (c) **资源窗**：UB 字节区间、L1 区、L0C tile，以及各自峰值（供 Wave C 做"每 mode 独立断言
   峰值 ≤ 248 KB / 512 KB / 256 KB"）；
   (d) **BufferID 清单**（核内）与 **flagId 清单 `(核型, mode, id, pipe)`**（核间 CrossCore），
   含相邻性说明；
   (e) **需要相位边界的位置**（哪几处需要全体 AIV 的 mode-0 barrier + `PipeBarrier<PIPE_ALL>`）；
   (f) **`m` / `pos` 语义**（本段接受哪些 m、位置从哪来）。
3. **Wave A 为每段预留一个默认关闭的编译期开关**，形态照 M100（**未合入**）的 `M15_PLE_WIRE`：
   其 `m15_layer_loop/m15_layer_loop.asc` 里是 `O.pleWire = (H_EnvU32("M15_PLE_WIRE", 0u) != 0u) ? 1u : 0u;`
   —— 缺省 0、接线开时置 1（M103 在 wt-100 上 `grep -rn "M15_PLE_WIRE" m15_layer_loop/` 命中 **25** 行）。
   ⇒ prefill 的融合就变成"**翻开关 + 填清单**"，而不是"再写一遍"。
   **默认关闭**这一条是硬要求：它保证 `runs=all` 的 decode 零回归在融合前后都可复跑。

### M103-3. 形状 / mask / pos / `slot_mapping`

- **`m=4097` 的形状**（来自 checkpoint config，正文 §1 已核）：`HIDDEN = 2560`、`q|gate` = 12288、
  `k|v` 各 512、indexer `index_qk` = 640；MoE 的 `x[m,2560]`、槽位 `S = m * topk`。
  **每个按行集的平面 = `4097 × 2560 × 2` = 20,976,640 B**（对比 `m15_loop_layout.h` 现在按
  `M_MAX = 64` 行的 `64 × 2560 × 2` = 327,680 B）。**同族的手算式一并给出**（复审 r1 的 P2-1 就是
  这一族的手算错：初稿写成 20,979,200，比真值多 2,560 B = 半行）：
  `q|gate` 12288 列 ⇒ `4097 × 12288 × 2` = 100,687,872 B；`qkvzba` 16480 列 ⇒
  `4097 × 16480 × 2` = 135,037,120 B；`opin` 6144 列 ⇒ `4097 × 6144 × 2` = 50,343,936 B。
- **`pos`**：当前只有 `m15_layer_loop/m15_layer_kernel.h` 的 `LayerArgs::apPos` —— **单个 `uint32`**。
  prefill 需要二选一：(a) **per-row 位置数组**；(b) **单请求连续** `[start, start+m)` 的约定 +
  `start` 标量（更省，且与官方 `query_start_loc` 的语义相容）。cos/sin 表**容量不缺**：
  `m15_layer_loop/m15_attn_prolog.h` 的 `CS_NPOS = M15Kv::PREFILL_M = 4097` 已按 prefill 预生成
  ⇒ 缺的只是"每行取哪一行表"。
- **`slot_mapping`**：**m15 树里不存在**。`20bd20d` 上在
  `m15_layer_loop/m15_layer_loop.asc` + `m15_layer_loop/m15_layer_kernel.h` +
  `m15_layer_loop/m15_attn_layer.h` 三个文件里搜 `slot_mapping` 命中 **1 处**（命令与读数见 M103-8 C3），
  且那一处是 `m15_layer_loop/m15_attn_layer.h` 的**注释**（M76 对早期 README 声明的证伪清单里列的
  缺失输入）。M82 的选择是"**所有地址经 `M15KV_KV_*` 宏由 `pos` 算**"，paged 契约写成
  "连续布局 = **恒等 `block_table`** 的特例"（`m15_layer_loop/m15_attn_kv.h` 的 D2，device 侧页号由
  `GetValue` 取）。⇒ **prefill 可以继续不引入 `slot_mapping` 数组**，但需要在接口层写出
  **"只支持单请求 / 恒等表"的显式契约**，否则 batch>1 时会静默错（M103-6 第 8 条）。
- **mask**：稀疏路径下因果性**预烘焙进 packed indices**（正文 §3.2(a)）；走稠密 causal 时核心必须
  自带三角掩码 + `s2` 上界。GDN 侧 **`decay_mask` 是两次不同的 mask**（L 处连对角清零、输出项含对角），
  正文 §3.1 的订正块（引官方 golden 行号）是权威，实现必须照它。
- **KV/cache 的写入时序契约**：主 KV 写**必须早于核心 gather**（正文 §3.2(b)）；compressed 行只在
  `(pos+1) % 4 == 0` 产生 ⇒ 4097 长度**写 1,024 行**（位置 3,7,…,4095），位置 4096 属"开放组"
  —— `m15_layer_loop/m15_attn_kv.h` 的 `M15KV_COMP_ROW_WRITTEN` 是这条的可执行定义。

### M103-4. 判据（对拍）可行性

**结论：`m=4097` 不能拿"官方输出"当判据，必须自建稠密 causal 参考。** 三条事实：

1. **唯一的官方 CPU 参考是 `m21_layer_ref/`（= M39 的交付物）**：单层、真实 checkpoint、逐段 dump +
   `m21_layer_ref/compare_dumps.py` 分级比较。`m21_layer_ref/run_reference.py` 的 `--m` **接受任意整数**，
   但**归档/验证过的档只有 m=1 与 m=64**（`m21_layer_ref/reference/` 的 6 个 tag）；
   **`--m 4097` 能否跑完（内存/时间）未确定**（M103-6 第 5 条）。
2. **`docs/17 §7` 明确禁止**把官方 QSA 的**稀疏**输出当稠密判据、并禁止"m=4097 对齐官方输出"这类声明
   （`m15_layer_loop/m15_attn_kv.h` 的 D3 照录了这条）。理由成立：`indexer_budget = 2048` 只覆盖 512 个
   4-token 块，而 4097 长度有 1025 个块 ⇒ 官方每行只看得到约一半历史，gather 集合与 softmax 分母都不同。
3. **现有的 `check_*.py` 都不收 m**：`m15_layer_loop/check_ref.py`（`M15_REF_LAYERS/STEPS`）、
   `m15_layer_loop/check_hc_ref.py`（`M15_HC_M` 默认 1）、`m15_layer_loop/check_chain_ref.py`
   （`M15_CHAIN_M` 默认 1，比对面 = `m21_layer_ref/ref/` 下的 torch 参考）、
   `m15_layer_loop/check_moe_ref.py`（无 m 参数，`M_MAX` 从 `moe_layout.txt` 读）
   ⇒ **逐段的 m=4097 判据要新建**。

**可行的分层判据**（每条都要能对"把被测对象弄坏"变红 —— `docs/17 §4` 的非空洞性 + 第五变体纪律）：

| 段 | 可用的独立参考 | 缺口 |
|---|---|---|
| GDN | `m18_gdn_prefill/check_ref.py`（numpy fp64 逐句参考；README §4.7 记录 m=4097 / 48 head 的 target 档已 PASS） | 它**不含** conv1d / l2norm / gating（那三步在核外）⇒ prefill 版多出的 P1/P2/P3 要另立参考 |
| attention 前端 + cache | `m15_layer_loop/evidence/attn_prolog/oracle_attn_prolog.py`（R1–R13 表）+ `m15_layer_loop/m15_attn_cache_host.h`（M98 的 host 侧独立填数学参考） | 两者现在都是单行/≤8 行尺度，要扩到 chunk 尺度 |
| attention core | **自建**稠密 causal 参考（numpy fp64：causal mask + partial RoPE + `×sigmoid(gate)`） | **不得**用官方 QSA 输出（`docs/17 §7`） |
| MoE | `m17_moe_real/check_ref.py` + `tools/golden/moe_block_ref.py`（现有链是 m=1 形态） | 要扩到 `Σt_e` 非均匀分布；`m15_layer_loop/README.md` §8 第 7 条已给"one-hot router 权重"构造法，但**未实现** |
| hc | `m20_hyperconn/check_ref.py` 的 `reference`（独立 numpy float64），已支持 `M15_HC_M` | 到 4097 行的内存/时间未验 |
| 整层 | `m21_layer_ref/` 单层参考可做逐段对拍（GDN / MoE / hc 段） | **attention 段必须换掉它的 QSA 输出**，否则就是 `docs/17 §7` 的禁止项 |

### M103-5. KV 与 cache 几何（**只用 `M15KV_KV_*` 宏**）

- **唯一权威 = `m15_layer_loop/m15_attn_kv.h`**。M98 **已更正**的主 KV 几何（该头文件的主 KV 段，
  带四路依据 + 一次独立交叉校验）：
  - 页 **32,768 B** = `2 head × 16 token × (K256 ‖ V256) × 2 B`；`KV_TOKEN_STRIDE = 1,024 B`
    （**同一 head 内 K‖V 相邻**，`off_V = off_K + 512`）；`KV_HEAD_PLANE_BYTES = 16,384 B`；
    `KV_BLOCK_TOKENS = 16`、`KV_HEADS = 2`。
  - 层 stride / 平面：`KV_LAYER_STRIDE = 8,421,376`（有 `static_assert`）、
    `KV_PLANE_BYTES = 101,056,512`（12 层）。
  - 寻址宏：`M15KV_KV_BLOCK_OF` / `M15KV_KV_SLOT_OF` / `M15KV_KV_IN_BLOCK_OFF(slot,head,kv,dim)` /
    `M15KV_KV_BYTE_OFF_PHYS` / `M15KV_KV_BYTE_OFF_CONTIG` / `M15KV_TBL_IDX`。
  - raw key ring：`RING_HEAD_SIZE = 140`、`RING_ROW_BYTES = 280`（**280 % 32 == 24 ⇒ 单行必须
    `DataCopyPad`**）；compressed：`COMP_ROW_BYTES = 256` + `M15KV_COMP_ROW_WRITTEN`；
    packed indices：`PACK_ROW_BYTES = 8208`（**8208 % 32 == 16 ⇒ 单行必须 `DataCopyPad`**）、
    `PACK_PLANE_BYTES = 33,628,176`。
- **容量已按 prefill 定尺**：`PREFILL_M = 4097`、`PREFILL_BLOCKS = 257`、`PREFILL_COMP_ROWS = 1028`、
  `PREFILL_COMP_ROWS_WRITTEN = 1024` ⇒ **prefill 的 KV 工作主要是"写入几何 + 时序"，不需要改容量**。
- **两处"第二份数字"是活的危险源**（已作为 finding 报塔，见 M103-7）：`m15_layer_loop/README.md`
  的主 KV 表与 `m15_layer_loop/m15_layer_loop.asc` 的 `attnKvDev` 注释仍写**旧几何**
  （`20bd20d` 上在 README 里命中 4 行：第 1050 / 1149 / 1154 / 1174 行；`.asc` 第 564 行 —— 命令与读数见
  M103-8 C5）。**按它们手算会得到恰好一半的 stride，而且是"写错地方也不越界"的静默错**。
  ⇒ 派 Wave B 时请**显式要求"只用宏，不引用 README/注释里的数字"**。

### M103-6. 未确定项与所需实验（**9 条；不许当事实引用**）

1. **GDN 选型未决**：复用 m18 的 VF 路线（结构已证、实测 15.51 ms/层、AIC 空转）还是走 mmad
   （正文 §6.3 的算力下限 1.37 ms/层，但要新建 32×32 前代 + 扫描 mmad + `h` 常驻 UB 方案）。
   **需要的实验**：① 每 chunk 扫描步的**固定开销标定**（正文 §9 存疑 5 就是这条，至今未做）；
   ② 把 m18 的 15.51 ms 与正文 §6.3 的 roofline 下限在同一档、同一机况下对齐复算一次
   （m18 README §9 自述它对外部负载敏感，早期曾观测到 65.7 ms）。
2. **相位 A 的 UB 预算未算**：GDN-prefill 的段窗 + PERSIST 是否 ≤ 248 KB（m18 自用 225.5 KB，
   GDN-decode 的 `UB_RC_END` 已是 227,328）。需要一份逐窗的静态预算表。
3. **`Bf16Gemm` 的 m-tile 循环能否直接跑 65 个 tile 未验**：代码路径在（`for mBlock < mLoop`、
   `srcDValue = K` 作行距），但契约注释写死 `mLoop == 1`，且 host 两处 launcher 都把 m 钉成 1。
   **需要的实验**：最小 `m = 128`（2 个 tile）的 GDN in_proj 对拍，看尾块与 `calcM = max(curM, 2)`
   的交互（3510 的 Nd2Nz quirk 在多 tile 下是否仍成立）。
4. **MoE 的 `mTiles > 1` 路径是否跑过未确定**：E=4/topk=4/m=1 下 `mTiles ≡ 1`；**没有找到任何传
   m>1 的入口**（两个 launcher 都 clamp 到 1；唯一的 m>1 守卫在独立的 `moe_relift/m95_e512/` 里，
   且它只测 router）。
5. **`m21_layer_ref/run_reference.py --m 4097` 能否跑完未确定**（只归档过 m=1/m=64）。
6. **mode-4 的 cross-core 在融合 kernel 里是否可用**：M101 的 core 用了 mode-4（依据是 `m10` 探针的
   真机实证），而 `m15_layer_resources.h` 的分节目前只覆盖 mode 0/2。融合前必须显式登记
   （**未确定**：mode-4 与既有 mode 是否共享同一个 id 池）。
7. **PLE 本体仍卡在 ngram 表 95.37 GiB**（`docs/14 §10` 第 1 条）。M100（未合入）解决到什么程度
   **未确定** —— M103 只读了它的 `git diff --stat` 与少量 hunk，未逐行读实现。
8. **"只支持单请求 / 恒等 `block_table`"的契约声明缺失**：照 M82 的 D2 走"`pos` 算地址 + 恒等表"
   需要在接口层显式写出这条限制，否则 batch>1 静默错。
9. **E=512 槽表改动会牵动 host arena 与 manifest**（`m15_layer_loop/slice_layer_manifest.py`、
   `m15_layer_loop/weights_manifest.txt`、`tools/weights/`），这些**不在 M103 的阅读范围**
   ⇒ 需要与 M93/M99 的 manifest 工作对齐后再派 Wave A 的 A-2。

### M103-7. 本轮报出的两条 out-of-scope finding

（两条都**不是** M103 的 scope，M103 未自行修；本节只登记，路由以塔为准。）

1. **`improve`｜主 KV 旧几何仍活在 README 与 `.asc` 注释里** ——
   `m15_layer_loop/README.md`（4 行）与 `m15_layer_loop/m15_layer_loop.asc`（第 564 行注释）仍写
   页 16,384 B / 每层 4,210,688 B / 12 层 50,528,256 B，而权威头已改成 32,768 B / 8,421,376 /
   101,056,512（M98）。M98 自己在 `m15_layer_loop/evidence/attn_cache/README.md` §5 第 1 项里
   如实披露"仍未做"。**它的危害是静默错**：读者手算得到一半的 stride，落在已分配的 101 MB 平面内。
2. **`bug`｜`DECODE_CTX = 4096` 与「m=1 时 context = 4097」的契约不符**（**M184 已修**）——
   `m15_layer_loop/m15_attn_kv.h` 的 `DECODE_CTX`（⇒ `DECODE_BLOCKS = 256`）。预填 4097 个 token 后
   decode 要读位置 0..4096 共 4097 个 token ⇒ `ceil(4097/16) = 257` 页。这两个常量当时只在该文件
   自己的 `static_assert` 里被引用（`20bd20d` 上全 `m15_layer_loop/` 搜 `DECODE_CTX|DECODE_BLOCKS|
   DECODE_COMP_ROWS` 命中 5 行，全在 `m15_layer_loop/m15_attn_kv.h` 内，命令见 M103-8 C4），
   所以当时不是活的读越界；但一旦 decode 侧拿 `DECODE_BLOCKS` 当读循环上界就会少读最后一页。
   注意 compressed 侧当时自洽（`DECODE_COMP_ROWS = 1024` = prefill 实际写入行数）。
   **M184（2026-10-05，M175 合入、`.asc` 释放后）已修**：`DECODE_CTX = 4097`（`m15_attn_kv.h:192`）⇒
   `DECODE_BLOCKS = 257`（`:202`）、`DECODE_COMP_ROWS = 1028`（`:203`）；`m15_layer_loop.asc:4856` 的
   guard 同步为 `DECODE_CTX == PREFILL_M`；`m15_attn_kv_host.h:155-164` 的 `runs=kv` decode 打印改为
   从 `DECODE_CTX` 派生（不再手写 4096）。设备读数见 `m15_layer_loop/evidence/m178_kv_geom/`。

### M103-8. 本节的复现命令与读数（全部实跑于 `20bd20d`）

命令都在 worktree 根目录下跑；`LC_ALL` 双跑一致（字符类只用 ASCII）。**下面给的是"命中行数"，
不是"零命中"这类断言**。

```
C1  基准 commit
    $ git rev-parse --short HEAD
    20bd20d

C2  M98 的 cache 填数学有没有接进融合 kernel（范围 = m15_layer_loop/ 下全部 .h 与 .asc，
    排除 m15_attn_cache.h 自身；"命中数 = 0"只在 20bd20d 与这个范围上成立）
    $ grep -n "M15AC::" m15_layer_loop/*.h m15_layer_loop/m15_layer_loop.asc \
        | grep -v "m15_layer_loop/m15_attn_cache.h" | wc -l
    0

C3  slot_mapping 在 m15 的哪些文件里出现（范围 = .asc + layer_kernel.h + attn_layer.h）
    $ LC_ALL=C grep -rn "slot_mapping" m15_layer_loop/m15_layer_loop.asc \
        m15_layer_loop/m15_layer_kernel.h m15_layer_loop/m15_attn_layer.h | wc -l
    1            # 即 m15_attn_layer.h 的注释；LC_ALL=C.UTF-8 同读数

C4  DECODE_* 常量的引用面（范围 = m15_layer_loop/ 下 *.h/*.asc/*.md）
    $ LC_ALL=C grep -rn "DECODE_CTX\|DECODE_BLOCKS\|DECODE_COMP_ROWS" m15_layer_loop/ \
        --include=*.h --include=*.asc --include=*.md | wc -l
    5            # 全部落在 m15_attn_kv.h 内（定义 3 行 + 2 条 static_assert）

C5  旧主 KV 几何还活在 README 里（范围 = m15_layer_loop/README.md 单文件）
    $ LC_ALL=C grep -c "16,384 B\|16384 B\|50,528,256\|4210688\|4,210,688" \
        m15_layer_loop/README.md
    4            # 第 1050 / 1149 / 1154 / 1174 行（20bd20d）

C6  三个段的包络常量（范围 = 三个资源头）
    $ grep -n "M_MAX = \|BASE_M = \|TOTAL_MAX = \|NUM_EXPERTS = \|TOPK_MAX = \|CHAIN_M = " \
        m15_layer_loop/m15_gdn_resources.h m15_layer_loop/m15_moe_resources.h \
        m15_layer_loop/m15_hc_resources.h
    m15_layer_loop/m15_gdn_resources.h:69:constexpr uint32_t M_MAX = 64;
    m15_layer_loop/m15_gdn_resources.h:70:constexpr uint32_t CHAIN_M = 1;

C6b 「CHAIN_M 被引用几次」取决于匹配口径（范围 = m15_layer_loop/ 全目录，含 *.md）
    $ LC_ALL=C grep -rn "CHAIN_M" m15_layer_loop/ | wc -l
    16           # 字面子串口径：M15_CHAIN_M / M15_CHAIN_MANUAL 也计入
    $ LC_ALL=C grep -rnE "\bCHAIN_M\b" m15_layer_loop/ | wc -l
    1            # 标识符口径：仅 m15_gdn_resources.h:70 的定义处
    （两个 locale 读数相同。）⇒ M103-1.1 那句必须写成**标识符口径**（"仅定义处"），
     否则与字面口径的 16 冲突 —— 复审 r1 的 advisory 指出的正是这一点。
    m15_layer_loop/m15_gdn_resources.h:88:constexpr uint32_t BASE_M = 64;
    m15_layer_loop/m15_moe_resources.h:70:constexpr uint32_t NUM_EXPERTS = 4;
    m15_layer_loop/m15_moe_resources.h:71:constexpr uint32_t TOPK_MAX = 4;
    m15_layer_loop/m15_moe_resources.h:72:constexpr uint32_t M_MAX = 64;
    m15_layer_loop/m15_moe_resources.h:73:constexpr uint32_t TOTAL_MAX = M_MAX * TOPK_MAX;
    m15_layer_loop/m15_moe_resources.h:292:constexpr uint32_t BASE_M = 64;
    m15_layer_loop/m15_hc_resources.h:43:constexpr uint32_t M_MAX = 64;
    m15_layer_loop/m15_hc_resources.h:108:constexpr uint32_t BASE_M = M_MAX;

C7  docs 引用完整性（语料层 = docs/*.md 除去扫描器自排除的 docs/17 自身）—— 三档读数
    $ python3 docs/scan_doc_refs.py ; echo rc=$?
    # 同一命令跑在三个树状态上（每档都两遍连跑、读数相同；LC_ALL=C 与 C.UTF-8 一致，字符类只用 ASCII）：
    #   ① 落库前（main 20bd20d）                doc refs=128  section refs=147  path refs=92   rc=0
    #   ② M103 首次落库后（tip af33b6e）        doc refs=129  section refs=156  path refs=231  rc=0
    #   ③ 复审 r1 修复后（本次）                doc refs=129  section refs=157  path refs=247  rc=0
    # ③ 的原样输出（本文件当前状态的读数）：
    docs scanned=15  doc refs=129 (unresolved=0)  section refs=157 (unresolved=0)
    in-repo path refs=247  gating=0 (distinct=0)  allowlisted=2 (distinct=2)
    RESULT: OK (15 files scanned, 129 doc refs, 157 section refs, 247 path refs, 2 allowlisted, 0 out-of-range)
    rc=0
    # ③ − ① = 本节净增 1 条 doc→doc 定位、10 条 `docs/NN §X` 章节点位引用、155 条路径式引用，全部可解析。
    # **口径纪律**：本节每改一次，这一栏必须重取（M103 初稿就在这上面犯过一次"半刷新"：C7 记的
    #   读数是在它自己写进文件之前取的，而 C7 文本里对另一份 docs 文件的提及又让那个数 +1）。
    $ python3 docs/scan_doc_refs.py --selftest ; echo rc=$?
    RESULT: OK (5 条负向对照全部成立)   rc=0

C8  Wave B 的 scope 两两不相交 + 自覆盖（对应 M103-2.6 的 stem/dir glob 写法）
    $ python3 - <<'EOF'
    from fnmatch import fnmatch
    stems = ['m15_gdn_prefill','m15_attn_prefill','m15_attn_fa_core','m15_moe_prefill','m15_hc_prefill']
    sibs  = {s: [('%s.h'%s), ('%s_host.h'%s), ('%s_res.h'%s), ('%s.asc'%s)] for s in stems}
    bad = [(a,b,f) for a in stems for b in stems if a!=b for f in sibs[b]
           if fnmatch('m15_layer_loop/'+f, 'm15_layer_loop/%s*'%a)]
    print('段体头 stem glob 两两相交对数 =', len(bad), bad)
    dirs = stems
    bad2 = [(a,b) for a in dirs for b in dirs if a!=b
            if fnmatch(b+'/evidence/run.log', a+'/**') or fnmatch(b+'/CMakeLists.txt', a+'/**')]
    print('目录 glob 两两相交对数 =', len(bad2), bad2)
    print('每 wave 自覆盖 =',
          all(all(fnmatch('m15_layer_loop/'+f,'m15_layer_loop/%s*'%s) for f in sibs[s]) for s in stems))
    EOF
    段体头 stem glob 两两相交对数 = 0 []
    目录 glob 两两相交对数 = 0 []
    每 wave 自覆盖 = True
    # 注：fnmatch 的 `*` 与 picomatch 一样不跨 `/`；`**` 的点文件语义见 M103-2.6 的写法说明
    #     （本机无 node ⇒ 那条语义取自塔裁与 picomatch 默认，M103 未能本地实证）。

C9  复审 P1-1 的修前复现 + 修后对照（文件都在 m15_layer_loop/ 下；这里用裸名，以免触发扫描器的路径门）
    $ python3 -c "
    from fnmatch import fnmatch
    b2='m15_attn_prefill*'; b3='m15_attn_prefill_core*'
    for f in ['m15_attn_prefill_core.h','m15_attn_prefill_core_host.h']:
        print(f, 'B2:',fnmatch(f,b2), 'B3:',fnmatch(f,b3))"
    m15_attn_prefill_core.h B2: True B3: True
    m15_attn_prefill_core_host.h B2: True B3: True
    # ⇒ 修前：B2 的 `m15_attn_prefill*` 把 B3 的两个文件一并吃掉 ⇒ 两条并行 mission 的 scope 相交。
    # 修后：B3 改名到 `m15_attn_fa_core*`；B2 的 stem glob 去匹配 B3 的新文件名 ⇒ 命中 (none)
    #       （C8 的「两两相交对数 = 0」是它的全局版）。

C10 复审 P2-1 的平面尺寸重算（同族手算式一并给）
    $ python3 -c "print(4097*2560*2, 4097*12288*2, 4097*16480*2, 4097*6144*2, 4096*2560*2)"
    20976640 100687872 135037120 50343936 20971520
    # ⇒ 4097 行 × 2560 列 × 2B = 20,976,640（初稿误写 20,979,200，多 2,560 B = 半行；
    #    `4096 × 2560 × 2` = 20,971,520 也不是它 —— 初稿那个数没有对应的行数）。
```

**本节未做的事（如实划界）**：① **零设备档** —— M103 全程没有跑任何 device 程序（塔的派单也写明
本 mission 不需要设备；`npu-smi info` 当时显示 NPU 0 上有别的 `m15_layer_loop` 进程在跑，
M103 未排队、未抢跑）；② 未逐一穷举 `m15_layer_loop/` 之外的文件对某个符号的引用；
③ M100/M101/M104 三个未合入分支只读了 `git diff --stat` / 分支名与少量 hunk，**未逐行读实现**
（M104 读时 **0 提交**）；④ 正文 §6 的 FLOP/时间估算未重算，本节只标注它们的现况是否仍然对得上；
⑤ **picomatch 的点文件语义未在本地实证**（本机无 node）—— 见 M103-2.6 的写法说明。

### M103-9. 修订记录（复审轮次）

**r1（2026-09-27；复审对象 = `af33b6e`；裁决 `p1-2items` / fix-then-merge）** —— 逐条处置：

| # | 复审条目 | 处置（落在哪一节） | 复核依据 |
|---|---|---|---|
| P1-1 | B2 的 `m15_attn_prefill*` 与 B3 的 `m15_attn_prefill_core*` 两个 glob 相交（fnmatch 实证 True/True） | 采纳复审解法的**改名**方案：B3 → `m15_attn_fa_core*`；并把全部 scope 改成 **stem glob / 目录 glob**（不再用前缀 `*` 去覆盖兄弟段），收敛到 M103-2.6 的**唯一权威清单** | C8（两两相交 = 0、自覆盖 = True）＋ C9（修前 True/True 复现） |
| P1-2 | B4 的「E=512 槽表全部落在自己的 `_res.h`、Wave A 只保留指针+分支」与 A-2 的「Wave A 重算 `MW_*`/`MOE_W_STRIDE`」互相矛盾 | **按塔裁统一**：`MW_*` / `MOE_W_STRIDE` 的**唯一 owner = Wave A 的 `m15_layer_resources.h`**；B4 **只读不重复定义**。原话作废并写明（M103-2.1 的 A-2 ＋ M103-2.2 的 B4 ＋ 共同契约第 3 条） | `MW_WGU_BYTES` 的算式与两档读数（6,553,600 / 838,860,800）写进 A-2 |
| P2-1 | `4097 × 2560 × 2` 写成 20,979,200 | 两处改成 **20,976,640**，并在两处**各附同族手算式**（12288 / 16480 / 6144 列） | C10 |
| P2-2 | Wave B 的 scope 不含 `evidence/`，也不含 README | M103-2.6 每条 Wave B 补 **`<dir>/**`** ＋ **`<dir>/evidence/.gitignore`**（显式列点文件）；各 wave 的 `README.md` 落在**自己的目录**内（不再依赖 `m15_layer_loop/README.md`） | M103-2.6 的清单 ＋ C8 的自覆盖读数 |
| Adv-1 | `m15_layer_loop/CMakeLists.txt` 落在任何 wave 之外、却自称"Wave B 唯一共享文件" | **按塔裁**：Wave B **一律不碰**它；每条 Wave B 的独立验证路（自己的 `CMakeLists.txt` ＋ `.asc`（含 `main()`）＋ host 头 ＋ README ＋ evidence）放进**自己的目录**，段体头仍交付到 `m15_layer_loop/`。原话作废并写明（M103-2.0 的 P1 ＋ 共同契约第 7 条 ＋ M103-2.6 的注） | 相撞矩阵里该文件已从 Wave B 接触面移除 |
| Adv-2 | `CHAIN_M` "被引用 0 次" 与字面 grep 的 16 冲突 | 改成**标识符口径**（`\bCHAIN_M\b` = 1 行 = 仅定义处），并把**两条读数**都写进 C6b | C6b |
| 塔裁新增 | 防「建好但从未集成」的 fork 堆积（先例 = `m18_gdn_prefill/`） | 新增 **M103-2.7**：段体头从第一天即可被融合 TU include；交付一份**融合清单**（挂载点 / `LayerArgs` 字段 / 资源窗 / BufferID / flagId / 相位边界 / `m`·`pos` 语义）；**Wave A 为每段留一个默认关闭的编译期开关**（形态照 M100 未合入分支的 `M15_PLE_WIRE`，缺省 0） | M100 的 `M15_PLE_WIRE` 读数（wt-100 上 25 行命中）标注为未合入分支 |
