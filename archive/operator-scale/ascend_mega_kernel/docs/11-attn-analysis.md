# Full-Attention 层分析（计算规格 · donor 盘点 · KV cache · mega kernel attention 段建议）

> 调研：M7 survey（2026-09-26，agent-attn）。范围：/workspace 下 vllm（qwen4_exp/qwen3_next 模型实现）、Qwen3.8-Flash-Next-MXFP4 config + safetensors 头、ops-transformer（flash_attn / fused_infer_attention_score / incre_flash_attention / posembedding）、vllm-ascend（只读）。
> 适用约束：docs/05-megakernel-design.md v1.2 §2/§6——per-layer kernel 边界、mix 1:2 启动、禁 Matmul 高阶 API / TPipe/TBuf/TQue → BufferID + CrossCore、裸指针基础 API；验收 m=1（decode）与 m=4097（prefill）。
> **范围裁决（tower 代判，2026-09-26；同日按 tower 的 QSA 裁决更新表述）**：首期 mega kernel 做 **dense GQA FA core + gather 接口预留**（indices 输入，nullptr=稠密）；indexer 段独立成段后续做。**验收基准 = 自建稠密 causal 参考，它不是官方行为**：官方 QSA 的 `indexer_budget=2048` 是 token 预算（block_topk = 2048/4 = 512），4097 上下文有 1025 个可见块 ⇒ 官方**只 attend 约一半历史**，稠密 causal 与官方输出**不可能一致**（裁决原文与影响见 `docs/17-verification-standard.md` §7 附则）。
> **交叉引用（tower 指令，2026-09-26）**：**索引算术与 cache 布局以 `docs/14` §7 与 `docs/15` §2.5 为准**；本文 §1.1 的投影表为早期版本，精度低于前者。另注意：本 checkpoint 的 12 层在 `layer_types` 里标成 `full_attention`，**QSA 判定必须靠 indexer 字段存在**而非字符串——否则会被误判成稠密 GQA。

## 0. 头号发现：12 个 "full_attention" 层实际是 QSA 稀疏检索注意力

layer_types 里的 `full_attention` ≠ 稠密全注意力。vLLM qwen4_exp 的层分派（`vllm/models/qwen4_exp/nvidia/model.py:216-236`）：

```python
use_qsa = layer_type == QSA_LAYER_TYPE or getattr(config, "indexer_n_heads", None) is not None
```

本模型 config 带 `indexer_n_heads=4` → **全部 12 个 full_attention 层走 `Qwen4ExpQSAAttention`**（`nvidia/qsa.py:252`，继承 Qwen3NextAttention）。safetensors 实证：层 3/11/15… 均有 `self_attn.indexer.index_qk_proj`（BF16 [640, 2560]）。数据流（`nvidia/qsa.py:500-518`）：

```text
hidden ──index_qk_proj[2560→640]──► indexer：GemmaRMSNorm(128) + rope(64维) + 写双 cache
   │                                     └─ q_idx[4头×128] · compressed_K[n/4,128] → top-512 块
   │                                        = 每 token 选 2048 个 KV token（packed [m,2051]+count 列）
   └─q_proj/k_proj/v_proj + q/k_norm + rope ─► sparse GQA（只对选中的 ≤2048 token 做注意力）► sigmoid(gate) ► o_proj
```

- **indexer 规格**（`nvidia/indexer_qsa.py:97-190`，config 值）：4 个 query 头 ×128 维 + 1 个 KV 头（MQA）×128 维；`indexer_budget=2048`（token_topk）、`indexer_compress_ratio=4`（每 4 token 压缩 1 个检索块，512 块 × 4 = 2048）；norm 是 GemmaRMSNorm(128, eps=1e-6)；rope 复用主 attention 的 rotary（64 ≤ 128 校验 `config.py:170-173`）。
- **双 cache**（`indexer_qsa.py:162-186`）：raw key cache（bf16，**每请求一个环**：key 每 token 128 维 **+ 3 个 int64 rope 位置尾** ⇒ **行宽 140 元素**；容量 `compress_ratio*ceil((compress_ratio+num_spec)/compress_ratio)` 行，`num_spec=0` 时 4 行 = 1,120 B/请求/层）+ compressed key cache（bf16 或 fp8-e4m3-无-scale，每 4 token 128 维——"Q/K 已 RMSNorm，fp8×fp8 直接点积"）。**原稿把 raw key 写成"每 token 128 维"，漏了位置尾与"每请求一个环"**（M89 更正，2026-09-27；依据 `common/qsa_cache.py:808-856`）。
- **sparse core**：`qsa_sparse_paged_attention`（`nvidia/ops/qsa.py:594`）——bf16（KV 也可 fp8-e4m3，本模型 attention 保持 bf16）；packed indices [m, 2052] int32（末列有效数）；head_dim 要求 2 的幂 ✓；输出过 output_gate。
- **MTP**：投机解码 step≥1 复用 step0 的 topk 选择（`skip_topk`，`indexer_qsa.py:129/385`）。
- 带宽账（每 token 每层，ctx=262144）：稠密 = K+V 各 2 头 ×256×2B ×262144 ≈ **512MB**；QSA = 扫 compressed cache 65536×128×2B=16MB + gather 2048×1KB=2MB ≈ **18MB（28×）**。decode 长下文场景 QSA 是带宽救星，也基本是必做项。

## 1. 注意力计算规格（全部经 safetensors 头 / vllm 源码验证）

### 1.1 投影与门控

| 权重 | 形状 | 说明 |
|---|---|---|
| q_proj | [12288, 2560] | 24 头 ×256 ×**2**（config 无 `attn_output_gate` 字段 → vllm 默认 True，`qwen3_next.py:307`）：前 6144 为 q，后 6144 为 gate；gate 不 norm 不旋转 |
| k_proj / v_proj | [512, 2560] | 2 KV 头 ×256 |
| o_proj | [2560, 6144] | 24×256 → 2560 |
| q_norm / k_norm | [256] | **GemmaRMSNorm**（乘 `1+w`，eps=1e-6），先 norm 后 rope（`qwen3_next.py:440-446`）。vLLM 里 `Qwen3NextRMSNorm` **就是 `GemmaRMSNorm` 的别名**（`qwen3_next.py:31`）⇒ 原词**不算错但会误导**（**不是**朴素 RMSNorm）；见 §1.1.1 |
| index_qk_proj | [640, 2560] | 见 §0 |

计算：`out = o_proj( attn(q,k,v) · sigmoid(gate) )`，scale = 256^-0.5。HF checkpoint 是分离 q/k/v_proj（不是 fused qkv_proj），mega kernel 内可融合为单次 GEMM 读三权重。

### 1.1.1 主 attention 的 q/k norm 是 `GemmaRMSNorm`（乘 `1+w`），**不是**朴素 RMSNorm（M89 更正，2026-09-27）

- **原稿写的是什么（M7 survey，2026-09-26）**：§1.1 表里那格写的是 `per-head_dim RMSNorm（Qwen3NextRMSNorm，eps=1e-6）`，措辞上把 `Qwen3NextRMSNorm` 与"朴素 per-head RMSNorm"并列，读者会直接按 `x·rsqrt(mean(x²)+eps)·w` 实现。
- **语义**：`y = x · rsqrt(mean(x²)+eps) · (1+w)`。`w` 由 checkpoint 加载 —— `layernorm.py:140-168`：`__init__` 里 `nn.Parameter(torch.zeros(hidden_size))`（**默认初始化是 0**，加载后被实测的非零值覆盖），`forward_native` 里 `weight = self.weight.float() + 1.0`（类 docstring 明说两处与 `RMSNorm` 不同：`x*(1+w)` 而非 `x*w`）。
- **它在本模型里叫什么**：主 attention 的 `q_norm`/`k_norm` 在 `qwen4_exp/nvidia/qsa.py:350-351` 构造为 `GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)`；而 **vLLM 里 `Qwen3NextRMSNorm` 就是 `GemmaRMSNorm` 的别名**（`vllm/model_executor/models/qwen3_next.py:31` = `from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm`）。⇒ 原词**不算错、但会误导**：它指向的是 `GemmaRMSNorm`，**不是**朴素 RMSNorm。原词仍留在 §1.1 的表里（便于读者查证来源）。
- **后果（checkpoint 实测）**：`layers.3.self_attn.q_norm.weight` 是 bf16 `[256]`，**256 个全部非零**，mean **0.2833** / std 0.0610 / min **−0.3457** / max 0.5938（**含负值**）；`layers.3.self_attn.k_norm.weight` mean **0.2734** / std 0.1190 / 范围 [−0.7891, 0.7773]。
  ⇒ 有效 scale 乘 `w` 约 **0.28**、乘 `(1+w)` 约 **1.28**，**差约 4.5×**；`w` 为负时 `1+w` 可以很小。
  **按朴素 RMSNorm 实现 ⇒ 数值错约 4.5×**（且是**方向级**错误：整条注意力的 q/k 幅度整体被缩/放）。
- **依据**：`qwen4_exp/nvidia/qsa.py:350-351`（**M88 的 oracle 复核**，其结论已被塔在 `2026-09-27T01:19Z` 的 inbox 裁决件确认；M88 分支 `feat/m88-attention-prolog-oracle-and-indexer` 截至 2026-09-27 **在 `main` 上还没有自己的 commit** —— 判据 `git rev-list --count main..feat/m88-attention-prolog-oracle-and-indexer`，在 tip 上可复跑；其 tip 与 `main` 齐平 ⇒ 引用时以此为界）+ `layernorm.py:140-168` + `qwen3_next.py:31`。上面那组 checkpoint 数值由 **M89 在读 safetensors header 后独立复算**（不是转录）。
- **indexer 路同族**：`qwen4_exp/nvidia/indexer_qsa.py:138-145` 的 `q_layernorm`/`k_layernorm` 同样构造为 `GemmaRMSNorm(index_head_dim, eps=rms_norm_eps)`；其 fused 实现里的 `weight = load(...) + 1.0`（`qwen4_exp/nvidia/ops/qsa_pre_indexer.py:69`）是同一口径的实证。
- **对本仓库的判据含义**：任何"把实现改成乘 `w` 而非 `(1+w)`"的形态都必须在判据里 FAIL（方向级错误不能被"逐元素界"放过）。

### 1.2 RoPE（partial 0.25 + mrope_interleaved）

- `rotary_dim = int(256 × 0.25) = 64`——**只旋每头前 64 维，后 192 维直通**；Q、K 同规则。
- neox 风格 rotate_half：`x1=x[...,:32], x2=x[...,32:64]`，`out=[x1·cos−x2·sin, x2·cos+x1·sin]`。
- `mrope_section=[11,11,10]`（和=32=rotary_dim/2）；config 带 `mrope_interleaved: true`。
- **本 checkpoint 走的是 `MRotaryEmbedding`，不是 `MRotaryEmbeddingInterleaved`**（M89 更正，2026-09-27）：`rope_type='default'`
  在 `rotary_embedding/__init__.py:110-121` 落到 `MRotaryEmbedding` 分支（`mrope_interleaved` 作为它的一个开关）；`MRotaryEmbeddingInterleaved`
  只在 `scaling_type == 'openpangu'` 才选（同文件 `:330-341`），而 `get_mrope_interleaved_id_list` 是**后者的**方法
  （`rotary_embedding/mrope_interleaved.py:139`）。
- **原稿写的是什么（M7 survey，2026-09-26）**：
  > 「`mrope_interleaved=true` 的含义：把 32 个频率对按三模态（t/h/w）计数生成**最少频次轮转置换**（`get_mrope_interleaved_id_list`），每位置对 64 维 cos/sin 重排。**文本-only 输入时三模态位置 id 相同 → 退化为一张静态置换表**（32 对频率的固定重排），kernel 内可实现为"查表换序 + 常数 cos/sin"，无运行时差异。」
  ⇒ 这段描述的是 **openpangu 分支**（`MRotaryEmbeddingInterleaved`）的机制，**对本 checkpoint 不成立**；而"文本-only 退化"这个方向**是对的，只是退化的结果不是置换表，而是恒等**（见下一条）。
- **文本-only 下 MRoPE 退化为普通 NeoX partial RoPE(64)**（M89 补证，2026-09-27）：
  - Qwen4Exp **自己覆盖了** `get_mrope_input_positions`（`qwen4_exp/nvidia/model.py:846-852`：`positions = torch.arange(len(input_tokens))`
    → `return positions.unsqueeze(0).expand(3, -1), 0`）⇒ **三个轴的 position id 恒等相同**。
  - `MRotaryEmbedding.forward_native` 在 `positions.ndim == 2` 且 `mrope_interleaved` 时对 cos/sin 调 `apply_interleaved_rope`
    （`rotary_embedding/mrope.py:399-403`）；该函数只在 `is_height` 取 `x[1]`、`is_width` 取 `x[2]`、其余取 `x[0]`
    （同文件 `:236-247`）⇒ **`x[0]==x[1]==x[2]` 时结果恒等于 `x[0]`**，三模态置换**在数值上恒等**。
  - ⇒ 主干 RoPE = **普通 neox partial RoPE（只旋每头前 64 维）**，与 indexer 路**共用同一套 cos/sin**
    （`rotary_embedding/base.py:104-126` 的 `_match_cos_sin_cache_dtype` 把 cos/sin 缓存转成 query dtype = bf16）；
    **不是"查表换序 + 常数 cos/sin"**。
- **两条边界（必须与上面一起读）**：
  - **(a) 位置尾照存**：`uses_mrope` 仍为真（= `_mrope_section(config) is not None`，本 config 有 `mrope_section`
    ⇒ `transformers_utils/config.py:676-712`），indexer 的 raw key ring 仍按 `cache_rope_positions=True` 存 **3 个 int64 位置尾**
    （`qwen4_exp/nvidia/indexer_qsa.py:164-167` + `common/qsa_cache.py:814-826`）⇒ **ring 行宽 140 不变**
    （128 key + 12 个 bf16 槽）。⇒ 本条**只说明"数值不变"**，**不构成退回 128 的理由**（M82 的 ring 契约照旧）。
  - **(b) 只在文本-only 下成立**：一旦接入**真多模态** MRoPE（三轴 position 不同），`apply_interleaved_rope` 的置换**就会生效**
    ⇒ **主干 RoPE 必须重新实现**（届时"共用 indexer 的 cos/sin"这一简并才成立）。
- **依据**：M88 的 oracle 复核（分支 `feat/m88-attention-prolog-oracle-and-indexer`，截至 2026-09-27 在 `main` 上还没有自己的 commit —— 判据同上）；M89 逐条重开上游源码复核（`model.py:846-852`、`mrope.py:236-247/399-403`、`__init__.py:110-121/330-341`、`transformers_utils/config.py:676-712`、`common/qsa_cache.py:814-826`）。
- θ=1e7，max_position 262144。cos/sin 精度：vllm 缓存 bf16；golden 对食用 fp32 缓存。

### 1.3 GQA broadcast 结构

24 q 头 / 2 KV 头，group **g=12**。donor 的标准做法（gS1-merge，`flash_attn_kernel_dn.h:296-297`）：把同一 KV 头对应的 12 个 q 头排进 cube M 维（12 行，pad 到 16），一个 KV tile 只读一次 → **KV GM 带宽 ÷12**。QK^T 每 KV 头独立（K 不跨头共享）。

## 2. Donor 盘点（head_dim=256 + GQA + bf16 支持度）

### 2.1 flash_attn arch35（A5 非量化 FA）——**主 donor**

入口 `op_kernel/flash_attn.cpp:83,92` 单 kernel `KERNEL_TYPE_MIX_AIC_1_2`；tiling key 5 模板参。

- **head_dim=256**：README 与配置表支持。**D=256 一律走 ND 模板**（DN 只注册 config 0/2/6）：prefill → config4（sOuter128/sInner128），decode m=1 → config5（sOuter32→64/sInner256：g·s1<64 判 decode，本模型 g=12 恰在其列）。
- **D=256 特殊处理**：mmRes UB 2 个（D>128 时）；BMM1 K=256 拆 2×128 kLoop（`MatmulK<128,128,128>`）；**BMM2 N=256 拆 128 列循环**（`IterateBmm2l0Split`——L0C 单 buffer 放不下 128×256 fp32）；KV L1 2×128KB（config5）或 4×64KB（config4）。
- **GQA**：gS1-merge（§1.3），checker 仅要求 n1≥n2 整除，g=12 无上限问题。
- **CV 融合数据面**：AIC `Fixpipe<...,FIXPIPE_ROW_MAJOR_UB>` L0C→AIV UB，`dualDstCtl=1` 按 M 拆两半写 2 个 AIV（`flash_attn_block_cube_nd.h:314,406`）；P（bf16）经 AIV 写 L1 3-buffer 回流给 BMM2 当 A 操作数；CrossCore **mode 4**（flash_attn 用 4；FIA 用 Buffer.Set/WaitCrossCore，语义同 mode 2 点对点）。
- **同步资产（对 BufferID 改造极友好）**：arch35 kernel 文件 **零 TQue/TBuf**，全部 `Mutex::Lock/Unlock<PIPE_*>(BUFFER_ID)`（ID 静态枚举 cube：Q_L1 0-1/KV_L1 2-5/L0A 6-7/L0B 8-9/L0C 10-13）+ CrossCoreSetFlag/WaitFlag；`SyncAll` 仅 FD 与 InitOutput。mmRes UB flag id 0-3（+16 触达第 2 个 AIV，`common/op_kernel/attn_buffer.h:36`）。
- **decode m=1**：无独立 decode kernel——metadata 驱动 split-K flash-decode（`flash_attn_metadata_aicpu.cpp:284-354`，SectionStreamK），partial 写 GM workspace，AIV-only `FlashDecode()` 归并（`flash_attn_block_vec_flashdecode.h:413-513`），combine 期间 cube 可跑下一 section。m=1 在 mmad 被 pad 成 16（`matmul.h:670-672` "m==1→16"——mega kernel 同样 m 走 16 行矩阵）。
- **prefill 流水**：调度器单 while 循环（bN2,gS1,s2）+ `PRELOAD_N=2` 任务前瞻、4 环任务缓存（`flash_attn_kernel_dn.h:221-273`）；**AIC 同帧跑 BMM1(L) + BMM2(L-2)，AIV 同帧跑 Vec1(L) + Vec2(L-2)**（`:434-456`）——两级流水，Q 每 gS1 行常驻 L1 双 buffer，KV L1 4 buffer 轮替。
- **softmax**：全 fp32 online softmax（RegBase VF：`FusedExpSub`、running max/sum、Vec2 `FlashUpdateNew/LastDivNew`）；LSE=log(sum)+max 仅在最后 s2 tile 算。
- dtype：仅 bf16/fp16 ✓ 与目标一致。

### 2.2 fused_infer_attention_score（FIA）——paged KV 契约主 donor

- **KV cache 布局契约**：三种模式，**paged 一等公民**（`blockTable!=nullptr`）：block table **int32 [B, ≥maxBlockNumPerBatch]**；cache 形状 BBH `[blocks, blockSize, N2*D]` / BNBD `[blocks, N2, blockSize, D]` / PA_NZ。block_size：非量化 GQA D≠64/128 时 **prefill 128 倍数 ≤1024、decode 16 倍数 ≤512**；**arch35 独有 strided（view）cache 支持**。连续布局（BSH/BNSD/TND）同 kernel 走 `KvLayoutType=0`。
- **arch35 内核**：noquant GQA/MLA + fullquant GQA/MLA/MXFP8/FP4 + antiquant，prefill(PFA) 与 decode(IFA) 同 op。
- **head_dim=256 支持度**：全局 D≤512 ✓（config 6/14/18 即 D256/DV256）。但注意：**重构版 noquant-GQA 模板只收 qkHeadDim=128**（`fia_tiling_nonquant_gqa.cpp:145`）→ D=256 实际落 prio 999 BASEAPI 兜底模板（vec 侧混 TQue/TBuf，改造量比 flash_attn 大）。**结论：D=256 时 flash_attn 是更干净的主 donor，FIA 抄布局契约与 cost model。**
- **FIXP**：`dualDstCtl=1` 双 AIV 直写；P 经 L1 `CROSS_CORE_SYNC_FORWARD` 3-buffer。
- **decode**：Q_S=1 ✓；s1==1 && D>128 → SOuter64/SInner128；FD split-KV cost model + AIV combine。
- **L2 hint 实证**：decode 形态（gSize·s1Size≤64）对 K/V GM 置 `CACHE_MODE_DISABLE`（流式无复用）；prefill 有 S1-outer split 保 L2 KV 复用。
- **KV 写回**：kernel 内不做 cache 写（调用方先写入）——mega kernel 要自己补这段（见 §3.3）。

### 2.3 incre_flash_attention（IFA）——950PR 上已死，仅作数据流参考

- **master 上 arch35 IFA 已删除**，aclnn V1-V4 在 DAV_3510 硬禁（"no longer supported on Ascend950"）；950 上 **FIA 是唯一入口**。donor 代码需从 `origin/9.1.0` 读。
- 价值：**decode 数据流最干净的样本**——bn2 切核；M=gDeal(≤12)→16、N=S2=256、K=256；L0C→UB fp32 直写；softmax `SoftmaxFlashV2_VF` + `FlashUpdate`；FD 按 `kvSplitPart=aicNum/bng` 切。同步是 TSCM 队列 + Matmul 框架级（须按 §2.1 的 BufferID 范式重建）。

### 2.4 关联 donor（prolog / indexer）

| 功能 | donor | 要点 |
|---|---|---|
| RMSNorm+RoPE+KV cache 写融合 | `ops-transformer/posembedding/kv_rms_norm_rope_cache/`（arch35 regbase 全套 + `_pa` 分页变体） | attention prolog 直接对标：norm→rope→写 paged cache 一 kernel |
| partial rotary 原语 | `posembedding/inplace_partial_rotary_mul/`（arch35；neox 版 + interleaved 版） | partial 0.25 = 只算前 64 维 |
| QSA indexer 对标 | `ops-transformer/attention/lightning_indexer*`、`dense_lightning_indexer_softmax_lse_v2`、`indexer_quant_cache`（均 arch35） | 检索 indexer 生产实现（PyPTO MegaKernel 家族），scan+topk 结构可借 |
| FIXP L0C→UB 交接范式 | `kv_quant_scfa_block_cube.h:300-320`（docs/05 §4.1 已固化） | get_buf(PIPE_FIX)+CrossCore mode 2+Fixpipe+drain |

## 3. KV cache 设计

### 3.1 布局选择：paged（与 serving 栈对齐）

- vLLM 侧事实：FullAttentionSpec paged，**block_size = 16 的倍数**（`qsa.py:106-108`），cache 形状 `[num_blocks, 2, block_size, 256]` bf16；QSA side caches：raw key = **每请求环形暂存**（bf16，**行宽 140 元素** = 128 key + 3 个 int64 位置尾，容量 `4*ceil((4+num_spec)/4)` 行 ⇒ `num_spec=0` 时 4 行 = **1,120 B/请求/层**）、compressed [tokens/4,128] bf16（fp8 可选）。**原稿写 raw key 为 `[tokens,128]` 是 M31 的早期口径**（M89 更正，2026-09-27；依据 `common/qsa_cache.py:808-856`，`docs/14` §7.5 同口径）。
- mega kernel 若置于 vLLM 之下跑，**必须消费 paged 布局**；donor 证明间接开销仅每 block chunk 一次 block table 查表（`BlockTableParser`），arch35 还支持 strided cache。**block_size 取 16**（QSA gather 粒度 = compress_ratio 4 token 的整数倍；小页让跨核 seq 切分粒度细）。
- 每层独立 cache（12 个 attn 层各一份）：cache 写 = 每 token 每 layer K+V 各 2×256×2B = **2KB**，kernel 内 MTE3 直写页内（block_table 偏移 + 页内偏移），无跨层竞争。

### 3.2 L2 hint 策略

- **decode（m=1）**：KV 流式无跨 tile 复用（gS1-merge 已让每 tile 单读）→ donor 实证置 `CACHE_MODE_DISABLE`；例外：QSA gather 随机 4-token 组同页多次命中有复用，保留默认或实测。
- **prefill**：KV 跨 m-tile 高复用 → 开 L2 + donor 的 S1-outer split（整段 S1 钉在同一核组）；128MB L2 可驻留多 KV 头整段（2 头×4097×1KB=8MB）。

### 3.3 decode m=1 cube tile 与 FIXP→UB 衔接（D=256, g=12）

形状（config5 系）：QK^T M=16（g=12 pad）、N=S2 tile=256、K=256→2×128 kLoop；PV M=16、N=256→**2×128 N-split**、K=S2。资源：L0C 4×64KB 轮替；KV L1 2×128KB 双 buffer；mmRes UB fp32 [8,256]×2；P L1 3×[16×256 bf16]。Q 常驻 L1（8KB）。

FIXP→UB 交接（docs/05 §4.1 范式 + donor 实证）：
1. AIC `get_buf(PIPE_FIX, id, false); rls_buf(...)` 等 FIXP 空 → CrossCore(mode 2) 等 AIV 消费完 → `Fixpipe(dstUB, srcL0C, {ROW_MAJOR, dualDstCtl=1})`（M 拆两半各写一个 AIV）→ `get_buf(PIPE_FIX,id,true); rls_buf(...)` 等写尽 → CrossCore 通知 AIV。
2. P 回流反向：AIV softmax 出 P（bf16 cast）→ DataCopy UB→L1 → CrossCore 通知 AIC → AIC 当 BMM2 A 操作数。
3. softmax 全 fp32 在 AIV（RegBase VF：Max/FusedExpSub/Sum，Vec2 FlashUpdate）。

### 3.4 QSA 段补充（正确性必做项，排期靠后）

indexer scan（decode m=1）：q_idx [4,128] × compressed_K [seq/4,128] GEMV + **top-512-of-65536** 选择（AIV `Sort32`/`MrgSort`，属 `docs/05` §6.1 规则 ⓒ 的无寄存器等价物例外，使用处须附注释指明依据）；gather 读按 packed indices 每 4 token 一组（1KB/K头）。raw cache 写带 rope 位置。

## 4. Mega kernel attention 段实现建议

### 4.1 AIC cube vs AIV softmax 配比（mix 1:2 固定）

硬件 cube:vector 算力 = **8:1**。注意力 FLOP 结构（D=256）：

- prefill 每 KV tile（M=128/AIC, N=128）：cube = 2×(2·128·128·256) = 16.8 MFLOPs；AIV 侧 softmax+update ≈ 0.34 MFLOPs → **FLOP 比 ~50:1，时间比 ~12:1**。
- decode m=1 每 KV tile：cube : AIV ≈ **~400:1**，AIV 几乎空转。

**结论：attention 单独成段时 AIV 必然大量空闲（decode 尤甚），必须吸收邻段向量工作**：① prolog 的 q/k norm + rope + gate split（AIV，与 QKV GEMM 的 cube 并行）；② epilogue sigmoid(gate)·out（AIV）；③ QSA indexer 的 topk sort 落在 AIV 空窗；④ 跨段错峰（MoE 段 vector 重）。这是 1:2 mix 下全层吞吐的关键设计点。

### 4.2 prefill m-tile 流水方案（m=4097）

- tile 静态：**gS1-merge 下 cube 的 M 维就是 GS1 行**（同一 KV 头的 g=12 个 q 头各占一行，KV tile 单读共享）。config4 每 AIC tile M=128 GS1 行，即每 tile 覆盖 ~10.7 token × 12 头；m=4097 → 49164 GS1 行 → **384 cube tile/KV 头**（尾 tile mask）。schedule 与 m 无关，tile 数由 main scalar 运行时算。
- 两级流水照抄 donor：AIC 帧内 BMM1(L)+BMM2(L-2)，AIV 帧内 Vec1(L)+Vec2(L-2)；PRELOAD_N=2 任务前瞻；KV L1 4×64KB 轮替 + Q L1 双 buffer；mmRes UB ×2 ping-pong；P L1 3-buffer。
- 跨 op 预取：attention 段的 o_proj 权重可在末段 KV tile 时由 MTE2 预取。
- causal 稀疏：s2 tile 上界随 m-tile 行变化，4097 长序列尾部 tile 大量跳过。

### 4.3 decode m=1 方案

- 每 (b=1, n2) 仅 2 个 KV 头 → 天然并行度 2 ≪ 28 AIC → **必须 FD split-K**：seq 切 28×2 份，partial（fp32 out + max/sum）写 GM workspace，AIV combine（或 CrossCore mode 0 all-AIC barrier 替代 SyncAll）。
- gS1-merge 后单 (b,n2) 的 cube M=12→pad 16；可选 2 KV 头并入一次 cube op（M=32），decode 带宽瓶颈下收益有限。
- 顺序：prolog GEMM（QKV+index_qk，cube）→ norm+rope+cache 写（AIV）→ indexer scan+topk（正确性必做项，仅排期靠后，见 §5 遗留事项 3）→ sparse/dense core（§3.3 流水）→ gate+o_proj（cube GEMV + AIV sigmoid·mul）。

### 4.4 资源预算（D=256 config，每 AIC/AIV 对）

| 资源 | 预算 | 出处 |
|---|---|---|
| L1 | KV 2×128KB（config5）/ 4×64KB（config4）+ Q 8-16KB + P 3×8KB ≈ 280-330KB / 512KB | flash_attn_block_cube_nd.h:101 |
| L0C | 4×64KB 轮替 | flash_attn_block_cube_dn.h:109-112 |
| UB (AIV) | mmRes 2×[8,256]fp32=16KB + softmax bufs ≈20KB + vec2 acc 16KB + P cast buf 8KB ≈ 60KB / 248KB | flash_attn_block_vec_dn.h:85-117 |
| CrossCore flagId | mmRes 0-3(+16)、P_L1 5-7、L0C 10-13、FD 2,4,8-11 → 0-15 窗口内复用规划 | attn_buffer.h:36 |

## 5. 遗留事项

1. ~~QSA 范围~~——**已裁决（2026-09-26 tower QSA 裁决，见 `docs/17-verification-standard.md` §7）**：首期 dense core + gather 预留；indexer 独立成段。**验收基准 = 自建稠密 causal 参考，且必须处处标注它不是官方行为**（官方只 attend 约一半历史，稠密与官方输出存在固有差异）；**不再写"m=4097 对齐官方输出"这类自相矛盾的表述**。
2. **QSA 的 cache 填充现在就要做**（同一裁决）：raw key ring + compressed key cache（含 off-by-one：压缩行在 position 3,7,…,4095 共 1024 行；pos 4096 属"开放组"只作 causal 尾部 token、不写 compressed cache）。缺这层 cache，后续 decode 接真 QSA 要把整条 KV 路径重做（docs/15 §0 #4(a)、§8.2）。
3. **打分 / topk / expand 是正确性必做项、不是性能选项**（同一裁决）：prefill 下 QSA 只省 ~1.1% 算力（docs/15 §0 #4），但**不做就与官方 token 不一致**；只排期靠后（等 attention core 通了再上），**不可砍**。
4. indexer KV dtype：首期 bf16 对齐 golden。
5. block_size = 16，需与上层 serving 栈对齐。
6. MTP skip_topk：投机解码 draft 步复用 step0 选择——mega kernel 若支持 MTP 需在 indexer 段加复用开关。
7. IFA 的 9.1.0 代码仅 antiquant 路径接线——作数据流参考，勿直接抄接线结构。
