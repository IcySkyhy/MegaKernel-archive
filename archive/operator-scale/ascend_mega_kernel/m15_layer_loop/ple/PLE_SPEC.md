# PLE 规格钉死（M85，task 1 交付物）

> 结论：**PLE 的规格在仓内是可以钉死的**，不需要"停下来报塔"。仓内已有可执行级别的伪码
> （`docs/14-hyperconnection-ple-indexer-spec.md` §4.1 / §6.1 / §6.2 / §6.3 / §6.4），
> 且本机有**语义权威源码** `/workspace/vllm/vllm/models/qwen4_exp/`（`nvidia/ple_layer.py`、
> `nvidia/ops/ple.py`、`nvidia/ngram_embedding.py`、`nvidia/model.py`、`nvidia/model_state.py`）
> 与**真实 checkpoint** `/workspace/Qwen3.8-Flash-Next-MXFP4`（131 分片）。本文把三者逐条对账后
> 固化出「输入 / 权重 / 输出 / 步序」四张表；所有形状与常量都经**本 mission 实测**（命令见 §7）。
>
> **本文不实现 PLE 本体**（那是 task 2）；本文只钉规格。凡与 `docs/14 §6.2` 有差异处，本文以
> **源码**为准并显式标注差异（本轮发现 2 处，见 §5）。
>
> 权威锚点基准：`/workspace/vllm`（本机 checkout；行号见各条引用）。

---

## 1. 输入

| # | 名称 | 形状 | dtype | 来源 / 去哪里读 | 备注 |
|---|---|---|---|---|---|
| I1 | `hidden_states` = **物化后的多流态 H'** | `[T, 10240]` | BF16 | 层 0 的 `attn_hc.combine` 输出（`MODE_COMBINE_ONLY` 落在 `HC_WS0 + WS_HCP`）；vLLM 侧 = `attn_hc.combine(hidden, prev_block_output, prev_injection)` 的返回（`nvidia/model.py:293-297`） | **不是**层输入 `[T,2560]`；10240 = `hc_count(4) × hidden_size(2560)` |
| I2 | `input_ids` | `[T]` | int32 | 本 step 的 token id | gather 前 `reshape(-1)`（`nvidia/ple_layer.py:401`） |
| I3 | `ngram_context` | `[num_reqs, 2]` | int32 | 每请求最近 2 个**已计算** token（新请求/不足时填 EOS=248044）；`nvidia/model_state.py:39-46,64-93` | 列序 = `[computed-2, computed-1]`（`ngram_context_offsets = arange(-2,0)`） |
| I4 | `query_start_loc` | `[num_reqs+1]` | int32 | 每请求在本 step 内的起始 token 偏移 | 用于求 `chunk_pos = t - qsl[r]` |
| I5 | 权重（见 §2） | — | — | checkpoint `model.language_model.layers.1.ple.*` | 只有 0-based **层 1** 挂 PLE |
| I6 | short-conv 状态 `conv_state` | `[slots, 10240, 9]` | BF16 | 每请求一行，跨 step 演化；TP-replicated；`state_len = (4-1)*3 = 9` | **独立于 GDN 的 MambaSpec**（`nvidia/ple_layer.py:185-191`） |

> **dtype 总则（实测）**：PLE 的 9 个小张量 + 128 个分片表 **全部是 BF16 / I64**，
> `config.json#quantization_config.quantized_tensors`（240 项）**不含任何 `ple.*`**（只含 MoE）。
> ⇒ PLE 走 bf16 直通，无 MXFP4 反量化。

---

## 2. 权重（role → 形状 → 切片方式 → 用在哪一步）

checkpoint 前缀 `model.language_model.layers.1.`（`docs/14 §6.3` 的 137 张量表，本轮 HDR 实测复核 ✓）

| role（checkpoint 张量名） | dtype | shape | 切片 / 装载方式 | 步 |
|---|---|---|---|---|
| `ple.ple_embedding.ngram_embedding.shard_0..127.weight` | BF16 | `[2500012, 160]` ×128 | **128 片按 shard id 顺序拼成一条** `[320001536, 160]`；行 id = 全局 id（不按片索引） | ② |
| `ple.key_proj.weight` | BF16 | `[10240, 2560]` | 行主序 `(out,in)`，直接 GEMV/GEMM | ③ |
| `ple.value_proj.weight` | BF16 | `[2560, 2560]` | idem；**与 key_proj 分开**（checkpoint 无 `kv_proj`） | ③ |
| `ple.conv1d.weight` | BF16 | `[10240, 1, 4]` | 先 `.squeeze(1)` → `w[10240,4]`，`w[c,k]`（`nvidia/ple_layer.py:373`） | ⑤ |
| `ple.norm_key.weight` | BF16 | `[10240]` | Gemma 约定 `(1 + w)`，逐元素；按 4×2560 分组 | ④ |
| `ple.norm_query.weight` | BF16 | `[10240]` | idem | ④ |
| `ple.norm_conv.weight` | BF16 | `[10240]` | idem | ④ |
| `ple.ple_embedding.layer_multipliers` | I64 | `[3]` | `m[0..2]`；**确定性生成**，可编译期固化 | ① |
| `ple.ple_embedding.ngram_heads_vocab_sizes` | I64 | `[16]` | `size[g]`；idem | ① |
| `ple.ple_embedding.ngram_heads_offsets` | I64 | `[16]` | `offset[g]`；idem | ① |

**9 个小张量 = 62.64 MiB（含 3×int64 的 280 B）；表 = 95.37 GiB**（`128 × 762.94 MiB`）。

**checkpoint 里 `kv_proj` 不存在**（NVIDIA 变体在装载时融合 `MergedColumnParallelLinear(2560,[10240,2560])`，
`nvidia/ple_layer.py:107-115`，顺序 **key 在前 / value 在后**）。本仓建议**按 checkpoint 原名分开**（省一次拼接，`docs/14 §11` 第 3 条）。

### 2.1 三个 int64 buffer 的确定性生成（可编译期固化，`docs/14 §6.4`）

```
max_multiplier = ((1<<63)-1)//248320 = 37,143,089,710,272 ; half_bound = 18,571,544,855,136
base_seed = 1234 (= seed 默认值 1234 + 10007*ple_dense_layer_id(0))
m[i]      = 2*(splitmix64(base_seed + 0x9E3779B97F4A7C15*(i+1)) % half_bound) + 1
size[h]   = nth_prime_after(20,000,000-1, h+1)
offset[h] = Σ_{j<h} size[j]
```
**词表尾部 padding 的通用公式**（本轮补记，`DIVERGENCES.md` G5）：
`total = Σ size[h]` 不是 128 的倍数，表按 128 分片 ⇒ 行数取
`padded = ceil(total / divisor) * divisor`，`divisor = text_config.make_ngram_vocab_size_divisible_by = 128`
（上游 `nvidia/ngram_embedding.py:200-201`）。本模型 `total = 320,001,446` ⇒ `padded = 320,001,536`，
每片行数 `padded / split_ngram_parts(128) = 2,500,012`（与 checkpoint 分片形状逐值相符）。

**本轮实测**（直接读 checkpoint 的 3 个 buffer，见 §7 命令 D）与上式**逐值相等**：
`m = (23703573157769, 20109073645365, 8052911324071)`、
`size = (20000003, 20000023, ...)`、`offset = (0, 20000003, 40000026, ...)`。⇒ 三者在 Ascend 侧**不必读 checkpoint**。

---

## 3. 输出

| # | 名称 | 形状 | dtype | 写到哪 | 与后续 mix 的关系 |
|---|---|---|---|---|---|
| O1 | `gated_output`（= 新的 `hidden_states`） | `[T, 10240]` | BF16 | **就地写回 I1 的缓冲**（`nvidia/ple_layer.py:428` 的第三参 `outer_residual=hidden_states` 被 `ple_conv` 写穿） | **PLE 之后**才跑 `attn_hc.mix(H'')`：`norm → down → silu → up → gate` → `BLK_attn [T,2560]` + `injection [T,4]`（`nvidia/model.py:298-303`） |
| O2 | `conv_state`（状态原位更新） | `[slots, 10240, 9]` | BF16 | 原地 | 跨 step 演化，每 step 必然改变（非空洞性判据） |
| O3 | `ids [T,16]`（内部量，可 dump 判 T1） | `[T,16]` | **int64** | 中间 dump | — |
| O4 | `key`/`value`/`gated`/`normed`（内部量） | `[T,10240]` / `[T,2560]` / `[T,10240]` / `[T,10240]` | BF16 | 中间 dump | — |

**值域**：`ids ∈ [0, 320,001,446)`（`total = Σ size[h]`）；

**PLE 不是纯加性 embedding**：它把 10240 宽的多流态本身当 **query**（④），输出是在该态上的**门控修正**
（`docs/14 §6.2` 末段；`nvidia/ple_layer.py:417-425` 把 `hidden_states` 传成 `query`）。

---

## 4. 步序（5 步；在 combine 与 mix 之间的确切插入点）

### 4.0 插入点与依赖（回答了「为什么 combine 必须先物化、mix 必须后跑」）

```
层 L=0:  … → mlp_hc.combine_and_mix → mlp_out 落成 pending BO(层 1 的 prev_block_output)
层 L=1（唯一挂 PLE 的层）:
   H' = attn_hc.combine(hidden, prev_BO, prev_inj)     # ★ 必须先物化：PLE 要写多流态
   H''= ple(H', input_ids, query_start_loc, ngram_context)   # ← 四相位 kernel 的"PLE 段"
   H'', BLK_attn, inj' = attn_hc.mix(H'')              # ★ mix 必须在 PLE 之后
   attn_out = linear_attn(BLK_attn)                    # 层 1 是 linear_attention（GDN）
```
- `nvidia/model.py:291-303`（组合并 + PLE + mix 分支）**逐句**对应上框；
- ⇒ `MODE_COMBINE_ONLY`（只 W0+S1、物化 bf16 H'）与 `MODE_MIX`（只 norm→…→gate）**必须存在**，
  且中间夹 PLE 段 —— 这正是 `m15_hc_resources.h:98` 的 `MODE_COMBINE_ONLY = 3` 的用途；
- 一 step 只跑**一次**；其余 47 层 `self.ple is None`（`nvidia/model.py:193-207` 的构造条件）；
- MTP 层**不挂** PLE（layer_idx 越界，`nvidia/model.py` 的 mtp 入口）。

### 4.1 五步（decode，T token/step；`nvidia/ops/ple.py` 逐行对账）

常量（`config.json#text_config`，§7 命令 A 实测）：`N=3, P=8, G=(N-1)*P=16, head_dim=He/G=160,
He=ple_embed_dim=2560, H=hidden_size=2560, hc=4, W=10240, K=ple_conv_kernel_size=4, dilation=N=3,
state_len=(K-1)*dilation=9, eps=rms_norm_eps=1e-6, eos=248044`。

```
① n-gram id（int64 位运算）——  nvidia/ops/ple.py:25-113
   c    = t - qsl[r]                      # 本 step 内 chunk 位置
   cur  = input_ids[t]
   # ★ lag = shift 的跨 chunk 候选列 = `NC - shift + c`（**依赖 c**，上游 ops/ple.py:81
   #   的 `ctx_col = NGRAM_CONTEXT_LEN - shift + chunk_pos`；eager 侧等价式见
   #   nvidia/ngram_embedding.py:322-337 的 `context = cat([ngram_context, packed])` +
   #   `shifted[s][c + NC] = context[c + NC - s]`）
   prev1= (c>=1) ? input_ids[t-1] : ctx[r, NC-1+c]     # c=0        → ctx[1]（= computed-1）
   prev2= (c>=2) ? input_ids[t-2] : ctx[r, NC-2+c]     # c=0 → ctx[0]；**c=1 → ctx[1]**
   # EOS 回退：从新到旧走 shift=1,2，一旦 candidate==EOS，更老的候选全部换成 EOS
   #   （"crossed" 累计位；ple.py:78-98）
   mixed = int64(cur) * m[0]
   mixed ^= (order>1) ? int64(prev1)*m[1] : 0     # order = g//P + 2 ⇒ g<8 只做 bigram
   mixed ^= (order>2) ? int64(prev2)*m[2] : 0
   ids[t,g] = (mixed mod size[g]) + offset[g]     # mod 取非负余数（torch.remainder 语义；ple.py:104-105）

② 行查找（稀疏 gather）——  ple.py 之外，nvidia/ngram_embedding.py
   emb16 = table[ids]            # [T,16,160] BF16
   emb   = emb16.flatten(-2)     # [T,2560]

③ key/value 投影（TP-replicated, disable_tp=True）——  nvidia/ple_layer.py:415-416
   kv = [key_proj ; value_proj] @ emb      # [T,12800]
   key = kv[:, 0:10240] ; value = kv[:, 10240:12800]

④ gate + 分组归一（逐 (t, stream) 一个 program，s=0..3）——  nvidia/ops/ple.py:185-281
   k = key[t, s*2560+·] ; q = H'[t, s*2560+·]        # ★ query = 当前多流态
   k_n = bf16( k_fp32 * rsqrt(mean(k^2)+eps) * (1+nk_w) )
   q_n = bf16( q_fp32 * rsqrt(mean(q^2)+eps) * (1+nq_w) )
   dot = fp32( bf16( Σ bf16(k_n*q_n) ) )
   d   = fp32( bf16( dot / sqrt(2560) ) )   # dot 本身 = fp32( bf16( Σ bf16(k_n·q_n) ) )（**和先物化**；ops/ple.py:218-219）
   g   = fp32( bf16( sigmoid( sign(d) * bf16(sqrt(max(|d|,1e-6))) ) ) )
   v   = value[t, 0:2560]                            # ★ 4 个 stream 共享同一 value
   gated[t, s*2560+·] = bf16( bf16(g) * v )
   normed[t,s*2560+·] = bf16( fp32(gated) * rsqrt(mean(gated^2)+eps) * (1+ncw_w) )

⑤ 膨胀深度卷积 + 状态更新 + 残差加 ——  nvidia/ops/ple.py:319-485
   #  taps（dilation=3, K=4）：h = j + 3k，h<=8 读 conv_state[c,h]，否则读 convin[t]（当前输入）
   #  ⇒ 四个 tap 的 **lag = 9, 6, 3, 0**（conv_state[0] 是最老的）
   acc   = fp32 Σ_{k=0..3} fp32(w[c,k]) * fp32(tap_k)
   conv  = bf16(acc)
   y     = fp32(conv) * sigmoid(fp32(conv))          # SiLU
   ple_output = bf16( bf16(gated) + bf16(y) )        # 先加卷积，再整体舍到 bf16
   out[t]     = bf16( fp32(H'[t]) + fp32(ple_output) )   # ★ 最后加外层残差（= 新的 hidden_states）
   #  状态更新（decode 融进同一 kernel）：state[c,i] = state[c,i+1] (i<8) ; state[c,8] = convin[t]
   #  （prefill/spec 走 ple.py:489-569 的独立 writeback kernel —— 本 mission 只做 decode）
```

**舍入点必须逐个复刻**（`nvidia/ops/ple.py:218` 的注释 "Match eager materialization at each
intermediate tensor boundary"）：上表里每个 `bf16(...)` 都是真实存在的舍入，不是装饰。

---

## 5. 与上游实现的差异（**完整清单在 `ple/DIVERGENCES.md`**）

本节只列**本 SPEC 自身曾经写错/写简**的四条；完整对照（含布局/接口/未实现项、以及每条"咬它的判据"）
见 **`ple/DIVERGENCES.md`** —— 那份表是 M85 r1 复审要求的"逐项对照"产物。

| # | 本 SPEC 的原写法 | 上游实际 | 影响 / 现状 |
|---|---|---|---|
| D1 | ③ 写 `kv = W_kv @ emb`，`W_kv = merge(key_proj, value_proj)` | checkpoint **没有** `ple.kv_proj`；融合发生在装载期（`MergedColumnParallelLinear(2560,[10240,2560])`，顺序 key、value） | 装载时按 `[key; value]` 拼或分开算；**顺序不可反** |
| D2 | ⑤ 写 `y[j] = silu(bf16(Σ w·tap))`、`gated_output = bf16(gated + y)`、`out = bf16(hidden + gated_output)` | 源码在 ⑤ 里还有**一次** `bf16` 落点：`conv = bf16(acc)` → `y = conv_fp32 * sigmoid(conv_fp32)` → `conv_output = bf16(y)`（`ple.py:438-440`）；即 **SiLU 在 bf16 取整之后的 fp32 值上算** | T3 判据的舍入链按源码写 |
| **D3**（**M85 r1 复审 P1，本轮修**） | `prev2 = (c>=2) ? input_ids[t-2] : ctx[r,0]` —— 跨 chunk 列**与 c 无关** | `ctx_col = NC - shift + c`（`ops/ple.py:81`）；eager 同：`context=cat([ngram_context,packed])`、`shifted[s][c+NC]`（`ngram_embedding.py:322-337`）。**c=1 时 lag-2 取 `ctx[r,1]`** | 只影响多 token chunk 的第二个 token（decode 的 `c=0` 不受影响，prefill 会静默继承）。已修 SPEC/kernel/参考三处；`B1.ids.A` 现在能咬住（变异 bit1） |
| **D4**（**M85 r1 复审 P2，本轮修**） | ④ 的 `dot = fp32(bf16(Σ bf16(k_n·q_n)))` 写对了，但**第一版 kernel 漏了这次取整** | `dot = tl.sum(products.to(f32)).to(dtype).to(f32)`（`ops/ple.py:218-219`）；AMD eager 侧同（`amd/ple_layer.py:1113-1117`） | "和先物化到 bf16 再除"；**只在部分输入上可观测**（~2%/gate 项）⇒ 随包数据按 `pick_observable_seed.py` 选种子使可观测；`B4.gated` 现在能咬住（变异 bit2） |

**与源码一致、不需要分歧条目的项**（本轮逐条复核）：taps lag 9/6/3/0、`(1+w)` Gemma 约定、
query = 物化后的多流态、value 4 流共享、`sign(d)·sqrt(max(|d|,1e-6))` 门控、4×2560 分组、
状态移位 `new[i]=old[i+1]`、`NULL_STATE_ID`（null 行仍写 out、不写状态）、
`layer_multipliers`/`sizes`/`offsets` 的确定性生成。

**`docs/14 §6.1` 的两条叙述也逐条核实**：PLE 挂 0-based layer 1 ✓（`ple_layer_ids=[2]` 是 1-based，
config 实测 ✓；checkpoint 张量名也是 `layers.1.ple.*` ✓）；`state_len=9` ✓；`split_ngram_parts=128` ✓。

---

## 6. 判据分档（task 3 的输入，按 `docs/17 §1.1`）

| 被验量 | 档 | 理由 |
|---|---|---|
| `ids [T,16]` int64（含 xor/mod/offset） | **T1 逐位** | 纯整数域 + 位运算；无浮点介入 |
| gather 行的**索引与取到的行** | **T1 逐字节** | 索引类；行内容 bf16 拷贝无算术 |
| `kv`（③ GEMM） | **T1 逐字节**（小档）/ **T3**（m 大） | ② 的输入是 bf16 真值、③ 是 m×k 的 mmad 累加（`k=2560` ⇒ 触发 T3 的 "mmad/cube 累加"条件）；小 m 单 tile 时若累加序可完整建模可报 T1 |
| `k_n/q_n/gated/normed/conv/out` | **T3 逐元素界** | 含 **Rsqrt**（超越近似）+ **Sigmoid**（超越近似）+ 跨 2560 的归约 ⇒ 三条触发条件全中 |
| `conv_state` 演化 | **T4 结构性** + T1（移位是纯拷贝，逐字节可判） | 状态演化非空洞性 |
| 负向对照 | **必须有**（`docs/17 §4`） | 见 README 的对照设计 |

**T3 的 ε 推导要求**（`docs/17 §1.1` 两条护栏）：须逐项列出 ε 来源（`Rsqrt`/`Sigmoid` 的官方精度规格、
2560 长度归约的 `k·2^-24`、bf16 每次物化的 `0.5·ulp`），**逐元素**检查，参考本身已量化到 bf16 时该项取 `1.0·ulp`。

---

## 7. 实测命令与输出（本文所有形状/常量的来源）

| # | 命令 | 输出（摘要） |
|---|---|---|
| A | `/usr/local/python3.12.13/bin/python3.12 /tmp/ple_cfg.py`（读 `config.json#text_config`） | `hidden_size=2560, hc_count=4, ngram_size=3, heads_per_ngram=8, ple_embed_dim=2560, ple_layer_ids=[2], ple_conv_kernel_size=4, split_ngram_parts=128, ngram_vocab_size_base=20000000, rms_norm_eps=1e-06, eos_token_id=248044`；`seed` 与 `ple_dense_layer_id` **缺省** |
| B | 同上脚本读 `quantized_tensors` | 240 项，**全部**是 `mlp.experts.{gate_up_proj,down_proj}` / `mlp.shared_expert.*`；`quant_method="ascend"`, `format="mxfp4-pack-quantized-e8m0"` |
| C | `python3.12 /tmp/ple_hdr.py`（扫 131 个 safetensors header） | 128 片 `BF16 [2500012,160]`；`key_proj BF16 [10240,2560]`（shard 5）；`value_proj BF16 [2560,2560]`；`conv1d BF16 [10240,1,4]`；`norm_key/query/conv BF16 [10240]`；3 个 `I64` buffer |
| D | 同上脚本读 3 个 int64 buffer 的字节 | `layer_multipliers=(23703573157769,20109073645365,8052911324071)`；`vocab_sizes`/`offsets` 与 `docs/14 §6.4` 的复算值**逐值相等** |

**未取证项（留给后续 mission，不在本 mission 范围）**：95.37 GiB 表在本容器的落地方式
（`docs/14 §10` 第 1 条仍是阻塞项）；prefill/spec 路径（`ple.py:489-569` 的独立 writeback）；
prefetch 机制（`start_prefetch`）。
